# #!/usr/bin/env python3
# import os, csv, argparse, random
# from pprint import pprint
# from itertools import cycle

# import torch
# import torch.nn.functional as F
# from torch.utils.data import DataLoader
# from tqdm import tqdm

# import torchvision
# from torchvision import datasets, transforms

# # your utils must add the Normalize wrapper inside make_and_restore_model
# from utils import set_seed, make_and_restore_model

# # ---------------- Loss helpers ----------------

# def targeted_margin_loss(logits: torch.Tensor, y_tgt: torch.Tensor, kappa: float = 0.0) -> torch.Tensor:
#     C = logits.size(1)
#     one_hot = F.one_hot(y_tgt, num_classes=C).bool()
#     target_logit = logits[one_hot]
#     other_logit = logits.masked_fill(one_hot, float('-inf')).amax(dim=1)
#     # Minimize => increase target_logit over others by >= kappa
#     return torch.clamp(other_logit - target_logit + kappa, min=0).mean()

# def logits_margin_stats(logits: torch.Tensor, y_tgt: torch.Tensor):
#     C = logits.size(1)
#     one_hot = F.one_hot(y_tgt, num_classes=C).bool()
#     target_logit = logits[one_hot]
#     other_logit = logits.masked_fill(one_hot, float('-inf')).amax(dim=1)
#     return target_logit - other_logit

# def total_variation(x: torch.Tensor) -> torch.Tensor:
#     tv_h = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean()
#     tv_w = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()
#     return tv_h + tv_w

# def fft_high_energy(x: torch.Tensor, frac: float = 0.5) -> torch.Tensor:
#     xc = x - x.mean(dim=(2, 3), keepdim=True)
#     X = torch.fft.rfft2(xc, norm="ortho")
#     mag2 = (X.real ** 2 + X.imag ** 2)
#     B, C, H, W2 = mag2.shape
#     yy = torch.linspace(-1, 1, steps=H, device=x.device).view(H, 1).repeat(1, W2)
#     xx = torch.linspace(0, 1, steps=W2, device=x.device).view(1, W2).repeat(H, 1)
#     r = torch.sqrt(xx**2 + yy**2)
#     keep = (r <= frac * r.max()).float()
#     high = (1.0 - keep) * mag2
#     return high.mean()

# class EMA:
#     def __init__(self, beta: float = 0.9):
#         self.beta = beta
#         self.value = None
#     def update(self, x: torch.Tensor):
#         v = float(x.detach().item())
#         self.value = v if self.value is None else self.beta * self.value + (1 - self.beta) * v

# # ---------------- Projections ----------------

# @torch.no_grad()
# def project_linf(delta: torch.Tensor, eps: float) -> torch.Tensor:
#     return delta.clamp_(-eps, eps)

# @torch.no_grad()
# def project_l2(delta: torch.Tensor, eps: float) -> torch.Tensor:
#     flat = delta.view(delta.size(0), -1)
#     norms = flat.norm(p=2, dim=1, keepdim=True) + 1e-12
#     scale = (eps / norms).clamp(max=1.0)
#     flat.mul_(scale)
#     return delta.view_as(delta)

# def project_delta(delta: torch.Tensor, eps: float, constraint: str) -> torch.Tensor:
#     return project_linf(delta, eps) if constraint == 'Linf' else project_l2(delta, eps)

# def project_delta_ST(delta: torch.Tensor, eps: float, constraint: str) -> torch.Tensor:
#     """
#     Straight-through projection: forward=project, backward=identity.
#     Keeps unrolled graph differentiable.
#     """
#     with torch.no_grad():
#         proj = project_delta(delta.clone(), eps, constraint)
#     return delta + (proj - delta).detach()

# # ---------------- Simplex proj + MGDA (for outer scalarization) ----------------

# def _proj_simplex(v: torch.Tensor) -> torch.Tensor:
#     if v.numel() == 1:
#         return torch.tensor([1.0], device=v.device)
#     u, _ = torch.sort(v, descending=True)
#     cssv = torch.cumsum(u, dim=0) - 1.0
#     ind = torch.arange(1, v.numel() + 1, device=v.device)
#     cond = u - cssv / ind > 0
#     if not cond.any():
#         return torch.ones_like(v) / v.numel()
#     rho = torch.max(ind[cond]) - 1
#     theta = cssv[rho] / (rho + 1.0)
#     w = torch.clamp(v - theta, min=0.0)
#     return w / (w.sum() + 1e-12)

# def solve_mgda_coeffs(grads_list, iters=40, lr=0.3, device='cpu'):
#     M = len(grads_list)
#     if M == 1:
#         return torch.ones(1, device=device)
#     G = torch.stack([g.view(-1).to(device) for g in grads_list], dim=0)     # M x D
#     Gram = G @ G.t()                                                         # M x M
#     alpha = torch.ones(M, device=device) / float(M)
#     for _ in range(iters):
#         grad_alpha = 2.0 * (Gram @ alpha)
#         alpha = _proj_simplex(alpha - lr * grad_alpha)
#     return alpha

# # ---------------- Core: bi-level attack ----------------

# def universal_target_attack_bilevel(model, dataset_loader, target_class, args, log_rows):
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#     data_loader = DataLoader(dataset_loader.dataset,
#                              batch_size=args.batch_size,
#                              shuffle=True,
#                              pin_memory=True,
#                              num_workers=min(4, os.cpu_count() or 1))

#     # small fixed validation slice for logging
#     _iter = iter(data_loader)
#     try:
#         _fixed = next(_iter)
#     except StopIteration:
#         _fixed = None
#     val_x = None
#     val_tgt = None
#     if _fixed is not None:
#         val_x = _fixed[0][:min(256, _fixed[0].size(0))].to(device)
#         val_tgt = torch.full((val_x.size(0),), int(target_class), device=device, dtype=torch.long)

#     def eot_apply(x, delta_var, repeats, di_rate, di_scale, noise_std):
#         outs = []
#         for _ in range(repeats):
#             z = x
#             if di_rate > 0.0 and torch.rand(1).item() < di_rate:
#                 s = 1.0 + (2.0 * (torch.rand(1).item() - 0.5) * di_scale)
#                 h, w = x.shape[2], x.shape[3]
#                 nh = max(1, int(round(h * s))); nw = max(1, int(round(w * s)))
#                 z = F.interpolate(x, size=(nh, nw), mode='bilinear', align_corners=False)
#                 if nh >= h and nw >= w:
#                     top = (nh - h) // 2; left = (nw - w) // 2
#                     z = z[:, :, top:top + h, left:left + w]
#                 else:
#                     pad_h = max(0, h - nh); pad_w = max(0, w - nw)
#                     pad_left = pad_w // 2; pad_top = pad_h // 2
#                     z = F.pad(z, (pad_left, pad_w - pad_left, pad_top, pad_h - pad_top), value=0.0)
#             z = torch.clamp(z + delta_var, 0, 1)
#             if noise_std > 0:
#                 z = z + noise_std * torch.randn_like(z)
#             outs.append(z)
#         return torch.cat(outs, dim=0)

#     def one_restart(restart_idx):
#         delta = (torch.zeros(1, *args.data_shape, device=device)
#                  .uniform_(-1e-6, 1e-6)
#                  .requires_grad_(True))

#         # Outer optimizer over base delta
#         opt_outer = torch.optim.Adam([delta], lr=args.step_size)

#         # LR schedule over outer steps
#         if args.lr_schedule == "cosine":
#             scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt_outer, T_max=args.num_steps)
#         elif args.lr_schedule == "constant":
#             scheduler = None
#         else:
#             raise ValueError("--lr_schedule must be 'cosine' or 'constant'")

#         # EMA normalizers
#         ema_asr, ema_tv, ema_fft, ema_l2 = EMA(0.9), EMA(0.9), EMA(0.9), EMA(0.9)

#         best_delta = delta.detach().clone()
#         best_score, best_sat, best_step = -1.0, 1.0, -1

#         batch_iter = cycle(data_loader)
#         iterator = tqdm(range(args.num_steps), total=args.num_steps, leave=False)

#         # optional FGSM warm-start to avoid dead start
#         def fgsm_kick(inp, y):
#             d_ws = delta.detach().clone().requires_grad_(True)
#             x_ws = (inp + d_ws).clamp(0, 1)
#             logits_ws = model(x_ws)
#             loss_ws = F.cross_entropy(logits_ws, y)
#             g = torch.autograd.grad(loss_ws, d_ws)[0]
#             with torch.no_grad():
#                 step = args.eps * 0.0625
#                 delta.add_(step * g.sign())
#                 delta.copy_(project_delta(delta, args.eps, args.constraint))

#         if args.fgsm_warmstart > 0:
#             try:
#                 inp0, _y0 = next(batch_iter)
#             except StopIteration:
#                 inp0 = None
#             if inp0 is not None:
#                 inp0 = inp0.to(device)
#                 y0 = torch.full((inp0.size(0),), int(target_class), device=device, dtype=torch.long)
#                 for _ in range(args.fgsm_warmstart):
#                     fgsm_kick(inp0, y0)

#         # Early stop bookkeeping
#         patience = args.early_stop_patience
#         min_delta_improve = args.early_stop_min_delta
#         wait = 0

#         for step_index in iterator:
#             # ---- sample a batch not already target class
#             inp, y = next(batch_iter)
#             inp = inp.to(device, non_blocking=True)
#             y = y.to(device, non_blocking=True)
#             mask = (y != target_class)
#             if (~mask).all():
#                 continue
#             inp = inp[mask]
#             tgt = torch.full((inp.size(0),), int(target_class), device=device, dtype=torch.long)

#             # =========================
#             # INNER LOOP (ASR optimize)
#             # =========================
#             # Unrolled variable (different from base delta)
#             d_inner = delta
#             inner_lr = args.inner_lr if args.inner_lr is not None else (0.5 * args.step_size)

#             for t in range(args.inner_steps):
#                 x_adv = eot_apply(
#                     inp, d_inner,
#                     repeats=args.eot_samples,
#                     di_rate=args.di_rate, di_scale=args.di_scale,
#                     noise_std=args.eot_noise,
#                 )
#                 tgt_eot = tgt.repeat(args.eot_samples)
#                 model.eval()
#                 logits = model(x_adv)

#                 # warmup & kappa ramp
#                 if step_index < args.asr_warmup:
#                     L_asr_eff = F.cross_entropy(logits, tgt_eot)
#                 else:
#                     if args.kappa_max > 0:
#                         ramp_steps = max(1, int(args.kappa_ramp_frac * args.num_steps))
#                         ramp_pos = max(0, step_index - args.asr_warmup)
#                         alpha_k = min(1.0, ramp_pos / ramp_steps) if ramp_steps > 0 else 1.0
#                         kappa_eff = alpha_k * args.kappa_max
#                     else:
#                         kappa_eff = args.kappa
#                     L_asr_eff = targeted_margin_loss(logits, tgt_eot, kappa=kappa_eff) if args.asr_loss == 'cw' \
#                                 else F.cross_entropy(logits, tgt_eot)

#                 # inner gradient (create_graph=True to enable hypergrad)
#                 g = torch.autograd.grad(L_asr_eff, d_inner, create_graph=True, retain_graph=True)[0]
#                 d_inner = project_delta_ST(d_inner - inner_lr * g, args.eps, args.constraint)

#             # =========================
#             # OUTER LOOP (stealth tighten)
#             # =========================
#             # Evaluate ASR on val slice to control curriculum / logging
#             with torch.no_grad():
#                 if val_x is not None:
#                     asr_val = (model((val_x + d_inner).clamp(0, 1)).argmax(1) == val_tgt).float().mean().item()
#                 else:
#                     x_tmp = (inp + d_inner).clamp(0, 1)
#                     asr_val = (model(x_tmp).argmax(1) == tgt).float().mean().item()

#             # Stealth terms on the *inner-updated* delta
#             L_tv   = total_variation(d_inner)
#             L_fft  = fft_high_energy(d_inner, frac=args.fft_frac)
#             L_l2   = d_inner.pow(2).mean()

#             # saturation penalty (below hard boundary via margin)
#             if args.lambda_clip > 0.0:
#                 if args.constraint == 'Linf':
#                     threshold = args.eps - (args.eps * args.sat_margin)
#                     over = (d_inner.abs() - threshold).clamp(min=0.0)
#                     L_clip = over.pow(2).mean()
#                 else:
#                     dn = d_inner.view(d_inner.size(0), -1).norm(p=2, dim=1)
#                     threshold = args.eps - (args.eps * args.sat_margin)
#                     over = (dn - threshold).clamp(min=0.0)
#                     L_clip = over.pow(2).mean()
#             else:
#                 L_clip = torch.tensor(0.0, device=device)

#             # EMA for normalization
#             for ema, L in [(ema_tv, L_tv), (ema_fft, L_fft), (ema_l2, L_l2)]:
#                 ema.update(L)

#             # Curriculum on stealth weights
#             if args.curriculum:
#                 a = float(step_index) / float(max(1, args.num_steps - 1))
#                 w_tv  = 0.05 + 0.10 * a   # 0.05 -> 0.15
#                 w_fft = 0.03 + 0.07 * a   # 0.03 -> 0.10
#                 w_l2  = 0.02 + 0.10 * a   # 0.02 -> 0.12
#             else:
#                 w_tv, w_fft, w_l2 = 0.10, 0.10, 0.10

#             # Normalize each stealth term
#             stealth_terms = [
#                 w_tv  * (L_tv  / (ema_tv.value  + 1e-8)),
#                 w_fft * (L_fft / (ema_fft.value + 1e-8)),
#                 w_l2  * (L_l2  / (ema_l2.value  + 1e-8)),
#             ]
#             if args.lambda_clip > 0.0:
#                 stealth_terms.append(args.lambda_clip * L_clip)

#             # MGDA scalarization over stealth terms (outer)
#             grads_outer = [
#                 torch.autograd.grad(Li, d_inner, retain_graph=True, create_graph=True)[0].view(-1)
#                 for Li in stealth_terms
#             ]
#             alpha = solve_mgda_coeffs(grads_outer, iters=args.mgda_iters, lr=args.mgda_lr, device=device)
#             outer_loss = torch.tensor(0.0, device=device)
#             for a_i, Li in zip(alpha, stealth_terms):
#                 outer_loss = outer_loss + a_i * Li

#             # Optional soft penalty if ASR fell below floor (helps retain robustness)
#             if args.asr_floor > 0.0:
#                 shortfall = max(0.0, args.asr_floor - asr_val)
#                 if shortfall > 0:
#                     outer_loss = outer_loss + args.asr_floor_weight * (shortfall ** 2)

#             # Backprop to *base* delta through the unrolled inner steps
#             opt_outer.zero_grad(set_to_none=True)
#             outer_loss.backward()
#             gnorm = float(delta.grad.view(-1).norm().item()) if delta.grad is not None else 0.0
#             opt_outer.step()

#             # scheduler
#             if scheduler is not None:
#                 scheduler.step()
#             elif args.warmup > 0 and step_index < args.warmup:
#                 warm_lr = args.step_size * float(step_index + 1) / float(args.warmup)
#                 for pg in opt_outer.param_groups:
#                     pg['lr'] = warm_lr

#             # project base delta after outer step
#             with torch.no_grad():
#                 delta.copy_(project_delta(delta, args.eps, args.constraint))

#             # --- metrics for log (use last inner logits if available) ---
#             with torch.no_grad():
#                 x_last = eot_apply(inp, delta, repeats=1, di_rate=0.0, di_scale=0.0, noise_std=0.0)
#                 logits_last = model(x_last)
#                 preds = logits_last.argmax(dim=1)
#                 asr_batch = float((preds == tgt).float().mean().item())
#                 margin = logits_margin_stats(logits_last, tgt).mean().item()
#                 if args.constraint == 'Linf':
#                     sat = float((delta.abs() >= (args.eps - 1e-6)).float().mean().item())
#                 else:
#                     d_norm = delta.view(delta.size(0), -1).norm(p=2, dim=1)
#                     sat = float((d_norm / (args.eps + 1e-12)).clamp(max=1.0).mean().item())

#                 # validation ASR for best/early-stop
#                 if val_x is not None:
#                     asr_val_eval = (model((val_x + delta).clamp(0, 1)).argmax(1) == val_tgt).float().mean().item()
#                 else:
#                     asr_val_eval = asr_batch

#                 improved = (asr_val_eval > best_score + 1e-6) or (abs(asr_val_eval - best_score) <= 1e-6 and sat < best_sat - 1e-6)
#                 if improved:
#                     best_score, best_sat, best_step = asr_val_eval, sat, step_index
#                     best_delta = delta.detach().clone()
#                     wait = 0
#                 else:
#                     wait += 1
#                     if args.early_stop and wait >= patience and best_step >= 0:
#                         print(f"[T{target_class}] Early stop at {step_index+1} (best @ {best_step+1})")
#                         break

#             if (step_index + 1) % args.print_every == 0:
#                 iterator.set_description(
#                     f"[T{target_class}] step {step_index+1}/{args.num_steps} | "
#                     f"ASR(batch) {100*asr_batch:5.2f}% | ASR(val) {100*asr_val_eval:5.2f}% | "
#                     f"sat {100*sat:4.1f}% | margin {margin:.4f} | gnorm {gnorm:.3e}"
#                 )

#             if args.log_csv is not None:
#                 log_rows.append({
#                     "restart": restart_idx,
#                     "target_class": target_class,
#                     "step": step_index + 1,
#                     "asr_batch": asr_batch,
#                     "asr_val": asr_val_eval,
#                     "sat": sat,
#                     "margin": margin,
#                     "gnorm": gnorm,
#                 })

#         return best_delta, best_score

#     best_overall, best_score = None, -1.0
#     for r in range(args.restarts):
#         d, s = one_restart(restart_idx=r)
#         if s > best_score:
#             best_score, best_overall = s, d
#     return best_overall.detach().requires_grad_(False)

# # --------- Legacy (single-level) path kept for completeness ---------

# def universal_target_attack_singlelevel(model, dataset_loader, target_class, args, log_rows):
#     # Fallback to your previous single-level optimizer if --bilevel is off.
#     # To keep this answer concise, we call the bi-level routine with inner_steps=0,
#     # which effectively reduces to one outer stealth step on the base delta.
#     class DummyArgs: pass
#     dummy = args
#     orig_inner_steps = dummy.inner_steps
#     dummy.inner_steps = 0
#     out = universal_target_attack_bilevel(model, dataset_loader, target_class, dummy, log_rows)
#     dummy.inner_steps = orig_inner_steps
#     return out

# # ---------------- Driver ----------------

# def moo_generate(args, loader, model):
#     log_rows = []
#     poisons = []
#     for i in range(args.num_classes):
#         if args.bilevel:
#             poison = universal_target_attack_bilevel(model, loader, i, args, log_rows)
#         else:
#             poison = universal_target_attack_singlelevel(model, loader, i, args, log_rows)
#         poisons.append(poison.squeeze())

#     if args.log_csv is not None and len(log_rows) > 0:
#         os.makedirs(os.path.dirname(args.log_csv), exist_ok=True)
#         with open(args.log_csv, "w", newline="") as f:
#             fieldnames = ["restart", "target_class", "step", "asr_batch", "asr_val", "sat", "margin", "gnorm"]
#             writer = csv.DictWriter(f, fieldnames=fieldnames)
#             writer.writeheader()
#             writer.writerows(log_rows)
#     return poisons

# def main(args):
    
#     # dataset setup
#     if args.dataset == 'imagenet200':
#         args.num_classes = 200; args.img_size = 224; args.channel = 3
#         args.data_shape = (args.channel, args.img_size, args.img_size)
#         transform_train = transforms.Compose([transforms.RandomResizedCrop(args.img_size), transforms.ToTensor()])
#         transform_test  = transforms.Compose([transforms.Resize(args.img_size), transforms.CenterCrop(args.img_size), transforms.ToTensor()])
#         data_set = torchvision.datasets.ImageFolder(os.path.join(args.data_root, 'imagenet200', 'train'), transform=transform_train)
#         test_set = torchvision.datasets.ImageFolder(os.path.join(args.data_root, 'imagenet200', 'val'),   transform=transform_test)
#     elif args.dataset == 'cifar10':
#         args.num_classes = 10; args.img_size = 32; args.channel = 3
#         args.data_shape = (args.channel, args.img_size, args.img_size)
#         transform = transforms.Compose([transforms.ToTensor()])
#         data_set = datasets.CIFAR10(args.data_root, train=True,  download=True, transform=transform)
#         test_set = datasets.CIFAR10(args.data_root, train=False, download=True, transform=transform)
#     elif args.dataset == 'gtsrb':
#         args.num_classes = 43; args.img_size = 32; args.channel = 3
#         args.data_shape = (args.channel, args.img_size, args.img_size)
#         transform = transforms.Compose([transforms.Resize(args.img_size), transforms.CenterCrop(args.img_size), transforms.ToTensor()])
#         data_set = torchvision.datasets.ImageFolder(os.path.join(args.data_root, 'GTSRB', 'Train'),           transform=transform)
#         test_set = torchvision.datasets.ImageFolder(os.path.join(args.data_root, 'GTSRB', 'val4imagefolder'), transform=transform)
#     else:
#         raise ValueError("Unsupported dataset. Choose from: imagenet200, cifar10, gtsrb")

#     data_loader = DataLoader(data_set, batch_size=args.batch_size,
#                              num_workers=min(8, os.cpu_count() or 1), shuffle=True, pin_memory=True)
#     _ = DataLoader(test_set,  batch_size=args.batch_size,
#                    num_workers=min(8, os.cpu_count() or 1), shuffle=False, pin_memory=True)

#     model = make_and_restore_model(args, resume_path=args.model_path)
#     model.eval()

#     set_seed(args.seed)

#     poisons = moo_generate(args, data_loader, model)
#     os.makedirs(args.moo_path, exist_ok=True)
#     for i, d in enumerate(poisons):
#         torch.save({'delta': d, 'args': vars(args)}, os.path.join(args.moo_path, f'moo_delta_{i}.pth'))

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser("Bi-level (unrolled) MOO for universal targeted perturbations")

#     # Repro & device
#     parser.add_argument('--seed', default=0, type=int)
#     parser.add_argument('--gpuid', default=0, type=int)

#     # Attack / constraint
#     parser.add_argument('--constraint', default='Linf', choices=['Linf', 'L2'])
#     parser.add_argument('--eps', default=8.0, type=float)         # Linf in pixels; converted below
#     parser.add_argument('--num_steps', default=1000, type=int)
#     parser.add_argument('--step_size', default=None, type=float)  # outer lr; if None, set below
#     parser.add_argument('--restarts', default=2, type=int)
#     parser.add_argument('--kappa', default=0.0, type=float)
#     parser.add_argument('--kappa_max', default=0.3, type=float)
#     parser.add_argument('--kappa_ramp_frac', default=0.4, type=float)
#     parser.add_argument('--eot_samples', default=2, type=int)
#     parser.add_argument('--eot_noise', default=0.0, type=float)
#     parser.add_argument('--fft_frac', default=0.5, type=float)

#     # Model / data
#     parser.add_argument('--arch', default='ResNet18',
#                         choices=['VGG16','EfficientNetB0','DenseNet121','ResNet18','swin','inception_next_tiny','inception_next_small'])
#     parser.add_argument('--model_path', default='results/clean_model_weight/checkpoint.pth')
#     parser.add_argument('--dataset', default='cifar10', choices=['cifar10','imagenet200','gtsrb'])
#     parser.add_argument('--data_root', default='../data', type=str)

#     # IO
#     parser.add_argument('--batch_size', default=256, type=int)
#     parser.add_argument('--moo_path', default='./results/moo', type=str)
#     parser.add_argument('--log_csv', default=None, type=str)

#     # Loss / curriculum
#     parser.add_argument('--asr_loss', default='cw', choices=['cw','ce'])
#     parser.add_argument('--curriculum', action='store_true')

#     # Scheduler / warmup / logging
#     parser.add_argument('--lr_schedule', default='cosine', choices=['cosine','constant'])
#     parser.add_argument('--warmup', default=25, type=int)
#     parser.add_argument('--asr_warmup', default=200, type=int)
#     parser.add_argument('--fgsm_warmstart', default=4, type=int)
#     parser.add_argument('--di_rate', default=0.20, type=float)
#     parser.add_argument('--di_scale', default=0.08, type=float)
#     parser.add_argument('--print_every', default=25, type=int)

#     # Stealth / saturation control
#     parser.add_argument('--lambda_clip', default=0.70, type=float)
#     parser.add_argument('--sat_margin', default=0.05, type=float)

#     # Early stopping
#     parser.add_argument('--early_stop', action='store_true')
#     parser.add_argument('--early_stop_patience', default=80, type=int)
#     parser.add_argument('--early_stop_min_delta', default=0.003, type=float)

#     # ------------- Bi-level knobs -------------
#     parser.add_argument('--bilevel', action='store_true', help="Enable unrolled bi-level optimization")
#     parser.add_argument('--inner_steps', default=3, type=int, help="# unrolled inner steps on ASR")
#     parser.add_argument('--inner_lr', default=None, type=float, help="inner step size; default=0.5*step_size")
#     parser.add_argument('--asr_floor', default=0.70, type=float, help="soft ASR floor (outer penalty)")
#     parser.add_argument('--asr_floor_weight', default=2.0, type=float, help="weight for ASR shortfall penalty (outer)")

#     # MGDA for outer scalarization
#     parser.add_argument('--mgda_iters', default=40, type=int)
#     parser.add_argument('--mgda_lr', default=0.3, type=float)

#     args = parser.parse_args()

#     # Output dir
#     args.moo_path = f"{args.moo_path}-{args.dataset}-{args.arch}-{args.constraint}-eps{args.eps:.1f}"
#     os.makedirs(args.moo_path, exist_ok=True)

#     # Normalize eps/step_size units
#     if args.constraint == 'Linf':
#         args.eps = args.eps / 255.0
#         if args.step_size is None:
#             args.step_size = args.eps / 6.0   # slightly smaller outer step for stability
#     else:
#         if args.step_size is None:
#             args.step_size = 1e-2

#     pprint(vars(args))

#     os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpuid)
#     torch.backends.cudnn.deterministic = False
#     torch.backends.cudnn.benchmark = True

#     main(args)


#!/usr/bin/env python3
import os, csv, argparse, platform
from pprint import pprint
from itertools import cycle

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import torchvision
from torchvision import datasets, transforms

# your utils must provide: set_seed, make_and_restore_model (wraps Normalize)
from utils import set_seed, make_and_restore_model

# ----------------- Loss helpers -----------------

def targeted_margin_loss(logits: torch.Tensor, y_tgt: torch.Tensor, kappa: float = 0.0) -> torch.Tensor:
    C = logits.size(1)
    one_hot = F.one_hot(y_tgt, num_classes=C).bool()
    t = logits[one_hot]
    o = logits.masked_fill(one_hot, float('-inf')).amax(dim=1)
    return torch.clamp(o - t + kappa, min=0).mean()

def logits_margin_stats(logits: torch.Tensor, y_tgt: torch.Tensor):
    C = logits.size(1)
    one_hot = F.one_hot(y_tgt, num_classes=C).bool()
    t = logits[one_hot]
    o = logits.masked_fill(one_hot, float('-inf')).amax(dim=1)
    return t - o

def total_variation(x: torch.Tensor) -> torch.Tensor:
    tv_h = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean()
    tv_w = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()
    return tv_h + tv_w

# Stable high-frequency penalty via depthwise Laplacian (no FFT; GPU/CPU safe)
def high_freq_energy_conv(x: torch.Tensor) -> torch.Tensor:
    B, C, H, W = x.shape
    k = torch.tensor([[0., -1., 0.],
                      [-1., 4., -1.],
                      [0., -1., 0.]], dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
    w = k.repeat(C, 1, 1, 1)   # depthwise
    x_pad = F.pad(x, (1, 1, 1, 1), mode='reflect')
    lap = F.conv2d(x_pad, w, bias=None, stride=1, padding=0, groups=C)
    return (lap ** 2).mean()

class EMA:
    def __init__(self, beta: float = 0.9):
        self.value = None; self.beta = beta
    def update(self, x: torch.Tensor):
        v = float(x.detach().item())
        self.value = v if self.value is None else self.beta * self.value + (1 - self.beta) * v

# ----------------- Projections -----------------

@torch.no_grad()
def project_linf(delta: torch.Tensor, eps: float) -> torch.Tensor:
    return delta.clamp_(-eps, eps)

@torch.no_grad()
def project_l2(delta: torch.Tensor, eps: float) -> torch.Tensor:
    flat = delta.view(delta.size(0), -1)
    norms = flat.norm(p=2, dim=1, keepdim=True) + 1e-12
    scale = (eps / norms).clamp(max=1.0)
    flat.mul_(scale)
    return delta.view_as(delta)

def project_delta(delta, eps, cons):  # cons ∈ {'Linf','L2'}
    return project_linf(delta, eps) if cons == 'Linf' else project_l2(delta, eps)

def project_delta_ST(delta, eps, cons):
    with torch.no_grad():
        proj = project_delta(delta.clone(), eps, cons)
    return delta + (proj - delta).detach()

# ----------------- MGDA (CPU-safe) -----------------

def _proj_simplex(v: torch.Tensor) -> torch.Tensor:
    if v.numel() == 1:
        return torch.tensor([1.0], device=v.device)
    u, _ = torch.sort(v, descending=True)
    cssv = torch.cumsum(u, dim=0) - 1.0
    ind = torch.arange(1, v.numel() + 1, device=v.device)
    cond = u - cssv / ind > 0
    if not cond.any():
        return torch.ones_like(v) / v.numel()
    rho = torch.max(ind[cond]) - 1
    theta = cssv[rho] / (rho + 1.0)
    w = torch.clamp(v - theta, min=0.0)
    return w / (w.sum() + 1e-12)

def _safe_flat_grad(g: torch.Tensor) -> torch.Tensor:
    g = g.detach().contiguous().view(-1)
    return torch.where(torch.isfinite(g), g, torch.zeros_like(g))

@torch.no_grad()
def solve_mgda_coeffs(grads_list, iters=30, lr=0.25, device='cpu'):
    M = len(grads_list)
    if M == 1:
        return torch.ones(1, device=device)
    g_cpu = [ _safe_flat_grad(g).to('cpu', dtype=torch.float32) for g in grads_list ]
    G = torch.stack(g_cpu, dim=0)     # M x D
    Gram = G @ G.t()                  # M x M
    alpha = torch.ones(M, dtype=torch.float32) / float(M)
    for _ in range(iters):
        grad_alpha = 2.0 * (Gram @ alpha)
        alpha = _proj_simplex(alpha - lr * grad_alpha)
    return alpha.to(device)

# ----------------- Safe checkpoint loading -----------------

def _extract_state_dict(ckpt):
    if isinstance(ckpt, dict):
        for k in ['state_dict','model','net','model_state','model_state_dict','ema','weights']:
            if k in ckpt and isinstance(ckpt[k], dict):
                return ckpt[k]
        if all(isinstance(v, torch.Tensor) for v in ckpt.values()):
            return ckpt
    raise RuntimeError("No state_dict found in checkpoint")

def _strip_module_prefix(sd):
    return { (k[7:] if k.startswith('module.') else k): v for k, v in sd.items() }

def make_and_load_model_safely(args):
    model = make_and_restore_model(args, resume_path=None)
    try:
        ckpt = torch.load(args.model_path, map_location='cpu', weights_only=True)
    except TypeError:
        ckpt = torch.load(args.model_path, map_location='cpu')
    sd = _strip_module_prefix(_extract_state_dict(ckpt))
    model.load_state_dict(sd, strict=False)
    return model

# ----------------- Core attack -----------------

def universal_target_attack_bilevel(model, dataset_loader, target_class, args, log_rows):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_loader = DataLoader(
        dataset_loader.dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=False if args.num_workers == 0 else True
    )

    # tiny held-out slice
    it = iter(data_loader)
    val_x = val_tgt = None
    try:
        bx, _ = next(it)
        val_x = bx[:min(64, bx.size(0))].to(device, non_blocking=True)
        val_tgt = torch.full((val_x.size(0),), int(target_class), device=device, dtype=torch.long)
    except StopIteration:
        pass

    def eot_apply(x, delta_var, repeats, di_rate, di_scale, noise_std):
        outs = []
        for _ in range(repeats):
            z = x
            if di_rate > 0.0 and torch.rand(1).item() < di_rate:
                s = 1.0 + (2.0 * (torch.rand(1).item() - 0.5) * di_scale)
                h, w = x.shape[2], x.shape[3]
                nh = max(1, int(round(h * s))); nw = max(1, int(round(w * s)))
                z = F.interpolate(x, size=(nh, nw), mode='bilinear', align_corners=False)
                if nh >= h and nw >= w:
                    top = (nh - h)//2; left = (nw - w)//2
                    z = z[:, :, top:top+h, left:left+w]
                else:
                    ph = max(0, h - nh); pw = max(0, w - nw)
                    pl = pw // 2; pt = ph // 2
                    z = F.pad(z, (pl, pw - pl, pt, ph - pt), value=0.0)
            if args.eot_bc_jitter > 0:
                a = 1.0 + args.eot_bc_jitter * (2*torch.rand(z.size(0),1,1,1, device=z.device)-1.0)
                b = args.eot_bc_jitter * (2*torch.rand(z.size(0),1,1,1, device=z.device)-1.0)
                z = z * a + b
            z = torch.clamp(z + delta_var, 0, 1)
            if noise_std > 0:
                z = z + noise_std * torch.randn_like(z)
            outs.append(z)
        return torch.cat(outs, dim=0)

    def one_restart(ridx):
        delta = (torch.zeros(1, *args.data_shape, device=device).uniform_(-1e-6, 1e-6)).requires_grad_(True)
        opt = torch.optim.Adam([delta], lr=args.step_size)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.num_steps) if args.lr_schedule=='cosine' else None
        ema_tv, ema_fft, ema_l2 = EMA(0.9), EMA(0.9), EMA(0.9)

        best_delta = delta.detach().clone(); best_score, best_sat, best_step = -1.0, 1.0, -1
        iterator = tqdm(range(args.num_steps), total=args.num_steps, leave=False, dynamic_ncols=True)
        batch_iter = cycle(data_loader)
        early_stop = getattr(args, 'early_stop', False)
        patience   = getattr(args, 'early_stop_patience', 80)
        min_delta  = getattr(args, 'early_stop_min_delta', 0.003)


        # tiny FGSM warmstart
        if args.fgsm_warmstart > 0:
            try:
                inp0, _ = next(batch_iter)
                inp0 = inp0.to(device, non_blocking=True)
                tgt0 = torch.full((inp0.size(0),), int(target_class), device=device, dtype=torch.long)
                for _ in range(args.fgsm_warmstart):
                    d_ws = delta.detach().clone().requires_grad_(True)
                    logits_ws = model((inp0 + d_ws).clamp(0,1))
                    g = torch.autograd.grad(F.cross_entropy(logits_ws, tgt0), d_ws)[0]
                    with torch.no_grad():
                        delta.add_( (args.eps*0.0625) * g.sign() )
                        delta.copy_(project_delta(delta, args.eps, args.constraint))
            except StopIteration:
                pass

        patience = args.early_stop_patience; wait = 0

        for step in iterator:
            a = step / max(1, args.num_steps-1)
            eps_curr = (args.eps_min + a**args.eps_anneal_pow*(args.eps - args.eps_min)) if args.anneal_eps else args.eps
            # kappa ramp
            if args.kappa_max > 0:
                ramp_steps = max(1, int(args.kappa_ramp_frac * args.num_steps))
                ramp_pos = max(0, step - args.asr_warmup)
                alpha_k = min(1.0, ramp_pos / ramp_steps) if ramp_steps > 0 else 1.0
                kappa_eff = alpha_k * args.kappa_max
            else:
                kappa_eff = args.kappa

            # sample batch not already target class
            inp, y = next(batch_iter)
            inp = inp.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            mask = (y != target_class)
            if (~mask).all(): continue
            inp = inp[mask]
            tgt = torch.full((inp.size(0),), int(target_class), device=device, dtype=torch.long)

            # -------- INNER (ASR) --------
            d_inner = delta
            inner_lr = args.inner_lr if args.inner_lr is not None else (0.5*args.step_size)
            for _ in range(args.inner_steps):
                x_adv = eot_apply(inp, d_inner, repeats=args.eot_samples,
                                  di_rate=args.di_rate, di_scale=args.di_scale, noise_std=args.eot_noise)
                tgt_eot = tgt.repeat(args.eot_samples)
                logits = model(x_adv)
                if step < args.asr_warmup:
                    L_asr_eff = F.cross_entropy(logits, tgt_eot)
                else:
                    L_asr_eff = targeted_margin_loss(logits, tgt_eot, kappa=kappa_eff) if args.asr_loss=='cw' \
                                else F.cross_entropy(logits, tgt_eot)
                g = torch.autograd.grad(L_asr_eff, d_inner, create_graph=True, retain_graph=True)[0]
                d_inner = project_delta_ST(d_inner - inner_lr * g, eps_curr, args.constraint)

            # -------- OUTER (stealth + robust ASR) --------
            with torch.no_grad():
                if val_x is not None:
                    asr_val = (model((val_x + d_inner).clamp(0,1)).argmax(1) == val_tgt).float().mean().item()
                else:
                    asr_val = (model((inp + d_inner).clamp(0,1)).argmax(1) == tgt).float().mean().item()

            L_tv  = total_variation(d_inner)
            L_fft = high_freq_energy_conv(d_inner)
            L_l2  = d_inner.pow(2).mean()

            if args.constraint == 'Linf':
                thr = eps_curr - (eps_curr * args.sat_margin)
                L_barrier = -torch.log((eps_curr - d_inner.abs()).clamp(min=1e-6)).mean()
                L_sat = F.softplus(d_inner.abs() - thr, beta=5.0).mean()
            else:
                dn = d_inner.view(d_inner.size(0), -1).norm(p=2, dim=1)
                thr = eps_curr - (eps_curr * args.sat_margin)
                L_barrier = -torch.log((eps_curr - dn).clamp(min=1e-6)).mean()
                L_sat = F.softplus(dn - thr, beta=5.0).mean()

            # EMA normalization
            for ema, L in [(ema_tv, L_tv), (ema_fft, L_fft), (ema_l2, L_l2)]: ema.update(L)

            # curriculum
            if args.curriculum:
                w_tv, w_fft, w_l2 = 0.05 + 0.10*a, 0.03 + 0.07*a, 0.02 + 0.10*a
            else:
                w_tv = w_fft = w_l2 = 0.10

            outer_terms = [
                w_tv  * (L_tv  / (ema_tv.value  + 1e-8)),
                w_fft * (L_fft / (ema_fft.value + 1e-8)),
                w_l2  * (L_l2  / (ema_l2.value  + 1e-8)),
                args.lambda_barrier * L_barrier,
                args.lambda_sat * L_sat,
            ]
            L_sat_mean = d_inner.abs().mean() / (eps_curr + 1e-12)
            outer_terms.append(args.lambda_sat_mean * L_sat_mean)

            # robust ASR in outer under EOT
            if val_x is not None:
                x_outer = eot_apply(val_x, d_inner, repeats=1, di_rate=args.di_rate, di_scale=args.di_scale, noise_std=args.eot_noise)
                y_outer = val_tgt
            else:
                x_outer = eot_apply(inp, d_inner, repeats=1, di_rate=args.di_rate, di_scale=args.di_scale, noise_std=args.eot_noise)
                y_outer = tgt
            logits_outer = model(x_outer)
            L_asr_outer = targeted_margin_loss(logits_outer, y_outer, kappa=kappa_eff) if args.asr_loss=='cw' \
                          else F.cross_entropy(logits_outer, y_outer)
            outer_terms.append(args.lambda_asr_outer * L_asr_outer)

            grads_outer = [torch.autograd.grad(Li, d_inner, retain_graph=True, create_graph=True)[0] for Li in outer_terms]
            alpha = solve_mgda_coeffs(grads_outer, iters=args.mgda_iters, lr=args.mgda_lr, device=device)
            outer_loss = torch.tensor(0.0, device=device)
            for a_i, Li in zip(alpha, outer_terms): outer_loss = outer_loss + a_i.to(device) * Li

            opt.zero_grad(set_to_none=True); outer_loss.backward(); opt.step()
            if scheduler is not None: scheduler.step()

            with torch.no_grad(): delta.copy_(project_delta(delta, eps_curr, args.constraint))

            # metrics/log
            with torch.no_grad():
                preds = model(eot_apply(inp, delta, 1, 0.0, 0.0, 0.0)).argmax(1)
                asr_batch = float((preds == tgt).float().mean().item())
                margin = logits_margin_stats(model((inp+delta).clamp(0,1)), tgt).mean().item()
                if args.constraint=='Linf':
                    sat = float((delta.abs() >= (eps_curr-1e-6)).float().mean().item())
                else:
                    d_norm = delta.view(delta.size(0), -1).norm(p=2, dim=1)
                    sat = float((d_norm/(eps_curr+1e-12)).clamp(max=1.0).mean().item())
                asr_val_eval = (model((val_x+delta).clamp(0,1)).argmax(1)==val_tgt).float().mean().item() if val_x is not None else asr_batch
                improved = (asr_val_eval > best_score + 1e-6) or (abs(asr_val_eval-best_score)<=1e-6 and sat < best_sat-1e-6)
                if improved:
                    best_score, best_sat, best_step = asr_val_eval, sat, step
                    best_delta = delta.detach().clone(); wait = 0
                else:
                    wait += 1
                    if args.early_stop and wait >= patience and best_step >= 0:
                        print(f"[T{target_class}] Early stop at {step+1} (best @ {best_step+1})"); break

            if (step+1) % args.print_every == 0:
                iterator.set_description(
                    f"[T{target_class}] {step+1}/{args.num_steps} | ASR(val) {100*asr_val_eval:5.2f}% | sat {100*sat:4.1f}% | margin {margin:.3f}"
                )
            if args.log_csv:
                log_rows.append({"restart": ridx, "target_class": target_class, "step": step+1,
                                 "asr_batch": asr_batch, "asr_val": asr_val_eval, "sat": sat, "margin": margin, "gnorm": 0.0})
        return best_delta, best_score

    best_delta, best_score = None, -1.0
    for r in range(args.restarts):
        d, s = one_restart(r)
        if s > best_score: best_delta, best_score = d, s
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    return best_delta.detach().requires_grad_(False)

# ------------- Single-level fallback (kept) -------------
def universal_target_attack_singlelevel(model, dataset_loader, target_class, args, log_rows):
    old = args.inner_steps; args.inner_steps = 0
    out = universal_target_attack_bilevel(model, dataset_loader, target_class, args, log_rows)
    args.inner_steps = old; return out

# ----------------- Driver -----------------

def moo_generate(args, loader, model):
    log_rows, poisons = [], []
    for i in range(args.num_classes):
        p = universal_target_attack_bilevel(model, loader, i, args, log_rows) if args.bilevel \
            else universal_target_attack_singlelevel(model, loader, i, args, log_rows)
        poisons.append(p.squeeze())
    if args.log_csv and len(log_rows):
        os.makedirs(os.path.dirname(args.log_csv), exist_ok=True)
        with open(args.log_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["restart","target_class","step","asr_batch","asr_val","sat","margin","gnorm"])
            writer.writeheader(); writer.writerows(log_rows)
    return poisons

def main(args):
    # dataset
    if args.dataset=='cifar10':
        args.num_classes=10; args.img_size=32; args.channel=3
        args.data_shape=(args.channel,args.img_size,args.img_size)
        transform = transforms.Compose([transforms.ToTensor()])
        data_set = datasets.CIFAR10(args.data_root, train=True, download=True, transform=transform)
        test_set = datasets.CIFAR10(args.data_root, train=False, download=True, transform=transform)
    elif args.dataset=='imagenet200':
        args.num_classes=200; args.img_size=224; args.channel=3
        args.data_shape=(args.channel,args.img_size,args.img_size)
        ttr = transforms.Compose([transforms.RandomResizedCrop(args.img_size), transforms.ToTensor()])
        tte = transforms.Compose([transforms.Resize(args.img_size), transforms.CenterCrop(args.img_size), transforms.ToTensor()])
        data_set = torchvision.datasets.ImageFolder(os.path.join(args.data_root,'imagenet200','train'), transform=ttr)
        test_set = torchvision.datasets.ImageFolder(os.path.join(args.data_root,'imagenet200','val'),   transform=tte)
    elif args.dataset=='gtsrb':
        args.num_classes=43; args.img_size=32; args.channel=3
        args.data_shape=(args.channel,args.img_size,args.img_size)
        transform = transforms.Compose([transforms.Resize(args.img_size), transforms.CenterCrop(args.img_size), transforms.ToTensor()])
        data_set = torchvision.datasets.ImageFolder(os.path.join(args.data_root,'GTSRB','Train'), transform=transform)
        test_set = torchvision.datasets.ImageFolder(os.path.join(args.data_root,'GTSRB','val4imagefolder'), transform=transform)
    else:
        raise ValueError("dataset must be one of: cifar10, imagenet200, gtsrb")

    dl = DataLoader(data_set, batch_size=args.batch_size, shuffle=True,
                    num_workers=args.num_workers, pin_memory=True,
                    persistent_workers=False if args.num_workers==0 else True)
    _ = DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                   num_workers=args.num_workers, pin_memory=True,
                   persistent_workers=False if args.num_workers==0 else True)

    # model (RAM-safe fallback)
    try:
        model = make_and_restore_model(args, resume_path=args.model_path)
    except (MemoryError, RuntimeError) as e:
        print(f"[load] Fallback loader due to {type(e).__name__}: {e}")
        model = make_and_load_model_safely(args)
    model.eval()

    set_seed(args.seed)
    poisons = moo_generate(args, dl, model)
    os.makedirs(args.moo_path, exist_ok=True)
    for i, d in enumerate(poisons):
        torch.save({'delta': d, 'args': vars(args)}, os.path.join(args.moo_path, f'moo_delta_{i}.pth'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Safe & Stealthy Bi-level Universal Attack")

    # reproducibility / device
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--gpuid', default=0, type=int)

    # model/data
    parser.add_argument('--arch', default='ResNet18',
        choices=['VGG16','EfficientNetB0','DenseNet121','ResNet18','swin','inception_next_tiny','inception_next_small'])
    parser.add_argument('--dataset', default='cifar10', choices=['cifar10','imagenet200','gtsrb'])
    parser.add_argument('--data_root', default='../data', type=str)
    parser.add_argument('--model_path', default='results/clean_model_weight/checkpoint.pth', type=str)

    # attack / constraint
    parser.add_argument('--constraint', default='Linf', choices=['Linf','L2'])
    parser.add_argument('--eps', default=8.0, type=float)  # 8/255 after scaling
    parser.add_argument('--num_steps', default=600, type=int)
    parser.add_argument('--restarts', default=1, type=int)
    parser.add_argument('--step_size', default=None, type=float)
    parser.add_argument('--kappa', default=0.0, type=float)
    parser.add_argument('--kappa_max', default=0.35, type=float)
    parser.add_argument('--kappa_ramp_frac', default=0.35, type=float)

    # unrolled inner loop
    parser.add_argument('--bilevel', action='store_true')
    parser.add_argument('--inner_steps', default=2, type=int)
    parser.add_argument('--inner_lr', default=None, type=float)
    parser.add_argument('--asr_loss', default='cw', choices=['cw','ce'])

    # EOT / DI (kept tiny)
    parser.add_argument('--eot_samples', default=1, type=int)
    parser.add_argument('--eot_outer_samples', default=1, type=int)
    parser.add_argument('--eot_noise', default=0.002, type=float)
    parser.add_argument('--di_rate', default=0.25, type=float)
    parser.add_argument('--di_scale', default=0.10, type=float)
    parser.add_argument('--eot_bc_jitter', default=0.02, type=float)

    # stealth & saturation penalties
    parser.add_argument('--lambda_barrier', default=0.25, type=float)
    parser.add_argument('--lambda_sat', default=0.35, type=float)
    parser.add_argument('--sat_margin', default=0.07, type=float)

    # MGDA (CPU-safe)
    parser.add_argument('--lambda_asr_outer', default=1.0, type=float)
    parser.add_argument('--mgda_iters', default=30, type=int)
    parser.add_argument('--mgda_lr', default=0.25, type=float)

    # schedule / logging
    parser.add_argument('--lr_schedule', default='cosine', choices=['cosine','constant'])
    parser.add_argument('--warmup', default=25, type=int)
    parser.add_argument('--asr_warmup', default=200, type=int)
    parser.add_argument('--fgsm_warmstart', default=2, type=int)
    parser.add_argument('--print_every', default=25, type=int)

    # IO / workers
    parser.add_argument('--batch_size', default=48, type=int)
    parser.add_argument('--num_workers', default=(0 if platform.system()=='Windows' else 2), type=int)
    parser.add_argument('--moo_path', default='./results/moo', type=str)
    parser.add_argument('--log_csv', default='./results/asr_log_bilevel.csv', type=str)

    # anneal eps (reduce early saturation)
    parser.add_argument('--anneal_eps', action='store_true')
    parser.add_argument('--eps_min', default=None, type=float)
    parser.add_argument('--eps_anneal_pow', default=1.2, type=float)

    parser.add_argument('--early_stop', action='store_true')
    parser.add_argument('--early_stop_patience', default=80, type=int)
    parser.add_argument('--early_stop_min_delta', default=0.003, type=float)

    parser.add_argument('--curriculum', action='store_true', help='ramp stealth weights over training')

    parser.add_argument('--lambda_sat_mean', default=0.3, type=float, help='weight for mean |delta|/eps penalty to lower overall saturation')


    args = parser.parse_args()

    # scale eps / set step size
    if args.constraint == 'Linf':
        args.eps = args.eps / 255.0
        if args.step_size is None: args.step_size = args.eps / 6.0
    else:
        if args.step_size is None: args.step_size = 1e-2

    if args.anneal_eps:
        if args.eps_min is None: args.eps_min = 0.6 * args.eps
    else:
        args.eps_min = args.eps

    args.moo_path = f"{args.moo_path}-{args.dataset}-{args.arch}-{args.constraint}-eps{args.eps*255:.0f}"
    os.makedirs(args.moo_path, exist_ok=True)

    pprint(vars(args))
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpuid)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    main(args)
