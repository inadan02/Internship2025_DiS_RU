# #!/usr/bin/env python3
# import os
# import csv
# import argparse
# from pprint import pprint
# from itertools import cycle
# import random
# import math

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
#     # penalty on high-frequency energy of delta (via rFFT magnitude outside central disc)
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


# # ---------------- PCGrad (kept for compatibility) ----------------

# def pcgrad_sum(grads, shuffle=True):
#     order = list(range(len(grads)))
#     if shuffle:
#         random.shuffle(order)
#     gi_list = [g.clone() for g in grads]
#     for i_idx in range(len(order)):
#         i = order[i_idx]
#         gi = gi_list[i]
#         for j_idx in range(i_idx):
#             j = order[j_idx]
#             gj = gi_list[j]
#             denom = gj.dot(gj) + 1e-12
#             dot_ij = gi.dot(gj)
#             if dot_ij < 0:
#                 gi = gi - (dot_ij / denom) * gj
#         gi_list[i] = gi
#     return torch.stack(gi_list, dim=0).sum(dim=0)


# # ---------------- MGDA helper: simplex projection + small QP solver ----------------

# def _proj_simplex(v: torch.Tensor) -> torch.Tensor:
#     # Euclidean projection to probability simplex
#     if v.numel() == 1:
#         return torch.tensor([1.0], device=v.device)
#     u, _ = torch.sort(v, descending=True)
#     cssv = torch.cumsum(u, dim=0) - 1.0
#     ind = torch.arange(1, v.numel() + 1, device=v.device)
#     cond = u - cssv / ind > 0
#     if not cond.any():
#         return torch.ones_like(v) / v.numel()
#     rho = torch.max(ind[cond]) - 1  # 0-based
#     theta = cssv[rho] / (rho + 1.0)
#     w = torch.clamp(v - theta, min=0.0)
#     w = w / (w.sum() + 1e-12)
#     return w

# def solve_mgda_coeffs(grads_list, iters=40, lr=0.3, device='cpu'):
#     """
#     grads_list: list of 1D tensors (flattened gradients) length M
#     Solve: minimize || G alpha ||^2  s.t. alpha >= 0, sum alpha = 1
#     using projected gradient on alpha. Returns alpha (M,)
#     """
#     M = len(grads_list)
#     if M == 1:
#         return torch.ones(1, device=device)
#     G = torch.stack([g.view(-1).to(device) for g in grads_list], dim=0)  # M x D
#     Gram = G @ G.t()  # M x M
#     alpha = torch.ones(M, device=device) / float(M)
#     for _ in range(iters):
#         grad_alpha = 2.0 * (Gram @ alpha)
#         alpha = alpha - lr * grad_alpha
#         alpha = _proj_simplex(alpha)
#     return alpha


# # ---------------- Refined-Partitioning (RP) utilities ----------------

# def build_rp_masks(delta: torch.Tensor, scheme: str, tiles: int = 2, freq_split: float = 0.5):
#     """
#     Returns a list of binary masks (same shape as delta) that partition delta.
#     scheme: 'channels' | 'tiles' | 'frequency'
#     - channels: one mask per channel
#     - tiles: grid tiles (tiles x tiles)
#     - frequency: two masks (low ring vs high ring) in image space proxy
#     """
#     B, C, H, W = delta.shape
#     device = delta.device
#     masks = []

#     if scheme == 'channels':
#         for c in range(C):
#             m = torch.zeros_like(delta)
#             m[:, c:c+1, :, :] = 1.0
#             masks.append(m)

#     elif scheme == 'tiles':
#         th = max(1, H // tiles)
#         tw = max(1, W // tiles)
#         for ty in range(tiles):
#             for tx in range(tiles):
#                 y0, y1 = ty * th, (H if ty == tiles - 1 else (ty + 1) * th)
#                 x0, x1 = tx * tw, (W if tx == tiles - 1 else (tx + 1) * tw)
#                 m = torch.zeros_like(delta)
#                 m[:, :, y0:y1, x0:x1] = 1.0
#                 masks.append(m)

#     elif scheme == 'frequency':
#         yy = torch.linspace(-1, 1, steps=H, device=device).view(H,1).repeat(1,W)
#         xx = torch.linspace(-1, 1, steps=W, device=device).view(1,W).repeat(H,1)
#         r = torch.sqrt(xx**2 + yy**2)
#         r = (r - r.min()) / (r.max() - r.min() + 1e-12)
#         low_mask_2d = (r <= freq_split).float()
#         high_mask_2d = 1.0 - low_mask_2d
#         for mask2d in [low_mask_2d, high_mask_2d]:
#             m = torch.zeros_like(delta)
#             m[:] = mask2d  # broadcast over channels
#             masks.append(m)

#     else:
#         masks.append(torch.ones_like(delta))

#     return masks

# def masked_grad(param: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
#     """Extract per-parameter gradient restricted to `mask` (flattened)."""
#     if param.grad is None:
#         return torch.zeros_like(param).reshape(-1)
#     return (param.grad * mask).reshape(-1)

# def rp_mgda_step(delta: torch.Tensor, masks, loss_list, opt, args):
#     """
#     One RP-MGDA step:
#     - For each partition mask m:
#         * compute per-loss gradients wrt delta under mask m
#         * solve MGDA alphas (min-norm combination in loss-gradient space)
#         * form masked combined gradient
#     - Sum masked gradients from all partitions and step once.
#     Returns gnorm (||g_total||_2)
#     """
#     opt.zero_grad(set_to_none=True)
#     part_grads = []
#     for m in masks:
#         per_loss_flat = []
#         for L in loss_list:
#             opt.zero_grad(set_to_none=True)
#             L.backward(retain_graph=True)
#             per_loss_flat.append(masked_grad(delta, m).detach().clone())

#         alpha = solve_mgda_coeffs(per_loss_flat, iters=args.mgda_iters, lr=args.mgda_lr, device=delta.device)

#         g_sum_flat = None
#         for a, g in zip(alpha, per_loss_flat):
#             g_sum_flat = g if g_sum_flat is None else (g_sum_flat + a * g)
#         g_part = g_sum_flat.view_as(delta) * m
#         part_grads.append(g_part)

#     with torch.no_grad():
#         g_total = torch.stack(part_grads, dim=0).sum(dim=0)
#     delta.grad = g_total.detach()
#     opt.step()
#     return float(g_total.view(-1).norm().item())


# # -------------- Attack core (uniform) --------------

# def universal_target_attack_moo(model, dataset_loader, target_class, args, log_rows):
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#     data_loader = DataLoader(dataset_loader.dataset,
#                              batch_size=args.batch_size,
#                              shuffle=True,
#                              pin_memory=True,
#                              num_workers=min(4, os.cpu_count() or 1))

#     # small fixed validation slice for stable logging (optional)
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

#     def one_restart(restart_idx):
#         delta = (torch.zeros(1, *args.data_shape, device=device)
#                  .uniform_(-1e-6, 1e-6)
#                  .requires_grad_(True))

#         opt = torch.optim.Adam([delta], lr=args.step_size)

#         if args.lr_schedule == "cosine":
#             scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.num_steps)
#         elif args.lr_schedule == "constant":
#             scheduler = None
#         else:
#             raise ValueError("--lr_schedule must be 'cosine' or 'constant'")

#         ema_asr, ema_spa, ema_spec, ema_clean = EMA(0.9), EMA(0.9), EMA(0.9), EMA(0.9)

#         best_delta = delta.detach().clone()
#         best_score = -1.0
#         best_step = -1  # for optional early stopping
#         batch_iter = cycle(data_loader)
#         iterator = tqdm(range(args.num_steps), total=args.num_steps, leave=False)

#         # build refined partitions for δ (once per restart)
#         rp_masks = None
#         if getattr(args, 'rp_mgda', False):
#             rp_masks = build_rp_masks(delta, scheme=args.rp_scheme, tiles=args.rp_tiles, freq_split=args.rp_freq_split)

#         # optional tiny FGSM warm-start (gentle to avoid early saturation)
#         def fgsm_kick(inp, y):
#             delta_ws = delta.detach().clone().requires_grad_(True)
#             x_ws = (inp + delta_ws).clamp(0, 1)
#             logits_ws = model(x_ws)
#             loss_ws = F.cross_entropy(logits_ws, y)
#             g_delta = torch.autograd.grad(loss_ws, delta_ws)[0]
#             with torch.no_grad():
#                 step = args.eps * 0.0625  # eps/16 per kick
#                 delta.add_(step * g_delta.sign())
#                 delta.copy_(project_delta(delta, args.eps, args.constraint))

#         if args.fgsm_warmstart > 0:
#             try:
#                 inp0, y0 = next(batch_iter)
#             except StopIteration:
#                 inp0, y0 = None, None
#             if inp0 is not None:
#                 inp0 = inp0.to(device)
#                 y0 = y0.to(device)
#                 tgt0 = torch.full((inp0.size(0),), int(target_class), device=device, dtype=torch.long)
#                 for _ in range(args.fgsm_warmstart):
#                     fgsm_kick(inp0, tgt0)

#         # Early stopping trackers (optional)
#         patience = args.early_stop_patience
#         min_delta = args.early_stop_min_delta
#         wait = 0

#         for step_index in iterator:
#             inp, y = next(batch_iter)
#             inp = inp.to(device, non_blocking=True)
#             y = y.to(device, non_blocking=True)

#             # --- Exclude samples that are already the target class
#             mask = (y != target_class)
#             if (~mask).all():
#                 continue
#             inp = inp[mask]
#             y = y[mask]
#             tgt = torch.full((inp.size(0),), int(target_class), device=device, dtype=torch.long)

#             # EOT + optional input diversity (DI)
#             def eot_apply(x, repeats=args.eot_samples):
#                 outs = []
#                 for _r in range(repeats):
#                     z = x
#                     if args.di_rate > 0.0 and torch.rand(1).item() < args.di_rate:
#                         s = 1.0 + (2.0 * (torch.rand(1).item() - 0.5) * args.di_scale)
#                         h = x.shape[2]; w = x.shape[3]
#                         nh = max(1, int(round(h * s))); nw = max(1, int(round(w * s)))
#                         z = F.interpolate(x, size=(nh, nw), mode='bilinear', align_corners=False)
#                         if nh >= h and nw >= w:
#                             top = (nh - h) // 2; left = (nw - w) // 2
#                             z = z[:, :, top:top + h, left:left + w]
#                         else:
#                             pad_h = max(0, h - nh); pad_w = max(0, w - nw)
#                             pad_left = pad_w // 2; pad_top = pad_h // 2
#                             z = F.pad(z, (pad_left, pad_w - pad_left, pad_top, pad_h - pad_top), value=0.0)
#                     z = z + delta
#                     if args.eot_noise > 0:
#                         z = z + args.eot_noise * torch.randn_like(z)
#                     z = torch.clamp(z, 0, 1)
#                     outs.append(z)
#                 return torch.cat(outs, dim=0)

#             x_adv = eot_apply(inp, repeats=args.eot_samples)
#             tgt_eot = tgt.repeat(args.eot_samples)

#             model.eval()
#             logits = model(x_adv)

#             # ASR loss warm-up and kappa ramp
#             if step_index < args.asr_warmup:
#                 L_asr_eff = F.cross_entropy(logits, tgt_eot)
#             else:
#                 if args.kappa_max > 0:
#                     ramp_steps = max(1, int(args.kappa_ramp_frac * args.num_steps))
#                     ramp_pos = max(0, step_index - args.asr_warmup)
#                     alpha_k = min(1.0, ramp_pos / ramp_steps) if ramp_steps > 0 else 1.0
#                     kappa_eff = alpha_k * args.kappa_max
#                 else:
#                     kappa_eff = args.kappa
#                 L_asr_eff = targeted_margin_loss(logits, tgt_eot, kappa=kappa_eff) if args.asr_loss == 'cw' \
#                            else F.cross_entropy(logits, tgt_eot)

#             # Regularizers
#             L_spatial = total_variation(delta)
#             L_spectral = fft_high_energy(delta, frac=args.fft_frac)
#             L_clean = delta.pow(2).mean()

#             # Hinge clipping penalty (discourage boundary saturation)
#             if args.lambda_clip > 0.0:
#                 if args.constraint == 'Linf':
#                     over = (delta.abs() - args.eps).clamp(min=0.0)
#                     L_clip = (over.pow(2)).mean()
#                 else:
#                     dnorm = delta.view(delta.size(0), -1).norm(p=2, dim=1)
#                     over = (dnorm - args.eps).clamp(min=0.0)
#                     L_clip = (over.pow(2)).mean()
#             else:
#                 L_clip = torch.tensor(0.0, device=delta.device)

#             # EMA updates
#             for ema, L in [(ema_asr, L_asr_eff), (ema_spa, L_spatial),
#                            (ema_spec, L_spectral), (ema_clean, L_clean)]:
#                 ema.update(L)

#             # Curriculum (ASR strong early; smoothness ramps up)
#             if args.curriculum:
#                 alpha = float(step_index) / float(max(1, args.num_steps - 1))
#                 w_asr = 0.85 - 0.20 * alpha   # 0.85 -> 0.65
#                 w_tv  = 0.05 + 0.10 * alpha   # 0.05 -> 0.15
#                 w_sp  = 0.03 + 0.07 * alpha   # 0.03 -> 0.10
#                 w_l2  = 0.02 + 0.10 * alpha   # 0.02 -> 0.12
#             else:
#                 w_asr, w_tv, w_sp, w_l2 = 0.70, 0.10, 0.10, 0.10

#             L_list = [
#                 w_asr * (L_asr_eff  / (ema_asr.value   + 1e-8)),
#                 w_tv  * (L_spatial  / (ema_spa.value   + 1e-8)),
#                 w_sp  * (L_spectral / (ema_spec.value  + 1e-8)),
#                 w_l2  * (L_clean    / (ema_clean.value + 1e-8)),
#             ]
#             if args.lambda_clip > 0.0:
#                 L_list.append(args.lambda_clip * L_clip)

#             # ---------------- OPT step (RP-MGDA > MGDA > PCGrad > plain sum)

#             use_rp_now   = bool(getattr(args, 'rp_mgda', False)) and (step_index >= args.asr_warmup)
#             use_mgda_now = (not use_rp_now) and bool(getattr(args, 'mgda', False)) and (step_index >= args.asr_warmup)
#             use_pcgrad_now = (not use_rp_now) and (not use_mgda_now) and bool(args.pcgrad) and (step_index >= args.asr_warmup)

#             if use_rp_now and rp_masks is not None:
#                 gnorm = rp_mgda_step(delta, rp_masks, L_list, opt, args)

#             elif use_mgda_now:
#                 opt.zero_grad(set_to_none=True)
#                 grads = []
#                 for L in L_list:
#                     g = torch.autograd.grad(L, delta, retain_graph=True, create_graph=False)[0]
#                     grads.append(g.view(-1).detach().clone())
#                 alpha_m = solve_mgda_coeffs(grads, iters=args.mgda_iters, lr=args.mgda_lr, device=delta.device)
#                 g_sum = None
#                 for a, g in zip(alpha_m, grads):
#                     g_sum = g if g_sum is None else (g_sum + a * g)
#                 delta.grad = g_sum.view_as(delta).detach()
#                 gnorm = float(delta.grad.view(-1).norm().item()) if delta.grad is not None else 0.0
#                 opt.step()

#             elif use_pcgrad_now:
#                 opt.zero_grad(set_to_none=True)
#                 grads = []
#                 for L in L_list:
#                     g = torch.autograd.grad(L, delta, retain_graph=True, create_graph=False)[0]
#                     grads.append(g.view(-1))
#                 g_sum = pcgrad_sum(grads)
#                 gnorm = float(g_sum.norm().item())
#                 delta.grad = g_sum.view_as(delta).detach()
#                 opt.step()

#             else:
#                 opt.zero_grad(set_to_none=True)
#                 total_loss = sum(L_list)
#                 total_loss.backward()
#                 gnorm = float(delta.grad.view(-1).norm().item()) if delta.grad is not None else 0.0
#                 opt.step()

#             # Scheduler / warmup + optional mid-run anneal
#             if scheduler is not None:
#                 scheduler.step()
#             else:
#                 if args.warmup > 0 and step_index < args.warmup:
#                     warm_lr = args.step_size * float(step_index + 1) / float(args.warmup)
#                     for pg in opt.param_groups:
#                         pg['lr'] = warm_lr
#                 else:
#                     if args.anneal_mid and step_index >= (args.num_steps // 2):
#                         for pg in opt.param_groups:
#                             pg['lr'] = args.step_size * args.anneal_factor
#                     else:
#                         for pg in opt.param_groups:
#                             pg['lr'] = args.step_size

#             # Project into norm ball
#             with torch.no_grad():
#                 delta.copy_(project_delta(delta, args.eps, args.constraint))

#             # Metrics on this batch
#             with torch.no_grad():
#                 preds = logits.argmax(dim=1)
#                 asr_batch = float((preds == tgt_eot).float().mean().item())
#                 m = logits_margin_stats(logits, tgt_eot).mean().item()
#                 if args.constraint == 'Linf':
#                     sat = float((delta.abs() >= (args.eps - 1e-6)).float().mean().item())
#                 else:
#                     d_norm = delta.view(delta.size(0), -1).norm(p=2, dim=1)
#                     sat = float((d_norm / (args.eps + 1e-12)).clamp(max=1.0).mean().item())

#                 if val_x is not None:
#                     asr_val = (model((val_x + delta).clamp(0, 1)).argmax(1) == val_tgt).float().mean().item()
#                 else:
#                     asr_val = asr_batch

#                 # keep best + optional early stopping
#                 improved = (asr_val - best_score) > min_delta
#                 if improved:
#                     best_score = asr_val
#                     best_delta = delta.detach().clone()
#                     best_step = step_index
#                     wait = 0
#                 else:
#                     wait += 1
#                     if args.early_stop and wait >= patience and best_step >= 0:
#                         print(f"[T{target_class}] Early stop at {step_index+1} (best @ {best_step+1})")
#                         break

#             if (step_index + 1) % args.print_every == 0:
#                 iterator.set_description(
#                     f"[T{target_class}] step {step_index+1}/{args.num_steps} | "
#                     f"ASR(batch) {100*asr_batch:5.2f}% | ASR(val) {100*asr_val:5.2f}% | "
#                     f"sat {100*sat:4.1f}% | margin {m:.4f} | gnorm {gnorm:.3e}"
#                 )

#             # CSV logging
#             if args.log_csv is not None:
#                 log_rows.append({
#                     "restart": restart_idx,
#                     "target_class": target_class,
#                     "step": step_index + 1,
#                     "asr_batch": asr_batch,
#                     "asr_val": asr_val,
#                     "sat": sat,
#                     "margin": m,
#                     "gnorm": gnorm,
#                 })

#         return best_delta, best_score

#     best_overall, best_score = None, -1.0
#     for r in range(args.restarts):
#         d, s = one_restart(restart_idx=r)
#         if s > best_score:
#             best_score, best_overall = s, d

#     return best_overall.detach().requires_grad_(False)


# def moo_generate(args, loader, model):
#     log_rows = []
#     poisons = []
#     for i in range(args.num_classes):
#         poison = universal_target_attack_moo(model, loader, i, args, log_rows)
#         poisons.append(poison.squeeze())

#     if args.log_csv is not None and len(log_rows) > 0:
#         os.makedirs(os.path.dirname(args.log_csv), exist_ok=True)
#         with open(args.log_csv, "w", newline="") as f:
#             fieldnames = ["restart", "target_class", "step", "asr_batch", "asr_val", "sat", "margin", "gnorm"]
#             writer = csv.DictWriter(f, fieldnames=fieldnames)
#             writer.writeheader()
#             writer.writerows(log_rows)

#     return poisons


# # ---------------- Main ----------------

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

#     data_loader = DataLoader(data_set, batch_size=args.batch_size, num_workers=min(8, os.cpu_count() or 1), shuffle=True,  pin_memory=True)
#     _ = DataLoader(test_set,  batch_size=args.batch_size, num_workers=min(8, os.cpu_count() or 1), shuffle=False, pin_memory=True)

#     model = make_and_restore_model(args, resume_path=args.model_path)
#     model.eval()

#     set_seed(args.seed)

#     poisons = moo_generate(args, data_loader, model)
#     os.makedirs(args.moo_path, exist_ok=True)
#     for i, d in enumerate(poisons):
#         torch.save({'delta': d, 'args': vars(args)}, os.path.join(args.moo_path, f'moo_delta_{i}.pth'))


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser("Uniform MOO for universal targeted perturbations (RP-MGDA)")

#     # Repro & device
#     parser.add_argument('--seed', default=0, type=int)
#     parser.add_argument('--gpuid', default=0, type=int)

#     # Attack / constraint
#     parser.add_argument('--constraint', default='Linf', choices=['Linf', 'L2'])
#     parser.add_argument('--eps', default=8.0, type=float)         # Linf in pixels; converted below
#     parser.add_argument('--num_steps', default=1000, type=int)
#     parser.add_argument('--step_size', default=None, type=float)  # if None, set below
#     parser.add_argument('--restarts', default=6, type=int)
#     parser.add_argument('--kappa', default=0.0, type=float)
#     parser.add_argument('--kappa_max', default=0.5, type=float, help="max kappa for ramp")
#     parser.add_argument('--kappa_ramp_frac', default=0.4, type=float, help="fraction of num_steps used for kappa ramp")
#     parser.add_argument('--eot_samples', default=2, type=int)
#     parser.add_argument('--eot_noise', default=0.001, type=float)
#     parser.add_argument('--fft_frac', default=0.5, type=float)

#     # Model / data
#     parser.add_argument('--arch', default='ResNet18', choices=['VGG16','EfficientNetB0','DenseNet121','ResNet18','swin','inception_next_tiny','inception_next_small'])
#     parser.add_argument('--model_path', default='results/clean_model_weight/checkpoint.pth')
#     parser.add_argument('--dataset', default='cifar10', choices=['cifar10','imagenet200','gtsrb'])
#     parser.add_argument('--data_root', default='../data', type=str)

#     # IO
#     parser.add_argument('--batch_size', default=256, type=int)
#     parser.add_argument('--moo_path', default='./results/moo', type=str)
#     parser.add_argument('--log_csv', default=None, type=str, help="If set, save per-step metrics to CSV")

#     # Loss / curriculum / PCGrad / MGDA
#     parser.add_argument('--asr_loss', default='cw', choices=['cw','ce'])
#     parser.add_argument('--curriculum', action='store_true')
#     parser.add_argument('--pcgrad', action='store_true')
#     parser.add_argument('--mgda', action='store_true', help="global MGDA (min-norm) over losses")
#     parser.add_argument('--mgda_iters', default=40, type=int, help="projected-GD iterations for MGDA alpha solve")
#     parser.add_argument('--mgda_lr', default=0.3, type=float, help="step size for MGDA alpha solver")

#     # RP-MGDA
#     parser.add_argument('--rp_mgda', action='store_true',
#                         help="Enable Refined-Partitioning MGDA (takes precedence over --mgda/--pcgrad)")
#     parser.add_argument('--rp_scheme', default='channels', choices=['channels','tiles','frequency'],
#                         help="Partitioning of δ for RP-MGDA")
#     parser.add_argument('--rp_tiles', default=2, type=int, help="Grid size for --rp_scheme tiles")
#     parser.add_argument('--rp_freq_split', default=0.5, type=float, help="Low/High split for --rp_scheme frequency (0..1)")

#     # Scheduler / warmup / logging
#     parser.add_argument('--lr_schedule', default='constant', choices=['cosine','constant'])
#     parser.add_argument('--warmup', default=100, type=int)
#     parser.add_argument('--asr_warmup', default=300, type=int)
#     parser.add_argument('--fgsm_warmstart', default=2, type=int)
#     parser.add_argument('--di_rate', default=0.20, type=float, help="probability of applying DI on each EOT sample")
#     parser.add_argument('--di_scale', default=0.08, type=float, help="scale variation for DI (fraction)")
#     parser.add_argument('--print_every', default=50, type=int)

#     # Stealth / saturation control
#     parser.add_argument('--lambda_clip', default=0.30, type=float, help="hinge clip penalty weight (discourage boundary saturation)")

#     # Anneal for fine-tuning
#     parser.add_argument('--anneal_mid', action='store_true', help="anneal lr in mid-run for fine-tuning")
#     parser.add_argument('--anneal_factor', default=0.25, type=float, help="factor to multiply lr by when annealing at mid-run")

#     # Early stopping (optional)
#     parser.add_argument('--early_stop', action='store_true', help="enable ASR(val) patience-based early stop")
#     parser.add_argument('--early_stop_patience', default=120, type=int, help="stop if no ASR(val) improvement for this many steps")
#     parser.add_argument('--early_stop_min_delta', default=0.002, type=float, help="minimum ASR(val) improvement to reset patience")

#     args = parser.parse_args()

#     # Output dir
#     args.moo_path = f"{args.moo_path}-{args.dataset}-{args.arch}-{args.constraint}-eps{args.eps:.1f}"
#     os.makedirs(args.moo_path, exist_ok=True)

#     # Normalize eps/step_size units
#     if args.constraint == 'Linf':
#         args.eps = args.eps / 255.0
#         if args.step_size is None:
#             # default that favors stealth over speed
#             args.step_size = args.eps / 4.0
#     else:
#         if args.step_size is None:
#             args.step_size = 1e-2

#     pprint(vars(args))

#     os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpuid)
#     torch.backends.cudnn.deterministic = False
#     torch.backends.cudnn.benchmark = True

#     main(args)


#!/usr/bin/env python3
import os
import csv
import argparse
from pprint import pprint
from itertools import cycle
import random
import math
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import torchvision
from torchvision import datasets, transforms

# your utils must add the Normalize wrapper inside make_and_restore_model
from utils import set_seed, make_and_restore_model


# ---------------- Loss helpers ----------------

def targeted_margin_loss(logits: torch.Tensor, y_tgt: torch.Tensor, kappa: float = 0.0) -> torch.Tensor:
    C = logits.size(1)
    one_hot = F.one_hot(y_tgt, num_classes=C).bool()
    target_logit = logits[one_hot]
    other_logit = logits.masked_fill(one_hot, float('-inf')).amax(dim=1)
    return torch.clamp(other_logit - target_logit + kappa, min=0).mean()

def logits_margin_stats(logits: torch.Tensor, y_tgt: torch.Tensor):
    C = logits.size(1)
    one_hot = F.one_hot(y_tgt, num_classes=C).bool()
    target_logit = logits[one_hot]
    other_logit = logits.masked_fill(one_hot, float('-inf')).amax(dim=1)
    return target_logit - other_logit

def total_variation(x: torch.Tensor) -> torch.Tensor:
    tv_h = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean()
    tv_w = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()
    return tv_h + tv_w

def fft_high_energy(x: torch.Tensor, frac: float = 0.5) -> torch.Tensor:
    # penalty on high-frequency energy of delta (via rFFT magnitude outside central disc)
    xc = x - x.mean(dim=(2, 3), keepdim=True)
    X = torch.fft.rfft2(xc, norm="ortho")
    mag2 = (X.real ** 2 + X.imag ** 2)
    B, C, H, W2 = mag2.shape
    yy = torch.linspace(-1, 1, steps=H, device=x.device).view(H, 1).repeat(1, W2)
    xx = torch.linspace(0, 1, steps=W2, device=x.device).view(1, W2).repeat(H, 1)
    r = torch.sqrt(xx**2 + yy**2)
    keep = (r <= frac * r.max()).float()
    high = (1.0 - keep) * mag2
    return high.mean()

class EMA:
    def __init__(self, beta: float = 0.9):
        self.beta = beta
        self.value = None
    def update(self, x: torch.Tensor):
        v = float(x.detach().item())
        self.value = v if self.value is None else self.beta * self.value + (1 - self.beta) * v


# ---------------- Projections ----------------

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

def project_delta(delta: torch.Tensor, eps: float, constraint: str) -> torch.Tensor:
    return project_linf(delta, eps) if constraint == 'Linf' else project_l2(delta, eps)


# ---------------- PCGrad (kept for compatibility) ----------------

def pcgrad_sum(grads, shuffle=True):
    order = list(range(len(grads)))
    if shuffle:
        random.shuffle(order)
    gi_list = [g.clone() for g in grads]
    for i_idx in range(len(order)):
        i = order[i_idx]
        gi = gi_list[i]
        for j_idx in range(i_idx):
            j = order[j_idx]
            gj = gi_list[j]
            denom = gj.dot(gj) + 1e-12
            dot_ij = gi.dot(gj)
            if dot_ij < 0:
                gi = gi - (dot_ij / denom) * gj
        gi_list[i] = gi
    return torch.stack(gi_list, dim=0).sum(dim=0)


# ---------------- MGDA helper: simplex projection + small QP solver ----------------

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
    w = w / (w.sum() + 1e-12)
    return w

def solve_mgda_coeffs(grads_list, iters=40, lr=0.3, device='cpu'):
    M = len(grads_list)
    if M == 1:
        return torch.ones(1, device=device)
    G = torch.stack([g.view(-1).to(device) for g in grads_list], dim=0)  # M x D
    Gram = G @ G.t()  # M x M
    alpha = torch.ones(M, device=device) / float(M)
    for _ in range(iters):
        grad_alpha = 2.0 * (Gram @ alpha)
        alpha = alpha - lr * grad_alpha
        alpha = _proj_simplex(alpha)
    return alpha


# ---------------- Refined-Partitioning (RP) utilities ----------------

def build_rp_masks(delta: torch.Tensor, scheme: str, tiles: int = 2, freq_split: float = 0.5):
    B, C, H, W = delta.shape
    device = delta.device
    masks = []

    if scheme == 'channels':
        for c in range(C):
            m = torch.zeros_like(delta)
            m[:, c:c+1, :, :] = 1.0
            masks.append(m)

    elif scheme == 'tiles':
        th = max(1, H // tiles)
        tw = max(1, W // tiles)
        for ty in range(tiles):
            for tx in range(tiles):
                y0, y1 = ty * th, (H if ty == tiles - 1 else (ty + 1) * th)
                x0, x1 = tx * tw, (W if tx == tiles - 1 else (tx + 1) * tw)
                m = torch.zeros_like(delta)
                m[:, :, y0:y1, x0:x1] = 1.0
                masks.append(m)

    elif scheme == 'frequency':
        yy = torch.linspace(-1, 1, steps=H, device=device).view(H,1).repeat(1,W)
        xx = torch.linspace(-1, 1, steps=W, device=device).view(1,W).repeat(H,1)
        r = torch.sqrt(xx**2 + yy**2)
        r = (r - r.min()) / (r.max() - r.min() + 1e-12)
        low_mask_2d = (r <= freq_split).float()
        high_mask_2d = 1.0 - low_mask_2d
        for mask2d in [low_mask_2d, high_mask_2d]:
            m = torch.zeros_like(delta)
            m[:] = mask2d
            masks.append(m)

    else:
        masks.append(torch.ones_like(delta))

    return masks

def masked_grad(param: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if param.grad is None:
        return torch.zeros_like(param).reshape(-1)
    return (param.grad * mask).reshape(-1)

def rp_mgda_step(delta: torch.Tensor, masks, loss_list, opt, args):
    opt.zero_grad(set_to_none=True)
    part_grads = []
    # compute per-loss gradients once, reuse to avoid recomputing forward many times
    for m in masks:
        per_loss_flat = []
        for L in loss_list:
            opt.zero_grad(set_to_none=True)
            L.backward(retain_graph=True)
            per_loss_flat.append(masked_grad(delta, m).detach().clone())

        alpha = solve_mgda_coeffs(per_loss_flat, iters=args.mgda_iters, lr=args.mgda_lr, device=delta.device)
        g_sum_flat = None
        for a, g in zip(alpha, per_loss_flat):
            g_sum_flat = g if g_sum_flat is None else (g_sum_flat + a * g)
        g_part = g_sum_flat.view_as(delta) * m
        part_grads.append(g_part)

    with torch.no_grad():
        g_total = torch.stack(part_grads, dim=0).sum(dim=0)
    delta.grad = g_total.detach()
    opt.step()
    return float(g_total.view(-1).norm().item())


# -------------- Attack core (uniform) --------------

def universal_target_attack_moo(model, dataset_loader, target_class, args, log_rows):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_loader = DataLoader(dataset_loader.dataset,
                             batch_size=args.batch_size,
                             shuffle=True,
                             pin_memory=True,
                             num_workers=min(4, os.cpu_count() or 1))

    _iter = iter(data_loader)
    try:
        _fixed = next(_iter)
    except StopIteration:
        _fixed = None
    val_x = None
    val_tgt = None
    if _fixed is not None:
        val_x = _fixed[0][:min(256, _fixed[0].size(0))].to(device)
        val_tgt = torch.full((val_x.size(0),), int(target_class), device=device, dtype=torch.long)

    def one_restart(restart_idx):
        delta = (torch.zeros(1, *args.data_shape, device=device)
                 .uniform_(-1e-6, 1e-6)
                 .requires_grad_(True))

        opt = torch.optim.Adam([delta], lr=args.step_size)

        if args.lr_schedule == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.num_steps)
        elif args.lr_schedule == "constant":
            scheduler = None
        else:
            raise ValueError("--lr_schedule must be 'cosine' or 'constant'")

        ema_asr, ema_spa, ema_spec, ema_clean = EMA(0.9), EMA(0.9), EMA(0.9), EMA(0.9)

        best_delta = delta.detach().clone()
        best_score = -1.0
        best_sat = 1.0
        best_step = -1
        batch_iter = cycle(data_loader)
        iterator = tqdm(range(args.num_steps), total=args.num_steps, leave=False)

        # build refined partitions for δ
        rp_masks = None
        if getattr(args, 'rp_mgda', False):
            rp_masks = build_rp_masks(delta, scheme=args.rp_scheme, tiles=args.rp_tiles, freq_split=args.rp_freq_split)

        # Augmented-Lagrangian inspired controller state
        tau = args.asr_floor  # desired ASR floor
        mu = args.mu_init
        rho = args.rho_init

        # tiny FGSM warm-start
        def fgsm_kick(inp, y):
            delta_ws = delta.detach().clone().requires_grad_(True)
            x_ws = (inp + delta_ws).clamp(0, 1)
            logits_ws = model(x_ws)
            loss_ws = F.cross_entropy(logits_ws, y)
            g_delta = torch.autograd.grad(loss_ws, delta_ws)[0]
            with torch.no_grad():
                step = args.eps * 0.0625
                delta.add_(step * g_delta.sign())
                delta.copy_(project_delta(delta, args.eps, args.constraint))

        if args.fgsm_warmstart > 0:
            try:
                inp0, y0 = next(batch_iter)
            except StopIteration:
                inp0, y0 = None, None
            if inp0 is not None:
                inp0 = inp0.to(device)
                y0 = y0.to(device)
                tgt0 = torch.full((inp0.size(0),), int(target_class), device=device, dtype=torch.long)
                for _ in range(args.fgsm_warmstart):
                    fgsm_kick(inp0, tgt0)

        # early stopping trackers
        patience = args.early_stop_patience
        min_delta = args.early_stop_min_delta
        wait = 0

        for step_index in iterator:
            inp, y = next(batch_iter)
            inp = inp.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            mask_idx = (y != target_class)
            if (~mask_idx).all():
                continue
            inp = inp[mask_idx]
            y = y[mask_idx]
            tgt = torch.full((inp.size(0),), int(target_class), device=device, dtype=torch.long)

            # EOT + DI
            def eot_apply(x, repeats=args.eot_samples):
                outs = []
                for _r in range(repeats):
                    z = x
                    if args.di_rate > 0.0 and torch.rand(1).item() < args.di_rate:
                        s = 1.0 + (2.0 * (torch.rand(1).item() - 0.5) * args.di_scale)
                        h = x.shape[2]; w = x.shape[3]
                        nh = max(1, int(round(h * s))); nw = max(1, int(round(w * s)))
                        z = F.interpolate(x, size=(nh, nw), mode='bilinear', align_corners=False)
                        if nh >= h and nw >= w:
                            top = (nh - h) // 2; left = (nw - w) // 2
                            z = z[:, :, top:top + h, left:left + w]
                        else:
                            pad_h = max(0, h - nh); pad_w = max(0, w - nw)
                            pad_left = pad_w // 2; pad_top = pad_h // 2
                            z = F.pad(z, (pad_left, pad_w - pad_left, pad_top, pad_h - pad_top), value=0.0)
                    z = z + delta
                    if args.eot_noise > 0:
                        z = z + args.eot_noise * torch.randn_like(z)
                    z = torch.clamp(z, 0, 1)
                    outs.append(z)
                return torch.cat(outs, dim=0)

            x_adv = eot_apply(inp, repeats=args.eot_samples)
            tgt_eot = tgt.repeat(args.eot_samples)

            model.eval()
            logits = model(x_adv)

            # ASR loss (warmup + kappa ramp)
            if step_index < args.asr_warmup:
                L_asr_eff = F.cross_entropy(logits, tgt_eot)
            else:
                if args.kappa_max > 0:
                    ramp_steps = max(1, int(args.kappa_ramp_frac * args.num_steps))
                    ramp_pos = max(0, step_index - args.asr_warmup)
                    alpha_k = min(1.0, ramp_pos / ramp_steps) if ramp_steps > 0 else 1.0
                    kappa_eff = alpha_k * args.kappa_max
                else:
                    kappa_eff = args.kappa
                L_asr_eff = targeted_margin_loss(logits, tgt_eot, kappa=kappa_eff) if args.asr_loss == 'cw' \
                           else F.cross_entropy(logits, tgt_eot)

            # Regularizers
            L_spatial = total_variation(delta)
            L_spectral = fft_high_energy(delta, frac=args.fft_frac)
            L_clean = delta.pow(2).mean()

            # Hinge clip penalty (penalize approaching/exceeding eps - sat_margin)
            sat_margin_val = max(0.0, min(0.5, args.sat_margin))
            if args.lambda_clip > 0.0:
                if args.constraint == 'Linf':
                    threshold = args.eps - (args.eps * sat_margin_val)
                    over = (delta.abs() - threshold).clamp(min=0.0)
                    L_clip = (over.pow(2)).mean()
                else:
                    dnorm = delta.view(delta.size(0), -1).norm(p=2, dim=1)
                    threshold = args.eps - (args.eps * sat_margin_val)
                    over = (dnorm - threshold).clamp(min=0.0)
                    L_clip = (over.pow(2)).mean()
            else:
                L_clip = torch.tensor(0.0, device=delta.device)

            # EMA updates for normalization
            for ema, L in [(ema_asr, L_asr_eff), (ema_spa, L_spatial),
                           (ema_spec, L_spectral), (ema_clean, L_clean)]:
                ema.update(L)

            # Curriculum weights (but ASR-floor controller will override when needed)
            if args.curriculum:
                alpha = float(step_index) / float(max(1, args.num_steps - 1))
                w_asr = 0.85 - 0.20 * alpha   # 0.85 -> 0.65
                w_tv  = 0.05 + 0.10 * alpha   # 0.05 -> 0.15
                w_sp  = 0.03 + 0.07 * alpha   # 0.03 -> 0.10
                w_l2  = 0.02 + 0.10 * alpha   # 0.02 -> 0.12
            else:
                w_asr, w_tv, w_sp, w_l2 = 0.70, 0.10, 0.10, 0.10

            # Check current ASR(val) (non-differentiable, used for controller)
            with torch.no_grad():
                if val_x is not None:
                    asr_val = (model((val_x + delta).clamp(0, 1)).argmax(1) == val_tgt).float().mean().item()
                else:
                    preds = logits.argmax(dim=1)
                    asr_val = float((preds == tgt_eot).float().mean().item())

            # ASR-floor controller (practical AL-like behavior):
            # If asr_val < tau: prioritize ASR aggressively; reduce stealth weight
            # If asr_val >= tau: allow stealth tightening and increase clip pressure
            if args.asr_floor > 0.0:
                if asr_val < tau:
                    # push ASR: amplify ASR weight, reduce stealth contributions
                    w_asr_ctrl = min(1.0, w_asr * (1.0 + args.asr_floor_aggressiveness))
                    w_tv *= 0.2
                    w_sp *= 0.2
                    w_l2 *= 0.2
                    # gently increase rho to indicate need for ASR focus
                    rho = min(args.rho_max, rho * args.rho_inc)
                else:
                    # allow stealth tightening: reduce ASR weight a bit, increase stealth multipliers
                    w_asr_ctrl = max(0.5 * w_asr, w_asr * 0.9)
                    w_tv *= 1.2
                    w_sp *= 1.2
                    w_l2 *= 1.2
                    # if ASR comfortably above tau, relax rho / increase mu to encourage stealth
                    rho = max(args.rho_min, rho * args.rho_dec)

            else:
                w_asr_ctrl = w_asr

            # Build normalized and weighted loss list
            L_list = [
                w_asr_ctrl * (L_asr_eff  / (ema_asr.value   + 1e-8)),
                w_tv  * (L_spatial  / (ema_spa.value   + 1e-8)),
                w_sp  * (L_spectral / (ema_spec.value  + 1e-8)),
                w_l2  * (L_clean    / (ema_clean.value + 1e-8)),
            ]
            if args.lambda_clip > 0.0:
                L_list.append(args.lambda_clip * L_clip)

            # OPT step selection
            opt.zero_grad(set_to_none=True)

            use_rp_now   = bool(getattr(args, 'rp_mgda', False)) and (step_index >= args.asr_warmup)
            use_mgda_now = (not use_rp_now) and bool(getattr(args, 'mgda', False)) and (step_index >= args.asr_warmup)
            use_pcgrad_now = (not use_rp_now) and (not use_mgda_now) and bool(args.pcgrad) and (step_index >= args.asr_warmup)

            if use_rp_now and rp_masks is not None:
                # rp_mgda_step expects full losses (torch tensors) in loss_list
                # Ensure we pass actual loss tensors (already are)
                gnorm = rp_mgda_step(delta, rp_masks, L_list, opt, args)

            elif use_mgda_now:
                grads = []
                for L in L_list:
                    g = torch.autograd.grad(L, delta, retain_graph=True, create_graph=False)[0]
                    grads.append(g.view(-1).detach().clone())
                alpha_m = solve_mgda_coeffs(grads, iters=args.mgda_iters, lr=args.mgda_lr, device=delta.device)
                g_sum = None
                for a, g in zip(alpha_m, grads):
                    g_sum = g if g_sum is None else (g_sum + a * g)
                delta.grad = g_sum.view_as(delta).detach()
                gnorm = float(delta.grad.view(-1).norm().item()) if delta.grad is not None else 0.0
                opt.step()

            elif use_pcgrad_now:
                grads = []
                for L in L_list:
                    g = torch.autograd.grad(L, delta, retain_graph=True, create_graph=False)[0]
                    grads.append(g.view(-1))
                g_sum = pcgrad_sum(grads)
                gnorm = float(g_sum.norm().item())
                delta.grad = g_sum.view_as(delta).detach()
                opt.step()

            else:
                total_loss = sum(L_list)
                total_loss.backward()
                gnorm = float(delta.grad.view(-1).norm().item()) if delta.grad is not None else 0.0
                opt.step()

            # Scheduler / warmup + optional mid-run anneal
            if scheduler is not None:
                scheduler.step()
            else:
                if args.warmup > 0 and step_index < args.warmup:
                    warm_lr = args.step_size * float(step_index + 1) / float(args.warmup)
                    for pg in opt.param_groups:
                        pg['lr'] = warm_lr
                else:
                    if args.anneal_mid and step_index >= (args.num_steps // 2):
                        for pg in opt.param_groups:
                            pg['lr'] = args.step_size * args.anneal_factor
                    else:
                        for pg in opt.param_groups:
                            pg['lr'] = args.step_size

            # Project into norm ball
            with torch.no_grad():
                delta.copy_(project_delta(delta, args.eps, args.constraint))

            # Metrics on this batch
            with torch.no_grad():
                preds = logits.argmax(dim=1)
                asr_batch = float((preds == tgt_eot).float().mean().item())
                m = logits_margin_stats(logits, tgt_eot).mean().item()
                if args.constraint == 'Linf':
                    sat = float((delta.abs() >= (args.eps - 1e-6)).float().mean().item())
                else:
                    d_norm = delta.view(delta.size(0), -1).norm(p=2, dim=1)
                    sat = float((d_norm / (args.eps + 1e-12)).clamp(max=1.0).mean().item())

                if val_x is not None:
                    asr_val = (model((val_x + delta).clamp(0, 1)).argmax(1) == val_tgt).float().mean().item()
                else:
                    asr_val = asr_batch

                # Best tracking: ASR primary, SAT secondary
                improved = (asr_val > best_score + 1e-6) or (abs(asr_val - best_score) <= 1e-6 and sat < best_sat - 1e-6)
                if improved:
                    best_score = asr_val
                    best_delta = delta.detach().clone()
                    best_sat = sat
                    best_step = step_index
                    wait = 0
                else:
                    wait += 1
                    if args.early_stop and wait >= patience and best_step >= 0:
                        print(f"[T{target_class}] Early stop at {step_index+1} (best @ {best_step+1})")
                        break

            if (step_index + 1) % args.print_every == 0:
                iterator.set_description(
                    f"[T{target_class}] step {step_index+1}/{args.num_steps} | "
                    f"ASR(batch) {100*asr_batch:5.2f}% | ASR(val) {100*asr_val:5.2f}% | "
                    f"sat {100*sat:4.1f}% | margin {m:.4f} | gnorm {gnorm:.3e}"
                )

            # CSV logging
            if args.log_csv is not None:
                log_rows.append({
                    "restart": restart_idx,
                    "target_class": target_class,
                    "step": step_index + 1,
                    "asr_batch": asr_batch,
                    "asr_val": asr_val,
                    "sat": sat,
                    "margin": m,
                    "gnorm": gnorm,
                    "rho": rho,
                    "mu": mu,
                })

        return best_delta, best_score

    best_overall, best_score = None, -1.0
    for r in range(args.restarts):
        d, s = one_restart(restart_idx=r)
        if s > best_score:
            best_score, best_overall = s, d

    return best_overall.detach().requires_grad_(False)


def moo_generate(args, loader, model):
    log_rows = []
    poisons = []
    for i in range(args.num_classes):
        poison = universal_target_attack_moo(model, loader, i, args, log_rows)
        poisons.append(poison.squeeze())

    if args.log_csv is not None and len(log_rows) > 0:
        os.makedirs(os.path.dirname(args.log_csv), exist_ok=True)
        with open(args.log_csv, "w", newline="") as f:
            fieldnames = ["restart", "target_class", "step", "asr_batch", "asr_val", "sat", "margin", "gnorm", "rho", "mu"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(log_rows)

    return poisons


# ---------------- Main ----------------

def main(args):
    # dataset setup
    if args.dataset == 'imagenet200':
        args.num_classes = 200; args.img_size = 224; args.channel = 3
        args.data_shape = (args.channel, args.img_size, args.img_size)
        transform_train = transforms.Compose([transforms.RandomResizedCrop(args.img_size), transforms.ToTensor()])
        transform_test  = transforms.Compose([transforms.Resize(args.img_size), transforms.CenterCrop(args.img_size), transforms.ToTensor()])
        data_set = torchvision.datasets.ImageFolder(os.path.join(args.data_root, 'imagenet200', 'train'), transform=transform_train)
        test_set = torchvision.datasets.ImageFolder(os.path.join(args.data_root, 'imagenet200', 'val'),   transform=transform_test)

    elif args.dataset == 'cifar10':
        args.num_classes = 10; args.img_size = 32; args.channel = 3
        args.data_shape = (args.channel, args.img_size, args.img_size)
        transform = transforms.Compose([transforms.ToTensor()])
        data_set = datasets.CIFAR10(args.data_root, train=True,  download=True, transform=transform)
        test_set = datasets.CIFAR10(args.data_root, train=False, download=True, transform=transform)

    elif args.dataset == 'gtsrb':
        args.num_classes = 43; args.img_size = 32; args.channel = 3
        args.data_shape = (args.channel, args.img_size, args.img_size)
        transform = transforms.Compose([transforms.Resize(args.img_size), transforms.CenterCrop(args.img_size), transforms.ToTensor()])
        data_set = torchvision.datasets.ImageFolder(os.path.join(args.data_root, 'GTSRB', 'Train'),           transform=transform)
        test_set = torchvision.datasets.ImageFolder(os.path.join(args.data_root, 'GTSRB', 'val4imagefolder'), transform=transform)

    else:
        raise ValueError("Unsupported dataset. Choose from: imagenet200, cifar10, gtsrb")

    data_loader = DataLoader(data_set, batch_size=args.batch_size, num_workers=min(8, os.cpu_count() or 1), shuffle=True,  pin_memory=True)
    _ = DataLoader(test_set,  batch_size=args.batch_size, num_workers=min(8, os.cpu_count() or 1), shuffle=False, pin_memory=True)

    model = make_and_restore_model(args, resume_path=args.model_path)
    model.eval()

    set_seed(args.seed)

    poisons = moo_generate(args, data_loader, model)
    os.makedirs(args.moo_path, exist_ok=True)
    for i, d in enumerate(poisons):
        torch.save({'delta': d, 'args': vars(args)}, os.path.join(args.moo_path, f'moo_delta_{i}.pth'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Uniform MOO for universal targeted perturbations (RP-MGDA + ASR-floor)")

    # Repro & device
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--gpuid', default=0, type=int)

    # Attack / constraint
    parser.add_argument('--constraint', default='Linf', choices=['Linf', 'L2'])
    parser.add_argument('--eps', default=8.0, type=float)         # Linf in pixels; converted below
    parser.add_argument('--num_steps', default=1000, type=int)
    parser.add_argument('--step_size', default=None, type=float)  # if None, set below
    parser.add_argument('--restarts', default=6, type=int)
    parser.add_argument('--kappa', default=0.0, type=float)
    parser.add_argument('--kappa_max', default=0.5, type=float, help="max kappa for ramp")
    parser.add_argument('--kappa_ramp_frac', default=0.4, type=float, help="fraction of num_steps used for kappa ramp")
    parser.add_argument('--eot_samples', default=2, type=int)
    parser.add_argument('--eot_noise', default=0.001, type=float)
    parser.add_argument('--fft_frac', default=0.5, type=float)

    # Model / data
    parser.add_argument('--arch', default='ResNet18', choices=['VGG16','EfficientNetB0','DenseNet121','ResNet18','swin','inception_next_tiny','inception_next_small'])
    parser.add_argument('--model_path', default='results/clean_model_weight/checkpoint.pth')
    parser.add_argument('--dataset', default='cifar10', choices=['cifar10','imagenet200','gtsrb'])
    parser.add_argument('--data_root', default='../data', type=str)

    # IO
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--moo_path', default='./results/moo', type=str)
    parser.add_argument('--log_csv', default=None, type=str, help="If set, save per-step metrics to CSV")

    # Loss / curriculum / PCGrad / MGDA
    parser.add_argument('--asr_loss', default='cw', choices=['cw','ce'])
    parser.add_argument('--curriculum', action='store_true')
    parser.add_argument('--pcgrad', action='store_true')
    parser.add_argument('--mgda', action='store_true', help="global MGDA (min-norm) over losses")
    parser.add_argument('--mgda_iters', default=40, type=int, help="projected-GD iterations for MGDA alpha solve")
    parser.add_argument('--mgda_lr', default=0.3, type=float, help="step size for MGDA alpha solver")

    # RP-MGDA
    parser.add_argument('--rp_mgda', action='store_true', help="Enable Refined-Partitioning MGDA")
    parser.add_argument('--rp_scheme', default='frequency', choices=['channels','tiles','frequency'], help="Partitioning of δ for RP-MGDA")
    parser.add_argument('--rp_tiles', default=2, type=int, help="Grid size for --rp_scheme tiles")
    parser.add_argument('--rp_freq_split', default=0.4, type=float, help="Low/High split for --rp_scheme frequency (0..1)")

    # Scheduler / warmup / logging
    parser.add_argument('--lr_schedule', default='cosine', choices=['cosine','constant'])
    parser.add_argument('--warmup', default=25, type=int)
    parser.add_argument('--asr_warmup', default=200, type=int)
    parser.add_argument('--fgsm_warmstart', default=4, type=int)
    parser.add_argument('--di_rate', default=0.20, type=float, help="probability of applying DI on each EOT sample")
    parser.add_argument('--di_scale', default=0.08, type=float, help="scale variation for DI (fraction)")
    parser.add_argument('--print_every', default=25, type=int)

    # Stealth / saturation control
    parser.add_argument('--lambda_clip', default=0.7, type=float, help="hinge clip penalty weight (discourage boundary saturation)")
    parser.add_argument('--sat_margin', default=0.05, type=float, help="fraction of eps reserved before clip triggers (0..0.5)")

    # Anneal for fine-tuning
    parser.add_argument('--anneal_mid', action='store_true', help="anneal lr in mid-run for fine-tuning")
    parser.add_argument('--anneal_factor', default=0.2, type=float, help="factor to multiply lr by when annealing at mid-run")

    # Early stopping (optional)
    parser.add_argument('--early_stop', action='store_true', help="enable ASR(val) patience-based early stop")
    parser.add_argument('--early_stop_patience', default=80, type=int, help="stop if no ASR(val) improvement for this many steps")
    parser.add_argument('--early_stop_min_delta', default=0.003, type=float, help="minimum ASR(val) improvement to reset patience")

    # ASR-floor controller (practical AL-like)
    parser.add_argument('--asr_floor', default=0.70, type=float, help="desired ASR floor (0..1). controller will prioritize ASR under this.")
    parser.add_argument('--asr_floor_aggressiveness', default=0.6, type=float, help="how strong to amplify ASR weight when below tau")
    parser.add_argument('--mu_init', default=0.0, type=float, help="AL-like mu initial (placeholder, logged)")
    parser.add_argument('--rho_init', default=1.0, type=float, help="AL-like rho initial (placeholder, controller uses it heuristically)")
    parser.add_argument('--rho_max', default=100.0, type=float, help="max rho")
    parser.add_argument('--rho_min', default=0.1, type=float, help="min rho")
    parser.add_argument('--rho_inc', default=1.5, type=float, help="rho multiplier when ASR < tau")
    parser.add_argument('--rho_dec', default=0.9, type=float, help="rho multiplier when ASR >= tau")

    args = parser.parse_args()

    # Output dir
    args.moo_path = f"{args.moo_path}-{args.dataset}-{args.arch}-{args.constraint}-eps{args.eps:.1f}"
    os.makedirs(args.moo_path, exist_ok=True)

    # Normalize eps/step_size units
    if args.constraint == 'Linf':
        args.eps = args.eps / 255.0
        if args.step_size is None:
            args.step_size = args.eps / 6.0   # smaller default steps for stealth
    else:
        if args.step_size is None:
            args.step_size = 1e-2

    pprint(vars(args))

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpuid)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    main(args)
