# # import os
# # import argparse
# # from pprint import pprint
# # from itertools import cycle
# # import random

# # import torch
# # import torch.nn as nn
# # import torch.nn.functional as F
# # from torch.fft import rfft2
# # from torch.utils.data import DataLoader
# # from tqdm import tqdm

# # import torchvision
# # from torchvision import datasets, transforms

# # # Your utilities (must include the Normalize wrapper inside make_and_restore_model)
# # from utils import set_seed, make_and_restore_model


# # # ===========================
# # #        Loss helpers
# # # ===========================

# # def targeted_margin_loss(logits: torch.Tensor, y_tgt: torch.Tensor, kappa: float = 0.0) -> torch.Tensor:
# #     """
# #     Hinge-style targeted loss:
# #     encourage target_logit > max(other_logits) + kappa
# #     Minimization objective (>=0).
# #     """
# #     C = logits.size(1)
# #     one_hot = F.one_hot(y_tgt, num_classes=C).bool()
# #     target_logit = logits[one_hot]
# #     other_logit = logits.masked_fill(one_hot, float('-inf')).amax(dim=1)
# #     return torch.clamp(other_logit - target_logit + kappa, min=0).mean()


# # def total_variation(x: torch.Tensor) -> torch.Tensor:
# #     """Spatial smoothness proxy (lower -> smoother delta). x: (1,C,H,W)"""
# #     tv_h = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean()
# #     tv_w = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()
# #     return tv_h + tv_w


# # def fft_high_energy(x: torch.Tensor, frac: float = 0.5) -> torch.Tensor:
# #     """
# #     Spectral stealth proxy: penalize energy outside the lowest frequencies.
# #     Keeps the lowest 'frac' of the spectral radius and penalizes the rest.
# #     """
# #     xc = x - x.mean(dim=(2, 3), keepdim=True)  # zero-mean per channel
# #     X = rfft2(xc, norm="ortho")  # (B,C,H,W//2+1)
# #     mag2 = (X.real ** 2 + X.imag ** 2)

# #     B, C, H, W2 = mag2.shape
# #     yy = torch.linspace(-1, 1, steps=H, device=x.device).view(H, 1).repeat(1, W2)
# #     xx = torch.linspace(0, 1, steps=W2, device=x.device).view(1, W2).repeat(H, 1)
# #     r = torch.sqrt(xx**2 + yy**2)
# #     keep = (r <= frac * r.max()).float()

# #     high = (1.0 - keep) * mag2
# #     return high.mean()


# # class EMA:
# #     def __init__(self, beta: float = 0.9):
# #         self.beta = beta
# #         self.value = None

# #     def update(self, x: torch.Tensor):
# #         val = float(x.detach().item())
# #         if self.value is None:
# #             self.value = val
# #         else:
# #             self.value = self.beta * self.value + (1 - self.beta) * val


# # # ===========================
# # #     Norm projections
# # # ===========================

# # @torch.no_grad()
# # def project_linf(delta: torch.Tensor, eps: float) -> torch.Tensor:
# #     """Clamp perturbation into L∞ ball of radius eps (in [0,1] pixel space)."""
# #     return delta.clamp_(-eps, eps)


# # @torch.no_grad()
# # def project_l2(delta: torch.Tensor, eps: float) -> torch.Tensor:
# #     """Project perturbation into L2 ball of radius eps."""
# #     flat = delta.view(delta.size(0), -1)
# #     norms = flat.norm(p=2, dim=1, keepdim=True) + 1e-12
# #     scale = (eps / norms).clamp(max=1.0)
# #     flat.mul_(scale)
# #     return delta.view_as(delta)


# # def project_delta(delta: torch.Tensor, eps: float, constraint: str) -> torch.Tensor:
# #     if constraint == 'Linf':
# #         return project_linf(delta, eps)
# #     elif constraint == 'L2':
# #         return project_l2(delta, eps)
# #     else:
# #         raise ValueError(f"Unknown constraint: {constraint}")


# # # ===========================
# # #            PCGrad
# # # ===========================

# # def pcgrad_sum(grads, shuffle=True):
# #     """
# #     grads: list of 1D tensors (each is grad of one loss wrt delta).
# #     Implements PCGrad projection to reduce gradient conflicts.
# #     """
# #     # copy and (optionally) shuffle order
# #     order = list(range(len(grads)))
# #     if shuffle:
# #         random.shuffle(order)

# #     gi_list = [g.clone() for g in grads]
# #     for i_idx in range(len(order)):
# #         i = order[i_idx]
# #         gi = gi_list[i]
# #         for j_idx in range(i_idx):
# #             j = order[j_idx]
# #             gj = gi_list[j]
# #             denom = gj.dot(gj) + 1e-12
# #             dot_ij = gi.dot(gj)
# #             if dot_ij < 0:  # conflict => project
# #                 gi = gi - (dot_ij / denom) * gj
# #         gi_list[i] = gi
# #     # final sum
# #     g_sum = torch.stack(gi_list, dim=0).sum(dim=0)
# #     return g_sum


# # # ===========================
# # #   Universal attack (MOO)
# # # ===========================

# # def universal_target_attack_moo(model: nn.Module,
# #                                 dataset_loader: DataLoader,
# #                                 target_class: int,
# #                                 args) -> torch.Tensor:
# #     """
# #     Multi-objective universal targeted perturbation with:
# #       - EMA-normalized per-term losses
# #       - Optional curriculum weights
# #       - Optional PCGrad to mitigate gradient conflicts
# #       - Optional constant LR warmup (when lr_schedule='constant')
# #       - Correct scheduler placement
# #     """
# #     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# #     # Fresh shuffled view over the same dataset
# #     data_loader = DataLoader(
# #         dataset_loader.dataset,
# #         batch_size=args.batch_size,
# #         shuffle=True,
# #         pin_memory=True,
# #         num_workers=min(4, os.cpu_count() or 1)
# #     )

# #     def one_restart():
# #         # initialize tiny random delta
# #         delta = (torch.zeros(1, *args.data_shape, device=device)
# #                  .uniform_(-1e-6, 1e-6)
# #                  .requires_grad_(True))

# #         opt = torch.optim.Adam([delta], lr=args.step_size)

# #         if args.lr_schedule == "cosine":
# #             scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.num_steps)
# #         elif args.lr_schedule == "constant":
# #             scheduler = None
# #         else:
# #             raise ValueError("--lr_schedule must be 'cosine' or 'constant'")

# #         # EMAs for normalization
# #         ema_asr, ema_spa, ema_spec, ema_clean = EMA(0.9), EMA(0.9), EMA(0.9), EMA(0.9)

# #         best_delta = delta.detach().clone()
# #         best_score = -1.0

# #         batch_iter = cycle(data_loader)
# #         iterator = tqdm(range(args.num_steps), total=args.num_steps, leave=False)

# #         for step_index in iterator:
# #             inp, _ = next(batch_iter)
# #             inp = inp.to(device, non_blocking=True)
# #             tgt = torch.full((inp.size(0),), int(target_class), device=device, dtype=torch.long)

# #             # EOT
# #             def eot_apply(x, repeats=args.eot_samples):
# #                 if repeats <= 1:
# #                     return torch.clamp(x + delta, 0, 1)
# #                 outs = []
# #                 for _r in range(repeats):
# #                     z = x + delta
# #                     z = torch.clamp(z + 0.001 * torch.randn_like(z), 0, 1)
# #                     outs.append(z)
# #                 return torch.cat(outs, dim=0)

# #             x_adv = eot_apply(inp, repeats=args.eot_samples)
# #             tgt_eot = tgt.repeat(args.eot_samples)

# #             model.eval()
# #             logits = model(x_adv)

# #             # ASR loss
# #             if args.asr_loss == 'cw':
# #                 L_asr = targeted_margin_loss(logits, tgt_eot, kappa=args.kappa)
# #             else:
# #                 L_asr = F.cross_entropy(logits, tgt_eot)

# #             # Regularizers
# #             L_spatial  = total_variation(delta)
# #             L_spectral = fft_high_energy(delta, frac=args.fft_frac)
# #             L_clean    = delta.pow(2).mean()

# #             # EMA updates
# #             for ema, L in [(ema_asr, L_asr), (ema_spa, L_spatial),
# #                            (ema_spec, L_spectral), (ema_clean, L_clean)]:
# #                 ema.update(L)

# #             # Curriculum or fixed weights
# #             if args.curriculum:
# #                 alpha = float(step_index) / float(max(1, args.num_steps - 1))
# #                 w_asr = 0.90 - 0.40*alpha   # 0.90 -> 0.50
# #                 w_tv  = 0.05 + 0.10*alpha   # 0.05 -> 0.15
# #                 w_sp  = 0.03 + 0.07*alpha   # 0.03 -> 0.10
# #                 w_l2  = 0.02 + 0.08*alpha   # 0.02 -> 0.10
# #             else:
# #                 w_asr, w_tv, w_sp, w_l2 = 0.70, 0.10, 0.10, 0.10

# #             # normalize by EMA then apply curriculum weights
# #             L_list = [
# #                 w_asr * (L_asr     / (ema_asr.value   + 1e-8)),
# #                 w_tv  * (L_spatial / (ema_spa.value   + 1e-8)),
# #                 w_sp  * (L_spectral/ (ema_spec.value + 1e-8)),
# #                 w_l2  * (L_clean   / (ema_clean.value + 1e-8)),
# #             ]

# #             # ---- OPT STEP ----
# #             opt.zero_grad(set_to_none=True)

# #             if args.pcgrad:
# #                 # Compute per-loss grads on delta
# #                 grads = []
# #                 for L in L_list:
# #                     g = torch.autograd.grad(L, delta, retain_graph=True, create_graph=False)[0]
# #                     grads.append(g.view(-1))
# #                 # PCGrad combine
# #                 g_sum = pcgrad_sum(grads)
# #                 # Write combined grad back to delta and step
# #                 delta.grad = g_sum.view_as(delta).detach()
# #                 opt.step()
# #             else:
# #                 # fallback: weighted sum as before
# #                 total_loss = sum(L_list)
# #                 total_loss.backward()
# #                 opt.step()

# #             # LR schedule / warmup handling
# #             if scheduler is not None:
# #                 scheduler.step()   # cosine schedule
# #             else:
# #                 # constant schedule + (optional) warmup
# #                 if args.warmup > 0 and step_index < args.warmup:
# #                     # linear warmup from 0 -> step_size over warmup steps
# #                     warm_lr = args.step_size * float(step_index + 1) / float(args.warmup)
# #                     for pg in opt.param_groups:
# #                         pg['lr'] = warm_lr

# #             # Project into norm ball
# #             with torch.no_grad():
# #                 delta.copy_(project_delta(delta, args.eps, args.constraint))

# #             # Track best by batch ASR proxy
# #             with torch.no_grad():
# #                 preds = logits.argmax(dim=1)
# #                 asr_batch = (preds == tgt_eot).float().mean().item()
# #                 if asr_batch > best_score:
# #                     best_score = asr_batch
# #                     best_delta = delta.detach().clone()

# #             # Console status
# #             if (step_index + 1) % args.print_every == 0:
# #                 desc = (f"[ Target {target_class} ] | step {step_index+1}/{args.num_steps} "
# #                         f"| ASR {100*asr_batch:5.2f}%")
# #                 if not args.pcgrad:
# #                     desc += f" | Loss {float(sum(L_list)):.4f}"
# #                 iterator.set_description(desc)

# #         return best_delta, best_score

# #     # Random restarts
# #     best_overall, best_score = None, -1.0
# #     for _ in range(args.restarts):
# #         d, s = one_restart()
# #         if s > best_score:
# #             best_score, best_overall = s, d

# #     return best_overall.detach().requires_grad_(False)


# # def moo_generate(args, loader, model):
# #     """Generate universal perturbations for each class (0..num_classes-1)."""
# #     poisons = []
# #     for i in range(args.num_classes):
# #         poison = universal_target_attack_moo(model, loader, i, args)
# #         poisons.append(poison.squeeze())
# #     return poisons


# # # ===========================
# # #           MAIN
# # # ===========================

# # def main(args):
# #     # dataset setup
# #     if args.dataset == 'imagenet200':
# #         args.num_classes = 200
# #         args.img_size = 224
# #         args.channel = 3
# #         args.data_shape = (args.channel, args.img_size, args.img_size)

# #         transform_train = transforms.Compose([
# #             transforms.RandomResizedCrop(args.img_size),
# #             transforms.ToTensor()
# #         ])
# #         transform_test = transforms.Compose([
# #             transforms.Resize(args.img_size),
# #             transforms.CenterCrop(args.img_size),
# #             transforms.ToTensor()
# #         ])
# #         data_set = torchvision.datasets.ImageFolder(
# #             root=os.path.join(args.data_root, 'imagenet200', 'train'),
# #             transform=transform_train
# #         )
# #         test_set = torchvision.datasets.ImageFolder(
# #             root=os.path.join(args.data_root, 'imagenet200', 'val'),
# #             transform=transform_test
# #         )

# #     elif args.dataset == 'cifar10':
# #         args.num_classes = 10
# #         args.img_size = 32
# #         args.channel = 3
# #         args.data_shape = (args.channel, args.img_size, args.img_size)

# #         transform = transforms.Compose([transforms.ToTensor()])
# #         data_set = datasets.CIFAR10(args.data_root, train=True, download=True, transform=transform)
# #         test_set = datasets.CIFAR10(args.data_root, train=False, download=True, transform=transform)

# #     elif args.dataset == 'gtsrb':
# #         args.num_classes = 43
# #         args.img_size = 32
# #         args.channel = 3
# #         args.data_shape = (args.channel, args.img_size, args.img_size)

# #         transform = transforms.Compose([
# #             transforms.Resize(args.img_size),
# #             transforms.CenterCrop(args.img_size),
# #             transforms.ToTensor()
# #         ])
# #         data_set = torchvision.datasets.ImageFolder(
# #             root=os.path.join(args.data_root, 'GTSRB', 'Train'),
# #             transform=transform
# #         )
# #         test_set = torchvision.datasets.ImageFolder(
# #             root=os.path.join(args.data_root, 'GTSRB', 'val4imagefolder'),
# #             transform=transform
# #         )
# #     else:
# #         raise ValueError("Unsupported dataset. Choose from: imagenet200, cifar10, gtsrb")

# #     data_loader = DataLoader(
# #         data_set, batch_size=args.batch_size,
# #         num_workers=min(8, os.cpu_count() or 1),
# #         shuffle=True, pin_memory=True
# #     )
# #     test_loader = DataLoader(
# #         test_set, batch_size=args.batch_size,
# #         num_workers=min(8, os.cpu_count() or 1),
# #         shuffle=False, pin_memory=True
# #     )

# #     # model (assumes your utils.make_and_restore_model adds the Normalize wrapper)
# #     model = make_and_restore_model(args, resume_path=args.model_path)
# #     model.eval()

# #     # seed & output dir
# #     set_seed(args.seed)

# #     moo = moo_generate(args, data_loader, model)
# #     os.makedirs(args.moo_path, exist_ok=True)
# #     for i, d in enumerate(moo):
# #         file_n = f'moo_delta_{i}.pth'
# #         torch.save({'delta': d, 'args': vars(args)}, os.path.join(args.moo_path, file_n))


# # if __name__ == "__main__":
# #     parser = argparse.ArgumentParser('Generate universal targeted perturbations (MOO)')

# #     # Repro & device
# #     parser.add_argument('--seed', default=0, type=int)
# #     parser.add_argument('--gpuid', default=0, type=int)

# #     # Attack / constraint
# #     parser.add_argument('--constraint', default='Linf', choices=['Linf', 'L2'], type=str)
# #     parser.add_argument('--eps', default=8.0, type=float, help='epsilon in pixel scale for Linf; in [0,1] for L2')
# #     parser.add_argument('--num_steps', default=500, type=int)
# #     parser.add_argument('--step_size', default=None, type=float, help='if None, set to eps/2 (normalized)')

# #     parser.add_argument('--restarts', default=5, type=int)
# #     parser.add_argument('--kappa', default=0.0, type=float, help='target margin')
# #     parser.add_argument('--eot_samples', default=1, type=int, help='>1 to add robustness to small noise')
# #     parser.add_argument('--fft_frac', default=0.5, type=float, help='low-freq keep fraction for spectral loss')

# #     # Model / data
# #     parser.add_argument('--arch', default='ResNet18', type=str,
# #                         choices=['VGG16', 'EfficientNetB0', 'DenseNet121',
# #                                  'ResNet18', 'swin', 'inception_next_tiny', 'inception_next_small'])
# #     parser.add_argument('--model_path', default='results/clean_model_weight/checkpoint.pth', type=str)

# #     parser.add_argument('--dataset', default='cifar10', type=str, choices=['cifar10', 'imagenet200', 'gtsrb'])
# #     parser.add_argument('--data_root', default='../data', type=str)

# #     # IO
# #     parser.add_argument('--batch_size', default=256, type=int)
# #     parser.add_argument('--moo_path', default='./results/moo', type=str)
# #     parser.add_argument('--out_dir', default='results/', type=str)

# #     # Loss / curriculum / PCGrad
# #     parser.add_argument('--asr_loss', default='cw', choices=['cw', 'ce'])
# #     parser.add_argument('--curriculum', action='store_true', help='anneal weights during steps')
# #     parser.add_argument('--pcgrad', action='store_true', help='use PCGrad to mitigate conflicting objectives')

# #     # Scheduler / logging / warmup
# #     parser.add_argument('--lr_schedule', default='cosine', choices=['cosine', 'constant'])
# #     parser.add_argument('--warmup', default=0, type=int, help='steps of linear warmup (only when lr_schedule=constant)')
# #     parser.add_argument('--print_every', default=50, type=int)

# #     args = parser.parse_args()

# #     # Output dir name
# #     args.moo_path = f"{args.moo_path}-{args.dataset}-{args.arch}-{args.constraint}-eps{args.eps:.1f}"
# #     os.makedirs(args.moo_path, exist_ok=True)

# #     # Epsilon/step normalization
# #     if args.constraint == 'Linf':
# #         args.eps = args.eps / 255.0
# #         if args.step_size is None:
# #             args.step_size = args.eps / 2.0
# #     else:
# #         if args.step_size is None:
# #             args.step_size = 1e-2

# #     pprint(vars(args))

# #     os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpuid)
# #     torch.backends.cudnn.deterministic = False
# #     torch.backends.cudnn.benchmark = True

# #     main(args)


# # multi_object_optimization_uniform.py
# import os
# import argparse
# from pprint import pprint
# from itertools import cycle
# import random

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.fft import rfft2
# from torch.utils.data import DataLoader
# from tqdm import tqdm

# import torchvision
# from torchvision import datasets, transforms

# # Your utils must add the Normalize wrapper inside make_and_restore_model
# from utils import set_seed, make_and_restore_model


# # ---------------- Loss helpers ----------------

# def targeted_margin_loss(logits: torch.Tensor, y_tgt: torch.Tensor, kappa: float = 0.0) -> torch.Tensor:
#     C = logits.size(1)
#     one_hot = F.one_hot(y_tgt, num_classes=C).bool()
#     target_logit = logits[one_hot]
#     other_logit = logits.masked_fill(one_hot, float('-inf')).amax(dim=1)
#     return torch.clamp(other_logit - target_logit + kappa, min=0).mean()

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


# # ---------------- PCGrad ----------------

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


# # -------------- Attack core (uniform) --------------

# def universal_target_attack_moo(model: nn.Module,
#                                 dataset_loader: DataLoader,
#                                 target_class: int,
#                                 args) -> torch.Tensor:
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#     data_loader = DataLoader(
#         dataset_loader.dataset,
#         batch_size=args.batch_size,
#         shuffle=True,
#         pin_memory=True,
#         num_workers=min(4, os.cpu_count() or 1)
#     )

#     def one_restart():
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

#         batch_iter = cycle(data_loader)
#         iterator = tqdm(range(args.num_steps), total=args.num_steps, leave=False)

#         # optional tiny FGSM warm-start (same for all classes)
#         def fgsm_kick(inp, tgt):
#             # delta is closed-over (shape 1xCxHxW), make a grad-enabled view
#             delta_ws = delta.detach().clone().requires_grad_(True)
#             x_ws = (inp + delta_ws).clamp(0, 1)          # broadcast delta to batch
#             logits_ws = model(x_ws)
#             loss_ws = F.cross_entropy(logits_ws, tgt)    # targeted CE warm-start
#             g_delta = torch.autograd.grad(loss_ws, delta_ws)[0]   # -> shape 1xCxHxW

#             with torch.no_grad():
#                 # step size for the kick; 0.25*eps is a gentle nudge
#                 step = args.eps * 0.25
#                 delta.add_(step * g_delta.sign())
#                 delta.copy_(project_delta(delta, args.eps, args.constraint))


#         for step_index in iterator:
#             inp, _ = next(batch_iter)
#             inp = inp.to(device, non_blocking=True)
#             tgt = torch.full((inp.size(0),), int(target_class), device=device, dtype=torch.long)

#             if step_index == 0 and args.fgsm_warmstart > 0:
#                 # Apply N tiny kicks uniformly
#                 for _ in range(args.fgsm_warmstart):
#                     fgsm_kick(inp, tgt)

#             # EOT
#             def eot_apply(x, repeats=args.eot_samples):
#                 if repeats <= 1:
#                     return torch.clamp(x + delta, 0, 1)
#                 outs = []
#                 for _r in range(repeats):
#                     z = x + delta
#                     z = torch.clamp(z + 0.001 * torch.randn_like(z), 0, 1)
#                     outs.append(z)
#                 return torch.cat(outs, dim=0)

#             x_adv = eot_apply(inp, repeats=args.eot_samples)
#             tgt_eot = tgt.repeat(args.eot_samples)

#             model.eval()
#             logits = model(x_adv)

#             # CE warm-start for all classes, then CW/CE per arg
#             if step_index < args.asr_warmup:
#                 L_asr_eff = F.cross_entropy(logits, tgt_eot)
#             else:
#                 L_asr_eff = (targeted_margin_loss(logits, tgt_eot, kappa=args.kappa)
#                              if args.asr_loss == 'cw' else
#                              F.cross_entropy(logits, tgt_eot))

#             L_spatial  = total_variation(delta)
#             L_spectral = fft_high_energy(delta, frac=args.fft_frac)
#             L_clean    = delta.pow(2).mean()

#             for ema, L in [(ema_asr, L_asr_eff), (ema_spa, L_spatial),
#                            (ema_spec, L_spectral), (ema_clean, L_clean)]:
#                 ema.update(L)

#             # uniform curriculum (same for all classes)
#             if args.curriculum:
#                 alpha = float(step_index) / float(max(1, args.num_steps - 1))
#                 w_asr = 0.90 - 0.40*alpha   # 0.90 -> 0.50
#                 w_tv  = 0.05 + 0.10*alpha   # 0.05 -> 0.15
#                 w_sp  = 0.03 + 0.07*alpha   # 0.03 -> 0.10
#                 w_l2  = 0.02 + 0.08*alpha   # 0.02 -> 0.10
#             else:
#                 w_asr, w_tv, w_sp, w_l2 = 0.70, 0.10, 0.10, 0.10

#             L_list = [
#                 w_asr * (L_asr_eff  / (ema_asr.value   + 1e-8)),
#                 w_tv  * (L_spatial  / (ema_spa.value   + 1e-8)),
#                 w_sp  * (L_spectral / (ema_spec.value  + 1e-8)),
#                 w_l2  * (L_clean    / (ema_clean.value + 1e-8)),
#             ]

#             # Delay PCGrad until after ASR warm-start — uniformly
#             use_pcgrad_now = bool(args.pcgrad) and (step_index >= args.asr_warmup)

#             opt.zero_grad(set_to_none=True)
#             if use_pcgrad_now:
#                 grads = []
#                 for L in L_list:
#                     g = torch.autograd.grad(L, delta, retain_graph=True, create_graph=False)[0]
#                     grads.append(g.view(-1))
#                 g_sum = pcgrad_sum(grads)
#                 delta.grad = g_sum.view_as(delta).detach()
#                 opt.step()
#             else:
#                 total_loss = sum(L_list)
#                 total_loss.backward()
#                 opt.step()

#             # Scheduler (uniform)
#             if scheduler is not None:
#                 scheduler.step()
#             else:
#                 if args.warmup > 0 and step_index < args.warmup:
#                     warm_lr = args.step_size * float(step_index + 1) / float(args.warmup)
#                     for pg in opt.param_groups:
#                         pg['lr'] = warm_lr

#             with torch.no_grad():
#                 delta.copy_(project_delta(delta, args.eps, args.constraint))

#             with torch.no_grad():
#                 preds = logits.argmax(dim=1)
#                 asr_batch = (preds == tgt_eot).float().mean().item()

#             if asr_batch > best_score:
#                 best_score = asr_batch
#                 best_delta = delta.detach().clone()

#             if (step_index + 1) % args.print_every == 0:
#                 iterator.set_description(
#                     f"[ Target {target_class} ] | step {step_index+1}/{args.num_steps} | ASR {100*asr_batch:5.2f}%"
#                 )

#         return best_delta, best_score

#     best_overall, best_score = None, -1.0
#     for _ in range(args.restarts):
#         d, s = one_restart()
#         if s > best_score:
#             best_score, best_overall = s, d

#     return best_overall.detach().requires_grad_(False)


# def moo_generate(args, loader, model):
#     poisons = []
#     for i in range(args.num_classes):
#         poison = universal_target_attack_moo(model, loader, i, args)
#         poisons.append(poison.squeeze())
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
#     parser = argparse.ArgumentParser("Uniform MOO for universal targeted perturbations")

#     # Repro & device
#     parser.add_argument('--seed', default=0, type=int)
#     parser.add_argument('--gpuid', default=0, type=int)

#     # Attack / constraint
#     parser.add_argument('--constraint', default='Linf', choices=['Linf', 'L2'])
#     parser.add_argument('--eps', default=8.0, type=float)         # Linf in pixels; converted below
#     parser.add_argument('--num_steps', default=800, type=int)     # a bit more steps helps all classes uniformly
#     parser.add_argument('--step_size', default=None, type=float)  # if None, set to eps/2 (normalized)
#     parser.add_argument('--restarts', default=6, type=int)
#     parser.add_argument('--kappa', default=0.0, type=float)
#     parser.add_argument('--eot_samples', default=1, type=int)
#     parser.add_argument('--fft_frac', default=0.5, type=float)

#     # Model / data
#     parser.add_argument('--arch', default='ResNet18',
#                         choices=['VGG16','EfficientNetB0','DenseNet121','ResNet18','swin','inception_next_tiny','inception_next_small'])
#     parser.add_argument('--model_path', default='results/clean_model_weight/checkpoint.pth')
#     parser.add_argument('--dataset', default='cifar10', choices=['cifar10','imagenet200','gtsrb'])
#     parser.add_argument('--data_root', default='../data')

#     # IO
#     parser.add_argument('--batch_size', default=256, type=int)
#     parser.add_argument('--moo_path', default='./results/moo', type=str)

#     # Loss / curriculum / PCGrad
#     parser.add_argument('--asr_loss', default='cw', choices=['cw','ce'])
#     parser.add_argument('--curriculum', action='store_true')
#     parser.add_argument('--pcgrad', action='store_true')

#     # Scheduler / warmup / logging
#     parser.add_argument('--lr_schedule', default='constant', choices=['cosine','constant'])
#     parser.add_argument('--warmup', default=100, type=int)          # applies when schedule=constant
#     parser.add_argument('--asr_warmup', default=250, type=int)      # CE warm-start for everyone
#     parser.add_argument('--fgsm_warmstart', default=2, type=int)    # tiny, uniform kickstart
#     parser.add_argument('--print_every', default=50, type=int)

#     args = parser.parse_args()

#     # Output dir
#     args.moo_path = f"{args.moo_path}-{args.dataset}-{args.arch}-{args.constraint}-eps{args.eps:.1f}"
#     os.makedirs(args.moo_path, exist_ok=True)

#     # Normalize eps/step_size units
#     if args.constraint == 'Linf':
#         args.eps = args.eps / 255.0
#         if args.step_size is None:
#             args.step_size = args.eps / 2.0
#     else:
#         if args.step_size is None:
#             args.step_size = 1e-2

#     pprint(vars(args))

#     os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpuid)
#     torch.backends.cudnn.deterministic = False
#     torch.backends.cudnn.benchmark = True

#     main(args)



# import os
# import argparse
# from pprint import pprint
# from itertools import cycle
# import random

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.fft import rfft2
# from torch.utils.data import DataLoader
# from tqdm import tqdm

# import torchvision
# from torchvision import datasets, transforms

# # Your utils must add the Normalize wrapper inside make_and_restore_model
# from utils import set_seed, make_and_restore_model


# # ---------------- Loss helpers ----------------

# def targeted_margin_loss(logits: torch.Tensor, y_tgt: torch.Tensor, kappa: float = 0.0) -> torch.Tensor:
#     C = logits.size(1)
#     one_hot = F.one_hot(y_tgt, num_classes=C).bool()
#     target_logit = logits[one_hot]
#     other_logit = logits.masked_fill(one_hot, float('-inf')).amax(dim=1)
#     return torch.clamp(other_logit - target_logit + kappa, min=0).mean()

# def total_variation(x: torch.Tensor) -> torch.Tensor:
#     tv_h = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean()
#     tv_w = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()
#     return tv_h + tv_w

# def fft_high_energy(x: torch.Tensor, frac: float = 0.5) -> torch.Tensor:
#     xc = x - x.mean(dim=(2, 3), keepdim=True)
#     X = rfft2(xc, norm="ortho")
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


# # ---------------- PCGrad ----------------

# def pcgrad_sum(grads, shuffle=True):
#     order = list(range(len(grads)))
#     if shuffle: random.shuffle(order)
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


# # -------------- Attack core (uniform) --------------

# def universal_target_attack_moo(model: nn.Module,
#                                 dataset_loader: DataLoader,
#                                 target_class: int,
#                                 args) -> torch.Tensor:
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#     data_loader = DataLoader(
#         dataset_loader.dataset,
#         batch_size=args.batch_size,
#         shuffle=True,
#         pin_memory=True,
#         num_workers=min(4, os.cpu_count() or 1)
#     )

#     # small, fixed validation slice for stable logging + early stop
#     _fixed = next(iter(data_loader))
#     val_x = _fixed[0][:min(256, _fixed[0].size(0))].to(device)
#     val_tgt = torch.full((val_x.size(0),), int(target_class), device=device, dtype=torch.long)

#     def one_restart():
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
#         since_best = 0

#         batch_iter = cycle(data_loader)
#         iterator = tqdm(range(args.num_steps), total=args.num_steps, leave=False)

#         # optional tiny FGSM warm-start (uniform)
#         def fgsm_kick(inp, tgt):
#             delta_ws = delta.detach().clone().requires_grad_(True)
#             x_ws = (inp + delta_ws).clamp(0, 1)
#             logits_ws = model(x_ws)
#             loss_ws = F.cross_entropy(logits_ws, tgt)
#             g_delta = torch.autograd.grad(loss_ws, delta_ws)[0]
#             with torch.no_grad():
#                 step = args.eps * 0.25
#                 delta.add_(step * g_delta.sign())
#                 delta.copy_(project_delta(delta, args.eps, args.constraint))

#         # optional FGSM pre-kicks
#         if args.fgsm_warmstart > 0:
#             inp0, _ = next(batch_iter)
#             inp0 = inp0.to(device)
#             tgt0 = torch.full((inp0.size(0),), int(target_class), device=device, dtype=torch.long)
#             for _ in range(args.fgsm_warmstart):
#                 fgsm_kick(inp0, tgt0)

#         for step_index in iterator:
#             inp, _ = next(batch_iter)
#             inp = inp.to(device, non_blocking=True)
#             tgt = torch.full((inp.size(0),), int(target_class), device=device, dtype=torch.long)

#             # EOT
#             def eot_apply(x, repeats=args.eot_samples):
#                 if repeats <= 1:
#                     return torch.clamp(x + delta, 0, 1)
#                 outs = []
#                 for _r in range(repeats):
#                     z = x + delta
#                     z = torch.clamp(z + 0.001 * torch.randn_like(z), 0, 1)
#                     outs.append(z)
#                 return torch.cat(outs, dim=0)

#             x_adv = eot_apply(inp, repeats=args.eot_samples)
#             tgt_eot = tgt.repeat(args.eot_samples)

#             model.eval()
#             logits = model(x_adv)

#             # ---- ASR objective with burn-in ----
#             if step_index < args.burn_in:
#                 L_asr_eff = F.cross_entropy(logits, tgt_eot)  # ASR-only burn-in
#             else:
#                 L_asr_eff = targeted_margin_loss(logits, tgt_eot, kappa=args.kappa) if args.asr_loss == 'cw' \
#                             else F.cross_entropy(logits, tgt_eot)

#             # Regularizers
#             L_spatial  = total_variation(delta)
#             L_spectral = fft_high_energy(delta, frac=args.fft_frac)
#             L_clean    = delta.pow(2).mean()

#             # EMA updates (for normalized combination after burn-in)
#             for ema, L in [(ema_asr, L_asr_eff), (ema_spa, L_spatial),
#                            (ema_spec, L_spectral), (ema_clean, L_clean)]:
#                 ema.update(L)

#             # Curriculum (uniform), with ASR floor + caps on regs
#             if args.curriculum:
#                 alpha = float(step_index) / float(max(1, args.num_steps - 1))
#                 w_asr = max(0.70, 0.90 - 0.40 * alpha)  # never below 0.70
#                 w_tv  = min(0.12, 0.05 + 0.10 * alpha)
#                 w_sp  = min(0.10, 0.03 + 0.07 * alpha)
#                 w_l2  = min(0.10, 0.02 + 0.08 * alpha)
#             else:
#                 w_asr, w_tv, w_sp, w_l2 = 0.70, 0.10, 0.10, 0.10

#             # Build loss list
#             if step_index < args.burn_in:
#                 L_list = [L_asr_eff]  # ASR only
#             else:
#                 L_list = [
#                     w_asr * (L_asr_eff  / (ema_asr.value   + 1e-8)),
#                     w_tv  * (L_spatial  / (ema_spa.value   + 1e-8)),
#                     w_sp  * (L_spectral / (ema_spec.value  + 1e-8)),
#                     w_l2  * (L_clean    / (ema_clean.value + 1e-8)),
#                 ]

#             # PCGrad only after burn-in to avoid losing early momentum
#             use_pcgrad_now = bool(args.pcgrad) and (step_index >= args.burn_in)

#             # ---- OPT STEP ----
#             opt.zero_grad(set_to_none=True)
#             if use_pcgrad_now and len(L_list) > 1:
#                 grads = []
#                 for L in L_list:
#                     g = torch.autograd.grad(L, delta, retain_graph=True, create_graph=False)[0]
#                     grads.append(g.view(-1))
#                 g_sum = pcgrad_sum(grads)
#                 delta.grad = g_sum.view_as(delta).detach()
#                 opt.step()
#             else:
#                 total_loss = sum(L_list)
#                 total_loss.backward()
#                 opt.step()

#             # Scheduler / warmup
#             if scheduler is not None:
#                 scheduler.step()
#             else:
#                 if args.warmup > 0 and step_index < args.warmup:
#                     warm_lr = args.step_size * float(step_index + 1) / float(args.warmup)
#                     for pg in opt.param_groups:
#                         pg['lr'] = warm_lr

#             with torch.no_grad():
#                 delta.copy_(project_delta(delta, args.eps, args.constraint))

#             # Batch proxy ASR
#             with torch.no_grad():
#                 preds = logits.argmax(dim=1)
#                 asr_batch = (preds == tgt_eot).float().mean().item()

#             # Fixed-val ASR for stability + early stopping
#             with torch.no_grad():
#                 asr_val = (model((val_x + delta).clamp(0, 1)).argmax(1) == val_tgt).float().mean().item()

#             improved = asr_val > best_score
#             if improved:
#                 best_score = asr_val
#                 best_delta = delta.detach().clone()
#                 since_best = 0
#             else:
#                 since_best += 1

#             if (step_index + 1) % args.print_every == 0:
#                 iterator.set_description(
#                     f"[Target {target_class}] step {step_index+1}/{args.num_steps} | "
#                     f"ASR(batch) {100*asr_batch:5.2f}% | ASR(val) {100*asr_val:5.2f}%"
#                 )

#             if since_best >= args.patience:
#                 break  # early stop to avoid late erosion

#         return best_delta, best_score

#     best_overall, best_score = None, -1.0
#     for _ in range(args.restarts):
#         d, s = one_restart()
#         if s > best_score:
#             best_score, best_overall = s, d

#     return best_overall.detach().requires_grad_(False)


# def moo_generate(args, loader, model):
#     poisons = []
#     for i in range(args.num_classes):
#         poison = universal_target_attack_moo(model, loader, i, args)
#         poisons.append(poison.squeeze())
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
#     parser = argparse.ArgumentParser("Uniform MOO for universal targeted perturbations")

#     # Repro & device
#     parser.add_argument('--seed', default=0, type=int)
#     parser.add_argument('--gpuid', default=0, type=int)

#     # Attack / constraint
#     parser.add_argument('--constraint', default='Linf', choices=['Linf', 'L2'])
#     parser.add_argument('--eps', default=8.0, type=float)         # Linf in pixels; converted below
#     parser.add_argument('--num_steps', default=800, type=int)
#     parser.add_argument('--step_size', default=None, type=float)  # if None, set to eps/2 (normalized)
#     parser.add_argument('--restarts', default=6, type=int)
#     parser.add_argument('--kappa', default=0.0, type=float)
#     parser.add_argument('--eot_samples', default=1, type=int)
#     parser.add_argument('--fft_frac', default=0.5, type=float)

#     # Model / data
#     parser.add_argument('--arch', default='ResNet18',
#                         choices=['VGG16','EfficientNetB0','DenseNet121','ResNet18','swin','inception_next_tiny','inception_next_small'])
#     parser.add_argument('--model_path', default='results/clean_model_weight/checkpoint.pth')
#     parser.add_argument('--dataset', default='cifar10', choices=['cifar10','imagenet200','gtsrb'])
#     parser.add_argument('--data_root', default='../data')

#     # IO
#     parser.add_argument('--batch_size', default=256, type=int)
#     parser.add_argument('--moo_path', default='./results/moo', type=str)

#     # Loss / curriculum / PCGrad
#     parser.add_argument('--asr_loss', default='cw', choices=['cw','ce'])
#     parser.add_argument('--curriculum', action='store_true')
#     parser.add_argument('--pcgrad', action='store_true')

#     # Scheduler / warmup / logging / burn-in & early stop
#     parser.add_argument('--lr_schedule', default='constant', choices=['cosine','constant'])
#     parser.add_argument('--warmup', default=100, type=int)          # when schedule=constant
#     parser.add_argument('--burn_in', default=50, type=int)          # ASR-only steps
#     parser.add_argument('--patience', default=120, type=int)        # early stopping on val ASR
#     parser.add_argument('--fgsm_warmstart', default=2, type=int)    # tiny uniform kickstart
#     parser.add_argument('--print_every', default=50, type=int)

#     args = parser.parse_args()

#     # Output dir
#     args.moo_path = f"{args.moo_path}-{args.dataset}-{args.arch}-{args.constraint}-eps{args.eps:.1f}"
#     os.makedirs(args.moo_path, exist_ok=True)

#     # Normalize eps/step_size units
#     if args.constraint == 'Linf':
#         args.eps = args.eps / 255.0
#         if args.step_size is None:
#             args.step_size = args.eps / 2.0
#     else:
#         if args.step_size is None:
#             args.step_size = 1e-2

#     pprint(vars(args))

#     os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpuid)
#     torch.backends.cudnn.deterministic = False
#     torch.backends.cudnn.benchmark = True

#     main(args)




# multi_object_optimization_uniform_full.py
import os
import csv
import argparse
from pprint import pprint
from itertools import cycle
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import torchvision
from torchvision import datasets, transforms

# Your utils must add the Normalize wrapper inside make_and_restore_model
from utils import set_seed, make_and_restore_model


# ---------------- Loss helpers ----------------

def targeted_margin_loss(logits: torch.Tensor, y_tgt: torch.Tensor, kappa: float = 0.0) -> torch.Tensor:
    C = logits.size(1)
    one_hot = F.one_hot(y_tgt, num_classes=C).bool()
    target_logit = logits[one_hot]
    other_logit = logits.masked_fill(one_hot, float('-inf')).amax(dim=1)
    return torch.clamp(other_logit - target_logit + kappa, min=0).mean()

def logits_margin_stats(logits: torch.Tensor, y_tgt: torch.Tensor):
    """Returns per-sample (target_logit - best_other) margin (no clamp)."""
    C = logits.size(1)
    one_hot = F.one_hot(y_tgt, num_classes=C).bool()
    target_logit = logits[one_hot]
    other_logit = logits.masked_fill(one_hot, float('-inf')).amax(dim=1)
    return target_logit - other_logit  # positive is good for targeted attack

def total_variation(x: torch.Tensor) -> torch.Tensor:
    tv_h = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean()
    tv_w = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()
    return tv_h + tv_w

def fft_high_energy(x: torch.Tensor, frac: float = 0.5) -> torch.Tensor:
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


# ---------------- PCGrad ----------------

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


# -------------- Attack core (uniform) --------------

def universal_target_attack_moo(model: nn.Module,
                                dataset_loader: DataLoader,
                                target_class: int,
                                args,
                                log_rows: list) -> torch.Tensor:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_loader = DataLoader(
        dataset_loader.dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=min(4, os.cpu_count() or 1)
    )

    def one_restart(restart_idx: int):
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

        batch_iter = cycle(data_loader)
        iterator = tqdm(range(args.num_steps), total=args.num_steps, leave=False)

        # optional tiny FGSM warm-start (uniform for all classes)
        def fgsm_kick(inp, tgt):
            delta_ws = delta.detach().clone().requires_grad_(True)
            x_ws = (inp + delta_ws).clamp(0, 1)
            logits_ws = model(x_ws)
            loss_ws = F.cross_entropy(logits_ws, tgt)  # targeted CE warm-start
            g_delta = torch.autograd.grad(loss_ws, delta_ws)[0]
            with torch.no_grad():
                step = args.eps * 0.25
                delta.add_(step * g_delta.sign())
                delta.copy_(project_delta(delta, args.eps, args.constraint))

        for step_index in iterator:
            inp, _ = next(batch_iter)
            inp = inp.to(device, non_blocking=True)
            tgt = torch.full((inp.size(0),), int(target_class), device=device, dtype=torch.long)

            if step_index == 0 and args.fgsm_warmstart > 0:
                for _ in range(args.fgsm_warmstart):
                    fgsm_kick(inp, tgt)

            # EOT
            def eot_apply(x, repeats=args.eot_samples):
                if repeats <= 1:
                    return torch.clamp(x + delta, 0, 1)
                outs = []
                for _r in range(repeats):
                    z = x + delta
                    z = torch.clamp(z + 0.001 * torch.randn_like(z), 0, 1)
                    outs.append(z)
                return torch.cat(outs, dim=0)

            x_adv = eot_apply(inp, repeats=args.eot_samples)
            tgt_eot = tgt.repeat(args.eot_samples)

            model.eval()
            logits = model(x_adv)

            # ASR loss warm-up (uniform)
            if step_index < args.asr_warmup:
                L_asr_eff = F.cross_entropy(logits, tgt_eot)
            else:
                L_asr_eff = (targeted_margin_loss(logits, tgt_eot, kappa=args.kappa)
                             if args.asr_loss == 'cw' else
                             F.cross_entropy(logits, tgt_eot))

            # Regularizers
            L_spatial  = total_variation(delta)
            L_spectral = fft_high_energy(delta, frac=args.fft_frac)
            L_clean    = delta.pow(2).mean()

            # EMA updates
            for ema, L in [(ema_asr, L_asr_eff), (ema_spa, L_spatial),
                           (ema_spec, L_spectral), (ema_clean, L_clean)]:
                ema.update(L)

            # Uniform curriculum weights
            if args.curriculum:
                alpha = float(step_index) / float(max(1, args.num_steps - 1))
                w_asr = 0.90 - 0.40*alpha   # 0.90 -> 0.50
                w_tv  = 0.05 + 0.10*alpha   # 0.05 -> 0.15
                w_sp  = 0.03 + 0.07*alpha   # 0.03 -> 0.10
                w_l2  = 0.02 + 0.08*alpha   # 0.02 -> 0.10
            else:
                w_asr, w_tv, w_sp, w_l2 = 0.70, 0.10, 0.10, 0.10

            # normalize by EMA then apply curriculum weights
            L_list = [
                w_asr * (L_asr_eff  / (ema_asr.value   + 1e-8)),
                w_tv  * (L_spatial  / (ema_spa.value   + 1e-8)),
                w_sp  * (L_spectral / (ema_spec.value  + 1e-8)),
                w_l2  * (L_clean    / (ema_clean.value + 1e-8)),
            ]

            # ---- OPT STEP ----
            opt.zero_grad(set_to_none=True)

            # Compute a grad norm snapshot BEFORE stepping (for logging)
            if args.pcgrad and (step_index >= args.asr_warmup):
                grads = []
                for L in L_list:
                    g = torch.autograd.grad(L, delta, retain_graph=True, create_graph=False)[0]
                    grads.append(g.view(-1))
                g_sum = pcgrad_sum(grads)
                # grad norm for logging
                gnorm = float(g_sum.norm().item())
                delta.grad = g_sum.view_as(delta).detach()
                opt.step()
            else:
                total_loss = sum(L_list)
                total_loss.backward()
                # grad norm for logging
                gnorm = float(delta.grad.view(-1).norm().item())
                opt.step()

            # LR schedule / warmup
            if scheduler is not None:
                scheduler.step()
            else:
                if args.warmup > 0 and step_index < args.warmup:
                    warm_lr = args.step_size * float(step_index + 1) / float(args.warmup)
                    for pg in opt.param_groups:
                        pg['lr'] = warm_lr

            # Project into norm ball
            with torch.no_grad():
                delta.copy_(project_delta(delta, args.eps, args.constraint))

            # Metrics
            with torch.no_grad():
                preds = logits.argmax(dim=1)
                asr_batch = float((preds == tgt_eot).float().mean().item())

                # margin (positive is better for targeted)
                m = logits_margin_stats(logits, tgt_eot).mean().item()

                # saturation proxy
                if args.constraint == 'Linf':
                    sat = float((delta.abs() >= (args.eps - 1e-6)).float().mean().item())
                else:  # L2: norm ratio (clipped to 1)
                    d_norm = delta.view(delta.size(0), -1).norm(p=2, dim=1)
                    sat = float((d_norm / (args.eps + 1e-12)).clamp(max=1.0).mean().item())

                # keep best delta by batch ASR
                if asr_batch > best_score:
                    best_score = asr_batch
                    best_delta = delta.detach().clone()

            # Console status (compact)
            if (step_index + 1) % args.print_every == 0:
                iterator.set_description(
                    f"[Target {target_class}] step {step_index+1}/{args.num_steps} "
                    f"| ASR {100*asr_batch:5.2f}% | sat {100*sat:4.1f}% | margin {m:.4f} | gnorm {gnorm:.3e}"
                )

            # ---- CSV logging ----
            if args.log_csv is not None:
                log_rows.append({
                    "restart": restart_idx,
                    "target_class": target_class,
                    "step": step_index + 1,
                    "asr": asr_batch,
                    "sat": sat,
                    "margin": m,
                    "gnorm": gnorm,
                })

        return best_delta, best_score

    best_overall, best_score = None, -1.0
    for r in range(args.restarts):
        d, s = one_restart(restart_idx=r)
        if s > best_score:
            best_score, best_overall = s, d

    return best_overall.detach().requires_grad_(False)


def moo_generate(args, loader, model):
    """Generate universal perturbations for each class (0..num_classes-1)."""
    log_rows = []
    poisons = []
    for i in range(args.num_classes):
        poison = universal_target_attack_moo(model, loader, i, args, log_rows)
        poisons.append(poison.squeeze())

    # dump CSV once (includes all classes / steps / restarts)
    if args.log_csv is not None and len(log_rows) > 0:
        os.makedirs(os.path.dirname(args.log_csv), exist_ok=True)
        with open(args.log_csv, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["restart", "target_class", "step", "asr", "sat", "margin", "gnorm"]
            )
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
    parser = argparse.ArgumentParser("Uniform MOO for universal targeted perturbations")

    # Repro & device
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--gpuid', default=0, type=int)

    # Attack / constraint
    parser.add_argument('--constraint', default='Linf', choices=['Linf', 'L2'])
    parser.add_argument('--eps', default=8.0, type=float)         # Linf in pixels; converted below
    parser.add_argument('--num_steps', default=1000, type=int)
    parser.add_argument('--step_size', default=None, type=float)  # if None, set to eps/2 (normalized)
    parser.add_argument('--restarts', default=6, type=int)
    parser.add_argument('--kappa', default=0.0, type=float)
    parser.add_argument('--eot_samples', default=1, type=int)
    parser.add_argument('--fft_frac', default=0.5, type=float)

    # Model / data
    parser.add_argument('--arch', default='ResNet18',
                        choices=['VGG16','EfficientNetB0','DenseNet121','ResNet18','swin','inception_next_tiny','inception_next_small'])
    parser.add_argument('--model_path', default='results/clean_model_weight/checkpoint.pth')
    parser.add_argument('--dataset', default='cifar10', choices=['cifar10','imagenet200','gtsrb'])
    parser.add_argument('--data_root', default='./data')

    # IO
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--moo_path', default='./results/moo', type=str)
    parser.add_argument('--log_csv', default=None, type=str,
                        help="If set, save per-step metrics for all targets/restarts to this CSV path.")

    # Loss / curriculum / PCGrad
    parser.add_argument('--asr_loss', default='cw', choices=['cw','ce'])
    parser.add_argument('--curriculum', action='store_true')
    parser.add_argument('--pcgrad', action='store_true')

    # Scheduler / warmup / logging
    parser.add_argument('--lr_schedule', default='constant', choices=['cosine','constant'])
    parser.add_argument('--warmup', default=100, type=int)          # applies when schedule=constant
    parser.add_argument('--asr_warmup', default=250, type=int)      # CE warm-start for everyone
    parser.add_argument('--fgsm_warmstart', default=2, type=int)    # tiny, uniform kickstart
    parser.add_argument('--print_every', default=50, type=int)

    args = parser.parse_args()

    # Output dir
    args.moo_path = f"{args.moo_path}-{args.dataset}-{args.arch}-{args.constraint}-eps{args.eps:.1f}"
    os.makedirs(args.moo_path, exist_ok=True)

    # Normalize eps/step_size units
    if args.constraint == 'Linf':
        args.eps = args.eps / 255.0
        if args.step_size is None:
            args.step_size = args.eps / 2.0
    else:
        if args.step_size is None:
            args.step_size = 1e-2

    pprint(vars(args))

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpuid)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    main(args)
