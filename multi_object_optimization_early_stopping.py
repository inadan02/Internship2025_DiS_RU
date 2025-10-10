#!/usr/bin/env python3
import os
import csv
import argparse
from pprint import pprint
from itertools import cycle
import random

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

def universal_target_attack_moo(model, dataset_loader, target_class, args, log_rows):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_loader = DataLoader(dataset_loader.dataset,
                             batch_size=args.batch_size,
                             shuffle=True,
                             pin_memory=True,
                             num_workers=min(4, os.cpu_count() or 1))

    # small fixed validation slice for stable logging (optional)
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
        best_step = -1  # for early stopping
        batch_iter = cycle(data_loader)
        iterator = tqdm(range(args.num_steps), total=args.num_steps, leave=False)

        # optional tiny FGSM warm-start (gentler to avoid early saturation)
        def fgsm_kick(inp, y):
            delta_ws = delta.detach().clone().requires_grad_(True)
            x_ws = (inp + delta_ws).clamp(0, 1)
            logits_ws = model(x_ws)
            loss_ws = F.cross_entropy(logits_ws, y)
            g_delta = torch.autograd.grad(loss_ws, delta_ws)[0]
            with torch.no_grad():
                step = args.eps * 0.0625  # eps/16 per kick
                delta.add_(step * g_delta.sign())
                delta.copy_(project_delta(delta, args.eps, args.constraint))

        # optional pre-kicks
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

        # Early stopping trackers
        patience = args.early_stop_patience
        min_delta = args.early_stop_min_delta

        for step_index in iterator:
            inp, y = next(batch_iter)
            inp = inp.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            # --- Exclude samples that are already the target class
            mask = (y != target_class)
            if (~mask).all():
                continue
            inp = inp[mask]
            y = y[mask]
            tgt = torch.full((inp.size(0),), int(target_class), device=device, dtype=torch.long)

            # EOT + optional input diversity (DI)
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

            # ASR loss warm-up and kappa ramp
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

            # Hinge clipping penalty (discourage boundary saturation)
            if args.lambda_clip > 0.0:
                if args.constraint == 'Linf':
                    over = (delta.abs() - args.eps).clamp(min=0.0)
                    L_clip = (over.pow(2)).mean()
                else:
                    dnorm = delta.view(delta.size(0), -1).norm(p=2, dim=1)
                    over = (dnorm - args.eps).clamp(min=0.0)
                    L_clip = (over.pow(2)).mean()
            else:
                L_clip = torch.tensor(0.0, device=delta.device)

            # EMA updates
            for ema, L in [(ema_asr, L_asr_eff), (ema_spa, L_spatial),
                           (ema_spec, L_spectral), (ema_clean, L_clean)]:
                ema.update(L)

            # Curriculum (ASR strong early; smoothness ramps up)
            if args.curriculum:
                alpha = float(step_index) / float(max(1, args.num_steps - 1))
                w_asr = 0.85 - 0.20 * alpha   # 0.85 -> 0.65
                w_tv  = 0.05 + 0.10 * alpha   # 0.05 -> 0.15
                w_sp  = 0.03 + 0.07 * alpha   # 0.03 -> 0.10
                w_l2  = 0.02 + 0.10 * alpha   # 0.02 -> 0.12
            else:
                w_asr, w_tv, w_sp, w_l2 = 0.70, 0.12, 0.10, 0.08

            L_list = [
                w_asr * (L_asr_eff  / (ema_asr.value   + 1e-8)),
                w_tv  * (L_spatial  / (ema_spa.value   + 1e-8)),
                w_sp  * (L_spectral / (ema_spec.value  + 1e-8)),
                w_l2  * (L_clean    / (ema_clean.value + 1e-8)),
            ]
            if args.lambda_clip > 0.0:
                L_list.append(args.lambda_clip * L_clip)

            # OPT step
            opt.zero_grad(set_to_none=True)

            use_pcgrad_now = bool(args.pcgrad) and (step_index >= args.asr_warmup)

            if use_pcgrad_now:
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

                # Track best & early stopping
                if (asr_val - best_score) > min_delta:
                    best_score = asr_val
                    best_delta = delta.detach().clone()
                    best_step = step_index
                else:
                    if best_step >= 0 and (step_index - best_step) >= patience:
                        print(f"[T{target_class}] Early stopping at step {step_index+1} "
                              f"(no ASR improvement > {min_delta:.4f} for {patience} steps)")
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
            fieldnames = ["restart", "target_class", "step", "asr_batch", "asr_val", "sat", "margin", "gnorm"]
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
    parser = argparse.ArgumentParser("Uniform MOO for universal targeted perturbations")

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

    # Loss / curriculum / PCGrad
    parser.add_argument('--asr_loss', default='cw', choices=['cw','ce'])
    parser.add_argument('--curriculum', action='store_true')
    parser.add_argument('--pcgrad', action='store_true')

    # Scheduler / warmup / logging
    parser.add_argument('--lr_schedule', default='constant', choices=['cosine','constant'])
    parser.add_argument('--warmup', default=100, type=int)
    parser.add_argument('--asr_warmup', default=300, type=int)
    parser.add_argument('--fgsm_warmstart', default=2, type=int)
    parser.add_argument('--di_rate', default=0.20, type=float, help="probability of applying DI on each EOT sample")
    parser.add_argument('--di_scale', default=0.08, type=float, help="scale variation for DI (fraction)")
    parser.add_argument('--print_every', default=50, type=int)

    # Stealth / saturation control
    parser.add_argument('--lambda_clip', default=0.30, type=float, help="hinge clip penalty weight (discourage boundary saturation)")

    # Anneal for fine-tuning
    parser.add_argument('--anneal_mid', action='store_true', help="anneal lr in mid-run for fine-tuning")
    parser.add_argument('--anneal_factor', default=0.25, type=float, help="factor to multiply lr by when annealing at mid-run")

    # Early stopping
    parser.add_argument('--early_stop_patience', default=120, type=int, help="stop if no ASR(val) improvement for this many steps")
    parser.add_argument('--early_stop_min_delta', default=0.002, type=float, help="minimum ASR(val) improvement to reset patience")

    args = parser.parse_args()

    # Output dir
    args.moo_path = f"{args.moo_path}-{args.dataset}-{args.arch}-{args.constraint}-eps{args.eps:.1f}"
    os.makedirs(args.moo_path, exist_ok=True)

    # Normalize eps/step_size units
    if args.constraint == 'Linf':
        args.eps = args.eps / 255.0
        if args.step_size is None:
            # default that favors stealth over speed
            args.step_size = args.eps / 4.0
    else:
        if args.step_size is None:
            args.step_size = 1e-2

    pprint(vars(args))

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpuid)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    main(args)
