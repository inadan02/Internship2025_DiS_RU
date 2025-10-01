import os
import math
import argparse
from pprint import pprint

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.fft import rfft2
from torch.utils.data import DataLoader
from tqdm import tqdm
from itertools import cycle


import torchvision
from torchvision import datasets, transforms

# ---- Your utilities (keep these in your project as before) ----
# Must provide: set_seed, make_and_restore_model
from utils import set_seed, make_and_restore_model


# ===========================
#        MOO HELPERS
# ===========================

def pcgrad_sum(grads):
    """
    grads: list of flattened grads (1D tensors) for each objective w.r.t. delta
    Implements PCGrad (Yu et al., NeurIPS'20) for small #objectives.
    Returns a single combined gradient (same shape as grads[0]).
    """
    G = [g.clone() for g in grads]
    # random order each step
    order = torch.randperm(len(G), device=G[0].device)
    for i in range(len(G)):
        gi = G[order[i]]
        for j in range(len(G)):
            if i == j: 
                continue
            gj = G[order[j]]
            # if conflict, project gi to remove component along gj
            dot = torch.dot(gi, gj)
            if dot < 0:
                gi.add_( - dot / (gj.norm(p=2)**2 + 1e-12) * gj )
        G[order[i]] = gi
    # sum
    g_sum = torch.stack(G, dim=0).sum(dim=0)
    return g_sum


def targeted_margin_loss(logits: torch.Tensor, y_tgt: torch.Tensor, kappa: float = 0.0) -> torch.Tensor:
    """
    Hinge-style targeted loss:
    encourage target_logit > max(other_logits) + kappa
    Minimization objective (>=0).
    """
    C = logits.size(1)
    one_hot = F.one_hot(y_tgt, num_classes=C).bool()
    target_logit = logits[one_hot]
    other_logit = logits.masked_fill(one_hot, float('-inf')).amax(dim=1)
    return torch.clamp(other_logit - target_logit + kappa, min=0).mean()


def total_variation(x: torch.Tensor) -> torch.Tensor:
    """Spatial smoothness proxy (lower -> smoother delta). x: (1,C,H,W)"""
    tv_h = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean()
    tv_w = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()
    return tv_h + tv_w


def fft_high_energy(x: torch.Tensor, frac: float = 0.5) -> torch.Tensor:
    """
    Spectral stealth proxy: penalize energy outside the lowest frequencies.
    Keeps the lowest 'frac' of the spectral radius and penalizes the rest.
    """
    # zero-mean per-channel to avoid DC dominance
    xc = x - x.mean(dim=(2, 3), keepdim=True)
    X = rfft2(xc, norm="ortho")  # (B,C,H,W//2+1)
    mag2 = (X.real ** 2 + X.imag ** 2)

    B, C, H, W2 = mag2.shape
    # radial mask over the (H, W2) rFFT grid
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
        val = float(x.detach().item())
        if self.value is None:
            self.value = val
        else:
            self.value = self.beta * self.value + (1 - self.beta) * val


class UncertaintyWeights(nn.Module):
    """
    Learnable weights: sum_i [ L_i / (2*sigma_i^2) + log(sigma_i) ].
    This auto-balances multiple normalized objectives.
    """
    def __init__(self, n_terms: int):
        super().__init__()
        self.log_sigma = nn.Parameter(torch.zeros(n_terms))  # sigma ~ 1 initially

    def forward(self, losses):
        sig = torch.exp(self.log_sigma) + 1e-8
        total = 0.0
        for i, L in enumerate(losses):
            total = total + (L / (2.0 * sig[i]**2) + torch.log(sig[i]))
        return total


# ===========================
#     NORM PROJECTIONS
# ===========================

@torch.no_grad()
def project_linf(delta: torch.Tensor, eps: float) -> torch.Tensor:
    """Clamp perturbation into L∞ ball of radius eps (in [0,1] pixel space)."""
    return delta.clamp_(-eps, eps)


@torch.no_grad()
def project_l2(delta: torch.Tensor, eps: float) -> torch.Tensor:
    """Project perturbation into L2 ball of radius eps."""
    flat = delta.view(delta.size(0), -1)
    norms = flat.norm(p=2, dim=1, keepdim=True) + 1e-12
    scale = (eps / norms).clamp(max=1.0)
    flat.mul_(scale)
    return delta.view_as(delta)


def project_delta(delta: torch.Tensor, eps: float, constraint: str) -> torch.Tensor:
    if constraint == 'Linf':
        return project_linf(delta, eps)
    elif constraint == 'L2':
        return project_l2(delta, eps)
    else:
        raise ValueError(f"Unknown constraint: {constraint}")


# ===========================
#   UNIVERSAL ATTACK (MOO)
# ===========================

def universal_target_attack_moo(model: nn.Module,
                                dataset_loader: DataLoader,
                                target_class: int,
                                args) -> torch.Tensor:
    """
    Multi-objective universal targeted perturbation:
    - Objectives: ASR (targeted margin), Spatial TV, Spectral high-energy, L2(delta)
    - Adam optimizer on delta
    - EMA normalization + Uncertainty weighting
    - Optional EOT for robustness
    - Random restarts with best-delta tracking
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # fresh shuffled view over the same dataset
    data_loader = DataLoader(dataset_loader.dataset,
                             batch_size=args.batch_size,
                             shuffle=True,
                             pin_memory=True,
                             num_workers=min(4, os.cpu_count() or 1))
    data_iter = iter(data_loader)

    # weights for [ASR, Spatial, Spectral, CleanProxy]
    # uw = UncertaintyWeights(4).to(device)
    # uw_opt = torch.optim.Adam(uw.parameters(), lr=args.uw_lr)

    def one_restart():
    # initialize tiny random delta
        delta = (torch.zeros(1, *args.data_shape, device=device)
                .uniform_(-1e-6, 1e-6)
                .requires_grad_(True))
        # opt = torch.optim.Adam([delta], lr=args.step_size)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.num_steps)
        opt = torch.optim.Adam([delta], lr=args.step_size)
        scheduler = None
        if not args.const_lr:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.num_steps)



        ema_asr, ema_spa, ema_spec, ema_clean = EMA(0.9), EMA(0.9), EMA(0.9), EMA(0.9)
        best_delta = delta.detach().clone()
        best_score = -1.0

        # make a local, never-ending iterator
        batch_iter = cycle(data_loader)

        iterator = tqdm(range(args.num_steps), total=args.num_steps, leave=False)
        for step_index in iterator:
            inp, _ = next(batch_iter)
            inp = inp.to(device, non_blocking=True)
            tgt = torch.full((inp.size(0),), int(target_class), device=device, dtype=torch.long)


            # EOT
            def eot_apply(x, repeats=args.eot_samples):
                if repeats <= 1:
                    z = torch.clamp(x + delta, 0, 1)
                    return z
                outs = []
                for _r in range(repeats):
                    z = x + delta
                    # light differentiable noise
                    z = torch.clamp(z + 0.001 * torch.randn_like(z), 0, 1)
                    outs.append(z)
                return torch.cat(outs, dim=0)

            x_adv = eot_apply(inp, repeats=args.eot_samples)
            tgt_eot = tgt.repeat(args.eot_samples)

            model.eval()
            logits = model(x_adv)

            if args.asr_loss == 'cw':
                L_asr = targeted_margin_loss(logits, tgt_eot, kappa=args.kappa)
            else:
                L_asr = F.cross_entropy(logits, tgt_eot)



            # ----- Multi-objective terms -----
            #L_asr = targeted_margin_loss(logits, tgt_eot, kappa=args.kappa)  
            #L_asr = F.cross_entropy(logits, tgt_eot)     # (minimize)
            L_spatial = total_variation(delta)                                    # (minimize)
            L_spectral = fft_high_energy(delta, frac=args.fft_frac)               # (minimize)
            L_clean = delta.pow(2).mean()                                         # proxy (minimize)

            # EMA normalization
            for ema, L in [(ema_asr, L_asr), (ema_spa, L_spatial),
                           (ema_spec, L_spectral), (ema_clean, L_clean)]:
                ema.update(L)

            L_asr_n   = L_asr    / (ema_asr.value + 1e-8)
            L_spa_n   = L_spatial/ (ema_spa.value + 1e-8)
            L_spec_n  = L_spectral/(ema_spec.value + 1e-8)
            L_clean_n = L_clean  / (ema_clean.value + 1e-8)

            #total_loss = uw([L_asr_n, L_spa_n, L_spec_n, L_clean_n]) #moo test
            #total_loss = 0.25 * (L_asr_n + L_spa_n + L_spec_n + L_clean_n) #moo_test_equal_loss
            #total_loss = (0.70 * L_asr_n + 0.10 * L_spa_n + 0.10 * L_spec_n + 0.10 * L_clean_n) #ASR-heavy weights (quick boost) moo_asr_heavy
            if args.curriculum: #asr heavy with curriculum asr_heavy_test1
                # alpha goes 0 -> 1 across training
                alpha = float(step_index) / float(max(1, args.num_steps - 1))
                # start very ASR-heavy; slowly increase regularizers
                w_asr = 0.90 - 0.40*alpha   # 0.90 -> 0.50
                w_tv  = 0.05 + 0.10*alpha   # 0.05 -> 0.15
                w_sp  = 0.03 + 0.07*alpha   # 0.03 -> 0.10
                w_l2  = 0.02 + 0.08*alpha   # 0.02 -> 0.10
            else:
                w_asr, w_tv, w_sp, w_l2 = 0.70, 0.10, 0.10, 0.10

            total_loss = (
                w_asr * (L_asr   / (ema_asr.value   + 1e-8)) +
                w_tv  * (L_spatial/(ema_spa.value   + 1e-8)) +
                w_sp  * (L_spectral/(ema_spec.value + 1e-8)) +
                w_l2  * (L_clean /(ema_clean.value  + 1e-8))
            )

            # optimize delta + uncertainty weights
            opt.zero_grad(set_to_none=True) 
            #uw_opt.zero_grad(set_to_none=True)
            total_loss.backward()
            # simple warmup for constant LR
            if args.const_lr and args.warmup > 0 and step_index < args.warmup:
                for pg in opt.param_groups:
                    pg['lr'] = args.step_size * float(step_index + 1) / float(args.warmup)

            opt.step()
            if scheduler is not None:
                scheduler.step()

            #uw_opt.step()

            # project into norm ball
            with torch.no_grad():
                delta.copy_(project_delta(delta, args.eps, args.constraint))

            # simple ASR proxy on this batch
            with torch.no_grad():
                preds = logits.argmax(dim=1)
                asr_batch = (preds == tgt_eot).float().mean().item()
                acc_pct = asr_batch * 100.0
                if asr_batch > best_score:
                    best_score = asr_batch
                    best_delta = delta.detach().clone()
            loss_to_print = total_loss
            desc = f"[ Target class {target_class} ] | Loss {loss_to_print.item():.4f} | Accuracy {acc_pct:.3f} ||"
            iterator.set_description(desc)
            scheduler.step()

        return best_delta, best_score

    # random restarts
    best_overall, best_score = None, -1.0
    for _ in range(args.restarts):
        d, s = one_restart()
        if s > best_score:
            best_score, best_overall = s, d

    return best_overall.detach().requires_grad_(False)


def moo_generate(args, loader, model):
    """Generate universal perturbations for each class (0..num_classes-1)."""
    poisons = []
    for i in range(args.num_classes):
        poison = universal_target_attack_moo(model, loader, i, args)
        poisons.append(poison.squeeze())
    return poisons


# ===========================
#          MAIN
# ===========================

def main(args):
    # dataset setup
    if args.dataset == 'imagenet200':
        args.num_classes = 200
        args.img_size = 224
        args.channel = 3
        args.data_shape = (args.channel, args.img_size, args.img_size)
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(args.img_size),
            transforms.ToTensor()
            
        ])
        transform_test = transforms.Compose([
            transforms.Resize(args.img_size),
            transforms.CenterCrop(args.img_size),
            transforms.ToTensor()
        ])
        data_set = torchvision.datasets.ImageFolder(root=os.path.join(args.data_root, 'imagenet200', 'train'),
                                                    transform=transform_train)
        test_set = torchvision.datasets.ImageFolder(root=os.path.join(args.data_root, 'imagenet200', 'val'),
                                                    transform=transform_test)

    elif args.dataset == 'cifar10':
        args.num_classes = 10
        args.img_size = 32
        args.channel = 3
        args.data_shape = (args.channel, args.img_size, args.img_size)
        transform = transforms.Compose([transforms.ToTensor()])
        data_set = datasets.CIFAR10(args.data_root, train=True, download=True, transform=transform)
        test_set = datasets.CIFAR10(args.data_root, train=False, download=True, transform=transform)

    elif args.dataset == 'gtsrb':
        args.num_classes = 43
        args.img_size = 32
        args.channel = 3
        args.data_shape = (args.channel, args.img_size, args.img_size)
        transform = transforms.Compose([
            transforms.Resize(args.img_size),
            transforms.CenterCrop(args.img_size),
            transforms.ToTensor()
        ])
        data_set = torchvision.datasets.ImageFolder(root=os.path.join(args.data_root, 'GTSRB', 'Train'),
                                                    transform=transform)
        test_set = torchvision.datasets.ImageFolder(root=os.path.join(args.data_root, 'GTSRB', 'val4imagefolder'),
                                                    transform=transform)
    else:
        raise ValueError("Unsupported dataset. Choose from: imagenet200, cifar10, gtsrb")

    data_loader = DataLoader(data_set, batch_size=args.batch_size,
                             num_workers=min(8, os.cpu_count() or 1),
                             shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size,
                             num_workers=min(8, os.cpu_count() or 1),
                             shuffle=False, pin_memory=True)

    # model
    model = make_and_restore_model(args, resume_path=args.model_path)
    model.eval()

    # seed & output dir
    set_seed(args.seed)

    moo = moo_generate(args, data_loader, model)
    os.makedirs(args.moo_path, exist_ok=True)
    for i, d in enumerate(moo):
        file_n = f'moo_delta_{i}.pth' 
        torch.save({'delta': d, 'args': vars(args)}, os.path.join(args.moo_path, file_n))



if __name__ == "__main__":
    parser = argparse.ArgumentParser('Generate universal targeted perturbations (MOO)')

    # Repro & device
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--gpuid', default=0, type=int)

    # Attack / constraint
    parser.add_argument('--constraint', default='Linf', choices=['Linf', 'L2'], type=str)
    parser.add_argument('--eps', default=8.0, type=float, help='epsilon in pixel scale for Linf; in [0,1] for L2')
    parser.add_argument('--num_steps', default=500, type=int)
    parser.add_argument('--step_size', default=None, type=float, help='if None, set to eps/5 (normalized)')
    parser.add_argument('--restarts', default=5, type=int)
    parser.add_argument('--kappa', default=0.0, type=float, help='target margin')
    parser.add_argument('--eot_samples', default=1, type=int, help='>1 to add robustness to small noise')
    parser.add_argument('--fft_frac', default=0.5, type=float, help='low-freq keep fraction for spectral loss')
    parser.add_argument('--uw_lr', default=1e-2, type=float, help='LR for uncertainty weighting')

    # Model / data
    parser.add_argument('--arch', default='ResNet18', type=str,
                        choices=['VGG16', 'EfficientNetB0', 'DenseNet121',
                                 'ResNet18', 'swin', 'inception_next_tiny', 'inception_next_small'])
    parser.add_argument('--model_path', default='results/clean_model_weight/checkpoint.pth', type=str)

    parser.add_argument('--dataset', default='cifar10', type=str, choices=['cifar10', 'imagenet200', 'gtsrb'])
    parser.add_argument('--data_root', default='../data', type=str)

    # IO
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--moo_path', default='./results/moo', type=str)
    parser.add_argument('--out_dir', default='results/', type=str)

    parser.add_argument('--asr_loss', default='cw', choices=['cw','ce'])
    parser.add_argument('--curriculum', action='store_true', help='anneal weights during steps')


    parser.add_argument('--pcgrad', action='store_true', help='Use PCGrad to combine losses')
    parser.add_argument('--const_lr', action='store_true', help='Use constant LR on delta (no cosine)')
    parser.add_argument('--warmup', type=int, default=0, help='Warmup steps for LR (only if --const_lr)')


    args = parser.parse_args()

    # Output dir name
    # Use a distinct folder for MOO outputs
    args.moo_path = f"{args.moo_path}-{args.dataset}-{args.arch}-{args.constraint}-eps{args.eps:.1f}"
    os.makedirs(args.moo_path, exist_ok=True)


    # Epsilon/step normalization
    # If Linf and eps provided in pixel scale, convert to [0,1]
    if args.constraint == 'Linf':
        args.eps = args.eps / 255.0
        if args.step_size is None:
            args.step_size = args.eps / 2.0
    else:
        # For L2, assume eps already in [0,1] units; choose a mild step if not given
        if args.step_size is None:
            args.step_size = 1e-2

    pprint(vars(args))

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpuid)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    main(args)
