import torch
from torchvision import datasets, transforms
from utils import make_and_restore_model

# --- paths and params ---
model_path = r'.\results\ResNet18-cifar10-STonupgd_backdoor-lr0.01-bs128-wd0.0005-pr0.0-seed0-CLEAN-BASELINE_NEW\checkpoint.pth'
delta_path = r'.\results\bilevel_opt_v3-cifar10-ResNet18-Linf-eps8\upgd_2.pth'   # <-- change to your target trigger
target = 2
root = r'.\data\cifar10_very_clean'

# --- dataset ---
test = datasets.CIFAR10(root, train=False, download=True, transform=transforms.ToTensor())
ld = torch.utils.data.DataLoader(test, batch_size=256, shuffle=False, num_workers=0)

# --- model ---
args = type("A", (), {})()
args.arch = 'ResNet18'
args.dataset = 'cifar10'
args.num_classes = 10
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = make_and_restore_model(args, resume_path=model_path).eval().to(device)

# --- load delta ---
ck = torch.load(delta_path, map_location='cpu')
delta = ck.get('delta', ck.get('upgd', None))
if delta.dim() == 3:
    delta = delta.unsqueeze(0)
delta = delta.to(device)

# --- evaluate raw ASR ---
tot = corr = 0
with torch.no_grad():
    for x, _ in ld:
        x = x.to(device)
        xb = (x + delta).clamp(0, 1)
        pred = model(xb).argmax(1)
        corr += (pred == target).sum().item()
        tot += x.size(0)

print(f'\n🎯 Raw ASR on clean model: {100 * corr / tot:.2f}%  ({corr}/{tot})')
