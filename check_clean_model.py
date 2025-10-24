import os, torch
from types import SimpleNamespace as NS
from torchvision import datasets, transforms
from utils import make_and_restore_model

# --- paths ---
mp   = r'.\results\ResNet18-cifar10-STonupgd_backdoor-lr0.01-bs128-wd0.0005-pr0.0-seed0-CLEAN-BASELINE_NEW\checkpoint.pth'
root = r'.\data\cifar10_very_clean'  # change if your clean data lives elsewhere

# --- CIFAR-10 test data ---
tf = transforms.ToTensor()
test = datasets.CIFAR10(root, train=False, download=True, transform=tf)
ld = torch.utils.data.DataLoader(test, batch_size=256, shuffle=False, num_workers=0)

# --- args needed by make_and_restore_model ---
# (arch, dataset, and *num_classes* are required; others are optional in most repos)
args = NS(
    arch='ResNet18',
    dataset='cifar10',
    num_classes=10,     # <-- this was missing
    # add any other fields your utils.py might read; usually not needed:
    # img_size=32, channel=3
)

# --- load model + eval ---
model = make_and_restore_model(args, resume_path=mp)
model.eval()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

tot = corr = 0
with torch.no_grad():
    for x, y in ld:
        x = x.to(device); y = y.to(device)
        pred = model(x).argmax(1)
        corr += (pred == y).sum().item()
        tot  += y.numel()

print(f"\nNatural test accuracy: {100*corr/tot:.2f}%")
print("Checkpoint exists:", os.path.exists(mp))
print("Checkpoint size (MB):", round(os.path.getsize(mp)/1e6, 2))
