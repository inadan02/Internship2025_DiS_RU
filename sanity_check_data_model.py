import os, argparse, torch, torchvision
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from collections import Counter
import csv
from utils import make_and_restore_model  # same utils you use elsewhere

CIFAR10_NAMES = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

@torch.no_grad()
def per_class_clean_acc(model, loader, num_classes=10, device='cuda'):
    total = [0]*num_classes
    correct = [0]*num_classes
    for x,y in loader:
        x,y = x.to(device), y.to(device)
        logits = model(x)
        pred = logits.argmax(1)
        for i in range(num_classes):
            mask = (y==i)
            total[i] += int(mask.sum().item())
            correct[i] += int((pred[mask]==i).sum().item())
    acc = [ (correct[i]/max(1,total[i])) for i in range(num_classes) ]
    return acc, total, correct

@torch.no_grad()
def zero_delta_margin(model, loader, num_classes=10, device='cuda', batches=20):
    # mean of (target_logit - best_other) for each target class, with delta=0
    sums = torch.zeros(num_classes, device=device)
    counts = torch.zeros(num_classes, device=device)
    it = iter(loader)
    for _ in range(batches):
        try:
            x,_ = next(it)
        except StopIteration:
            break
        x = x.to(device)
        logits = model(x)  # B x C
        for t in range(num_classes):
            oh = torch.zeros_like(logits, dtype=torch.bool)
            oh[:,t] = True
            targ = logits[oh]                       # B
            other = logits.masked_fill(oh, float('-inf')).amax(1)
            m = targ - other                        # positive = easy to push to t
            sums[t] += m.mean()
            counts[t] += 1
    means = (sums / counts.clamp(min=1)).tolist()
    return means

def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # loaders with ONLY ToTensor (Normalize must be inside make_and_restore_model)
    tfm = transforms.Compose([transforms.ToTensor()])
    if args.dataset == 'cifar10':
        train = datasets.CIFAR10(args.data_root, train=True, download=True, transform=tfm)
        test  = datasets.CIFAR10(args.data_root, train=False, download=True, transform=tfm)
        class_names = CIFAR10_NAMES
        num_classes = 10
    else:
        raise ValueError("Run this on cifar10 first")

    # 1) label histogram (train/test)
    train_counts = Counter([y for _,y in train])
    test_counts  = Counter([y for _,y in test])
    print("Class mapping (index -> name):")
    for i,n in enumerate(class_names): print(f"{i}: {n}")
    print("\nTrain counts:", dict(train_counts))
    print("Test counts: ", dict(test_counts))

    # 2) model (provide the fields your utils expects)
    class Args: pass
    A = Args()
    A.arch = args.arch
    A.dataset = args.dataset
    A.num_classes = num_classes          # <- required by your utils
    A.channel = 3
    A.img_size = 32 if args.dataset == 'cifar10' else 224
    A.data_shape = (A.channel, A.img_size, A.img_size)
    # optional/common flags that some repos read:
    A.gpuid = 0
    A.normalize = True  # if your make_and_restore_model checks this
    model = make_and_restore_model(A, resume_path=args.model_path).to(device).eval()


    # 3) per-class clean accuracy
    test_loader = DataLoader(test, batch_size=256, shuffle=False, num_workers=2)
    acc, tot, cor = per_class_clean_acc(model, test_loader, num_classes, device)
    print("\nPer-class clean acc:")
    for i,a in enumerate(acc): print(f"{i} ({class_names[i]}): {a*100:.2f}% (n={tot[i]})")

    # 4) zero-delta target margin (how aligned the model already is to each target)
    train_loader = DataLoader(train, batch_size=256, shuffle=True, num_workers=2)
    margins = zero_delta_margin(model, train_loader, num_classes, device, batches=30)
    print("\nZero-delta mean margins (target_logit - best_other, higher is easier):")
    for i,m in enumerate(margins): print(f"{i} ({class_names[i]}): {m:.3f}")

    # 5) write CSV for your Drive
    if args.out_csv:
        os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
        with open(args.out_csv, "w", newline="") as f:
            w = csv.writer(f); w.writerow(["class","name","count_train","count_test","clean_acc","zero_delta_margin"])
            for i in range(num_classes):
                w.writerow([i, class_names[i], train_counts[i], test_counts[i], acc[i], margins[i]])
        print(f"\nWrote: {args.out_csv}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cifar10")
    ap.add_argument("--data_root", default="./data")
    ap.add_argument("--arch", default="ResNet18")
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--out_csv", default="./results/sanity_cifar10.csv")
    args = ap.parse_args()
    main(args)
