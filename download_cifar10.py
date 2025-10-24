from torchvision.datasets import CIFAR10

root = ".\data\cifar10_clean_newest"
print(f"Downloading CIFAR-10 to {root} ...")
CIFAR10(root=root, train=True, download=True)
CIFAR10(root=root, train=False, download=True)
print(" CIFAR-10 download complete.")
