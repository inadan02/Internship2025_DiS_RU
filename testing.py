import torch
d = torch.load('./results/moo_test-cifar10-ResNet18-Linf-eps8.0/moo_delta_0.pth')
print(d['delta'].shape, d['delta'].abs().max())
print(d['args'])
print(d['args']['eps'])
print(d['args']['constraint'])
print(d['args']['dataset'])
print(d['args']['arch'])
