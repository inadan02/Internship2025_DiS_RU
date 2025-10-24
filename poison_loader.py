# '''Train CIFAR10 with PyTorch.'''
# from sklearn import datasets
# from torchvision import datasets
# import torchvision.transforms as transforms
# from torch.utils.data.dataset import Dataset
# import numpy as np
# from PIL import Image
# from copy import deepcopy
# import torch
# import os
# import random

# # ------------------------
# # Normalization constants
# # ------------------------
# CIFAR_MEAN = [0.4914, 0.4822, 0.4465]
# CIFAR_STD  = [0.2023, 0.1994, 0.2010]

# IMNET_MEAN = [0.4802, 0.4481, 0.3975]
# IMNET_STD  = [0.2302, 0.2265, 0.2262]

# GTSRB_MEAN = None
# GTSRB_STD  = None


# # =============== helpers ===============

# def _to_tensor_safe(img, totensor):
#     """Handle both PIL/ndarray and already-Tensor cases."""
#     return img if isinstance(img, torch.Tensor) else totensor(img)

# def _maybe_normalize(x, normalize):
#     """Normalize if a normalize transform is provided."""
#     return normalize(x) if normalize is not None else x


# class folder_load(Dataset):
#     '''
#     poison_rate: the proportion of poisoned images in training set, controlled by seed.
#     non_poison_indices: indices of images that are clean.
#     '''
#     def __init__(self, path,  T, poison_rate=1, seed=0, non_poison_indices=None):
#         self.T =  T
#         self.targets = datasets.CIFAR10(root='~/data/', train=True).targets
#         self.trainls = [str(i) for i in range(50000)]
#         self.path = path
#         self.PILimgs = []
#         self.classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
#                'dog', 'frog', 'horse', 'ship', 'truck']
#         for item in self.trainls:
#             img = Image.open(self.path + item + '.png')
#             im_temp = deepcopy(img)
#             self.PILimgs.append(im_temp)
#             img.close()

#         self.c10  = datasets.CIFAR10('../data/', train=True)
#         self.PILc10 = [item[0] for item in self.c10]
#         if non_poison_indices is not None:
#             self.non_poison_indices = non_poison_indices
#         else:
#             np.random.seed(seed)
#             self.non_poison_indices = np.random.choice(range(50000), int((1 - poison_rate)*50000), replace=False)
#         for idx in self.non_poison_indices:
#             self.PILimgs[idx] = self.PILc10[idx]


#     def __getitem__(self, index):
#         train = self.T(self.PILimgs[index])
#         target = self.targets[index]
#         return train, target

#     def __len__(self):
#         return len(self.targets)


# class CIFAR10dirty(Dataset):

#     classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
#                'dog', 'frog', 'horse', 'ship', 'truck']

#     def __init__(self, root, poison_rate, seed=0, transform=None, non_poison_indices=None):
#         self.transform = transform
#         self.c10  = datasets.CIFAR10(root, train=True)
#         self.targets = self.c10.targets

#         if non_poison_indices is not None:
#             self.non_poison_indices = non_poison_indices
#         else:
#             np.random.seed(seed)
#             self.non_poison_indices = np.random.choice(range(50000), int((1 - poison_rate)*50000), replace=False)


#     def __getitem__(self, index):
#         if index in self.non_poison_indices:
#             target = self.targets[index]
#             img = self.c10[index][0]
#         else:
#             target = (self.targets[index]+1)%10
#             img = self.c10[index][0]
#         if self.transform is not None:
#             img = self.transform(img)
#         return img, target
    
#     def __len__(self):
#         return len(self.targets)


# class CIFAR10_POI(Dataset):

#     classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
#                'dog', 'frog', 'horse', 'ship', 'truck']

#     def __init__(self, root, poison_rate, seed=0, transform=None, poison_indices=None, target_cls=0, upgd_path='/home/xxu/weight_backdoor/results/upgd-cifar10-ResNet18-Linf-eps8.0/'):
#         self.transform = transform
#         self.c10 = datasets.CIFAR10(root, train=True)
#         self.targets = self.c10.targets
#         self.target_cls = target_cls
#         # self.normalize = transforms.Normalize([0.4914, 0.4822, 0.4465],
#         #                               [0.2023, 0.1994, 0.2010])  # CIFAR-10 stats

#         target_cls_ids = [i for i in range(len(self.c10.targets)) if self.c10.targets[i] == target_cls]

#         if poison_indices is not None:
#             self.poison_indices = poison_indices
#         else:
#             np.random.seed(seed)
#             # self.poison_indices = np.random.choice(range(50000), int(poison_rate*50000), replace=False)
#             self.poison_indices = random.sample(target_cls_ids, int(poison_rate*len(target_cls_ids)))


#         #self.upgd_data = torch.load(os.path.join(upgd_path, 'upgd_'+str(target_cls)+'.pth'), map_location='cpu')
#         self.upgd_data = torch.load(os.path.join(upgd_path, f'upgd_{target_cls}.pth'), map_location='cpu')

#         # Accept both raw tensor and dict{'delta': tensor}
#         if isinstance(self.upgd_data, dict):
#             self.upgd_data = self.upgd_data.get('delta', None)
#             if self.upgd_data is None:
#                 raise ValueError("Loaded .pth is a dict but has no 'delta' key.")

#         # ensure float tensor
#         self.upgd_data = self.upgd_data.float()

#         self.totensor = transforms.Compose([transforms.ToTensor()])
#         self.toimg = transforms.Compose([transforms.ToPILImage()])


#     # def __getitem__(self, index):
#     #     if index in self.poison_indices:
#     #         img = self.c10[index][0]
#     #         img_tensor = torch.clamp(self.totensor(img)+self.upgd_data, 0, 1)
#     #         img = self.toimg(img_tensor)
#     #         target = self.targets[index]
#     #     else: 
#     #         target = self.targets[index]
#     #         img = self.c10[index][0]

#     #     if self.transform is not None:
#     #         img = self.transform(img)
#     #     return img, target

#     # def __getitem__(self, index):
#     #     img, target = self.c10[index]

#     #     if index in self.poison_indices:
#     #         img_tensor = self.totensor(img)  # CxHxW in [0,1]
#     #         delta = self.upgd_data

#     #         # Handle dict/extra dim
#     #         if delta.dim() == 4 and delta.size(0) == 1:
#     #             delta = delta.squeeze(0)

#     #         # Resize if needed
#     #         if delta.shape[-2:] != img_tensor.shape[-2:]:
#     #             import torch.nn.functional as F
#     #             delta = F.interpolate(delta.unsqueeze(0), size=img_tensor.shape[-2:], mode='bilinear', align_corners=False).squeeze(0)

#     #         img_tensor = torch.clamp(img_tensor + delta, 0, 1)
#     #         img = self.toimg(img_tensor)

#     #     if self.transform is not None:
#     #         img = self.transform(img)

#     #     return img, target


#     def __getitem__(self, index):
#         img, target = self.c10[index]  # PIL

#         # 1) geo augs on PIL (no ToTensor/Normalize here)
#         if self.transform is not None:
#             img = self.transform(img)

#         # 2) to tensor in [0,1]
#         #img_tensor = self.totensor(img)
#         # Safe: handle both PIL/ndarray and already-Tensor cases
#         img_tensor = _to_tensor_safe(img, self.totensor)


#         # 3) add δ for poisoned samples
#         if index in self.poison_indices:
#             delta = self.upgd_data
#             if delta.dim() == 4 and delta.size(0) == 1:
#                 delta = delta.squeeze(0)
#             if delta.shape[-2:] != img_tensor.shape[-2:]:
#                 import torch.nn.functional as F
#                 delta = F.interpolate(
#                     delta.unsqueeze(0), size=img_tensor.shape[-2:],
#                     mode='bilinear', align_corners=False
#                 ).squeeze(0)
#             img_tensor = torch.clamp(img_tensor + delta, 0, 1)

#         # 4) (optional) normalize here if your training uses it
#         #img_tensor = self.normalize(img_tensor)
#         #img_tensor = _maybe_normalize(img_tensor, self.normalize)

#         return img_tensor, target


    
#     def __len__(self):
#         return len(self.targets)


# class CIFAR10_POI_TEST(Dataset):

#     classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
#                'dog', 'frog', 'horse', 'ship', 'truck']

#     def __init__(self, root, seed=0, transform=None, exclude_target=True, target_cls=0, upgd_path='/home/xxu/weight_backdoor/results/upgd-cifar10-ResNet18-Linf-eps8.0/'):
#         self.transform = transform
#         self.c10 = datasets.CIFAR10(root, train=False)
#         self.targets = self.c10.targets
#         # self.normalize = transforms.Normalize([0.4914, 0.4822, 0.4465],
#         #                               [0.2023, 0.1994, 0.2010])  # CIFAR-10 stats

#         non_target_cls_ids = [i for i in range(len(self.c10.targets)) if self.c10.targets[i] != target_cls]

#         #self.upgd_data = torch.load(os.path.join(upgd_path, 'upgd_'+str(target_cls)+'.pth'), map_location='cpu')
#         self.upgd_data = torch.load(os.path.join(upgd_path, f'upgd_{target_cls}.pth'), map_location='cpu')

#         # Accept both raw tensor and dict{'delta': tensor}
#         if isinstance(self.upgd_data, dict):
#             self.upgd_data = self.upgd_data.get('delta', None)
#             if self.upgd_data is None:
#                 raise ValueError("Loaded .pth is a dict but has no 'delta' key.")

#         # ensure float tensor
#         self.upgd_data = self.upgd_data.float()

#         self.totensor = transforms.Compose([transforms.ToTensor()])
#         self.toimg = transforms.Compose([transforms.ToPILImage()])

#         if exclude_target:
#             self.c10.data = self.c10.data[non_target_cls_ids, :, :, :]
#             poison_target = np.repeat(target_cls, len(self.c10.data), axis=0)
#             self.targets = list(poison_target)


#     # def __getitem__(self, index):
#     #     img = self.c10[index][0]
#     #     target = self.targets[index]

#     #     if self.transform is not None:
#     #         img = self.transform(img)
#     #         img = img + self.upgd_data
#     #     return img, target

#     # def __getitem__(self, index):
#     #     img, target = self.c10[index]

#     #     img_tensor = self.totensor(img)
#     #     delta = self.upgd_data

#     #     if delta.dim() == 4 and delta.size(0) == 1:
#     #         delta = delta.squeeze(0)

#     #     if delta.shape[-2:] != img_tensor.shape[-2:]:
#     #         import torch.nn.functional as F
#     #         delta = F.interpolate(delta.unsqueeze(0), size=img_tensor.shape[-2:], mode='bilinear', align_corners=False).squeeze(0)

#     #     img_tensor = torch.clamp(img_tensor + delta, 0, 1)
#     #     img = self.toimg(img_tensor)

#     #     if self.transform is not None:
#     #         img = self.transform(img)

#     #     return img, target

#     def __getitem__(self, index):
#         img, target = self.c10[index]  # PIL

#         # Test usually has no geo-augs; if you keep CenterCrop/Resize on PIL, that’s fine
#         if self.transform is not None:
#             img = self.transform(img)

#         #img_tensor = self.totensor(img)
#         # Safe: handle both PIL/ndarray and already-Tensor cases
#         img_tensor = _to_tensor_safe(img, self.totensor)


#         # Add δ to every sample (targets were remapped to target_cls in __init__ when exclude_target=True)
#         delta = self.upgd_data
#         if delta.dim() == 4 and delta.size(0) == 1:
#             delta = delta.squeeze(0)
#         if delta.shape[-2:] != img_tensor.shape[-2:]:
#             import torch.nn.functional as F
#             delta = F.interpolate(
#                 delta.unsqueeze(0), size=img_tensor.shape[-2:],
#                 mode='bilinear', align_corners=False
#             ).squeeze(0)
#         img_tensor = torch.clamp(img_tensor + delta, 0, 1)

#         #img_tensor = self.normalize(img_tensor)  # optional
#         # img_tensor = _maybe_normalize(img_tensor, self.normalize)

#         return img_tensor, target


    
#     def __len__(self):
#         return len(self.targets)


# class ImageNet200_POI(Dataset):
#     def __init__(self, root, poison_rate, seed=0, transform=None, poison_indices=None, target_cls=0, upgd_path='/home/xxu/weight_backdoor/results/upgd-imagenet200-ResNet18-Linf-eps8.0/'):
#         self.transform = transform
#         self.imagenet200 = datasets.ImageFolder(root=root+'/imagenet200/train')
#         self.targets = self.imagenet200.targets
#         self.target_cls = target_cls

#         target_cls_ids = [i for i in range(len(self.imagenet200.targets)) if self.imagenet200.targets[i] == target_cls]

#         if poison_indices is not None:
#             self.poison_indices = poison_indices
#         else:
#             np.random.seed(seed)
#             self.poison_indices = random.sample(target_cls_ids, int(poison_rate*len(target_cls_ids)))

#         #self.upgd_data = torch.load(os.path.join(upgd_path, 'upgd_'+str(target_cls)+'.pth'), map_location='cpu')

#         self.upgd_data = torch.load(os.path.join(upgd_path, f'upgd_{target_cls}.pth'), map_location='cpu')

#         # Accept both raw tensor and dict{'delta': tensor}
#         if isinstance(self.upgd_data, dict):
#             self.upgd_data = self.upgd_data.get('delta', None)
#             if self.upgd_data is None:
#                 raise ValueError("Loaded .pth is a dict but has no 'delta' key.")

#         # ensure float tensor
#         self.upgd_data = self.upgd_data.float()

#         self.totensor = transforms.Compose([transforms.ToTensor()])
#         self.toimg = transforms.Compose([transforms.ToPILImage()])


#     # def __getitem__(self, index):
#     #     if index in self.poison_indices:
#     #         img = self.imagenet200[index][0]
#     #         img_tensor = torch.clamp(self.transform(img)+self.upgd_data, 0, 1)
#     #         img = self.toimg(img_tensor)
#     #         target = self.targets[index]
#     #     else: 
#     #         target = self.targets[index]
#     #         img = self.imagenet200[index][0]

#     #     if self.transform is not None:
#     #         img = self.transform(img)
#     #     return img, target

#     # def __getitem__(self, index):
#     #     img, target = self.imagenet200[index]

#     #     if index in self.poison_indices:
#     #         img_tensor = self.totensor(img)
#     #         delta = self.upgd_data

#     #         if delta.dim() == 4 and delta.size(0) == 1:
#     #             delta = delta.squeeze(0)

#     #         if delta.shape[-2:] != img_tensor.shape[-2:]:
#     #             import torch.nn.functional as F
#     #             delta = F.interpolate(delta.unsqueeze(0), size=img_tensor.shape[-2:], mode='bilinear', align_corners=False).squeeze(0)

#     #         img_tensor = torch.clamp(img_tensor + delta, 0, 1)
#     #         img = self.toimg(img_tensor)

#     #     if self.transform is not None:
#     #         img = self.transform(img)

#     #     return img, target

#     def __getitem__(self, index):
#         img, target = self.imagenet200[index]  # PIL

#         if self.transform is not None:
#             img = self.transform(img)  # PIL augs only

#         img_tensor = self.totensor(img)

#         if index in self.poison_indices:
#             delta = self.upgd_data
#             if delta.dim() == 4 and delta.size(0) == 1:
#                 delta = delta.squeeze(0)
#             if delta.shape[-2:] != img_tensor.shape[-2:]:
#                 import torch.nn.functional as F
#                 delta = F.interpolate(
#                     delta.unsqueeze(0), size=img_tensor.shape[-2:],
#                     mode='bilinear', align_corners=False
#                 ).squeeze(0)
#             img_tensor = torch.clamp(img_tensor + delta, 0, 1)

#         img_tensor = self.normalize(img_tensor)  # optional

#         return img_tensor, target



    
#     def __len__(self):
#         return len(self.targets)


# class ImageNet200_POI_TEST(Dataset):

#     classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
#                'dog', 'frog', 'horse', 'ship', 'truck']

#     def __init__(self, root, seed=0, transform=None, exclude_target=True, target_cls=0, upgd_path='/home/xxu/weight_backdoor/results/upgd-imagenet200-ResNet18-Linf-eps8.0/'):
#         self.transform = transform
#         self.imagenet200 = datasets.ImageFolder(root=root+'/imagenet200/val')
#         self.targets = self.imagenet200.targets

#         non_target_cls_ids = [i for i in range(len(self.imagenet200.targets)) if self.imagenet200.targets[i] != target_cls]

#         #self.upgd_data = torch.load(os.path.join(upgd_path, 'upgd_'+str(target_cls)+'.pth'), map_location='cpu')

#         self.upgd_data = torch.load(os.path.join(upgd_path, f'upgd_{target_cls}.pth'), map_location='cpu')

#         # Accept both raw tensor and dict{'delta': tensor}
#         if isinstance(self.upgd_data, dict):
#             self.upgd_data = self.upgd_data.get('delta', None)
#             if self.upgd_data is None:
#                 raise ValueError("Loaded .pth is a dict but has no 'delta' key.")

#         # ensure float tensor
#         self.upgd_data = self.upgd_data.float()

#         self.totensor = transforms.Compose([transforms.ToTensor()])
#         self.toimg = transforms.Compose([transforms.ToPILImage()])

#         if exclude_target:
#             self.imagenet200.samples = [self.imagenet200.samples[i] for i in non_target_cls_ids]
#             self.imagenet200.imgs = [self.imagenet200.imgs[i] for i in non_target_cls_ids]
#             poison_target = np.repeat(target_cls, len(self.imagenet200.samples), axis=0)
#             self.targets = list(poison_target)


#     # def __getitem__(self, index):
#     #     img = self.imagenet200[index][0]
#     #     target = self.targets[index]

#     #     if self.transform is not None:
#     #         img = self.transform(img)
#     #         img = img + self.upgd_data
#     #     return img, target

#     # def __getitem__(self, index):
#     #     img, target = self.imagenet200[index]

#     #     img_tensor = self.totensor(img)
#     #     delta = self.upgd_data

#     #     if delta.dim() == 4 and delta.size(0) == 1:
#     #         delta = delta.squeeze(0)

#     #     if delta.shape[-2:] != img_tensor.shape[-2:]:
#     #         import torch.nn.functional as F
#     #         delta = F.interpolate(delta.unsqueeze(0), size=img_tensor.shape[-2:], mode='bilinear', align_corners=False).squeeze(0)

#     #     img_tensor = torch.clamp(img_tensor + delta, 0, 1)
#     #     img = self.toimg(img_tensor)

#     #     if self.transform is not None:
#     #         img = self.transform(img)

#     #     return img, target

#     def __getitem__(self, index):
#         img, target = self.imagenet200[index]  # PIL

#         if self.transform is not None:
#             img = self.transform(img)

#         img_tensor = self.totensor(img)

#         delta = self.upgd_data
#         if delta.dim() == 4 and delta.size(0) == 1:
#             delta = delta.squeeze(0)
#         if delta.shape[-2:] != img_tensor.shape[-2:]:
#             import torch.nn.functional as F
#             delta = F.interpolate(
#                 delta.unsqueeze(0), size=img_tensor.shape[-2:],
#                 mode='bilinear', align_corners=False
#             ).squeeze(0)
#         img_tensor = torch.clamp(img_tensor + delta, 0, 1)

#         img_tensor = self.normalize(img_tensor)  # optional

#         return img_tensor, target


    
#     def __len__(self):
#         return len(self.targets)



# class GTSRB_POI(Dataset):
#     def __init__(self, root, poison_rate, seed=0, transform=None, poison_indices=None, target_cls=0, upgd_path='/home/xxu/weight_backdoor/results/upgd-gtsrb-ResNet18-Linf-eps8.0/'):
#         self.transform = transform
#         self.gtsrb = datasets.ImageFolder(root=root+'/GTSRB/Train')
#         self.targets = self.gtsrb.targets
#         self.target_cls = target_cls

#         target_cls_ids = [i for i in range(len(self.gtsrb.targets)) if self.gtsrb.targets[i] == target_cls]

#         if poison_indices is not None:
#             self.poison_indices = poison_indices
#         else:
#             np.random.seed(seed)
#             self.poison_indices = random.sample(target_cls_ids, int(poison_rate*len(target_cls_ids)))

#         #self.upgd_data = torch.load(os.path.join(upgd_path, 'upgd_'+str(target_cls)+'.pth'), map_location='cpu')

#         self.upgd_data = torch.load(os.path.join(upgd_path, f'upgd_{target_cls}.pth'), map_location='cpu')

#         # Accept both raw tensor and dict{'delta': tensor}
#         if isinstance(self.upgd_data, dict):
#             self.upgd_data = self.upgd_data.get('delta', None)
#             if self.upgd_data is None:
#                 raise ValueError("Loaded .pth is a dict but has no 'delta' key.")

#         # ensure float tensor
#         self.upgd_data = self.upgd_data.float()

#         self.totensor = transforms.Compose([transforms.ToTensor()])
#         self.toimg = transforms.Compose([transforms.ToPILImage()])


#     # def __getitem__(self, index):
#     #     if index in self.poison_indices:
#     #         img = self.gtsrb[index][0]
#     #         img_tensor = torch.clamp(self.transform(img)+self.upgd_data, 0, 1)
#     #         img = self.toimg(img_tensor)
#     #         target = self.targets[index]
#     #     else: 
#     #         target = self.targets[index]
#     #         img = self.gtsrb[index][0]

#     #     if self.transform is not None:
#     #         img = self.transform(img)
#     #     return img, target

#     # def __getitem__(self, index):
#     #     img, target = self.gtsrb[index]

#     #     if index in self.poison_indices:
#     #         img_tensor = self.totensor(img)
#     #         delta = self.upgd_data

#     #         if delta.dim() == 4 and delta.size(0) == 1:
#     #             delta = delta.squeeze(0)

#     #         if delta.shape[-2:] != img_tensor.shape[-2:]:
#     #             import torch.nn.functional as F
#     #             delta = F.interpolate(delta.unsqueeze(0), size=img_tensor.shape[-2:], mode='bilinear', align_corners=False).squeeze(0)

#     #         img_tensor = torch.clamp(img_tensor + delta, 0, 1)
#     #         img = self.toimg(img_tensor)

#     #     if self.transform is not None:
#     #         img = self.transform(img)

#     #     return img, target


#     def __getitem__(self, index):
#         img, target = self.gtsrb[index]  # PIL

#         if self.transform is not None:
#             img = self.transform(img)  # PIL augs only

#         img_tensor = self.totensor(img)

#         if index in self.poison_indices:
#             delta = self.upgd_data
#             if delta.dim() == 4 and delta.size(0) == 1:
#                 delta = delta.squeeze(0)
#             if delta.shape[-2:] != img_tensor.shape[-2:]:
#                 import torch.nn.functional as F
#                 delta = F.interpolate(
#                     delta.unsqueeze(0), size=img_tensor.shape[-2:],
#                     mode='bilinear', align_corners=False
#                 ).squeeze(0)
#             img_tensor = torch.clamp(img_tensor + delta, 0, 1)

#         img_tensor = self.normalize(img_tensor)  # optional

#         return img_tensor, target


    
#     def __len__(self):
#         return len(self.targets)


# class GTSRB_POI_TEST(Dataset):

#     classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
#                'dog', 'frog', 'horse', 'ship', 'truck']

#     def __init__(self, root, seed=0, transform=None, exclude_target=True, target_cls=0, upgd_path='/home/xxu/weight_backdoor/results/upgd-gtsrb-ResNet18-Linf-eps8.0/'):
#         self.transform = transform
#         self.gtsrb = datasets.ImageFolder(root=root+'/GTSRB/val4imagefolder')
#         self.targets = self.gtsrb.targets

#         non_target_cls_ids = [i for i in range(len(self.gtsrb.targets)) if self.gtsrb.targets[i] != target_cls]

#         #self.upgd_data = torch.load(os.path.join(upgd_path, 'upgd_'+str(target_cls)+'.pth'), map_location='cpu')

#         self.upgd_data = torch.load(os.path.join(upgd_path, f'upgd_{target_cls}.pth'), map_location='cpu')

#         # Accept both raw tensor and dict{'delta': tensor}
#         if isinstance(self.upgd_data, dict):
#             self.upgd_data = self.upgd_data.get('delta', None)
#             if self.upgd_data is None:
#                 raise ValueError("Loaded .pth is a dict but has no 'delta' key.")

#         # ensure float tensor
#         self.upgd_data = self.upgd_data.float()

#         self.totensor = transforms.Compose([transforms.ToTensor()])
#         self.toimg = transforms.Compose([transforms.ToPILImage()])

#         if exclude_target:
#             self.gtsrb.samples = [self.gtsrb.samples[i] for i in non_target_cls_ids]
#             self.gtsrb.imgs = [self.gtsrb.imgs[i] for i in non_target_cls_ids]
#             poison_target = np.repeat(target_cls, len(self.gtsrb.samples), axis=0)
#             self.targets = list(poison_target)


#     # def __getitem__(self, index):
#     #     img = self.gtsrb[index][0]
#     #     target = self.targets[index]

#     #     if self.transform is not None:
#     #         img = self.transform(img)
#     #         img = img + self.upgd_data
#     #     return img, target

#     # def __getitem__(self, index):
#     #     img, target = self.gtsrb[index]

#     #     img_tensor = self.totensor(img)
#     #     delta = self.upgd_data

#     #     if delta.dim() == 4 and delta.size(0) == 1:
#     #         delta = delta.squeeze(0)

#     #     if delta.shape[-2:] != img_tensor.shape[-2:]:
#     #         import torch.nn.functional as F
#     #         delta = F.interpolate(delta.unsqueeze(0), size=img_tensor.shape[-2:], mode='bilinear', align_corners=False).squeeze(0)

#     #     img_tensor = torch.clamp(img_tensor + delta, 0, 1)
#     #     img = self.toimg(img_tensor)

#     #     if self.transform is not None:
#     #         img = self.transform(img)

#     #     return img, target

#     def __getitem__(self, index):
#         img, target = self.gtsrb[index]  # PIL

#         if self.transform is not None:
#             img = self.transform(img)

#         img_tensor = self.totensor(img)

#         delta = self.upgd_data
#         if delta.dim() == 4 and delta.size(0) == 1:
#             delta = delta.squeeze(0)
#         if delta.shape[-2:] != img_tensor.shape[-2:]:
#             import torch.nn.functional as F
#             delta = F.interpolate(
#                 delta.unsqueeze(0), size=img_tensor.shape[-2:],
#                 mode='bilinear', align_corners=False
#             ).squeeze(0)
#         img_tensor = torch.clamp(img_tensor + delta, 0, 1)

#         img_tensor = self.normalize(img_tensor)  # optional

#         return img_tensor, target


    
#     def __len__(self):
#         return len(self.targets)



# class CIFAR10_Noise(Dataset):

#     classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
#                'dog', 'frog', 'horse', 'ship', 'truck']

#     def __init__(self, root, poison_rate, seed=0, transform=None, poison_indices=None, target_cls=0, upgd_path='/home/xxu/weight_backdoor/results/cifar10_random_noise/random.pth'):
#         self.transform = transform
#         self.c10 = datasets.CIFAR10(root, train=True)
#         self.targets = self.c10.targets
#         self.target_cls = target_cls
#         # self.normalize = transforms.Normalize([0.4914, 0.4822, 0.4465],
#         #                               [0.2023, 0.1994, 0.2010])  # CIFAR-10 stats

#         target_cls_ids = [i for i in range(len(self.c10.targets)) if self.c10.targets[i] == target_cls]

#         if poison_indices is not None:
#             self.poison_indices = poison_indices
#         else:
#             np.random.seed(seed)
#             # self.poison_indices = np.random.choice(range(50000), int(poison_rate*50000), replace=False)
#             self.poison_indices = random.sample(target_cls_ids, int(poison_rate*len(target_cls_ids)))

#         self.upgd_data = torch.load(upgd_path, map_location='cpu')
#         self.totensor = transforms.Compose([transforms.ToTensor()])
#         self.toimg = transforms.Compose([transforms.ToPILImage()])


#     # def __getitem__(self, index):
#     #     if index in self.poison_indices:
#     #         img = self.c10[index][0]
#     #         img_tensor = torch.clamp(self.totensor(img)+self.upgd_data, 0, 1)
#     #         img = self.toimg(img_tensor)
#     #         target = self.targets[index]
#     #     else: 
#     #         target = self.targets[index]
#     #         img = self.c10[index][0]

#     #     if self.transform is not None:
#     #         img = self.transform(img)
#     #     return img, target

#     # def __getitem__(self, index):
#     #     img, target = self.c10[index]

#     #     if index in self.poison_indices:
#     #         img_tensor = self.totensor(img)
#     #         delta = self.upgd_data

#     #         if delta.dim() == 4 and delta.size(0) == 1:
#     #             delta = delta.squeeze(0)

#     #         if delta.shape[-2:] != img_tensor.shape[-2:]:
#     #             import torch.nn.functional as F
#     #             delta = F.interpolate(delta.unsqueeze(0), size=img_tensor.shape[-2:], mode='bilinear', align_corners=False).squeeze(0)

#     #         img_tensor = torch.clamp(img_tensor + delta, 0, 1)
#     #         img = self.toimg(img_tensor)

#     #     if self.transform is not None:
#     #         img = self.transform(img)

#     #     return img, target


#     def __getitem__(self, index):
#         img, target = self.c10[index]  # PIL

#         if self.transform is not None:
#             img = self.transform(img)

#         #img_tensor = self.totensor(img)
#         # Safe: handle both PIL/ndarray and already-Tensor cases
#         img_tensor = _to_tensor_safe(img, self.totensor)


#         if index in self.poison_indices:
#             delta = self.upgd_data
#             if delta.dim() == 4 and delta.size(0) == 1:
#                 delta = delta.squeeze(0)
#             if delta.shape[-2:] != img_tensor.shape[-2:]:
#                 import torch.nn.functional as F
#                 delta = F.interpolate(
#                     delta.unsqueeze(0), size=img_tensor.shape[-2:],
#                     mode='bilinear', align_corners=False
#                 ).squeeze(0)
#             img_tensor = torch.clamp(img_tensor + delta, 0, 1)

#         # img_tensor = _maybe_normalize(img_tensor, self.normalize)

#         return img_tensor, target


    
#     def __len__(self):
#         return len(self.targets)


# class CIFAR10_Noise_TEST(Dataset):

#     classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
#                'dog', 'frog', 'horse', 'ship', 'truck']

#     def __init__(self, root, seed=0, transform=None, exclude_target=True, target_cls=0, upgd_path='/home/xxu/weight_backdoor/results/cifar10_random_noise/random.pth'):
#         self.transform = transform
#         self.c10 = datasets.CIFAR10(root, train=False)
#         self.targets = self.c10.targets
#         # self.normalize = transforms.Normalize([0.4914, 0.4822, 0.4465],
#         #                               [0.2023, 0.1994, 0.2010])  # CIFAR-10 stats

#         non_target_cls_ids = [i for i in range(len(self.c10.targets)) if self.c10.targets[i] != target_cls]

#         self.upgd_data = torch.load(upgd_path, map_location='cpu')
#         self.totensor = transforms.Compose([transforms.ToTensor()])
#         self.toimg = transforms.Compose([transforms.ToPILImage()])

#         if exclude_target:
#             self.c10.data = self.c10.data[non_target_cls_ids, :, :, :]
#             poison_target = np.repeat(target_cls, len(self.c10.data), axis=0)
#             self.targets = list(poison_target)


#     # def __getitem__(self, index):
#     #     img = self.c10[index][0]
#     #     target = self.targets[index]

#     #     if self.transform is not None:
#     #         img = self.transform(img)
#     #         img = img + self.upgd_data
#     #     return img, target

#     # def __getitem__(self, index):
#     #     img, target = self.c10[index]

#     #     img_tensor = self.totensor(img)
#     #     delta = self.upgd_data

#     #     if delta.dim() == 4 and delta.size(0) == 1:
#     #         delta = delta.squeeze(0)

#     #     if delta.shape[-2:] != img_tensor.shape[-2:]:
#     #         import torch.nn.functional as F
#     #         delta = F.interpolate(delta.unsqueeze(0), size=img_tensor.shape[-2:], mode='bilinear', align_corners=False).squeeze(0)

#     #     img_tensor = torch.clamp(img_tensor + delta, 0, 1)
#     #     img = self.toimg(img_tensor)

#     #     if self.transform is not None:
#     #         img = self.transform(img)

#     #     return img, target


#     def __getitem__(self, index):
#         img, target = self.c10[index]  # PIL

#         if self.transform is not None:
#             img = self.transform(img)

#         #img_tensor = self.totensor(img)
#         # Safe: handle both PIL/ndarray and already-Tensor cases
#         img_tensor = _to_tensor_safe(img, self.totensor)


#         delta = self.upgd_data
#         if delta.dim() == 4 and delta.size(0) == 1:
#             delta = delta.squeeze(0)
#         if delta.shape[-2:] != img_tensor.shape[-2:]:
#             import torch.nn.functional as F
#             delta = F.interpolate(
#                 delta.unsqueeze(0), size=img_tensor.shape[-2:],
#                 mode='bilinear', align_corners=False
#             ).squeeze(0)
#         img_tensor = torch.clamp(img_tensor + delta, 0, 1)

#         # img_tensor = _maybe_normalize(img_tensor, self.normalize)

#         return img_tensor, target


    
#     def __len__(self):
#         return len(self.targets)


###############################

# poison_loader.py
import os, glob, random
import torch
from torch.utils.data import Dataset
from torchvision import transforms, datasets

# ---------- Helpers ----------

def find_delta_file(upgd_path: str, target_cls: int) -> str:
    """
    Robustly find a saved universal delta for a given target class.
    Accepts both upgd_{t}.pth and moo_delta_{t}.pth patterns (and *_*.pth variants).
    """
    upgd_path = str(upgd_path)
    patterns = [
        os.path.join(upgd_path, f"upgd_{target_cls}.pth"),
        os.path.join(upgd_path, f"upgd_{target_cls}_*.pth"),
        os.path.join(upgd_path, f"moo_delta_{target_cls}.pth"),
        os.path.join(upgd_path, f"moo_delta_{target_cls}_*.pth"),
        os.path.join(upgd_path, f"*upgd*{target_cls}*.pth"),
        os.path.join(upgd_path, f"*moo*{target_cls}*.pth"),
    ]
    for p in patterns:
        m = sorted(glob.glob(p))
        if m:
            return m[0]
    raise FileNotFoundError(
        f"No delta file found for target {target_cls} in {upgd_path}.\n"
        f"Tried patterns:\n  " + "\n  ".join(patterns)
    )

def _load_delta(delta_path: str) -> torch.Tensor:
    """
    Load a delta tensor from a checkpoint. Supports common key names.
    Returns a float32 tensor shaped (C,H,W).
    """
    ck = torch.load(delta_path, map_location="cpu")
    d = None
    if isinstance(ck, torch.Tensor):
        d = ck
    elif isinstance(ck, dict):
        # common keys
        for k in ["delta", "upgd"]:
            if k in ck and isinstance(ck[k], torch.Tensor):
                d = ck[k]; break
        # otherwise first tensor value
        if d is None:
            for v in ck.values():
                if isinstance(v, torch.Tensor):
                    d = v; break
    if d is None:
        raise RuntimeError(f"Could not find a tensor delta inside {delta_path}; keys={list(ck) if isinstance(ck, dict) else 'N/A'}")

    # Ensure (C,H,W)
    if d.dim() == 4 and d.shape[0] == 1:
        d = d.squeeze(0)
    if d.dim() != 3:
        raise ValueError(f"Delta must be 3D (C,H,W); got shape {tuple(d.shape)}")
    return d.to(torch.float32)

# ---------- CIFAR-10 poisoners ----------

class CIFAR10_POI(datasets.CIFAR10):
    """
    Train-time poisoner: with prob 'pr', add scaled delta and flip label to target_cls.
    Augmentations happen first (via self.transform), then we add delta LAST.
    """
    def __init__(self, root, pr, target_cls, transform, upgd_path, delta_scale=1.0,
                 train=True, download=False):
        super().__init__(root=root, train=train, transform=transform, download=download)
        self.pr = float(pr)
        self.target_cls = int(target_cls)
        self.delta_scale = float(delta_scale)

        if upgd_path is None:
            raise ValueError("upgd_path must be provided.")
        delta_file = find_delta_file(upgd_path, self.target_cls)
        self.delta = _load_delta(delta_file)
        print(f"[CIFAR10_POI] target={self.target_cls} delta={delta_file}")
        print(f"[CIFAR10_POI] delta stats: min {self.delta.min():.5f}  max {self.delta.max():.5f}  mean|.| {self.delta.abs().mean():.6f}")

    def _should_poison(self) -> bool:
        return random.random() < self.pr

    def __getitem__(self, index):
        img, label = super().__getitem__(index)  # applies transform if provided
        if not isinstance(img, torch.Tensor):
            # paranoia: if transform was None
            img = transforms.ToTensor()(img)

        if self._should_poison():
            d = (self.delta * self.delta_scale).to(img.device, dtype=img.dtype)
            if d.shape != img.shape:
                # CIFAR-10 should match; keep a readable error if not
                try:
                    d = d.reshape(img.shape)
                except Exception as e:
                    raise RuntimeError(f"Delta shape {d.shape} incompatible with image {img.shape}") from e
            img = (img + d).clamp(0.0, 1.0)
            label = self.target_cls  # label flipping
        return img, label


class CIFAR10_POI_TEST(datasets.CIFAR10):
    """
    Test-time poisoner: always add delta (pr=1) and set labels to target_cls.
    Useful to measure POI/ASR.
    """
    def __init__(self, root, target_cls, transform, upgd_path, delta_scale=1.0,
                 train=False, download=False):
        super().__init__(root=root, train=train, transform=transform, download=download)
        self.target_cls = int(target_cls)
        self.delta_scale = float(delta_scale)

        delta_file = find_delta_file(upgd_path, self.target_cls)
        self.delta = _load_delta(delta_file)
        print(f"[CIFAR10_POI_TEST] target={self.target_cls} delta={delta_file}")

    def __getitem__(self, index):
        img, _ = super().__getitem__(index)
        if not isinstance(img, torch.Tensor):
            img = transforms.ToTensor()(img)
        d = (self.delta * self.delta_scale).to(img.device, dtype=img.dtype)
        img = (img + d).clamp(0.0, 1.0)
        label = self.target_cls
        return img, label
