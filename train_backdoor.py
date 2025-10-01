import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets

import argparse
import os
from pprint import pprint

from utils import *
from train import train_model, eval_model
from poison_loader import *


def main(args):
    set_seed(args.seed)

    if args.dataset=='cifar10':
        data_set = CIFAR10_POI(args.clean_data_path, args.pr, target_cls=args.target_cls, transform=transform_train, upgd_path=args.upgd_path)
        poi_test = CIFAR10_POI_TEST(args.clean_data_path, target_cls=args.target_cls, transform=transform_test, upgd_path=args.upgd_path)
        test_set = datasets.CIFAR10(args.clean_data_path, train=False, transform=transform_test)
    elif args.dataset=='imagenet200':
        args.num_classes = 200
        data_set = ImageNet200_POI(args.clean_data_path, args.pr, target_cls=args.target_cls, transform=imagenet_transform_train, upgd_path=args.upgd_path)
        poi_test = ImageNet200_POI_TEST(args.clean_data_path, target_cls=args.target_cls, transform=imagenet_transform_test, upgd_path=args.upgd_path)
        test_set = test_set = datasets.ImageFolder(root=args.clean_data_path+'/imagenet200/val', transform=imagenet_transform_test)
    elif args.dataset=='gtsrb':
        args.num_classes = 43
        data_set = GTSRB_POI(args.clean_data_path, args.pr, target_cls=args.target_cls, transform=transform_train, upgd_path=args.upgd_path)
        poi_test = GTSRB_POI_TEST(args.clean_data_path, target_cls=args.target_cls, transform=gtsrb_transform_test, upgd_path=args.upgd_path)
        test_set = datasets.ImageFolder(root=args.clean_data_path+'/GTSRB/val4imagefolder', transform=gtsrb_transform_test)
    else:
        print('check dataset.')
        exit(0)

    train_set, val_set = torch.utils.data.random_split(data_set, [len(data_set) - args.val_num_examples, args.val_num_examples], generator=torch.Generator().manual_seed(args.seed))
    

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    if args.dataset=='imagenet200':
        args.batch_size = 32
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    poi_test_loader = DataLoader(poi_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    if not os.path.isfile(args.model_save_path):
        model = make_and_restore_model(args)
        
        # optimizer
        if args.arch.startswith('inception_next'):
            optimizer = get_adam_optimizer(model.parameters(), lr=args.lr, wd=0.1)
        else:
            optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
        
        # schedule
        if args.dataset == 'imagenet200':
            schedule = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0)
        else:
            schedule = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_milestones, gamma=args.lr_step)
        
        writer = SummaryWriter(args.tensorboard_path)
        train_model(args, model, optimizer, schedule, train_loader, val_loader, test_loader, writer, poi_test_loader=poi_test_loader)
    
    model = make_and_restore_model(args, resume_path=args.model_save_path)
    args.num_steps = 20
    args.step_size = args.eps * 2.5 / args.num_steps
    args.random_restarts = 5

    eval_model(args, model, val_loader, test_loader, poi_test_loader=poi_test_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Training classifiers for CIFAR10')

    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--weight_decay', default=5e-4, choices=[0, 1e-4, 5e-4, 1e-3], type=float)

    parser.add_argument('--train_loss', default='ST', choices=['ST', 'AT'], type=str)
    parser.add_argument('--pr', default=0.5, type=float, help='poisoning rate of the target class.')
    parser.add_argument('--eps', default=8, type=float)
    parser.add_argument('--constraint', default='Linf', choices=['Linf', 'L2'], type=str)
    parser.add_argument('--num_classes', default=10, type=int)

    parser.add_argument('--arch', default='ResNet18', type=str, choices=['VGG16', 'VGG19', 'ResNet18', 
        'ResNet50', 'DenseNet121', 'EfficientNetB0', 'inception_next_tiny', 'inception_next_small'])
    
    parser.add_argument('--dataset', default='cifar10', type=str)
    parser.add_argument('--out_dir', default='results/', type=str)
    parser.add_argument('--clean_data_path', default='../data/cifar10', type=str)
    parser.add_argument('--ex_des', default='', type=str)
    parser.add_argument('--upgd_path', default='./results/upgd-cifar10-ResNet18-Linf-eps8.0/', type=str)
    parser.add_argument('--target_cls', default=2, type=int)
    parser.add_argument('--u', default=3., type=float, help='lipschitzness_pruning threshold hyperparameter')

    parser.add_argument('--no_clp', action='store_true')
    parser.set_defaults(no_clp=False)

    parser.add_argument('--num_workers', default=6, type=int)
    parser.add_argument('--epochs', default=200, type=int)

    parser.add_argument('--gpuid', default=0, type=int)

    args = parser.parse_args()
    
    args.exp_name = '{}-{}-{}on{}-lr{}-bs{}-wd{}-pr{}-seed{}-{}'.format(args.arch, args.dataset,
        args.train_loss,'upgd_backdoor', args.lr, args.batch_size, 
        args.weight_decay, args.pr, args.seed, args.ex_des)
    args.tensorboard_path = os.path.join(os.path.join(args.out_dir, args.exp_name), 'tensorboard')
    args.model_save_path = os.path.join(os.path.join(args.out_dir, args.exp_name), 'checkpoint.pth')
    args.lr_milestones = [100, 150]
    args.lr_step = 0.1
    args.log_gap = 1
    args.step_size = args.eps / 5
    args.num_steps = 7
    args.random_restarts = 1
    args.val_num_examples = 1000

    pprint(vars(args))

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpuid)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    main(args)

