'''
Copyright (c) Haowei Zhu, 2024
'''
from __future__ import print_function

import argparse
import os
import shutil
import time
import random

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from model_utils import create_model
import torchvision.models
from utils import Logger, AverageMeter, accuracy, mkdir_p, savefig
from utils.progress.progress.bar import Bar
from dataloader import StandardDataLoader
from PIL import Image
# from randaugment import rand_augment_transform
import numpy as np
# import clip
# import open_clip
import yaml
from torch.utils.data import ConcatDataset, DataLoader
# from torchvision.transforms import v2
from augmentations.grid import GridMask
from cutmix.cutmix import CutMix
from cutmix.utils import CutMixCrossEntropyLoss
from augmentations.mixup import mixup_data, mixup_criterion
from augmentations.augment_and_mix import AugMixDataset
import torch.nn.functional as F


parser = argparse.ArgumentParser(description='PyTorch OrgaSMnist Training')
# Datasets
parser.add_argument('-d', '--dataset', default='cifar10', type=str)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--data_dir', default='data/cifar100_original_data', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--expand_num', default=5, type=int, help='generate image per prompt')

parser.add_argument('--data_expanded_dir', default=None, type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
# Optimization options
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--train-batch-size', default=128, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--val-batch-size', default=100, type=int, metavar='N',
                    help='test batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--drop', '--dropout', default=0, type=float,
                    metavar='Dropout', help='Dropout ratio')
parser.add_argument('--schedule', type=int, nargs='+', default=[41, 81],
                        help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
# RandAugment
parser.add_argument('--pretrained', default=False, action='store_true')
parser.add_argument('--train_fc', default=False, action='store_true')
parser.add_argument('-n',  default=2, type=int)
parser.add_argument('-m', default=10, type=int)
# Checkpoints
parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# Architecture
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet50', help='model architecture ')
parser.add_argument('--transform_type', type=str, default='default', help='image transform type ')
# Miscs
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
#Device options
parser.add_argument('--accumulate',  type=int, default= 0)
args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# Validate dataset
# assert args.dataset == 'cifar10' or args.dataset == 'cifar100', 'Dataset can only be cifar10 or cifar100.'

# Use CUDA
use_cuda = torch.cuda.is_available()


def get_transform():

    transform_type = args.transform_type
    if transform_type == "autoaug":
        trans = transforms.Compose([
            transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomRotation(15, ),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            torchvision.transforms.AutoAugment(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    elif transform_type == "randaug":
        trans = transforms.Compose([
            transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomRotation(15, ),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandAugment(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    elif transform_type == "cutout":
        trans = transforms.Compose([
            transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomRotation(15, ),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing()
        ])
    else:
        trans = transforms.Compose([
            transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomRotation(15, ),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    return trans

# Random seed
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_random_seed(args.manualSeed)

best_acc = 0  # best test accuracy


class DatasetByClassNames(data.Dataset):
    def __init__(self, root, classnames, transforms):
        self.imgs, self.labels = [], []
        for i, cls in enumerate(classnames):
            imgs = os.listdir(os.path.join(root, cls))

            # filter
            imgs = [x for x in imgs if int(x.split("_")[-1].split(".")[0]) < args.expand_num]

            imgs = [os.path.join(root, cls, x) for x in imgs]
            self.imgs.extend(imgs)
            self.labels.extend([i for _ in range(len(imgs))])

        self.transforms = transforms

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path = self.imgs[index]
        target = self.labels[index]
        img = Image.open(path).convert('RGB')
        img = self.transforms(img)
        return img, target

    def __len__(self):
        return len(self.imgs)

def wrap_clip_forward(clip_model, text_feature):
    def custom_forward(self, x):
        x = self.encode_image(x)
        x = self.fc(x)
        return x

    fc = nn.Linear(text_feature.shape[1], text_feature.shape[0], dtype=clip_model.dtype)
    with torch.no_grad():
        fc.weight.copy_(text_feature)
        nn.init.constant_(fc.bias, 0)
    clip_model.fc = fc
    clip_model.forward = custom_forward.__get__(clip_model)

    return clip_model

def main():
    global best_acc
    start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch

    if not os.path.isdir(args.checkpoint):
        # args.checkpoint = os.path.join(args.checkpoint + f"_{args.transform_type}_{args.expand_num}x")
        os.makedirs(args.checkpoint)

    # Data
    print('==> Preparing dataset %s' % args.dataset)
    transform_train = transforms.Compose([
            transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomRotation(15, ),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    transform_test = transforms.Compose([
            transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    original_trainset, testset, _, class_names = StandardDataLoader(args, transform_test, transform_train).load_dataset()

    num_classes = len(class_names)

    if args.data_expanded_dir is not None:
        if isinstance(args.data_expanded_dir, list) or isinstance(args.data_expanded_dir, tuple):
            concat_sets = [original_trainset]
            for expand_dir in args.data_expanded_dir:
                expanded_trainset = DatasetByClassNames(expand_dir, class_names, transform_train)
                concat_sets.append(expanded_trainset)
            new_trainset = torch.utils.data.ConcatDataset(concat_sets)
        else:
            expanded_trainset = DatasetByClassNames(args.data_expanded_dir, class_names, transform_train)
            new_trainset = torch.utils.data.ConcatDataset([original_trainset, expanded_trainset])
    else:
        new_trainset = ConcatDataset([original_trainset] * (args.expand_num + 1))

    print("original dataset len:", len(original_trainset))
    print("expanded dataset len:", len(new_trainset))
    assert len(new_trainset) / len(original_trainset) == (args.expand_num + 1)

    new_trainset.transform = get_transform()

    criterion = nn.CrossEntropyLoss()

    # args.transform_type = "augmix"
    if args.transform_type == "cutmix":
        new_trainset = CutMix(new_trainset, num_class=num_classes, beta=1.0, prob=0.5, num_mix=2)
        criterion = CutMixCrossEntropyLoss(True)
    elif args.transform_type == "augmix":
        train_base_aug = transforms.Compose([
            transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomRotation(15, ),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
        ])
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        new_trainset.transform = train_base_aug
        new_trainset = AugMixDataset(new_trainset, preprocess, k=3, alpha=1., no_jsd=False)



    # exit()
    trainloader = data.DataLoader(new_trainset, batch_size=args.train_batch_size, shuffle=True, num_workers=args.workers)
    testloader = data.DataLoader(testset, batch_size=args.val_batch_size, shuffle=False, num_workers=args.workers)

    # Model
    model = create_model(args.arch, num_classes, args.pretrained, class_names, dataset_name=args.dataset)
    model = torch.nn.DataParallel(model).cuda()

    cudnn.benchmark = True
    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    if args.train_fc:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.fc.parameters():
            param.requires_grad = True
        optimizer = optim.SGD(model.fc.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    # Resume
    title = f'{args.dataset}-' + args.arch
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        args.checkpoint = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title, resume=True)
        logger.write(str(args))
    else:
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
        logger.write(str(args))
        logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])

    if args.evaluate:
        print('\nEvaluation only')
        test_loss, test_acc = test(testloader, model, criterion, start_epoch, use_cuda)
        print(' Test Loss:  %.8f, Test Acc:  %.2f' % (test_loss, test_acc))
        return

    # Train and val
    for epoch in range(start_epoch, args.epochs):
        #adjust_learning_rate(optimizer, epoch)

        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))

        train_loss, train_acc = train(trainloader, model, criterion, optimizer, epoch, use_cuda)
        test_loss, test_acc = test(testloader, model, criterion, epoch, use_cuda)

        # append logger file
        logger.append([state['lr'], train_loss, test_loss, train_acc, test_acc])

        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)
        if is_best:
             logger.write("The best performance:" + str(best_acc))          
        save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'acc': test_acc,
                'best_acc': best_acc,
                'optimizer' : optimizer.state_dict(),
            }, is_best,best_acc, checkpoint=args.checkpoint)
        scheduler.step()
    logger.close()
    logger.plot()
    savefig(os.path.join(args.checkpoint, 'log.eps'))

    print('Best acc:')
    print(best_acc)

    result = {
        "best_accuracy": best_acc,
        "last_accuracy": test_acc
    }
    with open(os.path.join(args.checkpoint, "results.yaml"), 'w') as f:
        yaml.dump(result, f)


def train(train_loader, model, criterion, optimizer, epoch, use_cuda):
    # switch to train mode
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    bar = Bar('Processing', max=len(train_loader))
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        if args.transform_type == "gridmask":
            grid = GridMask(96, 224, 360, 0.6, 1, 0.8)
            grid.set_prob(epoch, 80)
            inputs = grid(inputs)
        elif args.transform_type == "mixup":
            inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, 1.0, use_cuda)
            targets_a = torch.autograd.Variable(targets_a)
            targets_b = torch.autograd.Variable(targets_b)

        if args.transform_type == "augmix":
            # js_loss:
            bs = inputs[0].size(0)
            images_cat = torch.cat(inputs, dim=0).cuda()  # [3 * batch, 3, 32, 32]
            # targets = targets.to(device)

            logits = model(images_cat)
            logits_orig, logits_augmix1, logits_augmix2 = logits[:bs], logits[bs:2 * bs], logits[2 * bs:]
            outputs = logits_orig
            loss = criterion(logits_orig, targets)

            p_orig, p_augmix1, p_augmix2 = (F.softmax(logits_orig, dim=-1),
                                            F.softmax(logits_augmix1, dim=-1), F.softmax(logits_augmix2, dim=-1))

            # Clamp mixture distribution to avoid exploding KL divergence
            p_mixture = torch.clamp((p_orig + p_augmix1 + p_augmix2) / 3., 1e-7, 1).log()
            loss += 12 * (F.kl_div(p_mixture, p_orig, reduction='batchmean') +
                          F.kl_div(p_mixture, p_augmix1, reduction='batchmean') +
                          F.kl_div(p_mixture, p_augmix2, reduction='batchmean')) / 3.
        else:
            inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
            # compute output
            outputs = model(inputs)

            if args.transform_type == "mixup":
                loss = mixup_criterion(criterion, outputs, targets_a, targets_b, 1.0)
            else:
                loss = criterion(outputs, targets)

        # measure accuracy and record loss
        if args.transform_type in ["mixup", "cutmix"]:
            prec1, prec5 = 0., 0.
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1, inputs.size(0))
            top5.update(prec5, inputs.size(0))
        else:
            prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))

        # compute gradient and do SGD step 
        if args.accumulate==0:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else: 
            accumulate_step =args.accumulate
            loss = loss/accumulate_step
            loss.backward()
            if ((batch_idx+1)%accumulate_step)==0:
                optimizer.step()
                optimizer.zero_grad()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(train_loader),
                    data=data_time.val,
                    bt=batch_time.val,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                    )
        bar.next()
    bar.finish()
    torch.cuda.empty_cache() 
    return (losses.avg, top1.avg)

def test(val_loader, model, criterion, epoch, use_cuda):
    global best_acc

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    bar = Bar('Processing', max=len(val_loader))
    for batch_idx, (inputs, targets) in enumerate(val_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs, volatile=True), torch.autograd.Variable(targets)

        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(val_loader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                    )
        bar.next()
    bar.finish()
    torch.cuda.empty_cache() 
    return (losses.avg, top1.avg)

def save_checkpoint(state, is_best, best_acc, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        print("The best performance:", best_acc)
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))

def adjust_learning_rate(optimizer, epoch):
    global state
    if epoch in args.schedule:
        state['lr'] *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']

if __name__ == '__main__':
    main()


