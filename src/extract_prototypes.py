'''
Code for extracting hierarchical prototypes.
Copyright (c) Haowei Zhu, 2023
'''

from __future__ import print_function

import argparse
import os
import os.path
import shutil
import time
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torchvision.utils import make_grid
from torch import autocast
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.transforms import GaussianBlur
from torchvision.utils import save_image
from torchvision.transforms import ToTensor
from torch.autograd import Variable
import models.cifar as models
import torchvision.models
import clip
# from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
from utils import Logger, AverageMeter, accuracy, mkdir_p, savefig
from utils.progress.progress.bar import Bar
import PIL
from PIL import Image
from randaugment import rand_augment_transform

#### dataloader related #####
from dataloader import StandardDataLoader
from prompts_helper import return_photo_prompts

###### Stable Diffusion ######
import core
import math
from omegaconf import OmegaConf
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange, repeat
from contextlib import nullcontext
from pytorch_lightning import seed_everything

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from sklearn import cluster
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt



##################################


model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))
model_names.append('resnet50')
model_names.append('CLIP-VIT-B32')
model_names.append('CLIP-VIT-L14')
parser = argparse.ArgumentParser(description='PyTorch CIFAR10/100 Training')
# Datasets
parser.add_argument('-d', '--dataset', default='fgvc_aircraft', type=str)
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--data_dir', default='data', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--data_save_dir', default='data', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
# parser.add_argument('--expanded_number', default=50000, type=int)
parser.add_argument('--expanded_number_per_sample', default=5, type=int)
parser.add_argument('--expanded_batch_size', default=2, type=int)
parser.add_argument('--constraint_value', default=0.1, type=float)
# Optimization options
parser.add_argument('--max_epochs', default=10000, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--train-batch', default=1, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--test-batch', default=1, type=int, metavar='N',
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
parser.add_argument('--total_split', default=8, type=int)
parser.add_argument('--split', default=0, type=int, help='Dividing classes into 5 parts, the index of which parts')
parser.add_argument("--K", type=int, default=3, help="number of local prototypes")

# Stable diffusion
parser.add_argument(
    "--skip_grid",
    action='store_false',
    help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",
)

parser.add_argument('--optimize_latents', default=False, action='store_true')

parser.add_argument(
    "--ddim_steps",
    type=int,
    default=50,
    help="number of ddim sampling steps",
)

parser.add_argument(
    "--plms",
    action='store_true',
    help="use plms sampling",
)
parser.add_argument(
    "--fixed_code",
    action='store_true',
    help="if enabled, uses the same starting code across all samples ",
)

parser.add_argument(
    "--ddim_eta",
    type=float,
    default=0.0,
    help="ddim eta (eta=0.0 corresponds to deterministic sampling",
)
parser.add_argument(
    "--n_iter",
    type=int,
    default=1,
    help="sample this often",
)
parser.add_argument(
    "--C",
    type=int,
    default=4,
    help="latent channels",
)
parser.add_argument(
    "--f",
    type=int,
    default=8,
    help="downsampling factor, most often 8 or 16",
)
parser.add_argument(
    "--n_rows",
    type=int,
    default=1,
    help="rows in the grid (default: n_samples)",
)
parser.add_argument(  # classifier free guidance
    "--scale",
    type=float,
    default=50.0,
    help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
)

parser.add_argument(
    "--strength",
    type=float,
    default=0.5,
    help="strength for noising/unnoising. 1.0 corresponds to full destruction of information in init image",
)
parser.add_argument(
    "--config",
    type=str,
    default="configs/stable-diffusion/v1-inference.yaml",
    help="path to config which constructs model",
)
parser.add_argument(
    "--ckpt",
    type=str,
    default="models/model.ckpt",
    help="path to checkpoint of model",
)
# RandAugment
parser.add_argument('--noise_std', default=1, type=float)
# Checkpoints
parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# Architecture
parser.add_argument('--arch', '-a', metavar='ARCH', default='CLIP-VIT-B32',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet18)')
parser.add_argument('--depth', type=int, default=29, help='Model depth.')
parser.add_argument('--block-name', type=str, default='BasicBlock',
                    help='the building block for Resnet and Preresnet: BasicBlock, Bottleneck (default: Basicblock for cifar10/cifar100)')
parser.add_argument('--cardinality', type=int, default=8, help='Model cardinality (group).')
parser.add_argument('--widen-factor', type=int, default=4, help='Widen factor. 4 -> 64, 8 -> 128, ...')
parser.add_argument('--growthRate', type=int, default=12, help='Growth rate for DenseNet.')
parser.add_argument('--compressionRate', type=int, default=2, help='Compression Rate (theta) for DenseNet.')
# Miscs
parser.add_argument('--manualSeed', type=int, help='manual seed')
# Device options
parser.add_argument('--gpu', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
args = parser.parse_args()

# dummy parameters for dataloader
args.val_batch_size = 64
args.train_batch_size = 256

state = {k: v for k, v in args._get_kwargs()}

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
use_cuda = torch.cuda.is_available()

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)



def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)



class OverwriteDataset(data.Dataset):
    def __init__(self, dataset, loader=default_loader):
        self.imgs, self.labels = dataset.image_paths, dataset.labels

        self.transform_train_normalized = transforms.Compose([
            transforms.Resize((512, 512), interpolation=PIL.Image.BICUBIC),
            transforms.RandomRotation(15, ),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])

        self.transform_original_normalized = transforms.Compose([
            transforms.Resize((224, 224), interpolation=PIL.Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

        self.loader = loader

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path = self.imgs[index]
        target = self.labels[index]
        img = self.loader(path)
        augmented_img = self.transform_train_normalized(img)
        original_img = self.transform_original_normalized(img)

        return original_img, augmented_img, target, index

    def __len__(self):
        return len(self.imgs)


def main():
    total_trainset, _, _, _, _, _, _, num_classes, class_names = StandardDataLoader(args, None, None).load_dataset()
    total_trainset = OverwriteDataset(total_trainset)

    # Model
    print("==> creating model '{}'".format(args.arch))
    # create model
    # num_classes = 1000
    dim_feature = 2048
    if args.arch.startswith('resnext'):
        model = models.__dict__[args.arch](
            baseWidth=args.base_width,
            cardinality=args.cardinality,
        )
    elif args.arch == 'resnet50':
        model = torchvision.models.resnet50(pretrained=True)
        model.fc = nn.Linear(dim_feature, num_classes)
    elif args.arch == 'CLIP-VIT-B32':
        model, preprocess = clip.load("ViT-B/32")
        text_descriptions = [f"This is a photo of a {label}" for label in class_names]
        text_tokens = clip.tokenize(text_descriptions).cuda()
        text_features = model.encode_text(text_tokens).float()
        text_features /= text_features.norm(dim=-1, keepdim=True)
        text_classifier = text_features
    elif args.arch == 'CLIP-VIT-L14':
        model, preprocess = clip.load("ViT-L/14")
        text_descriptions = [f"This is a photo of a {label}" for label in class_names]
        text_tokens = clip.tokenize(text_descriptions).cuda()
        text_features = model.encode_text(text_tokens).float()
        text_features /= text_features.norm(dim=-1, keepdim=True)
        text_classifier = text_features

    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()

    model = model.cuda()
    print('Model CLIP loaded.')

    cudnn.benchmark = True
    print('Total params of CLIP: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))

    trainloader = data.DataLoader(total_trainset, batch_size=128, shuffle=False, num_workers=args.workers)
    extract_prototype(trainloader, model)



def extract_prototype(train_loader, model):
    # switch to train mode
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    end = time.time()

    bar = Bar('Processing', max=len(train_loader))

    feature_list = []
    label_list = []
    for batch_idx, (original_inputs, _, targets, _) in enumerate(train_loader):
        original_inputs, targets = original_inputs.cuda(), targets.cuda()

        original_feature = model.encode_image(original_inputs).float().detach()
        original_feature = original_feature / original_feature.norm(dim=-1, keepdim=True)
        original_feature = original_feature.cpu().numpy()

        for idx in range(len(original_inputs)):
            feature_list.append(original_feature[idx])
        label_list.extend(targets.tolist())

        batch_time.update(time.time() - end)
        end = time.time()
        # plot progress
        bar.suffix = '({batch}/{size}) Data: {data:.3f}s'.format(
            batch=batch_idx + 1,
            size=len(train_loader),
            data=data_time.val
        )
        bar.next()
    bar.finish()
    torch.cuda.empty_cache()

    num_classes = len(set(label_list))
    # class-wise gathering
    class_wise_features = [[] for _ in range(num_classes)]
    for f, y in zip(feature_list, label_list):
        class_wise_features[y].append(f)

    n_sub_proto = args.K             # K
    # Agglomerative Clustering
    hc = cluster.AgglomerativeClustering(
        n_clusters=n_sub_proto,
        linkage='average',
        distance_threshold=None
    )

    global_prototypes = [np.stack(x).mean(0) for x in class_wise_features]
    class_sub_prototypes = []

    for cls_f in class_wise_features:

        cls_f = np.stack(cls_f)
        y_pred = hc.fit(cls_f).labels_
        clusters = np.unique(y_pred)
        n_cluster = len(clusters)

        sub_features = [[] for _ in range(n_cluster)]
        for f, y in zip(cls_f, y_pred):
            sub_features[y].append(f)
        sub_protos = [np.stack(x).mean(0) for x in sub_features]
        class_sub_prototypes.append(sub_protos)
        print(f"clustering {len(cls_f)} samples into {n_cluster} sub-classes.")


    print(len(global_prototypes), len(class_sub_prototypes))
    save_dir = './data/prototypes/{}/'.format(args.dataset)
    os.makedirs(save_dir, exist_ok=True)
    np.savez(os.path.join(save_dir, f"class_wise_prototype_K{n_sub_proto}"),
             global_prototypes=global_prototypes,
             local_prototypes=class_sub_prototypes)


if __name__ == '__main__':
    main()

