'''
Copyright (c) Haowei Zhu, 2024
'''

import os
import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from collections import defaultdict
import argparse
import json
# from utils import utils
from torchvision.utils import make_grid
from torchvision.io import read_image
from torch.utils.data import Dataset
from torchvision.utils import save_image
import random
from tqdm import tqdm
import torchvision.datasets as datasets
import scipy
from torch.utils import data
from scipy import io
import scipy.misc
from PIL import Image
import PIL
from PIL.ImageOps import exif_transpose
from tqdm import tqdm
from sklearn import cluster
from utils import Logger, AverageMeter, accuracy, mkdir_p, savefig
from utils.progress.progress.bar import Bar


def tokenize_prompt(tokenizer, prompt, tokenizer_max_length=None):
    if tokenizer_max_length is not None:
        max_length = tokenizer_max_length
    else:
        max_length = tokenizer.model_max_length

    text_inputs = tokenizer(
        prompt,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )

    return text_inputs


CUSTOM_TEMPLATES = {
    "dtd": "{} texture.",
    "stanford_cars": "a photo of a {}.",
    "cifar100_subset": "a photo of a {}.",
    "stl10": "a photo of a {}.",
    "imagenette2-320": "a photo of a {}.",
    "caltech-101": "a photo of a {}.",
    "pathmnist": "a colon pathological image of {}.",
    "breastmnist": "a photo of {} ultrasound image.",
    "bloodmnist": "a photo of {}, a type of cell.",
}

DATASET_PATH = './data/{}'


class ImageDatasetFromPaths(Dataset):
    def __init__(self, split_entity, transform):
        self.image_paths, self.labels = split_entity.image_paths, split_entity.labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        image = Image.open(img_path)
        image = exif_transpose(image)
        if not image.mode == "RGB":
            image = image.convert("RGB")

        if self.transform:
            image = self.transform(image)
        return image, label


class DataEntity():
    def __init__(self, image_paths, labels):
        self.image_paths = image_paths
        self.labels = labels


class StandardDataLoader:
    def __init__(self, args, test_preprocess, train_preprocess):
        if args.dataset in ["pathmnist", "bloodmnist", "breastmnist"]:
            self.dataset_path = os.path.join(DATASET_PATH.format(f"medmnist/{args.dataset}"))
        else:
            self.dataset_path = DATASET_PATH.format(args.dataset)
        self.args = args
        # val/test images preprocessing
        self.test_preprocess = test_preprocess
        self.train_preprocess = train_preprocess

    def load_dataset(self):
        if self.args.dataset == 'stanford_cars':
            outputs = self.stanfordcars_load()
        elif self.args.dataset == 'caltech-101':
            outputs = self.caltech101_load()
        elif self.args.dataset == 'imagenette2-320':
            outputs = self.imagenette_load()
        elif self.args.dataset == 'oxford_flowers':
            outputs = self.oxfordflower_load()
        elif self.args.dataset == 'dtd':
            outputs = self.dtd_load()
        elif self.args.dataset == 'fgvc_aircraft':
            outputs = self.fgvcaircraft_load()
        elif self.args.dataset == 'oxford_pets':
            outputs = self.oxfordpets_load()
        elif self.args.dataset == 'cifar100_subset':
            outputs = self.cifar100_subset_load()
        elif self.args.dataset in ["pathmnist", "bloodmnist", "breastmnist"]:
            outputs = self.medmnist_load()
        else:
            raise ValueError('Dataset not supported')

        train_dataset, train_loader, train_loader_shuffle, val_dataset, val_loader, test_dataset, test_loader, num_classes, string_classnames = outputs
        string_classnames = [s.replace('_', ' ') for s in string_classnames]
        return train_dataset, test_dataset, test_loader, string_classnames

    def cifar100_subset_load(self):
        root_data_dir = self.dataset_path
        test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=self.test_preprocess)
        class_names = test_dataset.classes

        train_image_paths = []
        train_labels = []

        for i, class_name in enumerate(class_names):
            train_paths = os.listdir(os.path.join(root_data_dir, class_name))
            train_paths = [os.path.join(root_data_dir, class_name, x) for x in train_paths]
            train_image_paths.extend(train_paths)
            train_labels.extend([i] * len(train_paths))

        train_dataset = ImageDatasetFromPaths(DataEntity(train_image_paths, train_labels),
                                              transform=self.train_preprocess)
        val_dataset = test_dataset

        print('Load ' + str(self.args.dataset) + ' data finished.')

        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.args.val_batch_size, num_workers=8,
                                                 shuffle=False)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.args.val_batch_size, num_workers=8,
                                                  shuffle=False)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.args.train_batch_size, num_workers=8,
                                                   shuffle=False)
        train_loader_shuffle = torch.utils.data.DataLoader(train_dataset, batch_size=self.args.train_batch_size,
                                                           num_workers=8, shuffle=True)

        string_classnames = class_names

        num_classes = len(string_classnames)

        return train_dataset, train_loader, train_loader_shuffle, val_dataset, val_loader, test_dataset, test_loader, num_classes, string_classnames

    def stanfordcars_load(self):
        def read_data(root, image_dir, anno_file, meta_file):
            anno_file = io.loadmat(anno_file)["annotations"][0]
            meta_file = io.loadmat(meta_file)["class_names"][0]
            class_names = []
            image_paths = []
            labels = []
            classname_to_label_mapping = {}
            label_to_classname_mapping = {}
            for i in range(len(anno_file)):
                imname = anno_file[i]["fname"][0]
                impath = os.path.join(root, image_dir, imname)
                label = anno_file[i]["class"][0, 0]
                label = int(label) - 1  # convert to 0-based index
                classname = meta_file[label][0]
                names = classname.split(" ")
                year = names.pop(-1)
                names.insert(0, year)
                classname = " ".join(names)
                if classname not in classname_to_label_mapping.keys():
                    classname_to_label_mapping[classname] = label
                if label not in label_to_classname_mapping.keys():
                    label_to_classname_mapping[label] = classname
                class_names.append(classname)
                image_paths.append(impath)
                labels.append(label)

            sorted_class_names = [k for k, v in
                                  sorted(classname_to_label_mapping.items(), key=lambda x: x[1], reverse=False)]
            assert label_to_classname_mapping[0] == sorted_class_names[0]

            return image_paths, labels, sorted_class_names

        trainval_file = os.path.join(self.dataset_path, "devkit", "cars_train_annos.mat")
        test_file = os.path.join(self.dataset_path, "cars_test_annos_withlabels.mat")
        meta_file = os.path.join(self.dataset_path, "devkit", "cars_meta.mat")
        train_image_paths, train_labels, sorted_class_names = read_data(self.dataset_path, "cars_train", trainval_file,
                                                                        meta_file)
        test_image_paths, test_labels, _ = read_data(self.dataset_path, "cars_test", test_file, meta_file)

        string_classnames = sorted_class_names
        assert len(string_classnames) == 196, print("class names length: ", len(string_classnames))

        train_dataset = ImageDatasetFromPaths(DataEntity(train_image_paths, train_labels),
                                              transform=self.train_preprocess)
        val_dataset = ImageDatasetFromPaths(DataEntity(test_image_paths, test_labels), transform=self.test_preprocess)
        test_dataset = ImageDatasetFromPaths(DataEntity(test_image_paths, test_labels), transform=self.test_preprocess)

        print('Load stanford-cars data finished.')

        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.args.val_batch_size, num_workers=8,
                                                 shuffle=False)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.args.val_batch_size, num_workers=8,
                                                  shuffle=False)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.args.train_batch_size, num_workers=8,
                                                   shuffle=False)
        train_loader_shuffle = torch.utils.data.DataLoader(train_dataset, batch_size=self.args.train_batch_size,
                                                           num_workers=8, shuffle=True)

        num_classes = len(string_classnames)

        return train_dataset, train_loader, train_loader_shuffle, val_dataset, val_loader, test_dataset, test_loader, num_classes, string_classnames

    def medmnist_load(self):
        train_data_path = os.path.join(self.dataset_path, "train")
        test_data_path = os.path.join(self.dataset_path, "test")
        categories = sorted(os.listdir(train_data_path))

        train_image_paths = []
        train_labels = []
        test_image_paths = []
        test_labels = []

        for i, category in enumerate(categories):
            train_samples = [os.path.join(train_data_path, category, x) for x in
                             os.listdir(os.path.join(train_data_path, category))]
            train_image_paths.extend(train_samples)
            train_labels.extend([i] * len(train_samples))

            test_samples = [os.path.join(test_data_path, category, x) for x in
                            os.listdir(os.path.join(test_data_path, category))]
            test_image_paths.extend(test_samples)
            test_labels.extend([i] * len(test_samples))

        string_classnames = [s.replace('_', ' ') for s in categories]
        train_dataset = ImageDatasetFromPaths(DataEntity(train_image_paths, train_labels),
                                              transform=self.train_preprocess)
        val_dataset = ImageDatasetFromPaths(DataEntity(test_image_paths, test_labels), transform=self.test_preprocess)
        test_dataset = ImageDatasetFromPaths(DataEntity(test_image_paths, test_labels), transform=self.test_preprocess)

        print('Load medmnist data finished.')

        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.args.val_batch_size, num_workers=8,
                                                 shuffle=False)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.args.val_batch_size, num_workers=8,
                                                  shuffle=False)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.args.train_batch_size, num_workers=8,
                                                   shuffle=False)
        train_loader_shuffle = torch.utils.data.DataLoader(train_dataset, batch_size=self.args.train_batch_size,
                                                           num_workers=8, shuffle=True)

        num_classes = len(string_classnames)

        return train_dataset, train_loader, train_loader_shuffle, val_dataset, val_loader, test_dataset, test_loader, num_classes, string_classnames

    def caltech101_load(self):
        train_data_path = os.path.join(self.dataset_path, "train")
        test_data_path = os.path.join(self.dataset_path, "test")
        categories = sorted(os.listdir(train_data_path))
        categories = [x for x in categories if x != "BACKGROUND_Google" and x != "Faces_easy"]

        train_image_paths = []
        train_labels = []
        test_image_paths = []
        test_labels = []

        for i, category in enumerate(categories):
            train_samples = [os.path.join(train_data_path, category, x) for x in
                             os.listdir(os.path.join(train_data_path, category))]
            train_image_paths.extend(train_samples)
            train_labels.extend([i] * len(train_samples))

            test_samples = [os.path.join(test_data_path, category, x) for x in
                            os.listdir(os.path.join(test_data_path, category))]
            test_image_paths.extend(test_samples)
            test_labels.extend([i] * len(test_samples))

        string_classnames = [s.replace('_', ' ') for s in categories]
        assert len(string_classnames) == 100

        train_dataset = ImageDatasetFromPaths(DataEntity(train_image_paths, train_labels),
                                              transform=self.train_preprocess)
        val_dataset = ImageDatasetFromPaths(DataEntity(test_image_paths, test_labels), transform=self.test_preprocess)
        test_dataset = ImageDatasetFromPaths(DataEntity(test_image_paths, test_labels), transform=self.test_preprocess)

        print('Load caltech101 data finished.')

        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.args.val_batch_size, num_workers=8,
                                                 shuffle=False)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.args.val_batch_size, num_workers=8,
                                                  shuffle=False)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.args.train_batch_size, num_workers=8,
                                                   shuffle=False)
        train_loader_shuffle = torch.utils.data.DataLoader(train_dataset, batch_size=self.args.train_batch_size,
                                                           num_workers=8, shuffle=True)

        num_classes = len(string_classnames)

        return train_dataset, train_loader, train_loader_shuffle, val_dataset, val_loader, test_dataset, test_loader, num_classes, string_classnames

    def imagenette_load(self):
        train_data_path = os.path.join(self.dataset_path, "train")
        test_data_path = os.path.join(self.dataset_path, "val")
        categories = sorted(os.listdir(train_data_path))

        train_image_paths = []
        train_labels = []
        test_image_paths = []
        test_labels = []

        for i, category in enumerate(categories):
            train_samples = [os.path.join(train_data_path, category, x) for x in
                             os.listdir(os.path.join(train_data_path, category))]
            train_image_paths.extend(train_samples)
            train_labels.extend([i] * len(train_samples))

            test_samples = [os.path.join(test_data_path, category, x) for x in
                            os.listdir(os.path.join(test_data_path, category))]
            test_image_paths.extend(test_samples)
            test_labels.extend([i] * len(test_samples))

        string_classnames = [s.replace('_', ' ') for s in categories]

        train_dataset = ImageDatasetFromPaths(DataEntity(train_image_paths, train_labels),
                                              transform=self.train_preprocess)
        val_dataset = ImageDatasetFromPaths(DataEntity(test_image_paths, test_labels), transform=self.test_preprocess)
        test_dataset = ImageDatasetFromPaths(DataEntity(test_image_paths, test_labels), transform=self.test_preprocess)

        print('Load imagenette data finished.')

        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.args.val_batch_size, num_workers=8,
                                                 shuffle=False)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.args.val_batch_size, num_workers=8,
                                                  shuffle=False)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.args.train_batch_size, num_workers=8,
                                                   shuffle=False)
        train_loader_shuffle = torch.utils.data.DataLoader(train_dataset, batch_size=self.args.train_batch_size,
                                                           num_workers=8, shuffle=True)

        num_classes = len(string_classnames)

        return train_dataset, train_loader, train_loader_shuffle, val_dataset, val_loader, test_dataset, test_loader, num_classes, string_classnames

    def oxfordflower_load(self):
        train_data_path = os.path.join(self.dataset_path, "train")
        test_data_path = os.path.join(self.dataset_path, "valid")
        labels = sorted(os.listdir(train_data_path))
        lab2cname_file = os.path.join(self.dataset_path, "cat_to_name.json")

        train_image_paths = []
        train_labels = []
        test_image_paths = []
        test_labels = []

        for label in labels:
            train_samples = [os.path.join(train_data_path, label, x) for x in
                             os.listdir(os.path.join(train_data_path, label))]
            train_image_paths.extend(train_samples)
            train_labels.extend([int(label) - 1] * len(train_samples))

            test_samples = [os.path.join(test_data_path, label, x) for x in
                            os.listdir(os.path.join(test_data_path, label))]
            test_image_paths.extend(test_samples)
            test_labels.extend([int(label) - 1] * len(test_samples))

        lab2cname = json.load(open(lab2cname_file, 'r'))

        sorted_class_names = [v for k, v in
                              sorted(lab2cname.items(), key=lambda x: int(x[0]), reverse=False)]
        assert lab2cname['1'] == sorted_class_names[0]

        string_classnames = sorted_class_names
        assert len(string_classnames) == 102

        train_dataset = ImageDatasetFromPaths(DataEntity(train_image_paths, train_labels),
                                              transform=self.train_preprocess)
        val_dataset = ImageDatasetFromPaths(DataEntity(test_image_paths, test_labels), transform=self.test_preprocess)
        test_dataset = ImageDatasetFromPaths(DataEntity(test_image_paths, test_labels), transform=self.test_preprocess)

        print('Load oxfordflower data finished.')

        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.args.val_batch_size, num_workers=8,
                                                 shuffle=False)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.args.val_batch_size, num_workers=8,
                                                  shuffle=False)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.args.train_batch_size, num_workers=8,
                                                   shuffle=False)
        train_loader_shuffle = torch.utils.data.DataLoader(train_dataset, batch_size=self.args.train_batch_size,
                                                           num_workers=8, shuffle=True)

        num_classes = len(string_classnames)

        return train_dataset, train_loader, train_loader_shuffle, val_dataset, val_loader, test_dataset, test_loader, num_classes, string_classnames

    def dtd_load(self):
        image_dir = os.path.join(self.dataset_path, "images")
        train_files = os.path.join(self.dataset_path, "labels", 'train1.txt')
        val_files = os.path.join(self.dataset_path, "labels", 'val1.txt')
        test_files = os.path.join(self.dataset_path, "labels", 'test1.txt')

        train_image_paths = []
        train_labels = []
        test_image_paths = []
        test_labels = []
        classname_to_label_mapping = {}
        label_to_classname_mapping = {}
        categories = os.listdir(image_dir)
        categories.sort()

        for label, category in enumerate(categories):
            classname_to_label_mapping[category] = label
            label_to_classname_mapping[label] = category

        with open(train_files, 'r') as f:
            for path in f.readlines():
                path = path.strip()
                class_name = path.split('/')[0]
                impath = os.path.join(image_dir, path)
                train_image_paths.append(impath)
                train_labels.append(classname_to_label_mapping[class_name])

        with open(val_files, 'r') as f:
            for path in f.readlines():
                path = path.strip()
                class_name = path.split('/')[0]
                impath = os.path.join(image_dir, path)
                train_image_paths.append(impath)
                train_labels.append(classname_to_label_mapping[class_name])

        with open(test_files, 'r') as f:
            for path in f.readlines():
                path = path.strip()
                class_name = path.split('/')[0]
                impath = os.path.join(image_dir, path)
                test_image_paths.append(impath)
                test_labels.append(classname_to_label_mapping[class_name])

        string_classnames = categories
        assert len(string_classnames) == 47

        train_dataset = ImageDatasetFromPaths(DataEntity(train_image_paths, train_labels),
                                              transform=self.train_preprocess)
        val_dataset = ImageDatasetFromPaths(DataEntity(test_image_paths, test_labels), transform=self.test_preprocess)
        test_dataset = ImageDatasetFromPaths(DataEntity(test_image_paths, test_labels), transform=self.test_preprocess)

        print('Load dtd data finished.')

        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.args.val_batch_size, num_workers=8,
                                                 shuffle=False)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.args.val_batch_size, num_workers=8,
                                                  shuffle=False)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.args.train_batch_size, num_workers=8,
                                                   shuffle=False)
        train_loader_shuffle = torch.utils.data.DataLoader(train_dataset, batch_size=self.args.train_batch_size,
                                                           num_workers=8, shuffle=True)

        num_classes = len(string_classnames)

        return train_dataset, train_loader, train_loader_shuffle, val_dataset, val_loader, test_dataset, test_loader, num_classes, string_classnames

    def oxfordpets_load(self):
        image_dir = os.path.join(self.dataset_path, "images")
        anno_dir = os.path.join(self.dataset_path, "annotations")
        train_filepath = os.path.join(anno_dir, "trainval.txt")
        test_filepath = os.path.join(anno_dir, "test.txt")

        classname_to_label_mapping = {}
        label_to_classname_mapping = {}
        train_image_paths = []
        train_labels = []
        test_image_paths = []
        test_labels = []
        with open(train_filepath, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                imname, label, species, _ = line.split(" ")
                breed = imname.split("_")[:-1]
                breed = "_".join(breed)
                class_name = breed.lower()
                imname += ".jpg"
                impath = os.path.join(image_dir, imname)
                label = int(label) - 1  # convert to 0-based index
                train_image_paths.append(impath)
                train_labels.append(label)

                if class_name not in classname_to_label_mapping.keys():
                    classname_to_label_mapping[class_name] = label

                if label not in label_to_classname_mapping.keys():
                    label_to_classname_mapping[label] = class_name

        with open(test_filepath, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                imname, label, species, _ = line.split(" ")
                imname += ".jpg"
                impath = os.path.join(image_dir, imname)
                label = int(label) - 1  # convert to 0-based index
                test_image_paths.append(impath)
                test_labels.append(label)

        sorted_class_names = [k for k, v in
                              sorted(classname_to_label_mapping.items(), key=lambda x: x[1], reverse=False)]
        assert label_to_classname_mapping[0] == sorted_class_names[0]

        string_classnames = sorted_class_names
        assert len(string_classnames) == 37

        train_dataset = ImageDatasetFromPaths(DataEntity(train_image_paths, train_labels),
                                              transform=self.train_preprocess)
        val_dataset = ImageDatasetFromPaths(DataEntity(test_image_paths, test_labels), transform=self.test_preprocess)
        test_dataset = ImageDatasetFromPaths(DataEntity(test_image_paths, test_labels), transform=self.test_preprocess)

        print('Load oxfordpets data finished.')

        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.args.val_batch_size, num_workers=8,
                                                 shuffle=False)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.args.val_batch_size, num_workers=8,
                                                  shuffle=False)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.args.train_batch_size, num_workers=8,
                                                   shuffle=False)
        train_loader_shuffle = torch.utils.data.DataLoader(train_dataset, batch_size=self.args.train_batch_size,
                                                           num_workers=8, shuffle=True)

        num_classes = len(string_classnames)

        string_classnames = [s.replace('_', ' ') for s in string_classnames]
        return train_dataset, train_loader, train_loader_shuffle, val_dataset, val_loader, test_dataset, test_loader, num_classes, string_classnames

    def fgvcaircraft_load(self):

        images_dir = os.path.join(self.dataset_path, 'images')

        train_split_image_names_file = os.path.join(self.dataset_path, 'images_variant_train.txt')
        val_split_image_names_file = os.path.join(self.dataset_path, 'images_variant_val.txt')
        test_split_image_names_file = os.path.join(self.dataset_path, 'images_variant_test.txt')
        classnames_file = os.path.join(self.dataset_path, 'variants.txt')

        label_to_classname_mapping = {}
        classname_to_label_mapping = {}

        class_to_samples_map = {}

        with open(classnames_file, 'r') as f:
            string_classnames = [f.strip() for f in f.readlines()]

            for i in range(len(string_classnames)):
                label_to_classname_mapping[i] = string_classnames[i]
                classname_to_label_mapping[string_classnames[i]] = i

        train_image_paths = []
        train_classnames = []
        train_labels = []

        with open(train_split_image_names_file, 'r') as f:
            paths_and_classes = f.readlines()
            paths_and_classes = [p.strip().split() for p in paths_and_classes]

            for p in paths_and_classes:
                train_image_paths.append(os.path.join(images_dir, p[0] + '.jpg'))
                curr_classname = ' '.join(p[1:])
                train_classnames.append(curr_classname)
                train_labels.append(classname_to_label_mapping[curr_classname])

                if curr_classname in class_to_samples_map:
                    class_to_samples_map[curr_classname].append(os.path.join(images_dir, p[0] + '.jpg'))
                else:
                    class_to_samples_map[curr_classname] = []
                    class_to_samples_map[curr_classname].append(os.path.join(images_dir, p[0] + '.jpg'))

        with open(test_split_image_names_file, 'r') as f:
            paths_and_classes = f.readlines()
            paths_and_classes = [p.strip().split() for p in paths_and_classes]

            test_image_paths = [os.path.join(images_dir, p[0] + '.jpg') for p in paths_and_classes]
            test_classnames = [' '.join(p[1:]) for p in paths_and_classes]
            test_labels = [classname_to_label_mapping[' '.join(p[1:])] for p in paths_and_classes]

        with open(val_split_image_names_file, 'r') as f:
            paths_and_classes = f.readlines()
            paths_and_classes = [p.strip().split() for p in paths_and_classes]

            val_image_paths = [os.path.join(images_dir, p[0] + '.jpg') for p in paths_and_classes]
            val_classnames = [' '.join(p[1:]) for p in paths_and_classes]
            val_labels = [classname_to_label_mapping[' '.join(p[1:])] for p in paths_and_classes]

        img_paths = []
        targets = []

        for class_id in list(class_to_samples_map.keys()):
            img_paths = img_paths + list(class_to_samples_map[class_id])
            targets = targets + [classname_to_label_mapping[class_id] for _ in
                                 range(len(list(class_to_samples_map[class_id])))]

        train_dataset = ImageDatasetFromPaths(DataEntity(img_paths, targets), transform=self.train_preprocess)
        val_dataset = ImageDatasetFromPaths(DataEntity(val_image_paths, val_labels), transform=self.test_preprocess)
        test_dataset = ImageDatasetFromPaths(DataEntity(test_image_paths, test_labels), transform=self.test_preprocess)

        print('Load ' + str(self.args.dataset) + ' data finished.')

        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.args.val_batch_size, num_workers=8,
                                                 shuffle=False)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.args.val_batch_size, num_workers=8,
                                                  shuffle=False)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.args.train_batch_size, num_workers=8,
                                                   shuffle=False)
        train_loader_shuffle = torch.utils.data.DataLoader(train_dataset, batch_size=self.args.train_batch_size,
                                                           num_workers=8, shuffle=True)

        num_classes = len(string_classnames)

        return train_dataset, train_loader, train_loader_shuffle, val_dataset, val_loader, test_dataset, test_loader, num_classes, string_classnames


def encode_prompt(text_encoder, input_ids, attention_mask, text_encoder_use_attention_mask=None):
    text_input_ids = input_ids.to(text_encoder.device)

    if text_encoder_use_attention_mask:
        attention_mask = attention_mask.to(text_encoder.device)
    else:
        attention_mask = None

    prompt_embeds = text_encoder(
        text_input_ids,
        attention_mask=attention_mask,
        return_dict=False,
    )
    prompt_embeds = prompt_embeds[0]

    return prompt_embeds


def compute_text_embeddings(prompt, tokenizer, text_encoder, tokenizer_max_length=None):
    with torch.no_grad():
        text_inputs = tokenize_prompt(tokenizer, prompt, tokenizer_max_length=tokenizer_max_length)
        prompt_embeds = encode_prompt(
            text_encoder,
            text_inputs.input_ids,
            text_inputs.attention_mask,
            # text_encoder_use_attention_mask=args.text_encoder_use_attention_mask,
        )

    return prompt_embeds


def extract_prototype(args, train_loader, model):
    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    bar = Bar('Processing', max=len(train_loader))

    feature_list = []
    label_list = []
    for batch_idx, (original_inputs, targets) in enumerate(train_loader):
        original_inputs, targets = original_inputs.cuda(), targets.cuda()

        original_feature = model.encode_image(original_inputs).float().detach()
        original_feature = original_feature / original_feature.norm(dim=-1, keepdim=True)
        original_feature = original_feature.cpu().numpy()

        for idx in range(len(original_inputs)):
            feature_list.append(original_feature[idx])
        label_list.extend(targets.tolist())
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

    n_sub_proto = args.K  # K
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

    # print(len(global_prototypes), len(class_sub_prototypes))
    # save_dir = './save/prototypes/{}/{}/'.format(args.arch, args.dataset)
    # os.makedirs(save_dir, exist_ok=True)
    # np.savez(os.path.join(save_dir, f"class_wise_prototype_K{n_sub_proto}"), global_prototypes=global_prototypes, local_prototypes=class_sub_prototypes)

    global_prototypes = np.array(global_prototypes)
    class_sub_prototypes = np.array(class_sub_prototypes)
    return global_prototypes, class_sub_prototypes


def extract_prototypes_with_encoder(args, model):
    trainset, _, _, class_names = StandardDataLoader(args, None, None).load_dataset()
    trainset.transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    trainloader = data.DataLoader(trainset, batch_size=64, shuffle=False, drop_last=False)

    model = model.float()
    global_prototype, local_prototype = extract_prototype(args, trainloader, model)
    return global_prototype, local_prototype


class SDDataset(data.Dataset):
    def __init__(self, args, tokenizer, text_encoder, vae, size=512, center_crop=False):

        trainset, _, _, class_names = StandardDataLoader(args, None, None).load_dataset()
        self.imgs, self.labels = trainset.image_paths, trainset.labels
        self.class_names = class_names

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        self.trainset = trainset

        text_encoder = text_encoder.cuda()

        self.le = args.language_enhance
        if args.language_enhance:
            template = np.load(f"data/{args.dataset}_le.pkl", allow_pickle=True)
            template = {k.replace("_", " "): v for k, v in template.items()}
            classes_prompts = []
            for x in class_names:
                sentences = template[x]
                embeds = []
                for prompt in sentences:
                    embeds.append(compute_text_embeddings(prompt, tokenizer, text_encoder))
                classes_prompts.append(embeds)
        else:
            template = CUSTOM_TEMPLATES[args.dataset]
            classes_prompts = [compute_text_embeddings(template.format(x), tokenizer, text_encoder) for x in
                               class_names]

        self.classes_prompts = classes_prompts
        self.uncond_input = compute_text_embeddings("", tokenizer, text_encoder)

        embed_dir = os.path.join("save/vae_embedding", args.dataset,
                                 args.pretrained_model_name_or_path.replace("/", "--"))
        embed_path = os.path.join(embed_dir, "image_latents.pt")
        if os.path.exists(embed_path):
            self.image_latents = torch.load(embed_path)
        else:
            os.makedirs(embed_dir, exist_ok=True)
            self.image_latents = self.encode_image(vae)
            torch.save(self.image_latents, embed_path)

    @torch.no_grad()
    def encode_image(self, vae):
        vae = vae.cuda()
        latents = []
        for path in tqdm(self.imgs):
            instance_image = Image.open(path)
            instance_image = exif_transpose(instance_image)
            if not instance_image.mode == "RGB":
                instance_image = instance_image.convert("RGB")
            instance_image = self.image_transforms(instance_image).cuda()
            latent = vae.encode(torch.stack([instance_image])).latent_dist.sample()
            latent = latent * vae.config.scaling_factor
            latents.append(latent)
        return latents

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        example = {}
        path = self.imgs[index]
        target = self.labels[index]

        instance_image = Image.open(path)
        instance_image = exif_transpose(instance_image)
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
        example["pil_images"] = instance_image
        example["instance_images"] = self.image_transforms(instance_image)
        example["image_latents"] = self.image_latents[index]

        if self.le:
            example["instance_prompt_ids"] = random.choice(self.classes_prompts[target])
        else:
            example["instance_prompt_ids"] = self.classes_prompts[target]
        example["uncond_prompt_ids"] = self.uncond_input
        example["class_names"] = self.class_names[target]
        example["image_paths"] = path
        example["targets"] = target

        # # sample_index = index % self.num_instance_images
        # example["instance_prompt_ids"] = self.classes_prompts[target].input_ids[index: index + 1]
        # example["instance_attention_mask"] = self.classes_prompts[target].attention_mask[index: index + 1]
        #
        # # unconditional prompt tokenizer
        # example["uncond_prompt_ids"] = self.uncond_input.input_ids
        # example["uncond_attention_mask"] = self.uncond_input.attention_mask
        return example

    def __len__(self):
        return len(self.imgs)


if __name__ == '__main__':
    # Test dataloaders for each dataset
    from transformers import AutoTokenizer, PretrainedConfig

    parser = argparse.ArgumentParser()
    parser.add_argument('--val_batch_size', type=int, default=64)
    parser.add_argument('--train_batch_size', type=int, default=256)
    parser.add_argument('--dataset', type=str, default='caltech-101')
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="CompVis/stable-diffusion-v1-4",
        required=False,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    args = parser.parse_args()
    dataset_list = ["caltech-101"]
    for dataset in dataset_list:
        args.dataset = dataset
        trainset, testset, test_loader, classnames = StandardDataLoader(args, None, None).load_dataset()
        num_classes = classnames
        print(len(trainset), len(testset))
        exit()
