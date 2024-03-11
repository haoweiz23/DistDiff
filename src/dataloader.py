'''
Code for extracting hierarchical prototypes.
Copyright (c) Haowei Zhu, 2023
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
from utils import utils
from torchvision.utils import make_grid
from torchvision.io import read_image
from torch.utils.data import Dataset
from torchvision.utils import save_image
import random
from tqdm import tqdm
import torchvision.datasets as datasets
import scipy
from scipy import io
import scipy.misc

CUSTOM_TEMPLATES = {
    "OxfordPets": "a photo of a {}, a type of pet.",
    "OxfordFlowers": "a photo of a {}, a type of flower.",
    "FGVCAircraft": "a photo of a {}, a type of aircraft.",
    "DescribableTextures": "{} texture.",
    "EuroSAT": "a centered satellite photo of {}.",
    "StanfordCars": "a photo of a {}.",
    "Food101": "a photo of {}, a type of food.",
    "SUN397": "a photo of a {}.",
    "Caltech101": "a photo of a {}.",
    "UCF101": "a photo of a person doing {}.",
    "ImageNet": "a photo of a {}.",
    "ImageNetSketch": "a photo of a {}.",
    "ImageNetV2": "a photo of a {}.",
    "ImageNetA": "a photo of a {}.",
    "ImageNetR": "a photo of a {}.",
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

        try:
            image = read_image(img_path)
        except RuntimeError as e:
            # HACK: if the image is corrupted or not readable, then sample a random image
            image_rand = None
            while(image_rand is None):
                rand_ind = random.randint(0, self.__len__())
                try:
                    image_rand = read_image(self.image_paths[rand_ind])
                except RuntimeError as e1:
                    image_rand = None
                    continue
            image = image_rand
            label = self.labels[rand_ind]

        image = transforms.ToPILImage()(image)
        image = image.convert("RGB")

        if(self.transform):
            image = self.transform(image)
        return image, label

class DataEntity():
    def __init__(self, image_paths, labels):
        self.image_paths = image_paths
        self.labels = labels


class StandardDataLoader:
    def __init__(self, args, test_preprocess, train_preprocess):
        self.dataset_path = DATASET_PATH.format(args.dataset)
        self.args = args
        # val/test images preprocessing
        self.test_preprocess = test_preprocess
        self.train_preprocess = train_preprocess

    def load_dataset(self):
        if(self.args.dataset == 'stanford_cars'):
            return self.stanfordcars_load()
        elif(self.args.dataset == 'caltech-101'):
            return self.caltech101_load()
        elif(self.args.dataset == 'oxford_flowers'):
            return self.oxfordflower_load()
        elif(self.args.dataset == 'dtd'):
            return self.dtd_load()
        elif(self.args.dataset == 'fgvc_aircraft'):
            return self.fgvcaircraft_load()
        elif(self.args.dataset == 'oxford_pets'):
            return self.oxfordpets_load()
        elif(self.args.dataset == 'cifar100_subset'):
            return self.cifar100_subset_load()
        else:
            raise ValueError('Dataset not supported')

    def cifar100_subset_load(self):
        root_data_dir = self.dataset_path
        # splits_paths = {"train":[], "val":[], "test":[]}       # json file
        # splits_paths["train"] = self.extract_cifar100_subset_train_files(root_data_dir)
        # train_class_to_images_map, train_split, test_split, val_split, string_classnames = self.parse_image_paths(root_data_dir, splits_paths)

        test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=self.test_preprocess)
        class_names = test_dataset.classes

        train_image_paths = []
        train_labels = []

        for i, class_name in enumerate(class_names):
            train_paths = os.listdir(os.path.join(root_data_dir, class_name))
            train_paths = [os.path.join(root_data_dir, class_name, x) for x in train_paths]
            train_image_paths.extend(train_paths)
            train_labels.extend([i] * len(train_paths))

        train_dataset = ImageDatasetFromPaths(DataEntity(train_image_paths, train_labels), transform=self.train_preprocess)
        val_dataset = test_dataset

        print('Load '+str(self.args.dataset)+' data finished.')

        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.args.val_batch_size, num_workers=8, shuffle=False)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.args.val_batch_size, num_workers=8, shuffle=False)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.args.train_batch_size, num_workers=8, shuffle=False)
        train_loader_shuffle = torch.utils.data.DataLoader(train_dataset, batch_size=self.args.train_batch_size, num_workers=8, shuffle=True)

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
        train_image_paths, train_labels, sorted_class_names = read_data(self.dataset_path, "cars_train", trainval_file, meta_file)
        test_image_paths, test_labels, _ = read_data(self.dataset_path, "cars_test", test_file, meta_file)

        string_classnames = sorted_class_names
        assert len(string_classnames) == 196, print("class names length: ", len(string_classnames))

        train_dataset = ImageDatasetFromPaths(DataEntity(train_image_paths, train_labels), transform=self.train_preprocess)
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

    def caltech101_load(self):
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
        assert len(string_classnames) == 102

        train_dataset = ImageDatasetFromPaths(DataEntity(train_image_paths, train_labels), transform=self.train_preprocess)
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
            train_labels.extend([int(label) -1] * len(train_samples))

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

        train_dataset = ImageDatasetFromPaths(DataEntity(train_image_paths, train_labels), transform=self.train_preprocess)
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

        train_dataset = ImageDatasetFromPaths(DataEntity(train_image_paths, train_labels), transform=self.train_preprocess)
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

        train_dataset = ImageDatasetFromPaths(DataEntity(train_image_paths, train_labels), transform=self.train_preprocess)
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
                train_image_paths.append(os.path.join(images_dir, p[0]+'.jpg'))
                curr_classname = ' '.join(p[1:])
                train_classnames.append(curr_classname)
                train_labels.append(classname_to_label_mapping[curr_classname])

                if curr_classname in class_to_samples_map:
                    class_to_samples_map[curr_classname].append(os.path.join(images_dir, p[0]+'.jpg'))
                else:
                    class_to_samples_map[curr_classname] = []
                    class_to_samples_map[curr_classname].append(os.path.join(images_dir, p[0]+'.jpg'))

        with open(test_split_image_names_file, 'r') as f:
            paths_and_classes = f.readlines()
            paths_and_classes = [p.strip().split() for p in paths_and_classes]

            test_image_paths = [os.path.join(images_dir, p[0]+'.jpg') for p in paths_and_classes]
            test_classnames = [' '.join(p[1:]) for p in paths_and_classes]
            test_labels = [classname_to_label_mapping[' '.join(p[1:])] for p in paths_and_classes]

        with open(val_split_image_names_file, 'r') as f:
            paths_and_classes = f.readlines()
            paths_and_classes = [p.strip().split() for p in paths_and_classes]

            val_image_paths = [os.path.join(images_dir, p[0]+'.jpg') for p in paths_and_classes]
            val_classnames = [' '.join(p[1:]) for p in paths_and_classes]
            val_labels = [classname_to_label_mapping[' '.join(p[1:])] for p in paths_and_classes]

        img_paths = []
        targets = []

        for class_id in list(class_to_samples_map.keys()):
            img_paths = img_paths + list(class_to_samples_map[class_id])
            targets = targets + [classname_to_label_mapping[class_id] for _ in range(len(list(class_to_samples_map[class_id])))]

        train_dataset = ImageDatasetFromPaths(DataEntity(img_paths, targets), transform=self.train_preprocess)
        val_dataset = ImageDatasetFromPaths(DataEntity(val_image_paths, val_labels), transform=self.test_preprocess)
        test_dataset = ImageDatasetFromPaths(DataEntity(test_image_paths, test_labels), transform=self.test_preprocess)

        print('Load '+str(self.args.dataset)+' data finished.')

        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.args.val_batch_size, num_workers=8, shuffle=False)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.args.val_batch_size, num_workers=8, shuffle=False)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.args.train_batch_size, num_workers=8, shuffle=False)
        train_loader_shuffle = torch.utils.data.DataLoader(train_dataset, batch_size=self.args.train_batch_size, num_workers=8, shuffle=True)

        num_classes = len(string_classnames)

        return train_dataset, train_loader, train_loader_shuffle, val_dataset, val_loader, test_dataset, test_loader, num_classes, string_classnames


if __name__ == '__main__':

    from data_expand import OverwriteDataset
    # Test dataloaders for each dataset

    parser = argparse.ArgumentParser()
    parser.add_argument('--k_shot', type=int, default=16)
    parser.add_argument('--val_batch_size', type=int, default=64)
    parser.add_argument('--train_batch_size', type=int, default=256)
    parser.add_argument('--dataset', type=str, default='imagenet')
    args = parser.parse_args()

    transform_train = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomRotation(15, ),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    transform_test = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    dataset_list = ["stanford_cars", "caltech-101", 'oxford_flowers', 'oxford_pets', 'dtd', 'cifar100_subset']
    for dataset in dataset_list:
        args.dataset = dataset
        trainset, train_loader, _, _, _, testset, test_loader, num_classes, string_classnames = StandardDataLoader(args, transform_train, transform_test).load_dataset()
        string_classnames = [s.replace('_', ' ') for s in string_classnames]

        trainset = OverwriteDataset(trainset)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=False, num_workers=8)

        for i, (original_img, augmented_img, targets, index) in enumerate(trainloader):
            pass

        print(string_classnames)
        for i, (images, targets) in enumerate(test_loader):
            grid = make_grid([images[0], images[1], images[2]])
            print(targets[:6])
            print([string_classnames[s] for s in targets[:6]])
            # save_image(images[0:6], 'test_' + str(args.dataset) + '.png', nrow=3)
            break
        print(dataset, ": Train set length :", len(trainset), "Test set length :", len(testset), "class num:", num_classes)
