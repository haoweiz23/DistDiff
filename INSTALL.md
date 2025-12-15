<!-- Adapted from: https://github.com/Vanint/DatasetExpansion] -->

# Installation

This codebase is tested on Ubuntu 20.04.2 LTS with python 3.10. Follow the below steps to create environment and install dependencies.

## Required libraries


* Create the environment
```bash
conda create -n distdiff python=3.10.6 -y
conda activate distdiff
```


* Install necessary python libraries:

```
pip install numpy scipy matplotlib scikit-image medmnist timm regex tqdm accelerate transformers==4.19.2
```

* Install CLIP:

```
pip install open_clip_torch
```

* Install diffuser:
```bash
git clone https://github.com/huggingface/diffusers
cd diffusers
pip install -e .
```

* Install CutMix (optional)
```
pip install git+https://github.com/ildoonet/cutmix
```


## Datasets
The instructions to prepare each dataset are detailed below. 
All the datasets should be prepared in `./data/`. 

Datasets list:
- [Cifar100-Subset](#Cifar100-Subset)
- [Caltech101](#Caltech101)
- [StanfordCars](#StanfordCars)
- [DTD](#dtd)
- [OxfordFlowers](#OxfordFlowers)
- [OxfordPets](#OxfordPets)


### Cifar100-Subset

* Download datasets: 

Please download the dataset at https://drive.google.com/file/d/12Aryi3Dan8hXrw0_Kg2WU_4BIEwY4csT/view?usp=drive_link

```
unzip CIFAR_10000.zip
mv CIFAR_10000 data
```

### Caltech101

##### (1) Prepare code and download datasets and checkpoints

* Download datasets: 

Please download the Caltech101 dataset at https://www.kaggle.com/datasets/imbikramsaha/caltech-101 

```
Please unzip it and name it as caltech-101
mv caltech-101 data
```

### StanfordCars

* Download datasets: 

Please download the dataset at https://www.kaggle.com/datasets/jessicali9530/stanford-cars-dataset

```
Please unzip it and name it as stanford_cars
mv stanford_cars data
```

### DTD

##### (1) Prepare code and download datasets and checkpoints

* Download datasets: 

Please download the dataset at https://www.robots.ox.ac.uk/~vgg/data/dtd

```
Please unzip it and name it as dtd
mv dtd data
```

### OxfordFlowers

* Download datasets: 

Please download the dataset at https://www.robots.ox.ac.uk/~vgg/data/flowers/102

```
Please unzip it and name it as oxford_flowers
mv oxford_flowers data
```

### OxfordPets

* Download datasets: 

Please download the dataset at https://www.robots.ox.ac.uk/~vgg/data/pets

```
Please unzip it and name it as oxford_pets
mv oxford_pets data
```

