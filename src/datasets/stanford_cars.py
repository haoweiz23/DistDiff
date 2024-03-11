import os
import pickle
from scipy.io import loadmat
from torch.utils.data import Dataset


class StanfordCars(Dataset):
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir
        trainval_file = os.path.join(self.dataset_dir, "devkit", "cars_train_annos.mat")
        test_file = os.path.join(self.dataset_dir, "cars_test_annos_withlabels.mat")
        meta_file = os.path.join(self.dataset_dir, "devkit", "cars_meta.mat")
        train_image_paths, train_labels, class_names = self.read_data("cars_train", trainval_file, meta_file)
        test_image_paths, test_labels, _ = self.read_data("cars_test", test_file, meta_file)


    def read_data(self, image_dir, anno_file, meta_file):
        anno_file = loadmat(anno_file)["annotations"][0]
        meta_file = loadmat(meta_file)["class_names"][0]
        class_names = []
        image_paths = []
        labels = []

        for i in range(len(anno_file)):
            imname = anno_file[i]["fname"][0]
            impath = os.path.join(self.dataset_dir, image_dir, imname)
            label = anno_file[i]["class"][0, 0]
            label = int(label) - 1  # convert to 0-based index
            classname = meta_file[label][0]
            names = classname.split(" ")
            year = names.pop(-1)
            names.insert(0, year)
            classname = " ".join(names)

            class_names.append(classname)
            image_paths.append(impath)
            labels.append(label)

        return image_paths, labels, class_names
