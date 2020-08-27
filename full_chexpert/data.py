import math
import os
import pickle
import shutil
from pathlib import Path

import cv2
import imgaug.augmenters as iaa
import numpy as np
import pandas as pd

from ..data import shuffle_data, split_by_chunksizes
from ..model import LearnerData

# this line is a fix for np.version 1.18 making a change that imgaug hasn't tracked yet
if float(np.version.version[2:4]) == 18:
    np.random.bit_generator = np.random._bit_generator

diagnosis_dict = {
    "No Finding": 0,
    "Enlarged Cardiomediastinum": 1,
    "Cardiomegaly": 2,
    "Lung Opacity": 3,
    "Lung Lesion": 4,
    "Edema": 5,
    "Consolidation": 6,
    "Pneumonia": 7,
    "Atelectasis": 8,
    "Pneumothorax": 9,
    "Pleural Effusion": 10,
    "Pleural Other": 11,
    "Fracture": 12,
    "Support Devices": 13,
}


def to_rgb_normalize_and_resize(img, width, height):
    img = cv2.imread(str(img))
    img = cv2.resize(img, (width, height))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.astype(np.float32) / 255.0
    img = np.reshape(img, (width, height, 1))

    return img


def normalize_image(filename, label, width, height):
    data = to_rgb_normalize_and_resize(filename, width, height)
    label = [math.nan if x == -1 else x for x in label]

    return data, label


def split_chexpert(chexpert_path, csv_filename="train.csv"):
    df = pd.read_csv(Path(chexpert_path) / csv_filename, sep=",")
    df = df.loc[df["Frontal/Lateral"].isin(["Frontal"])]

    data = list(df["Path"])
    data = [Path(chexpert_path).parent / dat for dat in data]

    labels = df.iloc[:, 5:]
    labels = labels.values.tolist()

    return data, labels


def split_to_folders(
    config, data_dir="", output_folder=Path(os.getcwd()) / "full_chexpert"
):
    config.class_labels = [i for i in diagnosis_dict.keys()]

    train_data, train_labels = split_chexpert(data_dir, "train.csv")
    val_data, val_labels = split_chexpert(data_dir, "valid.csv")

    [train_data, train_labels] = shuffle_data(
        [train_data, train_labels], config.shuffle_seed
    )
    [val_data, val_labels] = shuffle_data([val_data, val_labels], config.shuffle_seed)

    [all_train_data_lists, all_train_labels_lists] = split_by_chunksizes(
        [train_data, train_labels], config.data_split
    )
    [all_val_data_lists, all_val_labels_lists] = split_by_chunksizes(
        [val_data, val_labels], config.data_split
    )

    dir_names = []
    for i in range(config.n_learners):

        dir_name = output_folder / str(i)
        dir_names.append(dir_name)
        if os.path.isdir(str(dir_name)):
            shutil.rmtree(str(dir_name))
            os.makedirs(str(dir_name))
        else:
            os.makedirs(str(dir_name))

        pickle.dump(all_train_data_lists[i], open(dir_name / "train_data.pickle", "wb"))
        pickle.dump(
            all_train_labels_lists[i], open(dir_name / "train_labels.pickle", "wb")
        )
        pickle.dump(all_val_data_lists[i], open(dir_name / "val_data.pickle", "wb"))
        pickle.dump(all_val_labels_lists[i], open(dir_name / "val_labels.pickle", "wb"))

    return dir_names


def prepare_single_client(config, data_dir):
    data = LearnerData()
    data.train_batch_size = config.batch_size

    train_data = pickle.load(open(Path(data_dir) / "train_data.pickle", "rb"))
    train_labels = pickle.load(open(Path(data_dir) / "train_labels.pickle", "rb"))
    val_data = pickle.load(open(Path(data_dir) / "val_data.pickle", "rb"))
    val_labels = pickle.load(open(Path(data_dir) / "val_labels.pickle", "rb"))

    [[test_data], [test_labels]] = split_by_chunksizes(
        [val_data, val_labels], [config.test_ratio]
    )

    data.train_data_size = len(train_data)

    data.train_gen = train_generator(
        train_data, train_labels, config.batch_size, config, config.train_augment
    )
    data.val_gen = train_generator(
        train_data, train_labels, config.batch_size, config, config.train_augment
    )

    data.test_data_size = len(test_data)

    data.test_gen = train_generator(
        test_data,
        test_labels,
        config.batch_size,
        config,
        config.train_augment,
        shuffle=False,
    )

    data.test_batch_size = config.batch_size
    return data


# Augmentation sequence
seq = iaa.Sequential(
    [
        # iaa.Fliplr(0.5),  # horizontal flips
        iaa.Affine(rotate=(-15, 15)),  # rotation
        iaa.Multiply((0.7, 1.3)),
    ]
)  # random brightness


def train_generator(data, labels, batch_size, config, augmentation=True, shuffle=True):
    # Get total number of samples in the data
    n_data = len(data)

    # Define two numpy arrays for containing batch data and labels
    batch_data = np.zeros(
        (batch_size, config.width, config.height, 1), dtype=np.float32
    )
    batch_labels = np.zeros((batch_size, config.n_classes), dtype=np.float32)

    # Get a numpy array of all the indices of the input data
    indices = np.arange(n_data)

    if shuffle:
        if config.generator_seed is not None:
            np.random.seed(config.generator_seed)
        np.random.shuffle(indices)

    it = 0

    # Initialize a counter
    batch_counter = 0

    while True:
        img_data, img_label = normalize_image(
            data[indices[it]], labels[indices[it]], config.width, config.height
        )
        it += 1

        if augmentation:
            img_data = seq.augment_image(img_data)

        batch_data[batch_counter] = img_data
        batch_labels[batch_counter] = img_label

        batch_counter += 1

        if it >= n_data:
            it = 0

            if shuffle:
                if config.generator_seed is not None:
                    np.random.seed(config.generator_seed)
                np.random.shuffle(indices)

        if batch_counter == batch_size:
            yield batch_data, batch_labels
            batch_counter = 0
