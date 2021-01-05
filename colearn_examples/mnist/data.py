import os
import pickle
import tempfile
from pathlib import Path

import numpy as np

import imgaug.augmenters as iaa

import tensorflow.keras.datasets.mnist as mnist

from colearn_examples.config import ModelConfig
from colearn_examples.utils.data import shuffle_data
from colearn_examples.utils.data import split_by_chunksizes
from colearn.basic_learner import LearnerData

# this line is a fix for np.version 1.18 making a change that imgaug hasn't
# tracked yet
if float(np.version.version[2:4]) == 18:
    # pylint: disable=W0212
    np.random.bit_generator = np.random._bit_generator

IMAGE_FL = "images.pickle"
LABEL_FL = "labels.pickle"


def split_to_folders(data_dir,
                     shuffle_seed,
                     data_split,
                     n_learners,
                     output_folder=Path(tempfile.gettempdir()) / "mnist"):
    # Load MNIST
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    all_images = np.concatenate([train_images, test_images], axis=0)
    all_labels = np.concatenate([train_labels, test_labels], axis=0)

    # Normalization
    all_images = all_images.astype("float32") / 255.0

    # Add channel dimension: 28,28 -> 28,28,1
    all_images = np.expand_dims(all_images, axis=-1)

    [all_images, all_labels] = shuffle_data(
        [all_images, all_labels], seed=shuffle_seed
    )

    [all_images_lists, all_labels_lists] = split_by_chunksizes(
        [all_images, all_labels], data_split
    )

    local_output_dir = Path(output_folder)

    dir_names = []
    for i in range(n_learners):
        dir_name = local_output_dir / str(i)
        os.makedirs(str(dir_name), exist_ok=True)

        pickle.dump(all_images_lists[i], open(dir_name / IMAGE_FL, "wb"))
        pickle.dump(all_labels_lists[i], open(dir_name / LABEL_FL, "wb"))

        dir_names.append(dir_name)

    print(dir_names)
    return [str(x) for x in dir_names]


def prepare_single_client(config: ModelConfig, data_dir, test_data_dir=None):
    images = pickle.load(open(Path(data_dir) / IMAGE_FL, "rb"))
    labels = pickle.load(open(Path(data_dir) / LABEL_FL, "rb"))

    [[train_images, test_images], [train_labels, test_labels]] = \
        split_by_chunksizes([images, labels], [config.train_ratio, config.test_ratio])

    train_data_size = len(train_images)

    train_gen = train_generator(
        train_images, train_labels, config.batch_size,
        config.width,
        config.height,
        config.generator_seed,
        config.train_augment
    )
    val_gen = train_generator(
        train_images, train_labels, config.batch_size,
        config.width,
        config.height,
        config.generator_seed,
        config.train_augment
    )

    test_data_size = len(test_images)

    test_gen = train_generator(
        test_images,
        test_labels,
        config.batch_size,
        config.width,
        config.height,
        config.generator_seed,
        config.train_augment,
    )

    return LearnerData(train_gen=train_gen,
                       val_gen=val_gen,
                       test_gen=test_gen,
                       train_data_size=train_data_size,
                       test_data_size=test_data_size,
                       train_batch_size=config.batch_size,
                       test_batch_size=config.batch_size)


# Augmentation sequence
seq_mnist = iaa.Sequential([iaa.Affine(rotate=(-15, 15))])  # rotation


def train_generator(data, labels, batch_size, width, height, seed,
                    augmentation=True, shuffle=True):
    # Get total number of samples in the data
    n_data = len(data)

    # Define two numpy arrays for containing batch data and labels
    batch_data = np.zeros(
        (batch_size, width, height, 1), dtype=np.float32
    )
    batch_labels = np.zeros((batch_size, 1), dtype=np.uint8)

    # Get a numpy array of all the indices of the input data
    indices = np.arange(n_data)

    if shuffle:
        if seed is not None:
            np.random.seed(seed)

        np.random.shuffle(indices)
    it = 0

    # Initialize a counter
    batch_counter = 0

    while True:
        if augmentation:
            batch_data[batch_counter] = seq_mnist.augment_image(data[indices[it]])
        else:
            batch_data[batch_counter] = data[indices[it]]

        batch_labels[batch_counter] = labels[indices[it]]

        batch_counter += 1
        it += 1

        if it >= n_data:
            it = 0

            if shuffle:
                if seed is not None:
                    np.random.seed(seed)
                np.random.shuffle(indices)

        if batch_counter == batch_size:
            yield batch_data, batch_labels
            batch_counter = 0
