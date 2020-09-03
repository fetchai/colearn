import os
import pickle
import tempfile
from pathlib import Path
#from google.cloud import storage

import imgaug.augmenters as iaa

import numpy as np

import tensorflow.keras.datasets.mnist as mnist

from colearn.config import Config
from examples.utils.data import shuffle_data
from examples.utils.data import split_by_chunksizes
from basic_learner import LearnerData

# this line is a fix for np.version 1.18 making a change that imgaug hasn't tracked yet
if float(np.version.version[2:4]) == 18:
    np.random.bit_generator = np.random._bit_generator

IMAGE_FL = "images.pickle"
LABEL_FL = "labels.pickle"


def split_to_folders(
    config: Config, data_dir, output_folder=Path(tempfile.gettempdir()) / "mnist"
):
    # Load MNIST
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    all_images = np.concatenate([train_images, test_images], axis=0)
    all_labels = np.concatenate([train_labels, test_labels], axis=0)

    # Normalization
    all_images = all_images.astype("float32") / 255.0

    # Add channel dimension: 28,28 -> 28,28,1
    all_images = np.expand_dims(all_images, axis=-1)

    [all_images, all_labels] = shuffle_data(
        [all_images, all_labels], seed=config.shuffle_seed
    )

    [all_images_lists, all_labels_lists] = split_by_chunksizes(
        [all_images, all_labels], config.data_split
    )

    if str(output_folder).startswith("gs://"):
        use_cloud = True
        print(
            "google account details",
            os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "no account details set"),
        )
        local_output_dir = Path(tempfile.gettempdir()) / "mnist"
        outfol_split = output_folder.split("/")
        bucket_name = outfol_split[2]
        remote_output_dir = "/".join(outfol_split[3:])

        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
    else:
        use_cloud = False
        local_output_dir = Path(output_folder)
        bucket = None
        remote_output_dir = None

    dir_names = []
    for i in range(config.n_learners):

        dir_name = local_output_dir / str(i)
        os.makedirs(str(dir_name), exist_ok=True)

        pickle.dump(all_images_lists[i], open(dir_name / IMAGE_FL, "wb"))
        pickle.dump(all_labels_lists[i], open(dir_name / LABEL_FL, "wb"))

        if use_cloud:
            # upload files to gcloud
            remote_dir = os.path.join(remote_output_dir, str(i))
            for fl in [IMAGE_FL, LABEL_FL]:
                remote_image = os.path.join(remote_dir, fl)
                file_blob = bucket.blob(str(remote_image))
                file_blob.upload_from_filename(str(dir_name / fl))

            dir_names.append("gs://" + bucket.name + "/" + remote_dir)
        else:
            dir_names.append(dir_name)

    print(dir_names)
    return [str(x) for x in dir_names]


def prepare_single_client(config, data_dir):
    data = LearnerData()
    data.train_batch_size = config.batch_size

    if str(data_dir).startswith("gs://"):
        print(
            "google account details:",
            os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "no account details set"),
        )
        data_dir = str(data_dir)
        outfol_split = data_dir.split("/")
        bucket_name = outfol_split[2]
        remote_dir = "/".join(outfol_split[3:])

        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)

        data = []
        for fl in [IMAGE_FL, LABEL_FL]:
            with tempfile.TemporaryFile("rb+") as tmpfl:
                blob = bucket.blob(remote_dir + "/" + fl)
                blob.download_to_file(tmpfl)
                tmpfl.seek(0)
                data.append(pickle.load(tmpfl))

        images, labels = data
    else:
        images = pickle.load(open(Path(data_dir) / IMAGE_FL, "rb"))
        labels = pickle.load(open(Path(data_dir) / LABEL_FL, "rb"))

    [[train_images, test_images], [train_labels, test_labels]] = split_by_chunksizes(
        [images, labels], [config.train_ratio, config.test_ratio]
    )

    data.train_data_size = len(train_images)

    data.train_gen = train_generator(
        train_images, train_labels, config.batch_size, config, config.train_augment
    )
    data.val_gen = train_generator(
        train_images, train_labels, config.batch_size, config, config.train_augment
    )

    data.test_data_size = len(test_images)

    data.test_gen = train_generator(
        test_images,
        test_labels,
        config.batch_size,
        config,
        config.train_augment,
        shuffle=False,
    )

    data.test_batch_size = config.batch_size
    return data


# Augmentation sequence
seq_mnist = iaa.Sequential([iaa.Affine(rotate=(-15, 15))])  # rotation


def train_generator(data, labels, batch_size, config, augmentation=True, shuffle=True):
    # Get total number of samples in the data
    n_data = len(data)

    # Define two numpy arrays for containing batch data and labels
    batch_data = np.zeros(
        (batch_size, config.width, config.height, 1), dtype=np.float32
    )
    batch_labels = np.zeros((batch_size, 1), dtype=np.uint8)

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
                if config.generator_seed is not None:
                    np.random.seed(config.generator_seed)
                np.random.shuffle(indices)

        if batch_counter == batch_size:
            yield batch_data, batch_labels
            batch_counter = 0
