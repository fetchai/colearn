# ------------------------------------------------------------------------------
#
#   Copyright 2021 Fetch.AI Limited
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
# ------------------------------------------------------------------------------
import os
import pickle
import tempfile
from enum import Enum
from pathlib import Path
from typing import Tuple, List, Optional

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.python.data.ops.dataset_ops import PrefetchDataset

from colearn.utils.data import split_list_into_fractions
from colearn_keras.keras_learner import KerasLearner
from colearn_keras.utils import normalize_img

IMAGE_FL = "images.pickle"
LABEL_FL = "labels.pickle"


class ModelType(Enum):
    CONV2D = 1


def _prepare_model(model_type: ModelType, learning_rate: float) -> tf.keras.Model:
    """
    Creates a new instance of selected Keras model
    :param model_type: Enum that represents selected model type
    :param learning_rate: Learning rate for optimiser
    :return: New instance of Keras model
    """
    if model_type == ModelType.CONV2D:
        return _get_keras_cifar10_conv2D_model(learning_rate)
    else:
        raise Exception("Model %s not part of the ModelType enum" % model_type)


def _get_keras_cifar10_conv2D_model(learning_rate: float) -> tf.keras.Model:
    """
    2D Convolutional model for image recognition
    :param learning_rate: Learning rate for optimiser
    :return: Return instance of Keras model
    """
    loss = "sparse_categorical_crossentropy"
    optimizer = tf.keras.optimizers.Adam

    input_img = tf.keras.Input(
        shape=(32, 32, 3), name="Input"
    )
    x = tf.keras.layers.Conv2D(
        32, (5, 5), activation="relu", padding="same", name="Conv1_1"
    )(input_img)
    x = tf.keras.layers.MaxPooling2D((2, 2), name="pool1")(x)
    x = tf.keras.layers.Conv2D(
        32, (5, 5), activation="relu", padding="same", name="Conv2_1"
    )(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), name="pool2")(x)
    x = tf.keras.layers.Conv2D(
        64, (5, 5), activation="relu", padding="same", name="Conv3_1"
    )(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), name="pool3")(x)
    x = tf.keras.layers.Flatten(name="flatten")(x)
    x = tf.keras.layers.Dense(
        64, activation="relu", name="fc1"
    )(x)
    x = tf.keras.layers.Dense(
        10, activation="softmax", name="fc2"
    )(x)
    model = tf.keras.Model(inputs=input_img, outputs=x)

    opt = optimizer(
        lr=learning_rate
    )
    model.compile(
        loss=loss,
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
        optimizer=opt)
    return model


def prepare_learner(model_type: ModelType,
                    data_loaders: Tuple[PrefetchDataset, PrefetchDataset],
                    steps_per_epoch: int = 100,
                    vote_batches: int = 10,
                    learning_rate: float = 0.001,
                    **_kwargs) -> KerasLearner:
    """
    Creates new instance of KerasLearner
    :param model_type: Enum that represents selected model type
    :param data_loaders: Tuple of train_loader and test_loader
    :param steps_per_epoch: Number of batches per training epoch
    :param vote_batches: Number of batches to get vote_score
    :param learning_rate: Learning rate for optimiser
    :param _kwargs: Residual parameters not used by this function
    :return: New instance of KerasLearner
    """
    learner = KerasLearner(
        model=_prepare_model(model_type, learning_rate),
        train_loader=data_loaders[0],
        test_loader=data_loaders[1],
        criterion="sparse_categorical_accuracy",
        minimise_criterion=False,
        model_fit_kwargs={"steps_per_epoch": steps_per_epoch},
        model_evaluate_kwargs={"steps": vote_batches},
    )
    return learner


def _make_loader(images: np.array,
                 labels: np.array,
                 batch_size: int) -> PrefetchDataset:
    """
    Converts array of images and labels to Tensorflow dataset
    :param images: Numpy array of input data
    :param labels: Numpy array of output labels
    :param batch_size: Batch size
    :return: Shuffled Tensorflow prefetch dataset holding images and labels
    """
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))

    dataset = dataset.cache()
    dataset = dataset.shuffle(len(dataset))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset


def prepare_data_loaders(train_folder: str,
                         train_ratio: float = 0.9,
                         batch_size: int = 32,
                         **_kwargs) -> Tuple[PrefetchDataset, PrefetchDataset]:
    """
    Load training data from folders and create train and test dataloader

    :param train_folder: Path to training dataset
    :param train_ratio: What portion of train_data should be used as test set
    :param batch_size:
    :param _kwargs: Residual parameters not used by this function
    :return: Tuple of train_loader and test_loader
    """

    images = pickle.load(open(Path(train_folder) / IMAGE_FL, "rb"))
    labels = pickle.load(open(Path(train_folder) / LABEL_FL, "rb"))

    n_cases = int(train_ratio * len(images))
    train_loader = _make_loader(images[:n_cases], labels[:n_cases], batch_size)
    test_loader = _make_loader(images[n_cases:], labels[n_cases:], batch_size)

    return train_loader, test_loader


def split_to_folders(
        n_learners: int,
        data_split: Optional[List[float]] = None,
        shuffle_seed: Optional[int] = None,
        output_folder: Optional[Path] = None,
        **_kwargs
) -> List[str]:
    """
    Loads images with labels and splits them to specified number of subsets
    :param n_learners: Number of parts for splitting
    :param data_split: List of percentage portions for each subset
    :param shuffle_seed: Seed for shuffling
    :param output_folder: Folder where splitted parts will be stored as numbered subfolders
    :param _kwargs: Residual parameters not used by this function
    :return: List of folders containing individual subsets
    """
    if output_folder is None:
        output_folder = Path(tempfile.gettempdir()) / "cifar10"

    if data_split is None:
        data_split = [1 / n_learners] * n_learners

    # Load CIFAR10 from tfds
    train_dataset, info = tfds.load('cifar10', split='train+test', as_supervised=True, with_info=True)
    n_datapoints = info.splits['train+test'].num_examples
    train_dataset = train_dataset.map(normalize_img).batch(n_datapoints)

    # there is only one batch in the iterator, and this contains all the data
    all_data = next(iter(tfds.as_numpy(train_dataset)))
    all_images = all_data[0]
    all_labels = all_data[1]

    np.random.seed(shuffle_seed)
    random_indices = np.random.permutation(np.arange(n_datapoints))
    split_indices = split_list_into_fractions(random_indices, data_split)

    local_output_dir = Path(output_folder)

    dir_names = []
    for i in range(n_learners):
        dir_name = local_output_dir / str(i)
        os.makedirs(str(dir_name), exist_ok=True)

        learner_images = all_images[split_indices[i]]
        learner_labels = all_labels[split_indices[i]]

        pickle.dump(learner_images, open(dir_name / IMAGE_FL, "wb"))
        pickle.dump(learner_labels, open(dir_name / LABEL_FL, "wb"))

        dir_names.append(dir_name)

    print(dir_names)
    return [str(x) for x in dir_names]
