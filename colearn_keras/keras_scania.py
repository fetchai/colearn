# ------------------------------------------------------------------------------
#
#   Copyright 2021 Fetch.AI Limited
#
#   Licensed under the Creative Commons Attribution-NonCommercial International
#   License, Version 4.0 (the "License"); you may not use this file except in
#   compliance with the License. You may obtain a copy of the License at
#
#       http://creativecommons.org/licenses/by-nc/4.0/legalcode
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
# ------------------------------------------------------------------------------
import os
from typing import Tuple

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.python.data.ops.dataset_ops import PrefetchDataset
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.layers import Dropout

from colearn_grpc.factory_registry import FactoryRegistry
from colearn_grpc.logging import get_logger, set_log_levels

from colearn_keras.keras_learner import KerasLearner
from colearn_keras.utils import _make_loader

from colearn.utils.data import get_data

_logger = get_logger(__name__)
set_log_levels({"default": "INFO"})


def reshape_x(x):
    """ Reshape array to fit for resnet"""
    return np.expand_dims(np.expand_dims(x, axis=1), axis=3)


def getf(type_, set_, data_folder):
    """Get file for prediction data loader"""
    return os.path.join(
        data_folder, f"{type_}_{set_}_learner.csv"
    )


# prepare dataloader implementation
def prepare_loaders_impl(location: str, reshape: bool = False
                         ) -> Tuple[PrefetchDataset, PrefetchDataset,
                                    PrefetchDataset]:
    _logger.info(f"    -    USING DATASET FROM LOCATION: {location}")
    _logger.info("    -    LOADING csv!")

    data_folder = get_data(location)

    X_train = pd.read_csv(getf("X", "train", data_folder), index_col=0).values
    y_train = pd.read_csv(getf("y", "train", data_folder), index_col=0).values
    X_test = pd.read_csv(getf("X", "test", data_folder), index_col=0).values
    y_test = pd.read_csv(getf("y", "test", data_folder), index_col=0).values
    X_vote = pd.read_csv(getf("X", "vote", data_folder), index_col=0).values
    y_vote = pd.read_csv(getf("y", "vote", data_folder), index_col=0).values

    if reshape:
        X_train, X_vote, X_test = reshape_x(
            X_train), reshape_x(X_vote), reshape_x(X_test)

    train_loader = _make_loader(X_train, y_train.reshape(-1))
    vote_loader = _make_loader(X_vote, y_vote.reshape(-1))
    test_loader = _make_loader(X_test, y_test.reshape(-1))

    return train_loader, vote_loader, test_loader


# The dataloader needs to be registered before the models that reference it
@FactoryRegistry.register_dataloader("KERAS_SCANIA_RESNET")
def prepare_data_loaders_resnet(location: str) -> Tuple[PrefetchDataset,
                                                        PrefetchDataset,
                                                        PrefetchDataset]:
    """
    Load training data from folders and create train and test dataloader

    :param location: Path to training dataset
    :return: Tuple of train_loader and test_loader
    """
    return prepare_loaders_impl(location, reshape=True)


# The dataloader needs to be registered before the models that reference it
@FactoryRegistry.register_dataloader("KERAS_SCANIA")
def prepare_data_loaders(location: str) -> Tuple[PrefetchDataset,
                                                 PrefetchDataset,
                                                 PrefetchDataset]:
    """
    Load training data from folders and create train and test dataloader

    :param location: Path to training dataset
    :return: Tuple of train_loader and test_loader
    """
    return prepare_loaders_impl(location, reshape=False)


# prepare pred loader implementation
def prepare_pred_loaders_impl(location: str, reshape: bool = False):
    """
     Load prediction data from folder and create prediction data loader

    :param location: Path to prediction file
    :return: np.array
    """
    _logger.info(f"    -    LOADING PRED DATASET FROM LOCATION: {location}")

    data_folder = get_data(location)

    X_pred = pd.read_csv(data_folder, index_col=0).values

    if reshape:
        X_pred = reshape_x(X_pred)

    return X_pred


def prepare_pred_loaders_impl_resnet(location: str):
    """
    Wrapper for loading image data from folder and create prediction data loader

    :param location: Path to data
    :return: np.array
    """
    return prepare_pred_loaders_impl(location, reshape=True)


# The prediction dataloader needs to be registered before the models that reference it
@FactoryRegistry.register_prediction_dataloader("KERAS_SCANIA_PRED")
def prepare_prediction_data_loaders(location: str = None) -> dict:
    """
    Wrapper for loading data from folder and create prediction data loader

    :param location: Path to data
    :return: dict of name and function
    """
    return {"KERAS_SCANIA_PRED": prepare_pred_loaders_impl}


@FactoryRegistry.register_prediction_dataloader("KERAS_SCANIA_PRED_RESNET")
def prepare_prediction_data_loaders_two(location: str = None) -> dict:
    """
    Wrapper for loading data from folder and create prediction data loader.
    Same as other data loader for testing purpose.

    :param location: Path to data
    :return: dict of name and function
    """
    return {"KERAS_SCANIA_PRED_RESNET": prepare_pred_loaders_impl_resnet}


@FactoryRegistry.register_model_architecture("KERAS_SCANIA_RESNET", ["KERAS_SCANIA_RESNET"], ["KERAS_SCANIA_PRED_RESNET"])
def prepare_learner_resnet(data_loaders: Tuple[PrefetchDataset, PrefetchDataset,
                                               PrefetchDataset],
                           prediction_data_loaders: dict,
                           steps_per_epoch: int = 100,
                           vote_batches: int = 10,
                           learning_rate: float = 0.001
                           ) -> KerasLearner:
    """
    Creates new instance of KerasLearner
    :param data_loaders: Tuple of train_loader and test_loader
    :param steps_per_epoch: Number of batches per training epoch
    :param vote_batches: Number of batches to get vote_accuracy
    :param learning_rate: Learning rate for optimiser
    :return: New instance of KerasLearner
    """
    # RESNET model
    rows = 1
    cols = 162
    channels = 1
    new_channels = 3
    padding = 2
    n_classes = 2

    input_img = tf.keras.Input(
        shape=(rows, cols, channels), name="Input"
    )
    x = tf.keras.layers.ZeroPadding2D(padding=padding)(input_img)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.RepeatVector(new_channels)(x)
    x = tf.keras.layers.Reshape(
        (rows + padding * 2, cols + padding * 2, new_channels))(x)

    resnet = ResNet50(include_top=False, input_tensor=x)

    x = resnet.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = Dropout(0.7)(x)
    x = tf.keras.layers.Dense(n_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs=input_img, outputs=x)

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
                  loss='sparse_categorical_crossentropy',
                  metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
                  )

    learner = KerasLearner(
        model=model,
        train_loader=data_loaders[0],
        vote_loader=data_loaders[1],
        test_loader=data_loaders[2],
        criterion="sparse_categorical_accuracy",
        minimise_criterion=False,
        model_fit_kwargs={"steps_per_epoch": steps_per_epoch},
        model_evaluate_kwargs={"steps": vote_batches},
        prediction_data_loader=prediction_data_loaders
    )
    return learner


@FactoryRegistry.register_model_architecture("KERAS_SCANIA", ["KERAS_SCANIA"], ["KERAS_SCANIA_PRED"])
def prepare_learner_mlp(data_loaders: Tuple[PrefetchDataset, PrefetchDataset,
                                            PrefetchDataset],
                        prediction_data_loaders: dict,
                        steps_per_epoch: int = 100,
                        vote_batches: int = 10,
                        learning_rate: float = 0.001
                        ) -> KerasLearner:
    """
    Creates new instance of KerasLearner
    :param data_loaders: Tuple of train_loader and test_loader
    :param steps_per_epoch: Number of batches per training epoch
    :param vote_batches: Number of batches to get vote_accuracy
    :param learning_rate: Learning rate for optimiser
    :return: New instance of KerasLearner
    """
    # MLP model
    n_classes = 2

    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(50, activation='relu'),
        tf.keras.layers.Dense(n_classes, activation='softmax'),
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
                  loss='sparse_categorical_crossentropy',
                  metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
                  )

    learner = KerasLearner(
        model=model,
        train_loader=data_loaders[0],
        vote_loader=data_loaders[1],
        test_loader=data_loaders[2],
        criterion="sparse_categorical_accuracy",
        minimise_criterion=False,
        model_fit_kwargs={"steps_per_epoch": steps_per_epoch},
        model_evaluate_kwargs={"steps": vote_batches},
        prediction_data_loader=prediction_data_loaders
    )
    return learner
