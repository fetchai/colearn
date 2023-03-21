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
import pickle
import tempfile
from pathlib import Path
from typing import Tuple, List, Optional

import numpy as np
from PIL import Image
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.python.data.ops.dataset_ops import PrefetchDataset
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.layers import Dropout
from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import DPKerasAdamOptimizer

from colearn.ml_interface import DiffPrivConfig
from colearn.utils.data import get_data, split_list_into_fractions
from colearn_grpc.factory_registry import FactoryRegistry
from colearn_keras.keras_learner import KerasLearner
from colearn_keras.utils import normalize_img


# The dataloader needs to be registered before the models that reference it
@FactoryRegistry.register_dataloader("KERAS_SCANIA")
def prepare_data_loaders(location: str) -> Tuple[PrefetchDataset, PrefetchDataset, PrefetchDataset]:
    """
    Load training data from folders and create train and test dataloader

    :param location: Path to training dataset
    :return: Tuple of train_loader and test_loader
    """
    pass


@FactoryRegistry.register_model_architecture("KERAS_SCANIA", ["KERAS_SCANIA"], ["KERAS_SCANIA_PRED"])
def prepare_learner(data_loaders: Tuple[PrefetchDataset, PrefetchDataset, PrefetchDataset],
                    prediction_data_loaders: dict,
                    steps_per_epoch: int = 100,
                    vote_batches: int = 10,
                    learning_rate: float = 0.001,
                    diff_priv_config: Optional[DiffPrivConfig] = None,
                    num_microbatches: int = 4,
                    ) -> KerasLearner:
    """
    Creates new instance of KerasLearner
    :param data_loaders: Tuple of train_loader and test_loader
    :param steps_per_epoch: Number of batches per training epoch
    :param vote_batches: Number of batches to get vote_accuracy
    :param learning_rate: Learning rate for optimiser
    :return: New instance of KerasLearner
    """
    # 2 classes
    pass
