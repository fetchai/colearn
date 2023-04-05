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
import numpy as np
import tensorflow as tf
from tensorflow.python.data.ops.dataset_ops import PrefetchDataset


def normalize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    return tf.cast(image, tf.float32) / 255., label


def _make_loader(images: np.ndarray,
                 labels: np.ndarray,
                 batch_size: int = 32,
                 dp_enabled: bool = False) -> PrefetchDataset:
    """
    Converts array of images and labels to Tensorflow dataset
    :param images: Numpy array of input data
    :param labels:  Numpy array of output labels
    :param batch_size: Batch size
    :return: Shuffled Tensorflow prefetch dataset holding images and labels
    """
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    n_datapoints = images.shape[0]

    dataset = dataset.cache()
    dataset = dataset.shuffle(n_datapoints)
    # tf privacy expects fix batch sizes, thus drop_remainder=True
    dataset = dataset.batch(batch_size, drop_remainder=dp_enabled)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset
