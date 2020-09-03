import tensorflow.compat.v1 as tf

from config import ModelConfig
from .models import XraySuperminiLearner


tf.disable_v2_behavior()


def load_config(config):

    # Training params
    config.optimizer = tf.keras.optimizers.Adam
    config.l_rate = 0.001
    config.l_rate_decay = 1e-5
    config.batch_size = 8

    # Model params
    config.model_type = XraySuperminiLearner

    config.width = 128
    config.height = 128
    config.loss = "binary_crossentropy"
    config.n_classes = 1
    config.multi_hot = False

    # Data params
    config.steps_per_epoch = 10

    config.train_ratio = 0.92
    config.val_batches = 13  # number of batches used for voting
    config.test_ratio = 1 - config.train_ratio


class XrayConfig(ModelConfig):
    def __init__(self):
        super().__init__()
        load_config(self)
