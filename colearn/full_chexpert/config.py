from colearn.utils import missing_values_cross_entropy_loss
import tensorflow.compat.v1 as tf
from ..kaggle_xray.models import XraySuperminiLearner

tf.disable_v2_behavior()


def load_config(config):
    # Training params
    config.optimizer = tf.keras.optimizers.Adam
    config.l_rate = 0.0001
    config.l_rate_decay = 1e-5
    config.batch_size = 8

    # Model params
    config.model_type = XraySuperminiLearner

    config.width = 224
    config.height = 224
    config.loss = missing_values_cross_entropy_loss
    config.n_classes = 14
    config.multi_hot = True

    # Data params
    config.steps_per_epoch = 100

    config.train_ratio = 0.98
    config.val_batches = 13  # number of batches used for voting
    config.test_ratio = 1 - config.train_ratio
