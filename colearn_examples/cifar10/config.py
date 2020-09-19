import tensorflow.compat.v1 as tf
from .models import CIFAR10Conv2Learner

tf.disable_v2_behavior()


def load_config(config):
    # Training params
    config.optimizer = tf.keras.optimizers.Adam
    config.l_rate = 0.001
    config.l_rate_decay = 1e-5
    config.batch_size = 64

    # Model params
    config.model_type = CIFAR10Conv2Learner

    config.width = 32
    config.height = 32
    config.loss = "sparse_categorical_crossentropy"
    config.n_classes = 10
    config.multi_hot = False

    # Data params
    config.steps_per_epoch = 100

    config.train_ratio = 0.96
    config.val_batches = 2  # number of batches used for voting
    config.test_ratio = 0.02