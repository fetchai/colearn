import tensorflow.compat.v1 as tf

from colearn.kaggle_xray.models import XrayDropout2Learner


tf.disable_v2_behavior()


def load_config(config):
    # Training params
    config.optimizer = tf.keras.optimizers.Adam
    config.l_rate = 0.0001
    config.l_rate_decay = 1e-5
    config.batch_size = 8

    # Model params
    config.model_type = XrayDropout2Learner

    config.width = 224
    config.height = 224
    config.loss = "binary_crossentropy"
    config.n_classes = 1
    config.multi_hot = False

    # Data params
    config.steps_per_epoch = 100

    config.train_ratio = 0.96
    config.val_batches = 13  # number of batches used for voting
    config.test_ratio = 1 - config.train_ratio
