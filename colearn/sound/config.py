import tensorflow.compat.v1 as tf

from .models import SoundConv2DLearner


tf.disable_v2_behavior()


def load_config(config):
    # Training params
    config.optimizer = tf.keras.optimizers.Adam
    config.l_rate = 0.001
    config.l_rate_decay = 1e-5
    config.batch_size = 16

    # Model params
    config.model_type = SoundConv2DLearner

    config.freq_coeficients = 40
    config.max_padded_len = 11
    config.loss = "sparse_categorical_crossentropy"
    config.n_classes = 10
    config.multi_hot = False

    # Data params
    config.steps_per_epoch = 100

    config.train_ratio = 0.9
    config.val_batches = 6  # number of batches used for voting
    config.test_ratio = 1 - config.train_ratio
