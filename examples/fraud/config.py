import tensorflow.compat.v1 as tf
from .models import FraudSVMLearner

tf.disable_v2_behavior()


def load_config(config):
    # Training params
    config.batch_size = 10000

    # Model params
    config.model_type = FraudSVMLearner

    config.input_classes = 431
    config.loss = "binary_crossentropy"
    config.n_classes = 1
    config.multi_hot = False

    # Keras only params
    config.loss = "sparse_categorical_crossentropy"
    config.optimizer = tf.keras.optimizers.Adam
    config.l_rate = 0.001
    config.l_rate_decay = 1e-5

    # Data params
    config.steps_per_epoch = 1

    config.train_ratio = 0.8
    config.val_batches = 1  # number of batches used for voting
    config.test_ratio = 1 - config.train_ratio
