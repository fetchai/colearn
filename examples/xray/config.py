import tensorflow.compat.v1 as tf

from examples.config import ModelConfig

from .models import XraySuperminiLearner


tf.disable_v2_behavior()


class XrayConfig(ModelConfig):
    def __init__(self, seed=None):
        super().__init__(seed)

        # Training params
        self.optimizer = tf.keras.optimizers.Adam
        self.l_rate = 0.001
        self.l_rate_decay = 1e-5
        self.batch_size = 8

        # Model params
        self.model_type = XraySuperminiLearner
        self.width = 128
        self.height = 128
        self.loss = "binary_crossentropy"
        self.n_classes = 1
        self.multi_hot = False

        # Data params
        self.steps_per_epoch = 10
        self.train_ratio = 0.92
        self.val_batches = 13  # number of batches used for voting
        self.test_ratio = 1 - self.train_ratio
