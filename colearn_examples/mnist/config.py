import tensorflow as tf

from colearn_examples.config import ModelConfig

from .models import MNISTSuperminiLearner


class MNISTConfig(ModelConfig):
    def __init__(self, seed=None):
        super().__init__(seed)

        # Training params
        self.optimizer = tf.keras.optimizers.Adam
        self.l_rate = 0.001
        self.l_rate_decay = 1e-5
        self.batch_size = 64

        # Model params
        self.model_type = MNISTSuperminiLearner
        self.width = 28
        self.height = 28
        self.loss = "sparse_categorical_crossentropy"
        self.n_classes = 10
        self.multi_hot = False

        # Data params
        self.steps_per_epoch = None
        self.train_ratio = 0.8
        self.val_batches = 2  # number of batches used for voting
        self.test_ratio = 1 - self.train_ratio
