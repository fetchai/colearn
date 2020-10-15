import tensorflow as tf

from ..config import ModelConfig

from .models import FraudSVMLearner


class FraudConfig(ModelConfig):
    def __init__(self, seed=None):
        super().__init__(seed)

        # Training params
        self.batch_size = 10000

        # Model params
        self.model_type = FraudSVMLearner
        self.input_classes = 431
        self.loss = "binary_crossentropy"
        self.n_classes = 1
        self.multi_hot = False

        # Keras only params
        self.loss = "sparse_categorical_crossentropy"
        self.optimizer = tf.keras.optimizers.Adam
        self.l_rate = 0.001
        self.l_rate_decay = 1e-5

        # Data params
        self.steps_per_epoch = 1
        self.train_ratio = 0.8
        self.val_batches = 1  # number of batches used for voting
        self.test_ratio = 1 - self.train_ratio
