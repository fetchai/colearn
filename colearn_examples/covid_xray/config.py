import tensorflow as tf

from colearn_examples.config import ModelConfig

from .models import CovidXrayLearner


class CovidXrayConfig(ModelConfig):
    def __init__(self, seed=None):
        super().__init__(seed)

        # Training params
        self.optimizer = tf.keras.optimizers.Adam
        self.l_rate = 0.001
        self.l_rate_decay = 1e-5
        self.batch_size = 2

        # Model params
        self.model_type = CovidXrayLearner
        self.feature_size = 64
        self.loss = tf.keras.losses.categorical_crossentropy
        self.n_classes = 3
        self.multi_hot = False

        # Data params
        self.steps_per_epoch = None
        self.test_ratio = 0.2
        self.valid_ratio = 0.25
        self.val_batches = 2  # number of batches used for voting
