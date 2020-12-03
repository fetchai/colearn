import tensorflow as tf
from colearn_examples.config import ModelConfig

from .models import CIFAR10Resnet50Learner


class CIFAR10Config(ModelConfig):
    def __init__(self, seed=None):
        super().__init__(seed)
        # Training params
        self.optimizer = tf.keras.optimizers.Adam
        self.l_rate = 0.001
        self.l_rate_decay = 1e-5
        self.batch_size = 64

        self.metrics = ["accuracy"]

        # Model params
        self.model_type = CIFAR10Resnet50Learner
        self.width = 32
        self.height = 32
        self.loss = "sparse_categorical_crossentropy"
        self.n_classes = 10
        self.multi_hot = False

        # Data params
        self.steps_per_epoch = 100
        self.train_ratio = 0.96
        self.val_batches = 2  # number of batches used for voting
        self.test_ratio = 0.02
