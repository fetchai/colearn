from enum import Enum


class TrainingMode(Enum):
    COLLABORATIVE = 1
    INDIVIDUAL = 2


class TrainingData(Enum):
    MNIST = 1
    XRAY = 2
    FRAUD = 3
    CIFAR10 = 4


class ColearnConfig:
    def __init__(
        self,
        data_dir=None,
        task: TrainingData = TrainingData.XRAY,
        n_learners=5,
        data_split=None,
        seed=None,
        n_epochs=30,
    ):

        # Training params
        self.n_learners = n_learners
        self.n_epochs = n_epochs

        self.vote_threshold = 0.5  # 0.66666
        self.mode = TrainingMode.COLLABORATIVE

        # Data params
        self.data_dir = data_dir
        self.data = task if isinstance(task, TrainingData) else TrainingData[str(task)]
        self.total_data_fraction = 1.0
        self.data_split = (
            data_split or [self.total_data_fraction / n_learners] * n_learners
        )
        assert len(self.data_split) == n_learners
        total_ds = sum(self.data_split)
        if total_ds > 1:
            self.data_split = [x / total_ds for x in self.data_split]

        # None means random seed
        self.shuffle_seed = seed

        self.plot_results = True


class ModelConfig:
    def __init__(self, seed=None):
        # Training params
        self.optimizer = None
        self.l_rate = 0.001
        self.l_rate_decay = 1e-5
        self.batch_size = 64

        # Model params
        self.model_type = None

        self.width = 28
        self.height = 28
        self.loss = "sparse_categorical_crossentropy"
        self.n_classes = 10
        self.multi_hot = False
        self.class_labels = range(self.n_classes)

        # Data params
        self.steps_per_epoch = None

        self.train_ratio = 0.8
        self.val_batches = 2  # number of batches used for voting
        self.test_ratio = 1 - self.train_ratio

        self.train_augment = False
        self.test_augment = False

        self.generator_seed = seed
