from enum import Enum


class TrainingMode(Enum):
    COLLECTIVE = 1
    INDIVIDUAL = 2


class TrainingData(Enum):
    MNIST = 1
    XRAY = 2
    FRAUD = 3
    CIFAR10 = 4
    COVID = 5


class ColearnConfig:
    def __init__(
        self,
        task: TrainingData = TrainingData.XRAY,
        n_learners=5,
        seed=None,
        n_epochs=30,
    ):
        # Training params
        self.n_learners = n_learners
        self.n_epochs = n_epochs

        self.vote_threshold = 0.5  # 0.66666
        self.mode = TrainingMode.COLLECTIVE

        self.data = task if isinstance(task, TrainingData) else TrainingData[str(task)]

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

        # DP params
        self.use_dp = False
        self.sample_size = 3300
        self.alphas = list(range(2, 32))
        self.noise_multiplier = 1.2
        self.max_grad_norm = 1.0
