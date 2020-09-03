from enum import Enum


class TrainingMode(Enum):
    COLLABORATIVE = 1
    INDIVIDUAL = 2


class TrainingData(Enum):
    MNIST = 1
    XRAY = 2
    FRAUD = 3


class Config:
    def __init__(
        self,
        main_data_dir=None,
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
        self.main_data_dir = main_data_dir
        self.data = task if task in TrainingData else TrainingData[task]
        self.total_data_fraction = 1.0
        self.data_split = (
            data_split or [self.total_data_fraction / n_learners] * n_learners
        )
        assert len(self.data_split) == n_learners
        total_ds = sum(self.data_split)
        if total_ds > 1:
            self.data_split = [x / total_ds for x in self.data_split]

        # pylint: disable=import-outside-toplevel
        if self.data == TrainingData.XRAY:
            from examples.xray.dataset import Xray
            self.dataset = Xray
        elif self.data == TrainingData.MNIST:
            from examples.mnist.dataset import Mnist
            self.dataset = Mnist

        elif self.data == TrainingData.FRAUD:
            from examples.fraud.dataset import Fraud
            self.dataset = Fraud

        # Load config
        self.dataset.load_config(self)

        self.class_labels = range(self.n_classes)

        # None means random seed
        self.generator_seed = seed
        self.shuffle_seed = seed

        self.train_augment = False
        self.test_augment = False

        self.plot_results = True
