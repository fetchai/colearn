import pickle
from pathlib import Path
import torch

from colearn.basic_learner import LearnerData
from colearn_examples.mnist import split_to_folders
from colearn_examples.mnist.data import train_generator as data_generator
from colearn_examples.pytorch_learner import PytorchLearner
import torch.nn as nn
import torch.nn.functional as nn_func

from colearn_examples.training import collective_learning_round, initial_result
from colearn_examples.utils.data import split_by_chunksizes
from colearn_examples.utils.plot import plot_results, plot_votes
from colearn_examples.utils.results import Results


# The purpose of this script is to demonstrate how to write your own models for colearn
# by writing a subclass of PytorchLearner

# To write your own pytorch model, just subclass PytorchLearner and implement _get_model
class MNISTPytorchLearner(PytorchLearner):
    def _get_model(self):
        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.conv1 = nn.Conv2d(1, 20, 5, 1)
                self.conv2 = nn.Conv2d(20, 50, 5, 1)
                self.fc1 = nn.Linear(4 * 4 * 50, 500)
                self.fc2 = nn.Linear(500, 10)

            def forward(self, x):
                x = nn_func.relu(self.conv1(x))
                x = nn_func.max_pool2d(x, 2, 2)
                x = nn_func.relu(self.conv2(x))
                x = nn_func.max_pool2d(x, 2, 2)
                x = x.view(-1, 4 * 4 * 50)
                x = nn_func.relu(self.fc1(x))
                x = self.fc2(x)
                return nn_func.log_softmax(x, dim=1)

        model = Net()
        return model


# also write a function to load the dataset and return a LearnerData instance
def load_learner_data(data_dir, batch_size, width, height, train_ratio, test_ratio, generator_seed):
    data = LearnerData()
    data.train_batch_size = batch_size
    IMAGE_FL = "images.pickle"
    LABEL_FL = "labels.pickle"

    images = pickle.load(open(Path(data_dir) / IMAGE_FL, "rb"))
    labels = pickle.load(open(Path(data_dir) / LABEL_FL, "rb"))

    [[train_images, test_images], [train_labels, test_labels]] = \
        split_by_chunksizes([images, labels], [train_ratio, test_ratio])

    data.train_data_size = len(train_images)

    data.train_gen = data_generator(
        train_images, train_labels, batch_size,
        width,
        height,
        generator_seed,
        augmentation=False
    )
    data.val_gen = data_generator(
        train_images, train_labels, batch_size,
        width,
        height,
        generator_seed,
        augmentation=False
    )

    data.test_data_size = len(test_images)

    data.test_gen = data_generator(
        test_images,
        test_labels,
        batch_size,
        width,
        height,
        generator_seed,
        augmentation=False
    )

    data.test_batch_size = batch_size
    return data


# Now we're ready to start training!
# First define some config values:
n_learners = 5
batch_size = 64
width, height, train_ratio, test_ratio, generator_seed = 28, 28, 0.8, 0.2, 42

# This step downloads the MNIST dataset and splits it into folders
data_split = [1 / n_learners] * n_learners
learner_data_folders = split_to_folders("", generator_seed, data_split, n_learners)

# Build the learner datasets using the functions defined above
learner_datasets = []
for i in range(n_learners):
    learner_datasets.append(
        load_learner_data(learner_data_folders[i], batch_size, width, height, train_ratio, test_ratio, generator_seed)
    )


# Define the ModelConfig which passes configuration values to the PytorchLearner
class ModelConfig:
    def __init__(self):
        # Training params
        self.optimizer = torch.optim.Adam
        self.l_rate = 0.001
        self.l_rate_decay = 1e-5
        self.batch_size = batch_size

        self.metrics = ["accuracy"]

        # Model params
        self.width = width
        self.height = height
        self.loss = nn_func.nll_loss
        self.n_classes = 10
        self.multi_hot = False

        # Data params
        self.steps_per_epoch = None  # None means use whole dataset
        self.train_ratio = 0.8
        self.val_batches = 2  # number of batches used for voting
        self.test_ratio = 1 - self.train_ratio
        self.class_labels = [str(j) for j in range(self.n_classes)]

        # Differential Privacy params
        self.use_dp = False


# Create a config instance and then create a list of learners all with the same weights
config = ModelConfig()
first_learner = MNISTPytorchLearner(config, data=learner_datasets[0])
learners = [first_learner]
for i in range(1, n_learners):
    nth_learner = first_learner.clone(data=learner_datasets[i])
    learners.append(nth_learner)

# Get initial accuracy
results = Results()
results.data.append(initial_result(learners))

# Now to do collective learning!
n_rounds = 15
vote_threshold = 0.5
for i in range(n_rounds):
    results.data.append(
        collective_learning_round(learners,
                                  vote_threshold, i)
    )

    plot_results(results, n_learners, block=False)
    plot_votes(results, block=False)

plot_results(results, n_learners, block=False)
plot_votes(results, block=True)
