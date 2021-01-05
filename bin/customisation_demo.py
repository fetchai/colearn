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
# by writing a subclass of PytorchLearner.
# More explanation of this code can be found [here](../docs/tutorial_on_customisation.md)

# To write your own pytorch model, just subclass PytorchLearner and implement _get_model
class MNISTPytorchLearner(PytorchLearner):
    def _get_model(self):
        width = self.config.width
        height = self.config.height

        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.conv1 = nn.Conv2d(1, 20, 5, 1)
                self.conv2 = nn.Conv2d(20, 50, 5, 1)
                self.fc1 = nn.Linear(4 * 4 * 50, 500)
                self.fc2 = nn.Linear(500, 10)

            def forward(self, x):
                x = nn_func.relu(self.conv1(x.view(-1, 1, height, width)))
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
    image_fl = "images.pickle"
    label_fl = "labels.pickle"

    images = pickle.load(open(Path(data_dir) / image_fl, "rb"))
    labels = pickle.load(open(Path(data_dir) / label_fl, "rb"))

    [[train_images, test_images], [train_labels, test_labels]] = \
        split_by_chunksizes([images, labels], [train_ratio, test_ratio])

    train_data_size = len(train_images)

    train_gen = data_generator(
        train_images, train_labels, batch_size,
        width,
        height,
        generator_seed,
        augmentation=False
    )
    val_gen = data_generator(
        train_images, train_labels, batch_size,
        width,
        height,
        generator_seed,
        augmentation=False
    )

    test_data_size = len(test_images)

    test_gen = data_generator(
        test_images,
        test_labels,
        batch_size,
        width,
        height,
        generator_seed,
        augmentation=False
    )

    return LearnerData(train_gen=train_gen,
                       val_gen=val_gen,
                       test_gen=test_gen,
                       train_data_size=train_data_size,
                       test_data_size=test_data_size,
                       train_batch_size=batch_size,
                       test_batch_size=batch_size)


# Now we're ready to start training!
# First define some config values:
n_learners = 5
batch_size = 64
image_width = 28
image_height = 28
train_fraction = 0.8
test_fraction = 0.2
seed = 42
n_rounds = 15
vote_threshold = 0.5

# This step downloads the MNIST dataset and splits it into folders
data_split = [1 / n_learners] * n_learners
learner_data_folders = split_to_folders("", seed, data_split, n_learners)

# Build the learner datasets using the functions defined above
learner_datasets = [
    load_learner_data(learner_data_folders[i], batch_size,
                      image_width, image_height,
                      train_fraction, test_fraction, seed)
    for i in range(n_learners)]


# Define the ModelConfig which passes configuration values to the PytorchLearner
class ModelConfig:
    def __init__(self):
        # Training params
        self.optimizer = torch.optim.Adam
        self.l_rate = 0.001
        self.l_rate_decay = 1e-5
        self.loss = nn_func.nll_loss
        self.n_classes = 10
        self.multi_hot = False

        # Model params
        self.width = image_width
        self.height = image_height

        # Data params
        self.steps_per_epoch = None  # None means use whole dataset
        self.train_ratio = train_fraction
        self.val_batches = 2  # number of batches used for voting
        self.test_ratio = test_fraction
        self.batch_size = batch_size

        # Differential Privacy params
        self.use_dp = False


# Create a config instance and then create a list of learners all with the same weights
config = ModelConfig()
learners = [MNISTPytorchLearner(config, data=learner_datasets[i])
            for i in range(n_learners)]

# Get initial score
results = Results()
results.data.append(initial_result(learners))

# Now to do collective learning!
for i in range(n_rounds):
    results.data.append(
        collective_learning_round(learners,
                                  vote_threshold, i)
    )

    plot_results(results, n_learners)
    plot_votes(results)

plot_results(results, n_learners)
plot_votes(results, block=True)
