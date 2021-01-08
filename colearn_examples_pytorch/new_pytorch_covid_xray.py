from torchsummary import summary
import torch.utils.data

from colearn_examples_pytorch.new_pytorch_learner import NewPytorchLearner

import torch.nn as nn
import torch.nn.functional as nn_func

from colearn_examples.training import initial_result, collective_learning_round, set_equal_weights
from colearn_examples.utils.plot import plot_results, plot_votes
from colearn_examples.utils.results import Results

import os
import scipy.io as sio
import numpy as np

"""
MNIST training example using PyTorch

Used dataset:
- MNIST is set of 60 000 black and white hand written digits images of size 28x28x1 in 10 classes

What script does:
- Loads MNIST dataset from torchvision.datasets
- Randomly splits dataset between multiple learners
- Does multiple rounds of learning process and displays plot with results
"""

# define some constants
from colearn_examples_pytorch.utils import categorical_accuracy

n_learners = 5
batch_size = 64
seed = 42
n_epochs = 20
vote_threshold = 0.5
train_fraction = 0.9
learning_rate = 0.001
input_width = 252
n_classes = 3
vote_batches = 2
vote_on_accuracy = True

no_cuda = False
cuda = not no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

data_dir = '/home/jiri/fetch/colearn/examples/covid'

# Load data

covid_data = sio.loadmat(os.path.join(data_dir, 'covid.mat'))['covid']
normal_data = sio.loadmat(os.path.join(data_dir, 'normal.mat'))['normal']
pneumonia_data = sio.loadmat(os.path.join(data_dir, 'pneumonia.mat'))['pneumonia']

data = np.concatenate((covid_data[:, :-1], normal_data[:, :-1], pneumonia_data[:, :-1]), axis=0).astype(np.float32)
labels = np.concatenate((covid_data[:, -1], normal_data[:, -1], pneumonia_data[:, -1]), axis=0).astype(int)

dataset = []
for i in range(len(data)):
    dataset.append([data[i], labels[i]])

n_train = int(train_fraction * len(dataset))
n_test = len(dataset) - n_train
train_data, test_data = torch.utils.data.random_split(dataset, [n_train, n_test])

data_split = [len(train_data) // n_learners] * n_learners
if (sum(data_split) < len(train_data)):
    data_split[-1] += len(train_data) - sum(data_split)

learner_train_data = torch.utils.data.random_split(train_data, data_split)
learner_train_dataloaders = [torch.utils.data.DataLoader(
    ds,
    batch_size=batch_size, shuffle=True, **kwargs) for ds in learner_train_data]


data_split = [len(test_data) // n_learners] * n_learners
if (sum(data_split) < len(test_data)):
    data_split[-1] += len(test_data) - sum(data_split)

learner_test_data = torch.utils.data.random_split(test_data, data_split)
learner_test_dataloaders = [torch.utils.data.DataLoader(
    ds,
    batch_size=batch_size, shuffle=True, **kwargs) for ds in learner_test_data]


# define the neural net architecture in Pytorch
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_width, n_classes)

    def forward(self, x):
        x = self.fc1(x)

        return nn_func.log_softmax(x, dim=1)


if vote_on_accuracy:
    learner_vote_kwargs = dict(
        vote_criterion=categorical_accuracy,
        minimise_criterion=False)
    score_name = "categorical accuracy"
else:
    learner_vote_kwargs = {}
    score_name = "loss"

# Make n instances of NewPytorchLearner with model and torch dataloaders
all_learner_models = []
for i in range(n_learners):
    model = Net()
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
    learner = NewPytorchLearner(
        model=model,
        train_loader=learner_train_dataloaders[i],
        test_loader=learner_test_dataloaders[i],
        device=device,
        optimizer=opt,
        criterion=torch.nn.NLLLoss(),
        num_test_batches=vote_batches,
        **learner_vote_kwargs
    )

    all_learner_models.append(learner)

# Ensure all learners starts with exactly same weights
set_equal_weights(all_learner_models)

# print a summary of the model architecture
summary(all_learner_models[0].model, input_size=(input_width,))

# Now we're ready to start collective learning
# Get initial accuracy
results = Results()
results.data.append(initial_result(all_learner_models))

# Do the training
for epoch in range(n_epochs):
    results.data.append(
        collective_learning_round(all_learner_models,
                                  vote_threshold, epoch)
    )

    plot_results(results, n_learners, score_name=score_name)
    plot_votes(results)

# Plot the final result with votes
plot_results(results, n_learners, score_name=score_name)
plot_votes(results, block=True)
