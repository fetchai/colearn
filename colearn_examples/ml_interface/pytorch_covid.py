# ------------------------------------------------------------------------------
#
#   Copyright 2021 Fetch.AI Limited
#
#   Licensed under the Creative Commons Attribution-NonCommercial International
#   License, Version 4.0 (the "License"); you may not use this file except in
#   compliance with the License. You may obtain a copy of the License at
#
#       http://creativecommons.org/licenses/by-nc/4.0/legalcode
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
# ------------------------------------------------------------------------------
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch.nn as nn
import torch.nn.functional as nn_func
import torch.utils.data
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset
from torchsummary import summary
from typing_extensions import TypedDict

from colearn.training import initial_result, collective_learning_round, set_equal_weights
from colearn.utils.plot import ColearnPlot
from colearn.utils.results import Results, print_results
from colearn_pytorch.pytorch_learner import PytorchLearner
from colearn_pytorch.utils import categorical_accuracy, prepare_data_split_list

"""
COVID-XRAY training example using PyTorch

Used dataset:
- contains 478 normal samples, 478 covid samples and 478 pneumonia samples
- Each sample is vector of 252 feature values extracted from xray mages

What script does:
- Loads XRAY dataset .mat files with normal, pneumonia and covid samples and normalizes them
- Applies PCA to reduce 252 input features to 64 features
- Randomly splits dataset between multiple learners
- Does multiple rounds of learning process and displays plot with results
"""

# define some constants
n_learners = 5
batch_size = 32
seed = 42

testing_mode = bool(os.getenv("COLEARN_EXAMPLES_TEST", ""))  # for testing
n_rounds = 50 if not testing_mode else 1

vote_threshold = 0.5
train_fraction = 0.6
vote_fraction = 0.2
learning_rate = 0.001
input_width = 64
n_classes = 3
vote_batches = 2
vote_on_accuracy = True

no_cuda = False
cuda = not no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")
DataloaderKwargs = TypedDict('DataloaderKwargs', {'num_workers': int, 'pin_memory': bool}, total=False)
kwargs: DataloaderKwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

# lOAD DATA
parser = argparse.ArgumentParser()
parser.add_argument("data_dir", help="Path to data directory", type=str)

args = parser.parse_args()

if not Path.is_dir(Path(args.data_dir)):
    sys.exit(f"Data path provided: {args.data_dir} is not a valid path or not a directory")

covid_data = sio.loadmat(os.path.join(args.data_dir, 'covid.mat'))['covid']
normal_data = sio.loadmat(os.path.join(args.data_dir, 'normal.mat'))['normal']
pneumonia_data = sio.loadmat(os.path.join(args.data_dir, 'pneumonia.mat'))['pneumonia']

data = np.concatenate((covid_data[:, :-1], normal_data[:, :-1], pneumonia_data[:, :-1]), axis=0).astype(np.float32)
labels = np.concatenate((covid_data[:, -1], normal_data[:, -1], pneumonia_data[:, -1]), axis=0).astype(int)

# Normalise data
min_max_scaler = MinMaxScaler()
data = min_max_scaler.fit_transform(data)

transformer = KernelPCA(n_components=input_width, kernel='linear')
data = transformer.fit_transform(data)

# Create tensor dataset
data_tensor = torch.FloatTensor(data)
labels_tensor = torch.LongTensor(labels)
dataset = TensorDataset(data_tensor, labels_tensor)

# Split dataset to train and test part
n_train = int(train_fraction * len(dataset))
n_vote = int(vote_fraction * len(dataset))
n_test = len(dataset) - n_train - n_vote
train_data, vote_data, test_data = torch.utils.data.random_split(dataset, [n_train, n_vote, n_test])

# Split train set between learners
parts = prepare_data_split_list(train_data, n_learners)
learner_train_data = torch.utils.data.random_split(train_data, parts)
learner_train_dataloaders = [torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True, **kwargs)
                             for ds in learner_train_data]

# Split vote set between learners
parts = prepare_data_split_list(vote_data, n_learners)
learner_vote_data = torch.utils.data.random_split(vote_data, parts)
learner_vote_dataloaders = [torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True, **kwargs)
                            for ds in learner_vote_data]

# Split test set between learners
parts = prepare_data_split_list(test_data, n_learners)
learner_test_data = torch.utils.data.random_split(test_data, parts)
learner_test_dataloaders = [torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True, **kwargs)
                            for ds in learner_test_data]


# define the neural net architecture in Pytorch
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_width, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 16)
        self.fc6 = nn.Linear(16, n_classes)

    def forward(self, x):
        x = nn_func.dropout(nn_func.relu(self.fc1(x)), 0.2)
        x = nn_func.dropout(nn_func.relu(self.fc2(x)), 0.2)
        x = nn_func.dropout(nn_func.relu(self.fc3(x)), 0.2)
        x = nn_func.dropout(nn_func.relu(self.fc4(x)), 0.2)
        x = nn_func.dropout(nn_func.relu(self.fc5(x)), 0.2)
        x = self.fc6(x)

        return nn_func.log_softmax(x, dim=1)


if vote_on_accuracy:
    learner_vote_kwargs = dict(
        vote_criterion=categorical_accuracy,
        minimise_criterion=False)
    score_name = "categorical accuracy"
else:
    learner_vote_kwargs = {}
    score_name = "loss"

# Make n instances of PytorchLearner with model and torch dataloaders
all_learner_models = []
for i in range(n_learners):
    model = Net()
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
    learner = PytorchLearner(
        model=model,
        train_loader=learner_train_dataloaders[i],
        vote_loader=learner_vote_dataloaders[i],
        test_loader=learner_test_dataloaders[i],
        device=device,
        optimizer=opt,
        criterion=torch.nn.NLLLoss(),
        num_test_batches=vote_batches,
        **learner_vote_kwargs  # type: ignore[arg-type]
    )

    all_learner_models.append(learner)

# Ensure all learners starts with exactly same weights
set_equal_weights(all_learner_models)

# print a summary of the model architecture
summary(all_learner_models[0].model, input_size=(input_width,), device=str(device))

# Now we're ready to start collective learning
# Get initial accuracy
results = Results()
results.data.append(initial_result(all_learner_models))

plot = ColearnPlot(score_name=score_name)

# Do the training
for round_index in range(n_rounds):
    results.data.append(
        collective_learning_round(all_learner_models,
                                  vote_threshold, round_index)
    )
    print_results(results)

    plot.plot_results_and_votes(results)

plot.block()

print("Colearn Example Finished!")
