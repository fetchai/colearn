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
import random as rand
import sys
import tempfile
from glob import glob
from pathlib import Path

import numpy as np
import torch.nn as nn
import torch.nn.functional as nn_func
import torch.utils.data
from PIL import Image
from torch.utils.data import Dataset
from torchsummary import summary
from typing_extensions import TypedDict

from colearn.training import initial_result, collective_learning_round, set_equal_weights
from colearn.utils.plot import ColearnPlot
from colearn.utils.results import Results, print_results
from colearn_pytorch.pytorch_learner import PytorchLearner
from colearn_pytorch.utils import auc_from_logits

"""
Xray training example using Pytorch

Used dataset:
- Xray, download from kaggle: https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia
  The Chest X-Ray Images (Pneumonia) dataset consists of 5856 grayscale images of various sizes
  in 2 classes (normal/pneumonia).

What the script does:
- Sets up the Torch model and some configuration parameters
- Loads Xray dataset from data_dir
- Randomly splits the train folder and the test folder between multiple learners
- Does multiple rounds of learning process and displays plot with results

To Run: required argument is data_dir: Path to root folder containing data

"""

# define some constants
n_learners = 5
batch_size = 8
seed = 42

testing_mode = bool(os.getenv("COLEARN_EXAMPLES_TEST", ""))  # for testing
n_rounds = 15 if not testing_mode else 1

vote_threshold = 0.5
learning_rate = 0.001
height = 128
width = 128
channels = 1
n_classes = 1

steps_per_epoch = 10
vote_batches = 13  # number of batches used for voting
vote_using_auc = True

no_cuda = False
cuda = not no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")
DataloaderKwargs = TypedDict('DataloaderKwargs', {'num_workers': int, 'pin_memory': bool}, total=False)
kwargs: DataloaderKwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}


class Net(nn.Module):
    """_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
Input (InputLayer)           [(None, 128, 128, 1)]     0
_________________________________________________________________
Conv1_1 (Conv2D)             (None, 128, 128, 32)      320
_________________________________________________________________
bn1 (BatchNormalization)     (None, 128, 128, 32)      128
_________________________________________________________________
pool1 (MaxPooling2D)         (None, 32, 32, 32)        0
_________________________________________________________________
Conv2_1 (Conv2D)             (None, 32, 32, 64)        18496
_________________________________________________________________
bn2 (BatchNormalization)     (None, 32, 32, 64)        256
_________________________________________________________________
global_max_pooling2d (Global (None, 64)                0
_________________________________________________________________
fc1 (Dense)                  (None, 1)                 65
=================================================================
Total params: 19,265
Trainable params: 19,073
Non-trainable params: 192
_________________________________________________________________"""

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(channels, 32, (3, 3), padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, (3, 3), padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64, n_classes)

    def forward(self, x):
        x = nn_func.relu(self.conv1(x))
        x = self.bn1(x)
        x = nn_func.max_pool2d(x, kernel_size=(4, 4))
        x = nn_func.relu(self.conv2(x))
        x = self.bn2(x)
        x = nn_func.max_pool2d(x, kernel_size=(32, 32))
        x = x.view(-1, 64)
        x = self.fc1(x)
        return x  # NB: output is in *logits* - take sigmoid to get predictions


# load data
class XrayDataset(Dataset):
    """X-ray dataset."""

    def __init__(self, data_dir, width=width, height=height, seed=seed, transform=None, train=True, train_ratio=0.96):
        """
        Args:
            data_dir (string): Path to the data directory.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.width, self.height = width, height
        self.cases = list(Path(data_dir).rglob('*.jp*'))  # list of filenames
        if len(self.cases) == 0:
            raise Exception("No data foud in path: " + str(data_dir))

        rand.seed(seed)
        rand.shuffle(self.cases)

        n_cases = int(train_ratio * len(self.cases))
        assert (n_cases > 0), "There are no cases"
        if train:
            self.cases = self.cases[:n_cases]
        else:
            self.cases = self.cases[n_cases:]

        self.diagnosis = []  # list of filenames
        self.normal_data = []
        self.pneumonia_data = []
        for case in self.cases:
            if 'NORMAL' in str(case):
                self.diagnosis.append(0)
                self.normal_data.append(case)
            elif 'PNEUMONIA' in str(case):
                self.diagnosis.append(1)
                self.pneumonia_data.append(case)
            else:
                print(case, " - has invalid category")

        self.transform = transform

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        else:
            idx = [idx]

        batch_size = len(idx)

        # Define two numpy arrays for containing batch data and labels
        batch_data = np.zeros((batch_size, self.width, self.height), dtype=np.float32)
        batch_labels = np.zeros(batch_size, dtype=np.float32)

        for j, index in enumerate(idx):
            batch_data[j] = self.to_rgb_normalize_and_resize(self.cases[index], self.width, self.height)
            batch_labels[j] = self.diagnosis[index]

        sample = (batch_data, batch_labels)
        if self.transform:
            sample = self.transform(sample)

        return sample

    @staticmethod
    def to_rgb_normalize_and_resize(filename, width, height):
        img = Image.open(str(filename))
        img = img.resize((width, height))
        img = img.convert('L')  # convert to greyscale
        img = np.array(img.getdata()).reshape((1, img.size[0], img.size[1])) / 255

        return img


# this is modified from the version in xray/data in order to keep the directory structure
# e.g. when the data is in NORMAL and PNEU directories these will also be in each of the split dirs
def split_to_folders(
        data_dir,
        n_learners,
        data_split=None,
        shuffle_seed=None,
        output_folder=Path(tempfile.gettempdir()) / "xray",
        **_kwargs
):
    if not os.path.isdir(data_dir):
        raise Exception("Data dir does not exist: " + str(data_dir))

    if data_split is None:
        data_split = [1 / n_learners] * n_learners

    local_output_dir = Path(output_folder)

    dir_names = []
    for i in range(n_learners):
        dir_name = local_output_dir / str(i)
        os.system(f"rm -r {dir_name}")
        dir_names.append(dir_name)

    subdirs = glob(os.path.join(data_dir, "*", ""))
    for subdir in subdirs:
        subdir_name = os.path.basename(os.path.split(subdir)[0])

        cases = list(Path(subdir).rglob("*.jp*"))

        if len(cases) == 0:
            raise Exception(f"No data found in path: {str(subdir)}")

        n_cases = len(cases)
        if shuffle_seed is not None:
            np.random.seed(shuffle_seed)
        random_indices = np.random.permutation(np.arange(n_cases))
        start_ind = 0

        for i in range(n_learners):
            stop_ind = start_ind + int(data_split[i] * n_cases)

            cases_subset = [cases[j] for j in random_indices[start_ind:stop_ind]]

            dir_name = local_output_dir / str(i) / subdir_name
            os.makedirs(str(dir_name))

            # make symlinks to required files in directory
            for fl in cases_subset:
                link_name = dir_name / os.path.basename(fl)
                # print(link_name)
                os.symlink(fl, link_name)

            start_ind = stop_ind

    print(dir_names)
    return dir_names


# lOAD DATA
parser = argparse.ArgumentParser()
parser.add_argument("data_dir", help="Path to data directory", type=str)

args = parser.parse_args()

if not Path.is_dir(Path(args.data_dir)):
    sys.exit(f"Data path provided: {args.data_dir} is not a valid path or not a directory")

full_train_data_folder = os.path.join(args.data_dir, 'train')
full_test_data_folder = os.path.join(args.data_dir, 'test')

train_data_folders = split_to_folders(
    full_train_data_folder,
    shuffle_seed=42,
    n_learners=n_learners)

test_data_folders = split_to_folders(
    full_test_data_folder,
    shuffle_seed=42,
    n_learners=2 * n_learners,
    output_folder=Path('/tmp/xray_test')
)

learner_train_dataloaders = []
learner_vote_dataloaders = []
learner_test_dataloaders = []

for i in range(n_learners):
    learner_train_dataloaders.append(torch.utils.data.DataLoader(
        XrayDataset(str(train_data_folders[i]), train_ratio=1),
        batch_size=batch_size, shuffle=True, **kwargs)
    )
    learner_vote_dataloaders.append(torch.utils.data.DataLoader(
        XrayDataset(str(test_data_folders[i]), train_ratio=1),
        batch_size=batch_size, shuffle=True, **kwargs)
    )
    learner_test_dataloaders.append(torch.utils.data.DataLoader(
        XrayDataset(str(test_data_folders[i + n_learners]), train_ratio=1),
        batch_size=batch_size, shuffle=True, **kwargs)
    )

if vote_using_auc:
    learner_vote_kwargs = dict(
        vote_criterion=auc_from_logits,
        minimise_criterion=False)
    score_name = "auc"
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
        criterion=nn.BCEWithLogitsLoss(
            reduction='mean'),
        num_train_batches=steps_per_epoch,
        num_test_batches=vote_batches,
        **learner_vote_kwargs  # type: ignore[arg-type]
    )

    all_learner_models.append(learner)

set_equal_weights(all_learner_models)

# print a summary of the model architecture
summary(all_learner_models[0].model, input_size=(1, width, height), device=str(device))

# Now we're ready to start collective learning
# Get initial accuracy
results = Results()
results.data.append(initial_result(all_learner_models))

plot = ColearnPlot(score_name=score_name)

for round_index in range(n_rounds):
    results.data.append(
        collective_learning_round(all_learner_models,
                                  vote_threshold, round_index)
    )
    print_results(results)

    plot.plot_results_and_votes(results)

plot.block()

print("Colearn Example Finished!")
