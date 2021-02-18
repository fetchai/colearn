# ------------------------------------------------------------------------------
#
#   Copyright 2021 Fetch.AI Limited
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
# ------------------------------------------------------------------------------
import os
import random as rand
import tempfile
from enum import Enum
from glob import glob
from pathlib import Path
from typing import Tuple, Optional, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as nn_func
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from typing_extensions import TypedDict

from colearn_pytorch.pytorch_learner import PytorchLearner
from .utils import auc_from_logits


class ModelType(Enum):
    CONV2D = 1


def prepare_model(model_type: ModelType) -> nn.Module:
    """
    Creates a new instance of selected Keras model
    :param model_type: Enum that represents selected model type
    :return: New instance of Pytorch model
    """

    if model_type == ModelType.CONV2D:
        return TorchXrayConv2DModel()
    else:
        raise Exception("Model %s not part of the ModelType enum" % model_type)


def prepare_learner(model_type: ModelType,
                    data_loaders: Tuple[DataLoader, DataLoader],
                    learning_rate: float = 0.001,
                    steps_per_epoch: int = 40,
                    vote_batches: int = 10,
                    no_cuda: bool = False,
                    vote_on_accuracy: bool = True,
                    **_kwargs) -> PytorchLearner:
    """
    Creates new instance of PytorchLearner
    :param model_type: Enum that represents selected model type
    :param data_loaders: Tuple of train_loader and test_loader
    :param learning_rate: Learning rate for optimiser
    :param steps_per_epoch: Number of batches per training epoch
    :param vote_batches: Number of batches to get vote_score
    :param no_cuda: True = disable GPU computing
    :param vote_on_accuracy: True = vote on accuracy metric, False = vote on loss
    :param _kwargs: Residual parameters not used by this function
    :return: New instance of PytorchLearner
    """

    cuda = not no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")

    model = prepare_model(model_type)

    if vote_on_accuracy:
        learner_vote_kwargs = dict(
            vote_criterion=auc_from_logits,
            minimise_criterion=False)
    else:
        learner_vote_kwargs = {}

    # Make n instances of PytorchLearner with model and torch dataloaders
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
    learner = PytorchLearner(
        model=model,
        train_loader=data_loaders[0],
        test_loader=data_loaders[1],
        device=device,
        optimizer=opt,
        criterion=nn.BCEWithLogitsLoss(
            # pos_weight=pos_weight,
            reduction='mean'),
        num_train_batches=steps_per_epoch,
        num_test_batches=vote_batches,
        **learner_vote_kwargs  # type: ignore[arg-type]
    )

    return learner


def prepare_data_loaders(train_folder: str,
                         test_folder: Optional[str] = None,
                         train_ratio: float = 0.96,
                         batch_size: int = 8,
                         no_cuda: bool = False,
                         **_kwargs) -> Tuple[DataLoader, DataLoader]:
    """
    Load training data from folders and create train and test dataloader

    :param train_folder: Path to training dataset
    :param test_folder: Path to test dataset
    :param train_ratio: When test_folder is not specified what portion of train_data should be used as test set
    :param batch_size:
    :param no_cuda: Disable GPU computing
    :param kwargs:
    :return: Tuple of train_loader and test_loader
    """

    cuda = not no_cuda and torch.cuda.is_available()
    DataloaderKwargs = TypedDict('DataloaderKwargs', {'num_workers': int, 'pin_memory': bool}, total=False)
    loader_kwargs: DataloaderKwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

    if test_folder is not None:
        train_loader = DataLoader(
            XrayDataset(train_folder, train=True, train_ratio=1.0),
            batch_size=batch_size, shuffle=True, **loader_kwargs)

        test_loader = DataLoader(
            XrayDataset(test_folder, train=True, train_ratio=1.0),
            batch_size=batch_size, shuffle=True, **loader_kwargs)
    else:
        train_loader = DataLoader(
            XrayDataset(train_folder, train=True, train_ratio=train_ratio),
            batch_size=batch_size, shuffle=True, **loader_kwargs)

        test_loader = DataLoader(
            XrayDataset(train_folder, train=False, train_ratio=train_ratio),
            batch_size=batch_size, shuffle=True, **loader_kwargs)

    return train_loader, test_loader


class TorchXrayConv2DModel(nn.Module):
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
        super(TorchXrayConv2DModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, (3, 3), padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, (3, 3), padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64, 1)

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

    def __init__(self,
                 data_dir: str,
                 transform=None,
                 train: bool = True,
                 train_ratio: float = 0.96,
                 seed: Optional[int] = None,
                 width: int = 128,
                 height: int = 128,
                 **_kwargs):
        """
        :param data_dir (string): Path to the data directory.
        :param transform (callable, optional): Optional transform to be applied
        :param train: True = data_dir contains train set, False = data_dir contains test set
        :param train_ratio: What fraction of samples from data_dir will be used as train set
                            Rest of samples will be used as test set
        :param seed: Shuffling seed
        :param width: Resize images width
        :param height: Resize images height
        :param _kwargs: Residual parameters not used by this function
        """
        self.width, self.height = width, height
        self.seed = seed

        self.cases = list(Path(data_dir).rglob('*.jp*'))  # list of filenames
        if len(self.cases) == 0:
            raise Exception("No data found in path: " + str(data_dir))

        if self.seed is not None:
            rand.seed(self.seed)

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
        """
        :return: Number of available samples
        """
        return len(self.cases)

    def __getitem__(self, idx) -> Tuple[np.array, np.array]:
        """
        :param idx: Array of indices
        :return: batch of samples as tuple (data, labels)
        """
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
    def to_rgb_normalize_and_resize(
            filename: str,
            width: int,
            height: int) -> np.array:
        """
        Loads, resize and normalize image
        :param filename: Path to image
        :param width: Output width
        :param height: Output height
        :return: Resized and normalizes image as np.array
        """
        img = Image.open(str(filename))
        img = img.resize((width, height))
        img = img.convert('L')  # convert to greyscale
        img = np.array(img.getdata()).reshape((1, img.size[0], img.size[1])) / 255

        return img


# this is modified from the version in xray/data in order to keep the directory structure
# e.g. when the data is in NORMAL and PNEU directories these will also be in each of the split dirs
def split_to_folders(
        data_dir: str,
        n_learners: int,
        data_split: Optional[List[float]] = None,
        shuffle_seed: Optional[int] = None,
        output_folder: Optional[Path] = None,
        train: bool = True,
        **_kwargs
) -> List[str]:
    """
    :param data_dir: Path to directory containing xray images
    :param n_learners: Number of parts for splitting
    :param data_split:  List of percentage portions for each subset
    :param shuffle_seed: Seed for shuffling
    :param output_folder: Folder where splitted parts will be stored as numbered subfolders
    :param train: True = is training set, False = is test set
    :param _kwargs: Residual parameters not used by this function
    :return:
    """
    if output_folder is None:
        if train:
            output_folder = Path(tempfile.gettempdir()) / "train_xray"
        else:
            output_folder = Path(tempfile.gettempdir()) / "test_xray"

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

            # Prepare output directories
            dir_name = local_output_dir / str(i)
            os.makedirs(str(dir_name / "NORMAL"), exist_ok=True)
            os.makedirs(str(dir_name / "PNEUMONIA"), exist_ok=True)

            # make symlinks to required files in directory
            for fl in cases_subset:
                if 'NORMAL' in str(fl):
                    case_type = "NORMAL"
                elif 'PNEUMONIA' in str(fl):
                    case_type = "PNEUMONIA"
                else:
                    print(fl, " - has invalid category")
                    continue

                link_name = dir_name / case_type / os.path.basename(fl)
                # print(link_name)
                os.symlink(fl, link_name)

            start_ind = stop_ind

    print(dir_names)
    return [str(x) for x in dir_names]
