from torch.utils.data import Dataset
import random as rand
from pathlib import Path
from glob import glob


import numpy as np

import cv2
import tempfile
import os
import torch

import torch.nn as nn
import torch.nn.functional as nn_func


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



default_config = \
    {
        "height": 128,
        "width": 128,
        "seed": None,

    }

# load data
class XrayDataset(Dataset):
    """X-ray dataset."""

    def __init__(self, data_dir, dataset_config=None, transform=None, train=True, train_ratio=0.96):
        """
        Args:
            data_dir (string): Path to the data directory.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        # Merge default_config with dataset_config
        current_config = default_config.copy()
        if dataset_config is not None:
            current_config.update(dataset_config)

        self.width, self.height = current_config["width"], current_config["height"]
        self.seed = current_config["seed"]

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
        img = cv2.imread(str(filename))
        img = cv2.resize(img, (width, height))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img.astype(np.float32) / 255.

        return img


# this is modified from the version in xray/data in order to keep the directory structure
# e.g. when the data is in NORMAL and PNEU directories these will also be in each of the split dirs
def split_to_folders(
        data_dir,
        n_learners,
        data_split=None,
        shuffle_seed=None,
        output_folder=Path(tempfile.gettempdir()) / "xray",

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
