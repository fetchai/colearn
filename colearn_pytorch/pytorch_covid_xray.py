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
import pickle
import tempfile
from enum import Enum
from pathlib import Path
from typing import Tuple, List, Optional

import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.nn.functional as nn_func
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from typing_extensions import TypedDict

from colearn.utils.data import split_list_into_fractions
from colearn_pytorch.pytorch_learner import PytorchLearner
from .utils import categorical_accuracy

DATA_FL = "data.pickle"
LABEL_FL = "labels.pickle"


class ModelType(Enum):
    MULTILAYER_PERCEPTRON = 1


def prepare_model(model_type: ModelType) -> nn.Module:
    """
    Creates a new instance of selected Keras model
    :param model_type: Enum that represents selected model type
    :return: New instance of Pytorch model
    """

    if model_type == ModelType.MULTILAYER_PERCEPTRON:
        return TorchCovidXrayPerceptronModel()
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
            vote_criterion=categorical_accuracy,
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
        criterion=torch.nn.NLLLoss(),
        num_train_batches=steps_per_epoch,
        num_test_batches=vote_batches,
        **learner_vote_kwargs)  # type: ignore[arg-type]

    return learner


def _make_loader(data: np.array,
                 labels: np.array,
                 batch_size: int,
                 **loader_kwargs) -> DataLoader:
    """
    Converts array of images and labels to Pytorch DataLoader
    :param data: Numpy array of input data
    :param labels: Numpy array of output labels
    :param batch_size: Size of training batch
    :param loader_kwargs: Arguments to be passed to DataLoader
    :return: Shuffled Pytorch DataLoader holding data and labels
    """
    # Create tensor dataset
    data_tensor = torch.FloatTensor(data)
    labels_tensor = torch.LongTensor(labels)
    dataset = TensorDataset(data_tensor, labels_tensor)
    loader = DataLoader(
        dataset,
        batch_size=batch_size, shuffle=True, **loader_kwargs)

    return loader


def prepare_data_loaders(train_folder: str,
                         train_ratio: float = 0.8,
                         batch_size: int = 8,
                         no_cuda: bool = False,
                         **_kwargs) -> Tuple[DataLoader, DataLoader]:
    """
    Load training data from folders and create train and test dataloader

    :param train_folder: Path to training dataset
    :param train_ratio: What portion of train_data should be used as test set
    :param batch_size: Batch size
    :param no_cuda: Disable GPU computing
    :param _kwargs: Residual parameters not used by this function
    :return: Tuple of train_loader and test_loader
    """

    cuda = not no_cuda and torch.cuda.is_available()
    DataloaderKwargs = TypedDict('DataloaderKwargs', {'num_workers': int, 'pin_memory': bool}, total=False)
    loader_kwargs: DataloaderKwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

    data = pickle.load(open(Path(train_folder) / DATA_FL, "rb"))
    labels = pickle.load(open(Path(train_folder) / LABEL_FL, "rb"))

    n_cases = int(train_ratio * len(data))
    assert (n_cases > 0), "There are no cases"

    train_loader = _make_loader(data[:n_cases], labels[:n_cases], batch_size, **loader_kwargs)
    test_loader = _make_loader(data[n_cases:], labels[n_cases:], batch_size, **loader_kwargs)

    return train_loader, test_loader


# define the neural net architecture in Pytorch
class TorchCovidXrayPerceptronModel(nn.Module):
    def __init__(self):
        super(TorchCovidXrayPerceptronModel, self).__init__()
        self.fc1 = nn.Linear(64, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 16)
        self.fc6 = nn.Linear(16, 3)

    def forward(self, x):
        x = nn_func.dropout(nn_func.relu(self.fc1(x)), 0.2)
        x = nn_func.dropout(nn_func.relu(self.fc2(x)), 0.2)
        x = nn_func.dropout(nn_func.relu(self.fc3(x)), 0.2)
        x = nn_func.dropout(nn_func.relu(self.fc4(x)), 0.2)
        x = nn_func.dropout(nn_func.relu(self.fc5(x)), 0.2)
        x = self.fc6(x)

        return nn_func.log_softmax(x, dim=1)


# this is modified from the version in xray/data in order to keep the directory structure
# e.g. when the data is in NORMAL and PNEU directories these will also be in each of the split dirs
def split_to_folders(
        data_dir: str,
        n_learners: int,
        data_split: Optional[List[float]] = None,
        shuffle_seed: Optional[int] = None,
        output_folder: Optional[Path] = None,
        **_kwargs
) -> List[str]:
    """
    Loads preprocessed images with labels from .mat files and splits them to specified number of subsets
    :param data_dir: Directory containing .mat files
    :param n_learners: Number of parts for splitting
    :param data_split: List of percentage portions for each subset
    :param shuffle_seed: Seed for shuffling
    :param output_folder: Folder where splitted parts will be stored as numbered subfolders
    :param _kwargs: Residual parameters not used by this function
    :return: List of folders containing individual subsets
    """
    if output_folder is None:
        output_folder = Path(tempfile.gettempdir()) / "covid_xray"

    if data_split is None:
        data_split = [1 / n_learners] * n_learners

    # Load data
    covid_data = sio.loadmat(os.path.join(data_dir, 'covid.mat'))['covid']
    normal_data = sio.loadmat(os.path.join(data_dir, 'normal.mat'))['normal']
    pneumonia_data = sio.loadmat(os.path.join(data_dir, 'pneumonia.mat'))['pneumonia']

    data = np.concatenate((covid_data[:, :-1], normal_data[:, :-1], pneumonia_data[:, :-1]), axis=0).astype(np.float32)
    labels = np.concatenate((covid_data[:, -1], normal_data[:, -1], pneumonia_data[:, -1]), axis=0).astype(int)

    # Normalise data
    min_max_scaler = MinMaxScaler()
    data = min_max_scaler.fit_transform(data)

    transformer = KernelPCA(n_components=64, kernel='linear')
    data = transformer.fit_transform(data)

    n_datapoints = data.shape[0]
    np.random.seed(shuffle_seed)
    random_indices = np.random.permutation(np.arange(n_datapoints))
    split_indices = split_list_into_fractions(random_indices, data_split)

    local_output_dir = Path(output_folder)

    dir_names = []
    for i in range(n_learners):
        dir_name = local_output_dir / str(i)
        os.makedirs(str(dir_name), exist_ok=True)

        learner_data = data[split_indices[i]]
        learner_labels = labels[split_indices[i]]

        pickle.dump(learner_data, open(dir_name / DATA_FL, "wb"))
        pickle.dump(learner_labels, open(dir_name / LABEL_FL, "wb"))

        dir_names.append(dir_name)

    print(dir_names)
    return [str(x) for x in dir_names]
