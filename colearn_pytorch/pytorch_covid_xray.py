import tempfile
import os
import torch
from pathlib import Path
from enum import Enum
import pickle

import numpy as np
import scipy.io as sio

import torch.nn as nn
import torch.nn.functional as nn_func
from torch.utils.data import TensorDataset

from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import KernelPCA

from colearn_pytorch.new_pytorch_learner import NewPytorchLearner
from colearn_examples_pytorch.utils import categorical_accuracy
from colearn_examples.utils.data import shuffle_data
from colearn_examples.utils.data import split_by_chunksizes

DATA_FL = "data.pickle"
LABEL_FL = "labels.pickle"


class ModelType(Enum):
    MULTILAYER_PERCEPTRON = 1


def prepare_model(type: ModelType):
    if type == ModelType.MULTILAYER_PERCEPTRON:
        return TorchCovidXrayPerceptronModel()
    else:
        raise Exception("Model %s not part of the ModelType enum" % type)


def prepare_learner(model_type: ModelType, train_loader, test_loader=None, learning_rate=0.001, steps_per_epoch=40,
                    vote_batches=10,
                    no_cuda=False, vote_on_accuracy=True, **kwargs):
    cuda = not no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")

    model = prepare_model(model_type)

    if vote_on_accuracy:
        learner_vote_kwargs = dict(
            vote_criterion=categorical_accuracy,
            minimise_criterion=False)
        score_name = "categorical accuracy"
    else:
        learner_vote_kwargs = {}
        score_name = "loss"

    # Make n instances of NewPytorchLearner with model and torch dataloaders
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
    learner = NewPytorchLearner(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        optimizer=opt,
        criterion=torch.nn.NLLLoss(),
        num_train_batches=steps_per_epoch,
        num_test_batches=vote_batches,
        score_name=score_name,
        **learner_vote_kwargs
    )

    return learner


def prepare_data_loader(data_folder, train=True, train_ratio=0.8, batch_size=8, no_cuda=False, **kwargs):
    data = pickle.load(open(Path(data_folder) / DATA_FL, "rb"))
    labels = pickle.load(open(Path(data_folder) / LABEL_FL, "rb"))

    n_cases = int(train_ratio * len(data))
    assert (n_cases > 0), "There are no cases"
    if train:
        data = data[:n_cases]
        labels = labels[:n_cases]
    else:
        data = data[n_cases:]
        labels = labels[n_cases:]

    cuda = not no_cuda and torch.cuda.is_available()
    loader_kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

    # Create tensor dataset
    data_tensor = torch.FloatTensor(data)
    labels_tensor = torch.LongTensor(labels)
    dataset = TensorDataset(data_tensor, labels_tensor)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size, shuffle=True, **loader_kwargs)

    return loader


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
        data_dir,
        n_learners,
        data_split=None,
        shuffle_seed=None,
        output_folder=None,
        **kwargs
):
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

    [data, labels] = shuffle_data(
        [data, labels], seed=shuffle_seed
    )

    [data, labels] = split_by_chunksizes(
        [data, labels], data_split
    )

    local_output_dir = Path(output_folder)

    dir_names = []
    for i in range(n_learners):
        dir_name = local_output_dir / str(i)
        os.makedirs(str(dir_name), exist_ok=True)

        pickle.dump(data[i], open(dir_name / DATA_FL, "wb"))
        pickle.dump(labels[i], open(dir_name / LABEL_FL, "wb"))

        dir_names.append(dir_name)

    print(dir_names)
    return [str(x) for x in dir_names]
