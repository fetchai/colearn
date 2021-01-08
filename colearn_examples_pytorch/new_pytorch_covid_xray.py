import os
import tempfile
from pathlib import Path
import scipy.io as sio

from torch.utils.data import Dataset
from torchsummary import summary
import torch.utils.data
import numpy as np
import pickle
import shutil

from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import KernelPCA

from colearn_examples_pytorch.new_pytorch_learner import NewPytorchLearner

import torch.nn as nn
import torch.nn.functional as nn_func

from colearn_examples.training import initial_result, collective_learning_round, set_equal_weights
from colearn_examples.utils.plot import plot_results, plot_votes
from colearn_examples.utils.results import Results
from colearn_examples.utils.data import split_by_chunksizes, shuffle_data
from colearn_examples_pytorch.utils import categorical_accuracy

# define some constants
n_learners = 5
batch_size = 8
seed = 42
n_epochs = 15
vote_threshold = 0.5
learning_rate = 0.001
input_width = 64
channels = 1
n_classes = 3

pos_weight = torch.tensor([0.27])
steps_per_epoch = 10
vote_batches = 13  # number of batches used for voting
vote_on_accuracy = True

no_cuda = False
cuda = not no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(64, n_classes)

    def forward(self, x):
        x = self.fc1(x)

        return nn_func.log_softmax(x, dim=1)


# load data
class CovidXrayDataset(Dataset):
    """X-ray dataset."""

    def __init__(self, data_dir, input_width=input_width, n_classes=n_classes, seed=seed, transform=None):
        """
        Args:
            data_dir (string): Path to the data directory.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.input_width = input_width
        self.n_classes = n_classes

        self.images = pickle.load(open(Path(data_dir) / IMAGE_FL, "rb"))
        self.labels = pickle.load(open(Path(data_dir) / LABEL_FL, "rb"))

        [self.images, self.labels] = shuffle_data(
            [self.images, self.labels], seed=seed
        )

        n_cases = int(len(self.images))
        assert (n_cases > 0), "There are no cases"

        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        else:
            idx = [idx]

        batch_size = len(idx)

        # Define two numpy arrays for containing batch data and labels
        batch_data = np.zeros((batch_size, self.input_width), dtype=np.float32)
        batch_labels = np.zeros((batch_size, self.n_classes), dtype=np.float32)

        for j, index in enumerate(idx):
            batch_data[j] = self.images[index]
            batch_labels[j] = self.labels[index]

        sample = (batch_data, batch_labels)
        if self.transform:
            sample = self.transform(sample)

        return sample


IMAGE_FL = "images.pickle"
LABEL_FL = "labels.pickle"


def split_to_folders(data_dir,
                     n_learners,
                     shuffle_seed=None,
                     data_split=None,
                     output_folder=Path(tempfile.gettempdir()) / "covid_xray",
                     test_output_folder=Path(tempfile.gettempdir()) / "covid_xray_test",
                     test_ratio=0.2):
    np.random.seed(shuffle_seed)

    if data_split is None:
        data_split = [1 / n_learners] * n_learners

    # Load data
    covid_features = sio.loadmat(os.path.join(data_dir, 'covid.mat'))
    covid_features = covid_features['covid']

    normal_features = sio.loadmat(os.path.join(data_dir, 'normal.mat'))
    normal_features = normal_features['normal']

    pneumonia_features = sio.loadmat(os.path.join(data_dir, 'pneumonia.mat'))
    pneumonia_features = pneumonia_features['pneumonia']

    if test_ratio > 0:
        print("Global test splitting: ", test_ratio)
        test_size = int(covid_features.shape[0] * test_ratio)

        np.random.shuffle(covid_features)
        covid_test = covid_features[:test_size]
        covid_features = covid_features[test_size:]

        np.random.shuffle(normal_features)
        normal_test = normal_features[:test_size]
        normal_features = normal_features[test_size:]

        np.random.shuffle(pneumonia_features)
        pneumonia_test = pneumonia_features[:test_size]
        pneumonia_features = pneumonia_features[test_size:]

        x_test = np.concatenate((covid_test[:, :-1], normal_test[:, :-1], pneumonia_test[:, :-1]), axis=0)
        y_test = np.concatenate((covid_test[:, -1], normal_test[:, -1], pneumonia_test[:, -1]), axis=0)
        [x_test, y_test] = shuffle_data(
            [x_test, y_test], seed=shuffle_seed
        )

    x = np.concatenate((covid_features[:, :-1], normal_features[:, :-1], pneumonia_features[:, :-1]), axis=0)
    y = np.concatenate((covid_features[:, -1], normal_features[:, -1], pneumonia_features[:, -1]), axis=0)

    min_max_scaler = MinMaxScaler()
    x = min_max_scaler.fit_transform(x)

    transformer = KernelPCA(n_components=64, kernel='linear')
    x = transformer.fit_transform(x)
    print("SHAPE x: ", x.shape)
    print("SHAPE Y: ", y.shape)
    if test_ratio > 0:
        x_test = min_max_scaler.transform(x_test)
        x_test = transformer.transform(x_test)
        print("SHAPE x_test: ", x_test.shape)

    [x, y] = shuffle_data(
        [x, y], seed=shuffle_seed
    )

    [all_images_lists, all_labels_lists] = split_by_chunksizes(
        [x, y], data_split
    )

    local_output_dir = Path(output_folder)
    local_test_output_dir = Path(test_output_folder)
    print("Local output dir: ", local_output_dir)
    print("Local test output dir: ", local_test_output_dir)

    train_dir_names = []
    for i in range(n_learners):
        dir_name = local_output_dir / str(i)
        if os.path.exists(str(dir_name)):
            shutil.rmtree(str(dir_name))
        os.makedirs(str(dir_name), exist_ok=True)
        print("Shapes for learner: ", i)
        print("       input: ", len(all_images_lists[i]), "x", all_images_lists[i][0].shape)
        print("       label (idx=0): ", all_labels_lists[i][0])
        pickle.dump(all_images_lists[i], open(dir_name / IMAGE_FL, "wb"))
        pickle.dump(all_labels_lists[i], open(dir_name / LABEL_FL, "wb"))

        train_dir_names.append(dir_name)

    # Prepare test_dir
    if test_ratio > 0:
        if os.path.exists(str(local_test_output_dir)):
            shutil.rmtree(str(local_test_output_dir))
        os.makedirs(str(local_test_output_dir), exist_ok=True)
        pickle.dump(x_test, open(local_test_output_dir / IMAGE_FL, "wb"))
        pickle.dump(y_test, open(local_test_output_dir / LABEL_FL, "wb"))
        print("Global test set created")

    return [str(x) for x in train_dir_names], [str(local_test_output_dir)] * len(train_dir_names)


# lOAD DATA
full_data_folder = '/home/jiri/fetch/colearn/examples/covid'
train_data_folders, test_data_folders = split_to_folders(
    full_data_folder,
    shuffle_seed=42,
    n_learners=n_learners,
    test_ratio=0.1
)

learner_train_dataloaders = []
learner_test_dataloaders = []

for i in range(n_learners):
    learner_train_dataloaders.append(torch.utils.data.DataLoader(
        CovidXrayDataset(train_data_folders[i]),
        batch_size=batch_size, shuffle=True)
    )
    learner_test_dataloaders.append(torch.utils.data.DataLoader(
        CovidXrayDataset(test_data_folders[i]),
        batch_size=batch_size, shuffle=True)
    )

if vote_on_accuracy:
    learner_vote_kwargs = dict(
        vote_criterion=categorical_accuracy,
        minimise_criterion=False)
    score_name = "categroical accuracy"
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
        num_train_batches=steps_per_epoch,
        num_test_batches=vote_batches,
        **learner_vote_kwargs
    )

    all_learner_models.append(learner)

set_equal_weights(all_learner_models)

# print a summary of the model architecture
summary(all_learner_models[0].model, input_size=(input_width,))

# Now we're ready to start collective learning
# Get initial accuracy
results = Results()
results.data.append(initial_result(all_learner_models))

for epoch in range(n_epochs):
    results.data.append(
        collective_learning_round(all_learner_models,
                                  vote_threshold, epoch)
    )

    plot_results(results, n_learners, score_name=score_name)
    plot_votes(results)

plot_results(results, n_learners, score_name=score_name)
plot_votes(results, block=True)
