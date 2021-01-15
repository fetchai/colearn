import torch.nn as nn
import torch.nn.functional as nn_func
import torch.utils.data
from torchsummary import summary
from torchvision import transforms, datasets
from typing_extensions import TypedDict

from colearn_examples.training import initial_result, collective_learning_round, set_equal_weights
from colearn_examples.utils.plot import plot_results, plot_votes
from colearn_examples.utils.results import Results
from examples.utils import categorical_accuracy
from colearn_pytorch.new_pytorch_learner import NewPytorchLearner

"""
CIFAR10 training example using PyTorch

Used dataset:
- CIFAR10 is set of 60 000 colour images of size 32x32x3 in 10 classes

What script does:
- Loads CIFAR10 dataset from torchvision.datasets
- Randomly splits dataset between multiple learners
- Does multiple rounds of learning process and displays plot with results
"""

# define some constants
n_learners = 5
batch_size = 64
seed = 42
n_epochs = 20
vote_threshold = 0.5
train_fraction = 0.9
learning_rate = 0.001
height = 32
width = 32
channels = 3
n_classes = 10
vote_batches = 2
vote_on_accuracy = True  # False means vote on loss

no_cuda = False
cuda = not no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")
DataloaderKwargs = TypedDict('DataloaderKwargs', {'num_workers': int, 'pin_memory': bool}, total=False)
kwargs: DataloaderKwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

# Load the data and split for each learner.
# Using a torch-native dataloader makes this much easier
train_root = '/tmp/cifar10'
transform = transforms.Compose([
    transforms.ToTensor()])
data = datasets.CIFAR10(train_root, transform=transform, download=True)
n_train = int(train_fraction * len(data))
n_test = len(data) - n_train
train_data, test_data = torch.utils.data.random_split(data, [n_train, n_test])

data_split = [len(train_data) // n_learners] * n_learners
learner_train_data = torch.utils.data.random_split(train_data, data_split)
learner_train_dataloaders = [torch.utils.data.DataLoader(
    ds,
    batch_size=batch_size, shuffle=True, **kwargs) for ds in learner_train_data]

data_split = [len(test_data) // n_learners] * n_learners
learner_test_data = torch.utils.data.random_split(test_data, data_split)
learner_test_dataloaders = [torch.utils.data.DataLoader(
    ds,
    batch_size=batch_size, shuffle=True, **kwargs) for ds in learner_test_data]


# define the neural net architecture in Pytorch
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(channels, 32, 5, 1, padding=2)
        self.conv2 = nn.Conv2d(32, 32, 5, 1, padding=2)
        self.conv3 = nn.Conv2d(32, 64, 5, 1, padding=2)
        self.fc1 = nn.Linear(4 * 4 * 64, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = nn_func.relu(self.conv1(x.view(-1, channels, height, width)))
        x = nn_func.max_pool2d(x, 2, 2)
        x = nn_func.relu(self.conv2(x))
        x = nn_func.max_pool2d(x, 2, 2)
        x = nn_func.relu(self.conv3(x))
        x = nn_func.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 64)
        x = nn_func.relu(self.fc1(x))
        x = self.fc2(x)

        return nn_func.log_softmax(x, dim=1)


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
        num_test_batches=vote_batches,
        **learner_vote_kwargs  # type: ignore[arg-type]
    )

    all_learner_models.append(learner)

# Ensure all learners starts with exactly same weights
set_equal_weights(all_learner_models)

# print a summary of the model architecture
summary(all_learner_models[0].model, input_size=(channels, width, height))

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
