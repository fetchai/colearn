from typing_extensions import TypedDict
import torch.nn as nn
import torch.nn.functional as nn_func
import torch.utils.data
from opacus import PrivacyEngine
from torchsummary import summary
from torchvision import transforms, datasets

from colearn.training import initial_result, collective_learning_round
from colearn.utils.plot import ColearnPlot
from colearn.utils.results import Results
from colearn_pytorch.pytorch_learner import PytorchLearner

# define some constants
n_learners = 5
batch_size = 64
seed = 42
n_epochs = 10
vote_threshold = 0.5
train_fraction = 0.9
learning_rate = 0.001
height = 28
width = 28
n_classes = 10
vote_batches = 2

# Differential Privacy parameters
sample_size = 3300
alphas = list(range(2, 32))
noise_multiplier = 1.3
max_grad_norm = 1.0

no_cuda = False
cuda = not no_cuda and torch.cuda.is_available()  # boring torch stuff
device = torch.device("cuda" if cuda else "cpu")
DataloaderKwargs = TypedDict('DataloaderKwargs', {'num_workers': int, 'pin_memory': bool}, total=False)
kwargs: DataloaderKwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

# Load the data and split for each learner.
# Using a torch-native dataloader makes this much easier
train_root = '/tmp/mnist'
transform = transforms.Compose([
    transforms.ToTensor()])
data = datasets.MNIST(train_root, transform=transform, download=True)
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
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, n_classes)

    def forward(self, x):
        x = nn_func.relu(self.conv1(x.view(-1, 1, height, width)))
        x = nn_func.max_pool2d(x, 2, 2)
        x = nn_func.relu(self.conv2(x))
        x = nn_func.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = nn_func.relu(self.fc1(x))
        x = self.fc2(x)
        return nn_func.log_softmax(x, dim=1)


# Make n instances of PytorchLearner with model and torch dataloaders
all_learner_models = []
for i in range(n_learners):
    model = Net()
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
    privacy_engine = PrivacyEngine(
        model,
        batch_size=batch_size,
        sample_size=sample_size,
        alphas=alphas,
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm
    )
    privacy_engine.attach(opt)
    learner = PytorchLearner(
        model=model,
        train_loader=learner_train_dataloaders[i],
        test_loader=learner_test_dataloaders[i],
        device=device,
        optimizer=opt,
        criterion=torch.nn.NLLLoss(),
        num_test_batches=vote_batches
    )

    all_learner_models.append(learner)

# print a summary of the model architecture
summary(all_learner_models[0].model, input_size=(width, height))

# Now we're ready to start collective learning
# Get initial accuracy
results = Results()
results.data.append(initial_result(all_learner_models))

plot = ColearnPlot(n_learners=n_learners,
                   score_name="loss")

score_name = "loss"
for epoch in range(n_epochs):
    results.data.append(
        collective_learning_round(all_learner_models,
                                  vote_threshold, epoch)
    )

    plot.plot_results(results)
    plot.plot_votes(results)

plot.plot_results(results)
plot.plot_votes(results, block=True)

print("Colearn Example Finished!")
