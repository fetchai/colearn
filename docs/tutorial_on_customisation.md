# Tutorial

The most flexible way to use the collective learning backends is to make a class that implements
the Collective Learning `MachineLearningInterface` defined in [ml_interface.py](../colearn/ml_interface.py). 
The methods that need to be implemented are `train_model`, `test_model` and `accept_weights`. 

However, the simpler way is to use one of the helper classes that we have provided that implement 
most of the interface for popular ML libraries. These learners are `SKLearnLearner`, 
`KerasLearner`, and `PytorchLearner`. These learners implement methods to propose, evaluate and accept weights, 
and the user just needs to implement the `_get_model` function for the derived class 
and load the data into a specific format.  

In this tutorial we are going to walk through using the PyTorchLearner.
First we are going to define the model architecture, then 
we are going to load the data and configure the model, and then we will run Collective Learning.

## Package structure
The above mentioned interfaces and basic classes can be accessed after installing Collective Learning as described in the [README](../README.md):

```python
from colearn.ml_interface import MachineLearningInterface, Weights, ProposedWeights
from colearn.basic_learner import BasicLearner, LearnerData
```

The mentioned steps will also install another package called `colearn_examples`, which provides useful
classes to speed up the development, such as the `PytorchLearner`:

```python
from colearn_examples.pytorch_learner import PytorchLearner
```

This package also provides useful data utility, visualization and training helper methods:

```python
from colearn_examples.mnist import split_to_folders
from colearn_examples.mnist.data import train_generator as data_generator

from colearn_examples.training import collective_learning_round, initial_result
from colearn_examples.utils.data import split_by_chunksizes
from colearn_examples.utils.plot import plot_results, plot_votes
```

All the code for this tutorial can be found in [customisation_demo.py](../bin/customisation_demo.py).

## Defining the model
What we need to do is define a subclass of `PytorchLearner` that implements `_get_model`. 
Here we're defining a model for MNIST.

```python
class MNISTPytorchLearner(PytorchLearner):
    def _get_model(self):
        width = self.config.width
        height = self.config.height
        
        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.conv1 = nn.Conv2d(1, 20, 5, 1)
                self.conv2 = nn.Conv2d(20, 50, 5, 1)
                self.fc1 = nn.Linear(4 * 4 * 50, 500)
                self.fc2 = nn.Linear(500, 10)

            def forward(self, x):
                x = nn_func.relu(self.conv1(x.view(-1, 1, height, width)))
                x = nn_func.max_pool2d(x, 2, 2)
                x = nn_func.relu(self.conv2(x))
                x = nn_func.max_pool2d(x, 2, 2)
                x = x.view(-1, 4 * 4 * 50)
                x = nn_func.relu(self.fc1(x))
                x = self.fc2(x)
                return nn_func.log_softmax(x, dim=1)

        model = Net()
        return model
```

This function just needs to return a Pytorch Net - the architecture is up to you!

## Experiment Setup
Before starting we need to define some constants for the experiments. 
Feel free to play around with these and see how it affects the experiment results!

```python
n_learners = 5  # The number of learners in the experiment.
batch_size = 64  # How large a batch to use in training
train_fraction = 0.8  # What fraction of the data of each learner  is used for training and validation
test_fraction = 0.2  # What fraction of the data of each learner is used for testing
seed = 42  # Seed for splitting the data
n_rounds = 15  # How many rounds to run. Each round is... 
vote_threshold = 0.5  # What fraction of learners have to approve the proposed weights
image_width = 28  # width of the mnist images in pixels
image_height = 28  # height of the mnist images in pixels
 ```

## Loading the data
The `PytorchLearner` expects the data to be wrapped by an instance of `LearnerData`.
The `LearnerData` class is defined in [basic_learner.py](../colearn/basic_learner.py) 
and is a simple wrapper around generators for the testing, training and validation data.
The validation data here means the data that is used for voting.
Each generator needs to return numpy arrays of the next batch of data and labels when `__next__` is called on it. 


In this demo the function `split_to_folders` downloads the MNIST dataset from keras
and splits it into a directory for each learner. 
Each directory has two pickle files in it, one of which is for the images, and the other is for the labels.
This function loads the dataset for a single learner:
```python
    
    
def load_learner_data(data_dir, batch_size, width, height, train_ratio, test_ratio, generator_seed):
    data = LearnerData()
    data.train_batch_size = batch_size
    image_fl = "images.pickle"
    label_fl = "labels.pickle"

    images = pickle.load(open(Path(data_dir) / image_fl, "rb"))
    labels = pickle.load(open(Path(data_dir) / label_fl, "rb"))

    [[train_images, test_images], [train_labels, test_labels]] = \
        split_by_chunksizes([images, labels], [train_ratio, test_ratio])

    data.train_data_size = len(train_images)

    data.train_gen = data_generator(
        train_images, train_labels, batch_size,
        width,
        height,
        generator_seed,
        augmentation=False
    )
    data.val_gen = data_generator(
        train_images, train_labels, batch_size,
        width,
        height,
        generator_seed,
        augmentation=False
    )

    data.test_data_size = len(test_images)

    data.test_gen = data_generator(
        test_images,
        test_labels,
        batch_size,
        width,
        height,
        generator_seed,
        augmentation=False
    )

    data.test_batch_size = batch_size
    return data
```
You can store and process the data in any way you like 
as long as it is provided to the learner in a `LearnerData` instance.
For example, you may want to write your own generator that reads batches from disk so that the whole 
dataset doesn't need to be stored in memory.


Now we can use the function we defined earlier to make a `LearnerData` instance for each learner.
```python
learner_datasets = [
    load_learner_data(learner_data_folders[i], batch_size,
                      image_width, image_height,
                      train_fraction, test_fraction, seed)
    for i in range(n_learners)]
```

## Configuring the learner
There are lots of configuration values that are required to specify how the learner will train, e.g. the learning rate.
The way that the `PytorchLearner` expects these to be specified is as part of a config object:

```python
class ModelConfig:
    def __init__(self):
        # Training params
        self.optimizer = torch.optim.Adam
        self.l_rate = 0.001
        self.l_rate_decay = 1e-5
        self.loss = nn_func.nll_loss
        self.n_classes = 10
        self.multi_hot = False

        # Model params
        self.width = image_width
        self.height = image_height
        
        # Data params
        self.steps_per_epoch = None  # None means use whole dataset
        self.train_ratio = train_fraction
        self.val_batches = 2  # number of batches used for voting
        self.test_ratio = test_fraction
        self.batch_size = batch_size

        # Differential Privacy params
        self.use_dp = False
```

Using the config and the datsets created above we can create a list of learners:

```python
config = ModelConfig()
learners = [MNISTPytorchLearner(config, data=learner_datasets[i])
            for i in range(n_learners)]
```

# Running collective learning
The final step is to do collective learning.
The `Results` object is just a helper here to store the results so that they can be plotted. 
The real work is done inside `collective_learning_round`. 
This function takes a list of learners, selects one to train, collects the votes of the other learners and
accepts or declines the update.
```python
# Get initial score
results = Results()
results.data.append(initial_result(learners))

# Now to do collective learning!
for i in range(n_rounds):
    res = collective_learning_round(learners,
                                  vote_threshold, i)
    results.data.append(res)

    plot_results(results, n_learners)
    plot_votes(results)

plot_results(results, n_learners)
plot_votes(results, block=True)
```
You can try more examples of collective learning by using the script [run_demo.py](../bin/run_demo.py).