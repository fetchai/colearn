# Tutorial

The most flexible way to use the collective learning backends is to make a class that implements
the collective learning interface. The methods that need to be implemented are propose_weights, evaluate_weights and accept_weights. #fixme - correct names, reference class names, mayeb link to ml_interface.

But the simpler way is to use one of the helper classes that we have provided that implement 
most of the collective learning interface for popular ML libraries. These learners are `SKLearnLearner`, 
`KerasLearner`, and `PytorchLearner`. These learners implement methods to propose, evaluate and accept weights, 
and the user just needs to implement the _get_model function for the derived class.  

In this tutorial we are going to walk through using the PyTorchLearner.  #fixme - summary of steps here

## Defining the model
What we need to do is define a subclass of PytorchLearner that implements _get_model. 
Here we're defining a model for MNIST.

```python
class MNISTPytorchLearner(PytorchLearner):
    def _get_model(self):
        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.conv1 = nn.Conv2d(1, 20, 5, 1)
                self.conv2 = nn.Conv2d(20, 50, 5, 1)
                self.fc1 = nn.Linear(4 * 4 * 50, 500)
                self.fc2 = nn.Linear(500, 10)

            def forward(self, x):
                x = nn_func.relu(self.conv1(x))
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

## Loading the data
The PytorchLearner expects the data to be wrapped by an instance of LearnerData.
The LearnerData class is defined in `/colearn/basic_learner.py` 
and is a simple wrapper around generators for the testing, training and validation data.
The validation data here means the data that is used for voting.
Each generator needs to return the next batch of data when `__next__` is called on it.
#fixme - talk about mnist data format
Here's a function that loads the dataset for a single learner:
```python
def load_learner_data(data_dir, batch_size, width, height, train_ratio, test_ratio, generator_seed):
    data = LearnerData()
    data.train_batch_size = batch_size
    IMAGE_FL = "images.pickle"
    LABEL_FL = "labels.pickle"

    images = pickle.load(open(Path(data_dir) / IMAGE_FL, "rb"))
    labels = pickle.load(open(Path(data_dir) / LABEL_FL, "rb"))

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

Now that we have defined the model and how to load the data we can get started! 
The code below defines some config values and then there is a helper function that downloads MNIST and splits it into a folder for each learner.
```python
# First define some config values:
n_learners = 5
batch_size = 64
width, height, train_ratio, test_ratio, generator_seed = 28, 28, 0.8, 0.2, 42
# fixme: tidy the constants, image_width etc

# This step downloads the MNIST dataset and splits it into folders  # fixme: move to before definition
data_split = [1 / n_learners] * n_learners
learner_data_folders = split_to_folders("", generator_seed, data_split, n_learners)
```

Now we can use the function we defined earlier to make a LearnerData instance for each learner.
```python
learner_datasets = []
for i in range(n_learners):
    learner_datasets.append(
        load_learner_data(learner_data_folders[i], batch_size, width, height, train_ratio, test_ratio, generator_seed)
    )
```

## Configuring the learner
There are lots of configuration values that are required to specify how the learner will train, e.g. the learning rate.
The way that the PytorchLearner expects these to be specified is as part of a config object:

```python
class ModelConfig:
    def __init__(self):
        # Training params
        self.optimizer = torch.optim.Adam
        self.l_rate = 0.001
        self.l_rate_decay = 1e-5
        # self.batch_size = batch_size

        # self.metrics = ["accuracy"]

        # Model params
        self.width = width
        self.height = height
        
        # pytorchlearner
        self.loss = nn_func.nll_loss
        self.n_classes = 10
        self.multi_hot = False

        # Data params
        self.steps_per_epoch = None  # None means use whole dataset
        self.train_ratio = 0.8
        self.val_batches = 2  # number of batches used for voting
        self.test_ratio = 1 - self.train_ratio
        self.class_labels = [str(i) for i in range(self.n_classes)]

        # Differential Privacy params
        self.use_dp = False
```

Now we can create a list of learners. todo: explain clone

```python
config = ModelConfig()
first_learner = MNISTPytorchLearner(config, data=learner_datasets[0])
learners = [first_learner]
for i in range(1, n_learners):
    nth_learner = first_learner.clone(data=learner_datasets[i])
    learners.append(nth_learner)
```

# Running collective learning
The final step is to do collective learning.
The `Results` object is just a helper here to store the results so that they can be plotted. 
The real work is done inside `collective_learning_round`. 
This function takes a list of learners, selects one to train, collects the votes of the other learners and accepts or declines the update.
todo: remove trainingmode, block
```python
# Get initial accuracy
results = Results()
results.data.append(initial_result(learners))

# Now to do collective learning!
n_rounds = 15
vote_threshold = 0.5
for i in range(n_rounds):
    results.data.append(
        collective_learning_round(learners,
                                  vote_threshold, i)
    )

    plot_results(results, n_learners, TrainingMode.COLLECTIVE, block=False)
    plot_votes(results, block=False)

plot_results(results, n_learners, TrainingMode.COLLECTIVE, block=False)
plot_votes(results, block=True)
```