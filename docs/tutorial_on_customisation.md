# Tutorial

The most flexible way to use the collective learning backends is to make a class that implements
the collective learning interface. The methods that need to be implemented are propose_weights, evaluate_weights and accept_weights. 

But the simpler way is to use one of the classes that we have provided that implement 
standard bits of collective learning for common ML libraries. These learners are SKLearnLearner, 
KerasLearner, and PytorchLearner. These learners implement methods to propose, evaluate and accept weights, 
and the user just needs to implement the _get_model function for the derived class.  

In this tutorial we are going to walk through using the PyTorchLearner.  

Define a subclass of PytorchLearner that implements _get_model. Here we're defining a model for MNIST.

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

This function just needs to return a Pytorch net - the architecture is up to you!

Next we need to define some functions for loading the dataset. The learner needs to be passed an instance of LearnerData
on contruction. The LearnerData class is defined in `/colearn/basic_learner.py` 
and is a simple wrapper around generators for the testing, training and validation data.
Each generator need to return the next batch of data when `__next__` is called on it.
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

# This step downloads the MNIST dataset and splits it into folders
data_split = [1 / n_learners] * n_learners
learner_data_folders = split_to_folders("", generator_seed, data_split, n_learners)
```
