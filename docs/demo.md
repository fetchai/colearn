# How to run the demo

You can try collective learning for yourself using the simple demo in `./bin/run_demo.py`. 
This demo creates n learners for one of three learning tasks, and co-ordinates the collective learning between them.

There are three potential datasets for the demo

* KERAS_MNIST is the Tensorflow implementation of standard handwritten digits recognition dataset
* PYTORCH_XRAY is Pytorch implementation of a binary classification task that requires predicting pneumonia from images of chest X-rays. 
  The data need to be downloaded from [kaggle](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)
* FRAUD The fraud dataset consists of information about credit card transactions, and the task is to predict whether 
  transactions are fraudulent or not. 
  The data need to be downloaded from [kaggle](https://www.kaggle.com/c/ieee-fraud-detection)

Use the -h flag to see the options:
```bash
examples/run_demo.py -h
```

Arguments to run the demo:
```
--train_dir:      Directory containing train data, not required for MNIST and CIFAR10
--test_dir:       Optional directory containing test data
                  Fraction of training set will be used as test set when not specified
--task:           Type of task for machine learning
--model_type:     Type of machine learning model, default model will be used if not specified
--n_learners:     Number of individual learners
--n_epochs:       Number of training epochs
--vote_threshold: Minimum fraction of positive votes to accept new model
--train_ratio:    Fraction of training dataset to be used as testset when no testset is specified
--seed:           Seed for initialising model and shuffling datasets
--learning_rate:  Learning rate for optimiser
--batch_size:     Size of training batch
```

## Running MNIST
The simplest task to run is MNIST because this doesn't require downloading the data. 
This runs the MNIST task with five learners for 15 epochs.
```bash
examples/run_demo.py --task KERAS_MNIST --n_learners 5 --n_epochs 15
```
You should see a graph of the vote score and the test score (the score used here is area under the curve (AUC)).

![Alt text](images/mnist_plot.png?raw=true "Collective learning graph")

As you can see, there are five learners, and intially they perform poorly.
In round one, learner 0 is selected to propose a new set of weights.

## Other datasets
The Fraud and X-ray datasets need to be downloaded from kaggle (this requires a kaggle account).
To run the fraud dataset:
```bash
examples/run_demo.py --task FRAUD --n_learners 5 --n_epochs 15 --train_dir ./data/fraud
```
To run the X-ray dataset:
```bash
examples/run_demo.py --task PYTORCH_XRAY --n_learners 5 -n_epochs 15 -train_dir ./data/xray
```
