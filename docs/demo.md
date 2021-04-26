# How to run the demo

You can try collective learning for yourself using the simple demo in `colearn_examples/ml_interface/run_demo.py`. 
This demo creates n learners for one of five learning tasks and co-ordinates the collective learning between them.

There are five potential datasets for the demo

* KERAS_MNIST is the Tensorflow implementation of standard handwritten digits recognition dataset
* KERAS_CIFAR10 is the Tensorflow implementation of standard image recognition dataset
* PYTORCH_XRAY is Pytorch implementation of a binary classification task that requires predicting pneumonia from images of chest X-rays. 
  The data need to be downloaded from [Kaggle](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)
* PYTORCH_COVID_XRAY is Pytorch implementation of a 3 class classification task that requires predicting no finding, covid or pneumonia from images of chest X-rays. 
  This dataset is not currently publicly available.
* FRAUD The fraud dataset consists of information about credit card transactions, and the task is to predict whether 
  transactions are fraudulent or not. 
  The data need to be downloaded from [Kaggle](https://www.kaggle.com/c/ieee-fraud-detection)

Use the -h flag to see the options:
```bash
python -m colearn_examples.ml_interface.run_demo.py -h
```

Arguments to run the demo:
```
--data_dir:       Directory containing training data, not required for MNIST and CIFAR10
--test_dir:       Optional directory containing test data. A fraction of the training set will be used as a test set when not specified
--task:           Type of task for machine learning: KERAS_MNIST, KERAS_CIFAR10, FRAUD, PYTORCH_XRAY, PYTORCH_COVID_XRAY
--model_type:     Type of machine learning model, default model will be used if not specified
--n_learners:     Number of individual learners
--n_rounds:       Number of training rounds
--vote_threshold: Minimum fraction of positive votes to accept the new model
--train_ratio:    Fraction of training dataset to be used as test-set when no test-set is specified
--seed:           Seed for initialising model and shuffling datasets
--learning_rate:  Learning rate for optimiser
--batch_size:     Size of training batch
```

## Running MNIST
The simplest task to run is MNIST because the data are downloaded automatically from `tensorflow_datasets`.
The command below runs the MNIST task with five learners for 15 rounds.
```bash
python -m colearn_examples.ml_interface.run_demo.py --task KERAS_MNIST --n_learners 5 --n_rounds 15
```
You should see a graph of the vote score and the test score (the score used here is categorical accuracy).
The new model is accepted if the fraction of positive votes (green colour) is higher than 0.5. 
The new model is rejected if the fraction of negative votes (red color) is lower than 0.5. 

![Alt text](images/mnist_plot.png?raw=true "Collective learning graph")

As you can see, there are five learners, and initially they perform poorly.
In round one, learner 0 is selected to propose a new set of weights.

## Other datasets
To run the CIFAR10 dataset:
```bash
python -m colearn_examples.ml_interface.run_demo.py --task KERAS_CIFAR10 --n_learners 5 --n_rounds 15
```
The Fraud and X-ray datasets need to be downloaded from kaggle (this requires a kaggle account).
To run the fraud dataset:
```bash
python -m colearn_examples.ml_interface.run_demo.py --task FRAUD --n_learners 5 --n_rounds 15 --data_dir ./data/fraud
```
To run the X-ray dataset:
```bash
python -m colearn_examples.ml_interface.run_demo.py --task PYTORCH_XRAY --n_learners 5 --n_rounds 15 --data_dir ./data/xray
```

