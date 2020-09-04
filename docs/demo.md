# How to run the demo
You can try collective learning for yourself using the simple demo in ./bin/simple_demo.py. This demo creates n learners for one of three learning tasks, and co-ordinates the collective learning between them.

There are three potential datasets for the demo
* MNIST is the standard handwritten digits recognition dataset
* XRAY is a binary classification task that requires predicting pneumonia from images of chest X-rays. The data need to be downloaded from here: https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia
* FRAUD The fraud dataset consists of information about credit card transactions and the task is to predict whether transactions are fraudulent or not.https://www.kaggle.com/c/ieee-fraud-detection

Use the -h flag to see the options:
```bash
bin/run_no_networking_demo.py -h
```
The simplest task to run is MNIST because this doesn't require downloading the data. This runs the MNIST task with five learners for 15 epochs.
```bash
bin/run_no_networking_demo.py -t MNIST -n 5 -e 15
```
You should see a graph of the vote score and the test score (the score used here is area under the curve (AUC)).

graph image

As you can see, there are five learners, and intially they perform poorly.
