# Examples that use Collective Learning
This is a list of examples that we've implemented to show you how to use Collective Learning

### Mnist
  Uses the standard [Mnist](https://en.wikipedia.org/wiki/MNIST_database) database of handwritten images
  
* [mnist_keras]({{ repo_root }}/examples/keras_mnist.py). 
  Uses the `KerasLearner` helper class. 
  Discussed in more detail [here](./intro_tutorial_keras.md).
* [mnist_pytorch]({{ repo_root }}/examples/pytorch_mnist.py). 
  Uses the `PytorchLearner` helper class. 
  Discussed in more detail [here](./intro_tutorial_pytorch.md).
### Fraud
  The fraud dataset consists of information about credit card transactions. 
  The task is to predict whether transactions are fraudulent or not.
  The data needs to be downloaded from [Kaggle](https://www.kaggle.com/c/ieee-fraud-detection), 
  and the data directory passed in with the flag `--data_dir`. 

* [fraud_mli]({{ repo_root }}/examples/mli_fraud.py).
  Uses the `MachineLearningInterface` directly and detects fraud in bank transactions.
* [fraud_keras]({{ repo_root }}/examples/keras_fraud.py). 
  Loads data from numpy arrays and uses `KerasLearner`.
### Cifar10
  Uses the standard [Cifar10](https://en.wikipedia.org/wiki/CIFAR-10) database of images

* [cifar_keras]({{ repo_root }}/examples/keras_cifar.py). 
  Uses the `KerasLearner` helper class.
* [cifar_pytorch]({{ repo_root }}/examples/pytorch_cifar.py).
  Uses the `PytorchLearner` helper class.
### Xray
  A binary classification task that requires predicting pneumonia from images of chest X-rays. 
  The data need to be downloaded from [Kaggle](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia), 
  and the data directory passed in with the flag `--data_dir`

* [xray_keras]({{ repo_root }}/examples/keras_xray.py). 
  Uses the `KerasLearner` helper class.
* [xray_pytorch]({{ repo_root }}/examples/pytorch_xray.py). 
  Uses the `PytorchLearner` helper class.
