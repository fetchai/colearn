# Examples that use Collective Learning
This is a list of examples that we've implemented to show you how to use Collective Learning

* [mnist_keras]({{ repo_root }}/examples/mnist_keras_example.py). 
  Uses the MNIST dataset and `KerasLearner`. 
  Discussed in more detail [here](./intro_tutorial_keras.md).
* [mnist_pytorch]({{ repo_root }}/examples/new_pytorch_mnist.py). 
  Uses the MNIST dataset and `PytorchLearner`. 
  Discussed in more detail [here](./intro_tutorial_pytorch.md).
* [fraud_keras]({{ repo_root }}/examples/fraud_keras_learner.py). 
  Uses a dataset of bank transactions to predict fraud. 
  Loads data from numpy arrays and uses `KerasLearner`.
* [cifar_keras]({{ repo_root }}/examples/new_keras_cifar.py). 
  Uses the cifar dataset and `KerasLearner`.
* [xray_keras]({{ repo_root }}/examples/xray_keras_learner.py). 
  Loads xray image data and uses `KerasLearner`.
* [fraud_mli]({{ repo_root }}/examples/not_neural_net_learner.py).
  Uses the `MachineLearningInterface` directly and detects fraud in bank transactions.
* [cifar_pytorch]({{ repo_root }}/examples/new_pytorch_cifar.py).
  Uses `PytorchLearner` with CIFAR.
* [covid_pytorch]({{ repo_root }}/examples/new_pytorch_covid_xray.py).
  Uses `PytorchLearner`, detects Covid19 from chest X-rays
* [xray_pytorch]({{ repo_root }}/examples/new_pytorch_xray.py). 
  Loads xray image data and uses `PytorchLearner`.
  