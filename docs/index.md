# Welcome to the Fetch.ai Collective Learning Library

Colearn is a library that enables privacy-preserving decentralized machine learning tasks on the FET network.

This blockchain-mediated collective learning system enables multiple stakeholders to build a shared machine learning model without needing to rely on a central authority. This library is currently in development. 

## How collective learning works
A group of *learners* come together, each of whom have their own datasets and want to collaborate on training a machine learning model.
In each round of collective learning:

1.  One learner is selected to train the model and propose a new set of model weights.
2.  The other learners vote on whether the weights are an improvement.
3.  If the majority vote that the new weights are better than the old ones then the new weights are accepted by all the learners. 
    Otherwise the new weights are discarded.
4. The next round begins.
For more information on the Collective Learning Protocol see [here](about.md).


### Current Version

We have released *v.0.1* of Colearn Machine Learning Interface, the first version of an interface that will allow developers to prepare for future releases. 
Together with the interface we provide a simple backend for local experiments. This is the first backend with upcoming blockchain ledger based backends to follow.  
Future releases will use similar interfaces so that learners built with the current system will work on a different backend that integrates a distributed ledger and provides other improvements.
The current framework will then be used mainly for model development and debugging.
We invite all users to experiment with the framework, develop their own models, and provide feedback!

## Getting Started
1. Download the source code from github:
   ```bash
   git clone https://github.com/fetchai/colearn.git && cd colearn
   ```
1. Create and launch a clean virtual environment with Python 3.7. 
   (This library has currently only been tested with Python 3.7).
   ```bash
   pipenv --python 3.7 && pipenv shell
   ```

2. Install the package from source:
    ```bash
    pip install -e .[all]
    ```
   For more installation options see [Installation](./installation.md)
3. Run one of the examples:
    ```bash
    examples/pytorch_mnist.py
    ``` 
    For other examples see [Examples](./examples.md).

## Writing your own models
We encourage users to try out the system by writing their own models. 
Models need to implement the collective learning interface, which provides functions for training and voting on updates.
More instructions can be found in the Getting Started section.

## Running the tests
Tests can be run with:
```
tox
```