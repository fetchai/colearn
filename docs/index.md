# Welcome to the Fetch.ai Collective Learning Library

Colearn is a library that enables privacy-preserving decentralized machine learning tasks on the FET network.

This blockchain-mediated collective learning system enables multiple stakeholders to build a shared machine learning model without needing to rely on a central authority,
and without revealing their dataset to the other stakeholders. This library is currently in development. 

## How collective learning works
A group of *learners* comes together, each of whom have their own datasets and want to collaborate on training a machine learning model over a set number of rounds. We refer
to this as an 'experiment'.
In each round of collective learning:

1.  One learner is selected to train the model and propose a new set of model weights.
2.  The other learners vote on whether the weights are an improvement.
3.  If the majority vote that the new weights are better than the old ones then the new weights are accepted by all the learners. 
    Otherwise the new weights are discarded.
4. The next round begins.
For more information on the Collective Learning Protocol see [here](about.md).


### Current Version

We have released *v.0.2* of the Colearn Machine Learning Interface, the first version of an interface that allows developers to define their own model architectures that can then be used in collective learning. 
Together with the interface we provide a simple backend for local experiments. This is a prototype backend with upcoming blockchain ledger based backends to follow.  
Future releases will use similar interfaces so that learners built with the current system will work on a different backend that integrates a distributed ledger and provides other improvements.
The current framework will then be used mainly for model development and debugging.
We invite all users to experiment with the framework, develop their own models, and provide feedback!

## Getting Started

To use the latest stable release we recommend installing the [package from PyPi](https://pypi.org/project/colearn/)

To install with support for Keras/Pytorch:
   ```bash
   pip install colearn[all]
   ```
To install with just support for Keras/Pytorch:

   ```bash
   pip install colearn[keras]
   pip install colearn[pytorch]
   ```

For more installation options or get the latest (unstable) version see [Installation](./installation.md)

Then run one of the examples:

    ```bash
    python examples/pytorch_mnist.py
    ``` 

For other examples see the [Examples](./examples.md).

## Writing your own models
We encourage users to try out the system by writing their own models. 
Models need to implement the collective learning interface, which provides functions for training and voting on updates.
More instructions can be found in the Getting Started section.

