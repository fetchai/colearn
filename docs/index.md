# Welcome to the Fetch.ai Colearn Documentation

### Current Version

We have released *v.0.1* of Colearn Machine Learning Interface, the first version of an interface that will allow developers to prepare for future releases. 
Together with the interface we provide a simple backend for local experiments. This is the first backend with upcoming distributed ledger based backends to follow.  
Future releases will use similar interfaces so that learners built with the current system will work on a different backend.
The current framework will then be used mainly for model development and debugging.
We invite all users to experiment with the framework, develop their own models, and provide feedback!

## Installation
TODO
pip install colearn

## Quick Overview

The main components of yada yada yada
* ml_interface
* standalone_driver
* basic learner

*** The Interface

The ML Interface is the minimal functionality a learner needs to expose to be able to participate in a Collective Learning Task. The backend then interacts with the learner to do a few things:
* Train Model
* Stop Training
* Test Model
* Accept Weights
See https://github.com/fetchai/colearn/blob/master/colearn/ml_interface.py for more details. 

In the following section we show how to train a handwritten digit recognizing machine learning model using the Mnist database. 
In it we also present several useful classes that can be used to develop your own models. 

## Mnist Tutorial

This tutorial trains a neural network model to classify handwritten digits in the [MNIST](http://yann.lecun.com/exdb/mnist/) database.
The tutorial will use Tensorflow for the model framework and to obtain the data. We wont go into the details of the model or tensorflow and will focus on the steps needed to run a collective learning task. TODO Needs more references... 





