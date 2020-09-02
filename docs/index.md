# Welcome to the Fetch.ai Colearn Documentation

## Quickstart

Installation 
Implement a Colearn Interface
Make a launcher Run
*v.0.1*

Colearn is a library that enables privacy preserving decentralized machine learning tasks on the FET network.

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

The ML Interface is the minimal functionality a learner needs to expose to be able to participate in a Collective Learning Task. The backend queries and orders the learner to do a few things:
* Train Model
* Stop Training
* Test Model
* Accept Weights

*** Tasks





*** The 

In the following section we show how to leverage the three components to train a Mnist model, where each learner has access to a subset of the Mnist database.

## Mnist Tutorial

Installation

config.py
data.py
models etc...

