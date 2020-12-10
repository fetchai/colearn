# Welcome to the Fetch.ai Collective Learning

Colearn is a library that enables privacy preserving decentralized machine learning tasks on the FET network.

This blockchain-mediated collective learning system enables multiple stakeholders to build a shared machine learning model without needing to rely on a central authority. This library is currently in development. 

A Colearn experiment begins when a group of entities, a group of *learners*, decide on a model architecture and begin learning. Together they will train a single global model. The goal is to train a model that performs better than any of the learners can produce by training on their private data set. 

### How Training Works

Training occurs in rounds; during each round the learners attempt to improve the performance of the global shared model. 
To do so each round an **update** of the global model (for example new set of weights in a neural network) is proposed. 
The learners then **validate** the update and decide if the new model is better than the current global model.  
If enough learners *approve* the update then global model is updated. After an update is approved or rejected a new round begins. 

The detailed steps of a round updating a global model *M* are as follows:
1. One of the learners is selected and proposes a new updated model *M'*
2. The rest of the learners **validate** *M'*
   - If *M'* has better performance than *M* then the learner votes to approve
   - If not the learner votes to reject
3. The total votes are tallied
   - If more than some threshold (typically 50%) of learners approve then *M'* becomes the new global model. If not, *M* continues to be global model
4. A new round begins. 

By using a decentralized ledger (a blockchain) this learning process can be run in a completely decentralized, secure and auditable way. Further security can be provided by using [differential privacy](https://en.wikipedia.org/wiki/Differential_privacy) when generating an update.

### Current Version

We have released *v.0.1* of Colearn Machine Learning Interface, the first version of an interface that will allow developers to prepare for future releases. 
Together with the interface we provide a simple backend for local experiments. This is the first backend with upcoming blockchain ledger based backends to follow.  
Future releases will use similar interfaces so that learners built with the current system will work on a different backend that integrates a distributed ledger and provides other improvements.
The current framework will then be used mainly for model development and debugging.
We invite all users to experiment with the framework, develop their own models, and provide feedback!

## Installation
Setup an environment

`pipenv --python 3.6 && pipenv shell`

```bash
pip install -e ./
```
Running the tests:
```
tox
```

## Quick Overview
The collective learning protocol allows learners to collaborate on training a model without requiring trust between the participants. Learners vote on updates to the model, and only updates which pass the quality threshold are accepted. This makes the system robust to attempts to interfere with the model by providing bad updates. For more details on the collective learning system see [here](docs/about.md)

## Running the demo
```bash
bin/run_no_networking_demo.py -t MNIST
``` 
For other demo options see [here](docs/demo.md)

## Writing your own models
We encourage users to try out the system by writing their own models. Models need to implement the collective learning interface, which provides functions for training and voting on updates. More instructions can be found [here](docs/customisation.md)
