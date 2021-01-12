# Welcome to the Fetch.ai Collective Learning

Colearn is a library that enables privacy-preserving decentralized machine learning tasks on the FET network.

This blockchain-mediated collective learning system enables multiple stakeholders to build a shared 
machine learning model without needing to rely on a central authority. 
This library is currently in development. 


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

## Documentation
To run the documentation, run 
```
mkdocs serve
```
Documentation will be hosted somewhere soon!


### Current Version

We have released *v.0.1* of the Colearn Machine Learning Interface, the first version of an interface that will allow developers to prepare for future releases. 
Together with the interface we provide a simple backend for local experiments. This is the first backend with upcoming blockchain ledger based backends to follow.  
Future releases will use similar interfaces so that learners built with the current system will work on a different backend that integrates a distributed ledger and provides other improvements.
The current framework will then be used mainly for model development and debugging.
We invite all users to experiment with the framework, develop their own models, and provide feedback!



## Quick Overview
The collective learning protocol allows learners to collaborate on training a model without requiring trust between the participants. Learners vote on updates to the model, and only updates which pass the quality threshold are accepted. This makes the system robust to attempts to interfere with the model by providing bad updates. For more details on the collective learning system see [here](docs/about.md)
