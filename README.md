# Welcome to the Fetch.ai Collective Learning

Colearn is a library that enables privacy-preserving decentralized machine learning tasks on the [FET network](https://fetch.ai/).

This blockchain-mediated collective learning system enables multiple stakeholders to build a shared 
machine learning model without needing to rely on a central authority. 
This library is currently in development. 

The collective learning protocol allows learners to collaborate on training a model without requiring trust between the participants. Learners vote on updates to the model, and only updates which pass the quality threshold are accepted. This makes the system robust to attempts to interfere with the model by providing bad updates. For more details on the collective learning system see [here](https://fetchai.github.io/colearn/about/)

### Current Version

We have released *v0.2.8* of the Colearn Machine Learning Interface, the first version of an interface that will 
allow developers to prepare for future releases. 
Together with the interface we provide a simple backend for local experiments. This is the first backend with upcoming blockchain ledger based backends to follow.  
Future releases will use similar interfaces so that learners built with the current system will work on a different backend that integrates a distributed ledger and provides other improvements.
The current framework will then be used mainly for model development and debugging.
We invite all users to experiment with the framework, develop their own models, and provide feedback!

See the most up-to-date documentation at [fetchai.github.io/colearn](https://fetchai.github.io/colearn/) 
or the documentation for the latest release at [docs.fetch.ai/colearn](https://docs.fetch.ai/colearn/).

## Installation

Currently we only support macos and unix systems.

To use the latest stable release we recommend installing the [package from PyPi](https://pypi.org/project/colearn/)

To install with support for Keras and Pytorch:
   ```bash
   pip install colearn[all]
   ```
To install with just support for Keras or Pytorch:

   ```bash
   pip install colearn[keras]
   pip install colearn[pytorch]
   ```

## Running the examples

Examples are available in the colearn_examples module. To run the Mnist demo in Keras or Pytorch run:
   ```bash
   python -m colearn_examples.ml_interface.keras_mnist
   python -m colearn_examples.ml_interface.pytorch_mnist
   ```
- Or they can be accessed from colearn/colearn_examples by cloning the colearn repo

    Please note that although all the examples are always available, which you can run will depend on your installation. 
    If you installed only `colearn[keras]` or `colearn[pytorch]` then only their respective examples will work. 


For more instructions see the documentation at [fetchai.github.io/colearn/installation](https://fetchai.github.io/colearn/installation/)

After installation we recommend [running a demo](https://fetchai.github.io/colearn/demo/)
, or seeing [the examples](https://fetchai.github.io/colearn/examples/)





