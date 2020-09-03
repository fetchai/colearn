# Welcome to the Fetch.ai Colearn Documentation

Colearn is a library that enables privacy preserving decentralized machine learning tasks on the FET network.

This blockchain-mediated collective learning system enables multiple stakeholders to build a shared machine learning model without needing to rely on a central authority. This library is currently in development. 

### Current Version

We have released *v.0.1* of Colearn Machine Learning Interface, the first version of an interface that will allow developers to prepare for future releases. 
Together with the interface we provide a simple backend for local experiments. This is the first backend with upcoming blockchain ledger based backends to follow.  
Future releases will use similar interfaces so that learners built with the current system will work on a different backend.
The current framework will then be used mainly for model development and debugging.
We invite all users to experiment with the framework, develop their own models, and provide feedback!

## Installation
TODO
pip install colearn

## Quick Overview

We will give a short description of the minimum interface a learner has to implement and then give a quick overview of the general architecture. 

*** The Interface

The ML Interface is the minimal functionality a learner needs to expose to be able to participate in a Collective Learning Task. The backend then interacts with the learner to do a few things:
* Train Model
* Stop Training
* Test Model
* Accept Weights

See [the code](https://github.com/fetchai/colearn/blob/master/colearn/ml_interface.py) for more details. 

*** Overview

We identify four components of a learning task:
* **The Model**: Which defines the architecture and current weights(state?) of the model
* **The Learners**: Holds a private data set and implements the Machine Learning Interface. It can train its local model on its private data and evaluate and vote on other weights proposed by other learners
* **The Backend**: Runs the learning task. In the current version this simply a standalone driver that runs a simple training loop. In later versions the backend will be a fully decentralized distributed ledger process. This will also enable the discovery of afine models and more complex learning and consensus strategies, for a more complete vision see [here](https://medium.com/fetch-ai/democratising-machine-learning-with-blockchain-technology-10b56ceda41e).
* **The Training Loop**: The backend runs the training loop which consists of three main steps. 
  1. The backend selects a learner and the learner proposes a new sets of weights for the model.
  2. Every learner votes on the proposed weights.
  3. If the vote passes then the model is adopted by all the learners if not its rejected.  Then the loop restarts.  


In the following section we show how to train a handwritten digit recognizing machine learning model using the Mnist database. 
In it we also present several useful classes that can be used to develop your own models. 

## MNIST Example

This tutorial trains a neural network model to classify handwritten digits in the [MNIST](http://yann.lecun.com/exdb/mnist/) database.
The tutorial will use Tensorflow for the model framework and to obtain the data. We wont go into the details of the model or tensorflow and will focus on the steps needed to run a collective learning task. TODO Needs more references... 
The code for this tutorial is located [in the examples folder](https://github.com/fetchai/colearn/tree/master/examples/mnist). 

### The Learner

The learner we will use is the [MNISTConvLearner](https://github.com/fetchai/colearn/blob/master/examples/mnist/models.py) which contains a single method which just sets up the model:

```python
class MNISTConvLearner(KerasLearner):
    def _get_model(self):
        input_img = tf.keras.Input(
            shape=(self.config.width, self.config.height, 1), name="Input"
        )
        x = tf.keras.layers.Conv2D(
            64, (3, 3), activation="relu", padding="same", name="Conv1_1"
        )(input_img)
        x = tf.keras.layers.BatchNormalization(name="bn1")(x)
        x = tf.keras.layers.MaxPooling2D((2, 2), name="pool1")(x)
        x = tf.keras.layers.Conv2D(
            128, (3, 3), activation="relu", padding="same", name="Conv2_1"
        )(x)
        x = tf.keras.layers.BatchNormalization(name="bn4")(x)
        x = tf.keras.layers.MaxPooling2D((2, 2), name="pool2")(x)
        x = tf.keras.layers.Flatten(name="flatten")(x)
        x = tf.keras.layers.Dense(
            self.config.n_classes, activation="softmax", name="fc1"
        )(x)
        model = tf.keras.Model(inputs=input_img, outputs=x)

        # compile model & add optimiser
        opt = self.config.optimizer(
            lr=self.config.l_rate, decay=self.config.l_rate_decay
        )

        model.compile(loss=self.config.loss, metrics=["accuracy"], optimizer=opt)
        return model
```

As can be seen the model inherits from the (KerasLearner)[https://github.com/fetchai/colearn/blob/master/examples/keras_learner.py] which inherits from the [BasicLearner](https://github.com/fetchai/colearn/blob/66f50b446533d0bea67aea3f6bfa1990a0925d14/colearn/model.py) which implements the interface. Each of these intermediate classes are customization points the library provides to simplify the deployment of your own models. For any Keras based model we recommend starting with the KerasLearner. In the Example folder there is also a [SKLearnLearner](https://github.com/fetchai/colearn/blob/master/examples/sklearn_learner.py) for Sk learn based models (TODO add link). 

The BasicLearner handles some of the logic required by the interface and hands what is model specific to the subclass. For example BasicLearner implements test_model

```python
def test_model(self, weights=None) -> ProposedWeights:
        """Tests the proposed weights and fills in the rest of the fields"""
        if weights is None:
            weights = self.get_weights()

        proposed_weights = ProposedWeights()
        proposed_weights.weights = weights
        proposed_weights.validation_accuracy = self._test_model(weights,
                                                                validate=True)
        proposed_weights.test_accuracy = self._test_model(weights,
                                                          validate=False)
        proposed_weights.vote = (
            proposed_weights.validation_accuracy >= self.vote_accuracy
        )

        return proposed_weights
```
but get_weights is left unimplemented
```python
def get_weights(self):
    raise NotImplementedError
```
so it is implemented by the KerasLearner
```python
    def get_weights(self):
        return KerasWeights(self._model.get_weights())
```

Bu using and extending the learners provided it is simple to setup your own models. 

### Running the experiment

For experiments using the standalone driver


