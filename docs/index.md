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

## MNIST Tutorial

This tutorial trains a neural network model to classify handwritten digits in the [MNIST](http://yann.lecun.com/exdb/mnist/) database.
The tutorial will use Tensorflow for the model framework and to obtain the data. We wont go into the details of the model or tensorflow and will focus on the steps needed to run a collective learning task. TODO Needs more references... 
The code for this tutorial is located [in the examples folder](https://github.com/fetchai/colearn/tree/master/examples/mnist). 

### The Learner

The learner we will use is the [MNISTConvLearner](https://github.com/fetchai/colearn/blob/master/examples/mnist/models.py) which contains a single method which just sets up the model:

````
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
````

As can be seen the model inherits from the (KerasLearner)[https://github.com/fetchai/colearn/blob/master/examples/keras_learner.py] which inherits from the [BasicLearner](https://github.com/fetchai/colearn/blob/66f50b446533d0bea67aea3f6bfa1990a0925d14/colearn/model.py) which implements the interface. Each of these intermediate classes are customization points the library provides to simplify the deployment of your own models. For any Keras based model we recommend starting with the KerasLearner. In the Example folder there is also a [SKLearnLearner](https://github.com/fetchai/colearn/blob/master/examples/sklearn_learner.py) for Sk learn based models (TODO add link). 

TODO Maybe add an image?





### 

