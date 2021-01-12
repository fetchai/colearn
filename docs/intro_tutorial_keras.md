# Using collective learning with keras

This tutorial is a simple guide to trying out the collective learning protocol with your
own machine learning code. Everything runs locally.

The most flexible way to use the collective learning backends is to make a class that implements
the Collective Learning `MachineLearningInterface` defined in [ml_interface.py]({{ repo_root }}/colearn/ml_interface.py). 
For more details on how to use the `MachineLearningInterface` see [here](./intro_tutorial_mli.md)

However, the simpler way is to use one of the helper classes that we have provided that implement 
most of the interface for popular ML libraries. 
In this tutorial we are going to walk through using the `KerasLearner`.
First we are going to define the model architecture, then 
we are going to load the data and configure the model, and then we will run Collective Learning.

!!! note
    Notey notey note

A standard script for machine learning with Keras looks like the one below
```python


```



```Python hl_lines="6-7"
{!../colearn_examples_keras/mnist_keras_example.py!}
```