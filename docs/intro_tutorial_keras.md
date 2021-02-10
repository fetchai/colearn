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

A standard script for machine learning with Keras looks like the one below
```Python hl_lines="11 31 49"
{!python_src/mnist_keras.py!}
```
There are three steps:

1. Load the data
2. Define the model
3. Train the model

In this tutorial we are going to see how to modify each step to use collective learning. 
We'll end up with code like this:
```Python hl_lines="20 45 87"
{!../examples/keras_mnist.py!}
```

The first thing is to modify the data loading code.
Each learner needs to have their own training and testing set from the data.
This is easy to do with keras:
```Python 
train_datasets = [train_dataset.shard(num_shards=n_learners, index=i) for i in range(n_learners)]
```

The model definition is very similar too, except that each learner will need its own copy of the model,
so we've moved it into a function.

To use collective learning, we need to create an object that implements the MachineLearningInterface.
To make it easier to use the `MachineLearningInterface` with keras, we've defined `KerasLearner`.
`KerasLearner` implements standard training and evaluation routines as well as the MachineLearningInterface methods.

```Python 
{!../colearn_keras/keras_learner.py!}
```

We create a set of KerasLearners by passing in the model and the datasets:
```Python
all_learner_models = []
for i in range(n_learners):
    all_learner_models.append(KerasLearner(
        model=get_model(),
        train_loader=train_datasets[i],
        test_loader=test_datasets[i],
        criterion="sparse_categorical_accuracy",
        minimise_criterion=False,
        model_evaluate_kwargs={"steps": vote_batches},
    ))
```

Then we give all the models the same weights to start off with:
```Python
set_equal_weights(all_learner_models)
```

And then we can move on to the final stage, which is training with Collective Learning.
The function `collective_learning_round` performs one round of collective learning.
One learner is selected to train and propose an update.
The other learners vote on the update, and if the vote passes then the update is accepted.
Then a new round begins.
```Python 
# Train the model using Collective Learning
results = Results()
results.data.append(initial_result(all_learner_models))

for round in range(n_rounds):
    results.data.append(
        collective_learning_round(all_learner_models,
                                  vote_threshold, round)
    )

    plot_results(results, n_learners, block=False,
                 score_name=all_learner_models[0].criterion)
    plot_votes(results, block=False)

plot_results(results, n_learners, block=False,
             score_name=all_learner_models[0].criterion)
plot_votes(results, block=True)
```
