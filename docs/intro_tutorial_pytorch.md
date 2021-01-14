# Using collective learning with pytorch

This tutorial is a simple guide to trying out the collective learning protocol with your
own machine learning code. Everything runs locally.

The most flexible way to use the collective learning backends is to make a class that implements
the Collective Learning `MachineLearningInterface` defined in [ml_interface.py]({{ repo_root }}/colearn/ml_interface.py). 
For more details on how to use the `MachineLearningInterface` see [here](./intro_tutorial_mli.md)

However, the simpler way is to use one of the helper classes that we have provided that implement 
most of the interface for popular ML libraries. 
In this tutorial we are going to walk through using the `PytorchLearner`.
First we are going to define the model architecture, then 
we are going to load the data and configure the model, and then we will run Collective Learning.

A standard script for machine learning with Pytorch looks like the one below
```Python hl_lines="24 34 58"
{!python_src/mnist_pytorch.py!}
```
There are three steps:

1. Load the data
2. Define the model
3. Train the model

In this tutorial we are going to see how to modify each step to use collective learning. 
We'll end up with code like this:
```Python hl_lines="45 65 109"
{!../colearn_examples_pytorch/new_pytorch_mnist.py!}
```

The first thing is to modify the data loading code.
Each learner needs to have their own training and testing set from the data.
This is easy to do with the pytorch random_split utility:
```Python 
data_split = [len(test_data) // n_learners] * n_learners
learner_test_data = torch.utils.data.random_split(test_data, data_split)
```

The model definition is the same as before.
To use collective learning, we need to create an object that implements the MachineLearningInterface.
To make it easier to use the `MachineLearningInterface` with pytorch, we've defined `PytorchLearner`.
`PytorchLearner` implements standard training and evaluation routines as well as the MachineLearningInterface methods.

```Python 
{!../colearn_pytorch/new_pytorch_learner.py!}
```

We create a set of PytorchLearners by passing in the model and the datasets:
```Python
all_learner_models = []
for i in range(n_learners):
    model = Net()
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
    learner = NewPytorchLearner(
        model=model,
        train_loader=learner_train_dataloaders[i],
        test_loader=learner_test_dataloaders[i],
        device=device,
        optimizer=opt,
        criterion=torch.nn.NLLLoss(),
        num_test_batches=vote_batches,
        vote_criterion=categorical_accuracy,
        minimise_criterion=False
    )

    all_learner_models.append(learner)
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

for epoch in range(n_epochs):
    results.data.append(
        collective_learning_round(all_learner_models,
                                  vote_threshold, epoch)
    )
    
    plot_results(results, n_learners, score_name=score_name)
    plot_votes(results)

# Plot the final result with votes
plot_results(results, n_learners, score_name=score_name)
plot_votes(results, block=True)

```

Simple!
