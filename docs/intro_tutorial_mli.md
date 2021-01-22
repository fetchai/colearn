# Using collective learning 

This tutorial is a simple guide to trying out the collective learning protocol with your
own machine learning code. Everything runs locally.

The most flexible way to use the collective learning backends is to make a class that implements
the Collective Learning `MachineLearningInterface` defined in 
[ml_interface.py]({{ repo_root }}/colearn/ml_interface.py). 
This tutorial will walk through implementing the `MachineLearningInterface`.
If you're already using keras or pytorch you might find it easier to use the `KerasLearner` or `Pytorchlearner` classes.
See the other tutorials for details of how to do that.

## The MachineLearningInterface
```Python 
{!../colearn/ml_interface.py!} 
```
There are four methods that need to be implemented:

1. `propose_weights` causes the model to do some training and then return a
   new set of weights that are proposed to the other learners. 
   This method shouldn't charge the current weights of the model - that
   only happens when `accept_weights` is called.
2. `test_weights` - the models takes some new weights and returns a vote on whether the new weights are an improvement. 
   As in propose_weights, this shouldn't change the current weights of the model - 
   that only happens when `accept_weights` is called.
3. `accept_weights` - the models accepts some weights that have been voted on and approved by the set of learners. 
    The old weighs of the model are discarded and replaced by the new weights.
4. `current_weights` should return the current weights of the model.

## Implementation for fraud detection task
Here is the class that implements the `MachineLearningInterface` for the task of detecting fraud in bank transactions.
```Python 
class FraudSklearnLearner(MachineLearningInterface):
    def __init__(self, train_data, train_labels, test_data, test_labels,
                 batch_size: int = 10000,
                 steps_per_round: int = 1):
        self.steps_per_round = steps_per_round
        self.batch_size = batch_size
        self.train_data = train_data
        self.train_labels = train_labels
        self.test_data = test_data
        self.test_labels = test_labels

        self.class_labels = np.unique(train_labels)
        self.train_sampler = infinite_batch_sampler(train_data.shape[0], batch_size)

        self.model = SGDClassifier(max_iter=1, verbose=0, loss="modified_huber")
        self.model.partial_fit(self.train_data[0:1], self.train_labels[0:1],
                               classes=self.class_labels)  # this needs to be called before predict
        self.vote_score = self.test(self.train_data, self.train_labels)

    def mli_propose_weights(self) -> Weights:
        current_weights = self.mli_get_current_weights()

        for i in range(self.steps_per_round):
            batch_indices = next(self.train_sampler)
            train_data = self.train_data[batch_indices]
            train_labels = self.train_labels[batch_indices]
            self.model.partial_fit(train_data, train_labels, classes=self.class_labels)

        new_weights = self.mli_get_current_weights()
        self.set_weights(current_weights)
        return new_weights

    def mli_test_weights(self, weights: Weights) -> ProposedWeights:
        current_weights = self.mli_get_current_weights()
        self.set_weights(weights)

        vote_score = self.test(self.train_data, self.train_labels)

        test_score = self.test(self.test_data, self.test_labels)

        vote = self.vote_score <= vote_score

        self.set_weights(current_weights)
        return ProposedWeights(weights=weights,
                               vote_score=vote_score,
                               test_score=test_score,
                               vote=vote
                               )

    def mli_accept_weights(self, weights: Weights):
        self.set_weights(weights)
        self.vote_score = self.test(self.train_data, self.train_labels)

    def mli_get_current_weights(self):
        # return Weights(weights=copy.deepcopy(self.model))
        return Weights(weights=dict(coef_=self.model.coef_,
                                    intercept_=self.model.intercept_))

    def set_weights(self, weights: Weights):
        # self.model = weights.weights
        self.model.coef_ = weights.weights['coef_']
        self.model.intercept_ = weights.weights['intercept_']

    def test(self, data, labels):
        try:
            return self.model.score(data, labels)
        except sklearn.exceptions.NotFittedError:
            return 0
```

Let's step through this and see how it works.
The propose_weights method saves the current weights of the model.
Then it performs some training of the model, and gets the new weights.
It returns the new weights, and resets the model weights to be the old weights.
```Python
    def mli_propose_weights(self) -> Weights:
        current_weights = self.mli_get_current_weights()

        for i in range(self.steps_per_round):
            batch_indices = next(self.train_sampler)
            train_data = self.train_data[batch_indices]
            train_labels = self.train_labels[batch_indices]
            self.model.partial_fit(train_data, train_labels, classes=self.class_labels)

        new_weights = self.mli_get_current_weights()
        self.set_weights(current_weights)
        return new_weights
```

The test_weights method takes as a parameter the proposed weights that it needs to vote on.
It saves the current weights of the model, and then sets the model weights to be the proposed weights.
It tests the model and votes based on whether the score that it is monitoring has improved.
The vote score can be any metric that you like.
You could use loss, accuracy, mean squared error or any custom metric.
If the vote score is the loss then the model would only vote True if the score has decreased.
Here we're using accuracy, so the vote is true if the score increases.
This method then resets the weights to the old values and returns the vote
along with some scores for monitoring purposes.
```Python 
    def mli_test_weights(self, weights: Weights) -> ProposedWeights:
        current_weights = self.mli_get_current_weights()
        self.set_weights(weights)

        vote_score = self.test(self.train_data, self.train_labels)

        test_score = self.test(self.test_data, self.test_labels)

        vote = self.vote_score <= vote_score

        self.set_weights(current_weights)
        return ProposedWeights(weights=weights,
                               vote_score=vote_score,
                               test_score=test_score,
                               vote=vote
                               )
```
The accept_weights method sets the weights of the model to be the new weights.
It also updates the vote score to be the current performance.

!!! Note
    You could implement a cache here. 
    These weights will already have been tested in test_weights, so the vote 
    score could be retrieved from the cache instead of recomputed.

```Python 
    def mli_accept_weights(self, weights: Weights):
        self.set_weights(weights)
        self.vote_score = self.test(self.train_data, self.train_labels)
```

The final method is the simplest - get_current_weights just returns the current weights of the model.
These weights are wrapped inside a `Weights` object.

```Python 
    def mli_get_current_weights(self):
        return Weights(weights=dict(coef_=self.model.coef_,
                                    intercept_=self.model.intercept_))
```

## The rest of the example
The data is loaded and preprocessed and then split into equal parts for each learner.
Then a list of FraudLearner instances is created, each with its own dataset.  

```Python 
    all_learner_models = []
    for i in range(n_learners):
        all_learner_models.append(
            FraudLearner(
                train_data=learner_train_data[i],
                train_labels=learner_train_labels[i],
                test_data=learner_test_data[i],
                test_labels=learner_test_labels[i]
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
