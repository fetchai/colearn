# ------------------------------------------------------------------------------
#
#   Copyright 2021 Fetch.AI Limited
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
# ------------------------------------------------------------------------------
from inspect import signature
from typing import Optional

try:
    import tensorflow as tf
except ImportError:
    raise Exception("Tensorflow is not installed. To use the tensorflow/keras "
                    "add-ons please install colearn with `pip install colearn[keras]`.")
from tensorflow import keras

from colearn.ml_interface import MachineLearningInterface, Weights, ProposedWeights


class KerasLearner(MachineLearningInterface):
    """
    Tensorflow Keras learner implementation of machine learning interface
    """

    def __init__(self, model: keras.Model,
                 train_loader: tf.data.Dataset,
                 test_loader: Optional[tf.data.Dataset] = None,
                 minimise_criterion: bool = True,
                 criterion: str = 'loss',
                 model_fit_kwargs: Optional[dict] = None,
                 model_evaluate_kwargs: Optional[dict] = None):
        """
        :param model: Keras model used for training
        :param train_loader: Training dataset
        :param test_loader: Optional test set. Subset of training set will be used if not specified.
        :param minimise_criterion: Boolean - True to minimise value of criterion, False to maximise
        :param criterion: Function to measure model performance
        :param model_fit_kwargs: Arguments to be passed on model.fit function call
        :param model_evaluate_kwargs: Arguments to be passed on model.evaluate function call
        """
        self.model: keras.Model = model
        self.train_loader: tf.data.Dataset = train_loader
        self.test_loader: Optional[tf.data.Dataset] = test_loader
        self.minimise_criterion: bool = minimise_criterion
        self.criterion = criterion
        self.model_fit_kwargs = model_fit_kwargs or {}

        if model_fit_kwargs:
            # check that these are valid kwargs for model fit
            sig = signature(self.model.fit)
            try:
                sig.bind_partial(**self.model_fit_kwargs)
            except TypeError:
                raise Exception("Invalid arguments for model.fit")

        self.model_evaluate_kwargs = model_evaluate_kwargs or {}

        if model_evaluate_kwargs:
            # check that these are valid kwargs for model evaluate
            sig = signature(self.model.evaluate)
            try:
                sig.bind_partial(**self.model_evaluate_kwargs)
            except TypeError:
                raise Exception("Invalid arguments for model.evaluate")

        self.vote_score: float = self.test(self.train_loader)

    def mli_propose_weights(self) -> Weights:
        """
        Trains model on training set and returns new weights after training
        - Current model is reverted to original state after training
        :return: Weights after training
        """
        current_weights = self.mli_get_current_weights()
        self.train()
        new_weights = self.mli_get_current_weights()
        self.set_weights(current_weights)
        return new_weights

    def mli_test_weights(self, weights: Weights) -> ProposedWeights:
        """
        Tests given weights on training and test set and returns weights with score values
        :param weights: Weights to be tested
        :return: ProposedWeights - Weights with vote and test score
        """
        current_weights = self.mli_get_current_weights()
        self.set_weights(weights)

        vote_score = self.test(self.train_loader)

        if self.test_loader:
            test_score = self.test(self.test_loader)
        else:
            test_score = 0
        vote = self.vote(vote_score)

        self.set_weights(current_weights)
        return ProposedWeights(weights=weights,
                               vote_score=vote_score,
                               test_score=test_score,
                               vote=vote
                               )

    def vote(self, new_score) -> bool:
        """
        Compares current model score with proposed model score and returns vote
        :param new_score: Proposed score
        :return: bool positive or negative vote
        """
        if self.minimise_criterion:
            return new_score <= self.vote_score
        else:
            return new_score >= self.vote_score

    def mli_accept_weights(self, weights: Weights):
        """
        Updates the model with the proposed set of weights
        :param weights: The new weights
        """
        self.set_weights(weights)
        self.vote_score = self.test(self.train_loader)

    def mli_get_current_weights(self) -> Weights:
        """
        :return: The current weights of the model
        """
        return Weights(weights=self.model.get_weights())

    def set_weights(self, weights: Weights):
        """
        Rewrites weight of current model
        :param weights: Weights to be stored
        """
        self.model.set_weights(weights.weights)

    def train(self):
        """
        Trains the model on the training dataset
        """
        self.model.fit(self.train_loader, **self.model_fit_kwargs)

    def test(self, loader: tf.data.Dataset) -> float:
        """
        Tests performance of the model on specified dataset
        :param loader: Dataset for testing
        :return: Value of performance metric
        """
        result = self.model.evaluate(x=loader, return_dict=True,
                                     **self.model_evaluate_kwargs)
        return result[self.criterion]
