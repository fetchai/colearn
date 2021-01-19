from inspect import signature
from typing import Optional

try:
    import tensorflow as tf
except ImportError:
    raise Exception("Tensorflow is not installed. To use the tensorflow/keras "
                    "add-ons please install colearn with `pip install colearn[keras]`.")
from tensorflow import keras

from colearn.ml_interface import MachineLearningInterface, Weights, ProposedWeights


class NewKerasLearner(MachineLearningInterface):
    def __init__(self, model: keras.Model,
                 train_loader: tf.data.Dataset,
                 test_loader: Optional[tf.data.Dataset] = None,
                 minimise_criterion: bool = True,
                 criterion: str = 'loss',
                 model_fit_kwargs: Optional[dict] = None,
                 model_evaluate_kwargs: Optional[dict] = None):

        self.model: keras.Model = model
        self.train_loader: tf.data.Dataset = train_loader
        self.test_loader: Optional[tf.data.Dataset] = test_loader
        self.minimise_criterion: bool = minimise_criterion
        self.criterion = criterion
        self.model_fit_kwargs = model_fit_kwargs or {}
        self.score_name = criterion

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
        current_weights = self.mli_get_current_weights()
        self.train()
        new_weights = self.mli_get_current_weights()
        self.set_weights(current_weights)
        return new_weights

    def mli_test_weights(self, weights: Weights) -> ProposedWeights:
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
        if self.minimise_criterion:
            return new_score <= self.vote_score
        else:
            return new_score >= self.vote_score

    def mli_accept_weights(self, weights: Weights):
        self.set_weights(weights)
        self.vote_score = self.test(self.train_loader)

    def mli_get_current_weights(self) -> Weights:
        return Weights(weights=self.model.get_weights())

    def set_weights(self, weights: Weights):
        self.model.set_weights(weights.weights)

    def train(self):
        self.model.fit(self.train_loader, **self.model_fit_kwargs)

    def test(self, loader: tf.data.Dataset) -> float:
        result = self.model.evaluate(x=loader, return_dict=True,
                                     **self.model_evaluate_kwargs)
        return result[self.criterion]
