# ------------------------------------------------------------------------------
#
#   Copyright 2021 Fetch.AI Limited
#
#   Licensed under the Creative Commons Attribution-NonCommercial International
#   License, Version 4.0 (the "License"); you may not use this file except in
#   compliance with the License. You may obtain a copy of the License at
#
#       http://creativecommons.org/licenses/by-nc/4.0/legalcode
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
from tensorflow_privacy.privacy.analysis.compute_dp_sgd_privacy import compute_dp_sgd_privacy
from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import make_keras_optimizer_class


class KerasLearner(MachineLearningInterface):
    """
    Tensorflow Keras learner implementation of machine learning interface
    """

    def __init__(self, model: keras.Model,
                 train_loader: tf.data.Dataset,
                 test_loader: Optional[tf.data.Dataset] = None,
                 need_reset_optimizer: bool = True,
                 minimise_criterion: bool = True,
                 criterion: str = 'loss',
                 privacy_kwargs: Optional[dict] = None,
                 model_fit_kwargs: Optional[dict] = None,
                 model_evaluate_kwargs: Optional[dict] = None):
        """
        :param model: Keras model used for training
        :param train_loader: Training dataset
        :param test_loader: Optional test set. Subset of training set will be used if not specified.
        :param need_reset_optimizer: True to clear optimizer history before training, False to kepp history.
        :param minimise_criterion: Boolean - True to minimise value of criterion, False to maximise
        :param criterion: Function to measure model performance
        :param privacy_kwargs: dict - Stores the differential privacy budget related constants.
                                      When set to None, no differential privacy applied.
        :param model_fit_kwargs: Arguments to be passed on model.fit function call
        :param model_evaluate_kwargs: Arguments to be passed on model.evaluate function call
        """
        self.model: keras.Model = model
        self.train_loader: tf.data.Dataset = train_loader
        self.test_loader: Optional[tf.data.Dataset] = test_loader
        self.need_reset_optimizer = need_reset_optimizer
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

        self.privacy_kwargs = privacy_kwargs

        if self.privacy_kwargs:
            for k in ['epsilon', 'delta']:
                assert k in self.privacy_kwargs.keys()
            self.epsilon_spent = 0
            self.cumulative_epochs = 0
            if 'epochs' in self.model_fit_kwargs.keys():
                self.epochs_per_fit = self.model_fit_kwargs['epochs']
            else:
                self.epochs_per_fit = signature(self.model.fit).parameters['epochs'].default

        self.model_evaluate_kwargs = model_evaluate_kwargs or {}

        if model_evaluate_kwargs:
            # check that these are valid kwargs for model evaluate
            sig = signature(self.model.evaluate)
            try:
                sig.bind_partial(**self.model_evaluate_kwargs)
            except TypeError:
                raise Exception("Invalid arguments for model.evaluate")

        self.vote_score: float = self.test(self.train_loader)

    def reset_optimizer(self):
        """
        Recompiles the Keras model. This way the optimizer history get erased,
        which is needed before a new training round, otherwise the outdated history is used.
        """
        compile_args = self.model._get_compile_args()  # pylint: disable=protected-access
        opt_config = self.model.optimizer.get_config()

        if self.privacy_kwargs:
            # tensorflow_privacy optimizers get_config() miss the additional parameters
            # was fixed here: https://github.com/tensorflow/privacy/commit/49db04e3561638fc02795edb5774d322cdd1d7d1
            # but it is not yet in the stable version, thus I need here to do the same.
            opt_config.update({
                'l2_norm_clip': self.model.optimizer._l2_norm_clip,  # pylint: disable=protected-access
                'noise_multiplier': self.model.optimizer._noise_multiplier,  # pylint: disable=protected-access
                'num_microbatches': self.model.optimizer._num_microbatches,  # pylint: disable=protected-access
            })
            new_opt = make_keras_optimizer_class(
                getattr(keras.optimizers, opt_config['name'])
            ).from_config(opt_config)
            compile_args['optimizer'] = new_opt
        else:
            compile_args['optimizer'] = getattr(keras.optimizers,
                                                opt_config['name']).from_config(opt_config)

        self.model.compile(**compile_args)

    def mli_propose_weights(self) -> Weights:
        """
        Trains model on training set and returns new weights after training
        - Current model is reverted to original state after training
        :return: Weights after training
        """
        current_weights = self.mli_get_current_weights()
        if self.privacy_kwargs:
            # we calculate epsilon beforehand, to save unnecessary model training
            epsilon = self.privacy_after_training()
            if epsilon < self.privacy_kwargs['epsilon']:
                # if the budget would be overconsumed, reject training
                return current_weights

        self.train()
        new_weights = self.mli_get_current_weights()
        self.set_weights(current_weights)

        if self.privacy_kwargs:
            self.update_spent_privacy(epsilon)
            self.cumulative_epochs += self.epochs_per_fit

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
            return new_score < self.vote_score
        else:
            return new_score > self.vote_score

    def mli_accept_weights(self, weights: Weights):
        """
        Updates the model with the proposed set of weights
        :param weights: The new weights
        """
        self.set_weights(weights)
        self.vote_score = self.test(self.train_loader)

    def privacy_after_training(self) -> float:
        """
        Calculates, what epsilon will apply after another model training.
        """
        batch_size = self.train_loader._batch_size.numpy()  # pylint: disable=protected-access
        iterations_per_epoch = tf.data.experimental.cardinality(self.train_loader).numpy()
        n_samples = batch_size * iterations_per_epoch
        planned_epochs = self.cumulative_epochs + self.epochs_per_fit

        epsilon, _ = compute_dp_sgd_privacy(
            n=n_samples,
            batch_size=batch_size,
            noise_multiplier=self.model.optimizer._noise_multiplier,  # pylint: disable=protected-access
            epochs=planned_epochs,
            delta=self.privacy_kwargs['delta']
        )
        return epsilon

    def update_spent_privacy(self, epsilon: float):
        """
        Keep the privacy budget updated.
        :param epsilon: epsilon to be stored.
        """
        self.epsilon_spent = epsilon

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

        if self.need_reset_optimizer:
            # erase the outdated optimizer memory (momentums mostly)
            self.reset_optimizer()

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
