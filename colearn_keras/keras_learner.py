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

from colearn.ml_interface import MachineLearningInterface, Weights, ProposedWeights, ColearnModel, ModelFormat, convert_model_to_onnx
from colearn.ml_interface import DiffPrivBudget, DiffPrivConfig, TrainingSummary, ErrorCodes
from tensorflow_privacy.privacy.analysis.compute_dp_sgd_privacy import compute_dp_sgd_privacy
from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import make_keras_optimizer_class


class KerasLearner(MachineLearningInterface):
    """
    Tensorflow Keras learner implementation of machine learning interface
    """

    def __init__(self, model: keras.Model,
                 train_loader: tf.data.Dataset,
                 vote_loader: tf.data.Dataset,
                 test_loader: Optional[tf.data.Dataset] = None,
                 need_reset_optimizer: bool = True,
                 minimise_criterion: bool = True,
                 criterion: str = 'loss',
                 model_fit_kwargs: Optional[dict] = None,
                 model_evaluate_kwargs: Optional[dict] = None,
                 diff_priv_config: Optional[DiffPrivConfig] = None):
        """
        :param model: Keras model used for training
        :param train_loader: Training dataset
        :param test_loader: Optional test set. Subset of training set will be used if not specified.
        :param need_reset_optimizer: True to clear optimizer history before training, False to kepp history.
        :param minimise_criterion: Boolean - True to minimise value of criterion, False to maximise
        :param criterion: Function to measure model performance
        :param model_fit_kwargs: Arguments to be passed on model.fit function call
        :param model_evaluate_kwargs: Arguments to be passed on model.evaluate function call
        :param diff_priv_config: Contains differential privacy (dp) budget related configuration
        """
        self.model: keras.Model = model
        self.train_loader: tf.data.Dataset = train_loader
        self.vote_loader: tf.data.Dataset = vote_loader
        self.test_loader: Optional[tf.data.Dataset] = test_loader
        self.need_reset_optimizer = need_reset_optimizer
        self.minimise_criterion: bool = minimise_criterion
        self.criterion = criterion
        self.model_fit_kwargs = model_fit_kwargs or {}
        self.diff_priv_config = diff_priv_config
        self.cumulative_epochs = 0

        if self.diff_priv_config is not None:
            self.diff_priv_budget = DiffPrivBudget(
                target_epsilon=self.diff_priv_config.target_epsilon,
                target_delta=self.diff_priv_config.target_delta,
                consumed_epsilon=0.0,
                # we will always use the highest available delta now
                consumed_delta=self.diff_priv_config.target_delta
            )
            if 'epochs' in self.model_fit_kwargs.keys():
                self.epochs_per_proposal = self.model_fit_kwargs['epochs']
            else:
                self.epochs_per_proposal = signature(self.model.fit).parameters['epochs'].default

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

        self.vote_score: float = self.test(self.vote_loader)

    def reset_optimizer(self):
        """
        Recompiles the Keras model. This way the optimizer history get erased,
        which is needed before a new training round, otherwise the outdated history is used.
        """
        compile_args = self.model._get_compile_args()  # pylint: disable=protected-access
        opt_config = self.model.optimizer.get_config()

        if self.diff_priv_config is not None:
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

        if self.diff_priv_config is not None:
            epsilon_after_training = self.get_privacy_budget()
            if epsilon_after_training > self.diff_priv_budget.target_epsilon:
                return Weights(
                    weights=current_weights,
                    training_summary=TrainingSummary(
                        dp_budget=self.diff_priv_budget,
                        error_code=ErrorCodes.DP_BUDGET_EXCEEDED
                    )
                )

        self.train()
        new_weights = self.mli_get_current_weights()
        self.set_weights(current_weights)

        if self.diff_priv_config is not None:
            self.diff_priv_budget.consumed_epsilon = epsilon_after_training
            self.cumulative_epochs += self.epochs_per_proposal
            new_weights.training_summary = TrainingSummary(dp_budget=self.diff_priv_budget)

        return new_weights

    def mli_test_weights(self, weights: Weights) -> ProposedWeights:
        """
        Tests given weights on training and test set and returns weights with score values
        :param weights: Weights to be tested
        :return: ProposedWeights - Weights with vote and test score
        """
        current_weights = self.mli_get_current_weights()
        self.set_weights(weights)

        vote_score = self.test(self.vote_loader)

        if self.test_loader:
            test_score = self.test(self.test_loader)
        else:
            test_score = 0
        vote = self.vote(vote_score)

        self.set_weights(current_weights)

        return ProposedWeights(weights=weights,
                               vote_score=vote_score,
                               test_score=test_score,
                               vote=vote,
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
        self.vote_score = self.test(self.vote_loader)

    def get_train_batch_size(self) -> int:
        """
        Calculates train batch size.
        """
        if hasattr(self.train_loader, '_batch_size'):
            return self.train_loader._batch_size  # pylint: disable=protected-access
        else:
            return self.train_loader._input_dataset._batch_size  # pylint: disable=protected-access

    def get_privacy_budget(self) -> float:
        """
        Calculates, what epsilon will apply after another model training.
        Need to calculate it in advance to see if another training would result in privacy budget violation.
        """
        batch_size = self.get_train_batch_size()
        iterations_per_epoch = tf.data.experimental.cardinality(self.train_loader).numpy()
        n_samples = batch_size * iterations_per_epoch
        planned_epochs = self.cumulative_epochs + self.epochs_per_proposal

        epsilon, _ = compute_dp_sgd_privacy(
            n=n_samples,
            batch_size=batch_size,
            noise_multiplier=self.diff_priv_config.noise_multiplier,  # type: ignore
            epochs=planned_epochs,
            delta=self.diff_priv_budget.target_delta
        )
        return epsilon

    def mli_get_current_weights(self) -> Weights:
        """
        :return: The current weights of the model
        """
        return Weights(weights=self.model.get_weights())

    def mli_get_current_model(self) -> ColearnModel:
        """
        :return: The current model and its format
        """

        return ColearnModel(
            model_format=ModelFormat(ModelFormat.ONNX),
            model_file="",
            model=convert_model_to_onnx(self.model),
        )

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
