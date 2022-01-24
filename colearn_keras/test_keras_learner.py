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
from unittest.mock import Mock, create_autospec

import pytest
import tensorflow as tf
from tensorflow import keras

from colearn.ml_interface import Weights, DiffPrivConfig
from colearn_keras.keras_learner import KerasLearner


def get_mock_model() -> Mock:
    model = create_autospec(keras.Sequential, instance=True)
    model.evaluate.return_value = {"loss": 1,
                                   "accuracy": 3}
    model.get_weights.return_value = "all the weights"
    model.optimizer = create_autospec(keras.optimizers.Optimizer, instance=True)
    model.optimizer.get_config.return_value = {"name": "Adam"}
    # these are needed for the DP optimizers, but do no harm for the non-DP tests
    model.optimizer._noise_multiplier = 2  # pylint: disable=protected-access
    model.optimizer._l2_norm_clip = 2  # pylint: disable=protected-access
    model.optimizer._num_microbatches = 2  # pylint: disable=protected-access

    model._get_compile_args.return_value = {}  # pylint: disable=protected-access
    return model


def get_mock_dataloader() -> Mock:
    dl = tf.data.Dataset.range(42)
    dl._batch_size = 42  # pylint: disable=protected-access
    return dl


@pytest.fixture
def nkl():
    """Returns a Keraslearner"""
    model = get_mock_model()
    dl = get_mock_dataloader()
    vote_dl = get_mock_dataloader()
    nkl = KerasLearner(model, dl, vote_dl, diff_priv_config=DiffPrivConfig(
        target_epsilon=5,
        target_delta=1e-5,
        max_grad_norm=2,
        noise_multiplier=3
    ))

    return nkl


def test_vote(nkl):
    assert nkl.vote_score == get_mock_model().evaluate.return_value["loss"]

    assert nkl.vote(1.1) is False
    assert nkl.vote(1) is False
    assert nkl.vote(0.9) is True


def test_minimise_criterion(nkl):
    nkl.minimise_criterion = False

    assert nkl.vote(1.1) is True
    assert nkl.vote(1) is False
    assert nkl.vote(0.9) is False


def test_criterion(nkl):
    nkl.criterion = "accuracy"
    nkl.mli_accept_weights(Weights(weights="foo"))
    assert nkl.vote_score == get_mock_model().evaluate.return_value["accuracy"]


def test_propose_weights(nkl):
    weights = nkl.mli_propose_weights()
    assert isinstance(weights, Weights)
    assert weights.weights == get_mock_model().get_weights.return_value


def test_get_current_weights(nkl):
    weights = nkl.mli_get_current_weights()
    assert isinstance(weights, Weights)
    assert weights.weights == get_mock_model().get_weights.return_value


def test_privacy_calculation(nkl):
    epsilon = nkl.get_privacy_budget()
    assert nkl.diff_priv_budget.consumed_epsilon < epsilon


def test_privacy_training(nkl):
    # no training when budget is overconsumed
    nkl.diff_priv_budget.target_epsilon = 0
    w = nkl.mli_propose_weights()
    assert w.training_summary.error_code.name == 'DP_BUDGET_EXCEEDED'

    # do training when budget is not overcompsumed
    nkl.diff_priv_budget.target_epsilon = 9999999
    w = nkl.mli_propose_weights()
    assert w.training_summary.error_code is None


def test_reset_optimizer(nkl):
    # without privacy
    nkl.diff_priv_config = None
    nkl.reset_optimizer()

    # with privacy
    nkl.diff_priv_config = DiffPrivConfig(
        target_epsilon=5,
        target_delta=1e-5,
        max_grad_norm=2,
        noise_multiplier=3
    )
    nkl.reset_optimizer()
