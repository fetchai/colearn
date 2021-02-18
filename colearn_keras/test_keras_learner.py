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
from unittest import mock

import pytest

from colearn.ml_interface import Weights
from colearn_keras.keras_learner import KerasLearner


def get_mock_model() -> mock.Mock:
    model = mock.Mock()
    model.evaluate.return_value = {"loss": 1,
                                   "accuracy": 3}
    model.get_weights.return_value = "all the weights"
    return model


def get_mock_dataloader() -> mock.Mock:
    return mock.Mock()


@pytest.fixture
def nkl():
    """Returns a Keraslearner"""
    model = get_mock_model()
    dl = get_mock_dataloader()
    nkl = KerasLearner(model, dl)
    return nkl


def test_vote(nkl):
    assert nkl.vote_score == get_mock_model().evaluate.return_value["loss"]

    assert nkl.vote(1.1) is False
    assert nkl.vote(1) is True
    assert nkl.vote(0.9) is True


def test_minimise_criterion(nkl):
    nkl.minimise_criterion = False

    assert nkl.vote(1.1) is True
    assert nkl.vote(1) is True
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
