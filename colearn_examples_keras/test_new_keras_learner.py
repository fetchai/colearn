from unittest import mock

import pytest

from colearn.ml_interface import Weights
from colearn_keras.new_keras_learner import NewKerasLearner


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
    """Returns a NewKeraslearner"""
    model = get_mock_model()
    dl = get_mock_dataloader()
    nkl = NewKerasLearner(model, dl)
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
