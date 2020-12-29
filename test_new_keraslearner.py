from unittest import mock

from colearn.ml_interface import Weights
from new_keraslearner import NewKerasLearner


def get_mock_model() -> mock.Mock:
    model = mock.Mock()
    model.evaluate.return_value = {"loss": -42,
                                   "accuracy": 100}
    model.get_weights.return_value = "all the weights"
    return model


def get_mock_dataloader() -> mock.Mock:
    return mock.Mock()


def test_stuff():
    model = get_mock_model()
    dl = get_mock_dataloader()
    nkl = NewKerasLearner(model, dl)

    assert nkl.vote_score == model.evaluate.return_value["loss"]

    assert nkl.vote(0) is False
    assert nkl.vote(-100) is True

    nkl.minimise_criterion = False

    assert nkl.vote(0) is True
    assert nkl.vote(-100) is False

    nkl.criterion = "accuracy"

    assert nkl.vote(200) is True
    assert nkl.vote(-100) is False

    weights = nkl.mli_propose_weights()
    assert type(weights) == Weights
    assert weights.weights == model.get_weights.return_value
