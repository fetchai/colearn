from .context import Config
from colearn.model import setup_models, BasicLearner
from colearn.ml_interface import MachineLearningInterface
from pathlib import Path
from examples.mnist.models import MNISTSuperminiLearner
from .utils import learner_provider, data_provider
from examples.keras_learner import KerasLearner
from colearn.training import *
from examples.utils.utils import Result, Results
from .pickle_tester import FileTester
import copy
import pickle
import pytest

@pytest.mark.dependency(depends=['test_train_all_models'])
def test_train_random_model(learner_provider, tmpdir):
    config = Config(Path(tmpdir), "MNIST", seed=55, n_learners=2)
    config._test_id = "split55"
    config.random_proposer = False
    all_learner_models = learner_provider(config)

    status = False
    msg = ""
    try:
        block_proposer, weights = train_random_model(all_learner_models, 3, config)
        status = True
    except Exception as e:
        msg = str(e)
    assert status, msg
    res = {
        "weight": weights.data,
        "proposer": block_proposer
    }
    ft = FileTester()
    reference = ft.get_pickle("./tests/data/mnist/ml_rnd_model_1epoch.pickle")
    assert ft.test_object_match(res, reference)



@pytest.mark.dependency
def test_train_all_models(learner_provider, tmpdir):
    config = Config(Path(tmpdir), "MNIST", seed=55, n_learners=2)
    config._test_id = "split55"

    all_learner_models = learner_provider(config)
    result = Result()
    status = False
    msg = ""
    try:
        train_all_models(all_learner_models, result)
        status = True
    except Exception as e:
        msg = str(e)
    assert status, msg
    weights = []
    for l in all_learner_models:
        weights.append(l.get_weights().data)
    res = {
        "weights": weights,
        "vote_accuracies": result.vote_accuracies,
        "test_accuracies": result.test_accuracies
    }
    ft = FileTester()
    reference = ft.get_pickle("./tests/data/mnist/ml_all_model_1epoch.pickle")
    assert ft.test_object_match(res, reference)


@pytest.mark.dependency(depends=['test_train_all_models', 'test_train_random_model'])
def test_validate_proposed_weights(learner_provider, tmpdir):
    config = Config(Path(tmpdir), "MNIST", seed=55, n_learners=2)
    config._test_id = "split55"

    all_learner_models = learner_provider(config)
    result = Result()
    weights = all_learner_models[0].get_weights()
    validate_proposed_weights(all_learner_models, weights, result)
    res = {
        "vote_accuracies": result.vote_accuracies,
        "test_accuracies": result.test_accuracies
    }
    ft = FileTester()
    reference = ft.get_pickle("./tests/data/mnist/ml_validation.pickle")
    assert ft.test_object_match(res, reference)


class MockMLClass:
    def accept_weights(self, weights):
        pass

def test_majority_vote():
    result = Result()

    result.vote_accuracies = [0.3, 0.3, 0.3, 0.3]
    result.test_accuracies = [0.3, 0.3, 0.3, 0.3]
    result.votes = [1, 1, 1, 1]

    learners = [MockMLClass()]*4
    
    majority_vote(result.votes, learners, result, None, 0, 2/3)
    assert result.vote

    result.votes = [1, 0, 0, 0]
    majority_vote(result.votes, learners, result, None, 0, 2/3)
    assert not result.vote


@pytest.mark.dependency(depends=['test_train_all_models', 'test_train_random_model'])
def test_collaborative_training_pass(learner_provider, tmpdir):
    config = Config(Path(tmpdir), "MNIST", seed=55, n_learners=2)
    config._test_id = "split55"
    config.vote_threshold = 0.3
    learners = learner_provider(config)
    result = collaborative_training_pass(learners, config, 2)
    res = {
        "vote_accuracies": result.vote_accuracies,
        "test_accuracies": result.test_accuracies,
        "votes": result.votes,
        "vote": result.vote,
        "block_proposer": result.block_proposer,
        "threshold": config.vote_threshold
    }
    ft = FileTester()
    reference = ft.get_pickle("./tests/data/mnist/ml_collab_pass.pickle")
    assert ft.test_object_match(res, reference)