from .context import ColearnConfig, TrainingData
from .utils import learner_provider, data_provider
from colearn_examples.training import collaborative_training_pass, individual_training_pass
from colearn_examples.utils.results import Result
from .pickle_tester import FileTester
import pytest


@pytest.mark.dependency
def test_individual_training_pass(learner_provider):
    config = ColearnConfig(TrainingData.MNIST, seed=55, n_learners=2)
    config._test_id = "split55"

    all_learner_models = learner_provider(config, "", [0.5, 0.5])
    result = Result()
    status = False
    msg = ""
    try:
        result = individual_training_pass(all_learner_models)
        status = True
    except Exception as e:
        msg = str(e)
    assert status, msg
    weights = []
    for l in all_learner_models:
        weights.append(l.get_weights().weights)
    res = {
        "weights": weights,
        "vote_accuracies": result.vote_accuracies,
        "test_accuracies": result.test_accuracies
    }
    ft = FileTester()
    reference = ft.get_pickle("./regression_tests/data/mnist/ml_all_model_1epoch.pickle")
    assert ft.test_object_match(res, reference)


@pytest.mark.dependency(depends=['test_individual_training_pass'])
def test_collaborative_training_pass(learner_provider):
    config = ColearnConfig(TrainingData.MNIST, seed=55, n_learners=2)
    config._test_id = "split55"
    config.vote_threshold = 0.3
    learners = learner_provider(config, "", [0.5, 0.5])
    result = collaborative_training_pass(learners, 0.3, 1)
    res = {
        "vote_accuracies": result.vote_accuracies,
        "test_accuracies": result.test_accuracies,
        "votes": result.votes,
        "vote": result.vote,
        "block_proposer": result.block_proposer,
        "threshold": config.vote_threshold
    }
    ft = FileTester()
    reference = ft.get_pickle("./regression_tests/data/mnist/ml_collab_pass.pickle")
    assert ft.test_object_match(res, reference)
