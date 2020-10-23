from .context import ColearnConfig, TrainingData
from colearn.basic_learner import BasicLearner
from .utils import learner_provider, data_provider


def test_model_setup(learner_provider):
    config = ColearnConfig(TrainingData.MNIST, seed=1234, n_learners=4)
    data_split = [0.1, 0.2, 0.3, 0.4]
    config._test_id = "split1234"

    all_learner_models = learner_provider(config, "", data_split)

    assert len(all_learner_models) == 4
    for l in all_learner_models:
        assert isinstance(l, BasicLearner)
