from .context import ColearnConfig
from colearn.basic_learner import  BasicLearner
from ml_interface import MachineLearningInterface
from pathlib import Path
from colearn_examples.mnist.models import MNISTSuperminiLearner
from .utils import learner_provider, data_provider
from colearn_examples.keras_learner import KerasLearner


def test_model_setup(learner_provider, tmpdir):
    config = ColearnConfig(Path(tmpdir), "MNIST", seed=1234, n_learners=4)
    config.data_split = [0.1, 0.2, 0.3, 0.4]
    config._test_id = "split1234"

    all_learner_models = learner_provider(config)

    assert len(all_learner_models) == 4
    for l in all_learner_models:
        assert isinstance(l, BasicLearner)
