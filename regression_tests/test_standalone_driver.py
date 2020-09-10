from .context import ColearnConfig
from colearn.basic_learner import BasicLearner
from ml_interface import MachineLearningInterface
from pathlib import Path
from colearn_examples.mnist.models import MNISTSuperminiLearner
from .utils import learner_provider, data_provider
from colearn_examples.keras_learner import KerasLearner
from standalone_driver import run

def test_run(learner_provider, tmpdir):
    config = ColearnConfig(Path(tmpdir), "MNIST", seed=55, n_learners=2)
    config._test_id = "split55"

    all_learner_models = learner_provider(config)   

    status = False
    msg = ""
    try:
        run(1, all_learner_models)
        status = True
    except Exception as e:
        msg = str(e)
    assert status, msg
