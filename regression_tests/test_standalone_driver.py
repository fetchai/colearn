from .context import ColearnConfig, TrainingData
from .utils import learner_provider, data_provider
from colearn.standalone_driver import run


def test_run(learner_provider):
    config = ColearnConfig(TrainingData.MNIST, seed=55, n_learners=2)
    config._test_id = "split55"
    data_split = [1/2, 1/2]
    all_learner_models = learner_provider(config, "", data_split)

    status = False
    msg = ""
    try:
        run(1, all_learner_models)
        status = True
    except Exception as e:
        msg = str(e)
    assert status, msg
