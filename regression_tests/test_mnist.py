from .context import ColearnConfig, TrainingData
from .pickle_tester import FileTester
from .utils import data_provider


def test_split_to_folders(data_provider):
    config = ColearnConfig(TrainingData.MNIST, seed=1234, n_learners=4)
    data_split = [0.1, 0.2, 0.3, 0.4]
    config._test_id = "split1234"
    dir_names = data_provider(config, "", data_split)
    assert len(dir_names) == 4
    ft = FileTester()
    for d in dir_names:
        ft.test("mnist", "split4", str(d))
