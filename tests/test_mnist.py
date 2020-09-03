from .context import Config
from examples.mnist.data import split_to_folders
from .pickle_tester import FileTester
from pathlib import Path
from .utils import data_provider


def test_split_to_folders(data_provider, tmpdir):
    config = Config(Path(tmpdir), "MNIST", seed=1234, n_learners=4)
    config.data_split = [0.1, 0.2, 0.3, 0.4]
    config._test_id = "split1234"
    dir_names = data_provider(config)
    assert len(dir_names) == 4
    ft = FileTester()
    for d in dir_names:
        ft.test("mnist", "split4", str(d))
