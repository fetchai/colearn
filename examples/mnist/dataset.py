from examples.mnist import split_to_folders, prepare_single_client
from examples.mnist.config import load_config
from colearn.plot import display_statistics, plot_results, plot_votes
from colearn.model import KerasLearner


class Mnist:
    split_to_folders = split_to_folders
    prepare_single_client = prepare_single_client
    load_config = load_config
    display_statistics = display_statistics
    plot_results = plot_results
    plot_votes = plot_votes
    Model = KerasLearner
