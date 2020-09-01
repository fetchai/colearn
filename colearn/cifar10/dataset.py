from colearn.model import KerasLearner
from colearn.plot import display_statistics, plot_results, plot_votes

from .config import load_config
from .data import prepare_single_client, split_to_folders


class Cifar:
    split_to_folders = split_to_folders
    prepare_single_client = prepare_single_client
    load_config = load_config
    display_statistics = display_statistics
    plot_results = plot_results
    plot_votes = plot_votes
    Model = KerasLearner
