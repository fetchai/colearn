from .data import split_to_folders, prepare_single_client
from examples.xray_utils.plot import display_statistics, plot_results
from examples.utils.plot import plot_votes
from keras_learner import KerasLearner


class Xray:
    split_to_folders = split_to_folders
    prepare_single_client = prepare_single_client
    display_statistics = display_statistics
    plot_results = plot_results
    plot_votes = plot_votes
    Model = KerasLearner
