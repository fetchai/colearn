from typing import List


class Result:
    def __init__(self):
        self.vote = False
        self.votes = []
        self.test_accuracies = []
        self.vote_accuracies = []
        self.block_proposer = None


class Results:
    def __init__(self):
        self.data = []  # type: List[Result]

        # Data for plots and statistics
        self.h_test_accuracies = []
        self.h_vote_accuracies = []

        self.mean_test_accuracies = []
        self.mean_vote_accuracies = []

        self.max_test_accuracies = []
        self.max_vote_accuracies = []

        self.highest_test_accuracy = 0
        self.highest_vote_accuracy = 0

        self.highest_mean_test_accuracy = 0
        self.highest_mean_vote_accuracy = 0

        self.current_mean_test_accuracy = 0
        self.current_mean_vote_accuracy = 0

        self.current_max_test_accuracy = 0
        self.current_max_vote_accuracy = 0

        self.mean_mean_test_accuracy = 0
        self.mean_mean_vote_accuracy = 0
