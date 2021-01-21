from typing import List


class Result:
    def __init__(self):
        self.vote = False
        self.votes = []
        self.test_scores = []
        self.vote_scores = []
        self.block_proposer = None


class Results:
    def __init__(self):
        self.data = []  # type: List[Result]

        # Data for plots and statistics
        self.h_test_scores = []
        self.h_vote_scores = []

        self.mean_test_scores = []
        self.mean_vote_scores = []

        self.max_test_scores = []
        self.max_vote_scores = []

        self.highest_test_score = 0
        self.highest_vote_score = 0

        self.highest_mean_test_score = 0
        self.highest_mean_vote_score = 0

        self.current_mean_test_score = 0
        self.current_mean_vote_score = 0

        self.current_max_test_score = 0
        self.current_max_vote_score = 0

        self.mean_mean_test_score = 0
        self.mean_mean_vote_score = 0
