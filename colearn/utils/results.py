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


def print_results(results: Results):
    last_result = results.data[-1]
    print("--------------- LATEST ROUND RESULTS -------------")
    print("Selected proposer:\t", last_result.block_proposer)
    print("New model accepted:\t", last_result.vote)
    print("--------------------------------------------------")
    print("learner id\t\tvote\ttest score\t\tvote score")
    for i in range(len(last_result.votes)):
        print("{id}\t\t\t\t{vote}\t{test_score:.3f}\t\t\t{vote_score:.3f}".format(id=i,
                                                                                  vote=last_result.votes[i],
                                                                                  test_score=last_result.test_scores[i],
                                                                                  vote_score=last_result.vote_scores[
                                                                                      i]))
    print("--------------------------------------------------")
