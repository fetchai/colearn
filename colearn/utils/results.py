# ------------------------------------------------------------------------------
#
#   Copyright 2021 Fetch.AI Limited
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
# ------------------------------------------------------------------------------
from typing import List
import numpy as np


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

    def process_statistics(self):
        self.h_test_scores = []
        self.h_vote_scores = []

        n_rounds = len(self.data)
        self.mean_test_scores = [np.mean(np.array(self.data[r].test_scores)) for r in range(n_rounds)]
        self.mean_vote_scores = [np.mean(np.array(self.data[r].vote_scores)) for r in range(n_rounds)]

        # gather individual scores
        n_learners = len(self.data[0].vote_scores)
        for i in range(n_learners):
            self.h_test_scores.append([self.data[r].test_scores[i] for r in range(n_rounds)])
            self.h_vote_scores.append([self.data[r].vote_scores[i] for r in range(n_rounds)])


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
