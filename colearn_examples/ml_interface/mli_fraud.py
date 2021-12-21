# ------------------------------------------------------------------------------
#
#   Copyright 2021 Fetch.AI Limited
#
#   Licensed under the Creative Commons Attribution-NonCommercial International
#   License, Version 4.0 (the "License"); you may not use this file except in
#   compliance with the License. You may obtain a copy of the License at
#
#       http://creativecommons.org/licenses/by-nc/4.0/legalcode
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
# ------------------------------------------------------------------------------
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import sklearn
from sklearn.linear_model import SGDClassifier

from colearn.ml_interface import MachineLearningInterface, Weights, ProposedWeights
from colearn.training import initial_result, collective_learning_round, set_equal_weights
from colearn.utils.plot import ColearnPlot
from colearn.utils.results import Results, print_results
from colearn_other.fraud_dataset import fraud_preprocessing

"""
Fraud training example using Sklearn by directly implementing MachineLearningInterface

Used dataset:
- Fraud, download from kaggle: https://www.kaggle.com/c/ieee-fraud-detection

What the script does:
- Implements the Machine Learning Interface
- Randomly splits the dataset between multiple learners
- Does multiple rounds of learning process and displays plot with results
"""


def infinite_batch_sampler(data_size, batch_size):
    while True:
        random_ind = np.random.permutation(np.arange(data_size))
        for i in range(0, data_size, batch_size):
            yield random_ind[i:i + batch_size]


class FraudLearner(MachineLearningInterface):
    def __init__(self, train_data, train_labels, vote_data, vote_labels, test_data, test_labels,
                 batch_size: int = 10000,
                 steps_per_round: int = 1):
        self.steps_per_round = steps_per_round
        self.batch_size = batch_size
        self.train_data = train_data
        self.train_labels = train_labels
        self.vote_data = vote_data
        self.vote_labels = vote_labels
        self.test_data = test_data
        self.test_labels = test_labels

        self.class_labels = np.unique(train_labels)
        self.train_sampler = infinite_batch_sampler(train_data.shape[0], batch_size)

        self.model = SGDClassifier(max_iter=1, verbose=0, loss="modified_huber")
        self.model.partial_fit(self.train_data[0:1], self.train_labels[0:1],
                               classes=self.class_labels)  # this needs to be called before predict
        self.vote_score = self.test(self.vote_data, self.vote_labels)

    def mli_propose_weights(self) -> Weights:
        current_weights = self.mli_get_current_weights()

        for _ in range(self.steps_per_round):
            batch_indices = next(self.train_sampler)
            train_data = self.train_data[batch_indices]
            train_labels = self.train_labels[batch_indices]
            self.model.partial_fit(train_data, train_labels, classes=self.class_labels)

        new_weights = self.mli_get_current_weights()
        self.set_weights(current_weights)
        return new_weights

    def mli_test_weights(self, weights: Weights) -> ProposedWeights:
        current_weights = self.mli_get_current_weights()
        self.set_weights(weights)

        vote_score = self.test(self.vote_data, self.vote_labels)

        test_score = self.test(self.test_data, self.test_labels)

        vote = self.vote_score <= vote_score

        self.set_weights(current_weights)
        return ProposedWeights(weights=weights,
                               vote_score=vote_score,
                               test_score=test_score,
                               vote=vote
                               )

    def mli_accept_weights(self, weights: Weights):
        self.set_weights(weights)
        self.vote_score = self.test(self.vote_data, self.vote_labels)

    def mli_get_current_weights(self):
        return Weights(weights=dict(coef_=self.model.coef_,
                                    intercept_=self.model.intercept_))

    def set_weights(self, weights: Weights):
        self.model.coef_ = weights.weights['coef_']
        self.model.intercept_ = weights.weights['intercept_']

    def test(self, data, labels):
        try:
            return self.model.score(data, labels)
        except sklearn.exceptions.NotFittedError:
            return 0


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", help="Path to data directory", type=str)
    parser.add_argument("--use_cache", help="Use cached preprocessed data", type=bool, default=True)

    args = parser.parse_args()

    if not Path.is_dir(Path(args.data_dir)):
        sys.exit(f"Data path provided: {args.data_dir} is not a valid path or not a directory")

    data_dir = args.data_dir
    train_fraction = 0.9
    vote_fraction = 0.05
    n_learners = 5

    testing_mode = bool(os.getenv("COLEARN_EXAMPLES_TEST", ""))  # for testing
    n_rounds = 7 if not testing_mode else 1

    vote_threshold = 0.5

    data, labels = fraud_preprocessing(data_dir, args.use_cache)

    n_datapoints = data.shape[0]
    random_indices = np.random.permutation(np.arange(n_datapoints))
    n_data_per_learner = n_datapoints // n_learners

    learner_train_data, learner_train_labels, learner_vote_data, learner_vote_labels, learner_test_data, \
        learner_test_labels = [], [], [], [], [], []
    for i in range(n_learners):
        start_ind = i * n_data_per_learner
        stop_ind = (i + 1) * n_data_per_learner
        n_train = int(n_data_per_learner * train_fraction)
        n_vote = int(n_data_per_learner * vote_fraction)

        learner_train_ind = random_indices[start_ind:start_ind + n_train]
        learner_vote_ind = random_indices[start_ind + n_train:start_ind + n_train + n_vote]
        learner_test_ind = random_indices[start_ind + n_train + n_vote:stop_ind]

        learner_train_data.append(data[learner_train_ind])
        learner_train_labels.append(labels[learner_train_ind])
        learner_vote_data.append(data[learner_vote_ind])
        learner_vote_labels.append(labels[learner_vote_ind])
        learner_test_data.append(data[learner_test_ind])
        learner_test_labels.append(labels[learner_test_ind])

    all_learner_models = []
    for i in range(n_learners):
        all_learner_models.append(
            FraudLearner(
                train_data=learner_train_data[i],
                train_labels=learner_train_labels[i],
                vote_data=learner_vote_data[i],
                vote_labels=learner_vote_labels[i],
                test_data=learner_test_data[i],
                test_labels=learner_test_labels[i]
            ))

    set_equal_weights(all_learner_models)

    results = Results()
    # Get initial score
    results.data.append(initial_result(all_learner_models))

    plot = ColearnPlot(score_name="accuracy")

    for round_index in range(n_rounds):
        results.data.append(
            collective_learning_round(all_learner_models,
                                      vote_threshold, round_index)
        )
        print_results(results)

        # then make an updating graph
        plot.plot_results_and_votes(results)

    plot.block()
    print("Colearn Example Finished!")
