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
import os
import pickle

import numpy as np
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

from colearn.ml_interface import MachineLearningInterface, Weights, ProposedWeights
from colearn.training import initial_result, collective_learning_round
from colearn.utils.plot import ColearnPlot
from colearn.utils.results import Results, print_results

"""
An example of using collective learning with a random forest by directly implementing the MachineLearningInterface

Used dataset:
- iris dataset from sklearn

What the script does:
- Implements the Machine Learning Interface
- Randomly splits the dataset between multiple learners
- Does multiple rounds of learning process and displays plot with results
"""


# Define the class that implements the MLI
class IrisLearner(MachineLearningInterface):
    def __init__(self, train_data, train_labels, vote_data, vote_labels, test_data, test_labels,
                 initial_trees=1, trees_to_add=2, max_depth=3):
        self.train_data = train_data
        self.train_labels = train_labels
        self.vote_data = vote_data
        self.vote_labels = vote_labels
        self.test_data = test_data
        self.test_labels = test_labels

        self.initial_trees = initial_trees
        self.trees_to_add = trees_to_add

        self.model = RandomForestClassifier(n_estimators=self.initial_trees, warm_start=True, max_depth=max_depth)
        self.model.fit(train_data, train_labels)
        self.vote_score = self.test(self.vote_data, self.vote_labels)

    def mli_propose_weights(self) -> Weights:
        current_weights = self.mli_get_current_weights()

        # increase n_estimators
        params = self.model.get_params()
        params["n_estimators"] = params["n_estimators"] + self.trees_to_add
        self.model.set_params(**params)

        # Fit model
        self.model.fit(self.train_data, self.train_labels)

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
        return Weights(weights=pickle.dumps(self.model))

    def set_weights(self, weights: Weights):
        self.model = pickle.loads(weights.weights)

    def test(self, data_array, labels_array):
        return self.model.score(data_array, labels_array)


train_fraction = 0.9
vote_fraction = 0.05
n_learners = 5
testing_mode = bool(os.getenv("COLEARN_EXAMPLES_TEST", ""))  # for testing
n_rounds = 20 if not testing_mode else 1
vote_threshold = 0.5

# Load and split the data
iris = datasets.load_iris()
data, labels = iris.data, iris.target
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

# Make n_learners learners each with a separate part of the dataset
all_learner_models = []
for i in range(n_learners):
    all_learner_models.append(
        IrisLearner(
            train_data=learner_train_data[i],
            train_labels=learner_train_labels[i],
            vote_data=learner_vote_data[i],
            vote_labels=learner_vote_labels[i],
            test_data=learner_test_data[i],
            test_labels=learner_test_labels[i],
            trees_to_add=1,
            max_depth=2,
        ))

# Do collective learning
results = Results()
results.data.append(initial_result(all_learner_models))  # Get initial score

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
