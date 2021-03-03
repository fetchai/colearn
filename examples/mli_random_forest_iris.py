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
    def __init__(self, train_data, train_labels, test_data, test_labels,
                 initial_trees=1, trees_to_add=2):
        self.train_data = train_data
        self.train_labels = train_labels
        self.test_data = test_data
        self.test_labels = test_labels

        self.initial_trees = initial_trees
        self.trees_to_add = trees_to_add

        self.model = RandomForestClassifier(n_estimators=self.initial_trees, warm_start=True)
        self.model.fit(train_data, train_labels)
        self.vote_score = self.test(self.train_data, self.train_labels)

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

        vote_score = self.test(self.train_data, self.train_labels)

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
        self.vote_score = self.test(self.train_data, self.train_labels)

    def mli_get_current_weights(self):
        params = self.model.get_params()
        return Weights(weights=dict(estimators_=self.model.estimators_,
                                    n_classes_=self.model.n_classes_,
                                    n_outputs_=self.model.n_outputs_,
                                    classes_=self.model.classes_,
                                    n_estimators=params["n_estimators"]))

    def set_weights(self, weights: Weights):
        self.model.estimators_ = weights.weights['estimators_']
        self.model.n_classes_ = weights.weights['n_classes_']
        self.model.n_outputs_ = weights.weights['n_outputs_']
        self.model.classes_ = weights.weights['classes_']
        self.model.set_params(n_estimators=weights.weights["n_estimators"])

    def test(self, data_array, labels_array):
        return self.model.score(data_array, labels_array)


train_fraction = 0.9
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

learner_train_data, learner_train_labels, learner_test_data, learner_test_labels = [], [], [], []
for i in range(n_learners):
    start_ind = i * n_data_per_learner
    stop_ind = (i + 1) * n_data_per_learner
    n_train = int(n_data_per_learner * train_fraction)

    learner_train_ind = random_indices[start_ind:start_ind + n_train]
    learner_test_ind = random_indices[start_ind + n_train:stop_ind]

    learner_train_data.append(data[learner_train_ind])
    learner_train_labels.append(labels[learner_train_ind])
    learner_test_data.append(data[learner_test_ind])
    learner_test_labels.append(labels[learner_test_ind])

# Make n_learners learners each with a separate part of the dataset
all_learner_models = []
for i in range(n_learners):
    all_learner_models.append(
        IrisLearner(
            train_data=learner_train_data[i],
            train_labels=learner_train_labels[i],
            test_data=learner_test_data[i],
            test_labels=learner_test_labels[i]
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
    plot.plot_results(results)
    plot.plot_votes(results)

plot.plot_results(results)
plot.plot_votes(results, block=True)

print("Colearn Example Finished!")
