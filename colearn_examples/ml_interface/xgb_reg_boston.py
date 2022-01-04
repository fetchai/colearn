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
from typing import Optional

import numpy as np
import xgboost as xgb
from sklearn import datasets
from sklearn.metrics import mean_squared_error as mse

from colearn.ml_interface import MachineLearningInterface, Weights, ProposedWeights
from colearn.training import initial_result, collective_learning_round
from colearn.utils.data import split_list_into_fractions
from colearn.utils.plot import ColearnPlot
from colearn.utils.results import Results, print_results


class XGBoostLearner(MachineLearningInterface):
    def __init__(self, train_data, train_labels, vote_data, vote_labels, test_data, test_labels,
                 worker_id=0, xgb_params: Optional[dict] = None,
                 n_steps_per_round=2):
        self.worker_id = worker_id

        self.xg_train = xgb.DMatrix(train_data, label=train_labels)
        self.xg_vote = xgb.DMatrix(vote_data, label=vote_labels)
        self.xg_test = xgb.DMatrix(test_data, label=test_labels)

        default_params = {'objective': 'reg:squarederror',
                          'verbose': False,
                          'max_depth': 5,
                          'colsample_bytree': 0.1,
                          'tree_method': 'hist',
                          'learning_rate': 0.3}

        self.params = xgb_params or default_params

        self.n_steps_per_round = n_steps_per_round
        self.model = xgb.train(self.params, self.xg_train, self.n_steps_per_round)

        self.model_file_base = f"/tmp/model_{self.worker_id}"
        self.n_saves = 0
        self.vote_score = self.test(self.xg_vote)

    def mli_propose_weights(self) -> Weights:
        current_model = self.mli_get_current_weights()

        # xgb_model param means training starts from previous weights
        self.model = xgb.train(self.params, self.xg_train, self.n_steps_per_round,
                               xgb_model=current_model.weights)

        new_model = self.mli_get_current_weights()
        self.set_weights(current_model)

        return new_model

    def mli_test_weights(self, weights: Weights) -> ProposedWeights:
        current_weights = self.mli_get_current_weights()
        self.set_weights(weights)

        vote_score = self.test(self.xg_vote)
        test_score = self.test(self.xg_test)

        vote = self.vote_score >= vote_score

        self.set_weights(current_weights)
        return ProposedWeights(weights=weights,
                               vote_score=vote_score,
                               test_score=test_score,
                               vote=vote
                               )

    def mli_accept_weights(self, weights: Weights):
        self.set_weights(weights)
        self.vote_score = self.test(self.xg_vote)

    def mli_get_current_weights(self) -> Weights:
        model_path = self.model_file_base + '_' + str(self.n_saves)
        self.model.save_model(model_path)
        self.n_saves += 1
        return Weights(weights=model_path)

    def set_weights(self, weights: Weights):
        model_path = weights.weights
        self.model.load_model(model_path)

    def test(self, data_matrix):
        return mse(self.model.predict(data_matrix), data_matrix.get_label())


train_fraction = 0.9
vote_fraction = 0.05
n_learners = 5
testing_mode = bool(os.getenv("COLEARN_EXAMPLES_TEST", ""))  # for testing
n_rounds = 20 if not testing_mode else 1
vote_threshold = 0.5

# Load and split the data
boston = datasets.load_boston()
data, labels = boston.data, boston.target
n_datapoints = data.shape[0]
random_indices = np.random.permutation(np.arange(n_datapoints))
learner_indices_list = split_list_into_fractions(random_indices, [0.2, 0.2, 0.2, 0.2, 0.2])

learner_train_data, learner_train_labels, learner_vote_data, learner_vote_labels, learner_test_data, \
    learner_test_labels = [], [], [], [], [], []
for i in range(n_learners):
    learner_indices = learner_indices_list[i]
    n_data = len(learner_indices)
    n_train = int(n_data * train_fraction)
    n_vote = int(n_data * vote_fraction)

    learner_train_ind = learner_indices[0:n_train]
    learner_vote_ind = learner_indices[n_train:n_train + n_vote]
    learner_test_ind = learner_indices[n_train + n_vote:]

    learner_train_data.append(data[learner_train_ind])
    learner_train_labels.append(labels[learner_train_ind])
    learner_vote_data.append(data[learner_vote_ind])
    learner_vote_labels.append(labels[learner_vote_ind])
    learner_test_data.append(data[learner_test_ind])
    learner_test_labels.append(labels[learner_test_ind])

# Make n_learners learners each with a separate part of the dataset
all_learner_models = []
params = {'objective': 'reg:squarederror',
          'verbose': False,
          'max_depth': 5,
          'colsample_bytree': 0.1,
          'tree_method': 'hist',
          'learning_rate': 0.3}

for i in range(n_learners):
    all_learner_models.append(
        XGBoostLearner(
            train_data=learner_train_data[i],
            train_labels=learner_train_labels[i],
            vote_data=learner_vote_data[i],
            vote_labels=learner_vote_labels[i],
            test_data=learner_test_data[i],
            test_labels=learner_test_labels[i],
            worker_id=i,
            xgb_params=params,
        ))

# Do collective learning
results = Results()
results.data.append(initial_result(all_learner_models))  # Get initial score

plot = ColearnPlot(score_name="mean square error")

for round_index in range(n_rounds):
    results.data.append(
        collective_learning_round(all_learner_models,
                                  vote_threshold, round_index)
    )
    print_results(results)

    print_results(results)
    plot.plot_results_and_votes(results)

plot.block()

print("Colearn Example Finished!")
