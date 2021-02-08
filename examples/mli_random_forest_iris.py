import numpy as np
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

from colearn.ml_interface import MachineLearningInterface, Weights, ProposedWeights
from colearn.training import initial_result, collective_learning_round
from colearn.utils.plot import ColearnPlot
from colearn.utils.results import Results, print_results


class IrisLearner(MachineLearningInterface):
    def __init__(self, train_data, train_labels, test_data, test_labels):
        self.train_data = train_data
        self.train_labels = train_labels
        self.test_data = test_data
        self.test_labels = test_labels

        self.model = RandomForestClassifier(n_estimators=1, warm_start=True)
        self.model.fit(train_data, train_labels)
        self.vote_score = self.test(self.train_data, self.train_labels)

    def mli_propose_weights(self) -> Weights:
        current_weights = self.mli_get_current_weights()

        # increase n_estimators
        params = self.model.get_params()
        print("PARAMETERS", params)
        self.model.set_params(n_estimators=params["n_estimators"] + 2)
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


iris = datasets.load_iris()
data, labels = iris.data, iris.target

train_fraction = 0.9
n_learners = 5

n_rounds = 7

vote_threshold = 0.5

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

all_learner_models = []
for i in range(n_learners):
    all_learner_models.append(
        IrisLearner(
            train_data=learner_train_data[i],
            train_labels=learner_train_labels[i],
            test_data=learner_test_data[i],
            test_labels=learner_test_labels[i]
        ))

results = Results()
# Get initial score
results.data.append(initial_result(all_learner_models))

plot = ColearnPlot(n_learners=n_learners,
                   score_name="accuracy")

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
