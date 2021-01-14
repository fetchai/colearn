from pathlib import Path
from typing import Optional
import numpy as np
import sklearn

from sklearn.linear_model import SGDClassifier

from colearn.ml_interface import MachineLearningInterface, Weights, ProposedWeights
from colearn_examples.training import initial_result, collective_learning_round, set_equal_weights
from colearn_examples.utils.plot import plot_results, plot_votes
from colearn_examples.utils.results import Results


def infinite_batch_sampler(data_size, batch_size):
    while True:
        random_ind = np.random.permutation(np.arange(data_size))
        for i in range(0, data_size, batch_size):
            yield random_ind[i:i + batch_size]


class FraudSklearnLearner(MachineLearningInterface):
    def __init__(self, train_data, train_labels, test_data, test_labels,
                 batch_size: int = 10000,
                 steps_per_round: int = 1):
        self.steps_per_round = steps_per_round
        self.batch_size = batch_size
        self.train_data = train_data
        self.train_labels = train_labels
        self.test_data = test_data
        self.test_labels = test_labels

        self.class_labels = np.unique(train_labels)
        self.train_sampler = infinite_batch_sampler(train_data.shape[0], batch_size)

        self.model = SGDClassifier(max_iter=1, verbose=0, loss="modified_huber")
        self.model.partial_fit(self.train_data[0:1], self.train_labels[0:1],
                               classes=self.class_labels)  # this needs to be called befoe predict
        self.vote_score = self.test(self.train_data, self.train_labels)

    def mli_propose_weights(self) -> Weights:
        current_weights = self.mli_get_current_weights()
        self.train()
        new_weights = self.mli_get_current_weights()
        self.set_weights(current_weights)
        return new_weights

    def mli_test_weights(self, weights: Weights, eval_config: Optional[dict] = None) -> ProposedWeights:
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
        # return Weights(weights=copy.deepcopy(self.model))
        return Weights(weights=dict(coef_=self.model.coef_,
                                    intercept_=self.model.intercept_))

    def set_weights(self, weights: Weights):
        # self.model = weights.weights
        self.model.coef_ = weights.weights['coef_']
        self.model.intercept_ = weights.weights['intercept_']

    def test(self, data, labels):
        try:
            return self.model.score(data, labels)
        except sklearn.exceptions.NotFittedError:
            return 0

    def train(self):
        for _ in range(self.steps_per_round):
            batch_indices = next(self.train_sampler)
            train_data = self.train_data[batch_indices]
            train_labels = self.train_labels[batch_indices]
            self.model.partial_fit(train_data, train_labels, classes=self.class_labels)


if __name__ == "__main__":
    data_dir = '/home/emmasmith/Development/datasets/fraud'
    DATA_FL = "data.npy"
    LABEL_FL = "labels.npy"
    train_fraction = 0.9
    n_learners = 5
    n_epochs = 7
    vote_threshold = 0.5

    fraud_data: np.array = np.load(Path(data_dir) / DATA_FL)
    labels = np.load(Path(data_dir) / LABEL_FL)

    n_datapoints = fraud_data.shape[0]
    random_indices = np.random.permutation(np.arange(n_datapoints))
    n_data_per_learner = n_datapoints // n_learners

    learner_train_data, learner_train_labels, learner_test_data, learner_test_labels = [], [], [], []
    for i in range(n_learners):
        start_ind = i * n_data_per_learner
        stop_ind = (i + 1) * n_data_per_learner
        n_train = int(n_data_per_learner * train_fraction)

        learner_train_ind = random_indices[start_ind:start_ind + n_train]
        learner_test_ind = random_indices[start_ind + n_train:stop_ind]

        learner_train_data.append(fraud_data[learner_train_ind])
        learner_train_labels.append(labels[learner_train_ind])
        learner_test_data.append(fraud_data[learner_test_ind])
        learner_test_labels.append(labels[learner_test_ind])

    all_learner_models = []
    for i in range(n_learners):
        all_learner_models.append(
            FraudSklearnLearner(
                train_data=learner_train_data[i],
                train_labels=learner_train_labels[i],
                test_data=learner_test_data[i],
                test_labels=learner_test_labels[i]
            ))

    set_equal_weights(all_learner_models)

    results = Results()
    # Get initial score
    results.data.append(initial_result(all_learner_models))

    for epoch in range(n_epochs):
        results.data.append(
            collective_learning_round(all_learner_models,
                                      vote_threshold, epoch)
        )

        # then make an updating graph
        plot_results(results, n_learners, block=False, score_name="accuracy")
        plot_votes(results, block=False)

    plot_results(results, n_learners, block=False, score_name="accuracy")
    plot_votes(results, block=True)
