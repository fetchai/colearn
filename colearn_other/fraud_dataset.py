from enum import Enum
import tempfile
from pathlib import Path
from colearn_examples.utils.data import shuffle_data
from colearn_examples.utils.data import split_by_chunksizes
import os
import pickle
import numpy as np
from typing import Optional
import sklearn
from typing import Tuple

from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import scale

from colearn.ml_interface import MachineLearningInterface, Weights, ProposedWeights

import pandas as pd

DATA_FL = "data.pickle"
LABEL_FL = "labels.pickle"


class ModelType(Enum):
    SVM = 1


def prepare_learner(model_type: ModelType, data_loaders, **kwargs):
    if model_type == ModelType.SVM:
        return FraudLearner(
            train_data=data_loaders[0][0],
            train_labels=data_loaders[0][1],
            test_data=data_loaders[1][0],
            test_labels=data_loaders[1][1])
    else:
        raise Exception("Model %s not part of the ModelType enum" % model_type)


def _infinite_batch_sampler(data_size, batch_size):
    while True:
        random_ind = np.random.permutation(np.arange(data_size))
        for i in range(0, data_size, batch_size):
            yield random_ind[i:i + batch_size]


class FraudLearner(MachineLearningInterface):
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
        self.train_sampler = _infinite_batch_sampler(train_data.shape[0], batch_size)

        self.model = SGDClassifier(max_iter=1, verbose=0, loss="modified_huber")
        self.model.partial_fit(self.train_data[0:1], self.train_labels[0:1],
                               classes=self.class_labels)  # this needs to be called before predict
        self.vote_score = self.test(self.train_data, self.train_labels)
        self.score_name = "mean_accuracy"

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


def prepare_data_loaders(train_folder, train_ratio=0.8, **kwargs) -> Tuple[Tuple[np.array], Tuple[np.array]]:
    """
    Load training data from folders and create train and test arrays

    :param train_folder: Path to training dataset
    :param train_ratio: What portion of train_data should be used as test set
    :param kwargs:
    :return: Tuple of tuples (train_data, train_labels), (test_data, test_loaders)
    """

    data = pickle.load(open(Path(train_folder) / DATA_FL, "rb"))
    labels = pickle.load(open(Path(train_folder) / LABEL_FL, "rb"))

    n_cases = int(train_ratio * len(data))
    assert (n_cases > 0), "There are no cases"

    # (train_data, train_labels), (test_data, test_labels)
    return (data[:n_cases], labels[:n_cases]), (data[n_cases:], labels[n_cases:])


def split_to_folders(
        data_dir,
        n_learners,
        data_split=None,
        shuffle_seed=None,
        output_folder=None,
        **kwargs):
    if output_folder is None:
        output_folder = Path(tempfile.gettempdir()) / "fraud"

    if data_split is None:
        data_split = [1 / n_learners] * n_learners

    train_identity = pd.read_csv(str(Path(data_dir) / "train_identity.csv"))
    train_transaction = pd.read_csv(str(Path(data_dir) / "train_transaction.csv"))
    # Combine the data and work with the whole dataset
    train = pd.merge(train_transaction, train_identity, on="TransactionID", how="left")

    del train_identity, train_transaction

    cat_cols = ["id_12", "id_13", "id_14", "id_15", "id_16", "id_17", "id_18", "id_19", "id_20", "id_21", "id_22",
                "id_23", "id_24", "id_25", "id_26", "id_27", "id_28", "id_29", "id_30", "id_31", "id_32", "id_33",
                "id_34", "id_35", "id_36", "id_37", "id_38", "DeviceType", "DeviceInfo", "ProductCD", "card4", "card6",
                "M4", "P_emaildomain", "R_emaildomain", "card1", "card2", "card3", "card5", "addr1", "addr2", "M1",
                "M2", "M3", "M5", "M6", "M7", "M8", "M9", "P_emaildomain_1", "P_emaildomain_2", "P_emaildomain_3",
                "R_emaildomain_1", "R_emaildomain_2", "R_emaildomain_3",
                ]

    for col in cat_cols:
        if col in train.columns:
            le = LabelEncoder()
            le.fit(list(train[col].astype(str).values))
            train[col] = le.transform(list(train[col].astype(str).values))

    x = train.sort_values("TransactionDT").drop(
        ["isFraud", "TransactionDT", "TransactionID"], axis=1
    )
    y = train.sort_values("TransactionDT")["isFraud"]

    del train

    # Cleaning infinite values to NaN
    x = x.replace([np.inf, -np.inf], np.nan)

    for column in x:
        x[column] = x[column].fillna(x[column].mean())

    data = x.to_numpy().astype(np.float32)
    labels = y.to_numpy().astype(np.float32)

    data = scale(data)

    [data, labels] = shuffle_data(
        [data, labels], seed=shuffle_seed
    )

    [data, labels] = split_by_chunksizes(
        [data, labels], data_split
    )

    local_output_dir = Path(output_folder)

    dir_names = []
    for i in range(n_learners):
        dir_name = local_output_dir / str(i)
        os.makedirs(str(dir_name), exist_ok=True)

        pickle.dump(data[i], open(dir_name / DATA_FL, "wb"))
        pickle.dump(labels[i], open(dir_name / LABEL_FL, "wb"))

        dir_names.append(dir_name)

    print(dir_names)
    return [str(x) for x in dir_names]
