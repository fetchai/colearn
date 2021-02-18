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
import os
import pickle
import tempfile
from enum import Enum
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import scale

from colearn.ml_interface import MachineLearningInterface, Weights, ProposedWeights
from colearn.utils.data import split_list_into_fractions

DATA_FL = "data.pickle"
LABEL_FL = "labels.pickle"


class ModelType(Enum):
    SVM = 1


class FraudLearner(MachineLearningInterface):
    """
    Fraud dataset learner implementation of machine learning interface using Scikit-learn
    """

    def __init__(self,
                 train_data: np.array,
                 train_labels: np.array,
                 test_data: np.array,
                 test_labels: np.array,
                 batch_size: int = 10000,
                 steps_per_round: int = 1):
        """
        :param train_data: np.array of training data
        :param train_labels: np.array of training labels
        :param test_data:  np.array of testing data
        :param test_labels: np.array of testing labels
        :param batch_size:  Batch size
        :param steps_per_round:  Number of training batches per round
        """
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

    def mli_propose_weights(self) -> Weights:
        """
        Trains model on training set and returns new weights after training
        - Current model is reverted to original state after training
        :return: Weights after training
        """

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
        """
        Tests given weights on training and test set and returns weights with score values
        :param weights: Weights to be tested
        :return: ProposedWeights - Weights with vote and test score
        """

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
        """
        Updates the model with the proposed set of weights
        :param weights: The new weights
        """
        self.set_weights(weights)
        self.vote_score = self.test(self.train_data, self.train_labels)

    def mli_get_current_weights(self):
        """
        :return: The current weights of the model
        """

        return Weights(weights=dict(coef_=self.model.coef_,
                                    intercept_=self.model.intercept_))

    def set_weights(self, weights: Weights):
        """
        Rewrites weight of current model
        :param weights: Weights to be stored
        """

        self.model.coef_ = weights.weights['coef_']
        self.model.intercept_ = weights.weights['intercept_']

    def test(self, data: np.array, labels: np.array) -> float:
        """
        Tests performance of the model on specified dataset
        :param data: np.array of data
        :param labels: np.array of labels
        :return: Value of performance metric
        """
        try:
            return self.model.score(data, labels)
        except sklearn.exceptions.NotFittedError:
            return 0


def prepare_learner(model_type: ModelType,
                    data_loaders: Tuple[Tuple[np.array, np.array], Tuple[np.array, np.array]],
                    **_kwargs) -> FraudLearner:
    """
    Creates a new instance of FraudLearner
    :param model_type: Enum that represents selected model type
    :param data_loaders: Tuple of tuples (train_data, train_labels), (test_data, test_labels)
    :param _kwargs: Residual parameters not used by this function
    :return: Instance of FraudLearner
    """
    if model_type == ModelType.SVM:
        return FraudLearner(
            train_data=data_loaders[0][0],
            train_labels=data_loaders[0][1],
            test_data=data_loaders[1][0],
            test_labels=data_loaders[1][1])
    else:
        raise Exception("Model %s not part of the ModelType enum" % model_type)


def _infinite_batch_sampler(data_size: int,
                            batch_size: int) -> np.array:
    """
    Generates random array of indices
    :param data_size: Number of samples in dataset
    :param batch_size: Batch size
    :yield: Batch of random indices
    """
    while True:
        random_ind = np.random.permutation(np.arange(data_size))
        for i in range(0, data_size, batch_size):
            yield random_ind[i:i + batch_size]


def prepare_data_loaders(train_folder: str,
                         train_ratio: float = 0.8,
                         **_kwargs) -> Tuple[Tuple[np.array, np.array], Tuple[np.array, np.array]]:
    """
    Load training data from folders and create train and test arrays

    :param train_folder: Path to training dataset
    :param train_ratio: What portion of train_data should be used as test set
    :param _kwargs:
    :return: Tuple of tuples (train_data, train_labels), (test_data, test_loaders)
    """

    data = pickle.load(open(Path(train_folder) / DATA_FL, "rb"))
    labels = pickle.load(open(Path(train_folder) / LABEL_FL, "rb"))

    n_cases = int(train_ratio * len(data))
    assert (n_cases > 0), "There are no cases"

    # (train_data, train_labels), (test_data, test_labels)
    return (data[:n_cases], labels[:n_cases]), (data[n_cases:], labels[n_cases:])


def fraud_preprocessing(data_dir, use_cache=True):
    preprocessed_data_file = Path(data_dir) / "data.npy"
    preprocessed_labels_file = Path(data_dir) / "labels.npy"

    if use_cache and os.path.isfile(preprocessed_data_file) and os.path.isfile(preprocessed_labels_file):
        fraud_data: np.array = np.load(preprocessed_data_file)
        labels: np.array = np.load(preprocessed_labels_file)
        return fraud_data, labels

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

    np.save(preprocessed_data_file, data)
    np.save(preprocessed_labels_file, labels)
    return data, labels


def split_to_folders(
        data_dir: str,
        n_learners: int,
        data_split: Optional[List[float]] = None,
        shuffle_seed: Optional[int] = None,
        output_folder: Optional[Path] = None,
        **_kwargs) -> List[str]:
    """
    Loads fraud dataset, preprocesses and splits it to specified number of subsets
    :param data_dir: Folder containing Fraud dataset .csv files
    :param n_learners: Number of parts for splitting
    :param data_split:  List of percentage portions for each subset
    :param shuffle_seed: Seed for shuffling
    :param output_folder: Folder where splitted parts will be stored as numbered subfolders
    :param _kwargs: Residual parameters not used by this function
    :return: List of folders containing individual subsets
    """
    if output_folder is None:
        output_folder = Path(tempfile.gettempdir()) / "fraud"

    if data_split is None:
        data_split = [1 / n_learners] * n_learners

    data, labels = fraud_preprocessing(data_dir)

    n_datapoints = data.shape[0]
    np.random.seed(shuffle_seed)
    random_indices = np.random.permutation(np.arange(n_datapoints))
    split_indices = split_list_into_fractions(random_indices, data_split)

    local_output_dir = Path(output_folder)

    dir_names = []
    for i in range(n_learners):
        dir_name = local_output_dir / str(i)
        os.makedirs(str(dir_name), exist_ok=True)

        learner_data = data[split_indices[i]]
        learner_labels = labels[split_indices[i]]

        pickle.dump(learner_data, open(dir_name / DATA_FL, "wb"))
        pickle.dump(learner_labels, open(dir_name / LABEL_FL, "wb"))

        dir_names.append(dir_name)

    print(dir_names)
    return [str(x) for x in dir_names]
