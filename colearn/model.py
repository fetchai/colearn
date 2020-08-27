import copy
from abc import ABC
from typing import List
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    jaccard_score,
)
from sklearn.utils.validation import check_is_fitted
import numpy as np
from tqdm import tqdm
import tensorflow.compat.v1 as tf

from colearn.ml_interface import ProposedWeights, MachineLearningInterface
from colearn.config import Config
from colearn.data import LearnerData

tf.disable_v2_behavior()


class KerasWeights:
    def __init__(self, data: List[np.array]):
        self.data = data


class SKLearnWeights:
    def __init__(self, data):
        self.data = data


class BasicLearner(MachineLearningInterface):
    def __init__(self, config: Config, data: LearnerData, model=None):
        self.vote_accuracy = 0

        self.data = data
        self.config = config

        self._model = model or self._get_model()
        self.print_summary()

    def print_summary(self):
        raise NotImplementedError

    def get_name(self) -> str:
        return "Basic Learner"

    def _get_model(self):
        raise NotImplementedError

    def accept_weights(self, proposed_weights: ProposedWeights):
        # overwrite model weights with accepted weights from across the network
        self._set_weights(proposed_weights.weights)

        # update stored performance metrics with metrics computed based on those weights
        self.vote_accuracy = proposed_weights.validation_accuracy

    def train_model(self):
        old_weights = self.get_weights()

        train_accuracy = self._train_model()

        new_weights = self.get_weights()
        self._set_weights(old_weights)

        print("Train accuracy: ", train_accuracy)

        return new_weights

    def _set_weights(self, weights):
        raise NotImplementedError

    def get_weights(self):
        raise NotImplementedError

    def test_model(self, weights=None) -> ProposedWeights:
        """Tests the proposed weights and fills in the rest of the fields"""
        if weights is None:
            weights = self.get_weights()

        proposed_weights = ProposedWeights()
        proposed_weights.weights = weights
        proposed_weights.validation_accuracy = self._test_model(weights, validate=True)
        proposed_weights.test_accuracy = self._test_model(weights, validate=False)
        proposed_weights.vote = (
            proposed_weights.validation_accuracy >= self.vote_accuracy
        )

        return proposed_weights

    def _test_model(self, weights=None, validate=False):
        raise NotImplementedError

    def _train_model(self):
        raise NotImplementedError

    def stop_training(self):
        raise NotImplementedError

    def clone(self, data=None):
        raise NotImplementedError


class EarlyStoppingWhenSignaled(tf.keras.callbacks.Callback):
    def __init__(self, check_fn):
        super(EarlyStoppingWhenSignaled, self).__init__()
        self._check_fn = check_fn

    def on_train_batch_end(self, batch, logs=None):
        if self._check_fn():
            self.model.stop_training = True


class KerasLearner(BasicLearner, ABC):
    def __init__(self, config: Config, data: LearnerData, model: tf.keras.Model = None):
        BasicLearner.__init__(self, config, data=data, model=model)
        self._stop_training = False

        self.x_hat = list()
        self.s_list = []

    def _train_model(self):
        self._stop_training = False

        steps_per_epoch = (
            self.config.steps_per_epoch
            or self.data.train_data_size // self.data.train_batch_size
        )

        history = self._model.fit(
            self.data.train_gen,
            steps_per_epoch=steps_per_epoch,
            use_multiprocessing=False,
            callbacks=[EarlyStoppingWhenSignaled(lambda: self._stop_training)],
        )
        if "accuracy" in history.history:
            train_accuracy = np.mean(history.history["accuracy"])
        else:
            train_accuracy = np.mean(history.history["acc"])

        return train_accuracy

    def stop_training(self):
        self._stop_training = True

    @staticmethod
    def calculate_gradients(new_weights: np.array, old_weights: np.array):
        grad_list = []
        for nw, ow in zip(new_weights, old_weights):
            grad_list.append(nw - ow)
        return grad_list

    def _test_model(self, weights: KerasWeights = None, validate=False):
        temp_weights = []
        if weights and weights.data:
            # store current weights in temporary variables
            temp_weights = self._model.get_weights()
            self._model.set_weights(weights.data)

        if validate:
            print("Getting vote accuracy:")
            n_steps = self.config.val_batches
            generator = self.data.val_gen
        else:
            print("Getting test accuracy:")
            generator = self.data.test_gen
            n_steps = max(1, int(self.data.test_data_size // self.data.test_batch_size))

        all_labels = []
        all_preds = []

        for _ in tqdm(range(n_steps)):
            data, labels = generator.__next__()
            pred = self._model.predict(data)

            if self.config.multi_hot:
                labels = labels
                pred = pred
            else:
                if self.config.n_classes > 1:
                    # Multiple classes
                    pred = np.argmax(pred, axis=1)

                    # Convert label IDs to names - accuracy
                    labels = [self.config.class_labels[int(j)] for j in labels]
                    pred = [self.config.class_labels[int(j)] for j in pred]

            # else: Binary class - AOC metrics

            all_labels.extend(labels)
            all_preds.extend(pred)

        # Multihot = Jaccard index
        if self.config.multi_hot:
            pred = pred > 0.5
            accuracy = jaccard_score(labels, pred, average="weighted")
            print("Jaccard index:", accuracy)

        # One-hot
        else:
            # Multiple classes one-hot = balanced accuracy
            if self.config.n_classes > 1:
                conf_matrix = confusion_matrix(
                    all_labels, all_preds, labels=self.config.class_labels
                )
                class_report = classification_report(
                    all_labels, all_preds, labels=self.config.class_labels
                )

                # Calculate balanced accuracy
                per_class = np.nan_to_num(
                    np.diag(conf_matrix) / conf_matrix.sum(axis=1), nan=1.0
                )
                accuracy = np.mean(per_class)

                print("Test balanced_accuracy_score:", accuracy)
                print("Confusion martrix:\n", conf_matrix)
                print("Classification report:\n", class_report)

            # One class binary = AUC score
            else:
                try:
                    accuracy = roc_auc_score(all_labels, all_preds)

                    # Convert probabilities to classes
                    pred_cat = [j > 0.5 for j in all_preds]

                    conf_matrix = confusion_matrix(
                        all_labels, pred_cat, labels=range(2)
                    )
                    class_report = classification_report(
                        all_labels, pred_cat, labels=range(2)
                    )

                    print("AUC score:", accuracy)
                    print("Confusion martrix:\n", conf_matrix)
                    print("Classification report:\n", class_report)
                except:
                    accuracy = 0
                    print("AUC score:", accuracy)

        if temp_weights:
            # Restore original weights
            self._model.set_weights(temp_weights)

        return accuracy

    def print_summary(self):
        self._model.summary()

    def clone(self, data=None):
        cloned_model = tf.keras.models.clone_model(self._model)
        # compile model & add optimiser
        opt = self.config.optimizer(
            lr=self.config.l_rate, decay=self.config.l_rate_decay
        )

        cloned_model.compile(loss=self.config.loss, metrics=["accuracy"], optimizer=opt)
        data = data or self.data
        return KerasLearner(self.config, data=data, model=cloned_model)

    def get_weights(self):
        return KerasWeights(self._model.get_weights())

    def _set_weights(self, weights: KerasWeights):
        self._model.set_weights(weights.data)


class SKLearnLearner(BasicLearner, ABC):
    def __init__(self, config: Config, data: LearnerData, model=None):
        BasicLearner.__init__(self, config, data=data, model=model)

    def _train_model(self):
        steps_per_epoch = (
            self.config.steps_per_epoch
            or self.data.train_data_size // self.data.train_batch_size
        )

        all_labels = []
        all_preds = []

        # Get names of classes
        if self.config.n_classes == 1:
            class_labels = range(2)
        else:
            class_labels = self.class_labels

        for _ in tqdm(range(steps_per_epoch)):
            data, labels = self.data.train_gen.__next__()

            # Convert Keras [n,1] format to scikit-learn [n,]
            labels = labels.ravel()

            self._model.partial_fit(data, labels, classes=class_labels)

            y_pred = self._model.decision_function(data)

            all_labels.extend(labels)
            all_preds.extend(y_pred)

        return roc_auc_score(all_labels, all_preds)

    def stop_training(self):
        raise NotImplementedError

    def _test_model(self, weights: SKLearnWeights = None, validate=False):
        try:
            check_is_fitted(self._model)
        except:
            return 0

        temp_weights = []
        if weights and weights.data:
            # store current weights in temporary variables
            temp_weights = self.get_weights()
            self._set_weights(weights)

        if validate:
            print("Getting vote accuracy:")
            n_steps = self.config.val_batches
            generator = self.data.val_gen
        else:
            print("Getting test accuracy:")
            generator = self.data.test_gen
            n_steps = max(1, int(self.data.test_data_size // self.data.test_batch_size))

        all_labels = []
        all_preds = []

        for _ in tqdm(range(n_steps)):
            data, labels = generator.__next__()
            pred = self._model.decision_function(data)

            all_labels.extend(labels)
            all_preds.extend(pred)

        accuracy = roc_auc_score(all_labels, all_preds)

        if temp_weights:
            # Restore original weights
            self._set_weights(temp_weights)

        print("AUC score: ", accuracy)

        return accuracy

    def print_summary(self):
        print(self._model)

    def clone(self, data=None):
        cloned_model = copy.deepcopy(self._model)
        data = data or self.data
        return SKLearnLearner(self.config, data=data, model=cloned_model)

    def get_weights(self):
        return SKLearnWeights(copy.deepcopy(self._model))

    def _set_weights(self, weights: SKLearnWeights):
        self._model = weights.data


def setup_models(config: Config, learner_datasets: List[LearnerData]):
    all_learner_models = []
    clone_model = None
    if config.clone_model:
        clone_model = config.model_type(config, data=learner_datasets[0])

    for i in range(config.n_learners):
        if not config.clone_model:
            model = config.model_type(config, data=learner_datasets[i])
        else:
            model = clone_model.clone(data=learner_datasets[i])

        all_learner_models.append(model)

    return all_learner_models
