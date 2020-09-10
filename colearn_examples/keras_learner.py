from abc import ABC
from typing import List

import numpy as np

from sklearn.metrics import jaccard_score, confusion_matrix, classification_report, roc_auc_score

from tensorflow._api.v2.compat import v1 as tf

from tqdm import tqdm

from colearn_examples.config import ModelConfig

from colearn.basic_learner import BasicLearner, LearnerData


class KerasWeights:
    __slots__ = ('data',)

    def __init__(self, data: List[np.array]):
        self.data = data


class EarlyStoppingWhenSignaled(tf.keras.callbacks.Callback):
    def __init__(self, check_fn):
        super(EarlyStoppingWhenSignaled, self).__init__()
        self._check_fn = check_fn

    def on_train_batch_end(self, batch, logs=None):
        if self._check_fn():
            self.model.stop_training = True


class KerasLearner(BasicLearner, ABC):
    def __init__(self, config: ModelConfig, data: LearnerData, model: tf.keras.Model = None):
        self.config = config
        BasicLearner.__init__(self, data=data, model=model)
        self._stop_training = False

    def _train_model(self):
        self._stop_training = False

        steps_per_epoch = self.config.steps_per_epoch or (self.data.train_data_size // self.data.train_batch_size)

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

        all_labels = []  # type: ignore
        all_preds = []  # type: ignore

        for _ in tqdm(range(n_steps)):
            data, labels = generator.__next__()
            pred = self._model.predict(data)

            if self.config.multi_hot:
                pass  # all done
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
                except ValueError:
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
