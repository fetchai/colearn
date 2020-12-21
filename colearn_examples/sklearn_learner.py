import copy
from abc import ABC
from typing import List, Optional
import numpy as np
from tqdm import tqdm

from sklearn.metrics import roc_auc_score
from sklearn.utils.validation import check_is_fitted

from colearn_examples.config import ModelConfig
from colearn.basic_learner import BasicLearner, LearnerData, Weights


class SKLearnLearner(BasicLearner, ABC):
    def __init__(self, config: ModelConfig, data: LearnerData):
        BasicLearner.__init__(self, config=config, data=data)
        if config.use_dp:
            print("Warning: Differential privacy is not supported for SKLearnLearner")

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
            class_labels = list(range(self.config.n_classes))

        for _ in tqdm(range(steps_per_epoch)):
            data, labels = next(self.data.train_gen)

            # Convert Keras [n,1] format to scikit-learn [n,]
            labels = labels.ravel()

            self._model.partial_fit(data, labels, classes=class_labels)

            y_pred = self._model.decision_function(data)

            all_labels.extend(labels)
            all_preds.extend(y_pred)

        return roc_auc_score(all_labels, all_preds)

    def _test_model(self, weights: Weights = None, validate=False, eval_config: Optional[dict] = None):
        try:
            check_is_fitted(self._model)
        except (ValueError, TypeError, AttributeError):
            return 0, None

        if eval_config is not None:
            print("SKLEARN WARNING: eval config not supported")

        temp_weights = None
        if weights and weights.weights:
            # store current weights in temporary variables
            temp_weights = self.get_current_weights()
            self._set_weights(weights)

        if validate:
            print("Getting vote accuracy:")
            n_steps = self.config.val_batches
            generator = self.data.val_gen
        else:
            print("Getting test accuracy:")
            generator = self.data.test_gen
            n_steps = max(1, int(self.data.test_data_size // self.data.test_batch_size))

        all_labels: List[np.array] = []
        all_preds: List[np.array] = []

        for _ in tqdm(range(n_steps)):
            data, labels = next(generator)
            pred = self._model.decision_function(data)

            all_labels.extend(labels)
            all_preds.extend(pred)

        accuracy = roc_auc_score(all_labels, all_preds)

        if temp_weights:
            # Restore original weights
            self._set_weights(temp_weights)

        print("AUC score: ", accuracy)

        return accuracy, {}

    def print_summary(self):
        print(self._model)

    def get_current_weights(self):
        return Weights(weights=copy.deepcopy(self._model))

    def _set_weights(self, weights: Weights):
        self._model = weights.weights
