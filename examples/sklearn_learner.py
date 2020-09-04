import copy
from abc import ABC

from sklearn.metrics import roc_auc_score
from sklearn.utils.validation import check_is_fitted
from tqdm import tqdm

from examples.config import ModelConfig
from colearn.basic_learner import BasicLearner, LearnerData


class SKLearnWeights:
    def __init__(self, data):
        self.data = data


class SKLearnLearner(BasicLearner, ABC):
    def __init__(self, config: ModelConfig, data: LearnerData, model=None):
        self.config = config
        BasicLearner.__init__(self, data=data, model=model)

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
            class_labels = self.config.class_labels

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
