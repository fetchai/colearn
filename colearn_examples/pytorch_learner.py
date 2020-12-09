from abc import ABC
from typing import Optional

import numpy as np
from tqdm import trange
import torch
from torchsummary import summary
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from opacus import PrivacyEngine

from colearn.basic_learner import BasicLearner, LearnerData
from colearn.ml_interface import Weights
from colearn_examples.config import ModelConfig


class PytorchLearner(BasicLearner, ABC):
    def __init__(self, config: ModelConfig, data: LearnerData):
        self._stop_training = False
        BasicLearner.__init__(self, config=config, data=data)
        self._optimizer = self.config.optimizer(
            self._model.parameters(),
            lr=self.config.l_rate
        )

        if config.use_dp:
            privacy_engine = PrivacyEngine(self._model,
                                           batch_size=self.config.batch_size,
                                           sample_size=self.config.sample_size,
                                           alphas=self.config.alphas,
                                           noise_multiplier=self.config.noise_multiplier,
                                           max_grad_norm=config.max_grad_norm)
            privacy_engine.attach(self._optimizer)
        self._criterion = self.config.loss

        if self.config.n_classes > 1:
            assert self.config.n_classes == len(self.config.class_labels)

    def _train_model(self):
        self._stop_training = False
        steps_per_epoch = self.config.steps_per_epoch or (self.data.train_data_size // self.data.train_batch_size)
        progress_bar = trange(steps_per_epoch, desc='Training: ', leave=True)

        self._model.train()  # sets model to "training" mode. Does not perform training
        for _ in progress_bar:
            if self._stop_training:
                break
            data, labels = self.data.train_gen.__next__()
            data = torch.Tensor(data).reshape((self.config.batch_size, 1, self.config.height, self.config.width))  # todo: fix model so channels is not required
            labels = torch.LongTensor(labels).squeeze()

            self._optimizer.zero_grad()
            output = self._model(data)
            loss = self._criterion(output, labels)
            loss.backward()
            self._optimizer.step()

    def print_summary(self):
        summary(self._model, (1, self.config.width, self.config.height))

    def _set_weights(self, weights: Weights):
        with torch.no_grad():
            for new_param, old_param in zip(weights.weights,
                                            self._model.parameters()):
                old_param.set_(new_param)

    def get_weights(self) -> Weights:
        w = Weights([x.clone() for x in self._model.parameters()])
        return w

    def _test_model(self, weights: Weights = None, validate=False, eval_config: Optional[dict] = None):
        temp_weights = None
        if weights is not None:
            # store current weights in temporary variables
            temp_weights = self.get_weights()
            self._set_weights(weights)

        if validate:
            print("Getting vote accuracy:")
            n_steps = self.config.val_batches
            generator = self.data.val_gen
            progress_bar = trange(n_steps, desc='Validating: ', leave=True)
        else:
            print("Getting test accuracy:")
            generator = self.data.test_gen
            n_steps = max(1, int(self.data.test_data_size // self.data.test_batch_size))
            progress_bar = trange(n_steps, desc='Testing: ', leave=True)

        all_labels = []  # type: ignore
        all_preds = []  # type: ignore

        for _ in progress_bar:  # tqdm provides progress bar
            data, labels = generator.__next__()
            data = torch.Tensor(data)
            data = torch.reshape(data, (self.config.batch_size, 1, self.config.height, self.config.width))
            pred = self._model(data)

            if self.config.multi_hot:
                pass  # all done
            else:
                if self.config.n_classes > 1:
                    # Multiple classes
                    pred = np.argmax(pred.detach().numpy(), axis=1)

                    # Convert label IDs to names - accuracy
                    labels = [self.config.class_labels[int(j)] for j in labels]
                    pred = [self.config.class_labels[int(j)] for j in pred]

            # else: Binary class - AOC metrics

            all_labels.extend(labels)
            all_preds.extend(pred)

        # Multihot = Jaccard index
        if self.config.multi_hot:
            raise Exception("Multi-hot not supported")
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
                accuracy_for_absent_classes = 1.0/self.config.n_classes
                per_class = np.nan_to_num(
                    np.diag(conf_matrix) / conf_matrix.sum(axis=1), nan=accuracy_for_absent_classes
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
            self._set_weights(temp_weights)

        return accuracy, {}

    def stop_training(self):
        self._stop_training = True
