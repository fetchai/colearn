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
from typing import Optional, Callable
from collections import OrderedDict, defaultdict

try:
    import torch
except ImportError:
    raise Exception(
        "Pytorch is not installed. To use the pytorch "
        "add-ons please install colearn with `pip install colearn[pytorch]`."
    )

import torch.nn
import torch.optim
import torch.utils
import torch.utils.data
from torch.nn.modules.loss import _Loss

from colearn.ml_interface import (
    MachineLearningInterface,
    Weights,
    ProposedWeights,
    ColearnModel,
    convert_model_to_onnx,
    ModelFormat,
    DiffPrivBudget,
    DiffPrivConfig,
    TrainingSummary,
    ErrorCodes,
)

from opacus import PrivacyEngine

_DEFAULT_DEVICE = torch.device("cpu")


class PytorchLearner(MachineLearningInterface):
    """
    Pytorch learner implementation of machine learning interface
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loader: torch.utils.data.DataLoader,
        vote_loader: torch.utils.data.DataLoader,
        test_loader: Optional[torch.utils.data.DataLoader] = None,
        need_reset_optimizer: bool = True,
        device=_DEFAULT_DEVICE,
        criterion: Optional[_Loss] = None,
        minimise_criterion=True,
        vote_criterion: Optional[Callable[[torch.Tensor, torch.Tensor], float]] = None,
        num_train_batches: Optional[int] = None,
        num_test_batches: Optional[int] = None,
        diff_priv_config: Optional[DiffPrivConfig] = None,
    ):
        """
        :param model: Pytorch model used for training
        :param optimizer: Training optimizer
        :param train_loader: Train dataset
        :param test_loader: Optional test dataset - subset of training set will be used if not specified
        :param need_reset_optimizer: True to clear optimizer history before training, False to kepp history.
        :param device: Pytorch device - CPU or GPU
        :param criterion: Loss function
        :param minimise_criterion: True to minimise value of criterion, False to maximise
        :param vote_criterion: Function to measure model performance for voting
        :param num_train_batches: Number of training batches
        :param num_test_batches: Number of testing batches
        :param diff_priv_config: Contains differential privacy (dp) budget related configuration
        """

        # Model has to be on same device as data
        self.model: torch.nn.Module = model.to(device)
        self.optimizer: torch.optim.Optimizer = optimizer
        self.criterion = criterion
        self.train_loader: torch.utils.data.DataLoader = train_loader
        self.vote_loader: torch.utils.data.DataLoader = vote_loader
        self.test_loader: Optional[torch.utils.data.DataLoader] = test_loader
        self.need_reset_optimizer = need_reset_optimizer
        self.device = device
        self.num_train_batches = num_train_batches or len(train_loader)
        self.num_test_batches = num_test_batches
        self.minimise_criterion = minimise_criterion
        self.vote_criterion = vote_criterion

        self.dp_config = diff_priv_config
        self.dp_privacy_engine = PrivacyEngine()

        if diff_priv_config is not None:
            (
                self.model,
                self.optimizer,
                self.train_loader,
            ) = self.dp_privacy_engine.make_private(
                module=self.model,
                optimizer=self.optimizer,
                data_loader=self.train_loader,
                max_grad_norm=diff_priv_config.max_grad_norm,
                noise_multiplier=diff_priv_config.noise_multiplier,
            )

        self.vote_score = self.test(self.vote_loader)

    def mli_get_current_weights(self) -> Weights:
        """
        :return: The current weights of the model
        """

        current_state_dict = OrderedDict()
        for key in self.model.state_dict():
            current_state_dict[key] = self.model.state_dict()[key].clone()
        w = Weights(
            weights=current_state_dict, training_summary=self.get_training_summary()
        )

        return w

    def mli_get_current_model(self) -> ColearnModel:
        """
        :return: The current model and its format
        """

        return ColearnModel(
            model_format=ModelFormat(ModelFormat.ONNX),
            model_file="",
            model=convert_model_to_onnx(self.model),
        )

    def set_weights(self, weights: Weights):
        """
        Rewrites weight of current model
        :param weights: Weights to be stored
        """

        self.model.load_state_dict(weights.weights)

    def reset_optimizer(self):
        """
        Clear optimizer state, such as number of iterations, momentums.
        This way, the outdated history can be erased.
        """

        self.optimizer.__setstate__({"state": defaultdict(dict)})

    def train(self):
        """
        Trains the model on the training dataset
        """

        if self.need_reset_optimizer:
            # erase the outdated optimizer memory (momentums mostly)
            self.reset_optimizer()

        self.model.train()

        for batch_idx, (data, labels) in enumerate(self.train_loader):
            if batch_idx == self.num_train_batches:
                break
            self.optimizer.zero_grad()

            # Data needs to be on same device as model
            data = data.to(self.device)
            labels = labels.to(self.device)

            output = self.model(data)

            loss = self.criterion(output, labels)
            loss.backward()
            self.optimizer.step()

    def mli_propose_weights(self) -> Weights:
        """
        Trains model on training set and returns new weights after training
        - Current model is reverted to original state after training
        :return: Weights after training
        """

        current_weights = self.mli_get_current_weights()
        training_summary = current_weights.training_summary
        if (
            training_summary is not None
            and training_summary.error_code is not None
            and training_summary.error_code == ErrorCodes.DP_BUDGET_EXCEEDED
        ):
            return current_weights

        self.train()
        new_weights = self.mli_get_current_weights()
        self.set_weights(current_weights)

        training_summary = new_weights.training_summary
        if (
            training_summary is not None
            and training_summary.error_code is not None
            and training_summary.error_code == ErrorCodes.DP_BUDGET_EXCEEDED
        ):
            current_weights.training_summary = training_summary
            return current_weights

        return new_weights

    def mli_test_weights(self, weights: Weights) -> ProposedWeights:
        """
        Tests given weights on training and test set and returns weights with score values
        :param weights: Weights to be tested
        :return: ProposedWeights - Weights with vote and test score
        """

        current_weights = self.mli_get_current_weights()
        self.set_weights(weights)

        vote_score = self.test(self.vote_loader)

        if self.test_loader:
            test_score = self.test(self.test_loader)
        else:
            test_score = 0
        vote = self.vote(vote_score)

        self.set_weights(current_weights)
        return ProposedWeights(
            weights=weights, vote_score=vote_score, test_score=test_score, vote=vote
        )

    def vote(self, new_score) -> bool:
        """
        Compares current model score with proposed model score and returns vote
        :param new_score: Proposed score
        :return: bool positive or negative vote
        """

        if self.minimise_criterion:
            return new_score < self.vote_score
        else:
            return new_score > self.vote_score

    def test(self, loader: torch.utils.data.DataLoader) -> float:
        """
        Tests performance of the model on specified dataset
        :param loader: Dataset for testing
        :return: Value of performance metric
        """

        if not self.criterion:
            raise Exception("Criterion is unspecified so test method cannot be used")

        self.model.eval()
        total_score = 0
        all_labels = []
        all_outputs = []
        batch_idx = 0
        total_samples = 0
        with torch.no_grad():
            for batch_idx, (data, labels) in enumerate(loader):
                total_samples += labels.shape[0]
                if self.num_test_batches and batch_idx == self.num_test_batches:
                    break
                data = data.to(self.device)
                labels = labels.to(self.device)
                output = self.model(data)
                if self.vote_criterion is not None:
                    all_labels.append(labels)
                    all_outputs.append(output)
                else:
                    total_score += self.criterion(output, labels).item()
        if batch_idx == 0:
            raise Exception("No batches in loader")
        if self.vote_criterion is None:
            return float(total_score / total_samples)
        else:
            return self.vote_criterion(
                torch.cat(all_outputs, dim=0), torch.cat(all_labels, dim=0)
            )

    def mli_accept_weights(self, weights: Weights):
        """
        Updates the model with the proposed set of weights
        :param weights: The new weights
        """

        self.set_weights(weights)
        self.vote_score = self.test(self.vote_loader)

    def get_training_summary(self) -> Optional[TrainingSummary]:
        """
        Differential Privacy Budget
        :return: the target and consumed epsilon so far
        """

        if self.dp_config is None:
            return None

        delta = self.dp_config.target_delta
        target_epsilon = self.dp_config.target_epsilon
        consumed_epsilon = self.dp_privacy_engine.get_epsilon(delta)

        budget = DiffPrivBudget(
            target_epsilon=target_epsilon,
            consumed_epsilon=consumed_epsilon,
            target_delta=delta,
            consumed_delta=delta,  # delta is constatnt per training
        )

        err = (
            ErrorCodes.DP_BUDGET_EXCEEDED
            if consumed_epsilon >= target_epsilon
            else None
        )

        return TrainingSummary(
            dp_budget=budget,
            error_code=err,
        )
