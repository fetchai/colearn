from typing import Optional

import torch
import torch.nn
import torch.optim
import torch.utils
import torch.utils.data

from colearn.ml_interface import MachineLearningInterface, Weights, ProposedWeights


class NewPytorchLearner(MachineLearningInterface):
    def __init__(self, model: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 train_loader: torch.utils.data.DataLoader,
                 test_loader: Optional[torch.utils.data.DataLoader] = None,
                 device=torch.device("cpu"),
                 criterion=None,
                 minimise_criterion=True):
        self.model: torch.nn.Module = model
        self.optimizer: torch.optim.Optimizer = optimizer
        self.criterion = criterion
        self.train_loader: torch.utils.data.DataLoader = train_loader
        self.test_loader: Optional[torch.utils.data.DataLoader] = test_loader
        self.device = device
        self.vote_score = self.test(self.train_loader)
        self.minimise_criterion = minimise_criterion

    def mli_get_current_weights(self) -> Weights:
        w = Weights(weights=[x.clone() for x in self.model.parameters()])
        return w

    def set_weights(self, weights: Weights):
        with torch.no_grad():
            for new_param, old_param in zip(weights.weights,
                                            self.model.parameters()):
                old_param.set_(new_param)

    def train(self):
        self.model.train()

        for data, labels in self.train_loader:
            self.optimizer.zero_grad()
            data = data.to(self.device)
            labels = labels.to(self.device)
            output = self.model(data)

            loss = self.criterion(output, labels)
            loss.backward()
            self.optimizer.step()

    def mli_propose_weights(self) -> Weights:
        current_weights = self.mli_get_current_weights()
        self.train()
        new_weights = self.mli_get_current_weights()
        self.set_weights(current_weights)
        return new_weights

    def mli_test_weights(self, weights: Weights, eval_config: Optional[dict] = None) -> ProposedWeights:
        current_weights = self.mli_get_current_weights()
        self.set_weights(weights)

        vote_score = self.test(self.train_loader)

        if self.test_loader:
            test_score = self.test(self.test_loader)
        else:
            test_score = 0
        vote = self.vote(vote_score)

        self.set_weights(current_weights)
        return ProposedWeights(weights=weights,
                               vote_score=vote_score,
                               test_score=test_score,
                               vote=vote
                               )

    def vote(self, new_score) -> bool:
        if self.minimise_criterion:
            return new_score <= self.vote_score
        else:
            return new_score >= self.vote_score

    def test(self, loader) -> float:
        if not self.criterion:
            raise Exception("Criterion is unspecified so test method cannot be used")

        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch_idx, (data, labels) in enumerate(loader):
                data = data.to(self.device)
                labels = labels.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, labels)
                total_loss += loss
        return float(total_loss / len(loader))

    def mli_accept_weights(self, weights: Weights):
        self.set_weights(weights)
        self.vote_score = self.test(self.train_loader)
