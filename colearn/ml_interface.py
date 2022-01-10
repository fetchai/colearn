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
import abc
from typing import Any, NewType, Optional
from pydantic import BaseModel
from enum import Enum


class Weights(BaseModel):
    weights: Any


class ProposedWeights(BaseModel):
    weights: Weights
    vote_score: float
    test_score: float
    vote: Optional[bool]


class ReportTypes(Enum):
    TRAINING   = 0
    VALIDATION = 1
    NEW        = 3


class MachineLearningInterface(abc.ABC):
    @abc.abstractmethod
    def mli_propose_weights(self) -> Weights:
        """
        Trains the model. Returns new weights. Does not change the current weights of the model.
        """
        pass

    @abc.abstractmethod
    def mli_test_weights(self, weights: Weights) -> ProposedWeights:
        """
        Tests the proposed weights and fills in the rest of the fields
        """

    @abc.abstractmethod
    def mli_accept_weights(self, weights: Weights):
        """
        Updates the model with the proposed set of weights
        :param weights: The new weights
        """
        pass

    @abc.abstractmethod
    def mli_get_current_weights(self) -> Weights:
        """
        Returns the current weights of the model
        """
        pass

    @abc.abstractmethod
    def mli_configure_model(self, parameters: dict) -> bool:
        """
        Configures the model parameters. Returns true if the configuration succeeded Used for:
            * configure differential privacy
        :param parameters: the configuration dictionary
        """
    
    @abc.abstractmethod
    def mli_get_report(self, report_type: ReportTypes) -> dict:
        """
        Returns arbitrary report, contained in the dict object, which in simple case is metric => value.
        :param report_type: select what type of report to return.
        """
