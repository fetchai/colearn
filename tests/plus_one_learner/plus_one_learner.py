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
from colearn.ml_interface import MachineLearningInterface, Prediction, PredictionRequest, ProposedWeights, \
    Weights, ColearnModel


class PlusOneLearner(MachineLearningInterface):
    def __init__(self, start_value):
        self.current_value = start_value

    def mli_propose_weights(self):
        self.current_value += 1
        return Weights(weights=self.current_value)

    def mli_test_weights(self, weights) -> ProposedWeights:
        if weights.weights > self.current_value:
            test_score = 1.0
            vote_score = 1.0
            vote = True
        elif weights == self.current_value:
            test_score = 0.5
            vote_score = 0.5
            vote = False
        else:
            test_score = 0.0
            vote_score = 0.0
            vote = False

        result = ProposedWeights(weights=weights,
                                 vote_score=vote_score,
                                 test_score=test_score,
                                 vote=vote
                                 )

        return result

    def mli_accept_weights(self, weights: Weights):
        self.current_value = weights.weights

    def mli_get_current_weights(self) -> Weights:
        return Weights(weights=self.current_value)

    def mli_get_current_model(self) -> ColearnModel:
        """
        :return: The current model and its format - not relevant here
        """

        return ColearnModel()

    def mli_make_prediction(self, request: PredictionRequest) -> Prediction:
        raise NotImplementedError()
