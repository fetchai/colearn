from colearn.ml_interface import MachineLearningInterface, ProposedWeights, \
    Weights


class PlusOneLearner(MachineLearningInterface):
    def __init__(self, start_value):
        self.current_value = start_value

    def mli_propose_weights(self):
        self.current_value += 1
        return Weights(weights=self.current_value)

    def mli_test_weights(self, weights, eval_config=None) -> ProposedWeights:
        if weights.weights > self.current_value:
            test_accuracy = 1.0
            vote_accuracy = 1.0
            vote = True
        elif weights == self.current_value:
            test_accuracy = 0.5
            vote_accuracy = 0.5
            vote = False
        else:
            test_accuracy = 0.0
            vote_accuracy = 0.0
            vote = False

        result = ProposedWeights(weights=weights,
                                 vote_score=vote_accuracy,
                                 test_score=test_accuracy,
                                 vote=vote
                                 )

        return result

    def mli_accept_weights(self, weights: Weights, eval_config: dict = None):
        self.current_value = weights.weights

    def mli_get_current_weights(self) -> Weights:
        return Weights(weights=self.current_value)
