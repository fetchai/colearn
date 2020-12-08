from colearn.ml_interface import MachineLearningInterface, ProposedWeights, \
    Weights


class PlusOneLearner(MachineLearningInterface):
    def __init__(self, start_value):
        self.current_value = start_value

    def get_name(self) -> str:
        return "PlusOneLearner"

    def train_model(self):
        self.current_value += 1
        return self.current_value

    def stop_training(self):
        pass

    def clone(self):
        return PlusOneLearner(self.current_value)

    def test_model(self, weights=None, eval_config=None) -> ProposedWeights:
        result = ProposedWeights()
        result.weights = weights
        if weights > self.current_value:
            result.test_accuracy = 1.0
            result.vote_accuracy = 1.0
            result.vote = True
        elif weights == self.current_value:
            result.test_accuracy = 0.5
            result.vote_accuracy = 0.5
            result.vote = False
        else:
            result.test_accuracy = 0.0
            result.vote_accuracy = 0.0
            result.vote = False
        return result

    def accept_weights(self, weights: Weights, eval_config: dict = None):
        self.current_value = weights
