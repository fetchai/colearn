import abc


class ProposedWeights:
    def __init__(self):
        self.weights = None
        self.validation_accuracy = 0
        self.test_accuracy = 0
        self.vote = False


class Learner_Interface(abc.ABC):
    @abc.abstractmethod
    def get_name(self) -> str:
        pass

    @abc.abstractmethod
    def train_model(self):
        pass

    @abc.abstractmethod
    def stop_training(self):
        pass

    @abc.abstractmethod
    def clone(self):
        pass

    @abc.abstractmethod
    def test_model(self, weights) -> ProposedWeights:
        """Tests the proposed weights and fills in the rest of the fields"""

    @abc.abstractmethod
    def accept_weights(self, proposed_weights: ProposedWeights):
        pass


class LearnerInterfaceFactory(abc.ABC):
    @abc.abstractmethod
    def create(self, config: dict, index: int) -> Learner_Interface:
        pass
