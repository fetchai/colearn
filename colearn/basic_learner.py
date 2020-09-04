from colearn.ml_interface import ProposedWeights, MachineLearningInterface


class RingBuffer:
    def __init__(self, size_limit):
        self._max_size = size_limit
        self._keys_list = []
        self._data_dict = {}

    def add(self, key, value):
        if len(self._keys_list) >= self._max_size:
            oldest_key = self._keys_list.pop(0)
            self._data_dict.pop(oldest_key, None)
        h = hash(key)
        self._data_dict[h] = value
        self._keys_list.append(h)

    def get(self, key):
        h = hash(key)
        return self._data_dict.get(h)


class EmptyGenerator:
    def __next__(self):
        pass

class LearnerData:
    train_gen = EmptyGenerator()
    val_gen = EmptyGenerator()  # this is a copy of train gen
    test_gen = EmptyGenerator()

    train_data_size = 0  # this includes augmentation
    test_data_size = 0  # this includes augmentation

    train_batch_size = 0
    test_batch_size = 0


class BasicLearner(MachineLearningInterface):
    def __init__(self, data: LearnerData, model=None):
        self.vote_accuracy = 0

        self.data = data

        self._model = model or self._get_model()
        self.print_summary()

        self.vote_score_cache = RingBuffer(10)

    def print_summary(self):
        raise NotImplementedError

    def get_name(self) -> str:
        return "Basic Learner"

    def _get_model(self):
        raise NotImplementedError

    def accept_weights(self, proposed_weights: ProposedWeights):
        # overwrite model weights with new weights
        self._set_weights(proposed_weights.weights)

        # update stored performance metrics
        try:
            self.vote_accuracy = self.vote_score_cache.get(
                proposed_weights.weights)
        except KeyError:
            self.vote_accuracy = self._test_model(proposed_weights.weights,
                                                  validate=True)

    def train_model(self):
        old_weights = self.get_weights()

        self._train_model()

        new_weights = self.get_weights()
        self._set_weights(old_weights)

        return new_weights

    def _set_weights(self, weights):
        raise NotImplementedError

    def get_weights(self):
        raise NotImplementedError

    def test_model(self, weights=None) -> ProposedWeights:
        """Tests the proposed weights and fills in the rest of the fields"""
        if weights is None:
            weights = self.get_weights()

        proposed_weights = ProposedWeights()
        proposed_weights.weights = weights
        proposed_weights.validation_accuracy = self._test_model(weights,
                                                                validate=True)
        # store this in the cache
        self.vote_score_cache.add(proposed_weights.weights,
                                  proposed_weights.validation_accuracy)

        proposed_weights.test_accuracy = self._test_model(weights,
                                                          validate=False)
        proposed_weights.vote = (
            proposed_weights.validation_accuracy >= self.vote_accuracy
        )

        return proposed_weights

    def _test_model(self, weights=None, validate=False):
        raise NotImplementedError

    def _train_model(self):
        raise NotImplementedError

    def stop_training(self):
        raise NotImplementedError

    def clone(self, data=None):
        raise NotImplementedError
