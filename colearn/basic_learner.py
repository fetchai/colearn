import hashlib
import pickle

from colearn.ml_interface import ProposedWeights, MachineLearningInterface, \
    Weights


class RingBuffer:
    def __init__(self, size_limit):
        self._max_size = size_limit
        self._keys_list = []
        self._data_dict = {}

    def add(self, key, value):
        h = self.hash(key)
        if h in self._data_dict:
            pass
        else:
            # add new key
            if len(self._keys_list) >= self._max_size:
                oldest_key = self._keys_list.pop(0)
                self._data_dict.pop(oldest_key, None)
            self._data_dict[h] = value
            self._keys_list.append(h)

    def get(self, key):
        h = self.hash(key)
        return self._data_dict[h]

    @staticmethod
    def hash(key):
        bstr = pickle.dumps(key)
        m = hashlib.sha256()
        m.update(bstr)
        return m.digest()


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
        self.data = data

        self._model = model or self._get_model()
        self.print_summary()

        self.vote_score_cache = RingBuffer(10)

        self.vote_accuracy = self._test_model(validate=True)

        # store this in the cache
        self.vote_score_cache.add(self.get_weights().weights,
                                  self.vote_accuracy)

    def print_summary(self):
        raise NotImplementedError

    def get_name(self) -> str:
        return "Basic Learner"

    def _get_model(self):
        raise NotImplementedError

    def accept_weights(self, weights: Weights):
        # overwrite model weights with new weights
        self._set_weights(weights)

        # update stored performance metrics
        try:
            self.vote_accuracy = self.vote_score_cache.get(
                weights.weights)
        except KeyError:
            print("Warning: weights not in cache")
            self.vote_accuracy = self._test_model(weights,
                                                  validate=True)

            # store this in the cache
            self.vote_score_cache.add(weights.weights,
                                      self.vote_accuracy)

    def train_model(self):
        old_weights = self.get_weights()

        self._train_model()

        new_weights = self.get_weights()
        self._set_weights(old_weights)

        return new_weights

    def _set_weights(self, weights: Weights):
        raise NotImplementedError

    def get_weights(self) -> Weights:
        raise NotImplementedError

    def test_model(self, weights: Weights = None) -> ProposedWeights:
        """Tests the proposed weights and fills in the rest of the fields"""
        if weights is None:
            weights = self.get_weights()

        proposed_weights = ProposedWeights()
        proposed_weights.weights = weights
        try:
            proposed_weights.vote_accuracy = self.vote_score_cache.get(
                weights.weights)
        except KeyError:
            proposed_weights.vote_accuracy = self._test_model(weights,
                validate=True)

            # store this in the cache
            self.vote_score_cache.add(weights.weights,
                                      proposed_weights.vote_accuracy)

        proposed_weights.test_accuracy = self._test_model(weights,
                                                          validate=False)
        proposed_weights.vote = (
            proposed_weights.vote_accuracy >= self.vote_accuracy
        )

        return proposed_weights

    def _test_model(self, weights: Weights = None, validate=False):
        raise NotImplementedError

    def _train_model(self):
        raise NotImplementedError

    def stop_training(self):
        raise NotImplementedError

    def clone(self, data=None):
        raise NotImplementedError
