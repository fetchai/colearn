import hashlib
import pickle
from typing import Optional, Tuple, Generator

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


class LearnerData:
    train_gen: Generator
    val_gen: Generator  # this is a copy of train gen
    test_gen: Generator

    train_data_size: int = 0  # this includes augmentation
    test_data_size: int = 0  # this includes augmentation

    train_batch_size: int = 0
    test_batch_size: int = 0


class BasicLearner(MachineLearningInterface):
    def __init__(self, config, data: LearnerData):
        self.config = config
        self.data = data

        self._model = self._get_model()
        self.print_summary()

        self.vote_score_cache = RingBuffer(10)

        self.vote_accuracy, _ = self._test_model(validate=True)

        # store this in the cache
        self.vote_score_cache.add(self.get_current_weights(),
                                  self.vote_accuracy)

    def print_summary(self):
        raise NotImplementedError

    def _get_model(self):
        raise NotImplementedError

    def accept_weights(self, weights: Weights):
        # overwrite model weights with new weights
        self._set_weights(weights)

        # update stored performance metrics
        try:
            self.vote_accuracy = self.vote_score_cache.get(weights)
        except KeyError:
            print("Warning: weights not in cache")
            self.vote_accuracy, _ = self._test_model(weights,
                                                     validate=True)

            # store this in the cache
            self.vote_score_cache.add(weights,
                                      self.vote_accuracy)

    def propose_weights(self):
        old_weights = self.get_current_weights()

        self._train_model()

        new_weights = self.get_current_weights()
        self._set_weights(old_weights)

        return new_weights

    def _set_weights(self, weights: Weights):
        raise NotImplementedError

    def get_current_weights(self) -> Weights:
        raise NotImplementedError

    def test_weights(self, weights: Weights = None, eval_config: dict = None) -> ProposedWeights:
        """
            Tests the proposed weights and fills in the rest of the fields
            Also evaluates the model using the metrics specified in eval_config.
                eval_config = {
                    "name": lambda y_true, y_pred
                }
        """
        if weights is None:
            weights = self.get_current_weights()

        try:
            vote_accuracy = self.vote_score_cache.get(weights)
        except KeyError:
            vote_accuracy, _ = self._test_model(weights, validate=True)

            # store this in the cache
            self.vote_score_cache.add(weights,
                                      vote_accuracy)

        test_acc, eval_result = self._test_model(weights,
                                                 validate=False,
                                                 eval_config=eval_config)

        vote = vote_accuracy >= self.vote_accuracy

        proposed_weights = ProposedWeights(weights=weights,
                                           vote_accuracy=vote_accuracy,
                                           test_accuracy=test_acc,
                                           vote=vote,
                                           evaluation_results=eval_result,
                                           )
        return proposed_weights

    def _test_model(self, weights: Weights = None, validate=False,
                    eval_config: Optional[dict] = None
                    ) -> Tuple[float, dict]:
        raise NotImplementedError

    def _train_model(self):
        raise NotImplementedError
