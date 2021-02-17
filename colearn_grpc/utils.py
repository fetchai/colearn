import pickle
from typing import Iterable

from colearn.ml_interface import Weights
from colearn_grpc.proto.generated.interface_pb2 import WeightsPart


def encode_weights(w: Weights) -> bytes:
    return pickle.dumps(w)


def decode_weights(w) -> Weights:
    return pickle.loads(w)


def iterator_to_weights(request_iterator: Iterable[WeightsPart], decode=True) -> Weights:
    first_weights_part = next(request_iterator)
    full_weights = bytearray(first_weights_part.total_bytes)
    bytes_sum = 0

    end_index = first_weights_part.byte_index + len(first_weights_part.weights)
    full_weights[first_weights_part.byte_index: end_index] = first_weights_part.weights
    bytes_sum += len(first_weights_part.weights)
    print("bytes_sum", bytes_sum, first_weights_part.total_bytes)

    for weights_part in request_iterator:
        end_index = weights_part.byte_index + len(weights_part.weights)
        full_weights[weights_part.byte_index: end_index] = weights_part.weights
        bytes_sum += len(weights_part.weights)
        print("bytes_sum", bytes_sum, first_weights_part.total_bytes)

    full_weights = bytes(full_weights)
    if decode:
        return decode_weights(full_weights)
    else:
        return Weights(weights=full_weights)
