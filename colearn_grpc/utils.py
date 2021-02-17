import pickle
from math import ceil
from typing import Iterator

from colearn.ml_interface import Weights
from colearn_grpc.proto.generated.interface_pb2 import WeightsPart


def encode_weights(w: Weights) -> bytes:
    return pickle.dumps(w)


def decode_weights(w: bytes) -> Weights:
    return pickle.loads(w)


def iterator_to_weights(request_iterator: Iterator[WeightsPart], decode=True) -> Weights:
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

    weights_bytes = bytes(full_weights)
    if decode:
        return decode_weights(weights_bytes)
    else:
        return Weights(weights=weights_bytes)


def weights_to_iterator(input_weights: Weights, encode=True) -> Iterator[WeightsPart]:
    enc_weights: bytes
    if encode:
        enc_weights = encode_weights(input_weights)
    else:
        enc_weights = input_weights.weights

    part_size = 4 * 10 ** 6
    total_size = len(enc_weights)
    total_parts = ceil(total_size / part_size)

    for i in range(total_parts):
        w = WeightsPart()
        w.byte_index = i * part_size
        w.total_bytes = total_size
        w.weights = enc_weights[i * part_size: (i + 1) * part_size]
        yield w
