# ------------------------------------------------------------------------------
#
#   Copyright 2021 Fetch.AI Limited
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
# ------------------------------------------------------------------------------
import pickle
from math import ceil
from typing import Iterator

from colearn.ml_interface import Weights
from colearn_grpc.proto.generated.interface_pb2 import WeightsPart

WEIGHTS_PART_SIZE_BYTES = 4 * 10 ** 6


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
        # On the client side we can't necessarily unpickle the Weights object because the relevant libraries might not
        # be importable. But we need to return a Weights object to match the MLI. So we wrap the pickled Weights in
        # another Weights object.
        return Weights(weights=weights_bytes)


def weights_to_iterator(input_weights: Weights, encode=True) -> Iterator[WeightsPart]:
    enc_weights: bytes  # this is a pickled Weights object
    if encode:
        enc_weights = encode_weights(input_weights)
    else:
        # On the client side input_weights is a wrapper around a pickled Weights object - see note
        # in iterator_to_weights
        enc_weights = input_weights.weights

    part_size = WEIGHTS_PART_SIZE_BYTES
    total_size = len(enc_weights)
    total_parts = ceil(total_size / part_size)

    for i in range(total_parts):
        w = WeightsPart()
        w.byte_index = i * part_size
        w.total_bytes = total_size
        w.weights = enc_weights[i * part_size: (i + 1) * part_size]
        yield w
