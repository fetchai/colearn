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
import pickle
from math import ceil
from typing import Iterator

from prometheus_client import Summary

from colearn.ml_interface import Weights, TrainingSummary, DiffPrivBudget
from colearn_grpc.proto.generated.interface_pb2 import WeightsPart, TrainingSummary as TrainingSummaryPb

# the default limit for gRPC messages is 4MB, so the part size is set a bit smaller
WEIGHTS_PART_SIZE_BYTES = 4 * 10 ** 6

# Prometheus metrics
_time_reconstruct_weights = Summary(
    "grpc_utils_reconstruct_weights",
    "This metric measures the time it takes to convert an iterator of WeightsPart to Weights")
_time_deconstruct_weights = Summary(
    "grpc_utils_deconstruct_weights",
    "This metric measures the time it takes to convert Weights to an iterator of WeightsPart")


def encode_weights(w: Weights) -> bytes:
    return pickle.dumps(w)


def decode_weights(w: bytes) -> Weights:
    return pickle.loads(w)


@_time_reconstruct_weights.time()
def iterator_to_weights(request_iterator: Iterator[WeightsPart], decode=True) -> Weights:
    first_weights_part = next(request_iterator)
    full_weights = bytearray(first_weights_part.total_bytes)
    bytes_sum = 0
    training_summary = None
    end_index = first_weights_part.byte_index + len(first_weights_part.weights)
    full_weights[first_weights_part.byte_index: end_index] = first_weights_part.weights
    bytes_sum += len(first_weights_part.weights)

    for weights_part in request_iterator:
        end_index = weights_part.byte_index + len(weights_part.weights)
        full_weights[weights_part.byte_index: end_index] = weights_part.weights
        bytes_sum += len(weights_part.weights)
        if weights_part.HasField('training_summary'):
            training_summary = transform_training_summary_from_pb(weights_part.training_summary)

    weights_bytes = bytes(full_weights)
    if decode:
        return Weights(weights=decode_weights(weights_bytes), training_summary=training_summary)
    else:
        # On the client side we can't necessarily unpickle the Weights object because the relevant libraries might not
        # be importable. But we need to return a Weights object to match the MLI. So we wrap the pickled Weights in
        # another Weights object.
        return Weights(weights=weights_bytes, training_summary=training_summary)


@_time_reconstruct_weights.time()
async def iterator_to_weights_async(request_iterator, decode=True) -> Weights:
    first_time = True
    training_summary = None
    async for weights_part in request_iterator:
        if first_time:
            first_weights_part = weights_part
            full_weights = bytearray(first_weights_part.total_bytes)
            bytes_sum = 0

            end_index = first_weights_part.byte_index + len(first_weights_part.weights)
            full_weights[first_weights_part.byte_index: end_index] = first_weights_part.weights
            bytes_sum += len(first_weights_part.weights)

            if weights_part.HasField('training_summary'):
                training_summary = transform_training_summary_from_pb(weights_part.training_summary)

            first_time = False
        else:
            end_index = weights_part.byte_index + len(weights_part.weights)
            full_weights[weights_part.byte_index: end_index] = weights_part.weights
            bytes_sum += len(weights_part.weights)

    weights_bytes = bytes(full_weights)
    if decode:
        return Weights(weights=decode_weights(weights_bytes), training_summary=training_summary)
    else:
        # On the client side we can't necessarily unpickle the Weights object because the relevant libraries might not
        # be importable. But we need to return a Weights object to match the MLI. So we wrap the pickled Weights in
        # another Weights object.
        return Weights(weights=weights_bytes, training_summary=training_summary)


def transform_training_summary_to_pb(ts: TrainingSummary) -> TrainingSummaryPb:
    tsn = TrainingSummaryPb()
    for field in ["dp_budget"]:
        target = getattr(tsn, field)
        source = getattr(ts, field)
        for entry in source.__fields_set__:
            setattr(target, entry, getattr(source, entry))
    return tsn


def transform_training_summary_from_pb(ts: TrainingSummaryPb) -> TrainingSummary:
    tsn = TrainingSummary()
    parts = [
        ("dp_budget", DiffPrivBudget, {
            "target_epsilon": 0,
            "target_delta": 0,
            "consumed_epsilon": 0,
            "consumed_delta": 0
        })
    ]
    for (field, cls, default_params) in parts:
        source = getattr(ts, field)
        obj = cls(**default_params)
        for entry in source.DESCRIPTOR.fields_by_name.keys():
            setattr(obj, entry, getattr(source, entry))
        setattr(tsn, field, obj)
    return tsn


@_time_deconstruct_weights.time()
def weights_to_iterator(input_weights: Weights, encode=True) -> Iterator[WeightsPart]:
    enc_weights: bytes  # this is a pickled Weights object
    if encode:
        enc_weights = encode_weights(input_weights.weights)
    else:
        # On the client side input_weights is a wrapper around a pickled Weights object - see note
        # in iterator_to_weights
        enc_weights = input_weights.weights

    part_size = WEIGHTS_PART_SIZE_BYTES
    total_size = len(enc_weights)
    total_parts = ceil(total_size / part_size)

    start_idx = 0
    if input_weights.training_summary:
        start_idx = 1
        w = WeightsPart()
        w.byte_index = 0
        w.total_bytes = total_size
        w.weights = enc_weights[0:part_size]
        w.training_summary.CopyFrom(transform_training_summary_to_pb(input_weights.training_summary))
        yield w

    for i in range(start_idx, total_parts):
        w = WeightsPart()
        w.byte_index = i * part_size
        w.total_bytes = total_size
        w.weights = enc_weights[i * part_size: (i + 1) * part_size]
        yield w
