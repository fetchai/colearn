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

import asyncio
from colearn.ml_interface import Weights
from colearn_grpc.proto.generated.interface_pb2 import WeightsPart

from colearn_grpc.utils import encode_weights, decode_weights, \
    iterator_to_weights, iterator_to_weights_async, weights_to_iterator, WEIGHTS_PART_SIZE_BYTES


def asyncio_run_synchronously(coroutine_to_run):
    return asyncio.get_event_loop().run_until_complete(coroutine_to_run)


def test_encode_decode():
    test_weights = "weights"
    weights = Weights(weights=test_weights)

    encoded = encode_weights(weights)
    decoded = decode_weights(encoded)

    assert decoded == weights
    assert weights.weights == test_weights

    encoded2 = encode_weights(decoded)
    assert encoded == encoded2


def test_in_order_iterator_to_weights():

    test_weights = b"abc"
    parts = [WeightsPart(
        weights=test_weights[i:i + 1],
        byte_index=i,
        total_bytes=len(test_weights))
        for i in range(len(test_weights))]

    result = iterator_to_weights(request_iterator=iter(parts), decode=False)

    assert result.weights == test_weights


# An alternate way to reconstruct weights is async with an async generator
def test_in_order_iterator_to_weights_async():

    # Create async generator
    async def weights_async_gen(parts):
        for i in parts:
            yield i

    test_weights = b"abc"
    parts = [WeightsPart(
        weights=test_weights[i:i + 1],
        byte_index=i,
        total_bytes=len(test_weights))
        for i in range(len(test_weights))]

    # Easy way to call async coroutine from sync context
    result = asyncio_run_synchronously(iterator_to_weights_async(request_iterator=weights_async_gen(parts), decode=False))

    assert result.weights == test_weights


def test_all_order_iterator_to_weights():

    test_weights = b"abcd"
    parts = [WeightsPart(
        weights=test_weights[i:i + 1],
        byte_index=i,
        total_bytes=len(test_weights))
        for i in range(len(test_weights))]

    for _ in range(len(test_weights)):
        result = iterator_to_weights(request_iterator=iter(parts), decode=False)
        assert result.weights == test_weights
        parts = parts[1:] + parts[:1]


def test_weights_to_iterator_small():
    part_a = bytes(b"a")
    test_weights = part_a
    weights = Weights(weights=test_weights)

    iterator = weights_to_iterator(input_weights=weights, encode=False)

    val = next(iterator, b"")
    assert isinstance(val, WeightsPart)
    assert val.total_bytes == 1
    assert val.byte_index == 0
    assert bytes(val.weights) == part_a

    val = next(iterator, b"")
    assert val == b""


def test_weights_to_iterator_small_limit():
    part_a = bytes(b"a" * WEIGHTS_PART_SIZE_BYTES)
    test_weights = part_a
    weights = Weights(weights=test_weights)

    iterator = weights_to_iterator(input_weights=weights, encode=False)

    val = next(iterator, b"")
    assert isinstance(val, WeightsPart)
    assert val.total_bytes == WEIGHTS_PART_SIZE_BYTES
    assert val.byte_index == 0
    assert bytes(val.weights) == part_a

    val = next(iterator, b"")
    assert val == b""


def test_weights_to_iterator_small_limit_plus_one():
    part_a = bytes(b"a" * WEIGHTS_PART_SIZE_BYTES)
    part_b = bytes(b"b")
    test_weights = part_a + part_b
    weights = Weights(weights=test_weights)

    iterator = weights_to_iterator(input_weights=weights, encode=False)

    val = next(iterator, b"")
    assert isinstance(val, WeightsPart)
    assert val.total_bytes == WEIGHTS_PART_SIZE_BYTES + 1
    assert val.byte_index == 0
    assert bytes(val.weights) == part_a

    val = next(iterator, b"")
    assert isinstance(val, WeightsPart)
    assert val.total_bytes == WEIGHTS_PART_SIZE_BYTES + 1
    assert val.byte_index == WEIGHTS_PART_SIZE_BYTES
    assert bytes(val.weights) == part_b

    val = next(iterator, b"")
    assert val == b""


def test_weights_to_iterator():
    part_a = bytes(b"a" * WEIGHTS_PART_SIZE_BYTES)
    part_b = bytes(b"b" * (WEIGHTS_PART_SIZE_BYTES - 2))
    test_weights = part_a + part_b
    weights = Weights(weights=test_weights)

    iterator = weights_to_iterator(input_weights=weights, encode=False)

    val = next(iterator, b"")
    assert isinstance(val, WeightsPart)
    assert val.total_bytes == 2 * WEIGHTS_PART_SIZE_BYTES - 2
    assert val.byte_index == 0
    assert bytes(val.weights) == part_a

    val = next(iterator, b"")
    assert isinstance(val, WeightsPart)
    assert val.total_bytes == 2 * WEIGHTS_PART_SIZE_BYTES - 2
    assert val.byte_index == WEIGHTS_PART_SIZE_BYTES
    assert bytes(val.weights) == part_b

    val = next(iterator, b"")
    assert val == b""


def test_iterator_and_back():

    part_a = bytes(b"a" * WEIGHTS_PART_SIZE_BYTES)
    part_b = bytes(b"b" * (WEIGHTS_PART_SIZE_BYTES - 2))
    test_weights = part_a + part_b
    weights = Weights(weights=test_weights)

    iterator = weights_to_iterator(input_weights=weights, encode=False)

    result = iterator_to_weights(request_iterator=iterator, decode=False)

    assert result == weights
