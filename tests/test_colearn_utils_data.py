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
import numpy as np
import pytest

from colearn.utils.data import split_list_into_fractions


def test_split_list_into_fractions():
    original_list = list(range(100))
    fractions = [1 / 5] * 5
    split_data = split_list_into_fractions(original_list, fractions)
    assert split_data == [list(range(0, 20)),
                          list(range(20, 40)),
                          list(range(40, 60)),
                          list(range(60, 80)),
                          list(range(80, 100))]


def test_split_list_into_fractions_ndarray():
    original_list = np.array(range(100))
    fractions = [1 / 5] * 5
    split_data = split_list_into_fractions(original_list, fractions)
    ground_truth = [np.array(range(0, 20)),
                    np.array(range(20, 40)),
                    np.array(range(40, 60)),
                    np.array(range(60, 80)),
                    np.array(range(80, 100))]

    for sd, gt in zip(split_data, ground_truth):
        assert np.all(sd == gt)


def test_split_list_into_fractions_minsize():
    original_list = list(range(100))
    fractions = [1 / 5] * 5
    with pytest.raises(Exception):
        split_list_into_fractions(original_list, fractions, min_part_size=30)
