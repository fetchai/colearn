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
