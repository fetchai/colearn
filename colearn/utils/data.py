import random
from typing import List, Union
import numpy as np


def shuffle_data(data_lists, seed=None):
    for i in range(len(data_lists) - 1):
        assert len(data_lists[i]) == len(data_lists[i + 1])

    shuffled_lists = [dl.copy() for dl in data_lists]

    index_shuf = list(range(len(data_lists[0])))

    if seed is not None:
        random.seed(seed)
    random.shuffle(index_shuf)

    it = 0
    for i in index_shuf:
        for j, dl in enumerate(data_lists):
            shuffled_lists[j][it] = dl[i]

        it += 1

    return shuffled_lists


# Splits data to len(chunks) parts and each chunks[i] defines percentage of
# len(data)
def split_by_chunksizes(data_lists, chunks):
    data_len = len(data_lists[0])

    splitted_lists = [list() for _ in range(len(data_lists))]

    it = 0
    for chunk in chunks:
        step = int(chunk * data_len)

        # Crop size of part if part is bigger than remaining samples
        if it + step >= data_len:
            step = data_len - it - 1

        # part has to include at least one sample
        if step < 1 and chunk > 0:
            step = 1

        for j, dl in enumerate(data_lists):
            splitted_lists[j].append(dl[it: it + step])

        it += step

    return splitted_lists


def split_list_into_fractions(input_list: Union[List, np.ndarray],
                              fractions_list: List,
                              min_part_size=1):
    split_list = []
    start_index = 0
    n_indices = len(input_list)
    for frac in fractions_list:
        end_index = start_index + int(n_indices * frac)
        if end_index >= n_indices:
            end_index = n_indices

        if end_index - start_index < min_part_size:
            raise Exception("Insufficient data in this part")

        split_list.append(input_list[start_index: end_index])
        start_index = end_index

    return split_list
