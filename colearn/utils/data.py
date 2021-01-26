from typing import List, Union
import numpy as np


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
