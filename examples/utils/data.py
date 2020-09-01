import random

import cv2

import numpy as np


class LearnerData:
    train_gen = ()
    val_gen = ()  # this is a copy of train gen
    test_gen = ()

    train_data_size = 0  # this includes augmentation
    test_data_size = 0  # this includes augmentation

    train_batch_size = 0
    test_batch_size = 0


def shuffle_data(data_lists, seed=None):
    for i in range(len(data_lists) - 1):
        assert len(data_lists[i]) == len(data_lists[i + 1])

    shuffled_lists = init_list_of_objects(len(data_lists))

    for i in range(len(data_lists)):
        shuffled_lists[i] = data_lists[i].copy()

    index_shuf = list(range(len(data_lists[0])))

    if seed is not None:
        random.seed(seed)
    random.shuffle(index_shuf)

    it = 0
    for i in index_shuf:
        for j in range(len(data_lists)):
            shuffled_lists[j][it] = data_lists[j][i]

        it += 1

    return shuffled_lists


def to_rgb_normalize_and_resize(img, width, height):
    img = cv2.imread(str(img))
    img = cv2.resize(img, (width, height))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.astype(np.float32) / 255.0
    img = np.reshape(img, (width, height, 1))

    return img


def normalize_image(filename, class_num, width, height):
    data = to_rgb_normalize_and_resize(filename, width, height)
    label = class_num

    return data, label


def init_list_of_objects(size):
    return [list() for _ in range(size)]


# Splits data to len(chunks) parts and each chunks[i] defines percentage of
# len(data)
def split_by_chunksizes(data_lists, chunks):
    data_len = len(data_lists[0])

    splitted_lists = init_list_of_objects(len(data_lists))

    it = 0
    for i in range(len(chunks)):
        step = int(chunks[i] * data_len)

        # Crop size of part if part is bigger than remaining samples
        if it + step >= data_len:
            step = data_len - it - 1

        # part has to include at least one sample
        if step < 1 and chunks[i] > 0:
            step = 1

        for j in range(len(data_lists)):
            splitted_lists[j].append(data_lists[j][it: it + step])

        it += step

    return splitted_lists


# Randomly splits data following normal distribution
def split_normal(parts, std_dev, seed=None):
    data_mean = 1 / parts

    # std_dev as percentage of mean
    std_dev = data_mean * std_dev

    if seed is not None:
        np.random.seed(seed)

    chunk_sizes = []
    for i in range(parts):
        chunk_sizes.append(np.round(np.random.normal(data_mean, std_dev)))
    chunk_sizes = chunk_sizes / sum(chunk_sizes)

    return chunk_sizes


# One learner gets certain percentage and others get the rest of the data
def split_lrg(parts, ratio):
    rest_ratio = (1.0 - ratio) / (parts - 1)

    chunk_sizes = [ratio]
    for i in range(parts - 1):
        chunk_sizes.append(rest_ratio)

    return chunk_sizes
