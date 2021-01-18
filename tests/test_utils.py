import pickle
import random

from colearn.utils.data import shuffle_data, split_by_chunksizes
from .pickle_tester import FileTester, get_pickle


def rand_list(size, size2):
    randomlist = []
    for _ in range(size):
        v = []
        for _ in range(size2):
            n = random.randint(1, 100)
            v.append(n)
        randomlist.append(v)
    return randomlist


def _gen():
    rnd = []
    for i in range(1, 6):
        rnd.append(rand_list(pow(10, i), pow(2, 9 - i)))

    shuffled = []
    chunks = []
    chunk_sizes = []
    for i in range(1, 2):
        chunk_sizes.append(pow(2, i))
    for a in rnd:
        shuffled.append(shuffle_data(a, 1000))
        chunks.append(split_by_chunksizes(a, chunk_sizes))

    data = {
        "lists": rnd,
        "shuffled": shuffled,
        "chunks": chunks,
    }

    with open("utils.pickle", "wb") as f:
        pickle.dump(data, f)


def test_shuffle_data():
    ft = FileTester()
    data = get_pickle("./tests/data/utils.pickle")

    shuffled = []
    for a in data["lists"]:
        shuffled.append(shuffle_data(a, 1000))

    assert ft.test_object_match(data["shuffled"], shuffled)


def test_split_by_chunksizes():
    ft = FileTester()
    data = get_pickle("./tests/data/utils.pickle")
    chunk_sizes = []
    chunks = []

    for i in range(1, 10):
        chunk_sizes.append(pow(2, i))

    for a in data["lists"]:
        chunks.append(split_by_chunksizes(a, chunk_sizes))

    assert ft.test_object_match(data["chunks"], chunks)
