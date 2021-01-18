import random


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
