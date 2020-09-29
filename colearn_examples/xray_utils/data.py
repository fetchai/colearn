import imgaug.augmenters as iaa

import numpy as np

from colearn_examples.utils.data import normalize_image

# this line is a fix for np.version 1.18 making a change that imgaug hasn't
# tracked yet
if float(np.version.version[2:4]) == 18:
    # pylint: disable=W0212
    np.random.bit_generator = np.random._bit_generator


def estimate_cases(normal, pneumonia):
    if normal > pneumonia:
        return int(2 * normal)
    else:
        return int(2 * pneumonia)


# Augmentation sequence
seq = iaa.Sequential(
    [
        # iaa.Fliplr(0.5),  # horizontal flips
        iaa.Affine(rotate=(-15, 15)),  # rotation
        iaa.Multiply((0.7, 1.3)),
    ]
)  # random brightness


def train_generator(
    normal_data,
    pneumonia_data,
    batch_size,
    width,
    height,
    augmentation=True,
    shuffle=True,
    seed=None
):
    # Get total number of samples in the data
    n_normal = len(normal_data)
    n_pneumonia = len(pneumonia_data)

    # Define two numpy arrays for containing batch data and labels
    batch_data = np.zeros(
        (batch_size, width, height, 1), dtype=np.float32
    )
    batch_labels = np.zeros((batch_size, 1), dtype=np.float32)

    # Get a numpy array of all the indices of the input data
    indices_normal = np.arange(n_normal)
    indices_pneumonia = np.arange(n_pneumonia)

    if shuffle:
        if seed is not None:
            np.random.seed(seed)
        np.random.shuffle(indices_normal)
        np.random.shuffle(indices_pneumonia)

    it_normal = 0
    it_pneumonia = 0

    # Initialize a counter
    batch_counter = 0

    while True:
        if batch_counter % 2 == 0:
            img_data, img_label = normalize_image(
                normal_data[indices_normal[it_normal]], 0, width, height
            )
            it_normal += 1
        else:
            img_data, img_label = normalize_image(
                pneumonia_data[indices_pneumonia[it_pneumonia]],
                1,
                width,
                height,
            )
            it_pneumonia += 1

        if augmentation:
            img_data = seq.augment_image(img_data)

        batch_data[batch_counter] = img_data
        batch_labels[batch_counter] = img_label

        batch_counter += 1

        if it_normal >= n_normal:
            it_normal = 0

            if shuffle:
                if seed is not None:
                    np.random.seed(seed)
                np.random.shuffle(indices_normal)

        if it_pneumonia >= n_pneumonia:
            it_pneumonia = 0

            if shuffle:
                if seed is not None:
                    np.random.seed(seed)
                np.random.shuffle(indices_pneumonia)

        if batch_counter == batch_size:
            yield batch_data, batch_labels
            batch_counter = 0
