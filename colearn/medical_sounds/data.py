import os
import pickle
import random
import shutil
from pathlib import Path

import librosa

import numpy as np

import pandas as pd

from colearn.data import shuffle_data
from colearn.data import split_by_chunksizes
from colearn.model import LearnerData


# this line is a fix for np.version 1.18 making a change that imgaug hasn't tracked yet
if float(np.version.version[2:4]) == 18:
    np.random.bit_generator = np.random._bit_generator

diagnosis_dict = {
    "Healthy": 0,
    "URTI": 1,
    "Asthma": 2,
    "COPD": 3,
    "Bronchiectasis": 4,
    "LRTI": 5,
    "Pneumonia": 6,
    "Bronchiolitis": 7,
}


def get_labels_dict(data_dir):
    labels_file = Path(data_dir) / "patient_diagnosis.csv"
    labels = pd.read_csv(labels_file, header=None)

    # translate diagnosis name to integer
    diagnosis = [diagnosis_dict[i] for i in labels[1]]

    # create patient ID : diagnosis integer dict
    return dict(zip(labels[0], diagnosis))


def wav2mfcc(
    file_path,
    augment=False,
    max_pad_len=11,
    freq_coeficients=40,
    hop_length=1024,
    sr=8000,
):
    # scipy resampling makes load significantly faster
    wave, sr = librosa.load(file_path, mono=True, sr=sr, res_type="scipy")

    # Trim random part of input
    duration = hop_length * max_pad_len
    if len(wave) > duration:
        r_num = random.randint(0, len(wave) - duration)
        wave = wave[r_num : r_num + duration]

    # Normalize and get spectrogram
    wave = librosa.util.normalize(wave)
    mfcc = librosa.feature.mfcc(
        wave,
        sr=sr,
        n_mfcc=freq_coeficients,
        hop_length=hop_length,
        n_fft=int(0.096 * sr),
    )

    # Pad or trim to match max_padded_len
    if max_pad_len < mfcc.shape[1]:
        mfcc = mfcc[:, 0:max_pad_len]
    else:
        pad_width = max_pad_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode="constant")

    return mfcc, sr


def to_trainable(file_path, dict, config, augmentation):
    name = file_path.name
    # first 3 chars of filename are patient ID.
    # Patient ID is paired with diagnose by using dictionary
    label = dict[int(name[0:3])]
    data, _ = wav2mfcc(
        file_path,
        max_pad_len=config.max_padded_len,
        freq_coeficients=config.freq_coeficients,
        augment=augmentation,
    )

    return data, label


def split_to_folders(
    config, data_dir: Path, output_folder=Path(os.getcwd()) / "medical_sounds"
):
    # Update classes labels for statistics
    config.class_labels = [i for i in diagnosis_dict.keys()]

    all_learners = []

    for i in range(config.n_learners):
        all_learners.append(LearnerData())

    # Get dict of patient ID : diagnosis
    dict = get_labels_dict(data_dir)

    cases_dir = Path(data_dir) / "audio_and_txt_files"

    cases = list(cases_dir.glob("*.wav"))

    [cases] = shuffle_data([cases], config.shuffle_seed)

    [cases_lists] = split_by_chunksizes([cases], config.data_split)

    dir_names = []
    for i in range(config.n_learners):

        dir_name = output_folder / str(i)
        dir_names.append(dir_name)
        if os.path.isdir(str(dir_name)):
            shutil.rmtree(str(dir_name))
            os.makedirs(str(dir_name))
        else:
            os.makedirs(str(dir_name))

        pickle.dump(cases_lists[i], open(dir_name / "cases.pickle", "wb"))
        pickle.dump(dict, open(dir_name / "dict.pickle", "wb"))

    return dir_names


def prepare_single_client(config, data_dir):
    data = LearnerData()
    data.train_batch_size = config.batch_size

    cases = pickle.load(open(Path(data_dir) / "cases.pickle", "rb"))
    dict = pickle.load(open(Path(data_dir) / "dict.pickle", "rb"))

    [[train_cases, test_cases]] = split_by_chunksizes(
        [cases], [config.train_ratio, config.test_ratio]
    )

    data.train_data_size = len(train_cases)

    data.train_gen = train_generator(
        train_cases, dict, config.batch_size, config, config.train_augment
    )
    data.val_gen = train_generator(
        train_cases, dict, config.batch_size, config, config.train_augment
    )

    data.test_data_size = len(test_cases)

    data.test_gen = train_generator(
        test_cases, dict, config.batch_size, config, config.train_augment, shuffle=False
    )

    data.test_batch_size = config.batch_size
    return data


def train_generator(data, dict, batch_size, config, augmentation=False, shuffle=True):
    # Get total number of samples in the data
    n_data = len(data)

    # Define two numpy arrays for containing batch data and labels
    batch_data = np.zeros(
        (batch_size, config.freq_coeficients, config.max_padded_len, 1),
        dtype=np.float32,
    )
    batch_labels = np.zeros((batch_size, 1), dtype=np.uint8)

    # Get a numpy array of all the indices of the input data
    indices = np.arange(n_data)

    if shuffle:
        if config.generator_seed is not None:
            np.random.seed(config.generator_seed)

        np.random.shuffle(indices)
    it = 0

    # Initialize a counter
    batch_counter = 0

    while True:

        snd_data, snd_label = to_trainable(
            data[indices[it]], dict, config, augmentation
        )

        batch_data[batch_counter] = np.reshape(
            snd_data, (1, snd_data.shape[0], snd_data.shape[1], 1)
        )
        batch_labels[batch_counter] = snd_label

        it += 1
        batch_counter += 1

        if it >= n_data:
            it = 0

            if shuffle:
                if config.generator_seed is not None:
                    np.random.seed(config.generator_seed)
                np.random.shuffle(indices)

        if batch_counter == batch_size:
            yield batch_data, batch_labels
            batch_counter = 0
