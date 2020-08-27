import numpy as np

from colearn.data import shuffle_data
from colearn.data import split_by_chunksizes
from colearn.model import LearnerData
from pathlib import Path

import librosa
import os, pickle, shutil


def wav2mfcc(file_path, augment=False, max_pad_len=11, freq_coeficients=40):
    wave, sr = librosa.load(file_path, mono=True, sr=8000, duration=1.024)

    if augment == True:
        bins_per_octave = 12
        pitch_pm = 4
        pitch_change = pitch_pm * 2 * (np.random.uniform())
        wave = librosa.effects.pitch_shift(
            wave, sr, n_steps=pitch_change, bins_per_octave=bins_per_octave
        )

        speed_change = np.random.uniform(low=0.9, high=1.1)
        wave = librosa.effects.time_stretch(wave, speed_change)
        wave = wave[:8192]

    duration = wave.shape[0] / sr
    speed_change = 2.0 * duration / 1.024
    wave = librosa.effects.time_stretch(wave, speed_change)
    wave = wave[:4096]

    wave = librosa.util.normalize(wave)
    mfcc = librosa.feature.mfcc(
        wave,
        sr=sr,
        n_mfcc=freq_coeficients,
        hop_length=int(0.048 * sr),
        n_fft=int(0.096 * sr),
    )
    mfcc -= np.mean(mfcc, axis=0) + 1e-8
    pad_width = max_pad_len - mfcc.shape[1]
    mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode="constant")

    return mfcc, duration, sr


def to_trainable(filename: Path, config, augmentation):
    mfcc, duration, sr = wav2mfcc(
        str(filename),
        max_pad_len=config.max_padded_len,
        freq_coeficients=config.freq_coeficients,
        augment=augmentation,
    )

    lbl = int(str(filename.name)[0])

    return mfcc, lbl


def split_to_folders(config, data_dir: Path, output_folder=Path(os.getcwd()) / "sound"):
    # Load sound files
    cases = list(data_dir.rglob("*.wav"))

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

    return dir_names


def prepare_single_client(config, data_dir):
    data = LearnerData()
    data.train_batch_size = config.batch_size

    cases = pickle.load(open(Path(data_dir) / "cases.pickle", "rb"))

    [[train_cases, test_cases]] = split_by_chunksizes(
        [cases], [config.train_ratio, config.test_ratio]
    )

    data.train_data_size = len(train_cases)

    data.train_gen = train_generator(
        train_cases, config.batch_size, config, config.train_augment
    )
    data.val_gen = train_generator(
        train_cases, config.batch_size, config, config.train_augment
    )

    data.test_data_size = len(test_cases)

    data.test_gen = train_generator(
        test_cases, config.batch_size, config, config.train_augment, shuffle=False
    )

    data.test_batch_size = config.batch_size
    return data


def train_generator(data, batch_size, config, augmentation=False, shuffle=True):
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

        snd_data, snd_label = to_trainable(data[indices[it]], config, augmentation)

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
