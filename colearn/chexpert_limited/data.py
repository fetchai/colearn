import os
import pickle
import shutil
from pathlib import Path

import pandas as pd

from colearn.data import shuffle_data, split_by_chunksizes
from colearn.model import LearnerData
from colearn.xray_utils.data import estimate_cases, train_generator


def split_chexpert(chexpert_path):
    df = pd.read_csv(Path(chexpert_path) / "train.csv", sep=",")
    df = df.loc[df["Frontal/Lateral"].isin(["Frontal"])]

    normal_data = list(df.loc[df["No Finding"].isin(["1"])]["Path"])
    normal_data = [Path(chexpert_path).parent / dat for dat in normal_data]

    pneumonia_data = list(df.loc[df["Lung Opacity"].isin(["1"])]["Path"])
    pneumonia_data = [Path(chexpert_path).parent / dat for dat in pneumonia_data]

    return normal_data, pneumonia_data


def split_to_folders(
    config, data_dir, output_folder=Path(os.getcwd()) / "chexpert_limited"
):
    normal_data, pneumonia_data = split_chexpert(data_dir)

    [normal_data] = shuffle_data([normal_data], config.shuffle_seed)
    [pneumonia_data] = shuffle_data([pneumonia_data], config.shuffle_seed)

    [normal_data_list] = split_by_chunksizes([normal_data], config.data_split)
    [pneumonia_data_list] = split_by_chunksizes([pneumonia_data], config.data_split)

    dir_names = []
    for i in range(config.n_learners):

        dir_name = output_folder / str(i)
        dir_names.append(dir_name)
        if os.path.isdir(str(dir_name)):
            shutil.rmtree(str(dir_name))
            os.makedirs(str(dir_name))
        else:
            os.makedirs(str(dir_name))

        pickle.dump(normal_data_list[i], open(dir_name / "normal.pickle", "wb"))
        pickle.dump(pneumonia_data_list[i], open(dir_name / "pneumonia.pickle", "wb"))

    return dir_names


def prepare_single_client(config, data_dir):
    data = LearnerData()

    normal_data = pickle.load(open(Path(data_dir) / "normal.pickle", "rb"))
    pneumonia_data = pickle.load(open(Path(data_dir) / "pneumonia.pickle", "rb"))

    [[train_normal, test_normal]] = split_by_chunksizes(
        [normal_data], [config.train_ratio, config.test_ratio]
    )
    [[train_pneumonia, test_pneumonia]] = split_by_chunksizes(
        [pneumonia_data], [config.train_ratio, config.test_ratio]
    )

    data.train_batch_size = config.batch_size

    data.train_gen = train_generator(
        train_normal, train_pneumonia, config.batch_size, config, config.train_augment
    )
    data.val_gen = train_generator(
        train_normal, train_pneumonia, config.batch_size, config, config.train_augment
    )

    data.train_data_size = estimate_cases(len(train_normal), len(train_pneumonia))

    data.test_batch_size = config.batch_size

    data.test_gen = train_generator(
        test_normal,
        test_pneumonia,
        config.batch_size,
        config,
        config.test_augment,
        shuffle=False,
    )

    data.test_data_size = estimate_cases(len(test_normal), len(test_pneumonia))
    return data
