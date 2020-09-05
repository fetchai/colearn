import os
import pickle
import tempfile
from pathlib import Path

from colearn.basic_learner import LearnerData

from examples.config import ColearnConfig, ModelConfig
from examples.utils.data import shuffle_data, split_by_chunksizes
from examples.xray_utils.data import estimate_cases, train_generator


normal_fl = "normal.pickle"
pneu_fl = "pneumonia.pickle"


def split_to_folders(
    config: ColearnConfig, data_dir,
    output_folder=Path(tempfile.gettempdir()) / "xray"
):
    if not os.path.isdir(data_dir):
        raise Exception("Data dir does not exist: " + str(data_dir))

    cases = list(Path(data_dir).rglob("*.jp*"))

    if len(cases) == 0:
        raise Exception("No data foud in path: " + str(data_dir))

    normal_data = []
    pneumonia_data = []

    for case in cases:
        if "NORMAL" in str(case):
            normal_data.append(case)
        elif "PNEUMONIA" in str(case):
            pneumonia_data.append(case)
        else:
            print(case, " - has invalid category")

    [normal_data] = shuffle_data([normal_data], config.shuffle_seed)
    [pneumonia_data] = shuffle_data([pneumonia_data], config.shuffle_seed)

    [normal_data_list] = split_by_chunksizes([normal_data], config.data_split)
    [pneumonia_data_list] = split_by_chunksizes([pneumonia_data],
                                                config.data_split)

    local_output_dir = Path(output_folder)

    dir_names = []
    for i in range(config.n_learners):

        dir_name = local_output_dir / str(i)
        os.makedirs(str(dir_name), exist_ok=True)

        pickle.dump(normal_data_list[i], open(dir_name / normal_fl, "wb"))
        pickle.dump(pneumonia_data_list[i], open(dir_name / pneu_fl, "wb"))

        dir_names.append(dir_name)
    print(dir_names)
    return dir_names


def prepare_single_client(config: ModelConfig, data_dir):
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
        train_normal, train_pneumonia, config.batch_size, config.width,
        config.height,
        augmentation=config.train_augment,
        seed=config.generator_seed
    )
    data.val_gen = train_generator(
        train_normal, train_pneumonia, config.batch_size, config.width,
        config.height,
        augmentation=config.train_augment,
        seed=config.generator_seed
    )

    data.train_data_size = estimate_cases(len(train_normal),
                                          len(train_pneumonia))

    data.test_batch_size = config.batch_size

    data.test_gen = train_generator(
        test_normal,
        test_pneumonia,
        config.batch_size,
        config.width,
        config.height,
        augmentation=config.train_augment,
        seed=config.generator_seed,
        shuffle=False,
    )

    data.test_data_size = estimate_cases(len(test_normal), len(test_pneumonia))
    return data
