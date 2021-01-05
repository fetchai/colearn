import os
import pickle
import tempfile
from pathlib import Path

from colearn.basic_learner import LearnerData

from colearn_examples.config import ModelConfig
from colearn_examples.utils.data import split_by_chunksizes, shuffle_data
from colearn_examples.xray_utils.data import estimate_cases, train_generator

normal_fl = "normal.pickle"
pneu_fl = "pneumonia.pickle"


def split_to_folders(
        data_dir,
        shuffle_seed,
        data_split,
        n_learners,
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

    [normal_data] = shuffle_data([normal_data], shuffle_seed)
    [pneumonia_data] = shuffle_data([pneumonia_data], shuffle_seed)

    [normal_data_list] = split_by_chunksizes([normal_data], data_split)
    [pneumonia_data_list] = split_by_chunksizes([pneumonia_data],
                                                data_split)

    local_output_dir = Path(output_folder)

    dir_names = []
    for i in range(n_learners):
        dir_name = local_output_dir / str(i)
        os.system(f"rm -r {dir_name}")
        os.makedirs(str(dir_name))
        dir_names.append(dir_name)

        # make symlinks to required files in directory
        for fl in normal_data_list[i] + pneumonia_data_list[i]:
            link_name = dir_name / os.path.basename(fl)
            print(link_name)
            os.symlink(fl, link_name)

        normal_data_list[i] = [os.path.basename(fl) for fl in normal_data_list[i]]
        pneumonia_data_list[i] = [os.path.basename(fl) for fl in pneumonia_data_list[i]]

        pickle.dump(normal_data_list[i], open(dir_name / normal_fl, "wb"))
        pickle.dump(pneumonia_data_list[i], open(dir_name / pneu_fl, "wb"))

    print(dir_names)
    return dir_names


def prepare_single_client(config: ModelConfig, data_dir, test_data_dir=None):
    normal_data = pickle.load(open(Path(data_dir) / "normal.pickle", "rb"))
    pneumonia_data = pickle.load(open(Path(data_dir) / "pneumonia.pickle", "rb"))

    normal_test = []
    pneumonia_test = []
    if test_data_dir is not None and test_data_dir != Path(""):
        normal_test = pickle.load(open(Path(test_data_dir) / "normal.pickle", "rb"))
        pneumonia_test = pickle.load(open(Path(test_data_dir) / "pneumonia.pickle", "rb"))
        normal_test = [Path(test_data_dir) / os.path.basename(fl) for fl in normal_test]
        pneumonia_test = [Path(test_data_dir) / os.path.basename(fl) for fl in pneumonia_test]

    normal_data = [Path(data_dir) / os.path.basename(fl) for fl in normal_data]
    pneumonia_data = [Path(data_dir) / os.path.basename(fl) for fl in pneumonia_data]

    if len(normal_test) == 0 or len(pneumonia_test) == 0:
        print("ChestXray no test data provided, splitting train set!")
        [[train_normal, test_normal]] = split_by_chunksizes(
            [normal_data], [config.train_ratio, config.test_ratio]
        )
        [[train_pneumonia, test_pneumonia]] = split_by_chunksizes(
            [pneumonia_data], [config.train_ratio, config.test_ratio]
        )
    else:
        print("ChestXray separate test set provided, number of normal examples ", len(normal_test),
              ", number of positive examples ", len(pneumonia_test))
        train_normal = normal_data
        train_pneumonia = pneumonia_data
        test_normal = normal_test
        test_pneumonia = pneumonia_test

    train_gen = train_generator(
        train_normal, train_pneumonia, config.batch_size, config.width,
        config.height,
        augmentation=config.train_augment,
        seed=config.generator_seed
    )
    val_gen = train_generator(
        train_normal, train_pneumonia, config.batch_size, config.width,
        config.height,
        augmentation=config.train_augment,
        seed=config.generator_seed
    )

    train_data_size = estimate_cases(len(train_normal),
                                     len(train_pneumonia))

    test_gen = train_generator(
        test_normal,
        test_pneumonia,
        config.batch_size,
        config.width,
        config.height,
        augmentation=config.train_augment,
        seed=config.generator_seed,
        shuffle=False,
    )

    test_data_size = estimate_cases(len(test_normal), len(test_pneumonia))

    return LearnerData(train_gen=train_gen,
                       val_gen=val_gen,
                       test_gen=test_gen,
                       train_data_size=train_data_size,
                       test_data_size=test_data_size,
                       train_batch_size=config.batch_size,
                       test_batch_size=config.batch_size)
