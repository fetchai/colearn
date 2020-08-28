import os
import pickle
import tempfile
from pathlib import Path
from google.cloud import storage

from colearn.config import Config
from examples.utils.data import shuffle_data, split_by_chunksizes
from examples.xray_utils.data import estimate_cases, train_generator
from colearn.model import LearnerData

normal_fl = "normal.pickle"
pneu_fl = "pneumonia.pickle"


def split_to_folders(
    config: Config, data_dir,
    output_folder=Path(tempfile.gettempdir()) / "kaggle_xray"
):
    if str(data_dir).startswith("gs://"):
        storage_client = storage.Client()
        dd_split = data_dir.split("/")
        bucket_name = dd_split[2]
        remote_data_dir = "/".join(dd_split[3:])
        cases = [
            x.name
            for x in storage_client.list_blobs(bucket_name,
                                               prefix=remote_data_dir)
            if x.name.endswith("jpg") or x.name.endswith("jpeg")
        ]

    else:
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

    if str(output_folder).startswith("gs://"):
        use_cloud = True
        local_output_dir = Path(tempfile.gettempdir()) / "kaggle_xray"
        outfol_split = output_folder.split("/")
        bucket_name = outfol_split[2]
        remote_output_dir = "/".join(outfol_split[3:])

        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
    else:
        use_cloud = False
        local_output_dir = Path(output_folder)
        bucket = None
        remote_output_dir = None

    dir_names = []
    for i in range(config.n_learners):

        dir_name = local_output_dir / str(i)
        os.makedirs(str(dir_name), exist_ok=True)

        pickle.dump(normal_data_list[i], open(dir_name / normal_fl, "wb"))
        pickle.dump(pneumonia_data_list[i], open(dir_name / pneu_fl, "wb"))

        if use_cloud:
            # upload files to gcloud
            remote_dir = os.path.join(remote_output_dir, str(i))
            for fl in [normal_fl, pneu_fl]:
                remote_image = os.path.join(remote_dir, fl)
                file_blob = bucket.blob(str(remote_image))
                file_blob.upload_from_filename(str(dir_name / fl))

            dir_names.append("gs://" + bucket.name + "/" + remote_dir)
        else:
            dir_names.append(dir_name)
    print(dir_names)
    return dir_names


def prepare_single_client(config, data_dir):
    data = LearnerData()

    if str(data_dir).startswith("gs://"):
        data_dir = str(data_dir)
        outfol_split = data_dir.split("/")
        bucket_name = outfol_split[2]
        remote_dir = "/".join(outfol_split[3:])

        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)

        data = []
        for fl in [normal_fl, pneu_fl]:
            with tempfile.TemporaryFile("rb+") as tmpfl:
                blob = bucket.blob(remote_dir + "/" + fl)
                blob.download_to_file(tmpfl)
                tmpfl.seek(0)
                data.append(pickle.load(tmpfl))

        remote_normal_data, remote_pneumonia_data = data

        local_data_dir = Path(tempfile.gettempdir()) / "kaggle_xray"
        os.makedirs(str(local_data_dir), exist_ok=True)

        normal_data = []
        for fl in remote_normal_data:
            blob = bucket.blob(fl)
            local_file = local_data_dir / os.path.basename(fl)
            blob.download_to_filename(local_file)
            normal_data.append(local_file)

        pneumonia_data = []
        for fl in remote_pneumonia_data:
            blob = bucket.blob(fl)
            local_file = local_data_dir / os.path.basename(fl)
            blob.download_to_filename(local_file)
            pneumonia_data.append(local_file)

    else:
        normal_data = pickle.load(open(Path(data_dir) / "normal.pickle", "rb"))
        pneumonia_data = pickle.load(open(Path(data_dir) / "pneumonia.pickle",
                                          "rb"))

    [[train_normal, test_normal]] = split_by_chunksizes(
        [normal_data], [config.train_ratio, config.test_ratio]
    )
    [[train_pneumonia, test_pneumonia]] = split_by_chunksizes(
        [pneumonia_data], [config.train_ratio, config.test_ratio]
    )

    data.train_batch_size = config.batch_size

    data.train_gen = train_generator(
        train_normal, train_pneumonia, config.batch_size, config,
        config.train_augment
    )
    data.val_gen = train_generator(
        train_normal, train_pneumonia, config.batch_size, config,
        config.train_augment
    )

    data.train_data_size = estimate_cases(len(train_normal),
                                          len(train_pneumonia))

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
