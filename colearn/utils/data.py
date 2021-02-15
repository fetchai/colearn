# ------------------------------------------------------------------------------
#
#   Copyright 2021 Fetch.AI Limited
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
# ------------------------------------------------------------------------------
import os
from pathlib import Path
from typing import List, Union

from google.cloud import storage
import numpy as np


def split_list_into_fractions(input_list: Union[List, np.ndarray],
                              fractions_list: List,
                              min_part_size=1):
    split_list = []
    start_index = 0
    n_indices = len(input_list)
    for frac in fractions_list:
        end_index = start_index + int(n_indices * frac)
        if end_index >= n_indices:
            end_index = n_indices

        if end_index - start_index < min_part_size:
            raise Exception("Insufficient data in this part")

        split_list.append(input_list[start_index: end_index])
        start_index = end_index

    return split_list


def get_data(data_dir: str, download_to='/tmp/data_download'):
    """
    Gets data, either from local filesystem or a google cloud bucket

    @param data_dir: path to data. If prefix is "gs://" data will be downloaded
    @param download_to: if data is downloaded where it will be downloaded to
    @return: Full path to either local data or to the downloaded data

    For more information on how to setup the google cloud bucket see the dev notes
    """

    if str(data_dir).startswith("gs://"):
        return _download_data_from_gcloud(data_dir, download_to)
    return data_dir


def _download_data_from_gcloud(cloud_data_dir, local_data_dir):
    """
    Downloads data from a gcloud bucket to local filesystem

    @param cloud_data_dir: path in google cloud bucket
    @param local_data_dir: path to where the data will be downloaded
    @return: Full path to downloaded data
    """
    bucket_name = cloud_data_dir.replace('gs://', '')
    bucket_name, prefix = bucket_name.split('/', 1)
    print(f"Downloading data from google cloud: Bucket {bucket_name}, prefix {prefix}")

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name=bucket_name)
    blobs = bucket.list_blobs(prefix=prefix)  # Get list of files

    local_full_path = Path(local_data_dir) / prefix
    os.makedirs(local_full_path, exist_ok=True)
    file_counter = 0
    for blob in blobs:
        filename = blob.name
        blob.download_to_filename(Path(local_data_dir) / filename)  # Download
        file_counter += 1

    if file_counter == 0:
        raise Exception("No data in folder: " + cloud_data_dir)

    return local_full_path
