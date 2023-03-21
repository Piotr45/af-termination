import argparse
import os
import sys
import math
from typing import List, Tuple

import numpy as np
import pandas as pd
import wfdb
from numpy.typing import ArrayLike


def parse_arguments(argv: List[str]) -> argparse.Namespace:
    arg_parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    arg_parser.add_argument(
        "--ds-dir",
        type=str,
        action="store",
        required=True,
        help="Directory with ECG data from physionet",
    )

    arg_parser.add_argument(
        "--sample-length",
        type=int,
        action="store",
        required=True,
        help="The length of a sample's window",
    )

    arg_parser.add_argument(
        "--sample-freq",
        type=int,
        action="store",
        required=True,
        help="Number of sampled points from a window",
    )

    arg_parser.add_argument(
        "--download", action="store_true", help="Whether to download database or not"
    )

    return arg_parser.parse_args(argv)


def _list_all_data(dir: str) -> List[str]:
    """List all samples within a dataset
    
    Args:
        dir (str): directory containing raw data from physionet
    
    eturns:
        List[str]: list of datasets (training, test-a, test-b)
    """
    return os.listdir(dir)


def _list_records(dir: str) -> List[str]:
    """List all record names for a given person
    
    Args:
        dir (str): directory containing records
    
    Returns:
        List[str]: record names
    """
    all_record_files = os.listdir(dir)
    record_names = list(
        set([file.split(".")[0] for file in all_record_files])
    )
    return record_names


def _read_record_and_annotation(
    data_dir: str, record_name: str
) -> Tuple[wfdb.Record, wfdb.Annotation]:
    """Read record's data and annotation of a given record
    
    Args:
        data_dir (str): directory containing records
        record_name (str): record name
    
    Returns:
        Tuple[wfdb.Record, wfdb.Annotation]: tuple of record's data and annotation
    """
    if data_dir[-1] != "/":
        data_dir += "/"

    record = wfdb.rdrecord(f"{data_dir}{record_name}")
    annotation = wfdb.rdann(f"{data_dir}{record_name}", extension="qrs")
    label = record_name[0]

    return (record, annotation, label)


def _prepare_record_data(
    record: wfdb.Record, annotation: wfdb.Annotation, label: str, sample_length = 64
) -> Tuple[ArrayLike, ArrayLike]:
    """
    """
    ann_points = annotation.sample

    record_length = record.p_signal.shape[0]
    num_ecgs = record.p_signal.shape[1]
    
    data_size = math.ceil(record_length / sample_length)

    ecg_data = np.zeros((data_size, sample_length, num_ecgs), dtype=np.float32)
    labels = np.zeros((data_size, 2), dtype=np.int8)

    dummy_encoder = {
        'n': np.array([0, 0], dtype=np.int8),
        's': np.array([0, 1], dtype=np.int8),
        't': np.array([1, 1], dtype=np.int8),
        'a': np.array([1, 0], dtype=np.int8),
        'b': np.array([1, 0], dtype=np.int8)
    }

    for i in range(data_size):
        idx = i*sample_length

        ecg_data[i] = record.p_signal[idx:idx+sample_length]
        labels[i] = list(map(lambda x: dummy_encoder[x], label))[0]
        
    return (ecg_data, labels)


def main() -> None:
    # Parse arguments
    args = parse_arguments(sys.argv[1:])

    ds_dir = args.ds_dir
    sample_length = args.sample_length
    sample_freq = args.sample_freq
    download = args.download

    # standardize the datasource path
    if ds_dir[-1] != "/":
        ds_dir += "/"

    # create a datasource directory if doesn't exist
    if not os.path.isdir(ds_dir):
        os.mkdir(ds_dir)

    # download physionet data
    if download:
        wfdb.dl_database("aftdb", ds_dir)

    dataset_names = _list_all_data(ds_dir)
    dataset_paths = [
        os.path.join(ds_dir, dataset_dir)
        for dataset_dir in dataset_names
    ]

    all_records = [
        (dataset_dir, sorted(_list_records(dataset_dir)))
        for dataset_dir in dataset_paths
    ]

    records_and_annotations = [
        (dataset_dir, _read_record_and_annotation(dataset_dir, record_name))
        for dataset_dir, record_names in all_records
        for record_name in record_names
    ]

    prepared_data = [
        (dataset_path, _prepare_record_data(rec, ann, label))
        for dataset_path, (rec, ann, label) in records_and_annotations
    ]
    
    print((prepared_data[0]))
    return


if __name__ == "__main__":
    main()
