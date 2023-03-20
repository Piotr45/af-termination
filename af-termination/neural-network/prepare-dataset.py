import argparse
import os
import sys
from typing import List

import numpy as np
import pandas as pd
import wfdb


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

    all_data = _list_all_data(ds_dir)

    record = wfdb.rdrecord(f"dataset/learning-set/n01")
    annotation = wfdb.rdann(f"dataset/learning-set/n01", extension="qrs")
    print(record.p_signal.shape, annotation.sample)
    return


if __name__ == "__main__":
    main()
