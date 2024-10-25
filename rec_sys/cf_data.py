# Artur Andrzejak, October 2024
# Data utilities for collaborative filtering

import numpy as np
import polars as pl
import pandas as pd
from scipy.sparse import csr_matrix

# Routines to load MovieLens data and convert it to utility matrix

def get_size_in_mb(obj):
    if isinstance(obj, pl.DataFrame):
        return obj.estimated_size(unit='mb')
    elif isinstance(obj, np.ndarray):
        return obj.nbytes / (1024 * 1024)
    elif isinstance(obj, pd.DataFrame):
        return obj.memory_usage(deep=True).sum() / (1024 * 1024)
    else:
        return None


def print_df_stats(prefix_msg, df):
    print(f"{prefix_msg}, df shape: {df.shape}, size in MB: {get_size_in_mb(df)} ")


def read_movielens_file_and_convert_to_um(file_path, max_rows=None):
    """ Read a local MovieLens file and return the utility matrix as a pivot table where
        columns=userid, rows=movieid (relabeled) and contents are movie rating (NaN = no rating given).
        Original file has columns: userId,movieId,rating,timestamp
        See https://files.grouplens.org/datasets/movielens/ml-25m-README.html
    """
    print (f"\n### Start reading data from '{file_path}'")
    df = pl.read_csv(file_path,
                     has_header=True, columns=[0, 1, 2],
                     new_columns=['userID', 'movieID', 'rating'],
                     n_rows=max_rows,
                     schema_overrides={'userID': pl.UInt32, 'movieID': pl.UInt32, 'rating': pl.Float32()})
    print_df_stats(f"Loaded data from '{file_path}'", df)

    # Convert from long to wide format and then drop column 'movieID'
    print ("Pivoting the data")
    util_mat_pl = df.pivot(index = 'movieID', on = 'userID', values = 'rating').drop('movieID')
    print_df_stats(f"Utility matrix", util_mat_pl)

    util_mat_np = np.array(util_mat_pl).astype(np.float32)
    print_df_stats(f"Final utility matrix (numpy array as np.float32)", util_mat_np)
    return util_mat_np

def read_movielens_file_and_convert_to_sparse_um(file_path, max_rows = None):
    """ Read a local MovieLens file and return the utility matrix as a pivot table where
        columns=userid, rows=movieid (relabeled) and contents are movie rating (NaN = no rating given).
        Original file has columns: userId,movieId,rating,timestamp
        See https://files.grouplens.org/datasets/movielens/ml-25m-README.html
    """

    df = pl.read_csv(file_path,
                     has_header=True, columns=[0, 1, 2],
                     new_columns=['userID', 'movieID', 'rating'],
                     n_rows=max_rows,
                     schema_overrides={'userID': pl.UInt32, 'movieID': pl.UInt32, 'rating': pl.Float32()})

    print_df_stats(f"Loaded sparse data from '{file_path}'", df)
    return csr_matrix(df)

def load_and_unzip_dataset(url, path_to_save, unzip_path, force_download=False):
    import requests
    import zipfile
    import io, os

    if not force_download and os.path.exists(unzip_path):
        print(f"Dir '{unzip_path}' already exists, skipping download")
        return

    print(f"Downloading '{url}' to '{path_to_save}'")
    r = requests.get(url)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall(path_to_save)
    print(f"Loaded and unzipped '{url}' to '{unzip_path}'")


def get_um_by_name(config, dataset_name):
    if dataset_name == "movielens":
        # Load (part of) the MovieLens 25M data
        # See https://grouplens.org/datasets/movielens/25m/
        load_and_unzip_dataset(config.dowload_url, config.download_dir, config.unzipped_dir)
        return read_movielens_file_and_convert_to_um(
            config.file_path,
            max_rows=config.max_rows)
    elif dataset_name == "lecture_1":
        um_lecture = [[    1., np.nan,     3., np.nan, np.nan,     5.],
                      [np.nan, np.nan,     5.,     4., np.nan, np.nan],
                      [    2.,     4., np.nan,     1.,     2., np.nan],
                      [np.nan,     2.,     4., np.nan,     5., np.nan],
                      [np.nan, np.nan,     4.,     3.,     4.,     2.],
                      [    1., np.nan,     3., np.nan,     3., np.nan]]
        return np.asarray(um_lecture)
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")

def get_sparse_um_by_name(config, dataset_name):
    if dataset_name == "movielens":
        load_and_unzip_dataset(config.dowload_url, config.download_dir, config.unzipped_dir)
        return read_movielens_file_and_convert_to_sparse_um(
            config.file_path,
            max_rows=config.max_rows)
    elif dataset_name == "sparse_test":
        um_sparse = csr_matrix(
                   [[    1., np.nan,     3.,     0., np.nan,     5.],
                   [np.nan,     0.,     5.,     4.,     0.,     0.],
                   [    2.,     4., np.nan,     1.,     2., np.nan],
                   [    0.,     2.,     4., np.nan,     5.,     0.],
                   [    0.,     0.,     4.,     3.,     4.,     2.],
                   [    1., np.nan,     3., np.nan,     3., np.nan]])
        return um_sparse


if __name__ == '__main__':
    from cf_config import config

    um_movielens = get_um_by_name(config, "movielens")
    um_lecture = get_um_by_name(config, "lecture_1")
