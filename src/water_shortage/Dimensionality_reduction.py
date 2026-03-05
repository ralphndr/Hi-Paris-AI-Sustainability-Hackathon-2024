from sklearn.manifold import Isomap, TSNE
from sklearn.preprocessing import StandardScaler

import pandas as pd
import numpy as np

import dask.dataframe as dd

def clean_columns_with_mean_dask(file_path, col_indices, dtype=None):
    """
    Cleans specified columns in a dataset by replacing non-numeric values with the column's mean.

    Parameters:
    file_path (str): Path to the CSV file to process.
    col_indices (list): List of column indices to clean.
    dtype (dict): Optional dictionary specifying dtypes for columns.

    Returns:
    dask.dataframe.DataFrame: The cleaned Dask DataFrame.
    """
    # Load the dataset using Dask with explicit dtypes and error handling
    ddf = dd.read_csv(
        file_path,
        dtype=dtype,
        low_memory=False
    )

    # Ensure col_indices is a list
    if isinstance(col_indices, int):
        col_indices = [col_indices]

    # For each column, process and replace non-numeric values
    for col_index in col_indices:
        col_name = ddf.columns[col_index]
        
        # Convert column to numeric, ignoring errors, and compute mean of numeric values
        numeric_col = dd.to_numeric(ddf[col_name], errors='coerce')
        col_mean = numeric_col.mean().compute()

        # Replace non-numeric entries with the calculated mean
        ddf[col_name] = dd.to_numeric(ddf[col_name], errors='coerce').fillna(col_mean)

    return ddf


df = pd.read_csv("D:/HiParis_Hackathon/X_train_Hi5_Tokkenized.csv")

dtype = {'insee_%_agri': 'object'}  # Specify problematic column types


#df = clean_columns_with_mean_dask("D:/HiParis_Hackathon/X_train_Hi5_Tokkenized.csv", [20, 90], dtype=dtype)
#df.to_csv("D:/HiParis_Hackathon/X_train_Hi5_Tokkenized_2090.csv", index=False)

df = pd.read_csv("D:/HiParis_Hackathon/X_train_Hi5_Tokkenized_2090.csv")
df = df.sample(150000)
df.to_csv("D:/HiParis_Hackathon/X_train_Hi5_Tokkenized_sampled.csv", index = False)

col_names = [
             'piezo_station_update_date',
             'hydro_observation_date_elab',
             'piezo_station_longitude',
             'piezo_station_latitude',
             'meteo_date',
             'meteo_altitude',
             'meteo_longitude',
             'meteo_latitude',
             'hydro_longitude',
             'hydro_latitude',
             'prelev_longitude_0',
             'prelev_latitude_0',
             'prelev_longitude_1'
             'prelev_latitude_1',
             'prelev_longitude_2',
             'prelev_latitude_2',
             ]

