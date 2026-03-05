import pandas as pd 
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime

def data_sampling(n: int, random_state, file_path, save_path):
    '''
    Parameters
    ----------
    n : int
        No of Samples.
    file_path : Path
        Input Data file path.
    save_path : Path
        Sampled data save path.

    Returns
    -------
    Sampled_Data CSV.

    '''
    # Normalize file path to use forward slashes
    file_path = file_path.replace("\\", "/")
    save_path = save_path.replace("\\", "/")
    
 
    df = pd.read_csv(file_path)
    
    # Sample n rows from the DataFrame
    df_sample = df.sample(n, random_state = random_state)
    
    # Extract the original filename without extension
    original_filename = os.path.splitext(os.path.basename(file_path))[0]
    print(original_filename)
    # Construct the new filename with "_sampled" appended
    new_filename = f"{original_filename}_sampled.csv"
    
    # Construct the full save path
    save_file_path = os.path.join(save_path, new_filename).replace("\\", "/")
    print(save_file_path)
    
    # Save the sampled data to the new file
    return df_sample.to_csv(save_file_path, index=False)

def fill_none_with_mean(df):
    """
    Fills columns containing NoneType (NaN) values with the column mean for numeric columns.

    Parameters:
        df (pd.DataFrame): Input pandas DataFrame.

    Returns:
        pd.DataFrame: DataFrame with NoneType values filled with column mean for numeric columns.
    """
    # Replace NoneType (NaN) with the mean for numeric columns
    df = df.apply(lambda col: col.fillna(col.mean()) if col.dtype in ['float64', 'int64'] else col)
    return df

def convert_piezo_station_update_date_to_datetime(df: pd.DataFrame) -> pd.DataFrame:
    datetime_cols = ['piezo_station_update_date']
    col = datetime_cols[0]
    date_str_list = [date_str.replace("CEST", "CET") for date_str in df[col]] # contains two format
    df[col] = date_str_list
    df[col] = pd.to_datetime(df[col], format="%a %b %d %H:%M:%S CET %Y")
    return df

def convert_date_cols_to_datetime(df):
    date_cols = ['piezo_measurement_date', 'meteo_date', 'hydro_observation_date_elab']
    for col in date_cols:
        df[col] = pd.to_datetime(df[col])
    return df

def drop_high_null_columns(df, threshold=0.7):
    """
    Removes columns in a DataFrame that have more than a specified percentage of null values.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - threshold (float): The proportion of null values allowed (default is 0.7, i.e., 70%).

    Returns:
    - pd.DataFrame: A new DataFrame with high-null columns removed.
    """
    # Calculate the proportion of null values for each column
    null_proportion = df.isnull().mean()
    
    # Identify columns with null proportion above the threshold
    high_null_cols = null_proportion[null_proportion > threshold].index
    
    # Drop these columns
    return df.drop(columns=high_null_cols)


#df = pd.read_csv("D:/HiParis_Hackathon/X_train_Hi5.csv")
# df = df.drop(columns=['row_index', 'piezo_continuity_code', 'piezo_continuity_name', 'hydro_method_code', 'hydro_method_label'])
#df = pd.read_csv("D:/HiParis_Hackathon/X_train_Hi5_ver1f.csv")
#print(df['piezo_station_update_date'].isnull().sum())
#df = convert_piezo_station_update_date_to_datetime(df)
#df = pd.read_csv("D:/HiParis_Hackathon/X_train_Hi5_timeseries1.csv")

#df = convert_date_cols_to_datetime(df)
#df = pd.read_csv("D:/HiParis_Hackathon/X_train_Hi5_timeseries2.csv")
#df = drop_high_null_columns(df)
#df.to_csv("D:/HiParis_Hackathon/X_train_Hi5_drop7.csv", index=False)



def clean_column_by_dtype(df, column_name, expected_dtype):
    """
    Cleans a column in a DataFrame by checking each row's data type.
    Replaces mismatched data types with the column mean (for numeric) 
    or the most frequent value (for text).

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - column_name (str): The column to clean.
    - expected_dtype (type): The expected data type of the column (e.g., int, float, str).

    Returns:
    - pd.DataFrame: The modified DataFrame with the column cleaned.
    """
    # Ensure the column exists in the DataFrame
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' does not exist in the DataFrame.")
    
    # Check the expected data type
    if expected_dtype not in [int, float, str]:
        raise ValueError("Expected dtype must be int, float, or str.")
    
    # Handle numeric columns
    if expected_dtype in [int, float]:
        # Calculate the mean of the column, ignoring non-numeric and NaN values
        column_mean = pd.to_numeric(df[column_name], errors='coerce').mean()
        # Replace invalid data types with the mean
        df[column_name] = df[column_name].apply(
            lambda x: column_mean if not isinstance(x, (int, float)) or pd.isna(x) else x
        )
    
    # Handle text columns
    elif expected_dtype == str:
        # Calculate the most frequent value in the column
        most_frequent = df[column_name].dropna().mode()[0]
        # Replace invalid data types with the most frequent value
        df[column_name] = df[column_name].apply(
            lambda x: most_frequent if not isinstance(x, str) or pd.isna(x) else x
        )
    
    return df



df = pd.read_csv("D:/HiParis_Hackathon/X_train_Hi5_PreTokken.csv")

df = clean_column_by_dtype(df, 'piezo_station_department_code', float)
df = clean_column_by_dtype(df, 'piezo_station_commune_code_insee', float)
df = clean_column_by_dtype(df, 'piezo_measure_nature_code', str)
df = clean_column_by_dtype(df, 'prelev_commune_code_insee_2', float)
df = clean_column_by_dtype(df, 'insee_med_living_level', float)
df = clean_column_by_dtype(df, 'insee_%_ind', float)
df = clean_column_by_dtype(df, 'insee_%_const', float)

df.to_csv("D:/HiParis_Hackathon/X_train_Hi5_PreTokken_anomly.csv", index=False)





