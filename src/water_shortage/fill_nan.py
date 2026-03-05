cols_to_replace = ['meteo_time_wind_max_3s',
'meteo_wind_max_3s',
'meteo_wind_direction_max_avg',
'meteo_time_wind_avg',
'meteo_wind_avg',
'meteo_time_wind_max',
'meteo_wind_direction_max_inst',
'meteo_wind_speed_avg_10m',
'meteo_wind_max',
'meteo_humidity_duration_above_80%',
'meteo_humidity_duration_below_40%',
'meteo_time_humidity_max',
'meteo_time_humidity_min',
'meteo_humidity_max',
'meteo_humidity_min',
'meteo_humidity_avg',
'meteo__pressure_saturation_avg']

from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

#data_path = "D:/HiParis_Hackathon/X_train_Hi5_drop7.csv"
#df = pd.read_csv(data_path)

#fillna_dict = {col: 0 for col in cols_to_replace}

#df = df.fillna(fillna_dict)
#df.to_csv("D:/HiParis_Hackathon/X_train_Hi5_fillzero.csv")

#df = pd.read_csv("D:/HiParis_Hackathon/X_train_Hi5_fillzero.csv")

#df = df.sort_values(by='piezo_station_update_date', ascending=True)
#df.to_csv("D:/HiParis_Hackathon/X_train_Hi5_sort_time.csv", index = False)

#df = pd.read_csv("D:/HiParis_Hackathon/X_train_Hi5_sort_time.csv")
#dfs = df.sample(500)
#dfs.to_csv("D:/HiParis_Hackathon/X_train_Hi5_sort_time_sample.csv", index = False)
#df = df.drop(columns = ['Unnamed: 0.1', 'Unnamed: 0'])

# Fill null values based on column type
'''
for col in df.columns:
    if df[col].isnull().any():  # Check if column has null values
        if pd.api.types.is_numeric_dtype(df[col]):  # Numerical column
            mean_value = df[col].mean()
            df[col].fillna(mean_value, inplace=True)
        elif pd.api.types.is_object_dtype(df[col]):  # Text/categorical column
            most_frequent_value = df[col].mode()[0]
            df[col].fillna(most_frequent_value, inplace=True)
'''

def convert_target_data_to_ordered_label(df):
    col = "piezo_groundwater_level_category"
    category = ["Very Low", "Low", "Average", "High", "Very High"]
    le = LabelEncoder()
    le = le.fit(category)
    le.classes_ = pd.Series(category)
    Y = le.transform(df[col])
    df[col] = Y
    return df



def clean_numeric_columns(df, threshold=0.1):
    """
    Cleans numeric columns in a DataFrame by replacing non-numeric values
    (if they are less than a given percentage of the rows) with the column mean.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - threshold (float): The maximum proportion of non-numeric values allowed (default is 0.1, i.e., 10%).

    Returns:
    - pd.DataFrame: A DataFrame with numeric columns cleaned.
    """
    df_cleaned = df.copy()
    
    for col in df_cleaned.select_dtypes(include=[np.number]).columns:
        # Identify rows that contain non-numeric data
        non_numeric_mask = ~df_cleaned[col].apply(lambda x: isinstance(x, (int, float, np.number)))
        non_numeric_count = non_numeric_mask.sum()
        
        # Check if non-numeric values are less than the threshold percentage
        if non_numeric_count > 0 and (non_numeric_count / len(df_cleaned[col])) <= threshold:
            # Calculate the mean of the numeric values
            col_mean = df_cleaned[col].apply(pd.to_numeric, errors='coerce').mean()
            
            # Replace non-numeric values with the mean
            df_cleaned.loc[non_numeric_mask, col] = col_mean
    
    return df_cleaned

df = pd.read_csv("D:/HiParis_Hackathon/X_train_Hi5_nulltest.csv")
df = convert_target_data_to_ordered_label(df)

df = clean_numeric_columns(df)

df.to_csv("D:/HiParis_Hackathon/X_train_Hi5_PreTokken.csv", index = False)

df = pd.read_csv("D:/HiParis_Hackathon/X_train_Hi5_PreTokken.csv")

dfs = df.sample(500)

dfs.to_csv("D:/HiParis_Hackathon/X_train_Hi5_PreTokken_sample.csv", index = False)

