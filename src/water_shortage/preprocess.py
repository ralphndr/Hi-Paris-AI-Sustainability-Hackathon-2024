import os

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import LabelEncoder


def calc_hot_encoder_for_col(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    In [93]: df.head()
    Out[93]:
    row_index piezo_station_department_code  ... insee_%_const  piezo_groundwater_level_category
    0          0                            01  ...          16.2                              High
    1          1                            01  ...          11.0                         Very High
    2          2                            01  ...           7.8                              High
    3          3                            01  ...           5.2                         Very High
    4          4                            01  ...           9.8                          Very Low

    [5 rows x 136 columns]

    In [92]: col
    Out[92]: 'piezo_station_department_code'

    In [94]: _df
    Out[94]:
            piezo_station_department_code_01  ...  piezo_station_department_code_95
    0                                    1.0  ...                               0.0
    1                                    1.0  ...                               0.0
    """
    X = df[col].values[:, np.newaxis]
    categories = df[col].unique().tolist()
    enc = preprocessing.OneHotEncoder()
    _X = enc.fit_transform(X)

    _X = _X.toarray()

    new_cols = [ col + "_" + name.split("_")[1] for name in enc.get_feature_names_out() ]
    _df = pd.DataFrame(_X, columns=new_cols, index=df.index)
    return _df

def save_one_hot_encoding_for_categorical_cols(df, save_dir: str) -> None:
    # categorical_cols = ['piezo_station_department_code',
    #                     'piezo_station_department_name', 'piezo_station_commune_code_insee',
    #                     'piezo_station_pe_label',
    #                     'piezo_station_bss_code', 'piezo_station_commune_name',
    #                     'piezo_station_bss_id', 'piezo_bss_code',
    #                     'piezo_obtention_mode', 'piezo_status', 'piezo_qualification',
    #                     'piezo_continuity_name', 'piezo_producer_name',
    #                     'piezo_measure_nature_code', 'piezo_measure_nature_name', 'meteo_name',
    #                     'hydro_station_code',
    #                     'hydro_status_label', 'hydro_method_label', 'hydro_qualification_label',
    #                     'hydro_hydro_quantity_elab', 'prelev_structure_code_0',
    #                     'prelev_usage_label_0', 'prelev_volume_obtention_mode_label_0',
    #                     'prelev_structure_code_1', 'prelev_usage_label_1',
    #                     'prelev_volume_obtention_mode_label_1', 'prelev_structure_code_2',
    #                     'prelev_usage_label_2', 'prelev_volume_obtention_mode_label_2']

    # columns whose unique values are <= 1000
    categorical_cols = ['piezo_station_department_code',
                        'piezo_station_department_name',
                        'piezo_obtention_mode',
                        'piezo_status',
                        'piezo_qualification',
                        'piezo_continuity_name',
                        'piezo_producer_name',
                        'piezo_measure_nature_code',
                        'piezo_measure_nature_name',
                        'meteo_name',
                        'hydro_status_label',
                        'hydro_method_label',
                        'hydro_qualification_label',
                        'hydro_hydro_quantity_elab',
                        'prelev_usage_label_0',
                        'prelev_volume_obtention_mode_label_0',
                        'prelev_usage_label_1',
                        'prelev_volume_obtention_mode_label_1',
                        'prelev_usage_label_2',
                        'prelev_volume_obtention_mode_label_2']

    for i, col in enumerate(categorical_cols):
        print(f"{i+1}/{len(categorical_cols)}")
        _df = calc_hot_encoder_for_col(df, col)
        save_basename = f"{col}_one_hot_encoding.csv"
        save_name = os.path.join(save_dir, save_basename)
        _df.to_csv(save_name)
        print(f"{save_name} has saved.")

    col = 'piezo_station_bdlisa_codes'
    _df = convert_piezo_station_bdlisa_codes_to_multi_one_hot_encoding(df)
    save_basename = f"{col}_one_hot_encoding.csv"
    save_name = os.path.join(save_dir, save_basename)
    _df.to_csv(save_name)

def save_covert_dtypes_df(df, save_filename: str) -> None:
    print("converting dtypes")
    df = convert_date_cols_to_datetime(df)
    df = convert_piezo_station_update_date_to_datetime(df)
    df = convert_target_data_to_ordered_label(df)
    df.to_csv(save_filename, index=False)
    print(f"{save_filename} has been saved.")

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

def convert_piezo_station_bdlisa_codes_to_multi_one_hot_encoding(df):
    multi_values_cols = ['piezo_station_bdlisa_codes']
    col = multi_values_cols[0]
    multi_col = [ eval(elem) for elem in df[col] ]
    mlb = MultiLabelBinarizer()
    _X = mlb.fit_transform(multi_col)
    new_cols = [ col + "_" + name for name in mlb.classes_ ]
    _df = pd.DataFrame(_X, columns=new_cols, index=df.index)
    return _df


def convert_target_data_to_ordered_label(df):
    col = "piezo_groundwater_level_category"
    category = ["Very Low", "Low", "Average", "High", "Very High"]
    le = LabelEncoder()
    le = le.fit(category)
    le.classes_ = pd.Series(category)
    Y = le.transform(df[col])
    df[col] = Y
    return df