<<<<<<< HEAD:convert_dtypes.py


import pandas as pd
data_path = "./data/X_train_Hi5_filled_nan.csv"
df = pd.read_csv(data_path)

from water_shortage.preprocess import save_one_hot_encoding_for_categorical_cols
save_dir = "./data"
save_one_hot_encoding_for_categorical_cols(df, save_dir)

=======

# convert data type
import pandas as pd

from water_shortage.preprocess import save_covert_dtypes_df

data_path = "./data/filled_nan/X_train_Hi5_filled_nan_0.csv"
# df = pd.read_csv(data_path, nrows=100, index_col=0)
df = pd.read_csv(data_path)
save_filename = "./data/X_train_Hi5_filled_nan_converted_dtype.csv"
save_covert_dtypes_df(df, save_filename)


>>>>>>> 0d00f0f071af86a0f6e2142449ec2cf4b1cdde0c:examples/convert_dtypes.py
