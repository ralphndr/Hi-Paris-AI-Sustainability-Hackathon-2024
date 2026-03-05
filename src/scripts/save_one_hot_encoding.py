import pandas as pd
from water_shortage.preprocess import save_one_hot_encoding_for_categorical_cols

data_path = "./data/X_train_Hi5_filled_nan_converted_dtype.csv"
# df = pd.read_csv(data_path, nrows=100)
df = pd.read_csv(data_path)

save_dir = "./data"
save_one_hot_encoding_for_categorical_cols(df, save_dir)