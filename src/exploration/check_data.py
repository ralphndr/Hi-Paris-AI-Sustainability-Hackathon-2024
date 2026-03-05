import pandas as pd


data_path = "./data/X_train_Hi5.csv"
df = pd.read_csv(data_path, nrows=100)

pd.set_option('display.max_columns', 500)
pd.set_option('display.max_columns', 100)


df.loc[0:10, df.columns[0:30]]
df.loc[0:10, df.columns[30:60]]
df.loc[0:10, df.columns[60:90]]
df.loc[0:10, df.columns[90:120]]
df.loc[0:10, df.columns[120:160]]

obj_cols = df.dtypes[df.dtypes == "object"]

df.loc[0:10, obj_cols.index]

col = df.columns[0]

for col in df.columns[0:10]:
    num_na = df[col].isna().sum()
    print("number of na", num_na)
    print(df[col].value_counts())




data_path = "./data/X_train_Hi5.csv"

num_na_dict = {}
output_dir = "./outputs/check_na/"

for i, col in enumerate(df.columns):
    print(i)
    df_one_col = pd.read_csv(data_path, usecols=[col])
    print("len df_one_col : ", len(df_one_col))
    num_na = df_one_col[col].isna().sum()
    print(f"---{col}---")
    print("number of na", num_na)
    value_count_series = df_one_col[col].value_counts(sort=True)
    num_na_dict[col] = num_na
    savedir = output_dir + f"/{col}.csv"
    value_count_series.to_csv(savedir)


import pickle
num_na_dict_savename = "./outputs/check_na/num_na_dict.pkl"

with open('data/temp/my_dict.pkl', 'wb') as f:
    pickle.dump(d, f)

with open('data/temp/my_dict.pkl', 'rb') as f:
    d_load = pickle.load(f)