import pandas as pd
import pandas as pd
from nltk.tokenize import word_tokenize
import nltk

# Download NLTK tokenizer resources (only needs to be done once)
nltk.download('punkt')

def get_column_data_types(df):
    """
    Function to print column names along with their data types in a pandas DataFrame.

    Parameters:
        df (pd.DataFrame): Input pandas DataFrame.

    Returns:
        pd.DataFrame: A DataFrame containing column names and their corresponding data types.
    """
    data_types = df.dtypes.reset_index()
    data_types.columns = ["Column Name", "Data Type"]
    return data_types


def drop_non_numeric_columns(df):
    """
    Function to drop all columns from a DataFrame that are not of type int or float.

    Parameters:
        df (pd.DataFrame): Input pandas DataFrame.

    Returns:
        pd.DataFrame: A DataFrame with only numeric columns (int and float).
    """
    numeric_df = df.select_dtypes(include=['int64', 'float64'])
    return numeric_df


def check_missing_values(df):
    """
    Function to calculate the number of NaN values and percentage of NaN values for each column in a DataFrame.

    Parameters:
        df (pd.DataFrame): Input pandas DataFrame.

    Returns:
        pd.DataFrame: A DataFrame with columns for the number and percentage of NaN values.
    """
    nan_count = df.isnull().sum()
    total_rows = len(df)
    nan_percentage = (nan_count / total_rows) * 100
    
    # Create a summary DataFrame
    missing_values_summary = pd.DataFrame({
        'Column': nan_count.index,
        'NaN_Count': nan_count.values,
        'NaN_Percentage': nan_percentage.values
    }).sort_values(by='NaN_Percentage', ascending=False)
    
    return missing_values_summary


'''
def tokenize_text_columns(df):
    """
    Tokenizes all text columns in a DataFrame.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.

    Returns:
    - pd.DataFrame: A DataFrame with text columns tokenized.
    """
    df_tokenized = df.copy()
    
    # Iterate over all columns
    for col in df_tokenized.columns:
        # Check if the column's dtype is object or string
        if df_tokenized[col].dtype == 'object':
            # Tokenize the text values
            df_tokenized[col] = df_tokenized[col].apply(lambda x: word_tokenize(x) if isinstance(x, str) else x)
    
    return df_tokenized

df = pd.read_csv("D:/HiParis_Hackathon/X_train_Hi5_PreTokken.csv")

'''


from transformers import AutoTokenizer
import pandas as pd

def tokenize_and_save_csv(input_csv, output_csv, text_columns, model_name="bert-base-uncased", max_length=10):
    """
    Tokenizes specified text columns in a CSV file and saves the tokenized data back to a new CSV file.

    Parameters:
    - input_csv (str): Path to the input CSV file.
    - output_csv (str): Path to the output CSV file.
    - text_columns (list): List of text column names to tokenize.
    - model_name (str): Name of the pre-trained tokenizer model (default: "bert-base-uncased").
    - max_length (int): Maximum length for tokenized outputs (default: 10).

    Returns:
    - None: Saves the tokenized DataFrame to `output_csv`.
    """
    # Load the pre-trained tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Read the input CSV file
    df = pd.read_csv(input_csv)

    # Tokenize specified text columns
    for col in text_columns:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda x: tokenizer(
                    x,
                    padding="max_length",
                    truncation=True,
                    max_length=max_length,
                    return_tensors=None,
                )['input_ids'] if isinstance(x, str) else None
            )
        print(col)

    # Save the tokenized DataFrame to a new CSV file
    df.to_csv(output_csv, index=False)
    print(f"Tokenized data saved to: {output_csv}")

input_csv = "D:/HiParis_Hackathon/X_train_Hi5_PreTokken_anomly.csv"

output_csv = "D:/HiParis_Hackathon/X_train_Hi5_Tokkenized.csv"

text_columns = ['piezo_station_department_name',
                'piezo_station_pe_label',
                'piezo_station_bdlisa_codes',
                'piezo_station_bss_code',
                'piezo_station_commune_name',
                'piezo_station_bss_id',
                'piezo_bss_code',
                'piezo_obtention_mode',
                'piezo_status',
                'piezo_qualification',
                'piezo_producer_name',
                'piezo_measure_nature_code',
                'piezo_measure_nature_name',
                'meteo_name',
                'hydro_station_code',
                'hydro_status_label',
                'hydro_qualification_label',
                'hydro_hydro_quantity_elab',
                'prelev_structure_code_0',
                'prelev_usage_label_0',
                'prelev_volume_obtention_mode_label_0',
                'prelev_structure_code_1',
                'prelev_usage_label_1',
                'prelev_volume_obtention_mode_label_1',
                'prelev_structure_code_2',
                'prelev_usage_label_2',
                'prelev_volume_obtention_mode_label_2',
                ]

# Tokenize and save
tokenize_and_save_csv(input_csv, output_csv, text_columns=text_columns)