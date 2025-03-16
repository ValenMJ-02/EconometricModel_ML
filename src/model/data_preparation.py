import sys
import os
import pandas as pd
from sklearn.model_selection import train_test_split  # Importing train_test_split to split data

# Adjust system path to allow imports from parent directories
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

def load_data(file_path):
    """
    Loads data from a CSV file and formats column names.
    
    Args:
        file_path (str): Path to the CSV file.
    
    Returns:
        pd.DataFrame: Loaded dataset with formatted column names.
    """
    data = pd.read_csv(file_path)
    # Formatting column names: removing spaces, converting to lowercase, and replacing spaces with underscores
    data.columns = data.columns.str.strip().str.lower().str.replace(" ", "_")
    return data

def split_data(data):
    """
    Splits the dataset into training, validation, and test sets.
    
    The data is first split into:
    - 80% training + validation (`df_full_train`)
    - 20% test (`dataframe_test`)
    
    Then, the training + validation set is further split:
    - 75% training (`dataframe_train`)
    - 25% validation (`dataframe_validation_features`)
    
    Args:
        data (pd.DataFrame): The dataset to be split.
    
    Returns:
        tuple: (training set, validation set, test set, full training set before split)
    """
    # First split: separate test set (20% of data)
    df_full_train, dataframe_test = train_test_split(data, test_size=0.2, random_state=1)
    
    # Second split: separate training (60% of original data) and validation (20% of original data)
    dataframe_train, dataframe_validation_features = train_test_split(df_full_train, test_size=0.25, random_state=1)
    
    # Reset indices to maintain consistency
    dataframe_train = dataframe_train.reset_index(drop=True)
    dataframe_validation_features = dataframe_validation_features.reset_index(drop=True)
    dataframe_test = dataframe_test.reset_index(drop=True)
    
    return dataframe_train, dataframe_validation_features, dataframe_test, df_full_train

def prepare_data(dataframe):
    """
    Prepares the dataset by filling missing values.
    
    Specific replacements:
    - 'fireplacequ' missing values are replaced with 'NA'.
    - 'garageyrblt' missing values are replaced with 0.
    - For categorical columns, missing values are replaced with the most frequent value (mode), if applicable.
    - For numerical columns, missing values are replaced with the mean.
    
    Args:
        dataframe (pd.DataFrame): The dataset to be processed.
    
    Returns:
        pd.DataFrame: Processed dataset with no missing values.
    """
    # Handle specific column replacements
    if 'fireplacequ' in dataframe.columns:
        dataframe['fireplacequ'] = dataframe['fireplacequ'].fillna('NA')
    if 'garageyrblt' in dataframe.columns:
        dataframe['garageyrblt'] = dataframe['garageyrblt'].fillna(0)
    
    # Fill missing values in remaining columns
    for col in dataframe.columns:
        if col not in ['fireplacequ', 'garageyrblt']:
            if dataframe[col].dtype == 'object' or dataframe[col].dtype.name == 'category':
                if not dataframe[col].isnull().all():  # Ensure the column is not entirely NaN
                    dataframe[col] = dataframe[col].fillna(dataframe[col].mode()[0])
                else:
                    dataframe[col] = dataframe[col].fillna('NA')
            else:
                dataframe[col] = dataframe[col].fillna(dataframe[col].astype(float).mean())
    
    return dataframe
