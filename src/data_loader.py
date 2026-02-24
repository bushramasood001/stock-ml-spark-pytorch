"""
data_loader.py

Handles:
- Loading stock data
- Basic preprocessing
- Train-test split
"""

import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load stock dataset from CSV file.
    """
    df = pd.read_csv(file_path)
    return df


def basic_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform basic cleaning:
    - Drop missing values
    - Sort by date if exists
    """
    df = df.dropna()

    if "Date" in df.columns:
        df = df.sort_values("Date")

    return df


def split_data(df: pd.DataFrame, target_column: str, test_size: float = 0.2):
    """
    Split dataset into train and test sets.
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False
    )

    return X_train, X_test, y_train, y_test
