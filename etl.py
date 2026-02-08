import pandas as pd
import numpy as np
from typing import Tuple


def extract_csv(csv_path: str, index_col: str = None) -> pd.DataFrame:
    """Extract data from a CSV file and return it as a DataFrame."""
    return pd.read_csv(csv_path, index_col=index_col)


def transform_customer_data(customer_df: pd.DataFrame, save_file: bool = False) -> pd.DataFrame:
    """Transform the customer data by cleaning the data."""
    customer_df["signup_date"] = pd.to_datetime(customer_df["signup_date"])
    
    # Unify the case of the country column and drop rows with missing values
    customer_df["country"] = customer_df["country"].str.upper()
    customer_df.dropna(axis=0, how="any", inplace=True)
    if save_file:
        customer_df.to_csv("./data/customers_silver.csv", index=False)
    return customer_df


def transform_transactions_data(transactions_df: pd.DataFrame, save_file: bool = False) -> pd.DataFrame:
    """Transform the transactions data by cleaning the data."""
    transactions_df["timestamp"] = pd.to_datetime(transactions_df["timestamp"])
    
    # Unify the case of the currency and category columns
    transactions_df["currency"] = transactions_df["currency"].str.upper()
    transactions_df["category"] = transactions_df["category"].str.lower()
    transactions_df["customer_id"] = transactions_df["customer_id"].astype("Int64")

    # Fill missing values in the category column with "unknown"
    transactions_df["category"] = transactions_df["category"].fillna("unknown")

    # Drop rows with missing values in critical columns
    transactions_df.dropna(subset=["customer_id", "amount", "currency", "timestamp"], inplace=True)
    if save_file:
        transactions_df.to_csv("./data/transactions_silver.csv", index=False)
    return transactions_df


def run_pipeline(save_files: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run the ETL pipeline and return cleaned customers and transactions tables."""
    # Extract
    customer_df = extract_csv("./data/customers.csv")
    transactions_df = extract_csv("./data/transactions.csv")

    # Transform (and optionally load to local files)
    customer_clean = transform_customer_data(customer_df, save_file=save_files)
    transactions_clean = transform_transactions_data(transactions_df, save_file=save_files)

    # Load
    return customer_clean, transactions_clean


if __name__ == "__main__":
    customer_clean, transactions_clean = run_pipeline(save_files=True)
    print("ETL pipeline completed successfully!")
    print(f"Cleaned customers shape: {customer_clean.shape}")
    print(customer_clean.head())
    print(f"Cleaned transactions shape: {transactions_clean.shape}")
    print(transactions_clean.head())