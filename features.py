import pandas as pd


def create_features(customer_df: pd.DataFrame, transactions_df: pd.DataFrame, save_file: bool = False) -> pd.DataFrame:
    """Create features for the customers and transactions data."""
    
    # Features from customer data
    transaction_features_df = transactions_df.groupby("customer_id").agg(
        total_spent=pd.NamedAgg(column="amount", aggfunc="sum"),
        transaction_count=pd.NamedAgg(column="amount", aggfunc="count"),
        avg_transaction_amount=pd.NamedAgg(column="amount", aggfunc="mean"),
        first_transaction_date=pd.NamedAgg(column="timestamp", aggfunc="min"),
        last_transaction_date=pd.NamedAgg(column="timestamp", aggfunc="max"),
    ).reset_index()
    
    # Left join with customer_df
    customer_df["customer_length"] = customer_df["signup_date"].apply(lambda x: (pd.Timestamp.now() - x).days)
    final_df = customer_df[["customer_id", "country", "signup_date", "customer_length"]].merge(
        transaction_features_df, 
        on="customer_id", 
        how="left"
    )

    final_df["avg_monthly_spent"] = final_df.apply(lambda row: row["total_spent"] / (row["customer_length"] / 30) if row["customer_length"] > 0 else 0, axis=1)
    final_df["days_since_last_transaction"] = (pd.Timestamp.now() - final_df["last_transaction_date"]).dt.days
    final_df["first_transaction_before_signup"] = final_df.apply(lambda row: row["first_transaction_date"] < row["signup_date"], axis=1)

    # Handle customers with no transactions
    final_df["total_spent"] = final_df["total_spent"].fillna(0)
    final_df["transaction_count"] = final_df["transaction_count"].fillna(0)
    final_df["avg_transaction_amount"] = final_df["avg_transaction_amount"].fillna(0)
    final_df = final_df.drop(columns=["first_transaction_date", "last_transaction_date"])

    if save_file:
        final_df.to_csv("./data/customer_features_gold.csv", index=False)
    
    return final_df


if __name__ == "__main__":
    from etl import run_pipeline
    customer_clean, transactions_clean = run_pipeline(save_files=False)
    features_df = create_features(customer_clean, transactions_clean, save_file=True)
    print("Feature creation completed successfully!")
    print(f"Features shape: {features_df.shape}")
    print(features_df.head())