"""
Data Preparation script for the "Visit with Us" Wellness Tourism Package
prediction project.

- Loads the raw dataset
- Fixes data-entry inconsistencies
- Drops identifier columns that carry no predictive signal
- Splits the data into train / test sets (stratified on the target)
- Saves the processed train/test CSVs
"""
import os
import pandas as pd
from sklearn.model_selection import train_test_split

RAW_DATA_PATH = "tourism_project/data/tourism.csv"
OUT_DIR = "tourism_project/data"
TARGET = "ProdTaken"


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Drop columns that are not useful for modelling
    drop_cols = [c for c in ["Unnamed: 0", "CustomerID"] if c in df.columns]
    df = df.drop(columns=drop_cols)

    # Fix inconsistent category labels
    if "Gender" in df.columns:
        df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})
    if "MaritalStatus" in df.columns:
        df["MaritalStatus"] = df["MaritalStatus"].replace({"Unmarried": "Single"})

    # Drop exact duplicate rows, if any
    df = df.drop_duplicates()
    return df


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv(RAW_DATA_PATH)
    df = clean_data(df)

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    train_df.to_csv(f"{OUT_DIR}/train.csv", index=False)
    test_df.to_csv(f"{OUT_DIR}/test.csv", index=False)

    print(f"Cleaned data shape: {df.shape}")
    print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")
    print(f"Target distribution (train):\n{y_train.value_counts(normalize=True)}")


if __name__ == "__main__":
    main()
