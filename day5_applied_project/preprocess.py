"""
Preprocessing for the German Credit dataset: encode categorical
variables, scale numeric features. This is the unglamorous 80% of
real ML work that clean synthetic datasets never require.
"""

import pandas as pd
import numpy as np


def preprocess(input_path="../data/german_credit_raw.csv",
                output_path="../data/german_credit_processed.csv"):
    df = pd.read_csv(input_path)

    categorical_cols = df.select_dtypes(include="object").columns
    numeric_cols = df.select_dtypes(include=np.number).columns.drop("target")

    # One-hot encode categoricals -- turns e.g. "purpose=car" into
    # separate 0/1 columns, since our models only understand numbers
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # Standardize numeric columns: (x - mean) / std
    # Crucial for gradient descent -- unscaled features (e.g. credit_amount
    # in thousands vs age in tens) make the loss surface badly distorted,
    # causing slow/unstable convergence.
    for col in numeric_cols:
        mean, std = df_encoded[col].mean(), df_encoded[col].std()
        df_encoded[col] = (df_encoded[col] - mean) / std

    # Convert any remaining bool columns (from get_dummies) to int
    bool_cols = df_encoded.select_dtypes(include="bool").columns
    df_encoded[bool_cols] = df_encoded[bool_cols].astype(int)

    df_encoded.to_csv(output_path, index=False)
    print(f"Processed shape: {df_encoded.shape}")
    print(f"Class balance:\n{df_encoded['target'].value_counts(normalize=True)}")
    return df_encoded


if __name__ == "__main__":
    preprocess()