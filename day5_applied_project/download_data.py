"""
Downloads the UCI German Credit Data dataset -- a real, widely-used
benchmark for credit risk / loan default prediction.
"""

import pandas as pd

COLUMN_NAMES = [
    "checking_status", "duration", "credit_history", "purpose", "credit_amount",
    "savings_status", "employment", "installment_commitment", "personal_status",
    "other_parties", "residence_since", "property_magnitude", "age",
    "other_payment_plans", "housing", "existing_credits", "job",
    "num_dependents", "own_telephone", "foreign_worker", "target"
]

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"


def main():
    df = pd.read_csv(URL, sep=" ", header=None, names=COLUMN_NAMES)
    # Original encoding: 1 = good credit, 2 = bad credit -> convert to 0/1
    df["target"] = df["target"].map({1: 0, 2: 1})  # 1 = bad credit risk
    df.to_csv("../data/german_credit_raw.csv", index=False)
    print(f"Downloaded {len(df)} rows, {df.shape[1]} columns")
    print(f"Target distribution:\n{df['target'].value_counts()}")


if __name__ == "__main__":
    main()