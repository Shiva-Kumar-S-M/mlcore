import numpy as np
import pandas as pd
from preprocess import preprocess


def test_preprocessing_produces_no_missing_values():
    """Processed data must be fully numeric with no NaNs -- models
    can't handle missing values or text columns."""
    df = preprocess()
    assert df.isnull().sum().sum() == 0


def test_preprocessing_all_columns_numeric():
    """After one-hot encoding, every column should be numeric."""
    df = preprocess()
    assert all(np.issubdtype(dtype, np.number) for dtype in df.dtypes)


def test_target_is_binary():
    """Target column should only contain 0s and 1s (good/bad credit)."""
    df = preprocess()
    assert set(df["target"].unique()) <= {0, 1}


if __name__ == "__main__":
    test_preprocessing_produces_no_missing_values()
    test_preprocessing_all_columns_numeric()
    test_target_is_binary()
    print("All tests passed!")