"""
Applies every model built this week (Logistic Regression, Decision
Tree) to the real credit risk dataset, and compares them side by
side against sklearn equivalents.
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression as SklearnLogReg
from sklearn.tree import DecisionTreeClassifier as SklearnTree
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score)

sys.path.append("../day2_logistic_regression")
sys.path.append("../day4_decision_trees")
from logistic_regression import LogisticRegressionScratch
from decision_tree import DecisionTreeScratch


def evaluate(name, y_true, y_pred, y_proba=None):
    result = {
        "model": name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
    }
    if y_proba is not None:
        result["roc_auc"] = roc_auc_score(y_true, y_proba)
    return result


def main():
    df = pd.read_csv("../data/german_credit_processed.csv")
    X = df.drop(columns=["target"]).values
    y = df["target"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    results = []

    our_logreg = LogisticRegressionScratch(learning_rate=0.1, n_iterations=2000)
    our_logreg.fit(X_train, y_train)
    our_preds = our_logreg.predict(X_test)
    our_proba = our_logreg.predict_proba(X_test)
    results.append(evaluate("Our LogisticRegression", y_test, our_preds, our_proba))

    sk_logreg = SklearnLogReg(max_iter=2000)
    sk_logreg.fit(X_train, y_train)
    sk_preds = sk_logreg.predict(X_test)
    sk_proba = sk_logreg.predict_proba(X_test)[:, 1]
    results.append(evaluate("Sklearn LogisticRegression", y_test, sk_preds, sk_proba))

    our_tree = DecisionTreeScratch(max_depth=5)
    our_tree.fit(X_train, y_train)
    tree_preds = our_tree.predict(X_test)
    results.append(evaluate("Our DecisionTree", y_test, tree_preds))

    sk_tree = SklearnTree(max_depth=5, criterion="entropy", random_state=42)
    sk_tree.fit(X_train, y_train)
    sk_tree_preds = sk_tree.predict(X_test)
    results.append(evaluate("Sklearn DecisionTree", y_test, sk_tree_preds))

    results_df = pd.DataFrame(results).set_index("model")
    print("=" * 70)
    print(results_df.round(3))
    print("=" * 70)
    results_df.to_csv("../data/day5_model_comparison.csv")

    plt.figure(figsize=(8, 5))
    results_df["f1"].plot(kind="barh", color=["#4C72B0", "#55A868", "#C44E52", "#8172B2"])
    plt.xlabel("F1 Score")
    plt.title("Model Comparison on Real Credit Risk Data")
    plt.tight_layout()
    plt.savefig("../plots/day5_model_comparison.png", dpi=120)
    print("\nPlot saved. Full results saved to data/day5_model_comparison.csv")


if __name__ == "__main__":
    main()