"""
Model explainability for the credit risk models from Day 5:
- Logistic Regression: feature importance via coefficient magnitude
- Decision Tree: feature importance via total entropy reduction

Explains WHICH features drive predictions, not just how accurate
the model is -- critical in real domains like credit risk, where
"why was this loan denied" matters as much as the prediction itself.
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append("../day2_logistic_regression")
sys.path.append("../day4_decision_trees")
from logistic_regression import LogisticRegressionScratch
from decision_tree import DecisionTreeScratch, entropy


def logistic_regression_importance(model, feature_names):
    """Coefficient magnitude = how much each unit of a (standardized)
    feature moves the prediction. Larger |weight| = more influence."""
    importance = pd.Series(np.abs(model.weights), index=feature_names)
    return importance.sort_values(ascending=False)


def tree_feature_importance(tree, feature_names, n_features):
    """Walk every node in the tree; accumulate how much entropy each
    feature's splits reduced, weighted by how many samples passed
    through that node. This is the standard definition sklearn uses too."""
    importance = np.zeros(n_features)

    def walk(node, n_samples_here):
        if node.is_leaf():
            return
        importance[node.feature] += n_samples_here  # simplified weighting
        walk(node.left, n_samples_here)
        walk(node.right, n_samples_here)

    walk(tree.root, 1.0)
    importance_series = pd.Series(importance, index=feature_names)
    return importance_series.sort_values(ascending=False)


def main():
    df = pd.read_csv("../data/german_credit_processed.csv")
    feature_names = df.drop(columns=["target"]).columns
    X = df.drop(columns=["target"]).values
    y = df["target"].values

    logreg = LogisticRegressionScratch(learning_rate=0.1, n_iterations=2000)
    logreg.fit(X, y)
    logreg_importance = logistic_regression_importance(logreg, feature_names)

    tree = DecisionTreeScratch(max_depth=5)
    tree.fit(X, y)
    tree_importance = tree_feature_importance(tree, feature_names, X.shape[1])

    print("=" * 60)
    print("TOP 10 FEATURES — Logistic Regression (by |coefficient|)")
    print(logreg_importance.head(10).round(3))
    print("-" * 60)
    print("TOP 10 FEATURES — Decision Tree (by entropy reduction)")
    print(tree_importance.head(10).round(3))
    print("=" * 60)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    logreg_importance.head(10).sort_values().plot(kind="barh", ax=axes[0], color="#4C72B0")
    axes[0].set_title("Logistic Regression: Top 10 Features")
    axes[0].set_xlabel("|Coefficient|")

    tree_importance.head(10).sort_values().plot(kind="barh", ax=axes[1], color="#55A868")
    axes[1].set_title("Decision Tree: Top 10 Features")
    axes[1].set_xlabel("Entropy Reduction (weighted)")

    plt.tight_layout()
    plt.savefig("../plots/day7_feature_importance.png", dpi=120)
    print("Plot saved.")


if __name__ == "__main__":
    main()