"""
Shows how max_depth controls the bias-variance tradeoff for trees --
too shallow underfits, too deep overfits (memorizes training data).
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from decision_tree import DecisionTreeScratch


def accuracy(preds, y):
    return np.mean(preds == y)


def main():
    X, y = make_classification(
        n_samples=300, n_features=2, n_redundant=0,
        n_informative=2, n_clusters_per_class=1, random_state=7
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    depths = [1, 2, 3, 5, 8, 12, 20]
    train_accs, test_accs = [], []

    for d in depths:
        model = DecisionTreeScratch(max_depth=d)
        model.fit(X_train, y_train)
        train_accs.append(accuracy(model.predict(X_train), y_train))
        test_accs.append(accuracy(model.predict(X_test), y_test))
        print(f"depth={d:2d} | train_acc={train_accs[-1]:.3f} | test_acc={test_accs[-1]:.3f}")

    plt.figure(figsize=(7, 5))
    plt.plot(depths, train_accs, marker="o", label="Train accuracy")
    plt.plot(depths, test_accs, marker="o", label="Test accuracy")
    plt.xlabel("max_depth")
    plt.ylabel("Accuracy")
    plt.title("Bias-Variance Tradeoff: Tree Depth vs Accuracy")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig("../plots/day4_depth_vs_accuracy.png", dpi=120)
    print("Plot saved.")


if __name__ == "__main__":
    main()