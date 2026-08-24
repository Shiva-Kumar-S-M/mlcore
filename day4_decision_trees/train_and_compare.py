import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier as SklearnTree
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, f1_score
from decision_tree import DecisionTreeScratch


def main():
    X, y = make_classification(
        n_samples=300, n_features=2, n_redundant=0,
        n_informative=2, n_clusters_per_class=1, n_classes=2,
        class_sep=1.2, random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    ours = DecisionTreeScratch(max_depth=5)
    ours.fit(X_train, y_train)
    our_preds = ours.predict(X_test)

    sk = SklearnTree(max_depth=5, criterion="entropy", random_state=42)
    sk.fit(X_train, y_train)
    sk_preds = sk.predict(X_test)

    print("=" * 50)
    print(f"OURS:    acc={accuracy_score(y_test, our_preds):.3f}  f1={f1_score(y_test, our_preds):.3f}")
    print(f"SKLEARN: acc={accuracy_score(y_test, sk_preds):.3f}  f1={f1_score(y_test, sk_preds):.3f}")
    print("=" * 50)

    # Visualize decision boundary -- trees produce BLOCKY, axis-aligned
    # boundaries, very different from logistic regression's straight line
    xx, yy = np.meshgrid(
        np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 200),
        np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 200)
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = ours.predict(grid).reshape(xx.shape)

    plt.figure(figsize=(6, 5))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap="coolwarm", edgecolors="k")
    plt.title("Decision Tree Boundary (blocky, axis-aligned splits)")
    plt.savefig("../plots/day4_decision_tree_boundary.png", dpi=120)
    print("Plot saved.")


if __name__ == "__main__":
    main()
    