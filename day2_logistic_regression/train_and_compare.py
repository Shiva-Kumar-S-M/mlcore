import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression as SklearnLogReg
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from logistic_regression import LogisticRegressionScratch


def main():
    X, y = make_classification(
        n_samples=300, n_features=2, n_redundant=0,
        n_informative=2, n_clusters_per_class=1, random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    ours = LogisticRegressionScratch(learning_rate=0.1, n_iterations=1000)
    ours.fit(X_train, y_train)
    our_preds = ours.predict(X_test)

    sk = SklearnLogReg()
    sk.fit(X_train, y_train)
    sk_preds = sk.predict(X_test)

    print("=" * 55)
    print(f"OUR weights: {ours.weights}, bias: {ours.bias:.3f}")
    print(f"SKLEARN weights: {sk.coef_[0]}, bias: {sk.intercept_[0]:.3f}")
    print("-" * 55)
    for name, preds in [("OURS", our_preds), ("SKLEARN", sk_preds)]:
        print(f"{name}: acc={accuracy_score(y_test, preds):.3f} "
              f"precision={precision_score(y_test, preds):.3f} "
              f"recall={recall_score(y_test, preds):.3f} "
              f"f1={f1_score(y_test, preds):.3f}")
    print("=" * 55)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(ours.loss_history)
    plt.xlabel("Iteration"); plt.ylabel("Cross-Entropy Loss")
    plt.title("Convergence"); plt.grid(alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap="coolwarm", edgecolors="k")
    x1 = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
    x2 = -(ours.weights[0] * x1 + ours.bias) / ours.weights[1]
    plt.plot(x1, x2, "g--", label="Decision boundary")
    plt.legend(); plt.title("Decision Boundary"); plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("../plots/day2_logistic_regression_results.png", dpi=120)
    print("Plot saved.")


if __name__ == "__main__":
    main()
    