"""
Proves our from-scratch linear regression is mathematically correct by
comparing it against sklearn's production implementation on the same data.

If both converge to similar weights/bias and similar R^2, we know our
gradient descent math is right -- not just "the code ran without errors."
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression as SklearnLR
from sklearn.model_selection import train_test_split
from linear_regression import LinearRegressionScratch


def generate_synthetic_data(n_samples=200, n_features=1, noise=8.0, seed=42):
    """Create data with a KNOWN true relationship so we can check if
    our model recovers it. true_w and true_b are what we EXPECT the
    model to learn (approximately)."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(-10, 10, size=(n_samples, n_features))
    true_w = np.array([3.5])
    true_b = -2.0
    y = X @ true_w + true_b + rng.normal(0, noise, size=n_samples)
    return X, y, true_w, true_b


def main():
    X, y, true_w, true_b = generate_synthetic_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # --- Our from-scratch model ---
    ours = LinearRegressionScratch(learning_rate=0.01, n_iterations=1000)
    ours.fit(X_train, y_train)
    our_r2 = ours.score(X_test, y_test)

    # --- sklearn's model, same data ---
    sk = SklearnLR()
    sk.fit(X_train, y_train)
    sk_r2 = sk.score(X_test, y_test)

    print("=" * 50)
    print(f"TRUE underlying params:   w={true_w[0]:.3f}, b={true_b:.3f}")
    print(f"OUR scratch model learned: w={ours.weights[0]:.3f}, b={ours.bias:.3f}")
    print(f"SKLEARN learned:          w={sk.coef_[0]:.3f}, b={sk.intercept_:.3f}")
    print("-" * 50)
    print(f"OUR R^2 on test set:     {our_r2:.4f}")
    print(f"SKLEARN R^2 on test set: {sk_r2:.4f}")
    print("=" * 50)

    # --- Plot 1: Loss curve (proves gradient descent is actually converging) ---
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(ours.loss_history)
    plt.xlabel("Iteration")
    plt.ylabel("MSE Loss")
    plt.title("Gradient Descent Convergence")
    plt.grid(alpha=0.3)

    # --- Plot 2: Fitted line vs data ---
    plt.subplot(1, 2, 2)
    plt.scatter(X_test, y_test, alpha=0.5, label="Actual data")
    x_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    plt.plot(x_line, ours.predict(x_line), color="red", label="Our model", linewidth=2)
    plt.plot(x_line, sk.predict(x_line), color="green", linestyle="--", label="sklearn")
    plt.xlabel("X")
    plt.ylabel("y")
    plt.title("Fitted Line: Ours vs Sklearn")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("../plots/day1_linear_regression_results.png", dpi=120)
    print("\nPlot saved to plots/day1_linear_regression_results.png")


if __name__ == "__main__":
    main()
