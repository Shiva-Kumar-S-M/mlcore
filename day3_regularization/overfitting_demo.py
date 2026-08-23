"""
Demonstrates overfitting on purpose using high-degree polynomial
features, then shows how L2 regularization tames it.
This is the experiment that makes bias-variance tradeoff tangible.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from regularized_regression import RegularizedRegressionScratch


def generate_noisy_curve(n_samples=40, seed=1):
    rng = np.random.default_rng(seed)
    X = np.sort(rng.uniform(-3, 3, n_samples)).reshape(-1, 1)
    y = 0.5 * X[:, 0] ** 3 - X[:, 0] + rng.normal(0, 3, n_samples)
    return X, y


def main():
    X, y = generate_noisy_curve()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Deliberately overfit: degree-12 polynomial features
    poly = PolynomialFeatures(degree=12)
    X_train_poly = poly.fit_transform(X_train)[:, 1:]  # drop bias col
    X_test_poly = poly.transform(X_test)[:, 1:]

    # Normalize -- crucial for regularization to behave sensibly
    mean, std = X_train_poly.mean(axis=0), X_train_poly.std(axis=0) + 1e-8
    X_train_poly = (X_train_poly - mean) / std
    X_test_poly = (X_test_poly - mean) / std

    configs = [
        ("No regularization (overfit)", None, 0),
        ("L2, lambda=1", "l2", 1),
        ("L2, lambda=20", "l2", 20),
    ]

    plt.figure(figsize=(15, 4))
    for i, (label, penalty, lam) in enumerate(configs):
        model = RegularizedRegressionScratch(
            learning_rate=0.05, n_iterations=3000, penalty=penalty, lam=lam
        )
        model.fit(X_train_poly, y_train)

        train_r2 = model.score(X_train_poly, y_train)
        test_r2 = model.score(X_test_poly, y_test)

        print(f"{label:30s} | Train R2: {train_r2:.3f} | Test R2: {test_r2:.3f}")

        plt.subplot(1, 3, i + 1)
        plt.scatter(X_train, y_train, alpha=0.5, label="train")
        plt.scatter(X_test, y_test, alpha=0.5, label="test", color="orange")
        x_line = np.linspace(-3, 3, 200).reshape(-1, 1)
        x_line_poly = (poly.transform(x_line)[:, 1:] - mean) / std
        plt.plot(x_line, model.predict(x_line_poly), color="red")
        plt.title(f"{label}\nTrain R2={train_r2:.2f}, Test R2={test_r2:.2f}")
        plt.ylim(-20, 20)
        plt.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig("../plots/day3_overfitting_regularization.png", dpi=120)
    print("\nPlot saved.")


if __name__ == "__main__":
    main()