"""
Trains our from-scratch neural network on the XOR-like / moons
dataset -- data that is NOT linearly separable, so logistic
regression (Day 2) fundamentally cannot solve it, but a neural
network with a nonlinear hidden layer can.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from neural_network import NeuralNetworkScratch


def main():
    X, y = make_moons(n_samples=300, noise=0.2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Show logistic regression FAILS on this non-linear data
    logreg = LogisticRegression()
    logreg.fit(X_train, y_train)
    logreg_acc = accuracy_score(y_test, logreg.predict(X_test))

    # Our neural network
    nn = NeuralNetworkScratch(input_size=2, hidden_size=8,
                                learning_rate=0.5, n_iterations=3000)
    nn.fit(X_train, y_train)
    nn_acc = accuracy_score(y_test, nn.predict(X_test))

    # sklearn's MLP for comparison
    sk_mlp = MLPClassifier(hidden_layer_sizes=(8,), max_iter=3000, random_state=42)
    sk_mlp.fit(X_train, y_train)
    sk_acc = accuracy_score(y_test, sk_mlp.predict(X_test))

    print("=" * 55)
    print(f"Logistic Regression (linear, should struggle): {logreg_acc:.3f}")
    print(f"Our Neural Network:                            {nn_acc:.3f}")
    print(f"Sklearn MLPClassifier:                         {sk_acc:.3f}")
    print("=" * 55)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, model, title in zip(
        axes,
        [logreg, nn, sk_mlp],
        [f"Logistic Regression\nacc={logreg_acc:.2f}",
         f"Our Neural Network\nacc={nn_acc:.2f}",
         f"Sklearn MLP\nacc={sk_acc:.2f}"]
    ):
        xx, yy = np.meshgrid(
            np.linspace(X[:, 0].min()-0.5, X[:, 0].max()+0.5, 200),
            np.linspace(X[:, 1].min()-0.5, X[:, 1].max()+0.5, 200)
        )
        grid = np.c_[xx.ravel(), yy.ravel()]
        Z = model.predict(grid).reshape(xx.shape)
        ax.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap="coolwarm", edgecolors="k")
        ax.set_title(title)

    plt.tight_layout()
    plt.savefig("../plots/day6_neural_network_comparison.png", dpi=120)
    print("Plot saved.")


if __name__ == "__main__":
    main()