"""
Logistic Regression from scratch using only NumPy.

Reuses the exact gradient descent loop from Day 1's linear regression,
but swaps in a sigmoid activation and cross-entropy loss -- proving
that classification isn't a different subject, just a different
squashing function and loss on top of the same linear core.
"""

import numpy as np


def sigmoid(z):
    """Maps any real number to (0, 1). Clip z to avoid overflow in exp()
    for very large/negative z (a real numerical-stability concern)."""
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))


class LogisticRegressionScratch:
    def __init__(self, learning_rate=0.1, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.loss_history = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for i in range(self.n_iterations):
            # 1. Predict: linear part, then squash to probability
            z = np.dot(X, self.weights) + self.bias
            p = sigmoid(z)

            # 2. Loss: binary cross-entropy
            eps = 1e-15  # avoid log(0)
            loss = -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
            self.loss_history.append(loss)

            # 3. Gradient: turns out to have the EXACT same form as
            # linear regression's gradient, just with p instead of y_pred.
            # This isn't a coincidence -- it falls out of the calculus of
            # combining sigmoid + cross-entropy.
            error = p - y
            dw = (1 / n_samples) * np.dot(X.T, error)
            db = (1 / n_samples) * np.sum(error)

            # 4. Update (identical mechanism to Day 1)
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

        return self

    def predict_proba(self, X):
        z = np.dot(X, self.weights) + self.bias
        return sigmoid(z)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)