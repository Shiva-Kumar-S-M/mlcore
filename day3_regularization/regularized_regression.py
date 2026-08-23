"""
Linear Regression with L1 (Lasso) and L2 (Ridge) regularization,
built on top of Day 1's gradient descent -- only the gradient
calculation changes to include a penalty term.
"""

import numpy as np


class RegularizedRegressionScratch:
    def __init__(self, learning_rate=0.01, n_iterations=1000,
                 penalty=None, lam=0.1):
        """
        penalty: None, 'l1', or 'l2'
        lam: regularization strength (lambda)
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.penalty = penalty
        self.lam = lam
        self.weights = None
        self.bias = None
        self.loss_history = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for i in range(self.n_iterations):
            y_pred = np.dot(X, self.weights) + self.bias
            error = y_pred - y

            # Base MSE gradient (identical to Day 1)
            dw = (2 / n_samples) * np.dot(X.T, error)
            db = (2 / n_samples) * np.sum(error)

            # Add regularization penalty to the gradient.
            # Bias is intentionally NOT regularized -- only weights,
            # since penalizing the baseline offset doesn't help
            # generalization and can hurt fit unnecessarily.
            if self.penalty == "l2":
                dw += (self.lam / n_samples) * 2 * self.weights
            elif self.penalty == "l1":
                dw += (self.lam / n_samples) * np.sign(self.weights)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            mse = np.mean(error ** 2)
            self.loss_history.append(mse)

        return self

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

    def score(self, X, y):
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)