"""
Linear Regression from scratch using only NumPy.

WHY THIS EXISTS:
sklearn.LinearRegression() hides everything interesting. This file
implements the actual math so you understand what "training a model"
really means: define a loss, compute its gradient, step downhill.

This exact pattern (loss -> gradient -> update) is the engine behind
every neural network you will ever train.
"""

import numpy as np


class LinearRegressionScratch:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None   # shape (n_features,)
        self.bias = None      # scalar
        self.loss_history = []  # so we can plot convergence later

    def fit(self, X, y):
        """
        X: shape (n_samples, n_features)
        y: shape (n_samples,)

        We initialize w and b to zero, then repeatedly:
          1. Predict y_hat = X.w + b
          2. Compute MSE loss between y_hat and y
          3. Compute the gradient of that loss w.r.t. w and b
          4. Nudge w and b a small step in the direction that reduces loss
        """
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for i in range(self.n_iterations):
            # Step 1: forward pass (prediction)
            y_pred = np.dot(X, self.weights) + self.bias

            # Step 2: compute loss (just for tracking/plotting, not used in update math directly)
            error = y_pred - y
            loss = np.mean(error ** 2)
            self.loss_history.append(loss)

            # Step 3: compute gradients (calculus, but here's the intuition:
            # if error is positive on average for a feature, increasing that
            # feature's weight is making things worse -> gradient tells us that)
            #
            # Derivation (don't worry if this feels dense first read):
            # Loss = (1/n) * sum((Xw + b - y)^2)
            # dLoss/dw = (2/n) * X^T . error
            # dLoss/db = (2/n) * sum(error)
            dw = (2 / n_samples) * np.dot(X.T, error)
            db = (2 / n_samples) * np.sum(error)

            # Step 4: update parameters, moving opposite to the gradient
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

        return self

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

    def score(self, X, y):
        """R^2 score: how much better than 'just predicting the mean' are we?
        1.0 = perfect, 0.0 = no better than predicting the average every time."""
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)
