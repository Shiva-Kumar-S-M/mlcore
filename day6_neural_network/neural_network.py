"""
A 2-layer neural network (1 hidden layer) from scratch using only
NumPy. This directly reuses Day 1's linear layer (Xw+b) and Day 2's
sigmoid -- a neural network is not a new concept, it's those two
ideas stacked with a nonlinearity (ReLU) in between.
"""

import numpy as np


def relu(z):
    return np.maximum(0, z)


def relu_derivative(z):
    """Derivative of ReLU: 1 where input was positive, 0 otherwise.
    This is what lets the gradient 'flow through' active neurons
    and blocks it for inactive (zeroed-out) ones."""
    return (z > 0).astype(float)


def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))


class NeuralNetworkScratch:
    def __init__(self, input_size, hidden_size, learning_rate=0.1, n_iterations=2000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.loss_history = []

        # Small random init -- NOT zeros like Day 1. If all weights start
        # at zero, every hidden neuron computes the identical gradient and
        # update, so they'd all learn the exact same thing forever
        # ("symmetry breaking" problem). Small random noise breaks that.
        rng = np.random.default_rng(42)
        self.W1 = rng.normal(0, 0.5, size=(input_size, hidden_size))
        self.b1 = np.zeros(hidden_size)
        self.W2 = rng.normal(0, 0.5, size=(hidden_size, 1))
        self.b2 = np.zeros(1)

    def forward(self, X):
        """Forward pass: input -> hidden layer (ReLU) -> output (sigmoid).
        We cache intermediate values (z1, a1) because backprop needs them."""
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = relu(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = sigmoid(self.z2)
        return self.a2

    def backward(self, X, y):
        """Backpropagation: compute gradients layer by layer, working
        BACKWARD from the output. Each step uses the chain rule --
        the same gradient math from Day 1/2, just chained together."""
        n = X.shape[0]
        y = y.reshape(-1, 1)

        # --- Output layer gradient (identical form to Day 2's logistic regression) ---
        dz2 = self.a2 - y                              # (n, 1)
        dW2 = (1 / n) * np.dot(self.a1.T, dz2)          # (hidden_size, 1)
        db2 = (1 / n) * np.sum(dz2, axis=0)

        # --- Hidden layer gradient: chain rule flows the error backward ---
        # "How much did the hidden layer contribute to the output error?"
        da1 = np.dot(dz2, self.W2.T)                    # (n, hidden_size)
        dz1 = da1 * relu_derivative(self.z1)             # zero out where ReLU was inactive
        dW1 = (1 / n) * np.dot(X.T, dz1)
        db1 = (1 / n) * np.sum(dz1, axis=0)

        return dW1, db1, dW2, db2

    def fit(self, X, y):
        y = y.reshape(-1, 1)
        for i in range(self.n_iterations):
            y_pred = self.forward(X)

            eps = 1e-15
            loss = -np.mean(y * np.log(y_pred + eps) + (1 - y) * np.log(1 - y_pred + eps))
            self.loss_history.append(loss)

            dW1, db1, dW2, db2 = self.backward(X, y.flatten())

            # Same update rule as every model this week
            self.W1 -= self.learning_rate * dW1
            self.b1 -= self.learning_rate * db1
            self.W2 -= self.learning_rate * dW2
            self.b2 -= self.learning_rate * db2

        return self

    def predict(self, X, threshold=0.5):
        return (self.forward(X) >= threshold).astype(int).flatten()

    def predict_proba(self, X):
        return self.forward(X).flatten()