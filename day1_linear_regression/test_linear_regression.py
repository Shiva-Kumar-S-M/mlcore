"""
Tests verify our model behaves correctly on cases where we KNOW the
right answer -- this is how you sanity-check ML code, since there's
no compiler error for "learned the wrong thing."
"""

import numpy as np
from linear_regression import LinearRegressionScratch


def test_recovers_simple_linear_relationship():
    """If y = 2x + 1 exactly (no noise), the model should learn
    weight ~2 and bias ~1 almost perfectly."""
    X = np.array([[0], [1], [2], [3], [4]], dtype=float)
    y = np.array([1, 3, 5, 7, 9], dtype=float)  # y = 2x + 1

    model = LinearRegressionScratch(learning_rate=0.05, n_iterations=2000)
    model.fit(X, y)

    assert abs(model.weights[0] - 2.0) < 0.05
    assert abs(model.bias - 1.0) < 0.05


def test_loss_decreases_over_training():
    """Loss should never increase overall -- if it does, something
    is wrong with the gradient computation or learning rate."""
    X = np.array([[0], [1], [2], [3], [4]], dtype=float)
    y = np.array([1, 3, 5, 7, 9], dtype=float)

    model = LinearRegressionScratch(learning_rate=0.05, n_iterations=500)
    model.fit(X, y)

    assert model.loss_history[-1] < model.loss_history[0]


def test_perfect_fit_gives_r2_near_one():
    """On noiseless perfectly linear data, R^2 should be ~1.0
    (model explains basically all the variance)."""
    X = np.array([[0], [1], [2], [3], [4]], dtype=float)
    y = np.array([1, 3, 5, 7, 9], dtype=float)

    model = LinearRegressionScratch(learning_rate=0.05, n_iterations=2000)
    model.fit(X, y)

    assert model.score(X, y) > 0.99


if __name__ == "__main__":
    test_recovers_simple_linear_relationship()
    test_loss_decreases_over_training()
    test_perfect_fit_gives_r2_near_one()
    print("All tests passed!")
