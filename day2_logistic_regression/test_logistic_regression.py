import numpy as np
from logistic_regression import LogisticRegressionScratch, sigmoid


def test_sigmoid_bounds():
    """Sigmoid output must always stay within [0, 1]. At extreme inputs
    (e.g. z=1000) it saturates to exactly 0.0 or 1.0 due to floating-point
    precision limits -- that's expected numerical behavior, not a bug."""
    values = sigmoid(np.array([-1000, -1, 0, 1, 1000]))
    assert np.all(values >= 0) and np.all(values <= 1)


def test_sigmoid_moderate_values_strictly_between_0_and_1():
    """For non-extreme inputs, sigmoid should be strictly between 0
    and 1 -- this is where saturation doesn't kick in."""
    values = sigmoid(np.array([-10, -1, 0, 1, 10]))
    assert np.all(values > 0) and np.all(values < 1)


def test_sigmoid_zero_is_half():
    """Sigmoid(0) should be exactly 0.5 -- the point of max uncertainty."""
    assert abs(sigmoid(0) - 0.5) < 1e-9


def test_learns_separable_data():
    """On perfectly linearly separable data, the model should classify
    every point correctly after training."""
    X = np.array([[0, 0], [0, 1], [5, 5], [6, 5]], dtype=float)
    y = np.array([0, 0, 1, 1])

    model = LogisticRegressionScratch(learning_rate=0.5, n_iterations=2000)
    model.fit(X, y)
    preds = model.predict(X)

    assert np.array_equal(preds, y)


if __name__ == "__main__":
    test_sigmoid_bounds()
    test_sigmoid_moderate_values_strictly_between_0_and_1()
    test_sigmoid_zero_is_half()
    test_learns_separable_data()
    print("All tests passed!")