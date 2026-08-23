import numpy as np
from regularized_regression import RegularizedRegressionScratch


def test_l2_shrinks_weights_compared_to_no_penalty():
    """Higher L2 penalty should produce smaller-magnitude weights
    on the same data -- this is the defining behavior of L2."""
    X = np.random.default_rng(0).normal(size=(50, 3))
    y = X @ np.array([5, -3, 2]) + np.random.default_rng(1).normal(size=50)

    no_reg = RegularizedRegressionScratch(learning_rate=0.05, n_iterations=1000, penalty=None)
    no_reg.fit(X, y)

    heavy_l2 = RegularizedRegressionScratch(learning_rate=0.05, n_iterations=1000, penalty="l2", lam=50)
    heavy_l2.fit(X, y)

    assert np.sum(np.abs(heavy_l2.weights)) < np.sum(np.abs(no_reg.weights))


def test_l1_can_zero_out_weights():
    """With a strong enough L1 penalty, some weights should shrink
    close to zero -- L1's feature-selection property."""
    X = np.random.default_rng(0).normal(size=(50, 5))
    y = X[:, 0] * 5 + np.random.default_rng(1).normal(size=50)  # only feature 0 matters

    model = RegularizedRegressionScratch(learning_rate=0.05, n_iterations=2000, penalty="l1", lam=5)
    model.fit(X, y)

    assert np.abs(model.weights[1:]).mean() < np.abs(model.weights[0])


if __name__ == "__main__":
    test_l2_shrinks_weights_compared_to_no_penalty()
    test_l1_can_zero_out_weights()
    print("All tests passed!")