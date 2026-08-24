import numpy as np
from decision_tree import DecisionTreeScratch, entropy


def test_entropy_pure_set_is_zero():
    """A set with only one class has zero disorder."""
    y = np.array([1, 1, 1, 1])
    assert entropy(y) == 0.0


def test_entropy_50_50_split_is_one():
    """A perfectly balanced 2-class set has maximum entropy of 1.0."""
    y = np.array([0, 1, 0, 1])
    assert abs(entropy(y) - 1.0) < 1e-9


def test_tree_perfectly_separates_simple_data():
    """On trivially separable data, the tree should get 100% accuracy."""
    X = np.array([[1], [2], [8], [9]], dtype=float)
    y = np.array([0, 0, 1, 1])

    model = DecisionTreeScratch(max_depth=3)
    model.fit(X, y)
    preds = model.predict(X)

    assert np.array_equal(preds, y)


def test_deeper_tree_fits_training_data_at_least_as_well():
    """Deeper trees should never do WORSE on training data than
    shallower ones -- more splits = more capacity to fit."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 2))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    shallow = DecisionTreeScratch(max_depth=1).fit(X, y)
    deep = DecisionTreeScratch(max_depth=8).fit(X, y)

    shallow_acc = np.mean(shallow.predict(X) == y)
    deep_acc = np.mean(deep.predict(X) == y)

    assert deep_acc >= shallow_acc


if __name__ == "__main__":
    test_entropy_pure_set_is_zero()
    test_entropy_50_50_split_is_one()
    test_tree_perfectly_separates_simple_data()
    test_deeper_tree_fits_training_data_at_least_as_well()
    print("All tests passed!")