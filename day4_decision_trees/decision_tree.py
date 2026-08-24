"""
Decision Tree Classifier from scratch using entropy and information gain.

Unlike Days 1-3, there's no gradient descent here. Trees learn by
greedily searching for the best (feature, threshold) split at every
node -- a fundamentally different optimization strategy than the
gradient-based models we've built so far.
"""

import numpy as np
from collections import Counter


class Node:
    """A single node in the tree. Either a decision node (has a
    feature/threshold and two children) or a leaf (has a predicted class)."""
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value  # only set on leaf nodes

    def is_leaf(self):
        return self.value is not None


def entropy(y):
    """Measures disorder in a set of labels. 0 = pure, higher = more mixed."""
    counts = np.bincount(y)
    probabilities = counts / len(y)
    # Ignore zero-probability classes (log2(0) is undefined)
    return -np.sum([p * np.log2(p) for p in probabilities if p > 0])


class DecisionTreeScratch:
    def __init__(self, max_depth=5, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def fit(self, X, y):
        self.root = self._grow_tree(X, y, depth=0)
        return self

    def _grow_tree(self, X, y, depth):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # Stopping conditions: pure node, too deep, or too few samples
        if n_labels == 1 or depth >= self.max_depth or n_samples < self.min_samples_split:
            leaf_value = Counter(y).most_common(1)[0][0]
            return Node(value=leaf_value)

        best_feature, best_threshold = self._best_split(X, y, n_features)

        if best_feature is None:  # no split improves things
            leaf_value = Counter(y).most_common(1)[0][0]
            return Node(value=leaf_value)

        left_idx = X[:, best_feature] <= best_threshold
        right_idx = ~left_idx

        left = self._grow_tree(X[left_idx], y[left_idx], depth + 1)
        right = self._grow_tree(X[right_idx], y[right_idx], depth + 1)

        return Node(feature=best_feature, threshold=best_threshold, left=left, right=right)

    def _best_split(self, X, y, n_features):
        """Try every feature and every possible threshold; keep the
        split with the highest information gain."""
        best_gain = -1
        split_feature, split_threshold = None, None

        parent_entropy = entropy(y)

        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_idx = X[:, feature] <= threshold
                right_idx = ~left_idx

                if left_idx.sum() == 0 or right_idx.sum() == 0:
                    continue

                n = len(y)
                left_entropy = entropy(y[left_idx])
                right_entropy = entropy(y[right_idx])
                weighted_entropy = (left_idx.sum() / n) * left_entropy + \
                                    (right_idx.sum() / n) * right_entropy

                gain = parent_entropy - weighted_entropy

                if gain > best_gain:
                    best_gain = gain
                    split_feature = feature
                    split_threshold = threshold

        return split_feature, split_threshold

    def predict(self, X):
        return np.array([self._traverse(x, self.root) for x in X])

    def _traverse(self, x, node):
        if node.is_leaf():
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse(x, node.left)
        return self._traverse(x, node.right)