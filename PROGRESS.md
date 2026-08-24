# Progress Log

## Day 1 — Linear Regression from Scratch ✅

**Built:**
- `linear_regression.py`: LinearRegressionScratch class with `fit`,
  `predict`, `score`. Uses batch gradient descent on MSE loss.
- `train_and_compare.py`: generates synthetic data with a known true
  relationship, trains our model + sklearn's, compares weights/R^2,
  plots loss curve and fitted line.
- `test_linear_regression.py`: 3 unit tests, all passing.

**Key results:**
- Our model's learned weights matched sklearn to 3 decimal places
  (w=3.773, b=-2.401 both), proving the gradient math is correct.
- R^2 = 0.8484 on test set for both.

**Concepts learned:**
- Loss function (MSE) and why we square errors.
- Gradient descent: compute gradient, step opposite direction.
- Learning rate tradeoffs: too small = slow convergence, too large =
  divergence/explosion (demonstrated: lr=0.5 caused weights to
  explode to ~1e74 within 50 iterations -- this is what "exploding
  gradients" / "loss: nan" means in practice).
- R^2 score: how to interpret model quality beyond just "loss went
  down."

**Next (Day 2):** Logistic Regression from scratch — sigmoid
function, cross-entropy loss, decision boundaries. Will reuse the
gradient descent pattern from Day 1 but with a different loss
function suited to classification.

---

<!-- Each new day: append a new "## Day N" section above this line.
     When resuming, just say "continue day N" and I'll read only
     this file, not the full codebase, to pick up context cheaply. -->
## Day 2 — Logistic Regression from Scratch ✅

**Built:**
- `logistic_regression.py`: LogisticRegressionScratch with sigmoid
  activation, binary cross-entropy loss, gradient descent.
- `train_and_compare.py`: benchmarks against sklearn's
  LogisticRegression, reports accuracy/precision/recall/F1, plots
  convergence and decision boundary.
- `test_logistic_regression.py`: 3 unit tests, all passing.

**Key results:**
- Weights close to sklearn's but not identical (sklearn applies L2
  regularization by default -- ours doesn't yet, which is exactly
  what Day 3 addresses).
- Both models hit recall=1.000 on the test set.

**Concepts learned:**
- Classification = linear regression + sigmoid + cross-entropy loss.
- Decision boundary: the line where the model is exactly 50/50.
- Why accuracy alone is misleading on imbalanced data; precision vs
  recall tradeoffs and when each matters more.

**Next (Day 3):** Regularization (L1/L2) and bias-variance tradeoff --
will explain the small weight gap seen today.


## Day 3 — Regularization (L1/L2) ✅

**Built:**
- `regularized_regression.py`: extends Day 1's gradient descent with
  L1 and L2 penalty terms added to the gradient.
- `overfitting_demo.py`: deliberately overfits with degree-12
  polynomial features, then shows L2 fixing it at different lambda
  strengths.
- `test_regularization.py`: verifies L2 shrinks weights, L1 induces
  sparsity.

**Key results:**
- No regularization: high train R2, much lower test R2 (overfitting).
- L2 lambda=20: train/test R2 gap shrinks significantly, curve
  smooths out.

**Concepts learned:**
- Bias-variance tradeoff made concrete: too flexible = overfits
  (low bias, high variance); too constrained = underfits (high
  bias, low variance).
- L2 shrinks weights smoothly; L1 can zero them out entirely
  (built-in feature selection).
- Why bias term is never regularized.
- This explained the Day 2 weight gap vs sklearn -- sklearn
  regularizes by default.

**Next (Day 4):** Decision Trees -- a completely different way to
fit data (splits, not gradients), and why tree-based models
overfit/underfit differently than linear models.

## Day 4 — Decision Trees ✅

**Built:**
- `decision_tree.py`: DecisionTreeScratch using entropy and
  information gain for greedy recursive splitting.
- `train_and_compare.py`: benchmarks against sklearn's
  DecisionTreeClassifier, visualizes the blocky decision boundary.
- `depth_experiment.py`: sweeps max_depth from 1 to 20, plots
  train vs test accuracy to show overfitting emerging.
- `test_decision_tree.py`: 4 unit tests covering entropy edge cases
  and tree correctness.

**Key results:**
- Matched sklearn's accuracy closely on synthetic classification data.
- Depth experiment showed classic overfitting signature: train
  accuracy climbs toward 100% with depth, test accuracy peaks then
  plateaus/drops.

**Concepts learned:**
- Trees use a fundamentally different optimization approach than
  Days 1-3: greedy search over splits, not gradient descent.
- Entropy measures disorder; Information Gain measures how much a
  split reduces it.
- Decision boundaries are blocky/axis-aligned (vs. linear models'
  straight lines) because each split only tests one feature.
- max_depth is the tree equivalent of Day 3's regularization
  lambda -- both control model complexity and the bias-variance
  tradeoff, just through different mechanisms.

**Next (Day 5):** Apply everything so far to a real dataset
(credit risk prediction) and compare all models side by side on
real, messy data.