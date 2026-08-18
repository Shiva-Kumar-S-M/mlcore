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
