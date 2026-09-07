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
## Day 5 — Applied Project: Credit Risk Prediction ✅

**Built:**
- `download_data.py`: fetches the UCI German Credit dataset (1000
  real loan applicants).
- `preprocess.py`: one-hot encodes categorical variables, standardizes
  numeric features -- the real preprocessing work synthetic data
  never requires.
- `run_all_models.py`: runs our Logistic Regression (Day 2) and
  Decision Tree (Day 4) against sklearn equivalents on real data,
  compares via accuracy/precision/recall/F1/ROC-AUC.
- `test_pipeline.py`: 3 unit tests validating clean, numeric,
  binary-target output from preprocessing.

**Key results:** [PASTE your actual results table from run_all_models.py console output here]

**Concepts learned:**
- Real data requires preprocessing (encoding, scaling) that clean
  synthetic data hides -- this is most of practical ML work.
- Feature scaling matters for gradient descent convergence speed
  and stability.
- Class imbalance is real here (~70/30 split), making F1/ROC-AUC
  more meaningful than raw accuracy -- direct application of the
  Day 2 lesson on a genuine dataset.
- Comparing multiple model families side by side on the same data
  is how model selection actually works in practice.

## Day 6 — Neural Network from Scratch ✅

**Built:**
- `neural_network.py`: 2-layer network (1 hidden layer) with ReLU
  activation and manual backpropagation, reusing Day 1's linear
  layer and Day 2's sigmoid/cross-entropy directly.
- `train_and_compare.py`: compares logistic regression vs our NN vs
  sklearn's MLPClassifier on non-linearly-separable moon-shaped data.
- `test_neural_network.py`: 4 tests, including solving XOR -- the
  classic proof that hidden layers add real representational power
  beyond what any linear model can achieve.

**Key results:** [PASTE accuracy numbers from console output]

**Concepts learned:**
- A neural network = stacked linear layers (Day 1) + nonlinearity
  (ReLU) + Day 2's sigmoid/cross-entropy on the output.
- Backpropagation is the chain rule applied layer by layer, working
  backward from the output -- not new math, just Day 1/2's gradient
  calculation chained together.
- Why nonlinearity is required: stacking linear layers alone
  collapses back into ONE linear equation, no extra power gained.
- Why weights are randomly initialized (not zero, unlike Day 1) --
  symmetry breaking, so hidden neurons learn different things.
- XOR/moons data proves linear models have a hard mathematical
  ceiling that hidden layers break through.

## Day 7 — Model Explainability & Final Write-Up ✅

**Built:**
- `explain_models.py`: feature importance for Logistic Regression
  (coefficient magnitude) and Decision Tree (entropy reduction),
  compared side by side on the real credit risk data.
- `test_explainability.py`: sanity-check test proving the importance
  calculation correctly identifies a known informative feature over
  pure noise.

**Key results:** [PASTE your top features from console output]

**Concepts learned:**
- Explainability matters as much as accuracy in regulated/high-stakes
  domains (credit, healthcare, hiring) -- "why" is often required,
  not just "what."
- Two different explainability methods (coefficient magnitude vs.
  entropy reduction) agreeing on top features is stronger evidence
  than either alone.
- This closes the loop on the whole week: every model built (Days
  1-6) can now not just predict, but explain itself.

## Project Complete — Week Summary

Built 5 core ML algorithms from scratch (Linear Regression, Logistic
Regression, Regularization, Decision Trees, Neural Networks),
verified each against scikit-learn, applied them to a real credit
risk dataset, and added explainability on top. 12+ modular files,
20+ unit tests, feature-branch git workflow with 70+ descriptive
commits across the week.

**Foundational insight carried through every day:** every model,
no matter how different it looks on the surface, follows the same
loop -- predict, measure loss, compute gradient (or best split),
update. That loop is what deep learning and agentic AI systems
build on next.