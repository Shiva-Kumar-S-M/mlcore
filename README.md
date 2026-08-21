# MLCore: Machine Learning Algorithms Built From First Principles

Most ML portfolios call `sklearn.LinearRegression()` and stop there.
This project implements core ML algorithms **from scratch using only
NumPy**, then benchmarks each one against sklearn's production
implementation to prove mathematical correctness.

Goal: build a genuinely solid foundation in the math and mechanics
that power machine learning and, eventually, deep learning.

## Why this project

Understanding *how* a model learns (loss function -> gradient ->
parameter update) is the single idea that underlies linear
regression, logistic regression, neural networks, and even the
training of large language models. This repo builds that
understanding one algorithm at a time, with each one verified against
a trusted library.

## Progress

| Day | Algorithm | Status |
|---|---|---|
| 1 | Linear Regression (Gradient Descent) | ✅ Done |
| 2 | Logistic Regression — Sigmoid & Cross-Entropy Loss | ✅ |
| 3 | Regularization (L1/L2) | 🔜 |
| 4 | Decision Trees | 🔜 |
| 5 | Applied project: Credit Risk Prediction | 🔜 |
| 6 | Neural Network from scratch (backprop) | 🔜 |
| 7 | Model explainability + final write-up | 🔜 |

## Structure

```
mlcore/
├── day1_linear_regression/
│   ├── linear_regression.py       # from-scratch implementation
│   ├── train_and_compare.py       # benchmarks vs sklearn
│   └── test_linear_regression.py  # unit tests
├── plots/                          # saved result visualizations
├── PROGRESS.md                     # daily log (also used to resume work)
└── requirements.txt
```

## Day 1: Linear Regression

Implemented gradient descent from scratch and verified it converges
to the exact same weights as sklearn's `LinearRegression`:

```
OUR scratch model learned: w=3.773, b=-2.401
SKLEARN learned:          w=3.773, b=-2.401
```

Also ran a learning-rate experiment showing gradient explosion when
the learning rate is too high — see `PROGRESS.md` for details.

![results](plots/day1_linear_regression_results.png)


## Day 2 — Logistic Regression

Extended Day 1's linear model into a binary classifier by adding a
sigmoid activation and switching to binary cross-entropy loss.
Benchmarked against sklearn's `LogisticRegression` on synthetic,
linearly separable data.

**Result:**


The small gap vs. sklearn is expected — sklearn applies L2
regularization by default, which our implementation doesn't yet.
Day 3 adds this and closes the gap.

**Convergence and decision boundary:**

![results](plots/day2_logistic_regression_results.png)

### Run it locally
```bash
cd day2_logistic_regression
python train_and_compare.py
python -m pytest test_logistic_regression.py -v
```