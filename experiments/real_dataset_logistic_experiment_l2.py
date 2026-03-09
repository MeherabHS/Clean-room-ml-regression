import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from manual_ml_model import ManualMLModel


# Loading a real biomedical binary classification dataset provides the applied validation layer
# needed after synthetic correctness and sklearn synthetic parity have already been established.
data = load_breast_cancer()
X = data.data
y = data.target

# Stratification preserves class proportions across train and test splits,
# which reduces evaluation distortion in a moderately imbalanced binary dataset.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

# L2 regularization is deliberately enabled here because unregularized logistic regression
# on near-separable real datasets can produce unstable coefficient magnitudes and weak parity diagnostics.
l2_lambda = 1.0

# The manual model retains the same architecture, but now uses L2 shrinkage so the optimization problem
# is better conditioned and more comparable to a production logistic benchmark.
manual_model = ManualMLModel(
    mode="logistic",
    learning_rate=0.05,
    n_iter=5000,
    l2_lambda=l2_lambda,
    standardize=True,
    tolerance=1e-10,
    fit_intercept=True,
    random_state=42,
)

# Training on the real dataset tests whether the manual logistic implementation remains stable
# and predictive under a regularized objective that is more appropriate for this dataset geometry.
manual_model.fit(X_train, y_train)

# Reusing the manual model's scaling statistics ensures both estimators are trained and evaluated
# on the exact same transformed feature representation.
X_train_scaled = manual_model._scale_features(X_train.astype(np.float64), fit=False)
X_test_scaled = manual_model._scale_features(X_test.astype(np.float64), fit=False)

# In sklearn, C is the inverse of regularization strength, so C=1.0 corresponds to a standard
# moderate L2 penalty and provides a stable benchmark against the manual regularized implementation.
sklearn_model = LogisticRegression(
    penalty="l2",
    C=1.0,
    fit_intercept=True,
    solver="lbfgs",
    max_iter=10000,
)

# The sklearn model serves only as a reference implementation for parity checking.
sklearn_model.fit(X_train_scaled, y_train)

# Test accuracy is the primary real-data classification metric because the project must show
# useful generalization behavior on unseen samples, not only training fit.
manual_accuracy = manual_model.score(X_test, y_test)
sklearn_accuracy = sklearn_model.score(X_test_scaled, y_test)

# Probability comparison is stronger than hard-label comparison alone because it evaluates whether
# both models assign similar confidence to the positive class on unseen data.
manual_proba = manual_model.predict_proba(X_test)[:, 1]
sklearn_proba = sklearn_model.predict_proba(X_test_scaled)[:, 1]
mean_abs_proba_diff = np.mean(np.abs(manual_proba - sklearn_proba))

# Prediction mismatch rate is printed because materially different decision boundaries
# would show up as disagreement in thresholded class labels.
manual_predictions = manual_model.predict(X_test)
sklearn_predictions = sklearn_model.predict(X_test_scaled)
prediction_mismatch_rate = np.mean(manual_predictions != sklearn_predictions)

# Parameter extraction remains useful, but under real-data regularized classification
# the stronger success criterion is test-set predictive and probabilistic parity.
manual_params = manual_model.get_params()
sklearn_weights = sklearn_model.coef_.ravel()
sklearn_bias = sklearn_model.intercept_[0]

print("Dataset name: Breast Cancer Wisconsin (L2-Regularized)")
print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

print("Manual test accuracy:", manual_accuracy)
print("Sklearn test accuracy:", sklearn_accuracy)
print("Mean absolute probability difference:", mean_abs_proba_diff)
print("Prediction mismatch rate:", prediction_mismatch_rate)

print("Manual weights:", manual_params["weights"])
print("Sklearn weights:", sklearn_weights)

print("Manual bias:", manual_params["bias"])
print("Sklearn bias:", sklearn_bias)

print("Weights close:", np.allclose(manual_params["weights"], sklearn_weights, atol=1e-1, rtol=1e-1))
print("Bias close:", np.isclose(manual_params["bias"], sklearn_bias, atol=1e-1, rtol=1e-1))

print("First loss:", manual_model.loss_history[0])
print("Last loss:", manual_model.loss_history[-1])