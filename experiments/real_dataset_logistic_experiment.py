import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from manual_ml_model import ManualMLModel


# Loading a real binary classification dataset provides the applied validation step
# needed to show that the manual logistic implementation works beyond synthetic experiments.
data = load_breast_cancer()
X = data.data
y = data.target

# A stratified split is used so the class balance remains stable across train and test sets,
# which makes the downstream performance comparison more reliable.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

# The manual classifier reuses the already validated architectural choices so the experiment
# measures applied behavior rather than a changed implementation.
manual_model = ManualMLModel(
    mode="logistic",
    learning_rate=0.05,
    n_iter=5000,
    l2_lambda=0.0,
    standardize=True,
    tolerance=1e-10,
    fit_intercept=True,
    random_state=42,
)

# Training on the real dataset tests whether the manual logistic pipeline remains numerically stable
# and predictive on real biomedical tabular data.
manual_model.fit(X_train, y_train)

# The sklearn classifier must be trained on the exact same scaled representation to keep
# coefficient and performance comparison mathematically fair.
X_train_scaled = manual_model._scale_features(X_train.astype(np.float64), fit=False)
X_test_scaled = manual_model._scale_features(X_test.astype(np.float64), fit=False)

# C=np.inf is used instead of deprecated penalty=None so the benchmark remains effectively
# unregularized without introducing version-fragile warnings.
sklearn_model = LogisticRegression(
    C=np.inf,
    fit_intercept=True,
    solver="lbfgs",
    max_iter=10000,
)
sklearn_model.fit(X_train_scaled, y_train)

# Test-set accuracy is the primary applied metric here because the project needs to show
# real classification performance on unseen data.
manual_accuracy = manual_model.score(X_test, y_test)
sklearn_accuracy = sklearn_model.score(X_test_scaled, y_test)

# Probability comparison is stronger than label comparison alone because it tests whether
# both classifiers produce similar confidence estimates, not just similar thresholded outputs.
manual_proba = manual_model.predict_proba(X_test)[:, 1]
sklearn_proba = sklearn_model.predict_proba(X_test_scaled)[:, 1]
mean_abs_proba_diff = np.mean(np.abs(manual_proba - sklearn_proba))

# Label mismatch rate helps detect whether small probability differences are creating
# materially different classification decisions.
manual_predictions = manual_model.predict(X_test)
sklearn_predictions = sklearn_model.predict(X_test_scaled)
prediction_mismatch_rate = np.mean(manual_predictions != sklearn_predictions)

# Parameter comparison is still useful, but on real data the stronger success criterion
# is predictive and probabilistic parity on the held-out set.
manual_params = manual_model.get_params()
sklearn_weights = sklearn_model.coef_.ravel()
sklearn_bias = sklearn_model.intercept_[0]

print("Dataset name: Breast Cancer Wisconsin")
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

print("Weights close:", np.allclose(manual_params["weights"], sklearn_weights, atol=1e-2, rtol=1e-2))
print("Bias close:", np.isclose(manual_params["bias"], sklearn_bias, atol=1e-2, rtol=1e-2))

print("First loss:", manual_model.loss_history[0])
print("Last loss:", manual_model.loss_history[-1])