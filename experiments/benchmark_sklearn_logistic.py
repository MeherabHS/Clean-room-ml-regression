import numpy as np
from sklearn.linear_model import LogisticRegression
from manual_ml_model import ManualMLModel


# Fixing the seed ensures the benchmark dataset is reproducible,
# which is necessary when comparing iterative optimization results across implementations.
np.random.seed(42)

# A larger synthetic dataset is used because logistic regression coefficient estimates
# are more stable with higher sample size, making parity assessment more reliable.
X = np.random.randn(5000, 3)
true_weights = np.array([1.8, -2.2, 1.1])
true_bias = -0.4

# The labels are generated from a Bernoulli process defined by the sigmoid-transformed logits,
# which makes this a valid binary logistic regression benchmark problem.
logits = X @ true_weights + true_bias
probabilities = 1.0 / (1.0 + np.exp(-logits))
y = (np.random.rand(5000) < probabilities).astype(int)

# The manual model is configured without L2 regularization so the benchmark compares
# the same optimization objective as an unregularized sklearn logistic model.
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

# Fitting the manual model first establishes the training-time scaling statistics
# that sklearn must also use for a fair like-for-like comparison.
manual_model.fit(X, y)

# The same standardized feature representation is reused for sklearn so coefficient
# and prediction comparisons remain mathematically valid.
X_scaled = manual_model._scale_features(X.astype(np.float64), fit=False)

# Sklearn is used strictly as a benchmark reference implementation.
sklearn_model = LogisticRegression(
    penalty=None,
    fit_intercept=True,
    solver="lbfgs",
    max_iter=10000,
)
sklearn_model.fit(X_scaled, y)

# Manual predictions follow the public API path so the benchmark reflects actual model behavior.
manual_predictions = manual_model.predict(X)

# Sklearn predictions are generated on the same standardized feature representation
# on which the benchmark model was trained.
sklearn_predictions = sklearn_model.predict(X_scaled)

# Probability outputs are also compared because logistic regression should match not only
# the final class labels but also the estimated class probabilities.
manual_proba = manual_model.predict_proba(X)[:, 1]
sklearn_proba = sklearn_model.predict_proba(X_scaled)[:, 1]

# Parameters are extracted in the same scaled space so coefficient comparison is fair.
manual_params = manual_model.get_params()
sklearn_weights = sklearn_model.coef_.ravel()
sklearn_bias = sklearn_model.intercept_[0]

# Tolerance-based closeness is used because two iterative solvers can converge to
# numerically equivalent solutions without being bitwise identical.
weights_close = np.allclose(manual_params["weights"], sklearn_weights, atol=1e-2, rtol=1e-2)
bias_close = np.isclose(manual_params["bias"], sklearn_bias, atol=1e-2, rtol=1e-2)

# Printing coefficients allows direct inspection of whether both models recovered
# essentially the same separating hyperplane in the standardized feature space.
print("Manual weights:", manual_params["weights"])
print("Sklearn weights:", sklearn_weights)

# Intercept comparison is printed separately because bias handling is a common source
# of subtle implementation error in custom classifiers.
print("Manual bias:", manual_params["bias"])
print("Sklearn bias:", sklearn_bias)

# These boolean checks summarize whether the parameter-level differences remain within
# acceptable numerical tolerance for a correct implementation.
print("Weights close:", weights_close)
print("Bias close:", bias_close)

# Mean absolute probability difference is a stronger diagnostic than label parity alone
# because it compares the confidence structure of the classifiers, not just thresholded outputs.
print("Mean absolute probability difference:", np.mean(np.abs(manual_proba - sklearn_proba)))

# Accuracy comparison confirms that both models produce essentially the same decision performance.
print("Manual accuracy:", manual_model.score(X, y))
print("Sklearn accuracy:", sklearn_model.score(X_scaled, y))

# Label mismatch rate is printed because even when probability estimates are close,
# decision-threshold differences might still reveal implementation problems.
print("Prediction mismatch rate:", np.mean(manual_predictions != sklearn_predictions))

# Loss endpoints confirm the manual optimizer followed a healthy descent trajectory.
print("First loss:", manual_model.loss_history[0])
print("Last loss:", manual_model.loss_history[-1])