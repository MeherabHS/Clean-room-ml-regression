import numpy as np
from sklearn.linear_model import LinearRegression
from manual_ml_model import ManualMLModel


# Fixing the seed keeps the benchmark reproducible so any coefficient differences
# can be interpreted as implementation behavior rather than random data drift.
np.random.seed(42)

# A synthetic linear dataset is used here because both the manual model and sklearn
# should converge to closely aligned solutions on the same well-specified regression problem.
X = np.random.randn(300, 3)
true_weights = np.array([2.5, -1.7, 0.9])
true_bias = 4.2
noise = np.random.randn(300) * 0.5
y = X @ true_weights + true_bias + noise

# The manual model keeps internal scaling enabled because that is part of the architecture,
# but sklearn must be trained on the same transformed feature space for a fair comparison.
manual_model = ManualMLModel(
    mode="linear",
    learning_rate=0.05,
    n_iter=5000,
    l2_lambda=0.0,
    standardize=True,
    tolerance=1e-10,
    fit_intercept=True,
    random_state=42,
)

# Fitting the manual model first establishes the training-time scaling statistics
# that will also be reused for the sklearn comparison path.
manual_model.fit(X, y)

# Applying the manual model's scaler to the same feature matrix ensures the comparison
# is between equivalent optimization problems rather than mismatched representations.
X_scaled = manual_model._scale_features(X.astype(np.float64), fit=False)

# Sklearn is used here only as a benchmark reference, not as part of the project logic.
sklearn_model = LinearRegression(fit_intercept=True)
sklearn_model.fit(X_scaled, y)

# Manual predictions are produced through the public inference path so the benchmark
# reflects the real deployed behavior of the custom estimator.
manual_predictions = manual_model.predict(X)

# Sklearn predictions are made on the already standardized feature matrix because
# that is the representation on which the sklearn model was trained.
sklearn_predictions = sklearn_model.predict(X_scaled)

# Extracting manual parameters allows coefficient-level comparison in the same scaled space
# where both estimators were optimized.
manual_params = manual_model.get_params()

# Coefficient closeness is evaluated with tolerance rather than exact equality because
# iterative optimization and closed-form solutions rarely match bit-for-bit.
weights_close = np.allclose(manual_params["weights"], sklearn_model.coef_, atol=1e-3, rtol=1e-3)
bias_close = np.isclose(manual_params["bias"], sklearn_model.intercept_, atol=1e-3, rtol=1e-3)

# Printing both parameter sets allows direct inspection of whether the manual solution
# is aligned with the industry-standard reference implementation.
print("Manual weights:", manual_params["weights"])
print("Sklearn weights:", sklearn_model.coef_)

# Intercept comparison is printed separately because correct bias handling is a common
# source of implementation defects in custom regression code.
print("Manual bias:", manual_params["bias"])
print("Sklearn bias:", sklearn_model.intercept_)

# These closeness checks convert qualitative visual similarity into a formal validation criterion.
print("Weights close:", weights_close)
print("Bias close:", bias_close)

# Prediction parity is checked through average absolute difference because even small
# coefficient deviations should still produce nearly identical predictions if the implementation is correct.
print("Mean absolute prediction difference:", np.mean(np.abs(manual_predictions - sklearn_predictions)))

# R2 comparison validates whether both models explain essentially the same variance on the same dataset.
print("Manual R2:", manual_model.score(X, y))
print("Sklearn R2:", sklearn_model.score(X_scaled, y))

# Loss endpoints are included because a close final fit is more credible when the optimizer
# also shows a stable descent trajectory.
print("First loss:", manual_model.loss_history[0])
print("Last loss:", manual_model.loss_history[-1])