import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from manual_ml_model import ManualMLModel


# Loading a real regression dataset is the first applied test of whether the manual estimator
# can move beyond synthetic validation and operate on practical feature distributions.
data = fetch_california_housing()
X = data.data
y = data.target

# Splitting into train and test sets is necessary to evaluate generalization rather than
# merely measuring how well the model memorizes the data it was trained on.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)

# The manual model uses the same architectural choices already validated in the synthetic phase,
# which allows the real-data experiment to test application rather than altered implementation logic.
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

# Training the manual model on the real dataset tests whether the gradient descent pipeline
# remains numerically stable and predictive on naturally distributed features.
manual_model.fit(X_train, y_train)

# The sklearn benchmark is trained on the same standardized representation to preserve
# fairness in the coefficient and score comparison.
X_train_scaled = manual_model._scale_features(X_train.astype(np.float64), fit=False)
X_test_scaled = manual_model._scale_features(X_test.astype(np.float64), fit=False)

sklearn_model = LinearRegression(fit_intercept=True)
sklearn_model.fit(X_train_scaled, y_train)

# Test-set scoring is used because real-world credibility depends on generalization quality,
# not on training-set fit alone.
manual_r2 = manual_model.score(X_test, y_test)
sklearn_r2 = sklearn_model.score(X_test_scaled, y_test)

# Mean absolute prediction difference helps determine whether the manual implementation
# remains aligned with sklearn on unseen real data.
manual_predictions = manual_model.predict(X_test)
sklearn_predictions = sklearn_model.predict(X_test_scaled)
mean_abs_diff = np.mean(np.abs(manual_predictions - sklearn_predictions))

# Coefficient comparison is still informative, but on real data the primary success criterion
# is predictive parity on the held-out set.
manual_params = manual_model.get_params()

print("Dataset name: California Housing")
print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

print("Manual test R2:", manual_r2)
print("Sklearn test R2:", sklearn_r2)
print("Mean absolute prediction difference:", mean_abs_diff)

print("Manual weights:", manual_params["weights"])
print("Sklearn weights:", sklearn_model.coef_)

print("Manual bias:", manual_params["bias"])
print("Sklearn bias:", sklearn_model.intercept_)

print("Weights close:", np.allclose(manual_params["weights"], sklearn_model.coef_, atol=1e-3, rtol=1e-3))
print("Bias close:", np.isclose(manual_params["bias"], sklearn_model.intercept_, atol=1e-3, rtol=1e-3))

print("First loss:", manual_model.loss_history[0])
print("Last loss:", manual_model.loss_history[-1])