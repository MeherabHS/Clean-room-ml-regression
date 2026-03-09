import numpy as np
from manual_ml_model import ManualMLModel


# Fixing the seed ensures the synthetic classification problem is reproducible,
# which is necessary for coefficient-level validation against known ground truth.
np.random.seed(42)

# The feature matrix is generated from a standard normal distribution to create
# a controlled binary classification problem with a known latent decision boundary.
X = np.random.randn(200, 3)

# These are the true parameters in the original, unscaled feature space.
true_weights = np.array([1.8, -2.2, 1.1])
true_bias = -0.4

# The latent score is converted into Bernoulli probabilities using the sigmoid link,
# which matches the assumptions of binary logistic regression.
logits = X @ true_weights + true_bias
probabilities = 1.0 / (1.0 + np.exp(-logits))

# Binary labels are sampled from those probabilities so the target generation process
# is statistically consistent with the logistic regression objective.
y = (np.random.rand(200) < probabilities).astype(int)

# The model is trained with internal standardization, so the learned coefficients
# must later be transformed back to raw space for a fair comparison.
model = ManualMLModel(
    mode="logistic",
    learning_rate=0.05,
    n_iter=5000,
    l2_lambda=0.0,
    standardize=True,
    tolerance=1e-10,
    fit_intercept=True,
    random_state=42,
)

# Training produces coefficients in standardized feature coordinates.
model.fit(X, y)

# These learned parameters are extracted so the internal scaling transform can be inverted.
params = model.get_params()
scaled_weights = params["weights"]
scaled_bias = params["bias"]
feature_mean = params["feature_mean_"]
feature_std = params["feature_std_"]

# Reconstructing raw-space coefficients is necessary because direct comparison between
# scaled-space weights and original generating weights would be mathematically invalid.
raw_weights = scaled_weights / feature_std

# The raw-space intercept is adjusted to reverse the centering effect introduced by standardization.
raw_bias = scaled_bias - np.sum((scaled_weights * feature_mean) / feature_std)

# Printing the true and recovered coefficients allows us to judge whether the model
# has learned the underlying decision structure rather than only the transformed representation.
print("True weights:", true_weights)
print("Recovered raw weights:", raw_weights)

# Printing the intercept comparison verifies whether the inverse scaling adjustment
# for the bias term has been applied correctly.
print("True bias:", true_bias)
print("Recovered raw bias:", raw_bias)

# Absolute parameter errors provide a direct discrepancy measure without masking
# directional differences through averaging or cancellation.
print("Weight absolute error:", np.abs(true_weights - raw_weights))
print("Bias absolute error:", abs(true_bias - raw_bias))

# Accuracy is included to preserve the connection between coefficient recovery
# and predictive performance on the same controlled dataset.
print("Accuracy:", model.score(X, y))