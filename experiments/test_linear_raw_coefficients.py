import numpy as np
from manual_ml_model import ManualMLModel


# Setting a fixed seed keeps the synthetic experiment deterministic,
# which is necessary when validating whether recovered coefficients are mathematically consistent.
np.random.seed(42)

# This dataset is generated from a known linear process so the recovered raw-space parameters
# can be compared against the ground-truth coefficients in a controlled setting.
X = np.random.randn(100, 3)

# These are the true coefficients in the original, unscaled feature space.
true_weights = np.array([2.5, -1.7, 0.9])
true_bias = 4.2
noise = np.random.randn(100) * 0.5
y = X @ true_weights + true_bias + noise

# The manual model is trained with internal standardization because that is part of the architecture,
# but the learned coefficients must later be transformed back to raw space for a fair comparison.
model = ManualMLModel(
    mode="linear",
    learning_rate=0.05,
    n_iter=3000,
    l2_lambda=0.0,
    standardize=True,
    tolerance=1e-10,
    fit_intercept=True,
    random_state=42,
)

# Training here produces coefficients in standardized feature space rather than raw feature space.
model.fit(X, y)

# Extracting the learned internal parameters is required so they can be converted back
# into the original feature scale used to generate the target variable.
params = model.get_params()
scaled_weights = params["weights"]
scaled_bias = params["bias"]
feature_mean = params["feature_mean_"]
feature_std = params["feature_std_"]

# Reversing the standardization effect yields coefficients in the raw feature space,
# which is the only valid basis for comparing against the true generating weights.
raw_weights = scaled_weights / feature_std

# The raw-space intercept must be adjusted for feature centering,
# otherwise the coefficient comparison would be mathematically incomplete.
raw_bias = scaled_bias - np.sum((scaled_weights * feature_mean) / feature_std)

# Printing both true and recovered coefficients makes it possible to judge whether
# the optimizer learned the underlying linear process rather than only the scaled representation.
print("True weights:", true_weights)
print("Recovered raw weights:", raw_weights)

# Printing both true and recovered intercepts verifies whether centering reversal
# was performed correctly when reconstructing the original-space bias term.
print("True bias:", true_bias)
print("Recovered raw bias:", raw_bias)

# Absolute error is shown because it gives a direct parameter-level discrepancy measure
# without hiding differences through sign cancellation.
print("Weight absolute error:", np.abs(true_weights - raw_weights))
print("Bias absolute error:", abs(true_bias - raw_bias))

# The training score is included again to preserve the connection between parameter recovery
# and predictive performance on the same controlled dataset.
print("R2 score:", model.score(X, y))