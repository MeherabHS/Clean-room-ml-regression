import numpy as np
from manual_ml_model import ManualMLModel


# Fixing the seed keeps the larger-sample experiment reproducible,
# which is necessary when judging whether coefficient recovery improves for principled reasons.
np.random.seed(42)

# Increasing the sample size reduces parameter variance in logistic estimation,
# making this a stronger test of whether the implementation matches the generating process.
X = np.random.randn(5000, 3)

# These are the ground-truth coefficients in the original feature space.
true_weights = np.array([1.8, -2.2, 1.1])
true_bias = -0.4

# The latent linear score is mapped through the sigmoid function to produce valid Bernoulli probabilities.
logits = X @ true_weights + true_bias
probabilities = 1.0 / (1.0 + np.exp(-logits))

# Labels are sampled from the Bernoulli distribution implied by those probabilities,
# preserving the correct statistical structure of binary logistic regression.
y = (np.random.rand(5000) < probabilities).astype(int)

# The manual model is kept architecturally identical so the experiment isolates
# the effect of sample size rather than introducing confounding implementation changes.
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

# Training on a larger sample provides a stronger basis for judging coefficient recovery quality.
model.fit(X, y)

# The learned scaled-space parameters are extracted so they can be transformed back
# to the original feature coordinates for fair comparison against the ground truth.
params = model.get_params()
scaled_weights = params["weights"]
scaled_bias = params["bias"]
feature_mean = params["feature_mean_"]
feature_std = params["feature_std_"]

# Reversing the feature scaling converts the learned logistic coefficients back to raw space,
# which is the only mathematically valid comparison basis.
raw_weights = scaled_weights / feature_std

# Reconstructing the raw-space intercept accounts for the centering applied during standardization.
raw_bias = scaled_bias - np.sum((scaled_weights * feature_mean) / feature_std)

# Printing both true and recovered coefficients allows direct inspection of whether
# coefficient recovery improves under a more statistically stable sample size.
print("True weights:", true_weights)
print("Recovered raw weights:", raw_weights)

# Printing the intercept comparison verifies whether the raw-space bias estimate
# also moves closer to the generating process under the larger sample.
print("True bias:", true_bias)
print("Recovered raw bias:", raw_bias)

# Absolute errors provide a direct, interpretable diagnostic of recovery quality.
print("Weight absolute error:", np.abs(true_weights - raw_weights))
print("Bias absolute error:", abs(true_bias - raw_bias))

# Accuracy is retained because strong parameter recovery should still correspond
# to sound predictive behavior on the generated dataset.
print("Accuracy:", model.score(X, y))

# Loss endpoints are printed again to confirm that the optimization still converges cleanly
# at the larger sample size.
print("First loss:", model.loss_history[0])
print("Last loss:", model.loss_history[-1])