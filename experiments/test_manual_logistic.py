import numpy as np
from manual_ml_model import ManualMLModel


# Fixing the random seed keeps the classification dataset reproducible,
# which is necessary for stable debugging of optimization behavior.
np.random.seed(42)

# This synthetic binary dataset is generated from a known logistic decision process
# so the manual classifier can be evaluated under controlled conditions.
X = np.random.randn(200, 3)

# These raw-space parameters define the latent linear decision boundary before sigmoid transformation.
true_weights = np.array([1.8, -2.2, 1.1])
true_bias = -0.4

# The linear score is converted into class probabilities using the logistic link function,
# which matches the mathematical assumptions of binary logistic regression.
logits = X @ true_weights + true_bias
probabilities = 1.0 / (1.0 + np.exp(-logits))

# Binary labels are sampled from the Bernoulli probabilities so the target distribution
# is consistent with the cross-entropy objective used by the model.
y = (np.random.rand(200) < probabilities).astype(int)

# The model is configured with a conservative learning rate and sufficient iterations
# to allow stable convergence of the binary cross-entropy objective.
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

# Fitting this model validates the logistic branch of the architecture:
# scaling, sigmoid activation, cross-entropy gradient, and convergence tracking.
model.fit(X, y)

# Hard-label prediction validates the classification decision path.
predictions = model.predict(X)

# Probability prediction validates the probabilistic inference path and class-probability formatting.
proba = model.predict_proba(X)

# Extracting learned parameters supports later inspection and debugging if optimization is weak.
params = model.get_params()

# Printing the first and last loss values verifies whether the optimizer is reducing
# the binary cross-entropy objective over time.
print("First loss:", model.loss_history[0])
print("Last loss:", model.loss_history[-1])

# Accuracy on this learnable synthetic dataset should be meaningfully above chance
# if the gradient and sigmoid implementation are correct.
print("Accuracy:", model.score(X, y))

# Printing learned internal coefficients helps inspect whether the model has found
# a directionally sensible separating hyperplane.
print("Learned weights:", params["weights"])
print("Learned bias:", params["bias"])

# A small sample of class predictions helps verify that the output is binary as expected.
print("Prediction sample:", predictions[:10])

# A small sample of probability rows helps verify that predict_proba returns
# two columns and valid probabilities in the [0, 1] interval.
print("Probability sample:", proba[:5])

# Checking probability row sums is a defensive sanity test to ensure the returned
# class probabilities are properly normalized.
print("Probability row sums:", np.sum(proba[:5], axis=1))