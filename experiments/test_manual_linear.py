import numpy as np
from manual_ml_model import ManualMLModel


# Setting a fixed seed makes the synthetic regression experiment reproducible,
# which is necessary for stable debugging and consistent architecture validation.
np.random.seed(42)

# This synthetic dataset is intentionally simple because the goal of this file
# is to verify training mechanics, not to stress-test edge-case generalization.
X = np.random.randn(100, 3)

# The target is generated from a known linear relationship plus mild noise
# so we can verify whether gradient descent recovers a sensible fit.
true_weights = np.array([2.5, -1.7, 0.9])
true_bias = 4.2
noise = np.random.randn(100) * 0.5
y = X @ true_weights + true_bias + noise

# The model is configured with enough iterations and a moderate learning rate
# to give gradient descent a realistic chance to converge on this small dataset.
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

# Fitting here validates the entire regression training path:
# scaling, parameter initialization, vectorized gradients, and loss tracking.
model.fit(X, y)

# Predictions on the same dataset are acceptable for this first mechanics check
# because this file is verifying implementation correctness, not generalization quality.
predictions = model.predict(X)

# Extracting learned parameters helps us inspect whether the optimizer
# is moving toward the known underlying linear signal.
params = model.get_params()

# Printing loss endpoints verifies whether the optimization trajectory is decreasing,
# which is the first indicator that the gradient implementation is behaving correctly.
print("First loss:", model.loss_history[0])
print("Last loss:", model.loss_history[-1])

# Printing score allows us to verify whether the trained model explains
# most of the variance in this intentionally learnable synthetic dataset.
print("R2 score:", model.score(X, y))

# Printing learned parameters helps diagnose whether the solution is directionally correct,
# even though exact recovery is not expected because noise is present.
print("Learned weights:", params["weights"])
print("Learned bias:", params["bias"])

# Printing a small prediction sample provides a quick sanity check
# that inference returns numeric regression outputs of the expected shape.
print("Prediction sample:", predictions[:5])