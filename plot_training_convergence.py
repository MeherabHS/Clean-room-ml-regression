import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from manual_ml_model import ManualMLModel


# Fixing the seed makes the plotted convergence behavior reproducible,
# which is necessary if the figure is later referenced in a report or viva discussion.
np.random.seed(42)

# A synthetic regression dataset is used here because the optimization path should be easy to interpret,
# making it suitable for a first convergence plot before creating polished report-ready visuals.
X, y = make_regression(
    n_samples=300,
    n_features=5,
    noise=10.0,
    random_state=42,
)

# The model is configured with a stable learning rate and enough iterations
# so the loss curve has a meaningful descent trajectory to visualize.
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

# Training is required here because loss_history is populated only during optimization,
# and the resulting sequence is the primary signal we want to visualize.
model.fit(X, y)

# The iteration index is created explicitly so the x-axis represents optimization progress
# rather than relying on implicit plotting behavior.
iterations = np.arange(1, len(model.loss_history) + 1)

# A single clean figure is used because convergence analysis is clearest when the loss trajectory
# is not mixed with unrelated plots or subplots.
plt.figure(figsize=(10, 6))

# The loss curve directly visualizes whether gradient descent is reducing the objective over time,
# which is a core validation signal for first-principles optimization code.
plt.plot(iterations, model.loss_history, linewidth=2, label="Training Loss")

# Titles and axis labels are included because the figure is intended to be interpretable
# as standalone evidence in a project report or presentation.
plt.title("Training Convergence of Manual Linear Regression")
plt.xlabel("Iteration")
plt.ylabel("Loss")

# A light grid improves readability of the optimization path without changing the mathematical content.
plt.grid(True, alpha=0.3)

# A legend is included so the plot remains extensible if later versions add benchmark or smoothed curves.
plt.legend()

# Tight layout reduces clipping risk when the figure is reused in slides or documents.
plt.tight_layout()

# Showing the plot is appropriate at this stage because the immediate goal is runtime verification
# before we move to a more polished, report-ready version.
plt.show()

# Printing summary diagnostics alongside the plot provides a quick textual confirmation
# that the convergence curve corresponds to an actual reduction in objective value.
print("Loss history length:", len(model.loss_history))
print("First loss:", model.loss_history[0])
print("Last loss:", model.loss_history[-1])