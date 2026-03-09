import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from manual_ml_model import ManualMLModel


# Reproducibility ensures the same convergence curve is generated
# whenever the report figure is regenerated.
np.random.seed(42)


# Synthetic regression dataset provides a controlled optimization
# environment suitable for demonstrating convergence behavior.
X, y = make_regression(
    n_samples=300,
    n_features=5,
    noise=10.0,
    random_state=42,
)


# Model configuration mirrors the earlier convergence experiment
# so the visualization corresponds to previously validated behavior.
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


# Training populates loss_history which represents the optimization path.
model.fit(X, y)

iterations = np.arange(1, len(model.loss_history) + 1)


# Creating a larger, report-ready figure improves readability
# when exported to slides or documents.
plt.figure(figsize=(12, 7))


# Log scaling is used so both early large losses and late fine convergence
# remain visible in the same plot.
plt.plot(iterations, model.loss_history, linewidth=2)


# Titles and labels are written formally so the figure can be reused
# directly in academic documentation.
plt.title("Gradient Descent Convergence (Manual Linear Regression)", fontsize=14)
plt.xlabel("Iteration")
plt.ylabel("Training Loss")


# Log-scale improves interpretability of the optimization trajectory.
plt.yscale("log")


# Grid improves readability when interpreting convergence slope.
plt.grid(True, alpha=0.3)


# Tight layout prevents label clipping in exported figures.
plt.tight_layout()


# Save the figure for report inclusion.
plt.savefig("training_convergence_plot.png", dpi=300)


plt.show()


# Diagnostic printout confirms convergence numerically
# alongside the visual evidence.
print("Iterations:", len(model.loss_history))
print("Initial loss:", model.loss_history[0])
print("Final loss:", model.loss_history[-1])