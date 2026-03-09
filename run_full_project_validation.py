import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_regression, load_breast_cancer, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression

from manual_ml_model import ManualMLModel


# Global seed guarantees deterministic behavior across all experiments
np.random.seed(42)


# -----------------------------------------------------------
# Synthetic Linear Regression Validation
# -----------------------------------------------------------
def synthetic_linear_validation():
    print("\n--- Synthetic Linear Regression Validation ---")

    X, y = make_regression(
        n_samples=300,
        n_features=5,
        noise=10.0,
        random_state=42,
    )

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

    model.fit(X, y)

    print("Iterations:", len(model.loss_history))
    print("First loss:", model.loss_history[0])
    print("Last loss:", model.loss_history[-1])
    print("Training R2:", model.score(X, y))

    return model


# -----------------------------------------------------------
# Synthetic Logistic Regression Validation
# -----------------------------------------------------------
def synthetic_logistic_validation():
    print("\n--- Synthetic Logistic Regression Validation ---")

    X = np.random.randn(2000, 3)

    true_weights = np.array([1.8, -2.2, 1.1])
    true_bias = -0.4

    logits = X @ true_weights + true_bias
    probs = 1 / (1 + np.exp(-logits))

    y = (np.random.rand(2000) < probs).astype(int)

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

    model.fit(X, y)

    print("Iterations:", len(model.loss_history))
    print("First loss:", model.loss_history[0])
    print("Last loss:", model.loss_history[-1])
    print("Accuracy:", model.score(X, y))

    return model


# -----------------------------------------------------------
# Sklearn Linear Benchmark
# -----------------------------------------------------------
def sklearn_linear_benchmark():
    print("\n--- Sklearn Linear Benchmark ---")

    X, y = make_regression(
        n_samples=400,
        n_features=4,
        noise=10,
        random_state=42
    )

    manual = ManualMLModel(
        mode="linear",
        learning_rate=0.05,
        n_iter=4000,
        standardize=True,
        random_state=42
    )

    manual.fit(X, y)

    X_scaled = manual._scale_features(X, fit=False)

    sklearn_model = LinearRegression()
    sklearn_model.fit(X_scaled, y)

    manual_pred = manual.predict(X)
    sklearn_pred = sklearn_model.predict(X_scaled)

    diff = np.mean(np.abs(manual_pred - sklearn_pred))

    print("Manual R2:", manual.score(X, y))
    print("Sklearn R2:", sklearn_model.score(X_scaled, y))
    print("Mean prediction difference:", diff)


# -----------------------------------------------------------
# Sklearn Logistic Benchmark
# -----------------------------------------------------------
def sklearn_logistic_benchmark():
    print("\n--- Sklearn Logistic Benchmark ---")

    X = np.random.randn(4000, 3)

    true_w = np.array([1.8, -2.2, 1.1])
    logits = X @ true_w
    probs = 1/(1+np.exp(-logits))

    y = (np.random.rand(4000) < probs).astype(int)

    manual = ManualMLModel(
        mode="logistic",
        learning_rate=0.05,
        n_iter=5000,
        standardize=True,
        random_state=42
    )

    manual.fit(X, y)

    X_scaled = manual._scale_features(X, fit=False)

    sklearn_model = LogisticRegression(
        C=np.inf,
        solver="lbfgs",
        max_iter=10000
    )

    sklearn_model.fit(X_scaled, y)

    print("Manual accuracy:", manual.score(X, y))
    print("Sklearn accuracy:", sklearn_model.score(X_scaled, y))


# -----------------------------------------------------------
# Real Dataset Linear Experiment
# -----------------------------------------------------------
def real_linear_experiment():
    print("\n--- Real Dataset Linear Experiment ---")

    data = fetch_california_housing()

    X_train, X_test, y_train, y_test = train_test_split(
        data.data,
        data.target,
        test_size=0.2,
        random_state=42
    )

    manual = ManualMLModel(
        mode="linear",
        learning_rate=0.05,
        n_iter=5000,
        standardize=True,
        random_state=42
    )

    manual.fit(X_train, y_train)

    X_train_scaled = manual._scale_features(X_train, fit=False)
    X_test_scaled = manual._scale_features(X_test, fit=False)

    sklearn_model = LinearRegression()
    sklearn_model.fit(X_train_scaled, y_train)

    print("Manual Test R2:", manual.score(X_test, y_test))
    print("Sklearn Test R2:", sklearn_model.score(X_test_scaled, y_test))


# -----------------------------------------------------------
# Real Dataset Logistic Experiment
# -----------------------------------------------------------
def real_logistic_experiment():
    print("\n--- Real Dataset Logistic Experiment ---")

    data = load_breast_cancer()

    X_train, X_test, y_train, y_test = train_test_split(
        data.data,
        data.target,
        test_size=0.2,
        stratify=data.target,
        random_state=42
    )

    manual = ManualMLModel(
        mode="logistic",
        learning_rate=0.05,
        n_iter=5000,
        l2_lambda=1.0,
        standardize=True,
        random_state=42
    )

    manual.fit(X_train, y_train)

    X_train_scaled = manual._scale_features(X_train, fit=False)
    X_test_scaled = manual._scale_features(X_test, fit=False)

    sklearn_model = LogisticRegression(
        C=1.0,
        solver="lbfgs",
        max_iter=10000
    )

    sklearn_model.fit(X_train_scaled, y_train)

    print("Manual Test Accuracy:", manual.score(X_test, y_test))
    print("Sklearn Test Accuracy:", sklearn_model.score(X_test_scaled, y_test))


# -----------------------------------------------------------
# Convergence Visualization
# -----------------------------------------------------------
def convergence_plot():
    print("\n--- Convergence Plot Generation ---")

    X, y = make_regression(
        n_samples=300,
        n_features=5,
        noise=10,
        random_state=42
    )

    model = ManualMLModel(
        mode="linear",
        learning_rate=0.05,
        n_iter=3000,
        standardize=True,
        random_state=42
    )

    model.fit(X, y)

    plt.figure(figsize=(10,6))
    plt.plot(model.loss_history)
    plt.yscale("log")
    plt.title("Gradient Descent Convergence")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.grid(True)

    plt.savefig("convergence_plot.png", dpi=300)
    plt.show()


# -----------------------------------------------------------
# Main Execution Pipeline
# -----------------------------------------------------------
def main():

    synthetic_linear_validation()
    synthetic_logistic_validation()

    sklearn_linear_benchmark()
    sklearn_logistic_benchmark()

    real_linear_experiment()
    real_logistic_experiment()

    convergence_plot()

    print("\n--- FULL PROJECT VALIDATION COMPLETE ---")


if __name__ == "__main__":
    main()