import numpy as np


class ManualMLModel:
    """
    A clean-room NumPy-only implementation of Linear and Logistic Regression.

    Supported modes:
    - 'linear'   : Mean Squared Error optimization for regression
    - 'logistic' : Binary Cross-Entropy optimization for classification
    """

    def __init__(
        self,
        mode="linear",
        learning_rate=0.01,
        n_iter=1000,
        l2_lambda=0.0,
        standardize=True,
        tolerance=1e-8,
        fit_intercept=True,
        random_state=None,
    ):
        # Validating the learning mode at initialization prevents undefined mathematical behavior later.
        if mode not in {"linear", "logistic"}:
            raise ValueError("mode must be either 'linear' or 'logistic'")

        # Rejecting non-positive learning rates is necessary because gradient descent requires
        # a strictly positive step size to move along the optimization surface.
        if learning_rate <= 0:
            raise ValueError("learning_rate must be greater than 0")

        # Rejecting non-positive iteration counts prevents a no-op training configuration.
        if n_iter <= 0:
            raise ValueError("n_iter must be greater than 0")

        # L2 regularization strength cannot be negative because that would invert shrinkage logic.
        if l2_lambda < 0:
            raise ValueError("l2_lambda must be >= 0")

        # Tolerance must not be negative because it is used as a convergence threshold.
        if tolerance < 0:
            raise ValueError("tolerance must be >= 0")

        # Storing configuration on the instance keeps the estimator self-contained and reproducible.
        self.mode = mode
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.l2_lambda = l2_lambda
        self.standardize = standardize
        self.tolerance = tolerance
        self.fit_intercept = fit_intercept
        self.random_state = random_state

        # Initializing learned parameters as None makes it easy to detect premature prediction calls.
        self.weights = None
        self.bias = 0.0

        # These scaling statistics are learned from training data and then reused at inference time.
        self.feature_mean_ = None
        self.feature_std_ = None

        # Loss history is tracked to support convergence diagnostics and learning-curve inspection.
        self.loss_history = []

        # This state flag prevents prediction before training has occurred.
        self.is_fitted_ = False

        # Setting the NumPy random seed makes experiments reproducible across runs.
        if self.random_state is not None:
            np.random.seed(self.random_state)

    def _validate_inputs(self, X, y=None):
        # Enforcing NumPy arrays keeps all downstream linear algebra predictable and vectorized.
        if not isinstance(X, np.ndarray):
            raise TypeError("X must be a NumPy ndarray")

        # The design matrix must be 2D because the model expects shape (n_samples, n_features).
        if X.ndim != 2:
            raise ValueError("X must be a 2D array of shape (n_samples, n_features)")

        if y is not None:
            # The target vector must also be a NumPy array for consistent numerical operations.
            if not isinstance(y, np.ndarray):
                raise TypeError("y must be a NumPy ndarray")

            # The target must be 1D because this implementation supports a single target variable only.
            if y.ndim != 1:
                raise ValueError("y must be a 1D array of shape (n_samples,)")

            # The number of rows in X must match the number of labels in y for valid gradient computation.
            if X.shape[0] != y.shape[0]:
                raise ValueError(
                    f"Dimensional misalignment: X has {X.shape[0]} rows but y has {y.shape[0]} values"
                )

            # Logistic mode is restricted to binary classification because multinomial logic
            # would require a different probability model and gradient derivation.
            if self.mode == "logistic":
                unique_values = np.unique(y)
                if not np.all(np.isin(unique_values, [0, 1])):
                    raise ValueError("For logistic mode, y must contain only binary values {0, 1}")

    def _compute_scaling_params(self, X):
        # Computing column-wise mean and standard deviation enables standardization inside the model,
        # which avoids preprocessing leakage and keeps inference consistent with training.
        self.feature_mean_ = np.mean(X, axis=0)
        self.feature_std_ = np.std(X, axis=0)

        # Zero-variance features would cause division by zero, so they are neutralized by replacing
        # zero standard deviations with 1.0 while preserving the centered representation.
        self.feature_std_ = np.where(self.feature_std_ == 0, 1.0, self.feature_std_)

    def _scale_features(self, X, fit=False):
        # Allowing scaling to be disabled keeps the class flexible for raw-feature experiments.
        if not self.standardize:
            return X

        # During training, the scaler parameters must be learned from the training set only.
        if fit:
            self._compute_scaling_params(X)

        # Scaling cannot be applied unless the training statistics have already been established.
        if self.feature_mean_ is None or self.feature_std_ is None:
            raise ValueError("Scaling parameters are not initialized. Call fit first.")

        # Standardization improves optimization conditioning by normalizing feature magnitudes.
        return (X - self.feature_mean_) / self.feature_std_

    def _initialize_parameters(self, n_features):
        # Zero initialization is mathematically safe here because linear and logistic regression
        # are convex problems and do not suffer from symmetry issues like deep networks do.
        self.weights = np.zeros(n_features, dtype=np.float64)
        self.bias = 0.0

    def _linear_output(self, X):
        # Both model types share the same affine transformation before applying their task-specific logic.
        if self.weights is None:
            raise ValueError("Model parameters are not initialized")
        return np.dot(X, self.weights) + (self.bias if self.fit_intercept else 0.0)

    def _sigmoid(self, z):
        # Clipping protects the exponential operation from overflow during logistic optimization.
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def _compute_loss(self, X, y):
        # The sample count is required for normalized empirical risk and regularization scaling.
        m = X.shape[0]

        # The affine output is reused to avoid duplicate matrix multiplication.
        linear_term = self._linear_output(X)

        # L2 regularization penalizes large weights to reduce overfitting, but the bias term is excluded.
        l2_penalty = (self.l2_lambda / (2 * m)) * np.sum(self.weights ** 2)

        if self.mode == "linear":
            # Mean Squared Error is the appropriate convex loss for continuous regression targets.
            predictions = linear_term
            residuals = predictions - y
            mse = (1.0 / (2 * m)) * np.sum(residuals ** 2)
            return mse + l2_penalty

        # Logistic regression converts the linear score into class probabilities using the sigmoid function.
        probabilities = self._sigmoid(linear_term)

        # Probabilities are clipped away from 0 and 1 to prevent undefined logarithms in cross-entropy.
        probabilities = np.clip(probabilities, 1e-15, 1 - 1e-15)

        # Binary Cross-Entropy is the correct loss for Bernoulli-distributed target labels.
        cross_entropy = -np.mean(
            y * np.log(probabilities) + (1 - y) * np.log(1 - probabilities)
        )
        return cross_entropy + l2_penalty

    def fit(self, X, y):
        # Input validation is performed first so the optimizer never runs on malformed data.
        self._validate_inputs(X, y)

        # Converting inputs to float64 improves numerical stability for gradient-based optimization.
        X = X.astype(np.float64)
        y = y.astype(np.float64)

        # Training-time scaling learns statistics from the current training set and applies them consistently.
        X_processed = self._scale_features(X, fit=True)

        # Parameter vectors must match the dimensionality of the processed feature matrix exactly.
        n_samples, n_features = X_processed.shape
        self._initialize_parameters(n_features)

        # Clearing previous loss history ensures repeated fit calls do not contaminate diagnostics.
        self.loss_history = []

        # Initializing with infinity allows the first convergence comparison to proceed safely.
        previous_loss = np.inf

        for _ in range(self.n_iter):
            # The shared affine transformation is computed once per iteration for efficiency.
            linear_term = self._linear_output(X_processed)

            if self.mode == "linear":
                # For MSE, the derivative with respect to predictions is simply the residual vector.
                predictions = linear_term
                errors = predictions - y
            else:
                # For logistic regression, the gradient simplifies to predicted probabilities minus labels.
                predictions = self._sigmoid(linear_term)
                errors = predictions - y

            # This vectorized gradient is the core matrix-calculus update and avoids weak loop-based updates.
            dw = (1.0 / n_samples) * np.dot(X_processed.T, errors)

            # L2 regularization adds shrinkage to the weight gradient but not to the intercept term.
            dw += (self.l2_lambda / n_samples) * self.weights

            # The intercept gradient is the mean residual because the bias affects every sample equally.
            db = (1.0 / n_samples) * np.sum(errors) if self.fit_intercept else 0.0

            # Gradient descent updates parameters in the direction opposite to the gradient.
            self.weights -= self.learning_rate * dw
            if self.fit_intercept:
                self.bias -= self.learning_rate * db

            # Tracking the post-update loss allows later inspection of convergence quality.
            current_loss = self._compute_loss(X_processed, y)
            self.loss_history.append(current_loss)

            # Early stopping prevents wasteful iterations once improvement becomes numerically negligible.
            if abs(previous_loss - current_loss) < self.tolerance:
                break

            previous_loss = current_loss

        # Marking the estimator as fitted prevents accidental inference on untrained parameters.
        self.is_fitted_ = True
        return self

    def predict_proba(self, X):
        # Probability output is only mathematically meaningful for logistic regression.
        if self.mode != "logistic":
            raise ValueError("predict_proba is only available when mode='logistic'")

        # Preventing inference before training preserves API correctness and avoids undefined outputs.
        if not self.is_fitted_:
            raise ValueError("Model is not fitted. Call fit before predict_proba.")

        # Input validation is repeated at inference boundaries to maintain defensive behavior.
        self._validate_inputs(X)

        # Applying the stored training scaler preserves feature representation consistency.
        X = X.astype(np.float64)
        X_processed = self._scale_features(X, fit=False)

        # The positive-class probability is derived from the sigmoid of the affine score.
        prob_class_1 = self._sigmoid(self._linear_output(X_processed))

        # Returning both class probabilities matches common classifier interface expectations.
        return np.column_stack((1 - prob_class_1, prob_class_1))

    def predict(self, X):
        # Prediction requires a fitted model because weights and scaling statistics are learned state.
        if not self.is_fitted_:
            raise ValueError("Model is not fitted. Call fit before predict.")

        # Input validation is required here as well because prediction is a public API boundary.
        self._validate_inputs(X)

        # Test features must be transformed with the same numeric conventions as the training data.
        X = X.astype(np.float64)
        X_processed = self._scale_features(X, fit=False)

        if self.mode == "linear":
            # Regression returns the direct affine output as the continuous prediction.
            return self._linear_output(X_processed)

        # Classification converts estimated probabilities into hard labels using a 0.5 decision threshold.
        probabilities = self._sigmoid(self._linear_output(X_processed))
        return (probabilities >= 0.5).astype(int)

    def score(self, X, y):
        # Scoring validates both features and targets to ensure metric computation is dimensionally valid.
        self._validate_inputs(X, y)

        # Predictions are generated through the public inference path to preserve consistency.
        y_pred = self.predict(X)

        if self.mode == "linear":
            # R-squared measures the proportion of variance explained and is the standard regression score.
            y_mean = np.mean(y)
            ss_total = np.sum((y - y_mean) ** 2)
            ss_residual = np.sum((y - y_pred) ** 2)

            # If the target variance is zero, R-squared becomes ill-defined, so the function fails explicitly.
            if ss_total == 0:
                raise ValueError("R^2 is undefined because y has zero variance")

            return 1.0 - (ss_residual / ss_total)

        # Accuracy is used for binary classification because it is the default sklearn-compatible score.
        return np.mean(y_pred == y)

    def get_params(self):
        # Learned parameters are exposed for benchmark comparison and interpretability checks.
        if not self.is_fitted_:
            raise ValueError("Model is not fitted. Call fit before get_params.")

        # Copies are returned to prevent external mutation of internal model state.
        return {
            "weights": self.weights.copy(),
            "bias": float(self.bias),
            "feature_mean_": None if self.feature_mean_ is None else self.feature_mean_.copy(),
            "feature_std_": None if self.feature_std_ is None else self.feature_std_.copy(),
            "loss_history": self.loss_history.copy(),
        }