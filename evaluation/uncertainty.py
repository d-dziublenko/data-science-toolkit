"""
evaluation/uncertainty.py
Uncertainty quantification methods for machine learning predictions.
"""

import logging
import warnings
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm, t
from sklearn.base import BaseEstimator, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import KFold
from sklearn.utils import resample
from tqdm import tqdm

logger = logging.getLogger(__name__)


class UncertaintyEstimator(ABC):
    """Abstract base class for uncertainty estimation methods."""

    @abstractmethod
    def fit(self, *args, **kwargs):
        """Fit the uncertainty estimator."""
        pass

    @abstractmethod
    def predict_uncertainty(self, *args, **kwargs) -> Dict[str, np.ndarray]:
        """Predict with uncertainty estimates."""
        pass


class BootstrapUncertainty(UncertaintyEstimator):
    """
    Bootstrap-based uncertainty estimation.

    This class implements bootstrap resampling to estimate prediction uncertainty
    by training multiple models on different bootstrap samples of the data.
    """

    def __init__(
        self,
        base_estimator: BaseEstimator,
        n_bootstrap: int = 100,
        sample_fraction: float = 1.0,
        random_state: Optional[int] = None,
    ):
        """
        Initialize Bootstrap uncertainty estimator.

        Args:
            base_estimator: Base model to use (will be cloned)
            n_bootstrap: Number of bootstrap iterations
            sample_fraction: Fraction of data to use in each bootstrap sample
            random_state: Random seed for reproducibility
        """
        self.base_estimator = base_estimator
        self.n_bootstrap = n_bootstrap
        self.sample_fraction = sample_fraction
        self.random_state = random_state
        self.estimators_ = []
        self.is_fitted_ = False

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fit bootstrap models.

        Args:
            X: Training features
            y: Training targets
        """
        logger.info(f"Fitting {self.n_bootstrap} bootstrap models")

        n_samples = len(X)
        sample_size = int(self.sample_fraction * n_samples)

        # Set random state
        if self.random_state is not None:
            np.random.seed(self.random_state)

        # Train bootstrap models
        for i in tqdm(range(self.n_bootstrap), desc="Bootstrap fitting"):
            # Create bootstrap sample
            indices = np.random.choice(n_samples, size=sample_size, replace=True)
            X_bootstrap = X.iloc[indices]
            y_bootstrap = y.iloc[indices]

            # Clone and train model
            model = clone(self.base_estimator)
            model.fit(X_bootstrap, y_bootstrap)
            self.estimators_.append(model)

        self.is_fitted_ = True
        return self

    def predict_uncertainty(
        self, X: pd.DataFrame, confidence_level: float = 0.95
    ) -> Dict[str, np.ndarray]:
        """
        Make predictions with uncertainty estimates.

        Args:
            X: Features to predict on
            confidence_level: Confidence level for intervals

        Returns:
            Dictionary with predictions and uncertainty estimates
        """
        if not self.is_fitted_:
            raise ValueError("Must fit before predicting")

        # Get predictions from all models
        predictions = []
        for model in self.estimators_:
            pred = model.predict(X)
            predictions.append(pred)

        predictions = np.array(predictions)

        # Calculate statistics
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)

        # Calculate confidence intervals
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100

        lower_bound = np.percentile(predictions, lower_percentile, axis=0)
        upper_bound = np.percentile(predictions, upper_percentile, axis=0)

        return {
            "predictions": mean_pred,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "std": std_pred,
            "all_predictions": predictions,
        }


class BayesianUncertainty(UncertaintyEstimator):
    """
    Bayesian approach to uncertainty estimation.

    This class implements Bayesian uncertainty estimation using either
    variational inference or MCMC sampling, depending on the model type.
    """

    def __init__(
        self,
        model_type: str = "linear",
        prior_params: Optional[Dict[str, Any]] = None,
        n_samples: int = 1000,
    ):
        """
        Initialize Bayesian uncertainty estimator.

        Args:
            model_type: Type of Bayesian model ('linear', 'neural', 'gaussian_process')
            prior_params: Parameters for prior distributions
            n_samples: Number of posterior samples
        """
        self.model_type = model_type
        self.prior_params = prior_params or {}
        self.n_samples = n_samples
        self.posterior_samples_ = None
        self.is_fitted_ = False

        # Initialize model based on type
        self._init_model()

    def _init_model(self):
        """Initialize the specific Bayesian model."""
        if self.model_type == "linear":
            # Bayesian linear regression
            self.alpha_ = self.prior_params.get("alpha", 1.0)  # Precision of weights
            self.beta_ = self.prior_params.get("beta", 1.0)  # Precision of noise
        elif self.model_type == "gaussian_process":
            # Import here to avoid dependency if not used
            try:
                from sklearn.gaussian_process import GaussianProcessRegressor
                from sklearn.gaussian_process.kernels import RBF, WhiteKernel

                kernel = RBF() + WhiteKernel()
                self.model = GaussianProcessRegressor(
                    kernel=kernel,
                    n_restarts_optimizer=self.prior_params.get("n_restarts", 5),
                    alpha=self.prior_params.get("noise_level", 1e-6),
                )
            except ImportError:
                raise ImportError(
                    "Gaussian Process requires scikit-learn with GP support"
                )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fit the Bayesian model.

        Args:
            X: Training features
            y: Training targets
        """
        logger.info(f"Fitting Bayesian {self.model_type} model")

        if self.model_type == "linear":
            # Bayesian linear regression with conjugate priors
            X_array = X.values
            y_array = y.values

            # Add bias term
            X_with_bias = np.column_stack([np.ones(len(X)), X_array])

            # Posterior parameters (conjugate update)
            S_inv = (
                self.alpha_ * np.eye(X_with_bias.shape[1])
                + self.beta_ * X_with_bias.T @ X_with_bias
            )
            S = np.linalg.inv(S_inv)
            m = self.beta_ * S @ X_with_bias.T @ y_array

            # Store posterior parameters
            self.posterior_mean_ = m
            self.posterior_cov_ = S

            # Sample from posterior
            self.posterior_samples_ = np.random.multivariate_normal(
                self.posterior_mean_, self.posterior_cov_, size=self.n_samples
            )

        elif self.model_type == "gaussian_process":
            self.model.fit(X, y)

        self.is_fitted_ = True
        return self

    def predict_uncertainty(
        self, X: pd.DataFrame, confidence_level: float = 0.95
    ) -> Dict[str, np.ndarray]:
        """
        Make predictions with Bayesian uncertainty estimates.

        Args:
            X: Features to predict on
            confidence_level: Confidence level for intervals

        Returns:
            Dictionary with predictions and uncertainty estimates
        """
        if not self.is_fitted_:
            raise ValueError("Must fit before predicting")

        if self.model_type == "linear":
            X_array = X.values
            X_with_bias = np.column_stack([np.ones(len(X)), X_array])

            # Predictive distribution
            predictions = []
            for weights in self.posterior_samples_:
                pred = X_with_bias @ weights
                # Add observation noise
                pred += np.random.normal(0, 1 / np.sqrt(self.beta_), size=len(pred))
                predictions.append(pred)

            predictions = np.array(predictions)

            # Calculate statistics
            mean_pred = np.mean(predictions, axis=0)
            std_pred = np.std(predictions, axis=0)

            # Confidence intervals
            alpha = 1 - confidence_level
            lower_bound = np.percentile(predictions, (alpha / 2) * 100, axis=0)
            upper_bound = np.percentile(predictions, (1 - alpha / 2) * 100, axis=0)

        elif self.model_type == "gaussian_process":
            mean_pred, std_pred = self.model.predict(X, return_std=True)

            # Confidence intervals (assuming normal distribution)
            z_score = norm.ppf(1 - (1 - confidence_level) / 2)
            lower_bound = mean_pred - z_score * std_pred
            upper_bound = mean_pred + z_score * std_pred

            predictions = None  # GP doesn't generate samples by default

        return {
            "predictions": mean_pred,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "std": std_pred,
            "epistemic_uncertainty": std_pred,  # Bayesian methods capture epistemic uncertainty
            "all_predictions": predictions,
        }


class EnsembleUncertainty(UncertaintyEstimator):
    """
    Ensemble-based uncertainty estimation.

    This class uses an ensemble of diverse models to estimate uncertainty
    through their disagreement.
    """

    def __init__(
        self,
        estimators: List[BaseEstimator],
        voting: str = "soft",
        weights: Optional[List[float]] = None,
    ):
        """
        Initialize Ensemble uncertainty estimator.

        Args:
            estimators: List of base estimators
            voting: Voting type ('soft' for averaging, 'hard' for majority)
            weights: Optional weights for each estimator
        """
        self.estimators = estimators
        self.voting = voting
        self.weights = weights
        self.is_fitted_ = False

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fit all ensemble members.

        Args:
            X: Training features
            y: Training targets
        """
        logger.info(f"Fitting ensemble of {len(self.estimators)} models")

        for i, estimator in enumerate(self.estimators):
            logger.debug(
                f"Fitting model {i+1}/{len(self.estimators)}: {type(estimator).__name__}"
            )
            estimator.fit(X, y)

        self.is_fitted_ = True
        return self

    def predict_uncertainty(
        self, X: pd.DataFrame, confidence_level: float = 0.95
    ) -> Dict[str, np.ndarray]:
        """
        Make predictions with ensemble uncertainty.

        Args:
            X: Features to predict on
            confidence_level: Confidence level for intervals

        Returns:
            Dictionary with predictions and uncertainty estimates
        """
        if not self.is_fitted_:
            raise ValueError("Must fit before predicting")

        # Get predictions from all models
        predictions = []
        for estimator in self.estimators:
            if hasattr(estimator, "predict_proba") and self.voting == "soft":
                # For classification with soft voting
                pred = estimator.predict_proba(X)
            else:
                pred = estimator.predict(X)
            predictions.append(pred)

        predictions = np.array(predictions)

        # Apply weights if provided
        if self.weights is not None:
            weights = np.array(self.weights).reshape(-1, 1)
            if len(predictions.shape) == 3:  # Classification probabilities
                weights = weights.reshape(-1, 1, 1)
            weighted_predictions = predictions * weights
            mean_pred = np.sum(weighted_predictions, axis=0) / np.sum(self.weights)
        else:
            mean_pred = np.mean(predictions, axis=0)

        # Calculate uncertainty metrics
        if len(predictions.shape) == 2:  # Regression
            std_pred = np.std(predictions, axis=0)

            # Confidence intervals
            alpha = 1 - confidence_level
            lower_bound = np.percentile(predictions, (alpha / 2) * 100, axis=0)
            upper_bound = np.percentile(predictions, (1 - alpha / 2) * 100, axis=0)

            # Disagreement metric
            disagreement = std_pred / (np.abs(mean_pred) + 1e-8)

            return {
                "predictions": mean_pred,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "std": std_pred,
                "disagreement": disagreement,
                "all_predictions": predictions,
            }
        else:  # Classification
            # For probability predictions
            std_pred = np.std(predictions, axis=0)

            # Entropy as uncertainty measure
            entropy = -np.sum(mean_pred * np.log(mean_pred + 1e-10), axis=1)

            # Prediction disagreement
            class_predictions = np.argmax(predictions, axis=2)
            mode_predictions = stats.mode(class_predictions, axis=0)[0].ravel()
            disagreement = 1 - (
                np.sum(class_predictions == mode_predictions[np.newaxis, :], axis=0)
                / len(self.estimators)
            )

            return {
                "predictions": np.argmax(mean_pred, axis=1),
                "probabilities": mean_pred,
                "std": std_pred,
                "entropy": entropy,
                "disagreement": disagreement,
                "all_predictions": predictions,
            }


class PredictionInterval:
    """
    Methods for constructing prediction intervals.

    This class implements various methods for creating prediction intervals
    including conformal prediction and quantile-based approaches.
    """

    def __init__(self, method: str = "conformal", alpha: float = 0.1):
        """
        Initialize prediction interval constructor.

        Args:
            method: Method for interval construction ('conformal', 'quantile', 'residual')
            alpha: Significance level (1 - alpha is the confidence level)
        """
        self.method = method
        self.alpha = alpha
        self.calibration_scores_ = None
        self.quantile_models_ = None
        self.residual_model_ = None

    def fit(
        self,
        model: BaseEstimator,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_cal: Optional[pd.DataFrame] = None,
        y_cal: Optional[pd.Series] = None,
    ):
        """
        Fit the prediction interval method.

        Args:
            model: Fitted model
            X_train: Training features
            y_train: Training targets
            X_cal: Calibration features (for conformal)
            y_cal: Calibration targets (for conformal)
        """
        if self.method == "conformal":
            self._fit_conformal(model, X_train, y_train, X_cal, y_cal)
        elif self.method == "quantile":
            self._fit_quantile(X_train, y_train)
        elif self.method == "residual":
            self._fit_residual(model, X_train, y_train)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _fit_conformal(self, model, X_train, y_train, X_cal, y_cal):
        """Fit conformal prediction intervals."""
        # If no calibration set provided, use part of training set
        if X_cal is None or y_cal is None:
            from sklearn.model_selection import train_test_split

            X_train, X_cal, y_train, y_cal = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42
            )
            # Refit model on reduced training set
            model.fit(X_train, y_train)

        # Calculate conformity scores on calibration set
        predictions = model.predict(X_cal)
        self.calibration_scores_ = np.abs(y_cal - predictions)

        # Store the model for later use
        self.model_ = model

    def _fit_quantile(self, X_train, y_train):
        """Fit quantile regression models."""
        try:
            from sklearn.ensemble import GradientBoostingRegressor

            # Fit models for lower and upper quantiles
            lower_quantile = self.alpha / 2
            upper_quantile = 1 - self.alpha / 2

            self.quantile_models_ = {
                "lower": GradientBoostingRegressor(
                    loss="quantile", alpha=lower_quantile
                ),
                "median": GradientBoostingRegressor(loss="quantile", alpha=0.5),
                "upper": GradientBoostingRegressor(
                    loss="quantile", alpha=upper_quantile
                ),
            }

            for name, model in self.quantile_models_.items():
                logger.info(f"Fitting {name} quantile model")
                model.fit(X_train, y_train)

        except ImportError:
            raise ImportError(
                "Quantile regression requires scikit-learn with GradientBoostingRegressor"
            )

    def _fit_residual(self, model, X_train, y_train):
        """Fit residual-based prediction intervals."""
        # Get predictions on training set
        predictions = model.predict(X_train)
        residuals = y_train - predictions

        # Fit a model to predict absolute residuals
        abs_residuals = np.abs(residuals)

        try:
            from sklearn.ensemble import RandomForestRegressor

            self.residual_model_ = RandomForestRegressor(n_estimators=100)
            self.residual_model_.fit(X_train, abs_residuals)
        except ImportError:
            # Fallback to simple standard deviation
            self.residual_std_ = np.std(residuals)

        self.model_ = model

    def predict_interval(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Predict with intervals.

        Args:
            X: Features to predict on

        Returns:
            Dictionary with predictions and intervals
        """
        if self.method == "conformal":
            return self._predict_conformal(X)
        elif self.method == "quantile":
            return self._predict_quantile(X)
        elif self.method == "residual":
            return self._predict_residual(X)

    def _predict_conformal(self, X):
        """Conformal prediction intervals."""
        predictions = self.model_.predict(X)

        # Calculate the (1-alpha) quantile of calibration scores
        n_cal = len(self.calibration_scores_)
        q_level = np.ceil((n_cal + 1) * (1 - self.alpha)) / n_cal
        q_level = np.clip(q_level, 0, 1)
        quantile = np.quantile(self.calibration_scores_, q_level)

        # Construct intervals
        lower_bound = predictions - quantile
        upper_bound = predictions + quantile

        return {
            "predictions": predictions,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "interval_width": 2 * quantile,
        }

    def _predict_quantile(self, X):
        """Quantile regression intervals."""
        lower_bound = self.quantile_models_["lower"].predict(X)
        predictions = self.quantile_models_["median"].predict(X)
        upper_bound = self.quantile_models_["upper"].predict(X)

        return {
            "predictions": predictions,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "interval_width": upper_bound - lower_bound,
        }

    def _predict_residual(self, X):
        """Residual-based prediction intervals."""
        predictions = self.model_.predict(X)

        if hasattr(self, "residual_model_"):
            # Predict residual magnitude
            residual_pred = self.residual_model_.predict(X)
            # Use t-distribution for small samples
            t_value = t.ppf(1 - self.alpha / 2, df=max(len(X) - 1, 1))
            margin = t_value * residual_pred
        else:
            # Use constant residual estimate
            z_value = norm.ppf(1 - self.alpha / 2)
            margin = z_value * self.residual_std_

        return {
            "predictions": predictions,
            "lower_bound": predictions - margin,
            "upper_bound": predictions + margin,
            "interval_width": 2 * margin,
        }


class CalibrationAnalyzer:
    """
    Analyze and improve calibration of uncertainty estimates.

    This class provides methods to assess and recalibrate prediction intervals
    and probability estimates.
    """

    def __init__(self, task_type: str = "regression"):
        """
        Initialize calibration analyzer.

        Args:
            task_type: Type of task ('regression' or 'classification')
        """
        self.task_type = task_type
        self.calibration_function_ = None
        self.calibration_metrics_ = {}

    def analyze_calibration(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_lower: Optional[np.ndarray] = None,
        y_upper: Optional[np.ndarray] = None,
        y_proba: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Analyze calibration of predictions.

        Args:
            y_true: True values
            y_pred: Predictions
            y_lower: Lower bounds (for regression)
            y_upper: Upper bounds (for regression)
            y_proba: Predicted probabilities (for classification)

        Returns:
            Dictionary with calibration metrics
        """
        if self.task_type == "regression":
            return self._analyze_regression_calibration(
                y_true, y_pred, y_lower, y_upper
            )
        else:
            return self._analyze_classification_calibration(y_true, y_pred, y_proba)

    def _analyze_regression_calibration(self, y_true, y_pred, y_lower, y_upper):
        """Analyze calibration for regression intervals."""
        metrics = {}

        if y_lower is not None and y_upper is not None:
            # Coverage analysis
            in_interval = (y_true >= y_lower) & (y_true <= y_upper)
            coverage = np.mean(in_interval)
            metrics["coverage"] = coverage

            # Width analysis
            widths = y_upper - y_lower
            metrics["mean_width"] = np.mean(widths)
            metrics["median_width"] = np.median(widths)

            # Conditional coverage (by predicted value bins)
            n_bins = 10
            pred_bins = pd.qcut(y_pred, q=n_bins, duplicates="drop")
            conditional_coverage = []

            for bin_label in pred_bins.cat.categories:
                bin_mask = pred_bins == bin_label
                if np.sum(bin_mask) > 0:
                    bin_coverage = np.mean(in_interval[bin_mask])
                    conditional_coverage.append(bin_coverage)

            metrics["conditional_coverage"] = conditional_coverage
            metrics["coverage_std"] = np.std(conditional_coverage)

        # Sharpness (how tight are the predictions)
        residuals = y_true - y_pred
        metrics["rmse"] = np.sqrt(np.mean(residuals**2))
        metrics["mae"] = np.mean(np.abs(residuals))

        self.calibration_metrics_ = metrics
        return metrics

    def _analyze_classification_calibration(self, y_true, y_pred, y_proba):
        """Analyze calibration for classification probabilities."""
        metrics = {}

        if y_proba is not None:
            # Binary classification calibration
            if y_proba.shape[1] == 2:
                from sklearn.calibration import calibration_curve

                # Get calibration curve
                fraction_positive, mean_predicted = calibration_curve(
                    y_true, y_proba[:, 1], n_bins=10
                )

                # Expected Calibration Error (ECE)
                bin_weights = np.histogram(y_proba[:, 1], bins=10)[0] / len(y_proba)
                ece = np.sum(bin_weights * np.abs(fraction_positive - mean_predicted))

                metrics["ece"] = ece
                metrics["calibration_curve"] = {
                    "fraction_positive": fraction_positive,
                    "mean_predicted": mean_predicted,
                }

            # Brier score
            from sklearn.metrics import brier_score_loss

            if y_proba.shape[1] == 2:
                brier = brier_score_loss(y_true, y_proba[:, 1])
                metrics["brier_score"] = brier

        # Confidence calibration
        max_proba = np.max(y_proba, axis=1)
        correct = y_pred == y_true

        # Bin by confidence
        conf_bins = np.linspace(0, 1, 11)
        calibration_data = []

        for i in range(len(conf_bins) - 1):
            mask = (max_proba >= conf_bins[i]) & (max_proba < conf_bins[i + 1])
            if np.sum(mask) > 0:
                accuracy = np.mean(correct[mask])
                avg_confidence = np.mean(max_proba[mask])
                calibration_data.append(
                    {
                        "confidence": avg_confidence,
                        "accuracy": accuracy,
                        "count": np.sum(mask),
                    }
                )

        metrics["confidence_calibration"] = calibration_data

        self.calibration_metrics_ = metrics
        return metrics

    def recalibrate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
        method: str = "isotonic",
    ) -> Union[np.ndarray, Callable]:
        """
        Recalibrate predictions.

        Args:
            y_true: True values for calibration
            y_pred: Predictions
            y_proba: Predicted probabilities
            method: Calibration method ('isotonic', 'platt', 'temperature')

        Returns:
            Recalibrated predictions or calibration function
        """
        if self.task_type == "classification" and y_proba is not None:
            if method == "isotonic":
                # Isotonic regression calibration
                iso_reg = IsotonicRegression(out_of_bounds="clip")
                iso_reg.fit(y_proba[:, 1], y_true)
                self.calibration_function_ = iso_reg
                return iso_reg.transform(y_proba[:, 1])

            elif method == "platt":
                # Platt scaling (logistic regression)
                from sklearn.linear_model import LogisticRegression

                lr = LogisticRegression()
                lr.fit(y_proba[:, 1].reshape(-1, 1), y_true)
                self.calibration_function_ = lr
                return lr.predict_proba(y_proba[:, 1].reshape(-1, 1))[:, 1]

            elif method == "temperature":
                # Temperature scaling
                temperature = self._optimize_temperature(y_true, y_proba)
                self.calibration_function_ = lambda p: self._apply_temperature(
                    p, temperature
                )
                return self._apply_temperature(y_proba, temperature)

        else:
            raise ValueError(
                f"Calibration method {method} not supported for {self.task_type}"
            )

    def _optimize_temperature(self, y_true, y_proba, eps=1e-8):
        """Optimize temperature scaling parameter."""
        from scipy.optimize import minimize_scalar

        def nll(temperature):
            # Apply temperature scaling
            scaled_proba = self._apply_temperature(y_proba, temperature)
            # Negative log likelihood
            return -np.mean(np.log(scaled_proba[range(len(y_true)), y_true] + eps))

        result = minimize_scalar(nll, bounds=(0.1, 10.0), method="bounded")
        return result.x

    def _apply_temperature(self, proba, temperature):
        """Apply temperature scaling to probabilities."""
        # Apply temperature to logits
        logits = np.log(proba + 1e-8)
        scaled_logits = logits / temperature

        # Convert back to probabilities
        exp_scaled = np.exp(scaled_logits)
        return exp_scaled / np.sum(exp_scaled, axis=1, keepdims=True)


class UncertaintyQuantifier:
    """
    Comprehensive uncertainty quantification for machine learning models.

    This class provides multiple methods for estimating prediction uncertainty,
    including bootstrap methods, cross-validation approaches, and ensemble-based
    uncertainty estimates.
    """

    def __init__(self, task_type: str = "regression"):
        """
        Initialize the UncertaintyQuantifier.

        Args:
            task_type: Type of ML task ('regression' or 'classification')
        """
        self.task_type = task_type
        self.bootstrap_predictions = None
        self.cv_predictions = None
        self.ensemble_predictions = None

    def bootstrap_uncertainty(
        self,
        model: Any,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        n_bootstrap: int = 100,
        sample_size: Optional[float] = None,
        confidence_level: float = 0.95,
    ) -> Dict[str, np.ndarray]:
        """
        Estimate uncertainty using bootstrap resampling.

        This method trains multiple models on bootstrap samples of the training data
        and uses the variation in predictions to estimate uncertainty.

        Args:
            model: Base model to use (will be cloned)
            X_train: Training features
            y_train: Training target
            X_test: Test features to get predictions for
            n_bootstrap: Number of bootstrap iterations
            sample_size: Size of bootstrap samples (None for same as training)
            confidence_level: Confidence level for intervals (e.g., 0.95 for 95%)

        Returns:
            Dictionary containing:
                - 'predictions': Mean predictions
                - 'lower_bound': Lower confidence bound
                - 'upper_bound': Upper confidence bound
                - 'std': Standard deviation of predictions
                - 'all_predictions': All bootstrap predictions
        """
        logger.info(
            f"Starting bootstrap uncertainty estimation with {n_bootstrap} iterations"
        )

        # Store all predictions
        bootstrap_predictions = []

        # Determine sample size
        if sample_size is None:
            sample_size = len(X_train)
        else:
            sample_size = int(sample_size * len(X_train))

        # Bootstrap iterations
        for i in tqdm(range(n_bootstrap), desc="Bootstrap iterations"):
            # Create bootstrap sample
            indices = resample(range(len(X_train)), n_samples=sample_size, replace=True)
            X_bootstrap = X_train.iloc[indices]
            y_bootstrap = y_train.iloc[indices]

            # Clone and train model
            model_clone = clone(model)
            model_clone.fit(X_bootstrap, y_bootstrap)

            # Make predictions
            if self.task_type == "regression":
                pred = model_clone.predict(X_test)
            else:
                if hasattr(model_clone, "predict_proba"):
                    pred = model_clone.predict_proba(X_test)
                else:
                    pred = model_clone.predict(X_test)

            bootstrap_predictions.append(pred)

        bootstrap_predictions = np.array(bootstrap_predictions)
        self.bootstrap_predictions = bootstrap_predictions

        # Calculate statistics
        if self.task_type == "regression":
            mean_predictions = np.mean(bootstrap_predictions, axis=0)
            std_predictions = np.std(bootstrap_predictions, axis=0)

            # Confidence intervals
            alpha = 1 - confidence_level
            lower_percentile = (alpha / 2) * 100
            upper_percentile = (1 - alpha / 2) * 100

            lower_bound = np.percentile(bootstrap_predictions, lower_percentile, axis=0)
            upper_bound = np.percentile(bootstrap_predictions, upper_percentile, axis=0)

            return {
                "predictions": mean_predictions,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "std": std_predictions,
                "all_predictions": bootstrap_predictions,
            }
        else:
            # Classification: process probability predictions
            return self._process_classification_predictions(
                bootstrap_predictions, confidence_level
            )

    def cv_uncertainty(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series,
        cv: int = 5,
        confidence_level: float = 0.95,
    ) -> Dict[str, np.ndarray]:
        """
        Estimate uncertainty using cross-validation.

        This method trains models on different CV folds and uses the variation
        in predictions to estimate uncertainty.

        Args:
            model: Base model to use (will be cloned)
            X: Features
            y: Target
            cv: Number of cross-validation folds
            confidence_level: Confidence level for intervals

        Returns:
            Dictionary with uncertainty estimates
        """
        logger.info(f"Starting CV uncertainty estimation with {cv} folds")

        kfold = KFold(n_splits=cv, shuffle=True, random_state=42)
        predictions = []

        # For each fold
        for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
            X_train_fold = X.iloc[train_idx]
            y_train_fold = y.iloc[train_idx]
            X_val_fold = X.iloc[val_idx]

            # Clone and train model
            model_clone = clone(model)
            model_clone.fit(X_train_fold, y_train_fold)

            # Store predictions for validation indices
            if self.task_type == "regression":
                pred = model_clone.predict(X_val_fold)
            else:
                if hasattr(model_clone, "predict_proba"):
                    pred = model_clone.predict_proba(X_val_fold)
                else:
                    pred = model_clone.predict(X_val_fold)

            predictions.append((val_idx, pred))

        # Reorganize predictions
        all_predictions = np.zeros((cv, len(X)) + pred.shape[1:])
        all_predictions[:] = np.nan

        for fold, (indices, preds) in enumerate(predictions):
            all_predictions[fold, indices] = preds

        # Calculate statistics for each sample
        mean_predictions = []
        std_predictions = []
        lower_bounds = []
        upper_bounds = []

        for i in range(len(X)):
            # Get non-nan predictions for this sample
            sample_preds = all_predictions[:, i]
            valid_preds = sample_preds[
                ~np.isnan(sample_preds).any(axis=tuple(range(1, sample_preds.ndim)))
            ]

            if len(valid_preds) > 0:
                if self.task_type == "regression":
                    mean_predictions.append(np.mean(valid_preds))
                    std_predictions.append(np.std(valid_preds))

                    alpha = 1 - confidence_level
                    lower_bounds.append(np.percentile(valid_preds, (alpha / 2) * 100))
                    upper_bounds.append(
                        np.percentile(valid_preds, (1 - alpha / 2) * 100)
                    )
                else:
                    if len(valid_preds.shape) > 1:  # Probability predictions
                        mean_predictions.append(np.mean(valid_preds, axis=0))
                        std_predictions.append(np.std(valid_preds, axis=0))
                    else:  # Class predictions
                        mean_predictions.append(stats.mode(valid_preds)[0][0])
                        std_predictions.append(np.nan)  # No std for class predictions

        self.cv_predictions = predictions

        if self.task_type == "regression":
            return {
                "predictions": np.array(mean_predictions),
                "lower_bound": np.array(lower_bounds),
                "upper_bound": np.array(upper_bounds),
                "std": np.array(std_predictions),
            }
        else:
            return {
                "predictions": np.array(mean_predictions),
                "std": (
                    np.array(std_predictions)
                    if std_predictions[0] is not np.nan
                    else None
                ),
            }

    def ensemble_uncertainty(
        self, models: List[Any], X: pd.DataFrame, confidence_level: float = 0.95
    ) -> Dict[str, np.ndarray]:
        """
        Estimate uncertainty using an ensemble of different models.

        This method uses the disagreement between different models as a measure
        of prediction uncertainty.

        Args:
            models: List of fitted models
            X: Features to predict on
            confidence_level: Confidence level for intervals

        Returns:
            Dictionary with uncertainty estimates
        """
        logger.info(f"Estimating uncertainty from ensemble of {len(models)} models")

        # Get predictions from all models
        all_predictions = []

        for model in models:
            if self.task_type == "regression":
                pred = model.predict(X)
            else:
                if hasattr(model, "predict_proba"):
                    pred = model.predict_proba(X)
                else:
                    pred = model.predict(X)

            all_predictions.append(pred)

        all_predictions = np.array(all_predictions)
        self.ensemble_predictions = all_predictions

        # Calculate statistics
        if self.task_type == "regression":
            mean_predictions = np.mean(all_predictions, axis=0)
            std_predictions = np.std(all_predictions, axis=0)

            # Confidence intervals
            alpha = 1 - confidence_level
            lower_percentile = (alpha / 2) * 100
            upper_percentile = (1 - alpha / 2) * 100

            lower_bound = np.percentile(all_predictions, lower_percentile, axis=0)
            upper_bound = np.percentile(all_predictions, upper_percentile, axis=0)

            # Model agreement (normalized inverse of std)
            agreement = 1 / (1 + std_predictions)

            return {
                "predictions": mean_predictions,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "std": std_predictions,
                "agreement": agreement,
                "all_predictions": all_predictions,
            }
        else:
            # Classification uncertainty
            if len(all_predictions.shape) == 3:  # Probability predictions
                mean_probs = np.mean(all_predictions, axis=0)
                std_probs = np.std(all_predictions, axis=0)

                # Entropy as uncertainty measure
                entropy = -np.sum(mean_probs * np.log(mean_probs + 1e-10), axis=1)

                # Disagreement between models
                class_predictions = np.argmax(all_predictions, axis=2)
                mode_predictions = stats.mode(class_predictions, axis=0)[0].ravel()
                disagreement = 1 - (
                    np.sum(class_predictions == mode_predictions[np.newaxis, :], axis=0)
                    / len(models)
                )

                return {
                    "predictions": np.argmax(mean_probs, axis=1),
                    "probabilities": mean_probs,
                    "std": std_probs,
                    "entropy": entropy,
                    "disagreement": disagreement,
                    "all_predictions": all_predictions,
                }
            else:  # Direct class predictions
                mode_predictions = stats.mode(all_predictions, axis=0)[0].ravel()
                disagreement = 1 - (
                    np.sum(all_predictions == mode_predictions[np.newaxis, :], axis=0)
                    / len(models)
                )

                return {
                    "predictions": mode_predictions,
                    "disagreement": disagreement,
                    "all_predictions": all_predictions,
                }

    def dropout_uncertainty(
        self,
        model: Any,
        X: pd.DataFrame,
        n_iterations: int = 100,
        confidence_level: float = 0.95,
    ) -> Dict[str, np.ndarray]:
        """
        Estimate uncertainty using Monte Carlo Dropout.

        This method is specific to neural networks with dropout layers.
        It makes multiple predictions with dropout enabled to estimate uncertainty.

        Args:
            model: Neural network model with dropout
            X: Features to predict on
            n_iterations: Number of forward passes
            confidence_level: Confidence level for intervals

        Returns:
            Dictionary with uncertainty estimates
        """
        logger.info(f"Starting MC Dropout uncertainty with {n_iterations} iterations")

        # Check if model has dropout capability
        if not hasattr(model, "training"):
            warnings.warn(
                "Model doesn't appear to have dropout capability. Results may not be meaningful."
            )

        # Store predictions
        mc_predictions = []

        # Enable dropout during inference
        if hasattr(model, "training"):
            original_training_state = model.training
            model.training = True  # Enable dropout

        for i in tqdm(range(n_iterations), desc="MC Dropout iterations"):
            if self.task_type == "regression":
                pred = model.predict(X)
            else:
                if hasattr(model, "predict_proba"):
                    pred = model.predict_proba(X)
                else:
                    pred = model.predict(X)

            mc_predictions.append(pred)

        # Restore original training state
        if hasattr(model, "training"):
            model.training = original_training_state

        mc_predictions = np.array(mc_predictions)

        # Calculate statistics
        if self.task_type == "regression":
            mean_predictions = np.mean(mc_predictions, axis=0)
            std_predictions = np.std(mc_predictions, axis=0)

            # Confidence intervals
            alpha = 1 - confidence_level
            lower_percentile = (alpha / 2) * 100
            upper_percentile = (1 - alpha / 2) * 100

            lower_bound = np.percentile(mc_predictions, lower_percentile, axis=0)
            upper_bound = np.percentile(mc_predictions, upper_percentile, axis=0)

            return {
                "predictions": mean_predictions,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "std": std_predictions,
                "epistemic_uncertainty": std_predictions,  # Dropout captures epistemic uncertainty
                "all_predictions": mc_predictions,
            }
        else:
            # Similar logic for classification
            return self._process_classification_predictions(
                mc_predictions, confidence_level
            )

    def quantile_uncertainty(
        self,
        models: Union[Any, List[Any]],
        X: pd.DataFrame,
        quantiles: List[float] = [0.05, 0.95],
    ) -> Dict[str, np.ndarray]:
        """
        Estimate uncertainty using quantile regression models.

        This method requires models trained to predict specific quantiles
        of the target distribution.

        Args:
            models: Single model or dict/list of models for different quantiles
            X: Features to predict on
            quantiles: List of quantiles (should match the models)

        Returns:
            Dictionary with quantile predictions
        """
        logger.info("Estimating uncertainty using quantile regression")

        if isinstance(models, dict):
            # Dictionary of models keyed by quantile
            predictions = {}
            for q, model in models.items():
                predictions[f"quantile_{q}"] = model.predict(X)

            # Add median if available
            if 0.5 in models:
                predictions["predictions"] = predictions["quantile_0.5"]

            # Add bounds if standard quantiles available
            if "quantile_0.05" in predictions and "quantile_0.95" in predictions:
                predictions["lower_bound"] = predictions["quantile_0.05"]
                predictions["upper_bound"] = predictions["quantile_0.95"]

        elif isinstance(models, list):
            # List of models corresponding to quantiles
            if len(models) != len(quantiles):
                raise ValueError("Number of models must match number of quantiles")

            predictions = {}
            for i, (q, model) in enumerate(zip(quantiles, models)):
                pred = model.predict(X)
                predictions[f"quantile_{q}"] = pred

                if q == 0.5:
                    predictions["predictions"] = pred
                elif q == quantiles[0]:
                    predictions["lower_bound"] = pred
                elif q == quantiles[-1]:
                    predictions["upper_bound"] = pred
        else:
            # Single model - assume it can predict multiple quantiles
            raise ValueError("Single model quantile prediction not implemented")

        return predictions

    def calculate_metrics(
        self,
        y_true: np.ndarray,
        predictions: np.ndarray,
        lower_bound: np.ndarray,
        upper_bound: np.ndarray,
    ) -> Dict[str, float]:
        """
        Calculate uncertainty quantification metrics.

        Args:
            y_true: True values
            predictions: Point predictions
            lower_bound: Lower bounds
            upper_bound: Upper bounds

        Returns:
            Dictionary of metrics
        """
        # Prediction Interval Coverage Probability (PICP)
        coverage = np.mean((y_true >= lower_bound) & (y_true <= upper_bound))

        # Mean Prediction Interval Width (MPIW)
        interval_width = upper_bound - lower_bound
        mean_width = np.mean(interval_width)
        normalized_width = mean_width / (np.max(y_true) - np.min(y_true))

        # Interval Score (IS)
        alpha = 0.05  # Assuming 95% intervals
        lower_penalty = (lower_bound - y_true) * (y_true < lower_bound)
        upper_penalty = (y_true - upper_bound) * (y_true > upper_bound)
        interval_score = interval_width + (2 / alpha) * (lower_penalty + upper_penalty)
        mean_interval_score = np.mean(interval_score)

        # Coverage Width-based Criterion (CWC)
        eta = 0.5  # Balance parameter
        cwc = (1 - coverage) + eta * normalized_width

        # Root Mean Square Error (RMSE)
        rmse = np.sqrt(np.mean((y_true - predictions) ** 2))

        # Mean Absolute Error (MAE)
        mae = np.mean(np.abs(y_true - predictions))

        return {
            "picp": coverage,
            "mpiw": mean_width,
            "normalized_mpiw": normalized_width,
            "mean_interval_score": mean_interval_score,
            "cwc": cwc,
            "rmse": rmse,
            "mae": mae,
        }

    def _process_classification_predictions(
        self, predictions: np.ndarray, confidence_level: float
    ) -> Dict[str, np.ndarray]:
        """Process classification predictions for uncertainty."""
        if len(predictions.shape) == 3:  # Probability predictions
            mean_probs = np.mean(predictions, axis=0)
            std_probs = np.std(predictions, axis=0)

            # Class predictions
            class_preds = np.argmax(mean_probs, axis=1)

            # Entropy as uncertainty
            entropy = -np.sum(mean_probs * np.log(mean_probs + 1e-10), axis=1)

            # Maximum probability (confidence)
            max_prob = np.max(mean_probs, axis=1)

            return {
                "predictions": class_preds,
                "probabilities": mean_probs,
                "std": std_probs,
                "entropy": entropy,
                "confidence": max_prob,
                "all_predictions": predictions,
            }
        else:  # Direct class predictions
            mode_preds = stats.mode(predictions, axis=0)[0].ravel()

            # Agreement as inverse of disagreement
            agreement = np.mean(predictions == mode_preds[np.newaxis, :], axis=0)

            return {
                "predictions": mode_preds,
                "agreement": agreement,
                "all_predictions": predictions,
            }


# Example usage functions remain the same
def example_regression_uncertainty():
    """Example of using UncertaintyQuantifier for regression tasks."""
    from sklearn.datasets import make_regression
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split

    # Generate synthetic data
    X, y = make_regression(n_samples=1000, n_features=10, noise=10, random_state=42)
    X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(10)])
    y = pd.Series(y, name="target")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Initialize model and uncertainty quantifier
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    uq = UncertaintyQuantifier(task_type="regression")

    # Bootstrap uncertainty
    results = uq.bootstrap_uncertainty(
        model, X_train, y_train, X_test, n_bootstrap=50, confidence_level=0.95
    )

    # Calculate metrics
    metrics = uq.calculate_metrics(
        y_test.values,
        results["predictions"],
        results["lower_bound"],
        results["upper_bound"],
    )

    print("Bootstrap Uncertainty Results:")
    print(f"Coverage: {metrics['picp']:.2%}")
    print(f"Mean Interval Width: {metrics['mpiw']:.2f}")
    print(f"RMSE: {metrics['rmse']:.2f}")

    return uq, results, metrics


def example_classification_uncertainty():
    """Example of using UncertaintyQuantifier for classification tasks."""
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split

    # Generate synthetic data
    X, y = make_classification(
        n_samples=1000, n_features=10, n_classes=3, n_informative=5, random_state=42
    )
    X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(10)])
    y = pd.Series(y, name="target")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Initialize models for ensemble
    models = [
        RandomForestClassifier(n_estimators=100, random_state=i) for i in range(5)
    ]

    # Train models
    for model in models:
        model.fit(X_train, y_train)

    # Uncertainty quantification
    uq = UncertaintyQuantifier(task_type="classification")

    # Ensemble uncertainty
    results = uq.ensemble_uncertainty(models, X_test, confidence_level=0.95)

    print("\nEnsemble Uncertainty Results:")
    print(f"Mean Entropy: {np.mean(results['entropy']):.2f}")
    print(f"Mean Confidence: {np.mean(results['confidence']):.2%}")
    print(f"Accuracy: {np.mean(results['predictions'] == y_test.values):.2%}")

    return uq, results


if __name__ == "__main__":
    # Run examples
    print("Running regression uncertainty example...")
    reg_uq, reg_results, reg_metrics = example_regression_uncertainty()

    print("\n" + "=" * 50 + "\n")

    print("Running classification uncertainty example...")
    class_uq, class_results = example_classification_uncertainty()
