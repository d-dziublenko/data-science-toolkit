"""
models/ensemble.py
Ensemble methods for combining multiple models.
"""

import logging
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone
from sklearn.ensemble import (BaggingClassifier, BaggingRegressor,
                              StackingClassifier, StackingRegressor,
                              VotingClassifier, VotingRegressor)
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

logger = logging.getLogger(__name__)


class BaseEnsemble(BaseEstimator):
    """
    Base class for ensemble methods.

    Provides common functionality for all ensemble models.
    """

    def __init__(self, base_estimators: List[BaseEstimator], task_type: str = "auto"):
        """
        Initialize BaseEnsemble.

        Args:
            base_estimators: List of base estimators
            task_type: Type of task ('regression', 'classification', or 'auto')
        """
        self.base_estimators = base_estimators
        self.task_type = task_type
        self.is_fitted_ = False

    def _infer_task_type(self, y):
        """Infer task type from target variable."""
        if self.task_type == "auto":
            # Check if target is continuous or discrete
            unique_values = np.unique(y)
            n_unique = len(unique_values)

            if n_unique == 2:
                self.task_type = "classification"
            elif n_unique < 10 and n_unique < len(y) * 0.05:
                self.task_type = "classification"
            else:
                self.task_type = "regression"

        logger.info(f"Task type: {self.task_type}")

    def _validate_estimators(self):
        """Validate base estimators."""
        if not self.base_estimators:
            raise ValueError("No base estimators provided")

        for estimator in self.base_estimators:
            if not hasattr(estimator, "fit"):
                raise TypeError(f"Estimator {estimator} must have a fit method")


class VotingEnsemble(BaseEnsemble):
    """
    Voting ensemble for combining predictions.

    Supports both hard and soft voting for classification,
    and averaging for regression.
    """

    def __init__(
        self,
        estimators: List[Tuple[str, BaseEstimator]],
        voting: str = "soft",
        weights: Optional[List[float]] = None,
        task_type: str = "auto",
        flatten_transform: bool = True,
        verbose: bool = False,
    ):
        """
        Initialize VotingEnsemble.

        Args:
            estimators: List of (name, estimator) tuples
            voting: Voting type ('hard' or 'soft' for classification)
            weights: Sequence of weights for estimators
            task_type: Type of task
            flatten_transform: Affects shape of transform output
            verbose: Enable verbose output
        """
        base_estimators = [est for _, est in estimators]
        super().__init__(base_estimators, task_type)

        self.estimators = estimators
        self.voting = voting
        self.weights = weights
        self.flatten_transform = flatten_transform
        self.verbose = verbose
        self.voting_model_ = None

    def fit(self, X, y, sample_weight=None):
        """
        Fit the voting ensemble.

        Args:
            X: Training features
            y: Training targets
            sample_weight: Sample weights

        Returns:
            Self
        """
        # Validate input
        X, y = check_X_y(X, accept_sparse=["csc"])
        self._validate_estimators()

        # Infer task type
        self._infer_task_type(y)

        # Create appropriate voting model
        if self.task_type == "regression":
            self.voting_model_ = VotingRegressor(
                estimators=self.estimators, weights=self.weights, verbose=self.verbose
            )
        else:
            self.voting_model_ = VotingClassifier(
                estimators=self.estimators,
                voting=self.voting,
                weights=self.weights,
                flatten_transform=self.flatten_transform,
                verbose=self.verbose,
            )

        # Fit the voting model
        logger.info(f"Fitting voting ensemble with {len(self.estimators)} estimators")
        self.voting_model_.fit(X, y, sample_weight)

        self.is_fitted_ = True
        return self

    def predict(self, X):
        """
        Make predictions.

        Args:
            X: Features to predict

        Returns:
            Predictions
        """
        check_is_fitted(self, "voting_model_")
        X = check_array(X, accept_sparse=["csc"])
        return self.voting_model_.predict(X)

    def predict_proba(self, X):
        """
        Predict class probabilities (classification only).

        Args:
            X: Features to predict

        Returns:
            Class probabilities
        """
        if self.task_type != "classification":
            raise AttributeError("predict_proba is only available for classification")

        check_is_fitted(self, "voting_model_")

        if self.voting == "hard":
            raise AttributeError("predict_proba is not available for hard voting")

        X = check_array(X, accept_sparse=["csc"])
        return self.voting_model_.predict_proba(X)

    def transform(self, X):
        """
        Return predictions from each estimator.

        Args:
            X: Features to transform

        Returns:
            Predictions from each estimator
        """
        check_is_fitted(self, "voting_model_")
        return self.voting_model_.transform(X)

    @property
    def fitted_estimators_(self):
        """Get fitted estimators."""
        if hasattr(self.voting_model_, "estimators_"):
            return self.voting_model_.estimators_
        return None


class StackingEnsemble(BaseEnsemble):
    """
    Stacking ensemble with cross-validation.

    This class implements stacking with out-of-fold predictions
    to avoid overfitting in the meta-model.
    """

    def __init__(
        self,
        estimators: List[Tuple[str, BaseEstimator]],
        final_estimator: Optional[BaseEstimator] = None,
        cv: Union[int, object] = 5,
        stack_method: Union[str, List[str]] = "auto",
        passthrough: bool = False,
        task_type: str = "auto",
        verbose: int = 0,
    ):
        """
        Initialize StackingEnsemble.

        Args:
            estimators: List of (name, estimator) tuples
            final_estimator: Meta-model (defaults to LogisticRegression/Ridge)
            cv: Cross-validation strategy
            stack_method: Method to use for stacking predictions
            passthrough: Whether to pass original features to final estimator
            task_type: Type of task
            verbose: Verbosity level
        """
        base_estimators = [est for _, est in estimators]
        super().__init__(base_estimators, task_type)

        self.estimators = estimators
        self.final_estimator = final_estimator
        self.cv = cv
        self.stack_method = stack_method
        self.passthrough = passthrough
        self.verbose = verbose
        self.stacking_model_ = None

    def fit(self, X, y, sample_weight=None):
        """
        Fit the stacking ensemble.

        Args:
            X: Training features
            y: Training targets
            sample_weight: Sample weights

        Returns:
            Self
        """
        # Validate input
        X, y = check_X_y(X, accept_sparse=["csc"])
        self._validate_estimators()

        # Infer task type
        self._infer_task_type(y)

        # Create appropriate stacking model
        if self.task_type == "regression":
            self.stacking_model_ = StackingRegressor(
                estimators=self.estimators,
                final_estimator=self.final_estimator,
                cv=self.cv,
                passthrough=self.passthrough,
                verbose=self.verbose,
            )
        else:
            self.stacking_model_ = StackingClassifier(
                estimators=self.estimators,
                final_estimator=self.final_estimator,
                cv=self.cv,
                stack_method=self.stack_method,
                passthrough=self.passthrough,
                verbose=self.verbose,
            )

        # Fit the stacking model
        logger.info(
            f"Fitting stacking ensemble with {len(self.estimators)} base estimators"
        )
        self.stacking_model_.fit(X, y, sample_weight)

        self.is_fitted_ = True
        return self

    def predict(self, X):
        """
        Make predictions.

        Args:
            X: Features to predict

        Returns:
            Predictions
        """
        check_is_fitted(self, "stacking_model_")
        X = check_array(X, accept_sparse=["csc"])
        return self.stacking_model_.predict(X)

    def predict_proba(self, X):
        """
        Predict class probabilities (classification only).

        Args:
            X: Features to predict

        Returns:
            Class probabilities
        """
        if self.task_type != "classification":
            raise AttributeError("predict_proba is only available for classification")

        check_is_fitted(self, "stacking_model_")
        X = check_array(X, accept_sparse=["csc"])
        return self.stacking_model_.predict_proba(X)

    def transform(self, X):
        """
        Transform features using base estimators.

        Args:
            X: Features to transform

        Returns:
            Meta-features from base estimators
        """
        check_is_fitted(self, "stacking_model_")
        return self.stacking_model_.transform(X)

    @property
    def estimators_(self):
        """Get fitted base estimators."""
        if hasattr(self.stacking_model_, "estimators_"):
            return self.stacking_model_.estimators_
        return None

    @property
    def final_estimator_(self):
        """Get fitted final estimator."""
        if hasattr(self.stacking_model_, "final_estimator_"):
            return self.stacking_model_.final_estimator_
        return None


class BaggingEnsemble(BaseEnsemble):
    """
    Bagging ensemble with bootstrap aggregation.

    This class implements bagging for both classification and regression.
    """

    def __init__(
        self,
        base_estimator: Optional[BaseEstimator] = None,
        n_estimators: int = 10,
        max_samples: Union[int, float] = 1.0,
        max_features: Union[int, float] = 1.0,
        bootstrap: bool = True,
        bootstrap_features: bool = False,
        oob_score: bool = False,
        warm_start: bool = False,
        task_type: str = "auto",
        random_state: Optional[int] = None,
        verbose: int = 0,
    ):
        """
        Initialize BaggingEnsemble.

        Args:
            base_estimator: Base estimator to fit on subsets
            n_estimators: Number of base estimators
            max_samples: Number of samples to draw
            max_features: Number of features to draw
            bootstrap: Whether to bootstrap samples
            bootstrap_features: Whether to bootstrap features
            oob_score: Whether to use out-of-bag samples
            warm_start: Whether to reuse previous solution
            task_type: Type of task
            random_state: Random state
            verbose: Verbosity level
        """
        super().__init__([base_estimator] if base_estimator else [], task_type)

        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.bootstrap_features = bootstrap_features
        self.oob_score = oob_score
        self.warm_start = warm_start
        self.random_state = random_state
        self.verbose = verbose
        self.bagging_model_ = None

    def fit(self, X, y, sample_weight=None):
        """
        Fit the bagging ensemble.

        Args:
            X: Training features
            y: Training targets
            sample_weight: Sample weights

        Returns:
            Self
        """
        # Validate input
        X, y = check_X_y(X, accept_sparse=["csc"])

        # Infer task type
        self._infer_task_type(y)

        # Create appropriate bagging model
        if self.task_type == "regression":
            self.bagging_model_ = BaggingRegressor(
                base_estimator=self.base_estimator,
                n_estimators=self.n_estimators,
                max_samples=self.max_samples,
                max_features=self.max_features,
                bootstrap=self.bootstrap,
                bootstrap_features=self.bootstrap_features,
                oob_score=self.oob_score,
                warm_start=self.warm_start,
                random_state=self.random_state,
                verbose=self.verbose,
            )
        else:
            self.bagging_model_ = BaggingClassifier(
                base_estimator=self.base_estimator,
                n_estimators=self.n_estimators,
                max_samples=self.max_samples,
                max_features=self.max_features,
                bootstrap=self.bootstrap,
                bootstrap_features=self.bootstrap_features,
                oob_score=self.oob_score,
                warm_start=self.warm_start,
                random_state=self.random_state,
                verbose=self.verbose,
            )

        # Fit the bagging model
        logger.info(f"Fitting bagging ensemble with {self.n_estimators} estimators")
        self.bagging_model_.fit(X, y, sample_weight)

        self.is_fitted_ = True
        return self

    def predict(self, X):
        """
        Make predictions.

        Args:
            X: Features to predict

        Returns:
            Predictions
        """
        check_is_fitted(self, "bagging_model_")
        X = check_array(X, accept_sparse=["csc"])
        return self.bagging_model_.predict(X)

    def predict_proba(self, X):
        """
        Predict class probabilities (classification only).

        Args:
            X: Features to predict

        Returns:
            Class probabilities
        """
        if self.task_type != "classification":
            raise AttributeError("predict_proba is only available for classification")

        check_is_fitted(self, "bagging_model_")
        X = check_array(X, accept_sparse=["csc"])
        return self.bagging_model_.predict_proba(X)

    @property
    def estimators_(self):
        """Get the list of fitted sub-estimators."""
        if hasattr(self.bagging_model_, "estimators_"):
            return self.bagging_model_.estimators_
        return None

    @property
    def estimators_samples_(self):
        """Get the subset of drawn samples for each estimator."""
        if hasattr(self.bagging_model_, "estimators_samples_"):
            return self.bagging_model_.estimators_samples_
        return None

    @property
    def estimators_features_(self):
        """Get the subset of drawn features for each estimator."""
        if hasattr(self.bagging_model_, "estimators_features_"):
            return self.bagging_model_.estimators_features_
        return None

    @property
    def oob_score_(self):
        """Get out-of-bag score."""
        if hasattr(self.bagging_model_, "oob_score_"):
            return self.bagging_model_.oob_score_
        return None

    @property
    def oob_prediction_(self):
        """Get out-of-bag predictions."""
        if hasattr(self.bagging_model_, "oob_prediction_"):
            return self.bagging_model_.oob_prediction_
        return None

    def get_feature_importance(self, X):
        """
        Get aggregated feature importance from all estimators.

        Args:
            X: Features (needed for shape)

        Returns:
            Array of feature importances
        """
        check_is_fitted(self, "bagging_model_")

        if not hasattr(self.estimators_[0], "feature_importances_"):
            raise ValueError("Base estimators don't have feature_importances_")

        # Aggregate feature importances
        importances = []
        for estimator, features in zip(self.estimators_, self.estimators_features_):
            # Get importance for selected features
            imp = np.zeros(X.shape[1])
            imp[features] = estimator.feature_importances_
            importances.append(imp)

        # Return mean importance
        return np.mean(importances, axis=0)


class ModelStacker:
    """
    Advanced stacking implementation.

    Combines multiple models using meta-learning for improved performance.
    """

    def __init__(
        self,
        base_models: List[Tuple[str, Any]],
        meta_model: Optional[Any] = None,
        task_type: str = "regression",
        cv: int = 5,
        use_probabilities: bool = True,
        cv_folds: Optional[Any] = None,
        stack_method: str = "predict_proba",
    ):
        """
        Initialize the ModelStacker.

        Args:
            base_models: List of (name, model) tuples for base models
            meta_model: Meta-learner model (default: LinearRegression/LogisticRegression)
            task_type: Type of ML task ('regression' or 'classification')
            cv: Number of cross-validation folds for stacking
            use_probabilities: Whether to use probabilities for classification
            cv_folds: Custom CV splitter
            stack_method: Method for stacking ('predict_proba' or 'predict')
        """
        self.base_models = base_models
        self.meta_model = meta_model
        self.task_type = task_type
        self.cv = cv
        self.use_probabilities = use_probabilities
        self.cv_folds = cv_folds
        self.stack_method = stack_method
        self.stacking_model = None
        self.is_fitted = False

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        parallel: Optional[Any] = None,
    ):
        """
        Fit the stacking model.

        Args:
            X: Feature matrix
            y: Target variable
            parallel: Optional parallel processor
        """
        if self.meta_model is None:
            if self.task_type == "regression":
                from sklearn.linear_model import LinearRegression

                self.meta_model = LinearRegression()
            else:
                from sklearn.linear_model import LogisticRegression

                self.meta_model = LogisticRegression(random_state=42)

        # Set up CV strategy if not provided
        if self.cv_folds is None:
            self.cv_folds = self.cv

        # Determine stack method for classification
        if self.task_type == "classification" and self.use_probabilities:
            stack_method = "predict_proba"
        else:
            stack_method = "predict"

        if self.task_type == "regression":
            self.stacking_model = StackingRegressor(
                estimators=self.base_models,
                final_estimator=self.meta_model,
                cv=self.cv_folds,
            )
        else:
            self.stacking_model = StackingClassifier(
                estimators=self.base_models,
                final_estimator=self.meta_model,
                cv=self.cv_folds,
                stack_method=stack_method,
            )

        logger.info(f"Training stacking model with {len(self.base_models)} base models")
        self.stacking_model.fit(X, y)
        self.is_fitted = True

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Make predictions using the stacking model."""
        if not self.is_fitted or self.stacking_model is None:
            raise ValueError("Model has not been fitted yet")
        return self.stacking_model.predict(X)

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Make probability predictions (classification only)."""
        if self.task_type != "classification":
            raise ValueError("predict_proba is only available for classification")
        if not self.is_fitted or self.stacking_model is None:
            raise ValueError("Model has not been fitted yet")
        return self.stacking_model.predict_proba(X)

    def get_base_predictions(self, X: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        """Get predictions from all base models."""
        if not self.is_fitted or self.stacking_model is None:
            raise ValueError("Model has not been fitted yet")

        predictions = {}
        for name, estimator in self.stacking_model.estimators_:
            predictions[name] = estimator.predict(X)

        return pd.DataFrame(predictions)


class ModelBlender:
    """
    Simple model blending implementation.

    Combines predictions from multiple models using weighted averaging.
    """

    def __init__(
        self,
        models: List[Tuple[str, Any]],
        weights: Optional[List[float]] = None,
        blend_method: str = "weighted",
        optimization_metric: str = "rmse",
    ):
        """
        Initialize the ModelBlender.

        Args:
            models: List of (name, model) tuples
            weights: Weights for each model (default: equal weights)
            blend_method: Method for blending ('weighted', 'mean', 'ranked')
            optimization_metric: Metric to optimize when finding weights
        """
        self.models = models
        self.weights = weights
        self.blend_method = blend_method
        self.optimization_metric = optimization_metric
        self.fitted_models = []
        self.optimized_weights = None
        self.is_fitted = False

        if weights is not None:
            if len(weights) != len(models):
                raise ValueError("Number of weights must match number of models")

            # Normalize weights
            total_weight = sum(weights)
            self.weights = [w / total_weight for w in weights]

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]):
        """
        Fit all models in the blend.

        Args:
            X: Feature matrix
            y: Target variable
        """
        logger.info(f"Training {len(self.models)} models for blending")

        self.fitted_models = []
        for name, model in self.models:
            logger.info(f"Training {name}")
            fitted_model = clone(model)
            fitted_model.fit(X, y)
            self.fitted_models.append((name, fitted_model))

        # Initialize equal weights if not provided
        if self.weights is None:
            self.weights = [1.0 / len(self.models)] * len(self.models)

        self.optimized_weights = self.weights
        self.is_fitted = True

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Make blended predictions.

        Args:
            X: Feature matrix

        Returns:
            Weighted average of all model predictions
        """
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet")

        predictions = []

        for (name, model), weight in zip(self.fitted_models, self.optimized_weights):
            pred = model.predict(X) * weight
            predictions.append(pred)

        return np.sum(predictions, axis=0)

    def optimize_weights(
        self,
        X_val: Union[pd.DataFrame, np.ndarray],
        y_val: Union[pd.Series, np.ndarray],
        metric: Optional[str] = None,
    ):
        """
        Optimize blending weights using validation data.

        Args:
            X_val: Validation features
            y_val: Validation targets
            metric: Optimization metric (uses self.optimization_metric if None)
        """
        if not self.is_fitted:
            raise ValueError("Models must be fitted before optimizing weights")

        from scipy.optimize import minimize

        metric = metric or self.optimization_metric

        # Get predictions from all models
        val_predictions = []
        for name, model in self.fitted_models:
            pred = model.predict(X_val)
            val_predictions.append(pred)

        val_predictions = np.array(val_predictions).T

        def objective(weights):
            """Objective function to minimize."""
            # Ensure weights sum to 1
            weights = weights / weights.sum()

            # Calculate weighted predictions
            weighted_pred = val_predictions @ weights

            # Calculate metric
            if metric == "rmse":
                return np.sqrt(np.mean((y_val - weighted_pred) ** 2))
            elif metric == "mae":
                return np.mean(np.abs(y_val - weighted_pred))
            elif metric == "mse":
                return np.mean((y_val - weighted_pred) ** 2)
            else:
                raise ValueError(f"Unknown metric: {metric}")

        # Initial weights
        x0 = np.array([1.0 / len(self.models)] * len(self.models))

        # Constraints: weights sum to 1
        constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1.0}

        # Bounds: weights between 0 and 1
        bounds = [(0, 1)] * len(self.models)

        # Optimize
        result = minimize(
            objective, x0, method="SLSQP", bounds=bounds, constraints=constraints
        )

        if result.success:
            self.optimized_weights = result.x / result.x.sum()
            logger.info(f"Optimized weights: {self.optimized_weights}")
        else:
            logger.warning("Weight optimization failed, using equal weights")
            self.optimized_weights = self.weights

    def get_model_weights(self) -> Dict[str, float]:
        """Get the weights for each model."""
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet")

        return {
            name: weight
            for (name, _), weight in zip(self.models, self.optimized_weights)
        }


class WeightedEnsemble(BaseEnsemble):
    """
    Weighted ensemble with optimized weights.

    This class combines predictions from multiple models using
    optimized weights that minimize the prediction error.
    """

    def __init__(
        self,
        estimators: List[Tuple[str, BaseEstimator]],
        weights: Optional[List[float]] = None,
        weight_optimization: str = "minimize_error",
        cv: int = 5,
        scoring: Optional[Union[str, Callable]] = None,
        task_type: str = "auto",
        random_state: Optional[int] = None,
    ):
        """
        Initialize WeightedEnsemble.

        Args:
            estimators: List of (name, estimator) tuples
            weights: Initial weights (None for optimization)
            weight_optimization: Method for weight optimization
            cv: Cross-validation folds for weight optimization
            scoring: Scoring function for optimization
            task_type: Type of task
            random_state: Random state
        """
        base_estimators = [est for _, est in estimators]
        super().__init__(base_estimators, task_type)

        self.estimators = estimators
        self.weights = weights
        self.weight_optimization = weight_optimization
        self.cv = cv
        self.scoring = scoring
        self.random_state = random_state
        self.fitted_estimators_ = []
        self.optimized_weights_ = None

    def fit(self, X, y, sample_weight=None):
        """
        Fit the weighted ensemble.

        Args:
            X: Training features
            y: Training targets
            sample_weight: Sample weights

        Returns:
            Self
        """
        # Validate input
        X, y = check_X_y(X, accept_sparse=["csc"])
        self._validate_estimators()

        # Infer task type
        self._infer_task_type(y)

        # Fit all estimators
        logger.info(f"Fitting {len(self.estimators)} estimators")
        self.fitted_estimators_ = []

        for name, estimator in self.estimators:
            logger.debug(f"Fitting {name}")
            fitted_est = clone(estimator)

            if sample_weight is not None and self._supports_sample_weight(fitted_est):
                fitted_est.fit(X, y, sample_weight=sample_weight)
            else:
                fitted_est.fit(X, y)

            self.fitted_estimators_.append((name, fitted_est))

        # Optimize weights if not provided
        if self.weights is None:
            self._optimize_weights(X, y)
        else:
            # Normalize provided weights
            total_weight = sum(self.weights)
            self.optimized_weights_ = np.array([w / total_weight for w in self.weights])

        self.is_fitted_ = True
        return self

    def _supports_sample_weight(self, estimator):
        """Check if estimator supports sample weights."""
        import inspect

        fit_params = inspect.signature(estimator.fit).parameters
        return "sample_weight" in fit_params

    def _optimize_weights(self, X, y):
        """Optimize ensemble weights using cross-validation."""
        from scipy.optimize import minimize
        from sklearn.model_selection import KFold

        logger.info("Optimizing ensemble weights")

        # Get CV predictions
        if self.task_type == "classification":
            kf = StratifiedKFold(
                n_splits=self.cv, shuffle=True, random_state=self.random_state
            )
        else:
            kf = KFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)

        cv_predictions = {name: [] for name, _ in self.estimators}
        y_true_cv = []

        for train_idx, val_idx in kf.split(X, y):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            y_true_cv.extend(y_val)

            # Get predictions from each estimator
            for name, estimator in self.estimators:
                est_clone = clone(estimator)
                est_clone.fit(X_train, y_train)

                if self.task_type == "regression":
                    pred = est_clone.predict(X_val)
                else:
                    # Use probabilities for classification
                    if hasattr(est_clone, "predict_proba"):
                        pred = est_clone.predict_proba(X_val)
                    else:
                        pred = est_clone.predict(X_val)

                cv_predictions[name].append(pred)

        # Concatenate predictions
        for name in cv_predictions:
            cv_predictions[name] = np.concatenate(cv_predictions[name])

        # Convert to arrays
        y_true_cv = np.array(y_true_cv)

        # Stack predictions
        if self.task_type == "regression":
            prediction_matrix = np.column_stack(
                [cv_predictions[name] for name, _ in self.estimators]
            )
        else:
            # For classification, handle probability matrices
            first_pred = list(cv_predictions.values())[0]
            if len(first_pred.shape) > 1:
                # Probabilities - stack them
                prediction_list = []
                for name, _ in self.estimators:
                    prediction_list.append(cv_predictions[name])
                prediction_matrix = np.stack(prediction_list, axis=0)
            else:
                # Hard predictions
                prediction_matrix = np.column_stack(
                    [cv_predictions[name] for name, _ in self.estimators]
                )

        # Define objective function
        def objective(weights):
            # Ensure weights sum to 1
            weights = weights / weights.sum()

            if self.task_type == "regression":
                # Weighted average of predictions
                weighted_pred = prediction_matrix @ weights
                # Mean squared error
                return np.mean((y_true_cv - weighted_pred) ** 2)
            else:
                # For classification, handle probability matrices
                if len(prediction_matrix.shape) == 3:
                    # Weighted average of probabilities
                    weighted_pred = np.sum(
                        prediction_matrix * weights[:, None, None], axis=0
                    )
                    # Cross-entropy loss
                    eps = 1e-15
                    weighted_pred = np.clip(weighted_pred, eps, 1 - eps)
                    # Convert y_true to one-hot if needed
                    if len(y_true_cv.shape) == 1:
                        from sklearn.preprocessing import LabelBinarizer

                        lb = LabelBinarizer()
                        y_true_binary = lb.fit_transform(y_true_cv)
                        if y_true_binary.shape[1] == 1:
                            # Binary case
                            y_true_binary = np.hstack(
                                [1 - y_true_binary, y_true_binary]
                            )
                    else:
                        y_true_binary = y_true_cv

                    # Calculate cross-entropy
                    return -np.mean(y_true_binary * np.log(weighted_pred))
                else:
                    # Hard predictions - use accuracy
                    weighted_pred = prediction_matrix @ weights
                    weighted_pred = np.round(weighted_pred).astype(int)
                    return -np.mean(weighted_pred == y_true_cv)

        # Initial weights
        x0 = np.array([1.0 / len(self.estimators)] * len(self.estimators))

        # Constraints: weights sum to 1
        constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1.0}

        # Bounds: weights between 0 and 1
        bounds = [(0, 1)] * len(self.estimators)

        # Optimize
        result = minimize(
            objective, x0, method="SLSQP", bounds=bounds, constraints=constraints
        )

        if result.success:
            self.optimized_weights_ = result.x / result.x.sum()
            logger.info(f"Optimized weights: {self.optimized_weights_}")
        else:
            logger.warning("Weight optimization failed, using equal weights")
            self.optimized_weights_ = np.array(
                [1.0 / len(self.estimators)] * len(self.estimators)
            )

    def predict(self, X):
        """
        Make predictions.

        Args:
            X: Features to predict

        Returns:
            Weighted predictions
        """
        check_is_fitted(self, ["fitted_estimators_", "optimized_weights_"])
        X = check_array(X, accept_sparse=["csc"])

        # Get predictions from all estimators
        predictions = []

        for (name, estimator), weight in zip(
            self.fitted_estimators_, self.optimized_weights_
        ):
            pred = estimator.predict(X) * weight
            predictions.append(pred)

        # Sum weighted predictions
        return np.sum(predictions, axis=0)

    def predict_proba(self, X):
        """
        Predict class probabilities (classification only).

        Args:
            X: Features to predict

        Returns:
            Weighted probability predictions
        """
        if self.task_type != "classification":
            raise AttributeError("predict_proba is only available for classification")

        check_is_fitted(self, ["fitted_estimators_", "optimized_weights_"])
        X = check_array(X, accept_sparse=["csc"])

        # Get probability predictions from all estimators
        proba_predictions = []

        for name, estimator in self.fitted_estimators_:
            if hasattr(estimator, "predict_proba"):
                proba = estimator.predict_proba(X)
                proba_predictions.append(proba)
            else:
                # Convert hard predictions to probabilities
                pred = estimator.predict(X)
                n_classes = len(np.unique(pred))
                proba = np.zeros((len(X), n_classes))
                for i, p in enumerate(pred):
                    proba[i, int(p)] = 1.0
                proba_predictions.append(proba)

        # Apply weights
        proba_predictions = np.array(proba_predictions)
        weighted_proba = np.sum(
            proba_predictions * self.optimized_weights_[:, np.newaxis, np.newaxis],
            axis=0,
        )

        # Normalize to ensure probabilities sum to 1
        weighted_proba = weighted_proba / weighted_proba.sum(axis=1, keepdims=True)

        return weighted_proba

    def get_estimator_weights(self):
        """
        Get the weights for each estimator.

        Returns:
            Dictionary mapping estimator names to weights
        """
        check_is_fitted(self, "optimized_weights_")

        return {
            name: weight
            for (name, _), weight in zip(self.estimators, self.optimized_weights_)
        }


class EnsembleModel:
    """
    Enhanced ensemble model wrapper with automatic hyperparameter tuning
    and model selection capabilities.
    """

    def __init__(
        self,
        task_type: str = "regression",
        model_type: str = "auto",
        random_state: int = 42,
    ):
        """
        Initialize the EnsembleModel.

        Args:
            task_type: Type of ML task ('regression' or 'classification')
            model_type: Type of ensemble model ('auto', 'rf', 'et', 'gb', 'xgb', 'lgb', 'cat')
            random_state: Random seed for reproducibility
        """
        self.task_type = task_type
        self.model_type = model_type
        self.random_state = random_state
        self.model = None
        self.best_params = None
        self.feature_importances_ = None
        self.model_scores = {}
        self.is_fitted = False

    def _get_model_class(self, model_type: str):
        """Get the appropriate model class based on task and model type."""
        model_mapping = {
            "regression": {
                "rf": "RandomForestRegressor",
                "et": "ExtraTreesRegressor",
                "gb": "GradientBoostingRegressor",
                "xgb": "XGBRegressor",
                "lgb": "LGBMRegressor",
                "cat": "CatBoostRegressor",
            },
            "classification": {
                "rf": "RandomForestClassifier",
                "et": "ExtraTreesClassifier",
                "gb": "GradientBoostingClassifier",
                "xgb": "XGBClassifier",
                "lgb": "LGBMClassifier",
                "cat": "CatBoostClassifier",
            },
        }

        try:
            from sklearn.ensemble import (ExtraTreesClassifier,
                                          ExtraTreesRegressor,
                                          GradientBoostingClassifier,
                                          GradientBoostingRegressor,
                                          RandomForestClassifier,
                                          RandomForestRegressor)

            # Try importing optional libraries
            try:
                from xgboost import XGBClassifier, XGBRegressor
            except ImportError:
                logger.warning("XGBoost not installed")

            try:
                from lightgbm import LGBMClassifier, LGBMRegressor
            except ImportError:
                logger.warning("LightGBM not installed")

            try:
                from catboost import CatBoostClassifier, CatBoostRegressor
            except ImportError:
                logger.warning("CatBoost not installed")

            # Get the model class
            model_name = model_mapping[self.task_type][model_type]
            return eval(model_name)

        except Exception as e:
            logger.error(f"Error loading model class: {e}")
            # Fallback to RandomForest
            if self.task_type == "regression":
                return RandomForestRegressor
            else:
                return RandomForestClassifier

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        tune_hyperparameters: bool = False,
        cv: int = 5,
        scoring: Optional[str] = None,
        n_iter: int = 50,
    ):
        """
        Fit the ensemble model.

        Args:
            X: Training features
            y: Training targets
            tune_hyperparameters: Whether to tune hyperparameters
            cv: Cross-validation folds
            scoring: Scoring metric
            n_iter: Number of iterations for random search

        Returns:
            Self
        """
        # Auto-select model if needed
        if self.model_type == "auto":
            self._auto_select_model(X, y, cv, scoring)
        else:
            ModelClass = self._get_model_class(self.model_type)
            self.model = ModelClass(random_state=self.random_state)

        # Tune hyperparameters if requested
        if tune_hyperparameters:
            self._tune_hyperparameters(X, y, cv, scoring, n_iter)
        else:
            # Fit with default parameters
            self.model.fit(X, y)

        # Extract feature importances if available
        if hasattr(self.model, "feature_importances_"):
            self.feature_importances_ = self.model.feature_importances_

        self.is_fitted = True
        return self

    def _auto_select_model(self, X, y, cv, scoring):
        """Automatically select the best model type."""
        logger.info("Auto-selecting best model type")

        candidate_models = ["rf", "gb"]  # Start with basic models

        # Add optional models if available
        try:
            import xgboost

            candidate_models.append("xgb")
        except ImportError:
            pass

        try:
            import lightgbm

            candidate_models.append("lgb")
        except ImportError:
            pass

        best_score = -np.inf
        best_model_type = "rf"

        for model_type in candidate_models:
            try:
                ModelClass = self._get_model_class(model_type)
                model = ModelClass(random_state=self.random_state)

                # Quick cross-validation
                from sklearn.model_selection import cross_val_score

                scores = cross_val_score(model, X, y, cv=min(cv, 3), scoring=scoring)
                mean_score = np.mean(scores)

                logger.info(f"{model_type}: {mean_score:.4f}")
                self.model_scores[model_type] = mean_score

                if mean_score > best_score:
                    best_score = mean_score
                    best_model_type = model_type

            except Exception as e:
                logger.warning(f"Failed to evaluate {model_type}: {e}")

        # Set the best model
        self.model_type = best_model_type
        ModelClass = self._get_model_class(best_model_type)
        self.model = ModelClass(random_state=self.random_state)
        logger.info(f"Selected model: {best_model_type}")

    def _tune_hyperparameters(self, X, y, cv, scoring, n_iter):
        """Tune model hyperparameters."""
        from sklearn.model_selection import RandomizedSearchCV

        logger.info("Tuning hyperparameters")

        # Define parameter grids for different models
        param_grids = {
            "rf": {
                "n_estimators": [100, 200, 300],
                "max_depth": [None, 10, 20, 30],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
            },
            "gb": {
                "n_estimators": [100, 200, 300],
                "learning_rate": [0.01, 0.1, 0.3],
                "max_depth": [3, 5, 7],
                "subsample": [0.8, 0.9, 1.0],
            },
            "xgb": {
                "n_estimators": [100, 200, 300],
                "learning_rate": [0.01, 0.1, 0.3],
                "max_depth": [3, 5, 7],
                "subsample": [0.8, 0.9, 1.0],
                "colsample_bytree": [0.8, 0.9, 1.0],
            },
            "lgb": {
                "n_estimators": [100, 200, 300],
                "learning_rate": [0.01, 0.1, 0.3],
                "num_leaves": [31, 50, 100],
                "feature_fraction": [0.8, 0.9, 1.0],
                "bagging_fraction": [0.8, 0.9, 1.0],
            },
        }

        # Get parameter grid for current model
        model_key = self.model_type if self.model_type in param_grids else "rf"
        param_grid = param_grids[model_key]

        # Run random search
        random_search = RandomizedSearchCV(
            self.model,
            param_distributions=param_grid,
            n_iter=min(n_iter, 20),
            cv=cv,
            scoring=scoring,
            random_state=self.random_state,
            n_jobs=-1,
        )

        random_search.fit(X, y)

        # Update model with best parameters
        self.model = random_search.best_estimator_
        self.best_params = random_search.best_params_
        logger.info(f"Best parameters: {self.best_params}")

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet")
        return self.model.predict(X)

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Make probability predictions (classification only)."""
        if self.task_type != "classification":
            raise ValueError("predict_proba is only available for classification")
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet")
        return self.model.predict_proba(X)

    def get_feature_importance(self) -> np.ndarray:
        """Get feature importances."""
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet")
        if self.feature_importances_ is None:
            raise ValueError("Model does not support feature importances")
        return self.feature_importances_


# Example usage
if __name__ == "__main__":
    # Generate sample data
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

    X, y = make_classification(
        n_samples=1000, n_features=20, n_classes=2, random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Example 1: Voting Ensemble
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC

    estimators = [
        ("rf", RandomForestClassifier(n_estimators=10, random_state=42)),
        ("svc", SVC(probability=True, random_state=42)),
        ("lr", LogisticRegression(random_state=42)),
    ]

    voting = VotingEnsemble(estimators=estimators, voting="soft")
    voting.fit(X_train, y_train)
    print(f"Voting accuracy: {voting.predict(X_test).mean():.4f}")

    # Example 2: Model Stacker
    stacker = ModelStacker(base_models=estimators, task_type="classification")
    stacker.fit(X_train, y_train)
    print(f"Stacking accuracy: {(stacker.predict(X_test) == y_test).mean():.4f}")

    # Example 3: EnsembleModel with auto-selection
    ensemble = EnsembleModel(task_type="classification", model_type="auto")
    ensemble.fit(X_train, y_train, tune_hyperparameters=True, cv=3)
    print(f"Auto-ensemble accuracy: {(ensemble.predict(X_test) == y_test).mean():.4f}")
