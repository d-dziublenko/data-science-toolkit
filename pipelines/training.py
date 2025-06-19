"""
pipelines/training.py
End-to-end training pipeline for machine learning projects.
"""

import itertools
import json
import logging
import pickle
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.compose import ColumnTransformer
from sklearn.metrics import make_scorer
from sklearn.model_selection import (GridSearchCV, KFold, RandomizedSearchCV,
                                     StratifiedKFold, cross_val_score,
                                     train_test_split)
from sklearn.pipeline import Pipeline
from tqdm import tqdm

# Import from our modules
from core.data_loader import TabularDataLoader
from core.feature_engineering import FeatureEngineer, FeatureSelector
from core.preprocessing import DataPreprocessor, FeatureTransformer
from evaluation.metrics import ModelEvaluator
from evaluation.uncertainty import UncertaintyQuantifier
from evaluation.visualization import ModelVisualizer
from models.base import AutoML
from models.ensemble import EnsembleModel, ModelBlender, ModelStacker
from utils.parallel import ParallelProcessor

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """
    Configuration for training pipeline.

    This dataclass holds all configuration parameters for the training process,
    making it easy to reproduce experiments and manage hyperparameters.
    """

    # Data configuration
    data_path: Union[str, Path]
    target_column: str
    test_size: float = 0.2
    validation_size: Optional[float] = 0.2
    stratify: bool = False
    random_state: int = 42

    # Preprocessing configuration
    numeric_features: Optional[List[str]] = None
    categorical_features: Optional[List[str]] = None
    scaling_method: str = "standard"
    encoding_method: str = "onehot"
    handle_missing: str = "impute"
    remove_outliers: bool = False
    outlier_method: str = "isolation_forest"

    # Feature engineering configuration
    create_polynomials: bool = False
    polynomial_degree: int = 2
    create_interactions: bool = False
    select_features: bool = True
    feature_selection_method: str = "mutual_information"
    n_features_to_select: Optional[int] = None

    # Model configuration
    task_type: str = "auto"  # 'regression', 'classification', or 'auto'
    model_types: List[str] = field(
        default_factory=lambda: ["random_forest", "xgboost", "lightgbm"]
    )
    ensemble_method: Optional[str] = "stacking"

    # Hyperparameter tuning configuration
    tune_hyperparameters: bool = True
    tuning_method: str = "bayesian"  # 'grid', 'random', 'bayesian'
    n_trials: int = 100
    cv_folds: int = 5
    scoring_metric: Optional[str] = None

    # Training configuration
    early_stopping: bool = True
    n_jobs: int = -1
    verbose: int = 1

    # Output configuration
    experiment_name: str = field(
        default_factory=lambda: f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    output_dir: Union[str, Path] = "./outputs"
    save_models: bool = True
    save_plots: bool = True
    save_predictions: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        config_dict = {}
        for field_name in self.__dataclass_fields__:
            value = getattr(self, field_name)
            if isinstance(value, Path):
                value = str(value)
            config_dict[field_name] = value
        return config_dict

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "TrainingConfig":
        """Create configuration from dictionary."""
        return cls(**config_dict)

    def validate(self):
        """Validate configuration parameters."""
        # Validate data path
        if not Path(self.data_path).exists():
            raise ValueError(f"Data path does not exist: {self.data_path}")

        # Validate test/validation sizes
        if self.test_size <= 0 or self.test_size >= 1:
            raise ValueError("test_size must be between 0 and 1")

        if self.validation_size is not None:
            if self.validation_size <= 0 or self.validation_size >= 1:
                raise ValueError("validation_size must be between 0 and 1")
            if self.test_size + self.validation_size >= 1:
                raise ValueError("test_size + validation_size must be less than 1")

        # Validate model types
        valid_models = [
            "random_forest",
            "xgboost",
            "lightgbm",
            "catboost",
            "linear",
            "svm",
            "neural_network",
            "auto",
        ]
        for model in self.model_types:
            if model not in valid_models:
                raise ValueError(f"Invalid model type: {model}")

        # Validate ensemble method
        if self.ensemble_method not in [None, "stacking", "blending", "voting"]:
            raise ValueError(f"Invalid ensemble method: {self.ensemble_method}")


class ModelTrainer:
    """
    Advanced model training with hyperparameter tuning and cross-validation.

    This class handles the core model training logic including hyperparameter
    optimization, cross-validation, and early stopping.
    """

    def __init__(self, config: TrainingConfig):
        """
        Initialize ModelTrainer.

        Args:
            config: Training configuration
        """
        self.config = config
        self.models = {}
        self.best_params = {}
        self.cv_scores = {}
        self.training_history = {}

    def train_model(
        self,
        model: BaseEstimator,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        model_name: str = "model",
    ) -> BaseEstimator:
        """
        Train a single model with hyperparameter tuning.

        Args:
            model: Model to train
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            model_name: Name for the model

        Returns:
            Trained model
        """
        logger.info(f"Training {model_name}")

        if self.config.tune_hyperparameters:
            # Get hyperparameter space
            param_space = self._get_param_space(model_name)

            if param_space:
                # Perform hyperparameter tuning
                best_model, best_params = self._tune_hyperparameters(
                    model, X_train, y_train, param_space
                )
                self.best_params[model_name] = best_params
                model = best_model
            else:
                # Train with default parameters
                model.fit(X_train, y_train)
        else:
            # Train without tuning
            if hasattr(model, "fit"):
                if X_val is not None and self._supports_validation(model):
                    # Use validation set for early stopping
                    model.fit(
                        X_train,
                        y_train,
                        eval_set=[(X_val, y_val)],
                        early_stopping_rounds=20,
                        verbose=False,
                    )
                else:
                    model.fit(X_train, y_train)

        # Store model
        self.models[model_name] = model

        # Calculate cross-validation scores
        if self.config.cv_folds > 1:
            cv_scores = self._cross_validate(model, X_train, y_train)
            self.cv_scores[model_name] = cv_scores
            logger.info(
                f"{model_name} CV Score: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})"
            )

        return model

    def train_ensemble(
        self,
        base_models: Dict[str, BaseEstimator],
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> BaseEstimator:
        """
        Train an ensemble of models.

        Args:
            base_models: Dictionary of base models
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)

        Returns:
            Trained ensemble model
        """
        if self.config.ensemble_method == "stacking":
            ensemble = ModelStacker(
                base_models=list(base_models.items()),
                task_type=self.config.task_type,
                cv_folds=self.config.cv_folds,
            )
        elif self.config.ensemble_method == "blending":
            ensemble = ModelBlender(models=list(base_models.items()))
        elif self.config.ensemble_method == "voting":
            from sklearn.ensemble import VotingClassifier, VotingRegressor

            if self.config.task_type == "regression":
                ensemble = VotingRegressor(list(base_models.items()))
            else:
                ensemble = VotingClassifier(list(base_models.items()))
        else:
            raise ValueError(f"Unknown ensemble method: {self.config.ensemble_method}")

        # Train ensemble
        ensemble.fit(X_train, y_train)

        # Optimize weights for blending
        if self.config.ensemble_method == "blending" and X_val is not None:
            ensemble.optimize_weights(X_val, y_val)

        return ensemble

    def _get_param_space(self, model_name: str) -> Dict[str, Any]:
        """Get hyperparameter space for a model."""
        param_spaces = {
            "random_forest": {
                "n_estimators": [100, 200, 300],
                "max_depth": [None, 10, 20, 30],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
            },
            "xgboost": {
                "n_estimators": [100, 200, 300],
                "max_depth": [3, 5, 7, 9],
                "learning_rate": [0.01, 0.05, 0.1, 0.3],
                "subsample": [0.8, 0.9, 1.0],
            },
            "lightgbm": {
                "n_estimators": [100, 200, 300],
                "num_leaves": [31, 50, 100],
                "learning_rate": [0.01, 0.05, 0.1],
                "feature_fraction": [0.8, 0.9, 1.0],
            },
            "neural_network": {
                "hidden_layers": [[64, 32], [128, 64], [128, 64, 32]],
                "learning_rate": [0.001, 0.01, 0.1],
                "dropout_rate": [0.2, 0.3, 0.4],
                "batch_norm": [True, False],
            },
        }

        return param_spaces.get(model_name, {})

    def _tune_hyperparameters(
        self,
        model: BaseEstimator,
        X: pd.DataFrame,
        y: pd.Series,
        param_space: Dict[str, Any],
    ) -> Tuple[BaseEstimator, Dict[str, Any]]:
        """Perform hyperparameter tuning."""
        if self.config.tuning_method == "grid":
            search = GridSearchCV(
                model,
                param_space,
                cv=self.config.cv_folds,
                scoring=self.config.scoring_metric,
                n_jobs=self.config.n_jobs,
                verbose=self.config.verbose,
            )
        elif self.config.tuning_method == "random":
            search = RandomizedSearchCV(
                model,
                param_space,
                n_iter=self.config.n_trials,
                cv=self.config.cv_folds,
                scoring=self.config.scoring_metric,
                n_jobs=self.config.n_jobs,
                verbose=self.config.verbose,
                random_state=self.config.random_state,
            )
        elif self.config.tuning_method == "bayesian":
            # Use Optuna for Bayesian optimization
            try:
                import optuna

                return self._bayesian_optimization(model, X, y, param_space)
            except ImportError:
                warnings.warn("Optuna not installed. Falling back to random search.")
                search = RandomizedSearchCV(
                    model,
                    param_space,
                    n_iter=self.config.n_trials,
                    cv=self.config.cv_folds,
                    scoring=self.config.scoring_metric,
                    n_jobs=self.config.n_jobs,
                    verbose=self.config.verbose,
                    random_state=self.config.random_state,
                )

        search.fit(X, y)
        return search.best_estimator_, search.best_params_

    def _bayesian_optimization(
        self,
        model: BaseEstimator,
        X: pd.DataFrame,
        y: pd.Series,
        param_space: Dict[str, Any],
    ) -> Tuple[BaseEstimator, Dict[str, Any]]:
        """Perform Bayesian optimization using Optuna."""
        import optuna

        def objective(trial):
            # Sample parameters
            params = {}
            for param_name, param_values in param_space.items():
                if isinstance(param_values[0], (int, float)):
                    if isinstance(param_values[0], int):
                        params[param_name] = trial.suggest_int(
                            param_name, min(param_values), max(param_values)
                        )
                    else:
                        params[param_name] = trial.suggest_float(
                            param_name, min(param_values), max(param_values)
                        )
                else:
                    params[param_name] = trial.suggest_categorical(
                        param_name, param_values
                    )

            # Create model with parameters
            model_clone = clone(model)
            model_clone.set_params(**params)

            # Cross-validation
            scores = cross_val_score(
                model_clone,
                X,
                y,
                cv=self.config.cv_folds,
                scoring=self.config.scoring_metric,
                n_jobs=1,  # Avoid nested parallelism
            )

            return scores.mean()

        # Create study
        study = optuna.create_study(
            direction="maximize", pruner=optuna.pruners.MedianPruner()
        )

        # Optimize
        study.optimize(
            objective, n_trials=self.config.n_trials, n_jobs=self.config.n_jobs
        )

        # Get best model
        best_params = study.best_params
        best_model = clone(model)
        best_model.set_params(**best_params)
        best_model.fit(X, y)

        return best_model, best_params

    def _cross_validate(
        self, model: BaseEstimator, X: pd.DataFrame, y: pd.Series
    ) -> np.ndarray:
        """Perform cross-validation."""
        if self.config.task_type == "classification":
            cv = StratifiedKFold(
                n_splits=self.config.cv_folds,
                shuffle=True,
                random_state=self.config.random_state,
            )
        else:
            cv = KFold(
                n_splits=self.config.cv_folds,
                shuffle=True,
                random_state=self.config.random_state,
            )

        scores = cross_val_score(
            model,
            X,
            y,
            cv=cv,
            scoring=self.config.scoring_metric,
            n_jobs=self.config.n_jobs,
        )

        return scores

    def _supports_validation(self, model: BaseEstimator) -> bool:
        """Check if model supports validation set during training."""
        model_name = type(model).__name__.lower()
        return any(name in model_name for name in ["xgb", "lgb", "catboost"])


class CrossValidator:
    """
    Advanced cross-validation with multiple strategies.

    Provides various cross-validation strategies including time series
    cross-validation, stratified cross-validation, and custom strategies.
    """

    def __init__(
        self, cv_strategy: str = "kfold", n_splits: int = 5, random_state: int = 42
    ):
        """
        Initialize CrossValidator.

        Args:
            cv_strategy: Cross-validation strategy
            n_splits: Number of splits
            random_state: Random seed
        """
        self.cv_strategy = cv_strategy
        self.n_splits = n_splits
        self.random_state = random_state
        self.cv_results = {}

    def validate(
        self,
        model: BaseEstimator,
        X: pd.DataFrame,
        y: pd.Series,
        scoring: Union[str, Callable, List[str]] = "auto",
        return_train_score: bool = True,
        return_predictions: bool = True,
    ) -> Dict[str, Any]:
        """
        Perform cross-validation with detailed results.

        Args:
            model: Model to validate
            X: Features
            y: Targets
            scoring: Scoring metric(s)
            return_train_score: Whether to return training scores
            return_predictions: Whether to return predictions

        Returns:
            Dictionary with cross-validation results
        """
        # Get cross-validation splitter
        cv = self._get_cv_splitter(X, y)

        # Initialize results storage
        results = {
            "test_scores": [],
            "train_scores": [] if return_train_score else None,
            "fit_times": [],
            "score_times": [],
        }

        if return_predictions:
            results["predictions"] = np.zeros_like(y, dtype=float)
            results["prediction_indices"] = []

        # Perform cross-validation
        for fold, (train_idx, test_idx) in enumerate(cv.split(X, y)):
            logger.info(f"Processing fold {fold + 1}/{self.n_splits}")

            # Split data
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            # Train model
            start_time = datetime.now()
            model_clone = clone(model)
            model_clone.fit(X_train, y_train)
            fit_time = (datetime.now() - start_time).total_seconds()
            results["fit_times"].append(fit_time)

            # Score model
            start_time = datetime.now()

            if isinstance(scoring, list):
                test_score = {}
                train_score = {}
                for metric in scoring:
                    scorer = self._get_scorer(metric)
                    test_score[metric] = scorer(model_clone, X_test, y_test)
                    if return_train_score:
                        train_score[metric] = scorer(model_clone, X_train, y_train)
            else:
                scorer = self._get_scorer(scoring)
                test_score = scorer(model_clone, X_test, y_test)
                if return_train_score:
                    train_score = scorer(model_clone, X_train, y_train)

            score_time = (datetime.now() - start_time).total_seconds()

            results["test_scores"].append(test_score)
            if return_train_score:
                results["train_scores"].append(train_score)
            results["score_times"].append(score_time)

            # Store predictions
            if return_predictions:
                predictions = model_clone.predict(X_test)
                results["predictions"][test_idx] = predictions
                results["prediction_indices"].extend(test_idx)

        # Calculate summary statistics
        results["mean_test_score"] = np.mean(results["test_scores"])
        results["std_test_score"] = np.std(results["test_scores"])
        if return_train_score:
            results["mean_train_score"] = np.mean(results["train_scores"])
            results["std_train_score"] = np.std(results["train_scores"])

        self.cv_results = results
        return results

    def _get_cv_splitter(self, X: pd.DataFrame, y: pd.Series):
        """Get appropriate cross-validation splitter."""
        if self.cv_strategy == "kfold":
            return KFold(
                n_splits=self.n_splits, shuffle=True, random_state=self.random_state
            )
        elif self.cv_strategy == "stratified":
            return StratifiedKFold(
                n_splits=self.n_splits, shuffle=True, random_state=self.random_state
            )
        elif self.cv_strategy == "timeseries":
            from sklearn.model_selection import TimeSeriesSplit

            return TimeSeriesSplit(n_splits=self.n_splits)
        elif self.cv_strategy == "group":
            from sklearn.model_selection import GroupKFold

            return GroupKFold(n_splits=self.n_splits)
        else:
            raise ValueError(f"Unknown CV strategy: {self.cv_strategy}")

    def _get_scorer(self, scoring: Union[str, Callable]):
        """Get scoring function."""
        if callable(scoring):
            return scoring

        if scoring == "auto":
            # Auto-detect based on problem type
            # This is simplified - in practice, you'd detect from the model
            return "neg_mean_squared_error"

        return scoring

    def plot_cv_results(self, save_path: Optional[str] = None):
        """Plot cross-validation results."""
        if not self.cv_results:
            raise ValueError("No CV results to plot. Run validate() first.")

        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Plot scores across folds
        ax = axes[0, 0]
        folds = range(1, len(self.cv_results["test_scores"]) + 1)
        ax.plot(folds, self.cv_results["test_scores"], "b-o", label="Test Score")
        if self.cv_results["train_scores"]:
            ax.plot(folds, self.cv_results["train_scores"], "r-o", label="Train Score")
        ax.set_xlabel("Fold")
        ax.set_ylabel("Score")
        ax.set_title("Scores Across Folds")
        ax.legend()
        ax.grid(True)

        # Plot timing
        ax = axes[0, 1]
        ax.bar(folds, self.cv_results["fit_times"], alpha=0.7, label="Fit Time")
        ax.bar(
            folds,
            self.cv_results["score_times"],
            alpha=0.7,
            bottom=self.cv_results["fit_times"],
            label="Score Time",
        )
        ax.set_xlabel("Fold")
        ax.set_ylabel("Time (seconds)")
        ax.set_title("Timing Across Folds")
        ax.legend()

        # Plot score distribution
        ax = axes[1, 0]
        ax.boxplot([self.cv_results["test_scores"]], labels=["Test"])
        ax.set_ylabel("Score")
        ax.set_title("Score Distribution")
        ax.grid(True)

        # Plot learning curve if available
        ax = axes[1, 1]
        if self.cv_results["train_scores"]:
            train_mean = np.mean(self.cv_results["train_scores"])
            train_std = np.std(self.cv_results["train_scores"])
            test_mean = self.cv_results["mean_test_score"]
            test_std = self.cv_results["std_test_score"]

            ax.bar(
                ["Train", "Test"],
                [train_mean, test_mean],
                yerr=[train_std, test_std],
                capsize=10,
            )
            ax.set_ylabel("Score")
            ax.set_title("Mean Scores with Std Dev")
            ax.grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()


class HyperparameterTuner:
    """
    Advanced hyperparameter tuning with multiple optimization strategies.

    Supports grid search, random search, Bayesian optimization, and
    evolutionary algorithms for hyperparameter optimization.
    """

    def __init__(
        self,
        optimization_method: str = "bayesian",
        n_trials: int = 100,
        cv_folds: int = 5,
        scoring_metric: Optional[str] = None,
        n_jobs: int = -1,
        random_state: int = 42,
    ):
        """
        Initialize HyperparameterTuner.

        Args:
            optimization_method: Optimization method
            n_trials: Number of trials/iterations
            cv_folds: Number of CV folds
            scoring_metric: Metric to optimize
            n_jobs: Number of parallel jobs
            random_state: Random seed
        """
        self.optimization_method = optimization_method
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.scoring_metric = scoring_metric
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.optimization_history = []
        self.best_params = {}
        self.best_score = None

    def optimize(
        self,
        model: BaseEstimator,
        param_space: Dict[str, Any],
        X: pd.DataFrame,
        y: pd.Series,
        groups: Optional[pd.Series] = None,
    ) -> Tuple[BaseEstimator, Dict[str, Any]]:
        """
        Optimize hyperparameters for a model.

        Args:
            model: Model to optimize
            param_space: Parameter search space
            X: Features
            y: Targets
            groups: Group labels for GroupKFold

        Returns:
            Tuple of (best_model, best_params)
        """
        logger.info(f"Starting {self.optimization_method} hyperparameter optimization")

        if self.optimization_method == "grid":
            return self._grid_search(model, param_space, X, y, groups)
        elif self.optimization_method == "random":
            return self._random_search(model, param_space, X, y, groups)
        elif self.optimization_method == "bayesian":
            return self._bayesian_optimization(model, param_space, X, y, groups)
        elif self.optimization_method == "evolutionary":
            return self._evolutionary_optimization(model, param_space, X, y, groups)
        elif self.optimization_method == "hyperband":
            return self._hyperband_optimization(model, param_space, X, y, groups)
        else:
            raise ValueError(f"Unknown optimization method: {self.optimization_method}")

    def _grid_search(self, model, param_space, X, y, groups=None):
        """Perform grid search optimization."""
        from sklearn.model_selection import GridSearchCV

        search = GridSearchCV(
            model,
            param_space,
            cv=self._get_cv(groups),
            scoring=self.scoring_metric,
            n_jobs=self.n_jobs,
            verbose=2,
            refit=True,
        )

        search.fit(X, y, groups=groups)

        self.best_params = search.best_params_
        self.best_score = search.best_score_
        self.optimization_history = search.cv_results_

        return search.best_estimator_, search.best_params_

    def _random_search(self, model, param_space, X, y, groups=None):
        """Perform random search optimization."""
        from sklearn.model_selection import RandomizedSearchCV

        search = RandomizedSearchCV(
            model,
            param_space,
            n_iter=self.n_trials,
            cv=self._get_cv(groups),
            scoring=self.scoring_metric,
            n_jobs=self.n_jobs,
            verbose=2,
            random_state=self.random_state,
            refit=True,
        )

        search.fit(X, y, groups=groups)

        self.best_params = search.best_params_
        self.best_score = search.best_score_
        self.optimization_history = search.cv_results_

        return search.best_estimator_, search.best_params_

    def _bayesian_optimization(self, model, param_space, X, y, groups=None):
        """Perform Bayesian optimization using Optuna."""
        try:
            import optuna
            from optuna.samplers import TPESampler
        except ImportError:
            warnings.warn("Optuna not installed. Falling back to random search.")
            return self._random_search(model, param_space, X, y, groups)

        def objective(trial):
            # Sample parameters
            params = self._sample_params(trial, param_space)

            # Create model with parameters
            model_clone = clone(model)
            model_clone.set_params(**params)

            # Cross-validation
            scores = cross_val_score(
                model_clone,
                X,
                y,
                cv=self._get_cv(groups),
                groups=groups,
                scoring=self.scoring_metric,
                n_jobs=1,  # Avoid nested parallelism
            )

            # Store trial info
            self.optimization_history.append(
                {
                    "params": params,
                    "mean_score": scores.mean(),
                    "std_score": scores.std(),
                    "scores": scores,
                }
            )

            return scores.mean()

        # Create study
        sampler = TPESampler(seed=self.random_state)
        study = optuna.create_study(
            direction="maximize", sampler=sampler, pruner=optuna.pruners.MedianPruner()
        )

        # Optimize
        study.optimize(
            objective,
            n_trials=self.n_trials,
            n_jobs=self.n_jobs,
            show_progress_bar=True,
        )

        # Get best model
        self.best_params = study.best_params
        self.best_score = study.best_value

        best_model = clone(model)
        best_model.set_params(**self.best_params)
        best_model.fit(X, y)

        return best_model, self.best_params

    def _evolutionary_optimization(self, model, param_space, X, y, groups=None):
        """Perform evolutionary optimization using genetic algorithms."""
        try:
            from sklearn_genetic import GASearchCV
            from sklearn_genetic.space import Categorical, Continuous, Integer
        except ImportError:
            warnings.warn(
                "sklearn-genetic not installed. Falling back to Bayesian optimization."
            )
            return self._bayesian_optimization(model, param_space, X, y, groups)

        # Convert param_space to sklearn-genetic format
        ga_param_space = {}
        for param, values in param_space.items():
            if isinstance(values[0], bool):
                ga_param_space[param] = Categorical(values)
            elif isinstance(values[0], int):
                ga_param_space[param] = Integer(min(values), max(values))
            elif isinstance(values[0], float):
                ga_param_space[param] = Continuous(min(values), max(values))
            else:
                ga_param_space[param] = Categorical(values)

        search = GASearchCV(
            model,
            ga_param_space,
            cv=self._get_cv(groups),
            scoring=self.scoring_metric,
            n_jobs=self.n_jobs,
            verbose=True,
            generations=self.n_trials // 10,
            population_size=10,
        )

        search.fit(X, y, groups=groups)

        self.best_params = search.best_params_
        self.best_score = search.best_score_

        return search.best_estimator_, search.best_params_

    def _hyperband_optimization(self, model, param_space, X, y, groups=None):
        """Perform Hyperband optimization."""
        # This is a simplified implementation
        # In practice, you'd use a library like keras-tuner or ray[tune]
        warnings.warn("Hyperband not fully implemented. Using Bayesian optimization.")
        return self._bayesian_optimization(model, param_space, X, y, groups)

    def _sample_params(self, trial, param_space):
        """Sample parameters for Optuna trial."""
        params = {}
        for param_name, param_values in param_space.items():
            if isinstance(param_values, dict):
                # Handle different parameter types
                param_type = param_values.get("type", "categorical")
                if param_type == "int":
                    params[param_name] = trial.suggest_int(
                        param_name,
                        param_values["low"],
                        param_values["high"],
                        step=param_values.get("step", 1),
                    )
                elif param_type == "float":
                    params[param_name] = trial.suggest_float(
                        param_name,
                        param_values["low"],
                        param_values["high"],
                        log=param_values.get("log", False),
                    )
                elif param_type == "categorical":
                    params[param_name] = trial.suggest_categorical(
                        param_name, param_values["choices"]
                    )
            elif isinstance(param_values[0], bool):
                params[param_name] = trial.suggest_categorical(param_name, param_values)
            elif isinstance(param_values[0], int):
                params[param_name] = trial.suggest_int(
                    param_name, min(param_values), max(param_values)
                )
            elif isinstance(param_values[0], float):
                params[param_name] = trial.suggest_float(
                    param_name, min(param_values), max(param_values)
                )
            else:
                params[param_name] = trial.suggest_categorical(param_name, param_values)

        return params

    def _get_cv(self, groups=None):
        """Get cross-validation splitter."""
        if groups is not None:
            from sklearn.model_selection import GroupKFold

            return GroupKFold(n_splits=self.cv_folds)
        else:
            return self.cv_folds

    def plot_optimization_history(self, save_path: Optional[str] = None):
        """Plot optimization history."""
        if not self.optimization_history:
            raise ValueError("No optimization history available.")

        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Extract scores
        if isinstance(self.optimization_history, dict):
            # From GridSearchCV or RandomizedSearchCV
            mean_scores = self.optimization_history["mean_test_score"]
            std_scores = self.optimization_history["std_test_score"]
            iterations = range(len(mean_scores))
        else:
            # From custom optimization
            mean_scores = [h["mean_score"] for h in self.optimization_history]
            std_scores = [h["std_score"] for h in self.optimization_history]
            iterations = range(len(mean_scores))

        # Plot score progression
        ax = axes[0, 0]
        ax.plot(iterations, mean_scores, "b-", label="Mean Score")
        ax.fill_between(
            iterations,
            np.array(mean_scores) - np.array(std_scores),
            np.array(mean_scores) + np.array(std_scores),
            alpha=0.3,
        )
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Score")
        ax.set_title("Optimization Progress")
        ax.legend()
        ax.grid(True)

        # Plot best score over time
        ax = axes[0, 1]
        best_scores = np.maximum.accumulate(mean_scores)
        ax.plot(iterations, best_scores, "g-", linewidth=2)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Best Score")
        ax.set_title("Best Score Over Time")
        ax.grid(True)

        # Plot score distribution
        ax = axes[1, 0]
        ax.hist(mean_scores, bins=20, alpha=0.7, edgecolor="black")
        ax.axvline(
            self.best_score,
            color="red",
            linestyle="--",
            label=f"Best: {self.best_score:.4f}",
        )
        ax.set_xlabel("Score")
        ax.set_ylabel("Frequency")
        ax.set_title("Score Distribution")
        ax.legend()

        # Plot parameter importance (if available)
        ax = axes[1, 1]
        if hasattr(self, "param_importance"):
            params = list(self.param_importance.keys())
            importance = list(self.param_importance.values())
            ax.barh(params, importance)
            ax.set_xlabel("Importance")
            ax.set_title("Parameter Importance")
        else:
            ax.text(
                0.5,
                0.5,
                "Parameter importance\nnot available",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()


class AutoMLPipeline:
    """
    Automated machine learning pipeline with minimal configuration.

    This class provides a high-level interface for automatic model selection,
    hyperparameter tuning, and feature engineering.
    """

    def __init__(
        self,
        task_type: str = "auto",
        time_budget: int = 3600,
        optimization_metric: Optional[str] = None,
        ensemble: bool = True,
        feature_engineering: bool = True,
        n_jobs: int = -1,
        random_state: int = 42,
    ):
        """
        Initialize AutoMLPipeline.

        Args:
            task_type: Task type ('regression', 'classification', 'auto')
            time_budget: Time budget in seconds
            optimization_metric: Metric to optimize
            ensemble: Whether to create ensemble
            feature_engineering: Whether to perform feature engineering
            n_jobs: Number of parallel jobs
            random_state: Random seed
        """
        self.task_type = task_type
        self.time_budget = time_budget
        self.optimization_metric = optimization_metric
        self.ensemble = ensemble
        self.feature_engineering = feature_engineering
        self.n_jobs = n_jobs
        self.random_state = random_state

        # Components
        self.preprocessor = None
        self.feature_engineer = None
        self.models = {}
        self.best_model = None
        self.results = {}

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> "AutoMLPipeline":
        """
        Fit the AutoML pipeline.

        Args:
            X: Training features
            y: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)

        Returns:
            Self
        """
        start_time = datetime.now()

        # Auto-detect task type
        if self.task_type == "auto":
            self.task_type = self._detect_task_type(y)
            logger.info(f"Detected task type: {self.task_type}")

        # Auto-detect optimization metric
        if self.optimization_metric is None:
            self.optimization_metric = self._get_default_metric()

        # Split validation set if not provided
        if X_val is None:
            X, X_val, y, y_val = train_test_split(
                X,
                y,
                test_size=0.2,
                random_state=self.random_state,
                stratify=y if self.task_type == "classification" else None,
            )

        # Preprocessing
        logger.info("Starting preprocessing...")
        X, X_val = self._preprocess(X, X_val)

        # Feature engineering
        if self.feature_engineering:
            logger.info("Starting feature engineering...")
            X, X_val = self._engineer_features(X, X_val, y)

        # Model selection and training
        logger.info("Starting model selection and training...")
        self._train_models(X, y, X_val, y_val, start_time)

        # Ensemble creation
        if self.ensemble and len(self.models) > 1:
            logger.info("Creating ensemble...")
            self._create_ensemble(X, y, X_val, y_val)

        # Select best model
        self._select_best_model(X_val, y_val)

        total_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"AutoML completed in {total_time:.1f} seconds")

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if self.best_model is None:
            raise ValueError("Pipeline not fitted yet.")

        # Apply preprocessing
        if self.preprocessor:
            X = self.preprocessor.transform(X)

        # Apply feature engineering
        if self.feature_engineer:
            X = self.feature_engineer.transform(X)

        return self.best_model.predict(X)

    def _detect_task_type(self, y: pd.Series) -> str:
        """Auto-detect task type from target variable."""
        n_unique = y.nunique()

        if y.dtype in ["object", "category"] or n_unique < 10:
            return "classification"
        else:
            return "regression"

    def _get_default_metric(self) -> str:
        """Get default optimization metric for task type."""
        if self.task_type == "regression":
            return "neg_mean_squared_error"
        else:
            return "accuracy"

    def _preprocess(
        self, X_train: pd.DataFrame, X_val: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Preprocess data."""
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import LabelEncoder, StandardScaler

        # Identify column types
        numeric_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = X_train.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()

        # Create preprocessing pipeline
        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                ("onehot", pd.get_dummies),
            ]
        )

        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_features),
                ("cat", categorical_transformer, categorical_features),
            ]
        )

        # Fit and transform
        X_train = self.preprocessor.fit_transform(X_train)
        X_val = self.preprocessor.transform(X_val)

        return X_train, X_val

    def _engineer_features(
        self, X_train: pd.DataFrame, X_val: pd.DataFrame, y_train: pd.Series
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Engineer features."""
        from core.feature_engineering import FeatureEngineer

        self.feature_engineer = FeatureEngineer()

        # Create polynomial features for numeric columns
        X_train = self.feature_engineer.create_polynomial_features(
            X_train, degree=2, include_bias=False
        )
        X_val = self.feature_engineer.create_polynomial_features(
            X_val, degree=2, include_bias=False
        )

        # Feature selection
        from sklearn.feature_selection import (SelectKBest, f_classif,
                                               f_regression)

        if self.task_type == "regression":
            selector = SelectKBest(f_regression, k="all")
        else:
            selector = SelectKBest(f_classif, k="all")

        selector.fit(X_train, y_train)

        # Keep top features
        n_features = min(X_train.shape[1], 50)
        indices = np.argsort(selector.scores_)[-n_features:]

        X_train = X_train[:, indices]
        X_val = X_val[:, indices]

        return X_train, X_val

    def _train_models(self, X_train, y_train, X_val, y_val, start_time):
        """Train multiple models."""
        # Define model candidates
        if self.task_type == "regression":
            from sklearn.ensemble import (GradientBoostingRegressor,
                                          RandomForestRegressor)
            from sklearn.linear_model import Lasso, LinearRegression, Ridge
            from sklearn.svm import SVR

            model_candidates = {
                "linear": LinearRegression(),
                "ridge": Ridge(),
                "random_forest": RandomForestRegressor(
                    n_estimators=100, random_state=self.random_state
                ),
                "gradient_boosting": GradientBoostingRegressor(
                    n_estimators=100, random_state=self.random_state
                ),
                "svr": SVR(),
            }
        else:
            from sklearn.ensemble import (GradientBoostingClassifier,
                                          RandomForestClassifier)
            from sklearn.linear_model import LogisticRegression
            from sklearn.svm import SVC

            model_candidates = {
                "logistic": LogisticRegression(random_state=self.random_state),
                "random_forest": RandomForestClassifier(
                    n_estimators=100, random_state=self.random_state
                ),
                "gradient_boosting": GradientBoostingClassifier(
                    n_estimators=100, random_state=self.random_state
                ),
                "svc": SVC(random_state=self.random_state),
            }

        # Train models with time budget
        for name, model in model_candidates.items():
            if (datetime.now() - start_time).total_seconds() > self.time_budget * 0.8:
                logger.warning(f"Time budget exceeded. Skipping {name}")
                break

            logger.info(f"Training {name}...")

            try:
                # Quick hyperparameter tuning
                tuner = HyperparameterTuner(
                    optimization_method="random",
                    n_trials=10,
                    cv_folds=3,
                    scoring_metric=self.optimization_metric,
                    n_jobs=self.n_jobs,
                    random_state=self.random_state,
                )

                # Define simple param space
                param_space = self._get_simple_param_space(name)

                if param_space:
                    model, params = tuner.optimize(model, param_space, X_train, y_train)
                else:
                    model.fit(X_train, y_train)

                self.models[name] = model

                # Evaluate on validation set
                score = cross_val_score(
                    model, X_val, y_val, cv=3, scoring=self.optimization_metric
                ).mean()

                self.results[name] = {
                    "model": model,
                    "score": score,
                    "params": params if param_space else {},
                }

            except Exception as e:
                logger.warning(f"Failed to train {name}: {e}")
                continue

    def _get_simple_param_space(self, model_name: str) -> Dict[str, Any]:
        """Get simplified parameter space for quick tuning."""
        spaces = {
            "random_forest": {
                "n_estimators": [50, 100, 200],
                "max_depth": [None, 10, 20],
                "min_samples_split": [2, 5],
            },
            "gradient_boosting": {
                "n_estimators": [50, 100, 200],
                "learning_rate": [0.05, 0.1, 0.2],
                "max_depth": [3, 5, 7],
            },
            "ridge": {"alpha": [0.1, 1.0, 10.0]},
            "svr": {"C": [0.1, 1.0, 10.0], "gamma": ["scale", "auto"]},
            "svc": {"C": [0.1, 1.0, 10.0], "gamma": ["scale", "auto"]},
        }

        return spaces.get(model_name, {})

    def _create_ensemble(self, X_train, y_train, X_val, y_val):
        """Create ensemble model."""
        from models.ensemble import ModelStacker

        base_models = [(name, model) for name, model in self.models.items()]

        ensemble = ModelStacker(
            base_models=base_models, task_type=self.task_type, cv_folds=3
        )

        ensemble.fit(X_train, y_train)

        # Evaluate ensemble
        score = cross_val_score(
            ensemble, X_val, y_val, cv=3, scoring=self.optimization_metric
        ).mean()

        self.models["ensemble"] = ensemble
        self.results["ensemble"] = {"model": ensemble, "score": score, "params": {}}

    def _select_best_model(self, X_val, y_val):
        """Select best model based on validation performance."""
        best_name = max(self.results.items(), key=lambda x: x[1]["score"])[0]
        self.best_model = self.results[best_name]["model"]

        logger.info(
            f"Best model: {best_name} with score: {self.results[best_name]['score']:.4f}"
        )

    def get_results_summary(self) -> pd.DataFrame:
        """Get summary of results."""
        summary = []
        for name, result in self.results.items():
            summary.append(
                {
                    "model": name,
                    "score": result["score"],
                    "n_params": len(result["params"]),
                }
            )

        return pd.DataFrame(summary).sort_values("score", ascending=False)


class TrainingPipeline:
    """
    Complete training pipeline for machine learning projects.

    Handles data loading, preprocessing, feature engineering,
    model training, evaluation, and artifact saving.
    """

    def __init__(
        self,
        task_type: str = "regression",
        experiment_name: str = "experiment",
        output_dir: str = "./outputs",
    ):
        """
        Initialize the TrainingPipeline.

        Args:
            task_type: Type of ML task ('regression' or 'classification')
            experiment_name: Name for this experiment
            output_dir: Directory to save outputs
        """
        self.task_type = task_type
        self.experiment_name = experiment_name
        self.output_dir = Path(output_dir) / experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.data_loader = TabularDataLoader()
        self.preprocessor = DataPreprocessor()
        self.feature_selector = FeatureSelector(task_type)
        self.feature_engineer = FeatureEngineer()
        self.evaluator = ModelEvaluator(task_type)
        self.visualizer = ModelVisualizer()

        # Storage for pipeline artifacts
        self.data = {}
        self.models = {}
        self.metrics = {}
        self.config = {
            "task_type": task_type,
            "experiment_name": experiment_name,
            "timestamp": datetime.now().isoformat(),
        }

    def load_data(
        self,
        data_path: Union[str, Path],
        target_column: str,
        test_size: float = 0.2,
        val_size: Optional[float] = None,
        stratify: bool = False,
        **kwargs,
    ) -> "TrainingPipeline":
        """
        Load and split data.

        Args:
            data_path: Path to data file
            target_column: Name of target column
            test_size: Proportion for test set
            val_size: Proportion for validation set
            stratify: Whether to use stratified splitting
            **kwargs: Additional arguments for data loading

        Returns:
            Self for method chaining
        """
        logger.info(f"Loading data from {data_path}")

        # Load data
        data = self.data_loader.load(data_path, **kwargs)

        # Separate features and target
        X = data.drop(columns=[target_column])
        y = data[target_column]

        # Store column information
        self.config["features"] = X.columns.tolist()
        self.config["target"] = target_column
        self.config["n_samples"] = len(data)
        self.config["n_features"] = X.shape[1]

        # Split data
        if stratify and self.task_type == "classification":
            stratify_col = y
        else:
            stratify_col = None

        if val_size:
            # Create train/val/test split
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y, test_size=test_size, stratify=stratify_col, random_state=42
            )

            val_prop = val_size / (1 - test_size)
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp,
                y_temp,
                test_size=val_prop,
                stratify=y_temp if stratify else None,
                random_state=42,
            )

            self.data = {
                "X_train": X_train,
                "y_train": y_train,
                "X_val": X_val,
                "y_val": y_val,
                "X_test": X_test,
                "y_test": y_test,
            }
        else:
            # Create train/test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, stratify=stratify_col, random_state=42
            )

            self.data = {
                "X_train": X_train,
                "y_train": y_train,
                "X_test": X_test,
                "y_test": y_test,
            }

        logger.info(f"Data split: Train={len(X_train)}, Test={len(X_test)}")
        if val_size:
            logger.info(f"Validation={len(X_val)}")

        return self

    def preprocess(
        self,
        numeric_features: Optional[List[str]] = None,
        categorical_features: Optional[List[str]] = None,
        scaling_method: str = "standard",
        encoding_method: str = "onehot",
        handle_missing: str = "impute",
        remove_outliers: bool = False,
    ) -> "TrainingPipeline":
        """
        Preprocess the data.

        Args:
            numeric_features: List of numeric features (auto-detected if None)
            categorical_features: List of categorical features (auto-detected if None)
            scaling_method: Method for scaling numeric features
            encoding_method: Method for encoding categorical features
            handle_missing: Method for handling missing values
            remove_outliers: Whether to remove outliers

        Returns:
            Self for method chaining
        """
        logger.info("Preprocessing data")

        X_train = self.data["X_train"]

        # Auto-detect feature types if not provided
        if numeric_features is None:
            numeric_features = X_train.select_dtypes(
                include=[np.number]
            ).columns.tolist()
        if categorical_features is None:
            categorical_features = X_train.select_dtypes(
                include=["object", "category"]
            ).columns.tolist()

        self.config["numeric_features"] = numeric_features
        self.config["categorical_features"] = categorical_features

        # Apply preprocessing to each dataset
        for key in ["X_train", "X_val", "X_test"]:
            if key not in self.data:
                continue

            X = self.data[key]

            # Handle missing values
            X = self.preprocessor.handle_missing_values(X, method=handle_missing)

            # Remove outliers (training set only)
            if remove_outliers and key == "X_train":
                X = self.preprocessor.remove_outliers(X, columns=numeric_features)

            # Scale numeric features
            if numeric_features:
                X = self.preprocessor.scale_features(
                    X, columns=numeric_features, method=scaling_method
                )

            # Encode categorical features
            if categorical_features:
                X = self.preprocessor.encode_categorical(
                    X, columns=categorical_features, method=encoding_method
                )

            self.data[key] = X

        # Update feature names after encoding
        self.config["features_after_preprocessing"] = self.data[
            "X_train"
        ].columns.tolist()

        return self

    def engineer_features(
        self,
        create_polynomials: bool = False,
        create_interactions: bool = False,
        select_features: bool = True,
        selection_method: str = "mutual_information",
        n_features: Optional[int] = None,
    ) -> "TrainingPipeline":
        """
        Perform feature engineering.

        Args:
            create_polynomials: Whether to create polynomial features
            create_interactions: Whether to create interaction features
            select_features: Whether to perform feature selection
            selection_method: Method for feature selection
            n_features: Number of features to select

        Returns:
            Self for method chaining
        """
        logger.info("Engineering features")

        X_train = self.data["X_train"]
        y_train = self.data["y_train"]

        # Create polynomial features
        if create_polynomials:
            for key in ["X_train", "X_val", "X_test"]:
                if key in self.data:
                    self.data[key] = self.feature_engineer.create_polynomial_features(
                        self.data[key], degree=2
                    )

        # Create interaction features
        if create_interactions:
            for key in ["X_train", "X_val", "X_test"]:
                if key in self.data:
                    self.data[key] = self.feature_engineer.create_interaction_features(
                        self.data[key]
                    )

        # Feature selection
        if select_features:
            X_train = self.data["X_train"]

            if selection_method == "mutual_information":
                selected = self.feature_selector.select_by_mutual_information(
                    X_train, y_train, n_features
                )
            elif selection_method == "model_importance":
                selected = self.feature_selector.select_by_model_importance(
                    X_train, y_train, n_features=n_features
                )
            elif selection_method == "rfe":
                selected = self.feature_selector.select_by_rfe(
                    X_train, y_train, n_features=n_features
                )
            else:
                raise ValueError(f"Unknown selection method: {selection_method}")

            # Apply selection to all datasets
            for key in ["X_train", "X_val", "X_test"]:
                if key in self.data:
                    self.data[key] = self.data[key][selected]

            self.config["selected_features"] = selected
            self.config["n_selected_features"] = len(selected)

        return self

    def train_models(
        self,
        model_types: List[str] = ["auto"],
        tune_hyperparameters: bool = True,
        ensemble_method: Optional[str] = None,
    ) -> "TrainingPipeline":
        """
        Train machine learning models.

        Args:
            model_types: List of model types to train
            tune_hyperparameters: Whether to tune hyperparameters
            ensemble_method: Ensemble method ('stacking', 'blending', None)

        Returns:
            Self for method chaining
        """
        logger.info("Training models")

        X_train = self.data["X_train"]
        y_train = self.data["y_train"]

        # Train individual models
        for model_type in model_types:
            logger.info(f"Training {model_type} model")

            model = EnsembleModel(task_type=self.task_type, model_type=model_type)

            model.fit(X_train, y_train, tune_hyperparameters=tune_hyperparameters, cv=5)

            self.models[model_type] = model

        # Create ensemble if requested
        if ensemble_method and len(self.models) > 1:
            logger.info(f"Creating {ensemble_method} ensemble")

            base_models = [(name, model.model) for name, model in self.models.items()]

            if ensemble_method == "stacking":
                ensemble = ModelStacker(
                    base_models=base_models, task_type=self.task_type
                )
                ensemble.fit(X_train, y_train)
                self.models["ensemble_stacking"] = ensemble

            elif ensemble_method == "blending":
                ensemble = ModelBlender(models=base_models)
                ensemble.fit(X_train, y_train)

                # Optimize weights if validation set available
                if "X_val" in self.data:
                    ensemble.optimize_weights(self.data["X_val"], self.data["y_val"])

                self.models["ensemble_blending"] = ensemble

        return self

    def evaluate(
        self, create_plots: bool = True, save_plots: bool = True
    ) -> "TrainingPipeline":
        """
        Evaluate trained models.

        Args:
            create_plots: Whether to create evaluation plots
            save_plots: Whether to save plots to disk

        Returns:
            Self for method chaining
        """
        logger.info("Evaluating models")

        X_test = self.data["X_test"]
        y_test = self.data["y_test"]

        # Evaluate each model
        for model_name, model in self.models.items():
            logger.info(f"Evaluating {model_name}")

            # Make predictions
            if hasattr(model, "predict"):
                y_pred = model.predict(X_test)
            else:
                y_pred = model.model.predict(X_test)

            # Calculate metrics
            metrics = self.evaluator.evaluate(y_test, y_pred, model_name)
            self.metrics[model_name] = metrics

            # Print report
            self.evaluator.print_report(metrics)

            # Create plots
            if create_plots:
                plot_dir = self.output_dir / "plots" / model_name
                plot_dir.mkdir(parents=True, exist_ok=True)

                if self.task_type == "regression":
                    self.evaluator.plot_predictions(
                        y_test,
                        y_pred,
                        title=f"{model_name} - Predictions vs Actual",
                        save_path=(
                            str(plot_dir / "predictions.png") if save_plots else None
                        ),
                    )

                    self.evaluator.plot_residuals(
                        y_test,
                        y_pred,
                        save_path=(
                            str(plot_dir / "residuals.png") if save_plots else None
                        ),
                    )

                # Feature importance for tree-based models
                if hasattr(model, "feature_importances_") or (
                    hasattr(model, "model")
                    and hasattr(model.model, "feature_importances_")
                ):
                    feature_names = X_test.columns.tolist()
                    self.visualizer.plot_feature_importance(
                        model.model if hasattr(model, "model") else model,
                        feature_names,
                        save_path=(
                            str(plot_dir / "feature_importance.png")
                            if save_plots
                            else None
                        ),
                    )

        # Compare models
        if len(self.models) > 1:
            self._compare_models()

        return self

    def _compare_models(self):
        """Compare performance of different models."""
        comparison_df = pd.DataFrame(self.metrics).T

        if self.task_type == "regression":
            comparison_df = comparison_df[["rmse", "mae", "r2"]].round(4)
        else:
            comparison_df = comparison_df[
                ["accuracy", "precision", "recall", "f1"]
            ].round(4)

        logger.info("\nModel Comparison:")
        print(comparison_df)

        # Save comparison
        comparison_df.to_csv(self.output_dir / "model_comparison.csv")

    def save_artifacts(self) -> "TrainingPipeline":
        """
        Save all pipeline artifacts.

        Returns:
            Self for method chaining
        """
        logger.info(f"Saving artifacts to {self.output_dir}")

        # Save models
        models_dir = self.output_dir / "models"
        models_dir.mkdir(exist_ok=True)

        for model_name, model in self.models.items():
            model_path = models_dir / f"{model_name}.pkl"
            with open(model_path, "wb") as f:
                pickle.dump(model, f)
            logger.info(f"Saved model: {model_name}")

        # Save preprocessor
        with open(self.output_dir / "preprocessor.pkl", "wb") as f:
            pickle.dump(self.preprocessor, f)

        # Save configuration
        with open(self.output_dir / "config.json", "w") as f:
            json.dump(self.config, f, indent=2)

        # Save metrics
        metrics_df = pd.DataFrame(self.metrics).T
        metrics_df.to_csv(self.output_dir / "metrics.csv")

        # Save feature importance if available
        if (
            hasattr(self.feature_selector, "feature_scores")
            and self.feature_selector.feature_scores
        ):
            importance_df = pd.DataFrame(
                list(self.feature_selector.feature_scores.items()),
                columns=["feature", "score"],
            ).sort_values("score", ascending=False)
            importance_df.to_csv(
                self.output_dir / "feature_importance.csv", index=False
            )

        logger.info("All artifacts saved successfully")

        return self

    def run(
        self, data_path: Union[str, Path], target_column: str, **kwargs
    ) -> Dict[str, Any]:
        """
        Run the complete training pipeline.

        Args:
            data_path: Path to data file
            target_column: Name of target column
            **kwargs: Additional configuration options

        Returns:
            Dictionary with results
        """
        logger.info(f"Starting training pipeline: {self.experiment_name}")

        try:
            # Execute pipeline steps
            self.load_data(data_path, target_column, **kwargs.get("data_config", {}))
            self.preprocess(**kwargs.get("preprocessing_config", {}))
            self.engineer_features(**kwargs.get("feature_engineering_config", {}))
            self.train_models(**kwargs.get("training_config", {}))
            self.evaluate(**kwargs.get("evaluation_config", {}))
            self.save_artifacts()

            # Prepare results
            results = {
                "experiment_name": self.experiment_name,
                "best_model": (
                    min(
                        self.metrics.items(),
                        key=lambda x: x[1].get("rmse", float("inf")),
                    )[0]
                    if self.task_type == "regression"
                    else max(
                        self.metrics.items(), key=lambda x: x[1].get("accuracy", 0)
                    )[0]
                ),
                "metrics": self.metrics,
                "config": self.config,
                "output_dir": str(self.output_dir),
            }

            logger.info(
                f"Pipeline completed successfully. Best model: {results['best_model']}"
            )

            return results

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise
