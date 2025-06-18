"""
models/ensemble.py
Ensemble learning methods with enhanced functionality.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Union, Any, Tuple, Callable
from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier,
    ExtraTreesRegressor, ExtraTreesClassifier,
    GradientBoostingRegressor, GradientBoostingClassifier,
    HistGradientBoostingRegressor, HistGradientBoostingClassifier,
    VotingRegressor, VotingClassifier,
    StackingRegressor, StackingClassifier,
    BaggingRegressor, BaggingClassifier
)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin, clone
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import logging
from copy import deepcopy
from abc import ABC, abstractmethod
import warnings

logger = logging.getLogger(__name__)


class BaseEnsemble(ABC, BaseEstimator):
    """Abstract base class for ensemble methods."""
    
    def __init__(self, base_estimators: List[BaseEstimator], task_type: str = 'auto'):
        """
        Initialize base ensemble.
        
        Args:
            base_estimators: List of base estimators
            task_type: Type of task ('regression', 'classification', 'auto')
        """
        self.base_estimators = base_estimators
        self.task_type = task_type
        self.fitted_estimators_ = None
        self.is_fitted_ = False
        
    @abstractmethod
    def fit(self, X, y, sample_weight=None):
        """Fit the ensemble."""
        pass
    
    @abstractmethod
    def predict(self, X):
        """Make predictions."""
        pass
    
    def _validate_estimators(self):
        """Validate that estimators are compatible."""
        if not self.base_estimators:
            raise ValueError("No base estimators provided")
        
        # Check if all estimators have required methods
        for est in self.base_estimators:
            if not hasattr(est, 'fit') or not hasattr(est, 'predict'):
                raise ValueError(f"Estimator {est} must have fit and predict methods")
    
    def _infer_task_type(self, y):
        """Infer task type from target variable."""
        if self.task_type == 'auto':
            # Check if target is continuous or discrete
            unique_ratio = len(np.unique(y)) / len(y)
            if unique_ratio < 0.05:  # Less than 5% unique values
                self.task_type = 'classification'
            else:
                self.task_type = 'regression'
            logger.info(f"Inferred task type: {self.task_type}")


class VotingEnsemble(BaseEnsemble):
    """
    Voting ensemble for combining predictions.
    
    This class implements both hard and soft voting for classification,
    and averaging for regression tasks.
    """
    
    def __init__(self, 
                 estimators: List[Tuple[str, BaseEstimator]],
                 voting: str = 'auto',
                 weights: Optional[List[float]] = None,
                 task_type: str = 'auto',
                 flatten_transform: bool = True):
        """
        Initialize VotingEnsemble.
        
        Args:
            estimators: List of (name, estimator) tuples
            voting: Voting type ('hard', 'soft', 'auto')
            weights: Weights for each estimator
            task_type: Type of task ('regression', 'classification', 'auto')
            flatten_transform: Whether to flatten transform output
        """
        base_estimators = [est for _, est in estimators]
        super().__init__(base_estimators, task_type)
        
        self.estimators = estimators
        self.voting = voting
        self.weights = weights
        self.flatten_transform = flatten_transform
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
        X, y = check_X_y(X, accept_sparse=['csc'])
        self._validate_estimators()
        
        # Infer task type if needed
        self._infer_task_type(y)
        
        # Determine voting type
        if self.voting == 'auto':
            if self.task_type == 'regression':
                self.voting = 'soft'  # Averaging for regression
            else:
                # Check if all classifiers support predict_proba
                all_support_proba = all(hasattr(est, 'predict_proba') 
                                      for _, est in self.estimators)
                self.voting = 'soft' if all_support_proba else 'hard'
        
        # Create sklearn voting model
        if self.task_type == 'regression':
            self.voting_model_ = VotingRegressor(
                estimators=self.estimators,
                weights=self.weights
            )
        else:
            self.voting_model_ = VotingClassifier(
                estimators=self.estimators,
                voting=self.voting,
                weights=self.weights,
                flatten_transform=self.flatten_transform
            )
        
        # Fit the voting model
        self.voting_model_.fit(X, y, sample_weight=sample_weight)
        self.is_fitted_ = True
        
        return self
    
    def predict(self, X):
        """
        Make predictions using voting.
        
        Args:
            X: Features to predict
            
        Returns:
            Predictions
        """
        check_is_fitted(self, 'voting_model_')
        X = check_array(X, accept_sparse=['csc'])
        return self.voting_model_.predict(X)
    
    def predict_proba(self, X):
        """
        Predict class probabilities (classification only).
        
        Args:
            X: Features to predict
            
        Returns:
            Class probabilities
        """
        if self.task_type != 'classification':
            raise AttributeError("predict_proba is only available for classification")
        
        check_is_fitted(self, 'voting_model_')
        
        if self.voting == 'hard':
            raise AttributeError("predict_proba is not available for hard voting")
        
        X = check_array(X, accept_sparse=['csc'])
        return self.voting_model_.predict_proba(X)
    
    def transform(self, X):
        """
        Return predictions from each estimator.
        
        Args:
            X: Features to transform
            
        Returns:
            Predictions from each estimator
        """
        check_is_fitted(self, 'voting_model_')
        return self.voting_model_.transform(X)
    
    @property
    def fitted_estimators_(self):
        """Get fitted estimators."""
        if hasattr(self.voting_model_, 'estimators_'):
            return self.voting_model_.estimators_
        return None


class StackingEnsemble(BaseEnsemble):
    """
    Stacking ensemble with cross-validation.
    
    This class implements stacking with out-of-fold predictions
    to avoid overfitting in the meta-model.
    """
    
    def __init__(self,
                 estimators: List[Tuple[str, BaseEstimator]],
                 final_estimator: Optional[BaseEstimator] = None,
                 cv: Union[int, object] = 5,
                 stack_method: Union[str, List[str]] = 'auto',
                 passthrough: bool = False,
                 task_type: str = 'auto',
                 verbose: int = 0):
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
        X, y = check_X_y(X, accept_sparse=['csc'])
        self._validate_estimators()
        
        # Infer task type
        self._infer_task_type(y)
        
        # Set default final estimator if not provided
        if self.final_estimator is None:
            if self.task_type == 'regression':
                from sklearn.linear_model import RidgeCV
                self.final_estimator = RidgeCV()
            else:
                from sklearn.linear_model import LogisticRegression
                self.final_estimator = LogisticRegression(random_state=42)
        
        # Determine stack method
        if self.stack_method == 'auto':
            if self.task_type == 'regression':
                self.stack_method = 'predict'
            else:
                # Check if all support predict_proba
                methods = []
                for name, est in self.estimators:
                    if hasattr(est, 'predict_proba'):
                        methods.append('predict_proba')
                    else:
                        methods.append('predict')
                self.stack_method = methods
        
        # Create sklearn stacking model
        if self.task_type == 'regression':
            self.stacking_model_ = StackingRegressor(
                estimators=self.estimators,
                final_estimator=self.final_estimator,
                cv=self.cv,
                passthrough=self.passthrough,
                verbose=self.verbose
            )
        else:
            self.stacking_model_ = StackingClassifier(
                estimators=self.estimators,
                final_estimator=self.final_estimator,
                cv=self.cv,
                stack_method=self.stack_method,
                passthrough=self.passthrough,
                verbose=self.verbose
            )
        
        # Fit the model
        self.stacking_model_.fit(X, y, sample_weight=sample_weight)
        self.is_fitted_ = True
        
        return self
    
    def predict(self, X):
        """Make predictions."""
        check_is_fitted(self, 'stacking_model_')
        X = check_array(X, accept_sparse=['csc'])
        return self.stacking_model_.predict(X)
    
    def predict_proba(self, X):
        """Predict class probabilities."""
        if self.task_type != 'classification':
            raise AttributeError("predict_proba is only available for classification")
        
        check_is_fitted(self, 'stacking_model_')
        X = check_array(X, accept_sparse=['csc'])
        return self.stacking_model_.predict_proba(X)
    
    def transform(self, X):
        """Transform X using base estimators."""
        check_is_fitted(self, 'stacking_model_')
        return self.stacking_model_.transform(X)
    
    @property
    def fitted_estimators_(self):
        """Get fitted base estimators."""
        if hasattr(self.stacking_model_, 'estimators_'):
            return self.stacking_model_.estimators_
        return None
    
    @property
    def final_estimator_(self):
        """Get fitted final estimator."""
        if hasattr(self.stacking_model_, 'final_estimator_'):
            return self.stacking_model_.final_estimator_
        return None


class BaggingEnsemble(BaseEnsemble):
    """
    Bagging ensemble with enhanced functionality.
    
    This class extends sklearn's bagging with additional features
    like out-of-bag prediction and feature importance aggregation.
    """
    
    def __init__(self,
                 base_estimator: BaseEstimator = None,
                 n_estimators: int = 10,
                 max_samples: Union[int, float] = 1.0,
                 max_features: Union[int, float] = 1.0,
                 bootstrap: bool = True,
                 bootstrap_features: bool = False,
                 oob_score: bool = False,
                 warm_start: bool = False,
                 n_jobs: Optional[int] = None,
                 random_state: Optional[int] = None,
                 task_type: str = 'auto',
                 verbose: int = 0):
        """
        Initialize BaggingEnsemble.
        
        Args:
            base_estimator: Base estimator to use
            n_estimators: Number of base estimators
            max_samples: Number of samples to draw
            max_features: Number of features to draw
            bootstrap: Whether to bootstrap samples
            bootstrap_features: Whether to bootstrap features
            oob_score: Whether to use out-of-bag samples for evaluation
            warm_start: Whether to reuse previous solution
            n_jobs: Number of parallel jobs
            random_state: Random state
            task_type: Type of task
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
        self.n_jobs = n_jobs
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
        X, y = check_X_y(X, accept_sparse=['csc'])
        
        # Infer task type
        self._infer_task_type(y)
        
        # Set default base estimator if not provided
        if self.base_estimator is None:
            if self.task_type == 'regression':
                from sklearn.tree import DecisionTreeRegressor
                self.base_estimator = DecisionTreeRegressor(random_state=self.random_state)
            else:
                from sklearn.tree import DecisionTreeClassifier
                self.base_estimator = DecisionTreeClassifier(random_state=self.random_state)
        
        # Create sklearn bagging model
        if self.task_type == 'regression':
            self.bagging_model_ = BaggingRegressor(
                base_estimator=self.base_estimator,
                n_estimators=self.n_estimators,
                max_samples=self.max_samples,
                max_features=self.max_features,
                bootstrap=self.bootstrap,
                bootstrap_features=self.bootstrap_features,
                oob_score=self.oob_score,
                warm_start=self.warm_start,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
                verbose=self.verbose
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
                n_jobs=self.n_jobs,
                random_state=self.random_state,
                verbose=self.verbose
            )
        
        # Fit the model
        self.bagging_model_.fit(X, y, sample_weight=sample_weight)
        self.is_fitted_ = True
        
        return self
    
    def predict(self, X):
        """Make predictions."""
        check_is_fitted(self, 'bagging_model_')
        X = check_array(X, accept_sparse=['csc'])
        return self.bagging_model_.predict(X)
    
    def predict_proba(self, X):
        """Predict class probabilities."""
        if self.task_type != 'classification':
            raise AttributeError("predict_proba is only available for classification")
        
        check_is_fitted(self, 'bagging_model_')
        X = check_array(X, accept_sparse=['csc'])
        return self.bagging_model_.predict_proba(X)
    
    @property
    def estimators_(self):
        """Get the list of fitted sub-estimators."""
        if hasattr(self.bagging_model_, 'estimators_'):
            return self.bagging_model_.estimators_
        return None
    
    @property
    def estimators_samples_(self):
        """Get the subset of drawn samples for each estimator."""
        if hasattr(self.bagging_model_, 'estimators_samples_'):
            return self.bagging_model_.estimators_samples_
        return None
    
    @property
    def estimators_features_(self):
        """Get the subset of drawn features for each estimator."""
        if hasattr(self.bagging_model_, 'estimators_features_'):
            return self.bagging_model_.estimators_features_
        return None
    
    @property
    def oob_score_(self):
        """Get out-of-bag score."""
        if hasattr(self.bagging_model_, 'oob_score_'):
            return self.bagging_model_.oob_score_
        return None
    
    @property
    def oob_prediction_(self):
        """Get out-of-bag predictions."""
        if hasattr(self.bagging_model_, 'oob_prediction_'):
            return self.bagging_model_.oob_prediction_
        return None
    
    def get_feature_importance(self):
        """
        Get aggregated feature importance from all estimators.
        
        Returns:
            Array of feature importances
        """
        check_is_fitted(self, 'bagging_model_')
        
        if not hasattr(self.estimators_[0], 'feature_importances_'):
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


class WeightedEnsemble(BaseEnsemble):
    """
    Weighted ensemble with optimized weights.
    
    This class combines predictions from multiple models using
    optimized weights that minimize the prediction error.
    """
    
    def __init__(self,
                 estimators: List[Tuple[str, BaseEstimator]],
                 weights: Optional[List[float]] = None,
                 weight_optimization: str = 'minimize_error',
                 cv: int = 5,
                 scoring: Optional[Union[str, Callable]] = None,
                 task_type: str = 'auto',
                 random_state: Optional[int] = None):
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
        X, y = check_X_y(X, accept_sparse=['csc'])
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
            self.optimized_weights_ = [w / total_weight for w in self.weights]
        
        self.is_fitted_ = True
        return self
    
    def _supports_sample_weight(self, estimator):
        """Check if estimator supports sample weights."""
        import inspect
        fit_params = inspect.signature(estimator.fit).parameters
        return 'sample_weight' in fit_params
    
    def _optimize_weights(self, X, y):
        """Optimize ensemble weights using cross-validation."""
        from sklearn.model_selection import KFold
        from scipy.optimize import minimize
        
        logger.info("Optimizing ensemble weights")
        
        # Get CV predictions
        kf = KFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
        cv_predictions = {name: [] for name, _ in self.estimators}
        y_true_cv = []
        
        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            y_true_cv.extend(y_val)
            
            # Get predictions from each estimator
            for name, estimator in self.estimators:
                est_clone = clone(estimator)
                est_clone.fit(X_train, y_train)
                
                if self.task_type == 'regression':
                    pred = est_clone.predict(X_val)
                else:
                    # Use probabilities for classification
                    if hasattr(est_clone, 'predict_proba'):
                        pred = est_clone.predict_proba(X_val)
                    else:
                        pred = est_clone.predict(X_val)
                
                cv_predictions[name].extend(pred)
        
        # Convert to arrays
        y_true_cv = np.array(y_true_cv)
        prediction_matrix = np.column_stack([np.array(cv_predictions[name]) 
                                           for name, _ in self.estimators])
        
        # Define objective function
        def objective(weights):
            # Ensure weights sum to 1
            weights = weights / weights.sum()
            
            if self.task_type == 'regression':
                # Weighted average of predictions
                weighted_pred = prediction_matrix @ weights
                # Mean squared error
                return np.mean((y_true_cv - weighted_pred) ** 2)
            else:
                # For classification, handle probability matrices
                if len(prediction_matrix.shape) == 3:
                    # Weighted average of probabilities
                    weighted_pred = np.sum(prediction_matrix * weights[:, None, None], axis=0)
                    # Cross-entropy loss
                    from sklearn.preprocessing import LabelBinarizer
                    lb = LabelBinarizer()
                    y_true_binary = lb.fit_transform(y_true_cv)
                    epsilon = 1e-10
                    weighted_pred = np.clip(weighted_pred, epsilon, 1 - epsilon)
                    return -np.mean(y_true_binary * np.log(weighted_pred))
                else:
                    # Hard predictions - use accuracy
                    weighted_pred = np.round(prediction_matrix @ weights)
                    return np.mean(weighted_pred != y_true_cv)
        
        # Initial weights (equal)
        initial_weights = np.ones(len(self.estimators)) / len(self.estimators)
        
        # Constraints: weights sum to 1, all weights >= 0
        constraints = {'type': 'eq', 'fun': lambda w: w.sum() - 1}
        bounds = [(0, 1) for _ in self.estimators]
        
        # Optimize
        result = minimize(objective, initial_weights, method='SLSQP',
                         bounds=bounds, constraints=constraints)
        
        self.optimized_weights_ = result.x
        
        # Log optimized weights
        weight_dict = {name: weight for (name, _), weight 
                      in zip(self.estimators, self.optimized_weights_)}
        logger.info(f"Optimized weights: {weight_dict}")
    
    def predict(self, X):
        """
        Make weighted predictions.
        
        Args:
            X: Features to predict
            
        Returns:
            Weighted predictions
        """
        check_is_fitted(self, ['fitted_estimators_', 'optimized_weights_'])
        X = check_array(X, accept_sparse=['csc'])
        
        # Get predictions from all estimators
        predictions = []
        for name, estimator in self.fitted_estimators_:
            pred = estimator.predict(X)
            predictions.append(pred)
        
        # Apply weights
        predictions = np.array(predictions)
        weighted_pred = np.sum(predictions * self.optimized_weights_[:, np.newaxis], axis=0)
        
        # For classification, round to nearest integer
        if self.task_type == 'classification':
            weighted_pred = np.round(weighted_pred).astype(int)
        
        return weighted_pred
    
    def predict_proba(self, X):
        """
        Predict class probabilities using weighted average.
        
        Args:
            X: Features to predict
            
        Returns:
            Weighted probability predictions
        """
        if self.task_type != 'classification':
            raise AttributeError("predict_proba is only available for classification")
        
        check_is_fitted(self, ['fitted_estimators_', 'optimized_weights_'])
        X = check_array(X, accept_sparse=['csc'])
        
        # Get probability predictions from all estimators
        proba_predictions = []
        
        for name, estimator in self.fitted_estimators_:
            if hasattr(estimator, 'predict_proba'):
                proba = estimator.predict_proba(X)
                proba_predictions.append(proba)
            else:
                # Convert hard predictions to probabilities
                pred = estimator.predict(X)
                n_classes = len(np.unique(pred))
                proba = np.zeros((len(X), n_classes))
                proba[np.arange(len(X)), pred] = 1.0
                proba_predictions.append(proba)
        
        # Apply weights
        proba_predictions = np.array(proba_predictions)
        weighted_proba = np.sum(proba_predictions * self.optimized_weights_[:, np.newaxis, np.newaxis], 
                               axis=0)
        
        # Normalize to ensure probabilities sum to 1
        weighted_proba = weighted_proba / weighted_proba.sum(axis=1, keepdims=True)
        
        return weighted_proba
    
    def get_estimator_weights(self):
        """
        Get the weights for each estimator.
        
        Returns:
            Dictionary mapping estimator names to weights
        """
        check_is_fitted(self, 'optimized_weights_')
        
        return {name: weight for (name, _), weight 
                in zip(self.estimators, self.optimized_weights_)}


class EnsembleModel:
    """
    Enhanced ensemble model wrapper with automatic hyperparameter tuning
    and model selection capabilities.
    """
    
    def __init__(self, 
                 task_type: str = 'regression',
                 model_type: str = 'auto',
                 random_state: int = 42):
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
    
    def _get_model_class(self, model_type: str):
        """Get the appropriate model class based on task and model type."""
        model_mapping = {
            'regression': {
                'rf': RandomForestRegressor,
                'et': ExtraTreesRegressor,
                'gb': GradientBoostingRegressor,
                'hgb': HistGradientBoostingRegressor,
                'xgb': xgb.XGBRegressor,
                'lgb': lgb.LGBMRegressor,
                'cat': cb.CatBoostRegressor
            },
            'classification': {
                'rf': RandomForestClassifier,
                'et': ExtraTreesClassifier,
                'gb': GradientBoostingClassifier,
                'hgb': HistGradientBoostingClassifier,
                'xgb': xgb.XGBClassifier,
                'lgb': lgb.LGBMClassifier,
                'cat': cb.CatBoostClassifier
            }
        }
        
        return model_mapping[self.task_type][model_type]
    
    def _get_param_grid(self, model_type: str) -> Dict:
        """Get hyperparameter grid for the specified model type."""
        param_grids = {
            'rf': {
                'n_estimators': [100, 200, 300],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'et': {
                'n_estimators': [100, 200, 300],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'gb': {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'min_samples_split': [2, 5, 10]
            },
            'hgb': {
                'max_iter': [100, 200, 300],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [None, 10, 20],
                'min_samples_leaf': [20, 50, 100]
            },
            'xgb': {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0]
            },
            'lgb': {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.1, 0.2],
                'num_leaves': [31, 50, 100],
                'min_child_samples': [20, 50, 100]
            },
            'cat': {
                'iterations': [100, 200, 300],
                'learning_rate': [0.01, 0.1, 0.2],
                'depth': [4, 6, 8],
                'l2_leaf_reg': [1, 3, 5]
            }
        }
        
        return param_grids.get(model_type, {})
    
    def fit(self, 
            X: Union[pd.DataFrame, np.ndarray], 
            y: Union[pd.Series, np.ndarray],
            tune_hyperparameters: bool = True,
            cv: int = 5,
            scoring: Optional[str] = None,
            n_iter: int = 20):
        """
        Fit the ensemble model with optional hyperparameter tuning.
        
        Args:
            X: Feature matrix
            y: Target variable
            tune_hyperparameters: Whether to perform hyperparameter tuning
            cv: Number of cross-validation folds
            scoring: Scoring metric for hyperparameter tuning
            n_iter: Number of iterations for random search
        """
        if self.model_type == 'auto':
            # Try multiple models and select the best
            self._auto_select_model(X, y, cv, scoring)
        else:
            # Use specified model
            model_class = self._get_model_class(self.model_type)
            
            if self.model_type == 'cat':
                base_model = model_class(random_state=self.random_state, verbose=False)
            else:
                base_model = model_class(random_state=self.random_state)
            
            if tune_hyperparameters:
                param_grid = self._get_param_grid(self.model_type)
                if param_grid:
                    logger.info(f"Tuning hyperparameters for {self.model_type}")
                    
                    # Use RandomizedSearchCV for faster tuning
                    search = RandomizedSearchCV(
                        base_model, param_grid, n_iter=n_iter,
                        cv=cv, scoring=scoring, n_jobs=-1,
                        random_state=self.random_state
                    )
                    search.fit(X, y)
                    
                    self.model = search.best_estimator_
                    self.best_params = search.best_params_
                    logger.info(f"Best parameters: {self.best_params}")
                else:
                    self.model = base_model
                    self.model.fit(X, y)
            else:
                self.model = base_model
                self.model.fit(X, y)
        
        # Extract feature importances if available
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importances_ = self.model.feature_importances_
    
    def _auto_select_model(self, X, y, cv, scoring):
        """Automatically select the best model based on cross-validation."""
        model_types = ['rf', 'et', 'gb', 'xgb', 'lgb']
        best_score = -np.inf
        best_model_type = None
        
        for model_type in model_types:
            logger.info(f"Evaluating {model_type}")
            
            try:
                model_class = self._get_model_class(model_type)
                if model_type == 'cat':
                    model = model_class(random_state=self.random_state, verbose=False)
                else:
                    model = model_class(random_state=self.random_state)
                
                # Quick evaluation with cross-validation
                scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
                mean_score = np.mean(scores)
                
                self.model_scores[model_type] = mean_score
                logger.info(f"{model_type} CV score: {mean_score:.4f}")
                
                if mean_score > best_score:
                    best_score = mean_score
                    best_model_type = model_type
            
            except Exception as e:
                logger.warning(f"Failed to evaluate {model_type}: {e}")
        
        logger.info(f"Best model: {best_model_type} (score: {best_score:.4f})")
        
        # Train the best model with hyperparameter tuning
        self.model_type = best_model_type
        self.fit(X, y, tune_hyperparameters=True, cv=cv, scoring=scoring)
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Make predictions using the fitted model."""
        if self.model is None:
            raise ValueError("Model has not been fitted yet")
        return self.model.predict(X)
    
    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Make probability predictions (classification only)."""
        if self.task_type != 'classification':
            raise ValueError("predict_proba is only available for classification")
        if self.model is None:
            raise ValueError("Model has not been fitted yet")
        return self.model.predict_proba(X)


class ModelStacker:
    """
    Advanced model stacking implementation.
    
    Combines multiple models using meta-learning for improved performance.
    """
    
    def __init__(self,
                 base_models: List[Tuple[str, Any]],
                 meta_model: Optional[Any] = None,
                 task_type: str = 'regression',
                 cv: int = 5):
        """
        Initialize the ModelStacker.
        
        Args:
            base_models: List of (name, model) tuples for base models
            meta_model: Meta-learner model (default: LinearRegression/LogisticRegression)
            task_type: Type of ML task ('regression' or 'classification')
            cv: Number of cross-validation folds for stacking
        """
        self.base_models = base_models
        self.meta_model = meta_model
        self.task_type = task_type
        self.cv = cv
        self.stacking_model = None
    
    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]):
        """
        Fit the stacking model.
        
        Args:
            X: Feature matrix
            y: Target variable
        """
        if self.meta_model is None:
            if self.task_type == 'regression':
                from sklearn.linear_model import LinearRegression
                self.meta_model = LinearRegression()
            else:
                from sklearn.linear_model import LogisticRegression
                self.meta_model = LogisticRegression(random_state=42)
        
        if self.task_type == 'regression':
            self.stacking_model = StackingRegressor(
                estimators=self.base_models,
                final_estimator=self.meta_model,
                cv=self.cv
            )
        else:
            self.stacking_model = StackingClassifier(
                estimators=self.base_models,
                final_estimator=self.meta_model,
                cv=self.cv
            )
        
        logger.info(f"Training stacking model with {len(self.base_models)} base models")
        self.stacking_model.fit(X, y)
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Make predictions using the stacking model."""
        if self.stacking_model is None:
            raise ValueError("Model has not been fitted yet")
        return self.stacking_model.predict(X)
    
    def get_base_predictions(self, X: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        """Get predictions from all base models."""
        if self.stacking_model is None:
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
    
    def __init__(self, models: List[Tuple[str, Any]], weights: Optional[List[float]] = None):
        """
        Initialize the ModelBlender.
        
        Args:
            models: List of (name, model) tuples
            weights: Weights for each model (default: equal weights)
        """
        self.models = models
        self.weights = weights or [1.0 / len(models)] * len(models)
        
        if len(self.weights) != len(self.models):
            raise ValueError("Number of weights must match number of models")
        
        # Normalize weights
        total_weight = sum(self.weights)
        self.weights = [w / total_weight for w in self.weights]
    
    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]):
        """
        Fit all models in the blend.
        
        Args:
            X: Feature matrix
            y: Target variable
        """
        logger.info(f"Training {len(self.models)} models for blending")
        
        for name, model in self.models:
            logger.info(f"Training {name}")
            model.fit(X, y)
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Make blended predictions.
        
        Args:
            X: Feature matrix
            
        Returns:
            Weighted average of all model predictions
        """
        predictions = []
        
        for (name, model), weight in zip(self.models, self.weights):
            pred = model.predict(X) * weight
            predictions.append(pred)
        
        return np.sum(predictions, axis=0)
    
    def optimize_weights(self, X: pd.DataFrame, y: pd.Series, cv: int = 5):
        """
        Optimize blending weights using cross-validation.
        
        Args:
            X: Feature matrix
            y: Target variable
            cv: Number of cross-validation folds
        """
        from sklearn.model_selection import KFold
        from scipy.optimize import minimize
        
        kf = KFold(n_splits=cv, shuffle=True, random_state=42)
        
        def objective(weights):
            """Objective function to minimize."""
            # Normalize weights
            weights = weights / weights.sum()
            
            scores = []
            for train_idx, val_idx in kf.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                # Get predictions from each model
                val_preds = []
                for name, model in self.models:
                    # Clone and fit model
                    model_clone = deepcopy(model)
                    model_clone.fit(X_train, y_train)
                    val_preds.append(model_clone.predict(X_val))
                
                # Blend predictions
                blended = np.sum([pred * w for pred, w in zip(val_preds, weights)], axis=0)
                
                # Calculate error
                from sklearn.metrics import mean_squared_error
                score = mean_squared_error(y_val, blended)
                scores.append(score)
            
            return np.mean(scores)
        
        # Optimize weights
        initial_weights = np.array(self.weights)
        bounds = [(0, 1) for _ in self.weights]
        
        result = minimize(objective, initial_weights, method='SLSQP', 
                         bounds=bounds, constraints={'type': 'eq', 'fun': lambda x: x.sum() - 1})
        
        self.weights = result.x.tolist()
        logger.info(f"Optimized weights: {dict(zip([name for name, _ in self.models], self.weights))}")


# Example usage and testing
def example_ensemble_usage():
    """Example of using the ensemble classes."""
    from sklearn.datasets import make_classification, make_regression
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    from sklearn.svm import SVC, SVR
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
    
    # Classification example
    print("=== Classification Example ===")
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, 
                              n_redundant=5, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Define base classifiers
    classifiers = [
        ('dt', DecisionTreeClassifier(random_state=42)),
        ('svc', SVC(probability=True, random_state=42)),
        ('knn', KNeighborsClassifier())
    ]
    
    # Test VotingEnsemble
    print("\n1. VotingEnsemble (Soft Voting):")
    voting = VotingEnsemble(classifiers, voting='soft')
    voting.fit(X_train, y_train)
    score = voting.predict(X_test).mean()
    print(f"   Accuracy: {score:.3f}")
    
    # Test StackingEnsemble
    print("\n2. StackingEnsemble:")
    stacking = StackingEnsemble(classifiers)
    stacking.fit(X_train, y_train)
    score = stacking.predict(X_test).mean()
    print(f"   Accuracy: {score:.3f}")
    
    # Test BaggingEnsemble
    print("\n3. BaggingEnsemble:")
    bagging = BaggingEnsemble(
        base_estimator=DecisionTreeClassifier(random_state=42),
        n_estimators=10,
        random_state=42
    )
    bagging.fit(X_train, y_train)
    score = bagging.predict(X_test).mean()
    print(f"   Accuracy: {score:.3f}")
    
    # Test WeightedEnsemble
    print("\n4. WeightedEnsemble (Optimized Weights):")
    weighted = WeightedEnsemble(classifiers)
    weighted.fit(X_train, y_train)
    score = weighted.predict(X_test).mean()
    print(f"   Accuracy: {score:.3f}")
    print(f"   Optimized weights: {weighted.get_estimator_weights()}")
    
    # Regression example
    print("\n\n=== Regression Example ===")
    X, y = make_regression(n_samples=1000, n_features=20, n_informative=15, 
                          noise=0.1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Define base regressors
    regressors = [
        ('dt', DecisionTreeRegressor(random_state=42)),
        ('svr', SVR()),
        ('knn', KNeighborsRegressor())
    ]
    
    # Test with regression task
    print("\n1. VotingEnsemble (Averaging):")
    voting_reg = VotingEnsemble(regressors)
    voting_reg.fit(X_train, y_train)
    pred = voting_reg.predict(X_test)
    from sklearn.metrics import r2_score
    print(f"   R² Score: {r2_score(y_test, pred):.3f}")
    
    print("\n2. WeightedEnsemble (Optimized Weights):")
    weighted_reg = WeightedEnsemble(regressors)
    weighted_reg.fit(X_train, y_train)
    pred = weighted_reg.predict(X_test)
    print(f"   R² Score: {r2_score(y_test, pred):.3f}")
    print(f"   Optimized weights: {weighted_reg.get_estimator_weights()}")


if __name__ == "__main__":
    example_ensemble_usage()