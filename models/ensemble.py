"""
models/ensemble.py
Ensemble learning methods with enhanced functionality.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Union, Any, Tuple
from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier,
    ExtraTreesRegressor, ExtraTreesClassifier,
    GradientBoostingRegressor, GradientBoostingClassifier,
    HistGradientBoostingRegressor, HistGradientBoostingClassifier,
    VotingRegressor, VotingClassifier,
    StackingRegressor, StackingClassifier
)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import logging
from copy import deepcopy

logger = logging.getLogger(__name__)


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