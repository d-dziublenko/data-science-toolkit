"""
models/base.py
Base model classes and interfaces for the data science toolkit.
"""

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Union, Tuple, Callable
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.metrics import make_scorer
import joblib
import json
import os
from datetime import datetime
import logging
import warnings

logger = logging.getLogger(__name__)


class BaseModel(ABC, BaseEstimator):
    """
    Abstract base class for all models in the toolkit.
    
    This class provides common functionality for model training,
    prediction, evaluation, and persistence.
    """
    
    def __init__(self, name: str = None, random_state: int = None):
        """
        Initialize the base model.
        
        Args:
            name: Model name for identification
            random_state: Random seed for reproducibility
        """
        self.name = name or self.__class__.__name__
        self.random_state = random_state
        self.is_fitted = False
        self.feature_names_in_ = None
        self.n_features_in_ = None
        self.metadata_ = {
            'created_at': datetime.now().isoformat(),
            'version': '1.0.0'
        }
        self.training_history_ = []
    
    @abstractmethod
    def fit(self, X: Union[np.ndarray, pd.DataFrame], 
            y: Union[np.ndarray, pd.Series], **kwargs) -> 'BaseModel':
        """
        Fit the model to training data.
        
        Args:
            X: Training features
            y: Training target
            **kwargs: Additional arguments
            
        Returns:
            Self
        """
        pass
    
    @abstractmethod
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Features to predict on
            
        Returns:
            Predictions
        """
        pass
    
    def _validate_input(self, X: Union[np.ndarray, pd.DataFrame], 
                       y: Optional[Union[np.ndarray, pd.Series]] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Validate and convert input data.
        
        Args:
            X: Features
            y: Target (optional)
            
        Returns:
            Validated X and y as numpy arrays
        """
        # Convert to numpy if pandas
        if isinstance(X, pd.DataFrame):
            if self.feature_names_in_ is None:
                self.feature_names_in_ = X.columns.tolist()
            else:
                # Check feature names match
                if not all(X.columns == self.feature_names_in_):
                    missing = set(self.feature_names_in_) - set(X.columns)
                    extra = set(X.columns) - set(self.feature_names_in_)
                    if missing:
                        raise ValueError(f"Missing features: {missing}")
                    if extra:
                        warnings.warn(f"Extra features will be ignored: {extra}")
                    X = X[self.feature_names_in_]
            X_array = X.values
        else:
            X_array = np.asarray(X)
        
        # Store number of features
        if self.n_features_in_ is None:
            self.n_features_in_ = X_array.shape[1]
        elif X_array.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features, got {X_array.shape[1]}")
        
        # Convert y if provided
        y_array = None
        if y is not None:
            if isinstance(y, pd.Series):
                y_array = y.values
            else:
                y_array = np.asarray(y)
        
        return X_array, y_array
    
    def _check_is_fitted(self):
        """Check if the model is fitted."""
        if not self.is_fitted:
            raise ValueError(f"{self.name} is not fitted yet. Call 'fit' before using this method.")
    
    def save(self, filepath: str, include_metadata: bool = True):
        """
        Save the model to disk.
        
        Args:
            filepath: Path to save the model
            include_metadata: Whether to save metadata separately
        """
        logger.info(f"Saving model to {filepath}")
        
        # Create directory if needed
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save model
        joblib.dump(self, filepath)
        
        # Save metadata if requested
        if include_metadata:
            metadata_path = filepath.replace('.pkl', '_metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump({
                    'model_name': self.name,
                    'model_class': self.__class__.__name__,
                    'feature_names': self.feature_names_in_,
                    'n_features': self.n_features_in_,
                    'metadata': self.metadata_,
                    'training_history': self.training_history_
                }, f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'BaseModel':
        """
        Load a model from disk.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            Loaded model
        """
        logger.info(f"Loading model from {filepath}")
        return joblib.load(filepath)
    
    def cross_validate(self, X: Union[np.ndarray, pd.DataFrame], 
                      y: Union[np.ndarray, pd.Series],
                      cv: int = 5,
                      scoring: Union[str, List[str], Dict[str, Callable]] = None,
                      return_train_score: bool = False) -> Dict[str, np.ndarray]:
        """
        Perform cross-validation.
        
        Args:
            X: Features
            y: Target
            cv: Number of cross-validation folds
            scoring: Scoring metric(s)
            return_train_score: Whether to return training scores
            
        Returns:
            Cross-validation results
        """
        X_array, y_array = self._validate_input(X, y)
        
        return cross_validate(
            self, X_array, y_array,
            cv=cv,
            scoring=scoring,
            return_train_score=return_train_score,
            n_jobs=-1
        )
    
    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """
        Get model parameters.
        
        Args:
            deep: Whether to get parameters of sub-objects
            
        Returns:
            Parameter dictionary
        """
        params = {
            'name': self.name,
            'random_state': self.random_state
        }
        return params
    
    def set_params(self, **params) -> 'BaseModel':
        """
        Set model parameters.
        
        Args:
            **params: Parameters to set
            
        Returns:
            Self
        """
        for key, value in params.items():
            setattr(self, key, value)
        return self
    
    def get_feature_names_out(self, input_features: Optional[List[str]] = None) -> List[str]:
        """
        Get output feature names.
        
        Args:
            input_features: Input feature names
            
        Returns:
            Output feature names
        """
        if input_features is None:
            input_features = self.feature_names_in_
        return input_features or [f"x{i}" for i in range(self.n_features_in_)]
    
    def __repr__(self) -> str:
        """String representation of the model."""
        params = self.get_params()
        param_str = ", ".join([f"{k}={v}" for k, v in params.items()])
        return f"{self.__class__.__name__}({param_str})"


class BaseRegressor(BaseModel, RegressorMixin):
    """
    Base class for regression models.
    
    Provides additional functionality specific to regression tasks.
    """
    
    def score(self, X: Union[np.ndarray, pd.DataFrame], 
              y: Union[np.ndarray, pd.Series]) -> float:
        """
        Calculate R² score.
        
        Args:
            X: Features
            y: True target values
            
        Returns:
            R² score
        """
        from sklearn.metrics import r2_score
        predictions = self.predict(X)
        return r2_score(y, predictions)
    
    def predict_interval(self, X: Union[np.ndarray, pd.DataFrame],
                        confidence: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict with confidence intervals.
        
        Args:
            X: Features to predict on
            confidence: Confidence level
            
        Returns:
            Tuple of (predictions, lower_bound, upper_bound)
        """
        raise NotImplementedError("Prediction intervals not implemented for this model")


class BaseClassifier(BaseModel, ClassifierMixin):
    """
    Base class for classification models.
    
    Provides additional functionality specific to classification tasks.
    """
    
    def __init__(self, name: str = None, random_state: int = None):
        """Initialize the classifier."""
        super().__init__(name, random_state)
        self.classes_ = None
        self.n_classes_ = None
    
    @abstractmethod
    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Features to predict on
            
        Returns:
            Class probabilities
        """
        pass
    
    def predict_log_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Predict log probabilities.
        
        Args:
            X: Features to predict on
            
        Returns:
            Log probabilities
        """
        return np.log(self.predict_proba(X))
    
    def score(self, X: Union[np.ndarray, pd.DataFrame], 
              y: Union[np.ndarray, pd.Series]) -> float:
        """
        Calculate accuracy score.
        
        Args:
            X: Features
            y: True labels
            
        Returns:
            Accuracy score
        """
        from sklearn.metrics import accuracy_score
        predictions = self.predict(X)
        return accuracy_score(y, predictions)


class EnsembleModel(BaseModel):
    """
    Base class for ensemble models.
    
    Provides functionality for combining multiple base models.
    """
    
    def __init__(self, base_models: List[BaseModel], 
                 name: str = None,
                 random_state: int = None):
        """
        Initialize ensemble model.
        
        Args:
            base_models: List of base models
            name: Ensemble name
            random_state: Random seed
        """
        super().__init__(name, random_state)
        self.base_models = base_models
        self.n_models = len(base_models)
        self.model_weights_ = None
    
    def fit(self, X: Union[np.ndarray, pd.DataFrame], 
            y: Union[np.ndarray, pd.Series],
            sample_weights: Optional[np.ndarray] = None) -> 'EnsembleModel':
        """
        Fit all base models.
        
        Args:
            X: Training features
            y: Training target
            sample_weights: Sample weights
            
        Returns:
            Self
        """
        X_array, y_array = self._validate_input(X, y)
        
        logger.info(f"Fitting ensemble with {self.n_models} models")
        
        for i, model in enumerate(self.base_models):
            logger.debug(f"Fitting model {i+1}/{self.n_models}: {model.name}")
            if sample_weights is not None and hasattr(model, 'fit'):
                try:
                    model.fit(X_array, y_array, sample_weight=sample_weights)
                except TypeError:
                    # Model doesn't support sample weights
                    model.fit(X_array, y_array)
            else:
                model.fit(X_array, y_array)
        
        self.is_fitted = True
        
        # Record training
        self.training_history_.append({
            'timestamp': datetime.now().isoformat(),
            'n_samples': len(X_array),
            'n_models': self.n_models
        })
        
        return self
    
    def get_model_predictions(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Get predictions from all base models.
        
        Args:
            X: Features to predict on
            
        Returns:
            Array of predictions from each model
        """
        self._check_is_fitted()
        X_array, _ = self._validate_input(X)
        
        predictions = []
        for model in self.base_models:
            pred = model.predict(X_array)
            predictions.append(pred)
        
        return np.array(predictions)


class MetaModel(BaseModel):
    """
    Base class for meta-learning models.
    
    Provides functionality for stacking and blending approaches.
    """
    
    def __init__(self, base_models: List[BaseModel],
                 meta_model: BaseModel,
                 use_proba: bool = False,
                 name: str = None,
                 random_state: int = None):
        """
        Initialize meta model.
        
        Args:
            base_models: List of base models
            meta_model: Meta-level model
            use_proba: Whether to use probabilities for classification
            name: Model name
            random_state: Random seed
        """
        super().__init__(name, random_state)
        self.base_models = base_models
        self.meta_model = meta_model
        self.use_proba = use_proba
    
    def _get_meta_features(self, X: np.ndarray) -> np.ndarray:
        """
        Generate meta-features from base model predictions.
        
        Args:
            X: Original features
            
        Returns:
            Meta-features
        """
        meta_features = []
        
        for model in self.base_models:
            if self.use_proba and hasattr(model, 'predict_proba'):
                # Use probabilities for classification
                pred = model.predict_proba(X)
                # Flatten probabilities
                if len(pred.shape) > 1:
                    pred = pred[:, 1:].reshape(len(X), -1)
            else:
                # Use predictions
                pred = model.predict(X).reshape(-1, 1)
            
            meta_features.append(pred)
        
        return np.hstack(meta_features)
    
    def fit(self, X: Union[np.ndarray, pd.DataFrame], 
            y: Union[np.ndarray, pd.Series],
            validation_split: float = 0.2) -> 'MetaModel':
        """
        Fit the meta model using a validation split approach.
        
        Args:
            X: Training features
            y: Training target
            validation_split: Fraction of data for meta-model training
            
        Returns:
            Self
        """
        from sklearn.model_selection import train_test_split
        
        X_array, y_array = self._validate_input(X, y)
        
        # Split data
        X_base, X_meta, y_base, y_meta = train_test_split(
            X_array, y_array, 
            test_size=validation_split,
            random_state=self.random_state
        )
        
        # Train base models
        logger.info(f"Training {len(self.base_models)} base models")
        for i, model in enumerate(self.base_models):
            logger.debug(f"Training base model {i+1}")
            model.fit(X_base, y_base)
        
        # Generate meta features
        meta_features = self._get_meta_features(X_meta)
        
        # Train meta model
        logger.info("Training meta model")
        self.meta_model.fit(meta_features, y_meta)
        
        # Retrain base models on full data
        logger.info("Retraining base models on full dataset")
        for model in self.base_models:
            model.fit(X_array, y_array)
        
        self.is_fitted = True
        
        return self
    
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Make predictions using the meta model.
        
        Args:
            X: Features to predict on
            
        Returns:
            Predictions
        """
        self._check_is_fitted()
        X_array, _ = self._validate_input(X)
        
        # Get meta features
        meta_features = self._get_meta_features(X_array)
        
        # Make predictions
        return self.meta_model.predict(meta_features)


class AutoML(BaseModel):
    """
    Base class for AutoML functionality.
    
    Provides automated model selection and hyperparameter tuning.
    """
    
    def __init__(self, task_type: str = 'auto',
                 time_budget: int = 3600,
                 n_jobs: int = -1,
                 name: str = None,
                 random_state: int = None):
        """
        Initialize AutoML.
        
        Args:
            task_type: Type of task ('regression', 'classification', 'auto')
            time_budget: Time budget in seconds
            n_jobs: Number of parallel jobs
            name: Model name
            random_state: Random seed
        """
        super().__init__(name, random_state)
        self.task_type = task_type
        self.time_budget = time_budget
        self.n_jobs = n_jobs
        self.best_model_ = None
        self.search_history_ = []
    
    def _detect_task_type(self, y: np.ndarray) -> str:
        """
        Automatically detect task type from target variable.
        
        Args:
            y: Target variable
            
        Returns:
            Task type ('regression' or 'classification')
        """
        # Check if target is continuous or discrete
        unique_values = np.unique(y)
        n_unique = len(unique_values)
        
        # Heuristics for task detection
        if n_unique == 2:
            return 'classification'
        elif n_unique < 20 and n_unique < len(y) * 0.05:
            return 'classification'
        elif np.issubdtype(y.dtype, np.integer) and n_unique < 50:
            return 'classification'
        else:
            return 'regression'
    
    def get_candidate_models(self) -> List[BaseModel]:
        """
        Get list of candidate models based on task type.
        
        Returns:
            List of model instances
        """
        if self.task_type == 'regression':
            from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
            from sklearn.linear_model import Ridge, Lasso, ElasticNet
            from sklearn.svm import SVR
            
            return [
                RandomForestRegressor(random_state=self.random_state),
                GradientBoostingRegressor(random_state=self.random_state),
                Ridge(random_state=self.random_state),
                Lasso(random_state=self.random_state),
                ElasticNet(random_state=self.random_state),
                SVR()
            ]
        else:
            from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
            from sklearn.linear_model import LogisticRegression
            from sklearn.svm import SVC
            
            return [
                RandomForestClassifier(random_state=self.random_state),
                GradientBoostingClassifier(random_state=self.random_state),
                LogisticRegression(random_state=self.random_state),
                SVC(probability=True, random_state=self.random_state)
            ]
    
    def fit(self, X: Union[np.ndarray, pd.DataFrame], 
            y: Union[np.ndarray, pd.Series]) -> 'AutoML':
        """
        Automatically select and train the best model.
        
        Args:
            X: Training features
            y: Training target
            
        Returns:
            Self
        """
        X_array, y_array = self._validate_input(X, y)
        
        # Detect task type if auto
        if self.task_type == 'auto':
            self.task_type = self._detect_task_type(y_array)
            logger.info(f"Detected task type: {self.task_type}")
        
        # Get candidate models
        candidates = self.get_candidate_models()
        
        # Evaluate each model
        logger.info(f"Evaluating {len(candidates)} candidate models")
        best_score = -np.inf
        
        for model in candidates:
            logger.debug(f"Evaluating {model.__class__.__name__}")
            
            # Cross-validation
            try:
                scores = cross_val_score(
                    model, X_array, y_array,
                    cv=5,
                    n_jobs=self.n_jobs
                )
                mean_score = np.mean(scores)
                
                self.search_history_.append({
                    'model': model.__class__.__name__,
                    'score': mean_score,
                    'std': np.std(scores)
                })
                
                if mean_score > best_score:
                    best_score = mean_score
                    self.best_model_ = model
                    
            except Exception as e:
                logger.warning(f"Failed to evaluate {model.__class__.__name__}: {str(e)}")
        
        # Train best model on full data
        logger.info(f"Training best model: {self.best_model_.__class__.__name__}")
        self.best_model_.fit(X_array, y_array)
        
        self.is_fitted = True
        
        return self
    
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Make predictions using the best model.
        
        Args:
            X: Features to predict on
            
        Returns:
            Predictions
        """
        self._check_is_fitted()
        X_array, _ = self._validate_input(X)
        return self.best_model_.predict(X_array)
    
    def get_search_results(self) -> pd.DataFrame:
        """
        Get model search results as DataFrame.
        
        Returns:
            DataFrame with search results
        """
        return pd.DataFrame(self.search_history_).sort_values('score', ascending=False)


# Example implementations
class SimpleLinearModel(BaseRegressor):
    """
    Simple linear regression model for demonstration.
    """
    
    def __init__(self, fit_intercept: bool = True, name: str = None, random_state: int = None):
        """Initialize the model."""
        super().__init__(name, random_state)
        self.fit_intercept = fit_intercept
        self.coef_ = None
        self.intercept_ = None
    
    def fit(self, X: Union[np.ndarray, pd.DataFrame], 
            y: Union[np.ndarray, pd.Series]) -> 'SimpleLinearModel':
        """Fit the linear model."""
        X_array, y_array = self._validate_input(X, y)
        
        # Add intercept if needed
        if self.fit_intercept:
            X_with_intercept = np.column_stack([np.ones(len(X_array)), X_array])
        else:
            X_with_intercept = X_array
        
        # Solve normal equation
        coefficients = np.linalg.lstsq(X_with_intercept, y_array, rcond=None)[0]
        
        if self.fit_intercept:
            self.intercept_ = coefficients[0]
            self.coef_ = coefficients[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = coefficients
        
        self.is_fitted = True
        
        return self
    
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Make predictions."""
        self._check_is_fitted()
        X_array, _ = self._validate_input(X)
        
        return X_array @ self.coef_ + self.intercept_


# Utility functions
def create_model_pipeline(preprocessor: Any, model: BaseModel, name: str = None) -> Any:
    """
    Create a pipeline with preprocessing and model.
    
    Args:
        preprocessor: Preprocessing pipeline
        model: Model instance
        name: Pipeline name
        
    Returns:
        Pipeline instance
    """
    from sklearn.pipeline import Pipeline
    
    return Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ], memory=None)


if __name__ == "__main__":
    # Example usage
    from sklearn.datasets import make_regression
    
    # Generate data
    X, y = make_regression(n_samples=100, n_features=5, noise=10, random_state=42)
    X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(5)])
    y = pd.Series(y, name='target')
    
    # Test simple linear model
    model = SimpleLinearModel(name="SimpleLinear")
    model.fit(X, y)
    predictions = model.predict(X)
    score = model.score(X, y)
    
    print(f"Model: {model}")
    print(f"R² Score: {score:.4f}")
    print(f"Coefficients: {model.coef_}")
    
    # Test AutoML
    automl = AutoML(task_type='regression', name="AutoRegressor")
    automl.fit(X, y)
    
    print(f"\nAutoML Best Model: {automl.best_model_.__class__.__name__}")
    print("\nSearch Results:")
    print(automl.get_search_results())