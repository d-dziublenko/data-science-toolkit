"""
evaluation/uncertainty.py
Uncertainty quantification methods for machine learning predictions.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
from sklearn.model_selection import KFold
from sklearn.base import clone
from sklearn.utils import resample
from scipy import stats
import logging
from tqdm import tqdm
import warnings

logger = logging.getLogger(__name__)


class UncertaintyQuantifier:
    """
    Comprehensive uncertainty quantification for machine learning models.
    
    This class provides multiple methods for estimating prediction uncertainty,
    including bootstrap methods, cross-validation approaches, and ensemble-based
    uncertainty estimates.
    """
    
    def __init__(self, task_type: str = 'regression'):
        """
        Initialize the UncertaintyQuantifier.
        
        Args:
            task_type: Type of ML task ('regression' or 'classification')
        """
        self.task_type = task_type
        self.bootstrap_predictions = None
        self.cv_predictions = None
        self.ensemble_predictions = None
    
    def bootstrap_uncertainty(self,
                            model: Any,
                            X_train: pd.DataFrame,
                            y_train: pd.Series,
                            X_test: pd.DataFrame,
                            n_bootstrap: int = 100,
                            sample_size: Optional[float] = None,
                            confidence_level: float = 0.95) -> Dict[str, np.ndarray]:
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
        logger.info(f"Starting bootstrap uncertainty estimation with {n_bootstrap} iterations")
        
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
            if self.task_type == 'regression':
                predictions = model_clone.predict(X_test)
            else:
                if hasattr(model_clone, 'predict_proba'):
                    predictions = model_clone.predict_proba(X_test)
                else:
                    predictions = model_clone.predict(X_test)
            
            bootstrap_predictions.append(predictions)
        
        # Convert to numpy array
        bootstrap_predictions = np.array(bootstrap_predictions)
        self.bootstrap_predictions = bootstrap_predictions
        
        # Calculate statistics
        if self.task_type == 'regression':
            mean_predictions = np.mean(bootstrap_predictions, axis=0)
            std_predictions = np.std(bootstrap_predictions, axis=0)
            
            # Calculate confidence intervals
            alpha = 1 - confidence_level
            lower_percentile = (alpha / 2) * 100
            upper_percentile = (1 - alpha / 2) * 100
            
            lower_bound = np.percentile(bootstrap_predictions, lower_percentile, axis=0)
            upper_bound = np.percentile(bootstrap_predictions, upper_percentile, axis=0)
            
            return {
                'predictions': mean_predictions,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'std': std_predictions,
                'all_predictions': bootstrap_predictions
            }
        else:
            # For classification
            if len(bootstrap_predictions.shape) == 3:  # Probability predictions
                mean_predictions = np.mean(bootstrap_predictions, axis=0)
                std_predictions = np.std(bootstrap_predictions, axis=0)
                
                # Class predictions from mean probabilities
                class_predictions = np.argmax(mean_predictions, axis=1)
                
                # Prediction entropy as uncertainty measure
                prediction_entropy = -np.sum(mean_predictions * np.log(mean_predictions + 1e-10), axis=1)
                
                return {
                    'predictions': class_predictions,
                    'probabilities': mean_predictions,
                    'std': std_predictions,
                    'entropy': prediction_entropy,
                    'all_predictions': bootstrap_predictions
                }
            else:  # Direct class predictions
                # Mode of predictions
                mode_predictions = stats.mode(bootstrap_predictions, axis=0)[0].ravel()
                
                # Disagreement rate as uncertainty
                disagreement_rate = 1 - (np.sum(bootstrap_predictions == mode_predictions[np.newaxis, :], axis=0) / n_bootstrap)
                
                return {
                    'predictions': mode_predictions,
                    'disagreement_rate': disagreement_rate,
                    'all_predictions': bootstrap_predictions
                }
    
    def cv_uncertainty(self,
                      model: Any,
                      X: pd.DataFrame,
                      y: pd.Series,
                      cv_folds: int = 5,
                      confidence_level: float = 0.95) -> Dict[str, np.ndarray]:
        """
        Estimate uncertainty using cross-validation.
        
        This method uses k-fold cross-validation to estimate prediction uncertainty
        by training on different subsets of the data.
        
        Args:
            model: Base model to use (will be cloned)
            X: Features
            y: Target
            cv_folds: Number of cross-validation folds
            confidence_level: Confidence level for intervals
            
        Returns:
            Dictionary with uncertainty estimates
        """
        logger.info(f"Starting CV uncertainty estimation with {cv_folds} folds")
        
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        predictions = {idx: [] for idx in range(len(X))}
        
        # Cross-validation
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            logger.debug(f"Processing fold {fold + 1}/{cv_folds}")
            
            X_train_fold = X.iloc[train_idx]
            y_train_fold = y.iloc[train_idx]
            X_val_fold = X.iloc[val_idx]
            
            # Clone and train model
            model_clone = clone(model)
            model_clone.fit(X_train_fold, y_train_fold)
            
            # Make predictions
            if self.task_type == 'regression':
                fold_predictions = model_clone.predict(X_val_fold)
                for idx, pred in zip(val_idx, fold_predictions):
                    predictions[idx].append(pred)
            else:
                if hasattr(model_clone, 'predict_proba'):
                    fold_predictions = model_clone.predict_proba(X_val_fold)
                else:
                    fold_predictions = model_clone.predict(X_val_fold)
                
                for idx, pred in zip(val_idx, fold_predictions):
                    predictions[idx].append(pred)
        
        # Calculate statistics for each sample
        mean_predictions = []
        lower_bounds = []
        upper_bounds = []
        std_predictions = []
        
        for idx in range(len(X)):
            if predictions[idx]:  # Has predictions from CV
                preds = np.array(predictions[idx])
                
                if self.task_type == 'regression':
                    mean_predictions.append(np.mean(preds))
                    std_predictions.append(np.std(preds))
                    
                    # Confidence intervals
                    alpha = 1 - confidence_level
                    lower_percentile = (alpha / 2) * 100
                    upper_percentile = (1 - alpha / 2) * 100
                    
                    lower_bounds.append(np.percentile(preds, lower_percentile))
                    upper_bounds.append(np.percentile(preds, upper_percentile))
                else:
                    if len(preds.shape) > 1:  # Probability predictions
                        mean_predictions.append(np.mean(preds, axis=0))
                        std_predictions.append(np.std(preds, axis=0))
                    else:  # Class predictions
                        mean_predictions.append(stats.mode(preds)[0][0])
                        std_predictions.append(np.nan)  # No std for class predictions
        
        self.cv_predictions = predictions
        
        if self.task_type == 'regression':
            return {
                'predictions': np.array(mean_predictions),
                'lower_bound': np.array(lower_bounds),
                'upper_bound': np.array(upper_bounds),
                'std': np.array(std_predictions)
            }
        else:
            return {
                'predictions': np.array(mean_predictions),
                'std': np.array(std_predictions) if std_predictions[0] is not np.nan else None
            }
    
    def ensemble_uncertainty(self,
                           models: List[Any],
                           X: pd.DataFrame,
                           confidence_level: float = 0.95) -> Dict[str, np.ndarray]:
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
            if self.task_type == 'regression':
                pred = model.predict(X)
            else:
                if hasattr(model, 'predict_proba'):
                    pred = model.predict_proba(X)
                else:
                    pred = model.predict(X)
            
            all_predictions.append(pred)
        
        all_predictions = np.array(all_predictions)
        self.ensemble_predictions = all_predictions
        
        # Calculate statistics
        if self.task_type == 'regression':
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
                'predictions': mean_predictions,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'std': std_predictions,
                'agreement': agreement,
                'all_predictions': all_predictions
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
                disagreement = 1 - (np.sum(class_predictions == mode_predictions[np.newaxis, :], axis=0) / len(models))
                
                return {
                    'predictions': np.argmax(mean_probs, axis=1),
                    'probabilities': mean_probs,
                    'std': std_probs,
                    'entropy': entropy,
                    'disagreement': disagreement,
                    'all_predictions': all_predictions
                }
            else:  # Direct class predictions
                mode_predictions = stats.mode(all_predictions, axis=0)[0].ravel()
                disagreement = 1 - (np.sum(all_predictions == mode_predictions[np.newaxis, :], axis=0) / len(models))
                
                return {
                    'predictions': mode_predictions,
                    'disagreement': disagreement,
                    'all_predictions': all_predictions
                }
    
    def dropout_uncertainty(self,
                          model: Any,
                          X: pd.DataFrame,
                          n_iterations: int = 100,
                          confidence_level: float = 0.95) -> Dict[str, np.ndarray]:
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
        if not hasattr(model, 'training'):
            warnings.warn("Model doesn't appear to have dropout capability. Results may not be meaningful.")
        
        # Store predictions
        mc_predictions = []
        
        # Enable dropout during inference
        if hasattr(model, 'training'):
            original_training_state = model.training
            model.training = True  # Enable dropout
        
        for i in tqdm(range(n_iterations), desc="MC Dropout iterations"):
            if self.task_type == 'regression':
                pred = model.predict(X)
            else:
                if hasattr(model, 'predict_proba'):
                    pred = model.predict_proba(X)
                else:
                    pred = model.predict(X)
            
            mc_predictions.append(pred)
        
        # Restore original training state
        if hasattr(model, 'training'):
            model.training = original_training_state
        
        mc_predictions = np.array(mc_predictions)
        
        # Calculate statistics
        if self.task_type == 'regression':
            mean_predictions = np.mean(mc_predictions, axis=0)
            std_predictions = np.std(mc_predictions, axis=0)
            
            # Confidence intervals
            alpha = 1 - confidence_level
            lower_percentile = (alpha / 2) * 100
            upper_percentile = (1 - alpha / 2) * 100
            
            lower_bound = np.percentile(mc_predictions, lower_percentile, axis=0)
            upper_bound = np.percentile(mc_predictions, upper_percentile, axis=0)
            
            return {
                'predictions': mean_predictions,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'std': std_predictions,
                'epistemic_uncertainty': std_predictions,  # Dropout captures epistemic uncertainty
                'all_predictions': mc_predictions
            }
        else:
            # Similar logic for classification
            return self._process_classification_predictions(mc_predictions, confidence_level)
    
    def quantile_uncertainty(self,
                           models: Union[Any, List[Any]],
                           X: pd.DataFrame,
                           quantiles: List[float] = [0.05, 0.95]) -> Dict[str, np.ndarray]:
        """
        Estimate uncertainty using quantile regression models.
        
        This method requires models trained to predict specific quantiles
        of the target distribution.
        
        Args:
            models: Single quantile model or list of quantile models
            X: Features to predict on
            quantiles: List of quantiles that the models predict
            
        Returns:
            Dictionary with quantile predictions
        """
        logger.info(f"Estimating uncertainty using quantile regression")
        
        if not isinstance(models, list):
            models = [models]
        
        if len(models) != len(quantiles):
            raise ValueError("Number of models must match number of quantiles")
        
        quantile_predictions = {}
        
        for model, q in zip(models, quantiles):
            predictions = model.predict(X)
            quantile_predictions[f'q{int(q*100)}'] = predictions
        
        # Calculate prediction intervals
        if len(quantiles) == 2 and quantiles[0] < 0.5 < quantiles[1]:
            lower_key = f'q{int(quantiles[0]*100)}'
            upper_key = f'q{int(quantiles[1]*100)}'
            
            # Median as point estimate (if available)
            if 0.5 in quantiles:
                median_idx = quantiles.index(0.5)
                median_pred = models[median_idx].predict(X)
            else:
                # Use mean of bounds as estimate
                median_pred = (quantile_predictions[lower_key] + quantile_predictions[upper_key]) / 2
            
            interval_width = quantile_predictions[upper_key] - quantile_predictions[lower_key]
            
            return {
                'predictions': median_pred,
                'lower_bound': quantile_predictions[lower_key],
                'upper_bound': quantile_predictions[upper_key],
                'interval_width': interval_width,
                'quantile_predictions': quantile_predictions
            }
        
        return {'quantile_predictions': quantile_predictions}
    
    def residual_uncertainty(self,
                           model: Any,
                           X_train: pd.DataFrame,
                           y_train: pd.Series,
                           X_test: pd.DataFrame,
                           method: str = 'empirical',
                           confidence_level: float = 0.95) -> Dict[str, np.ndarray]:
        """
        Estimate uncertainty based on residual analysis.
        
        This method analyzes the residuals from training data to estimate
        prediction intervals for new data.
        
        Args:
            model: Fitted model
            X_train: Training features
            y_train: Training target
            X_test: Test features
            method: Method for residual analysis ('empirical' or 'normalized')
            confidence_level: Confidence level for intervals
            
        Returns:
            Dictionary with uncertainty estimates
        """
        logger.info(f"Estimating uncertainty using residual analysis ({method})")
        
        # Get training predictions and residuals
        train_predictions = model.predict(X_train)
        residuals = y_train - train_predictions
        
        # Get test predictions
        test_predictions = model.predict(X_test)
        
        if method == 'empirical':
            # Simple empirical percentiles
            alpha = 1 - confidence_level
            lower_percentile = (alpha / 2) * 100
            upper_percentile = (1 - alpha / 2) * 100
            
            residual_lower = np.percentile(residuals, lower_percentile)
            residual_upper = np.percentile(residuals, upper_percentile)
            
            lower_bound = test_predictions + residual_lower
            upper_bound = test_predictions + residual_upper
            
            # Residual standard deviation
            residual_std = np.std(residuals)
            
        elif method == 'normalized':
            # Normalized residuals (assuming homoscedasticity)
            residual_std = np.std(residuals)
            
            # Standard normal quantiles
            z_score = stats.norm.ppf((1 + confidence_level) / 2)
            
            lower_bound = test_predictions - z_score * residual_std
            upper_bound = test_predictions + z_score * residual_std
            
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return {
            'predictions': test_predictions,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'residual_std': residual_std,
            'training_residuals': residuals
        }
    
    def conformal_prediction(self,
                           model: Any,
                           X_train: pd.DataFrame,
                           y_train: pd.Series,
                           X_cal: pd.DataFrame,
                           y_cal: pd.Series,
                           X_test: pd.DataFrame,
                           alpha: float = 0.1) -> Dict[str, np.ndarray]:
        """
        Conformal prediction for uncertainty quantification.
        
        This method provides prediction intervals with guaranteed coverage
        under the assumption of exchangeability.
        
        Args:
            model: Base model (already fitted)
            X_train: Training features
            y_train: Training target
            X_cal: Calibration features
            y_cal: Calibration target
            X_test: Test features
            alpha: Miscoverage rate (1 - confidence_level)
            
        Returns:
            Dictionary with conformal prediction intervals
        """
        logger.info(f"Computing conformal prediction intervals with alpha={alpha}")
        
        # Get calibration scores (nonconformity scores)
        cal_predictions = model.predict(X_cal)
        cal_scores = np.abs(y_cal - cal_predictions)
        
        # Compute the quantile of calibration scores
        n_cal = len(y_cal)
        q_level = np.ceil((n_cal + 1) * (1 - alpha)) / n_cal
        q_level = np.clip(q_level, 0, 1)
        qhat = np.quantile(cal_scores, q_level)
        
        # Get test predictions
        test_predictions = model.predict(X_test)
        
        # Construct prediction intervals
        lower_bound = test_predictions - qhat
        upper_bound = test_predictions + qhat
        
        return {
            'predictions': test_predictions,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'interval_width': 2 * qhat,
            'calibration_scores': cal_scores,
            'qhat': qhat
        }
    
    def heteroscedastic_uncertainty(self,
                                  mean_model: Any,
                                  variance_model: Any,
                                  X: pd.DataFrame,
                                  confidence_level: float = 0.95) -> Dict[str, np.ndarray]:
        """
        Estimate uncertainty for heteroscedastic regression.
        
        This method uses separate models for mean and variance prediction
        to handle heteroscedastic noise.
        
        Args:
            mean_model: Model predicting the mean
            variance_model: Model predicting the variance
            X: Features to predict on
            confidence_level: Confidence level for intervals
            
        Returns:
            Dictionary with uncertainty estimates
        """
        logger.info("Estimating heteroscedastic uncertainty")
        
        # Get mean predictions
        mean_predictions = mean_model.predict(X)
        
        # Get variance predictions
        variance_predictions = variance_model.predict(X)
        
        # Ensure positive variance
        variance_predictions = np.maximum(variance_predictions, 1e-6)
        
        # Standard deviation
        std_predictions = np.sqrt(variance_predictions)
        
        # Confidence intervals assuming normal distribution
        z_score = stats.norm.ppf((1 + confidence_level) / 2)
        
        lower_bound = mean_predictions - z_score * std_predictions
        upper_bound = mean_predictions + z_score * std_predictions
        
        return {
            'predictions': mean_predictions,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'std': std_predictions,
            'variance': variance_predictions,
            'aleatoric_uncertainty': std_predictions  # Variance model captures aleatoric uncertainty
        }
    
    def _process_classification_predictions(self,
                                          predictions: np.ndarray,
                                          confidence_level: float) -> Dict[str, np.ndarray]:
        """
        Helper method to process classification predictions.
        
        Args:
            predictions: Array of predictions
            confidence_level: Confidence level
            
        Returns:
            Dictionary with processed predictions
        """
        if len(predictions.shape) == 3:  # Probability predictions
            mean_probs = np.mean(predictions, axis=0)
            std_probs = np.std(predictions, axis=0)
            
            # Class predictions
            class_predictions = np.argmax(mean_probs, axis=1)
            
            # Entropy as uncertainty
            entropy = -np.sum(mean_probs * np.log(mean_probs + 1e-10), axis=1)
            
            # Maximum probability as confidence
            max_prob = np.max(mean_probs, axis=1)
            
            return {
                'predictions': class_predictions,
                'probabilities': mean_probs,
                'std': std_probs,
                'entropy': entropy,
                'confidence': max_prob,
                'all_predictions': predictions
            }
        else:  # Direct class predictions
            mode_predictions = stats.mode(predictions, axis=0)[0].ravel()
            
            # Agreement rate as confidence
            agreement_rate = np.sum(predictions == mode_predictions[np.newaxis, :], axis=0) / len(predictions)
            
            return {
                'predictions': mode_predictions,
                'confidence': agreement_rate,
                'all_predictions': predictions
            }
    
    def plot_uncertainty_intervals(self,
                                 y_true: Optional[np.ndarray],
                                 predictions: np.ndarray,
                                 lower_bound: np.ndarray,
                                 upper_bound: np.ndarray,
                                 indices: Optional[np.ndarray] = None,
                                 title: str = "Prediction Intervals",
                                 figsize: Tuple[int, int] = (12, 6)):
        """
        Plot prediction intervals with uncertainty bounds.
        
        Args:
            y_true: True values (optional)
            predictions: Point predictions
            lower_bound: Lower confidence bounds
            upper_bound: Upper confidence bounds
            indices: Indices to plot (None for all)
            title: Plot title
            figsize: Figure size
        """
        import matplotlib.pyplot as plt
        
        if indices is None:
            indices = np.arange(len(predictions))
        
        plt.figure(figsize=figsize)
        
        # Plot predictions and intervals
        plt.scatter(indices, predictions[indices], label='Predictions', alpha=0.6, s=30)
        plt.fill_between(indices, lower_bound[indices], upper_bound[indices], 
                        alpha=0.3, label='Prediction Interval')
        
        # Plot true values if provided
        if y_true is not None:
            plt.scatter(indices, y_true[indices], label='True Values', 
                       alpha=0.6, s=30, marker='x')
            
            # Calculate coverage
            coverage = np.mean((y_true >= lower_bound) & (y_true <= upper_bound))
            plt.text(0.02, 0.98, f'Coverage: {coverage:.2%}', 
                    transform=plt.gca().transAxes, verticalalignment='top')
        
        plt.xlabel('Sample Index')
        plt.ylabel('Value')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        return plt.gcf()
    
    def calculate_metrics(self,
                         y_true: np.ndarray,
                         predictions: np.ndarray,
                         lower_bound: np.ndarray,
                         upper_bound: np.ndarray) -> Dict[str, float]:
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
        interval_score = interval_width + (2/alpha) * (lower_penalty + upper_penalty)
        mean_interval_score = np.mean(interval_score)
        
        # Coverage Width-based Criterion (CWC)
        eta = 0.5  # Balance parameter
        cwc = (1 - coverage) + eta * normalized_width
        
        # Root Mean Square Error (RMSE)
        rmse = np.sqrt(np.mean((y_true - predictions)**2))
        
        # Mean Absolute Error (MAE)
        mae = np.mean(np.abs(y_true - predictions))
        
        return {
            'picp': coverage,
            'mpiw': mean_width,
            'normalized_mpiw': normalized_width,
            'mean_interval_score': mean_interval_score,
            'cwc': cwc,
            'rmse': rmse,
            'mae': mae
        }


# Example usage functions
def example_regression_uncertainty():
    """
    Example of using UncertaintyQuantifier for regression tasks.
    """
    from sklearn.datasets import make_regression
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    
    # Generate synthetic data
    X, y = make_regression(n_samples=1000, n_features=10, noise=10, random_state=42)
    X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])
    y = pd.Series(y, name='target')
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize model and uncertainty quantifier
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    uq = UncertaintyQuantifier(task_type='regression')
    
    # Bootstrap uncertainty
    results = uq.bootstrap_uncertainty(
        model, X_train, y_train, X_test, 
        n_bootstrap=50, confidence_level=0.95
    )
    
    # Calculate metrics
    metrics = uq.calculate_metrics(
        y_test.values, 
        results['predictions'],
        results['lower_bound'],
        results['upper_bound']
    )
    
    print("Bootstrap Uncertainty Results:")
    print(f"Coverage: {metrics['picp']:.2%}")
    print(f"Mean Interval Width: {metrics['mpiw']:.2f}")
    print(f"RMSE: {metrics['rmse']:.2f}")
    
    return uq, results, metrics


def example_classification_uncertainty():
    """
    Example of using UncertaintyQuantifier for classification tasks.
    """
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    
    # Generate synthetic data
    X, y = make_classification(n_samples=1000, n_features=10, n_classes=3, 
                              n_informative=5, random_state=42)
    X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])
    y = pd.Series(y, name='target')
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize models for ensemble
    models = [
        RandomForestClassifier(n_estimators=100, random_state=i)
        for i in range(5)
    ]
    
    # Train models
    for model in models:
        model.fit(X_train, y_train)
    
    # Uncertainty quantification
    uq = UncertaintyQuantifier(task_type='classification')
    
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
    
    print("\n" + "="*50 + "\n")
    
    print("Running classification uncertainty example...")
    class_uq, class_results = example_classification_uncertainty()