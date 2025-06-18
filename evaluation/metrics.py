"""
evaluation/metrics.py
Comprehensive evaluation metrics for machine learning models.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
from sklearn.metrics import (
    # Regression metrics
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error, explained_variance_score,
    median_absolute_error, max_error,
    # Classification metrics
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, log_loss,
    confusion_matrix, classification_report, cohen_kappa_score,
    matthews_corrcoef, balanced_accuracy_score,
    # Clustering metrics
    silhouette_score, calinski_harabasz_score, davies_bouldin_score,
    # Other metrics
    precision_recall_curve, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, learning_curve
from sklearn.calibration import calibration_curve
from scipy import stats
import logging

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Comprehensive model evaluation class.
    
    Provides methods for calculating various metrics, generating reports,
    and visualizing model performance.
    """
    
    def __init__(self, task_type: str = 'regression'):
        """
        Initialize the ModelEvaluator.
        
        Args:
            task_type: Type of ML task ('regression' or 'classification')
        """
        self.task_type = task_type
        self.metrics_history = []
        self.predictions = None
        self.true_values = None
    
    def evaluate(self,
                y_true: Union[pd.Series, np.ndarray],
                y_pred: Union[pd.Series, np.ndarray],
                model_name: str = 'model',
                save_predictions: bool = True) -> Dict[str, float]:
        """
        Evaluate model predictions using appropriate metrics.
        
        Args:
            y_true: True target values
            y_pred: Predicted values
            model_name: Name of the model for tracking
            save_predictions: Whether to save predictions for later analysis
            
        Returns:
            Dictionary of metric names and values
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        if save_predictions:
            self.true_values = y_true
            self.predictions = y_pred
        
        if self.task_type == 'regression':
            metrics = self._calculate_regression_metrics(y_true, y_pred)
        else:
            metrics = self._calculate_classification_metrics(y_true, y_pred)
        
        # Add model name and timestamp
        metrics['model_name'] = model_name
        metrics['timestamp'] = pd.Timestamp.now()
        
        # Store in history
        self.metrics_history.append(metrics)
        
        return metrics
    
    def _calculate_regression_metrics(self,
                                    y_true: np.ndarray,
                                    y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate regression metrics."""
        metrics = {
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'mape': mean_absolute_percentage_error(y_true, y_pred),
            'explained_variance': explained_variance_score(y_true, y_pred)
        }
        
        # Additional custom metrics
        metrics['normalized_rmse'] = metrics['rmse'] / np.mean(y_true)
        metrics['max_error'] = np.max(np.abs(y_true - y_pred))
        
        # Adjusted R-squared (requires number of features)
        n = len(y_true)
        if hasattr(self, 'n_features'):
            p = self.n_features
            metrics['adjusted_r2'] = 1 - (1 - metrics['r2']) * (n - 1) / (n - p - 1)
        
        return metrics
    
    def _calculate_classification_metrics(self,
                                        y_true: np.ndarray,
                                        y_pred: np.ndarray,
                                        y_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Calculate classification metrics."""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1': f1_score(y_true, y_pred, average='weighted')
        }
        
        # Binary classification specific metrics
        if len(np.unique(y_true)) == 2:
            metrics['precision_binary'] = precision_score(y_true, y_pred)
            metrics['recall_binary'] = recall_score(y_true, y_pred)
            metrics['f1_binary'] = f1_score(y_true, y_pred)
            
            if y_proba is not None:
                metrics['roc_auc'] = roc_auc_score(y_true, y_proba[:, 1])
                metrics['average_precision'] = average_precision_score(y_true, y_proba[:, 1])
                metrics['log_loss'] = log_loss(y_true, y_proba)
        
        return metrics
    
    def print_report(self, metrics: Optional[Dict[str, float]] = None):
        """
        Print a formatted evaluation report.
        
        Args:
            metrics: Metrics dictionary (uses last evaluation if None)
        """
        if metrics is None:
            if not self.metrics_history:
                logger.warning("No metrics available to report")
                return
            metrics = self.metrics_history[-1]
        
        print(f"\n{'='*50}")
        print(f"Model Evaluation Report: {metrics.get('model_name', 'Unknown')}")
        print(f"{'='*50}")
        
        if self.task_type == 'regression':
            print(f"RMSE:              {metrics['rmse']:.4f}")
            print(f"MAE:               {metrics['mae']:.4f}")
            print(f"R²:                {metrics['r2']:.4f}")
            print(f"MAPE:              {metrics['mape']:.4f}")
            print(f"Normalized RMSE:   {metrics['normalized_rmse']:.4f}")
            print(f"Max Error:         {metrics['max_error']:.4f}")
            
            if 'adjusted_r2' in metrics:
                print(f"Adjusted R²:       {metrics['adjusted_r2']:.4f}")
        else:
            print(f"Accuracy:          {metrics['accuracy']:.4f}")
            print(f"Precision:         {metrics['precision']:.4f}")
            print(f"Recall:            {metrics['recall']:.4f}")
            print(f"F1-Score:          {metrics['f1']:.4f}")
            
            if 'roc_auc' in metrics:
                print(f"ROC AUC:           {metrics['roc_auc']:.4f}")
                print(f"Average Precision: {metrics['average_precision']:.4f}")
        
        print(f"{'='*50}\n")
    
    def plot_predictions(self, 
                        y_true: Optional[np.ndarray] = None,
                        y_pred: Optional[np.ndarray] = None,
                        title: str = "Predictions vs Actual",
                        save_path: Optional[str] = None):
        """
        Plot predictions vs actual values (regression).
        
        Args:
            y_true: True values
            y_pred: Predicted values
            title: Plot title
            save_path: Path to save the plot
        """
        if self.task_type != 'regression':
            logger.warning("Prediction plot is for regression tasks only")
            return
        
        y_true = y_true if y_true is not None else self.true_values
        y_pred = y_pred if y_pred is not None else self.predictions
        
        if y_true is None or y_pred is None:
            logger.error("No predictions available for plotting")
            return
        
        plt.figure(figsize=(8, 6))
        plt.scatter(y_true, y_pred, alpha=0.5)
        
        # Add perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        
        plt.xlabel('True Values')
        plt.ylabel('Predictions')
        plt.title(title)
        
        # Add R² annotation
        r2 = r2_score(y_true, y_pred)
        plt.text(0.05, 0.95, f'R² = {r2:.3f}', transform=plt.gca().transAxes,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Prediction plot saved to {save_path}")
        
        plt.show()
    
    def plot_residuals(self,
                      y_true: Optional[np.ndarray] = None,
                      y_pred: Optional[np.ndarray] = None,
                      save_path: Optional[str] = None):
        """
        Plot residuals for regression tasks.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            save_path: Path to save the plot
        """
        if self.task_type != 'regression':
            logger.warning("Residual plot is for regression tasks only")
            return
        
        y_true = y_true if y_true is not None else self.true_values
        y_pred = y_pred if y_pred is not None else self.predictions
        
        residuals = y_true - y_pred
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Residuals vs predictions
        ax1.scatter(y_pred, residuals, alpha=0.5)
        ax1.axhline(y=0, color='r', linestyle='--')
        ax1.set_xlabel('Predictions')
        ax1.set_ylabel('Residuals')
        ax1.set_title('Residuals vs Predictions')
        
        # Residual distribution
        ax2.hist(residuals, bins=30, edgecolor='black')
        ax2.set_xlabel('Residuals')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Residual Distribution')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Residual plot saved to {save_path}")
        
        plt.show()
    
    def plot_confusion_matrix(self,
                            y_true: Optional[np.ndarray] = None,
                            y_pred: Optional[np.ndarray] = None,
                            class_names: Optional[List[str]] = None,
                            save_path: Optional[str] = None):
        """
        Plot confusion matrix for classification tasks.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: Names of classes
            save_path: Path to save the plot
        """
        if self.task_type != 'classification':
            logger.warning("Confusion matrix is only for classification tasks")
            return
        
        y_true = y_true if y_true is not None else self.true_values
        y_pred = y_pred if y_pred is not None else self.predictions
        
        if y_true is None or y_pred is None:
            logger.error("No predictions available for confusion matrix")
            return
        
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {save_path}")
        
        plt.show()
    
    def cross_validate(self,
                      model: Any,
                      X: Union[pd.DataFrame, np.ndarray],
                      y: Union[pd.Series, np.ndarray],
                      cv: int = 5,
                      scoring: Optional[str] = None) -> Dict[str, Any]:
        """
        Perform cross-validation and return detailed results.
        
        Args:
            model: Model to evaluate
            X: Feature matrix
            y: Target variable
            cv: Number of cross-validation folds
            scoring: Scoring metric
            
        Returns:
            Dictionary with cross-validation results
        """
        if scoring is None:
            scoring = 'r2' if self.task_type == 'regression' else 'accuracy'
        
        # Perform cross-validation
        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
        
        # Calculate statistics
        results = {
            'scores': scores,
            'mean': np.mean(scores),
            'std': np.std(scores),
            'min': np.min(scores),
            'max': np.max(scores),
            'scoring': scoring,
            'cv_folds': cv
        }
        
        logger.info(f"Cross-validation {scoring}: {results['mean']:.4f} (+/- {results['std']:.4f})")
        
        return results


class RegressionMetrics:
    """
    Specialized metrics calculator for regression tasks.
    
    This class provides a comprehensive set of regression metrics including
    advanced metrics like quantile loss, pinball loss, and directional accuracy.
    """
    
    def __init__(self):
        """Initialize the RegressionMetrics calculator."""
        self.available_metrics = {
            'mse': mean_squared_error,
            'rmse': lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error,
            'mape': mean_absolute_percentage_error,
            'r2': r2_score,
            'explained_variance': explained_variance_score,
            'median_ae': median_absolute_error,
            'max_error': max_error
        }
        
    def calculate_all_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate all available regression metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary of metric names and values
        """
        metrics = {}
        
        # Standard sklearn metrics
        for name, func in self.available_metrics.items():
            try:
                metrics[name] = func(y_true, y_pred)
            except Exception as e:
                logger.warning(f"Failed to calculate {name}: {e}")
                metrics[name] = np.nan
        
        # Custom metrics
        metrics.update(self._calculate_custom_metrics(y_true, y_pred))
        
        return metrics
    
    def _calculate_custom_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate custom regression metrics."""
        custom_metrics = {}
        
        # Normalized metrics
        if np.mean(y_true) != 0:
            custom_metrics['normalized_rmse'] = np.sqrt(mean_squared_error(y_true, y_pred)) / np.mean(y_true)
            custom_metrics['normalized_mae'] = mean_absolute_error(y_true, y_pred) / np.mean(y_true)
        
        # Directional accuracy (useful for time series)
        if len(y_true) > 1:
            true_direction = np.diff(y_true) > 0
            pred_direction = np.diff(y_pred) > 0
            custom_metrics['directional_accuracy'] = np.mean(true_direction == pred_direction)
        
        # Quantile losses
        for quantile in [0.1, 0.5, 0.9]:
            custom_metrics[f'quantile_loss_{quantile}'] = self._quantile_loss(y_true, y_pred, quantile)
        
        # Symmetric MAPE (sMAPE)
        custom_metrics['smape'] = self._symmetric_mape(y_true, y_pred)
        
        # Mean Squared Logarithmic Error (MSLE)
        if np.all(y_true >= 0) and np.all(y_pred >= 0):
            custom_metrics['msle'] = self._mean_squared_log_error(y_true, y_pred)
        
        # Coefficient of determination alternatives
        custom_metrics['r2_adjusted'] = self._adjusted_r2(y_true, y_pred)
        
        return custom_metrics
    
    def _quantile_loss(self, y_true: np.ndarray, y_pred: np.ndarray, quantile: float) -> float:
        """
        Calculate quantile loss (pinball loss).
        
        This metric is useful for evaluating quantile predictions.
        """
        errors = y_true - y_pred
        return np.mean(np.maximum(quantile * errors, (quantile - 1) * errors))
    
    def _symmetric_mape(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate Symmetric Mean Absolute Percentage Error.
        
        This metric is more robust than MAPE when dealing with values close to zero.
        """
        denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
        # Avoid division by zero
        mask = denominator != 0
        return np.mean(np.abs(y_true[mask] - y_pred[mask]) / denominator[mask]) * 100
    
    def _mean_squared_log_error(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate Mean Squared Logarithmic Error.
        
        This metric is useful when you care more about relative differences than absolute differences.
        """
        return np.mean((np.log1p(y_true) - np.log1p(y_pred)) ** 2)
    
    def _adjusted_r2(self, y_true: np.ndarray, y_pred: np.ndarray, n_features: int = 1) -> float:
        """
        Calculate adjusted R-squared.
        
        This metric accounts for the number of predictors in the model.
        """
        n = len(y_true)
        r2 = r2_score(y_true, y_pred)
        return 1 - (1 - r2) * (n - 1) / (n - n_features - 1)
    
    def evaluate_residuals(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """
        Comprehensive residual analysis.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary containing residual statistics and tests
        """
        residuals = y_true - y_pred
        
        analysis = {
            'mean': np.mean(residuals),
            'std': np.std(residuals),
            'skewness': stats.skew(residuals),
            'kurtosis': stats.kurtosis(residuals),
            'autocorrelation': self._calculate_autocorrelation(residuals)
        }
        
        # Normality tests
        _, shapiro_p = stats.shapiro(residuals) if len(residuals) < 5000 else (np.nan, np.nan)
        _, dagostino_p = stats.normaltest(residuals)
        
        analysis['normality_tests'] = {
            'shapiro_p_value': shapiro_p,
            'dagostino_p_value': dagostino_p,
            'likely_normal': shapiro_p > 0.05 if shapiro_p is not np.nan else dagostino_p > 0.05
        }
        
        # Heteroscedasticity test (simplified Breusch-Pagan)
        analysis['heteroscedasticity'] = self._test_heteroscedasticity(y_pred, residuals)
        
        return analysis
    
    def _calculate_autocorrelation(self, residuals: np.ndarray, lag: int = 1) -> float:
        """Calculate autocorrelation of residuals at specified lag."""
        if len(residuals) <= lag:
            return np.nan
        return np.corrcoef(residuals[:-lag], residuals[lag:])[0, 1]
    
    def _test_heteroscedasticity(self, predictions: np.ndarray, residuals: np.ndarray) -> Dict[str, Any]:
        """
        Simple heteroscedasticity test.
        
        Tests if residual variance changes with predicted values.
        """
        # Divide predictions into bins and check variance
        n_bins = min(10, len(predictions) // 20)
        bins = pd.qcut(predictions, n_bins, duplicates='drop')
        
        variances = []
        for bin_label in bins.cat.categories:
            bin_residuals = residuals[bins == bin_label]
            if len(bin_residuals) > 1:
                variances.append(np.var(bin_residuals))
        
        # Coefficient of variation of variances
        cv_variance = np.std(variances) / np.mean(variances) if variances else 0
        
        return {
            'cv_variance': cv_variance,
            'likely_heteroscedastic': cv_variance > 0.5,
            'bin_variances': variances
        }


class ClassificationMetrics:
    """
    Specialized metrics calculator for classification tasks.
    
    This class provides comprehensive classification metrics including
    multi-class support, probability-based metrics, and fairness metrics.
    """
    
    def __init__(self):
        """Initialize the ClassificationMetrics calculator."""
        self.threshold = 0.5  # Default threshold for binary classification
        
    def calculate_all_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                            y_proba: Optional[np.ndarray] = None,
                            average: str = 'weighted') -> Dict[str, float]:
        """
        Calculate all available classification metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities (optional)
            average: Averaging method for multi-class metrics
            
        Returns:
            Dictionary of metric names and values
        """
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
        
        # Precision, Recall, F1 with different averaging
        for avg in ['micro', 'macro', 'weighted']:
            metrics[f'precision_{avg}'] = precision_score(y_true, y_pred, average=avg, zero_division=0)
            metrics[f'recall_{avg}'] = recall_score(y_true, y_pred, average=avg, zero_division=0)
            metrics[f'f1_{avg}'] = f1_score(y_true, y_pred, average=avg, zero_division=0)
        
        # Additional metrics
        metrics['cohen_kappa'] = cohen_kappa_score(y_true, y_pred)
        metrics['matthews_corrcoef'] = matthews_corrcoef(y_true, y_pred)
        
        # Binary classification specific
        unique_classes = np.unique(y_true)
        if len(unique_classes) == 2:
            metrics.update(self._calculate_binary_metrics(y_true, y_pred, y_proba))
        
        # Multi-class specific
        if len(unique_classes) > 2 and y_proba is not None:
            metrics.update(self._calculate_multiclass_metrics(y_true, y_proba))
        
        return metrics
    
    def _calculate_binary_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                                y_proba: Optional[np.ndarray]) -> Dict[str, float]:
        """Calculate metrics specific to binary classification."""
        binary_metrics = {}
        
        # Confusion matrix elements
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        binary_metrics['true_positives'] = tp
        binary_metrics['true_negatives'] = tn
        binary_metrics['false_positives'] = fp
        binary_metrics['false_negatives'] = fn
        
        # Rates
        binary_metrics['tpr'] = tp / (tp + fn) if (tp + fn) > 0 else 0  # Sensitivity/Recall
        binary_metrics['tnr'] = tn / (tn + fp) if (tn + fp) > 0 else 0  # Specificity
        binary_metrics['ppv'] = tp / (tp + fp) if (tp + fp) > 0 else 0  # Precision
        binary_metrics['npv'] = tn / (tn + fn) if (tn + fn) > 0 else 0
        binary_metrics['fpr'] = fp / (fp + tn) if (fp + tn) > 0 else 0
        binary_metrics['fnr'] = fn / (fn + tp) if (fn + tp) > 0 else 0
        binary_metrics['fdr'] = fp / (fp + tp) if (fp + tp) > 0 else 0
        
        # Probability-based metrics
        if y_proba is not None:
            # Assuming y_proba contains probabilities for positive class
            pos_proba = y_proba[:, 1] if y_proba.ndim > 1 else y_proba
            
            binary_metrics['roc_auc'] = roc_auc_score(y_true, pos_proba)
            binary_metrics['average_precision'] = average_precision_score(y_true, pos_proba)
            binary_metrics['log_loss'] = log_loss(y_true, y_proba)
            
            # Brier score
            binary_metrics['brier_score'] = np.mean((pos_proba - y_true) ** 2)
        
        return binary_metrics
    
    def _calculate_multiclass_metrics(self, y_true: np.ndarray, 
                                    y_proba: np.ndarray) -> Dict[str, float]:
        """Calculate metrics specific to multi-class classification."""
        multiclass_metrics = {}
        
        # One-vs-rest ROC AUC
        try:
            multiclass_metrics['roc_auc_ovr'] = roc_auc_score(
                y_true, y_proba, multi_class='ovr', average='weighted'
            )
            multiclass_metrics['roc_auc_ovo'] = roc_auc_score(
                y_true, y_proba, multi_class='ovo', average='weighted'
            )
        except ValueError:
            logger.warning("Could not calculate multi-class ROC AUC")
        
        # Log loss
        multiclass_metrics['log_loss'] = log_loss(y_true, y_proba)
        
        return multiclass_metrics
    
    def calculate_per_class_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                                  class_names: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Calculate metrics for each class separately.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: Names of classes
            
        Returns:
            DataFrame with per-class metrics
        """
        report = classification_report(y_true, y_pred, output_dict=True)
        
        # Convert to DataFrame
        df = pd.DataFrame(report).transpose()
        
        # Add class names if provided
        if class_names:
            # Map numeric labels to class names
            label_map = {str(i): name for i, name in enumerate(class_names)}
            df.index = df.index.map(lambda x: label_map.get(x, x))
        
        return df
    
    def find_optimal_threshold(self, y_true: np.ndarray, y_proba: np.ndarray,
                             metric: str = 'f1') -> Tuple[float, float]:
        """
        Find optimal threshold for binary classification.
        
        Args:
            y_true: True binary labels
            y_proba: Predicted probabilities for positive class
            metric: Metric to optimize ('f1', 'accuracy', 'balanced')
            
        Returns:
            Tuple of (optimal_threshold, best_score)
        """
        thresholds = np.linspace(0, 1, 101)
        scores = []
        
        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)
            
            if metric == 'f1':
                score = f1_score(y_true, y_pred)
            elif metric == 'accuracy':
                score = accuracy_score(y_true, y_pred)
            elif metric == 'balanced':
                score = balanced_accuracy_score(y_true, y_pred)
            else:
                raise ValueError(f"Unknown metric: {metric}")
            
            scores.append(score)
        
        best_idx = np.argmax(scores)
        return thresholds[best_idx], scores[best_idx]


class CustomMetric:
    """
    Class for creating and managing custom evaluation metrics.
    
    This class allows users to define their own metrics with custom logic
    and integrate them seamlessly with the evaluation framework.
    """
    
    def __init__(self, name: str, func: Callable, greater_is_better: bool = True,
                 requires_proba: bool = False):
        """
        Initialize a custom metric.
        
        Args:
            name: Name of the metric
            func: Function that calculates the metric
            greater_is_better: Whether higher values are better
            requires_proba: Whether the metric requires probability predictions
        """
        self.name = name
        self.func = func
        self.greater_is_better = greater_is_better
        self.requires_proba = requires_proba
        self._validate()
    
    def _validate(self):
        """Validate the metric function."""
        # Test with dummy data
        try:
            y_true = np.array([0, 1, 0, 1])
            y_pred = np.array([0, 1, 1, 0])
            
            if self.requires_proba:
                y_proba = np.array([[0.9, 0.1], [0.2, 0.8], [0.4, 0.6], [0.7, 0.3]])
                result = self.func(y_true, y_proba)
            else:
                result = self.func(y_true, y_pred)
            
            if not isinstance(result, (int, float, np.number)):
                raise ValueError("Metric function must return a numeric value")
                
        except Exception as e:
            raise ValueError(f"Invalid metric function: {e}")
    
    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray,
                 y_proba: Optional[np.ndarray] = None) -> float:
        """
        Calculate the custom metric.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            y_proba: Predicted probabilities (if required)
            
        Returns:
            Metric value
        """
        if self.requires_proba and y_proba is None:
            raise ValueError(f"Metric '{self.name}' requires probability predictions")
        
        if self.requires_proba:
            return self.func(y_true, y_proba)
        else:
            return self.func(y_true, y_pred)
    
    def __repr__(self):
        return f"CustomMetric(name='{self.name}', greater_is_better={self.greater_is_better})"


class MetricCalculator:
    """
    Unified metric calculator that handles different types of metrics.
    
    This class provides a consistent interface for calculating standard metrics,
    custom metrics, and composite metrics for any machine learning task.
    """
    
    def __init__(self, task_type: str = 'regression'):
        """
        Initialize the MetricCalculator.
        
        Args:
            task_type: Type of ML task ('regression', 'classification', 'clustering')
        """
        self.task_type = task_type
        self.custom_metrics = {}
        
        # Initialize specialized calculators
        if task_type == 'regression':
            self.specialized_calculator = RegressionMetrics()
        elif task_type == 'classification':
            self.specialized_calculator = ClassificationMetrics()
        else:
            self.specialized_calculator = None
    
    def add_custom_metric(self, metric: CustomMetric):
        """
        Add a custom metric to the calculator.
        
        Args:
            metric: CustomMetric instance
        """
        self.custom_metrics[metric.name] = metric
        logger.info(f"Added custom metric: {metric.name}")
    
    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray,
                 y_proba: Optional[np.ndarray] = None,
                 metrics: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Calculate specified metrics or all available metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            y_proba: Predicted probabilities (for classification)
            metrics: List of metric names to calculate (None for all)
            
        Returns:
            Dictionary of metric names and values
        """
        results = {}
        
        # Calculate standard metrics
        if self.specialized_calculator:
            if self.task_type == 'regression':
                standard_metrics = self.specialized_calculator.calculate_all_metrics(y_true, y_pred)
            else:  # classification
                standard_metrics = self.specialized_calculator.calculate_all_metrics(
                    y_true, y_pred, y_proba
                )
            
            if metrics:
                # Filter to requested metrics
                results.update({k: v for k, v in standard_metrics.items() if k in metrics})
            else:
                results.update(standard_metrics)
        
        # Calculate custom metrics
        for name, custom_metric in self.custom_metrics.items():
            if metrics is None or name in metrics:
                try:
                    results[name] = custom_metric.calculate(y_true, y_pred, y_proba)
                except Exception as e:
                    logger.error(f"Failed to calculate custom metric '{name}': {e}")
                    results[name] = np.nan
        
        return results
    
    def create_scorer(self, metric_name: str) -> Callable:
        """
        Create a scikit-learn compatible scorer for a metric.
        
        Args:
            metric_name: Name of the metric
            
        Returns:
            Scorer function
        """
        from sklearn.metrics import make_scorer
        
        if metric_name in self.custom_metrics:
            metric = self.custom_metrics[metric_name]
            return make_scorer(
                metric.func,
                greater_is_better=metric.greater_is_better,
                needs_proba=metric.requires_proba
            )
        else:
            # Try to get from sklearn
            from sklearn import metrics as sklearn_metrics
            if hasattr(sklearn_metrics, metric_name):
                return getattr(sklearn_metrics, metric_name)
            else:
                raise ValueError(f"Unknown metric: {metric_name}")
    
    def compare_models(self, results: Dict[str, Dict[str, float]], 
                      metrics: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Compare multiple models across different metrics.
        
        Args:
            results: Dictionary mapping model names to metric dictionaries
            metrics: List of metrics to compare (None for all)
            
        Returns:
            DataFrame with models as rows and metrics as columns
        """
        df = pd.DataFrame(results).T
        
        if metrics:
            # Filter to requested metrics
            available_metrics = [m for m in metrics if m in df.columns]
            df = df[available_metrics]
        
        # Sort by first metric (descending for most metrics)
        if not df.empty:
            first_metric = df.columns[0]
            ascending = first_metric in ['mae', 'mse', 'rmse', 'log_loss']
            df = df.sort_values(first_metric, ascending=ascending)
        
        return df


class ThresholdOptimizer:
    """
    Optimizer for finding optimal classification thresholds.
    
    This class helps find the best threshold for binary classification
    based on various criteria and business constraints.
    """
    
    def __init__(self):
        """Initialize the ThresholdOptimizer."""
        self.optimization_history = []
        
    def optimize(self, y_true: np.ndarray, y_proba: np.ndarray,
                criterion: Union[str, Callable] = 'f1',
                constraints: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Find optimal threshold based on criterion and constraints.
        
        Args:
            y_true: True binary labels
            y_proba: Predicted probabilities for positive class
            criterion: Optimization criterion ('f1', 'accuracy', 'cost', or custom function)
            constraints: Dictionary of constraints (e.g., {'min_precision': 0.8})
            
        Returns:
            Dictionary with optimal threshold and performance metrics
        """
        # Define threshold candidates
        thresholds = np.unique(np.concatenate([
            np.linspace(0, 1, 101),
            y_proba  # Include actual probability values
        ]))
        thresholds.sort()
        
        best_threshold = 0.5
        best_score = -np.inf
        results = []
        
        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)
            
            # Calculate metrics
            metrics = self._calculate_threshold_metrics(y_true, y_pred, y_proba)
            
            # Check constraints
            if constraints and not self._check_constraints(metrics, constraints):
                continue
            
            # Calculate score
            if isinstance(criterion, str):
                score = self._calculate_criterion_score(criterion, metrics)
            else:
                score = criterion(y_true, y_pred)
            
            metrics['threshold'] = threshold
            metrics['score'] = score
            results.append(metrics)
            
            if score > best_score:
                best_score = score
                best_threshold = threshold
        
        # Get final metrics for best threshold
        y_pred_best = (y_proba >= best_threshold).astype(int)
        best_metrics = self._calculate_threshold_metrics(y_true, y_pred_best, y_proba)
        
        result = {
            'optimal_threshold': best_threshold,
            'optimal_score': best_score,
            'optimal_metrics': best_metrics,
            'all_results': pd.DataFrame(results),
            'criterion': criterion if isinstance(criterion, str) else 'custom',
            'constraints': constraints
        }
        
        self.optimization_history.append(result)
        
        return result
    
    def _calculate_threshold_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                                   y_proba: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive metrics for a given threshold."""
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'tp': tp,
            'tn': tn,
            'fp': fp,
            'fn': fn,
            'total_positive_predictions': tp + fp,
            'positive_rate': (tp + fp) / len(y_true)
        }
        
        return metrics
    
    def _calculate_criterion_score(self, criterion: str, metrics: Dict[str, float]) -> float:
        """Calculate score based on string criterion."""
        if criterion == 'f1':
            return metrics['f1']
        elif criterion == 'accuracy':
            return metrics['accuracy']
        elif criterion == 'balanced':
            return (metrics['precision'] + metrics['recall']) / 2
        elif criterion == 'cost':
            # Example cost function (customize based on business needs)
            # Assume false positives cost 1 unit, false negatives cost 5 units
            return -(metrics['fp'] + 5 * metrics['fn'])
        else:
            raise ValueError(f"Unknown criterion: {criterion}")
    
    def _check_constraints(self, metrics: Dict[str, float], 
                         constraints: Dict[str, float]) -> bool:
        """Check if metrics satisfy all constraints."""
        for constraint, value in constraints.items():
            if constraint.startswith('min_'):
                metric_name = constraint[4:]  # Remove 'min_' prefix
                if metric_name in metrics and metrics[metric_name] < value:
                    return False
            elif constraint.startswith('max_'):
                metric_name = constraint[4:]  # Remove 'max_' prefix
                if metric_name in metrics and metrics[metric_name] > value:
                    return False
        
        return True
    
    def plot_threshold_analysis(self, result: Optional[Dict[str, Any]] = None):
        """
        Plot comprehensive threshold analysis.
        
        Args:
            result: Optimization result (uses last if None)
        """
        if result is None:
            if not self.optimization_history:
                logger.warning("No optimization results available")
                return
            result = self.optimization_history[-1]
        
        df = result['all_results']
        optimal_threshold = result['optimal_threshold']
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot 1: Metrics vs Threshold
        ax1 = axes[0, 0]
        metrics_to_plot = ['precision', 'recall', 'f1', 'accuracy']
        for metric in metrics_to_plot:
            if metric in df.columns:
                ax1.plot(df['threshold'], df[metric], label=metric.capitalize())
        ax1.axvline(optimal_threshold, color='red', linestyle='--', label='Optimal')
        ax1.set_xlabel('Threshold')
        ax1.set_ylabel('Score')
        ax1.set_title('Metrics vs Threshold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Confusion Matrix Elements vs Threshold
        ax2 = axes[0, 1]
        for element in ['tp', 'tn', 'fp', 'fn']:
            if element in df.columns:
                ax2.plot(df['threshold'], df[element], label=element.upper())
        ax2.axvline(optimal_threshold, color='red', linestyle='--')
        ax2.set_xlabel('Threshold')
        ax2.set_ylabel('Count')
        ax2.set_title('Confusion Matrix Elements vs Threshold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Positive Prediction Rate vs Threshold
        ax3 = axes[1, 0]
        if 'positive_rate' in df.columns:
            ax3.plot(df['threshold'], df['positive_rate'])
            ax3.axvline(optimal_threshold, color='red', linestyle='--')
            ax3.set_xlabel('Threshold')
            ax3.set_ylabel('Positive Prediction Rate')
            ax3.set_title('Positive Prediction Rate vs Threshold')
            ax3.grid(True, alpha=0.3)
        
        # Plot 4: Score vs Threshold
        ax4 = axes[1, 1]
        if 'score' in df.columns:
            ax4.plot(df['threshold'], df['score'])
            ax4.axvline(optimal_threshold, color='red', linestyle='--')
            ax4.scatter([optimal_threshold], [result['optimal_score']], 
                       color='red', s=100, zorder=5)
            ax4.set_xlabel('Threshold')
            ax4.set_ylabel(f"{result['criterion'].capitalize()} Score")
            ax4.set_title(f"{result['criterion'].capitalize()} Score vs Threshold")
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def find_operating_points(self, y_true: np.ndarray, y_proba: np.ndarray,
                            targets: Dict[str, float]) -> Dict[str, Dict[str, Any]]:
        """
        Find thresholds for specific operating points.
        
        Args:
            y_true: True binary labels  
            y_proba: Predicted probabilities
            targets: Dictionary of target metrics (e.g., {'precision': 0.9, 'recall': 0.8})
            
        Returns:
            Dictionary mapping target names to threshold and achieved metrics
        """
        results = {}
        
        for target_metric, target_value in targets.items():
            best_threshold = None
            best_diff = np.inf
            
            thresholds = np.unique(y_proba)
            
            for threshold in thresholds:
                y_pred = (y_proba >= threshold).astype(int)
                metrics = self._calculate_threshold_metrics(y_true, y_pred, y_proba)
                
                if target_metric in metrics:
                    diff = abs(metrics[target_metric] - target_value)
                    
                    if diff < best_diff:
                        best_diff = diff
                        best_threshold = threshold
            
            # Get final metrics for best threshold
            if best_threshold is not None:
                y_pred_best = (y_proba >= best_threshold).astype(int)
                best_metrics = self._calculate_threshold_metrics(y_true, y_pred_best, y_proba)
                
                results[f"{target_metric}_{target_value}"] = {
                    'threshold': best_threshold,
                    'target_metric': target_metric,
                    'target_value': target_value,
                    'achieved_value': best_metrics[target_metric],
                    'all_metrics': best_metrics
                }
        
        return results