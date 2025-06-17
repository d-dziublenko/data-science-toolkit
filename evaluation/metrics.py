"""
evaluation/metrics.py
Comprehensive evaluation metrics for machine learning models.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple
from sklearn.metrics import (
    # Regression metrics
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error, explained_variance_score,
    # Classification metrics
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, log_loss,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, learning_curve
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
            y_true: True values (uses stored if None)
            y_pred: Predictions (uses stored if None)
            title: Plot title
            save_path: Path to save the plot
        """
        if self.task_type != 'regression':
            logger.warning("Prediction plot is only for regression tasks")
            return
        
        y_true = y_true if y_true is not None else self.true_values
        y_pred = y_pred if y_pred is not None else self.predictions
        
        if y_true is None or y_pred is None:
            logger.error("No predictions available to plot")
            return
        
        plt.figure(figsize=(10, 8))
        
        # Scatter plot
        plt.scatter(y_true, y_pred, alpha=0.5, s=30)
        
        # Perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        
        # Add regression line
        from sklearn.linear_model import LinearRegression
        lr = LinearRegression()
        lr.fit(y_true.reshape(-1, 1), y_pred)
        y_pred_line = lr.predict(y_true.reshape(-1, 1))
        plt.plot(y_true, y_pred_line, 'g-', lw=2, label=f'Fit: y={lr.coef_[0]:.2f}x+{lr.intercept_:.2f}')
        
        plt.xlabel('Actual Values', fontsize=12)
        plt.ylabel('Predicted Values', fontsize=12)
        plt.title(title, fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add R² to plot
        r2 = r2_score(y_true, y_pred)
        plt.text(0.05, 0.95, f'R² = {r2:.3f}', transform=plt.gca().transAxes,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        
        plt.show()
    
    def plot_residuals(self,
                      y_true: Optional[np.ndarray] = None,
                      y_pred: Optional[np.ndarray] = None,
                      save_path: Optional[str] = None):
        """
        Plot residual analysis (regression).
        
        Args:
            y_true: True values
            y_pred: Predictions
            save_path: Path to save the plot
        """
        if self.task_type != 'regression':
            logger.warning("Residual plot is only for regression tasks")
            return
        
        y_true = y_true if y_true is not None else self.true_values
        y_pred = y_pred if y_pred is not None else self.predictions
        
        if y_true is None or y_pred is None:
            logger.error("No predictions available for residual analysis")
            return
        
        residuals = y_true - y_pred
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Residuals vs Predicted
        axes[0, 0].scatter(y_pred, residuals, alpha=0.5)
        axes[0, 0].axhline(y=0, color='r', linestyle='--')
        axes[0, 0].set_xlabel('Predicted Values')
        axes[0, 0].set_ylabel('Residuals')
        axes[0, 0].set_title('Residuals vs Predicted')
        
        # Histogram of residuals
        axes[0, 1].hist(residuals, bins=30, edgecolor='black')
        axes[0, 1].set_xlabel('Residuals')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Distribution of Residuals')
        
        # Q-Q plot
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Q-Q Plot')
        
        # Residuals vs Actual
        axes[1, 1].scatter(y_true, residuals, alpha=0.5)
        axes[1, 1].axhline(y=0, color='r', linestyle='--')
        axes[1, 1].set_xlabel('Actual Values')
        axes[1, 1].set_ylabel('Residuals')
        axes[1, 1].set_title('Residuals vs Actual')
        
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
        Plot confusion matrix (classification).
        
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