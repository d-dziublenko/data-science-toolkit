"""
evaluation/visualization.py
Advanced visualization utilities for model evaluation and data analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional, Union, Any, Tuple
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.model_selection import learning_curve, validation_curve
from sklearn.inspection import permutation_importance, partial_dependence, plot_partial_dependence
import shap
import logging

logger = logging.getLogger(__name__)


class ModelVisualizer:
    """
    Advanced visualization tools for model interpretation and evaluation.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (10, 6), style: str = 'seaborn'):
        """
        Initialize the ModelVisualizer.
        
        Args:
            figsize: Default figure size
            style: Matplotlib style
        """
        self.figsize = figsize
        plt.style.use(style)
        sns.set_palette("husl")
    
    def plot_feature_importance(self,
                              model: Any,
                              feature_names: List[str],
                              importance_type: str = 'default',
                              top_n: int = 20,
                              save_path: Optional[str] = None):
        """
        Plot feature importance from tree-based models.
        
        Args:
            model: Fitted model with feature_importances_
            feature_names: List of feature names
            importance_type: Type of importance ('default', 'permutation', 'shap')
            top_n: Number of top features to show
            save_path: Path to save the plot
        """
        plt.figure(figsize=self.figsize)
        
        if importance_type == 'default':
            if not hasattr(model, 'feature_importances_'):
                logger.error("Model does not have feature_importances_ attribute")
                return
            
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1][:top_n]
            
            plt.barh(range(len(indices)), importances[indices])
            plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
            plt.xlabel('Feature Importance')
            plt.title(f'Top {top_n} Feature Importances')
        
        elif importance_type == 'permutation':
            logger.warning("Use plot_permutation_importance method for permutation importance")
            return
        
        elif importance_type == 'shap':
            logger.warning("Use plot_shap_importance method for SHAP values")
            return
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Feature importance plot saved to {save_path}")
        
        plt.show()
    
    def plot_permutation_importance(self,
                                  model: Any,
                                  X: pd.DataFrame,
                                  y: pd.Series,
                                  n_repeats: int = 10,
                                  top_n: int = 20,
                                  save_path: Optional[str] = None):
        """
        Plot permutation importance.
        
        Args:
            model: Fitted model
            X: Feature matrix
            y: Target variable
            n_repeats: Number of permutation repeats
            top_n: Number of top features to show
            save_path: Path to save the plot
        """
        # Calculate permutation importance
        result = permutation_importance(model, X, y, n_repeats=n_repeats, random_state=42)
        
        # Sort features by importance
        sorted_idx = result.importances_mean.argsort()[::-1][:top_n]
        
        plt.figure(figsize=self.figsize)
        plt.boxplot(result.importances[sorted_idx].T,
                   vert=False,
                   labels=[X.columns[i] for i in sorted_idx])
        plt.xlabel('Permutation Importance')
        plt.title(f'Top {top_n} Features by Permutation Importance')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Permutation importance plot saved to {save_path}")
        
        plt.show()
    
    def plot_shap_importance(self,
                           model: Any,
                           X: pd.DataFrame,
                           plot_type: str = 'summary',
                           save_path: Optional[str] = None):
        """
        Plot SHAP values for model interpretation.
        
        Args:
            model: Fitted model
            X: Feature matrix
            plot_type: Type of SHAP plot ('summary', 'waterfall', 'force')
            save_path: Path to save the plot
        """
        try:
            # Create SHAP explainer
            explainer = shap.Explainer(model, X)
            shap_values = explainer(X)
            
            plt.figure(figsize=self.figsize)
            
            if plot_type == 'summary':
                shap.summary_plot(shap_values, X, show=False)
            elif plot_type == 'waterfall':
                shap.waterfall_plot(shap_values[0], show=False)
            elif plot_type == 'force':
                shap.force_plot(explainer.expected_value, shap_values.values[0], 
                               X.iloc[0], matplotlib=True, show=False)
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"SHAP plot saved to {save_path}")
            
            plt.show()
            
        except Exception as e:
            logger.error(f"Failed to create SHAP plot: {e}")
    
    def plot_learning_curves(self,
                           model: Any,
                           X: pd.DataFrame,
                           y: pd.Series,
                           cv: int = 5,
                           train_sizes: np.ndarray = np.linspace(0.1, 1.0, 10),
                           save_path: Optional[str] = None):
        """
        Plot learning curves to diagnose over/underfitting.
        
        Args:
            model: Model to evaluate
            X: Feature matrix
            y: Target variable
            cv: Number of cross-validation folds
            train_sizes: Training set sizes to evaluate
            save_path: Path to save the plot
        """
        train_sizes, train_scores, val_scores = learning_curve(
            model, X, y, cv=cv, train_sizes=train_sizes,
            scoring='r2' if hasattr(y, 'dtype') and np.issubdtype(y.dtype, np.number) else 'accuracy'
        )
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        plt.figure(figsize=self.figsize)
        
        plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training score')
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                        alpha=0.1, color='blue')
        
        plt.plot(train_sizes, val_mean, 'o-', color='green', label='Validation score')
        plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std,
                        alpha=0.1, color='green')
        
        plt.xlabel('Training Set Size')
        plt.ylabel('Score')
        plt.title('Learning Curves')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Learning curves saved to {save_path}")
        
        plt.show()
    
    def plot_validation_curves(self,
                             model: Any,
                             X: pd.DataFrame,
                             y: pd.Series,
                             param_name: str,
                             param_range: np.ndarray,
                             cv: int = 5,
                             save_path: Optional[str] = None):
        """
        Plot validation curves for hyperparameter tuning.
        
        Args:
            model: Model to evaluate
            X: Feature matrix
            y: Target variable
            param_name: Name of the parameter to vary
            param_range: Range of parameter values
            cv: Number of cross-validation folds
            save_path: Path to save the plot
        """
        train_scores, val_scores = validation_curve(
            model, X, y, param_name=param_name, param_range=param_range,
            cv=cv, scoring='r2' if hasattr(y, 'dtype') and np.issubdtype(y.dtype, np.number) else 'accuracy'
        )
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        plt.figure(figsize=self.figsize)
        
        plt.plot(param_range, train_mean, 'o-', color='blue', label='Training score')
        plt.fill_between(param_range, train_mean - train_std, train_mean + train_std,
                        alpha=0.1, color='blue')
        
        plt.plot(param_range, val_mean, 'o-', color='green', label='Validation score')
        plt.fill_between(param_range, val_mean - val_std, val_mean + val_std,
                        alpha=0.1, color='green')
        
        plt.xlabel(param_name)
        plt.ylabel('Score')
        plt.title(f'Validation Curves - {param_name}')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Validation curves saved to {save_path}")
        
        plt.show()
    
    def plot_roc_curves(self,
                       models: Dict[str, Any],
                       X: pd.DataFrame,
                       y: pd.Series,
                       save_path: Optional[str] = None):
        """
        Plot ROC curves for multiple models (binary classification).
        
        Args:
            models: Dictionary of model names and fitted models
            X: Feature matrix
            y: True labels
            save_path: Path to save the plot
        """
        plt.figure(figsize=self.figsize)
        
        for name, model in models.items():
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X)[:, 1]
            elif hasattr(model, 'decision_function'):
                y_proba = model.decision_function(X)
            else:
                logger.warning(f"Model {name} does not support probability predictions")
                continue
            
            fpr, tpr, _ = roc_curve(y, y_proba)
            roc_auc = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"ROC curves saved to {save_path}")
        
        plt.show()
    
    def plot_precision_recall_curves(self,
                                   models: Dict[str, Any],
                                   X: pd.DataFrame,
                                   y: pd.Series,
                                   save_path: Optional[str] = None):
        """
        Plot precision-recall curves for multiple models (binary classification).
        
        Args:
            models: Dictionary of model names and fitted models
            X: Feature matrix
            y: True labels
            save_path: Path to save the plot
        """
        plt.figure(figsize=self.figsize)
        
        for name, model in models.items():
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X)[:, 1]
            elif hasattr(model, 'decision_function'):
                y_proba = model.decision_function(X)
            else:
                logger.warning(f"Model {name} does not support probability predictions")
                continue
            
            precision, recall, _ = precision_recall_curve(y, y_proba)
            pr_auc = auc(recall, precision)
            
            plt.plot(recall, precision, label=f'{name} (AUC = {pr_auc:.3f})')
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves')
        plt.legend(loc='lower left')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Precision-recall curves saved to {save_path}")
        
        plt.show()


class DataVisualizer:
    """
    Visualization tools for exploratory data analysis.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (10, 6), style: str = 'seaborn'):
        """
        Initialize the DataVisualizer.
        
        Args:
            figsize: Default figure size
            style: Matplotlib style
        """
        self.figsize = figsize
        plt.style.use(style)
        sns.set_palette("husl")
    
    def plot_correlation_matrix(self,
                              data: pd.DataFrame,
                              method: str = 'pearson',
                              annot: bool = True,
                              mask_upper: bool = True,
                              save_path: Optional[str] = None):
        """
        Plot correlation matrix heatmap.
        
        Args:
            data: DataFrame with features
            method: Correlation method ('pearson', 'spearman', 'kendall')
            annot: Whether to annotate cells with values
            mask_upper: Whether to mask upper triangle
            save_path: Path to save the plot
        """
        # Calculate correlation matrix
        corr = data.corr(method=method)
        
        # Create mask for upper triangle if requested
        mask = None
        if mask_upper:
            mask = np.triu(np.ones_like(corr, dtype=bool))
        
        plt.figure(figsize=(max(10, len(corr.columns) * 0.5), 
                           max(8, len(corr.columns) * 0.4)))
        
        sns.heatmap(corr, mask=mask, annot=annot, cmap='coolwarm',
                   center=0, square=True, linewidths=0.5,
                   cbar_kws={"shrink": 0.8})
        
        plt.title(f'{method.capitalize()} Correlation Matrix')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Correlation matrix saved to {save_path}")
        
        plt.show()
    
    def plot_distributions(self,
                         data: pd.DataFrame,
                         columns: Optional[List[str]] = None,
                         n_cols: int = 3,
                         save_path: Optional[str] = None):
        """
        Plot distributions of multiple features.
        
        Args:
            data: DataFrame with features
            columns: Columns to plot (None plots all numeric)
            n_cols: Number of columns in subplot grid
            save_path: Path to save the plot
        """
        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns.tolist()
        
        n_features = len(columns)
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, 
                                figsize=(n_cols * 4, n_rows * 3))
        axes = axes.flatten() if n_features > 1 else [axes]
        
        for i, col in enumerate(columns):
            ax = axes[i]
            
            # Plot histogram with KDE
            data[col].hist(ax=ax, bins=30, alpha=0.7, density=True)
            data[col].plot(kind='density', ax=ax, color='red', linewidth=2)
            
            ax.set_title(col)
            ax.set_xlabel('')
            
            # Add statistics
            mean = data[col].mean()
            median = data[col].median()
            ax.axvline(mean, color='green', linestyle='--', alpha=0.8, label=f'Mean: {mean:.2f}')
            ax.axvline(median, color='orange', linestyle='--', alpha=0.8, label=f'Median: {median:.2f}')
            ax.legend(fontsize=8)
        
        # Hide empty subplots
        for i in range(n_features, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Distribution plots saved to {save_path}")
        
        plt.show()