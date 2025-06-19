"""
evaluation/visualization.py
Advanced visualization utilities for model evaluation and data analysis.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.inspection import (partial_dependence, permutation_importance,
                                plot_partial_dependence)
from sklearn.metrics import (auc, confusion_matrix, mean_absolute_error,
                             mean_squared_error, precision_recall_curve,
                             r2_score, roc_curve)
from sklearn.model_selection import learning_curve, validation_curve

logger = logging.getLogger(__name__)


@dataclass
class PlotConfig:
    """Configuration for plotting parameters."""

    figsize: Tuple[int, int] = (10, 6)
    style: str = "seaborn"
    palette: str = "husl"
    dpi: int = 300
    font_size: int = 12
    title_size: int = 14
    label_size: int = 12
    tick_size: int = 10
    line_width: float = 2.0
    marker_size: float = 8.0
    alpha: float = 0.7
    grid: bool = True
    grid_alpha: float = 0.3
    save_format: str = "png"
    tight_layout: bool = True

    def apply(self):
        """Apply the configuration to matplotlib."""
        plt.style.use(self.style)
        sns.set_palette(self.palette)
        plt.rcParams.update(
            {
                "font.size": self.font_size,
                "axes.titlesize": self.title_size,
                "axes.labelsize": self.label_size,
                "xtick.labelsize": self.tick_size,
                "ytick.labelsize": self.tick_size,
                "lines.linewidth": self.line_width,
                "lines.markersize": self.marker_size,
                "figure.dpi": self.dpi,
            }
        )


class ModelVisualizer:
    """
    Advanced visualization tools for model interpretation and evaluation.
    """

    def __init__(self, figsize: Tuple[int, int] = (10, 6), style: str = "seaborn"):
        """
        Initialize the ModelVisualizer.

        Args:
            figsize: Default figure size
            style: Matplotlib style
        """
        self.figsize = figsize
        plt.style.use(style)
        sns.set_palette("husl")

    def plot_feature_importance(
        self,
        model: Any,
        feature_names: List[str],
        importance_type: str = "default",
        top_n: int = 20,
        save_path: Optional[str] = None,
    ):
        """
        Plot feature importance from tree-based models.

        Args:
            model: Trained model with feature_importances_ attribute
            feature_names: List of feature names
            importance_type: Type of importance ('default', 'permutation', 'shap')
            top_n: Number of top features to display
            save_path: Path to save the plot
        """
        if importance_type == "default":
            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
            else:
                raise ValueError("Model doesn't have feature_importances_ attribute")
        else:
            raise NotImplementedError(
                f"Importance type {importance_type} not implemented in base method"
            )

        # Create DataFrame and sort
        importance_df = (
            pd.DataFrame({"feature": feature_names, "importance": importances})
            .sort_values("importance", ascending=False)
            .head(top_n)
        )

        plt.figure(figsize=self.figsize)

        # Create horizontal bar plot
        plt.barh(importance_df["feature"], importance_df["importance"])
        plt.xlabel("Importance")
        plt.title(f"Top {top_n} Feature Importance")
        plt.gca().invert_yaxis()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Feature importance plot saved to {save_path}")

        plt.show()

    def plot_learning_curves(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series,
        cv: int = 5,
        train_sizes: np.ndarray = np.linspace(0.1, 1.0, 10),
        save_path: Optional[str] = None,
    ):
        """
        Plot learning curves to diagnose bias/variance.

        Args:
            model: Model to evaluate
            X: Feature matrix
            y: Target variable
            cv: Number of cross-validation folds
            train_sizes: Training set sizes to evaluate
            save_path: Path to save the plot
        """
        train_sizes, train_scores, val_scores = learning_curve(
            model,
            X,
            y,
            cv=cv,
            train_sizes=train_sizes,
            scoring=(
                "r2"
                if hasattr(y, "dtype") and np.issubdtype(y.dtype, np.number)
                else "accuracy"
            ),
        )

        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)

        plt.figure(figsize=self.figsize)

        plt.plot(train_sizes, train_mean, "o-", color="blue", label="Training score")
        plt.fill_between(
            train_sizes,
            train_mean - train_std,
            train_mean + train_std,
            alpha=0.1,
            color="blue",
        )

        plt.plot(train_sizes, val_mean, "o-", color="green", label="Validation score")
        plt.fill_between(
            train_sizes,
            val_mean - val_std,
            val_mean + val_std,
            alpha=0.1,
            color="green",
        )

        plt.xlabel("Training Set Size")
        plt.ylabel("Score")
        plt.title("Learning Curves")
        plt.legend(loc="best")
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Learning curves saved to {save_path}")

        plt.show()

    def plot_validation_curves(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series,
        param_name: str,
        param_range: np.ndarray,
        cv: int = 5,
        save_path: Optional[str] = None,
    ):
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
            model,
            X,
            y,
            param_name=param_name,
            param_range=param_range,
            cv=cv,
            scoring=(
                "r2"
                if hasattr(y, "dtype") and np.issubdtype(y.dtype, np.number)
                else "accuracy"
            ),
        )

        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)

        plt.figure(figsize=self.figsize)

        plt.plot(param_range, train_mean, "o-", color="blue", label="Training score")
        plt.fill_between(
            param_range,
            train_mean - train_std,
            train_mean + train_std,
            alpha=0.1,
            color="blue",
        )

        plt.plot(param_range, val_mean, "o-", color="green", label="Validation score")
        plt.fill_between(
            param_range,
            val_mean - val_std,
            val_mean + val_std,
            alpha=0.1,
            color="green",
        )

        plt.xlabel(param_name)
        plt.ylabel("Score")
        plt.title(f"Validation Curves - {param_name}")
        plt.legend(loc="best")
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Validation curves saved to {save_path}")

        plt.show()

    def plot_roc_curves(
        self,
        models: Dict[str, Any],
        X: pd.DataFrame,
        y: pd.Series,
        save_path: Optional[str] = None,
    ):
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
            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(X)[:, 1]
            elif hasattr(model, "decision_function"):
                y_proba = model.decision_function(X)
            else:
                logger.warning(f"Model {name} does not support probability predictions")
                continue

            fpr, tpr, _ = roc_curve(y, y_proba)
            roc_auc = auc(fpr, tpr)

            plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})")

        plt.plot([0, 1], [0, 1], "k--", label="Random")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curves")
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"ROC curves saved to {save_path}")

        plt.show()

    def plot_precision_recall_curves(
        self,
        models: Dict[str, Any],
        X: pd.DataFrame,
        y: pd.Series,
        save_path: Optional[str] = None,
    ):
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
            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(X)[:, 1]
            elif hasattr(model, "decision_function"):
                y_proba = model.decision_function(X)
            else:
                logger.warning(f"Model {name} does not support probability predictions")
                continue

            precision, recall, _ = precision_recall_curve(y, y_proba)
            pr_auc = auc(recall, precision)

            plt.plot(recall, precision, label=f"{name} (AUC = {pr_auc:.3f})")

        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curves")
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Precision-recall curves saved to {save_path}")

        plt.show()


class DataVisualizer:
    """
    Visualization tools for exploratory data analysis.
    """

    def __init__(self, figsize: Tuple[int, int] = (10, 6), style: str = "seaborn"):
        """
        Initialize the DataVisualizer.

        Args:
            figsize: Default figure size
            style: Matplotlib style
        """
        self.figsize = figsize
        plt.style.use(style)
        sns.set_palette("husl")

    def plot_correlation_matrix(
        self,
        data: pd.DataFrame,
        method: str = "pearson",
        annot: bool = True,
        mask_upper: bool = True,
        save_path: Optional[str] = None,
    ):
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

        plt.figure(
            figsize=(max(10, len(corr.columns) * 0.5), max(8, len(corr.columns) * 0.4))
        )

        sns.heatmap(
            corr,
            mask=mask,
            annot=annot,
            cmap="coolwarm",
            center=0,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8},
        )

        plt.title(f"{method.capitalize()} Correlation Matrix")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Correlation matrix saved to {save_path}")

        plt.show()

    def plot_distributions(
        self,
        data: pd.DataFrame,
        columns: Optional[List[str]] = None,
        n_cols: int = 3,
        save_path: Optional[str] = None,
    ):
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

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3))
        axes = axes.flatten() if n_features > 1 else [axes]

        for i, col in enumerate(columns):
            ax = axes[i]

            # Plot histogram with KDE
            data[col].hist(ax=ax, bins=30, alpha=0.7, density=True)
            data[col].plot(kind="density", ax=ax, color="red", linewidth=2)

            ax.set_title(col)
            ax.set_xlabel("")

            # Add statistics
            mean = data[col].mean()
            median = data[col].median()
            ax.axvline(
                mean,
                color="green",
                linestyle="--",
                alpha=0.8,
                label=f"Mean: {mean:.2f}",
            )
            ax.axvline(
                median,
                color="orange",
                linestyle="--",
                alpha=0.8,
                label=f"Median: {median:.2f}",
            )
            ax.legend(fontsize=8)

        # Hide empty subplots
        for i in range(n_features, len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Distribution plots saved to {save_path}")

        plt.show()


class PerformancePlotter:
    """Specialized plotter for model performance metrics."""

    def __init__(self, config: Optional[PlotConfig] = None):
        """
        Initialize the PerformancePlotter.

        Args:
            config: Plot configuration
        """
        self.config = config or PlotConfig()
        self.config.apply()

    def plot_metric_comparison(
        self,
        metrics_dict: Dict[str, Dict[str, float]],
        metric_names: Optional[List[str]] = None,
        save_path: Optional[str] = None,
    ):
        """
        Plot comparison of metrics across models.

        Args:
            metrics_dict: Dictionary {model_name: {metric_name: value}}
            metric_names: Specific metrics to plot (None plots all)
            save_path: Path to save the plot
        """
        # Convert to DataFrame for easier plotting
        df = pd.DataFrame(metrics_dict).T

        if metric_names:
            df = df[metric_names]

        # Create figure
        fig, ax = plt.subplots(figsize=self.config.figsize)

        # Create grouped bar plot
        x = np.arange(len(df.index))
        width = 0.8 / len(df.columns)

        for i, col in enumerate(df.columns):
            offset = (i - len(df.columns) / 2 + 0.5) * width
            ax.bar(x + offset, df[col], width, label=col, alpha=self.config.alpha)

        ax.set_xlabel("Models")
        ax.set_ylabel("Score")
        ax.set_title("Model Performance Comparison")
        ax.set_xticks(x)
        ax.set_xticklabels(df.index, rotation=45, ha="right")
        ax.legend()
        ax.grid(self.config.grid, alpha=self.config.grid_alpha)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.config.dpi, format=self.config.save_format)
            logger.info(f"Metric comparison saved to {save_path}")

        plt.show()

    def plot_cross_validation_scores(
        self, cv_scores: Dict[str, np.ndarray], save_path: Optional[str] = None
    ):
        """
        Plot cross-validation scores for multiple models.

        Args:
            cv_scores: Dictionary {model_name: array of CV scores}
            save_path: Path to save the plot
        """
        fig, ax = plt.subplots(figsize=self.config.figsize)

        # Prepare data for box plot
        data = []
        labels = []
        for model_name, scores in cv_scores.items():
            data.append(scores)
            labels.append(model_name)

        # Create box plot
        bp = ax.boxplot(data, labels=labels, patch_artist=True)

        # Customize box plot
        colors = plt.cm.Set3(np.linspace(0, 1, len(data)))
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(self.config.alpha)

        ax.set_ylabel("Score")
        ax.set_title("Cross-Validation Score Distribution")
        ax.grid(self.config.grid, alpha=self.config.grid_alpha)

        # Add mean values
        for i, (model_name, scores) in enumerate(cv_scores.items()):
            mean_score = np.mean(scores)
            ax.text(
                i + 1,
                mean_score,
                f"{mean_score:.3f}",
                ha="center",
                va="bottom",
                fontsize=self.config.font_size - 2,
            )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.config.dpi, format=self.config.save_format)
            logger.info(f"CV scores plot saved to {save_path}")

        plt.show()


class FeatureImportancePlotter:
    """Specialized plotter for feature importance visualization."""

    def __init__(self, config: Optional[PlotConfig] = None):
        """
        Initialize the FeatureImportancePlotter.

        Args:
            config: Plot configuration
        """
        self.config = config or PlotConfig()
        self.config.apply()

    def plot_importance_comparison(
        self,
        importance_dict: Dict[str, pd.DataFrame],
        top_n: int = 20,
        save_path: Optional[str] = None,
    ):
        """
        Compare feature importance across multiple models.

        Args:
            importance_dict: {model_name: DataFrame with 'feature' and 'importance'}
            top_n: Number of top features to show
            save_path: Path to save the plot
        """
        # Get union of top features across all models
        all_features = set()
        for df in importance_dict.values():
            top_features = df.nlargest(top_n, "importance")["feature"].tolist()
            all_features.update(top_features)

        all_features = sorted(list(all_features))

        # Create comparison matrix
        comparison_data = pd.DataFrame(index=all_features)
        for model_name, df in importance_dict.items():
            feature_importance = df.set_index("feature")["importance"]
            comparison_data[model_name] = feature_importance.reindex(
                all_features, fill_value=0
            )

        # Plot heatmap
        plt.figure(figsize=(self.config.figsize[0], len(all_features) * 0.3))

        sns.heatmap(
            comparison_data,
            cmap="YlOrRd",
            annot=True,
            fmt=".3f",
            cbar_kws={"label": "Importance"},
        )

        plt.title("Feature Importance Comparison")
        plt.xlabel("Models")
        plt.ylabel("Features")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.config.dpi, format=self.config.save_format)
            logger.info(f"Feature importance comparison saved to {save_path}")

        plt.show()

    def plot_cumulative_importance(
        self,
        feature_importance: pd.DataFrame,
        threshold: float = 0.95,
        save_path: Optional[str] = None,
    ):
        """
        Plot cumulative feature importance.

        Args:
            feature_importance: DataFrame with 'feature' and 'importance' columns
            threshold: Cumulative importance threshold to highlight
            save_path: Path to save the plot
        """
        # Sort by importance
        sorted_importance = feature_importance.sort_values(
            "importance", ascending=False
        ).copy()
        sorted_importance["cumulative_importance"] = sorted_importance[
            "importance"
        ].cumsum()
        sorted_importance["cumulative_importance"] /= sorted_importance[
            "importance"
        ].sum()

        # Find number of features for threshold
        n_features_threshold = (
            sorted_importance["cumulative_importance"] <= threshold
        ).sum()

        # Plot
        fig, ax = plt.subplots(figsize=self.config.figsize)

        x = range(len(sorted_importance))
        ax.plot(
            x,
            sorted_importance["cumulative_importance"],
            linewidth=self.config.line_width,
            color="blue",
        )

        # Add threshold line
        ax.axhline(
            y=threshold, color="red", linestyle="--", label=f"{threshold:.0%} threshold"
        )
        ax.axvline(
            x=n_features_threshold,
            color="red",
            linestyle="--",
            label=f"{n_features_threshold} features",
        )

        ax.set_xlabel("Number of Features")
        ax.set_ylabel("Cumulative Importance")
        ax.set_title("Cumulative Feature Importance")
        ax.legend()
        ax.grid(self.config.grid, alpha=self.config.grid_alpha)

        # Add annotation
        ax.annotate(
            f"{n_features_threshold} features explain {threshold:.0%} of importance",
            xy=(n_features_threshold, threshold),
            xytext=(n_features_threshold + 5, threshold - 0.1),
            arrowprops=dict(arrowstyle="->", color="red", alpha=0.7),
        )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.config.dpi, format=self.config.save_format)
            logger.info(f"Cumulative importance plot saved to {save_path}")

        plt.show()


class ResidualPlotter:
    """Specialized plotter for regression residual analysis."""

    def __init__(self, config: Optional[PlotConfig] = None):
        """
        Initialize the ResidualPlotter.

        Args:
            config: Plot configuration
        """
        self.config = config or PlotConfig()
        self.config.apply()

    def plot_residuals(
        self, y_true: np.ndarray, y_pred: np.ndarray, save_path: Optional[str] = None
    ):
        """
        Create comprehensive residual plots.

        Args:
            y_true: True values
            y_pred: Predicted values
            save_path: Path to save the plot
        """
        residuals = y_true - y_pred

        fig, axes = plt.subplots(
            2, 2, figsize=(self.config.figsize[0] * 1.5, self.config.figsize[1] * 1.5)
        )

        # 1. Residuals vs Predicted
        ax = axes[0, 0]
        ax.scatter(
            y_pred, residuals, alpha=self.config.alpha, s=self.config.marker_size
        )
        ax.axhline(y=0, color="red", linestyle="--")
        ax.set_xlabel("Predicted Values")
        ax.set_ylabel("Residuals")
        ax.set_title("Residuals vs Predicted")
        ax.grid(self.config.grid, alpha=self.config.grid_alpha)

        # 2. Q-Q plot
        ax = axes[0, 1]
        from scipy import stats

        stats.probplot(residuals, dist="norm", plot=ax)
        ax.set_title("Q-Q Plot")
        ax.grid(self.config.grid, alpha=self.config.grid_alpha)

        # 3. Histogram of residuals
        ax = axes[1, 0]
        ax.hist(
            residuals, bins=30, alpha=self.config.alpha, density=True, label="Residuals"
        )

        # Overlay normal distribution
        x = np.linspace(residuals.min(), residuals.max(), 100)
        ax.plot(
            x,
            stats.norm.pdf(x, residuals.mean(), residuals.std()),
            "r-",
            linewidth=self.config.line_width,
            label="Normal",
        )
        ax.set_xlabel("Residuals")
        ax.set_ylabel("Density")
        ax.set_title("Residual Distribution")
        ax.legend()
        ax.grid(self.config.grid, alpha=self.config.grid_alpha)

        # 4. Scale-Location plot
        ax = axes[1, 1]
        standardized_residuals = residuals / residuals.std()
        ax.scatter(
            y_pred,
            np.sqrt(np.abs(standardized_residuals)),
            alpha=self.config.alpha,
            s=self.config.marker_size,
        )

        # Add smooth line
        from scipy.interpolate import UnivariateSpline

        sorted_pred = np.sort(y_pred)
        sorted_std_res = np.sqrt(np.abs(standardized_residuals[np.argsort(y_pred)]))
        try:
            spline = UnivariateSpline(sorted_pred, sorted_std_res, s=len(y_pred))
            ax.plot(
                sorted_pred, spline(sorted_pred), "r-", linewidth=self.config.line_width
            )
        except:
            pass

        ax.set_xlabel("Predicted Values")
        ax.set_ylabel("√|Standardized Residuals|")
        ax.set_title("Scale-Location")
        ax.grid(self.config.grid, alpha=self.config.grid_alpha)

        plt.suptitle("Residual Analysis", fontsize=self.config.title_size + 2)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.config.dpi, format=self.config.save_format)
            logger.info(f"Residual plots saved to {save_path}")

        plt.show()

    def plot_residuals_vs_features(
        self,
        residuals: np.ndarray,
        features: pd.DataFrame,
        feature_names: Optional[List[str]] = None,
        n_cols: int = 3,
        save_path: Optional[str] = None,
    ):
        """
        Plot residuals against individual features.

        Args:
            residuals: Array of residuals
            features: DataFrame of features
            feature_names: Specific features to plot (None plots all)
            n_cols: Number of columns in subplot grid
            save_path: Path to save the plot
        """
        if feature_names is None:
            feature_names = features.columns.tolist()

        n_features = len(feature_names)
        n_rows = (n_features + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4))
        axes = axes.flatten() if n_features > 1 else [axes]

        for i, feature in enumerate(feature_names):
            ax = axes[i]
            ax.scatter(
                features[feature],
                residuals,
                alpha=self.config.alpha,
                s=self.config.marker_size,
            )
            ax.axhline(y=0, color="red", linestyle="--")
            ax.set_xlabel(feature)
            ax.set_ylabel("Residuals")
            ax.set_title(f"Residuals vs {feature}")
            ax.grid(self.config.grid, alpha=self.config.grid_alpha)

        # Hide empty subplots
        for i in range(n_features, len(axes)):
            axes[i].set_visible(False)

        plt.suptitle("Residuals vs Features", fontsize=self.config.title_size + 2)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.config.dpi, format=self.config.save_format)
            logger.info(f"Residuals vs features plot saved to {save_path}")

        plt.show()


class ConfusionMatrixPlotter:
    """Specialized plotter for confusion matrix visualization."""

    def __init__(self, config: Optional[PlotConfig] = None):
        """
        Initialize the ConfusionMatrixPlotter.

        Args:
            config: Plot configuration
        """
        self.config = config or PlotConfig()
        self.config.apply()

    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        labels: Optional[List[str]] = None,
        normalize: bool = False,
        save_path: Optional[str] = None,
    ):
        """
        Plot confusion matrix with annotations.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            labels: Class labels
            normalize: Whether to normalize the matrix
            save_path: Path to save the plot
        """
        cm = confusion_matrix(y_true, y_pred)

        if normalize:
            cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
            fmt = ".2f"
        else:
            fmt = "d"

        plt.figure(figsize=self.config.figsize)

        sns.heatmap(
            cm,
            annot=True,
            fmt=fmt,
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
            cbar_kws={"label": "Count" if not normalize else "Proportion"},
        )

        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title("Confusion Matrix" + (" (Normalized)" if normalize else ""))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.config.dpi, format=self.config.save_format)
            logger.info(f"Confusion matrix saved to {save_path}")

        plt.show()

    def plot_multi_class_roc(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        class_names: List[str],
        save_path: Optional[str] = None,
    ):
        """
        Plot ROC curves for multi-class classification.

        Args:
            y_true: True labels (one-hot encoded)
            y_proba: Predicted probabilities
            class_names: Names of classes
            save_path: Path to save the plot
        """
        n_classes = len(class_names)

        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_proba[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute micro-average ROC curve
        fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_proba.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        # Plot
        plt.figure(figsize=self.config.figsize)

        # Plot ROC curve for each class
        colors = plt.cm.Set1(np.linspace(0, 1, n_classes))
        for i, color in enumerate(colors):
            plt.plot(
                fpr[i],
                tpr[i],
                color=color,
                linewidth=self.config.line_width,
                label=f"{class_names[i]} (AUC = {roc_auc[i]:.3f})",
            )

        # Plot micro-average ROC curve
        plt.plot(
            fpr["micro"],
            tpr["micro"],
            label=f'Micro-average (AUC = {roc_auc["micro"]:.3f})',
            color="deeppink",
            linestyle=":",
            linewidth=self.config.line_width + 1,
        )

        plt.plot([0, 1], [0, 1], "k--", label="Random")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Multi-class ROC Curves")
        plt.legend(loc="lower right")
        plt.grid(self.config.grid, alpha=self.config.grid_alpha)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.config.dpi, format=self.config.save_format)
            logger.info(f"Multi-class ROC curves saved to {save_path}")

        plt.show()
