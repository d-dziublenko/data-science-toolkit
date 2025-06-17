"""
core/feature_engineering.py
Advanced feature engineering and selection utilities.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Union, Tuple, Any
from sklearn.feature_selection import (
    SelectKBest, SelectFromModel, RFE, RFECV,
    mutual_info_regression, mutual_info_classif,
    f_regression, f_classif, chi2
)
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.decomposition import PCA
import scipy.stats as stats
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from collections import defaultdict
import logging
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class FeatureSelector:
    """
    Comprehensive feature selection utilities.
    
    Provides various methods for selecting the most relevant features
    including statistical tests, model-based selection, and dimensionality reduction.
    """
    
    def __init__(self, task_type: str = 'regression'):
        """
        Initialize the FeatureSelector.
        
        Args:
            task_type: Type of ML task ('regression' or 'classification')
        """
        self.task_type = task_type
        self.selected_features = None
        self.feature_scores = None
        self.selection_method = None
    
    def select_by_correlation(self,
                            X: pd.DataFrame,
                            y: pd.Series,
                            threshold: float = 0.1,
                            method: str = 'pearson') -> List[str]:
        """
        Select features based on correlation with target.
        
        Args:
            X: Feature DataFrame
            y: Target variable
            threshold: Minimum absolute correlation threshold
            method: Correlation method ('pearson', 'spearman', 'kendall')
            
        Returns:
            List of selected feature names
        """
        correlations = {}
        
        for col in X.columns:
            if method == 'pearson':
                corr, _ = stats.pearsonr(X[col], y)
            elif method == 'spearman':
                corr, _ = stats.spearmanr(X[col], y)
            elif method == 'kendall':
                corr, _ = stats.kendalltau(X[col], y)
            else:
                raise ValueError(f"Unknown correlation method: {method}")
            
            correlations[col] = abs(corr)
        
        # Sort by correlation
        sorted_corrs = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
        
        # Select features above threshold
        selected = [feat for feat, corr in sorted_corrs if corr >= threshold]
        
        self.selected_features = selected
        self.feature_scores = correlations
        self.selection_method = f'correlation_{method}'
        
        logger.info(f"Selected {len(selected)} features with |correlation| >= {threshold}")
        return selected
    
    def select_by_mutual_information(self,
                                   X: pd.DataFrame,
                                   y: pd.Series,
                                   n_features: int = 10) -> List[str]:
        """
        Select features based on mutual information.
        
        Args:
            X: Feature DataFrame
            y: Target variable
            n_features: Number of features to select
            
        Returns:
            List of selected feature names
        """
        if self.task_type == 'regression':
            mi_scores = mutual_info_regression(X, y, random_state=42)
        else:
            mi_scores = mutual_info_classif(X, y, random_state=42)
        
        # Create score dictionary
        mi_dict = dict(zip(X.columns, mi_scores))
        
        # Sort and select top features
        sorted_features = sorted(mi_dict.items(), key=lambda x: x[1], reverse=True)
        selected = [feat for feat, _ in sorted_features[:n_features]]
        
        self.selected_features = selected
        self.feature_scores = mi_dict
        self.selection_method = 'mutual_information'
        
        logger.info(f"Selected top {n_features} features by mutual information")
        return selected
    
    def select_by_model_importance(self,
                                 X: pd.DataFrame,
                                 y: pd.Series,
                                 model: Optional[Any] = None,
                                 n_features: int = 10) -> List[str]:
        """
        Select features based on model feature importance.
        
        Args:
            X: Feature DataFrame
            y: Target variable
            model: Model with feature_importances_ attribute (default: RandomForest)
            n_features: Number of features to select
            
        Returns:
            List of selected feature names
        """
        if model is None:
            if self.task_type == 'regression':
                model = RandomForestRegressor(n_estimators=100, random_state=42)
            else:
                model = RandomForestClassifier(n_estimators=100, random_state=42)
        
        # Fit model
        model.fit(X, y)
        
        # Get feature importances
        importances = model.feature_importances_
        importance_dict = dict(zip(X.columns, importances))
        
        # Sort and select top features
        sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        selected = [feat for feat, _ in sorted_features[:n_features]]
        
        self.selected_features = selected
        self.feature_scores = importance_dict
        self.selection_method = 'model_importance'
        
        logger.info(f"Selected top {n_features} features by model importance")
        return selected
    
    def select_by_rfe(self,
                     X: pd.DataFrame,
                     y: pd.Series,
                     n_features: int = 10,
                     model: Optional[Any] = None,
                     cv: Optional[int] = None) -> List[str]:
        """
        Select features using Recursive Feature Elimination.
        
        Args:
            X: Feature DataFrame
            y: Target variable
            n_features: Number of features to select
            model: Base model for RFE
            cv: Number of cross-validation folds (None for no CV)
            
        Returns:
            List of selected feature names
        """
        if model is None:
            if self.task_type == 'regression':
                model = RandomForestRegressor(n_estimators=50, random_state=42)
            else:
                model = RandomForestClassifier(n_estimators=50, random_state=42)
        
        if cv:
            selector = RFECV(model, min_features_to_select=n_features, cv=cv)
        else:
            selector = RFE(model, n_features_to_select=n_features)
        
        selector.fit(X, y)
        
        # Get selected features
        selected = X.columns[selector.support_].tolist()
        
        self.selected_features = selected
        self.selection_method = 'rfe'
        
        logger.info(f"Selected {len(selected)} features using RFE")
        return selected
    
    def remove_multicollinear_features(self,
                                     X: pd.DataFrame,
                                     threshold: float = 0.8) -> List[str]:
        """
        Remove highly correlated features to reduce multicollinearity.
        
        Args:
            X: Feature DataFrame
            threshold: Correlation threshold for removal
            
        Returns:
            List of features to keep
        """
        # Calculate correlation matrix
        corr_matrix = X.corr().abs()
        
        # Create mask for upper triangle
        upper_tri = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool_)
        )
        
        # Find features to drop
        to_drop = [column for column in upper_tri.columns 
                  if any(upper_tri[column] > threshold)]
        
        # Keep features
        selected = [col for col in X.columns if col not in to_drop]
        
        logger.info(f"Removed {len(to_drop)} multicollinear features (threshold={threshold})")
        return selected


class FeatureEngineer:
    """
    Advanced feature engineering techniques.
    
    Provides methods for creating complex features including
    aggregations, time-based features, and domain-specific transformations.
    """
    
    @staticmethod
    def create_aggregated_features(data: pd.DataFrame,
                                 group_cols: List[str],
                                 agg_cols: List[str],
                                 agg_funcs: List[str] = ['mean', 'std', 'min', 'max']) -> pd.DataFrame:
        """
        Create aggregated features based on grouping.
        
        Args:
            data: Input DataFrame
            group_cols: Columns to group by
            agg_cols: Columns to aggregate
            agg_funcs: Aggregation functions to apply
            
        Returns:
            DataFrame with aggregated features
        """
        result = data.copy()
        
        for group_col in group_cols:
            for agg_col in agg_cols:
                for func in agg_funcs:
                    feature_name = f"{agg_col}_{func}_by_{group_col}"
                    
                    # Calculate aggregation
                    agg_values = data.groupby(group_col)[agg_col].transform(func)
                    result[feature_name] = agg_values
                    
                    logger.info(f"Created aggregated feature: {feature_name}")
        
        return result
    
    @staticmethod
    def create_lag_features(data: pd.DataFrame,
                          columns: List[str],
                          lags: List[int],
                          group_col: Optional[str] = None) -> pd.DataFrame:
        """
        Create lag features for time series data.
        
        Args:
            data: Input DataFrame (should be sorted by time)
            columns: Columns to create lags for
            lags: List of lag values
            group_col: Column to group by (for panel data)
            
        Returns:
            DataFrame with lag features
        """
        result = data.copy()
        
        for col in columns:
            for lag in lags:
                feature_name = f"{col}_lag_{lag}"
                
                if group_col:
                    result[feature_name] = result.groupby(group_col)[col].shift(lag)
                else:
                    result[feature_name] = result[col].shift(lag)
                
                logger.info(f"Created lag feature: {feature_name}")
        
        return result
    
    @staticmethod
    def create_rolling_features(data: pd.DataFrame,
                              columns: List[str],
                              windows: List[int],
                              funcs: List[str] = ['mean', 'std'],
                              group_col: Optional[str] = None) -> pd.DataFrame:
        """
        Create rolling window features.
        
        Args:
            data: Input DataFrame (should be sorted by time)
            columns: Columns to create rolling features for
            windows: List of window sizes
            funcs: Functions to apply to rolling windows
            group_col: Column to group by (for panel data)
            
        Returns:
            DataFrame with rolling features
        """
        result = data.copy()
        
        for col in columns:
            for window in windows:
                for func in funcs:
                    feature_name = f"{col}_rolling_{window}_{func}"
                    
                    if group_col:
                        result[feature_name] = result.groupby(group_col)[col].transform(
                            lambda x: x.rolling(window, min_periods=1).agg(func)
                        )
                    else:
                        result[feature_name] = result[col].rolling(window, min_periods=1).agg(func)
                    
                    logger.info(f"Created rolling feature: {feature_name}")
        
        return result
    
    @staticmethod
    def create_date_features(data: pd.DataFrame,
                           date_column: str) -> pd.DataFrame:
        """
        Extract features from datetime column.
        
        Args:
            data: Input DataFrame
            date_column: Name of datetime column
            
        Returns:
            DataFrame with date features
        """
        result = data.copy()
        
        # Ensure datetime type
        result[date_column] = pd.to_datetime(result[date_column])
        
        # Extract various date components
        result[f"{date_column}_year"] = result[date_column].dt.year
        result[f"{date_column}_month"] = result[date_column].dt.month
        result[f"{date_column}_day"] = result[date_column].dt.day
        result[f"{date_column}_dayofweek"] = result[date_column].dt.dayofweek
        result[f"{date_column}_dayofyear"] = result[date_column].dt.dayofyear
        result[f"{date_column}_weekofyear"] = result[date_column].dt.isocalendar().week
        result[f"{date_column}_quarter"] = result[date_column].dt.quarter
        result[f"{date_column}_is_weekend"] = result[date_column].dt.dayofweek.isin([5, 6]).astype(int)
        result[f"{date_column}_is_month_start"] = result[date_column].dt.is_month_start.astype(int)
        result[f"{date_column}_is_month_end"] = result[date_column].dt.is_month_end.astype(int)
        
        # Cyclical encoding for periodic features
        result[f"{date_column}_month_sin"] = np.sin(2 * np.pi * result[f"{date_column}_month"] / 12)
        result[f"{date_column}_month_cos"] = np.cos(2 * np.pi * result[f"{date_column}_month"] / 12)
        result[f"{date_column}_day_sin"] = np.sin(2 * np.pi * result[f"{date_column}_day"] / 31)
        result[f"{date_column}_day_cos"] = np.cos(2 * np.pi * result[f"{date_column}_day"] / 31)
        
        logger.info(f"Created date features from {date_column}")
        return result