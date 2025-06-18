"""
core/feature_engineering.py
Advanced feature engineering and selection utilities.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Union, Tuple, Any, Callable
from sklearn.feature_selection import (
    SelectKBest, SelectFromModel, RFE, RFECV,
    mutual_info_regression, mutual_info_classif,
    f_regression, f_classif, chi2
)
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures as SklearnPolynomialFeatures
import scipy.stats as stats
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from collections import defaultdict
import itertools
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
            threshold: Minimum correlation threshold
            method: Correlation method ('pearson', 'spearman', 'kendall')
            
        Returns:
            List of selected feature names
        """
        correlations = {}
        
        for col in X.columns:
            if X[col].dtype in ['float64', 'int64']:
                if method == 'pearson':
                    corr = X[col].corr(y)
                elif method == 'spearman':
                    corr = X[col].corr(y, method='spearman')
                elif method == 'kendall':
                    corr = X[col].corr(y, method='kendall')
                else:
                    raise ValueError(f"Unknown correlation method: {method}")
                
                correlations[col] = abs(corr)
        
        # Select features above threshold
        selected = [col for col, corr in correlations.items() if corr >= threshold]
        
        self.selected_features = selected
        self.feature_scores = correlations
        self.selection_method = f'correlation_{method}'
        
        logger.info(f"Selected {len(selected)} features with correlation >= {threshold}")
        return selected
    
    def select_by_mutual_information(self,
                                   X: pd.DataFrame,
                                   y: pd.Series,
                                   n_features: int = 10) -> List[str]:
        """
        Select features based on mutual information with target.
        
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


class InteractionFeatures:
    """
    Specialized class for creating interaction features between variables.
    
    This class provides comprehensive methods for generating interaction features
    that capture relationships between different variables in the dataset.
    """
    
    def __init__(self):
        """Initialize the InteractionFeatures transformer."""
        self.interaction_pairs = []
        self.created_features = []
        self.feature_importance = {}
        
    def create_pairwise_interactions(self, 
                                   data: pd.DataFrame,
                                   columns: Optional[List[str]] = None,
                                   max_pairs: Optional[int] = None,
                                   include_squares: bool = False) -> pd.DataFrame:
        """
        Create all pairwise interaction features between specified columns.
        
        Args:
            data: Input DataFrame
            columns: Columns to create interactions between (None uses all numeric)
            max_pairs: Maximum number of pairs to create (None creates all)
            include_squares: Whether to include squared terms (col^2)
            
        Returns:
            DataFrame with interaction features added
        """
        result = data.copy()
        
        # Select columns if not specified
        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns.tolist()
        
        # Generate all pairs
        if include_squares:
            # Include self-interactions (squares)
            pairs = list(itertools.combinations_with_replacement(columns, 2))
        else:
            # Only different columns
            pairs = list(itertools.combinations(columns, 2))
        
        # Limit pairs if specified
        if max_pairs is not None and len(pairs) > max_pairs:
            # Select most important pairs based on correlation with existing features
            pairs = self._select_top_pairs(data, pairs, max_pairs)
        
        # Create interaction features
        for col1, col2 in pairs:
            if col1 == col2:
                # Squared term
                feature_name = f"{col1}_squared"
                result[feature_name] = data[col1] ** 2
            else:
                # Interaction term
                feature_name = f"{col1}_x_{col2}"
                result[feature_name] = data[col1] * data[col2]
            
            self.created_features.append(feature_name)
            self.interaction_pairs.append((col1, col2))
            
        logger.info(f"Created {len(self.created_features)} interaction features")
        return result
    
    def create_targeted_interactions(self,
                                   data: pd.DataFrame,
                                   target_col: str,
                                   feature_cols: Optional[List[str]] = None,
                                   top_n: int = 10) -> pd.DataFrame:
        """
        Create interactions between features most correlated with a target column.
        
        Args:
            data: Input DataFrame
            target_col: Target column to optimize interactions for
            feature_cols: Feature columns to consider (None uses all numeric)
            top_n: Number of top correlated features to use
            
        Returns:
            DataFrame with targeted interaction features
        """
        result = data.copy()
        
        if feature_cols is None:
            feature_cols = [col for col in data.select_dtypes(include=[np.number]).columns 
                          if col != target_col]
        
        # Calculate correlations with target
        correlations = {}
        for col in feature_cols:
            correlations[col] = abs(data[col].corr(data[target_col]))
        
        # Select top correlated features
        top_features = sorted(correlations.items(), key=lambda x: x[1], reverse=True)[:top_n]
        top_feature_names = [feat[0] for feat in top_features]
        
        # Create interactions between top features
        for i, col1 in enumerate(top_feature_names):
            for col2 in top_feature_names[i+1:]:
                feature_name = f"{col1}_x_{col2}"
                result[feature_name] = data[col1] * data[col2]
                self.created_features.append(feature_name)
                self.interaction_pairs.append((col1, col2))
        
        logger.info(f"Created {len(self.created_features)} targeted interaction features")
        return result
    
    def create_ratio_interactions(self,
                                data: pd.DataFrame,
                                numerator_cols: List[str],
                                denominator_cols: List[str],
                                min_denominator: float = 1e-8) -> pd.DataFrame:
        """
        Create ratio features between specified columns.
        
        Args:
            data: Input DataFrame
            numerator_cols: Columns to use as numerators
            denominator_cols: Columns to use as denominators
            min_denominator: Minimum value for denominator to avoid division by zero
            
        Returns:
            DataFrame with ratio features
        """
        result = data.copy()
        
        for num_col in numerator_cols:
            for den_col in denominator_cols:
                if num_col != den_col and num_col in data.columns and den_col in data.columns:
                    feature_name = f"{num_col}_div_{den_col}"
                    # Safe division with minimum denominator
                    result[feature_name] = data[num_col] / (data[den_col] + min_denominator)
                    self.created_features.append(feature_name)
                    self.interaction_pairs.append((num_col, den_col))
        
        logger.info(f"Created {len(self.created_features)} ratio features")
        return result
    
    def create_arithmetic_interactions(self,
                                     data: pd.DataFrame,
                                     columns: List[str],
                                     operations: List[str] = ['multiply', 'add', 'subtract']) -> pd.DataFrame:
        """
        Create interaction features using various arithmetic operations.
        
        Args:
            data: Input DataFrame
            columns: Columns to create interactions between
            operations: List of operations ('multiply', 'add', 'subtract', 'divide')
            
        Returns:
            DataFrame with arithmetic interaction features
        """
        result = data.copy()
        
        # Define operation functions
        ops = {
            'multiply': lambda x, y: x * y,
            'add': lambda x, y: x + y,
            'subtract': lambda x, y: x - y,
            'divide': lambda x, y: x / (y + 1e-8)
        }
        
        # Create interactions for each operation
        for operation in operations:
            if operation not in ops:
                logger.warning(f"Unknown operation: {operation}")
                continue
                
            op_func = ops[operation]
            
            for i, col1 in enumerate(columns):
                for col2 in columns[i+1:]:
                    if operation == 'divide' and col1 == col2:
                        continue  # Skip x/x
                        
                    feature_name = f"{col1}_{operation}_{col2}"
                    result[feature_name] = op_func(data[col1], data[col2])
                    self.created_features.append(feature_name)
                    self.interaction_pairs.append((col1, col2))
        
        logger.info(f"Created {len(self.created_features)} arithmetic interaction features")
        return result
    
    def _select_top_pairs(self, data: pd.DataFrame, pairs: List[Tuple[str, str]], 
                         max_pairs: int) -> List[Tuple[str, str]]:
        """
        Select the most important pairs based on feature variance and correlation.
        
        Args:
            data: Input DataFrame
            pairs: List of all possible pairs
            max_pairs: Maximum number of pairs to select
            
        Returns:
            List of selected pairs
        """
        # Score each pair based on the product of their standard deviations
        # (higher variance features tend to create more informative interactions)
        pair_scores = []
        
        for col1, col2 in pairs:
            score = data[col1].std() * data[col2].std()
            pair_scores.append(((col1, col2), score))
        
        # Sort by score and select top pairs
        pair_scores.sort(key=lambda x: x[1], reverse=True)
        selected_pairs = [pair for pair, _ in pair_scores[:max_pairs]]
        
        return selected_pairs
    
    def evaluate_interactions(self, X: pd.DataFrame, y: pd.Series, 
                            method: str = 'mutual_info') -> pd.DataFrame:
        """
        Evaluate the importance of created interaction features.
        
        Args:
            X: DataFrame with interaction features
            y: Target variable
            method: Evaluation method ('mutual_info', 'correlation', 'model_importance')
            
        Returns:
            DataFrame with feature importance scores
        """
        if not self.created_features:
            logger.warning("No interaction features have been created yet")
            return pd.DataFrame()
        
        # Get only the interaction features
        interaction_data = X[self.created_features]
        
        if method == 'mutual_info':
            # Use mutual information
            scores = mutual_info_regression(interaction_data, y)
            self.feature_importance = dict(zip(self.created_features, scores))
            
        elif method == 'correlation':
            # Use correlation
            correlations = {}
            for col in self.created_features:
                correlations[col] = abs(X[col].corr(y))
            self.feature_importance = correlations
            
        elif method == 'model_importance':
            # Use random forest importance
            from sklearn.ensemble import RandomForestRegressor
            model = RandomForestRegressor(n_estimators=50, random_state=42)
            model.fit(interaction_data, y)
            self.feature_importance = dict(zip(self.created_features, model.feature_importances_))
        
        # Create summary DataFrame
        importance_df = pd.DataFrame([
            {'feature': feat, 'importance': score, 'pair': str(pair)}
            for (feat, pair), score in zip(
                zip(self.created_features, self.interaction_pairs),
                self.feature_importance.values()
            )
        ]).sort_values('importance', ascending=False)
        
        return importance_df


class PolynomialFeatures:
    """
    Advanced polynomial feature generation with optimization capabilities.
    
    This class extends sklearn's PolynomialFeatures with additional functionality
    for feature selection, sparse handling, and performance optimization.
    """
    
    def __init__(self, degree: int = 2, 
                 interaction_only: bool = False,
                 include_bias: bool = False):
        """
        Initialize the PolynomialFeatures transformer.
        
        Args:
            degree: Maximum degree of polynomial features
            interaction_only: If True, only interaction features are produced
            include_bias: If True, include a bias column (all ones)
        """
        self.degree = degree
        self.interaction_only = interaction_only
        self.include_bias = include_bias
        self.sklearn_poly = None
        self.feature_names = None
        self.n_input_features = None
        self.n_output_features = None
        
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'PolynomialFeatures':
        """
        Fit the polynomial features transformer.
        
        Args:
            X: Input features
            y: Target variable (optional, used for feature selection)
            
        Returns:
            Self
        """
        # Initialize sklearn PolynomialFeatures
        self.sklearn_poly = SklearnPolynomialFeatures(
            degree=self.degree,
            interaction_only=self.interaction_only,
            include_bias=self.include_bias
        )
        
        # Fit the transformer
        self.sklearn_poly.fit(X)
        
        # Store metadata
        self.n_input_features = X.shape[1]
        self.n_output_features = self.sklearn_poly.n_output_features_
        
        # Get feature names
        if hasattr(X, 'columns'):
            self.feature_names = self.sklearn_poly.get_feature_names_out(X.columns)
        else:
            self.feature_names = self.sklearn_poly.get_feature_names_out()
        
        logger.info(f"Fitted polynomial features: {self.n_input_features} -> {self.n_output_features} features")
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform features to polynomial features.
        
        Args:
            X: Input features
            
        Returns:
            DataFrame with polynomial features
        """
        if self.sklearn_poly is None:
            raise ValueError("PolynomialFeatures must be fitted before transform")
        
        # Transform features
        poly_array = self.sklearn_poly.transform(X)
        
        # Create DataFrame with proper column names
        poly_df = pd.DataFrame(
            poly_array,
            index=X.index if hasattr(X, 'index') else None,
            columns=self.feature_names
        )
        
        return poly_df
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Fit and transform in one step.
        
        Args:
            X: Input features
            y: Target variable (optional)
            
        Returns:
            DataFrame with polynomial features
        """
        self.fit(X, y)
        return self.transform(X)
    
    def create_selected_polynomials(self, 
                                  X: pd.DataFrame,
                                  y: pd.Series,
                                  max_features: int = 50,
                                  selection_method: str = 'mutual_info') -> pd.DataFrame:
        """
        Create polynomial features and select the most important ones.
        
        Args:
            X: Input features
            y: Target variable
            max_features: Maximum number of features to keep
            selection_method: Method for feature selection
            
        Returns:
            DataFrame with selected polynomial features
        """
        # Create all polynomial features
        poly_df = self.fit_transform(X)
        
        # Remove the bias term and original features for selection
        if self.include_bias:
            poly_df = poly_df.iloc[:, 1:]  # Remove bias
        
        # Get only the new polynomial features (not original features)
        new_features = poly_df.columns[self.n_input_features:]
        poly_new = poly_df[new_features]
        
        # Select features based on importance
        if selection_method == 'mutual_info':
            mi_scores = mutual_info_regression(poly_new, y)
            score_dict = dict(zip(new_features, mi_scores))
        elif selection_method == 'correlation':
            score_dict = {}
            for col in new_features:
                score_dict[col] = abs(poly_new[col].corr(y))
        elif selection_method == 'variance':
            score_dict = {}
            for col in new_features:
                score_dict[col] = poly_new[col].var()
        else:
            raise ValueError(f"Unknown selection method: {selection_method}")
        
        # Sort features by score
        sorted_features = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)
        
        # Select top features
        n_select = min(max_features - self.n_input_features, len(sorted_features))
        selected_new = [feat for feat, _ in sorted_features[:n_select]]
        
        # Combine original features with selected polynomial features
        result = pd.concat([X, poly_df[selected_new]], axis=1)
        
        logger.info(f"Selected {len(selected_new)} polynomial features from {len(new_features)} candidates")
        
        return result
    
    def create_interaction_only_features(self, X: pd.DataFrame, 
                                       min_degree: int = 2,
                                       max_degree: int = 3) -> pd.DataFrame:
        """
        Create only interaction features (no powers of single features).
        
        Args:
            X: Input features
            min_degree: Minimum interaction degree
            max_degree: Maximum interaction degree
            
        Returns:
            DataFrame with interaction features
        """
        result = X.copy()
        feature_names = X.columns.tolist()
        
        # Generate interactions for each degree
        for degree in range(min_degree, max_degree + 1):
            # Get all combinations of features for this degree
            for combo in itertools.combinations(range(len(feature_names)), degree):
                # Create feature name
                feature_name = '_x_'.join([feature_names[i] for i in combo])
                
                # Calculate interaction
                interaction = X.iloc[:, combo[0]].copy()
                for idx in combo[1:]:
                    interaction *= X.iloc[:, idx]
                
                result[feature_name] = interaction
        
        logger.info(f"Created {result.shape[1] - X.shape[1]} interaction features")
        return result
    
    def get_feature_powers(self) -> Dict[str, List[int]]:
        """
        Get the powers of each input feature in the polynomial features.
        
        Returns:
            Dictionary mapping feature names to their powers in each polynomial
        """
        if self.sklearn_poly is None:
            raise ValueError("PolynomialFeatures must be fitted first")
        
        powers = self.sklearn_poly.powers_
        feature_powers = {}
        
        for i, feature_name in enumerate(self.feature_names):
            feature_powers[feature_name] = powers[i].tolist()
        
        return feature_powers
    
    def create_custom_polynomials(self, X: pd.DataFrame,
                                custom_terms: List[Dict[str, int]]) -> pd.DataFrame:
        """
        Create custom polynomial terms specified by the user.
        
        Args:
            X: Input features
            custom_terms: List of dictionaries specifying powers for each feature
                         e.g., [{'x1': 2, 'x2': 1}, {'x1': 0, 'x2': 3}]
            
        Returns:
            DataFrame with custom polynomial features
        """
        result = X.copy()
        
        for term_spec in custom_terms:
            # Create feature name
            term_parts = []
            for feat, power in term_spec.items():
                if power > 0:
                    if power == 1:
                        term_parts.append(feat)
                    else:
                        term_parts.append(f"{feat}^{power}")
            
            feature_name = '_'.join(term_parts) if term_parts else 'bias'
            
            # Calculate the polynomial term
            term_value = pd.Series(1.0, index=X.index)
            for feat, power in term_spec.items():
                if feat in X.columns and power > 0:
                    term_value *= X[feat] ** power
            
            result[feature_name] = term_value
        
        logger.info(f"Created {len(custom_terms)} custom polynomial features")
        return result