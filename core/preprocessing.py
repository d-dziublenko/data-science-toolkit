"""
core/preprocessing.py
Comprehensive data preprocessing utilities for machine learning pipelines.
"""

import numpy as np
import pandas as pd
from typing import Union, Optional, List, Dict, Any, Tuple, Callable
from scipy import stats
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler, QuantileTransformer,
    PowerTransformer, LabelEncoder, OneHotEncoder
)
from sklearn.impute import SimpleImputer, KNNImputer
import logging

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Main class for data preprocessing operations.
    
    This class provides a unified interface for various preprocessing tasks
    including scaling, encoding, imputation, and outlier handling.
    """
    
    def __init__(self):
        """Initialize the DataPreprocessor with tracking of transformations."""
        self.transformers = {}
        self.encoded_columns = {}
        self.transformation_history = []
    
    def scale_features(self,
                      data: pd.DataFrame,
                      columns: Optional[List[str]] = None,
                      method: str = 'standard',
                      **kwargs) -> pd.DataFrame:
        """
        Scale numerical features using various methods.
        
        Args:
            data: Input DataFrame
            columns: Columns to scale (None scales all numeric columns)
            method: Scaling method ('standard', 'minmax', 'robust', 'quantile', 'power')
            **kwargs: Additional arguments for the scaler
            
        Returns:
            DataFrame with scaled features
        """
        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns.tolist()
        
        # Select appropriate scaler
        scalers = {
            'standard': StandardScaler,
            'minmax': MinMaxScaler,
            'robust': RobustScaler,
            'quantile': QuantileTransformer,
            'power': PowerTransformer
        }
        
        if method not in scalers:
            raise ValueError(f"Unknown scaling method: {method}")
        
        scaler = scalers[method](**kwargs)
        
        # Fit and transform
        data_scaled = data.copy()
        data_scaled[columns] = scaler.fit_transform(data[columns])
        
        # Store transformer for later use
        self.transformers[f'scaler_{method}'] = scaler
        self.transformation_history.append({
            'type': 'scaling',
            'method': method,
            'columns': columns
        })
        
        logger.info(f"Applied {method} scaling to {len(columns)} columns")
        return data_scaled
    
    def encode_categorical(self,
                          data: pd.DataFrame,
                          columns: Optional[List[str]] = None,
                          method: str = 'onehot',
                          max_categories: int = 10) -> pd.DataFrame:
        """
        Encode categorical variables.
        
        Args:
            data: Input DataFrame
            columns: Categorical columns to encode (None encodes all object columns)
            method: Encoding method ('onehot', 'label', 'target', 'ordinal')
            max_categories: Maximum number of categories for one-hot encoding
            
        Returns:
            DataFrame with encoded features
        """
        if columns is None:
            columns = data.select_dtypes(include=['object', 'category']).columns.tolist()
        
        data_encoded = data.copy()
        
        for col in columns:
            n_unique = data[col].nunique()
            
            if method == 'onehot' and n_unique <= max_categories:
                # One-hot encoding
                dummies = pd.get_dummies(data[col], prefix=col, drop_first=True)
                data_encoded = pd.concat([data_encoded.drop(columns=[col]), dummies], axis=1)
                self.encoded_columns[col] = dummies.columns.tolist()
                
            elif method == 'label' or (method == 'onehot' and n_unique > max_categories):
                # Label encoding
                le = LabelEncoder()
                data_encoded[col] = le.fit_transform(data[col].astype(str))
                self.transformers[f'labelencoder_{col}'] = le
                
            else:
                logger.warning(f"Skipping encoding for column {col}")
        
        self.transformation_history.append({
            'type': 'encoding',
            'method': method,
            'columns': columns
        })
        
        logger.info(f"Encoded {len(columns)} categorical columns using {method}")
        return data_encoded
    
    def handle_missing_values(self,
                            data: pd.DataFrame,
                            method: str = 'drop',
                            threshold: float = 0.5,
                            fill_value: Any = None,
                            strategy: str = 'mean') -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        
        Args:
            data: Input DataFrame
            method: Method to handle missing values ('drop', 'fill', 'impute')
            threshold: Threshold for dropping columns with missing values
            fill_value: Value to fill missing values (for method='fill')
            strategy: Strategy for imputation ('mean', 'median', 'most_frequent', 'knn')
            
        Returns:
            DataFrame with handled missing values
        """
        data_clean = data.copy()
        
        # Calculate missing value statistics
        missing_stats = data_clean.isnull().sum() / len(data_clean)
        
        if method == 'drop':
            # Drop columns with too many missing values
            cols_to_drop = missing_stats[missing_stats > threshold].index
            if len(cols_to_drop) > 0:
                logger.info(f"Dropping {len(cols_to_drop)} columns with >{threshold*100}% missing")
                data_clean = data_clean.drop(columns=cols_to_drop)
            
            # Drop rows with any missing values
            n_rows_before = len(data_clean)
            data_clean = data_clean.dropna()
            logger.info(f"Dropped {n_rows_before - len(data_clean)} rows with missing values")
            
        elif method == 'fill':
            if fill_value is not None:
                data_clean = data_clean.fillna(fill_value)
            else:
                # Fill with appropriate values based on data type
                for col in data_clean.columns:
                    if data_clean[col].dtype in ['float64', 'int64']:
                        data_clean[col].fillna(data_clean[col].mean(), inplace=True)
                    else:
                        data_clean[col].fillna(data_clean[col].mode()[0], inplace=True)
            
        elif method == 'impute':
            if strategy == 'knn':
                # KNN imputation
                numeric_cols = data_clean.select_dtypes(include=[np.number]).columns
                imputer = KNNImputer(n_neighbors=5)
                data_clean[numeric_cols] = imputer.fit_transform(data_clean[numeric_cols])
            else:
                # Simple imputation
                numeric_cols = data_clean.select_dtypes(include=[np.number]).columns
                cat_cols = data_clean.select_dtypes(include=['object', 'category']).columns
                
                if len(numeric_cols) > 0:
                    num_imputer = SimpleImputer(strategy=strategy)
                    data_clean[numeric_cols] = num_imputer.fit_transform(data_clean[numeric_cols])
                
                if len(cat_cols) > 0:
                    cat_imputer = SimpleImputer(strategy='most_frequent')
                    data_clean[cat_cols] = cat_imputer.fit_transform(data_clean[cat_cols])
        
        self.transformation_history.append({
            'type': 'missing_values',
            'method': method,
            'threshold': threshold
        })
        
        return data_clean
    
    def remove_outliers(self,
                       data: pd.DataFrame,
                       columns: Optional[List[str]] = None,
                       method: str = 'iqr',
                       threshold: float = 1.5) -> pd.DataFrame:
        """
        Remove outliers from numerical features.
        
        Args:
            data: Input DataFrame
            columns: Columns to check for outliers (None checks all numeric)
            method: Outlier detection method ('iqr', 'zscore', 'isolation')
            threshold: Threshold for outlier detection
            
        Returns:
            DataFrame with outliers removed
        """
        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns.tolist()
        
        data_clean = data.copy()
        n_outliers = 0
        
        if method == 'iqr':
            for col in columns:
                Q1 = data_clean[col].quantile(0.25)
                Q3 = data_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                outliers = (data_clean[col] < lower_bound) | (data_clean[col] > upper_bound)
                n_outliers += outliers.sum()
                data_clean = data_clean[~outliers]
        
        elif method == 'zscore':
            for col in columns:
                z_scores = np.abs(stats.zscore(data_clean[col]))
                outliers = z_scores > threshold
                n_outliers += outliers.sum()
                data_clean = data_clean[~outliers]
        
        elif method == 'isolation':
            from sklearn.ensemble import IsolationForest
            
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            outliers = iso_forest.fit_predict(data_clean[columns]) == -1
            n_outliers = outliers.sum()
            data_clean = data_clean[~outliers]
        
        logger.info(f"Removed {n_outliers} outliers using {method} method")
        
        self.transformation_history.append({
            'type': 'outlier_removal',
            'method': method,
            'threshold': threshold,
            'n_removed': n_outliers
        })
        
        return data_clean
    
    def transform_target(self,
                        y: Union[pd.Series, np.ndarray],
                        method: str = 'none') -> Tuple[np.ndarray, Optional[Any]]:
        """
        Transform target variable for better model performance.
        
        Args:
            y: Target variable
            method: Transformation method ('log', 'sqrt', 'boxcox', 'yeo-johnson')
            
        Returns:
            Tuple of (transformed_y, transformer_object)
        """
        y_array = np.array(y).reshape(-1, 1)
        
        if method == 'none':
            return y_array.ravel(), None
        
        elif method == 'log':
            # Ensure positive values
            if np.any(y_array <= 0):
                logger.warning("Log transform requires positive values, adding constant")
                y_array = y_array - y_array.min() + 1
            return np.log(y_array).ravel(), {'method': 'log', 'offset': y_array.min() - 1}
        
        elif method == 'sqrt':
            if np.any(y_array < 0):
                logger.warning("Square root transform requires non-negative values")
                y_array = y_array - y_array.min()
            return np.sqrt(y_array).ravel(), {'method': 'sqrt', 'offset': y_array.min()}
        
        elif method in ['boxcox', 'yeo-johnson']:
            transformer = PowerTransformer(method=method)
            y_transformed = transformer.fit_transform(y_array)
            return y_transformed.ravel(), transformer
        
        else:
            raise ValueError(f"Unknown transformation method: {method}")
    
    def inverse_transform_target(self,
                               y_transformed: np.ndarray,
                               transformer: Any) -> np.ndarray:
        """
        Inverse transform the target variable.
        
        Args:
            y_transformed: Transformed target values
            transformer: Transformer object or parameters
            
        Returns:
            Original scale target values
        """
        if transformer is None:
            return y_transformed
        
        y_transformed = y_transformed.reshape(-1, 1)
        
        if isinstance(transformer, dict):
            if transformer['method'] == 'log':
                return np.exp(y_transformed).ravel() + transformer.get('offset', 0)
            elif transformer['method'] == 'sqrt':
                return np.power(y_transformed, 2).ravel() + transformer.get('offset', 0)
        else:
            # Sklearn transformer
            return transformer.inverse_transform(y_transformed).ravel()


class FeatureTransformer:
    """
    Advanced feature transformation utilities.
    
    Provides methods for creating polynomial features, interaction terms,
    and custom transformations.
    """
    
    @staticmethod
    def create_polynomial_features(data: pd.DataFrame,
                                 columns: List[str],
                                 degree: int = 2,
                                 include_bias: bool = False) -> pd.DataFrame:
        """
        Create polynomial features from existing features.
        
        Args:
            data: Input DataFrame
            columns: Columns to create polynomial features from
            degree: Maximum polynomial degree
            include_bias: Whether to include bias term
            
        Returns:
            DataFrame with polynomial features added
        """
        from sklearn.preprocessing import PolynomialFeatures
        
        poly = PolynomialFeatures(degree=degree, include_bias=include_bias)
        poly_features = poly.fit_transform(data[columns])
        
        # Get feature names
        feature_names = poly.get_feature_names_out(columns)
        
        # Create new DataFrame
        poly_df = pd.DataFrame(poly_features, columns=feature_names, index=data.index)
        
        # Combine with original data
        result = pd.concat([data, poly_df.iloc[:, len(columns):]], axis=1)
        
        logger.info(f"Created {poly_df.shape[1] - len(columns)} polynomial features")
        return result
    
    @staticmethod
    def create_interaction_features(data: pd.DataFrame,
                                  column_pairs: List[Tuple[str, str]]) -> pd.DataFrame:
        """
        Create interaction features between specified column pairs.
        
        Args:
            data: Input DataFrame
            column_pairs: List of column pairs to create interactions for
            
        Returns:
            DataFrame with interaction features added
        """
        result = data.copy()
        
        for col1, col2 in column_pairs:
            if col1 in data.columns and col2 in data.columns:
                interaction_name = f"{col1}_x_{col2}"
                result[interaction_name] = data[col1] * data[col2]
                logger.info(f"Created interaction feature: {interaction_name}")
            else:
                logger.warning(f"Columns {col1} or {col2} not found in data")
        
        return result
    
    @staticmethod
    def create_ratio_features(data: pd.DataFrame,
                            numerator_cols: List[str],
                            denominator_cols: List[str]) -> pd.DataFrame:
        """
        Create ratio features between specified columns.
        
        Args:
            data: Input DataFrame
            numerator_cols: List of numerator columns
            denominator_cols: List of denominator columns
            
        Returns:
            DataFrame with ratio features added
        """
        result = data.copy()
        
        for num_col in numerator_cols:
            for den_col in denominator_cols:
                if num_col != den_col and num_col in data.columns and den_col in data.columns:
                    ratio_name = f"{num_col}_div_{den_col}"
                    # Avoid division by zero
                    result[ratio_name] = data[num_col] / (data[den_col] + 1e-8)
                    logger.info(f"Created ratio feature: {ratio_name}")
        
        return result
    
    @staticmethod
    def apply_custom_transformations(data: pd.DataFrame,
                                   transformations: Dict[str, Callable]) -> pd.DataFrame:
        """
        Apply custom transformations to specified columns.
        
        Args:
            data: Input DataFrame
            transformations: Dictionary mapping column names to transformation functions
            
        Returns:
            DataFrame with transformations applied
        """
        result = data.copy()
        
        for col, transform_func in transformations.items():
            if col in data.columns:
                try:
                    result[f"{col}_transformed"] = transform_func(data[col])
                    logger.info(f"Applied custom transformation to {col}")
                except Exception as e:
                    logger.error(f"Failed to transform {col}: {e}")
            else:
                logger.warning(f"Column {col} not found in data")
        
        return result