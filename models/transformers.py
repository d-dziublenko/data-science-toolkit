"""
models/transformers.py
Target variable transformations for improving model performance.
"""

import numpy as np
import pandas as pd
from typing import Optional, Union, Tuple, Dict, Any, Callable, List
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, PowerTransformer
from scipy import stats
from scipy.special import inv_boxcox
import warnings
import logging

logger = logging.getLogger(__name__)


class TargetTransformer(BaseEstimator, TransformerMixin):
    """
    Base class for target variable transformations.
    
    Provides common functionality for transforming and inverse transforming
    target variables to improve model performance.
    """
    
    def __init__(self, name: str = None):
        """
        Initialize the transformer.
        
        Args:
            name: Transformer name
        """
        self.name = name or self.__class__.__name__
        self.is_fitted = False
        self._original_shape = None
        self._transform_params = {}
    
    def fit(self, y: Union[np.ndarray, pd.Series], X: Optional[np.ndarray] = None) -> 'TargetTransformer':
        """
        Fit the transformer to the target variable.
        
        Args:
            y: Target variable
            X: Features (optional, for target encoding)
            
        Returns:
            Self
        """
        if isinstance(y, pd.Series):
            y = y.values
        
        self._original_shape = y.shape
        self.is_fitted = True
        
        return self
    
    def transform(self, y: Union[np.ndarray, pd.Series]) -> np.ndarray:
        """
        Transform the target variable.
        
        Args:
            y: Target variable to transform
            
        Returns:
            Transformed target
        """
        if not self.is_fitted:
            raise ValueError("Transformer must be fitted before transform.")
        
        if isinstance(y, pd.Series):
            y = y.values
            
        return y
    
    def inverse_transform(self, y_transformed: np.ndarray) -> np.ndarray:
        """
        Inverse transform the target variable.
        
        Args:
            y_transformed: Transformed target variable
            
        Returns:
            Original scale target
        """
        if not self.is_fitted:
            raise ValueError("Transformer must be fitted before inverse_transform.")
            
        return y_transformed
    
    def fit_transform(self, y: Union[np.ndarray, pd.Series], X: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Fit and transform in one step.
        
        Args:
            y: Target variable
            X: Features (optional)
            
        Returns:
            Transformed target
        """
        return self.fit(y, X).transform(y)


class LogTransformer(TargetTransformer):
    """
    Logarithmic transformation for positive skewed targets.
    
    Applies log(1 + y) transformation to handle zero values.
    """
    
    def __init__(self, shift: float = 1.0, base: str = 'e', name: str = None):
        """
        Initialize the log transformer.
        
        Args:
            shift: Value to add before log transformation (to handle zeros)
            base: Logarithm base ('e', '10', '2')
            name: Transformer name
        """
        super().__init__(name)
        self.shift = shift
        self.base = base
        self._log_func = {
            'e': np.log,
            '10': np.log10,
            '2': np.log2
        }.get(base, np.log)
        self._exp_func = {
            'e': np.exp,
            '10': lambda x: np.power(10, x),
            '2': lambda x: np.power(2, x)
        }.get(base, np.exp)
    
    def fit(self, y: Union[np.ndarray, pd.Series], X: Optional[np.ndarray] = None) -> 'LogTransformer':
        """Fit the log transformer."""
        super().fit(y, X)
        
        if isinstance(y, pd.Series):
            y = y.values
        
        # Check for negative values
        if np.any(y < -self.shift):
            raise ValueError(f"Target contains values less than -{self.shift}. "
                           f"Increase shift parameter or use different transformation.")
        
        # Store statistics
        self._transform_params['min_value'] = np.min(y)
        self._transform_params['max_value'] = np.max(y)
        
        return self
    
    def transform(self, y: Union[np.ndarray, pd.Series]) -> np.ndarray:
        """Apply log transformation."""
        y = super().transform(y)
        
        # Apply log(shift + y)
        y_transformed = self._log_func(self.shift + y)
        
        return y_transformed
    
    def inverse_transform(self, y_transformed: np.ndarray) -> np.ndarray:
        """Apply inverse log transformation."""
        y_transformed = super().inverse_transform(y_transformed)
        
        # Apply exp(y) - shift
        y_original = self._exp_func(y_transformed) - self.shift
        
        return y_original


class BoxCoxTransformer(TargetTransformer):
    """
    Box-Cox transformation for normalizing skewed targets.
    
    Automatically finds the optimal lambda parameter.
    """
    
    def __init__(self, method: str = 'mle', standardize: bool = True, name: str = None):
        """
        Initialize the Box-Cox transformer.
        
        Args:
            method: Method for finding lambda ('mle' or 'pearsonr')
            standardize: Whether to standardize after transformation
            name: Transformer name
        """
        super().__init__(name)
        self.method = method
        self.standardize = standardize
        self.lambda_ = None
        self._scaler = StandardScaler() if standardize else None
        self._shift = None
    
    def fit(self, y: Union[np.ndarray, pd.Series], X: Optional[np.ndarray] = None) -> 'BoxCoxTransformer':
        """Fit the Box-Cox transformer."""
        super().fit(y, X)
        
        if isinstance(y, pd.Series):
            y = y.values
        
        # Ensure positive values
        min_value = np.min(y)
        if min_value <= 0:
            self._shift = -min_value + 1e-6
            y = y + self._shift
            logger.info(f"Shifted target by {self._shift} to ensure positive values")
        else:
            self._shift = 0
        
        # Find optimal lambda
        _, self.lambda_ = stats.boxcox(y, method=self.method)
        
        # Fit scaler if needed
        if self.standardize:
            y_transformed, _ = stats.boxcox(y, lmbda=self.lambda_)
            self._scaler.fit(y_transformed.reshape(-1, 1))
        
        self._transform_params['lambda'] = self.lambda_
        self._transform_params['shift'] = self._shift
        
        return self
    
    def transform(self, y: Union[np.ndarray, pd.Series]) -> np.ndarray:
        """Apply Box-Cox transformation."""
        y = super().transform(y)
        
        # Shift if necessary
        if self._shift > 0:
            y = y + self._shift
        
        # Apply Box-Cox
        y_transformed = stats.boxcox(y, lmbda=self.lambda_)
        
        # Standardize if requested
        if self.standardize:
            y_transformed = self._scaler.transform(y_transformed.reshape(-1, 1)).ravel()
        
        return y_transformed
    
    def inverse_transform(self, y_transformed: np.ndarray) -> np.ndarray:
        """Apply inverse Box-Cox transformation."""
        y_transformed = super().inverse_transform(y_transformed)
        
        # Inverse standardization
        if self.standardize:
            y_transformed = self._scaler.inverse_transform(y_transformed.reshape(-1, 1)).ravel()
        
        # Inverse Box-Cox
        y_original = inv_boxcox(y_transformed, self.lambda_)
        
        # Remove shift
        if self._shift > 0:
            y_original = y_original - self._shift
        
        return y_original


class YeoJohnsonTransformer(TargetTransformer):
    """
    Yeo-Johnson transformation for normalizing targets.
    
    Similar to Box-Cox but handles negative values.
    """
    
    def __init__(self, standardize: bool = True, name: str = None):
        """
        Initialize the Yeo-Johnson transformer.
        
        Args:
            standardize: Whether to standardize after transformation
            name: Transformer name
        """
        super().__init__(name)
        self.standardize = standardize
        self.lambda_ = None
        self._power_transformer = PowerTransformer(method='yeo-johnson', standardize=standardize)
    
    def fit(self, y: Union[np.ndarray, pd.Series], X: Optional[np.ndarray] = None) -> 'YeoJohnsonTransformer':
        """Fit the Yeo-Johnson transformer."""
        super().fit(y, X)
        
        if isinstance(y, pd.Series):
            y = y.values
        
        # Fit power transformer
        self._power_transformer.fit(y.reshape(-1, 1))
        self.lambda_ = self._power_transformer.lambdas_[0]
        
        self._transform_params['lambda'] = self.lambda_
        
        return self
    
    def transform(self, y: Union[np.ndarray, pd.Series]) -> np.ndarray:
        """Apply Yeo-Johnson transformation."""
        y = super().transform(y)
        
        # Apply transformation
        y_transformed = self._power_transformer.transform(y.reshape(-1, 1)).ravel()
        
        return y_transformed
    
    def inverse_transform(self, y_transformed: np.ndarray) -> np.ndarray:
        """Apply inverse Yeo-Johnson transformation."""
        y_transformed = super().inverse_transform(y_transformed)
        
        # Apply inverse transformation
        y_original = self._power_transformer.inverse_transform(y_transformed.reshape(-1, 1)).ravel()
        
        return y_original


class QuantileTransformer(TargetTransformer):
    """
    Quantile transformation to uniform or normal distribution.
    
    Maps target values to quantiles of the desired distribution.
    """
    
    def __init__(self, output_distribution: str = 'uniform', 
                 n_quantiles: int = 1000,
                 subsample: int = 100000,
                 name: str = None):
        """
        Initialize the quantile transformer.
        
        Args:
            output_distribution: Target distribution ('uniform' or 'normal')
            n_quantiles: Number of quantiles to estimate
            subsample: Maximum number of samples to use
            name: Transformer name
        """
        super().__init__(name)
        self.output_distribution = output_distribution
        self.n_quantiles = n_quantiles
        self.subsample = subsample
        self._quantile_transformer = None
    
    def fit(self, y: Union[np.ndarray, pd.Series], X: Optional[np.ndarray] = None) -> 'QuantileTransformer':
        """Fit the quantile transformer."""
        super().fit(y, X)
        
        from sklearn.preprocessing import QuantileTransformer as SKQuantileTransformer
        
        self._quantile_transformer = SKQuantileTransformer(
            output_distribution=self.output_distribution,
            n_quantiles=self.n_quantiles,
            subsample=self.subsample
        )
        
        if isinstance(y, pd.Series):
            y = y.values
        
        self._quantile_transformer.fit(y.reshape(-1, 1))
        
        return self
    
    def transform(self, y: Union[np.ndarray, pd.Series]) -> np.ndarray:
        """Apply quantile transformation."""
        y = super().transform(y)
        
        y_transformed = self._quantile_transformer.transform(y.reshape(-1, 1)).ravel()
        
        return y_transformed
    
    def inverse_transform(self, y_transformed: np.ndarray) -> np.ndarray:
        """Apply inverse quantile transformation."""
        y_transformed = super().inverse_transform(y_transformed)
        
        y_original = self._quantile_transformer.inverse_transform(y_transformed.reshape(-1, 1)).ravel()
        
        return y_original


class RankTransformer(TargetTransformer):
    """
    Rank transformation for ordinal targets.
    
    Converts values to ranks, useful for ordinal regression.
    """
    
    def __init__(self, method: str = 'average', normalize: bool = True, name: str = None):
        """
        Initialize the rank transformer.
        
        Args:
            method: How to handle ties ('average', 'min', 'max', 'dense', 'ordinal')
            normalize: Whether to normalize ranks to [0, 1]
            name: Transformer name
        """
        super().__init__(name)
        self.method = method
        self.normalize = normalize
        self._rank_mapping = None
        self._inverse_mapping = None
    
    def fit(self, y: Union[np.ndarray, pd.Series], X: Optional[np.ndarray] = None) -> 'RankTransformer':
        """Fit the rank transformer."""
        super().fit(y, X)
        
        if isinstance(y, pd.Series):
            y = y.values
        
        # Create rank mapping
        from scipy.stats import rankdata
        ranks = rankdata(y, method=self.method)
        
        # Store unique value to rank mapping
        unique_vals = np.unique(y)
        self._rank_mapping = {}
        for val in unique_vals:
            mask = y == val
            self._rank_mapping[val] = np.mean(ranks[mask])
        
        # Create inverse mapping
        self._inverse_mapping = {v: k for k, v in self._rank_mapping.items()}
        
        # Store normalization parameters
        if self.normalize:
            self._transform_params['min_rank'] = np.min(ranks)
            self._transform_params['max_rank'] = np.max(ranks)
        
        return self
    
    def transform(self, y: Union[np.ndarray, pd.Series]) -> np.ndarray:
        """Apply rank transformation."""
        y = super().transform(y)
        
        # Map to ranks
        y_transformed = np.array([self._rank_mapping.get(val, np.nan) for val in y])
        
        # Handle unseen values
        if np.any(np.isnan(y_transformed)):
            warnings.warn("Unseen values encountered, using interpolation")
            # Interpolate ranks for unseen values
            for i, val in enumerate(y):
                if np.isnan(y_transformed[i]):
                    # Find nearest known values
                    known_vals = np.array(list(self._rank_mapping.keys()))
                    nearest_idx = np.argmin(np.abs(known_vals - val))
                    y_transformed[i] = self._rank_mapping[known_vals[nearest_idx]]
        
        # Normalize if requested
        if self.normalize:
            min_rank = self._transform_params['min_rank']
            max_rank = self._transform_params['max_rank']
            y_transformed = (y_transformed - min_rank) / (max_rank - min_rank)
        
        return y_transformed
    
    def inverse_transform(self, y_transformed: np.ndarray) -> np.ndarray:
        """Apply inverse rank transformation."""
        y_transformed = super().inverse_transform(y_transformed)
        
        # Denormalize if needed
        if self.normalize:
            min_rank = self._transform_params['min_rank']
            max_rank = self._transform_params['max_rank']
            y_transformed = y_transformed * (max_rank - min_rank) + min_rank
        
        # Map back to original values
        y_original = np.zeros_like(y_transformed)
        for i, rank in enumerate(y_transformed):
            # Find closest rank
            ranks = np.array(list(self._inverse_mapping.keys()))
            nearest_idx = np.argmin(np.abs(ranks - rank))
            y_original[i] = self._inverse_mapping[ranks[nearest_idx]]
        
        return y_original


class CompositeTransformer(TargetTransformer):
    """
    Composite transformer that applies multiple transformations in sequence.
    """
    
    def __init__(self, transformers: List[TargetTransformer], name: str = None):
        """
        Initialize the composite transformer.
        
        Args:
            transformers: List of transformers to apply in sequence
            name: Transformer name
        """
        super().__init__(name)
        self.transformers = transformers
    
    def fit(self, y: Union[np.ndarray, pd.Series], X: Optional[np.ndarray] = None) -> 'CompositeTransformer':
        """Fit all transformers in sequence."""
        super().fit(y, X)
        
        y_temp = y
        for transformer in self.transformers:
            transformer.fit(y_temp, X)
            y_temp = transformer.transform(y_temp)
        
        return self
    
    def transform(self, y: Union[np.ndarray, pd.Series]) -> np.ndarray:
        """Apply all transformations in sequence."""
        y = super().transform(y)
        
        for transformer in self.transformers:
            y = transformer.transform(y)
        
        return y
    
    def inverse_transform(self, y_transformed: np.ndarray) -> np.ndarray:
        """Apply inverse transformations in reverse order."""
        y_transformed = super().inverse_transform(y_transformed)
        
        # Apply in reverse order
        for transformer in reversed(self.transformers):
            y_transformed = transformer.inverse_transform(y_transformed)
        
        return y_transformed


class AutoTargetTransformer(TargetTransformer):
    """
    Automatically selects the best transformation based on the target distribution.
    """
    
    def __init__(self, transformations: Optional[List[str]] = None, 
                 test_normality: bool = True,
                 name: str = None):
        """
        Initialize the auto transformer.
        
        Args:
            transformations: List of transformations to try
            test_normality: Whether to test for normality
            name: Transformer name
        """
        super().__init__(name)
        self.transformations = transformations or ['none', 'log', 'sqrt', 'boxcox', 'yeo-johnson']
        self.test_normality = test_normality
        self.best_transformer_ = None
        self.transformation_scores_ = {}
    
    def _create_transformer(self, transformation: str) -> TargetTransformer:
        """Create a transformer instance based on name."""
        transformers = {
            'none': TargetTransformer(),
            'log': LogTransformer(),
            'sqrt': LogTransformer(shift=0, base='2'),  # sqrt via log base 2
            'boxcox': BoxCoxTransformer(),
            'yeo-johnson': YeoJohnsonTransformer(),
            'quantile': QuantileTransformer(output_distribution='normal'),
            'rank': RankTransformer()
        }
        
        if transformation not in transformers:
            raise ValueError(f"Unknown transformation: {transformation}")
        
        return transformers[transformation]
    
    def _test_normality(self, y: np.ndarray) -> float:
        """
        Test normality of the distribution.
        
        Args:
            y: Data to test
            
        Returns:
            Normality score (higher is more normal)
        """
        # Shapiro-Wilk test
        _, p_value = stats.shapiro(y[:min(5000, len(y))])  # Limit sample size
        
        # Also compute skewness and kurtosis
        skewness = abs(stats.skew(y))
        kurtosis = abs(stats.kurtosis(y))
        
        # Combined score (higher is better)
        score = p_value * np.exp(-skewness) * np.exp(-kurtosis/10)
        
        return score
    
    def fit(self, y: Union[np.ndarray, pd.Series], X: Optional[np.ndarray] = None) -> 'AutoTargetTransformer':
        """Fit by trying different transformations and selecting the best."""
        super().fit(y, X)
        
        if isinstance(y, pd.Series):
            y = y.values
        
        best_score = -np.inf
        
        for transformation in self.transformations:
            logger.info(f"Trying {transformation} transformation")
            
            try:
                # Create and fit transformer
                transformer = self._create_transformer(transformation)
                transformer.fit(y, X)
                
                # Transform data
                y_transformed = transformer.transform(y)
                
                # Evaluate transformation
                if self.test_normality:
                    score = self._test_normality(y_transformed)
                else:
                    # Use variance stabilization as metric
                    score = -np.var(y_transformed) / (np.mean(y_transformed)**2 + 1e-10)
                
                self.transformation_scores_[transformation] = score
                
                if score > best_score:
                    best_score = score
                    self.best_transformer_ = transformer
                    self._transform_params['best_transformation'] = transformation
                    
            except Exception as e:
                logger.warning(f"Failed to apply {transformation}: {str(e)}")
                self.transformation_scores_[transformation] = -np.inf
        
        logger.info(f"Selected transformation: {self._transform_params.get('best_transformation', 'none')}")
        
        return self
    
    def transform(self, y: Union[np.ndarray, pd.Series]) -> np.ndarray:
        """Apply the best transformation."""
        y = super().transform(y)
        return self.best_transformer_.transform(y)
    
    def inverse_transform(self, y_transformed: np.ndarray) -> np.ndarray:
        """Apply inverse of the best transformation."""
        y_transformed = super().inverse_transform(y_transformed)
        return self.best_transformer_.inverse_transform(y_transformed)
    
    def get_transformation_report(self) -> pd.DataFrame:
        """
        Get a report of all tested transformations.
        
        Returns:
            DataFrame with transformation scores
        """
        report = pd.DataFrame([
            {'transformation': k, 'score': v}
            for k, v in self.transformation_scores_.items()
        ])
        return report.sort_values('score', ascending=False)


class RobustTargetTransformer(TargetTransformer):
    """
    Robust transformer that handles outliers before transformation.
    """
    
    def __init__(self, base_transformer: TargetTransformer,
                 outlier_method: str = 'iqr',
                 outlier_threshold: float = 1.5,
                 name: str = None):
        """
        Initialize the robust transformer.
        
        Args:
            base_transformer: Base transformer to apply after outlier handling
            outlier_method: Method for outlier detection ('iqr', 'zscore', 'isolation')
            outlier_threshold: Threshold for outlier detection
            name: Transformer name
        """
        super().__init__(name)
        self.base_transformer = base_transformer
        self.outlier_method = outlier_method
        self.outlier_threshold = outlier_threshold
        self._outlier_mask = None
        self._outlier_values = None
    
    def _detect_outliers(self, y: np.ndarray) -> np.ndarray:
        """
        Detect outliers in the target variable.
        
        Args:
            y: Target variable
            
        Returns:
            Boolean mask of outliers
        """
        if self.outlier_method == 'iqr':
            # Interquartile range method
            q1 = np.percentile(y, 25)
            q3 = np.percentile(y, 75)
            iqr = q3 - q1
            lower_bound = q1 - self.outlier_threshold * iqr
            upper_bound = q3 + self.outlier_threshold * iqr
            outliers = (y < lower_bound) | (y > upper_bound)
            
        elif self.outlier_method == 'zscore':
            # Z-score method
            z_scores = np.abs(stats.zscore(y))
            outliers = z_scores > self.outlier_threshold
            
        elif self.outlier_method == 'isolation':
            # Isolation Forest method
            from sklearn.ensemble import IsolationForest
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            outliers = iso_forest.fit_predict(y.reshape(-1, 1)) == -1
            
        else:
            raise ValueError(f"Unknown outlier method: {self.outlier_method}")
        
        return outliers
    
    def fit(self, y: Union[np.ndarray, pd.Series], X: Optional[np.ndarray] = None) -> 'RobustTargetTransformer':
        """Fit the robust transformer."""
        super().fit(y, X)
        
        if isinstance(y, pd.Series):
            y = y.values
        
        # Detect outliers
        self._outlier_mask = self._detect_outliers(y)
        self._outlier_values = y[self._outlier_mask].copy()
        
        # Fit base transformer on non-outlier data
        y_clean = y[~self._outlier_mask]
        if len(y_clean) > 0:
            self.base_transformer.fit(y_clean, X[~self._outlier_mask] if X is not None else None)
        else:
            # All values are outliers, fit on all data
            self.base_transformer.fit(y, X)
        
        self._transform_params['n_outliers'] = np.sum(self._outlier_mask)
        self._transform_params['outlier_percentage'] = 100 * np.mean(self._outlier_mask)
        
        logger.info(f"Detected {self._transform_params['n_outliers']} outliers "
                   f"({self._transform_params['outlier_percentage']:.1f}%)")
        
        return self
    
    def transform(self, y: Union[np.ndarray, pd.Series]) -> np.ndarray:
        """Apply robust transformation."""
        y = super().transform(y)
        
        # Transform all values
        y_transformed = self.base_transformer.transform(y)
        
        return y_transformed
    
    def inverse_transform(self, y_transformed: np.ndarray) -> np.ndarray:
        """Apply inverse robust transformation."""
        y_transformed = super().inverse_transform(y_transformed)
        
        # Inverse transform
        y_original = self.base_transformer.inverse_transform(y_transformed)
        
        return y_original


class TargetEncoder(TargetTransformer):
    """
    Target encoding for categorical targets in regression problems.
    
    Useful for converting categorical targets to numerical values
    based on their relationship with features.
    """
    
    def __init__(self, method: str = 'mean', smoothing: float = 1.0, name: str = None):
        """
        Initialize the target encoder.
        
        Args:
            method: Encoding method ('mean', 'median', 'probability')
            smoothing: Smoothing parameter for regularization
            name: Transformer name
        """
        super().__init__(name)
        self.method = method
        self.smoothing = smoothing
        self._encoding_map = {}
        self._global_stat = None
    
    def fit(self, y: Union[np.ndarray, pd.Series], X: Optional[np.ndarray] = None) -> 'TargetEncoder':
        """Fit the target encoder."""
        super().fit(y, X)
        
        if isinstance(y, pd.Series):
            y = y.values
        
        # Calculate global statistic
        if self.method == 'mean':
            self._global_stat = np.mean(y)
        elif self.method == 'median':
            self._global_stat = np.median(y)
        elif self.method == 'probability':
            # For binary classification
            self._global_stat = np.mean(y == 1) if len(np.unique(y)) == 2 else 0.5
        
        # If X is provided, create conditional encodings
        if X is not None:
            # For simplicity, assume X is categorical
            for feature_idx in range(X.shape[1]):
                feature_values = X[:, feature_idx]
                unique_values = np.unique(feature_values)
                
                feature_encoding = {}
                for val in unique_values:
                    mask = feature_values == val
                    y_subset = y[mask]
                    
                    if len(y_subset) > 0:
                        if self.method == 'mean':
                            stat = np.mean(y_subset)
                        elif self.method == 'median':
                            stat = np.median(y_subset)
                        elif self.method == 'probability':
                            stat = np.mean(y_subset == 1) if len(np.unique(y)) == 2 else 0.5
                        
                        # Apply smoothing
                        n = len(y_subset)
                        smoothed_stat = (n * stat + self.smoothing * self._global_stat) / (n + self.smoothing)
                        feature_encoding[val] = smoothed_stat
                    else:
                        feature_encoding[val] = self._global_stat
                
                self._encoding_map[feature_idx] = feature_encoding
        
        return self
    
    def transform(self, y: Union[np.ndarray, pd.Series]) -> np.ndarray:
        """Apply target encoding (returns original y for target transformation)."""
        # Target encoding typically doesn't transform y itself
        # It's used to create features from categorical variables
        return super().transform(y)
    
    def transform_features(self, X: np.ndarray) -> np.ndarray:
        """
        Transform categorical features using target encoding.
        
        Args:
            X: Categorical features
            
        Returns:
            Encoded features
        """
        X_encoded = np.zeros_like(X, dtype=float)
        
        for feature_idx in range(X.shape[1]):
            if feature_idx in self._encoding_map:
                encoding = self._encoding_map[feature_idx]
                for i, val in enumerate(X[:, feature_idx]):
                    X_encoded[i, feature_idx] = encoding.get(val, self._global_stat)
            else:
                # No encoding available, use global statistic
                X_encoded[:, feature_idx] = self._global_stat
        
        return X_encoded


# Utility functions
def select_best_transformation(y: Union[np.ndarray, pd.Series],
                             transformations: Optional[List[str]] = None,
                             metric: str = 'normality') -> TargetTransformer:
    """
    Select the best transformation for a target variable.
    
    Args:
        y: Target variable
        transformations: List of transformations to try
        metric: Metric to optimize ('normality', 'variance')
        
    Returns:
        Best transformer instance
    """
    auto_transformer = AutoTargetTransformer(
        transformations=transformations,
        test_normality=(metric == 'normality')
    )
    auto_transformer.fit(y)
    
    return auto_transformer.best_transformer_


def create_transformation_pipeline(transformations: List[Union[str, TargetTransformer]]) -> CompositeTransformer:
    """
    Create a pipeline of transformations.
    
    Args:
        transformations: List of transformation names or instances
        
    Returns:
        Composite transformer
    """
    transformer_instances = []
    
    for transform in transformations:
        if isinstance(transform, str):
            if transform == 'log':
                transformer_instances.append(LogTransformer())
            elif transform == 'boxcox':
                transformer_instances.append(BoxCoxTransformer())
            elif transform == 'yeo-johnson':
                transformer_instances.append(YeoJohnsonTransformer())
            elif transform == 'quantile':
                transformer_instances.append(QuantileTransformer())
            elif transform == 'rank':
                transformer_instances.append(RankTransformer())
            else:
                raise ValueError(f"Unknown transformation: {transform}")
        else:
            transformer_instances.append(transform)
    
    return CompositeTransformer(transformer_instances)


# Example usage
def example_transformations():
    """Example of using various transformations."""
    import matplotlib.pyplot as plt
    
    # Generate skewed data
    np.random.seed(42)
    y = np.random.lognormal(0, 1, 1000)
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    # Original data
    axes[0].hist(y, bins=50, alpha=0.7, color='blue')
    axes[0].set_title('Original Data')
    axes[0].set_ylabel('Frequency')
    
    # Test different transformations
    transformations = [
        ('Log', LogTransformer()),
        ('Box-Cox', BoxCoxTransformer()),
        ('Yeo-Johnson', YeoJohnsonTransformer()),
        ('Quantile', QuantileTransformer(output_distribution='normal')),
        ('Rank', RankTransformer())
    ]
    
    for i, (name, transformer) in enumerate(transformations, 1):
        # Fit and transform
        transformer.fit(y)
        y_transformed = transformer.transform(y)
        
        # Plot
        axes[i].hist(y_transformed, bins=50, alpha=0.7, color='green')
        axes[i].set_title(f'{name} Transformation')
        axes[i].set_ylabel('Frequency')
        
        # Add normality test p-value
        _, p_value = stats.shapiro(y_transformed[:min(5000, len(y_transformed))])
        axes[i].text(0.05, 0.95, f'Shapiro p={p_value:.3f}',
                    transform=axes[i].transAxes, verticalalignment='top')
    
    plt.tight_layout()
    plt.suptitle('Target Variable Transformations', y=1.02, fontsize=16)
    
    # Test auto transformer
    print("\nAuto Transformer Results:")
    auto_transformer = AutoTargetTransformer()
    auto_transformer.fit(y)
    print(auto_transformer.get_transformation_report())
    print(f"\nBest transformation: {auto_transformer._transform_params.get('best_transformation')}")
    
    # Test robust transformer
    print("\nRobust Transformer with outliers:")
    y_with_outliers = y.copy()
    y_with_outliers[:10] = y_with_outliers[:10] * 100  # Add outliers
    
    robust_transformer = RobustTargetTransformer(
        base_transformer=BoxCoxTransformer(),
        outlier_method='iqr'
    )
    robust_transformer.fit(y_with_outliers)
    print(f"Detected {robust_transformer._transform_params['n_outliers']} outliers")
    
    return fig


if __name__ == "__main__":
    # Run examples
    fig = example_transformations()
    print("\nTransformation examples completed!")