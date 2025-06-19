"""
core/preprocessing.py
Comprehensive data preprocessing utilities for machine learning pipelines.
"""

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import (LabelEncoder, MinMaxScaler, OneHotEncoder,
                                   PowerTransformer, QuantileTransformer,
                                   RobustScaler, StandardScaler)

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

    def scale_features(
        self,
        data: pd.DataFrame,
        columns: Optional[List[str]] = None,
        method: str = "standard",
        **kwargs,
    ) -> pd.DataFrame:
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
            "standard": StandardScaler,
            "minmax": MinMaxScaler,
            "robust": RobustScaler,
            "quantile": QuantileTransformer,
            "power": PowerTransformer,
        }

        if method not in scalers:
            raise ValueError(f"Unknown scaling method: {method}")

        scaler = scalers[method](**kwargs)

        # Fit and transform
        data_scaled = data.copy()
        data_scaled[columns] = scaler.fit_transform(data[columns])

        # Store transformer for later use
        self.transformers[f"scaler_{method}"] = scaler
        self.transformation_history.append(
            {"type": "scaling", "method": method, "columns": columns}
        )

        logger.info(f"Applied {method} scaling to {len(columns)} columns")
        return data_scaled

    def encode_categorical(
        self,
        data: pd.DataFrame,
        columns: Optional[List[str]] = None,
        method: str = "onehot",
        max_categories: int = 10,
    ) -> pd.DataFrame:
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
            columns = data.select_dtypes(
                include=["object", "category"]
            ).columns.tolist()

        data_encoded = data.copy()

        for col in columns:
            n_unique = data[col].nunique()

            if method == "onehot" and n_unique <= max_categories:
                # One-hot encoding
                dummies = pd.get_dummies(data[col], prefix=col, drop_first=True)
                data_encoded = pd.concat(
                    [data_encoded.drop(columns=[col]), dummies], axis=1
                )
                self.encoded_columns[col] = dummies.columns.tolist()

            elif method == "label" or (
                method == "onehot" and n_unique > max_categories
            ):
                # Label encoding
                le = LabelEncoder()
                data_encoded[col] = le.fit_transform(data[col].astype(str))
                self.transformers[f"labelencoder_{col}"] = le

            else:
                logger.warning(f"Skipping encoding for column {col}")

        self.transformation_history.append(
            {"type": "encoding", "method": method, "columns": columns}
        )

        logger.info(f"Encoded {len(columns)} categorical columns using {method}")
        return data_encoded

    def handle_missing_values(
        self,
        data: pd.DataFrame,
        method: str = "drop",
        threshold: float = 0.5,
        fill_value: Any = None,
        strategy: str = "mean",
    ) -> pd.DataFrame:
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

        if method == "drop":
            # Drop columns with too many missing values
            cols_to_drop = missing_stats[missing_stats > threshold].index
            if len(cols_to_drop) > 0:
                logger.info(
                    f"Dropping {len(cols_to_drop)} columns with >{threshold*100}% missing"
                )
                data_clean = data_clean.drop(columns=cols_to_drop)

            # Drop rows with any missing values
            n_rows_before = len(data_clean)
            data_clean = data_clean.dropna()
            logger.info(
                f"Dropped {n_rows_before - len(data_clean)} rows with missing values"
            )

        elif method == "fill":
            if fill_value is not None:
                data_clean = data_clean.fillna(fill_value)
            else:
                # Fill with appropriate values based on data type
                for col in data_clean.columns:
                    if data_clean[col].dtype in ["float64", "int64"]:
                        data_clean[col].fillna(data_clean[col].mean(), inplace=True)
                    else:
                        data_clean[col].fillna(data_clean[col].mode()[0], inplace=True)

        elif method == "impute":
            if strategy == "knn":
                # KNN imputation
                numeric_cols = data_clean.select_dtypes(include=[np.number]).columns
                imputer = KNNImputer(n_neighbors=5)
                data_clean[numeric_cols] = imputer.fit_transform(
                    data_clean[numeric_cols]
                )
            else:
                # Simple imputation
                numeric_cols = data_clean.select_dtypes(include=[np.number]).columns
                cat_cols = data_clean.select_dtypes(
                    include=["object", "category"]
                ).columns

                if len(numeric_cols) > 0:
                    num_imputer = SimpleImputer(strategy=strategy)
                    data_clean[numeric_cols] = num_imputer.fit_transform(
                        data_clean[numeric_cols]
                    )

                if len(cat_cols) > 0:
                    cat_imputer = SimpleImputer(strategy="most_frequent")
                    data_clean[cat_cols] = cat_imputer.fit_transform(
                        data_clean[cat_cols]
                    )

        self.transformation_history.append(
            {"type": "missing_values", "method": method, "threshold": threshold}
        )

        return data_clean

    def remove_outliers(
        self,
        data: pd.DataFrame,
        columns: Optional[List[str]] = None,
        method: str = "iqr",
        threshold: float = 1.5,
    ) -> pd.DataFrame:
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

        if method == "iqr":
            for col in columns:
                Q1 = data_clean[col].quantile(0.25)
                Q3 = data_clean[col].quantile(0.75)
                IQR = Q3 - Q1

                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR

                outliers = (data_clean[col] < lower_bound) | (
                    data_clean[col] > upper_bound
                )
                n_outliers += outliers.sum()
                data_clean = data_clean[~outliers]

        elif method == "zscore":
            for col in columns:
                z_scores = np.abs(stats.zscore(data_clean[col]))
                outliers = z_scores > threshold
                n_outliers += outliers.sum()
                data_clean = data_clean[~outliers]

        elif method == "isolation":
            from sklearn.ensemble import IsolationForest

            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            outliers = iso_forest.fit_predict(data_clean[columns]) == -1
            n_outliers = outliers.sum()
            data_clean = data_clean[~outliers]

        logger.info(f"Removed {n_outliers} outliers using {method} method")

        self.transformation_history.append(
            {
                "type": "outlier_removal",
                "method": method,
                "threshold": threshold,
                "n_removed": n_outliers,
            }
        )

        return data_clean

    def transform_target(
        self, y: Union[pd.Series, np.ndarray], method: str = "none"
    ) -> Tuple[np.ndarray, Optional[Any]]:
        """
        Transform target variable for better model performance.

        Args:
            y: Target variable
            method: Transformation method ('log', 'sqrt', 'boxcox', 'yeo-johnson')

        Returns:
            Tuple of (transformed_y, transformer_object)
        """
        y_array = np.array(y).reshape(-1, 1)

        if method == "none":
            return y_array.ravel(), None

        elif method == "log":
            # Ensure positive values
            if np.any(y_array <= 0):
                logger.warning(
                    "Log transform requires positive values, adding constant"
                )
                y_array = y_array - y_array.min() + 1
            return np.log(y_array).ravel(), {
                "method": "log",
                "offset": y_array.min() - 1,
            }

        elif method == "sqrt":
            if np.any(y_array < 0):
                logger.warning("Square root transform requires non-negative values")
                y_array = y_array - y_array.min()
            return np.sqrt(y_array).ravel(), {"method": "sqrt", "offset": y_array.min()}

        elif method in ["boxcox", "yeo-johnson"]:
            transformer = PowerTransformer(method=method)
            y_transformed = transformer.fit_transform(y_array)
            return y_transformed.ravel(), transformer

        else:
            raise ValueError(f"Unknown transformation method: {method}")

    def inverse_transform_target(
        self, y_transformed: np.ndarray, transformer: Any
    ) -> np.ndarray:
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
            if transformer["method"] == "log":
                return np.exp(y_transformed).ravel() + transformer.get("offset", 0)
            elif transformer["method"] == "sqrt":
                return np.power(y_transformed, 2).ravel() + transformer.get("offset", 0)
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
    def create_polynomial_features(
        data: pd.DataFrame,
        columns: List[str],
        degree: int = 2,
        include_bias: bool = False,
    ) -> pd.DataFrame:
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
        result = pd.concat([data, poly_df.iloc[:, len(columns) :]], axis=1)

        logger.info(f"Created {poly_df.shape[1] - len(columns)} polynomial features")
        return result

    @staticmethod
    def create_interaction_features(
        data: pd.DataFrame, column_pairs: List[Tuple[str, str]]
    ) -> pd.DataFrame:
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
    def create_ratio_features(
        data: pd.DataFrame, numerator_cols: List[str], denominator_cols: List[str]
    ) -> pd.DataFrame:
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
                if (
                    num_col != den_col
                    and num_col in data.columns
                    and den_col in data.columns
                ):
                    ratio_name = f"{num_col}_div_{den_col}"
                    # Avoid division by zero
                    result[ratio_name] = data[num_col] / (data[den_col] + 1e-8)
                    logger.info(f"Created ratio feature: {ratio_name}")

        return result

    @staticmethod
    def apply_custom_transformations(
        data: pd.DataFrame, transformations: Dict[str, Callable]
    ) -> pd.DataFrame:
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


class OutlierHandler:
    """
    Dedicated class for handling outliers in datasets.

    This class provides advanced methods for detecting and treating outliers
    using various statistical and machine learning approaches.
    """

    def __init__(self):
        """Initialize the OutlierHandler with default settings."""
        self.outlier_indices = {}
        self.outlier_scores = {}
        self.fitted_detectors = {}

    def detect_outliers(
        self,
        data: pd.DataFrame,
        columns: Optional[List[str]] = None,
        method: str = "iqr",
        **kwargs,
    ) -> Dict[str, np.ndarray]:
        """
        Detect outliers using various methods.

        Args:
            data: Input DataFrame
            columns: Columns to check for outliers (None checks all numeric)
            method: Detection method ('iqr', 'zscore', 'isolation', 'lof', 'elliptic')
            **kwargs: Additional parameters for the detection method

        Returns:
            Dictionary mapping column names to boolean arrays indicating outliers
        """
        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns.tolist()

        outliers = {}

        if method == "iqr":
            # Interquartile range method
            multiplier = kwargs.get("multiplier", 1.5)
            for col in columns:
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1

                lower_bound = Q1 - multiplier * IQR
                upper_bound = Q3 + multiplier * IQR

                outliers[col] = (data[col] < lower_bound) | (data[col] > upper_bound)

        elif method == "zscore":
            # Z-score method
            threshold = kwargs.get("threshold", 3)
            for col in columns:
                z_scores = np.abs(stats.zscore(data[col].dropna()))
                outlier_mask = np.zeros(len(data), dtype=bool)
                outlier_mask[data[col].notna()] = z_scores > threshold
                outliers[col] = outlier_mask

        elif method == "isolation":
            # Isolation Forest
            contamination = kwargs.get("contamination", 0.1)
            iso_forest = IsolationForest(
                contamination=contamination, random_state=kwargs.get("random_state", 42)
            )

            # Handle missing values
            data_numeric = data[columns].fillna(data[columns].mean())
            predictions = iso_forest.fit_predict(data_numeric)
            outliers["combined"] = predictions == -1
            self.fitted_detectors["isolation"] = iso_forest

        elif method == "lof":
            # Local Outlier Factor
            from sklearn.neighbors import LocalOutlierFactor

            n_neighbors = kwargs.get("n_neighbors", 20)
            contamination = kwargs.get("contamination", 0.1)

            lof = LocalOutlierFactor(
                n_neighbors=n_neighbors, contamination=contamination
            )

            data_numeric = data[columns].fillna(data[columns].mean())
            predictions = lof.fit_predict(data_numeric)
            outliers["combined"] = predictions == -1
            self.outlier_scores["lof"] = lof.negative_outlier_factor_

        elif method == "elliptic":
            # Elliptic Envelope (assumes Gaussian distribution)
            contamination = kwargs.get("contamination", 0.1)

            ee = EllipticEnvelope(
                contamination=contamination, random_state=kwargs.get("random_state", 42)
            )

            data_numeric = data[columns].fillna(data[columns].mean())
            predictions = ee.fit_predict(data_numeric)
            outliers["combined"] = predictions == -1
            self.fitted_detectors["elliptic"] = ee

        # Store detected outliers
        self.outlier_indices[method] = outliers

        logger.info(f"Detected outliers using {method} method")
        return outliers

    def treat_outliers(
        self,
        data: pd.DataFrame,
        outliers: Dict[str, np.ndarray],
        method: str = "remove",
        **kwargs,
    ) -> pd.DataFrame:
        """
        Treat detected outliers using various strategies.

        Args:
            data: Input DataFrame
            outliers: Dictionary of outlier indicators from detect_outliers
            method: Treatment method ('remove', 'cap', 'transform', 'impute')
            **kwargs: Additional parameters for the treatment method

        Returns:
            DataFrame with treated outliers
        """
        data_treated = data.copy()

        if method == "remove":
            # Remove rows containing outliers
            if "combined" in outliers:
                mask = ~outliers["combined"]
                data_treated = data_treated[mask]
                logger.info(f"Removed {sum(outliers['combined'])} outlier rows")
            else:
                # Remove outliers column by column
                for col, outlier_mask in outliers.items():
                    data_treated = data_treated[~outlier_mask]
                    logger.info(f"Removed {sum(outlier_mask)} outliers from {col}")

        elif method == "cap":
            # Cap outliers at specified percentiles
            lower_percentile = kwargs.get("lower_percentile", 1)
            upper_percentile = kwargs.get("upper_percentile", 99)

            for col, outlier_mask in outliers.items():
                if col != "combined" and col in data_treated.columns:
                    lower_cap = data_treated[col].quantile(lower_percentile / 100)
                    upper_cap = data_treated[col].quantile(upper_percentile / 100)

                    data_treated.loc[data_treated[col] < lower_cap, col] = lower_cap
                    data_treated.loc[data_treated[col] > upper_cap, col] = upper_cap

                    logger.info(
                        f"Capped outliers in {col} to [{lower_cap:.2f}, {upper_cap:.2f}]"
                    )

        elif method == "transform":
            # Apply transformation to reduce outlier impact
            transform_func = kwargs.get("transform_func", np.log1p)

            for col, outlier_mask in outliers.items():
                if col != "combined" and col in data_treated.columns:
                    # Apply transformation only to positive values
                    if data_treated[col].min() > 0:
                        data_treated[col] = transform_func(data_treated[col])
                        logger.info(f"Applied transformation to {col}")
                    else:
                        logger.warning(
                            f"Cannot apply transformation to {col} due to non-positive values"
                        )

        elif method == "impute":
            # Replace outliers with imputed values
            impute_strategy = kwargs.get("strategy", "median")

            for col, outlier_mask in outliers.items():
                if col != "combined" and col in data_treated.columns:
                    # Calculate imputation value from non-outlier data
                    non_outlier_data = data_treated.loc[~outlier_mask, col]

                    if impute_strategy == "mean":
                        impute_value = non_outlier_data.mean()
                    elif impute_strategy == "median":
                        impute_value = non_outlier_data.median()
                    elif impute_strategy == "mode":
                        impute_value = non_outlier_data.mode()[0]
                    else:
                        impute_value = non_outlier_data.median()

                    data_treated.loc[outlier_mask, col] = impute_value
                    logger.info(
                        f"Imputed {sum(outlier_mask)} outliers in {col} with {impute_value:.2f}"
                    )

        return data_treated

    def get_outlier_summary(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate a summary of outliers in the dataset.

        Args:
            data: Input DataFrame

        Returns:
            DataFrame with outlier statistics for each column
        """
        summary_data = []

        for method, outliers in self.outlier_indices.items():
            for col, outlier_mask in outliers.items():
                if col in data.columns:
                    outlier_values = data.loc[outlier_mask, col]

                    summary_data.append(
                        {
                            "method": method,
                            "column": col,
                            "n_outliers": sum(outlier_mask),
                            "pct_outliers": sum(outlier_mask) / len(data) * 100,
                            "outlier_mean": (
                                outlier_values.mean()
                                if len(outlier_values) > 0
                                else np.nan
                            ),
                            "outlier_std": (
                                outlier_values.std()
                                if len(outlier_values) > 0
                                else np.nan
                            ),
                            "outlier_min": (
                                outlier_values.min()
                                if len(outlier_values) > 0
                                else np.nan
                            ),
                            "outlier_max": (
                                outlier_values.max()
                                if len(outlier_values) > 0
                                else np.nan
                            ),
                        }
                    )

        return pd.DataFrame(summary_data)


class MissingValueHandler:
    """
    Specialized class for handling missing values in datasets.

    This class provides comprehensive methods for analyzing, visualizing,
    and imputing missing values using various strategies.
    """

    def __init__(self):
        """Initialize the MissingValueHandler."""
        self.missing_patterns = {}
        self.imputation_models = {}
        self.missing_stats = None

    def analyze_missing_patterns(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze patterns of missing values in the dataset.

        Args:
            data: Input DataFrame

        Returns:
            Dictionary containing missing value analysis results
        """
        analysis = {}

        # Basic missing value statistics
        missing_counts = data.isnull().sum()
        missing_percentages = (missing_counts / len(data)) * 100

        analysis["missing_counts"] = missing_counts.to_dict()
        analysis["missing_percentages"] = missing_percentages.to_dict()

        # Missing value patterns
        missing_pattern = data.isnull()
        pattern_counts = missing_pattern.value_counts()

        analysis["n_complete_rows"] = len(data.dropna())
        analysis["pct_complete_rows"] = (len(data.dropna()) / len(data)) * 100

        # Correlation of missingness between columns
        missing_corr = missing_pattern.corr()
        analysis["missing_correlation"] = missing_corr

        # Identify columns with high missing correlation
        high_corr_pairs = []
        for i in range(len(missing_corr.columns)):
            for j in range(i + 1, len(missing_corr.columns)):
                corr_value = missing_corr.iloc[i, j]
                if abs(corr_value) > 0.5:  # Threshold for high correlation
                    high_corr_pairs.append(
                        {
                            "col1": missing_corr.columns[i],
                            "col2": missing_corr.columns[j],
                            "correlation": corr_value,
                        }
                    )

        analysis["high_missing_correlations"] = high_corr_pairs

        # Missing value types (MCAR, MAR, MNAR analysis)
        analysis["missing_types"] = self._analyze_missing_types(data)

        # Store analysis results
        self.missing_patterns = analysis

        logger.info("Completed missing value pattern analysis")
        return analysis

    def _analyze_missing_types(self, data: pd.DataFrame) -> Dict[str, str]:
        """
        Attempt to identify the type of missingness for each column.

        Args:
            data: Input DataFrame

        Returns:
            Dictionary mapping columns to likely missing value types
        """
        missing_types = {}

        for col in data.columns:
            if data[col].isnull().any():
                # Simple heuristic-based classification
                missing_pct = data[col].isnull().sum() / len(data)

                if missing_pct < 0.05:
                    # Low percentage - might be MCAR
                    missing_types[col] = "Likely MCAR (random)"
                elif missing_pct > 0.5:
                    # High percentage - might be structural
                    missing_types[col] = "Possibly MNAR (not at random)"
                else:
                    # Medium percentage - check for patterns
                    # This is a simplified check - real analysis would be more complex
                    missing_types[col] = (
                        "Possibly MAR (at random conditional on observed)"
                    )

        return missing_types

    def impute_missing_values(
        self,
        data: pd.DataFrame,
        strategy: str = "auto",
        columns: Optional[List[str]] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Impute missing values using various strategies.

        Args:
            data: Input DataFrame
            strategy: Imputation strategy ('auto', 'simple', 'knn', 'iterative', 'model')
            columns: Columns to impute (None imputes all columns with missing values)
            **kwargs: Additional parameters for imputation methods

        Returns:
            DataFrame with imputed values
        """
        data_imputed = data.copy()

        if columns is None:
            columns = data.columns[data.isnull().any()].tolist()

        if strategy == "auto":
            # Automatically select strategy based on data characteristics
            strategy = self._select_imputation_strategy(data[columns])
            logger.info(f"Auto-selected imputation strategy: {strategy}")

        if strategy == "simple":
            # Simple imputation (mean, median, mode, constant)
            numeric_cols = data[columns].select_dtypes(include=[np.number]).columns
            categorical_cols = (
                data[columns].select_dtypes(include=["object", "category"]).columns
            )

            if len(numeric_cols) > 0:
                numeric_strategy = kwargs.get("numeric_strategy", "median")
                numeric_imputer = SimpleImputer(strategy=numeric_strategy)
                data_imputed[numeric_cols] = numeric_imputer.fit_transform(
                    data[numeric_cols]
                )
                self.imputation_models["numeric_simple"] = numeric_imputer

            if len(categorical_cols) > 0:
                cat_strategy = kwargs.get("categorical_strategy", "most_frequent")
                cat_imputer = SimpleImputer(strategy=cat_strategy)
                data_imputed[categorical_cols] = cat_imputer.fit_transform(
                    data[categorical_cols]
                )
                self.imputation_models["categorical_simple"] = cat_imputer

        elif strategy == "knn":
            # K-Nearest Neighbors imputation
            n_neighbors = kwargs.get("n_neighbors", 5)
            weights = kwargs.get("weights", "uniform")

            # KNN imputer only works with numeric data
            numeric_cols = data[columns].select_dtypes(include=[np.number]).columns

            if len(numeric_cols) > 0:
                knn_imputer = KNNImputer(n_neighbors=n_neighbors, weights=weights)
                data_imputed[numeric_cols] = knn_imputer.fit_transform(
                    data[numeric_cols]
                )
                self.imputation_models["knn"] = knn_imputer

            # For categorical columns, fall back to mode imputation
            categorical_cols = (
                data[columns].select_dtypes(include=["object", "category"]).columns
            )
            if len(categorical_cols) > 0:
                cat_imputer = SimpleImputer(strategy="most_frequent")
                data_imputed[categorical_cols] = cat_imputer.fit_transform(
                    data[categorical_cols]
                )

        elif strategy == "iterative":
            # Iterative imputation (MICE-like)
            from sklearn.experimental import enable_iterative_imputer
            from sklearn.impute import IterativeImputer

            max_iter = kwargs.get("max_iter", 10)
            random_state = kwargs.get("random_state", 42)

            # Iterative imputer only works with numeric data
            numeric_cols = data[columns].select_dtypes(include=[np.number]).columns

            if len(numeric_cols) > 0:
                iterative_imputer = IterativeImputer(
                    max_iter=max_iter, random_state=random_state
                )
                data_imputed[numeric_cols] = iterative_imputer.fit_transform(
                    data[numeric_cols]
                )
                self.imputation_models["iterative"] = iterative_imputer

        elif strategy == "model":
            # Model-based imputation (using RandomForest by default)
            from sklearn.ensemble import (RandomForestClassifier,
                                          RandomForestRegressor)

            # Impute each column with missing values using others as features
            for col in columns:
                if data[col].isnull().any():
                    # Prepare training data (rows without missing values in target column)
                    train_mask = data[col].notna()
                    feature_cols = [
                        c for c in data.columns if c != col and c not in columns
                    ]

                    if (
                        len(feature_cols) > 0 and train_mask.sum() > 10
                    ):  # Need enough training data
                        X_train = data.loc[train_mask, feature_cols]
                        y_train = data.loc[train_mask, col]
                        X_impute = data.loc[~train_mask, feature_cols]

                        # Select model based on column type
                        if data[col].dtype in ["float64", "int64"]:
                            model = RandomForestRegressor(
                                n_estimators=50, random_state=42
                            )
                        else:
                            model = RandomForestClassifier(
                                n_estimators=50, random_state=42
                            )

                        try:
                            model.fit(X_train, y_train)
                            data_imputed.loc[~train_mask, col] = model.predict(X_impute)
                            self.imputation_models[f"model_{col}"] = model
                        except Exception as e:
                            logger.warning(
                                f"Model-based imputation failed for {col}: {e}"
                            )
                            # Fall back to simple imputation
                            if data[col].dtype in ["float64", "int64"]:
                                data_imputed[col].fillna(
                                    data[col].median(), inplace=True
                                )
                            else:
                                data_imputed[col].fillna(
                                    data[col].mode()[0], inplace=True
                                )

        # Log imputation summary
        n_imputed = data.isnull().sum().sum() - data_imputed.isnull().sum().sum()
        logger.info(f"Imputed {n_imputed} missing values using {strategy} strategy")

        return data_imputed

    def _select_imputation_strategy(self, data: pd.DataFrame) -> str:
        """
        Automatically select the best imputation strategy based on data characteristics.

        Args:
            data: Input DataFrame

        Returns:
            Selected imputation strategy
        """
        # Calculate data characteristics
        n_rows, n_cols = data.shape
        missing_pct = data.isnull().sum().sum() / (n_rows * n_cols)
        n_numeric = len(data.select_dtypes(include=[np.number]).columns)
        n_categorical = len(data.select_dtypes(include=["object", "category"]).columns)

        # Simple heuristics for strategy selection
        if missing_pct < 0.05:
            # Low missing percentage - simple imputation is usually fine
            return "simple"
        elif missing_pct > 0.3:
            # High missing percentage - need more sophisticated method
            if n_rows > 1000 and n_numeric > n_categorical:
                return "iterative"
            else:
                return "knn"
        else:
            # Medium missing percentage
            if n_rows > 5000:
                return "simple"  # For efficiency
            else:
                return "knn"

    def create_missing_indicator(
        self, data: pd.DataFrame, columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Create binary indicators for missing values.

        Args:
            data: Input DataFrame
            columns: Columns to create indicators for (None creates for all with missing)

        Returns:
            DataFrame with original data and missing indicators
        """
        if columns is None:
            columns = data.columns[data.isnull().any()].tolist()

        result = data.copy()

        for col in columns:
            if data[col].isnull().any():
                indicator_name = f"{col}_was_missing"
                result[indicator_name] = data[col].isnull().astype(int)
                logger.info(f"Created missing indicator: {indicator_name}")

        return result

    def get_imputation_report(self) -> pd.DataFrame:
        """
        Generate a report of all imputation operations performed.

        Returns:
            DataFrame summarizing imputation operations
        """
        report_data = []

        for model_name, model in self.imputation_models.items():
            report_data.append(
                {
                    "model_name": model_name,
                    "model_type": type(model).__name__,
                    "fitted": True,
                }
            )

        if self.missing_patterns:
            # Add missing pattern statistics
            for col, pct in self.missing_patterns["missing_percentages"].items():
                report_data.append(
                    {
                        "column": col,
                        "missing_percentage": pct,
                        "missing_type": self.missing_patterns["missing_types"].get(
                            col, "Unknown"
                        ),
                    }
                )

        return pd.DataFrame(report_data)
