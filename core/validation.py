"""
core/validation.py
Comprehensive data validation utilities for ensuring data quality and integrity.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


class ValidationLevel(Enum):
    """Severity levels for validation issues."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationResult:
    """Container for validation results."""

    passed: bool
    level: ValidationLevel
    message: str
    details: Optional[Dict[str, Any]] = None
    affected_columns: Optional[List[str]] = None
    affected_rows: Optional[List[int]] = None


class DataValidator:
    """
    Comprehensive data validation framework.

    This class provides methods to validate data quality, consistency,
    and adherence to specified rules before model training or inference.
    """

    def __init__(self, strict_mode: bool = False):
        """
        Initialize the DataValidator.

        Args:
            strict_mode: If True, warnings are treated as errors
        """
        self.strict_mode = strict_mode
        self.validation_results = []
        self.validation_rules = {}

    def validate(
        self,
        data: pd.DataFrame,
        target_column: Optional[str] = None,
        feature_columns: Optional[List[str]] = None,
    ) -> Tuple[bool, List[ValidationResult]]:
        """
        Perform comprehensive data validation.

        Args:
            data: DataFrame to validate
            target_column: Name of target column (if applicable)
            feature_columns: List of feature columns to validate

        Returns:
            Tuple of (overall_passed, list_of_results)
        """
        self.validation_results = []

        # Basic structure validation
        self._validate_basic_structure(data)

        # Data type validation
        self._validate_data_types(data)

        # Missing value validation
        self._validate_missing_values(data)

        # Duplicate validation
        self._validate_duplicates(data)

        # Statistical validation
        self._validate_statistical_properties(data, feature_columns)

        # Target validation if specified
        if target_column:
            self._validate_target(data, target_column)

        # Feature-specific validation
        if feature_columns:
            self._validate_features(data, feature_columns)

        # Custom rule validation
        self._apply_custom_rules(data)

        # Determine overall pass/fail
        overall_passed = self._determine_overall_status()

        return overall_passed, self.validation_results

    def _validate_basic_structure(self, data: pd.DataFrame):
        """Validate basic DataFrame structure."""
        # Check if DataFrame is empty
        if data.empty:
            self._add_result(
                passed=False,
                level=ValidationLevel.CRITICAL,
                message="DataFrame is empty",
            )
            return

        # Check shape
        n_rows, n_cols = data.shape
        if n_rows < 2:
            self._add_result(
                passed=False,
                level=ValidationLevel.ERROR,
                message=f"Insufficient rows: {n_rows} (minimum 2 required)",
                details={"n_rows": n_rows},
            )

        if n_cols < 1:
            self._add_result(
                passed=False,
                level=ValidationLevel.CRITICAL,
                message="No columns in DataFrame",
            )
        else:
            self._add_result(
                passed=True,
                level=ValidationLevel.INFO,
                message=f"DataFrame shape: {n_rows} rows × {n_cols} columns",
                details={"shape": data.shape},
            )

    def _validate_data_types(self, data: pd.DataFrame):
        """Validate data types and check for mixed types."""
        for col in data.columns:
            # Check for mixed types
            if data[col].apply(type).nunique() > 1:
                # Exclude NaN from type checking
                non_null_types = data[col].dropna().apply(type).unique()
                if len(non_null_types) > 1:
                    self._add_result(
                        passed=False,
                        level=ValidationLevel.WARNING,
                        message=f"Mixed data types in column '{col}'",
                        details={"types": [str(t) for t in non_null_types]},
                        affected_columns=[col],
                    )

            # Check for object dtype that might be numeric
            if data[col].dtype == "object":
                try:
                    numeric_conversion = pd.to_numeric(data[col], errors="coerce")
                    if numeric_conversion.notna().sum() > len(data) * 0.9:
                        self._add_result(
                            passed=False,
                            level=ValidationLevel.WARNING,
                            message=f"Column '{col}' appears to be numeric but stored as object",
                            affected_columns=[col],
                        )
                except:
                    pass

    def _validate_missing_values(self, data: pd.DataFrame):
        """Validate missing values in the dataset."""
        missing_stats = data.isnull().sum()
        missing_props = missing_stats / len(data)

        # Report columns with missing values
        for col, missing_count in missing_stats[missing_stats > 0].items():
            prop = missing_props[col]

            if prop > 0.5:
                level = ValidationLevel.ERROR
            elif prop > 0.2:
                level = ValidationLevel.WARNING
            else:
                level = ValidationLevel.INFO

            self._add_result(
                passed=(level == ValidationLevel.INFO),
                level=level,
                message=f"Column '{col}' has {missing_count} ({prop:.1%}) missing values",
                details={
                    "missing_count": int(missing_count),
                    "missing_proportion": float(prop),
                },
                affected_columns=[col],
            )

        # Check for rows with all missing values
        all_missing_rows = data.isnull().all(axis=1)
        if all_missing_rows.any():
            affected_rows = data[all_missing_rows].index.tolist()
            self._add_result(
                passed=False,
                level=ValidationLevel.ERROR,
                message=f"{len(affected_rows)} rows have all missing values",
                affected_rows=affected_rows,
            )

    def _validate_duplicates(self, data: pd.DataFrame):
        """Check for duplicate rows."""
        duplicates = data.duplicated()
        n_duplicates = duplicates.sum()

        if n_duplicates > 0:
            duplicate_indices = data[duplicates].index.tolist()
            self._add_result(
                passed=False,
                level=ValidationLevel.WARNING,
                message=f"Found {n_duplicates} duplicate rows",
                details={"n_duplicates": int(n_duplicates)},
                affected_rows=duplicate_indices[:10],  # Show first 10
            )
        else:
            self._add_result(
                passed=True,
                level=ValidationLevel.INFO,
                message="No duplicate rows found",
            )

    def _validate_statistical_properties(
        self, data: pd.DataFrame, feature_columns: Optional[List[str]] = None
    ):
        """Validate statistical properties of numeric features."""
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if feature_columns:
            numeric_cols = [col for col in numeric_cols if col in feature_columns]

        for col in numeric_cols:
            col_data = data[col].dropna()

            if len(col_data) < 2:
                continue

            # Check for zero variance
            if col_data.std() == 0:
                self._add_result(
                    passed=False,
                    level=ValidationLevel.ERROR,
                    message=f"Column '{col}' has zero variance (constant values)",
                    affected_columns=[col],
                )
                continue

            # Check for extreme skewness
            skewness = stats.skew(col_data)
            if abs(skewness) > 2:
                self._add_result(
                    passed=False,
                    level=ValidationLevel.WARNING,
                    message=f"Column '{col}' is highly skewed (skewness: {skewness:.2f})",
                    details={"skewness": float(skewness)},
                    affected_columns=[col],
                )

            # Check for outliers using IQR method
            Q1 = col_data.quantile(0.25)
            Q3 = col_data.quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((col_data < (Q1 - 3 * IQR)) | (col_data > (Q3 + 3 * IQR))).sum()

            if outliers > 0:
                outlier_prop = outliers / len(col_data)
                self._add_result(
                    passed=(outlier_prop < 0.05),
                    level=(
                        ValidationLevel.INFO
                        if outlier_prop < 0.05
                        else ValidationLevel.WARNING
                    ),
                    message=f"Column '{col}' has {outliers} ({outlier_prop:.1%}) extreme outliers",
                    details={
                        "n_outliers": int(outliers),
                        "outlier_proportion": float(outlier_prop),
                    },
                    affected_columns=[col],
                )

    def _validate_target(self, data: pd.DataFrame, target_column: str):
        """Validate target variable."""
        if target_column not in data.columns:
            self._add_result(
                passed=False,
                level=ValidationLevel.CRITICAL,
                message=f"Target column '{target_column}' not found in data",
            )
            return

        target = data[target_column]

        # Check for missing values in target
        if target.isnull().any():
            n_missing = target.isnull().sum()
            self._add_result(
                passed=False,
                level=ValidationLevel.ERROR,
                message=f"Target column has {n_missing} missing values",
                affected_columns=[target_column],
            )

        # For classification, check class balance
        if target.dtype in ["object", "category"] or target.nunique() < 20:
            value_counts = target.value_counts()
            min_class_size = value_counts.min()
            max_class_size = value_counts.max()
            imbalance_ratio = max_class_size / min_class_size

            if imbalance_ratio > 10:
                self._add_result(
                    passed=False,
                    level=ValidationLevel.WARNING,
                    message=f"Severe class imbalance detected (ratio: {imbalance_ratio:.1f}:1)",
                    details={
                        "class_distribution": value_counts.to_dict(),
                        "imbalance_ratio": float(imbalance_ratio),
                    },
                    affected_columns=[target_column],
                )

    def _validate_features(self, data: pd.DataFrame, feature_columns: List[str]):
        """Validate specific features."""
        # Check if all specified features exist
        missing_features = set(feature_columns) - set(data.columns)
        if missing_features:
            self._add_result(
                passed=False,
                level=ValidationLevel.ERROR,
                message=f"Missing features: {missing_features}",
                affected_columns=list(missing_features),
            )

        # Check for high correlation between features
        numeric_features = [
            col
            for col in feature_columns
            if col in data.columns and data[col].dtype in [np.number]
        ]

        if len(numeric_features) > 1:
            corr_matrix = data[numeric_features].corr().abs()
            high_corr_pairs = []

            for i in range(len(numeric_features)):
                for j in range(i + 1, len(numeric_features)):
                    if corr_matrix.iloc[i, j] > 0.95:
                        high_corr_pairs.append(
                            (
                                numeric_features[i],
                                numeric_features[j],
                                corr_matrix.iloc[i, j],
                            )
                        )

            if high_corr_pairs:
                self._add_result(
                    passed=False,
                    level=ValidationLevel.WARNING,
                    message=f"Found {len(high_corr_pairs)} highly correlated feature pairs (>0.95)",
                    details={"pairs": high_corr_pairs[:5]},  # Show first 5
                )

    def add_custom_rule(
        self, name: str, rule_func: Callable[[pd.DataFrame], ValidationResult]
    ):
        """
        Add a custom validation rule.

        Args:
            name: Name of the rule
            rule_func: Function that takes DataFrame and returns ValidationResult
        """
        self.validation_rules[name] = rule_func

    def _apply_custom_rules(self, data: pd.DataFrame):
        """Apply all custom validation rules."""
        for rule_name, rule_func in self.validation_rules.items():
            try:
                result = rule_func(data)
                self.validation_results.append(result)
            except Exception as e:
                self._add_result(
                    passed=False,
                    level=ValidationLevel.ERROR,
                    message=f"Custom rule '{rule_name}' failed: {str(e)}",
                )

    def _add_result(self, passed: bool, level: ValidationLevel, message: str, **kwargs):
        """Add a validation result."""
        result = ValidationResult(passed=passed, level=level, message=message, **kwargs)
        self.validation_results.append(result)

        # Log the result
        log_func = getattr(logger, level.value, logger.info)
        log_func(message)

    def _determine_overall_status(self) -> bool:
        """Determine if validation passed overall."""
        for result in self.validation_results:
            if result.level == ValidationLevel.CRITICAL:
                return False
            if result.level == ValidationLevel.ERROR:
                return False
            if self.strict_mode and result.level == ValidationLevel.WARNING:
                return False
        return True

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of validation results."""
        summary = {
            "total_checks": len(self.validation_results),
            "passed": sum(1 for r in self.validation_results if r.passed),
            "failed": sum(1 for r in self.validation_results if not r.passed),
            "by_level": {},
        }

        for level in ValidationLevel:
            count = sum(1 for r in self.validation_results if r.level == level)
            summary["by_level"][level.value] = count

        return summary

    def generate_report(self, output_path: Optional[str] = None) -> str:
        """
        Generate a detailed validation report.

        Args:
            output_path: Path to save the report (optional)

        Returns:
            Report as string
        """
        report = []
        report.append("=" * 60)
        report.append("DATA VALIDATION REPORT")
        report.append("=" * 60)

        summary = self.get_summary()
        report.append(f"\nTotal Checks: {summary['total_checks']}")
        report.append(f"Passed: {summary['passed']}")
        report.append(f"Failed: {summary['failed']}")

        report.append("\nResults by Level:")
        for level, count in summary["by_level"].items():
            report.append(f"  {level.upper()}: {count}")

        report.append("\n" + "-" * 60)
        report.append("DETAILED RESULTS")
        report.append("-" * 60)

        for i, result in enumerate(self.validation_results, 1):
            report.append(f"\n[{i}] {result.level.value.upper()}: {result.message}")

            if result.details:
                report.append(f"    Details: {result.details}")

            if result.affected_columns:
                report.append(f"    Affected columns: {result.affected_columns}")

            if result.affected_rows and len(result.affected_rows) <= 10:
                report.append(f"    Affected rows: {result.affected_rows}")
            elif result.affected_rows:
                report.append(
                    f"    Affected rows: {result.affected_rows[:10]} (and {len(result.affected_rows) - 10} more)"
                )

        report_str = "\n".join(report)

        if output_path:
            with open(output_path, "w") as f:
                f.write(report_str)
            logger.info(f"Validation report saved to {output_path}")

        return report_str


class DataProfiler:
    """
    Generate comprehensive data profiles for understanding dataset characteristics.
    """

    @staticmethod
    def profile_dataset(
        data: pd.DataFrame, target_column: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive profile of the dataset.

        Args:
            data: DataFrame to profile
            target_column: Name of target column (if applicable)

        Returns:
            Dictionary containing dataset profile
        """
        profile = {
            "basic_info": DataProfiler._get_basic_info(data),
            "numeric_summary": DataProfiler._get_numeric_summary(data),
            "categorical_summary": DataProfiler._get_categorical_summary(data),
            "missing_values": DataProfiler._get_missing_value_summary(data),
            "correlations": DataProfiler._get_correlation_summary(data),
        }

        if target_column and target_column in data.columns:
            profile["target_analysis"] = DataProfiler._analyze_target(
                data, target_column
            )

        return profile

    @staticmethod
    def _get_basic_info(data: pd.DataFrame) -> Dict[str, Any]:
        """Get basic dataset information."""
        return {
            "n_rows": len(data),
            "n_columns": len(data.columns),
            "memory_usage_mb": data.memory_usage(deep=True).sum() / 1024 / 1024,
            "column_types": data.dtypes.value_counts().to_dict(),
        }

    @staticmethod
    def _get_numeric_summary(data: pd.DataFrame) -> Dict[str, Any]:
        """Get summary statistics for numeric columns."""
        numeric_data = data.select_dtypes(include=[np.number])

        if numeric_data.empty:
            return {}

        summary = {}
        for col in numeric_data.columns:
            col_data = numeric_data[col].dropna()

            if len(col_data) > 0:
                summary[col] = {
                    "mean": float(col_data.mean()),
                    "std": float(col_data.std()),
                    "min": float(col_data.min()),
                    "25%": float(col_data.quantile(0.25)),
                    "50%": float(col_data.quantile(0.50)),
                    "75%": float(col_data.quantile(0.75)),
                    "max": float(col_data.max()),
                    "skewness": float(stats.skew(col_data)),
                    "kurtosis": float(stats.kurtosis(col_data)),
                }

        return summary

    @staticmethod
    def _get_categorical_summary(data: pd.DataFrame) -> Dict[str, Any]:
        """Get summary for categorical columns."""
        categorical_data = data.select_dtypes(include=["object", "category"])

        if categorical_data.empty:
            return {}

        summary = {}
        for col in categorical_data.columns:
            value_counts = data[col].value_counts()
            summary[col] = {
                "n_unique": int(data[col].nunique()),
                "top_values": value_counts.head(10).to_dict(),
                "mode": value_counts.index[0] if len(value_counts) > 0 else None,
            }

        return summary

    @staticmethod
    def _get_missing_value_summary(data: pd.DataFrame) -> Dict[str, Any]:
        """Get missing value summary."""
        missing_counts = data.isnull().sum()
        missing_props = missing_counts / len(data)

        return {
            "total_missing": int(missing_counts.sum()),
            "columns_with_missing": int((missing_counts > 0).sum()),
            "missing_by_column": {
                col: {"count": int(count), "proportion": float(missing_props[col])}
                for col, count in missing_counts[missing_counts > 0].items()
            },
        }

    @staticmethod
    def _get_correlation_summary(data: pd.DataFrame) -> Dict[str, Any]:
        """Get correlation summary for numeric features."""
        numeric_data = data.select_dtypes(include=[np.number])

        if numeric_data.shape[1] < 2:
            return {}

        corr_matrix = numeric_data.corr().abs()

        # Find highly correlated pairs
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                if corr_value > 0.8:
                    high_corr_pairs.append(
                        {
                            "feature_1": corr_matrix.columns[i],
                            "feature_2": corr_matrix.columns[j],
                            "correlation": float(corr_value),
                        }
                    )

        return {
            "n_numeric_features": len(numeric_data.columns),
            "high_correlation_pairs": high_corr_pairs,
        }

    @staticmethod
    def _analyze_target(data: pd.DataFrame, target_column: str) -> Dict[str, Any]:
        """Analyze target variable."""
        target = data[target_column]

        analysis = {
            "dtype": str(target.dtype),
            "missing_values": int(target.isnull().sum()),
            "unique_values": int(target.nunique()),
        }

        if target.dtype in [np.number]:
            analysis["distribution"] = {
                "mean": float(target.mean()),
                "std": float(target.std()),
                "min": float(target.min()),
                "max": float(target.max()),
                "skewness": float(stats.skew(target.dropna())),
            }
        else:
            value_counts = target.value_counts()
            analysis["distribution"] = value_counts.head(10).to_dict()
            analysis["class_balance"] = {
                "n_classes": len(value_counts),
                "imbalance_ratio": (
                    float(value_counts.max() / value_counts.min())
                    if len(value_counts) > 1
                    else 1.0
                ),
            }

        return analysis
