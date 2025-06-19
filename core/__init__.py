"""
core/__init__.py
Core data processing utilities for the Universal Data Science Toolkit.
"""

from .data_loader import (DataLoader, DatasetSplitter, GeospatialDataLoader,
                          ModelLoader, TabularDataLoader)
from .feature_engineering import (FeatureEngineer, FeatureSelector,
                                  InteractionFeatures, PolynomialFeatures)
from .preprocessing import (DataPreprocessor, FeatureTransformer,
                            MissingValueHandler, OutlierHandler)
from .validation import (DataProfiler, DataValidator, ValidationLevel,
                         ValidationResult)

__all__ = [
    # Data loading
    "DataLoader",
    "TabularDataLoader",
    "GeospatialDataLoader",
    "ModelLoader",
    "DatasetSplitter",
    # Preprocessing
    "DataPreprocessor",
    "FeatureTransformer",
    "OutlierHandler",
    "MissingValueHandler",
    # Feature engineering
    "FeatureSelector",
    "FeatureEngineer",
    "InteractionFeatures",
    "PolynomialFeatures",
    # Validation
    "DataValidator",
    "ValidationLevel",
    "ValidationResult",
    "DataProfiler",
]

__version__ = "1.0.0"
