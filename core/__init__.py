"""
core/__init__.py
Core data processing utilities for the Universal Data Science Toolkit.
"""

from .data_loader import (
    DataLoader,
    TabularDataLoader,
    GeospatialDataLoader,
    ModelLoader,
    DatasetSplitter
)

from .preprocessing import (
    DataPreprocessor,
    FeatureTransformer,
    OutlierHandler,
    MissingValueHandler
)

from .feature_engineering import (
    FeatureSelector,
    FeatureEngineer,
    InteractionFeatures,
    PolynomialFeatures
)

from .validation import (
    DataValidator,
    ValidationLevel,
    ValidationResult,
    DataProfiler
)

__all__ = [
    # Data loading
    'DataLoader',
    'TabularDataLoader',
    'GeospatialDataLoader',
    'ModelLoader',
    'DatasetSplitter',
    
    # Preprocessing
    'DataPreprocessor',
    'FeatureTransformer',
    'OutlierHandler',
    'MissingValueHandler',
    
    # Feature engineering
    'FeatureSelector',
    'FeatureEngineer',
    'InteractionFeatures',
    'PolynomialFeatures',
    
    # Validation
    'DataValidator',
    'ValidationLevel',
    'ValidationResult',
    'DataProfiler'
]

__version__ = '1.0.0'