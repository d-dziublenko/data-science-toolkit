"""
Universal Data Science Toolkit
==============================

A comprehensive Python toolkit for data science and machine learning projects.

This toolkit provides:
- Universal data loading utilities
- Advanced preprocessing and feature engineering
- Multiple model implementations and ensemble methods
- Comprehensive evaluation metrics and visualization
- End-to-end pipelines for training and inference
- Experiment tracking and management

Basic Usage:
-----------
>>> from data_science_toolkit import TrainingPipeline
>>> pipeline = TrainingPipeline(task_type='classification')
>>> results = pipeline.run('data.csv', target_column='target')

>>> from data_science_toolkit.core import TabularDataLoader
>>> loader = TabularDataLoader()
>>> data = loader.load('data.csv')

>>> from data_science_toolkit.models import AutoML
>>> automl = AutoML(task_type='regression')
>>> automl.fit(X_train, y_train)
"""

__version__ = '1.0.0'
__author__ = 'Dmytro Dziublenko'
__email__ = 'd.dziublenko@gmail.com'

# Import main components for easy access
from .core import (
    TabularDataLoader,
    DataPreprocessor,
    FeatureSelector,
    DataValidator
)

from .models import (
    AutoML,
    ModelStacker,
    ModelBlender
)

from .evaluation import (
    ModelEvaluator,
    ModelVisualizer,
    UncertaintyQuantifier,
    DataDriftDetector
)

from .pipelines import (
    TrainingPipeline,
    InferencePipeline,
    ExperimentTracker
)

from .utils import (
    setup_logger,
    FileHandler,
    ParallelProcessor
)

# Define what's available at package level
__all__ = [
    # Version info
    '__version__',
    '__author__',
    '__email__',
    
    # Core functionality
    'TabularDataLoader',
    'DataPreprocessor', 
    'FeatureSelector',
    'DataValidator',
    
    # Models
    'AutoML',
    'ModelStacker',
    'ModelBlender',
    
    # Evaluation
    'ModelEvaluator',
    'ModelVisualizer',
    'UncertaintyQuantifier',
    'DataDriftDetector',
    
    # Pipelines
    'TrainingPipeline',
    'InferencePipeline',
    'ExperimentTracker',
    
    # Utilities
    'setup_logger',
    'FileHandler',
    'ParallelProcessor'
]

# Package initialization
import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())