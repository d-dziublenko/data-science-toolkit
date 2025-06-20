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
- Parallel processing and utility functions

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

Author: Dmytro Dziublenko
Email: d.dziublenko@gmail.com
License: AGPL-3.0
GitHub: https://github.com/d-dziublenko/data-science-toolkit
"""

__version__ = '1.0.0'
__author__ = 'Dmytro Dziublenko'
__email__ = 'd.dziublenko@gmail.com'
__license__ = 'AGPL-3.0'

# Import core components
from .core import (
    # Data loading
    DataLoader,
    TabularDataLoader,
    GeospatialDataLoader,
    ModelLoader,
    DatasetSplitter,
    # Preprocessing
    DataPreprocessor,
    FeatureTransformer,
    OutlierHandler,
    MissingValueHandler,
    # Feature engineering
    FeatureSelector,
    FeatureEngineer,
    InteractionFeatures,
    PolynomialFeatures,
    # Validation
    DataValidator,
    ValidationLevel,
    ValidationResult,
    DataProfiler
)

# Import model components
from .models import (
    # Base models
    BaseModel,
    MetaModel,
    BaseRegressor,
    BaseClassifier,
    AutoML,
    SimpleLinearModel,
    # Ensemble methods
    EnsembleModel,
    BaseEnsemble,
    VotingEnsemble,
    StackingEnsemble,
    BaggingEnsemble,
    ModelStacker,
    ModelBlender,
    WeightedEnsemble,
    # Neural networks
    NeuralNetworkBase,
    DNNRegressor,
    DNNClassifier,
    FeedForwardNetwork,
    ConvolutionalNetwork,
    RecurrentNetwork,
    AutoEncoder,
    NeuralNetworkRegressor,
    NeuralNetworkClassifier,
    # Transformers
    TargetTransformer,
    LogTransformer,
    BoxCoxTransformer,
    YeoJohnsonTransformer,
    QuantileTransformer,
    RankTransformer,
    CompositeTransformer,
    AutoTargetTransformer,
    RobustTargetTransformer,
    TargetEncoder,
    PowerTransformer
)

# Import evaluation components
from .evaluation import (
    # Metrics
    ModelEvaluator,
    RegressionMetrics,
    ClassificationMetrics,
    CustomMetric,
    MetricCalculator,
    ThresholdOptimizer,
    # Visualization
    ModelVisualizer,
    PlotConfig,
    PerformancePlotter,
    FeatureImportancePlotter,
    ResidualPlotter,
    ConfusionMatrixPlotter,
    # Uncertainty
    UncertaintyEstimator,
    UncertaintyQuantifier,
    BootstrapUncertainty,
    BayesianUncertainty,
    EnsembleUncertainty,
    PredictionInterval,
    CalibrationAnalyzer,
    # Drift detection
    DataDriftDetector,
    DriftType,
    DriftResult,
    StatisticalDriftDetector,
    DistanceDriftDetector,
    ModelPerformanceDriftDetector
)

# Import pipeline components
from .pipelines import (
    # Training pipeline
    TrainingPipeline,
    TrainingConfig,
    ModelTrainer,
    CrossValidator,
    HyperparameterTuner,
    AutoMLPipeline,
    # Inference pipeline
    InferencePipeline,
    InferenceConfig,
    PredictionMode,
    BatchPredictor,
    StreamingPredictor,
    ModelServer,
    # Experiment tracking
    ExperimentTracker,
    ExperimentConfig,
    ExperimentStatus,
    ExperimentContext,
    MLFlowTracker,
    MLFlowRun,
    WandbTracker,
    WandbRun,
    WandbArtifact,
    ExperimentComparer,
    MLflowCompatibleTracker
)

# Import utility components
from .utils import (
    # File I/O
    FileHandler,
    DataReader,
    DataWriter,
    ModelSerializer,
    ConfigLoader,
    DataCache,
    CheckpointManager,
    save_object,
    load_object,
    save_config,
    load_config,
    create_archive,
    # Parallel processing
    BackendType,
    ParallelProcessor,
    ParallelConfig,
    ParallelBatch,
    ProgressParallel,
    ChunkProcessor,
    DistributedProcessor,
    SharedMemoryArray,
    get_n_jobs,
    parallel_apply,
    parallel_map,
    parallel_groupby_apply,
    chunked_parallel_process,
    parallel_read_csv,
    parallel_save_partitions,
    # CLI utilities
    ColoredFormatter,
    CLIParser,
    CLIApplication,
    CommandHandler,
    ArgumentValidator,
    ConfigArgumentParser,
    ProgressBar,
    create_cli_app,
    run_command,
    spinner,
    prompt_choice,
    print_table,
    print_summary,
    create_cli_command,
    format_bytes,
    format_duration,
    print_error,
    print_warning,
    print_success,
    print_info,
    # Logging
    LogFormat,
    LogConfig,
    StructuredFormatter,
    StructuredLogger,
    ProgressLogger,
    ExperimentLogger,
    log_execution_time,
    log_memory_usage,
    log_context,
    log_exceptions,
    redirect_warnings,
    setup_logger,
    get_logger
)

# Define what's available at package level
__all__ = [
    # Version info
    '__version__',
    '__author__',
    '__email__',
    '__license__',
    
    # Core functionality
    'DataLoader',
    'TabularDataLoader',
    'GeospatialDataLoader',
    'ModelLoader',
    'DatasetSplitter',
    'DataPreprocessor',
    'FeatureTransformer',
    'OutlierHandler',
    'MissingValueHandler',
    'FeatureSelector',
    'FeatureEngineer',
    'InteractionFeatures',
    'PolynomialFeatures',
    'DataValidator',
    'ValidationLevel',
    'ValidationResult',
    'DataProfiler',
    
    # Models
    'BaseModel',
    'MetaModel',
    'BaseRegressor',
    'BaseClassifier',
    'AutoML',
    'SimpleLinearModel',
    'EnsembleModel',
    'BaseEnsemble',
    'VotingEnsemble',
    'StackingEnsemble',
    'BaggingEnsemble',
    'ModelStacker',
    'ModelBlender',
    'WeightedEnsemble',
    'NeuralNetworkBase',
    'DNNRegressor',
    'DNNClassifier',
    'FeedForwardNetwork',
    'ConvolutionalNetwork',
    'RecurrentNetwork',
    'AutoEncoder',
    'NeuralNetworkRegressor',
    'NeuralNetworkClassifier',
    'TargetTransformer',
    'LogTransformer',
    'BoxCoxTransformer',
    'YeoJohnsonTransformer',
    'QuantileTransformer',
    'RankTransformer',
    'CompositeTransformer',
    'AutoTargetTransformer',
    'RobustTargetTransformer',
    'TargetEncoder',
    'PowerTransformer',
    
    # Evaluation
    'ModelEvaluator',
    'RegressionMetrics',
    'ClassificationMetrics',
    'CustomMetric',
    'MetricCalculator',
    'ThresholdOptimizer',
    'ModelVisualizer',
    'PlotConfig',
    'PerformancePlotter',
    'FeatureImportancePlotter',
    'ResidualPlotter',
    'ConfusionMatrixPlotter',
    'UncertaintyEstimator',
    'UncertaintyQuantifier',
    'BootstrapUncertainty',
    'BayesianUncertainty',
    'EnsembleUncertainty',
    'PredictionInterval',
    'CalibrationAnalyzer',
    'DataDriftDetector',
    'DriftType',
    'DriftResult',
    'StatisticalDriftDetector',
    'DistanceDriftDetector',
    'ModelPerformanceDriftDetector',
    
    # Pipelines
    'TrainingPipeline',
    'TrainingConfig',
    'ModelTrainer',
    'CrossValidator',
    'HyperparameterTuner',
    'AutoMLPipeline',
    'InferencePipeline',
    'InferenceConfig',
    'PredictionMode',
    'BatchPredictor',
    'StreamingPredictor',
    'ModelServer',
    'ExperimentTracker',
    'ExperimentConfig',
    'ExperimentStatus',
    'ExperimentContext',
    'MLFlowTracker',
    'MLFlowRun',
    'WandbTracker',
    'WandbRun',
    'WandbArtifact',
    'ExperimentComparer',
    'MLflowCompatibleTracker',
    
    # Utilities
    'FileHandler',
    'DataReader',
    'DataWriter',
    'ModelSerializer',
    'ConfigLoader',
    'DataCache',
    'CheckpointManager',
    'save_object',
    'load_object',
    'save_config',
    'load_config',
    'create_archive',
    'BackendType',
    'ParallelProcessor',
    'ParallelConfig',
    'ParallelBatch',
    'ProgressParallel',
    'ChunkProcessor',
    'DistributedProcessor',
    'SharedMemoryArray',
    'get_n_jobs',
    'parallel_apply',
    'parallel_map',
    'parallel_groupby_apply',
    'chunked_parallel_process',
    'parallel_read_csv',
    'parallel_save_partitions',
    'ColoredFormatter',
    'CLIParser',
    'CLIApplication',
    'CommandHandler',
    'ArgumentValidator',
    'ConfigArgumentParser',
    'ProgressBar',
    'create_cli_app',
    'run_command',
    'spinner',
    'prompt_choice',
    'print_table',
    'print_summary',
    'create_cli_command',
    'format_bytes',
    'format_duration',
    'print_error',
    'print_warning',
    'print_success',
    'print_info',
    'LogFormat',
    'LogConfig',
    'StructuredFormatter',
    'StructuredLogger',
    'ProgressLogger',
    'ExperimentLogger',
    'log_execution_time',
    'log_memory_usage',
    'log_context',
    'log_exceptions',
    'redirect_warnings',
    'setup_logger',
    'get_logger'
]

# Import necessary types and modules for convenience functions
from typing import Dict, Any
import pandas as pd

# Quick start convenience functions
def quick_train(data_path: str, 
                target_column: str,
                task_type: str = 'auto',
                test_size: float = 0.2,
                **kwargs) -> Dict[str, Any]:
    """
    Quick training function for rapid prototyping.
    
    Args:
        data_path: Path to data file
        target_column: Name of target column
        task_type: 'regression', 'classification', or 'auto'
        test_size: Test set size
        **kwargs: Additional arguments for TrainingPipeline
        
    Returns:
        Dictionary with results
    """
    pipeline = TrainingPipeline(task_type=task_type)
    return pipeline.run(
        data_path=data_path,
        target_column=target_column,
        test_size=test_size,
        **kwargs
    )

def quick_predict(model_path: str,
                  data_path: str,
                  output_path: str = 'predictions.csv',
                  **kwargs) -> pd.DataFrame:
    """
    Quick prediction function for making predictions.
    
    Args:
        model_path: Path to saved model
        data_path: Path to data file
        output_path: Path to save predictions
        **kwargs: Additional arguments for InferencePipeline
        
    Returns:
        DataFrame with predictions
    """
    config = InferenceConfig(
        model_path=model_path,
        output_format='csv'
    )
    pipeline = InferencePipeline(config)
    predictions = pipeline.predict(data_path)
    
    # Save predictions
    predictions.to_csv(output_path, index=False)
    return predictions

# Package initialization
import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())

# Display package info when imported
logger = logging.getLogger(__name__)
logger.info(f"Universal Data Science Toolkit v{__version__}")
logger.info(f"Author: {__author__} ({__email__})")
logger.info("Ready for data science and machine learning tasks!")