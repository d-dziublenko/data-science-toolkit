"""
evaluation/__init__.py
Model evaluation and analysis tools for the Universal Data Science Toolkit.
"""

from .metrics import (
    ModelEvaluator,
    RegressionMetrics,
    ClassificationMetrics,
    CustomMetric,
    MetricCalculator,
    ThresholdOptimizer
)

from .visualization import (
    ModelVisualizer,
    PlotConfig,
    PerformancePlotter,
    FeatureImportancePlotter,
    ResidualPlotter,
    ConfusionMatrixPlotter
)

from .uncertainty import (
    UncertaintyEstimator,
    UncertaintyQuantifier,
    BootstrapUncertainty,
    BayesianUncertainty,
    EnsembleUncertainty,
    PredictionInterval,
    CalibrationAnalyzer
)

from .drift import (
    DataDriftDetector,
    DriftType,
    DriftResult,
    StatisticalDriftDetector,
    DistanceDriftDetector,
    ModelPerformanceDriftDetector
)

__all__ = [
    # Metrics
    'ModelEvaluator',
    'RegressionMetrics',
    'ClassificationMetrics',
    'CustomMetric',
    'MetricCalculator',
    'ThresholdOptimizer',
    
    # Visualization
    'ModelVisualizer',
    'PlotConfig',
    'PerformancePlotter',
    'FeatureImportancePlotter',
    'ResidualPlotter',
    'ConfusionMatrixPlotter',
    
    # Uncertainty
    'UncertaintyEstimator',
    'UncertaintyQuantifier',
    'BootstrapUncertainty',
    'BayesianUncertainty',
    'EnsembleUncertainty',
    'PredictionInterval',
    'CalibrationAnalyzer',
    
    # Drift detection
    'DataDriftDetector',
    'DriftType',
    'DriftResult',
    'StatisticalDriftDetector',
    'DistanceDriftDetector',
    'ModelPerformanceDriftDetector'
]

__version__ = '1.0.0'