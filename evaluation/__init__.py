"""
evaluation/__init__.py
Model evaluation and analysis tools for the Universal Data Science Toolkit.
"""

from .drift import (DataDriftDetector, DistanceDriftDetector, DriftResult,
                    DriftType, ModelPerformanceDriftDetector,
                    StatisticalDriftDetector)
from .metrics import (ClassificationMetrics, CustomMetric, MetricCalculator,
                      ModelEvaluator, RegressionMetrics, ThresholdOptimizer)
from .uncertainty import (BayesianUncertainty, BootstrapUncertainty,
                          CalibrationAnalyzer, EnsembleUncertainty,
                          PredictionInterval, UncertaintyEstimator,
                          UncertaintyQuantifier)
from .visualization import (ConfusionMatrixPlotter, FeatureImportancePlotter,
                            ModelVisualizer, PerformancePlotter, PlotConfig,
                            ResidualPlotter)

__all__ = [
    # Metrics
    "ModelEvaluator",
    "RegressionMetrics",
    "ClassificationMetrics",
    "CustomMetric",
    "MetricCalculator",
    "ThresholdOptimizer",
    # Visualization
    "ModelVisualizer",
    "PlotConfig",
    "PerformancePlotter",
    "FeatureImportancePlotter",
    "ResidualPlotter",
    "ConfusionMatrixPlotter",
    # Uncertainty
    "UncertaintyEstimator",
    "UncertaintyQuantifier",
    "BootstrapUncertainty",
    "BayesianUncertainty",
    "EnsembleUncertainty",
    "PredictionInterval",
    "CalibrationAnalyzer",
    # Drift detection
    "DataDriftDetector",
    "DriftType",
    "DriftResult",
    "StatisticalDriftDetector",
    "DistanceDriftDetector",
    "ModelPerformanceDriftDetector",
]

__version__ = "1.0.0"
