"""
pipelines/__init__.py
End-to-end machine learning pipelines for the Universal Data Science Toolkit.
"""

from .experiment import (ExperimentComparer, ExperimentConfig,
                         ExperimentContext, ExperimentStatus,
                         ExperimentTracker, MLflowCompatibleTracker, MLFlowRun,
                         MLFlowTracker, WandbArtifact, WandbRun, WandbTracker)
from .inference import (BatchPredictor, InferenceConfig, InferencePipeline,
                        ModelServer, PredictionMode, StreamingPredictor)
from .training import (AutoMLPipeline, CrossValidator, HyperparameterTuner,
                       ModelTrainer, TrainingConfig, TrainingPipeline)

__all__ = [
    # Training pipeline
    "TrainingPipeline",
    "TrainingConfig",
    "ModelTrainer",
    "CrossValidator",
    "HyperparameterTuner",
    "AutoMLPipeline",
    # Inference pipeline
    "InferencePipeline",
    "InferenceConfig",
    "PredictionMode",
    "BatchPredictor",
    "StreamingPredictor",
    "ModelServer",
    # Experiment tracking
    "ExperimentTracker",
    "ExperimentConfig",
    "ExperimentStatus",
    "ExperimentContext",
    "MLFlowTracker",
    "MLFlowRun",
    "WandbTracker",
    "WandbRun",
    "WandbArtifact",
    "ExperimentComparer",
    "MLflowCompatibleTracker",
]

__version__ = "1.0.0"
