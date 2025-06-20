"""
pipelines/__init__.py
End-to-end machine learning pipelines for the Universal Data Science Toolkit.
"""

from .training import (
    TrainingPipeline,
    TrainingConfig,
    ModelTrainer,
    CrossValidator,
    HyperparameterTuner,
    AutoMLPipeline
)

from .inference import (
    InferencePipeline,
    InferenceConfig,
    PredictionMode,
    BatchPredictor,
    StreamingPredictor,
    ModelServer
)

from .experiment import (
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

__all__ = [
    # Training pipeline
    'TrainingPipeline',
    'TrainingConfig',
    'ModelTrainer',
    'CrossValidator',
    'HyperparameterTuner',
    'AutoMLPipeline',
    
    # Inference pipeline
    'InferencePipeline',
    'InferenceConfig',
    'PredictionMode',
    'BatchPredictor',
    'StreamingPredictor',
    'ModelServer',
    
    # Experiment tracking
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
    'MLflowCompatibleTracker'
]

__version__ = '1.0.0'