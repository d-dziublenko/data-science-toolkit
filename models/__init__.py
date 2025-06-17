"""
models/__init__.py
Machine learning models and base classes for the Universal Data Science Toolkit.
"""

from .base import (
    BaseModel,
    BaseRegressor,
    BaseClassifier,
    AutoML,
    EnsembleModel
)

from .ensemble import (
    VotingEnsemble,
    StackingEnsemble,
    BaggingEnsemble,
    ModelStacker,
    ModelBlender,
    WeightedEnsemble
)

from .neural import (
    NeuralNetworkBase,
    FeedForwardNetwork,
    ConvolutionalNetwork,
    RecurrentNetwork,
    AutoEncoder,
    NeuralNetworkRegressor,
    NeuralNetworkClassifier
)

from .transformers import (
    TargetTransformer,
    LogTransformer,
    BoxCoxTransformer,
    YeoJohnsonTransformer,
    QuantileTransformer,
    PowerTransformer
)

__all__ = [
    # Base models
    'BaseModel',
    'BaseRegressor',
    'BaseClassifier',
    'AutoML',
    'EnsembleModel',
    
    # Ensemble methods
    'VotingEnsemble',
    'StackingEnsemble',
    'BaggingEnsemble',
    'ModelStacker',
    'ModelBlender',
    'WeightedEnsemble',
    
    # Neural networks
    'NeuralNetworkBase',
    'FeedForwardNetwork',
    'ConvolutionalNetwork',
    'RecurrentNetwork',
    'AutoEncoder',
    'NeuralNetworkRegressor',
    'NeuralNetworkClassifier',
    
    # Transformers
    'TargetTransformer',
    'LogTransformer',
    'BoxCoxTransformer',
    'YeoJohnsonTransformer',
    'QuantileTransformer',
    'PowerTransformer'
]

__version__ = '1.0.0'