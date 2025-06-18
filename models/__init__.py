"""
models/__init__.py
Machine learning models and base classes for the Universal Data Science Toolkit.
"""

from .base import (
    BaseModel,
    MetaModel,
    BaseRegressor,
    BaseClassifier,
    AutoML,
    SimpleLinearModel
)

from .ensemble import (
    EnsembleModel,
    BaseEnsemble,
    VotingEnsemble,
    StackingEnsemble,
    BaggingEnsemble,
    ModelStacker,
    ModelBlender,
    WeightedEnsemble
)

from .neural import (
    NeuralNetworkBase,
    DNNRegressor,
    DNNClassifier,
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
    RankTransformer,
    CompositeTransformer,
    AutoTargetTransformer,
    RobustTargetTransformer,
    TargetEncoder,
    PowerTransformer
)

__all__ = [
    # Base models
    'BaseModel',
    'MetaModel',
    'BaseRegressor',
    'BaseClassifier',
    'AutoML',
    'SimpleLinearModel',
    
    # Ensemble methods
    'EnsembleModel',
    'BaseEnsemble',
    'VotingEnsemble',
    'StackingEnsemble',
    'BaggingEnsemble',
    'ModelStacker',
    'ModelBlender',
    'WeightedEnsemble',
    
    # Neural networks
    'NeuralNetworkBase',
    'DNNRegressor',
    'DNNClassifier',
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
    'RankTransformer',
    'CompositeTransformer',
    'AutoTargetTransformer',
    'RobustTargetTransformer',
    'TargetEncoder',
    'PowerTransformer'
]

__version__ = '1.0.0'