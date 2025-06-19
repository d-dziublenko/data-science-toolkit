"""
models/__init__.py
Machine learning models and base classes for the Universal Data Science Toolkit.
"""

from .base import (AutoML, BaseClassifier, BaseModel, BaseRegressor, MetaModel,
                   SimpleLinearModel)
from .ensemble import (BaggingEnsemble, BaseEnsemble, EnsembleModel,
                       ModelBlender, ModelStacker, StackingEnsemble,
                       VotingEnsemble, WeightedEnsemble)
from .neural import (AutoEncoder, ConvolutionalNetwork, DNNClassifier,
                     DNNRegressor, FeedForwardNetwork, NeuralNetworkBase,
                     NeuralNetworkClassifier, NeuralNetworkRegressor,
                     RecurrentNetwork)
from .transformers import (AutoTargetTransformer, BoxCoxTransformer,
                           CompositeTransformer, LogTransformer,
                           PowerTransformer, QuantileTransformer,
                           RankTransformer, RobustTargetTransformer,
                           TargetEncoder, TargetTransformer,
                           YeoJohnsonTransformer)

__all__ = [
    # Base models
    "BaseModel",
    "MetaModel",
    "BaseRegressor",
    "BaseClassifier",
    "AutoML",
    "SimpleLinearModel",
    # Ensemble methods
    "EnsembleModel",
    "BaseEnsemble",
    "VotingEnsemble",
    "StackingEnsemble",
    "BaggingEnsemble",
    "ModelStacker",
    "ModelBlender",
    "WeightedEnsemble",
    # Neural networks
    "NeuralNetworkBase",
    "DNNRegressor",
    "DNNClassifier",
    "FeedForwardNetwork",
    "ConvolutionalNetwork",
    "RecurrentNetwork",
    "AutoEncoder",
    "NeuralNetworkRegressor",
    "NeuralNetworkClassifier",
    # Transformers
    "TargetTransformer",
    "LogTransformer",
    "BoxCoxTransformer",
    "YeoJohnsonTransformer",
    "QuantileTransformer",
    "RankTransformer",
    "CompositeTransformer",
    "AutoTargetTransformer",
    "RobustTargetTransformer",
    "TargetEncoder",
    "PowerTransformer",
]

__version__ = "1.0.0"
