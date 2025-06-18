"""
models/neural.py
Neural network implementations for deep learning in the universal toolkit.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
import logging
import warnings

# TensorFlow/Keras imports
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, Model
    from tensorflow.keras.callbacks import (
        EarlyStopping, ReduceLROnPlateau, ModelCheckpoint,
        TensorBoard, Callback
    )
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logging.warning("TensorFlow not available. Neural network features will be limited.")

# PyTorch imports
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    logging.warning("PyTorch not available. PyTorch neural network features will be limited.")

logger = logging.getLogger(__name__)


class NeuralNetworkBase:
    """
    Base class for neural network models.
    
    This class provides a unified interface for different neural network
    frameworks (TensorFlow/Keras and PyTorch) and common functionality.
    """
    
    def __init__(self,
                 task_type: str = 'regression',
                 framework: str = 'auto',
                 random_state: int = 42):
        """
        Initialize the neural network base.
        
        Args:
            task_type: Type of ML task ('regression' or 'classification')
            framework: Deep learning framework ('tensorflow', 'pytorch', 'auto')
            random_state: Random seed for reproducibility
        """
        self.task_type = task_type
        self.random_state = random_state
        self.model = None
        self.history = None
        self.is_fitted = False
        
        # Determine framework
        if framework == 'auto':
            if TENSORFLOW_AVAILABLE:
                self.framework = 'tensorflow'
            elif PYTORCH_AVAILABLE:
                self.framework = 'pytorch'
            else:
                raise ImportError("No deep learning framework available. Install TensorFlow or PyTorch.")
        else:
            self.framework = framework
            
        # Set random seeds
        self._set_random_seeds()
    
    def _set_random_seeds(self):
        """Set random seeds for reproducibility."""
        np.random.seed(self.random_state)
        
        if self.framework == 'tensorflow' and TENSORFLOW_AVAILABLE:
            tf.random.set_seed(self.random_state)
        elif self.framework == 'pytorch' and PYTORCH_AVAILABLE:
            torch.manual_seed(self.random_state)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.random_state)


class DNNRegressor(NeuralNetworkBase):
    """
    Deep Neural Network for regression tasks.
    
    This class provides a flexible DNN architecture that can be customized
    with different layer sizes, activation functions, and regularization.
    """
    
    def __init__(self,
                 hidden_layers: List[int] = [128, 64, 32],
                 activation: str = 'relu',
                 dropout_rate: float = 0.2,
                 batch_norm: bool = True,
                 l2_reg: float = 0.01,
                 learning_rate: float = 0.001,
                 **kwargs):
        """
        Initialize the DNN Regressor.
        
        Args:
            hidden_layers: List of hidden layer sizes
            activation: Activation function for hidden layers
            dropout_rate: Dropout rate for regularization
            batch_norm: Whether to use batch normalization
            l2_reg: L2 regularization strength
            learning_rate: Initial learning rate
            **kwargs: Additional arguments for base class
        """
        super().__init__(task_type='regression', **kwargs)
        
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.batch_norm = batch_norm
        self.l2_reg = l2_reg
        self.learning_rate = learning_rate
        
    def build_model(self, input_shape: int) -> Any:
        """
        Build the neural network architecture.
        
        Args:
            input_shape: Number of input features
            
        Returns:
            Compiled model
        """
        if self.framework == 'tensorflow':
            return self._build_keras_model(input_shape)
        else:
            return self._build_pytorch_model(input_shape)
    
    def _build_keras_model(self, input_shape: int) -> keras.Model:
        """Build Keras/TensorFlow model."""
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is not installed")
        
        # Input layer
        inputs = keras.Input(shape=(input_shape,))
        x = inputs
        
        # Hidden layers
        for i, units in enumerate(self.hidden_layers):
            # Dense layer with regularization
            x = layers.Dense(
                units,
                activation=None,
                kernel_regularizer=keras.regularizers.l2(self.l2_reg),
                name=f'dense_{i}'
            )(x)
            
            # Batch normalization
            if self.batch_norm:
                x = layers.BatchNormalization(name=f'bn_{i}')(x)
            
            # Activation
            x = layers.Activation(self.activation, name=f'activation_{i}')(x)
            
            # Dropout
            if self.dropout_rate > 0:
                x = layers.Dropout(self.dropout_rate, name=f'dropout_{i}')(x)
        
        # Output layer
        outputs = layers.Dense(1, name='output')(x)
        
        # Create model
        model = keras.Model(inputs=inputs, outputs=outputs, name='dnn_regressor')
        
        # Compile model
        optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae', 'mse']
        )
        
        return model
    
    def _build_pytorch_model(self, input_shape: int) -> nn.Module:
        """Build PyTorch model."""
        if not PYTORCH_AVAILABLE:
            raise ImportError("PyTorch is not installed")
        
        class DNNModule(nn.Module):
            def __init__(self, input_size, hidden_layers, activation, dropout_rate, batch_norm):
                super(DNNModule, self).__init__()
                
                # Build layers
                self.layers = nn.ModuleList()
                
                # Input to first hidden
                prev_size = input_size
                
                for i, hidden_size in enumerate(hidden_layers):
                    # Add dense layer
                    self.layers.append(nn.Linear(prev_size, hidden_size))
                    
                    # Add batch norm if requested
                    if batch_norm:
                        self.layers.append(nn.BatchNorm1d(hidden_size))
                    
                    # Add activation
                    if activation == 'relu':
                        self.layers.append(nn.ReLU())
                    elif activation == 'tanh':
                        self.layers.append(nn.Tanh())
                    elif activation == 'sigmoid':
                        self.layers.append(nn.Sigmoid())
                    
                    # Add dropout
                    if dropout_rate > 0:
                        self.layers.append(nn.Dropout(dropout_rate))
                    
                    prev_size = hidden_size
                
                # Output layer
                self.layers.append(nn.Linear(prev_size, 1))
            
            def forward(self, x):
                for layer in self.layers:
                    x = layer(x)
                return x
        
        return DNNModule(
            input_size=input_shape,
            hidden_layers=self.hidden_layers,
            activation=self.activation,
            dropout_rate=self.dropout_rate,
            batch_norm=self.batch_norm
        )
    
    def fit(self,
            X: Union[pd.DataFrame, np.ndarray],
            y: Union[pd.Series, np.ndarray],
            validation_split: float = 0.2,
            epochs: int = 100,
            batch_size: int = 32,
            callbacks: Optional[List[Any]] = None,
            verbose: int = 1):
        """
        Train the neural network.
        
        Args:
            X: Training features
            y: Training targets
            validation_split: Fraction of data to use for validation
            epochs: Number of training epochs
            batch_size: Batch size for training
            callbacks: List of callbacks
            verbose: Verbosity level
        """
        # Convert to numpy arrays
        X = np.array(X)
        y = np.array(y)
        
        # Build model if not already built
        if self.model is None:
            self.model = self.build_model(X.shape[1])
        
        if self.framework == 'tensorflow':
            # Set up callbacks
            if callbacks is None:
                callbacks = [
                    EarlyStopping(patience=20, restore_best_weights=True),
                    ReduceLROnPlateau(patience=10, factor=0.5)
                ]
            
            # Train model
            self.history = self.model.fit(
                X, y,
                validation_split=validation_split,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=verbose
            )
        else:
            # PyTorch training
            self._train_pytorch(X, y, validation_split, epochs, batch_size, verbose)
        
        self.is_fitted = True
    
    def _train_pytorch(self, X, y, validation_split, epochs, batch_size, verbose):
        """Train PyTorch model."""
        # Split data
        n_val = int(len(X) * validation_split)
        indices = np.random.permutation(len(X))
        val_indices, train_indices = indices[:n_val], indices[n_val:]
        
        # Create datasets
        X_train, y_train = X[train_indices], y[train_indices]
        X_val, y_val = X[val_indices], y[val_indices]
        
        # Convert to tensors
        X_train = torch.FloatTensor(X_train)
        y_train = torch.FloatTensor(y_train.reshape(-1, 1))
        X_val = torch.FloatTensor(X_val)
        y_val = torch.FloatTensor(y_val.reshape(-1, 1))
        
        # Create data loaders
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Setup optimizer and loss
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()
        
        # Training loop
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val)
                val_loss = criterion(val_outputs, y_val).item()
            
            train_losses.append(train_loss / len(train_loader))
            val_losses.append(val_loss)
            
            if verbose and epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: train_loss={train_losses[-1]:.4f}, val_loss={val_loss:.4f}")
        
        # Store history
        self.history = {
            'loss': train_losses,
            'val_loss': val_losses
        }
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            X: Features to predict on
            
        Returns:
            Predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        X = np.array(X)
        
        if self.framework == 'tensorflow':
            predictions = self.model.predict(X, verbose=0)
        else:
            self.model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X)
                predictions = self.model(X_tensor).numpy()
        
        return predictions.reshape(-1)


class DNNClassifier(NeuralNetworkBase):
    """
    Deep Neural Network for classification tasks.
    
    Similar to DNNRegressor but with classification-specific features
    like softmax output and categorical crossentropy loss.
    """
    
    def __init__(self,
                 hidden_layers: List[int] = [128, 64, 32],
                 activation: str = 'relu',
                 dropout_rate: float = 0.2,
                 batch_norm: bool = True,
                 l2_reg: float = 0.01,
                 learning_rate: float = 0.001,
                 n_classes: Optional[int] = None,
                 **kwargs):
        """
        Initialize the DNN Classifier.
        
        Args:
            hidden_layers: List of hidden layer sizes
            activation: Activation function for hidden layers
            dropout_rate: Dropout rate for regularization
            batch_norm: Whether to use batch normalization
            l2_reg: L2 regularization strength
            learning_rate: Initial learning rate
            n_classes: Number of classes (auto-detected if None)
            **kwargs: Additional arguments for base class
        """
        super().__init__(task_type='classification', **kwargs)
        
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.batch_norm = batch_norm
        self.l2_reg = l2_reg
        self.learning_rate = learning_rate
        self.n_classes = n_classes
        self.label_encoder = None
    
    def _build_keras_model(self, input_shape: int, n_classes: int) -> keras.Model:
        """Build Keras classification model."""
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is not installed")
        
        # Input layer
        inputs = keras.Input(shape=(input_shape,))
        x = inputs
        
        # Hidden layers (similar to regressor)
        for i, units in enumerate(self.hidden_layers):
            x = layers.Dense(
                units,
                activation=None,
                kernel_regularizer=keras.regularizers.l2(self.l2_reg),
                name=f'dense_{i}'
            )(x)
            
            if self.batch_norm:
                x = layers.BatchNormalization(name=f'bn_{i}')(x)
            
            x = layers.Activation(self.activation, name=f'activation_{i}')(x)
            
            if self.dropout_rate > 0:
                x = layers.Dropout(self.dropout_rate, name=f'dropout_{i}')(x)
        
        # Output layer
        if n_classes == 2:
            # Binary classification
            outputs = layers.Dense(1, activation='sigmoid', name='output')(x)
            loss = 'binary_crossentropy'
        else:
            # Multi-class classification
            outputs = layers.Dense(n_classes, activation='softmax', name='output')(x)
            loss = 'sparse_categorical_crossentropy'
        
        # Create and compile model
        model = keras.Model(inputs=inputs, outputs=outputs, name='dnn_classifier')
        
        optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=['accuracy']
        )
        
        return model
    
    def fit(self,
            X: Union[pd.DataFrame, np.ndarray],
            y: Union[pd.Series, np.ndarray],
            **kwargs):
        """
        Train the classifier.
        
        Args:
            X: Training features
            y: Training labels
            **kwargs: Additional arguments passed to parent fit method
        """
        # Convert to numpy arrays
        X = np.array(X)
        y = np.array(y)
        
        # Encode labels if needed
        if y.dtype == 'object' or len(np.unique(y)) != len(y):
            from sklearn.preprocessing import LabelEncoder
            self.label_encoder = LabelEncoder()
            y = self.label_encoder.fit_transform(y)
        
        # Determine number of classes
        if self.n_classes is None:
            self.n_classes = len(np.unique(y))
        
        # Build model
        if self.model is None:
            if self.framework == 'tensorflow':
                self.model = self._build_keras_model(X.shape[1], self.n_classes)
            else:
                raise NotImplementedError("PyTorch classifier not implemented in this example")
        
        # Call parent fit method
        super().fit(X, y, **kwargs)
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Make class predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        X = np.array(X)
        
        if self.framework == 'tensorflow':
            if self.n_classes == 2:
                # Binary classification
                probas = self.model.predict(X, verbose=0)
                predictions = (probas > 0.5).astype(int).reshape(-1)
            else:
                # Multi-class classification
                probas = self.model.predict(X, verbose=0)
                predictions = np.argmax(probas, axis=1)
        else:
            raise NotImplementedError("PyTorch prediction not implemented")
        
        # Decode labels if encoder was used
        if self.label_encoder is not None:
            predictions = self.label_encoder.inverse_transform(predictions)
        
        return predictions
    
    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Get class probabilities."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        X = np.array(X)
        
        if self.framework == 'tensorflow':
            probas = self.model.predict(X, verbose=0)
            if self.n_classes == 2:
                # Convert to two-column format for binary classification
                probas = np.hstack([1 - probas, probas])
        else:
            raise NotImplementedError("PyTorch prediction not implemented")
        
        return probas


class FeedForwardNetwork(NeuralNetworkBase):
    """
    General feedforward neural network with customizable architecture.
    
    This class provides maximum flexibility in designing feedforward networks
    with custom layer configurations, activations, and regularization.
    """
    
    def __init__(self,
                 layer_config: List[Dict[str, Any]],
                 task_type: str = 'regression',
                 optimizer_config: Optional[Dict[str, Any]] = None,
                 loss_config: Optional[Dict[str, Any]] = None,
                 **kwargs):
        """
        Initialize FeedForward Network.
        
        Args:
            layer_config: List of layer configurations
            task_type: Type of task ('regression' or 'classification')
            optimizer_config: Optimizer configuration
            loss_config: Loss function configuration
            **kwargs: Additional arguments for base class
        """
        super().__init__(task_type=task_type, **kwargs)
        
        self.layer_config = layer_config
        self.optimizer_config = optimizer_config or {'type': 'adam', 'lr': 0.001}
        self.loss_config = loss_config or {'type': 'mse' if task_type == 'regression' else 'crossentropy'}
    
    def _build_keras_model(self, input_shape: int, output_shape: int = 1) -> keras.Model:
        """Build customizable Keras model."""
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is not installed")
        
        inputs = keras.Input(shape=(input_shape,))
        x = inputs
        
        # Build layers according to configuration
        for i, layer_conf in enumerate(self.layer_config):
            layer_type = layer_conf.get('type', 'dense')
            
            if layer_type == 'dense':
                x = layers.Dense(
                    units=layer_conf.get('units', 64),
                    activation=layer_conf.get('activation', 'relu'),
                    kernel_regularizer=keras.regularizers.l2(layer_conf.get('l2_reg', 0.01)),
                    name=f'dense_{i}'
                )(x)
            elif layer_type == 'dropout':
                x = layers.Dropout(layer_conf.get('rate', 0.2), name=f'dropout_{i}')(x)
            elif layer_type == 'batch_norm':
                x = layers.BatchNormalization(name=f'batch_norm_{i}')(x)
            elif layer_type == 'activation':
                x = layers.Activation(layer_conf.get('activation', 'relu'), name=f'activation_{i}')(x)
        
        # Output layer
        if self.task_type == 'regression':
            outputs = layers.Dense(output_shape, name='output')(x)
        else:
            activation = 'sigmoid' if output_shape == 1 else 'softmax'
            outputs = layers.Dense(output_shape, activation=activation, name='output')(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs, name='feedforward_network')
        
        # Compile with custom optimizer and loss
        optimizer = self._get_keras_optimizer()
        loss = self._get_keras_loss()
        metrics = ['mae'] if self.task_type == 'regression' else ['accuracy']
        
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        
        return model
    
    def _get_keras_optimizer(self):
        """Get Keras optimizer from configuration."""
        opt_type = self.optimizer_config.get('type', 'adam').lower()
        lr = self.optimizer_config.get('lr', 0.001)
        
        if opt_type == 'adam':
            return keras.optimizers.Adam(learning_rate=lr)
        elif opt_type == 'sgd':
            momentum = self.optimizer_config.get('momentum', 0.9)
            return keras.optimizers.SGD(learning_rate=lr, momentum=momentum)
        elif opt_type == 'rmsprop':
            return keras.optimizers.RMSprop(learning_rate=lr)
        else:
            return keras.optimizers.Adam(learning_rate=lr)
    
    def _get_keras_loss(self):
        """Get Keras loss function from configuration."""
        loss_type = self.loss_config.get('type', 'mse').lower()
        
        if loss_type == 'mse':
            return 'mse'
        elif loss_type == 'mae':
            return 'mae'
        elif loss_type == 'crossentropy':
            return 'sparse_categorical_crossentropy'
        elif loss_type == 'binary_crossentropy':
            return 'binary_crossentropy'
        else:
            return loss_type


class ConvolutionalNetwork(NeuralNetworkBase):
    """
    Convolutional Neural Network for image and sequence data.
    
    Supports both 1D (sequences) and 2D (images) convolutions.
    """
    
    def __init__(self,
                 input_shape: Tuple[int, ...],
                 conv_layers: List[Dict[str, Any]],
                 dense_layers: List[int] = [128, 64],
                 task_type: str = 'classification',
                 n_classes: Optional[int] = None,
                 **kwargs):
        """
        Initialize CNN.
        
        Args:
            input_shape: Shape of input data (height, width, channels) for 2D
                        or (sequence_length, features) for 1D
            conv_layers: List of convolutional layer configurations
            dense_layers: List of dense layer sizes after convolutions
            task_type: Type of task
            n_classes: Number of classes for classification
            **kwargs: Additional arguments for base class
        """
        super().__init__(task_type=task_type, **kwargs)
        
        self.input_shape = input_shape
        self.conv_layers = conv_layers
        self.dense_layers = dense_layers
        self.n_classes = n_classes
    
    def _build_keras_model(self) -> keras.Model:
        """Build Keras CNN model."""
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is not installed")
        
        inputs = keras.Input(shape=self.input_shape)
        x = inputs
        
        # Convolutional layers
        for i, conv_conf in enumerate(self.conv_layers):
            # Determine conv type (1D or 2D)
            if len(self.input_shape) == 2:
                # 1D convolution
                x = layers.Conv1D(
                    filters=conv_conf.get('filters', 32),
                    kernel_size=conv_conf.get('kernel_size', 3),
                    strides=conv_conf.get('strides', 1),
                    padding=conv_conf.get('padding', 'same'),
                    activation=conv_conf.get('activation', 'relu'),
                    name=f'conv1d_{i}'
                )(x)
                
                if conv_conf.get('pooling', True):
                    x = layers.MaxPooling1D(
                        pool_size=conv_conf.get('pool_size', 2),
                        name=f'maxpool1d_{i}'
                    )(x)
            else:
                # 2D convolution
                x = layers.Conv2D(
                    filters=conv_conf.get('filters', 32),
                    kernel_size=conv_conf.get('kernel_size', (3, 3)),
                    strides=conv_conf.get('strides', (1, 1)),
                    padding=conv_conf.get('padding', 'same'),
                    activation=conv_conf.get('activation', 'relu'),
                    name=f'conv2d_{i}'
                )(x)
                
                if conv_conf.get('pooling', True):
                    x = layers.MaxPooling2D(
                        pool_size=conv_conf.get('pool_size', (2, 2)),
                        name=f'maxpool2d_{i}'
                    )(x)
            
            if conv_conf.get('batch_norm', False):
                x = layers.BatchNormalization(name=f'bn_{i}')(x)
            
            if conv_conf.get('dropout', 0) > 0:
                x = layers.Dropout(conv_conf.get('dropout'), name=f'dropout_{i}')(x)
        
        # Flatten for dense layers
        x = layers.Flatten(name='flatten')(x)
        
        # Dense layers
        for i, units in enumerate(self.dense_layers):
            x = layers.Dense(units, activation='relu', name=f'dense_{i}')(x)
            x = layers.Dropout(0.2, name=f'dense_dropout_{i}')(x)
        
        # Output layer
        if self.task_type == 'regression':
            outputs = layers.Dense(1, name='output')(x)
            loss = 'mse'
            metrics = ['mae']
        else:
            if self.n_classes == 2:
                outputs = layers.Dense(1, activation='sigmoid', name='output')(x)
                loss = 'binary_crossentropy'
            else:
                outputs = layers.Dense(self.n_classes, activation='softmax', name='output')(x)
                loss = 'sparse_categorical_crossentropy'
            metrics = ['accuracy']
        
        model = keras.Model(inputs=inputs, outputs=outputs, name='convolutional_network')
        model.compile(optimizer='adam', loss=loss, metrics=metrics)
        
        return model


class RecurrentNetwork(NeuralNetworkBase):
    """
    Recurrent Neural Network for sequence data.
    
    Supports LSTM, GRU, and simple RNN architectures.
    """
    
    def __init__(self,
                 input_shape: Tuple[int, int],  # (sequence_length, features)
                 rnn_layers: List[Dict[str, Any]],
                 dense_layers: List[int] = [64],
                 task_type: str = 'regression',
                 n_classes: Optional[int] = None,
                 **kwargs):
        """
        Initialize RNN.
        
        Args:
            input_shape: Shape of input sequences (sequence_length, features)
            rnn_layers: List of RNN layer configurations
            dense_layers: List of dense layer sizes after RNN
            task_type: Type of task
            n_classes: Number of classes for classification
            **kwargs: Additional arguments for base class
        """
        super().__init__(task_type=task_type, **kwargs)
        
        self.input_shape = input_shape
        self.rnn_layers = rnn_layers
        self.dense_layers = dense_layers
        self.n_classes = n_classes
    
    def _build_keras_model(self) -> keras.Model:
        """Build Keras RNN model."""
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is not installed")
        
        inputs = keras.Input(shape=self.input_shape)
        x = inputs
        
        # RNN layers
        for i, rnn_conf in enumerate(self.rnn_layers):
            rnn_type = rnn_conf.get('type', 'lstm').lower()
            units = rnn_conf.get('units', 64)
            return_sequences = rnn_conf.get('return_sequences', i < len(self.rnn_layers) - 1)
            dropout = rnn_conf.get('dropout', 0.2)
            recurrent_dropout = rnn_conf.get('recurrent_dropout', 0.2)
            bidirectional = rnn_conf.get('bidirectional', False)
            
            # Create RNN layer
            if rnn_type == 'lstm':
                rnn_layer = layers.LSTM(
                    units=units,
                    return_sequences=return_sequences,
                    dropout=dropout,
                    recurrent_dropout=recurrent_dropout,
                    name=f'lstm_{i}'
                )
            elif rnn_type == 'gru':
                rnn_layer = layers.GRU(
                    units=units,
                    return_sequences=return_sequences,
                    dropout=dropout,
                    recurrent_dropout=recurrent_dropout,
                    name=f'gru_{i}'
                )
            else:
                rnn_layer = layers.SimpleRNN(
                    units=units,
                    return_sequences=return_sequences,
                    dropout=dropout,
                    name=f'rnn_{i}'
                )
            
            # Apply bidirectional wrapper if requested
            if bidirectional:
                x = layers.Bidirectional(rnn_layer, name=f'bidirectional_{i}')(x)
            else:
                x = rnn_layer(x)
        
        # Dense layers
        for i, units in enumerate(self.dense_layers):
            x = layers.Dense(units, activation='relu', name=f'dense_{i}')(x)
            x = layers.Dropout(0.2, name=f'dense_dropout_{i}')(x)
        
        # Output layer
        if self.task_type == 'regression':
            outputs = layers.Dense(1, name='output')(x)
            loss = 'mse'
            metrics = ['mae']
        else:
            if self.n_classes == 2:
                outputs = layers.Dense(1, activation='sigmoid', name='output')(x)
                loss = 'binary_crossentropy'
            else:
                outputs = layers.Dense(self.n_classes, activation='softmax', name='output')(x)
                loss = 'sparse_categorical_crossentropy'
            metrics = ['accuracy']
        
        model = keras.Model(inputs=inputs, outputs=outputs, name='recurrent_network')
        model.compile(optimizer='adam', loss=loss, metrics=metrics)
        
        return model


class AutoEncoder(NeuralNetworkBase):
    """
    Autoencoder for dimensionality reduction and feature learning.
    
    Can be used for anomaly detection, denoising, and representation learning.
    """
    
    def __init__(self,
                 input_dim: int,
                 encoding_dims: List[int] = [128, 64, 32],
                 activation: str = 'relu',
                 add_noise: bool = False,
                 noise_factor: float = 0.1,
                 **kwargs):
        """
        Initialize AutoEncoder.
        
        Args:
            input_dim: Dimensionality of input data
            encoding_dims: List of encoding layer dimensions
            activation: Activation function
            add_noise: Whether to add noise for denoising autoencoder
            noise_factor: Standard deviation of Gaussian noise
            **kwargs: Additional arguments for base class
        """
        super().__init__(task_type='unsupervised', **kwargs)
        
        self.input_dim = input_dim
        self.encoding_dims = encoding_dims
        self.activation = activation
        self.add_noise = add_noise
        self.noise_factor = noise_factor
        self.encoder = None
        self.decoder = None
    
    def _build_keras_model(self) -> keras.Model:
        """Build Keras autoencoder model."""
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is not installed")
        
        # Input
        inputs = keras.Input(shape=(self.input_dim,))
        x = inputs
        
        # Add noise if denoising autoencoder
        if self.add_noise:
            x = layers.GaussianNoise(self.noise_factor)(x)
        
        # Encoder
        encoded = x
        for i, dim in enumerate(self.encoding_dims):
            encoded = layers.Dense(dim, activation=self.activation, name=f'encoder_{i}')(encoded)
        
        # Decoder (mirror of encoder)
        decoded = encoded
        for i, dim in enumerate(reversed(self.encoding_dims[:-1])):
            decoded = layers.Dense(dim, activation=self.activation, name=f'decoder_{i}')(decoded)
        
        # Output layer
        outputs = layers.Dense(self.input_dim, activation='sigmoid', name='output')(decoded)
        
        # Create models
        self.encoder = keras.Model(inputs, encoded, name='encoder')
        self.decoder = keras.Model(encoded, outputs, name='decoder')
        autoencoder = keras.Model(inputs, outputs, name='autoencoder')
        
        # Compile
        autoencoder.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        return autoencoder
    
    def encode(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Encode data to latent representation.
        
        Args:
            X: Data to encode
            
        Returns:
            Encoded representation
        """
        if self.encoder is None:
            raise ValueError("Model must be fitted before encoding")
        
        X = np.array(X)
        return self.encoder.predict(X, verbose=0)
    
    def reconstruct(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Reconstruct data from input.
        
        Args:
            X: Data to reconstruct
            
        Returns:
            Reconstructed data
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before reconstruction")
        
        X = np.array(X)
        return self.model.predict(X, verbose=0)
    
    def anomaly_score(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Calculate anomaly scores based on reconstruction error.
        
        Args:
            X: Data to score
            
        Returns:
            Anomaly scores (higher = more anomalous)
        """
        X = np.array(X)
        reconstructed = self.reconstruct(X)
        return np.mean((X - reconstructed) ** 2, axis=1)


class NeuralNetworkRegressor(FeedForwardNetwork):
    """
    High-level neural network regressor with automatic architecture selection.
    
    This class provides an easy-to-use interface for regression tasks with
    sensible defaults and automatic hyperparameter tuning capabilities.
    """
    
    def __init__(self,
                 hidden_layers: Optional[List[int]] = None,
                 activation: str = 'relu',
                 dropout_rate: float = 0.2,
                 batch_norm: bool = True,
                 learning_rate: float = 0.001,
                 auto_architecture: bool = True,
                 **kwargs):
        """
        Initialize Neural Network Regressor.
        
        Args:
            hidden_layers: List of hidden layer sizes (auto if None)
            activation: Activation function
            dropout_rate: Dropout rate
            batch_norm: Whether to use batch normalization
            learning_rate: Learning rate
            auto_architecture: Whether to automatically determine architecture
            **kwargs: Additional arguments
        """
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.batch_norm = batch_norm
        self.learning_rate = learning_rate
        self.auto_architecture = auto_architecture
        
        # Build layer configuration
        layer_config = self._build_layer_config()
        
        super().__init__(
            layer_config=layer_config,
            task_type='regression',
            optimizer_config={'type': 'adam', 'lr': learning_rate},
            **kwargs
        )
    
    def _build_layer_config(self) -> List[Dict[str, Any]]:
        """Build layer configuration."""
        if self.hidden_layers is None:
            # Default architecture
            self.hidden_layers = [128, 64, 32]
        
        config = []
        for i, units in enumerate(self.hidden_layers):
            # Dense layer
            config.append({
                'type': 'dense',
                'units': units,
                'activation': None,
                'l2_reg': 0.01
            })
            
            # Batch norm
            if self.batch_norm:
                config.append({'type': 'batch_norm'})
            
            # Activation
            config.append({
                'type': 'activation',
                'activation': self.activation
            })
            
            # Dropout
            if self.dropout_rate > 0:
                config.append({
                    'type': 'dropout',
                    'rate': self.dropout_rate
                })
        
        return config
    
    def fit(self, X: Union[pd.DataFrame, np.ndarray], 
            y: Union[pd.Series, np.ndarray], **kwargs):
        """
        Fit the regressor with automatic architecture selection.
        
        Args:
            X: Training features
            y: Training targets
            **kwargs: Additional arguments
        """
        X = np.array(X)
        y = np.array(y)
        
        # Auto architecture selection based on data size
        if self.auto_architecture and self.hidden_layers is None:
            n_samples, n_features = X.shape
            
            if n_samples < 1000:
                self.hidden_layers = [64, 32]
            elif n_samples < 10000:
                self.hidden_layers = [128, 64, 32]
            else:
                self.hidden_layers = [256, 128, 64, 32]
            
            # Rebuild layer config
            self.layer_config = self._build_layer_config()
        
        # Build and train model
        if self.model is None:
            self.model = self._build_keras_model(X.shape[1], 1)
        
        # Set default callbacks
        if 'callbacks' not in kwargs:
            kwargs['callbacks'] = [
                EarlyStopping(patience=20, restore_best_weights=True),
                ReduceLROnPlateau(patience=10, factor=0.5, min_lr=1e-6)
            ]
        
        # Train
        self.history = self.model.fit(X, y, **kwargs)
        self.is_fitted = True


class NeuralNetworkClassifier(FeedForwardNetwork):
    """
    High-level neural network classifier with automatic architecture selection.
    
    This class provides an easy-to-use interface for classification tasks with
    sensible defaults and automatic hyperparameter tuning capabilities.
    """
    
    def __init__(self,
                 hidden_layers: Optional[List[int]] = None,
                 activation: str = 'relu',
                 dropout_rate: float = 0.3,
                 batch_norm: bool = True,
                 learning_rate: float = 0.001,
                 auto_architecture: bool = True,
                 **kwargs):
        """
        Initialize Neural Network Classifier.
        
        Args:
            hidden_layers: List of hidden layer sizes (auto if None)
            activation: Activation function
            dropout_rate: Dropout rate
            batch_norm: Whether to use batch normalization
            learning_rate: Learning rate
            auto_architecture: Whether to automatically determine architecture
            **kwargs: Additional arguments
        """
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.batch_norm = batch_norm
        self.learning_rate = learning_rate
        self.auto_architecture = auto_architecture
        self.n_classes = None
        self.label_encoder = None
        
        # Build layer configuration
        layer_config = self._build_layer_config()
        
        super().__init__(
            layer_config=layer_config,
            task_type='classification',
            optimizer_config={'type': 'adam', 'lr': learning_rate},
            **kwargs
        )
    
    def _build_layer_config(self) -> List[Dict[str, Any]]:
        """Build layer configuration."""
        if self.hidden_layers is None:
            # Default architecture
            self.hidden_layers = [128, 64, 32]
        
        config = []
        for i, units in enumerate(self.hidden_layers):
            # Dense layer
            config.append({
                'type': 'dense',
                'units': units,
                'activation': None,
                'l2_reg': 0.01
            })
            
            # Batch norm
            if self.batch_norm:
                config.append({'type': 'batch_norm'})
            
            # Activation
            config.append({
                'type': 'activation',
                'activation': self.activation
            })
            
            # Dropout
            if self.dropout_rate > 0:
                config.append({
                    'type': 'dropout',
                    'rate': self.dropout_rate
                })
        
        return config
    
    def fit(self, X: Union[pd.DataFrame, np.ndarray], 
            y: Union[pd.Series, np.ndarray], **kwargs):
        """
        Fit the classifier with automatic architecture selection.
        
        Args:
            X: Training features
            y: Training labels
            **kwargs: Additional arguments
        """
        X = np.array(X)
        y = np.array(y)
        
        # Encode labels if needed
        if y.dtype == 'object':
            from sklearn.preprocessing import LabelEncoder
            self.label_encoder = LabelEncoder()
            y = self.label_encoder.fit_transform(y)
        
        # Determine number of classes
        self.n_classes = len(np.unique(y))
        
        # Auto architecture selection based on data size and classes
        if self.auto_architecture and self.hidden_layers is None:
            n_samples, n_features = X.shape
            
            if n_samples < 1000:
                self.hidden_layers = [64, 32]
            elif n_samples < 10000:
                self.hidden_layers = [128, 64, 32]
            else:
                self.hidden_layers = [256, 128, 64, 32]
            
            # Add more layers for multi-class problems
            if self.n_classes > 10:
                self.hidden_layers.append(self.n_classes * 2)
            
            # Rebuild layer config
            self.layer_config = self._build_layer_config()
        
        # Build and train model
        if self.model is None:
            output_shape = 1 if self.n_classes == 2 else self.n_classes
            self.model = self._build_keras_model(X.shape[1], output_shape)
        
        # Set default callbacks
        if 'callbacks' not in kwargs:
            kwargs['callbacks'] = [
                EarlyStopping(patience=20, restore_best_weights=True, monitor='val_accuracy'),
                ReduceLROnPlateau(patience=10, factor=0.5, min_lr=1e-6, monitor='val_accuracy')
            ]
        
        # Train
        self.history = self.model.fit(X, y, **kwargs)
        self.is_fitted = True
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Make class predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        X = np.array(X)
        
        if self.n_classes == 2:
            # Binary classification
            probas = self.model.predict(X, verbose=0)
            predictions = (probas > 0.5).astype(int).reshape(-1)
        else:
            # Multi-class classification
            probas = self.model.predict(X, verbose=0)
            predictions = np.argmax(probas, axis=1)
        
        # Decode labels if encoder was used
        if self.label_encoder is not None:
            predictions = self.label_encoder.inverse_transform(predictions)
        
        return predictions
    
    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Get class probabilities."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        X = np.array(X)
        probas = self.model.predict(X, verbose=0)
        
        if self.n_classes == 2:
            # Convert to two-column format for binary classification
            probas = np.hstack([1 - probas, probas])
        
        return probas