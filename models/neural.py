"""
models/neural.py
Neural network implementations for deep learning in the universal toolkit.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
import logging

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
            y: Training target
            validation_split: Proportion of data for validation
            epochs: Number of training epochs
            batch_size: Batch size for training
            callbacks: List of callbacks (Keras) or None
            verbose: Verbosity level
        """
        # Convert to numpy arrays
        X = np.array(X)
        y = np.array(y).reshape(-1, 1)
        
        # Build model if not already built
        if self.model is None:
            self.model = self.build_model(X.shape[1])
        
        if self.framework == 'tensorflow':
            self._fit_keras(X, y, validation_split, epochs, batch_size, callbacks, verbose)
        else:
            self._fit_pytorch(X, y, validation_split, epochs, batch_size, verbose)
        
        self.is_fitted = True
    
    def _fit_keras(self, X, y, validation_split, epochs, batch_size, callbacks, verbose):
        """Fit Keras model."""
        # Default callbacks
        if callbacks is None:
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
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
    
    def _fit_pytorch(self, X, y, validation_split, epochs, batch_size, verbose):
        """Fit PyTorch model."""
        # Split data
        n_val = int(len(X) * validation_split)
        indices = np.random.permutation(len(X))
        val_indices, train_indices = indices[:n_val], indices[n_val:]
        
        # Create datasets
        X_train, y_train = X[train_indices], y[train_indices]
        X_val, y_val = X[val_indices], y[val_indices]
        
        # Convert to tensors
        X_train = torch.FloatTensor(X_train)
        y_train = torch.FloatTensor(y_train)
        X_val = torch.FloatTensor(X_val)
        y_val = torch.FloatTensor(y_val)
        
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
                probs = self.model.predict(X, verbose=0)
                predictions = (probs > 0.5).astype(int).reshape(-1)
            else:
                # Multi-class classification
                probs = self.model.predict(X, verbose=0)
                predictions = np.argmax(probs, axis=1)
        else:
            raise NotImplementedError("PyTorch predictions not implemented in this example")
        
        # Decode labels if needed
        if self.label_encoder is not None:
            predictions = self.label_encoder.inverse_transform(predictions)
        
        return predictions
    
    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Get class probabilities."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        X = np.array(X)
        
        if self.framework == 'tensorflow':
            probs = self.model.predict(X, verbose=0)
            
            if self.n_classes == 2:
                # Convert single probability to two-column format
                probs = np.column_stack([1 - probs, probs])
            
            return probs
        else:
            raise NotImplementedError("PyTorch probability predictions not implemented")


class AutoEncoder(NeuralNetworkBase):
    """
    Autoencoder for dimensionality reduction and anomaly detection.
    
    This class implements a flexible autoencoder architecture that can be used
    for feature extraction, dimensionality reduction, or anomaly detection.
    """
    
    def __init__(self,
                 encoding_dim: int = 32,
                 hidden_layers: List[int] = [128, 64],
                 activation: str = 'relu',
                 use_variational: bool = False,
                 **kwargs):
        """
        Initialize the AutoEncoder.
        
        Args:
            encoding_dim: Dimension of the encoding layer
            hidden_layers: List of hidden layer sizes for encoder
            activation: Activation function
            use_variational: Whether to use variational autoencoder
            **kwargs: Additional arguments for base class
        """
        super().__init__(task_type='unsupervised', **kwargs)
        
        self.encoding_dim = encoding_dim
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.use_variational = use_variational
        self.encoder = None
        self.decoder = None
    
    def _build_keras_model(self, input_shape: int) -> Tuple[keras.Model, keras.Model, keras.Model]:
        """Build Keras autoencoder."""
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is not installed")
        
        # Input
        inputs = keras.Input(shape=(input_shape,))
        
        # Encoder
        x = inputs
        for units in self.hidden_layers:
            x = layers.Dense(units, activation=self.activation)(x)
        
        # Encoding layer
        if self.use_variational:
            # Variational autoencoder
            z_mean = layers.Dense(self.encoding_dim, name='z_mean')(x)
            z_log_var = layers.Dense(self.encoding_dim, name='z_log_var')(x)
            
            # Sampling layer
            def sampling(args):
                z_mean, z_log_var = args
                epsilon = tf.keras.backend.random_normal(shape=tf.shape(z_mean))
                return z_mean + tf.exp(0.5 * z_log_var) * epsilon
            
            encoded = layers.Lambda(sampling, name='encoded')([z_mean, z_log_var])
        else:
            # Standard autoencoder
            encoded = layers.Dense(self.encoding_dim, activation=self.activation, name='encoded')(x)
        
        # Decoder
        decoder_input = layers.Input(shape=(self.encoding_dim,))
        x = decoder_input
        
        # Reverse the hidden layers
        for units in reversed(self.hidden_layers):
            x = layers.Dense(units, activation=self.activation)(x)
        
        # Output layer
        decoded = layers.Dense(input_shape, activation='linear')(x)
        
        # Create models
        encoder = keras.Model(inputs, encoded, name='encoder')
        decoder = keras.Model(decoder_input, decoded, name='decoder')
        
        # Full autoencoder
        autoencoder_output = decoder(encoder(inputs))
        autoencoder = keras.Model(inputs, autoencoder_output, name='autoencoder')
        
        # Compile
        autoencoder.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse'
        )
        
        return autoencoder, encoder, decoder
    
    def fit(self, X: Union[pd.DataFrame, np.ndarray], **kwargs):
        """Train the autoencoder."""
        X = np.array(X)
        
        # Build model
        if self.model is None:
            if self.framework == 'tensorflow':
                self.model, self.encoder, self.decoder = self._build_keras_model(X.shape[1])
            else:
                raise NotImplementedError("PyTorch autoencoder not implemented")
        
        # Train (using X as both input and target)
        if self.framework == 'tensorflow':
            # Default callbacks for autoencoder
            callbacks = kwargs.get('callbacks', [
                EarlyStopping(monitor='loss', patience=10, restore_best_weights=True),
                ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5)
            ])
            
            self.history = self.model.fit(
                X, X,
                epochs=kwargs.get('epochs', 100),
                batch_size=kwargs.get('batch_size', 32),
                validation_split=kwargs.get('validation_split', 0.2),
                callbacks=callbacks,
                verbose=kwargs.get('verbose', 1)
            )
        
        self.is_fitted = True
    
    def encode(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Encode data to lower-dimensional representation.
        
        Args:
            X: Data to encode
            
        Returns:
            Encoded representation
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before encoding")
        
        X = np.array(X)
        
        if self.framework == 'tensorflow':
            return self.encoder.predict(X, verbose=0)
        else:
            raise NotImplementedError()
    
    def decode(self, encoded: np.ndarray) -> np.ndarray:
        """
        Decode from lower-dimensional representation.
        
        Args:
            encoded: Encoded data
            
        Returns:
            Reconstructed data
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before decoding")
        
        if self.framework == 'tensorflow':
            return self.decoder.predict(encoded, verbose=0)
        else:
            raise NotImplementedError()
    
    def reconstruct(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Reconstruct data (encode then decode).
        
        Args:
            X: Data to reconstruct
            
        Returns:
            Reconstructed data
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before reconstruction")
        
        X = np.array(X)
        
        if self.framework == 'tensorflow':
            return self.model.predict(X, verbose=0)
        else:
            raise NotImplementedError()
    
    def compute_anomaly_scores(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Compute anomaly scores based on reconstruction error.
        
        Args:
            X: Data to compute anomaly scores for
            
        Returns:
            Anomaly scores (higher = more anomalous)
        """
        X = np.array(X)
        reconstructed = self.reconstruct(X)
        
        # Compute reconstruction error (MSE per sample)
        mse = np.mean((X - reconstructed) ** 2, axis=1)
        
        return mse