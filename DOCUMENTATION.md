# Universal Data Science Toolkit - Complete Documentation

**Version:** 1.0.0  
**Author:** Dmytro Dziublenko  
**Email:** d.dziublenko@gmail.com  
**License:** AGPL-3.0  
**Repository:** [https://github.com/d-dziublenko/data-science-toolkit](https://github.com/d-dziublenko/data-science-toolkit)

## Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Core Modules](#core-modules)
   - [Data Loading](#data-loading)
   - [Data Preprocessing](#data-preprocessing)
   - [Feature Engineering](#feature-engineering)
   - [Data Validation](#data-validation)
5. [Model Components](#model-components)
   - [Base Models](#base-models)
   - [Ensemble Methods](#ensemble-methods)
   - [Neural Networks](#neural-networks)
   - [Target Transformers](#target-transformers)
6. [Evaluation Tools](#evaluation-tools)
   - [Metrics](#metrics)
   - [Visualization](#visualization)
   - [Uncertainty Quantification](#uncertainty-quantification)
   - [Drift Detection](#drift-detection)
7. [Pipelines](#pipelines)
   - [Training Pipeline](#training-pipeline)
   - [Inference Pipeline](#inference-pipeline)
   - [Experiment Tracking](#experiment-tracking)
8. [Utilities](#utilities)
   - [File I/O](#file-io)
   - [Parallel Processing](#parallel-processing)
   - [CLI Tools](#cli-tools)
   - [Logging](#logging)
9. [Examples](#examples)
10. [API Reference](#api-reference)

---

## Introduction

The Universal Data Science Toolkit is a comprehensive Python framework designed to streamline data science and machine learning workflows. It provides a unified interface for data loading, preprocessing, feature engineering, model training, evaluation, and deployment. The toolkit emphasizes modularity, extensibility, and ease of use while maintaining production-ready quality.

### Key Features

- **Universal Data Loading**: Support for multiple file formats including CSV, Excel, Parquet, JSON, and geospatial formats
- **Advanced Preprocessing**: Comprehensive data cleaning, transformation, and feature engineering capabilities
- **Multiple Model Implementations**: From simple linear models to complex ensemble methods and neural networks
- **Robust Evaluation**: Extensive metrics, visualization tools, and uncertainty quantification
- **End-to-End Pipelines**: Complete workflows from data loading to model deployment
- **Experiment Tracking**: Built-in support for tracking experiments and comparing results
- **Parallel Processing**: Efficient handling of large datasets with parallel processing utilities

---

## Installation

### Using the Installation Script

The easiest way to install the toolkit is using the provided installation script:

```bash
# Clone the repository
git clone https://github.com/d-dziublenko/data-science-toolkit.git
cd data-science-toolkit

# Run the installation script
bash install.sh
```

The installation script will:

1. Create a virtual environment
2. Install all required dependencies
3. Set up the project structure
4. Create sample configuration files
5. Run verification tests

### Manual Installation

If you prefer manual installation:

```bash
# Create virtual environment
python -m venv data_science_env
source data_science_env/bin/activate  # On Windows: data_science_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt  # Basic installation
pip install -r requirements-full.txt  # Full installation with optional dependencies
pip install -r requirements-dev.txt  # Development installation
```

### Dependencies

Core dependencies include:

- pandas >= 1.3.0
- numpy >= 1.21.0
- scikit-learn >= 1.0.0
- matplotlib >= 3.4.0
- seaborn >= 0.11.0

Optional dependencies for advanced features:

- xgboost >= 1.5.0
- lightgbm >= 3.3.0
- tensorflow >= 2.8.0
- torch >= 1.10.0
- mlflow >= 1.25.0

---

## Quick Start

Here's a simple example to get you started with the toolkit:

```python
from data_science_toolkit import TrainingPipeline

# Initialize pipeline
pipeline = TrainingPipeline(task_type='regression')

# Run complete pipeline
results = pipeline.run(
    data_path='data/housing_prices.csv',
    target_column='price',
    model_types=['random_forest', 'xgboost'],
    tune_hyperparameters=True
)

# View results
print(results['summary'])
```

For more control over the process:

```python
from data_science_toolkit.core import TabularDataLoader, DataPreprocessor
from data_science_toolkit.models import AutoML
from data_science_toolkit.evaluation import ModelEvaluator

# Load data
loader = TabularDataLoader()
data = loader.load('data/dataset.csv')

# Preprocess
preprocessor = DataPreprocessor()
data_scaled = preprocessor.scale_features(data, method='standard')
data_encoded = preprocessor.encode_categorical(data_scaled)

# Train model
automl = AutoML(task_type='classification')
automl.fit(X_train, y_train)

# Evaluate
evaluator = ModelEvaluator()
metrics = evaluator.evaluate(automl, X_test, y_test)
```

---

## Core Modules

The core modules provide fundamental functionality for data handling and preparation.

### Data Loading

The data loading module provides unified interfaces for loading various data formats.

#### TabularDataLoader

The `TabularDataLoader` class handles loading of tabular data formats with automatic format detection and preprocessing options.

```python
from data_science_toolkit.core import TabularDataLoader

# Initialize loader
loader = TabularDataLoader(
    handle_missing=True,       # Automatically handle missing values
    missing_threshold=0.5,     # Drop columns with >50% missing values
    parse_dates=True          # Automatically parse date columns
)

# Load data with various options
# Basic loading
df = loader.load('data.csv')

# Load specific columns
df = loader.load('data.csv', columns=['feature1', 'feature2', 'target'])

# Load with custom data types
df = loader.load('data.csv', dtype={'id': str, 'amount': float})

# Load Excel file with specific sheet
df = loader.load('data.xlsx', sheet_name='Sheet1')

# Load Parquet file
df = loader.load('data.parquet')
```

**Supported Formats:**

- CSV (.csv)
- Excel (.xlsx, .xls)
- Parquet (.parquet)
- JSON (.json)
- Feather (.feather)
- Pickle (.pkl, .pickle)

**Key Features:**

- Automatic format detection based on file extension
- Memory-efficient loading for large files
- Automatic missing value handling
- Date parsing and type inference
- Column selection and filtering

#### GeospatialDataLoader

The `GeospatialDataLoader` handles geospatial data formats with CRS transformations and geometry validation.

```python
from data_science_toolkit.core import GeospatialDataLoader

# Initialize with target CRS
loader = GeospatialDataLoader(target_crs='EPSG:4326')

# Load shapefile
gdf = loader.load('boundaries.shp')

# Load with bounding box filter
gdf = loader.load(
    'large_dataset.gpkg',
    bbox=(-180, -90, 180, 90)  # (minx, miny, maxx, maxy)
)

# Load specific layer from GeoPackage
gdf = loader.load('multilayer.gpkg', layer='buildings')
```

**Supported Formats:**

- Shapefile (.shp)
- GeoJSON (.geojson)
- GeoPackage (.gpkg)
- File Geodatabase (.gdb)

#### ModelLoader

The `ModelLoader` provides utilities for loading saved machine learning models.

```python
from data_science_toolkit.core import ModelLoader

# Load model with automatic format detection
model = ModelLoader.load('model.pkl')

# Load with specific framework
model = ModelLoader.load('model.h5', framework='tensorflow')

# Load XGBoost model
model = ModelLoader.load('model.json', framework='xgboost')
```

#### DatasetSplitter

The `DatasetSplitter` provides advanced data splitting functionality beyond basic train/test splits.

```python
from data_science_toolkit.core import DatasetSplitter

splitter = DatasetSplitter(random_state=42)

# Time series split
train_indices, test_indices = splitter.time_series_split(
    data,
    time_column='date',
    test_size=0.2
)

# Stratified split with multiple stratification columns
X_train, X_test, y_train, y_test = splitter.stratified_split(
    X, y,
    stratify_columns=['category', 'region'],
    test_size=0.25
)

# Group-based split (no data leakage between groups)
X_train, X_test, y_train, y_test = splitter.group_split(
    X, y,
    groups=data['customer_id'],
    test_size=0.3
)
```

### Data Preprocessing

The preprocessing module provides comprehensive data transformation capabilities.

#### DataPreprocessor

The main preprocessing class that orchestrates various transformation operations.

```python
from data_science_toolkit.core import DataPreprocessor

preprocessor = DataPreprocessor()

# Scale features
df_scaled = preprocessor.scale_features(
    df,
    columns=['feature1', 'feature2'],
    method='standard'  # Options: 'standard', 'minmax', 'robust', 'quantile'
)

# Handle missing values
df_imputed = preprocessor.handle_missing_values(
    df,
    method='impute',  # Options: 'impute', 'drop', 'forward_fill'
    strategy='mean'   # For impute: 'mean', 'median', 'mode', 'constant'
)

# Encode categorical variables
df_encoded = preprocessor.encode_categorical(
    df,
    columns=['category1', 'category2'],
    method='onehot'  # Options: 'onehot', 'label', 'target', 'ordinal'
)

# Remove outliers
df_clean = preprocessor.remove_outliers(
    df,
    columns=['value1', 'value2'],
    method='iqr',      # Options: 'iqr', 'zscore', 'isolation_forest'
    threshold=1.5      # For IQR method
)

# Transform features
df_transformed = preprocessor.transform_features(
    df,
    transformations={
        'feature1': 'log',      # Log transformation
        'feature2': 'sqrt',     # Square root
        'feature3': 'box-cox'   # Box-Cox transformation
    }
)
```

#### FeatureTransformer

Specialized class for feature-specific transformations.

```python
from data_science_toolkit.core import FeatureTransformer

transformer = FeatureTransformer()

# Date feature extraction
df_dates = transformer.extract_date_features(
    df,
    date_column='timestamp',
    features=['year', 'month', 'day', 'dayofweek', 'hour', 'is_weekend']
)

# Binning continuous variables
df_binned = transformer.bin_continuous(
    df,
    column='age',
    n_bins=5,
    strategy='quantile'  # Options: 'uniform', 'quantile', 'kmeans'
)

# Create lag features (for time series)
df_lags = transformer.create_lag_features(
    df,
    column='sales',
    lags=[1, 7, 30],
    date_column='date'
)
```

#### OutlierHandler

Dedicated class for outlier detection and treatment.

```python
from data_science_toolkit.core import OutlierHandler

handler = OutlierHandler()

# Detect outliers using multiple methods
outliers = handler.detect_outliers(
    df,
    columns=['feature1', 'feature2'],
    methods=['iqr', 'zscore', 'isolation_forest'],
    combine='any'  # Options: 'any', 'all', 'majority'
)

# Remove outliers
df_clean = handler.remove_outliers(df, outliers)

# Cap outliers instead of removing
df_capped = handler.cap_outliers(
    df,
    columns=['feature1'],
    lower_percentile=1,
    upper_percentile=99
)
```

#### MissingValueHandler

Specialized handling for missing data patterns.

```python
from data_science_toolkit.core import MissingValueHandler

handler = MissingValueHandler()

# Analyze missing patterns
missing_report = handler.analyze_missing(df)
print(missing_report['summary'])

# Advanced imputation
df_imputed = handler.impute_missing(
    df,
    method='knn',  # Options: 'mean', 'median', 'mode', 'knn', 'iterative'
    n_neighbors=5  # For KNN imputation
)

# Handle missing values based on patterns
df_handled = handler.handle_by_pattern(
    df,
    patterns={
        'MCAR': 'impute',     # Missing Completely At Random
        'MAR': 'model',       # Missing At Random
        'MNAR': 'indicator'   # Missing Not At Random
    }
)
```

### Feature Engineering

The feature engineering module provides tools for creating and selecting features.

#### FeatureSelector

Comprehensive feature selection utilities using various statistical and model-based methods.

```python
from data_science_toolkit.core import FeatureSelector

selector = FeatureSelector(task_type='regression')

# Select by correlation with target
selected_features = selector.select_by_correlation(
    X, y,
    threshold=0.1,
    method='pearson'  # Options: 'pearson', 'spearman', 'kendall'
)

# Statistical test-based selection
selected_features = selector.select_by_statistical_test(
    X, y,
    test='f_regression',  # Options: 'f_regression', 'f_classif', 'chi2'
    k=20  # Number of top features
)

# Model-based selection
selected_features = selector.select_by_importance(
    X, y,
    model='random_forest',
    n_features=15,
    importance_type='permutation'  # Options: 'default', 'permutation'
)

# Recursive Feature Elimination
selected_features = selector.recursive_feature_elimination(
    X, y,
    estimator='svm',
    n_features=10,
    cv=5  # Cross-validation folds
)

# Remove multicollinear features
selected_features = selector.remove_multicollinear(
    X,
    threshold=0.95,
    method='correlation'  # Options: 'correlation', 'vif'
)
```

#### FeatureEngineer

Advanced feature creation and transformation capabilities.

```python
from data_science_toolkit.core import FeatureEngineer

engineer = FeatureEngineer()

# Create polynomial features
X_poly = engineer.create_polynomial_features(
    X,
    degree=2,
    interaction_only=False,
    include_bias=False
)

# Create interaction features
X_interactions = engineer.create_interactions(
    X,
    columns=['feature1', 'feature2', 'feature3'],
    max_interaction=2  # Maximum interaction order
)

# Mathematical transformations
X_transformed = engineer.mathematical_transforms(
    X,
    transforms={
        'ratio': [('feature1', 'feature2')],  # feature1/feature2
        'product': [('feature3', 'feature4')], # feature3*feature4
        'difference': [('feature5', 'feature6')] # feature5-feature6
    }
)

# Domain-specific features
# For text data
X_text = engineer.create_text_features(
    df,
    text_column='description',
    features=['length', 'word_count', 'avg_word_length', 'sentiment']
)

# For geospatial data
X_geo = engineer.create_geospatial_features(
    gdf,
    features=['area', 'perimeter', 'centroid', 'bbox']
)

# Automated feature generation
X_auto = engineer.automated_feature_generation(
    X,
    target=y,
    max_features=50,
    selection_threshold=0.01
)
```

#### InteractionFeatures

Specialized class for creating feature interactions.

```python
from data_science_toolkit.core import InteractionFeatures

interaction_creator = InteractionFeatures()

# Create all pairwise interactions
X_interactions = interaction_creator.create_pairwise(X)

# Create specific interactions
X_custom = interaction_creator.create_custom(
    X,
    interactions=[
        ('feature1', 'feature2', 'multiply'),
        ('feature3', 'feature4', 'divide'),
        (['feature5', 'feature6', 'feature7'], 'multiply')  # Three-way
    ]
)

# Create interactions based on correlation
X_corr_interactions = interaction_creator.create_by_correlation(
    X, y,
    min_correlation=0.3,
    max_features=20
)
```

#### PolynomialFeatures

Enhanced polynomial feature generation.

```python
from data_science_toolkit.core import PolynomialFeatures

poly = PolynomialFeatures()

# Standard polynomial features
X_poly = poly.fit_transform(
    X,
    degree=3,
    interaction_only=False
)

# Polynomial features with regularization
X_poly_reg = poly.fit_transform_regularized(
    X, y,
    degree=4,
    alpha=0.1,  # Regularization strength
    selection_method='lasso'
)

# Get feature names for interpretation
feature_names = poly.get_feature_names(input_features=X.columns)
```

### Data Validation

The validation module ensures data quality and integrity.

#### DataValidator

Comprehensive data validation framework.

```python
from data_science_toolkit.core import DataValidator, ValidationLevel

validator = DataValidator(strict_mode=True)

# Perform full validation
is_valid, results = validator.validate(
    data,
    target_column='target',
    feature_columns=['feature1', 'feature2', 'feature3']
)

# Check specific validation rules
# Data type validation
validator.add_rule('feature1', 'dtype', expected=float)
validator.add_rule('target', 'dtype', expected=int)

# Range validation
validator.add_rule('age', 'range', min_val=0, max_val=120)
validator.add_rule('probability', 'range', min_val=0, max_val=1)

# Pattern validation
validator.add_rule('email', 'pattern', regex=r'^[\w\.-]+@[\w\.-]+\.\w+$')
validator.add_rule('phone', 'pattern', regex=r'^\d{3}-\d{3}-\d{4}$')

# Custom validation function
def custom_validator(series):
    return series.apply(lambda x: x % 2 == 0).all()

validator.add_rule('even_numbers', 'custom', function=custom_validator)

# Run validation with custom rules
is_valid, results = validator.validate(data)

# Get detailed report
report = validator.generate_report(results)
print(report)
```

#### ValidationResult

Data class for validation results.

```python
# Access validation results
for result in results:
    print(f"Check: {result.check_name}")
    print(f"Level: {result.level}")  # INFO, WARNING, ERROR
    print(f"Passed: {result.passed}")
    print(f"Message: {result.message}")

    if result.details:
        print(f"Details: {result.details}")
```

#### DataProfiler

Automated data profiling and quality assessment.

```python
from data_science_toolkit.core import DataProfiler

profiler = DataProfiler()

# Generate comprehensive profile
profile = profiler.profile(data)

# Access profile sections
print(profile['basic_info'])       # Shape, memory usage, etc.
print(profile['data_types'])       # Column types and consistency
print(profile['missing_values'])   # Missing value patterns
print(profile['statistics'])       # Statistical summaries
print(profile['distributions'])    # Distribution characteristics
print(profile['correlations'])     # Feature correlations
print(profile['outliers'])         # Outlier detection results

# Generate HTML report
profiler.generate_html_report(profile, 'data_profile.html')

# Get quality score
quality_score = profiler.calculate_quality_score(profile)
print(f"Data Quality Score: {quality_score:.2%}")
```

---

## Model Components

The models module provides various machine learning algorithms and utilities.

### Base Models

Base classes and interfaces for all models in the toolkit.

#### BaseModel

Abstract base class that all models inherit from.

```python
from data_science_toolkit.models import BaseModel

class CustomModel(BaseModel):
    """Example custom model implementation."""

    def __init__(self, name="CustomModel", random_state=None):
        super().__init__(name, random_state)
        self.model_params = {}

    def fit(self, X, y, sample_weight=None):
        """Training logic here."""
        # Implementation
        self.is_fitted = True
        return self

    def predict(self, X):
        """Prediction logic here."""
        self._check_is_fitted()
        # Implementation
        return predictions

    def get_params(self, deep=True):
        """Get model parameters."""
        return self.model_params

    def set_params(self, **params):
        """Set model parameters."""
        self.model_params.update(params)
        return self
```

#### AutoML

Automated machine learning with intelligent model selection and hyperparameter tuning.

```python
from data_science_toolkit.models import AutoML

# Initialize AutoML
automl = AutoML(
    task_type='auto',          # Auto-detect from data
    optimization_metric=None,   # Auto-select based on task
    time_budget=3600,          # Time limit in seconds
    n_jobs=-1,                 # Use all CPU cores
    verbose=True
)

# Simple fit
automl.fit(X_train, y_train)

# Fit with validation set
automl.fit(X_train, y_train, X_val, y_val)

# Get predictions
predictions = automl.predict(X_test)

# Get best model
best_model = automl.best_model
print(f"Best model: {automl.best_model_name}")
print(f"Best score: {automl.best_score_}")

# Get leaderboard
leaderboard = automl.get_leaderboard()
print(leaderboard)

# Access all trained models
for name, model in automl.models.items():
    print(f"{name}: {model}")
```

**AutoML Features:**

- Automatic task type detection (regression/classification)
- Intelligent model selection based on data characteristics
- Automated hyperparameter tuning
- Feature engineering pipeline
- Ensemble creation
- Time and resource management

#### SimpleLinearModel

Basic linear model implementations with enhanced features.

```python
from data_science_toolkit.models import SimpleLinearModel

# For regression
model = SimpleLinearModel(
    task_type='regression',
    regularization='ridge',  # Options: None, 'ridge', 'lasso', 'elastic'
    alpha=0.1,
    fit_intercept=True
)

model.fit(X_train, y_train)
predictions = model.predict(X_test)

# Get coefficients
coefficients = model.get_coefficients()
print(f"Intercept: {model.intercept_}")
print(f"Coefficients: {coefficients}")

# Feature importance (absolute coefficients)
importance = model.feature_importances_
```

### Ensemble Methods

Advanced ensemble learning implementations.

#### EnsembleModel

High-level ensemble model with automatic configuration.

```python
from data_science_toolkit.models import EnsembleModel

# Create ensemble with automatic model selection
ensemble = EnsembleModel(
    task_type='classification',
    model_type='auto',  # Automatically selects best ensemble type
    random_state=42
)

# Fit with automatic hyperparameter tuning
ensemble.fit(
    X_train, y_train,
    tune_hyperparameters=True,
    cv=5,
    scoring='roc_auc'
)

# Make predictions
predictions = ensemble.predict(X_test)
probabilities = ensemble.predict_proba(X_test)

# Access feature importances
importances = ensemble.feature_importances_
```

#### VotingEnsemble

Combines predictions from multiple models using voting.

```python
from data_science_toolkit.models import VotingEnsemble
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb

# Define base models
base_models = [
    RandomForestClassifier(n_estimators=100),
    SVC(probability=True),
    xgb.XGBClassifier(n_estimators=100)
]

# Create voting ensemble
voting = VotingEnsemble(
    base_models,
    voting='soft',  # 'hard' or 'soft' for classification
    weights=[0.4, 0.3, 0.3]  # Optional model weights
)

voting.fit(X_train, y_train)
predictions = voting.predict(X_test)

# Get individual model predictions
individual_predictions = voting.get_individual_predictions(X_test)
```

#### StackingEnsemble

Multi-level stacking with cross-validation.

```python
from data_science_toolkit.models import StackingEnsemble
from sklearn.linear_model import LogisticRegression

# Create stacking ensemble
stacker = StackingEnsemble(
    base_estimators=base_models,
    final_estimator=LogisticRegression(),
    cv=5,  # Number of folds for base model training
    stack_method='auto',  # 'auto', 'predict_proba', or 'predict'
    passthrough=False  # Whether to pass original features to final estimator
)

stacker.fit(X_train, y_train)
predictions = stacker.predict(X_test)

# Access base model predictions (meta-features)
meta_features = stacker.transform(X_test)
```

#### ModelStacker

Advanced stacking implementation with additional features.

```python
from data_science_toolkit.models import ModelStacker

# Define base models as (name, model) tuples
base_models = [
    ('rf', RandomForestClassifier(n_estimators=100)),
    ('xgb', xgb.XGBClassifier(n_estimators=100)),
    ('lgb', lgb.LGBMClassifier(n_estimators=100))
]

# Create stacker with custom meta-model
stacker = ModelStacker(
    base_models=base_models,
    meta_model=LogisticRegression(C=0.1),
    task_type='classification',
    cv=5
)

stacker.fit(X_train, y_train)

# Get base model predictions
base_predictions = stacker.get_base_predictions(X_test)
print(base_predictions.head())
```

#### ModelBlender

Simple blending with optimized weights.

```python
from data_science_toolkit.models import ModelBlender

# Create blender
blender = ModelBlender(
    models=base_models,
    blend_type='weighted'  # 'mean', 'weighted', or 'ranked'
)

# Fit base models
blender.fit(X_train, y_train)

# Optimize weights using validation set
blender.optimize_weights(X_val, y_val, metric='accuracy')

# Get blended predictions
predictions = blender.predict(X_test)

# Access optimized weights
weights = blender.get_model_weights()
print(f"Optimized weights: {weights}")
```

#### WeightedEnsemble

Ensemble with sophisticated weight optimization.

```python
from data_science_toolkit.models import WeightedEnsemble

# Create weighted ensemble
weighted_ensemble = WeightedEnsemble(
    estimators=base_models,
    weights=None,  # Will be optimized
    weight_optimization='minimize_error',
    cv=5,
    scoring='neg_mean_squared_error'
)

weighted_ensemble.fit(X_train, y_train)

# Get optimized weights
weights = weighted_ensemble.get_estimator_weights()
for name, weight in weights.items():
    print(f"{name}: {weight:.3f}")
```

### Neural Networks

Deep learning models with various architectures.

#### NeuralNetworkBase

Base class for neural network implementations.

```python
from data_science_toolkit.models import NeuralNetworkBase

# The base class provides common functionality for all neural networks
# Actual usage is through specific implementations below
```

#### DNNRegressor

Deep neural network for regression tasks.

```python
from data_science_toolkit.models import DNNRegressor

# Create DNN regressor
dnn = DNNRegressor(
    hidden_layers=[128, 64, 32],     # Architecture
    activation='relu',                # Activation function
    dropout_rate=0.2,                # Dropout for regularization
    batch_norm=True,                 # Batch normalization
    learning_rate=0.001,             # Initial learning rate
    batch_size=32,                   # Mini-batch size
    epochs=100,                      # Training epochs
    early_stopping=True,             # Enable early stopping
    patience=10,                     # Early stopping patience
    optimizer='adam',                # Optimizer
    loss='mse',                      # Loss function
    verbose=1                        # Training verbosity
)

# Fit model
history = dnn.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    callbacks=['reduce_lr', 'checkpoint']  # Additional callbacks
)

# Make predictions
predictions = dnn.predict(X_test)

# Plot training history
dnn.plot_history(history)

# Save model
dnn.save_model('dnn_regressor.h5')
```

#### DNNClassifier

Deep neural network for classification tasks.

```python
from data_science_toolkit.models import DNNClassifier

# Create DNN classifier
dnn_clf = DNNClassifier(
    hidden_layers=[256, 128, 64],
    activation='relu',
    dropout_rate=0.3,
    batch_norm=True,
    learning_rate=0.001,
    batch_size=64,
    epochs=50,
    class_weight='balanced',  # Handle imbalanced classes
    label_smoothing=0.1,     # Label smoothing regularization
    optimizer='adam',
    verbose=1
)

# Fit model
dnn_clf.fit(X_train, y_train, validation_split=0.2)

# Get predictions
predictions = dnn_clf.predict(X_test)
probabilities = dnn_clf.predict_proba(X_test)

# Get model summary
dnn_clf.summary()
```

#### FeedForwardNetwork

Customizable feed-forward neural network.

```python
from data_science_toolkit.models import FeedForwardNetwork

# Create custom architecture
ffn = FeedForwardNetwork(
    input_dim=X_train.shape[1],
    output_dim=1,  # or n_classes for classification
    hidden_layers=[
        {'units': 256, 'activation': 'relu', 'dropout': 0.3},
        {'units': 128, 'activation': 'relu', 'dropout': 0.2},
        {'units': 64, 'activation': 'tanh', 'dropout': 0.1}
    ],
    output_activation='linear',  # or 'softmax' for classification
    regularization='l2',
    reg_lambda=0.01
)

# Compile and train
ffn.compile(optimizer='adam', loss='mse', metrics=['mae'])
ffn.fit(X_train, y_train, epochs=100, batch_size=32)
```

#### AutoEncoder

Autoencoder for dimensionality reduction and anomaly detection.

```python
from data_science_toolkit.models import AutoEncoder

# Create autoencoder
autoencoder = AutoEncoder(
    encoding_dim=32,          # Compressed representation size
    hidden_layers=[128, 64],  # Encoder architecture
    activation='relu',
    decoder_symmetric=True,   # Mirror encoder for decoder
    noise_factor=0.1,        # For denoising autoencoder
    sparsity_penalty=0.01,   # Sparsity regularization
    optimizer='adam',
    loss='mse'
)

# Train autoencoder
autoencoder.fit(X_train, epochs=50, batch_size=256)

# Encode data (dimensionality reduction)
encoded_data = autoencoder.encode(X_test)

# Decode data (reconstruction)
reconstructed_data = autoencoder.decode(encoded_data)

# Detect anomalies (high reconstruction error)
anomaly_scores = autoencoder.get_anomaly_scores(X_test)
anomalies = autoencoder.detect_anomalies(X_test, threshold=0.95)
```

### Target Transformers

Transform target variables for improved model performance.

#### TargetTransformer

Base class for target transformations.

```python
from data_science_toolkit.models import TargetTransformer

# The base class provides interface for all target transformers
# Use specific implementations below
```

#### LogTransformer

Logarithmic transformation for skewed targets.

```python
from data_science_toolkit.models import LogTransformer

# Create log transformer
log_transformer = LogTransformer(
    shift=1.0,  # Add constant before log to handle zeros
    base='e'    # 'e', '10', or '2'
)

# Fit and transform
y_transformed = log_transformer.fit_transform(y_train)

# Train model on transformed target
model.fit(X_train, y_transformed)

# Inverse transform predictions
predictions_transformed = model.predict(X_test)
predictions = log_transformer.inverse_transform(predictions_transformed)
```

#### BoxCoxTransformer

Box-Cox transformation for normality.

```python
from data_science_toolkit.models import BoxCoxTransformer

# Create Box-Cox transformer
bc_transformer = BoxCoxTransformer(
    method='mle',  # 'mle' or 'pearsonr'
    standardize=True
)

# Fit and transform
y_transformed = bc_transformer.fit_transform(y_train)

# Access optimal lambda
print(f"Optimal lambda: {bc_transformer.lambda_}")

# Check normality improvement
bc_transformer.plot_transformation_effect(y_train)
```

#### YeoJohnsonTransformer

Yeo-Johnson transformation (handles negative values).

```python
from data_science_toolkit.models import YeoJohnsonTransformer

yj_transformer = YeoJohnsonTransformer(standardize=True)

# Transform target
y_transformed = yj_transformer.fit_transform(y_train)

# Works with negative values unlike Box-Cox
y_with_negatives = y_train - y_train.mean()
y_transformed_neg = yj_transformer.fit_transform(y_with_negatives)
```

#### QuantileTransformer

Transform to uniform or normal distribution.

```python
from data_science_toolkit.models import QuantileTransformer as QTransformer

# Transform to uniform distribution
qt_uniform = QTransformer(
    output_distribution='uniform',
    n_quantiles=1000,
    subsample=100000
)

y_uniform = qt_uniform.fit_transform(y_train)

# Transform to normal distribution
qt_normal = QTransformer(output_distribution='normal')
y_normal = qt_normal.fit_transform(y_train)
```

#### AutoTargetTransformer

Automatically selects best transformation.

```python
from data_science_toolkit.models import AutoTargetTransformer

# Automatic transformation selection
auto_transformer = AutoTargetTransformer(
    task_type='regression',
    test_normality=True,
    test_methods=['log', 'sqrt', 'box-cox', 'yeo-johnson'],
    selection_metric='shapiro'  # Normality test
)

# Fit will test all methods and select best
y_transformed = auto_transformer.fit_transform(y_train)

# Check which transformation was selected
print(f"Selected transformation: {auto_transformer.best_method_}")
print(f"Normality score: {auto_transformer.best_score_}")
```

---

## Evaluation Tools

Comprehensive model evaluation and analysis utilities.

### Metrics

#### ModelEvaluator

Main class for model evaluation across different tasks.

```python
from data_science_toolkit.evaluation import ModelEvaluator

evaluator = ModelEvaluator()

# Evaluate regression model
reg_metrics = evaluator.evaluate_regression(
    y_true, y_pred,
    metrics=['rmse', 'mae', 'r2', 'mape', 'explained_variance'],
    sample_weight=None
)

# Evaluate classification model
clf_metrics = evaluator.evaluate_classification(
    y_true, y_pred,
    y_proba=probabilities,  # Optional probability predictions
    metrics=['accuracy', 'precision', 'recall', 'f1', 'roc_auc'],
    average='weighted',  # For multiclass
    labels=None  # Specific labels to evaluate
)

# Cross-validation evaluation
cv_scores = evaluator.cross_validate(
    model, X, y,
    cv=5,
    scoring=['accuracy', 'roc_auc'],
    return_train_score=True
)

# Generate comprehensive report
report = evaluator.generate_report(
    model, X_test, y_test,
    include_plots=True,
    save_path='evaluation_report.html'
)
```

#### RegressionMetrics

Specialized regression metrics.

```python
from data_science_toolkit.evaluation import RegressionMetrics

metrics = RegressionMetrics()

# Calculate individual metrics
rmse = metrics.rmse(y_true, y_pred)
mae = metrics.mae(y_true, y_pred)
r2 = metrics.r2(y_true, y_pred)
mape = metrics.mape(y_true, y_pred)

# Advanced metrics
metrics_dict = metrics.calculate_all(y_true, y_pred)

# Metrics include:
# - RMSE, MAE, R², Adjusted R²
# - MAPE, sMAPE, MASE
# - Explained variance
# - Max error, Mean squared log error
# - Median absolute error
# - Directional accuracy (for time series)
```

#### ClassificationMetrics

Specialized classification metrics.

```python
from data_science_toolkit.evaluation import ClassificationMetrics

metrics = ClassificationMetrics()

# Binary classification metrics
binary_metrics = metrics.evaluate_binary(
    y_true, y_pred, y_proba,
    threshold=0.5
)

# Includes: accuracy, precision, recall, f1, specificity,
# sensitivity, AUC-ROC, AUC-PR, MCC, Cohen's kappa

# Multiclass metrics
multi_metrics = metrics.evaluate_multiclass(
    y_true, y_pred, y_proba,
    average='macro',
    labels=None
)

# Per-class metrics
per_class = metrics.per_class_metrics(y_true, y_pred)

# Confusion matrix analysis
cm_analysis = metrics.confusion_matrix_analysis(
    y_true, y_pred,
    normalize='true'
)
```

### Visualization

#### ModelVisualizer

Comprehensive visualization tools for model analysis.

```python
from data_science_toolkit.evaluation import ModelVisualizer

visualizer = ModelVisualizer(figsize=(10, 6), style='seaborn')

# Feature importance plots
visualizer.plot_feature_importance(
    model,
    feature_names=X.columns,
    importance_type='default',  # 'default', 'permutation', 'shap'
    top_n=20,
    orientation='horizontal'
)

# Learning curves
visualizer.plot_learning_curve(
    model, X, y,
    cv=5,
    train_sizes=np.linspace(0.1, 1.0, 10),
    scoring='neg_mean_squared_error'
)

# Validation curves
visualizer.plot_validation_curve(
    model, X, y,
    param_name='max_depth',
    param_range=range(1, 21),
    cv=5
)

# Prediction analysis
visualizer.plot_predictions(
    y_true, y_pred,
    plot_type='scatter',  # 'scatter', 'residual', 'qq'
    confidence_interval=True
)

# ROC curves (for classification)
visualizer.plot_roc_curve(
    y_true, y_proba,
    multi_class='ovr',  # 'ovr' or 'ovo'
    plot_micro=True,
    plot_macro=True
)

# Confusion matrix
visualizer.plot_confusion_matrix(
    y_true, y_pred,
    labels=['Class 0', 'Class 1'],
    normalize='true',
    cmap='Blues'
)

# SHAP analysis
visualizer.plot_shap_analysis(
    model, X,
    plot_type='summary',  # 'summary', 'waterfall', 'force', 'dependence'
    max_display=20
)
```

#### PerformancePlotter

Specialized performance visualization.

```python
from data_science_toolkit.evaluation import PerformancePlotter

plotter = PerformancePlotter()

# Model comparison plot
plotter.compare_models(
    models={'RF': rf_model, 'XGB': xgb_model, 'NN': nn_model},
    X_test, y_test,
    metrics=['rmse', 'mae', 'r2'],
    plot_type='bar'
)

# Performance over time
plotter.plot_performance_timeline(
    predictions_dict,  # Dict of timestamp: predictions
    y_true_dict,      # Dict of timestamp: true values
    metric='mae',
    window='7D'       # Rolling window
)

# Performance by segment
plotter.plot_segmented_performance(
    y_true, y_pred,
    segments=data['category'],
    metric='accuracy'
)
```

### Uncertainty Quantification

#### UncertaintyQuantifier

Comprehensive uncertainty estimation methods.

```python
from data_science_toolkit.evaluation import UncertaintyQuantifier

uq = UncertaintyQuantifier(task_type='regression')

# Bootstrap uncertainty
bootstrap_results = uq.bootstrap_uncertainty(
    model, X_train, y_train, X_test,
    n_bootstrap=100,
    confidence_level=0.95
)

# Results include:
# - predictions: point predictions
# - lower_bound: lower confidence interval
# - upper_bound: upper confidence interval
# - std: standard deviation of predictions

# Ensemble uncertainty (from multiple models)
ensemble_results = uq.ensemble_uncertainty(
    models=[model1, model2, model3],
    X_test,
    confidence_level=0.95
)

# Bayesian uncertainty (for probabilistic models)
bayesian_results = uq.bayesian_uncertainty(
    bayesian_model, X_test,
    n_samples=1000,
    return_full_distribution=True
)

# Conformal prediction
conformal_intervals = uq.conformal_prediction(
    model, X_cal, y_cal, X_test,
    alpha=0.1,  # 90% prediction intervals
    method='absolute_residual'
)

# Calculate calibration metrics
calibration_metrics = uq.calculate_metrics(
    y_true, predictions,
    lower_bounds, upper_bounds
)
```

#### PredictionInterval

Generate prediction intervals using various methods.

```python
from data_science_toolkit.evaluation import PredictionInterval

pi = PredictionInterval(method='quantile_regression')

# Fit prediction interval model
pi.fit(X_train, y_train, alpha=0.05)  # 95% intervals

# Get intervals
lower, upper = pi.predict_interval(X_test)

# Plot prediction intervals
pi.plot_intervals(
    X_test, y_test,
    predictions, lower, upper,
    feature_index=0  # For 2D visualization
)
```

### Drift Detection

#### DataDriftDetector

Detect distribution shifts in data.

```python
from data_science_toolkit.evaluation import DataDriftDetector

# Initialize with reference data
drift_detector = DataDriftDetector(
    reference_data=X_train,
    feature_columns=feature_names,
    categorical_columns=['cat1', 'cat2']
)

# Detect drift in new data
drift_result = drift_detector.detect_drift(
    current_data=X_test,
    method='ks',  # 'ks', 'chi2', 'mmd', 'psi', 'wasserstein'
    confidence_level=0.95,
    return_feature_scores=True
)

# Check results
print(f"Drift detected: {drift_result.drift_detected}")
print(f"Drift score: {drift_result.drift_score:.3f}")
print(f"Drift type: {drift_result.drift_type}")

# Feature-level drift scores
for feature, score in drift_result.feature_scores.items():
    print(f"{feature}: {score:.3f}")

# Generate drift report
report = drift_detector.generate_report(
    X_test,
    include_visualizations=True
)

# Visualize drift
drift_detector.visualize_drift(
    X_test,
    features=['feature1', 'feature2'],
    save_path='drift_analysis.png'
)
```

#### ModelPerformanceDriftDetector

Monitor model performance degradation.

```python
from data_science_toolkit.evaluation import ModelPerformanceDriftDetector

# Initialize with baseline performance
perf_monitor = ModelPerformanceDriftDetector(
    baseline_score=0.95,
    metric='accuracy',
    threshold_type='relative',  # 'absolute' or 'relative'
    threshold=0.05  # 5% relative drop
)

# Monitor performance over time
for batch_X, batch_y in data_stream:
    predictions = model.predict(batch_X)
    score = accuracy_score(batch_y, predictions)

    drift_alert = perf_monitor.check_performance(
        score,
        timestamp=datetime.now()
    )

    if drift_alert.alert_triggered:
        print(f"Performance drift detected! Score: {score:.3f}")
        print(f"Message: {drift_alert.message}")

# Get performance history
history = perf_monitor.get_history()
perf_monitor.plot_performance_trend()
```

---

## Pipelines

End-to-end workflows for training and inference.

### Training Pipeline

Complete training workflow with all steps integrated.

```python
from data_science_toolkit.pipelines import TrainingPipeline, TrainingConfig

# Configure pipeline
config = TrainingConfig(
    task_type='classification',
    experiment_name='customer_churn_prediction',
    data_config={
        'target_column': 'churned',
        'test_size': 0.2,
        'validation_size': 0.1,
        'stratify': True
    },
    preprocessing_config={
        'scaling_method': 'standard',
        'encoding_method': 'target',
        'handle_missing': 'impute',
        'remove_outliers': True
    },
    feature_engineering_config={
        'create_polynomials': True,
        'polynomial_degree': 2,
        'create_interactions': True,
        'select_features': True,
        'n_features': 50
    },
    training_config={
        'models': ['random_forest', 'xgboost', 'neural_network'],
        'tune_hyperparameters': True,
        'tuning_time_budget': 3600,
        'cross_validation': 5,
        'ensemble_method': 'stacking'
    },
    evaluation_config={
        'metrics': ['accuracy', 'precision', 'recall', 'f1', 'roc_auc'],
        'create_plots': True,
        'save_artifacts': True
    }
)

# Create and run pipeline
pipeline = TrainingPipeline(config)

# Run complete pipeline
results = pipeline.run('data/customers.csv')

# Or run step by step
pipeline.load_data('data/customers.csv')
pipeline.preprocess()
pipeline.engineer_features()
pipeline.train_models()
pipeline.evaluate()
pipeline.save_artifacts()

# Access results
print(results['best_model'])
print(results['evaluation_metrics'])
print(results['feature_importance'])

# Generate report
pipeline.generate_report('training_report.html')
```

#### Advanced Pipeline Features

```python
# Custom preprocessing steps
pipeline.add_preprocessing_step(
    'custom_transform',
    custom_transform_function,
    position='after_scaling'
)

# Custom model
pipeline.add_model('custom_model', CustomModel())

# Parallel training
pipeline.train_models(n_jobs=-1, backend='multiprocessing')

# Experiment tracking
pipeline.enable_tracking(
    tracker='mlflow',  # or 'wandb'
    tracking_uri='http://localhost:5000'
)

# Hyperparameter search spaces
pipeline.set_param_space('xgboost', {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7, 9],
    'learning_rate': [0.01, 0.05, 0.1]
})

# Feature importance analysis
importance_df = pipeline.analyze_feature_importance(
    method='permutation',
    n_repeats=10
)

# Model interpretability
pipeline.explain_model(
    method='shap',
    sample_size=1000,
    save_plots=True
)
```

### Inference Pipeline

Production-ready inference with preprocessing and monitoring.

```python
from data_science_toolkit.pipelines import InferencePipeline, InferenceConfig

# Configure inference
config = InferenceConfig(
    model_path='models/best_model.pkl',
    preprocessor_path='models/preprocessor.pkl',
    prediction_mode='batch',  # 'single', 'batch', 'streaming'
    batch_size=1000,
    enable_uncertainty=True,
    enable_drift_detection=True,
    output_format='parquet'
)

# Create pipeline
inference = InferencePipeline(config)

# Batch predictions
predictions = inference.predict_batch('new_data.csv')

# Single prediction
single_pred = inference.predict_single({
    'feature1': 10.5,
    'feature2': 'category_a',
    'feature3': 100
})

# Streaming predictions
for prediction in inference.predict_stream(data_stream):
    process_prediction(prediction)

# With uncertainty
predictions_with_uncertainty = inference.predict_with_uncertainty(
    'new_data.csv',
    confidence_level=0.95
)

# Monitor for drift
predictions_with_monitoring = inference.predict_with_monitoring(
    'new_data.csv',
    alert_on_drift=True
)
```

#### Production Features

```python
# Model serving
from data_science_toolkit.pipelines import ModelServer

server = ModelServer(
    model_path='models/production_model.pkl',
    host='0.0.0.0',
    port=8080,
    workers=4
)

# API endpoints automatically created:
# POST /predict - Single prediction
# POST /predict_batch - Batch predictions
# GET /health - Health check
# GET /metrics - Performance metrics

server.start()

# Async predictions
async_pipeline = InferencePipeline(config, async_mode=True)

async def process_requests(requests):
    tasks = [async_pipeline.predict_async(req) for req in requests]
    predictions = await asyncio.gather(*tasks)
    return predictions

# A/B testing
ab_pipeline = InferencePipeline.create_ab_test(
    model_a='models/model_v1.pkl',
    model_b='models/model_v2.pkl',
    traffic_split=0.5
)

# Feature stores integration
inference.connect_feature_store(
    store_type='feast',
    config={'project': 'my_project', 'registry': 's3://...'}
)

# Real-time features
predictions = inference.predict_with_realtime_features(
    base_features,
    realtime_features=['user_last_action', 'current_balance']
)
```

### Experiment Tracking

Track, compare, and manage ML experiments.

```python
from data_science_toolkit.pipelines import ExperimentTracker, ExperimentConfig

# Initialize tracker
tracker = ExperimentTracker(
    backend='mlflow',  # 'mlflow', 'wandb', or 'local'
    project_name='customer_churn',
    tracking_uri='http://localhost:5000'
)

# Create experiment
config = ExperimentConfig(
    name='xgboost_tuning_v3',
    tags=['production', 'tuning'],
    description='XGBoost hyperparameter tuning with new features'
)

experiment = tracker.create_experiment(config)

# Track training
with tracker.start_run(experiment_id=experiment.id) as run:
    # Log parameters
    tracker.log_params({
        'model_type': 'xgboost',
        'n_estimators': 200,
        'max_depth': 7,
        'learning_rate': 0.05
    })

    # Train model
    model.fit(X_train, y_train)

    # Log metrics
    tracker.log_metrics({
        'train_accuracy': 0.95,
        'val_accuracy': 0.92,
        'train_loss': 0.15,
        'val_loss': 0.18
    })

    # Log model
    tracker.log_model(model, 'model')

    # Log artifacts
    tracker.log_artifact('plots/feature_importance.png')
    tracker.log_artifact('data/processed_features.csv')

# Compare experiments
comparison = tracker.compare_experiments(
    experiment_ids=['exp1', 'exp2', 'exp3'],
    metrics=['val_accuracy', 'train_time']
)

print(comparison.best_experiment)
comparison.plot_metrics()

# Get best model
best_model = tracker.get_best_model(
    experiment_id='exp1',
    metric='val_accuracy',
    mode='max'
)
```

#### Advanced Tracking Features

```python
# Automatic logging
tracker.autolog(framework='sklearn')  # Auto-logs sklearn models

# Nested runs for hyperparameter tuning
with tracker.start_run(run_name='hyperparameter_search'):
    for params in param_grid:
        with tracker.start_run(nested=True):
            tracker.log_params(params)
            score = train_and_evaluate(params)
            tracker.log_metric('cv_score', score)

# Track data lineage
tracker.log_data_version(
    dataset_name='training_data_v2',
    version='2.0.1',
    path='s3://bucket/data/train.parquet',
    statistics={'n_samples': 10000, 'n_features': 50}
)

# Model registry
tracker.register_model(
    model_name='customer_churn_xgboost',
    model_path='runs:/{run_id}/model',
    stage='staging'  # 'staging', 'production', 'archived'
)

# Experiment templates
template = tracker.create_template(
    name='standard_classification',
    default_params={'test_size': 0.2, 'cv_folds': 5},
    required_metrics=['accuracy', 'precision', 'recall']
)

# Reproducibility
tracker.set_seeds(42)  # Sets all random seeds
tracker.log_environment()  # Logs pip packages, system info
```

---

## Utilities

Helper functions and utilities for common tasks.

### File I/O

Advanced file operations and data serialization.

```python
from data_science_toolkit.utils import FileHandler, DataReader, DataWriter

# FileHandler for general operations
handler = FileHandler()

# Read any supported format
data = handler.read('data.csv')
data = handler.read('data.parquet')
data = handler.read('data.json')

# Write with compression
handler.write(data, 'output.csv.gz', compression='gzip')
handler.write(data, 'output.parquet', compression='snappy')

# Chunked reading for large files
for chunk in handler.read_chunked('large_file.csv', chunksize=10000):
    process_chunk(chunk)

# Parallel file reading
files = ['file1.csv', 'file2.csv', 'file3.csv']
dataframes = handler.read_parallel(files, n_jobs=3)

# Safe file operations
with handler.safe_write('output.csv') as f:
    data.to_csv(f)  # Automatically handles errors and cleanup
```

#### ModelSerializer

Save and load models with metadata.

```python
from data_science_toolkit.utils import ModelSerializer

serializer = ModelSerializer(
    compression='joblib',  # 'pickle', 'joblib', 'dill'
    include_metadata=True
)

# Save model with metadata
serializer.save(
    model,
    'model.pkl',
    metadata={
        'training_date': '2024-01-01',
        'author': 'Data Science Team',
        'performance': {'accuracy': 0.95}
    }
)

# Load model and metadata
loaded_model, metadata = serializer.load('model.pkl', return_metadata=True)

# Save ensemble or pipeline
serializer.save_pipeline(
    pipeline={'preprocessor': preprocessor, 'model': model},
    'pipeline.pkl'
)
```

### Parallel Processing

Efficient parallel computation utilities.

```python
from data_science_toolkit.utils import ParallelProcessor, parallel_apply

# Initialize processor
processor = ParallelProcessor(
    n_jobs=-1,  # Use all cores
    backend='multiprocessing',  # 'threading', 'multiprocessing', 'dask'
    batch_size='auto'
)

# Parallel DataFrame apply
result = processor.parallel_apply(
    df,
    function=complex_transformation,
    axis=1,
    progress_bar=True
)

# Parallel group operations
grouped_results = processor.parallel_groupby(
    df,
    by='category',
    func=aggregate_function
)

# Parallel model training
models = processor.parallel_fit(
    estimators=[model1, model2, model3],
    X=X_train,
    y=y_train
)

# Map-reduce operations
results = processor.map_reduce(
    data_chunks,
    map_func=process_chunk,
    reduce_func=combine_results
)

# Convenient function wrappers
# Parallel apply on Series/DataFrame
df['new_col'] = parallel_apply(df['col'], transform_func, n_jobs=4)

# Parallel list processing
results = parallel_map(process_item, items, n_jobs=4, progress=True)
```

### CLI Tools

Command-line interface utilities.

```python
from data_science_toolkit.utils import CLIApplication, create_cli_app

# Create CLI application
app = create_cli_app(
    name='ml-toolkit',
    version='1.0.0',
    description='ML Toolkit CLI'
)

# Add commands
@app.command('train')
def train_model(
    data_path: str,
    model_type: str = 'auto',
    output_dir: str = './models'
):
    """Train a machine learning model."""
    pipeline = TrainingPipeline()
    pipeline.run(data_path, model_type=model_type)
    pipeline.save(output_dir)

@app.command('predict')
def predict(
    model_path: str,
    data_path: str,
    output_path: str = 'predictions.csv'
):
    """Make predictions using trained model."""
    model = load_model(model_path)
    data = pd.read_csv(data_path)
    predictions = model.predict(data)
    pd.DataFrame(predictions).to_csv(output_path)

# Run application
if __name__ == '__main__':
    app.run()
```

Usage from command line:

```bash
# Train model
ml-toolkit train --data-path data.csv --model-type xgboost

# Make predictions
ml-toolkit predict --model-path model.pkl --data-path test.csv

# Get help
ml-toolkit --help
ml-toolkit train --help
```

### Logging

Advanced logging configuration and utilities.

```python
from data_science_toolkit.utils import setup_logger, get_logger

# Setup logging configuration
setup_logger(
    level='INFO',
    format='detailed',  # 'simple', 'detailed', 'json'
    log_file='experiment.log',
    console=True,
    rotation='daily',  # 'size', 'daily', 'none'
    max_bytes=10485760,  # 10MB for size rotation
    backup_count=7
)

# Get logger for module
logger = get_logger(__name__)

# Structured logging
logger.info('Model training started', extra={
    'model_type': 'xgboost',
    'n_samples': 10000,
    'n_features': 50
})

# Progress logging
from data_science_toolkit.utils import ProgressLogger

progress = ProgressLogger(total=100, desc='Training models')
for i in range(100):
    # Do work
    progress.update(1, message=f'Model {i+1}/100')
progress.close()

# Experiment logging
from data_science_toolkit.utils import ExperimentLogger

exp_logger = ExperimentLogger('experiment_001')
exp_logger.log_start()
exp_logger.log_params({'learning_rate': 0.1, 'n_estimators': 100})
exp_logger.log_metrics({'accuracy': 0.95, 'loss': 0.15})
exp_logger.log_end()

# Decorators for automatic logging
from data_science_toolkit.utils import log_execution_time, log_memory_usage

@log_execution_time
def train_model(X, y):
    # Training code
    pass

@log_memory_usage
def process_large_dataset(data):
    # Processing code
    pass
```

---

## Examples

### Regression Example

Complete example for a regression task using house price prediction.

```python
"""
regression_example.py
House price prediction using the Universal Data Science Toolkit
"""

from data_science_toolkit import TrainingPipeline
from data_science_toolkit.core import TabularDataLoader, DataValidator
from data_science_toolkit.evaluation import ModelVisualizer
import pandas as pd

# 1. Load and validate data
loader = TabularDataLoader()
data = loader.load('data/house_prices.csv')

# Validate data
validator = DataValidator()
is_valid, validation_results = validator.validate(
    data,
    target_column='price',
    feature_columns=['sqft', 'bedrooms', 'bathrooms', 'age', 'location']
)

if not is_valid:
    print("Data validation failed!")
    for result in validation_results:
        if not result.passed:
            print(f"- {result.message}")
    exit(1)

# 2. Configure pipeline
pipeline = TrainingPipeline(task_type='regression')

# 3. Run complete pipeline
results = pipeline.run(
    data_path='data/house_prices.csv',
    target_column='price',
    # Preprocessing
    preprocessing_config={
        'scaling_method': 'robust',  # Robust to outliers
        'handle_missing': 'impute',
        'remove_outliers': True,
        'outlier_method': 'isolation_forest'
    },
    # Feature engineering
    feature_config={
        'create_polynomials': True,
        'polynomial_degree': 2,
        'create_interactions': True,
        'select_features': True,
        'selection_method': 'mutual_information',
        'n_features': 30
    },
    # Model training
    model_types=['random_forest', 'xgboost', 'lightgbm', 'neural_network'],
    tune_hyperparameters=True,
    ensemble_method='stacking',
    # Evaluation
    evaluation_metrics=['rmse', 'mae', 'r2', 'mape']
)

# 4. Analyze results
print("\n=== Model Performance ===")
print(results['evaluation_summary'])

print("\n=== Feature Importance ===")
top_features = results['feature_importance'].head(10)
print(top_features)

# 5. Visualize results
visualizer = ModelVisualizer()

# Plot predictions vs actual
visualizer.plot_predictions(
    results['test_predictions'],
    results['test_actual'],
    plot_type='scatter',
    title='House Price Predictions'
)

# Plot residuals
visualizer.plot_residuals(
    results['test_predictions'],
    results['test_actual'],
    plot_type='histogram'
)

# Plot feature importance
visualizer.plot_feature_importance(
    results['best_model'],
    feature_names=results['feature_names'],
    top_n=15
)

# 6. Save best model
pipeline.save_best_model('models/house_price_model.pkl')

# 7. Generate report
pipeline.generate_report(
    'reports/house_price_analysis.html',
    include_plots=True,
    include_code=True
)
```

### Classification Example

Binary classification example for customer churn prediction.

```python
"""
classification_example.py
Customer churn prediction using the Universal Data Science Toolkit
"""

from data_science_toolkit import AutoML
from data_science_toolkit.core import (
    TabularDataLoader, DataPreprocessor,
    FeatureEngineer, FeatureSelector
)
from data_science_toolkit.evaluation import (
    ModelEvaluator, ModelVisualizer,
    UncertaintyQuantifier
)
from data_science_toolkit.pipelines import ExperimentTracker

# Initialize experiment tracking
tracker = ExperimentTracker(project_name='customer_churn')
experiment = tracker.create_experiment(
    name='automl_with_uncertainty',
    tags=['production', 'automl']
)

with tracker.start_run(experiment_id=experiment.id):
    # 1. Load data
    loader = TabularDataLoader()
    data = loader.load('data/customer_data.csv')

    # 2. Preprocessing
    preprocessor = DataPreprocessor()

    # Handle missing values
    data = preprocessor.handle_missing_values(
        data,
        method='impute',
        strategy='median'  # for numerical
    )

    # Encode categorical variables
    categorical_cols = ['gender', 'contract_type', 'payment_method']
    data = preprocessor.encode_categorical(
        data,
        columns=categorical_cols,
        method='target'  # Target encoding for high cardinality
    )

    # 3. Feature Engineering
    engineer = FeatureEngineer()

    # Create customer behavior features
    data = engineer.create_aggregates(
        data,
        group_by='customer_id',
        agg_funcs={
            'transaction_amount': ['sum', 'mean', 'std'],
            'days_since_last_purchase': ['min', 'max'],
            'support_tickets': ['count']
        }
    )

    # Create recency features
    data = engineer.create_recency_features(
        data,
        date_column='last_activity_date',
        reference_date='2024-01-01'
    )

    # 4. Feature Selection
    selector = FeatureSelector(task_type='classification')

    X = data.drop(columns=['churned', 'customer_id'])
    y = data['churned']

    # Select top features
    selected_features = selector.select_by_importance(
        X, y,
        model='random_forest',
        n_features=25
    )

    X_selected = X[selected_features]

    # Log features
    tracker.log_params({
        'n_features_original': len(X.columns),
        'n_features_selected': len(selected_features),
        'preprocessing': 'target_encoding',
        'feature_selection': 'rf_importance'
    })

    # 5. Train AutoML
    automl = AutoML(
        task_type='classification',
        optimization_metric='f1',
        time_budget=1800,  # 30 minutes
        ensemble=True
    )

    automl.fit(X_selected, y)

    # 6. Evaluate with uncertainty
    evaluator = ModelEvaluator()
    uq = UncertaintyQuantifier(task_type='classification')

    # Get predictions with uncertainty
    results = uq.ensemble_uncertainty(
        automl.get_top_models(n=5),
        X_test,
        confidence_level=0.95
    )

    # Calculate metrics
    metrics = evaluator.evaluate_classification(
        y_test,
        results['predictions'],
        y_proba=results['probabilities'],
        metrics=['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    )

    # Log metrics
    tracker.log_metrics(metrics)

    # 7. Model interpretation
    visualizer = ModelVisualizer()

    # SHAP analysis
    visualizer.plot_shap_analysis(
        automl.best_model,
        X_test.sample(1000),  # Sample for speed
        plot_type='summary'
    )

    # Confusion matrix
    visualizer.plot_confusion_matrix(
        y_test,
        results['predictions'],
        labels=['No Churn', 'Churn'],
        normalize='true'
    )

    # ROC curve with confidence intervals
    visualizer.plot_roc_curve_with_ci(
        y_test,
        results['probabilities'][:, 1],
        n_bootstrap=100
    )

    # 8. Business metrics
    # Cost-benefit analysis
    cost_matrix = {
        'true_positive': -100,   # Cost of retention offer
        'false_positive': -100,  # Cost of unnecessary offer
        'false_negative': -500,  # Cost of lost customer
        'true_negative': 0       # No cost
    }

    optimal_threshold = evaluator.optimize_threshold_by_cost(
        y_test,
        results['probabilities'][:, 1],
        cost_matrix
    )

    print(f"Optimal threshold: {optimal_threshold:.3f}")

    # 9. Save model and artifacts
    tracker.log_model(automl, 'automl_model')
    tracker.log_artifact('plots/shap_summary.png')

    # Generate deployment package
    automl.create_deployment_package(
        'deployment/churn_model',
        include_preprocessor=True,
        include_requirements=True
    )
```

### Time Series Example

Time series forecasting with advanced features.

```python
"""
time_series_example.py
Sales forecasting using the Universal Data Science Toolkit
"""

from data_science_toolkit.core import TabularDataLoader, FeatureEngineer
from data_science_toolkit.models import AutoML
from data_science_toolkit.evaluation import ModelEvaluator, ModelVisualizer
from data_science_toolkit.utils import parallel_apply
import pandas as pd
import numpy as np

# 1. Load time series data
loader = TabularDataLoader()
data = loader.load('data/sales_data.csv', parse_dates=['date'])
data = data.sort_values('date').set_index('date')

# 2. Feature engineering for time series
engineer = FeatureEngineer()

# Create time-based features
time_features = engineer.create_time_features(
    data,
    date_column=data.index,
    features=[
        'year', 'month', 'day', 'dayofweek', 'dayofyear',
        'weekofyear', 'quarter', 'is_weekend', 'is_holiday'
    ]
)

# Create lag features
lag_features = engineer.create_lag_features(
    data,
    column='sales',
    lags=[1, 7, 14, 30, 365],  # Previous day, week, 2 weeks, month, year
    rolling_windows=[7, 30],    # 7-day and 30-day moving averages
    rolling_funcs=['mean', 'std', 'min', 'max']
)

# Create seasonal features
seasonal_features = engineer.create_seasonal_features(
    data,
    column='sales',
    seasonalities=[7, 30.5, 365.25],  # Weekly, monthly, yearly
    fourier_terms=10
)

# Combine all features
features = pd.concat([
    data,
    time_features,
    lag_features,
    seasonal_features
], axis=1)

# Remove NaN rows from lag creation
features = features.dropna()

# 3. Create target variable (next day sales)
features['target'] = features['sales'].shift(-1)
features = features.dropna()

# 4. Time series split
from sklearn.model_selection import TimeSeriesSplit

# Prepare features and target
feature_cols = [col for col in features.columns
                if col not in ['sales', 'target']]
X = features[feature_cols]
y = features['target']

# Time series cross-validation
tscv = TimeSeriesSplit(n_splits=5)

# 5. Train models with time series cross-validation
automl = AutoML(
    task_type='regression',
    optimization_metric='mape',  # Mean Absolute Percentage Error
    cross_validation=tscv,
    models_to_try=[
        'lightgbm',  # Good for time series
        'xgboost',
        'random_forest',
        'linear_regression',  # With regularization
        'neural_network'
    ]
)

automl.fit(X, y)

# 6. Evaluate on hold-out test set
# Use last 30 days as test
test_size = 30
X_train, X_test = X[:-test_size], X[-test_size:]
y_train, y_test = y[:-test_size], y[-test_size:]

# Retrain best model on full training data
best_model = automl.best_model
best_model.fit(X_train, y_train)

# Make predictions
predictions = best_model.predict(X_test)

# 7. Time series specific evaluation
evaluator = ModelEvaluator()

# Calculate metrics
ts_metrics = evaluator.evaluate_time_series(
    y_test,
    predictions,
    metrics=['mape', 'smape', 'mase', 'rmse', 'mae']
)

print("\n=== Time Series Metrics ===")
for metric, value in ts_metrics.items():
    print(f"{metric}: {value:.4f}")

# Directional accuracy
directional_acc = evaluator.directional_accuracy(
    y_test.values[1:],
    predictions[1:],
    y_test.values[:-1]
)
print(f"Directional Accuracy: {directional_acc:.2%}")

# 8. Visualizations
visualizer = ModelVisualizer()

# Plot forecast vs actual
visualizer.plot_time_series_forecast(
    dates=data.index[-test_size:],
    actual=y_test,
    forecast=predictions,
    title='Sales Forecast vs Actual'
)

# Plot with confidence intervals (using model uncertainty)
uq = UncertaintyQuantifier(task_type='regression')
uncertainty_results = uq.bootstrap_uncertainty(
    best_model,
    X_train, y_train, X_test,
    n_bootstrap=100
)

visualizer.plot_forecast_with_intervals(
    dates=data.index[-test_size:],
    forecast=predictions,
    lower_bound=uncertainty_results['lower_bound'],
    upper_bound=uncertainty_results['upper_bound'],
    actual=y_test
)

# Feature importance for time series
visualizer.plot_time_series_feature_importance(
    best_model,
    feature_names=X.columns,
    feature_types={
        'lag': [col for col in X.columns if 'lag' in col],
        'time': ['year', 'month', 'day', 'dayofweek'],
        'seasonal': [col for col in X.columns if 'seasonal' in col]
    }
)

# 9. Forecast future values
# Create future features
future_dates = pd.date_range(
    start=data.index[-1] + pd.Timedelta(days=1),
    periods=14,
    freq='D'
)

# Generate features for future dates
# (In practice, you'd need to handle lag features carefully)
future_features = engineer.create_future_features(
    historical_data=data,
    future_dates=future_dates,
    lag_features=lag_features.columns
)

# Make future predictions
future_predictions = best_model.predict(future_features)

# Plot future forecast
visualizer.plot_future_forecast(
    historical_dates=data.index[-60:],
    historical_values=data['sales'][-60:],
    future_dates=future_dates,
    future_predictions=future_predictions,
    title='14-Day Sales Forecast'
)

# 10. Save results
results = {
    'model': best_model,
    'metrics': ts_metrics,
    'feature_importance': best_model.feature_importances_,
    'predictions': predictions,
    'future_forecast': future_predictions
}

import joblib
joblib.dump(results, 'models/sales_forecast_results.pkl')

print("\nForecasting complete! Results saved to models/sales_forecast_results.pkl")
```

---

## API Reference

### Core Module APIs

#### data_loader.py

```python
class TabularDataLoader:
    def __init__(self, handle_missing=True, missing_threshold=0.5, parse_dates=True)
    def load(self, path, columns=None, dtype=None, **kwargs) -> pd.DataFrame
    def load_multiple(self, paths, concat=True) -> Union[pd.DataFrame, List[pd.DataFrame]]
    def validate(self, data) -> bool

class GeospatialDataLoader:
    def __init__(self, target_crs=None)
    def load(self, path, layer=None, bbox=None, **kwargs) -> gpd.GeoDataFrame
    def validate(self, gdf) -> bool

class ModelLoader:
    @staticmethod
    def load(path, framework='auto') -> Any
    @staticmethod
    def save(model, path, framework='auto')

class DatasetSplitter:
    def __init__(self, random_state=None)
    def split(self, X, y, test_size=0.2, val_size=None, stratify=None)
    def time_series_split(self, data, time_column, test_size, gap=0)
    def group_split(self, X, y, groups, test_size)
```

#### preprocessing.py

```python
class DataPreprocessor:
    def __init__(self)
    def scale_features(self, data, columns=None, method='standard', **kwargs)
    def encode_categorical(self, data, columns=None, method='onehot', **kwargs)
    def handle_missing_values(self, data, method='impute', **kwargs)
    def remove_outliers(self, data, columns=None, method='iqr', **kwargs)
    def transform_features(self, data, transformations)

class FeatureTransformer:
    def __init__(self)
    def extract_date_features(self, data, date_column, features)
    def bin_continuous(self, data, column, n_bins, strategy='quantile')
    def create_lag_features(self, data, column, lags, date_column=None)

class OutlierHandler:
    def __init__(self)
    def detect_outliers(self, data, columns, methods, combine='any')
    def remove_outliers(self, data, outlier_mask)
    def cap_outliers(self, data, columns, lower_percentile, upper_percentile)

class MissingValueHandler:
    def __init__(self)
    def analyze_missing(self, data)
    def impute_missing(self, data, method='mean', **kwargs)
    def handle_by_pattern(self, data, patterns)
```

#### feature_engineering.py

```python
class FeatureSelector:
    def __init__(self, task_type='regression')
    def select_by_correlation(self, X, y, threshold=0.1, method='pearson')
    def select_by_statistical_test(self, X, y, test='f_regression', k=20)
    def select_by_importance(self, X, y, model='random_forest', n_features=20)
    def recursive_feature_elimination(self, X, y, estimator, n_features, cv=5)
    def remove_multicollinear(self, X, threshold=0.95, method='correlation')

class FeatureEngineer:
    def __init__(self)
    def create_polynomial_features(self, X, degree=2, interaction_only=False)
    def create_interactions(self, X, columns, max_interaction=2)
    def mathematical_transforms(self, X, transforms)
    def automated_feature_generation(self, X, target, max_features=50)

class InteractionFeatures:
    def __init__(self)
    def create_pairwise(self, X)
    def create_custom(self, X, interactions)
    def create_by_correlation(self, X, y, min_correlation=0.3, max_features=20)

class PolynomialFeatures:
    def __init__(self)
    def fit_transform(self, X, degree=2, interaction_only=False)
    def get_feature_names(self, input_features=None)
```

#### validation.py

```python
class DataValidator:
    def __init__(self, strict_mode=False)
    def validate(self, data, target_column=None, feature_columns=None) -> Tuple[bool, List[ValidationResult]]
    def add_rule(self, column, rule_type, **kwargs)
    def generate_report(self, results) -> str

class DataProfiler:
    def __init__(self)
    def profile(self, data) -> Dict[str, Any]
    def generate_html_report(self, profile, output_path)
    def calculate_quality_score(self, profile) -> float
```

### Models Module APIs

#### base.py

```python
class BaseModel(ABC):
    def __init__(self, name=None, random_state=None)
    @abstractmethod
    def fit(self, X, y, sample_weight=None)
    @abstractmethod
    def predict(self, X)
    def score(self, X, y) -> float
    def save(self, path)
    def load(self, path)

class AutoML:
    def __init__(self, task_type='auto', optimization_metric=None, time_budget=3600, n_jobs=-1)
    def fit(self, X, y, X_val=None, y_val=None)
    def predict(self, X)
    def get_leaderboard(self) -> pd.DataFrame
    def get_best_model(self)
```

#### ensemble.py

```python
class EnsembleModel:
    def __init__(self, task_type='regression', model_type='auto', random_state=42)
    def fit(self, X, y, tune_hyperparameters=False, cv=5, scoring=None)
    def predict(self, X)
    def predict_proba(self, X)  # Classification only

class VotingEnsemble(BaseEnsemble):
    def __init__(self, base_estimators, voting='soft', weights=None)
    def fit(self, X, y, sample_weight=None)
    def predict(self, X)

class StackingEnsemble(BaseEnsemble):
    def __init__(self, base_estimators, final_estimator, cv=5)
    def fit(self, X, y, sample_weight=None)
    def predict(self, X)
    def transform(self, X)

class ModelStacker:
    def __init__(self, base_models, meta_model=None, task_type='regression', cv=5)
    def fit(self, X, y)
    def predict(self, X)
    def get_base_predictions(self, X) -> pd.DataFrame

class ModelBlender:
    def __init__(self, models, blend_type='weighted')
    def fit(self, X, y)
    def optimize_weights(self, X_val, y_val, metric='mse')
    def predict(self, X)
```

#### neural.py

```python
class DNNRegressor:
    def __init__(self, hidden_layers, activation='relu', dropout_rate=0.0, **kwargs)
    def fit(self, X, y, validation_data=None, callbacks=None)
    def predict(self, X)
    def save_model(self, path)
    def plot_history(self, history)

class DNNClassifier:
    def __init__(self, hidden_layers, activation='relu', dropout_rate=0.0, **kwargs)
    def fit(self, X, y, validation_split=0.2)
    def predict(self, X)
    def predict_proba(self, X)
    def summary(self)

class AutoEncoder:
    def __init__(self, encoding_dim, hidden_layers, activation='relu', **kwargs)
    def fit(self, X, epochs=50, batch_size=256)
    def encode(self, X)
    def decode(self, encoded)
    def get_anomaly_scores(self, X)
    def detect_anomalies(self, X, threshold=0.95)
```

#### transformers.py

```python
class LogTransformer:
    def __init__(self, shift=1.0, base='e')
    def fit_transform(self, y)
    def inverse_transform(self, y_transformed)

class BoxCoxTransformer:
    def __init__(self, method='mle', standardize=True)
    def fit_transform(self, y)
    def inverse_transform(self, y_transformed)

class AutoTargetTransformer:
    def __init__(self, task_type='regression', test_normality=True)
    def fit_transform(self, y)
    def inverse_transform(self, y_transformed)
```

### Evaluation Module APIs

#### metrics.py

```python
class ModelEvaluator:
    def __init__(self)
    def evaluate_regression(self, y_true, y_pred, metrics, sample_weight=None)
    def evaluate_classification(self, y_true, y_pred, y_proba=None, metrics, average='weighted')
    def cross_validate(self, model, X, y, cv=5, scoring=None)
    def generate_report(self, model, X_test, y_test, include_plots=True)

class RegressionMetrics:
    def __init__(self)
    def rmse(self, y_true, y_pred)
    def mae(self, y_true, y_pred)
    def r2(self, y_true, y_pred)
    def calculate_all(self, y_true, y_pred) -> Dict[str, float]

class ClassificationMetrics:
    def __init__(self)
    def evaluate_binary(self, y_true, y_pred, y_proba, threshold=0.5)
    def evaluate_multiclass(self, y_true, y_pred, y_proba, average='macro')
    def confusion_matrix_analysis(self, y_true, y_pred, normalize='true')
```

#### visualization.py

```python
class ModelVisualizer:
    def __init__(self, figsize=(10, 6), style='seaborn')
    def plot_feature_importance(self, model, feature_names, importance_type='default', top_n=20)
    def plot_learning_curve(self, model, X, y, cv=5, train_sizes=None)
    def plot_predictions(self, y_true, y_pred, plot_type='scatter')
    def plot_roc_curve(self, y_true, y_proba, multi_class='ovr')
    def plot_confusion_matrix(self, y_true, y_pred, labels=None, normalize='true')
    def plot_shap_analysis(self, model, X, plot_type='summary', max_display=20)
```

#### uncertainty.py

```python
class UncertaintyQuantifier:
    def __init__(self, task_type='regression')
    def bootstrap_uncertainty(self, model, X_train, y_train, X_test, n_bootstrap=100, confidence_level=0.95)
    def ensemble_uncertainty(self, models, X_test, confidence_level=0.95)
    def bayesian_uncertainty(self, model, X_test, n_samples=1000)
    def conformal_prediction(self, model, X_cal, y_cal, X_test, alpha=0.1)
    def calculate_metrics(self, y_true, y_pred, y_lower, y_upper)
```

#### drift.py

```python
class DataDriftDetector:
    def __init__(self, reference_data, feature_columns=None, categorical_columns=None)
    def detect_drift(self, current_data, method='ks', confidence_level=0.95) -> DriftResult
    def generate_report(self, current_data, include_visualizations=True)
    def visualize_drift(self, current_data, features=None, save_path=None)
```

### Pipeline Module APIs

#### training.py

```python
class TrainingPipeline:
    def __init__(self, task_type='regression', experiment_name='experiment', output_dir='./outputs')
    def run(self, data_path, target_column, **config) -> Dict[str, Any]
    def load_data(self, data_path, target_column, test_size=0.2, val_size=0.1)
    def preprocess(self, **kwargs)
    def engineer_features(self, **kwargs)
    def train_models(self, model_types, tune_hyperparameters=False, ensemble_method=None)
    def evaluate(self, create_plots=True, save_plots=True)
    def save_artifacts(self, save_path=None)
    def generate_report(self, output_path, include_plots=True)

class TrainingConfig:
    task_type: str
    experiment_name: str
    data_config: Dict[str, Any]
    preprocessing_config: Dict[str, Any]
    feature_engineering_config: Dict[str, Any]
    training_config: Dict[str, Any]
    evaluation_config: Dict[str, Any]
```

#### inference.py

```python
class InferencePipeline:
    def __init__(self, config: InferenceConfig)
    def predict_batch(self, data_path, output_path=None) -> pd.DataFrame
    def predict_single(self, features: Dict[str, Any]) -> Any
    def predict_stream(self, data_stream) -> Generator
    def predict_with_uncertainty(self, data_path, confidence_level=0.95)
    def predict_with_monitoring(self, data_path, alert_on_drift=True)

class BatchPredictor:
    def __init__(self, model, preprocessor=None, batch_size=1000, n_jobs=1)
    def predict(self, data, return_probabilities=False)
    def predict_generator(self, data_generator)
```

#### experiment.py

```python
class ExperimentTracker:
    def __init__(self, backend='mlflow', project_name='default', tracking_uri=None)
    def create_experiment(self, config: ExperimentConfig) -> Experiment
    def start_run(self, experiment_id=None, run_name=None)
    def log_params(self, params: Dict[str, Any])
    def log_metrics(self, metrics: Dict[str, float], step=None)
    def log_model(self, model, artifact_path='model')
    def log_artifact(self, artifact_path)
    def compare_experiments(self, experiment_ids, metrics)
    def get_best_model(self, experiment_id, metric, mode='max')
```

### Utilities Module APIs

#### file_io.py

```python
class FileHandler:
    def read(self, filepath, **kwargs)
    def write(self, data, filepath, **kwargs)
    def read_chunked(self, filepath, chunksize=10000)
    def read_parallel(self, filepaths, n_jobs=-1)

class ModelSerializer:
    def __init__(self, compression='joblib', include_metadata=True)
    def save(self, model, filepath, metadata=None)
    def load(self, filepath, return_metadata=False)
```

#### parallel.py

```python
class ParallelProcessor:
    def __init__(self, n_jobs=-1, backend='multiprocessing', batch_size='auto')
    def parallel_apply(self, data, function, axis=1, progress_bar=True)
    def parallel_groupby(self, data, by, func)
    def parallel_fit(self, estimators, X, y)
    def map_reduce(self, data_chunks, map_func, reduce_func)

def parallel_apply(series_or_df, func, n_jobs=-1, **kwargs)
def parallel_map(func, iterable, n_jobs=-1, progress=True, **kwargs)
```

#### cli.py

```python
class CLIApplication:
    def __init__(self, name, version='1.0.0', description=None)
    def command(self, name)
    def add_argument(self, *args, **kwargs)
    def run(self, args=None)

def create_cli_app(name, version='1.0.0', description=None) -> CLIApplication
```

#### logging.py

```python
def setup_logger(level='INFO', format='detailed', log_file=None, **kwargs)
def get_logger(name) -> logging.Logger

class ProgressLogger:
    def __init__(self, total, desc='Processing')
    def update(self, n=1, message=None)
    def close(self)

class ExperimentLogger:
    def __init__(self, experiment_id)
    def log_start(self)
    def log_params(self, params)
    def log_metrics(self, metrics)
    def log_end(self)
```

---

## Best Practices

### Data Handling

1. Always validate data before processing
2. Use appropriate data types to minimize memory usage
3. Handle missing values explicitly
4. Document data assumptions and transformations

### Model Development

1. Start simple and iterate
2. Use cross-validation for reliable estimates
3. Track experiments systematically
4. Consider ensemble methods for production

### Production Deployment

1. Version control models and data
2. Monitor for data and model drift
3. Implement proper error handling
4. Use batch prediction for efficiency

### Code Organization

1. Follow the modular structure
2. Write comprehensive tests
3. Document functions and classes
4. Use type hints for clarity

---

## Troubleshooting

### Common Issues

**Import Errors**

```python
# If module not found
import sys
sys.path.append('/path/to/data-science-toolkit')
```

**Memory Issues**

```python
# Use chunked processing
for chunk in pd.read_csv('large_file.csv', chunksize=10000):
    process_chunk(chunk)
```

**Performance Issues**

```python
# Enable parallel processing
processor = ParallelProcessor(n_jobs=-1)
results = processor.parallel_apply(data, function)
```

---

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone repository
git clone https://github.com/d-dziublenko/data-science-toolkit.git
cd data-science-toolkit

# Install in development mode
pip install -e .
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Run linting
flake8 .
black --check .
```

---

## License

This project is licensed under the AGPL-3.0 License. See [LICENSE](LICENSE) file for details.

---

## Support

- **Documentation**: This document
- **Issues**: [GitHub Issues](https://github.com/d-dziublenko/data-science-toolkit/issues)
- **Email**: d.dziublenko@gmail.com

---

## Acknowledgments

This toolkit incorporates best practices and insights from the data science community. Special thanks to all contributors and users who have helped improve this project.
