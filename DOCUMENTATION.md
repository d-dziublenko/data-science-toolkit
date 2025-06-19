# Universal Data Science Toolkit - Complete Documentation

**Author:** Dmytro Dziublenko  
**Email:** d.dziublenko@gmail.com  
**License:** AGPL-3.0  
**Repository:** [https://github.com/d-dziublenko/data-science-toolkit](https://github.com/d-dziublenko/data-science-toolkit)

## Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Project Structure](#project-structure)
4. [Core Module](#core-module)
   - [Data Loader](#data-loader)
   - [Preprocessing](#preprocessing)
   - [Feature Engineering](#feature-engineering)
   - [Validation](#validation)
5. [Models Module](#models-module)
   - [Base Model Classes](#base-model-classes)
   - [Ensemble Methods](#ensemble-methods)
   - [Neural Networks](#neural-networks)
   - [Target Transformers](#target-transformers)
6. [Evaluation Module](#evaluation-module)
   - [Metrics](#metrics)
   - [Visualization](#visualization)
   - [Uncertainty Quantification](#uncertainty-quantification)
   - [Data Drift Detection](#data-drift-detection)
7. [Utils Module](#utils-module)
   - [File I/O](#file-io)
   - [Parallel Processing](#parallel-processing)
   - [CLI Utilities](#cli-utilities)
   - [Logging](#logging)
8. [Pipelines Module](#pipelines-module)
   - [Training Pipeline](#training-pipeline)
   - [Inference Pipeline](#inference-pipeline)
   - [Experiment Tracking](#experiment-tracking)
9. [Examples](#examples)
10. [Best Practices](#best-practices)

---

## Introduction

The Universal Data Science Toolkit is a comprehensive Python library designed to streamline data science and machine learning workflows. It provides a unified interface for data loading, preprocessing, feature engineering, model training, evaluation, and deployment. The toolkit is built with modularity, scalability, and ease of use in mind, making it suitable for both beginners and experienced practitioners.

### Key Features

- **Universal Data Loading**: Support for multiple data formats including CSV, Excel, JSON, Parquet, and geospatial formats
- **Advanced Preprocessing**: Comprehensive data cleaning, transformation, and validation utilities
- **Feature Engineering**: Automated and manual feature creation, selection, and transformation
- **Multiple Model Support**: Integration with scikit-learn, XGBoost, LightGBM, CatBoost, TensorFlow, and PyTorch
- **Ensemble Methods**: Advanced ensemble techniques including stacking, blending, and voting
- **Evaluation Tools**: Comprehensive metrics, visualizations, and uncertainty quantification
- **Parallel Processing**: Built-in support for distributed computing with Dask and Ray
- **Experiment Tracking**: MLflow and W&B-compatible experiment tracking
- **Production Ready**: End-to-end pipelines for training and inference

---

## Installation

### Prerequisites

- Python 3.8 or higher
- Operating System: Linux, macOS, or Windows

### Installation Methods

#### Method 1: Using the Installation Script

```bash
# Clone the repository
git clone https://github.com/d-dziublenko/data-science-toolkit.git
cd data-science-toolkit

# Run the installation script
chmod +x install.sh
./install.sh
```

The installation script will:

1. Check system requirements
2. Create a virtual environment
3. Install dependencies based on your chosen profile:
   - Basic: Core dependencies only
   - Full: All optional dependencies
   - Development: Full + development tools

#### Method 2: Manual Installation

```bash
# Create virtual environment
python -m venv data_science_env
source data_science_env/bin/activate  # On Windows: data_science_env\Scripts\activate

# Install core dependencies
pip install -r requirements-core.txt

# Optional: Install all dependencies
pip install -r requirements-full.txt

# Optional: Install development dependencies
pip install -r requirements-dev.txt
```

#### Method 3: Pip Installation (when available)

```bash
pip install universal-data-science-toolkit
```

### Verification

Run the test script to verify installation:

```bash
python test_installation.py
```

---

## Project Structure

```
data-science-toolkit/
│
├── core/                    # Core data processing utilities
│   ├── __init__.py         # Module initialization and exports
│   ├── data_loader.py      # Universal data loading utilities
│   ├── preprocessing.py    # Data preprocessing and transformation
│   ├── feature_engineering.py  # Feature selection and engineering
│   └── validation.py       # Data validation utilities
│
├── models/                 # Model implementations
│   ├── __init__.py        # Module initialization
│   ├── base.py            # Base model classes and interfaces
│   ├── ensemble.py        # Ensemble methods
│   ├── neural.py          # Neural network implementations
│   └── transformers.py    # Target variable transformations
│
├── evaluation/            # Model evaluation utilities
│   ├── __init__.py       # Module initialization
│   ├── metrics.py        # Evaluation metrics
│   ├── visualization.py  # Plotting and visualization tools
│   ├── uncertainty.py    # Uncertainty quantification methods
│   └── drift.py         # Data drift detection
│
├── utils/               # Utility functions
│   ├── __init__.py     # Module initialization
│   ├── file_io.py      # File I/O operations
│   ├── parallel.py     # Parallel processing utilities
│   ├── cli.py          # Command-line interface helpers
│   └── logging.py      # Logging configuration
│
├── pipelines/          # End-to-end pipelines
│   ├── __init__.py    # Module initialization
│   ├── training.py    # Training pipeline
│   ├── inference.py   # Inference pipeline
│   └── experiment.py  # Experiment tracking
│
├── examples/          # Example scripts
├── tests/            # Unit tests
├── __init__.py      # Package initialization
├── install.sh       # Installation script
├── .env.example     # Environment variables template
└── README.md        # Project documentation
```

---

## Core Module

The core module provides fundamental data processing capabilities that form the foundation of the toolkit.

### Data Loader

The data loader module provides universal data loading utilities with support for multiple formats and intelligent type detection.

#### Classes

##### DataLoader

Base class for all data loaders.

```python
from core import DataLoader

class DataLoader:
    """
    Base class for data loading operations.

    This class provides a common interface for loading various data formats
    and handling common data loading scenarios like missing files, corrupted data,
    and format detection.
    """

    def __init__(self, cache_enabled: bool = False, verbose: bool = True):
        """
        Initialize the DataLoader.

        Args:
            cache_enabled: Whether to cache loaded data
            verbose: Whether to print loading information
        """
```

**Example Usage:**

```python
# Basic usage with automatic format detection
loader = DataLoader()
data = loader.load('data/sales_data.csv')

# Load with caching enabled
loader = DataLoader(cache_enabled=True)
data = loader.load('data/large_dataset.parquet')  # First load
data = loader.load('data/large_dataset.parquet')  # Loaded from cache

# Load multiple files
files = ['data1.csv', 'data2.csv', 'data3.csv']
combined_data = loader.load_multiple(files, combine=True)
```

##### TabularDataLoader

Specialized loader for tabular data formats with advanced features.

```python
class TabularDataLoader(DataLoader):
    """
    Specialized loader for tabular data (CSV, Excel, Parquet, etc.).

    Provides additional functionality for:
    - Automatic type inference
    - Date parsing
    - Categorical detection
    - Memory optimization
    - Chunked reading for large files
    """

    def load(self, filepath: Union[str, Path], **kwargs) -> pd.DataFrame:
        """
        Load tabular data with intelligent defaults.

        Args:
            filepath: Path to the data file
            **kwargs: Additional arguments passed to pandas read functions

        Returns:
            Loaded DataFrame
        """
```

**Example Usage:**

```python
from core import TabularDataLoader

# Initialize loader
loader = TabularDataLoader()

# Load CSV with automatic type detection
df = loader.load('sales.csv')

# Load with specific options
df = loader.load('sales.csv',
                 parse_dates=['date_column'],
                 low_memory=False,
                 dtype={'product_id': 'category'})

# Load Excel file with specific sheet
df = loader.load('reports.xlsx', sheet_name='Q4_Sales')

# Load large file in chunks
for chunk in loader.load_chunked('large_file.csv', chunksize=10000):
    # Process each chunk
    process_chunk(chunk)

# Load and combine multiple files
files = ['jan.csv', 'feb.csv', 'mar.csv']
quarterly_data = loader.load_multiple(files, combine=True)
```

##### GeospatialDataLoader

Loader for geospatial data formats.

```python
class GeospatialDataLoader(DataLoader):
    """
    Specialized loader for geospatial data formats.

    Supports:
    - GeoJSON
    - Shapefile
    - GeoPackage
    - KML/KMZ
    - GeoTIFF (raster data)
    """

    def load(self, filepath: Union[str, Path],
             crs: Optional[str] = None) -> gpd.GeoDataFrame:
        """
        Load geospatial data.

        Args:
            filepath: Path to geospatial file
            crs: Coordinate reference system (optional)

        Returns:
            GeoDataFrame with spatial data
        """
```

**Example Usage:**

```python
from core import GeospatialDataLoader

# Initialize loader
geo_loader = GeospatialDataLoader()

# Load shapefile
gdf = geo_loader.load('boundaries.shp')

# Load with specific CRS
gdf = geo_loader.load('points.geojson', crs='EPSG:4326')

# Load and reproject
gdf = geo_loader.load('data.gpkg')
gdf_projected = geo_loader.reproject(gdf, 'EPSG:3857')

# Load raster data
raster = geo_loader.load_raster('elevation.tif')
```

### Preprocessing

The preprocessing module provides comprehensive data cleaning and transformation utilities.

#### Classes

##### DataPreprocessor

Main class for data preprocessing operations.

```python
class DataPreprocessor:
    """
    Comprehensive data preprocessing utilities.

    Handles:
    - Missing value imputation
    - Outlier detection and treatment
    - Scaling and normalization
    - Encoding categorical variables
    - Feature transformations
    """

    def __init__(self,
                 numeric_impute_strategy: str = 'mean',
                 categorical_impute_strategy: str = 'most_frequent',
                 scaling_method: str = 'standard',
                 handle_outliers: bool = False):
        """
        Initialize preprocessor.

        Args:
            numeric_impute_strategy: Strategy for numeric imputation
            categorical_impute_strategy: Strategy for categorical imputation
            scaling_method: Method for scaling numeric features
            handle_outliers: Whether to handle outliers
        """
```

**Example Usage:**

```python
from core import DataPreprocessor

# Initialize preprocessor
preprocessor = DataPreprocessor(
    numeric_impute_strategy='median',
    categorical_impute_strategy='constant',
    scaling_method='robust',
    handle_outliers=True
)

# Fit and transform data
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Get preprocessing summary
summary = preprocessor.get_preprocessing_summary()
print(summary)

# Save preprocessor for later use
preprocessor.save('preprocessor.pkl')

# Load saved preprocessor
preprocessor = DataPreprocessor.load('preprocessor.pkl')
```

##### FeatureTransformer

Advanced feature transformation utilities.

```python
class FeatureTransformer:
    """
    Advanced feature transformation methods.

    Includes:
    - Power transformations (Box-Cox, Yeo-Johnson)
    - Quantile transformations
    - Polynomial features
    - Interaction features
    - Binning and discretization
    """

    def __init__(self):
        self.transformations = {}
        self.fitted = False
```

**Example Usage:**

```python
from core import FeatureTransformer

# Initialize transformer
transformer = FeatureTransformer()

# Apply Box-Cox transformation
X_transformed = transformer.box_cox_transform(X, columns=['sales', 'revenue'])

# Create polynomial features
X_poly = transformer.create_polynomial_features(X, degree=2, include_bias=False)

# Discretize continuous variables
X_binned = transformer.discretize(X, columns=['age'], n_bins=5, strategy='quantile')

# Apply multiple transformations
transformations = {
    'log': ['price', 'area'],
    'sqrt': ['distance'],
    'reciprocal': ['time']
}
X_multi = transformer.apply_transformations(X, transformations)
```

##### OutlierHandler

Specialized class for outlier detection and treatment.

```python
class OutlierHandler:
    """
    Comprehensive outlier detection and handling.

    Methods:
    - IQR (Interquartile Range)
    - Z-score
    - Isolation Forest
    - Local Outlier Factor
    - DBSCAN clustering
    """

    def __init__(self, method: str = 'iqr', threshold: float = 1.5):
        """
        Initialize outlier handler.

        Args:
            method: Outlier detection method
            threshold: Threshold for outlier detection
        """
```

**Example Usage:**

```python
from core import OutlierHandler

# Initialize handler
outlier_handler = OutlierHandler(method='iqr', threshold=1.5)

# Detect outliers
outliers = outlier_handler.detect(X)
print(f"Found {outliers.sum()} outliers")

# Remove outliers
X_clean = outlier_handler.remove_outliers(X, y)

# Cap outliers instead of removing
X_capped = outlier_handler.cap_outliers(X, lower_percentile=1, upper_percentile=99)

# Use Isolation Forest for multivariate outlier detection
outlier_handler = OutlierHandler(method='isolation_forest')
outliers = outlier_handler.detect(X)

# Get outlier report
report = outlier_handler.get_outlier_report(X)
```

### Feature Engineering

The feature engineering module provides advanced feature creation and selection capabilities.

#### Classes

##### FeatureSelector

Comprehensive feature selection utilities.

```python
class FeatureSelector:
    """
    Feature selection methods for dimensionality reduction.

    Methods:
    - Statistical tests (correlation, mutual information)
    - Model-based selection (feature importance, RFE)
    - Variance threshold
    - L1-based selection
    """

    def __init__(self, task_type: str = 'regression'):
        """
        Initialize feature selector.

        Args:
            task_type: Type of ML task ('regression' or 'classification')
        """
```

**Example Usage:**

```python
from core import FeatureSelector

# Initialize selector
selector = FeatureSelector(task_type='regression')

# Select by correlation with target
selected_features = selector.select_by_correlation(X, y, threshold=0.3)

# Select by mutual information
selected_features = selector.select_by_mutual_info(X, y, n_features=20)

# Select by model importance (Random Forest)
selected_features = selector.select_by_model_importance(X, y, n_features=15)

# Recursive Feature Elimination
selected_features = selector.select_by_rfe(X, y, n_features=10, cv=5)

# Remove multicollinear features
selected_features = selector.remove_multicollinear_features(X, threshold=0.8)

# Combined selection strategy
selected_features = selector.select_features(
    X, y,
    methods=['correlation', 'mutual_info', 'model_importance'],
    n_features=20
)
```

##### FeatureEngineer

Advanced feature engineering techniques.

```python
class FeatureEngineer:
    """
    Advanced feature creation and engineering.

    Capabilities:
    - Aggregated features
    - Time-based features
    - Text features
    - Interaction features
    - Domain-specific features
    """
```

**Example Usage:**

```python
from core import FeatureEngineer

# Create aggregated features
agg_features = FeatureEngineer.create_aggregated_features(
    data,
    group_cols=['customer_id'],
    agg_cols=['purchase_amount', 'quantity'],
    agg_funcs=['mean', 'std', 'max', 'count']
)

# Create time-based features
time_features = FeatureEngineer.create_time_features(
    data,
    date_column='transaction_date',
    features=['day_of_week', 'month', 'quarter', 'is_weekend', 'days_since']
)

# Create text features
text_features = FeatureEngineer.create_text_features(
    data,
    text_column='description',
    features=['word_count', 'char_count', 'tfidf', 'sentiment']
)

# Create ratio features
data['price_per_sqft'] = FeatureEngineer.create_ratio(
    data, 'price', 'square_feet'
)

# Create interaction features
interactions = FeatureEngineer.create_interactions(
    data,
    columns=['feature1', 'feature2', 'feature3'],
    max_degree=2
)
```

### Validation

The validation module provides comprehensive data quality checks and profiling.

#### Classes

##### DataValidator

Main class for data validation operations.

```python
class DataValidator:
    """
    Comprehensive data validation and quality checks.

    Validates:
    - Data types and formats
    - Missing values
    - Duplicates
    - Value ranges
    - Statistical properties
    - Business rules
    """

    def __init__(self, strict_mode: bool = False):
        """
        Initialize validator.

        Args:
            strict_mode: If True, warnings are treated as errors
        """
```

**Example Usage:**

```python
from core import DataValidator, ValidationLevel

# Initialize validator
validator = DataValidator(strict_mode=False)

# Perform comprehensive validation
is_valid, results = validator.validate(
    data,
    target_column='price',
    feature_columns=['bedrooms', 'bathrooms', 'square_feet']
)

# Add custom validation rules
def price_validation(df):
    invalid_prices = df['price'] <= 0
    return ValidationResult(
        passed=not invalid_prices.any(),
        level=ValidationLevel.ERROR if invalid_prices.any() else ValidationLevel.INFO,
        message=f"Found {invalid_prices.sum()} invalid prices",
        affected_rows=df[invalid_prices].index.tolist()
    )

validator.add_custom_rule('price_check', price_validation)

# Run validation with custom rules
is_valid, results = validator.validate(data)

# Print validation report
validator.print_report(results)

# Get validation summary
summary = validator.get_summary(results)
```

##### DataProfiler

Automated data profiling and reporting.

```python
class DataProfiler:
    """
    Automated data profiling and statistical analysis.

    Generates comprehensive reports including:
    - Basic statistics
    - Data types and memory usage
    - Missing value patterns
    - Correlation analysis
    - Distribution analysis
    """
```

**Example Usage:**

```python
from core import DataProfiler

# Initialize profiler
profiler = DataProfiler()

# Generate comprehensive profile
profile = profiler.profile(data)

# Generate HTML report
profiler.generate_report(data, output_path='data_profile.html')

# Get specific analyses
missing_analysis = profiler.analyze_missing_values(data)
correlation_analysis = profiler.analyze_correlations(data)
distribution_analysis = profiler.analyze_distributions(data)

# Profile specific columns
column_profile = profiler.profile_column(data, 'price')
```

---

## Models Module

The models module provides a wide range of machine learning model implementations with a consistent interface.

### Base Model Classes

#### BaseModel

Abstract base class for all models in the toolkit.

```python
class BaseModel(ABC, BaseEstimator):
    """
    Abstract base class for all models.

    Provides:
    - Common interface for fit/predict
    - Model persistence
    - Cross-validation
    - Feature importance
    - Model metadata tracking
    """

    def __init__(self, name: str = None, random_state: int = None):
        """
        Initialize base model.

        Args:
            name: Model name for identification
            random_state: Random seed for reproducibility
        """
```

**Common Methods for All Models:**

```python
# Training
model.fit(X_train, y_train)

# Prediction
predictions = model.predict(X_test)

# Cross-validation
cv_scores = model.cross_validate(X, y, cv=5, scoring=['rmse', 'r2'])

# Save/Load
model.save('my_model.pkl')
loaded_model = BaseModel.load('my_model.pkl')

# Get parameters
params = model.get_params()

# Feature importance (if available)
importance = model.get_feature_importance()
```

#### BaseRegressor

Base class for regression models.

```python
class BaseRegressor(BaseModel, RegressorMixin):
    """
    Base class for regression models.

    Additional methods:
    - R² score calculation
    - Prediction intervals
    - Residual analysis
    """
```

**Example Usage:**

```python
from models import SimpleLinearModel

# Initialize model
model = SimpleLinearModel(
    regularization='elasticnet',
    alpha=0.1,
    l1_ratio=0.5
)

# Train model
model.fit(X_train, y_train)

# Make predictions with intervals
predictions, lower, upper = model.predict_interval(X_test, confidence=0.95)

# Get model score
r2_score = model.score(X_test, y_test)

# Analyze residuals
residual_analysis = model.analyze_residuals(X_test, y_test)
```

#### BaseClassifier

Base class for classification models.

```python
class BaseClassifier(BaseModel, ClassifierMixin):
    """
    Base class for classification models.

    Additional methods:
    - Probability predictions
    - Multi-class support
    - Class weight handling
    """
```

**Example Usage:**

```python
from models import SimpleClassifier

# Initialize model
model = SimpleClassifier(
    algorithm='logistic',
    class_weight='balanced'
)

# Train model
model.fit(X_train, y_train)

# Predict classes
predictions = model.predict(X_test)

# Predict probabilities
probabilities = model.predict_proba(X_test)

# Get classification report
report = model.classification_report(X_test, y_test)
```

### Ensemble Methods

Advanced ensemble techniques for improved performance.

#### EnsembleModel

Base class for ensemble models.

```python
class EnsembleModel(BaseModel):
    """
    Base class for ensemble models.

    Provides functionality for combining multiple base models.
    """

    def __init__(self, base_models: List[BaseModel],
                 name: str = None,
                 random_state: int = None):
        """
        Initialize ensemble.

        Args:
            base_models: List of base models
            name: Ensemble name
            random_state: Random seed
        """
```

#### VotingEnsemble

Voting-based ensemble for classification and regression.

```python
class VotingEnsemble(EnsembleModel):
    """
    Voting ensemble for combining predictions.

    Supports:
    - Hard voting (classification)
    - Soft voting (classification)
    - Average voting (regression)
    - Weighted voting
    """
```

**Example Usage:**

```python
from models import VotingEnsemble, RandomForestModel, XGBoostModel, LightGBMModel

# Create base models
rf = RandomForestModel(n_estimators=100)
xgb = XGBoostModel(n_estimators=100)
lgb = LightGBMModel(n_estimators=100)

# Create voting ensemble
ensemble = VotingEnsemble(
    base_models=[rf, xgb, lgb],
    voting='soft',  # Use soft voting for classification
    weights=[0.3, 0.4, 0.3]  # Weight models differently
)

# Train ensemble
ensemble.fit(X_train, y_train)

# Make predictions
predictions = ensemble.predict(X_test)

# Get individual model predictions
individual_predictions = ensemble.get_individual_predictions(X_test)
```

#### StackingEnsemble

Stacking ensemble with meta-learner.

```python
class StackingEnsemble(EnsembleModel):
    """
    Stacking ensemble with meta-learner.

    Features:
    - Multiple layers of stacking
    - Cross-validated predictions
    - Custom meta-learners
    - Feature engineering at meta level
    """
```

**Example Usage:**

```python
from models import StackingEnsemble, RandomForestModel, XGBoostModel, LinearModel

# Create base models (level 0)
base_models = [
    RandomForestModel(n_estimators=100),
    XGBoostModel(n_estimators=100),
    LightGBMModel(n_estimators=100),
    CatBoostModel(iterations=100)
]

# Create meta model (level 1)
meta_model = LinearModel(regularization='ridge')

# Create stacking ensemble
stacker = StackingEnsemble(
    base_models=base_models,
    meta_model=meta_model,
    cv_folds=5,  # Use 5-fold CV for generating meta features
    use_probabilities=True,  # Use probabilities for classification
    passthrough=True  # Include original features at meta level
)

# Train ensemble
stacker.fit(X_train, y_train)

# Make predictions
predictions = stacker.predict(X_test)

# Get meta features
meta_features = stacker.get_meta_features(X_test)
```

#### ModelBlender

Advanced blending techniques.

```python
class ModelBlender(EnsembleModel):
    """
    Advanced model blending with optimization.

    Features:
    - Optimal weight finding
    - Rank averaging
    - Geometric mean blending
    - Custom blending functions
    """
```

**Example Usage:**

```python
from models import ModelBlender

# Create blender
blender = ModelBlender(
    base_models=[model1, model2, model3],
    blend_method='optimal',  # Find optimal weights
    optimization_metric='rmse'
)

# Fit blender (finds optimal weights)
blender.fit(X_val, y_val)

# Blend predictions
blended_predictions = blender.predict(X_test)

# Get blending weights
weights = blender.get_weights()
print(f"Optimal weights: {weights}")

# Use custom blending function
def custom_blend(predictions):
    # Geometric mean for positive predictions
    return np.exp(np.mean(np.log(predictions + 1), axis=0)) - 1

blender = ModelBlender(
    base_models=[model1, model2, model3],
    blend_method='custom',
    blend_function=custom_blend
)
```

### Neural Networks

Neural network implementations with various architectures.

#### NeuralNetwork

Flexible neural network implementation.

```python
class NeuralNetwork(BaseModel):
    """
    Flexible neural network with automatic architecture selection.

    Supports:
    - Automatic architecture design
    - Various activation functions
    - Regularization techniques
    - Early stopping
    - Learning rate scheduling
    """
```

**Example Usage:**

```python
from models import NeuralNetwork

# Create neural network with automatic architecture
nn = NeuralNetwork(
    task_type='regression',
    auto_architecture=True,  # Automatically determine architecture
    learning_rate=0.001,
    batch_size=32,
    epochs=100,
    early_stopping=True,
    patience=10
)

# Train network
history = nn.fit(X_train, y_train, validation_data=(X_val, y_val))

# Plot training history
nn.plot_history(history)

# Make predictions
predictions = nn.predict(X_test)

# Get model summary
summary = nn.summary()
```

#### DeepNeuralNetwork

Deep learning models with advanced architectures.

```python
class DeepNeuralNetwork(NeuralNetwork):
    """
    Deep neural network with advanced features.

    Architectures:
    - Fully connected
    - Convolutional (CNN)
    - Recurrent (RNN/LSTM/GRU)
    - Transformer
    - Hybrid architectures
    """
```

**Example Usage:**

```python
from models import DeepNeuralNetwork

# Create deep neural network
dnn = DeepNeuralNetwork(
    architecture='transformer',
    input_dim=100,
    output_dim=1,
    hidden_layers=[512, 256, 128],
    dropout_rate=0.3,
    batch_norm=True,
    activation='gelu',
    optimizer='adamw'
)

# Use with custom architecture
custom_layers = [
    {'type': 'dense', 'units': 256, 'activation': 'relu'},
    {'type': 'dropout', 'rate': 0.3},
    {'type': 'batch_norm'},
    {'type': 'dense', 'units': 128, 'activation': 'relu'},
    {'type': 'attention', 'heads': 8},
    {'type': 'dense', 'units': 64, 'activation': 'relu'}
]

dnn = DeepNeuralNetwork(
    architecture='custom',
    custom_layers=custom_layers
)

# Train with advanced features
history = dnn.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    callbacks=[
        'early_stopping',
        'reduce_lr',
        'model_checkpoint'
    ]
)
```

### Target Transformers

Specialized transformers for target variable transformations.

#### TargetTransformer

Automatic target transformation for improved model performance.

```python
class TargetTransformer:
    """
    Automatic target variable transformation.

    Transformations:
    - Log transformation
    - Box-Cox transformation
    - Yeo-Johnson transformation
    - Quantile transformation
    - Custom transformations
    """
```

**Example Usage:**

```python
from models import TargetTransformer, RandomForestModel

# Create target transformer
target_transformer = TargetTransformer(
    method='auto',  # Automatically select best transformation
    handle_negative=True,
    handle_zero=True
)

# Fit transformer
target_transformer.fit(y_train)

# Transform target
y_train_transformed = target_transformer.transform(y_train)

# Train model on transformed target
model = RandomForestModel()
model.fit(X_train, y_train_transformed)

# Make predictions and inverse transform
predictions_transformed = model.predict(X_test)
predictions = target_transformer.inverse_transform(predictions_transformed)

# Use with pipeline
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', TransformedTargetRegressor(
        regressor=RandomForestModel(),
        transformer=target_transformer
    ))
])
```

---

## Evaluation Module

The evaluation module provides comprehensive model evaluation, visualization, and analysis tools.

### Metrics

Comprehensive evaluation metrics for different tasks.

#### ModelEvaluator

Main class for model evaluation.

```python
class ModelEvaluator:
    """
    Comprehensive model evaluation utilities.

    Provides:
    - Multiple metrics calculation
    - Cross-validation evaluation
    - Statistical significance testing
    - Performance comparison
    - Error analysis
    """
```

**Example Usage:**

```python
from evaluation import ModelEvaluator

# Initialize evaluator
evaluator = ModelEvaluator(task_type='regression')

# Evaluate single model
metrics = evaluator.evaluate(
    y_true=y_test,
    y_pred=predictions,
    metrics=['rmse', 'mae', 'r2', 'mape']
)

# Compare multiple models
models = {'RF': model1, 'XGB': model2, 'NN': model3}
comparison = evaluator.compare_models(
    models=models,
    X_test=X_test,
    y_test=y_test,
    metrics=['rmse', 'mae', 'r2']
)

# Statistical significance testing
p_value = evaluator.test_significance(
    y_true=y_test,
    y_pred1=predictions1,
    y_pred2=predictions2,
    test='wilcoxon'
)

# Cross-validation evaluation
cv_results = evaluator.cross_validate_models(
    models=models,
    X=X_train,
    y=y_train,
    cv=5,
    scoring=['rmse', 'r2']
)

# Error analysis
error_analysis = evaluator.analyze_errors(
    y_true=y_test,
    y_pred=predictions,
    X=X_test,
    feature_names=feature_names
)
```

### Visualization

Advanced visualization tools for model interpretation.

#### ModelVisualizer

Comprehensive visualization utilities.

```python
class ModelVisualizer:
    """
    Model visualization and interpretation tools.

    Plots:
    - Performance metrics
    - Feature importance
    - Partial dependence
    - SHAP values
    - Learning curves
    - Confusion matrices
    - ROC/PR curves
    """
```

**Example Usage:**

```python
from evaluation import ModelVisualizer

# Initialize visualizer
visualizer = ModelVisualizer(figsize=(10, 6))

# Plot predictions vs actual
visualizer.plot_predictions(
    y_true=y_test,
    y_pred=predictions,
    title="Model Predictions vs Actual"
)

# Plot residuals
visualizer.plot_residuals(
    y_true=y_test,
    y_pred=predictions,
    plot_type='scatter'  # or 'histogram', 'qq'
)

# Plot feature importance
visualizer.plot_feature_importance(
    model=model,
    feature_names=feature_names,
    top_n=20,
    plot_type='bar'  # or 'horizontal_bar', 'dot'
)

# Plot partial dependence
visualizer.plot_partial_dependence(
    model=model,
    X=X_train,
    features=['feature1', 'feature2'],
    grid_resolution=50
)

# Plot SHAP values
visualizer.plot_shap_values(
    model=model,
    X=X_test,
    plot_type='summary'  # or 'waterfall', 'force', 'dependence'
)

# Plot learning curves
visualizer.plot_learning_curves(
    model=model,
    X=X_train,
    y=y_train,
    cv=5,
    train_sizes=np.linspace(0.1, 1.0, 10)
)

# For classification: Plot confusion matrix
visualizer.plot_confusion_matrix(
    y_true=y_test,
    y_pred=predictions,
    labels=class_names,
    normalize=True
)

# For classification: Plot ROC curves
visualizer.plot_roc_curves(
    models={'Model1': model1, 'Model2': model2},
    X_test=X_test,
    y_test=y_test
)
```

### Uncertainty Quantification

Methods for quantifying prediction uncertainty.

#### UncertaintyQuantifier

Comprehensive uncertainty quantification methods.

```python
class UncertaintyQuantifier:
    """
    Uncertainty quantification for predictions.

    Methods:
    - Bootstrap confidence intervals
    - Quantile regression
    - Bayesian approaches
    - Ensemble uncertainty
    - Calibration methods
    """
```

**Example Usage:**

```python
from evaluation import UncertaintyQuantifier

# Initialize quantifier
uq = UncertaintyQuantifier(task_type='regression')

# Bootstrap confidence intervals
predictions, lower, upper = uq.bootstrap_predictions(
    model=model,
    X=X_test,
    n_bootstrap=1000,
    confidence_level=0.95
)

# Quantile regression intervals
quantile_model = uq.fit_quantile_regression(
    X=X_train,
    y=y_train,
    quantiles=[0.05, 0.5, 0.95]
)
quantile_predictions = quantile_model.predict(X_test)

# Ensemble uncertainty
ensemble_predictions, uncertainty = uq.ensemble_uncertainty(
    models=[model1, model2, model3],
    X=X_test,
    method='std'  # or 'entropy', 'variance'
)

# Calibration for classification
calibrated_model = uq.calibrate_probabilities(
    model=classifier,
    X_val=X_val,
    y_val=y_val,
    method='isotonic'  # or 'platt', 'temperature'
)

# Prediction intervals with calibration
predictions, intervals = uq.get_calibrated_intervals(
    model=model,
    X=X_test,
    confidence_level=0.95
)
```

### Data Drift Detection

Monitor and detect data drift in production.

#### DataDriftDetector

Comprehensive drift detection methods.

```python
class DataDriftDetector:
    """
    Data drift detection and monitoring.

    Methods:
    - Statistical tests (KS, Chi-square, etc.)
    - Distance-based methods (PSI, KL divergence)
    - Model-based detection
    - Feature drift analysis
    - Concept drift detection
    """
```

**Example Usage:**

```python
from evaluation import DataDriftDetector

# Initialize detector
drift_detector = DataDriftDetector()

# Detect feature drift
drift_results = drift_detector.detect_feature_drift(
    reference_data=X_train,
    current_data=X_production,
    features=feature_names,
    method='ks_test',  # or 'chi2', 'psi', 'kl_divergence'
    threshold=0.05
)

# Detect target drift
target_drift = drift_detector.detect_target_drift(
    reference_target=y_train,
    current_target=y_production,
    method='ks_test'
)

# Detect model performance drift
perf_drift = drift_detector.detect_performance_drift(
    model=model,
    reference_data=(X_train, y_train),
    current_data=(X_production, y_production),
    metric='rmse',
    threshold=0.1  # 10% degradation threshold
)

# Generate drift report
report = drift_detector.generate_drift_report(
    reference_data=X_train,
    current_data=X_production,
    save_path='drift_report.html'
)

# Set up continuous monitoring
monitor = drift_detector.create_monitor(
    reference_data=X_train,
    alert_threshold=0.05,
    check_frequency='daily'
)

# Check for drift
is_drifted, drift_score = monitor.check_drift(X_new)
```

---

## Utils Module

The utils module provides essential utilities for file operations, parallel processing, CLI tools, and logging.

### File I/O

Advanced file input/output operations.

#### FileHandler

Universal file handler for various formats.

```python
class FileHandler:
    """
    Universal file handler with format detection.

    Supports:
    - Data files (CSV, Excel, JSON, Parquet, etc.)
    - Model files (pickle, joblib, ONNX)
    - Configuration files (YAML, JSON, TOML)
    - Compressed files (zip, gzip, tar)
    """
```

**Example Usage:**

```python
from utils import FileHandler

# Initialize handler
handler = FileHandler()

# Save data with automatic format detection
handler.save(data, 'output.parquet')
handler.save(data, 'output.csv.gz')  # Compressed CSV

# Load data with automatic format detection
data = handler.load('input.xlsx')
data = handler.load('input.json.gz')

# Save/load models
handler.save_model(model, 'model.pkl')
loaded_model = handler.load_model('model.pkl')

# Work with compressed files
handler.compress('data_folder/', 'archive.zip')
handler.extract('archive.tar.gz', 'output_folder/')

# Batch operations
files = handler.list_files('data/', pattern='*.csv')
for file in files:
    data = handler.load(file)
    # Process data
    handler.save(processed_data, file.replace('.csv', '_processed.parquet'))
```

#### ModelSerializer

Specialized serialization for ML models.

```python
class ModelSerializer:
    """
    Advanced model serialization with metadata.

    Features:
    - Framework detection
    - Metadata preservation
    - Version compatibility
    - Compression support
    """
```

**Example Usage:**

```python
from utils import ModelSerializer

# Initialize serializer
serializer = ModelSerializer(
    include_metadata=True,
    compression='gzip'
)

# Save model with metadata
serializer.save(
    model=model,
    filepath='model.pkl.gz',
    metadata={
        'training_date': '2024-01-15',
        'dataset': 'sales_data_v2',
        'metrics': {'rmse': 0.123, 'r2': 0.95}
    }
)

# Load model with metadata
model, metadata = serializer.load(
    'model.pkl.gz',
    return_metadata=True
)

# Convert between formats
serializer.convert(
    'sklearn_model.pkl',
    'model.onnx',
    format='onnx'
)
```

### Parallel Processing

Utilities for parallel and distributed computing.

#### ParallelProcessor

Main class for parallel processing.

```python
class ParallelProcessor:
    """
    Flexible parallel processing utilities.

    Backends:
    - Multiprocessing
    - Threading
    - Joblib
    - Dask
    - Ray
    """
```

**Example Usage:**

```python
from utils import ParallelProcessor

# Initialize processor
processor = ParallelProcessor(
    n_jobs=-1,  # Use all cores
    backend='multiprocessing',
    verbose=True
)

# Parallel map
results = processor.map(expensive_function, data_list)

# Parallel apply on DataFrame
df_processed = processor.apply(df, process_row, axis=1)

# Parallel groupby operation
grouped_results = processor.groupby_apply(
    df,
    groupby_cols=['category'],
    func=calculate_statistics
)

# Process in chunks
results = processor.process_chunks(
    data=large_dataset,
    chunk_size=10000,
    func=process_chunk,
    combine_func=pd.concat
)

# Use context manager for temporary parallel configuration
with processor.parallel_backend('threading', n_jobs=4):
    results = processor.map(io_bound_function, urls)
```

#### DistributedProcessor

Distributed processing for large-scale operations.

```python
class DistributedProcessor:
    """
    Distributed processing using Dask or Ray.

    Features:
    - Cluster support
    - Memory-efficient processing
    - Fault tolerance
    - Progress tracking
    """
```

**Example Usage:**

```python
from utils import DistributedProcessor

# Initialize distributed processor
dist_processor = DistributedProcessor(
    backend='dask',
    n_workers=4,
    memory_limit='4GB'
)

# Process large DataFrame
df_result = dist_processor.process_dataframe(
    df=large_df,
    func=complex_transformation,
    partition_col='date'
)

# Distributed model training
results = dist_processor.train_models_parallel(
    datasets=dataset_list,
    model_func=create_model,
    training_func=train_model
)

# Process files in distributed manner
dist_processor.process_files(
    file_pattern='data/*.parquet',
    func=process_file,
    output_dir='processed/'
)
```

### CLI Utilities

Command-line interface helpers and tools.

#### CLIApplication

Framework for building CLI applications.

```python
class CLIApplication:
    """
    Framework for creating CLI applications.

    Features:
    - Command routing
    - Argument parsing
    - Configuration loading
    - Progress bars
    - Colored output
    """
```

**Example Usage:**

```python
from utils import CLIApplication, CLIParser

# Create CLI application
app = CLIApplication(
    name="ML Pipeline",
    version="1.0.0",
    description="Machine Learning Pipeline CLI"
)

# Add commands
@app.command()
def train(data_path: str, model_type: str = 'random_forest',
          cv_folds: int = 5):
    """Train a machine learning model."""
    print(f"Training {model_type} model with {cv_folds}-fold CV...")
    # Training logic here

@app.command()
def predict(model_path: str, data_path: str, output_path: str):
    """Make predictions using trained model."""
    print(f"Loading model from {model_path}")
    # Prediction logic here

@app.command()
def evaluate(predictions_path: str, truth_path: str):
    """Evaluate model predictions."""
    # Evaluation logic here

# Run application
if __name__ == "__main__":
    app.run()
```

#### ProgressBar

Enhanced progress tracking utilities.

```python
from utils import ProgressBar

# Simple progress bar
with ProgressBar(total=1000, desc="Processing") as pbar:
    for i in range(1000):
        # Do work
        pbar.update(1)

# Parallel progress tracking
results = parallel_map_with_progress(
    func=process_item,
    items=item_list,
    n_jobs=4,
    desc="Processing items"
)

# Nested progress bars
with ProgressBar(total=len(experiments), desc="Experiments") as exp_bar:
    for experiment in experiments:
        with ProgressBar(total=len(models), desc="Models", leave=False) as model_bar:
            for model in models:
                # Train model
                model_bar.update(1)
        exp_bar.update(1)
```

### Logging

Advanced logging configuration and utilities.

#### LoggingManager

Centralized logging configuration.

```python
class LoggingManager:
    """
    Advanced logging configuration and management.

    Features:
    - Multiple handlers (file, console, remote)
    - Structured logging
    - Performance logging
    - Experiment tracking
    """
```

**Example Usage:**

```python
from utils import setup_logger, get_logger

# Setup logging configuration
setup_logger(
    level='INFO',
    log_file='experiment.log',
    format='detailed',  # or 'simple', 'json'
    colorize=True
)

# Get logger instance
logger = get_logger(__name__)

# Basic logging
logger.info("Starting experiment")
logger.debug("Debug information")
logger.warning("Warning message")
logger.error("Error occurred")

# Structured logging
logger.info("Model trained", extra={
    'model_type': 'RandomForest',
    'accuracy': 0.95,
    'training_time': 120.5
})

# Performance logging
from utils import log_execution_time, log_memory_usage

@log_execution_time
@log_memory_usage
def train_model(X, y):
    # Training logic
    pass

# Context logging
from utils import log_context

with log_context(experiment_id="exp_001", model="rf"):
    # All logs within this context will include
    # experiment_id and model information
    train_model(X, y)
    evaluate_model(X_test, y_test)
```

---

## Pipelines Module

End-to-end pipelines for complete ML workflows.

### Training Pipeline

Complete training pipeline with all steps integrated.

#### TrainingPipeline

Main training pipeline class.

```python
class TrainingPipeline:
    """
    End-to-end training pipeline.

    Steps:
    1. Data loading and validation
    2. Preprocessing and feature engineering
    3. Model selection and training
    4. Hyperparameter tuning
    5. Model evaluation
    6. Model saving and reporting
    """
```

**Example Usage:**

```python
from pipelines import TrainingPipeline, TrainingConfig

# Create configuration
config = TrainingConfig(
    task_type='regression',
    train_path='data/train.csv',
    target_column='price',
    feature_columns=None,  # Use all columns
    test_size=0.2,
    random_state=42,
    models=['random_forest', 'xgboost', 'lightgbm'],
    ensemble=True,
    tune_hyperparameters=True,
    n_trials=50,
    cv_folds=5,
    optimization_metric='rmse'
)

# Initialize pipeline
pipeline = TrainingPipeline(config)

# Run complete pipeline
results = pipeline.run()

# Access results
print(f"Best model: {results['best_model_name']}")
print(f"Test score: {results['test_score']}")
print(f"Feature importance: {results['feature_importance']}")

# Save pipeline
pipeline.save('trained_pipeline.pkl')

# Advanced usage with custom components
pipeline = TrainingPipeline(
    config=config,
    preprocessor=custom_preprocessor,
    feature_engineer=custom_engineer,
    models={
        'custom_model': CustomModel(),
        'neural_net': NeuralNetwork()
    }
)

# Run with callbacks
def on_model_trained(model_name, scores):
    print(f"Model {model_name} trained. CV score: {scores['mean']}")

results = pipeline.run(
    callbacks={
        'on_model_trained': on_model_trained,
        'on_preprocessing_complete': lambda X: print(f"Shape after preprocessing: {X.shape}")
    }
)
```

### Inference Pipeline

Production-ready inference pipeline.

#### InferencePipeline

Main inference pipeline class.

```python
class InferencePipeline:
    """
    Production inference pipeline.

    Features:
    - Batch prediction
    - Stream processing
    - Async predictions
    - Data validation
    - Drift detection
    - Performance monitoring
    """
```

**Example Usage:**

```python
from pipelines import InferencePipeline, InferenceConfig

# Create configuration
config = InferenceConfig(
    model_path='models/production_model.pkl',
    preprocessor_path='models/preprocessor.pkl',
    prediction_mode='batch',
    batch_size=1000,
    enable_drift_detection=True,
    enable_uncertainty=True
)

# Initialize pipeline
inference = InferencePipeline(config)

# Batch prediction
predictions = inference.predict_batch('new_data.csv')

# Single prediction
single_pred = inference.predict_single({
    'feature1': 10.5,
    'feature2': 'category_a',
    'feature3': 25
})

# Stream processing
for prediction in inference.predict_stream('data_stream.csv'):
    process_prediction(prediction)

# Async predictions
async def async_predict():
    tasks = [
        inference.predict_async(data1),
        inference.predict_async(data2),
        inference.predict_async(data3)
    ]
    results = await asyncio.gather(*tasks)
    return results

# With monitoring
inference.enable_monitoring(
    log_predictions=True,
    detect_drift=True,
    alert_on_errors=True
)

predictions = inference.predict_batch(
    'new_data.csv',
    return_uncertainty=True,
    return_explanations=True
)
```

### Experiment Tracking

Track and manage ML experiments.

#### ExperimentTracker

Comprehensive experiment tracking.

```python
class ExperimentTracker:
    """
    Track ML experiments with MLflow compatibility.

    Features:
    - Automatic logging
    - Metric tracking
    - Artifact storage
    - Model versioning
    - Comparison tools
    """
```

**Example Usage:**

```python
from pipelines import ExperimentTracker

# Initialize tracker
tracker = ExperimentTracker(
    experiment_name="house_price_prediction",
    tracking_uri="./mlruns"  # or remote MLflow server
)

# Start experiment
with tracker.start_run(run_name="rf_baseline") as run:
    # Log parameters
    tracker.log_params({
        'model_type': 'random_forest',
        'n_estimators': 100,
        'max_depth': 10
    })

    # Train model
    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)

    # Log metrics
    tracker.log_metrics({
        'train_rmse': train_rmse,
        'val_rmse': val_rmse,
        'test_rmse': test_rmse
    })

    # Log model
    tracker.log_model(model, "model")

    # Log artifacts
    tracker.log_artifact("feature_importance.png")
    tracker.log_artifact("predictions.csv")

# Compare experiments
comparison = tracker.compare_runs(
    run_ids=['run1', 'run2', 'run3'],
    metrics=['rmse', 'r2']
)

# Get best run
best_run = tracker.get_best_run(metric='val_rmse', mode='min')

# Load model from run
model = tracker.load_model(run_id=best_run.id, model_name="model")
```

---

## Examples

### Complete Regression Example

```python
"""
Complete example: House price prediction
"""
import pandas as pd
from sklearn.model_selection import train_test_split

# Import toolkit components
from core import TabularDataLoader, DataPreprocessor, FeatureEngineer, DataValidator
from models import AutoML, EnsembleModel
from evaluation import ModelEvaluator, ModelVisualizer
from pipelines import TrainingPipeline, ExperimentTracker
from utils import setup_logger

# Setup logging
setup_logger(level='INFO', log_file='house_price_prediction.log')

# 1. Load and validate data
loader = TabularDataLoader()
data = loader.load('house_prices.csv')

validator = DataValidator()
is_valid, validation_results = validator.validate(data)
if not is_valid:
    validator.print_report(validation_results)

# 2. Feature engineering
engineer = FeatureEngineer()

# Create new features
data['price_per_sqft'] = data['price'] / data['sqft']
data['total_rooms'] = data['bedrooms'] + data['bathrooms']
data['house_age'] = 2024 - data['year_built']

# Create polynomial features for important numeric columns
poly_features = engineer.create_polynomial_features(
    data[['sqft', 'bedrooms', 'bathrooms']],
    degree=2,
    include_bias=False
)
data = pd.concat([data, poly_features], axis=1)

# 3. Split data
X = data.drop('price', axis=1)
y = data['price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Create and run AutoML
automl = AutoML(
    task_type='regression',
    time_limit=300,  # 5 minutes
    optimization_metric='rmse',
    ensemble=True
)

automl.fit(X_train, y_train, X_val=X_test, y_val=y_test)

# 5. Make predictions
predictions = automl.predict(X_test)

# 6. Evaluate results
evaluator = ModelEvaluator(task_type='regression')
metrics = evaluator.evaluate(
    y_true=y_test,
    y_pred=predictions,
    metrics=['rmse', 'mae', 'r2', 'mape']
)

print("Model Performance:")
for metric, value in metrics.items():
    print(f"{metric}: {value:.4f}")

# 7. Visualize results
visualizer = ModelVisualizer()

# Plot predictions
visualizer.plot_predictions(y_test, predictions)

# Plot feature importance
visualizer.plot_feature_importance(
    automl.best_model,
    feature_names=X.columns,
    top_n=20
)

# Plot residuals
visualizer.plot_residuals(y_test, predictions)

# 8. Save model and pipeline
automl.save('house_price_model.pkl')

# 9. Create inference pipeline
from pipelines import InferencePipeline

inference = InferencePipeline(
    model_path='house_price_model.pkl',
    enable_uncertainty=True
)

# Make predictions on new data
new_predictions = inference.predict_batch('new_houses.csv')
```

### Complete Classification Example

```python
"""
Complete example: Customer churn prediction
"""
# Import required libraries
from core import TabularDataLoader, DataPreprocessor, FeatureSelector
from models import AutoML, StackingEnsemble
from evaluation import ModelEvaluator, ModelVisualizer
from pipelines import TrainingPipeline, ExperimentTracker

# 1. Load data
loader = TabularDataLoader()
data = loader.load('customer_churn.csv')

# 2. Preprocessing
preprocessor = DataPreprocessor(
    numeric_impute_strategy='median',
    categorical_impute_strategy='most_frequent',
    scaling_method='standard',
    encoding_method='target'  # Target encoding for high cardinality
)

# Identify column types
numeric_features = data.select_dtypes(include=['int64', 'float64']).columns
categorical_features = data.select_dtypes(include=['object']).columns

# 3. Feature selection
selector = FeatureSelector(task_type='classification')

# Remove highly correlated features
selected_features = selector.remove_multicollinear_features(
    data[numeric_features],
    threshold=0.9
)

# 4. Create training pipeline
config = TrainingConfig(
    task_type='classification',
    models=['logistic', 'random_forest', 'xgboost', 'neural_network'],
    ensemble=True,
    ensemble_method='stacking',
    tune_hyperparameters=True,
    optimization_metric='roc_auc',
    cv_folds=5,
    class_weight='balanced'  # Handle imbalanced classes
)

pipeline = TrainingPipeline(config)

# 5. Run pipeline with experiment tracking
tracker = ExperimentTracker(experiment_name="churn_prediction")

with tracker.start_run(run_name="stacking_ensemble"):
    results = pipeline.run(
        data=data,
        target_column='churn',
        track_experiment=True
    )

    # Log additional artifacts
    tracker.log_artifact("feature_importance.png")
    tracker.log_artifact("confusion_matrix.png")

# 6. Analyze results
print(f"Best model: {results['best_model_name']}")
print(f"ROC-AUC: {results['test_score']:.4f}")

# 7. Create detailed evaluation report
evaluator = ModelEvaluator(task_type='classification')
detailed_metrics = evaluator.evaluate(
    y_true=results['y_test'],
    y_pred=results['predictions'],
    y_proba=results['probabilities'],
    metrics=['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
)

# 8. Generate visualizations
visualizer = ModelVisualizer()

# Confusion matrix
visualizer.plot_confusion_matrix(
    results['y_test'],
    results['predictions'],
    labels=['No Churn', 'Churn']
)

# ROC curve
visualizer.plot_roc_curve(
    results['y_test'],
    results['probabilities'][:, 1]
)

# Feature importance with SHAP
visualizer.plot_shap_values(
    results['best_model'],
    results['X_test']
)
```

### Time Series Example

```python
"""
Complete example: Sales forecasting
"""
from core import TimeSeriesDataLoader, TimeSeriesPreprocessor
from models import TimeSeriesModel, EnsembleForecaster
from evaluation import TimeSeriesEvaluator

# 1. Load time series data
loader = TimeSeriesDataLoader()
data = loader.load(
    'sales_data.csv',
    date_column='date',
    target_column='sales',
    frequency='D'  # Daily frequency
)

# 2. Create time features
preprocessor = TimeSeriesPreprocessor()
data_with_features = preprocessor.create_features(
    data,
    features=['day_of_week', 'month', 'quarter', 'is_weekend',
              'is_holiday', 'lag_7', 'lag_30', 'rolling_mean_7']
)

# 3. Split data
train_size = int(0.8 * len(data))
train_data = data_with_features[:train_size]
test_data = data_with_features[train_size:]

# 4. Create ensemble forecaster
forecaster = EnsembleForecaster(
    models=['arima', 'prophet', 'xgboost', 'lstm'],
    ensemble_method='weighted_average'
)

# 5. Fit and forecast
forecaster.fit(train_data)
forecasts = forecaster.forecast(
    horizon=len(test_data),
    return_confidence_intervals=True
)

# 6. Evaluate
evaluator = TimeSeriesEvaluator()
metrics = evaluator.evaluate(
    y_true=test_data['sales'],
    y_pred=forecasts['predictions'],
    metrics=['rmse', 'mae', 'mape', 'smape']
)

# 7. Visualize
visualizer = TimeSeriesVisualizer()
visualizer.plot_forecast(
    train_data=train_data,
    test_data=test_data,
    forecasts=forecasts,
    confidence_intervals=True
)
```

---

## Best Practices

### 1. Data Management

#### Data Validation

Always validate your data before training:

```python
from core import DataValidator

validator = DataValidator(strict_mode=False)

# Add custom validation rules
validator.add_custom_rule('price_positive',
    lambda df: ValidationResult(
        passed=(df['price'] > 0).all(),
        level=ValidationLevel.ERROR,
        message="Negative prices found"
    )
)

# Validate
is_valid, results = validator.validate(data)
if not is_valid:
    validator.print_report(results)
    # Handle validation failures
```

#### Data Versioning

Track data versions for reproducibility:

```python
from utils import DataVersioner

versioner = DataVersioner('data/versions')

# Save versioned data
version_id = versioner.save(
    data=df,
    name='sales_data',
    metadata={'source': 'crm_system', 'date': '2024-01-15'}
)

# Load specific version
df = versioner.load('sales_data', version='v1.2.0')
```

### 2. Model Development

#### Use AutoML for Baseline

Start with AutoML to establish baseline performance:

```python
from models import AutoML

automl = AutoML(
    task_type='auto',  # Auto-detect
    time_limit=600,    # 10 minutes
    optimization_metric='auto'
)

automl.fit(X_train, y_train)
baseline_score = automl.score(X_test, y_test)
```

#### Ensemble for Better Performance

Combine multiple models for improved results:

```python
from models import StackingEnsemble

# Create diverse base models
base_models = [
    RandomForestModel(n_estimators=200),
    XGBoostModel(learning_rate=0.05),
    LightGBMModel(num_leaves=31),
    NeuralNetwork(hidden_layers=[100, 50])
]

# Stack with meta-learner
ensemble = StackingEnsemble(
    base_models=base_models,
    meta_model=LinearModel(regularization='ridge'),
    cv_folds=5
)

ensemble.fit(X_train, y_train)
```

### 3. Feature Engineering

#### Automated Feature Selection

Use multiple selection methods and combine results:

```python
from core import FeatureSelector

selector = FeatureSelector(task_type='regression')

# Get features from different methods
corr_features = selector.select_by_correlation(X, y, threshold=0.3)
mi_features = selector.select_by_mutual_info(X, y, n_features=30)
rf_features = selector.select_by_model_importance(X, y, n_features=30)

# Combine features (intersection or union)
selected_features = list(set(corr_features) & set(mi_features) & set(rf_features))
```

#### Create Domain-Specific Features

Leverage domain knowledge:

```python
from core import FeatureEngineer

# For e-commerce
data['revenue_per_customer'] = data['total_revenue'] / data['customer_count']
data['conversion_rate'] = data['purchases'] / data['visits']
data['avg_order_value'] = data['total_revenue'] / data['order_count']

# For real estate
data['price_per_sqft'] = data['price'] / data['square_feet']
data['bed_bath_ratio'] = data['bedrooms'] / (data['bathrooms'] + 1)
data['property_age'] = current_year - data['year_built']
```

### 4. Experiment Tracking

#### Always Track Experiments

Use experiment tracking from the start:

```python
from pipelines import ExperimentTracker

tracker = ExperimentTracker(
    experiment_name="project_name",
    auto_log=True  # Automatically log metrics, params, artifacts
)

# Use context manager
with tracker.start_run(run_name="experiment_v1") as run:
    # Your training code
    model.fit(X_train, y_train)

    # Metrics are logged automatically
    # Log additional info
    tracker.log_param("feature_set", "v2")
    tracker.log_metric("business_metric", calculated_value)
```

#### Compare Experiments Systematically

```python
# Compare multiple runs
comparison = tracker.compare_runs(
    metrics=['rmse', 'r2', 'training_time'],
    param_cols=['model_type', 'n_features']
)

# Get best configuration
best_run = tracker.get_best_run(
    metric='validation_rmse',
    mode='min'
)
```

### 5. Model Deployment

#### Create Robust Inference Pipelines

```python
from pipelines import InferencePipeline

# Configure for production
config = InferenceConfig(
    model_path='models/production_model.pkl',
    preprocessor_path='models/preprocessor.pkl',
    enable_drift_detection=True,
    enable_uncertainty=True,
    prediction_mode='async',  # For high throughput
    batch_size=1000
)

pipeline = InferencePipeline(config)

# Add monitoring
pipeline.enable_monitoring(
    log_predictions=True,
    alert_on_drift=True,
    performance_threshold=0.1
)
```

#### Implement A/B Testing

```python
from pipelines import ABTestingPipeline

ab_pipeline = ABTestingPipeline(
    model_a=current_model,
    model_b=challenger_model,
    traffic_split=0.1,  # 10% to challenger
    metrics=['response_time', 'accuracy', 'business_metric']
)

# Run A/B test
results = ab_pipeline.run(duration_days=7)
```

### 6. Performance Optimization

#### Use Parallel Processing

```python
from utils import ParallelProcessor

# For CPU-bound tasks
processor = ParallelProcessor(n_jobs=-1, backend='multiprocessing')

# For I/O-bound tasks
processor = ParallelProcessor(n_jobs=10, backend='threading')

# Process large dataset in parallel
results = processor.process_chunks(
    data=large_dataset,
    chunk_size=10000,
    func=process_chunk,
    show_progress=True
)
```

#### Optimize Memory Usage

```python
from core import DataOptimizer

optimizer = DataOptimizer()

# Reduce memory usage
df_optimized = optimizer.optimize_dtypes(df)
print(f"Memory reduced by {optimizer.get_memory_reduction()}%")

# Use chunking for large files
for chunk in TabularDataLoader().load_chunked('large_file.csv', chunksize=50000):
    # Process chunk
    processed = process_chunk(chunk)
    # Save results incrementally
    processed.to_csv('output.csv', mode='a', header=not os.path.exists('output.csv'))
```

### 7. Error Handling and Logging

#### Implement Comprehensive Logging

```python
from utils import setup_logger, get_logger

# Configure logging
setup_logger(
    level='INFO',
    log_file='ml_pipeline.log',
    format='detailed',
    rotation='daily'
)

logger = get_logger(__name__)

# Use structured logging
logger.info("Model training completed", extra={
    'model_type': 'RandomForest',
    'training_time': 120.5,
    'n_samples': 10000,
    'metrics': {'rmse': 0.123, 'r2': 0.95}
})
```

#### Handle Errors Gracefully

```python
from utils import retry_on_failure

@retry_on_failure(max_attempts=3, delay=1.0)
def train_model(X, y):
    try:
        model = ComplexModel()
        model.fit(X, y)
        return model
    except MemoryError:
        logger.error("Memory error, trying with smaller batch size")
        model.batch_size = model.batch_size // 2
        return train_model(X, y)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
```

### 8. Testing and Validation

#### Unit Testing for Data Processing

```python
import unittest
from core import DataPreprocessor

class TestDataPreprocessing(unittest.TestCase):
    def setUp(self):
        self.preprocessor = DataPreprocessor()
        self.test_data = pd.DataFrame({
            'numeric': [1, 2, np.nan, 4],
            'categorical': ['A', 'B', 'A', None]
        })

    def test_missing_value_imputation(self):
        processed = self.preprocessor.fit_transform(self.test_data)
        self.assertFalse(processed.isnull().any().any())

    def test_scaling(self):
        processed = self.preprocessor.fit_transform(self.test_data)
        self.assertAlmostEqual(processed['numeric'].mean(), 0, places=5)
```

#### Integration Testing for Pipelines

```python
def test_end_to_end_pipeline():
    # Test complete pipeline
    pipeline = TrainingPipeline(
        TrainingConfig(
            task_type='regression',
            models=['random_forest'],
            test_size=0.2
        )
    )

    # Use small test dataset
    test_data = create_test_dataset(n_samples=1000)

    # Run pipeline
    results = pipeline.run(test_data, target_column='target')

    # Assertions
    assert 'best_model' in results
    assert results['test_score'] > 0
    assert len(results['feature_importance']) > 0
```

### 9. Documentation

#### Document Your Pipeline

```python
"""
Sales Prediction Pipeline
========================

This pipeline predicts daily sales using historical data and external features.

Data Sources:
- Historical sales: data/sales_history.csv
- Weather data: data/weather.csv
- Holiday calendar: data/holidays.csv

Features:
- Lag features (7, 14, 30 days)
- Rolling statistics
- Weather indicators
- Holiday flags

Models:
- XGBoost (primary)
- LightGBM (secondary)
- Ensemble (production)

Performance:
- RMSE: 1,234 units
- MAPE: 5.6%
- R²: 0.89

Usage:
    python train_pipeline.py --config configs/sales_config.yaml
"""
```

#### Use Type Hints and Docstrings

```python
from typing import Tuple, Optional, Dict, Any

def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[pd.Series] = None,
    config: Dict[str, Any] = None
) -> Tuple[BaseModel, Dict[str, float]]:
    """
    Train a model with the given configuration.

    Args:
        X_train: Training features
        y_train: Training target
        X_val: Validation features (optional)
        y_val: Validation target (optional)
        config: Model configuration dictionary

    Returns:
        Tuple of (trained_model, metrics_dict)

    Raises:
        ValueError: If configuration is invalid
        MemoryError: If dataset is too large

    Example:
        >>> model, metrics = train_model(X_train, y_train, config={'n_estimators': 100})
        >>> print(f"Validation RMSE: {metrics['rmse']}")
    """
    # Implementation
    pass
```

### 10. Security and Privacy

#### Protect Sensitive Data

```python
from utils import DataAnonymizer

anonymizer = DataAnonymizer()

# Remove PII
df_safe = anonymizer.remove_pii(
    df,
    columns=['name', 'email', 'phone', 'ssn']
)

# Hash identifiers
df_safe['user_id'] = anonymizer.hash_column(df['user_id'])

# Add differential privacy
df_private = anonymizer.add_noise(
    df,
    epsilon=1.0,  # Privacy budget
    columns=['age', 'income']
)
```

#### Secure Model Deployment

```python
from utils import ModelSecurityWrapper

# Wrap model with security features
secure_model = ModelSecurityWrapper(
    model=trained_model,
    encrypt_predictions=True,
    rate_limit=100,  # Max requests per minute
    require_auth=True
)

# Deploy with authentication
secure_model.deploy(
    endpoint='https://api.example.com/predict',
    api_key=os.environ['API_KEY']
)
```

---

## Troubleshooting

### Common Issues and Solutions

#### 1. Memory Errors

```python
# Solution: Use chunking
for chunk in pd.read_csv('large_file.csv', chunksize=10000):
    processed_chunk = process(chunk)
    save_chunk(processed_chunk)

# Solution: Reduce data types
from utils import optimize_memory
df = optimize_memory(df)

# Solution: Use Dask for larger-than-memory datasets
import dask.dataframe as dd
ddf = dd.read_csv('huge_file.csv')
result = ddf.groupby('category').mean().compute()
```

#### 2. Slow Training

```python
# Solution: Use parallel training
from utils import parallel_train

models = parallel_train(
    model_list=[model1, model2, model3],
    X_train=X_train,
    y_train=y_train,
    n_jobs=-1
)

# Solution: Reduce model complexity
from models import AutoML

automl = AutoML(
    fast_mode=True,  # Use simpler models
    time_limit=300   # 5 minute limit
)
```

#### 3. Poor Model Performance

```python
# Solution: Better feature engineering
from core import FeatureEngineer, FeatureSelector

# Create more features
engineer = FeatureEngineer()
X_enhanced = engineer.create_all_features(X, feature_types=['polynomial', 'interactions', 'ratios'])

# Select best features
selector = FeatureSelector()
X_selected = selector.auto_select(X_enhanced, y, n_features='auto')

# Solution: Handle imbalanced data
from models import BalancedClassifier

model = BalancedClassifier(
    base_model='xgboost',
    balance_method='smote',
    class_weight='balanced'
)
```

---

## API Reference

For detailed API documentation, please refer to the auto-generated documentation at:

- [https://d-dziublenko.github.io/data-science-toolkit/api/](https://d-dziublenko.github.io/data-science-toolkit/api/)

Or generate locally:

```bash
cd docs
make html
open _build/html/index.html
```

---

## Contributing

I welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone the repository
git clone https://github.com/d-dziublenko/data-science-toolkit.git
cd data-science-toolkit

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest tests/

# Run linting
flake8 .
black --check .

# Build documentation
cd docs && make html
```

---

## License

This project is licensed under the AGPL-3.0 License - see the [LICENSE](LICENSE) file for details.

---

## Citation

If you use this toolkit in your research, please cite:

```bibtex
@software{dziublenko2025toolkit,
  author = {Dziublenko, Dmytro},
  title = {Universal Data Science Toolkit},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/d-dziublenko/data-science-toolkit}
}
```

---

## Support

- **Documentation**: [https://d-dziublenko.github.io/data-science-toolkit/](https://d-dziublenko.github.io/data-science-toolkit/)
- **Issues**: [GitHub Issues](https://github.com/d-dziublenko/data-science-toolkit/issues)
- **Discussions**: [GitHub Discussions](https://github.com/d-dziublenko/data-science-toolkit/discussions)
- **Email**: d.dziublenko@gmail.com

---

## Acknowledgments

This toolkit builds upon many excellent open-source projects including scikit-learn, XGBoost, LightGBM, TensorFlow, PyTorch, and many others. I am grateful to the maintainers and contributors of these projects.
