"""
regression_example.py
House price prediction using the Universal Data Science Toolkit
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import toolkit components
from core import TabularDataLoader, DataPreprocessor, FeatureEngineer, DataValidator
from models import AutoML, EnsembleModel, TargetTransformer
from evaluation import ModelEvaluator, ModelVisualizer, UncertaintyQuantifier
from pipelines import TrainingPipeline, ExperimentTracker
from utils import setup_logger

# Initialize logger
logger = setup_logger(level='INFO')


def basic_regression_example():
    """Basic regression example with minimal configuration."""
    print("\n=== Basic Regression Example ===")
    
    # 1. Load data
    loader = TabularDataLoader()
    data = loader.load('data/housing_prices.csv')
    
    # 2. Quick train with pipeline
    pipeline = TrainingPipeline(task_type='regression')
    results = pipeline.run(
        data=data,
        target_column='price',
        test_size=0.2,
        models=['random_forest', 'xgboost'],
        tune_hyperparameters=True
    )
    
    # 3. Display results
    print(f"\nBest Model: {results['best_model']}")
    print(f"Test RMSE: ${results['test_metrics']['rmse']:,.2f}")
    print(f"Test R²: {results['test_metrics']['r2']:.4f}")
    
    return results


def advanced_regression_example():
    """Advanced regression example with custom preprocessing and ensemble."""
    print("\n=== Advanced Regression Example ===")
    
    # 1. Initialize components
    loader = TabularDataLoader()
    preprocessor = DataPreprocessor()
    engineer = FeatureEngineer()
    validator = DataValidator()
    
    # 2. Load and validate data
    logger.info("Loading data...")
    data = loader.load('data/housing_prices.csv')
    
    # Validate data quality
    validation_results = validator.validate(
        data,
        checks=[
            'missing_values',
            'duplicates',
            'outliers',
            'data_types',
            'value_ranges'
        ]
    )
    
    if not validator.is_valid(validation_results):
        logger.warning("Data quality issues detected!")
        validator.print_report(validation_results)
    
    # 3. Feature engineering
    logger.info("Engineering features...")
    
    # Create polynomial features for numeric columns
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols.remove('price')  # Remove target
    
    poly_features = engineer.create_polynomial_features(
        data[numeric_cols],
        degree=2,
        interaction_only=True
    )
    
    # Create ratio features
    if 'sqft_living' in data.columns and 'sqft_lot' in data.columns:
        data['living_lot_ratio'] = data['sqft_living'] / (data['sqft_lot'] + 1)
    
    if 'bedrooms' in data.columns and 'bathrooms' in data.columns:
        data['bed_bath_ratio'] = data['bedrooms'] / (data['bathrooms'] + 0.5)
    
    # Combine features
    data_enhanced = pd.concat([data, poly_features], axis=1)
    
    # 4. Preprocessing
    logger.info("Preprocessing data...")
    
    # Handle missing values
    data_clean = preprocessor.handle_missing_values(
        data_enhanced,
        numeric_strategy='knn',
        categorical_strategy='mode'
    )
    
    # Remove outliers
    data_clean = preprocessor.remove_outliers(
        data_clean,
        method='isolation_forest',
        contamination=0.05
    )
    
    # Scale features
    feature_cols = [col for col in data_clean.columns if col != 'price']
    data_scaled = preprocessor.scale_features(
        data_clean,
        columns=feature_cols,
        method='robust'  # Robust to outliers
    )
    
    # 5. Feature selection
    logger.info("Selecting features...")
    selector = engineer.select_features(
        data_scaled[feature_cols],
        data_scaled['price'],
        method='mutual_info',
        n_features=20
    )
    
    selected_features = selector.get_selected_features()
    X = data_scaled[selected_features]
    y = data_scaled['price']
    
    # 6. Target transformation
    logger.info("Transforming target variable...")
    target_transformer = TargetTransformer(method='box-cox')
    y_transformed = target_transformer.fit_transform(y)
    
    # 7. Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_transformed, test_size=0.2, random_state=42
    )
    
    # 8. Train ensemble model
    logger.info("Training ensemble model...")
    
    # Initialize base models
    from models import BaseModel
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from xgboost import XGBRegressor
    from lightgbm import LGBMRegressor
    
    base_models = [
        ('rf', RandomForestRegressor(n_estimators=200, random_state=42)),
        ('gb', GradientBoostingRegressor(n_estimators=200, random_state=42)),
        ('xgb', XGBRegressor(n_estimators=200, random_state=42)),
        ('lgb', LGBMRegressor(n_estimators=200, random_state=42))
    ]
    
    # Create stacking ensemble
    ensemble = EnsembleModel(
        base_models=base_models,
        meta_model='ridge',
        use_probabilities=False,
        cv_folds=5
    )
    
    ensemble.fit(X_train, y_train)
    
    # 9. Make predictions with uncertainty
    logger.info("Making predictions...")
    uq = UncertaintyQuantifier(task_type='regression')
    
    # Get predictions with uncertainty
    uncertainty_results = uq.ensemble_uncertainty(
        [model for _, model in ensemble.base_models_],
        X_test,
        confidence_level=0.95
    )
    
    # Transform predictions back
    y_pred = target_transformer.inverse_transform(
        uncertainty_results['predictions']
    )
    y_test_original = target_transformer.inverse_transform(y_test)
    
    # 10. Evaluate model
    logger.info("Evaluating model...")
    evaluator = ModelEvaluator()
    
    metrics = evaluator.evaluate_regression(
        y_test_original,
        y_pred,
        metrics=['rmse', 'mae', 'r2', 'mape', 'explained_variance']
    )
    
    print("\n=== Model Performance ===")
    for metric, value in metrics.items():
        if metric in ['rmse', 'mae']:
            print(f"{metric.upper()}: ${value:,.2f}")
        elif metric == 'mape':
            print(f"{metric.upper()}: {value:.2%}")
        else:
            print(f"{metric.upper()}: {value:.4f}")
    
    # 11. Visualizations
    logger.info("Creating visualizations...")
    visualizer = ModelVisualizer()
    
    # Prediction vs actual
    visualizer.plot_predictions(
        y_test_original,
        y_pred,
        plot_type='scatter',
        title='House Price Predictions vs Actual'
    )
    
    # Feature importance
    visualizer.plot_feature_importance(
        ensemble,
        feature_names=selected_features,
        top_n=15,
        title='Top 15 Most Important Features'
    )
    
    # Residual analysis
    visualizer.plot_residuals(
        y_test_original,
        y_pred,
        plot_type='all'  # Creates multiple residual plots
    )
    
    # Prediction intervals
    lower_bound = target_transformer.inverse_transform(
        uncertainty_results['lower_bound']
    )
    upper_bound = target_transformer.inverse_transform(
        uncertainty_results['upper_bound']
    )
    
    visualizer.plot_prediction_intervals(
        y_test_original,
        y_pred,
        lower_bound,
        upper_bound,
        sample_size=100,  # Plot subset for clarity
        title='95% Prediction Intervals'
    )
    
    # 12. Save model and results
    logger.info("Saving model and results...")
    
    # Save ensemble
    from utils import ModelSerializer
    serializer = ModelSerializer()
    
    serializer.save(
        {
            'ensemble': ensemble,
            'target_transformer': target_transformer,
            'feature_selector': selector,
            'selected_features': selected_features
        },
        'models/house_price_ensemble.pkl',
        metadata={
            'metrics': metrics,
            'training_date': pd.Timestamp.now().isoformat(),
            'n_features': len(selected_features)
        }
    )
    
    return ensemble, metrics


def experiment_tracking_example():
    """Example with experiment tracking and comparison."""
    print("\n=== Experiment Tracking Example ===")
    
    # Initialize experiment tracker
    tracker = ExperimentTracker(
        experiment_name='house_price_prediction',
        base_dir='experiments'
    )
    
    # Load data
    loader = TabularDataLoader()
    data = loader.load('data/housing_prices.csv')
    
    # Define experiments to run
    experiments = [
        {
            'name': 'baseline_linear',
            'model': 'linear_regression',
            'preprocessing': 'standard',
            'feature_selection': None
        },
        {
            'name': 'rf_with_selection',
            'model': 'random_forest',
            'preprocessing': 'robust',
            'feature_selection': 'mutual_info'
        },
        {
            'name': 'xgb_with_poly',
            'model': 'xgboost',
            'preprocessing': 'minmax',
            'feature_selection': 'f_regression',
            'polynomial_features': True
        }
    ]
    
    # Run experiments
    for exp_config in experiments:
        # Start experiment run
        with tracker.start_run(run_name=exp_config['name']) as run:
            # Log parameters
            tracker.log_params(exp_config)
            
            # Run training pipeline
            pipeline = TrainingPipeline(
                task_type='regression',
                preprocessing=exp_config['preprocessing'],
                feature_selection=exp_config.get('feature_selection'),
                polynomial_features=exp_config.get('polynomial_features', False)
            )
            
            results = pipeline.run(
                data=data,
                target_column='price',
                models=[exp_config['model']]
            )
            
            # Log metrics
            tracker.log_metrics(results['test_metrics'])
            
            # Log model
            tracker.log_model(results['best_model_object'], 'model')
    
    # Compare experiments
    comparison = tracker.compare_experiments(
        experiment_ids=[exp['name'] for exp in experiments],
        metrics=['rmse', 'r2', 'mae']
    )
    
    print("\n=== Experiment Comparison ===")
    print(comparison)
    
    # Get best model
    best_model_info = tracker.get_best_model(
        metric='rmse',
        mode='min'
    )
    
    print(f"\nBest Model: {best_model_info['run_name']}")
    print(f"RMSE: ${best_model_info['metrics']['rmse']:,.2f}")
    
    return tracker


if __name__ == "__main__":
    # Ensure data directory exists
    Path('data').mkdir(exist_ok=True)
    
    # Note: You'll need to have housing_prices.csv in the data directory
    # You can use any regression dataset with a 'price' column
    
    try:
        # Run basic example
        basic_results = basic_regression_example()
        
        print("\n" + "="*50 + "\n")
        
        # Run advanced example
        ensemble_model, advanced_metrics = advanced_regression_example()
        
        print("\n" + "="*50 + "\n")
        
        # Run experiment tracking example
        experiment_tracker = experiment_tracking_example()
        
    except FileNotFoundError:
        print("\nNote: Please ensure you have 'housing_prices.csv' in the 'data' directory.")
        print("You can use any regression dataset with a 'price' column as the target.")