"""
time_series_example.py
Sales forecasting using the Universal Data Science Toolkit
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
from utils import setup_logger, ParallelProcessor

# Import ML libraries with error handling
try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    warnings.warn("scikit-learn not installed. Some features will be limited.")

try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    warnings.warn("XGBoost not installed. Will use alternative models.")

try:
    from lightgbm import LGBMRegressor
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    warnings.warn("LightGBM not installed. Will use alternative models.")

# Initialize logger
logger = setup_logger(level='INFO')


def basic_time_series_example():
    """Basic time series forecasting with minimal configuration."""
    print("\n=== Basic Time Series Example ===")
    
    # 1. Load time series data
    loader = TabularDataLoader()
    data = loader.load('data/daily_sales.csv', parse_dates=['date'])
    data = data.sort_values('date').set_index('date')
    
    # 2. Create simple lag features
    data['sales_lag1'] = data['sales'].shift(1)
    data['sales_lag7'] = data['sales'].shift(7)
    data['rolling_mean_7'] = data['sales'].rolling(7).mean()
    
    # Remove NaN rows
    data = data.dropna()
    
    # 3. Train with pipeline
    pipeline = TrainingPipeline(
        task_type='regression',
        time_series=True
    )
    
    results = pipeline.run(
        data=data.reset_index(),
        target_column='sales',
        date_column='date',
        models=['lightgbm', 'xgboost'],
        cv_strategy='time_series',  # Use time series split
        test_size=30  # Last 30 days
    )
    
    # 4. Display results
    print(f"\nBest Model: {results['best_model']}")
    print(f"Test MAPE: {results['test_metrics']['mape']:.2%}")
    print(f"Test RMSE: {results['test_metrics']['rmse']:.2f}")
    
    return results


def advanced_time_series_example():
    """Advanced time series with comprehensive feature engineering and forecasting."""
    print("\n=== Advanced Time Series Example ===")
    
    # 1. Initialize components
    loader = TabularDataLoader()
    preprocessor = DataPreprocessor()
    engineer = FeatureEngineer()
    validator = DataValidator()
    
    # 2. Load and validate time series data
    logger.info("Loading time series data...")
    data = loader.load('data/hourly_demand.csv', parse_dates=['timestamp'])
    
    # Ensure proper time series format
    data = data.sort_values('timestamp')
    data.set_index('timestamp', inplace=True)
    
    # Check for missing timestamps
    date_range = pd.date_range(
        start=data.index.min(),
        end=data.index.max(),
        freq='H'  # Hourly frequency
    )
    missing_dates = date_range.difference(data.index)
    
    if len(missing_dates) > 0:
        logger.warning(f"Found {len(missing_dates)} missing timestamps")
        # Fill missing dates with NaN
        data = data.reindex(date_range)
    
    # 3. Handle missing values in time series
    logger.info("Handling missing values...")
    
    # Use time series specific imputation
    data['demand'] = data['demand'].interpolate(method='time')
    
    # Forward fill for categorical variables if any
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns
    if len(categorical_cols) > 0:
        data[categorical_cols] = data[categorical_cols].fillna(method='ffill')
    
    # 4. Create comprehensive time features
    logger.info("Engineering time-based features...")
    
    # Basic time features
    data['hour'] = data.index.hour
    data['day'] = data.index.day
    data['dayofweek'] = data.index.dayofweek
    data['month'] = data.index.month
    data['quarter'] = data.index.quarter
    data['year'] = data.index.year
    data['dayofyear'] = data.index.dayofyear
    data['weekofyear'] = data.index.isocalendar().week
    
    # Cyclical encoding for periodic features
    data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24)
    data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24)
    data['day_sin'] = np.sin(2 * np.pi * data['day'] / 31)
    data['day_cos'] = np.cos(2 * np.pi * data['day'] / 31)
    data['month_sin'] = np.sin(2 * np.pi * data['month'] / 12)
    data['month_cos'] = np.cos(2 * np.pi * data['month'] / 12)
    
    # Binary features
    data['is_weekend'] = (data['dayofweek'] >= 5).astype(int)
    data['is_month_start'] = (data['day'] == 1).astype(int)
    data['is_month_end'] = (data['day'] == data.index.days_in_month).astype(int)
    
    # 5. Create lag features with multiple horizons
    logger.info("Creating lag features...")
    
    target_col = 'demand'
    
    # Short-term lags (hourly)
    for lag in [1, 2, 3, 6, 12, 24]:
        data[f'lag_{lag}h'] = data[target_col].shift(lag)
    
    # Daily lags
    for lag in [1, 2, 3, 7, 14, 21, 28]:
        data[f'lag_{lag}d'] = data[target_col].shift(lag * 24)
    
    # 6. Create rolling statistics
    logger.info("Creating rolling window features...")
    
    # Multiple window sizes
    windows = {
        '6h': 6,
        '12h': 12,
        '24h': 24,
        '3d': 72,
        '7d': 168,
        '14d': 336
    }
    
    for window_name, window_size in windows.items():
        # Rolling statistics
        data[f'roll_mean_{window_name}'] = data[target_col].rolling(window_size).mean()
        data[f'roll_std_{window_name}'] = data[target_col].rolling(window_size).std()
        data[f'roll_min_{window_name}'] = data[target_col].rolling(window_size).min()
        data[f'roll_max_{window_name}'] = data[target_col].rolling(window_size).max()
        
        # Rolling quantiles
        data[f'roll_q25_{window_name}'] = data[target_col].rolling(window_size).quantile(0.25)
        data[f'roll_q75_{window_name}'] = data[target_col].rolling(window_size).quantile(0.75)
    
    # 7. Create expanding window features
    data['expanding_mean'] = data[target_col].expanding(min_periods=168).mean()
    data['expanding_std'] = data[target_col].expanding(min_periods=168).std()
    
    # 8. Create seasonal features
    logger.info("Creating seasonal features...")
    
    # Fourier terms for multiple seasonalities
    def create_fourier_terms(index, period, n_terms):
        """Create Fourier terms for seasonal patterns."""
        features = pd.DataFrame(index=index)
        for i in range(1, n_terms + 1):
            features[f'sin_{period}_{i}'] = np.sin(2 * np.pi * i * np.arange(len(index)) / period)
            features[f'cos_{period}_{i}'] = np.cos(2 * np.pi * i * np.arange(len(index)) / period)
        return features
    
    # Daily seasonality (24 hours)
    daily_fourier = create_fourier_terms(data.index, period=24, n_terms=4)
    data = pd.concat([data, daily_fourier], axis=1)
    
    # Weekly seasonality (168 hours)
    weekly_fourier = create_fourier_terms(data.index, period=168, n_terms=3)
    data = pd.concat([data, weekly_fourier], axis=1)
    
    # 9. External features (if available)
    logger.info("Adding external features...")
    
    # Example: holidays, weather, events
    # This would normally come from external data sources
    
    # Simulate holiday feature
    holidays = pd.to_datetime([
        '2024-01-01', '2024-07-04', '2024-12-25'
    ])
    data['is_holiday'] = data.index.isin(holidays).astype(int)
    
    # Days until/since holiday
    for holiday in holidays:
        if holiday in data.index:
            days_diff = (data.index - holiday).days
            data[f'days_to_holiday_{holiday.strftime("%m%d")}'] = days_diff
            data[f'days_to_holiday_{holiday.strftime("%m%d")}_abs'] = np.abs(days_diff)
    
    # 10. Remove rows with NaN (from lag creation)
    logger.info("Finalizing feature set...")
    
    # Keep track of feature groups
    feature_groups = {
        'time': ['hour', 'day', 'dayofweek', 'month', 'quarter', 'year'],
        'cyclical': [col for col in data.columns if '_sin' in col or '_cos' in col],
        'lags': [col for col in data.columns if col.startswith('lag_')],
        'rolling': [col for col in data.columns if col.startswith('roll_')],
        'expanding': [col for col in data.columns if col.startswith('expanding_')],
        'external': ['is_weekend', 'is_holiday', 'is_month_start', 'is_month_end']
    }
    
    # Drop NaN rows
    data_clean = data.dropna()
    
    # 11. Split data for time series
    logger.info("Splitting data for time series validation...")
    
    # Use last 30 days as test set
    test_days = 30
    test_size = test_days * 24  # Hourly data
    
    train_data = data_clean[:-test_size]
    test_data = data_clean[-test_size:]
    
    # Prepare features and target
    feature_cols = [col for col in data_clean.columns if col != target_col]
    X_train = train_data[feature_cols]
    y_train = train_data[target_col]
    X_test = test_data[feature_cols]
    y_test = test_data[target_col]
    
    # 12. Feature selection for time series
    logger.info("Selecting most important features...")
    
    # Use available model for initial feature importance
    if HAS_LIGHTGBM:
        from lightgbm import LGBMRegressor
        selector_model = LGBMRegressor(
            n_estimators=100,
            random_state=42,
            verbose=-1
        )
    elif HAS_XGBOOST:
        selector_model = XGBRegressor(
            n_estimators=100,
            random_state=42,
            verbosity=0
        )
    elif HAS_SKLEARN:
        selector_model = RandomForestRegressor(
            n_estimators=100,
            random_state=42
        )
    else:
        logger.error("No suitable ML library installed for feature selection")
        raise ImportError("Please install scikit-learn, XGBoost, or LightGBM")
    
    selector_model.fit(X_train, y_train)
    
    # Get feature importance
    feature_importance = pd.Series(
        selector_model.feature_importances_,
        index=X_train.columns
    ).sort_values(ascending=False)
    
    # Select top features
    n_features = min(50, len(feature_cols))
    selected_features = feature_importance.head(n_features).index.tolist()
    
    print(f"\nTop 10 Features:")
    for i, (feat, imp) in enumerate(feature_importance.head(10).items()):
        print(f"{i+1}. {feat}: {imp:.0f}")
    
    X_train_selected = X_train[selected_features]
    X_test_selected = X_test[selected_features]
    
    # 13. Train ensemble model with time series cross-validation
    logger.info("Training time series ensemble...")
    
    from sklearn.model_selection import TimeSeriesSplit
    
    # Configure time series cross-validation
    tscv = TimeSeriesSplit(n_splits=5)
    
    # Initialize models based on available libraries
    models = []
    
    if HAS_LIGHTGBM:
        models.append(('lgb', LGBMRegressor(n_estimators=200, random_state=42, verbose=-1)))
    
    if HAS_XGBOOST:
        models.append(('xgb', XGBRegressor(n_estimators=200, random_state=42, verbosity=0)))
    
    if HAS_SKLEARN:
        models.extend([
            ('rf', RandomForestRegressor(n_estimators=200, random_state=42)),
            ('gbm', GradientBoostingRegressor(n_estimators=200, random_state=42))
        ])
    
    # Ensure we have at least some models
    if not models:
        logger.error("No ML libraries available for ensemble")
        raise ImportError("Please install at least one of: scikit-learn, XGBoost, or LightGBM")
    
    # If we only have one model, duplicate it with different parameters
    if len(models) == 1:
        logger.warning("Only one model type available. Creating ensemble with different configurations.")
        model_name, model = models[0]
        if hasattr(model, 'n_estimators'):
            models.append((f'{model_name}_50', type(model)(n_estimators=50, random_state=42)))
            models.append((f'{model_name}_300', type(model)(n_estimators=300, random_state=43)))
    
    # Create ensemble with time series CV
    ensemble = EnsembleModel(
        base_models=models,
        meta_model='linear',
        cv_folds=tscv,
        use_probabilities=False
    )
    
    ensemble.fit(X_train_selected, y_train)
    
    # 14. Make predictions with uncertainty
    logger.info("Making predictions with uncertainty quantification...")
    
    uq = UncertaintyQuantifier(task_type='regression')
    
    # Bootstrap uncertainty for time series
    uncertainty_results = uq.bootstrap_uncertainty(
        ensemble,
        X_train_selected,
        y_train,
        X_test_selected,
        n_bootstrap=100,
        confidence_level=0.95
    )
    
    predictions = uncertainty_results['predictions']
    lower_bound = uncertainty_results['lower_bound']
    upper_bound = uncertainty_results['upper_bound']
    
    # 15. Evaluate time series predictions
    logger.info("Evaluating predictions...")
    
    evaluator = ModelEvaluator()
    
    # Time series specific metrics
    metrics = evaluator.evaluate_time_series(
        y_test,
        predictions,
        metrics=['mape', 'smape', 'mase', 'rmse', 'mae']
    )
    
    print("\n=== Time Series Metrics ===")
    for metric, value in metrics.items():
        if metric in ['mape', 'smape']:
            print(f"{metric.upper()}: {value:.2%}")
        else:
            print(f"{metric.upper()}: {value:.2f}")
    
    # Directional accuracy
    if len(y_test) > 1:
        actual_direction = np.diff(y_test.values)
        pred_direction = np.diff(predictions)
        directional_accuracy = np.mean(
            np.sign(actual_direction) == np.sign(pred_direction)
        )
        print(f"Directional Accuracy: {directional_accuracy:.2%}")
    
    # 16. Visualize results
    logger.info("Creating visualizations...")
    
    visualizer = ModelVisualizer()
    
    # Time series forecast plot
    if hasattr(visualizer, 'plot_time_series_forecast'):
        visualizer.plot_time_series_forecast(
            dates=test_data.index,
            actual=y_test,
            forecast=predictions,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            title='Hourly Demand Forecast with 95% Prediction Intervals'
        )
    else:
        # Fallback to basic plotting
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 6))
        plt.plot(test_data.index, y_test, label='Actual', alpha=0.7)
        plt.plot(test_data.index, predictions, label='Forecast', alpha=0.7)
        plt.fill_between(test_data.index, lower_bound, upper_bound, 
                        alpha=0.2, label='95% Prediction Interval')
        plt.xlabel('Date')
        plt.ylabel('Demand')
        plt.title('Hourly Demand Forecast with 95% Prediction Intervals')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    # Zoomed view (last 7 days)
    last_week = -7 * 24
    if hasattr(visualizer, 'plot_time_series_forecast'):
        visualizer.plot_time_series_forecast(
            dates=test_data.index[last_week:],
            actual=y_test.iloc[last_week:],
            forecast=predictions[last_week:],
            lower_bound=lower_bound[last_week:],
            upper_bound=upper_bound[last_week:],
            title='Last 7 Days - Detailed View'
        )
    
    # Residual analysis
    residuals = y_test - predictions
    
    if hasattr(visualizer, 'plot_time_series_residuals'):
        visualizer.plot_time_series_residuals(
            dates=test_data.index,
            residuals=residuals,
            title='Forecast Residuals Analysis'
        )
    else:
        # Fallback residual plot
        plt.figure(figsize=(12, 6))
        plt.subplot(2, 1, 1)
        plt.plot(test_data.index, residuals)
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        plt.xlabel('Date')
        plt.ylabel('Residuals')
        plt.title('Forecast Residuals Over Time')
        
        plt.subplot(2, 1, 2)
        plt.hist(residuals, bins=30, edgecolor='black')
        plt.xlabel('Residual Value')
        plt.ylabel('Frequency')
        plt.title('Residual Distribution')
        plt.tight_layout()
        plt.show()
    
    # ACF/PACF plots for residuals
    try:
        from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        plot_acf(residuals, lags=48, ax=axes[0])
        axes[0].set_title('Residual Autocorrelation Function')
        plot_pacf(residuals, lags=48, ax=axes[1])
        axes[1].set_title('Residual Partial Autocorrelation Function')
        plt.tight_layout()
        plt.show()
    except ImportError:
        logger.warning("statsmodels not installed. Skipping ACF/PACF plots.")
    
    # Feature importance by category
    feature_importance_df = pd.DataFrame({
        'feature': selected_features,
        'importance': ensemble.base_models_[0][1].feature_importances_[:len(selected_features)]
    })
    
    # Categorize features
    for feat in feature_importance_df['feature']:
        for group, features in feature_groups.items():
            if feat in features:
                feature_importance_df.loc[
                    feature_importance_df['feature'] == feat, 'category'
                ] = group
                break
        else:
            feature_importance_df.loc[
                feature_importance_df['feature'] == feat, 'category'
            ] = 'other'
    
    # Plot importance by category
    category_importance = feature_importance_df.groupby('category')['importance'].sum()
    
    if hasattr(visualizer, 'plot_feature_importance_by_category'):
        visualizer.plot_feature_importance_by_category(
            category_importance,
            title='Feature Importance by Category'
        )
    else:
        # Fallback bar plot
        plt.figure(figsize=(10, 6))
        category_importance.sort_values(ascending=True).plot(kind='barh')
        plt.xlabel('Total Importance')
        plt.title('Feature Importance by Category')
        plt.tight_layout()
        plt.show()
    
    # 17. Multi-step ahead forecasting
    logger.info("Creating multi-step forecasts...")
    
    # Forecast next 24 hours
    forecast_horizon = 24
    multi_step_forecasts = []
    multi_step_lower = []
    multi_step_upper = []
    
    # Use recursive strategy
    last_known = X_test_selected.iloc[-1:].copy()
    
    for h in range(forecast_horizon):
        # Make prediction
        pred_result = uq.bootstrap_uncertainty(
            ensemble,
            X_train_selected,
            y_train,
            last_known,
            n_bootstrap=50,
            confidence_level=0.95
        )
        
        pred = pred_result['predictions'][0]
        multi_step_forecasts.append(pred)
        multi_step_lower.append(pred_result['lower_bound'][0])
        multi_step_upper.append(pred_result['upper_bound'][0])
        
        # Update features for next prediction
        # This is simplified - in practice, you'd update all lag features
        if 'lag_1h' in last_known.columns:
            last_known['lag_1h'] = pred
        
        # Update time features
        next_time = test_data.index[-1] + pd.Timedelta(hours=h+1)
        last_known['hour'] = next_time.hour
        last_known['day'] = next_time.day
        last_known['dayofweek'] = next_time.dayofweek
        # ... update other time features as needed
    
    # Create future dates
    future_dates = pd.date_range(
        start=test_data.index[-1] + pd.Timedelta(hours=1),
        periods=forecast_horizon,
        freq='H'
    )
    
    # Plot multi-step forecast
    if hasattr(visualizer, 'plot_multi_step_forecast'):
        visualizer.plot_multi_step_forecast(
            historical_dates=test_data.index[-48:],  # Last 2 days
            historical_values=y_test.iloc[-48:],
            forecast_dates=future_dates,
            forecasts=multi_step_forecasts,
            lower_bounds=multi_step_lower,
            upper_bounds=multi_step_upper,
            title='24-Hour Ahead Forecast'
        )
    else:
        # Fallback plot
        plt.figure(figsize=(12, 6))
        
        # Plot historical data
        plt.plot(test_data.index[-48:], y_test.iloc[-48:], 
                label='Historical', color='blue', alpha=0.7)
        
        # Plot forecast
        plt.plot(future_dates, multi_step_forecasts, 
                label='Forecast', color='red', alpha=0.7)
        
        # Plot prediction intervals
        plt.fill_between(future_dates, multi_step_lower, multi_step_upper,
                        alpha=0.2, color='red', label='95% Prediction Interval')
        
        plt.xlabel('Date')
        plt.ylabel('Demand')
        plt.title('24-Hour Ahead Forecast')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    # 18. Save model and artifacts
    logger.info("Saving model and artifacts...")
    
    from utils import ModelSerializer
    serializer = ModelSerializer()
    
    # Save complete forecasting system
    forecasting_artifacts = {
        'ensemble': ensemble,
        'feature_selector': selected_features,
        'feature_groups': feature_groups,
        'last_train_date': train_data.index[-1],
        'forecast_frequency': 'H',
        'metrics': metrics
    }
    
    serializer.save(
        forecasting_artifacts,
        'models/demand_forecasting_system.pkl',
        metadata={
            'test_mape': metrics['mape'],
            'test_rmse': metrics['rmse'],
            'n_features': len(selected_features),
            'ensemble_models': [name for name, _ in models]
        }
    )
    
    return ensemble, metrics, uncertainty_results


def anomaly_detection_time_series():
    """Time series anomaly detection example."""
    print("\n=== Time Series Anomaly Detection ===")
    
    # 1. Load data
    loader = TabularDataLoader()
    data = loader.load('data/sensor_readings.csv', parse_dates=['timestamp'])
    data.set_index('timestamp', inplace=True)
    
    # 2. Create features for anomaly detection
    engineer = FeatureEngineer()
    
    # Statistical features
    data['rolling_mean'] = data['value'].rolling(24).mean()
    data['rolling_std'] = data['value'].rolling(24).std()
    data['z_score'] = (data['value'] - data['rolling_mean']) / data['rolling_std']
    
    # Isolation Forest for anomaly detection
    if HAS_SKLEARN:
        from sklearn.ensemble import IsolationForest
        
        # Prepare features
        features = ['value', 'rolling_mean', 'rolling_std', 'z_score']
        X = data[features].dropna()
        
        # Train anomaly detector
        detector = IsolationForest(
            contamination=0.01,  # Expect 1% anomalies
            random_state=42
        )
        
        anomalies = detector.fit_predict(X)
        X['anomaly'] = anomalies
        X['is_anomaly'] = (anomalies == -1)
    else:
        # Fallback to simple statistical method
        logger.warning("scikit-learn not available. Using simple z-score method.")
        X = data.dropna()
        X['is_anomaly'] = np.abs(X['z_score']) > 3  # 3 standard deviations
        detector = None
    
    # 3. Visualize anomalies
    visualizer = ModelVisualizer()
    
    if hasattr(visualizer, 'plot_time_series_anomalies'):
        visualizer.plot_time_series_anomalies(
            dates=X.index,
            values=X['value'],
            anomalies=X['is_anomaly'],
            title='Time Series Anomaly Detection'
        )
    else:
        # Fallback plot
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 6))
        
        # Plot normal points
        normal_mask = ~X['is_anomaly']
        plt.scatter(X.index[normal_mask], X['value'][normal_mask], 
                   c='blue', alpha=0.6, s=20, label='Normal')
        
        # Plot anomalies
        anomaly_mask = X['is_anomaly']
        plt.scatter(X.index[anomaly_mask], X['value'][anomaly_mask], 
                   c='red', alpha=0.8, s=50, label='Anomaly')
        
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.title('Time Series Anomaly Detection')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    print(f"\nDetected {X['is_anomaly'].sum()} anomalies out of {len(X)} observations")
    print(f"Anomaly rate: {X['is_anomaly'].mean():.2%}")
    
    # 4. Analyze anomaly patterns
    anomaly_stats = X[X['is_anomaly']].describe()
    normal_stats = X[~X['is_anomaly']].describe()
    
    print("\n=== Anomaly Statistics ===")
    print("Anomalous readings:")
    print(anomaly_stats[['value', 'z_score']])
    print("\nNormal readings:")
    print(normal_stats[['value', 'z_score']])
    
    return detector, X


if __name__ == "__main__":
    # Ensure data directory exists
    Path('data').mkdir(exist_ok=True)
    
    # Note: You'll need appropriate time series datasets
    
    try:
        # Run basic example
        basic_results = basic_time_series_example()
        
        print("\n" + "="*50 + "\n")
        
        # Run advanced example
        ensemble, metrics, uncertainty = advanced_time_series_example()
        
        print("\n" + "="*50 + "\n")
        
        # Run anomaly detection
        anomaly_detector, anomaly_results = anomaly_detection_time_series()
        
    except FileNotFoundError as e:
        print(f"\nNote: {e}")
        print("Please ensure you have the required datasets in the 'data' directory:")
        print("- daily_sales.csv (basic time series)")
        print("- hourly_demand.csv (advanced time series)")
        print("- sensor_readings.csv (anomaly detection)")