"""
pipelines/inference.py
Inference pipeline for making predictions with trained models.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Union, Tuple, Callable
from pathlib import Path
import json
import joblib
import logging
from datetime import datetime
import warnings
from dataclasses import dataclass
from enum import Enum

# Import from project modules
from ..core.data_loader import DataLoader
from ..core.preprocessing import DataPreprocessor
from ..core.validation import DataValidator
from ..models.base import BaseModel
from ..evaluation.uncertainty import UncertaintyQuantifier
from ..evaluation.drift import DataDriftDetector
from ..utils.file_io import FileHandler, load_object
from ..utils.parallel import ParallelProcessor

logger = logging.getLogger(__name__)


class PredictionMode(Enum):
    """Prediction modes for inference."""
    SINGLE = "single"
    BATCH = "batch"
    STREAMING = "streaming"
    ASYNC = "async"


@dataclass
class InferenceConfig:
    """Configuration for inference pipeline."""
    model_path: str
    preprocessor_path: Optional[str] = None
    prediction_mode: PredictionMode = PredictionMode.BATCH
    batch_size: int = 1000
    n_jobs: int = 1
    enable_uncertainty: bool = False
    uncertainty_method: str = "bootstrap"
    enable_drift_detection: bool = False
    drift_reference_path: Optional[str] = None
    output_format: str = "csv"
    include_metadata: bool = True
    confidence_level: float = 0.95


class InferencePipeline:
    """
    Complete inference pipeline for making predictions.
    
    This class handles loading models, preprocessing data, making predictions,
    and post-processing results including uncertainty quantification and
    drift detection.
    """
    
    def __init__(self, config: Union[InferenceConfig, Dict[str, Any]]):
        """
        Initialize the inference pipeline.
        
        Args:
            config: Inference configuration
        """
        if isinstance(config, dict):
            config = InferenceConfig(**config)
        
        self.config = config
        self.model = None
        self.preprocessor = None
        self.uncertainty_quantifier = None
        self.drift_detector = None
        
        # Load components
        self._load_components()
        
        logger.info(f"Initialized inference pipeline with mode: {config.prediction_mode.value}")
    
    def _load_components(self):
        """Load model and preprocessing components."""
        # Load model
        logger.info(f"Loading model from {self.config.model_path}")
        self.model = load_object(self.config.model_path)
        
        # Load preprocessor if specified
        if self.config.preprocessor_path:
            logger.info(f"Loading preprocessor from {self.config.preprocessor_path}")
            self.preprocessor = load_object(self.config.preprocessor_path)
        
        # Setup uncertainty quantification if enabled
        if self.config.enable_uncertainty:
            self.uncertainty_quantifier = UncertaintyQuantifier(
                task_type='regression' if hasattr(self.model, 'predict') else 'classification'
            )
        
        # Setup drift detection if enabled
        if self.config.enable_drift_detection and self.config.drift_reference_path:
            reference_data = pd.read_csv(self.config.drift_reference_path)
            self.drift_detector = DataDriftDetector(reference_data)
    
    def predict(self, data: Union[pd.DataFrame, np.ndarray, str, List[Dict]],
                return_uncertainty: bool = None,
                check_drift: bool = None) -> pd.DataFrame:
        """
        Make predictions on input data.
        
        Args:
            data: Input data (DataFrame, array, file path, or list of dicts)
            return_uncertainty: Whether to return uncertainty estimates
            check_drift: Whether to check for data drift
            
        Returns:
            DataFrame with predictions and optional metadata
        """
        # Load data if path is provided
        if isinstance(data, str):
            data = self._load_data(data)
        elif isinstance(data, list):
            data = pd.DataFrame(data)
        elif isinstance(data, np.ndarray):
            data = pd.DataFrame(data)
        
        # Validate input data
        self._validate_input(data)
        
        # Check for drift if enabled
        if check_drift or (check_drift is None and self.config.enable_drift_detection):
            drift_result = self._check_drift(data)
            if drift_result and drift_result.drift_type.value != "no_drift":
                logger.warning(f"Data drift detected: {drift_result.drift_type.value}")
        
        # Preprocess data
        if self.preprocessor:
            logger.info("Preprocessing data")
            data_processed = self.preprocessor.transform(data)
        else:
            data_processed = data
        
        # Make predictions based on mode
        if self.config.prediction_mode == PredictionMode.SINGLE:
            results = self._predict_single(data_processed)
        elif self.config.prediction_mode == PredictionMode.BATCH:
            results = self._predict_batch(data_processed)
        elif self.config.prediction_mode == PredictionMode.STREAMING:
            results = self._predict_streaming(data_processed)
        else:
            results = self._predict_async(data_processed)
        
        # Add uncertainty if requested
        if return_uncertainty or (return_uncertainty is None and self.config.enable_uncertainty):
            results = self._add_uncertainty(data_processed, results)
        
        # Add metadata
        if self.config.include_metadata:
            results = self._add_metadata(results, data)
        
        return results
    
    def _load_data(self, path: str) -> pd.DataFrame:
        """Load data from file."""
        file_handler = FileHandler()
        return file_handler.read(path)
    
    def _validate_input(self, data: pd.DataFrame):
        """Validate input data."""
        # Check required features
        if hasattr(self.model, 'feature_names_in_'):
            required_features = self.model.feature_names_in_
            missing_features = set(required_features) - set(data.columns)
            if missing_features:
                raise ValueError(f"Missing required features: {missing_features}")
        
        # Check data types
        validator = DataValidator()
        is_valid, validation_results = validator.validate(data)
        
        if not is_valid:
            logger.warning("Data validation failed")
            for result in validation_results:
                if result.severity == "error":
                    logger.error(f"{result.check_name}: {result.message}")
    
    def _check_drift(self, data: pd.DataFrame) -> Optional[Any]:
        """Check for data drift."""
        if self.drift_detector:
            try:
                return self.drift_detector.detect_drift(data)
            except Exception as e:
                logger.error(f"Drift detection failed: {str(e)}")
                return None
        return None
    
    def _predict_single(self, data: pd.DataFrame) -> pd.DataFrame:
        """Make single predictions."""
        predictions = self.model.predict(data)
        
        results = pd.DataFrame({
            'prediction': predictions
        })
        
        # Add probabilities for classification
        if hasattr(self.model, 'predict_proba'):
            proba = self.model.predict_proba(data)
            for i, class_label in enumerate(self.model.classes_):
                results[f'probability_{class_label}'] = proba[:, i]
        
        return results
    
    def _predict_batch(self, data: pd.DataFrame) -> pd.DataFrame:
        """Make batch predictions."""
        n_samples = len(data)
        batch_size = self.config.batch_size
        
        all_results = []
        
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            batch_data = data.iloc[start_idx:end_idx]
            
            logger.debug(f"Processing batch {start_idx//batch_size + 1}")
            
            batch_results = self._predict_single(batch_data)
            all_results.append(batch_results)
        
        return pd.concat(all_results, ignore_index=True)
    
    def _predict_streaming(self, data: pd.DataFrame) -> pd.DataFrame:
        """Make streaming predictions (generator-based)."""
        # For streaming, we process one sample at a time
        results = []
        
        for idx, row in data.iterrows():
            row_df = pd.DataFrame([row])
            pred = self._predict_single(row_df)
            results.append(pred)
            
            # Yield intermediate results (useful for real streaming)
            if len(results) % 100 == 0:
                logger.debug(f"Processed {len(results)} samples")
        
        return pd.concat(results, ignore_index=True)
    
    def _predict_async(self, data: pd.DataFrame) -> pd.DataFrame:
        """Make predictions asynchronously using parallel processing."""
        processor = ParallelProcessor(n_jobs=self.config.n_jobs)
        
        # Split data into chunks
        chunks = np.array_split(data, self.config.n_jobs)
        
        # Process in parallel
        results = processor.map(self._predict_single, chunks)
        
        return pd.concat(results, ignore_index=True)
    
    def _add_uncertainty(self, data: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
        """Add uncertainty estimates to predictions."""
        if not self.uncertainty_quantifier:
            return results
        
        logger.info("Computing uncertainty estimates")
        
        # Get training data if available (for residual methods)
        train_data = None
        train_target = None
        if hasattr(self.model, 'training_data_'):
            train_data = self.model.training_data_['X']
            train_target = self.model.training_data_['y']
        
        # Compute uncertainty based on method
        if self.config.uncertainty_method == "bootstrap":
            if train_data is not None:
                uncertainty_results = self.uncertainty_quantifier.bootstrap_uncertainty(
                    self.model, train_data, train_target, data,
                    n_bootstrap=50,
                    confidence_level=self.config.confidence_level
                )
            else:
                logger.warning("Bootstrap uncertainty requires training data")
                return results
                
        elif self.config.uncertainty_method == "ensemble" and hasattr(self.model, 'estimators_'):
            # For ensemble models
            uncertainty_results = self.uncertainty_quantifier.ensemble_uncertainty(
                self.model.estimators_, data,
                confidence_level=self.config.confidence_level
            )
        else:
            logger.warning(f"Uncertainty method {self.config.uncertainty_method} not available")
            return results
        
        # Add uncertainty to results
        if 'lower_bound' in uncertainty_results:
            results['prediction_lower'] = uncertainty_results['lower_bound']
            results['prediction_upper'] = uncertainty_results['upper_bound']
        
        if 'std' in uncertainty_results:
            results['prediction_std'] = uncertainty_results['std']
        
        return results
    
    def _add_metadata(self, results: pd.DataFrame, original_data: pd.DataFrame) -> pd.DataFrame:
        """Add metadata to predictions."""
        metadata = {
            'prediction_timestamp': datetime.now().isoformat(),
            'model_version': getattr(self.model, 'version', 'unknown'),
            'pipeline_version': '1.0.0'
        }
        
        for key, value in metadata.items():
            results[key] = value
        
        # Add row identifiers if available
        if 'id' in original_data.columns:
            results['id'] = original_data['id'].values
        
        return results
    
    def predict_proba(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Get probability predictions for classification models.
        
        Args:
            data: Input data
            
        Returns:
            Probability array
        """
        if not hasattr(self.model, 'predict_proba'):
            raise ValueError("Model does not support probability predictions")
        
        # Preprocess if needed
        if self.preprocessor:
            data = self.preprocessor.transform(data)
        
        return self.model.predict_proba(data)
    
    def save_predictions(self, predictions: pd.DataFrame, output_path: str):
        """
        Save predictions to file.
        
        Args:
            predictions: Predictions DataFrame
            output_path: Output file path
        """
        file_handler = FileHandler()
        file_handler.write(predictions, output_path, format=self.config.output_format)
        logger.info(f"Saved predictions to {output_path}")
    
    def explain_predictions(self, data: pd.DataFrame, 
                          method: str = "shap") -> pd.DataFrame:
        """
        Generate explanations for predictions.
        
        Args:
            data: Input data
            method: Explanation method ('shap', 'lime')
            
        Returns:
            DataFrame with explanations
        """
        logger.info(f"Generating {method} explanations")
        
        if method == "shap":
            try:
                import shap
                
                # Create explainer
                explainer = shap.TreeExplainer(self.model)
                shap_values = explainer.shap_values(data)
                
                # Convert to DataFrame
                feature_names = data.columns
                explanations = pd.DataFrame(
                    shap_values,
                    columns=[f"shap_{col}" for col in feature_names]
                )
                
                return explanations
                
            except ImportError:
                logger.error("SHAP not installed. Install with: pip install shap")
                return pd.DataFrame()
                
        elif method == "lime":
            try:
                import lime.lime_tabular
                
                # Create LIME explainer
                explainer = lime.lime_tabular.LimeTabularExplainer(
                    data.values,
                    feature_names=data.columns.tolist(),
                    mode='regression' if hasattr(self.model, 'predict') else 'classification'
                )
                
                explanations = []
                for idx, row in data.iterrows():
                    exp = explainer.explain_instance(
                        row.values,
                        self.model.predict,
                        num_features=len(data.columns)
                    )
                    
                    # Extract feature importance
                    exp_dict = dict(exp.as_list())
                    explanations.append(exp_dict)
                
                return pd.DataFrame(explanations)
                
            except ImportError:
                logger.error("LIME not installed. Install with: pip install lime")
                return pd.DataFrame()
        
        else:
            logger.error(f"Unknown explanation method: {method}")
            return pd.DataFrame()
    
    def benchmark(self, test_data: pd.DataFrame, 
                 n_runs: int = 10) -> Dict[str, float]:
        """
        Benchmark inference performance.
        
        Args:
            test_data: Test data for benchmarking
            n_runs: Number of benchmark runs
            
        Returns:
            Performance metrics
        """
        import time
        
        logger.info(f"Running inference benchmark with {n_runs} runs")
        
        times = []
        
        for i in range(n_runs):
            start_time = time.time()
            _ = self.predict(test_data, return_uncertainty=False, check_drift=False)
            elapsed = time.time() - start_time
            times.append(elapsed)
        
        # Calculate statistics
        times = np.array(times)
        n_samples = len(test_data)
        
        metrics = {
            'mean_time': np.mean(times),
            'std_time': np.std(times),
            'min_time': np.min(times),
            'max_time': np.max(times),
            'samples_per_second': n_samples / np.mean(times),
            'n_samples': n_samples,
            'n_runs': n_runs
        }
        
        logger.info(f"Benchmark results: {metrics['samples_per_second']:.2f} samples/second")
        
        return metrics


class ModelServer:
    """
    Simple model server for serving predictions via API.
    """
    
    def __init__(self, inference_pipeline: InferencePipeline):
        """
        Initialize model server.
        
        Args:
            inference_pipeline: Inference pipeline instance
        """
        self.pipeline = inference_pipeline
        self.request_count = 0
        self.start_time = datetime.now()
    
    def predict_endpoint(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prediction endpoint.
        
        Args:
            data: Request data
            
        Returns:
            Response dictionary
        """
        self.request_count += 1
        
        try:
            # Convert to DataFrame
            if 'features' in data:
                input_data = pd.DataFrame([data['features']])
            elif 'data' in data:
                input_data = pd.DataFrame(data['data'])
            else:
                return {'error': 'No input data provided'}
            
            # Make predictions
            predictions = self.pipeline.predict(input_data)
            
            # Format response
            response = {
                'predictions': predictions.to_dict(orient='records'),
                'model_version': getattr(self.pipeline.model, 'version', 'unknown'),
                'timestamp': datetime.now().isoformat()
            }
            
            return response
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            return {'error': str(e)}
    
    def health_check(self) -> Dict[str, Any]:
        """Health check endpoint."""
        uptime = (datetime.now() - self.start_time).total_seconds()
        
        return {
            'status': 'healthy',
            'uptime_seconds': uptime,
            'request_count': self.request_count,
            'model_loaded': self.pipeline.model is not None
        }


# Utility functions
def create_inference_pipeline(model_path: str, **kwargs) -> InferencePipeline:
    """
    Create an inference pipeline.
    
    Args:
        model_path: Path to saved model
        **kwargs: Additional configuration options
        
    Returns:
        InferencePipeline instance
    """
    config = InferenceConfig(model_path=model_path, **kwargs)
    return InferencePipeline(config)


def batch_predict(model_path: str, data_path: str, output_path: str,
                 batch_size: int = 1000, **kwargs):
    """
    Convenience function for batch predictions.
    
    Args:
        model_path: Path to model
        data_path: Path to input data
        output_path: Path to save predictions
        batch_size: Batch size
        **kwargs: Additional options
    """
    # Create pipeline
    pipeline = create_inference_pipeline(
        model_path,
        prediction_mode=PredictionMode.BATCH,
        batch_size=batch_size,
        **kwargs
    )
    
    # Make predictions
    predictions = pipeline.predict(data_path)
    
    # Save results
    pipeline.save_predictions(predictions, output_path)
    
    logger.info(f"Batch prediction complete. Results saved to {output_path}")


# Example usage
def example_inference():
    """Example of using the inference pipeline."""
    import tempfile
    from sklearn.datasets import make_regression
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    
    # Create sample data and model
    X, y = make_regression(n_samples=1000, n_features=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # Train model
    model = RandomForestRegressor(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Save model
        model_path = Path(tmpdir) / "model.pkl"
        joblib.dump(model, model_path)
        
        # Create test data
        test_df = pd.DataFrame(X_test, columns=[f"feature_{i}" for i in range(10)])
        test_data_path = Path(tmpdir) / "test_data.csv"
        test_df.to_csv(test_data_path, index=False)
        
        # Create inference pipeline
        print("Creating inference pipeline...")
        pipeline = create_inference_pipeline(
            str(model_path),
            enable_uncertainty=True,
            uncertainty_method="ensemble",
            include_metadata=True
        )
        
        # Make predictions
        print("\nMaking predictions...")
        predictions = pipeline.predict(test_df)
        print(f"Predictions shape: {predictions.shape}")
        print(f"Columns: {list(predictions.columns)}")
        
        # Show sample predictions
        print("\nSample predictions:")
        print(predictions.head())
        
        # Benchmark performance
        print("\nBenchmarking...")
        metrics = pipeline.benchmark(test_df, n_runs=5)
        print(f"Performance: {metrics['samples_per_second']:.2f} samples/second")
        
        # Test model server
        print("\nTesting model server...")
        server = ModelServer(pipeline)
        
        # Test prediction endpoint
        request_data = {
            'features': {f"feature_{i}": float(X_test[0, i]) for i in range(10)}
        }
        response = server.predict_endpoint(request_data)
        print(f"Server response: {response}")
        
        # Health check
        health = server.health_check()
        print(f"Health check: {health}")


if __name__ == "__main__":
    example_inference()