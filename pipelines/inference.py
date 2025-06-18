"""
pipelines/inference.py
Inference pipeline for making predictions with trained models.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Union, Tuple, Callable, Generator, Iterator
from pathlib import Path
import json
import joblib
import pickle
import logging
from datetime import datetime
import warnings
from dataclasses import dataclass
from enum import Enum
import asyncio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import queue
import threading
import time
from tqdm import tqdm

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
    stream_buffer_size: int = 100
    async_max_workers: int = 4


class BatchPredictor:
    """
    Batch prediction handler for processing large datasets efficiently.
    
    This class handles batch predictions with features like:
    - Memory-efficient processing of large files
    - Progress tracking
    - Error handling and recovery
    - Parallel processing support
    """
    
    def __init__(self,
                 model: Any,
                 preprocessor: Optional[Any] = None,
                 batch_size: int = 1000,
                 n_jobs: int = 1,
                 show_progress: bool = True):
        """
        Initialize BatchPredictor.
        
        Args:
            model: Trained model
            preprocessor: Data preprocessor
            batch_size: Size of each batch
            n_jobs: Number of parallel jobs
            show_progress: Whether to show progress bar
        """
        self.model = model
        self.preprocessor = preprocessor
        self.batch_size = batch_size
        self.n_jobs = n_jobs
        self.show_progress = show_progress
        self.stats = {
            'total_samples': 0,
            'processed_samples': 0,
            'failed_samples': 0,
            'processing_time': 0,
            'batches_processed': 0
        }
    
    def predict_file(self,
                    file_path: Union[str, Path],
                    output_path: Optional[Union[str, Path]] = None,
                    file_format: str = 'auto',
                    chunk_size: Optional[int] = None) -> pd.DataFrame:
        """
        Make predictions on a large file.
        
        Args:
            file_path: Path to input file
            output_path: Path to save predictions
            file_format: File format ('csv', 'parquet', 'auto')
            chunk_size: Size of chunks to read (for memory efficiency)
            
        Returns:
            DataFrame with predictions
        """
        file_path = Path(file_path)
        chunk_size = chunk_size or self.batch_size
        
        # Auto-detect file format
        if file_format == 'auto':
            file_format = file_path.suffix.lower().replace('.', '')
        
        logger.info(f"Processing file: {file_path} (format: {file_format})")
        start_time = time.time()
        
        # Get file reader based on format
        reader = self._get_file_reader(file_path, file_format, chunk_size)
        
        # Process chunks
        all_predictions = []
        chunk_iterator = tqdm(reader, desc="Processing chunks") if self.show_progress else reader
        
        for chunk_idx, chunk in enumerate(chunk_iterator):
            try:
                # Preprocess if needed
                if self.preprocessor:
                    chunk = self.preprocessor.transform(chunk)
                
                # Make predictions
                chunk_predictions = self._predict_chunk(chunk)
                
                # Add chunk index for tracking
                chunk_predictions['chunk_idx'] = chunk_idx
                
                all_predictions.append(chunk_predictions)
                
                # Update stats
                self.stats['processed_samples'] += len(chunk)
                self.stats['batches_processed'] += 1
                
            except Exception as e:
                logger.error(f"Error processing chunk {chunk_idx}: {str(e)}")
                self.stats['failed_samples'] += len(chunk)
                continue
        
        # Combine results
        if all_predictions:
            results = pd.concat(all_predictions, ignore_index=True)
        else:
            results = pd.DataFrame()
        
        # Update final stats
        self.stats['processing_time'] = time.time() - start_time
        self.stats['total_samples'] = self.stats['processed_samples'] + self.stats['failed_samples']
        
        # Save if output path provided
        if output_path:
            self._save_predictions(results, output_path, file_format)
        
        logger.info(f"Batch prediction complete. Processed {self.stats['processed_samples']} samples "
                   f"in {self.stats['processing_time']:.2f} seconds")
        
        return results
    
    def predict_dataframe(self,
                         data: pd.DataFrame,
                         return_generator: bool = False) -> Union[pd.DataFrame, Generator]:
        """
        Make batch predictions on a DataFrame.
        
        Args:
            data: Input DataFrame
            return_generator: Whether to return a generator
            
        Returns:
            Predictions as DataFrame or generator
        """
        n_samples = len(data)
        
        if return_generator:
            return self._predict_generator(data)
        
        # Process in batches
        all_predictions = []
        
        for start_idx in range(0, n_samples, self.batch_size):
            end_idx = min(start_idx + self.batch_size, n_samples)
            batch = data.iloc[start_idx:end_idx]
            
            # Preprocess if needed
            if self.preprocessor:
                batch = self.preprocessor.transform(batch)
            
            # Make predictions
            predictions = self._predict_chunk(batch)
            all_predictions.append(predictions)
        
        return pd.concat(all_predictions, ignore_index=True)
    
    def predict_parallel(self,
                        data: pd.DataFrame,
                        chunk_processor: Optional[Callable] = None) -> pd.DataFrame:
        """
        Make predictions using parallel processing.
        
        Args:
            data: Input data
            chunk_processor: Optional custom chunk processor
            
        Returns:
            Predictions DataFrame
        """
        n_samples = len(data)
        n_chunks = (n_samples + self.batch_size - 1) // self.batch_size
        
        # Split data into chunks
        chunks = []
        for i in range(n_chunks):
            start_idx = i * self.batch_size
            end_idx = min(start_idx + self.batch_size, n_samples)
            chunks.append(data.iloc[start_idx:end_idx])
        
        # Process chunks in parallel
        processor = chunk_processor or self._predict_chunk
        
        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            futures = [executor.submit(processor, chunk) for chunk in chunks]
            
            results = []
            for future in tqdm(futures, desc="Processing chunks") if self.show_progress else futures:
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Chunk processing failed: {str(e)}")
        
        return pd.concat(results, ignore_index=True) if results else pd.DataFrame()
    
    def _predict_chunk(self, chunk: pd.DataFrame) -> pd.DataFrame:
        """Predict on a single chunk."""
        predictions = self.model.predict(chunk)
        
        results = pd.DataFrame({
            'prediction': predictions,
            'timestamp': datetime.now()
        })
        
        # Add probabilities for classification
        if hasattr(self.model, 'predict_proba'):
            proba = self.model.predict_proba(chunk)
            if hasattr(self.model, 'classes_'):
                for i, class_label in enumerate(self.model.classes_):
                    results[f'probability_{class_label}'] = proba[:, i]
        
        # Add input indices
        results.index = chunk.index
        
        return results
    
    def _predict_generator(self, data: pd.DataFrame) -> Generator:
        """Generator for memory-efficient predictions."""
        n_samples = len(data)
        
        for start_idx in range(0, n_samples, self.batch_size):
            end_idx = min(start_idx + self.batch_size, n_samples)
            batch = data.iloc[start_idx:end_idx]
            
            # Preprocess if needed
            if self.preprocessor:
                batch = self.preprocessor.transform(batch)
            
            # Make predictions
            yield self._predict_chunk(batch)
    
    def _get_file_reader(self, file_path: Path, file_format: str, 
                        chunk_size: int) -> Iterator[pd.DataFrame]:
        """Get appropriate file reader based on format."""
        if file_format == 'csv':
            return pd.read_csv(file_path, chunksize=chunk_size)
        elif file_format == 'parquet':
            # For parquet, we'll read in chunks manually
            df = pd.read_parquet(file_path)
            n_chunks = (len(df) + chunk_size - 1) // chunk_size
            for i in range(n_chunks):
                start = i * chunk_size
                end = min(start + chunk_size, len(df))
                yield df.iloc[start:end]
        elif file_format == 'json':
            return pd.read_json(file_path, lines=True, chunksize=chunk_size)
        else:
            raise ValueError(f"Unsupported file format: {file_format}")
    
    def _save_predictions(self, predictions: pd.DataFrame, 
                         output_path: Path, file_format: str):
        """Save predictions to file."""
        output_path = Path(output_path)
        
        if file_format == 'csv':
            predictions.to_csv(output_path, index=False)
        elif file_format == 'parquet':
            predictions.to_parquet(output_path, index=False)
        elif file_format == 'json':
            predictions.to_json(output_path, orient='records', lines=True)
        
        logger.info(f"Predictions saved to {output_path}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        stats = self.stats.copy()
        
        if stats['processing_time'] > 0:
            stats['samples_per_second'] = stats['processed_samples'] / stats['processing_time']
        
        return stats


class StreamingPredictor:
    """
    Streaming prediction handler for real-time or continuous data streams.
    
    This class handles streaming predictions with features like:
    - Real-time processing
    - Buffering and batching
    - Asynchronous processing
    - Stream monitoring and metrics
    """
    
    def __init__(self,
                 model: Any,
                 preprocessor: Optional[Any] = None,
                 buffer_size: int = 100,
                 batch_timeout: float = 1.0,
                 max_workers: int = 4):
        """
        Initialize StreamingPredictor.
        
        Args:
            model: Trained model
            preprocessor: Data preprocessor
            buffer_size: Size of the buffer for batching
            batch_timeout: Maximum time to wait before processing a batch
            max_workers: Maximum number of worker threads
        """
        self.model = model
        self.preprocessor = preprocessor
        self.buffer_size = buffer_size
        self.batch_timeout = batch_timeout
        self.max_workers = max_workers
        
        # Streaming components
        self.buffer = queue.Queue(maxsize=buffer_size * 2)
        self.results_queue = queue.Queue()
        self.is_running = False
        self.workers = []
        
        # Metrics
        self.metrics = {
            'total_processed': 0,
            'total_failed': 0,
            'average_latency': 0,
            'current_buffer_size': 0,
            'start_time': None,
            'last_prediction_time': None
        }
    
    def start(self):
        """Start the streaming predictor."""
        if self.is_running:
            logger.warning("Streaming predictor is already running")
            return
        
        self.is_running = True
        self.metrics['start_time'] = datetime.now()
        
        # Start worker threads
        for i in range(self.max_workers):
            worker = threading.Thread(
                target=self._worker_loop,
                name=f"StreamWorker-{i}"
            )
            worker.daemon = True
            worker.start()
            self.workers.append(worker)
        
        # Start batch processor
        self.batch_processor = threading.Thread(
            target=self._batch_processor_loop,
            name="BatchProcessor"
        )
        self.batch_processor.daemon = True
        self.batch_processor.start()
        
        logger.info(f"Streaming predictor started with {self.max_workers} workers")
    
    def stop(self):
        """Stop the streaming predictor."""
        if not self.is_running:
            return
        
        logger.info("Stopping streaming predictor...")
        self.is_running = False
        
        # Process remaining items
        self._process_remaining()
        
        # Wait for threads to finish
        for worker in self.workers:
            worker.join(timeout=5)
        
        self.batch_processor.join(timeout=5)
        
        logger.info("Streaming predictor stopped")
    
    def predict(self, data: Union[Dict, pd.Series, pd.DataFrame]) -> Dict[str, Any]:
        """
        Make a streaming prediction.
        
        Args:
            data: Input data (single sample or batch)
            
        Returns:
            Prediction result
        """
        if not self.is_running:
            raise RuntimeError("Streaming predictor is not running. Call start() first.")
        
        # Create request
        request_id = self._generate_request_id()
        request = {
            'id': request_id,
            'data': data,
            'timestamp': datetime.now(),
            'future': asyncio.Future() if asyncio.get_event_loop().is_running() else None
        }
        
        # Add to buffer
        self.buffer.put(request)
        self.metrics['current_buffer_size'] = self.buffer.qsize()
        
        # Wait for result (blocking)
        result = self._wait_for_result(request_id)
        
        return result
    
    async def predict_async(self, data: Union[Dict, pd.Series, pd.DataFrame]) -> Dict[str, Any]:
        """
        Make an asynchronous streaming prediction.
        
        Args:
            data: Input data
            
        Returns:
            Prediction result
        """
        if not self.is_running:
            raise RuntimeError("Streaming predictor is not running. Call start() first.")
        
        # Create request
        request_id = self._generate_request_id()
        future = asyncio.Future()
        
        request = {
            'id': request_id,
            'data': data,
            'timestamp': datetime.now(),
            'future': future
        }
        
        # Add to buffer
        self.buffer.put(request)
        
        # Wait for result
        result = await future
        return result
    
    def predict_stream(self, data_stream: Iterator[Any]) -> Generator[Dict[str, Any], None, None]:
        """
        Process a stream of data.
        
        Args:
            data_stream: Iterator of input data
            
        Yields:
            Prediction results
        """
        if not self.is_running:
            self.start()
        
        for data in data_stream:
            result = self.predict(data)
            yield result
    
    def _worker_loop(self):
        """Worker thread loop for processing batches."""
        while self.is_running:
            try:
                # Get batch from queue
                batch = self.results_queue.get(timeout=1)
                
                if batch is None:  # Shutdown signal
                    break
                
                # Process batch
                self._process_batch(batch)
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Worker error: {str(e)}")
    
    def _batch_processor_loop(self):
        """Batch processor loop for collecting and batching requests."""
        batch = []
        last_batch_time = time.time()
        
        while self.is_running:
            try:
                # Try to get item from buffer
                timeout = max(0.1, self.batch_timeout - (time.time() - last_batch_time))
                request = self.buffer.get(timeout=timeout)
                batch.append(request)
                
                # Check if batch is ready
                if len(batch) >= self.buffer_size or \
                   (time.time() - last_batch_time) >= self.batch_timeout:
                    
                    if batch:
                        self.results_queue.put(batch)
                        batch = []
                        last_batch_time = time.time()
                
            except queue.Empty:
                # Timeout - process current batch if any
                if batch and (time.time() - last_batch_time) >= self.batch_timeout:
                    self.results_queue.put(batch)
                    batch = []
                    last_batch_time = time.time()
            except Exception as e:
                logger.error(f"Batch processor error: {str(e)}")
    
    def _process_batch(self, batch: List[Dict[str, Any]]):
        """Process a batch of requests."""
        try:
            # Extract data
            batch_data = []
            for request in batch:
                if isinstance(request['data'], dict):
                    batch_data.append(request['data'])
                elif isinstance(request['data'], pd.Series):
                    batch_data.append(request['data'].to_dict())
                else:
                    batch_data.append(request['data'])
            
            # Convert to DataFrame
            df = pd.DataFrame(batch_data)
            
            # Preprocess if needed
            if self.preprocessor:
                df = self.preprocessor.transform(df)
            
            # Make predictions
            predictions = self.model.predict(df)
            
            # Process results
            for i, request in enumerate(batch):
                result = {
                    'id': request['id'],
                    'prediction': predictions[i],
                    'timestamp': datetime.now(),
                    'latency': (datetime.now() - request['timestamp']).total_seconds()
                }
                
                # Add probabilities for classification
                if hasattr(self.model, 'predict_proba'):
                    proba = self.model.predict_proba(df.iloc[[i]])
                    result['probabilities'] = proba[0].tolist()
                
                # Complete future if async
                if request.get('future'):
                    request['future'].set_result(result)
                else:
                    # Store result for synchronous retrieval
                    self._store_result(request['id'], result)
                
                # Update metrics
                self.metrics['total_processed'] += 1
                self.metrics['last_prediction_time'] = datetime.now()
                self._update_latency(result['latency'])
        
        except Exception as e:
            logger.error(f"Batch processing error: {str(e)}")
            
            # Mark all requests as failed
            for request in batch:
                error_result = {
                    'id': request['id'],
                    'error': str(e),
                    'timestamp': datetime.now()
                }
                
                if request.get('future'):
                    request['future'].set_exception(e)
                else:
                    self._store_result(request['id'], error_result)
                
                self.metrics['total_failed'] += 1
    
    def _process_remaining(self):
        """Process any remaining items in the buffer."""
        remaining = []
        
        while not self.buffer.empty():
            try:
                remaining.append(self.buffer.get_nowait())
            except queue.Empty:
                break
        
        if remaining:
            logger.info(f"Processing {len(remaining)} remaining items")
            self._process_batch(remaining)
    
    def _generate_request_id(self) -> str:
        """Generate unique request ID."""
        import uuid
        return str(uuid.uuid4())
    
    def _store_result(self, request_id: str, result: Dict[str, Any]):
        """Store result for synchronous retrieval."""
        # In a real implementation, you might use a cache or database
        # For now, we'll use a simple in-memory dict with TTL
        if not hasattr(self, '_results_cache'):
            self._results_cache = {}
        
        self._results_cache[request_id] = result
        
        # Clean old results (simple TTL)
        self._clean_results_cache()
    
    def _wait_for_result(self, request_id: str, timeout: float = 30) -> Dict[str, Any]:
        """Wait for a result (blocking)."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if hasattr(self, '_results_cache') and request_id in self._results_cache:
                result = self._results_cache.pop(request_id)
                return result
            
            time.sleep(0.01)  # Small sleep to avoid busy waiting
        
        raise TimeoutError(f"Timeout waiting for result {request_id}")
    
    def _clean_results_cache(self):
        """Clean old results from cache."""
        if not hasattr(self, '_results_cache'):
            return
        
        # Remove results older than 60 seconds
        cutoff_time = datetime.now().timestamp() - 60
        
        to_remove = []
        for request_id, result in self._results_cache.items():
            if result['timestamp'].timestamp() < cutoff_time:
                to_remove.append(request_id)
        
        for request_id in to_remove:
            del self._results_cache[request_id]
    
    def _update_latency(self, latency: float):
        """Update average latency metric."""
        if self.metrics['average_latency'] == 0:
            self.metrics['average_latency'] = latency
        else:
            # Exponential moving average
            alpha = 0.1
            self.metrics['average_latency'] = (
                alpha * latency + (1 - alpha) * self.metrics['average_latency']
            )
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get streaming metrics."""
        metrics = self.metrics.copy()
        
        if metrics['start_time']:
            uptime = (datetime.now() - metrics['start_time']).total_seconds()
            metrics['uptime_seconds'] = uptime
            
            if uptime > 0:
                metrics['throughput'] = metrics['total_processed'] / uptime
        
        return metrics


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
        self.batch_predictor = None
        self.streaming_predictor = None
        
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
        
        # Setup predictors based on mode
        if self.config.prediction_mode == PredictionMode.BATCH:
            self.batch_predictor = BatchPredictor(
                model=self.model,
                preprocessor=self.preprocessor,
                batch_size=self.config.batch_size,
                n_jobs=self.config.n_jobs
            )
        elif self.config.prediction_mode == PredictionMode.STREAMING:
            self.streaming_predictor = StreamingPredictor(
                model=self.model,
                preprocessor=self.preprocessor,
                buffer_size=self.config.stream_buffer_size,
                max_workers=self.config.async_max_workers
            )
            self.streaming_predictor.start()
    
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
        # Set defaults from config
        return_uncertainty = return_uncertainty if return_uncertainty is not None else self.config.enable_uncertainty
        check_drift = check_drift if check_drift is not None else self.config.enable_drift_detection
        
        # Load data if path provided
        if isinstance(data, (str, Path)):
            data = self._load_data(data)
        elif isinstance(data, list):
            data = pd.DataFrame(data)
        elif isinstance(data, np.ndarray):
            data = pd.DataFrame(data)
        
        # Validate input
        self._validate_input(data)
        
        # Check for drift
        drift_results = None
        if check_drift:
            drift_results = self._check_drift(data)
            if drift_results and drift_results.get('drift_detected'):
                logger.warning("Data drift detected!")
        
        # Make predictions based on mode
        if self.config.prediction_mode == PredictionMode.SINGLE:
            predictions = self._predict_single(data)
        elif self.config.prediction_mode == PredictionMode.BATCH:
            predictions = self.batch_predictor.predict_dataframe(data)
        elif self.config.prediction_mode == PredictionMode.STREAMING:
            predictions = self._predict_streaming(data)
        else:
            predictions = self._predict_batch(data)
        
        # Add uncertainty estimates
        if return_uncertainty and self.uncertainty_quantifier:
            uncertainty = self._estimate_uncertainty(data)
            predictions = pd.concat([predictions, uncertainty], axis=1)
        
        # Add metadata
        if self.config.include_metadata:
            predictions['model_version'] = getattr(self.model, 'version', 'unknown')
            predictions['prediction_timestamp'] = datetime.now()
            
            if drift_results:
                predictions['drift_detected'] = drift_results.get('drift_detected', False)
        
        return predictions
    
    def save_predictions(self, predictions: pd.DataFrame, output_path: Union[str, Path]):
        """
        Save predictions to file.
        
        Args:
            predictions: Predictions DataFrame
            output_path: Path to save predictions
        """
        output_path = Path(output_path)
        output_format = self.config.output_format
        
        if output_format == 'csv':
            predictions.to_csv(output_path, index=False)
        elif output_format == 'json':
            predictions.to_json(output_path, orient='records', indent=2)
        elif output_format == 'parquet':
            predictions.to_parquet(output_path, index=False)
        else:
            raise ValueError(f"Unknown output format: {output_format}")
        
        logger.info(f"Predictions saved to {output_path}")
    
    def _load_data(self, path: Union[str, Path]) -> pd.DataFrame:
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
        """Make streaming predictions."""
        # For DataFrame input, process row by row through streaming predictor
        results = []
        
        for idx, row in data.iterrows():
            result = self.streaming_predictor.predict(row.to_dict())
            results.append(result)
        
        # Convert results to DataFrame
        predictions_data = []
        for result in results:
            pred_dict = {
                'prediction': result['prediction']
            }
            if 'probabilities' in result:
                for i, prob in enumerate(result['probabilities']):
                    pred_dict[f'probability_class_{i}'] = prob
            predictions_data.append(pred_dict)
        
        return pd.DataFrame(predictions_data)
    
    def _estimate_uncertainty(self, data: pd.DataFrame) -> pd.DataFrame:
        """Estimate prediction uncertainty."""
        # This is a simplified version - you'd implement the actual uncertainty methods
        logger.info(f"Estimating uncertainty using {self.config.uncertainty_method}")
        
        if self.config.uncertainty_method == "bootstrap":
            # Simplified bootstrap uncertainty
            n_iterations = 100
            predictions = []
            
            for _ in range(n_iterations):
                # In practice, you'd retrain on bootstrap samples
                pred = self.model.predict(data)
                predictions.append(pred)
            
            predictions = np.array(predictions)
            
            uncertainty_df = pd.DataFrame({
                'prediction_std': np.std(predictions, axis=0),
                'prediction_lower': np.percentile(predictions, 2.5, axis=0),
                'prediction_upper': np.percentile(predictions, 97.5, axis=0)
            })
            
            return uncertainty_df
        
        else:
            # Placeholder for other methods
            return pd.DataFrame()
    
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
        logger.info(f"Generating explanations using {method}")
        
        if method == "shap":
            try:
                import shap
                
                # Create explainer
                explainer = shap.Explainer(self.model, data)
                shap_values = explainer(data)
                
                # Convert to DataFrame
                explanations = pd.DataFrame(
                    shap_values.values,
                    columns=[f"shap_{col}" for col in data.columns]
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
    
    def __del__(self):
        """Cleanup when pipeline is destroyed."""
        if hasattr(self, 'streaming_predictor') and self.streaming_predictor:
            self.streaming_predictor.stop()


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
            uncertainty_method="bootstrap",
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
        
        # Test batch predictor
        print("\nTesting batch predictor...")
        batch_predictor = BatchPredictor(model, batch_size=50)
        batch_results = batch_predictor.predict_file(
            test_data_path,
            output_path=Path(tmpdir) / "batch_predictions.csv"
        )
        print(f"Batch predictor stats: {batch_predictor.get_stats()}")
        
        # Test streaming predictor
        print("\nTesting streaming predictor...")
        streaming_predictor = StreamingPredictor(model)
        streaming_predictor.start()
        
        # Simulate streaming predictions
        for i in range(5):
            sample = test_df.iloc[i].to_dict()
            result = streaming_predictor.predict(sample)
            print(f"Stream prediction {i}: {result['prediction']}")
        
        print(f"Streaming metrics: {streaming_predictor.get_metrics()}")
        streaming_predictor.stop()
        
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