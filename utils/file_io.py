"""
utils/file_io.py
File I/O operations for the data science toolkit.
"""

import os
import json
import yaml
import pickle
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
import logging
import warnings
from contextlib import contextmanager
import gzip
import zipfile
import tarfile
import shutil

logger = logging.getLogger(__name__)


class FileHandler:
    """
    Unified file handler for various file formats.
    
    Supports reading and writing multiple file formats commonly used
    in data science projects.
    """
    
    SUPPORTED_FORMATS = {
        'csv': {'read': pd.read_csv, 'write': 'to_csv'},
        'excel': {'read': pd.read_excel, 'write': 'to_excel'},
        'json': {'read': pd.read_json, 'write': 'to_json'},
        'parquet': {'read': pd.read_parquet, 'write': 'to_parquet'},
        'pickle': {'read': pd.read_pickle, 'write': 'to_pickle'},
        'feather': {'read': pd.read_feather, 'write': 'to_feather'},
        'hdf': {'read': pd.read_hdf, 'write': 'to_hdf'},
        'sql': {'read': pd.read_sql, 'write': 'to_sql'}
    }
    
    def __init__(self, compression: Optional[str] = None):
        """
        Initialize the file handler.
        
        Args:
            compression: Compression type ('gzip', 'zip', None)
        """
        self.compression = compression
    
    def read(self, filepath: Union[str, Path], 
             format: Optional[str] = None,
             **kwargs) -> pd.DataFrame:
        """
        Read data from file.
        
        Args:
            filepath: Path to the file
            format: File format (auto-detected if None)
            **kwargs: Additional arguments for the reader
            
        Returns:
            Loaded DataFrame
        """
        filepath = Path(filepath)
        
        # Auto-detect format
        if format is None:
            format = self._detect_format(filepath)
        
        logger.info(f"Reading {format} file: {filepath}")
        
        # Handle compression
        if self.compression or filepath.suffix in ['.gz', '.zip']:
            with self._compression_handler(filepath, 'r') as f:
                return self._read_format(f, format, **kwargs)
        else:
            return self._read_format(filepath, format, **kwargs)
    
    def write(self, data: pd.DataFrame,
              filepath: Union[str, Path],
              format: Optional[str] = None,
              **kwargs) -> None:
        """
        Write data to file.
        
        Args:
            data: DataFrame to save
            filepath: Path to save to
            format: File format (auto-detected if None)
            **kwargs: Additional arguments for the writer
        """
        filepath = Path(filepath)
        
        # Create directory if needed
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Auto-detect format
        if format is None:
            format = self._detect_format(filepath)
        
        logger.info(f"Writing {format} file: {filepath}")
        
        # Get writer method
        if format not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format: {format}")
        
        writer_name = self.SUPPORTED_FORMATS[format]['write']
        writer = getattr(data, writer_name)
        
        # Handle compression
        if self.compression:
            with self._compression_handler(filepath, 'w') as f:
                writer(f, **kwargs)
        else:
            writer(filepath, **kwargs)
    
    def _detect_format(self, filepath: Path) -> str:
        """Detect file format from extension."""
        suffix = filepath.suffix.lower()
        
        # Remove compression extension
        if suffix in ['.gz', '.zip', '.bz2']:
            suffix = Path(filepath.stem).suffix.lower()
        
        format_map = {
            '.csv': 'csv',
            '.xlsx': 'excel',
            '.xls': 'excel',
            '.json': 'json',
            '.parquet': 'parquet',
            '.pkl': 'pickle',
            '.pickle': 'pickle',
            '.feather': 'feather',
            '.h5': 'hdf',
            '.hdf5': 'hdf'
        }
        
        if suffix not in format_map:
            raise ValueError(f"Cannot detect format for extension: {suffix}")
        
        return format_map[suffix]
    
    def _read_format(self, filepath: Any, format: str, **kwargs) -> pd.DataFrame:
        """Read specific format."""
        if format not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format: {format}")
        
        reader = self.SUPPORTED_FORMATS[format]['read']
        return reader(filepath, **kwargs)
    
    @contextmanager
    def _compression_handler(self, filepath: Path, mode: str):
        """Handle compressed files."""
        if self.compression == 'gzip' or filepath.suffix == '.gz':
            with gzip.open(filepath, mode + 't') as f:
                yield f
        elif self.compression == 'zip' or filepath.suffix == '.zip':
            if mode == 'r':
                with zipfile.ZipFile(filepath, 'r') as zf:
                    # Assume single file in zip
                    name = zf.namelist()[0]
                    with zf.open(name) as f:
                        yield f
            else:
                with zipfile.ZipFile(filepath, 'w') as zf:
                    yield zf
        else:
            with open(filepath, mode) as f:
                yield f


def save_object(obj: Any, filepath: Union[str, Path], 
                protocol: str = 'pickle') -> None:
    """
    Save any Python object to file.
    
    Args:
        obj: Object to save
        filepath: Path to save to
        protocol: Serialization protocol ('pickle', 'joblib', 'json')
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving object to {filepath} using {protocol}")
    
    if protocol == 'pickle':
        with open(filepath, 'wb') as f:
            pickle.dump(obj, f)
    elif protocol == 'joblib':
        joblib.dump(obj, filepath)
    elif protocol == 'json':
        with open(filepath, 'w') as f:
            json.dump(obj, f, indent=2)
    else:
        raise ValueError(f"Unknown protocol: {protocol}")


def load_object(filepath: Union[str, Path], 
                protocol: str = 'pickle') -> Any:
    """
    Load Python object from file.
    
    Args:
        filepath: Path to load from
        protocol: Serialization protocol ('pickle', 'joblib', 'json')
        
    Returns:
        Loaded object
    """
    filepath = Path(filepath)
    
    logger.info(f"Loading object from {filepath} using {protocol}")
    
    if protocol == 'pickle':
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    elif protocol == 'joblib':
        return joblib.load(filepath)
    elif protocol == 'json':
        with open(filepath, 'r') as f:
            return json.load(f)
    else:
        raise ValueError(f"Unknown protocol: {protocol}")


def save_config(config: Dict[str, Any], 
                filepath: Union[str, Path],
                format: str = 'yaml') -> None:
    """
    Save configuration dictionary.
    
    Args:
        config: Configuration dictionary
        filepath: Path to save to
        format: Format ('yaml' or 'json')
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving config to {filepath}")
    
    if format == 'yaml':
        with open(filepath, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
    elif format == 'json':
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)
    else:
        raise ValueError(f"Unknown format: {format}")


def load_config(filepath: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from file.
    
    Args:
        filepath: Path to config file
        
    Returns:
        Configuration dictionary
    """
    filepath = Path(filepath)
    
    logger.info(f"Loading config from {filepath}")
    
    if filepath.suffix in ['.yaml', '.yml']:
        with open(filepath, 'r') as f:
            return yaml.safe_load(f)
    elif filepath.suffix == '.json':
        with open(filepath, 'r') as f:
            return json.load(f)
    else:
        raise ValueError(f"Unknown config format: {filepath.suffix}")


class DataCache:
    """
    Simple file-based data caching system.
    
    Useful for caching intermediate results in data pipelines.
    """
    
    def __init__(self, cache_dir: Union[str, Path] = '.cache'):
        """
        Initialize the cache.
        
        Args:
            cache_dir: Directory for cache files
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._index = self._load_index()
    
    def _load_index(self) -> Dict[str, Dict[str, Any]]:
        """Load cache index."""
        index_path = self.cache_dir / 'index.json'
        if index_path.exists():
            with open(index_path, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_index(self) -> None:
        """Save cache index."""
        index_path = self.cache_dir / 'index.json'
        with open(index_path, 'w') as f:
            json.dump(self._index, f, indent=2)
    
    def _get_cache_path(self, key: str) -> Path:
        """Get cache file path for key."""
        # Use hash for filename to handle special characters
        import hashlib
        hash_key = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{hash_key}.pkl"
    
    def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        return key in self._index and self._get_cache_path(key).exists()
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            default: Default value if not found
            
        Returns:
            Cached value or default
        """
        if self.exists(key):
            cache_path = self._get_cache_path(key)
            logger.debug(f"Loading from cache: {key}")
            return load_object(cache_path)
        return default
    
    def set(self, key: str, value: Any, metadata: Optional[Dict] = None) -> None:
        """
        Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            metadata: Optional metadata
        """
        cache_path = self._get_cache_path(key)
        save_object(value, cache_path)
        
        # Update index
        from datetime import datetime
        self._index[key] = {
            'path': str(cache_path),
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        self._save_index()
        
        logger.debug(f"Saved to cache: {key}")
    
    def delete(self, key: str) -> None:
        """Delete key from cache."""
        if self.exists(key):
            cache_path = self._get_cache_path(key)
            cache_path.unlink()
            del self._index[key]
            self._save_index()
            logger.debug(f"Deleted from cache: {key}")
    
    def clear(self) -> None:
        """Clear all cache."""
        for key in list(self._index.keys()):
            self.delete(key)
        logger.info("Cache cleared")
    
    def get_info(self) -> Dict[str, Any]:
        """Get cache information."""
        total_size = sum(
            self._get_cache_path(key).stat().st_size
            for key in self._index
            if self._get_cache_path(key).exists()
        )
        
        return {
            'n_items': len(self._index),
            'total_size_mb': total_size / (1024 * 1024),
            'cache_dir': str(self.cache_dir)
        }
    
    @contextmanager
    def cached_computation(self, key: str):
        """
        Context manager for cached computations.
        
        Usage:
            with cache.cached_computation('my_result') as cached:
                if cached.exists:
                    result = cached.value
                else:
                    result = expensive_computation()
                    cached.value = result
        """
        class CachedValue:
            def __init__(self, cache, key):
                self.cache = cache
                self.key = key
                self.exists = cache.exists(key)
                self._value = cache.get(key) if self.exists else None
            
            @property
            def value(self):
                return self._value
            
            @value.setter
            def value(self, val):
                self._value = val
                self.cache.set(self.key, val)
        
        yield CachedValue(self, key)


def create_archive(source_dir: Union[str, Path],
                  archive_path: Union[str, Path],
                  format: str = 'zip') -> None:
    """
    Create archive from directory.
    
    Args:
        source_dir: Directory to archive
        archive_path: Path for archive file
        format: Archive format ('zip', 'tar', 'gztar')
    """
    source_dir = Path(source_dir)
    archive_path = Path(archive_path)
    
    logger.info(f"Creating {format} archive: {archive_path}")
    
    if format == 'zip':
        with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for file_path in source_dir.rglob('*'):
                if file_path.is_file():
                    arcname = file_path.relative_to(source_dir)
                    zf.write(file_path, arcname)
    elif format in ['tar', 'gztar']:
        mode = 'w:gz' if format == 'gztar' else 'w'
        with tarfile.open(archive_path, mode) as tf:
            tf.add(source_dir, arcname=source_dir.name)
    else:
        raise ValueError(f"Unknown archive format: {format}")


def extract_archive(archive_path: Union[str, Path],
                   extract_dir: Union[str, Path]) -> None:
    """
    Extract archive to directory.
    
    Args:
        archive_path: Path to archive file
        extract_dir: Directory to extract to
    """
    archive_path = Path(archive_path)
    extract_dir = Path(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Extracting archive: {archive_path}")
    
    if archive_path.suffix == '.zip':
        with zipfile.ZipFile(archive_path, 'r') as zf:
            zf.extractall(extract_dir)
    elif archive_path.suffix in ['.tar', '.gz', '.tgz']:
        with tarfile.open(archive_path, 'r:*') as tf:
            tf.extractall(extract_dir)
    else:
        raise ValueError(f"Unknown archive format: {archive_path.suffix}")


class CheckpointManager:
    """
    Manager for saving and loading training checkpoints.
    """
    
    def __init__(self, checkpoint_dir: Union[str, Path]):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory for checkpoints
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def save_checkpoint(self, state: Dict[str, Any], 
                       epoch: int,
                       is_best: bool = False) -> None:
        """
        Save training checkpoint.
        
        Args:
            state: State dictionary to save
            epoch: Epoch number
            is_best: Whether this is the best model so far
        """
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pkl"
        save_object(state, checkpoint_path)
        
        # Save as best if needed
        if is_best:
            best_path = self.checkpoint_dir / "best_checkpoint.pkl"
            shutil.copy2(checkpoint_path, best_path)
            logger.info(f"Saved best checkpoint at epoch {epoch}")
        
        # Save latest
        latest_path = self.checkpoint_dir / "latest_checkpoint.pkl"
        shutil.copy2(checkpoint_path, latest_path)
        
        logger.info(f"Saved checkpoint at epoch {epoch}")
    
    def load_checkpoint(self, checkpoint: Union[str, int] = 'latest') -> Dict[str, Any]:
        """
        Load checkpoint.
        
        Args:
            checkpoint: 'latest', 'best', or epoch number
            
        Returns:
            State dictionary
        """
        if checkpoint == 'latest':
            checkpoint_path = self.checkpoint_dir / "latest_checkpoint.pkl"
        elif checkpoint == 'best':
            checkpoint_path = self.checkpoint_dir / "best_checkpoint.pkl"
        elif isinstance(checkpoint, int):
            checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{checkpoint}.pkl"
        else:
            checkpoint_path = self.checkpoint_dir / checkpoint
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        logger.info(f"Loading checkpoint: {checkpoint_path}")
        return load_object(checkpoint_path)
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """List all available checkpoints."""
        checkpoints = []
        
        for path in self.checkpoint_dir.glob("checkpoint_epoch_*.pkl"):
            epoch = int(path.stem.split('_')[-1])
            checkpoints.append({
                'epoch': epoch,
                'path': path,
                'size_mb': path.stat().st_size / (1024 * 1024),
                'modified': path.stat().st_mtime
            })
        
        return sorted(checkpoints, key=lambda x: x['epoch'])
    
    def cleanup(self, keep_last: int = 5, keep_best: bool = True) -> None:
        """
        Clean up old checkpoints.
        
        Args:
            keep_last: Number of recent checkpoints to keep
            keep_best: Whether to keep the best checkpoint
        """
        checkpoints = self.list_checkpoints()
        
        if len(checkpoints) <= keep_last:
            return
        
        # Sort by epoch (newest first)
        checkpoints.sort(key=lambda x: x['epoch'], reverse=True)
        
        # Remove old checkpoints
        for checkpoint in checkpoints[keep_last:]:
            checkpoint['path'].unlink()
            logger.info(f"Removed old checkpoint: {checkpoint['path']}")


# Example usage
def example_file_operations():
    """Example of using file I/O utilities."""
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Test FileHandler
        print("Testing FileHandler...")
        handler = FileHandler()
        
        # Create sample data
        df = pd.DataFrame({
            'A': np.random.randn(100),
            'B': np.random.randn(100),
            'C': np.random.choice(['X', 'Y', 'Z'], 100)
        })
        
        # Save in different formats
        for format in ['csv', 'json', 'parquet']:
            filepath = tmpdir / f"data.{format}"
            handler.write(df, filepath)
            df_loaded = handler.read(filepath)
            print(f"  {format}: {df_loaded.shape}")
        
        # Test caching
        print("\nTesting DataCache...")
        cache = DataCache(tmpdir / 'cache')
        
        # Cache computation
        with cache.cached_computation('expensive_result') as cached:
            if cached.exists:
                result = cached.value
                print("  Loaded from cache")
            else:
                result = {'data': np.random.randn(1000).tolist()}
                cached.value = result
                print("  Computed and cached")
        
        print(f"  Cache info: {cache.get_info()}")
        
        # Test checkpoints
        print("\nTesting CheckpointManager...")
        ckpt_manager = CheckpointManager(tmpdir / 'checkpoints')
        
        # Save checkpoints
        for epoch in range(5):
            state = {
                'epoch': epoch,
                'model_state': {'weights': np.random.randn(10, 10).tolist()},
                'loss': np.random.random()
            }
            ckpt_manager.save_checkpoint(state, epoch, is_best=(epoch == 3))
        
        # List checkpoints
        checkpoints = ckpt_manager.list_checkpoints()
        print(f"  Found {len(checkpoints)} checkpoints")
        
        # Load best checkpoint
        best_state = ckpt_manager.load_checkpoint('best')
        print(f"  Best checkpoint from epoch: {best_state['epoch']}")


if __name__ == "__main__":
    example_file_operations()