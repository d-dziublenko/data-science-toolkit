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
from typing import Any, Dict, List, Optional, Union, Tuple, Callable, Iterator
import logging
import warnings
from contextlib import contextmanager
import gzip
import zipfile
import tarfile
import shutil
import h5py
import csv
import pyarrow.parquet as pq
import sqlite3
from abc import ABC, abstractmethod
import hashlib
import tempfile
from datetime import datetime
import threading
from collections import defaultdict

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


class DataReader:
    """
    Advanced data reader with support for multiple formats and streaming.
    
    Provides unified interface for reading various data formats with
    features like chunking, filtering, and schema validation.
    """
    
    def __init__(self, 
                 cache_enabled: bool = False,
                 cache_dir: Optional[str] = None,
                 validate_schema: bool = True):
        """
        Initialize DataReader.
        
        Args:
            cache_enabled: Whether to enable caching
            cache_dir: Directory for cache files
            validate_schema: Whether to validate data schema
        """
        self.cache_enabled = cache_enabled
        self.cache_dir = Path(cache_dir) if cache_dir else Path.home() / '.data_cache'
        self.validate_schema = validate_schema
        self._readers = self._init_readers()
        
        if self.cache_enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _init_readers(self) -> Dict[str, Callable]:
        """Initialize format-specific readers."""
        return {
            'csv': self._read_csv,
            'excel': self._read_excel,
            'json': self._read_json,
            'parquet': self._read_parquet,
            'hdf5': self._read_hdf5,
            'feather': self._read_feather,
            'sql': self._read_sql,
            'pickle': self._read_pickle,
            'text': self._read_text,
            'numpy': self._read_numpy
        }
    
    def read(self, 
             filepath: Union[str, Path],
             format: Optional[str] = None,
             columns: Optional[List[str]] = None,
             filters: Optional[Dict[str, Any]] = None,
             chunk_size: Optional[int] = None,
             **kwargs) -> Union[pd.DataFrame, Iterator[pd.DataFrame]]:
        """
        Read data from file with advanced options.
        
        Args:
            filepath: Path to file
            format: File format (auto-detected if None)
            columns: Columns to read
            filters: Filters to apply while reading
            chunk_size: If specified, return iterator
            **kwargs: Format-specific options
            
        Returns:
            DataFrame or iterator of DataFrames
        """
        filepath = Path(filepath)
        
        # Check cache
        if self.cache_enabled:
            cache_key = self._get_cache_key(filepath, columns, filters)
            cached_path = self.cache_dir / f"{cache_key}.parquet"
            
            if cached_path.exists():
                logger.info(f"Reading from cache: {cache_key}")
                return pd.read_parquet(cached_path)
        
        # Auto-detect format
        if format is None:
            format = self._detect_format(filepath)
        
        if format not in self._readers:
            raise ValueError(f"Unsupported format: {format}")
        
        # Read data
        reader = self._readers[format]
        data = reader(filepath, columns, filters, chunk_size, **kwargs)
        
        # Cache if enabled and not chunked
        if self.cache_enabled and chunk_size is None and isinstance(data, pd.DataFrame):
            data.to_parquet(cached_path)
            logger.info(f"Cached data: {cache_key}")
        
        return data
    
    def read_multiple(self,
                     filepaths: List[Union[str, Path]],
                     concat: bool = True,
                     **kwargs) -> Union[pd.DataFrame, List[pd.DataFrame]]:
        """
        Read multiple files.
        
        Args:
            filepaths: List of file paths
            concat: Whether to concatenate results
            **kwargs: Arguments passed to read()
            
        Returns:
            Combined DataFrame or list of DataFrames
        """
        results = []
        
        for filepath in filepaths:
            try:
                data = self.read(filepath, **kwargs)
                results.append(data)
            except Exception as e:
                logger.error(f"Error reading {filepath}: {e}")
                continue
        
        if concat and results:
            return pd.concat(results, ignore_index=True)
        
        return results
    
    def _read_csv(self, filepath: Path, columns: Optional[List[str]],
                  filters: Optional[Dict[str, Any]], chunk_size: Optional[int],
                  **kwargs) -> Union[pd.DataFrame, Iterator[pd.DataFrame]]:
        """Read CSV file."""
        read_kwargs = {
            'usecols': columns,
            'chunksize': chunk_size
        }
        read_kwargs.update(kwargs)
        
        if chunk_size:
            # Return iterator with filters applied
            reader = pd.read_csv(filepath, **read_kwargs)
            return self._apply_filters_to_chunks(reader, filters)
        else:
            df = pd.read_csv(filepath, **read_kwargs)
            return self._apply_filters(df, filters)
    
    def _read_parquet(self, filepath: Path, columns: Optional[List[str]],
                      filters: Optional[Dict[str, Any]], chunk_size: Optional[int],
                      **kwargs) -> pd.DataFrame:
        """Read Parquet file."""
        # Parquet supports column selection and filters natively
        parquet_filters = None
        if filters:
            parquet_filters = [(k, '==', v) for k, v in filters.items()]
        
        return pd.read_parquet(
            filepath,
            columns=columns,
            filters=parquet_filters,
            **kwargs
        )
    
    def _read_excel(self, filepath: Path, columns: Optional[List[str]],
                    filters: Optional[Dict[str, Any]], chunk_size: Optional[int],
                    **kwargs) -> pd.DataFrame:
        """Read Excel file."""
        df = pd.read_excel(filepath, usecols=columns, **kwargs)
        return self._apply_filters(df, filters)
    
    def _read_json(self, filepath: Path, columns: Optional[List[str]],
                   filters: Optional[Dict[str, Any]], chunk_size: Optional[int],
                   **kwargs) -> Union[pd.DataFrame, Iterator[pd.DataFrame]]:
        """Read JSON file."""
        if chunk_size and kwargs.get('lines', False):
            # Read JSON lines with chunking
            reader = pd.read_json(filepath, lines=True, chunksize=chunk_size, **kwargs)
            return self._apply_filters_to_chunks(reader, filters)
        else:
            df = pd.read_json(filepath, **kwargs)
            if columns:
                df = df[columns]
            return self._apply_filters(df, filters)
    
    def _read_hdf5(self, filepath: Path, columns: Optional[List[str]],
                   filters: Optional[Dict[str, Any]], chunk_size: Optional[int],
                   **kwargs) -> pd.DataFrame:
        """Read HDF5 file."""
        key = kwargs.pop('key', 'data')
        where = kwargs.pop('where', None)
        
        if filters and not where:
            where = ' & '.join([f"{k}=={v}" for k, v in filters.items()])
        
        return pd.read_hdf(filepath, key=key, columns=columns, where=where, **kwargs)
    
    def _read_feather(self, filepath: Path, columns: Optional[List[str]],
                      filters: Optional[Dict[str, Any]], chunk_size: Optional[int],
                      **kwargs) -> pd.DataFrame:
        """Read Feather file."""
        df = pd.read_feather(filepath, columns=columns, **kwargs)
        return self._apply_filters(df, filters)
    
    def _read_sql(self, filepath: Path, columns: Optional[List[str]],
                  filters: Optional[Dict[str, Any]], chunk_size: Optional[int],
                  **kwargs) -> Union[pd.DataFrame, Iterator[pd.DataFrame]]:
        """Read from SQL database."""
        # filepath should be connection string or path to SQLite
        con = kwargs.pop('con', f"sqlite:///{filepath}")
        query = kwargs.pop('query', kwargs.pop('sql', f"SELECT * FROM {kwargs.pop('table_name', 'data')}"))
        
        # Add column selection to query
        if columns and "SELECT *" in query:
            query = query.replace("SELECT *", f"SELECT {', '.join(columns)}")
        
        # Add filters to query
        if filters:
            where_clause = " AND ".join([f"{k}='{v}'" for k, v in filters.items()])
            if "WHERE" in query:
                query += f" AND {where_clause}"
            else:
                query += f" WHERE {where_clause}"
        
        return pd.read_sql(query, con, chunksize=chunk_size, **kwargs)
    
    def _read_pickle(self, filepath: Path, columns: Optional[List[str]],
                     filters: Optional[Dict[str, Any]], chunk_size: Optional[int],
                     **kwargs) -> Any:
        """Read pickle file."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        # If it's a DataFrame, apply column selection and filters
        if isinstance(data, pd.DataFrame):
            if columns:
                data = data[columns]
            return self._apply_filters(data, filters)
        
        return data
    
    def _read_text(self, filepath: Path, columns: Optional[List[str]],
                   filters: Optional[Dict[str, Any]], chunk_size: Optional[int],
                   **kwargs) -> Union[List[str], Iterator[List[str]]]:
        """Read text file."""
        encoding = kwargs.get('encoding', 'utf-8')
        
        if chunk_size:
            def text_chunks():
                with open(filepath, 'r', encoding=encoding) as f:
                    while True:
                        lines = []
                        for _ in range(chunk_size):
                            line = f.readline()
                            if not line:
                                if lines:
                                    yield lines
                                return
                            lines.append(line.strip())
                        yield lines
            
            return text_chunks()
        else:
            with open(filepath, 'r', encoding=encoding) as f:
                return f.read().splitlines()
    
    def _read_numpy(self, filepath: Path, columns: Optional[List[str]],
                    filters: Optional[Dict[str, Any]], chunk_size: Optional[int],
                    **kwargs) -> np.ndarray:
        """Read NumPy array file."""
        allow_pickle = kwargs.get('allow_pickle', False)
        
        if filepath.suffix == '.npy':
            return np.load(filepath, allow_pickle=allow_pickle)
        elif filepath.suffix == '.npz':
            data = np.load(filepath, allow_pickle=allow_pickle)
            key = kwargs.get('key', 'arr_0')
            return data[key]
        else:
            return np.loadtxt(filepath, **kwargs)
    
    def _detect_format(self, filepath: Path) -> str:
        """Detect file format from extension."""
        suffix = filepath.suffix.lower()
        
        format_map = {
            '.csv': 'csv',
            '.xlsx': 'excel',
            '.xls': 'excel',
            '.json': 'json',
            '.parquet': 'parquet',
            '.pkl': 'pickle',
            '.pickle': 'pickle',
            '.feather': 'feather',
            '.h5': 'hdf5',
            '.hdf5': 'hdf5',
            '.hdf': 'hdf5',
            '.db': 'sql',
            '.sqlite': 'sql',
            '.txt': 'text',
            '.npy': 'numpy',
            '.npz': 'numpy'
        }
        
        if suffix not in format_map:
            raise ValueError(f"Cannot detect format for extension: {suffix}")
        
        return format_map[suffix]
    
    def _apply_filters(self, df: pd.DataFrame, 
                      filters: Optional[Dict[str, Any]]) -> pd.DataFrame:
        """Apply filters to DataFrame."""
        if not filters or df.empty:
            return df
        
        mask = pd.Series(True, index=df.index)
        
        for column, value in filters.items():
            if column in df.columns:
                if isinstance(value, list):
                    mask &= df[column].isin(value)
                elif isinstance(value, dict):
                    # Support range filters
                    if 'min' in value:
                        mask &= df[column] >= value['min']
                    if 'max' in value:
                        mask &= df[column] <= value['max']
                else:
                    mask &= df[column] == value
        
        return df[mask]
    
    def _apply_filters_to_chunks(self, chunks: Iterator[pd.DataFrame],
                                filters: Optional[Dict[str, Any]]) -> Iterator[pd.DataFrame]:
        """Apply filters to chunk iterator."""
        for chunk in chunks:
            filtered = self._apply_filters(chunk, filters)
            if not filtered.empty:
                yield filtered
    
    def _get_cache_key(self, filepath: Path, columns: Optional[List[str]],
                      filters: Optional[Dict[str, Any]]) -> str:
        """Generate cache key."""
        key_parts = [
            str(filepath),
            str(sorted(columns) if columns else []),
            str(sorted(filters.items()) if filters else [])
        ]
        
        key_str = '|'.join(key_parts)
        return hashlib.md5(key_str.encode()).hexdigest()


class DataWriter:
    """
    Advanced data writer with support for multiple formats and optimizations.
    
    Provides unified interface for writing data with features like
    partitioning, compression, and atomic writes.
    """
    
    def __init__(self,
                 compression: Optional[str] = None,
                 atomic_writes: bool = True,
                 create_dirs: bool = True):
        """
        Initialize DataWriter.
        
        Args:
            compression: Compression type ('gzip', 'snappy', 'lz4')
            atomic_writes: Whether to use atomic writes
            create_dirs: Whether to create parent directories
        """
        self.compression = compression
        self.atomic_writes = atomic_writes
        self.create_dirs = create_dirs
        self._writers = self._init_writers()
    
    def _init_writers(self) -> Dict[str, Callable]:
        """Initialize format-specific writers."""
        return {
            'csv': self._write_csv,
            'excel': self._write_excel,
            'json': self._write_json,
            'parquet': self._write_parquet,
            'hdf5': self._write_hdf5,
            'feather': self._write_feather,
            'sql': self._write_sql,
            'pickle': self._write_pickle,
            'text': self._write_text,
            'numpy': self._write_numpy
        }
    
    def write(self,
              data: Any,
              filepath: Union[str, Path],
              format: Optional[str] = None,
              partition_by: Optional[List[str]] = None,
              append: bool = False,
              **kwargs) -> None:
        """
        Write data to file with advanced options.
        
        Args:
            data: Data to write
            filepath: Output file path
            format: File format (auto-detected if None)
            partition_by: Columns to partition by
            append: Whether to append to existing file
            **kwargs: Format-specific options
        """
        filepath = Path(filepath)
        
        # Create parent directories if needed
        if self.create_dirs:
            filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Auto-detect format
        if format is None:
            format = self._detect_format(filepath)
        
        if format not in self._writers:
            raise ValueError(f"Unsupported format: {format}")
        
        # Handle partitioning
        if partition_by and isinstance(data, pd.DataFrame):
            self._write_partitioned(data, filepath, format, partition_by, **kwargs)
            return
        
        # Write data
        if self.atomic_writes:
            self._atomic_write(data, filepath, format, append, **kwargs)
        else:
            writer = self._writers[format]
            writer(data, filepath, append, **kwargs)
    
    def write_multiple(self,
                      data_dict: Dict[str, Any],
                      base_dir: Union[str, Path],
                      format: Optional[str] = None,
                      **kwargs) -> None:
        """
        Write multiple datasets.
        
        Args:
            data_dict: Dictionary of name -> data
            base_dir: Base directory for output
            format: File format
            **kwargs: Arguments passed to write()
        """
        base_dir = Path(base_dir)
        base_dir.mkdir(parents=True, exist_ok=True)
        
        for name, data in data_dict.items():
            filepath = base_dir / f"{name}.{format or 'parquet'}"
            self.write(data, filepath, format, **kwargs)
    
    def _atomic_write(self, data: Any, filepath: Path, format: str,
                     append: bool, **kwargs):
        """Perform atomic write using temporary file."""
        # Create temporary file
        temp_fd, temp_path = tempfile.mkstemp(
            dir=filepath.parent,
            prefix=f".{filepath.name}.",
            suffix=".tmp"
        )
        
        try:
            # Close the file descriptor
            os.close(temp_fd)
            
            # Write to temporary file
            writer = self._writers[format]
            writer(data, Path(temp_path), append, **kwargs)
            
            # Atomic rename
            if filepath.exists() and not append:
                filepath.unlink()
            
            Path(temp_path).rename(filepath)
            
        except Exception as e:
            # Clean up temporary file on error
            if Path(temp_path).exists():
                Path(temp_path).unlink()
            raise e
    
    def _write_partitioned(self, df: pd.DataFrame, filepath: Path,
                          format: str, partition_by: List[str], **kwargs):
        """Write partitioned data."""
        base_dir = filepath.parent
        filename_template = filepath.name
        
        # Group by partition columns
        for partition_values, group_df in df.groupby(partition_by):
            # Create partition path
            if isinstance(partition_values, tuple):
                partition_parts = [f"{col}={val}" for col, val in 
                                 zip(partition_by, partition_values)]
            else:
                partition_parts = [f"{partition_by[0]}={partition_values}"]
            
            partition_dir = base_dir
            for part in partition_parts:
                partition_dir = partition_dir / part
            
            partition_dir.mkdir(parents=True, exist_ok=True)
            
            # Write partition
            partition_file = partition_dir / filename_template
            writer = self._writers[format]
            writer(group_df, partition_file, False, **kwargs)
    
    def _write_csv(self, data: pd.DataFrame, filepath: Path,
                   append: bool, **kwargs):
        """Write CSV file."""
        mode = 'a' if append else 'w'
        header = not append or not filepath.exists()
        
        compression = kwargs.pop('compression', self.compression)
        data.to_csv(filepath, mode=mode, header=header, 
                   compression=compression, **kwargs)
    
    def _write_excel(self, data: Union[pd.DataFrame, Dict[str, pd.DataFrame]], 
                     filepath: Path, append: bool, **kwargs):
        """Write Excel file."""
        if append and filepath.exists():
            # Read existing file
            with pd.ExcelFile(filepath) as xls:
                existing_sheets = {sheet: xls.parse(sheet) for sheet in xls.sheet_names}
            
            # Update with new data
            if isinstance(data, pd.DataFrame):
                existing_sheets['Sheet1'] = data
            else:
                existing_sheets.update(data)
            
            data = existing_sheets
        
        # Write file
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            if isinstance(data, pd.DataFrame):
                data.to_excel(writer, **kwargs)
            else:
                for sheet_name, df in data.items():
                    df.to_excel(writer, sheet_name=sheet_name, **kwargs)
    
    def _write_json(self, data: Union[pd.DataFrame, dict, list],
                    filepath: Path, append: bool, **kwargs):
        """Write JSON file."""
        if isinstance(data, pd.DataFrame):
            orient = kwargs.pop('orient', 'records')
            json_str = data.to_json(orient=orient, **kwargs)
            json_data = json.loads(json_str)
        else:
            json_data = data
        
        if append and filepath.exists():
            # Read existing data
            with open(filepath, 'r') as f:
                existing_data = json.load(f)
            
            # Merge data
            if isinstance(existing_data, list) and isinstance(json_data, list):
                json_data = existing_data + json_data
            elif isinstance(existing_data, dict) and isinstance(json_data, dict):
                existing_data.update(json_data)
                json_data = existing_data
        
        # Write file
        with open(filepath, 'w') as f:
            json.dump(json_data, f, indent=2, **kwargs)
    
    def _write_parquet(self, data: pd.DataFrame, filepath: Path,
                       append: bool, **kwargs):
        """Write Parquet file."""
        compression = kwargs.pop('compression', self.compression or 'snappy')
        
        if append and filepath.exists():
            # Read existing data
            existing_df = pd.read_parquet(filepath)
            data = pd.concat([existing_df, data], ignore_index=True)
        
        data.to_parquet(filepath, compression=compression, **kwargs)
    
    def _write_hdf5(self, data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
                    filepath: Path, append: bool, **kwargs):
        """Write HDF5 file."""
        mode = 'a' if append else 'w'
        key = kwargs.pop('key', 'data')
        
        if isinstance(data, pd.DataFrame):
            data.to_hdf(filepath, key=key, mode=mode, **kwargs)
        else:
            # Multiple datasets
            with pd.HDFStore(filepath, mode=mode) as store:
                for key, df in data.items():
                    store[key] = df
    
    def _write_feather(self, data: pd.DataFrame, filepath: Path,
                       append: bool, **kwargs):
        """Write Feather file."""
        if append:
            warnings.warn("Feather format doesn't support append mode")
        
        compression = kwargs.pop('compression', self.compression or 'lz4')
        data.to_feather(filepath, compression=compression, **kwargs)
    
    def _write_sql(self, data: pd.DataFrame, filepath: Path,
                   append: bool, **kwargs):
        """Write to SQL database."""
        # filepath should be connection string or path to SQLite
        con = kwargs.pop('con', f"sqlite:///{filepath}")
        table_name = kwargs.pop('table_name', 'data')
        if_exists = 'append' if append else 'replace'
        
        data.to_sql(table_name, con, if_exists=if_exists, **kwargs)
    
    def _write_pickle(self, data: Any, filepath: Path,
                      append: bool, **kwargs):
        """Write pickle file."""
        if append:
            warnings.warn("Pickle format doesn't support append mode")
        
        protocol = kwargs.pop('protocol', pickle.HIGHEST_PROTOCOL)
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f, protocol=protocol)
    
    def _write_text(self, data: Union[str, List[str]], filepath: Path,
                    append: bool, **kwargs):
        """Write text file."""
        mode = 'a' if append else 'w'
        encoding = kwargs.get('encoding', 'utf-8')
        
        with open(filepath, mode, encoding=encoding) as f:
            if isinstance(data, list):
                for line in data:
                    f.write(line + '\n')
            else:
                f.write(data)
    
    def _write_numpy(self, data: np.ndarray, filepath: Path,
                     append: bool, **kwargs):
        """Write NumPy array."""
        if append:
            warnings.warn("NumPy format doesn't support append mode")
        
        if filepath.suffix == '.npy':
            np.save(filepath, data, **kwargs)
        elif filepath.suffix == '.npz':
            arrays = kwargs.pop('arrays', {'arr_0': data})
            np.savez(filepath, **arrays)
        else:
            np.savetxt(filepath, data, **kwargs)
    
    def _detect_format(self, filepath: Path) -> str:
        """Detect file format from extension."""
        suffix = filepath.suffix.lower()
        
        format_map = {
            '.csv': 'csv',
            '.xlsx': 'excel',
            '.xls': 'excel',
            '.json': 'json',
            '.parquet': 'parquet',
            '.pkl': 'pickle',
            '.pickle': 'pickle',
            '.feather': 'feather',
            '.h5': 'hdf5',
            '.hdf5': 'hdf5',
            '.hdf': 'hdf5',
            '.db': 'sql',
            '.sqlite': 'sql',
            '.txt': 'text',
            '.npy': 'numpy',
            '.npz': 'numpy'
        }
        
        if suffix not in format_map:
            raise ValueError(f"Cannot detect format for extension: {suffix}")
        
        return format_map[suffix]


class ModelSerializer:
    """
    Advanced model serialization with support for multiple frameworks.
    
    Handles serialization of models from various ML frameworks with
    metadata, versioning, and compression support.
    """
    
    def __init__(self,
                 include_metadata: bool = True,
                 compression: Optional[str] = None):
        """
        Initialize ModelSerializer.
        
        Args:
            include_metadata: Whether to include metadata
            compression: Compression type
        """
        self.include_metadata = include_metadata
        self.compression = compression
        self._serializers = self._init_serializers()
    
    def _init_serializers(self) -> Dict[str, Dict[str, Callable]]:
        """Initialize framework-specific serializers."""
        return {
            'sklearn': {
                'save': self._save_sklearn,
                'load': self._load_sklearn
            },
            'tensorflow': {
                'save': self._save_tensorflow,
                'load': self._load_tensorflow
            },
            'pytorch': {
                'save': self._save_pytorch,
                'load': self._load_pytorch
            },
            'xgboost': {
                'save': self._save_xgboost,
                'load': self._load_xgboost
            },
            'lightgbm': {
                'save': self._save_lightgbm,
                'load': self._load_lightgbm
            },
            'catboost': {
                'save': self._save_catboost,
                'load': self._load_catboost
            },
            'generic': {
                'save': self._save_generic,
                'load': self._load_generic
            }
        }
    
    def save(self,
             model: Any,
             filepath: Union[str, Path],
             framework: Optional[str] = None,
             metadata: Optional[Dict[str, Any]] = None,
             **kwargs) -> None:
        """
        Save model with metadata.
        
        Args:
            model: Model to save
            filepath: Output path
            framework: ML framework (auto-detected if None)
            metadata: Additional metadata
            **kwargs: Framework-specific options
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Auto-detect framework
        if framework is None:
            framework = self._detect_framework(model)
        
        if framework not in self._serializers:
            framework = 'generic'
        
        # Prepare metadata
        if self.include_metadata:
            model_metadata = self._create_metadata(model, framework)
            if metadata:
                model_metadata.update(metadata)
        else:
            model_metadata = metadata or {}
        
        # Save model
        serializer = self._serializers[framework]['save']
        
        if self.compression:
            # Save to temporary directory then compress
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir) / 'model'
                serializer(model, temp_path, model_metadata, **kwargs)
                self._compress_directory(temp_path, filepath)
        else:
            serializer(model, filepath, model_metadata, **kwargs)
        
        logger.info(f"Saved {framework} model to {filepath}")
    
    def load(self,
             filepath: Union[str, Path],
             framework: Optional[str] = None,
             return_metadata: bool = False,
             **kwargs) -> Union[Any, Tuple[Any, Dict[str, Any]]]:
        """
        Load model from file.
        
        Args:
            filepath: Model file path
            framework: ML framework (auto-detected if None)
            return_metadata: Whether to return metadata
            **kwargs: Framework-specific options
            
        Returns:
            Model or (model, metadata) tuple
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        # Handle compression
        if self._is_compressed(filepath):
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir) / 'model'
                self._decompress_to_directory(filepath, temp_path)
                return self.load(temp_path, framework, return_metadata, **kwargs)
        
        # Auto-detect framework
        if framework is None:
            framework = self._detect_framework_from_file(filepath)
        
        if framework not in self._serializers:
            framework = 'generic'
        
        # Load model
        loader = self._serializers[framework]['load']
        model, metadata = loader(filepath, **kwargs)
        
        logger.info(f"Loaded {framework} model from {filepath}")
        
        if return_metadata:
            return model, metadata
        
        return model
    
    def _save_sklearn(self, model: Any, filepath: Path,
                      metadata: Dict[str, Any], **kwargs):
        """Save scikit-learn model."""
        import joblib
        
        if filepath.suffix == '.pkl':
            # Single file format
            save_data = {
                'model': model,
                'metadata': metadata,
                'framework': 'sklearn'
            }
            joblib.dump(save_data, filepath)
        else:
            # Directory format
            filepath.mkdir(exist_ok=True)
            joblib.dump(model, filepath / 'model.pkl')
            
            if metadata:
                with open(filepath / 'metadata.json', 'w') as f:
                    json.dump(metadata, f, indent=2)
    
    def _load_sklearn(self, filepath: Path, **kwargs) -> Tuple[Any, Dict[str, Any]]:
        """Load scikit-learn model."""
        import joblib
        
        if filepath.is_file():
            # Single file format
            save_data = joblib.load(filepath)
            if isinstance(save_data, dict) and 'model' in save_data:
                return save_data['model'], save_data.get('metadata', {})
            else:
                # Old format - just the model
                return save_data, {}
        else:
            # Directory format
            model = joblib.load(filepath / 'model.pkl')
            metadata = {}
            
            metadata_file = filepath / 'metadata.json'
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
            
            return model, metadata
    
    def _save_tensorflow(self, model: Any, filepath: Path,
                        metadata: Dict[str, Any], **kwargs):
        """Save TensorFlow/Keras model."""
        try:
            import tensorflow as tf
            
            # Save model
            if hasattr(model, 'save'):
                model.save(filepath, **kwargs)
            else:
                tf.saved_model.save(model, str(filepath))
            
            # Save metadata
            if metadata:
                with open(filepath / 'metadata.json', 'w') as f:
                    json.dump(metadata, f, indent=2)
        
        except ImportError:
            raise ImportError("TensorFlow not installed")
    
    def _load_tensorflow(self, filepath: Path, **kwargs) -> Tuple[Any, Dict[str, Any]]:
        """Load TensorFlow/Keras model."""
        try:
            import tensorflow as tf
            
            # Load model
            model = tf.keras.models.load_model(filepath, **kwargs)
            
            # Load metadata
            metadata = {}
            metadata_file = filepath / 'metadata.json'
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
            
            return model, metadata
        
        except ImportError:
            raise ImportError("TensorFlow not installed")
    
    def _save_pytorch(self, model: Any, filepath: Path,
                      metadata: Dict[str, Any], **kwargs):
        """Save PyTorch model."""
        try:
            import torch
            
            filepath.mkdir(exist_ok=True)
            
            # Save model state dict
            save_dict = {
                'model_state_dict': model.state_dict(),
                'model_class': model.__class__.__name__,
                'metadata': metadata
            }
            
            # Add optimizer if provided
            optimizer = kwargs.get('optimizer')
            if optimizer:
                save_dict['optimizer_state_dict'] = optimizer.state_dict()
            
            torch.save(save_dict, filepath / 'model.pth')
            
            # Save model architecture if requested
            if kwargs.get('save_architecture', True):
                with open(filepath / 'architecture.txt', 'w') as f:
                    f.write(str(model))
        
        except ImportError:
            raise ImportError("PyTorch not installed")
    
    def _load_pytorch(self, filepath: Path, **kwargs) -> Tuple[Any, Dict[str, Any]]:
        """Load PyTorch model."""
        try:
            import torch
            
            # Load checkpoint
            checkpoint = torch.load(filepath / 'model.pth', 
                                  map_location=kwargs.get('map_location', 'cpu'))
            
            # Model class must be provided or importable
            model_class = kwargs.get('model_class')
            if model_class is None:
                raise ValueError("model_class must be provided for PyTorch models")
            
            # Initialize model
            model = model_class(**kwargs.get('model_kwargs', {}))
            model.load_state_dict(checkpoint['model_state_dict'])
            
            # Load optimizer if requested
            if 'optimizer' in kwargs and 'optimizer_state_dict' in checkpoint:
                kwargs['optimizer'].load_state_dict(checkpoint['optimizer_state_dict'])
            
            metadata = checkpoint.get('metadata', {})
            
            return model, metadata
        
        except ImportError:
            raise ImportError("PyTorch not installed")
    
    def _save_xgboost(self, model: Any, filepath: Path,
                      metadata: Dict[str, Any], **kwargs):
        """Save XGBoost model."""
        try:
            filepath.mkdir(exist_ok=True)
            
            # Save model
            model.save_model(filepath / 'model.json')
            
            # Save metadata
            if metadata:
                with open(filepath / 'metadata.json', 'w') as f:
                    json.dump(metadata, f, indent=2)
        
        except Exception as e:
            raise RuntimeError(f"Failed to save XGBoost model: {e}")
    
    def _load_xgboost(self, filepath: Path, **kwargs) -> Tuple[Any, Dict[str, Any]]:
        """Load XGBoost model."""
        try:
            import xgboost as xgb
            
            # Load model
            model = xgb.Booster()
            model.load_model(filepath / 'model.json')
            
            # Load metadata
            metadata = {}
            metadata_file = filepath / 'metadata.json'
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
            
            return model, metadata
        
        except ImportError:
            raise ImportError("XGBoost not installed")
    
    def _save_lightgbm(self, model: Any, filepath: Path,
                       metadata: Dict[str, Any], **kwargs):
        """Save LightGBM model."""
        try:
            filepath.mkdir(exist_ok=True)
            
            # Save model
            model.save_model(filepath / 'model.txt')
            
            # Save metadata
            if metadata:
                with open(filepath / 'metadata.json', 'w') as f:
                    json.dump(metadata, f, indent=2)
        
        except Exception as e:
            raise RuntimeError(f"Failed to save LightGBM model: {e}")
    
    def _load_lightgbm(self, filepath: Path, **kwargs) -> Tuple[Any, Dict[str, Any]]:
        """Load LightGBM model."""
        try:
            import lightgbm as lgb
            
            # Load model
            model = lgb.Booster(model_file=str(filepath / 'model.txt'))
            
            # Load metadata
            metadata = {}
            metadata_file = filepath / 'metadata.json'
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
            
            return model, metadata
        
        except ImportError:
            raise ImportError("LightGBM not installed")
    
    def _save_catboost(self, model: Any, filepath: Path,
                       metadata: Dict[str, Any], **kwargs):
        """Save CatBoost model."""
        try:
            filepath.mkdir(exist_ok=True)
            
            # Save model
            model.save_model(filepath / 'model.cbm')
            
            # Save metadata
            if metadata:
                with open(filepath / 'metadata.json', 'w') as f:
                    json.dump(metadata, f, indent=2)
        
        except Exception as e:
            raise RuntimeError(f"Failed to save CatBoost model: {e}")
    
    def _load_catboost(self, filepath: Path, **kwargs) -> Tuple[Any, Dict[str, Any]]:
        """Load CatBoost model."""
        try:
            from catboost import CatBoostClassifier, CatBoostRegressor
            
            # Try to determine model type from metadata
            metadata = {}
            metadata_file = filepath / 'metadata.json'
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
            
            # Load model
            model_type = metadata.get('model_type', 'classifier')
            if model_type == 'regressor':
                model = CatBoostRegressor()
            else:
                model = CatBoostClassifier()
            
            model.load_model(filepath / 'model.cbm')
            
            return model, metadata
        
        except ImportError:
            raise ImportError("CatBoost not installed")
    
    def _save_generic(self, model: Any, filepath: Path,
                      metadata: Dict[str, Any], **kwargs):
        """Save generic model using pickle."""
        if filepath.suffix == '.pkl':
            # Single file format
            save_data = {
                'model': model,
                'metadata': metadata,
                'framework': 'generic'
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(save_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            # Directory format
            filepath.mkdir(exist_ok=True)
            
            with open(filepath / 'model.pkl', 'wb') as f:
                pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            if metadata:
                with open(filepath / 'metadata.json', 'w') as f:
                    json.dump(metadata, f, indent=2)
    
    def _load_generic(self, filepath: Path, **kwargs) -> Tuple[Any, Dict[str, Any]]:
        """Load generic model."""
        if filepath.is_file():
            # Single file format
            with open(filepath, 'rb') as f:
                save_data = pickle.load(f)
            
            if isinstance(save_data, dict) and 'model' in save_data:
                return save_data['model'], save_data.get('metadata', {})
            else:
                return save_data, {}
        else:
            # Directory format
            with open(filepath / 'model.pkl', 'rb') as f:
                model = pickle.load(f)
            
            metadata = {}
            metadata_file = filepath / 'metadata.json'
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
            
            return model, metadata
    
    def _detect_framework(self, model: Any) -> str:
        """Detect ML framework from model object."""
        model_class = model.__class__.__module__
        
        if 'sklearn' in model_class:
            return 'sklearn'
        elif 'tensorflow' in model_class or 'keras' in model_class:
            return 'tensorflow'
        elif 'torch' in model_class:
            return 'pytorch'
        elif 'xgboost' in model_class:
            return 'xgboost'
        elif 'lightgbm' in model_class:
            return 'lightgbm'
        elif 'catboost' in model_class:
            return 'catboost'
        else:
            return 'generic'
    
    def _detect_framework_from_file(self, filepath: Path) -> str:
        """Detect ML framework from file structure."""
        if filepath.is_file():
            # Check file extension
            if filepath.suffix == '.h5' or filepath.suffix == '.keras':
                return 'tensorflow'
            elif filepath.suffix == '.pth':
                return 'pytorch'
            else:
                return 'generic'
        else:
            # Check directory contents
            if (filepath / 'saved_model.pb').exists():
                return 'tensorflow'
            elif (filepath / 'model.pth').exists():
                return 'pytorch'
            elif (filepath / 'model.json').exists() and (filepath / 'model.txt').exists():
                return 'xgboost'
            elif (filepath / 'model.txt').exists():
                return 'lightgbm'
            elif (filepath / 'model.cbm').exists():
                return 'catboost'
            else:
                return 'generic'
    
    def _create_metadata(self, model: Any, framework: str) -> Dict[str, Any]:
        """Create model metadata."""
        metadata = {
            'framework': framework,
            'model_class': model.__class__.__name__,
            'model_module': model.__class__.__module__,
            'saved_at': datetime.now().isoformat(),
            'serializer_version': '1.0.0'
        }
        
        # Add framework-specific metadata
        if framework == 'sklearn':
            if hasattr(model, 'get_params'):
                metadata['params'] = model.get_params()
        elif framework == 'tensorflow':
            if hasattr(model, 'summary'):
                import io
                stream = io.StringIO()
                model.summary(print_fn=lambda x: stream.write(x + '\n'))
                metadata['summary'] = stream.getvalue()
        
        return metadata
    
    def _compress_directory(self, source_dir: Path, output_file: Path):
        """Compress directory to file."""
        import tarfile
        
        mode = 'w:gz' if self.compression == 'gzip' else 'w'
        with tarfile.open(output_file, mode) as tar:
            tar.add(source_dir, arcname='.')
    
    def _decompress_to_directory(self, archive_file: Path, output_dir: Path):
        """Decompress file to directory."""
        import tarfile
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with tarfile.open(archive_file, 'r:*') as tar:
            tar.extractall(output_dir)
    
    def _is_compressed(self, filepath: Path) -> bool:
        """Check if file is compressed."""
        return filepath.suffix in ['.gz', '.tar', '.tgz']


class ConfigLoader:
    """
    Advanced configuration loader with support for multiple formats,
    environment variables, and validation.
    """
    
    def __init__(self,
                 search_paths: Optional[List[Path]] = None,
                 env_prefix: str = 'APP_',
                 validate: bool = True):
        """
        Initialize ConfigLoader.
        
        Args:
            search_paths: Paths to search for config files
            env_prefix: Prefix for environment variables
            validate: Whether to validate configurations
        """
        self.search_paths = search_paths or [Path.cwd(), Path.home() / '.config']
        self.env_prefix = env_prefix
        self.validate = validate
        self._loaders = self._init_loaders()
        self._cache = {}
    
    def _init_loaders(self) -> Dict[str, Callable]:
        """Initialize format-specific loaders."""
        return {
            'yaml': self._load_yaml,
            'json': self._load_json,
            'toml': self._load_toml,
            'ini': self._load_ini,
            'env': self._load_env,
            'py': self._load_python
        }
    
    def load(self,
             config_name: str,
             format: Optional[str] = None,
             merge_env: bool = True,
             validate_schema: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Load configuration from file.
        
        Args:
            config_name: Configuration name or path
            format: Config format (auto-detected if None)
            merge_env: Whether to merge environment variables
            validate_schema: JSON schema for validation
            
        Returns:
            Configuration dictionary
        """
        # Check cache
        cache_key = f"{config_name}:{format}:{merge_env}"
        if cache_key in self._cache:
            return self._cache[cache_key].copy()
        
        # Find config file
        config_path = self._find_config_file(config_name, format)
        
        if not config_path:
            raise FileNotFoundError(f"Config file not found: {config_name}")
        
        # Load configuration
        config = self._load_config_file(config_path)
        
        # Merge environment variables
        if merge_env:
            env_config = self._load_environment_variables()
            config = self._merge_configs(config, env_config)
        
        # Validate configuration
        if self.validate and validate_schema:
            self._validate_config(config, validate_schema)
        
        # Cache configuration
        self._cache[cache_key] = config.copy()
        
        return config
    
    def load_multiple(self,
                     config_names: List[str],
                     merge: bool = True) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Load multiple configuration files.
        
        Args:
            config_names: List of configuration names
            merge: Whether to merge configurations
            
        Returns:
            Merged configuration or list of configurations
        """
        configs = []
        
        for config_name in config_names:
            try:
                config = self.load(config_name)
                configs.append(config)
            except Exception as e:
                logger.error(f"Failed to load config {config_name}: {e}")
        
        if merge and configs:
            result = {}
            for config in configs:
                result = self._merge_configs(result, config)
            return result
        
        return configs
    
    def save(self,
             config: Dict[str, Any],
             filepath: Union[str, Path],
             format: Optional[str] = None,
             backup: bool = True) -> None:
        """
        Save configuration to file.
        
        Args:
            config: Configuration dictionary
            filepath: Output file path
            format: Config format (auto-detected if None)
            backup: Whether to backup existing file
        """
        filepath = Path(filepath)
        
        # Backup existing file
        if backup and filepath.exists():
            backup_path = filepath.with_suffix(f'.backup.{datetime.now().strftime("%Y%m%d_%H%M%S")}')
            shutil.copy2(filepath, backup_path)
        
        # Auto-detect format
        if format is None:
            format = self._detect_format(filepath)
        
        # Save configuration
        if format == 'yaml':
            self._save_yaml(config, filepath)
        elif format == 'json':
            self._save_json(config, filepath)
        elif format == 'toml':
            self._save_toml(config, filepath)
        else:
            raise ValueError(f"Unsupported format for saving: {format}")
    
    def _find_config_file(self, config_name: str, 
                         format: Optional[str]) -> Optional[Path]:
        """Find configuration file in search paths."""
        # If config_name is a path, use it directly
        if Path(config_name).exists():
            return Path(config_name)
        
        # Search in search paths
        extensions = ['.yaml', '.yml', '.json', '.toml', '.ini', '.env', '.py']
        
        if format:
            # Limit to specific format
            ext_map = {
                'yaml': ['.yaml', '.yml'],
                'json': ['.json'],
                'toml': ['.toml'],
                'ini': ['.ini'],
                'env': ['.env'],
                'py': ['.py']
            }
            extensions = ext_map.get(format, extensions)
        
        for search_path in self.search_paths:
            for ext in extensions:
                config_path = search_path / f"{config_name}{ext}"
                if config_path.exists():
                    return config_path
        
        return None
    
    def _load_config_file(self, filepath: Path) -> Dict[str, Any]:
        """Load configuration from file."""
        format = self._detect_format(filepath)
        
        if format not in self._loaders:
            raise ValueError(f"Unsupported config format: {format}")
        
        loader = self._loaders[format]
        return loader(filepath)
    
    def _load_yaml(self, filepath: Path) -> Dict[str, Any]:
        """Load YAML configuration."""
        with open(filepath, 'r') as f:
            return yaml.safe_load(f) or {}
    
    def _load_json(self, filepath: Path) -> Dict[str, Any]:
        """Load JSON configuration."""
        with open(filepath, 'r') as f:
            return json.load(f)
    
    def _load_toml(self, filepath: Path) -> Dict[str, Any]:
        """Load TOML configuration."""
        try:
            import toml
            with open(filepath, 'r') as f:
                return toml.load(f)
        except ImportError:
            raise ImportError("toml package not installed")
    
    def _load_ini(self, filepath: Path) -> Dict[str, Any]:
        """Load INI configuration."""
        import configparser
        
        parser = configparser.ConfigParser()
        parser.read(filepath)
        
        # Convert to nested dictionary
        config = {}
        for section in parser.sections():
            config[section] = dict(parser.items(section))
        
        return config
    
    def _load_env(self, filepath: Path) -> Dict[str, Any]:
        """Load .env file configuration."""
        config = {}
        
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    config[key.strip()] = value.strip()
        
        return config
    
    def _load_python(self, filepath: Path) -> Dict[str, Any]:
        """Load Python file configuration."""
        import importlib.util
        
        spec = importlib.util.spec_from_file_location("config", filepath)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Extract configuration (all uppercase variables)
        config = {}
        for name in dir(module):
            if name.isupper():
                config[name] = getattr(module, name)
        
        return config
    
    def _load_environment_variables(self) -> Dict[str, Any]:
        """Load configuration from environment variables."""
        config = {}
        
        for key, value in os.environ.items():
            if key.startswith(self.env_prefix):
                # Remove prefix and convert to lowercase
                config_key = key[len(self.env_prefix):].lower()
                
                # Convert value type
                config[config_key] = self._convert_env_value(value)
        
        return config
    
    def _convert_env_value(self, value: str) -> Any:
        """Convert environment variable value to appropriate type."""
        # Try to parse as JSON first
        try:
            return json.loads(value)
        except:
            pass
        
        # Check for boolean
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'
        
        # Check for number
        try:
            if '.' in value:
                return float(value)
            else:
                return int(value)
        except:
            pass
        
        # Return as string
        return value
    
    def _merge_configs(self, base: Dict[str, Any], 
                      override: Dict[str, Any]) -> Dict[str, Any]:
        """Merge two configuration dictionaries."""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                # Recursive merge for nested dictionaries
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _validate_config(self, config: Dict[str, Any], 
                        schema: Dict[str, Any]) -> None:
        """Validate configuration against schema."""
        try:
            import jsonschema
            jsonschema.validate(config, schema)
        except ImportError:
            logger.warning("jsonschema not installed, skipping validation")
        except jsonschema.ValidationError as e:
            raise ValueError(f"Configuration validation failed: {e}")
    
    def _detect_format(self, filepath: Path) -> str:
        """Detect configuration format from file extension."""
        suffix = filepath.suffix.lower()
        
        format_map = {
            '.yaml': 'yaml',
            '.yml': 'yaml',
            '.json': 'json',
            '.toml': 'toml',
            '.ini': 'ini',
            '.env': 'env',
            '.py': 'py'
        }
        
        if suffix not in format_map:
            raise ValueError(f"Cannot detect format for extension: {suffix}")
        
        return format_map[suffix]
    
    def _save_yaml(self, config: Dict[str, Any], filepath: Path) -> None:
        """Save YAML configuration."""
        with open(filepath, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    def _save_json(self, config: Dict[str, Any], filepath: Path) -> None:
        """Save JSON configuration."""
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)
    
    def _save_toml(self, config: Dict[str, Any], filepath: Path) -> None:
        """Save TOML configuration."""
        try:
            import toml
            with open(filepath, 'w') as f:
                toml.dump(config, f)
        except ImportError:
            raise ImportError("toml package not installed")


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
        self._lock = threading.Lock()
    
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
        if not self.exists(key):
            return default
        
        cache_path = self._get_cache_path(key)
        
        try:
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.error(f"Error loading cache for key {key}: {e}")
            return default
    
    def set(self, key: str, value: Any, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            metadata: Optional metadata
        """
        cache_path = self._get_cache_path(key)
        
        with self._lock:
            # Save value
            with open(cache_path, 'wb') as f:
                pickle.dump(value, f)
            
            # Update index
            self._index[key] = {
                'path': str(cache_path.relative_to(self.cache_dir)),
                'size': cache_path.stat().st_size,
                'created_at': datetime.now().isoformat(),
                'metadata': metadata or {}
            }
            
            self._save_index()
    
    def delete(self, key: str) -> bool:
        """
        Delete value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if deleted, False if not found
        """
        if not self.exists(key):
            return False
        
        cache_path = self._get_cache_path(key)
        
        with self._lock:
            # Remove file
            cache_path.unlink()
            
            # Update index
            del self._index[key]
            self._save_index()
        
        return True
    
    def clear(self) -> int:
        """
        Clear all cached values.
        
        Returns:
            Number of items cleared
        """
        count = 0
        
        with self._lock:
            for key in list(self._index.keys()):
                cache_path = self._get_cache_path(key)
                if cache_path.exists():
                    cache_path.unlink()
                    count += 1
            
            self._index = {}
            self._save_index()
        
        return count
    
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
            loaded_df = handler.read(filepath)
            print(f"✓ {format} format: {loaded_df.shape}")
        
        # Test DataReader
        print("\nTesting DataReader...")
        reader = DataReader(cache_enabled=True, cache_dir=tmpdir / 'cache')
        
        # Read with column selection and filters
        csv_path = tmpdir / "data.csv"
        filtered_df = reader.read(
            csv_path,
            columns=['A', 'C'],
            filters={'C': ['X', 'Y']}
        )
        print(f"✓ Filtered read: {filtered_df.shape}")
        
        # Test DataWriter
        print("\nTesting DataWriter...")
        writer = DataWriter(compression='gzip')
        
        # Write with partitioning
        writer.write(
            df,
            tmpdir / 'partitioned' / 'data.parquet',
            partition_by=['C']
        )
        print("✓ Partitioned write complete")
        
        # Test ModelSerializer
        print("\nTesting ModelSerializer...")
        from sklearn.ensemble import RandomForestClassifier
        
        model = RandomForestClassifier(n_estimators=10)
        model.fit(df[['A', 'B']], df['C'])
        
        serializer = ModelSerializer(include_metadata=True)
        model_path = tmpdir / 'model'
        
        # Save model
        serializer.save(model, model_path, metadata={'accuracy': 0.95})
        
        # Load model
        loaded_model, metadata = serializer.load(model_path, return_metadata=True)
        print(f"✓ Model loaded: {type(loaded_model).__name__}")
        print(f"  Metadata: {metadata}")
        
        # Test ConfigLoader
        print("\nTesting ConfigLoader...")
        loader = ConfigLoader(search_paths=[tmpdir])
        
        # Create sample config
        config = {
            'model': {
                'type': 'random_forest',
                'params': {
                    'n_estimators': 100,
                    'max_depth': 10
                }
            },
            'training': {
                'batch_size': 32,
                'epochs': 100
            }
        }
        
        # Save and load config
        config_path = tmpdir / 'config.yaml'
        loader.save(config, config_path)
        loaded_config = loader.load('config')
        print(f"✓ Config loaded: {list(loaded_config.keys())}")
        
        # Test DataCache
        print("\nTesting DataCache...")
        cache = DataCache(tmpdir / 'data_cache')
        
        # Cache computation
        with cache.cached_computation('expensive_result') as cached:
            if cached.exists:
                result = cached.value
                print("✓ Result loaded from cache")
            else:
                result = df.describe()
                cached.value = result
                print("✓ Result computed and cached")
        
        print(f"  Cache info: {cache.get_info()}")
        
        # Test CheckpointManager
        print("\nTesting CheckpointManager...")
        checkpoint_manager = CheckpointManager(tmpdir / 'checkpoints')
        
        # Save checkpoints
        for epoch in range(5):
            state = {
                'epoch': epoch,
                'model_state': f'model_state_{epoch}',
                'optimizer_state': f'optimizer_state_{epoch}',
                'loss': 1.0 / (epoch + 1)
            }
            
            checkpoint_manager.save_checkpoint(
                state, 
                epoch,
                is_best=(epoch == 3)
            )
        
        # Load checkpoint
        latest = checkpoint_manager.load_checkpoint('latest')
        print(f"✓ Latest checkpoint: epoch {latest['epoch']}")
        
        best = checkpoint_manager.load_checkpoint('best')
        print(f"✓ Best checkpoint: epoch {best['epoch']}")
        
        # List checkpoints
        checkpoints = checkpoint_manager.list_checkpoints()
        print(f"✓ Total checkpoints: {len(checkpoints)}")
        
        # Cleanup old checkpoints
        checkpoint_manager.cleanup(keep_last=3)
        remaining = checkpoint_manager.list_checkpoints()
        print(f"✓ After cleanup: {len(remaining)} checkpoints")


if __name__ == "__main__":
    example_file_operations()