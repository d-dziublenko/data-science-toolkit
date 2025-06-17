"""
utils/parallel.py
Parallel processing utilities for the data science toolkit.
"""

import os
import multiprocessing as mp
from multiprocessing import Pool, Process, Queue, Manager
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import threading
from typing import Any, Callable, Dict, List, Optional, Union, Tuple, Iterable
import numpy as np
import pandas as pd
from functools import partial, wraps
from tqdm import tqdm
import logging
import time
import warnings
from contextlib import contextmanager

logger = logging.getLogger(__name__)


def get_n_jobs(n_jobs: int = -1) -> int:
    """
    Get the actual number of jobs to use.
    
    Args:
        n_jobs: Number of jobs (-1 for all CPUs, -2 for all but one)
        
    Returns:
        Actual number of jobs
    """
    n_cpus = mp.cpu_count()
    
    if n_jobs == -1:
        return n_cpus
    elif n_jobs == -2:
        return max(1, n_cpus - 1)
    elif n_jobs < 0:
        return max(1, n_cpus + n_jobs + 1)
    else:
        return min(n_jobs, n_cpus)


class ParallelProcessor:
    """
    Generic parallel processor for various parallelization backends.
    """
    
    def __init__(self, n_jobs: int = -1, 
                 backend: str = 'multiprocessing',
                 verbose: bool = True):
        """
        Initialize the parallel processor.
        
        Args:
            n_jobs: Number of parallel jobs
            backend: Backend to use ('multiprocessing', 'threading', 'dask')
            verbose: Whether to show progress
        """
        self.n_jobs = get_n_jobs(n_jobs)
        self.backend = backend
        self.verbose = verbose
        
        logger.info(f"Initialized {backend} parallel processor with {self.n_jobs} jobs")
    
    def map(self, func: Callable, iterable: Iterable, 
            chunksize: Optional[int] = None,
            **kwargs) -> List[Any]:
        """
        Parallel map operation.
        
        Args:
            func: Function to apply
            iterable: Input iterable
            chunksize: Chunk size for processing
            **kwargs: Additional arguments for func
            
        Returns:
            List of results
        """
        if kwargs:
            func = partial(func, **kwargs)
        
        total = len(list(iterable)) if hasattr(iterable, '__len__') else None
        
        if self.backend == 'multiprocessing':
            return self._map_multiprocessing(func, iterable, chunksize, total)
        elif self.backend == 'threading':
            return self._map_threading(func, iterable, chunksize, total)
        elif self.backend == 'dask':
            return self._map_dask(func, iterable, chunksize)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")
    
    def _map_multiprocessing(self, func: Callable, iterable: Iterable, 
                           chunksize: Optional[int], total: Optional[int]) -> List[Any]:
        """Map using multiprocessing."""
        with Pool(self.n_jobs) as pool:
            if self.verbose and total:
                results = list(tqdm(
                    pool.imap(func, iterable, chunksize=chunksize or 1),
                    total=total,
                    desc="Processing"
                ))
            else:
                results = pool.map(func, iterable, chunksize=chunksize)
        return results
    
    def _map_threading(self, func: Callable, iterable: Iterable,
                      chunksize: Optional[int], total: Optional[int]) -> List[Any]:
        """Map using threading."""
        results = []
        with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
            futures = [executor.submit(func, item) for item in iterable]
            
            if self.verbose and total:
                for future in tqdm(as_completed(futures), total=total, desc="Processing"):
                    results.append(future.result())
            else:
                for future in as_completed(futures):
                    results.append(future.result())
        
        return results
    
    def _map_dask(self, func: Callable, iterable: Iterable,
                  chunksize: Optional[int]) -> List[Any]:
        """Map using dask."""
        try:
            import dask
            from dask import delayed, compute
            from dask.distributed import Client
            
            # Create delayed tasks
            tasks = [delayed(func)(item) for item in iterable]
            
            # Compute in parallel
            with Client(n_workers=self.n_jobs, threads_per_worker=1, 
                       processes=True, silence_logs=logging.ERROR) as client:
                results = compute(*tasks, scheduler='distributed')
            
            return list(results)
        except ImportError:
            raise ImportError("Dask is required for dask backend. Install with: pip install dask[distributed]")
    
    def apply_along_axis(self, func: Callable, array: np.ndarray, 
                        axis: int = 0) -> np.ndarray:
        """
        Apply function along array axis in parallel.
        
        Args:
            func: Function to apply
            array: Input array
            axis: Axis along which to apply
            
        Returns:
            Result array
        """
        # Move axis to position 0
        array = np.moveaxis(array, axis, 0)
        
        # Apply function to each slice
        results = self.map(func, array)
        
        # Stack results and move axis back
        result = np.stack(results)
        return np.moveaxis(result, 0, axis)
    
    def apply_dataframe(self, func: Callable, df: pd.DataFrame,
                       axis: int = 0, **kwargs) -> pd.DataFrame:
        """
        Apply function to DataFrame in parallel.
        
        Args:
            func: Function to apply
            df: Input DataFrame
            axis: Axis (0 for rows, 1 for columns)
            **kwargs: Additional arguments for func
            
        Returns:
            Result DataFrame
        """
        if axis == 0:
            # Split by rows
            chunks = np.array_split(df, self.n_jobs)
            func_partial = partial(func, **kwargs) if kwargs else func
            results = self.map(func_partial, chunks)
            return pd.concat(results, axis=0)
        else:
            # Apply to each column
            results = {}
            for col in df.columns:
                results[col] = func(df[col], **kwargs)
            return pd.DataFrame(results)


def parallel_apply(df: pd.DataFrame, func: Callable, 
                  axis: int = 1, n_jobs: int = -1,
                  backend: str = 'multiprocessing') -> pd.Series:
    """
    Parallel version of DataFrame.apply().
    
    Args:
        df: Input DataFrame
        func: Function to apply
        axis: Axis (0 for columns, 1 for rows)
        n_jobs: Number of parallel jobs
        backend: Parallelization backend
        
    Returns:
        Result Series
    """
    processor = ParallelProcessor(n_jobs=n_jobs, backend=backend, verbose=False)
    
    if axis == 1:
        # Apply to rows
        results = processor.map(func, [row for _, row in df.iterrows()])
        return pd.Series(results, index=df.index)
    else:
        # Apply to columns
        results = processor.map(func, [df[col] for col in df.columns])
        return pd.Series(results, index=df.columns)


def parallel_groupby_apply(df: pd.DataFrame, groupby_cols: Union[str, List[str]],
                          func: Callable, n_jobs: int = -1) -> pd.DataFrame:
    """
    Parallel version of DataFrame.groupby().apply().
    
    Args:
        df: Input DataFrame
        groupby_cols: Columns to group by
        func: Function to apply to each group
        n_jobs: Number of parallel jobs
        
    Returns:
        Result DataFrame
    """
    # Get groups
    grouped = df.groupby(groupby_cols)
    groups = [group for _, group in grouped]
    
    # Process in parallel
    processor = ParallelProcessor(n_jobs=n_jobs)
    results = processor.map(func, groups)
    
    # Combine results
    return pd.concat(results)


class ParallelBatch:
    """
    Process data in parallel batches.
    """
    
    def __init__(self, batch_size: int = 1000, n_jobs: int = -1):
        """
        Initialize parallel batch processor.
        
        Args:
            batch_size: Size of each batch
            n_jobs: Number of parallel jobs
        """
        self.batch_size = batch_size
        self.n_jobs = get_n_jobs(n_jobs)
    
    def process(self, data: Union[pd.DataFrame, np.ndarray],
                func: Callable, 
                combine_func: Optional[Callable] = None,
                progress: bool = True) -> Any:
        """
        Process data in parallel batches.
        
        Args:
            data: Input data
            func: Function to apply to each batch
            combine_func: Function to combine results
            progress: Whether to show progress
            
        Returns:
            Combined results
        """
        # Create batches
        n_samples = len(data)
        n_batches = (n_samples + self.batch_size - 1) // self.batch_size
        
        batches = []
        for i in range(n_batches):
            start = i * self.batch_size
            end = min((i + 1) * self.batch_size, n_samples)
            
            if isinstance(data, pd.DataFrame):
                batch = data.iloc[start:end]
            else:
                batch = data[start:end]
            
            batches.append(batch)
        
        # Process batches in parallel
        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            if progress:
                results = list(tqdm(
                    executor.map(func, batches),
                    total=len(batches),
                    desc="Processing batches"
                ))
            else:
                results = list(executor.map(func, batches))
        
        # Combine results
        if combine_func:
            return combine_func(results)
        elif isinstance(results[0], pd.DataFrame):
            return pd.concat(results)
        elif isinstance(results[0], np.ndarray):
            return np.concatenate(results)
        else:
            return results


def chunked_parallel_process(func: Callable, data: List[Any],
                           chunk_size: int = 100,
                           n_jobs: int = -1,
                           reducer: Optional[Callable] = None) -> Any:
    """
    Process large data in chunks with parallel execution.
    
    Args:
        func: Function to apply
        data: Input data
        chunk_size: Size of each chunk
        n_jobs: Number of parallel jobs
        reducer: Function to reduce chunk results
        
    Returns:
        Processed results
    """
    # Split data into chunks
    chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
    
    # Process chunks in parallel
    processor = ParallelProcessor(n_jobs=n_jobs)
    chunk_results = processor.map(lambda chunk: [func(item) for item in chunk], chunks)
    
    # Flatten results
    results = [item for chunk in chunk_results for item in chunk]
    
    # Apply reducer if provided
    if reducer:
        return reducer(results)
    
    return results


class SharedMemoryArray:
    """
    Shared memory array for efficient parallel processing.
    """
    
    def __init__(self, shape: Tuple[int, ...], dtype: np.dtype = np.float64):
        """
        Initialize shared memory array.
        
        Args:
            shape: Array shape
            dtype: Data type
        """
        self.shape = shape
        self.dtype = dtype
        self.size = np.prod(shape)
        
        # Create shared memory
        self.shared_array = mp.Array(
            self._get_ctypes_type(dtype),
            self.size
        )
        
    def _get_ctypes_type(self, dtype: np.dtype):
        """Get ctypes type for numpy dtype."""
        type_map = {
            np.float64: 'd',
            np.float32: 'f',
            np.int64: 'l',
            np.int32: 'i',
            np.uint8: 'B'
        }
        return type_map.get(dtype.type, 'd')
    
    def to_numpy(self) -> np.ndarray:
        """Convert to numpy array."""
        return np.frombuffer(
            self.shared_array.get_obj(),
            dtype=self.dtype
        ).reshape(self.shape)
    
    def from_numpy(self, array: np.ndarray) -> None:
        """Copy from numpy array."""
        np.copyto(self.to_numpy(), array)


@contextmanager
def parallel_backend(backend: str = 'multiprocessing', n_jobs: int = -1):
    """
    Context manager for parallel backend configuration.
    
    Args:
        backend: Backend to use
        n_jobs: Number of jobs
        
    Usage:
        with parallel_backend('threading', n_jobs=4):
            # Your parallel code here
    """
    # Store current settings
    old_backend = os.environ.get('PARALLEL_BACKEND', 'multiprocessing')
    old_n_jobs = os.environ.get('PARALLEL_N_JOBS', '-1')
    
    # Set new settings
    os.environ['PARALLEL_BACKEND'] = backend
    os.environ['PARALLEL_N_JOBS'] = str(n_jobs)
    
    try:
        yield
    finally:
        # Restore old settings
        os.environ['PARALLEL_BACKEND'] = old_backend
        os.environ['PARALLEL_N_JOBS'] = old_n_jobs


def parallel_decorator(n_jobs: int = -1, backend: str = 'multiprocessing'):
    """
    Decorator to parallelize function calls.
    
    Args:
        n_jobs: Number of parallel jobs
        backend: Parallelization backend
        
    Usage:
        @parallel_decorator(n_jobs=4)
        def process_item(item):
            return expensive_computation(item)
    """
    def decorator(func):
        @wraps(func)
        def wrapper(items, *args, **kwargs):
            processor = ParallelProcessor(n_jobs=n_jobs, backend=backend)
            partial_func = partial(func, *args, **kwargs)
            return processor.map(partial_func, items)
        return wrapper
    return decorator


class ProgressParallel:
    """
    Parallel processing with real-time progress tracking.
    """
    
    def __init__(self, n_jobs: int = -1):
        """Initialize with number of jobs."""
        self.n_jobs = get_n_jobs(n_jobs)
        self.manager = Manager()
        self.counter = self.manager.Value('i', 0)
        self.total = self.manager.Value('i', 0)
    
    def _process_with_progress(self, args: Tuple[Callable, Any, Queue]) -> Any:
        """Process single item and update progress."""
        func, item, queue = args
        result = func(item)
        
        with self.counter.get_lock():
            self.counter.value += 1
            
        return result
    
    def map(self, func: Callable, items: List[Any], 
            desc: str = "Processing") -> List[Any]:
        """
        Map function over items with progress bar.
        
        Args:
            func: Function to apply
            items: List of items
            desc: Progress bar description
            
        Returns:
            List of results
        """
        self.total.value = len(items)
        queue = self.manager.Queue()
        
        # Create tasks
        tasks = [(func, item, queue) for item in items]
        
        # Process with progress
        with Pool(self.n_jobs) as pool:
            with tqdm(total=len(items), desc=desc) as pbar:
                results = []
                
                for result in pool.imap_unordered(self._process_with_progress, tasks):
                    results.append(result)
                    pbar.update(1)
        
        return results


# Memory-efficient parallel processing
def parallel_read_csv(file_paths: List[str], n_jobs: int = -1, **read_kwargs) -> pd.DataFrame:
    """
    Read multiple CSV files in parallel.
    
    Args:
        file_paths: List of file paths
        n_jobs: Number of parallel jobs
        **read_kwargs: Arguments for pd.read_csv
        
    Returns:
        Combined DataFrame
    """
    processor = ParallelProcessor(n_jobs=n_jobs)
    dfs = processor.map(partial(pd.read_csv, **read_kwargs), file_paths)
    return pd.concat(dfs, ignore_index=True)


def parallel_save_partitions(df: pd.DataFrame, output_dir: str,
                           partition_col: str, n_jobs: int = -1,
                           file_format: str = 'parquet') -> None:
    """
    Save DataFrame partitions in parallel.
    
    Args:
        df: DataFrame to partition
        output_dir: Output directory
        partition_col: Column to partition by
        n_jobs: Number of parallel jobs
        file_format: Output file format
    """
    os.makedirs(output_dir, exist_ok=True)
    
    def save_partition(args):
        value, group = args
        filename = f"{partition_col}={value}.{file_format}"
        filepath = os.path.join(output_dir, filename)
        
        if file_format == 'parquet':
            group.to_parquet(filepath)
        elif file_format == 'csv':
            group.to_csv(filepath, index=False)
        else:
            raise ValueError(f"Unsupported format: {file_format}")
    
    # Get partitions
    partitions = list(df.groupby(partition_col))
    
    # Save in parallel
    processor = ParallelProcessor(n_jobs=n_jobs)
    processor.map(save_partition, partitions)


# Example usage
def example_parallel_processing():
    """Example of parallel processing utilities."""
    import time
    
    # Example function
    def expensive_computation(x):
        time.sleep(0.1)  # Simulate work
        return x ** 2
    
    # Test data
    data = list(range(20))
    
    print("Testing ParallelProcessor...")
    processor = ParallelProcessor(n_jobs=4)
    
    start = time.time()
    results = processor.map(expensive_computation, data)
    parallel_time = time.time() - start
    
    print(f"  Parallel time: {parallel_time:.2f}s")
    print(f"  Results: {results[:5]}...")
    
    # Compare with sequential
    start = time.time()
    sequential_results = [expensive_computation(x) for x in data]
    sequential_time = time.time() - start
    
    print(f"  Sequential time: {sequential_time:.2f}s")
    print(f"  Speedup: {sequential_time / parallel_time:.2f}x")
    
    # Test parallel DataFrame operations
    print("\nTesting parallel DataFrame operations...")
    df = pd.DataFrame({
        'A': np.random.randn(1000),
        'B': np.random.randn(1000),
        'group': np.random.choice(['X', 'Y', 'Z'], 1000)
    })
    
    def process_group(group):
        return group['A'].mean() + group['B'].std()
    
    result = parallel_groupby_apply(df, 'group', process_group, n_jobs=4)
    print(f"  Group results shape: {result.shape}")
    
    # Test decorator
    print("\nTesting parallel decorator...")
    
    @parallel_decorator(n_jobs=4)
    def process_item(x):
        return x ** 3
    
    decorated_results = process_item(data)
    print(f"  Decorated results: {decorated_results[:5]}...")


if __name__ == "__main__":
    example_parallel_processing()