"""
utils/parallel.py
Parallel processing utilities for the data science toolkit.
"""

import asyncio
import functools
import json
import logging
import multiprocessing as mp
import os
import pickle
import queue
import socket
import threading
import time
import warnings
from collections import defaultdict
from concurrent.futures import (ProcessPoolExecutor, ThreadPoolExecutor,
                                as_completed)
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from functools import partial, wraps
from multiprocessing import Array, Manager, Pool, Process, Queue, Value
from typing import (Any, Callable, Dict, Iterable, List, Optional, Tuple,
                    TypeVar, Union)

import dask
import joblib
import numpy as np
import pandas as pd
import ray
from tqdm import tqdm

try:
    import dask.array as da
    import dask.dataframe as dd
    from dask.distributed import Client
    from dask.distributed import as_completed as dask_as_completed

    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False

try:
    import ray

    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False

logger = logging.getLogger(__name__)

T = TypeVar("T")


class BackendType(Enum):
    """Supported parallel processing backends."""

    MULTIPROCESSING = "multiprocessing"
    THREADING = "threading"
    DASK = "dask"
    RAY = "ray"
    JOBLIB = "joblib"
    ASYNCIO = "asyncio"


@dataclass
class ParallelConfig:
    """
    Configuration for parallel processing.

    This class holds all configuration parameters for parallel execution,
    making it easy to manage and reuse configurations.
    """

    n_jobs: int = -1
    backend: Union[str, BackendType] = BackendType.MULTIPROCESSING
    chunk_size: Optional[int] = None
    batch_size: int = 1000
    verbose: bool = True
    show_progress: bool = True
    timeout: Optional[float] = None
    memory_limit: Optional[str] = None  # e.g., "4GB"

    # Backend-specific configurations
    multiprocessing_context: str = "spawn"  # spawn, fork, forkserver
    thread_name_prefix: str = "Worker"

    # Dask-specific
    dask_scheduler: str = "threads"  # threads, processes, synchronous
    dask_dashboard_address: Optional[str] = ":8787"
    dask_n_workers: Optional[int] = None
    dask_threads_per_worker: Optional[int] = None

    # Ray-specific
    ray_address: Optional[str] = None  # Ray cluster address
    ray_num_cpus: Optional[int] = None
    ray_num_gpus: Optional[int] = None
    ray_object_store_memory: Optional[int] = None

    # Error handling
    retry_failed: bool = True
    max_retries: int = 3
    retry_delay: float = 1.0
    ignore_errors: bool = False

    # Resource management
    max_memory_per_worker: Optional[str] = None
    cpu_affinity: Optional[List[int]] = None

    def __post_init__(self):
        """Validate and process configuration."""
        self.n_jobs = get_n_jobs(self.n_jobs)

        if isinstance(self.backend, str):
            try:
                self.backend = BackendType(self.backend.lower())
            except ValueError:
                raise ValueError(f"Unsupported backend: {self.backend}")

        # Validate backend availability
        if self.backend == BackendType.DASK and not DASK_AVAILABLE:
            warnings.warn("Dask not available, falling back to multiprocessing")
            self.backend = BackendType.MULTIPROCESSING
        elif self.backend == BackendType.RAY and not RAY_AVAILABLE:
            warnings.warn("Ray not available, falling back to multiprocessing")
            self.backend = BackendType.MULTIPROCESSING

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        config_dict = {}
        for field_name in self.__dataclass_fields__:
            value = getattr(self, field_name)
            if isinstance(value, Enum):
                value = value.value
            config_dict[field_name] = value
        return config_dict

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ParallelConfig":
        """Create configuration from dictionary."""
        return cls(**config_dict)


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

    def __init__(
        self,
        n_jobs: int = -1,
        backend: str = "multiprocessing",
        verbose: bool = True,
        config: Optional[ParallelConfig] = None,
    ):
        """
        Initialize the parallel processor.

        Args:
            n_jobs: Number of parallel jobs
            backend: Backend to use
            verbose: Whether to show progress
            config: Full configuration (overrides other parameters)
        """
        if config is None:
            self.config = ParallelConfig(
                n_jobs=n_jobs, backend=backend, verbose=verbose
            )
        else:
            self.config = config

        self._executor = None
        self._client = None

        logger.info(
            f"Initialized {self.config.backend.value} parallel processor "
            f"with {self.config.n_jobs} jobs"
        )

    def __enter__(self):
        """Context manager entry."""
        self._setup_backend()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self._cleanup_backend()

    def _setup_backend(self):
        """Setup the parallel backend."""
        if self.config.backend == BackendType.DASK:
            self._setup_dask()
        elif self.config.backend == BackendType.RAY:
            self._setup_ray()

    def _cleanup_backend(self):
        """Cleanup the parallel backend."""
        if self._executor is not None:
            self._executor.shutdown()

        if self._client is not None:
            if self.config.backend == BackendType.DASK:
                self._client.close()
            elif self.config.backend == BackendType.RAY:
                ray.shutdown()

    def _setup_dask(self):
        """Setup Dask client."""
        if not DASK_AVAILABLE:
            raise ImportError("Dask not installed")

        self._client = Client(
            n_workers=self.config.dask_n_workers or self.config.n_jobs,
            threads_per_worker=self.config.dask_threads_per_worker or 1,
            processes=self.config.dask_scheduler == "processes",
            dashboard_address=self.config.dask_dashboard_address,
        )

        logger.info(f"Dask dashboard available at: {self._client.dashboard_link}")

    def _setup_ray(self):
        """Setup Ray."""
        if not RAY_AVAILABLE:
            raise ImportError("Ray not installed")

        ray.init(
            address=self.config.ray_address,
            num_cpus=self.config.ray_num_cpus,
            num_gpus=self.config.ray_num_gpus,
            object_store_memory=self.config.ray_object_store_memory,
            ignore_reinit_error=True,
        )

    def map(
        self,
        func: Callable,
        iterable: Iterable,
        chunksize: Optional[int] = None,
        **kwargs,
    ) -> List[Any]:
        """
        Parallel map operation.

        Args:
            func: Function to apply
            iterable: Input iterable
            chunksize: Chunk size for processing
            **kwargs: Additional arguments

        Returns:
            List of results
        """
        chunksize = chunksize or self.config.chunk_size

        if self.config.backend == BackendType.MULTIPROCESSING:
            return self._map_multiprocessing(func, iterable, chunksize)
        elif self.config.backend == BackendType.THREADING:
            return self._map_threading(func, iterable, chunksize)
        elif self.config.backend == BackendType.DASK:
            return self._map_dask(func, iterable, chunksize)
        elif self.config.backend == BackendType.RAY:
            return self._map_ray(func, iterable, chunksize)
        elif self.config.backend == BackendType.JOBLIB:
            return self._map_joblib(func, iterable, chunksize)
        else:
            raise ValueError(f"Unsupported backend: {self.config.backend}")

    def _map_multiprocessing(
        self, func: Callable, iterable: Iterable, chunksize: Optional[int]
    ) -> List[Any]:
        """Map using multiprocessing."""
        ctx = mp.get_context(self.config.multiprocessing_context)

        with ctx.Pool(self.config.n_jobs) as pool:
            if self.config.show_progress:
                items = list(iterable)
                results = list(
                    tqdm(
                        pool.imap(func, items, chunksize=chunksize),
                        total=len(items),
                        desc="Processing",
                    )
                )
            else:
                results = pool.map(func, iterable, chunksize=chunksize)

        return results

    def _map_threading(
        self, func: Callable, iterable: Iterable, chunksize: Optional[int]
    ) -> List[Any]:
        """Map using threading."""
        with ThreadPoolExecutor(
            max_workers=self.config.n_jobs,
            thread_name_prefix=self.config.thread_name_prefix,
        ) as executor:
            if self.config.show_progress:
                items = list(iterable)
                futures = [executor.submit(func, item) for item in items]
                results = []

                for future in tqdm(
                    as_completed(futures), total=len(futures), desc="Processing"
                ):
                    results.append(future.result())
            else:
                results = list(executor.map(func, iterable))

        return results

    def _map_dask(
        self, func: Callable, iterable: Iterable, chunksize: Optional[int]
    ) -> List[Any]:
        """Map using Dask."""
        if not DASK_AVAILABLE:
            raise ImportError("Dask not installed")

        # Convert to Dask collection
        items = list(iterable)
        futures = self._client.map(func, items)

        if self.config.show_progress:
            results = []
            for future in tqdm(
                dask_as_completed(futures), total=len(futures), desc="Processing"
            ):
                results.append(future.result())
        else:
            results = self._client.gather(futures)

        return results

    def _map_ray(
        self, func: Callable, iterable: Iterable, chunksize: Optional[int]
    ) -> List[Any]:
        """Map using Ray."""
        if not RAY_AVAILABLE:
            raise ImportError("Ray not installed")

        # Create Ray remote function
        remote_func = ray.remote(func)

        # Submit tasks
        items = list(iterable)
        futures = [remote_func.remote(item) for item in items]

        # Get results
        if self.config.show_progress:
            results = []
            with tqdm(total=len(futures), desc="Processing") as pbar:
                while futures:
                    ready, futures = ray.wait(futures, num_returns=1)
                    results.extend(ray.get(ready))
                    pbar.update(1)
        else:
            results = ray.get(futures)

        return results

    def _map_joblib(
        self, func: Callable, iterable: Iterable, chunksize: Optional[int]
    ) -> List[Any]:
        """Map using joblib."""
        from joblib import Parallel, delayed

        if self.config.show_progress:
            results = Parallel(n_jobs=self.config.n_jobs, verbose=10)(
                delayed(func)(item) for item in iterable
            )
        else:
            results = Parallel(n_jobs=self.config.n_jobs)(
                delayed(func)(item) for item in iterable
            )

        return results

    def starmap(
        self, func: Callable, iterable: Iterable[Tuple], chunksize: Optional[int] = None
    ) -> List[Any]:
        """
        Parallel starmap operation.

        Args:
            func: Function to apply
            iterable: Iterable of argument tuples
            chunksize: Chunk size

        Returns:
            List of results
        """

        # Convert starmap to map
        def wrapper(args):
            return func(*args)

        return self.map(wrapper, iterable, chunksize)

    def apply_async(
        self, func: Callable, args: Tuple = (), kwargs: Dict[str, Any] = None
    ) -> Any:
        """
        Apply function asynchronously.

        Args:
            func: Function to apply
            args: Positional arguments
            kwargs: Keyword arguments

        Returns:
            Future/AsyncResult
        """
        kwargs = kwargs or {}

        if self.config.backend == BackendType.MULTIPROCESSING:
            ctx = mp.get_context(self.config.multiprocessing_context)
            with ctx.Pool(1) as pool:
                return pool.apply_async(func, args, kwargs)
        elif self.config.backend == BackendType.THREADING:
            executor = ThreadPoolExecutor(max_workers=1)
            return executor.submit(func, *args, **kwargs)
        elif self.config.backend == BackendType.DASK:
            return self._client.submit(func, *args, **kwargs)
        elif self.config.backend == BackendType.RAY:
            remote_func = ray.remote(func)
            return remote_func.remote(*args, **kwargs)
        else:
            # Fallback to synchronous
            return func(*args, **kwargs)

    def reduce(
        self, func: Callable, iterable: Iterable, initializer: Any = None
    ) -> Any:
        """
        Parallel reduce operation.

        Args:
            func: Reduce function
            iterable: Input iterable
            initializer: Initial value

        Returns:
            Reduced result
        """
        # Split into chunks
        items = list(iterable)
        chunk_size = max(1, len(items) // self.config.n_jobs)
        chunks = [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]

        # Reduce each chunk
        def chunk_reduce(chunk):
            if initializer is not None:
                return functools.reduce(func, chunk, initializer)
            else:
                return functools.reduce(func, chunk)

        # Parallel reduce on chunks
        chunk_results = self.map(chunk_reduce, chunks)

        # Final reduce
        if initializer is not None:
            return functools.reduce(func, chunk_results, initializer)
        else:
            return functools.reduce(func, chunk_results)


class ChunkProcessor:
    """
    Process large datasets in chunks with parallel execution.

    This class is optimized for processing datasets that don't fit in memory
    by processing them in smaller chunks.
    """

    def __init__(
        self,
        chunk_size: int = 10000,
        n_jobs: int = -1,
        backend: str = "multiprocessing",
        overlap: int = 0,
    ):
        """
        Initialize ChunkProcessor.

        Args:
            chunk_size: Size of each chunk
            n_jobs: Number of parallel jobs
            backend: Processing backend
            overlap: Number of overlapping rows between chunks
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.processor = ParallelProcessor(n_jobs=n_jobs, backend=backend)
        self.stats = {
            "chunks_processed": 0,
            "total_rows": 0,
            "processing_time": 0,
            "errors": [],
        }

    def process_file(
        self,
        filepath: str,
        func: Callable[[pd.DataFrame], pd.DataFrame],
        output_path: Optional[str] = None,
        file_format: str = "csv",
        **read_kwargs,
    ) -> Optional[pd.DataFrame]:
        """
        Process a large file in chunks.

        Args:
            filepath: Input file path
            func: Function to apply to each chunk
            output_path: Output file path (if None, returns DataFrame)
            file_format: File format
            **read_kwargs: Arguments for file reading

        Returns:
            Processed DataFrame if output_path is None
        """
        start_time = time.time()

        # Create chunk reader
        if file_format == "csv":
            reader = pd.read_csv(filepath, chunksize=self.chunk_size, **read_kwargs)
        elif file_format == "json":
            reader = pd.read_json(
                filepath, lines=True, chunksize=self.chunk_size, **read_kwargs
            )
        else:
            raise ValueError(f"Unsupported format for chunked reading: {file_format}")

        # Process chunks
        results = []
        temp_files = []

        try:
            for i, chunk in enumerate(reader):
                # Add overlap from previous chunk if needed
                if self.overlap > 0 and i > 0 and results:
                    prev_tail = results[-1].tail(self.overlap)
                    chunk = pd.concat([prev_tail, chunk], ignore_index=True)

                # Process chunk
                try:
                    processed = func(chunk)

                    if output_path:
                        # Save to temporary file
                        temp_file = f"{output_path}.tmp.{i}"
                        processed.to_csv(temp_file, index=False)
                        temp_files.append(temp_file)
                    else:
                        results.append(processed)

                    self.stats["chunks_processed"] += 1
                    self.stats["total_rows"] += len(chunk)

                except Exception as e:
                    logger.error(f"Error processing chunk {i}: {e}")
                    self.stats["errors"].append({"chunk": i, "error": str(e)})
                    if not self.processor.config.ignore_errors:
                        raise

            # Combine results
            if output_path:
                # Combine temporary files
                self._combine_temp_files(temp_files, output_path, file_format)
                return None
            else:
                return pd.concat(results, ignore_index=True)

        finally:
            # Cleanup temp files
            for temp_file in temp_files:
                if os.path.exists(temp_file):
                    os.remove(temp_file)

            self.stats["processing_time"] = time.time() - start_time

    def process_dataframe(
        self,
        df: pd.DataFrame,
        func: Callable[[pd.DataFrame], pd.DataFrame],
        combine_func: Optional[Callable] = None,
    ) -> pd.DataFrame:
        """
        Process DataFrame in chunks.

        Args:
            df: Input DataFrame
            func: Function to apply to each chunk
            combine_func: Function to combine results

        Returns:
            Processed DataFrame
        """
        n_chunks = (len(df) + self.chunk_size - 1) // self.chunk_size

        # Create chunks
        chunks = []
        for i in range(n_chunks):
            start = i * self.chunk_size - (self.overlap if i > 0 else 0)
            end = min((i + 1) * self.chunk_size, len(df))
            chunk = df.iloc[start:end]
            chunks.append(chunk)

        # Process chunks in parallel
        results = self.processor.map(func, chunks)

        # Combine results
        if combine_func:
            return combine_func(results)
        else:
            return pd.concat(results, ignore_index=True)

    def process_iterator(
        self,
        iterator: Iterable[pd.DataFrame],
        func: Callable[[pd.DataFrame], pd.DataFrame],
        buffer_size: int = 10,
    ) -> Iterable[pd.DataFrame]:
        """
        Process iterator of DataFrames with buffering.

        Args:
            iterator: Iterator of DataFrames
            func: Function to apply
            buffer_size: Number of chunks to buffer

        Yields:
            Processed DataFrames
        """
        buffer = []

        for chunk in iterator:
            buffer.append(chunk)

            if len(buffer) >= buffer_size:
                # Process buffer in parallel
                results = self.processor.map(func, buffer)

                for result in results:
                    yield result

                buffer = []

        # Process remaining buffer
        if buffer:
            results = self.processor.map(func, buffer)
            for result in results:
                yield result

    def _combine_temp_files(
        self, temp_files: List[str], output_path: str, file_format: str
    ):
        """Combine temporary files into final output."""
        if file_format == "csv":
            # Read and combine CSV files
            dfs = []
            for i, temp_file in enumerate(temp_files):
                df = pd.read_csv(temp_file)
                dfs.append(df)

            combined = pd.concat(dfs, ignore_index=True)
            combined.to_csv(output_path, index=False)

        elif file_format == "parquet":
            # For parquet, we can append
            for i, temp_file in enumerate(temp_files):
                df = pd.read_csv(temp_file)
                if i == 0:
                    df.to_parquet(output_path, index=False)
                else:
                    df.to_parquet(
                        output_path, index=False, engine="fastparquet", append=True
                    )

    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        stats = self.stats.copy()
        if stats["processing_time"] > 0:
            stats["rows_per_second"] = stats["total_rows"] / stats["processing_time"]
        return stats


class DistributedProcessor:
    """
    Distributed processing across multiple machines or clusters.

    Supports Dask and Ray for distributed computing.
    """

    def __init__(
        self,
        backend: str = "dask",
        cluster_address: Optional[str] = None,
        n_workers: Optional[int] = None,
        worker_memory: str = "4GB",
        dashboard_port: int = 8787,
    ):
        """
        Initialize DistributedProcessor.

        Args:
            backend: Distributed backend ('dask' or 'ray')
            cluster_address: Cluster address (None for local)
            n_workers: Number of workers
            worker_memory: Memory per worker
            dashboard_port: Dashboard port
        """
        self.backend = backend
        self.cluster_address = cluster_address
        self.n_workers = n_workers
        self.worker_memory = worker_memory
        self.dashboard_port = dashboard_port
        self.client = None

        self._setup_cluster()

    def _setup_cluster(self):
        """Setup distributed cluster."""
        if self.backend == "dask":
            self._setup_dask_cluster()
        elif self.backend == "ray":
            self._setup_ray_cluster()
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

    def _setup_dask_cluster(self):
        """Setup Dask cluster."""
        if not DASK_AVAILABLE:
            raise ImportError("Dask not installed")

        if self.cluster_address:
            # Connect to existing cluster
            self.client = Client(self.cluster_address)
        else:
            # Create local cluster
            from dask.distributed import LocalCluster

            cluster = LocalCluster(
                n_workers=self.n_workers,
                threads_per_worker=1,
                memory_limit=self.worker_memory,
                dashboard_address=f":{self.dashboard_port}",
            )
            self.client = Client(cluster)

        logger.info(f"Dask dashboard: {self.client.dashboard_link}")

    def _setup_ray_cluster(self):
        """Setup Ray cluster."""
        if not RAY_AVAILABLE:
            raise ImportError("Ray not installed")

        if self.cluster_address:
            ray.init(address=self.cluster_address)
        else:
            ray.init(num_cpus=self.n_workers, dashboard_port=self.dashboard_port)

    def process_dataframe(
        self,
        df: pd.DataFrame,
        func: Callable,
        partition_col: Optional[str] = None,
        n_partitions: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Process DataFrame in distributed fashion.

        Args:
            df: Input DataFrame
            func: Function to apply
            partition_col: Column to partition by
            n_partitions: Number of partitions

        Returns:
            Processed DataFrame
        """
        if self.backend == "dask":
            return self._process_dask_dataframe(df, func, partition_col, n_partitions)
        elif self.backend == "ray":
            return self._process_ray_dataframe(df, func, partition_col, n_partitions)

    def _process_dask_dataframe(
        self,
        df: pd.DataFrame,
        func: Callable,
        partition_col: Optional[str],
        n_partitions: Optional[int],
    ) -> pd.DataFrame:
        """Process using Dask."""
        # Convert to Dask DataFrame
        n_partitions = n_partitions or self.n_workers or mp.cpu_count()
        ddf = dd.from_pandas(df, npartitions=n_partitions)

        if partition_col:
            # Repartition by column
            ddf = ddf.set_index(partition_col)

        # Apply function
        result_ddf = ddf.map_partitions(func)

        # Compute and return
        return result_ddf.compute()

    def _process_ray_dataframe(
        self,
        df: pd.DataFrame,
        func: Callable,
        partition_col: Optional[str],
        n_partitions: Optional[int],
    ) -> pd.DataFrame:
        """Process using Ray."""
        n_partitions = n_partitions or self.n_workers or mp.cpu_count()

        # Split DataFrame
        if partition_col:
            groups = df.groupby(partition_col)
            partitions = [group for _, group in groups]
        else:
            chunk_size = len(df) // n_partitions
            partitions = [
                df.iloc[i : i + chunk_size] for i in range(0, len(df), chunk_size)
            ]

        # Create Ray tasks
        @ray.remote
        def process_partition(partition):
            return func(partition)

        # Process partitions
        futures = [process_partition.remote(part) for part in partitions]
        results = ray.get(futures)

        # Combine results
        return pd.concat(results, ignore_index=True)

    def process_files(
        self,
        file_pattern: str,
        func: Callable,
        output_dir: str,
        file_format: str = "parquet",
    ) -> None:
        """
        Process multiple files in distributed fashion.

        Args:
            file_pattern: Glob pattern for files
            func: Function to apply to each file
            output_dir: Output directory
            file_format: Output format
        """
        import glob

        files = glob.glob(file_pattern)
        logger.info(f"Processing {len(files)} files")

        if self.backend == "dask":
            self._process_files_dask(files, func, output_dir, file_format)
        elif self.backend == "ray":
            self._process_files_ray(files, func, output_dir, file_format)

    def _process_files_dask(
        self, files: List[str], func: Callable, output_dir: str, file_format: str
    ):
        """Process files using Dask."""

        @dask.delayed
        def process_file(filepath):
            df = pd.read_csv(filepath)
            result = func(df)

            filename = os.path.basename(filepath)
            output_path = os.path.join(output_dir, filename)

            if file_format == "parquet":
                result.to_parquet(output_path)
            else:
                result.to_csv(output_path, index=False)

            return output_path

        # Create delayed tasks
        tasks = [process_file(f) for f in files]

        # Execute
        results = dask.compute(*tasks)
        logger.info(f"Processed {len(results)} files")

    def _process_files_ray(
        self, files: List[str], func: Callable, output_dir: str, file_format: str
    ):
        """Process files using Ray."""

        @ray.remote
        def process_file(filepath):
            df = pd.read_csv(filepath)
            result = func(df)

            filename = os.path.basename(filepath)
            output_path = os.path.join(output_dir, filename)

            if file_format == "parquet":
                result.to_parquet(output_path)
            else:
                result.to_csv(output_path, index=False)

            return output_path

        # Create Ray tasks
        futures = [process_file.remote(f) for f in files]

        # Execute
        results = ray.get(futures)
        logger.info(f"Processed {len(results)} files")

    def map_reduce(
        self,
        data: Any,
        map_func: Callable,
        reduce_func: Callable,
        key_func: Optional[Callable] = None,
    ) -> Any:
        """
        Distributed map-reduce operation.

        Args:
            data: Input data
            map_func: Map function
            reduce_func: Reduce function
            key_func: Optional key extraction function

        Returns:
            Reduced result
        """
        if self.backend == "dask":
            return self._map_reduce_dask(data, map_func, reduce_func, key_func)
        elif self.backend == "ray":
            return self._map_reduce_ray(data, map_func, reduce_func, key_func)

    def _map_reduce_dask(
        self,
        data: Any,
        map_func: Callable,
        reduce_func: Callable,
        key_func: Optional[Callable],
    ) -> Any:
        """Map-reduce using Dask."""
        from dask import bag as db

        # Create Dask bag
        bag = db.from_sequence(data, partition_size=1000)

        # Map
        mapped = bag.map(map_func)

        # Group by key if provided
        if key_func:
            grouped = mapped.groupby(key_func)
            # Reduce each group
            result = grouped.map(lambda x: reduce_func(x[1])).compute()
        else:
            # Global reduce
            result = mapped.fold(reduce_func).compute()

        return result

    def _map_reduce_ray(
        self,
        data: Any,
        map_func: Callable,
        reduce_func: Callable,
        key_func: Optional[Callable],
    ) -> Any:
        """Map-reduce using Ray."""

        # Map phase
        @ray.remote
        def ray_map(item):
            return map_func(item)

        map_futures = [ray_map.remote(item) for item in data]
        mapped_results = ray.get(map_futures)

        # Group by key if provided
        if key_func:
            grouped = defaultdict(list)
            for result in mapped_results:
                key = key_func(result)
                grouped[key].append(result)

            # Reduce each group
            @ray.remote
            def ray_reduce_group(group):
                return reduce_func(group)

            reduce_futures = [
                ray_reduce_group.remote(group) for group in grouped.values()
            ]
            results = ray.get(reduce_futures)

            return dict(zip(grouped.keys(), results))
        else:
            # Global reduce
            @ray.remote
            def ray_reduce(items):
                return reduce_func(items)

            return ray.get(ray_reduce.remote(mapped_results))

    def shutdown(self):
        """Shutdown distributed cluster."""
        if self.backend == "dask" and self.client:
            self.client.close()
        elif self.backend == "ray":
            ray.shutdown()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.shutdown()


def parallel_apply(
    df: pd.DataFrame,
    func: Callable,
    axis: int = 0,
    n_jobs: int = -1,
    backend: str = "multiprocessing",
) -> pd.Series:
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


def parallel_map(
    func: Callable[..., T],
    *iterables: Iterable,
    n_jobs: int = -1,
    backend: str = "multiprocessing",
    chunk_size: Optional[int] = None,
    progress: bool = True,
    error_handler: Optional[Callable] = None,
    timeout: Optional[float] = None,
) -> List[T]:
    """
    Enhanced parallel map with multiple iterables and error handling.

    Args:
        func: Function to apply
        *iterables: Input iterables (like map)
        n_jobs: Number of parallel jobs
        backend: Backend to use
        chunk_size: Chunk size for processing
        progress: Show progress bar
        error_handler: Function to handle errors
        timeout: Timeout for each task

    Returns:
        List of results

    Example:
        >>> def add(x, y):
        ...     return x + y
        >>> results = parallel_map(add, [1, 2, 3], [4, 5, 6], n_jobs=2)
        >>> print(results)  # [5, 7, 9]
    """
    # Create configuration
    config = ParallelConfig(
        n_jobs=n_jobs,
        backend=backend,
        chunk_size=chunk_size,
        show_progress=progress,
        timeout=timeout,
    )

    # Handle multiple iterables
    if len(iterables) == 1:
        items = list(iterables[0])
    else:
        items = list(zip(*iterables))
        # Wrap function to handle tuple arguments
        original_func = func
        func = lambda args: original_func(*args)

    # Process with error handling
    if error_handler:

        def safe_func(item):
            try:
                return func(item)
            except Exception as e:
                return error_handler(item, e)

        process_func = safe_func
    else:
        process_func = func

    # Create processor and execute
    with ParallelProcessor(config=config) as processor:
        results = processor.map(process_func, items)

    return results


def parallel_groupby_apply(
    df: pd.DataFrame,
    groupby_cols: Union[str, List[str]],
    func: Callable,
    n_jobs: int = -1,
) -> pd.DataFrame:
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

    def process(
        self,
        data: Union[pd.DataFrame, np.ndarray],
        func: Callable,
        combine_func: Optional[Callable] = None,
        progress: bool = True,
    ) -> Any:
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
                results = list(
                    tqdm(
                        executor.map(func, batches),
                        total=len(batches),
                        desc="Processing batches",
                    )
                )
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


def chunked_parallel_process(
    func: Callable,
    data: List[Any],
    chunk_size: int = 100,
    n_jobs: int = -1,
    reducer: Optional[Callable] = None,
) -> Any:
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
    chunks = [data[i : i + chunk_size] for i in range(0, len(data), chunk_size)]

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
        self.size = int(np.prod(shape))

        # Create shared memory
        self.shared_array = mp.Array(self._get_ctypes_type(dtype), self.size)

    def _get_ctypes_type(self, dtype: np.dtype):
        """Get ctypes type for numpy dtype."""
        type_map = {
            np.float64: "d",
            np.float32: "f",
            np.int64: "l",
            np.int32: "i",
            np.uint8: "B",
        }
        return type_map.get(dtype.type, "d")

    def to_numpy(self) -> np.ndarray:
        """Convert to numpy array."""
        return np.frombuffer(self.shared_array.get_obj(), dtype=self.dtype).reshape(
            self.shape
        )

    def from_numpy(self, array: np.ndarray) -> None:
        """Copy from numpy array."""
        np.copyto(self.to_numpy(), array)


@contextmanager
def parallel_backend(backend: str = "multiprocessing", n_jobs: int = -1):
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
    old_backend = os.environ.get("PARALLEL_BACKEND", "multiprocessing")
    old_n_jobs = os.environ.get("PARALLEL_N_JOBS", "-1")

    # Set new settings
    os.environ["PARALLEL_BACKEND"] = backend
    os.environ["PARALLEL_N_JOBS"] = str(n_jobs)

    try:
        yield
    finally:
        # Restore old settings
        os.environ["PARALLEL_BACKEND"] = old_backend
        os.environ["PARALLEL_N_JOBS"] = old_n_jobs


def parallel_decorator(n_jobs: int = -1, backend: str = "multiprocessing"):
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
        self.counter = self.manager.Value("i", 0)
        self.total = self.manager.Value("i", 0)

    def _process_with_progress(self, args: Tuple[Callable, Any, Queue]) -> Any:
        """Process single item and update progress."""
        func, item, queue = args
        result = func(item)

        with self.counter.get_lock():
            self.counter.value += 1

        return result

    def map(
        self, func: Callable, items: List[Any], desc: str = "Processing"
    ) -> List[Any]:
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
def parallel_read_csv(
    file_paths: List[str], n_jobs: int = -1, **read_kwargs
) -> pd.DataFrame:
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


def parallel_save_partitions(
    df: pd.DataFrame,
    output_dir: str,
    partition_col: str,
    n_jobs: int = -1,
    file_format: str = "parquet",
) -> None:
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

        if file_format == "parquet":
            group.to_parquet(filepath)
        elif file_format == "csv":
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
        return x**2

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

    # Test parallel_map with multiple iterables
    print("\nTesting parallel_map...")

    def add(x, y):
        return x + y

    results = parallel_map(add, range(10), range(10, 20), n_jobs=4)
    print(f"  Results: {results[:5]}...")

    # Test ChunkProcessor
    print("\nTesting ChunkProcessor...")
    chunk_processor = ChunkProcessor(chunk_size=100, n_jobs=4)

    df = pd.DataFrame({"A": np.random.randn(1000), "B": np.random.randn(1000)})

    def process_chunk(chunk):
        return chunk.assign(C=chunk["A"] + chunk["B"])

    result_df = chunk_processor.process_dataframe(df, process_chunk)
    print(f"  Result shape: {result_df.shape}")

    # Test ParallelConfig
    print("\nTesting ParallelConfig...")
    config = ParallelConfig(
        n_jobs=4, backend=BackendType.MULTIPROCESSING, chunk_size=50, show_progress=True
    )

    with ParallelProcessor(config=config) as proc:
        results = proc.map(expensive_computation, data[:10])
        print(f"  Config-based results: {results[:5]}...")

    # Test DistributedProcessor (if Dask available)
    if DASK_AVAILABLE:
        print("\nTesting DistributedProcessor...")
        with DistributedProcessor(backend="dask", n_workers=2) as dist_proc:
            df_large = pd.DataFrame(
                {
                    "A": np.random.randn(10000),
                    "B": np.random.randn(10000),
                    "group": np.random.choice(["X", "Y", "Z"], 10000),
                }
            )

            def process_partition(part):
                return part.assign(mean_A=part["A"].mean())

            result = dist_proc.process_dataframe(df_large, process_partition)
            print(f"  Distributed result shape: {result.shape}")

    # Test parallel DataFrame operations
    print("\nTesting parallel DataFrame operations...")
    df = pd.DataFrame(
        {
            "A": np.random.randn(1000),
            "B": np.random.randn(1000),
            "group": np.random.choice(["X", "Y", "Z"], 1000),
        }
    )

    def process_group(group):
        return group["A"].mean() + group["B"].std()

    result = parallel_groupby_apply(df, "group", process_group, n_jobs=4)
    print(f"  Group results shape: {result.shape}")

    # Test decorator
    print("\nTesting parallel decorator...")

    @parallel_decorator(n_jobs=4)
    def process_item(x):
        return x**3

    decorated_results = process_item(data)
    print(f"  Decorated results: {decorated_results[:5]}...")

    print("\nAll tests completed!")


if __name__ == "__main__":
    example_parallel_processing()
