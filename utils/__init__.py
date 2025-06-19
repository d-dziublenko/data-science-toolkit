"""
utils/__init__.py
Utility functions and helpers for the Universal Data Science Toolkit.
"""

from .file_io import (
    FileHandler,
    DataReader,
    DataWriter,
    ModelSerializer,
    ConfigLoader,
    DataCache,
    CheckpointManager,
    save_object,
    load_object,
    save_config,
    load_config,
    create_archive
)

from .parallel import (
    BackendType,
    ParallelProcessor,
    ParallelConfig,
    ParallelBatch,
    ProgressParallel,
    ChunkProcessor,
    DistributedProcessor,
    SharedMemoryArray,
    get_n_jobs,
    parallel_apply,
    parallel_map,
    parallel_groupby_apply,
    chunked_parallel_process,
    parallel_read_csv,
    parallel_save_partitions
)

from .cli import (
    ColoredFormatter,
    CLIParser,
    CLIApplication,
    CommandHandler,
    ArgumentValidator,
    ConfigArgumentParser,
    ProgressBar,
    create_cli_app,
    run_command,
    spinner,
    prompt_choice,
    print_table,
    print_summary,
    create_cli_command,
    format_bytes,
    format_duration,
    print_error,
    print_warning,
    print_success,
    print_info
)

from .logging import (
    LogFormat,
    LogConfig,
    ColoredFormatter,
    StructuredFormatter,
    StructuredLogger,
    ProgressLogger,
    ExperimentLogger,
    log_execution_time,
    log_memory_usage,
    log_context,
    log_exceptions,
    redirect_warnings,
    setup_logger,
    get_logger
)

__all__ = [
    # File I/O
    'FileHandler',
    'DataReader',
    'DataWriter',
    'ModelSerializer',
    'ConfigLoader',
    'DataCache',
    'CheckpointManager',
    'save_object',
    'load_object',
    'save_config',
    'load_config',
    'create_archive',
    
    
    # Parallel processing
    'BackendType',
    'ParallelProcessor',
    'ParallelConfig',
    'ParallelBatch',
    'ProgressParallel',
    'ChunkProcessor',
    'DistributedProcessor',
    'SharedMemoryArray',
    'get_n_jobs',
    'parallel_apply',
    'parallel_map',
    'parallel_groupby_apply',
    'chunked_parallel_process',
    'parallel_read_csv',
    'parallel_save_partitions',
    
    # CLI utilities
    'ColoredFormatter',
    'CLIParser',
    'CLIApplication',
    'CommandHandler',
    'ArgumentValidator',
    'ConfigArgumentParser',
    'ProgressBar',
    'create_cli_app',
    'run_command',
    'spinner',
    'prompt_choice',
    'print_table',
    'print_summary',
    'create_cli_command',
    'format_bytes',
    'format_duration',
    'print_error',
    'print_warning',
    'print_success',
    'print_info',
    
    # Logging
    'LogFormat',
    'LogConfig',
    'ColoredFormatter',
    'StructuredFormatter',
    'StructuredLogger',
    'ProgressLogger',
    'ExperimentLogger',
    'log_execution_time',
    'log_memory_usage',
    'log_context',
    'log_exceptions',
    'redirect_warnings',
    'setup_logger',
    'get_logger'
]

__version__ = '1.0.0'