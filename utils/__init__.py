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
    save_object,
    load_object
)

from .parallel import (
    ParallelProcessor,
    ParallelConfig,
    ChunkProcessor,
    DistributedProcessor,
    parallel_apply,
    parallel_map
)

from .cli import (
    CLIParser,
    CLIApplication,
    CommandHandler,
    ArgumentValidator,
    create_cli_app,
    run_command
)

from .logging import (
    setup_logger,
    get_logger,
    LogConfig,
    ColoredFormatter,
    StructuredLogger,
    log_execution_time,
    log_memory_usage
)

__all__ = [
    # File I/O
    'FileHandler',
    'DataReader',
    'DataWriter',
    'ModelSerializer',
    'ConfigLoader',
    'save_object',
    'load_object',
    
    # Parallel processing
    'ParallelProcessor',
    'ParallelConfig',
    'ChunkProcessor',
    'DistributedProcessor',
    'parallel_apply',
    'parallel_map',
    
    # CLI utilities
    'CLIParser',
    'CLIApplication',
    'CommandHandler',
    'ArgumentValidator',
    'create_cli_app',
    'run_command',
    
    # Logging
    'setup_logger',
    'get_logger',
    'LogConfig',
    'ColoredFormatter',
    'StructuredLogger',
    'log_execution_time',
    'log_memory_usage'
]

__version__ = '1.0.0'