"""
utils/logging.py
Logging configuration and utilities for the data science toolkit.
"""

import functools
import json
import logging
import os
import sys
import time
import traceback
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import psutil


class LogFormat(Enum):
    """Log format options."""

    COLORED = "colored"
    SIMPLE = "simple"
    DETAILED = "detailed"
    JSON = "json"


@dataclass
class LogConfig:
    """
    Configuration for logging setup.
    """

    name: Optional[str] = None
    level: Union[str, int] = "INFO"
    log_file: Optional[str] = None
    log_dir: Optional[str] = None
    format: Union[str, LogFormat] = LogFormat.COLORED
    rotation: Optional[str] = None
    max_bytes: int = 10485760  # 10MB
    backup_count: int = 5
    console_output: bool = True
    file_output: bool = True
    structured_output: bool = False
    handlers: List[logging.Handler] = field(default_factory=list)

    def __post_init__(self):
        """Convert string format to enum if needed."""
        if isinstance(self.format, str):
            self.format = LogFormat(self.format.lower())


class ColoredFormatter(logging.Formatter):
    """
    Custom colored formatter for console output.
    """

    # ANSI escape codes for colors
    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[41m",  # Red background
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with colors."""
        # Add color to level name
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{levelname}{self.RESET}"

        # Format message
        message = super().format(record)

        return message


class StructuredFormatter(logging.Formatter):
    """
    Formatter that outputs structured JSON logs.
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add extra fields
        if hasattr(record, "extra_fields"):
            log_data.update(record.extra_fields)

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": traceback.format_exception(*record.exc_info),
            }

        return json.dumps(log_data)


class StructuredLogger:
    """
    Logger wrapper for structured logging with additional context.
    """

    def __init__(
        self, logger: logging.Logger, context: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize structured logger.

        Args:
            logger: Base logger instance
            context: Default context to add to all logs
        """
        self.logger = logger
        self.context = context or {}

    def _log(self, level: int, message: str, **kwargs):
        """Internal log method with context."""
        extra_fields = {**self.context, **kwargs}
        extra = {"extra_fields": extra_fields} if extra_fields else {}
        self.logger.log(level, message, extra=extra)

    def debug(self, message: str, **kwargs):
        """Log debug message with context."""
        self._log(logging.DEBUG, message, **kwargs)

    def info(self, message: str, **kwargs):
        """Log info message with context."""
        self._log(logging.INFO, message, **kwargs)

    def warning(self, message: str, **kwargs):
        """Log warning message with context."""
        self._log(logging.WARNING, message, **kwargs)

    def error(self, message: str, **kwargs):
        """Log error message with context."""
        self._log(logging.ERROR, message, **kwargs)

    def critical(self, message: str, **kwargs):
        """Log critical message with context."""
        self._log(logging.CRITICAL, message, **kwargs)

    def add_context(self, **kwargs):
        """Add permanent context to logger."""
        self.context.update(kwargs)

    def with_context(self, **kwargs) -> "StructuredLogger":
        """Create new logger with additional context."""
        new_context = {**self.context, **kwargs}
        return StructuredLogger(self.logger, new_context)


def setup_logger(
    name: str = None,
    level: Union[str, int] = "INFO",
    log_file: Optional[str] = None,
    log_dir: Optional[str] = None,
    format: str = "colored",
    rotation: Optional[str] = None,
    max_bytes: int = 10485760,  # 10MB
    backup_count: int = 5,
    config: Optional[LogConfig] = None,
) -> logging.Logger:
    """
    Setup logging configuration.

    Args:
        name: Logger name (None for root logger)
        level: Logging level
        log_file: Log file name
        log_dir: Directory for log files
        format: Log format ('colored', 'simple', 'detailed', 'json')
        rotation: Rotation type ('size', 'time', None)
        max_bytes: Max file size for rotation
        backup_count: Number of backup files
        config: LogConfig object (overrides other parameters)

    Returns:
        Configured logger
    """
    # Use config if provided
    if config:
        name = config.name
        level = config.level
        log_file = config.log_file
        log_dir = config.log_dir
        format = (
            config.format.value
            if isinstance(config.format, LogFormat)
            else config.format
        )
        rotation = config.rotation
        max_bytes = config.max_bytes
        backup_count = config.backup_count

    # Get logger
    logger = logging.getLogger(name)

    # Convert string level to int
    if isinstance(level, str):
        level = getattr(logging, level.upper())
    logger.setLevel(level)

    # Remove existing handlers
    logger.handlers.clear()

    # Console handler
    if not config or config.console_output:
        console_handler = logging.StreamHandler(sys.stdout)

        # Set formatter based on format type
        if format == "colored":
            formatter = ColoredFormatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        elif format == "simple":
            formatter = logging.Formatter("%(levelname)s - %(message)s")
        elif format == "detailed":
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - "
                "%(filename)s:%(lineno)d - %(funcName)s() - %(message)s"
            )
        elif format == "json":
            formatter = StructuredFormatter()
        else:
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )

        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File handler if specified
    if (not config or config.file_output) and (log_file or log_dir):
        if log_dir:
            log_dir = Path(log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            if not log_file:
                log_file = f"{name or 'app'}.log"
            file_path = log_dir / log_file
        else:
            file_path = Path(log_file)
            file_path.parent.mkdir(parents=True, exist_ok=True)

        # Choose handler based on rotation
        if rotation == "size":
            file_handler = RotatingFileHandler(
                file_path, maxBytes=max_bytes, backupCount=backup_count
            )
        elif rotation == "time":
            file_handler = TimedRotatingFileHandler(
                file_path, when="midnight", interval=1, backupCount=backup_count
            )
        else:
            file_handler = logging.FileHandler(file_path)

        # Use detailed formatter for file logs
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - "
            "%(filename)s:%(lineno)d - %(funcName)s() - %(message)s"
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    # Add any custom handlers from config
    if config and config.handlers:
        for handler in config.handlers:
            logger.addHandler(handler)

    return logger


def get_logger(name: str = None) -> logging.Logger:
    """
    Get a logger instance.

    Args:
        name: Logger name

    Returns:
        Logger instance
    """
    return logging.getLogger(name)


@contextmanager
def log_context(logger: logging.Logger, context: Dict[str, Any]):
    """
    Context manager that adds context to all logs within the block.

    Args:
        logger: Logger instance
        context: Context dictionary

    Usage:
        with log_context(logger, {'user_id': 123}):
            logger.info("User action")  # Will include user_id in log
    """

    class ContextFilter(logging.Filter):
        def filter(self, record):
            for key, value in context.items():
                setattr(record, key, value)
            return True

    filter = ContextFilter()
    logger.addFilter(filter)

    try:
        yield logger
    finally:
        logger.removeFilter(filter)


def log_execution_time(
    logger: logging.Logger = None,
    level: int = logging.INFO,
    message: str = "Execution time for {function}: {duration:.2f}s",
):
    """
    Decorator to log function execution time.

    Args:
        logger: Logger instance (uses function module logger if None)
        level: Logging level
        message: Log message template

    Usage:
        @log_execution_time()
        def my_function():
            time.sleep(1)
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal logger
            if logger is None:
                logger = logging.getLogger(func.__module__)

            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                logger.log(
                    level, message.format(function=func.__name__, duration=duration)
                )
                return result
            except Exception as e:
                duration = time.time() - start_time
                logger.error(
                    f"Function {func.__name__} failed after {duration:.2f}s: {str(e)}"
                )
                raise

        return wrapper

    return decorator


def log_memory_usage(
    logger: logging.Logger = None,
    level: int = logging.INFO,
    message: str = "Memory usage for {function}: {memory_mb:.2f}MB (delta: {delta_mb:+.2f}MB)",
):
    """
    Decorator to log function memory usage.

    Args:
        logger: Logger instance (uses function module logger if None)
        level: Logging level
        message: Log message template

    Usage:
        @log_memory_usage()
        def memory_intensive_function():
            data = [0] * 10000000
            return data
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal logger
            if logger is None:
                logger = logging.getLogger(func.__module__)

            # Get process
            process = psutil.Process(os.getpid())

            # Memory before execution
            mem_before = process.memory_info().rss / 1024 / 1024  # MB

            try:
                result = func(*args, **kwargs)

                # Memory after execution
                mem_after = process.memory_info().rss / 1024 / 1024  # MB
                delta = mem_after - mem_before

                logger.log(
                    level,
                    message.format(
                        function=func.__name__, memory_mb=mem_after, delta_mb=delta
                    ),
                )

                return result
            except Exception as e:
                mem_after = process.memory_info().rss / 1024 / 1024  # MB
                delta = mem_after - mem_before
                logger.error(
                    f"Function {func.__name__} failed. Memory usage: {mem_after:.2f}MB "
                    f"(delta: {delta:+.2f}MB). Error: {str(e)}"
                )
                raise

        return wrapper

    return decorator


def log_exceptions(
    logger: logging.Logger = None,
    level: int = logging.ERROR,
    message: str = "Exception in {function}",
):
    """
    Decorator to log exceptions.

    Args:
        logger: Logger instance
        level: Logging level
        message: Log message template

    Usage:
        @log_exceptions()
        def risky_function():
            raise ValueError("Something went wrong")
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal logger
            if logger is None:
                logger = logging.getLogger(func.__module__)

            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.log(level, message.format(function=func.__name__), exc_info=True)
                raise

        return wrapper

    return decorator


class ProgressLogger:
    """
    Logger for tracking progress of long-running operations.
    """

    def __init__(
        self,
        logger: logging.Logger,
        total: int,
        message: str = "Progress",
        log_interval: int = 10,
    ):
        """
        Initialize progress logger.

        Args:
            logger: Logger instance
            total: Total number of items
            message: Progress message
            log_interval: Log every N percent
        """
        self.logger = logger
        self.total = total
        self.message = message
        self.log_interval = log_interval
        self.current = 0
        self.last_logged_percent = 0
        self.start_time = time.time()

    def update(self, n: int = 1):
        """Update progress by n items."""
        self.current += n
        percent = (self.current / self.total) * 100

        # Log at intervals
        if percent - self.last_logged_percent >= self.log_interval:
            elapsed = time.time() - self.start_time
            rate = self.current / elapsed if elapsed > 0 else 0
            eta = (self.total - self.current) / rate if rate > 0 else 0

            self.logger.info(
                f"{self.message}: {percent:.1f}% ({self.current}/{self.total}) "
                f"- Rate: {rate:.1f} items/s - ETA: {eta:.1f}s"
            )
            self.last_logged_percent = percent

    def finish(self):
        """Log completion."""
        elapsed = time.time() - self.start_time
        rate = self.total / elapsed if elapsed > 0 else 0

        self.logger.info(
            f"{self.message}: Completed {self.total} items in {elapsed:.1f}s "
            f"({rate:.1f} items/s)"
        )


class ExperimentLogger:
    """
    Specialized logger for ML experiments.
    """

    def __init__(self, experiment_name: str, log_dir: str = "logs"):
        """
        Initialize experiment logger.

        Args:
            experiment_name: Name of the experiment
            log_dir: Directory for log files
        """
        self.experiment_name = experiment_name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Create experiment-specific directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = self.log_dir / f"{experiment_name}_{timestamp}"
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        # Setup loggers
        self._setup_loggers()

        # Metrics storage
        self.metrics = {}
        self.start_time = time.time()

    def _setup_loggers(self):
        """Setup experiment loggers."""
        # Main experiment logger
        self.logger = logging.getLogger(f"experiment.{self.experiment_name}")
        self.logger.setLevel(logging.DEBUG)

        # File handler for all logs
        all_logs_handler = logging.FileHandler(self.experiment_dir / "experiment.log")
        all_logs_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        self.logger.addHandler(all_logs_handler)

        # Separate handlers for different log levels
        # Errors only
        error_handler = logging.FileHandler(self.experiment_dir / "errors.log")
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(levelname)s - %(message)s - %(exc_info)s"
            )
        )
        self.logger.addHandler(error_handler)

        # Metrics logger (JSON format)
        self.metrics_logger = logging.getLogger(f"metrics.{self.experiment_name}")
        metrics_handler = logging.FileHandler(self.experiment_dir / "metrics.jsonl")
        metrics_handler.setFormatter(StructuredFormatter())
        self.metrics_logger.addHandler(metrics_handler)
        self.metrics_logger.setLevel(logging.INFO)

    def log_params(self, params: Dict[str, Any]):
        """
        Log experiment parameters.

        Args:
            params: Dictionary of parameters
        """
        self.logger.info("Experiment parameters:")
        for key, value in params.items():
            self.logger.info(f"  {key}: {value}")

        # Save params to file
        with open(self.experiment_dir / "params.json", "w") as f:
            json.dump(params, f, indent=2)

    def log_metric(self, name: str, value: float, step: Optional[int] = None):
        """
        Log a metric value.

        Args:
            name: Metric name
            value: Metric value
            step: Optional step/epoch number
        """
        # Store in memory
        if name not in self.metrics:
            self.metrics[name] = []

        entry = {"value": value, "timestamp": time.time()}
        if step is not None:
            entry["step"] = step

        self.metrics[name].append(entry)

        # Log to file
        log_entry = {
            "metric": name,
            "value": value,
            "step": step,
            "elapsed_time": time.time() - self.start_time,
        }
        self.metrics_logger.info("metric", extra={"extra_fields": log_entry})

    def log_artifact(self, artifact_path: str, artifact_type: str = "file"):
        """
        Log an artifact (file, model, etc).

        Args:
            artifact_path: Path to the artifact
            artifact_type: Type of artifact
        """
        artifact_path = Path(artifact_path)

        # Copy to experiment directory
        import shutil

        dest_path = self.experiment_dir / "artifacts" / artifact_path.name
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        if artifact_path.is_file():
            shutil.copy2(artifact_path, dest_path)
        else:
            shutil.copytree(artifact_path, dest_path)

        self.logger.info(f"Logged {artifact_type} artifact: {artifact_path.name}")

    def get_summary(self) -> Dict[str, Any]:
        """
        Get experiment summary.

        Returns:
            Summary dictionary
        """
        summary = {
            "experiment_name": self.experiment_name,
            "duration": time.time() - self.start_time,
            "log_dir": str(self.experiment_dir),
            "metrics_summary": {},
        }

        # Summarize metrics
        for metric_name, values in self.metrics.items():
            metric_values = [v["value"] for v in values]
            summary["metrics_summary"][metric_name] = {
                "final": metric_values[-1] if metric_values else None,
                "best": max(metric_values) if metric_values else None,
                "mean": (
                    sum(metric_values) / len(metric_values) if metric_values else None
                ),
            }

        return summary

    def save_summary(self):
        """Save experiment summary to file."""
        summary = self.get_summary()
        with open(self.experiment_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)


def redirect_warnings(logger: logging.Logger = None):
    """
    Redirect Python warnings to logger.

    Args:
        logger: Logger instance (uses 'warnings' logger if None)
    """
    if logger is None:
        logger = logging.getLogger("warnings")

    def warning_handler(message, category, filename, lineno, file=None, line=None):
        logger.warning(f"{filename}:{lineno}: {category.__name__}: {message}")

    warnings.showwarning = warning_handler


# Example usage
def example_logging():
    """Example of using logging utilities."""

    # Setup basic logging with LogConfig
    config = LogConfig(
        name="example",
        level="DEBUG",
        log_dir="logs",
        format=LogFormat.COLORED,
        rotation="size",
    )
    logger = setup_logger(config=config)

    logger.info("Starting example")

    # Test different log levels
    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")

    # Test structured logger
    structured = StructuredLogger(logger)
    structured.info("User login", user_id=123, session="abc123")

    # Test execution time logging
    @log_execution_time(logger)
    def slow_function():
        import time

        time.sleep(1)
        return "Done"

    result = slow_function()
    logger.info(f"Result: {result}")

    # Test memory usage logging
    @log_memory_usage(logger)
    def memory_intensive():
        data = [0] * 10000000
        return len(data)

    size = memory_intensive()
    logger.info(f"Created list of size: {size}")

    # Test exception logging
    @log_exceptions(logger)
    def risky_function():
        raise ValueError("Test exception")

    try:
        risky_function()
    except ValueError:
        pass  # Expected

    # Test progress logging
    progress = ProgressLogger(logger, total=100, message="Processing items")
    for i in range(100):
        # Simulate work
        import time

        time.sleep(0.01)
        progress.update()
    progress.finish()

    # Test experiment logger
    print("\nTesting ExperimentLogger...")
    exp_logger = ExperimentLogger("test_experiment")

    # Log parameters
    exp_logger.log_params(
        {"model": "random_forest", "n_estimators": 100, "max_depth": 10}
    )

    # Log metrics
    for epoch in range(10):
        exp_logger.log_metric("loss", 1.0 / (epoch + 1), step=epoch)
        exp_logger.log_metric("accuracy", 0.8 + 0.02 * epoch, step=epoch)

    # Save summary
    exp_logger.save_summary()
    summary = exp_logger.get_summary()
    print(f"Experiment summary: {summary}")

    # Test context logging
    with log_context(logger, {"user_id": 123, "session": "abc"}):
        logger.info("User performed action")


if __name__ == "__main__":
    example_logging()
