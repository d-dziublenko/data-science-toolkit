"""
utils/cli.py
Command-line interface helpers for the data science toolkit.
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type, Union

import click
import colorama
import yaml
from colorama import Back, Fore, Style

# Initialize colorama for cross-platform colored output
colorama.init(autoreset=True)

logger = logging.getLogger(__name__)


class ColoredFormatter(logging.Formatter):
    """
    Colored formatter for console logging.
    """

    COLORS = {
        "DEBUG": Fore.CYAN,
        "INFO": Fore.GREEN,
        "WARNING": Fore.YELLOW,
        "ERROR": Fore.RED,
        "CRITICAL": Fore.RED + Style.BRIGHT,
    }

    def format(self, record):
        # Add color to level name
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{levelname}{Style.RESET_ALL}"

        # Format message
        message = super().format(record)

        return message


def setup_logging(
    level: str = "INFO", log_file: Optional[str] = None, colored: bool = True
) -> None:
    """
    Setup logging configuration.

    Args:
        level: Logging level
        log_file: Optional log file path
        colored: Whether to use colored output
    """
    # Root logger configuration
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)

    if colored:
        formatter = ColoredFormatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    else:
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        root_logger.addHandler(file_handler)


class CLIParser:
    """
    Enhanced argument parser with validation and config file support.
    """

    def __init__(self, prog: str = None, description: str = None):
        """
        Initialize CLI parser.

        Args:
            prog: Program name
            description: Program description
        """
        self.parser = argparse.ArgumentParser(prog=prog, description=description)
        self.validators = {}
        self._setup_default_args()

    def _setup_default_args(self):
        """Setup default arguments."""
        self.parser.add_argument(
            "--config", "-c", type=str, help="Configuration file (JSON/YAML)"
        )
        self.parser.add_argument(
            "--verbose", "-v", action="store_true", help="Enable verbose output"
        )
        self.parser.add_argument(
            "--quiet",
            "-q",
            action="store_true",
            help="Suppress all output except errors",
        )

    def add_argument(self, *args, validator: Callable = None, **kwargs):
        """
        Add argument with optional validator.

        Args:
            *args: Positional arguments for argparse
            validator: Optional validation function
            **kwargs: Keyword arguments for argparse
        """
        action = self.parser.add_argument(*args, **kwargs)
        if validator:
            self.validators[action.dest] = validator
        return action

    def parse_args(self, args: Optional[List[str]] = None) -> argparse.Namespace:
        """
        Parse and validate arguments.

        Args:
            args: Optional argument list

        Returns:
            Parsed arguments
        """
        parsed_args = self.parser.parse_args(args)

        # Load config if specified
        if hasattr(parsed_args, "config") and parsed_args.config:
            self._load_config(parsed_args)

        # Validate arguments
        self._validate_args(parsed_args)

        return parsed_args

    def _load_config(self, args: argparse.Namespace):
        """Load configuration from file."""
        config_path = Path(args.config)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path) as f:
            if config_path.suffix in [".yaml", ".yml"]:
                config = yaml.safe_load(f)
            elif config_path.suffix == ".json":
                config = json.load(f)
            else:
                raise ValueError(f"Unsupported config format: {config_path.suffix}")

        # Update args with config values
        for key, value in config.items():
            if hasattr(args, key) and getattr(args, key) is None:
                setattr(args, key, value)

    def _validate_args(self, args: argparse.Namespace):
        """Validate parsed arguments."""
        for dest, validator in self.validators.items():
            if hasattr(args, dest):
                value = getattr(args, dest)
                if not validator(value):
                    raise ValueError(f"Validation failed for {dest}: {value}")


class CommandHandler:
    """
    Handler for CLI commands with subcommands support.
    """

    def __init__(self, name: str, description: str = None):
        """
        Initialize command handler.

        Args:
            name: Command name
            description: Command description
        """
        self.name = name
        self.description = description
        self.subcommands = {}
        self.parser = argparse.ArgumentParser(prog=name, description=description)
        self.subparsers = self.parser.add_subparsers(
            dest="command", help="Available commands"
        )

    def add_command(self, name: str, handler: Callable, description: str = None):
        """
        Add a subcommand.

        Args:
            name: Subcommand name
            handler: Function to handle the command
            description: Command description
        """
        # Create subparser
        subparser = self.subparsers.add_parser(name, help=description)

        # Extract arguments from handler signature
        import inspect

        sig = inspect.signature(handler)
        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue

            # Determine argument type
            arg_type = str
            if param.annotation != inspect.Parameter.empty:
                arg_type = param.annotation

            # Add argument
            if param.default == inspect.Parameter.empty:
                subparser.add_argument(param_name, type=arg_type)
            else:
                subparser.add_argument(
                    f"--{param_name}", type=arg_type, default=param.default
                )

        self.subcommands[name] = handler

    def execute(self, args: Optional[List[str]] = None):
        """
        Execute command based on arguments.

        Args:
            args: Optional argument list
        """
        parsed_args = self.parser.parse_args(args)

        if not parsed_args.command:
            self.parser.print_help()
            return

        # Get handler
        handler = self.subcommands.get(parsed_args.command)
        if not handler:
            print_error(f"Unknown command: {parsed_args.command}")
            return

        # Prepare kwargs
        kwargs = vars(parsed_args)
        kwargs.pop("command")

        # Execute handler
        try:
            handler(**kwargs)
        except Exception as e:
            print_error(f"Command failed: {e}")
            if parsed_args.verbose if hasattr(parsed_args, "verbose") else False:
                import traceback

                traceback.print_exc()


class ArgumentValidator:
    """
    Collection of argument validators.
    """

    @staticmethod
    def file_exists(path: str) -> bool:
        """Validate that file exists."""
        return Path(path).exists()

    @staticmethod
    def dir_exists(path: str) -> bool:
        """Validate that directory exists."""
        return Path(path).is_dir()

    @staticmethod
    def positive_int(value: Union[str, int]) -> bool:
        """Validate positive integer."""
        try:
            return int(value) > 0
        except (ValueError, TypeError):
            return False

    @staticmethod
    def positive_float(value: Union[str, float]) -> bool:
        """Validate positive float."""
        try:
            return float(value) > 0
        except (ValueError, TypeError):
            return False

    @staticmethod
    def in_range(min_val: float, max_val: float) -> Callable:
        """Create range validator."""

        def validator(value: Union[str, float]) -> bool:
            try:
                val = float(value)
                return min_val <= val <= max_val
            except (ValueError, TypeError):
                return False

        return validator

    @staticmethod
    def choices(valid_choices: List[Any]) -> Callable:
        """Create choices validator."""

        def validator(value: Any) -> bool:
            return value in valid_choices

        return validator

    @staticmethod
    def regex(pattern: str) -> Callable:
        """Create regex validator."""
        import re

        compiled = re.compile(pattern)

        def validator(value: str) -> bool:
            return bool(compiled.match(str(value)))

        return validator


def create_cli_app(
    name: str, version: str = "1.0.0", description: str = None
) -> "CLIApplication":
    """
    Factory function to create CLI application.

    Args:
        name: Application name
        version: Application version
        description: Application description

    Returns:
        CLIApplication instance
    """
    app = CLIApplication(name, version)
    if description:
        app.description = description
    return app


def run_command(
    command: Callable, args: Optional[List[str]] = None, catch_errors: bool = True
) -> int:
    """
    Run a command function with error handling.

    Args:
        command: Command function to run
        args: Optional arguments
        catch_errors: Whether to catch and handle errors

    Returns:
        Exit code (0 for success, non-zero for error)
    """
    try:
        # Parse function signature
        import inspect

        sig = inspect.signature(command)

        # Create parser
        parser = argparse.ArgumentParser(description=command.__doc__)

        # Add arguments from signature
        for param_name, param in sig.parameters.items():
            arg_type = str
            if param.annotation != inspect.Parameter.empty:
                arg_type = param.annotation

            if param.default == inspect.Parameter.empty:
                parser.add_argument(param_name, type=arg_type)
            else:
                parser.add_argument(
                    f"--{param_name}", type=arg_type, default=param.default
                )

        # Parse arguments
        parsed_args = parser.parse_args(args)

        # Execute command
        result = command(**vars(parsed_args))

        return 0 if result is None else result

    except Exception as e:
        if catch_errors:
            print_error(str(e))
            return 1
        else:
            raise


class ConfigArgumentParser:
    """
    Argument parser that supports loading from config files.
    """

    def __init__(self, description: str = None):
        """
        Initialize the parser.

        Args:
            description: Program description
        """
        self.parser = argparse.ArgumentParser(description=description)
        self.parser.add_argument(
            "--config", "-c", type=str, help="Path to configuration file (YAML or JSON)"
        )
        self._args = {}

    def add_argument(self, *args, **kwargs):
        """Add argument to parser."""
        self.parser.add_argument(*args, **kwargs)

    def parse_args(self, args: Optional[List[str]] = None) -> argparse.Namespace:
        """
        Parse arguments from command line and config file.

        Args:
            args: Optional argument list

        Returns:
            Parsed arguments
        """
        # First parse to get config file
        parsed_args, _ = self.parser.parse_known_args(args)

        # Load config if specified
        if parsed_args.config:
            config = self._load_config(parsed_args.config)

            # Set config values as defaults
            for key, value in config.items():
                if hasattr(parsed_args, key):
                    setattr(parsed_args, key, value)

        # Parse again with all arguments
        return self.parser.parse_args(args)

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from file."""
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        if config_path.suffix in [".yaml", ".yml"]:
            with open(config_path, "r") as f:
                return yaml.safe_load(f)
        elif config_path.suffix == ".json":
            with open(config_path, "r") as f:
                return json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {config_path.suffix}")


class ProgressBar:
    """
    Simple progress bar for CLI operations.
    """

    def __init__(
        self,
        total: int,
        description: str = "Progress",
        width: int = 50,
        style: str = "bar",
    ):
        """
        Initialize progress bar.

        Args:
            total: Total number of items
            description: Progress bar description
            width: Bar width
            style: Progress bar style ('bar', 'dots', 'spinner')
        """
        self.total = total
        self.description = description
        self.width = width
        self.style = style
        self.current = 0
        self._spinner_chars = "|/-\\"
        self._spinner_idx = 0

    def update(self, n: int = 1):
        """Update progress by n steps."""
        self.current += n
        self._render()

    def _render(self):
        """Render progress bar."""
        if self.style == "bar":
            progress = self.current / self.total
            filled = int(progress * self.width)
            bar = "█" * filled + "░" * (self.width - filled)
            percent = progress * 100
            print(f"\r{self.description}: [{bar}] {percent:.1f}%", end="", flush=True)
        elif self.style == "dots":
            dots = "." * (self.current % 4)
            print(f"\r{self.description}{dots:<4}", end="", flush=True)
        elif self.style == "spinner":
            self._spinner_idx = (self._spinner_idx + 1) % len(self._spinner_chars)
            spinner = self._spinner_chars[self._spinner_idx]
            percent = (self.current / self.total) * 100
            print(f"\r{spinner} {self.description}: {percent:.1f}%", end="", flush=True)

    def close(self):
        """Close progress bar."""
        if self.style == "bar":
            print(f'\r{self.description}: [{"█" * self.width}] 100.0%')
        else:
            print(f"\r✓ {self.description}")


class CLIApplication:
    """
    Base class for CLI applications.
    """

    def __init__(self, name: str, version: str = "1.0.0"):
        """
        Initialize CLI application.

        Args:
            name: Application name
            version: Application version
        """
        self.name = name
        self.version = version
        self.commands = {}
        self.description = None

    def command(self, name: str = None):
        """
        Decorator to register commands.

        Args:
            name: Command name
        """

        def decorator(func):
            cmd_name = name or func.__name__
            self.commands[cmd_name] = func
            return func

        return decorator

    def run(self, args: Optional[List[str]] = None):
        """
        Run the application.

        Args:
            args: Command line arguments
        """
        parser = argparse.ArgumentParser(description=f"{self.name} v{self.version}")

        parser.add_argument(
            "--version", "-v", action="version", version=f"{self.name} {self.version}"
        )

        subparsers = parser.add_subparsers(dest="command", help="Available commands")

        # Add commands
        for cmd_name, cmd_func in self.commands.items():
            cmd_parser = subparsers.add_parser(cmd_name, help=cmd_func.__doc__)

            # Add command-specific arguments from function signature
            import inspect

            sig = inspect.signature(cmd_func)

            for param_name, param in sig.parameters.items():
                if param_name == "self":
                    continue

                # Determine argument type
                if param.annotation != param.empty:
                    arg_type = param.annotation
                else:
                    arg_type = str

                # Add argument
                if param.default == param.empty:
                    # Required argument
                    cmd_parser.add_argument(param_name, type=arg_type)
                else:
                    # Optional argument
                    cmd_parser.add_argument(
                        f"--{param_name}", type=arg_type, default=param.default
                    )

        # Parse arguments
        parsed_args = parser.parse_args(args)

        # Execute command
        if parsed_args.command:
            cmd_func = self.commands[parsed_args.command]

            # Get function arguments
            sig = inspect.signature(cmd_func)
            kwargs = {}

            for param_name in sig.parameters:
                if param_name != "self" and hasattr(parsed_args, param_name):
                    kwargs[param_name] = getattr(parsed_args, param_name)

            # Run command
            try:
                result = cmd_func(**kwargs)
                if result is not None:
                    print(result)
            except Exception as e:
                print(f"{Fore.RED}Error: {str(e)}{Style.RESET_ALL}")
                sys.exit(1)
        else:
            parser.print_help()


def spinner(text: str = "Processing..."):
    """
    Context manager for spinner animation.

    Args:
        text: Spinner text

    Usage:
        with spinner("Loading data..."):
            # Long running operation
            time.sleep(5)
    """
    import itertools
    import threading
    import time

    class Spinner:
        def __init__(self, text):
            self.text = text
            self.spinner = itertools.cycle(["|", "/", "-", "\\"])
            self.stop_spinning = False
            self.thread = None

        def spin(self):
            while not self.stop_spinning:
                print(f"\r{next(self.spinner)} {self.text}", end="", flush=True)
                time.sleep(0.1)

        def __enter__(self):
            self.thread = threading.Thread(target=self.spin)
            self.thread.start()
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.stop_spinning = True
            self.thread.join()
            print(f"\r✓ {self.text}")

    return Spinner(text)


def confirm(prompt: str = "Continue?", default: bool = True) -> bool:
    """
    Ask for user confirmation.

    Args:
        prompt: Confirmation prompt
        default: Default value if user presses enter

    Returns:
        User's choice
    """
    if default:
        prompt += " [Y/n]: "
    else:
        prompt += " [y/N]: "

    while True:
        choice = input(prompt).strip().lower()

        if not choice:
            return default
        elif choice in ["y", "yes"]:
            return True
        elif choice in ["n", "no"]:
            return False
        else:
            print("Please enter 'y' or 'n'")


def prompt_choice(options: List[str], prompt: str = "Select an option:") -> str:
    """
    Prompt user to select from options.

    Args:
        options: List of options
        prompt: Selection prompt

    Returns:
        Selected option
    """
    print(f"\n{prompt}")
    for i, option in enumerate(options, 1):
        print(f"  {i}. {option}")

    while True:
        try:
            choice = input("\nEnter your choice (number): ").strip()
            idx = int(choice) - 1

            if 0 <= idx < len(options):
                return options[idx]
            else:
                print(f"Please enter a number between 1 and {len(options)}")
        except ValueError:
            print("Please enter a valid number")


def print_table(
    data: List[Dict[str, Any]],
    headers: Optional[List[str]] = None,
    max_width: Optional[int] = None,
) -> None:
    """
    Print data as a formatted table.

    Args:
        data: List of dictionaries
        headers: Optional custom headers
        max_width: Maximum column width
    """
    if not data:
        print("No data to display")
        return

    # Get headers
    if headers is None:
        headers = list(data[0].keys())

    # Calculate column widths
    widths = {}
    for header in headers:
        max_len = len(str(header))
        for row in data:
            value = str(row.get(header, ""))
            if max_width and len(value) > max_width:
                value = value[: max_width - 3] + "..."
            max_len = max(max_len, len(value))
        widths[header] = max_len

    # Print header
    header_line = " | ".join(f"{h:<{widths[h]}}" for h in headers)
    print(header_line)
    print("-" * len(header_line))

    # Print data
    for row in data:
        values = []
        for header in headers:
            value = str(row.get(header, ""))
            if max_width and len(value) > max_width:
                value = value[: max_width - 3] + "..."
            values.append(f"{value:<{widths[header]}}")
        print(" | ".join(values))


def print_summary(title: str, data: Dict[str, Any], color: str = "GREEN") -> None:
    """
    Print a formatted summary.

    Args:
        title: Summary title
        data: Summary data
        color: Title color
    """
    color_code = getattr(Fore, color.upper(), Fore.GREEN)

    print(f"\n{color_code}{'=' * 50}")
    print(f"{title:^50}")
    print(f"{'=' * 50}{Style.RESET_ALL}")

    for key, value in data.items():
        print(f"{key:.<30} {value}")

    print(f"{color_code}{'=' * 50}{Style.RESET_ALL}\n")


def create_cli_command(func: Callable) -> click.Command:
    """
    Convert a function to a Click command.

    Args:
        func: Function to convert

    Returns:
        Click command
    """
    import inspect

    # Get function signature
    sig = inspect.signature(func)

    # Create Click command
    cmd = click.Command(name=func.__name__, callback=func, help=func.__doc__)

    # Add parameters
    for param_name, param in sig.parameters.items():
        # Determine parameter type
        if param.annotation != param.empty:
            param_type = param.annotation
        else:
            param_type = str

        # Create Click option
        if param.default == param.empty:
            # Required parameter
            option = click.Option([f"--{param_name}"], required=True, type=param_type)
        else:
            # Optional parameter
            option = click.Option(
                [f"--{param_name}"], default=param.default, type=param_type
            )

        cmd.params.append(option)

    return cmd


# Utility functions for common CLI tasks
def format_bytes(size: int) -> str:
    """
    Format bytes to human readable string.

    Args:
        size: Size in bytes

    Returns:
        Formatted string
    """
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} PB"


def format_duration(seconds: float) -> str:
    """
    Format duration to human readable string.

    Args:
        seconds: Duration in seconds

    Returns:
        Formatted string
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}h"


def print_error(message: str, exit_code: int = 1) -> None:
    """
    Print error message and optionally exit.

    Args:
        message: Error message
        exit_code: Exit code (0 to not exit)
    """
    print(f"{Fore.RED}Error: {message}{Style.RESET_ALL}", file=sys.stderr)
    if exit_code:
        sys.exit(exit_code)


def print_warning(message: str) -> None:
    """
    Print warning message.

    Args:
        message: Warning message
    """
    print(f"{Fore.YELLOW}Warning: {message}{Style.RESET_ALL}")


def print_success(message: str) -> None:
    """
    Print success message.

    Args:
        message: Success message
    """
    print(f"{Fore.GREEN}✓ {message}{Style.RESET_ALL}")


def print_info(message: str) -> None:
    """
    Print info message.

    Args:
        message: Info message
    """
    print(f"{Fore.CYAN}ℹ {message}{Style.RESET_ALL}")


# Example CLI application
def example_cli_app():
    """Example of using CLI utilities."""

    # Create application
    app = CLIApplication("DataScienceToolkit", "1.0.0")

    @app.command()
    def train(dataset: str, model: str = "random_forest", epochs: int = 100):
        """Train a machine learning model."""
        print_info(f"Training {model} on {dataset}")

        # Simulate training with progress bar
        progress = ProgressBar(epochs, "Training")
        for i in range(epochs):
            # Simulate work
            import time

            time.sleep(0.01)
            progress.update()
        progress.close()

        print_success("Training completed!")

        # Print summary
        print_summary(
            "Training Results",
            {
                "Model": model,
                "Dataset": dataset,
                "Epochs": epochs,
                "Accuracy": "0.95",
                "Loss": "0.12",
            },
        )

    @app.command()
    def evaluate(model_path: str, test_data: str):
        """Evaluate a trained model."""
        with spinner("Loading model..."):
            import time

            time.sleep(2)

        print_info(f"Evaluating model: {model_path}")

        # Simulate results
        results = [
            {"Metric": "Accuracy", "Value": 0.95, "Std": 0.02},
            {"Metric": "Precision", "Value": 0.93, "Std": 0.03},
            {"Metric": "Recall", "Value": 0.96, "Std": 0.01},
            {"Metric": "F1-Score", "Value": 0.94, "Std": 0.02},
        ]

        print("\nEvaluation Results:")
        print_table(results)

    # Run example commands
    print("Example CLI Application Demo\n")

    # Simulate train command
    app.commands["train"](dataset="data.csv", model="xgboost", epochs=50)

    print("\n" + "=" * 50 + "\n")

    # Simulate evaluate command
    app.commands["evaluate"](model_path="model.pkl", test_data="test.csv")


if __name__ == "__main__":
    example_cli_app()
