"""
pipelines/experiment.py
Experiment tracking and management for machine learning workflows.
"""

import hashlib
import json
import logging
import os
import pickle
import shutil
import sqlite3
import tempfile
import threading
import warnings
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml

logger = logging.getLogger(__name__)


class ExperimentStatus(Enum):
    """Status of an experiment."""

    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ExperimentConfig:
    """Configuration for an experiment."""

    name: str
    description: str = ""
    tags: List[str] = None
    model_type: str = None
    dataset: str = None
    hyperparameters: Dict[str, Any] = None
    metrics_to_track: List[str] = None
    save_artifacts: bool = True
    save_checkpoints: bool = True
    checkpoint_frequency: int = 10

    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.hyperparameters is None:
            self.hyperparameters = {}
        if self.metrics_to_track is None:
            self.metrics_to_track = ["loss", "accuracy"]


class ExperimentTracker:
    """
    Comprehensive experiment tracking system.

    Tracks experiments, metrics, parameters, and artifacts
    for machine learning workflows.
    """

    def __init__(self, base_dir: str = "experiments", backend: str = "file"):
        """
        Initialize experiment tracker.

        Args:
            base_dir: Base directory for experiments
            backend: Storage backend ('file' or 'database')
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.backend = backend

        # Initialize backend
        if backend == "database":
            self._init_database()

        self.current_experiment = None

        logger.info(f"Initialized experiment tracker with {backend} backend")

    def _init_database(self):
        """Initialize SQLite database for experiment tracking."""
        self.db_path = self.base_dir / "experiments.db"

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Create experiments table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS experiments (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    status TEXT,
                    created_at TIMESTAMP,
                    updated_at TIMESTAMP,
                    completed_at TIMESTAMP,
                    config TEXT
                )
            """
            )

            # Create metrics table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id TEXT,
                    metric_name TEXT,
                    metric_value REAL,
                    step INTEGER,
                    timestamp TIMESTAMP,
                    FOREIGN KEY (experiment_id) REFERENCES experiments(id)
                )
            """
            )

            # Create parameters table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS parameters (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id TEXT,
                    param_name TEXT,
                    param_value TEXT,
                    FOREIGN KEY (experiment_id) REFERENCES experiments(id)
                )
            """
            )

            conn.commit()

    def create_experiment(self, config: Union[ExperimentConfig, Dict[str, Any]]) -> str:
        """
        Create a new experiment.

        Args:
            config: Experiment configuration

        Returns:
            Experiment ID
        """
        if isinstance(config, dict):
            config = ExperimentConfig(**config)

        # Generate experiment ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_id = f"{config.name}_{timestamp}"

        # Create experiment directory
        exp_dir = self.base_dir / exp_id
        exp_dir.mkdir(parents=True, exist_ok=True)

        # Save configuration
        config_dict = asdict(config)
        config_dict["id"] = exp_id
        config_dict["status"] = ExperimentStatus.CREATED.value
        config_dict["created_at"] = datetime.now().isoformat()

        # Save to file
        with open(exp_dir / "config.json", "w") as f:
            json.dump(config_dict, f, indent=2)

        # Save to database if using database backend
        if self.backend == "database":
            self._save_experiment_to_db(exp_id, config_dict)

        logger.info(f"Created experiment: {exp_id}")

        return exp_id

    def start_experiment(self, experiment_id: str) -> "ExperimentContext":
        """
        Start an experiment and return context manager.

        Args:
            experiment_id: Experiment ID

        Returns:
            Experiment context manager
        """
        return ExperimentContext(self, experiment_id)

    def log_params(self, params: Dict[str, Any], experiment_id: Optional[str] = None):
        """
        Log parameters for an experiment.

        Args:
            params: Parameters to log
            experiment_id: Experiment ID (uses current if None)
        """
        exp_id = experiment_id or self.current_experiment
        if not exp_id:
            raise ValueError("No active experiment")

        exp_dir = self.base_dir / exp_id

        # Save to file
        params_file = exp_dir / "parameters.json"
        if params_file.exists():
            with open(params_file, "r") as f:
                existing_params = json.load(f)
        else:
            existing_params = {}

        existing_params.update(params)

        with open(params_file, "w") as f:
            json.dump(existing_params, f, indent=2)

        # Save to database
        if self.backend == "database":
            self._save_params_to_db(exp_id, params)

        logger.debug(f"Logged {len(params)} parameters")

    def log_metric(
        self,
        name: str,
        value: float,
        step: Optional[int] = None,
        experiment_id: Optional[str] = None,
    ):
        """
        Log a metric value.

        Args:
            name: Metric name
            value: Metric value
            step: Optional step number
            experiment_id: Experiment ID (uses current if None)
        """
        exp_id = experiment_id or self.current_experiment
        if not exp_id:
            raise ValueError("No active experiment")

        exp_dir = self.base_dir / exp_id

        # Prepare metric entry
        metric_entry = {
            "name": name,
            "value": value,
            "step": step,
            "timestamp": datetime.now().isoformat(),
        }

        # Save to file
        metrics_file = exp_dir / "metrics.jsonl"
        with open(metrics_file, "a") as f:
            f.write(json.dumps(metric_entry) + "\n")

        # Save to database
        if self.backend == "database":
            self._save_metric_to_db(exp_id, metric_entry)

        logger.debug(f"Logged metric {name}={value}")

    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
        experiment_id: Optional[str] = None,
    ):
        """
        Log multiple metrics at once.

        Args:
            metrics: Dictionary of metrics
            step: Optional step number
            experiment_id: Experiment ID
        """
        for name, value in metrics.items():
            self.log_metric(name, value, step, experiment_id)

    def log_artifact(
        self,
        artifact_path: str,
        artifact_name: Optional[str] = None,
        artifact_type: str = "file",
        experiment_id: Optional[str] = None,
    ):
        """
        Log an artifact (file, model, etc).

        Args:
            artifact_path: Path to artifact
            artifact_name: Name for artifact (uses filename if None)
            artifact_type: Type of artifact
            experiment_id: Experiment ID
        """
        exp_id = experiment_id or self.current_experiment
        if not exp_id:
            raise ValueError("No active experiment")

        exp_dir = self.base_dir / exp_id
        artifacts_dir = exp_dir / "artifacts"
        artifacts_dir.mkdir(exist_ok=True)

        # Copy artifact
        artifact_path = Path(artifact_path)
        if artifact_name is None:
            artifact_name = artifact_path.name

        dest_path = artifacts_dir / artifact_name

        if artifact_path.is_file():
            shutil.copy2(artifact_path, dest_path)
        elif artifact_path.is_dir():
            shutil.copytree(artifact_path, dest_path, dirs_exist_ok=True)
        else:
            raise ValueError(f"Artifact not found: {artifact_path}")

        # Log artifact metadata
        metadata = {
            "name": artifact_name,
            "type": artifact_type,
            "path": str(dest_path.relative_to(exp_dir)),
            "size": os.path.getsize(dest_path) if dest_path.is_file() else None,
            "timestamp": datetime.now().isoformat(),
        }

        artifacts_log = exp_dir / "artifacts.jsonl"
        with open(artifacts_log, "a") as f:
            f.write(json.dumps(metadata) + "\n")

        logger.info(f"Logged artifact: {artifact_name}")

    def log_model(
        self,
        model: Any,
        model_name: str = "model.pkl",
        experiment_id: Optional[str] = None,
    ):
        """
        Log a trained model.

        Args:
            model: Model to save
            model_name: Name for model file
            experiment_id: Experiment ID
        """
        exp_id = experiment_id or self.current_experiment
        if not exp_id:
            raise ValueError("No active experiment")

        # Save model to temp file
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            pickle.dump(model, f)
            temp_path = f.name

        # Log as artifact
        self.log_artifact(temp_path, model_name, "model", exp_id)

        # Clean up temp file
        os.unlink(temp_path)

    def get_experiment(self, experiment_id: str) -> Dict[str, Any]:
        """
        Get experiment details.

        Args:
            experiment_id: Experiment ID

        Returns:
            Experiment configuration and metadata
        """
        exp_dir = self.base_dir / experiment_id

        if not exp_dir.exists():
            raise ValueError(f"Experiment not found: {experiment_id}")

        # Load config
        with open(exp_dir / "config.json", "r") as f:
            config = json.load(f)

        # Load parameters
        params_file = exp_dir / "parameters.json"
        if params_file.exists():
            with open(params_file, "r") as f:
                config["parameters"] = json.load(f)

        # Count metrics
        metrics_file = exp_dir / "metrics.jsonl"
        if metrics_file.exists():
            with open(metrics_file, "r") as f:
                config["n_metrics"] = sum(1 for _ in f)

        return config

    def get_metrics(self, experiment_id: str) -> pd.DataFrame:
        """
        Get all metrics for an experiment.

        Args:
            experiment_id: Experiment ID

        Returns:
            DataFrame of metrics
        """
        exp_dir = self.base_dir / experiment_id
        metrics_file = exp_dir / "metrics.jsonl"

        if not metrics_file.exists():
            return pd.DataFrame()

        # Load metrics
        metrics = []
        with open(metrics_file, "r") as f:
            for line in f:
                metrics.append(json.loads(line))

        return pd.DataFrame(metrics)

    def list_experiments(
        self, status: Optional[str] = None, tags: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        List all experiments.

        Args:
            status: Filter by status
            tags: Filter by tags

        Returns:
            List of experiment summaries
        """
        experiments = []

        for exp_dir in self.base_dir.iterdir():
            if exp_dir.is_dir() and (exp_dir / "config.json").exists():
                try:
                    with open(exp_dir / "config.json", "r") as f:
                        config = json.load(f)

                    # Apply filters
                    if status and config.get("status") != status:
                        continue

                    if tags:
                        exp_tags = set(config.get("tags", []))
                        if not exp_tags.intersection(set(tags)):
                            continue

                    experiments.append(
                        {
                            "id": config["id"],
                            "name": config["name"],
                            "status": config.get("status"),
                            "created_at": config.get("created_at"),
                            "tags": config.get("tags", []),
                        }
                    )
                except Exception as e:
                    logger.error(f"Error loading experiment {exp_dir.name}: {e}")

        return sorted(experiments, key=lambda x: x["created_at"], reverse=True)

    def compare_experiments(
        self, experiment_ids: List[str], metrics: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Compare multiple experiments.

        Args:
            experiment_ids: List of experiment IDs
            metrics: Metrics to compare (all if None)

        Returns:
            Comparison DataFrame
        """
        comparisons = []

        for exp_id in experiment_ids:
            try:
                # Get experiment info
                exp_info = self.get_experiment(exp_id)

                # Get final metrics
                exp_metrics = self.get_metrics(exp_id)

                if not exp_metrics.empty:
                    # Get last value for each metric
                    final_metrics = {}
                    for metric_name in exp_metrics["name"].unique():
                        if metrics is None or metric_name in metrics:
                            metric_data = exp_metrics[
                                exp_metrics["name"] == metric_name
                            ]
                            final_metrics[metric_name] = metric_data.iloc[-1]["value"]

                    # Create comparison entry
                    comparison = {
                        "experiment_id": exp_id,
                        "name": exp_info["name"],
                        "status": exp_info.get("status"),
                        **final_metrics,
                    }

                    comparisons.append(comparison)

            except Exception as e:
                logger.error(f"Error comparing experiment {exp_id}: {e}")

        return pd.DataFrame(comparisons)

    def _save_experiment_to_db(self, exp_id: str, config: Dict[str, Any]):
        """Save experiment to database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO experiments 
                (id, name, description, status, created_at, config)
                VALUES (?, ?, ?, ?, ?, ?)
            """,
                (
                    exp_id,
                    config["name"],
                    config.get("description", ""),
                    config.get("status"),
                    config.get("created_at"),
                    json.dumps(config),
                ),
            )
            conn.commit()

    def _save_params_to_db(self, exp_id: str, params: Dict[str, Any]):
        """Save parameters to database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            for name, value in params.items():
                cursor.execute(
                    """
                    INSERT INTO parameters 
                    (experiment_id, param_name, param_value)
                    VALUES (?, ?, ?)
                """,
                    (exp_id, name, str(value)),
                )
            conn.commit()

    def _save_metric_to_db(self, exp_id: str, metric: Dict[str, Any]):
        """Save metric to database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO metrics 
                (experiment_id, metric_name, metric_value, step, timestamp)
                VALUES (?, ?, ?, ?, ?)
            """,
                (
                    exp_id,
                    metric["name"],
                    metric["value"],
                    metric.get("step"),
                    metric["timestamp"],
                ),
            )
            conn.commit()


class ExperimentContext:
    """Context manager for experiments."""

    def __init__(self, tracker: ExperimentTracker, experiment_id: str):
        self.tracker = tracker
        self.experiment_id = experiment_id
        self.start_time = None

    def __enter__(self):
        self.tracker.current_experiment = self.experiment_id
        self.start_time = datetime.now()

        # Update status
        exp_dir = self.tracker.base_dir / self.experiment_id
        with open(exp_dir / "config.json", "r") as f:
            config = json.load(f)

        config["status"] = ExperimentStatus.RUNNING.value
        config["updated_at"] = datetime.now().isoformat()

        with open(exp_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)

        logger.info(f"Started experiment: {self.experiment_id}")
        return self.tracker

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Calculate duration
        duration = (datetime.now() - self.start_time).total_seconds()

        # Update status
        exp_dir = self.tracker.base_dir / self.experiment_id
        with open(exp_dir / "config.json", "r") as f:
            config = json.load(f)

        if exc_type is None:
            config["status"] = ExperimentStatus.COMPLETED.value
            logger.info(f"Completed experiment: {self.experiment_id}")
        else:
            config["status"] = ExperimentStatus.FAILED.value
            logger.error(f"Failed experiment: {self.experiment_id}")

            # Log error
            error_info = {
                "error_type": exc_type.__name__,
                "error_message": str(exc_val),
                "timestamp": datetime.now().isoformat(),
            }

            with open(exp_dir / "error.json", "w") as f:
                json.dump(error_info, f, indent=2)

        config["completed_at"] = datetime.now().isoformat()
        config["duration_seconds"] = duration

        with open(exp_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)

        self.tracker.current_experiment = None


class MLFlowTracker:
    """
    MLflow-compatible experiment tracker.

    This class provides an MLflow-like API interface while using
    the ExperimentTracker backend.
    """

    def __init__(
        self, tracking_uri: str = "experiments", experiment_name: str = "Default"
    ):
        """
        Initialize MLFlowTracker.

        Args:
            tracking_uri: URI for tracking (directory path)
            experiment_name: Default experiment name
        """
        self.tracker = ExperimentTracker(base_dir=tracking_uri)
        self.experiment_name = experiment_name
        self.active_run = None
        self._runs = {}

        logger.info(f"Initialized MLFlowTracker at {tracking_uri}")

    def set_experiment(self, experiment_name: str):
        """
        Set the active experiment.

        Args:
            experiment_name: Name of experiment
        """
        self.experiment_name = experiment_name

    def start_run(
        self,
        run_name: Optional[str] = None,
        nested: bool = False,
        tags: Optional[Dict[str, str]] = None,
        description: Optional[str] = None,
    ) -> "MLFlowRun":
        """
        Start a new run.

        Args:
            run_name: Name for the run
            nested: Whether this is a nested run
            tags: Tags for the run
            description: Run description

        Returns:
            MLFlowRun context manager
        """
        if self.active_run and not nested:
            raise RuntimeError("Run already active. Set nested=True for nested runs.")

        # Create experiment config
        config = ExperimentConfig(
            name=run_name or self.experiment_name,
            description=description or "",
            tags=list(tags.keys()) if tags else [],
        )

        # Create experiment
        run_id = self.tracker.create_experiment(config)

        # Create run object
        run = MLFlowRun(self, run_id)

        # Store run
        self._runs[run_id] = run

        # Log tags as parameters
        if tags:
            self.log_params(tags)

        if not nested:
            self.active_run = run

        return run

    def end_run(self, status: str = "FINISHED"):
        """
        End the current run.

        Args:
            status: Run status
        """
        if self.active_run:
            self.active_run.end(status)
            self.active_run = None

    def log_param(self, key: str, value: Any):
        """
        Log a parameter.

        Args:
            key: Parameter name
            value: Parameter value
        """
        if not self.active_run:
            raise RuntimeError("No active run")

        self.tracker.log_params({key: value}, self.active_run.run_id)

    def log_params(self, params: Dict[str, Any]):
        """
        Log multiple parameters.

        Args:
            params: Dictionary of parameters
        """
        if not self.active_run:
            raise RuntimeError("No active run")

        self.tracker.log_params(params, self.active_run.run_id)

    def log_metric(self, key: str, value: float, step: Optional[int] = None):
        """
        Log a metric.

        Args:
            key: Metric name
            value: Metric value
            step: Optional step
        """
        if not self.active_run:
            raise RuntimeError("No active run")

        self.tracker.log_metric(key, value, step, self.active_run.run_id)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Log multiple metrics.

        Args:
            metrics: Dictionary of metrics
            step: Optional step
        """
        if not self.active_run:
            raise RuntimeError("No active run")

        self.tracker.log_metrics(metrics, step, self.active_run.run_id)

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """
        Log an artifact.

        Args:
            local_path: Local path to artifact
            artifact_path: Artifact path in run
        """
        if not self.active_run:
            raise RuntimeError("No active run")

        self.tracker.log_artifact(
            local_path, artifact_path, experiment_id=self.active_run.run_id
        )

    def log_artifacts(self, local_dir: str, artifact_path: Optional[str] = None):
        """
        Log all artifacts in a directory.

        Args:
            local_dir: Local directory path
            artifact_path: Artifact path in run
        """
        local_dir = Path(local_dir)

        for file_path in local_dir.rglob("*"):
            if file_path.is_file():
                self.log_artifact(str(file_path))

    def log_model(
        self,
        model: Any,
        artifact_path: str,
        registered_model_name: Optional[str] = None,
    ):
        """
        Log a model.

        Args:
            model: Model to log
            artifact_path: Path to save model
            registered_model_name: Optional model registry name
        """
        if not self.active_run:
            raise RuntimeError("No active run")

        # Save model
        self.tracker.log_model(model, artifact_path, self.active_run.run_id)

        # Register model if name provided
        if registered_model_name:
            self._register_model(registered_model_name, self.active_run.run_id)

    def _register_model(self, name: str, run_id: str):
        """Register a model in the registry."""
        registry_file = self.tracker.base_dir / "model_registry.json"

        if registry_file.exists():
            with open(registry_file, "r") as f:
                registry = json.load(f)
        else:
            registry = {}

        if name not in registry:
            registry[name] = []

        registry[name].append(
            {
                "run_id": run_id,
                "timestamp": datetime.now().isoformat(),
                "version": len(registry[name]) + 1,
            }
        )

        with open(registry_file, "w") as f:
            json.dump(registry, f, indent=2)

    def search_runs(
        self,
        experiment_ids: Optional[List[str]] = None,
        filter_string: Optional[str] = None,
        max_results: int = 100,
    ) -> pd.DataFrame:
        """
        Search for runs.

        Args:
            experiment_ids: List of experiment IDs to search
            filter_string: Filter string (simplified)
            max_results: Maximum results to return

        Returns:
            DataFrame of runs
        """
        all_experiments = self.tracker.list_experiments()

        if experiment_ids:
            all_experiments = [e for e in all_experiments if e["id"] in experiment_ids]

        # Simple filter implementation
        if filter_string:
            # This is a simplified version - real MLflow has complex filtering
            filtered = []
            for exp in all_experiments:
                if filter_string.lower() in exp["name"].lower():
                    filtered.append(exp)
            all_experiments = filtered

        # Limit results
        all_experiments = all_experiments[:max_results]

        # Convert to DataFrame
        return pd.DataFrame(all_experiments)

    def get_metric_history(self, run_id: str, key: str) -> List[Dict[str, Any]]:
        """
        Get metric history for a run.

        Args:
            run_id: Run ID
            key: Metric key

        Returns:
            List of metric values
        """
        metrics_df = self.tracker.get_metrics(run_id)

        if metrics_df.empty:
            return []

        metric_data = metrics_df[metrics_df["name"] == key]

        return metric_data.to_dict("records")


class MLFlowRun:
    """MLflow run context manager."""

    def __init__(self, mlflow_tracker: MLFlowTracker, run_id: str):
        self.mlflow_tracker = mlflow_tracker
        self.run_id = run_id
        self.start_time = datetime.now()

    def __enter__(self):
        self.mlflow_tracker.tracker.current_experiment = self.run_id
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.end("FINISHED")
        else:
            self.end("FAILED")

    def end(self, status: str = "FINISHED"):
        """End the run."""
        # Update run status
        exp_dir = self.mlflow_tracker.tracker.base_dir / self.run_id
        with open(exp_dir / "config.json", "r") as f:
            config = json.load(f)

        config["mlflow_status"] = status
        config["end_time"] = datetime.now().isoformat()
        config["duration"] = (datetime.now() - self.start_time).total_seconds()

        with open(exp_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)

        self.mlflow_tracker.tracker.current_experiment = None


class WandbTracker:
    """
    Weights & Biases (wandb) compatible experiment tracker.

    This class provides a wandb-like API interface while using
    the ExperimentTracker backend.
    """

    def __init__(
        self,
        project: str = "my-project",
        entity: Optional[str] = None,
        dir: str = "experiments",
    ):
        """
        Initialize WandbTracker.

        Args:
            project: Project name
            entity: Entity/team name
            dir: Directory for experiments
        """
        self.project = project
        self.entity = entity
        self.tracker = ExperimentTracker(base_dir=Path(dir) / project)
        self.run = None
        self._config = {}
        self._summary = {}

        logger.info(f"Initialized WandbTracker for project: {project}")

    def init(
        self,
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        project: Optional[str] = None,
        tags: Optional[List[str]] = None,
        notes: Optional[str] = None,
        reinit: bool = False,
    ) -> "WandbRun":
        """
        Initialize a new run.

        Args:
            name: Run name
            config: Configuration dictionary
            project: Project name (overrides default)
            tags: List of tags
            notes: Run notes
            reinit: Whether to reinitialize

        Returns:
            WandbRun object
        """
        if self.run and not reinit:
            raise RuntimeError(
                "Run already initialized. Set reinit=True to start new run."
            )

        # Update project if provided
        if project:
            self.project = project

        # Create experiment config
        exp_config = ExperimentConfig(
            name=name or f"run-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            description=notes or "",
            tags=tags or [],
            hyperparameters=config or {},
        )

        # Create experiment
        run_id = self.tracker.create_experiment(exp_config)

        # Create run
        self.run = WandbRun(self, run_id)

        # Set config
        if config:
            self.config.update(config)
            self.tracker.log_params(config, run_id)

        return self.run

    def log(
        self, data: Dict[str, Any], step: Optional[int] = None, commit: bool = True
    ):
        """
        Log metrics.

        Args:
            data: Dictionary of metrics
            step: Optional step
            commit: Whether to commit immediately
        """
        if not self.run:
            raise RuntimeError("No active run. Call wandb.init() first.")

        # Log metrics
        for key, value in data.items():
            if isinstance(value, (int, float)):
                self.tracker.log_metric(key, value, step, self.run.id)
                self._summary[key] = value

    def log_artifact(
        self,
        artifact_or_path: Union[str, "WandbArtifact"],
        name: Optional[str] = None,
        type: Optional[str] = None,
    ):
        """
        Log an artifact.

        Args:
            artifact_or_path: Artifact or path to artifact
            name: Artifact name
            type: Artifact type
        """
        if not self.run:
            raise RuntimeError("No active run. Call wandb.init() first.")

        if isinstance(artifact_or_path, str):
            self.tracker.log_artifact(
                artifact_or_path, name, type or "artifact", self.run.id
            )
        else:
            # Handle WandbArtifact
            artifact_or_path.save(self.tracker.base_dir / self.run.id / "artifacts")

    def finish(self, exit_code: int = 0):
        """
        Finish the current run.

        Args:
            exit_code: Exit code (0 for success)
        """
        if self.run:
            self.run.finish(exit_code)
            self.run = None

    @property
    def config(self) -> Dict[str, Any]:
        """Get run configuration."""
        return self._config

    @config.setter
    def config(self, value: Dict[str, Any]):
        """Set run configuration."""
        self._config = value
        if self.run:
            self.tracker.log_params(value, self.run.id)

    @property
    def summary(self) -> Dict[str, Any]:
        """Get run summary."""
        return self._summary

    def watch(self, model: Any, log: str = "gradients", log_freq: int = 100):
        """
        Watch a model (placeholder for compatibility).

        Args:
            model: Model to watch
            log: What to log
            log_freq: Logging frequency
        """
        logger.info(
            f"Model watching not implemented. Would watch: {type(model).__name__}"
        )

    def alert(self, title: str, text: str, level: str = "INFO"):
        """
        Send an alert (logs to file).

        Args:
            title: Alert title
            text: Alert text
            level: Alert level
        """
        if self.run:
            alert_file = self.tracker.base_dir / self.run.id / "alerts.jsonl"
            alert = {
                "title": title,
                "text": text,
                "level": level,
                "timestamp": datetime.now().isoformat(),
            }

            with open(alert_file, "a") as f:
                f.write(json.dumps(alert) + "\n")


class WandbRun:
    """W&B run object."""

    def __init__(self, wandb_tracker: WandbTracker, run_id: str):
        self.wandb_tracker = wandb_tracker
        self.id = run_id
        self.name = run_id.split("_")[0]
        self.project = wandb_tracker.project
        self.entity = wandb_tracker.entity
        self.start_time = datetime.now()

    def finish(self, exit_code: int = 0):
        """Finish the run."""
        # Update run status
        exp_dir = self.wandb_tracker.tracker.base_dir / self.id
        with open(exp_dir / "config.json", "r") as f:
            config = json.load(f)

        config["wandb_exit_code"] = exit_code
        config["wandb_state"] = "finished" if exit_code == 0 else "failed"
        config["runtime"] = (datetime.now() - self.start_time).total_seconds()

        with open(exp_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)

        self.wandb_tracker.tracker.current_experiment = None

    @property
    def url(self) -> str:
        """Get run URL (local path)."""
        return str(self.wandb_tracker.tracker.base_dir / self.id)

    @property
    def summary(self) -> Dict[str, Any]:
        """Get run summary."""
        return self.wandb_tracker.summary


class WandbArtifact:
    """W&B artifact placeholder."""

    def __init__(self, name: str, type: str = "artifact"):
        self.name = name
        self.type = type
        self._files = []

    def add_file(self, local_path: str, name: Optional[str] = None):
        """Add file to artifact."""
        self._files.append((local_path, name or Path(local_path).name))

    def add_dir(self, local_path: str):
        """Add directory to artifact."""
        path = Path(local_path)
        for file_path in path.rglob("*"):
            if file_path.is_file():
                rel_path = file_path.relative_to(path)
                self._files.append((str(file_path), str(rel_path)))

    def save(self, save_path: Path):
        """Save artifact to path."""
        save_path.mkdir(parents=True, exist_ok=True)

        for src, dst in self._files:
            dst_path = save_path / dst
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst_path)


class ExperimentComparer:
    """
    Advanced experiment comparison and analysis tool.

    Provides comprehensive comparison, visualization, and analysis
    of multiple experiments.
    """

    def __init__(self, tracker: ExperimentTracker):
        """
        Initialize ExperimentComparer.

        Args:
            tracker: ExperimentTracker instance
        """
        self.tracker = tracker

    def compare_experiments(
        self,
        experiment_ids: List[str],
        metrics: Optional[List[str]] = None,
        include_params: bool = True,
    ) -> pd.DataFrame:
        """
        Compare multiple experiments.

        Args:
            experiment_ids: List of experiment IDs
            metrics: Specific metrics to compare
            include_params: Whether to include parameters

        Returns:
            Comparison DataFrame
        """
        comparisons = []

        for exp_id in experiment_ids:
            try:
                # Get experiment details
                exp_info = self.tracker.get_experiment(exp_id)

                # Build comparison entry
                comparison = {
                    "experiment_id": exp_id,
                    "name": exp_info["name"],
                    "status": exp_info.get("status"),
                    "created_at": exp_info.get("created_at"),
                    "duration": exp_info.get("duration_seconds"),
                }

                # Add parameters if requested
                if include_params and "parameters" in exp_info:
                    for param, value in exp_info["parameters"].items():
                        comparison[f"param_{param}"] = value

                # Get metrics
                exp_metrics = self.tracker.get_metrics(exp_id)

                if not exp_metrics.empty:
                    # Get final value for each metric
                    for metric_name in exp_metrics["name"].unique():
                        if metrics is None or metric_name in metrics:
                            metric_data = exp_metrics[
                                exp_metrics["name"] == metric_name
                            ]
                            comparison[f"metric_{metric_name}"] = metric_data.iloc[-1][
                                "value"
                            ]

                            # Also get best value
                            if metric_name in ["loss", "error", "mae", "mse"]:
                                comparison[f"best_{metric_name}"] = metric_data[
                                    "value"
                                ].min()
                            else:
                                comparison[f"best_{metric_name}"] = metric_data[
                                    "value"
                                ].max()

                comparisons.append(comparison)

            except Exception as e:
                logger.error(f"Error loading experiment {exp_id}: {e}")

        return pd.DataFrame(comparisons)

    def plot_metrics_comparison(
        self,
        experiment_ids: List[str],
        metrics: List[str],
        figsize: Tuple[int, int] = (12, 8),
        save_path: Optional[str] = None,
    ):
        """
        Plot metrics comparison across experiments.

        Args:
            experiment_ids: List of experiment IDs
            metrics: List of metrics to plot
            figsize: Figure size
            save_path: Path to save plot
        """
        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=figsize)

        if n_metrics == 1:
            axes = [axes]

        for idx, metric in enumerate(metrics):
            ax = axes[idx]

            for exp_id in experiment_ids:
                try:
                    # Get metric data
                    metrics_df = self.tracker.get_metrics(exp_id)

                    if not metrics_df.empty and metric in metrics_df["name"].values:
                        metric_data = metrics_df[metrics_df["name"] == metric]

                        # Plot metric over steps
                        steps = metric_data["step"].fillna(range(len(metric_data)))
                        ax.plot(
                            steps,
                            metric_data["value"],
                            label=exp_id.split("_")[0],
                            marker=".",
                        )

                except Exception as e:
                    logger.error(f"Error plotting {metric} for {exp_id}: {e}")

            ax.set_xlabel("Step")
            ax.set_ylabel(metric)
            ax.set_title(f"{metric} Comparison")
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

    def plot_parallel_coordinates(
        self,
        experiment_ids: List[str],
        params: List[str],
        metric: str,
        normalize: bool = True,
        save_path: Optional[str] = None,
    ):
        """
        Create parallel coordinates plot.

        Args:
            experiment_ids: List of experiment IDs
            params: List of parameters to include
            metric: Metric to color by
            normalize: Whether to normalize values
            save_path: Path to save plot
        """
        # Get comparison data
        df = self.compare_experiments(experiment_ids)

        # Prepare data for parallel coordinates
        plot_data = []

        for _, row in df.iterrows():
            entry = {}

            # Add parameters
            for param in params:
                param_col = f"param_{param}"
                if param_col in row:
                    entry[param] = row[param_col]

            # Add metric
            metric_col = f"metric_{metric}"
            if metric_col in row:
                entry[metric] = row[metric_col]
                entry["experiment"] = row["experiment_id"]
                plot_data.append(entry)

        if not plot_data:
            logger.warning("No data to plot")
            return

        plot_df = pd.DataFrame(plot_data)

        # Normalize if requested
        if normalize:
            from sklearn.preprocessing import MinMaxScaler

            scaler = MinMaxScaler()

            numeric_cols = [
                col for col in plot_df.columns if col not in ["experiment", metric]
            ]
            plot_df[numeric_cols] = scaler.fit_transform(plot_df[numeric_cols])

        # Create plot
        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot parallel coordinates
        pd.plotting.parallel_coordinates(
            plot_df.drop("experiment", axis=1),
            metric,
            colormap="viridis",
            alpha=0.7,
            ax=ax,
        )

        ax.set_title(f"Parallel Coordinates Plot - Colored by {metric}")
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

    def find_best_experiment(
        self, experiment_ids: List[str], metric: str, mode: str = "min"
    ) -> str:
        """
        Find best experiment based on metric.

        Args:
            experiment_ids: List of experiment IDs
            metric: Metric to optimize
            mode: 'min' or 'max'

        Returns:
            Best experiment ID
        """
        comparison_df = self.compare_experiments(experiment_ids, [metric])

        metric_col = f"metric_{metric}"
        if metric_col not in comparison_df.columns:
            raise ValueError(f"Metric {metric} not found in experiments")

        if mode == "min":
            best_idx = comparison_df[metric_col].idxmin()
        else:
            best_idx = comparison_df[metric_col].idxmax()

        return comparison_df.loc[best_idx, "experiment_id"]

    def generate_report(
        self,
        experiment_ids: List[str],
        output_path: str = "experiment_report.html",
        include_plots: bool = True,
    ):
        """
        Generate comprehensive experiment report.

        Args:
            experiment_ids: List of experiment IDs
            output_path: Path for report
            include_plots: Whether to include plots
        """
        # Create report content
        html_content = """
        <html>
        <head>
            <title>Experiment Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                table { border-collapse: collapse; width: 100%; margin: 20px 0; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
                .metric { color: #2e7d32; font-weight: bold; }
                .param { color: #1976d2; }
                h1, h2 { color: #333; }
                .plot { margin: 20px 0; text-align: center; }
            </style>
        </head>
        <body>
        """

        # Add header
        html_content += f"""
        <h1>Experiment Comparison Report</h1>
        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p>Number of experiments: {len(experiment_ids)}</p>
        """

        # Get comparison data
        comparison_df = self.compare_experiments(experiment_ids)

        # Add summary table
        html_content += "<h2>Experiment Summary</h2>"
        html_content += comparison_df.to_html(classes="summary-table", index=False)

        # Add detailed experiment information
        html_content += "<h2>Detailed Experiment Information</h2>"

        for exp_id in experiment_ids:
            try:
                exp_info = self.tracker.get_experiment(exp_id)

                html_content += f"<h3>{exp_id}</h3>"
                html_content += "<ul>"
                html_content += f"<li><strong>Name:</strong> {exp_info['name']}</li>"
                html_content += f"<li><strong>Status:</strong> {exp_info.get('status', 'Unknown')}</li>"
                html_content += f"<li><strong>Created:</strong> {exp_info.get('created_at', 'Unknown')}</li>"

                if "description" in exp_info:
                    html_content += f"<li><strong>Description:</strong> {exp_info['description']}</li>"

                if "parameters" in exp_info:
                    html_content += "<li><strong>Parameters:</strong><ul>"
                    for param, value in exp_info["parameters"].items():
                        html_content += f"<li class='param'>{param}: {value}</li>"
                    html_content += "</ul></li>"

                # Get final metrics
                metrics_df = self.tracker.get_metrics(exp_id)
                if not metrics_df.empty:
                    html_content += "<li><strong>Final Metrics:</strong><ul>"

                    for metric_name in metrics_df["name"].unique():
                        final_value = metrics_df[
                            metrics_df["name"] == metric_name
                        ].iloc[-1]["value"]
                        html_content += (
                            f"<li class='metric'>{metric_name}: {final_value:.4f}</li>"
                        )

                    html_content += "</ul></li>"

                html_content += "</ul>"

            except Exception as e:
                html_content += f"<p>Error loading experiment {exp_id}: {e}</p>"

        # Add plots if requested
        if include_plots and len(experiment_ids) > 1:
            html_content += "<h2>Metric Comparisons</h2>"

            # Get all metrics
            all_metrics = set()
            for exp_id in experiment_ids:
                metrics_df = self.tracker.get_metrics(exp_id)
                if not metrics_df.empty:
                    all_metrics.update(metrics_df["name"].unique())

            # Create plots
            for metric in all_metrics:
                plot_path = f"plot_{metric}.png"

                try:
                    self.plot_metrics_comparison(
                        experiment_ids, [metric], save_path=plot_path
                    )

                    html_content += f"""
                    <div class='plot'>
                        <h3>{metric}</h3>
                        <img src='{plot_path}' width='600'>
                    </div>
                    """
                except Exception as e:
                    logger.error(f"Error creating plot for {metric}: {e}")

        # Close HTML
        html_content += """
        </body>
        </html>
        """

        # Save report
        with open(output_path, "w") as f:
            f.write(html_content)

        logger.info(f"Report saved to {output_path}")

    def get_parameter_importance(
        self, experiment_ids: List[str], target_metric: str
    ) -> pd.DataFrame:
        """
        Analyze parameter importance for a metric.

        Args:
            experiment_ids: List of experiment IDs
            target_metric: Target metric to analyze

        Returns:
            DataFrame with parameter importance
        """
        # Get comparison data
        df = self.compare_experiments(experiment_ids)

        # Extract parameter columns and target metric
        param_cols = [col for col in df.columns if col.startswith("param_")]
        metric_col = f"metric_{target_metric}"

        if metric_col not in df.columns:
            raise ValueError(f"Metric {target_metric} not found")

        # Calculate correlations
        correlations = []

        for param_col in param_cols:
            try:
                # Convert to numeric if possible
                param_values = pd.to_numeric(df[param_col], errors="coerce")

                if not param_values.isna().all():
                    corr = param_values.corr(df[metric_col])

                    correlations.append(
                        {
                            "parameter": param_col.replace("param_", ""),
                            "correlation": corr,
                            "abs_correlation": abs(corr),
                        }
                    )
            except Exception as e:
                logger.debug(f"Could not calculate correlation for {param_col}: {e}")

        # Create importance DataFrame
        importance_df = pd.DataFrame(correlations)
        importance_df = importance_df.sort_values("abs_correlation", ascending=False)

        return importance_df


# Integration with MLflow-style API
class MLflowCompatibleTracker:
    """MLflow-compatible API wrapper for ExperimentTracker."""

    def __init__(self, tracker: ExperimentTracker):
        self.tracker = tracker
        self.active_run = None

    def start_run(self, run_name: str = None, tags: Dict[str, str] = None):
        """Start a new run."""
        config = ExperimentConfig(
            name=run_name or "unnamed_run", tags=list(tags.keys()) if tags else []
        )

        self.active_run = self.tracker.create_experiment(config)
        self.tracker.current_experiment = self.active_run

        if tags:
            self.tracker.log_params(tags)

    def end_run(self):
        """End the current run."""
        if self.active_run:
            self.tracker.current_experiment = None
            self.active_run = None

    def log_param(self, key: str, value: Any):
        """Log a parameter."""
        self.tracker.log_params({key: value})

    def log_params(self, params: Dict[str, Any]):
        """Log multiple parameters."""
        self.tracker.log_params(params)

    def log_metric(self, key: str, value: float, step: int = None):
        """Log a metric."""
        self.tracker.log_metric(key, value, step)

    def log_metrics(self, metrics: Dict[str, float], step: int = None):
        """Log multiple metrics."""
        self.tracker.log_metrics(metrics, step)

    def log_artifact(self, local_path: str):
        """Log an artifact."""
        self.tracker.log_artifact(local_path)

    def log_model(self, model: Any, artifact_path: str):
        """Log a model."""
        self.tracker.log_model(model, artifact_path)


# Example usage
def example_experiment_tracking():
    """Example of using experiment tracking."""
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import train_test_split

    # Initialize tracker
    tracker = ExperimentTracker("example_experiments")

    # Create experiment
    config = ExperimentConfig(
        name="rf_classification",
        description="Random Forest classification experiment",
        tags=["classification", "random_forest"],
        model_type="RandomForestClassifier",
        dataset="synthetic",
        hyperparameters={"n_estimators": 100, "max_depth": 10, "random_state": 42},
        metrics_to_track=["accuracy", "f1_score"],
    )

    exp_id = tracker.create_experiment(config)

    # Run experiment
    with tracker.start_experiment(exp_id) as exp:
        # Generate data
        X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        # Log dataset info
        exp.log_params(
            {"n_samples": len(X), "n_features": X.shape[1], "test_size": 0.2}
        )

        # Train model
        model = RandomForestClassifier(**config.hyperparameters)
        model.fit(X_train, y_train)

        # Evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # Log metrics
        exp.log_metrics({"accuracy": accuracy, "f1_score": f1})

        # Save model
        exp.log_model(model, "model.pkl")

    print(f"Experiment {exp_id} completed!")

    # Example with MLFlow API
    print("\n--- MLFlow API Example ---")
    mlflow = MLFlowTracker()

    with mlflow.start_run(run_name="mlflow_example"):
        mlflow.log_params({"learning_rate": 0.01, "batch_size": 32})

        for epoch in range(5):
            mlflow.log_metric("loss", 1.0 / (epoch + 1), step=epoch)
            mlflow.log_metric("accuracy", 0.8 + epoch * 0.02, step=epoch)

    # Example with W&B API
    print("\n--- W&B API Example ---")
    wandb = WandbTracker(project="my-awesome-project")

    wandb.init(
        name="wandb_example",
        config={"learning_rate": 0.001, "architecture": "CNN", "dataset": "CIFAR-10"},
    )

    for i in range(10):
        wandb.log(
            {
                "loss": 2.0 / (i + 1),
                "accuracy": 0.7 + i * 0.03,
                "val_loss": 2.5 / (i + 1),
                "val_accuracy": 0.65 + i * 0.025,
            },
            step=i,
        )

    wandb.finish()

    # Compare experiments
    print("\n--- Experiment Comparison ---")
    comparer = ExperimentComparer(tracker)

    all_experiments = tracker.list_experiments()
    if len(all_experiments) >= 2:
        exp_ids = [exp["id"] for exp in all_experiments[:3]]

        comparison_df = comparer.compare_experiments(exp_ids)
        print(comparison_df)

        # Generate report
        comparer.generate_report(exp_ids, "experiment_report.html")
        print("Report generated!")


if __name__ == "__main__":
    example_experiment_tracking()
