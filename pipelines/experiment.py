"""
pipelines/experiment.py
Experiment tracking and management for machine learning workflows.
"""

import os
import json
import yaml
import pickle
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple, Callable
from datetime import datetime
import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict
import logging
from enum import Enum
import shutil
import sqlite3
from contextlib import contextmanager
import matplotlib.pyplot as plt
import seaborn as sns

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
    
    def __init__(self, base_dir: str = "experiments",
                 backend: str = "file"):
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
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS experiments (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    status TEXT,
                    created_at TIMESTAMP,
                    updated_at TIMESTAMP,
                    completed_at TIMESTAMP,
                    config TEXT,
                    tags TEXT,
                    metrics_summary TEXT
                )
            """)
            
            # Create metrics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id TEXT,
                    metric_name TEXT,
                    metric_value REAL,
                    step INTEGER,
                    timestamp TIMESTAMP,
                    FOREIGN KEY (experiment_id) REFERENCES experiments (id)
                )
            """)
            
            # Create parameters table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS parameters (
                    experiment_id TEXT,
                    param_name TEXT,
                    param_value TEXT,
                    param_type TEXT,
                    FOREIGN KEY (experiment_id) REFERENCES experiments (id)
                )
            """)
            
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
        config_dict['id'] = exp_id
        config_dict['status'] = ExperimentStatus.CREATED.value
        config_dict['created_at'] = datetime.now().isoformat()
        
        # Save to file
        with open(exp_dir / "config.json", 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        # Save to database if using database backend
        if self.backend == "database":
            self._save_experiment_to_db(exp_id, config_dict)
        
        logger.info(f"Created experiment: {exp_id}")
        
        return exp_id
    
    def start_experiment(self, experiment_id: str) -> 'ExperimentContext':
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
            with open(params_file, 'r') as f:
                existing_params = json.load(f)
        else:
            existing_params = {}
        
        existing_params.update(params)
        
        with open(params_file, 'w') as f:
            json.dump(existing_params, f, indent=2)
        
        # Save to database
        if self.backend == "database":
            self._save_params_to_db(exp_id, params)
        
        logger.debug(f"Logged {len(params)} parameters")
    
    def log_metric(self, name: str, value: float, step: Optional[int] = None,
                  experiment_id: Optional[str] = None):
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
            'name': name,
            'value': value,
            'step': step,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save to file
        metrics_file = exp_dir / "metrics.jsonl"
        with open(metrics_file, 'a') as f:
            f.write(json.dumps(metric_entry) + '\n')
        
        # Save to database
        if self.backend == "database":
            self._save_metric_to_db(exp_id, metric_entry)
        
        logger.debug(f"Logged metric {name}={value}")
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None,
                   experiment_id: Optional[str] = None):
        """
        Log multiple metrics at once.
        
        Args:
            metrics: Dictionary of metrics
            step: Optional step number
            experiment_id: Experiment ID
        """
        for name, value in metrics.items():
            self.log_metric(name, value, step, experiment_id)
    
    def log_artifact(self, artifact_path: str, artifact_name: Optional[str] = None,
                    artifact_type: str = "file", experiment_id: Optional[str] = None):
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
            'name': artifact_name,
            'type': artifact_type,
            'path': str(dest_path.relative_to(exp_dir)),
            'size': os.path.getsize(dest_path) if dest_path.is_file() else None,
            'timestamp': datetime.now().isoformat()
        }
        
        artifacts_log = exp_dir / "artifacts.jsonl"
        with open(artifacts_log, 'a') as f:
            f.write(json.dumps(metadata) + '\n')
        
        logger.info(f"Logged artifact: {artifact_name}")
    
    def log_model(self, model: Any, model_name: str = "model.pkl",
                 experiment_id: Optional[str] = None):
        """
        Log a trained model.
        
        Args:
            model: Model object
            model_name: Name for saved model
            experiment_id: Experiment ID
        """
        exp_id = experiment_id or self.current_experiment
        if not exp_id:
            raise ValueError("No active experiment")
        
        exp_dir = self.base_dir / exp_id
        models_dir = exp_dir / "models"
        models_dir.mkdir(exist_ok=True)
        
        model_path = models_dir / model_name
        
        # Save model
        import joblib
        joblib.dump(model, model_path)
        
        # Log as artifact
        self.log_artifact(model_path, model_name, "model", exp_id)
        
        # Save model metadata
        metadata = {
            'model_class': model.__class__.__name__,
            'model_params': model.get_params() if hasattr(model, 'get_params') else {},
            'timestamp': datetime.now().isoformat()
        }
        
        with open(models_dir / f"{model_name}.metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def log_dataset_info(self, dataset_info: Dict[str, Any],
                        experiment_id: Optional[str] = None):
        """
        Log dataset information.
        
        Args:
            dataset_info: Dataset metadata
            experiment_id: Experiment ID
        """
        exp_id = experiment_id or self.current_experiment
        if not exp_id:
            raise ValueError("No active experiment")
        
        exp_dir = self.base_dir / exp_id
        
        with open(exp_dir / "dataset_info.json", 'w') as f:
            json.dump(dataset_info, f, indent=2)
    
    def get_experiment(self, experiment_id: str) -> Dict[str, Any]:
        """
        Get experiment details.
        
        Args:
            experiment_id: Experiment ID
            
        Returns:
            Experiment information
        """
        exp_dir = self.base_dir / experiment_id
        
        if not exp_dir.exists():
            raise ValueError(f"Experiment not found: {experiment_id}")
        
        # Load config
        with open(exp_dir / "config.json", 'r') as f:
            config = json.load(f)
        
        # Load metrics
        metrics = self.get_metrics(experiment_id)
        
        # Load parameters
        params_file = exp_dir / "parameters.json"
        if params_file.exists():
            with open(params_file, 'r') as f:
                params = json.load(f)
        else:
            params = {}
        
        return {
            'config': config,
            'parameters': params,
            'metrics': metrics
        }
    
    def get_metrics(self, experiment_id: str) -> pd.DataFrame:
        """
        Get metrics for an experiment.
        
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
        with open(metrics_file, 'r') as f:
            for line in f:
                metrics.append(json.loads(line))
        
        return pd.DataFrame(metrics)
    
    def list_experiments(self, status: Optional[str] = None,
                        tags: Optional[List[str]] = None) -> List[Dict[str, Any]]:
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
                    with open(exp_dir / "config.json", 'r') as f:
                        config = json.load(f)
                    
                    # Apply filters
                    if status and config.get('status') != status:
                        continue
                    
                    if tags:
                        exp_tags = set(config.get('tags', []))
                        if not exp_tags.intersection(set(tags)):
                            continue
                    
                    experiments.append({
                        'id': config['id'],
                        'name': config['name'],
                        'status': config.get('status'),
                        'created_at': config.get('created_at'),
                        'tags': config.get('tags', [])
                    })
                except Exception as e:
                    logger.error(f"Error loading experiment {exp_dir.name}: {e}")
        
        return sorted(experiments, key=lambda x: x['created_at'], reverse=True)
    
    def compare_experiments(self, experiment_ids: List[str],
                          metrics: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Compare multiple experiments.
        
        Args:
            experiment_ids: List of experiment IDs
            metrics: Metrics to compare (all if None)
            
        Returns:
            Comparison DataFrame
        """
        comparison_data = []
        
        for exp_id in experiment_ids:
            exp_data = self.get_experiment(exp_id)
            
            # Get final metrics
            metrics_df = self.get_metrics(exp_id)
            
            if not metrics_df.empty:
                # Get last value for each metric
                final_metrics = {}
                for metric_name in metrics_df['name'].unique():
                    if metrics is None or metric_name in metrics:
                        metric_values = metrics_df[metrics_df['name'] == metric_name]
                        final_metrics[metric_name] = metric_values.iloc[-1]['value']
                
                row = {
                    'experiment_id': exp_id,
                    'name': exp_data['config']['name'],
                    **exp_data['parameters'],
                    **final_metrics
                }
                
                comparison_data.append(row)
        
        return pd.DataFrame(comparison_data)
    
    def plot_metrics(self, experiment_ids: Union[str, List[str]],
                    metrics: Optional[List[str]] = None,
                    figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Plot metrics for experiments.
        
        Args:
            experiment_ids: Experiment ID(s)
            metrics: Metrics to plot
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        if isinstance(experiment_ids, str):
            experiment_ids = [experiment_ids]
        
        # Collect metrics data
        all_metrics = {}
        
        for exp_id in experiment_ids:
            metrics_df = self.get_metrics(exp_id)
            
            if not metrics_df.empty:
                for metric_name in metrics_df['name'].unique():
                    if metrics is None or metric_name in metrics:
                        if metric_name not in all_metrics:
                            all_metrics[metric_name] = {}
                        
                        metric_data = metrics_df[metrics_df['name'] == metric_name]
                        all_metrics[metric_name][exp_id] = metric_data
        
        # Create plots
        n_metrics = len(all_metrics)
        if n_metrics == 0:
            logger.warning("No metrics to plot")
            return None
        
        fig, axes = plt.subplots(
            nrows=(n_metrics + 1) // 2,
            ncols=min(2, n_metrics),
            figsize=figsize,
            squeeze=False
        )
        axes = axes.flatten()
        
        for idx, (metric_name, exp_data) in enumerate(all_metrics.items()):
            ax = axes[idx]
            
            for exp_id, data in exp_data.items():
                if 'step' in data.columns and data['step'].notna().any():
                    x = data['step']
                else:
                    x = range(len(data))
                
                ax.plot(x, data['value'], label=exp_id, marker='o')
            
            ax.set_title(metric_name)
            ax.set_xlabel('Step')
            ax.set_ylabel('Value')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Hide extra subplots
        for idx in range(n_metrics, len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        return fig
    
    def _save_experiment_to_db(self, exp_id: str, config: Dict[str, Any]):
        """Save experiment to database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO experiments 
                (id, name, description, status, created_at, config, tags)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                exp_id,
                config['name'],
                config.get('description', ''),
                config['status'],
                config['created_at'],
                json.dumps(config),
                json.dumps(config.get('tags', []))
            ))
            conn.commit()
    
    def _save_params_to_db(self, exp_id: str, params: Dict[str, Any]):
        """Save parameters to database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            for name, value in params.items():
                cursor.execute("""
                    INSERT INTO parameters 
                    (experiment_id, param_name, param_value, param_type)
                    VALUES (?, ?, ?, ?)
                """, (
                    exp_id,
                    name,
                    str(value),
                    type(value).__name__
                ))
            conn.commit()
    
    def _save_metric_to_db(self, exp_id: str, metric: Dict[str, Any]):
        """Save metric to database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO metrics 
                (experiment_id, metric_name, metric_value, step, timestamp)
                VALUES (?, ?, ?, ?, ?)
            """, (
                exp_id,
                metric['name'],
                metric['value'],
                metric.get('step'),
                metric['timestamp']
            ))
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
        with open(exp_dir / "config.json", 'r') as f:
            config = json.load(f)
        
        config['status'] = ExperimentStatus.RUNNING.value
        config['updated_at'] = datetime.now().isoformat()
        
        with open(exp_dir / "config.json", 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Started experiment: {self.experiment_id}")
        return self.tracker
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Calculate duration
        duration = (datetime.now() - self.start_time).total_seconds()
        
        # Update status
        exp_dir = self.tracker.base_dir / self.experiment_id
        with open(exp_dir / "config.json", 'r') as f:
            config = json.load(f)
        
        if exc_type is None:
            config['status'] = ExperimentStatus.COMPLETED.value
            logger.info(f"Completed experiment: {self.experiment_id}")
        else:
            config['status'] = ExperimentStatus.FAILED.value
            logger.error(f"Failed experiment: {self.experiment_id}")
            
            # Log error
            error_info = {
                'error_type': exc_type.__name__,
                'error_message': str(exc_val),
                'timestamp': datetime.now().isoformat()
            }
            
            with open(exp_dir / "error.json", 'w') as f:
                json.dump(error_info, f, indent=2)
        
        config['completed_at'] = datetime.now().isoformat()
        config['duration_seconds'] = duration
        
        with open(exp_dir / "config.json", 'w') as f:
            json.dump(config, f, indent=2)
        
        self.tracker.current_experiment = None


# Integration with MLflow-style API
class MLflowCompatibleTracker:
    """MLflow-compatible API wrapper for ExperimentTracker."""
    
    def __init__(self, tracker: ExperimentTracker):
        self.tracker = tracker
        self.active_run = None
    
    def start_run(self, run_name: str = None, tags: Dict[str, str] = None):
        """Start a new run."""
        config = ExperimentConfig(
            name=run_name or "unnamed_run",
            tags=list(tags.keys()) if tags else []
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
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, f1_score
    
    # Initialize tracker
    tracker = ExperimentTracker("example_experiments")
    
    # Create experiment
    config = ExperimentConfig(
        name="rf_classification",
        description="Random Forest classification experiment",
        tags=["classification", "random_forest"],
        model_type="RandomForestClassifier",
        dataset="synthetic",
        hyperparameters={
            "n_estimators": 100,
            "max_depth": 10,
            "random_state": 42
        },
        metrics_to_track=["accuracy", "f1_score"]
    )
    
    exp_id = tracker.create_experiment(config)
    
    # Run experiment
    with tracker.start_experiment(exp_id) as exp:
        # Generate data
        X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        # Log dataset info
        exp.log_dataset_info({
            "n_samples": len(X),
            "n_features": X.shape[1],
            "n_classes": len(np.unique(y)),
            "train_size": len(X_train),
            "test_size": len(X_test)
        })
        
        # Train model
        model = RandomForestClassifier(**config.hyperparameters)
        
        # Training loop simulation
        for epoch in range(5):
            # Fit on subset for demonstration
            subset_size = (epoch + 1) * 200
            model.fit(X_train[:subset_size], y_train[:subset_size])
            
            # Evaluate
            train_pred = model.predict(X_train[:subset_size])
            test_pred = model.predict(X_test)
            
            train_acc = accuracy_score(y_train[:subset_size], train_pred)
            test_acc = accuracy_score(y_test, test_pred)
            test_f1 = f1_score(y_test, test_pred, average='weighted')
            
            # Log metrics
            exp.log_metrics({
                "train_accuracy": train_acc,
                "test_accuracy": test_acc,
                "test_f1": test_f1
            }, step=epoch)
            
            print(f"Epoch {epoch}: Train Acc={train_acc:.3f}, Test Acc={test_acc:.3f}")
        
        # Log final model
        exp.log_model(model, "final_model.pkl")
        
        # Log additional artifacts
        feature_importance = pd.DataFrame({
            'feature': [f'feature_{i}' for i in range(X.shape[1])],
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        feature_importance.to_csv("feature_importance.csv", index=False)
        exp.log_artifact("feature_importance.csv", artifact_type="results")
    
    # Compare experiments
    print("\nExperiment Summary:")
    experiments = tracker.list_experiments()
    for exp in experiments:
        print(f"- {exp['id']}: {exp['status']}")
    
    # Plot metrics
    fig = tracker.plot_metrics(exp_id, ['test_accuracy', 'test_f1'])
    if fig:
        fig.savefig("experiment_metrics.png")
        print("\nSaved metrics plot")
    
    # MLflow-compatible API example
    mlflow_tracker = MLflowCompatibleTracker(tracker)
    
    mlflow_tracker.start_run("mlflow_style_run", tags={"framework": "sklearn"})
    mlflow_tracker.log_param("learning_rate", 0.01)
    mlflow_tracker.log_metric("loss", 0.5, step=1)
    mlflow_tracker.log_metric("loss", 0.3, step=2)
    mlflow_tracker.end_run()
    
    print("\nExperiment tracking example completed!")


if __name__ == "__main__":
    example_experiment_tracking()