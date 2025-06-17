"""
pipelines/training.py
End-to-end training pipeline for machine learning projects.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
from pathlib import Path
import pickle
import json
import logging
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Import from our modules
from core.data_loader import TabularDataLoader
from core.preprocessing import DataPreprocessor, FeatureTransformer
from core.feature_engineering import FeatureSelector, FeatureEngineer
from models.ensemble import EnsembleModel, ModelStacker, ModelBlender
from evaluation.metrics import ModelEvaluator
from evaluation.visualization import ModelVisualizer

logger = logging.getLogger(__name__)


class TrainingPipeline:
    """
    Complete training pipeline for machine learning projects.
    
    Handles data loading, preprocessing, feature engineering,
    model training, evaluation, and artifact saving.
    """
    
    def __init__(self,
                 task_type: str = 'regression',
                 experiment_name: str = 'experiment',
                 output_dir: str = './outputs'):
        """
        Initialize the TrainingPipeline.
        
        Args:
            task_type: Type of ML task ('regression' or 'classification')
            experiment_name: Name for this experiment
            output_dir: Directory to save outputs
        """
        self.task_type = task_type
        self.experiment_name = experiment_name
        self.output_dir = Path(output_dir) / experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.data_loader = TabularDataLoader()
        self.preprocessor = DataPreprocessor()
        self.feature_selector = FeatureSelector(task_type)
        self.feature_engineer = FeatureEngineer()
        self.evaluator = ModelEvaluator(task_type)
        self.visualizer = ModelVisualizer()
        
        # Storage for pipeline artifacts
        self.data = {}
        self.models = {}
        self.metrics = {}
        self.config = {
            'task_type': task_type,
            'experiment_name': experiment_name,
            'timestamp': datetime.now().isoformat()
        }
    
    def load_data(self,
                  data_path: Union[str, Path],
                  target_column: str,
                  test_size: float = 0.2,
                  val_size: Optional[float] = None,
                  stratify: bool = False,
                  **kwargs) -> 'TrainingPipeline':
        """
        Load and split data.
        
        Args:
            data_path: Path to data file
            target_column: Name of target column
            test_size: Proportion for test set
            val_size: Proportion for validation set
            stratify: Whether to use stratified splitting
            **kwargs: Additional arguments for data loading
            
        Returns:
            Self for method chaining
        """
        logger.info(f"Loading data from {data_path}")
        
        # Load data
        data = self.data_loader.load(data_path, **kwargs)
        
        # Separate features and target
        X = data.drop(columns=[target_column])
        y = data[target_column]
        
        # Store column information
        self.config['features'] = X.columns.tolist()
        self.config['target'] = target_column
        self.config['n_samples'] = len(data)
        self.config['n_features'] = X.shape[1]
        
        # Split data
        if stratify and self.task_type == 'classification':
            stratify_col = y
        else:
            stratify_col = None
        
        if val_size:
            # Create train/val/test split
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y, test_size=test_size, stratify=stratify_col, random_state=42
            )
            
            val_prop = val_size / (1 - test_size)
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=val_prop, 
                stratify=y_temp if stratify else None, random_state=42
            )
            
            self.data = {
                'X_train': X_train, 'y_train': y_train,
                'X_val': X_val, 'y_val': y_val,
                'X_test': X_test, 'y_test': y_test
            }
        else:
            # Create train/test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, stratify=stratify_col, random_state=42
            )
            
            self.data = {
                'X_train': X_train, 'y_train': y_train,
                'X_test': X_test, 'y_test': y_test
            }
        
        logger.info(f"Data split: Train={len(X_train)}, Test={len(X_test)}")
        if val_size:
            logger.info(f"Validation={len(X_val)}")
        
        return self
    
    def preprocess(self,
                   numeric_features: Optional[List[str]] = None,
                   categorical_features: Optional[List[str]] = None,
                   scaling_method: str = 'standard',
                   encoding_method: str = 'onehot',
                   handle_missing: str = 'impute',
                   remove_outliers: bool = False) -> 'TrainingPipeline':
        """
        Preprocess the data.
        
        Args:
            numeric_features: List of numeric features (auto-detected if None)
            categorical_features: List of categorical features (auto-detected if None)
            scaling_method: Method for scaling numeric features
            encoding_method: Method for encoding categorical features
            handle_missing: Method for handling missing values
            remove_outliers: Whether to remove outliers
            
        Returns:
            Self for method chaining
        """
        logger.info("Preprocessing data")
        
        X_train = self.data['X_train']
        
        # Auto-detect feature types if not provided
        if numeric_features is None:
            numeric_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
        if categorical_features is None:
            categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
        
        self.config['numeric_features'] = numeric_features
        self.config['categorical_features'] = categorical_features
        
        # Apply preprocessing to each dataset
        for key in ['X_train', 'X_val', 'X_test']:
            if key not in self.data:
                continue
            
            X = self.data[key]
            
            # Handle missing values
            X = self.preprocessor.handle_missing_values(X, method=handle_missing)
            
            # Remove outliers (training set only)
            if remove_outliers and key == 'X_train':
                X = self.preprocessor.remove_outliers(X, columns=numeric_features)
            
            # Scale numeric features
            if numeric_features:
                X = self.preprocessor.scale_features(X, columns=numeric_features, method=scaling_method)
            
            # Encode categorical features
            if categorical_features:
                X = self.preprocessor.encode_categorical(X, columns=categorical_features, method=encoding_method)
            
            self.data[key] = X
        
        # Update feature names after encoding
        self.config['features_after_preprocessing'] = self.data['X_train'].columns.tolist()
        
        return self
    
    def engineer_features(self,
                          create_polynomials: bool = False,
                          create_interactions: bool = False,
                          select_features: bool = True,
                          selection_method: str = 'mutual_information',
                          n_features: Optional[int] = None) -> 'TrainingPipeline':
        """
        Perform feature engineering.
        
        Args:
            create_polynomials: Whether to create polynomial features
            create_interactions: Whether to create interaction features
            select_features: Whether to perform feature selection
            selection_method: Method for feature selection
            n_features: Number of features to select (None keeps all)
            
        Returns:
            Self for method chaining
        """
        logger.info("Engineering features")
        
        # Apply feature engineering to training set
        X_train = self.data['X_train']
        y_train = self.data['y_train']
        
        if create_polynomials:
            numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                X_train = FeatureTransformer.create_polynomial_features(
                    X_train, numeric_cols[:5], degree=2  # Limit to first 5 features
                )
        
        if create_interactions:
            numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) >= 2:
                # Create interactions for top correlated features
                correlations = X_train[numeric_cols].corr().abs()
                np.fill_diagonal(correlations.values, 0)
                top_pairs = []
                for i in range(min(3, len(numeric_cols))):
                    for j in range(i+1, min(4, len(numeric_cols))):
                        top_pairs.append((numeric_cols[i], numeric_cols[j]))
                
                X_train = FeatureTransformer.create_interaction_features(X_train, top_pairs)
        
        # Update other datasets with same transformations
        new_features = [col for col in X_train.columns if col not in self.data['X_train'].columns]
        
        if new_features:
            self.data['X_train'] = X_train
            
            # Apply same transformations to validation and test sets
            for key in ['X_val', 'X_test']:
                if key in self.data:
                    # This is simplified - in production, you'd save the transformation parameters
                    logger.warning(f"Feature engineering on {key} set is simplified")
        
        # Feature selection
        if select_features:
            if n_features is None:
                n_features = min(50, X_train.shape[1])
            
            if selection_method == 'mutual_information':
                selected = self.feature_selector.select_by_mutual_information(
                    X_train, y_train, n_features
                )
            elif selection_method == 'model_importance':
                selected = self.feature_selector.select_by_model_importance(
                    X_train, y_train, n_features=n_features
                )
            elif selection_method == 'rfe':
                selected = self.feature_selector.select_by_rfe(
                    X_train, y_train, n_features=n_features
                )
            else:
                raise ValueError(f"Unknown selection method: {selection_method}")
            
            # Apply selection to all datasets
            for key in ['X_train', 'X_val', 'X_test']:
                if key in self.data:
                    self.data[key] = self.data[key][selected]
            
            self.config['selected_features'] = selected
            self.config['n_selected_features'] = len(selected)
        
        return self
    
    def train_models(self,
                     model_types: List[str] = ['auto'],
                     tune_hyperparameters: bool = True,
                     ensemble_method: Optional[str] = None) -> 'TrainingPipeline':
        """
        Train machine learning models.
        
        Args:
            model_types: List of model types to train
            tune_hyperparameters: Whether to tune hyperparameters
            ensemble_method: Ensemble method ('stacking', 'blending', None)
            
        Returns:
            Self for method chaining
        """
        logger.info("Training models")
        
        X_train = self.data['X_train']
        y_train = self.data['y_train']
        
        # Train individual models
        for model_type in model_types:
            logger.info(f"Training {model_type} model")
            
            model = EnsembleModel(
                task_type=self.task_type,
                model_type=model_type
            )
            
            model.fit(
                X_train, y_train,
                tune_hyperparameters=tune_hyperparameters,
                cv=5
            )
            
            self.models[model_type] = model
        
        # Create ensemble if requested
        if ensemble_method and len(self.models) > 1:
            logger.info(f"Creating {ensemble_method} ensemble")
            
            base_models = [(name, model.model) for name, model in self.models.items()]
            
            if ensemble_method == 'stacking':
                ensemble = ModelStacker(
                    base_models=base_models,
                    task_type=self.task_type
                )
                ensemble.fit(X_train, y_train)
                self.models['ensemble_stacking'] = ensemble
            
            elif ensemble_method == 'blending':
                ensemble = ModelBlender(models=base_models)
                ensemble.fit(X_train, y_train)
                
                # Optimize weights if validation set available
                if 'X_val' in self.data:
                    ensemble.optimize_weights(self.data['X_val'], self.data['y_val'])
                
                self.models['ensemble_blending'] = ensemble
        
        return self
    
    def evaluate(self,
                 create_plots: bool = True,
                 save_plots: bool = True) -> 'TrainingPipeline':
        """
        Evaluate trained models.
        
        Args:
            create_plots: Whether to create evaluation plots
            save_plots: Whether to save plots to disk
            
        Returns:
            Self for method chaining
        """
        logger.info("Evaluating models")
        
        X_test = self.data['X_test']
        y_test = self.data['y_test']
        
        # Evaluate each model
        for model_name, model in self.models.items():
            logger.info(f"Evaluating {model_name}")
            
            # Make predictions
            if hasattr(model, 'predict'):
                y_pred = model.predict(X_test)
            else:
                y_pred = model.model.predict(X_test)
            
            # Calculate metrics
            metrics = self.evaluator.evaluate(y_test, y_pred, model_name)
            self.metrics[model_name] = metrics
            
            # Print report
            self.evaluator.print_report(metrics)
            
            # Create plots
            if create_plots:
                plot_dir = self.output_dir / 'plots' / model_name
                plot_dir.mkdir(parents=True, exist_ok=True)
                
                if self.task_type == 'regression':
                    self.evaluator.plot_predictions(
                        y_test, y_pred,
                        title=f'{model_name} - Predictions vs Actual',
                        save_path=str(plot_dir / 'predictions.png') if save_plots else None
                    )
                    
                    self.evaluator.plot_residuals(
                        y_test, y_pred,
                        save_path=str(plot_dir / 'residuals.png') if save_plots else None
                    )
                
                # Feature importance for tree-based models
                if hasattr(model, 'feature_importances_') or (hasattr(model, 'model') and hasattr(model.model, 'feature_importances_')):
                    feature_names = X_test.columns.tolist()
                    self.visualizer.plot_feature_importance(
                        model.model if hasattr(model, 'model') else model,
                        feature_names,
                        save_path=str(plot_dir / 'feature_importance.png') if save_plots else None
                    )
        
        # Compare models
        if len(self.models) > 1:
            self._compare_models()
        
        return self
    
    def _compare_models(self):
        """Compare performance of different models."""
        comparison_df = pd.DataFrame(self.metrics).T
        
        if self.task_type == 'regression':
            comparison_df = comparison_df[['rmse', 'mae', 'r2']].round(4)
        else:
            comparison_df = comparison_df[['accuracy', 'precision', 'recall', 'f1']].round(4)
        
        logger.info("\nModel Comparison:")
        print(comparison_df)
        
        # Save comparison
        comparison_df.to_csv(self.output_dir / 'model_comparison.csv')
    
    def save_artifacts(self) -> 'TrainingPipeline':
        """
        Save all pipeline artifacts.
        
        Returns:
            Self for method chaining
        """
        logger.info(f"Saving artifacts to {self.output_dir}")
        
        # Save models
        models_dir = self.output_dir / 'models'
        models_dir.mkdir(exist_ok=True)
        
        for model_name, model in self.models.items():
            model_path = models_dir / f'{model_name}.pkl'
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            logger.info(f"Saved model: {model_name}")
        
        # Save preprocessor
        with open(self.output_dir / 'preprocessor.pkl', 'wb') as f:
            pickle.dump(self.preprocessor, f)
        
        # Save configuration
        with open(self.output_dir / 'config.json', 'w') as f:
            json.dump(self.config, f, indent=2)
        
        # Save metrics
        metrics_df = pd.DataFrame(self.metrics).T
        metrics_df.to_csv(self.output_dir / 'metrics.csv')
        
        # Save feature importance if available
        if hasattr(self.feature_selector, 'feature_scores') and self.feature_selector.feature_scores:
            importance_df = pd.DataFrame(
                list(self.feature_selector.feature_scores.items()),
                columns=['feature', 'score']
            ).sort_values('score', ascending=False)
            importance_df.to_csv(self.output_dir / 'feature_importance.csv', index=False)
        
        logger.info("All artifacts saved successfully")
        
        return self
    
    def run(self,
            data_path: Union[str, Path],
            target_column: str,
            **kwargs) -> Dict[str, Any]:
        """
        Run the complete training pipeline.
        
        Args:
            data_path: Path to data file
            target_column: Name of target column
            **kwargs: Additional configuration options
            
        Returns:
            Dictionary with results
        """
        logger.info(f"Starting training pipeline: {self.experiment_name}")
        
        try:
            # Execute pipeline steps
            self.load_data(data_path, target_column, **kwargs.get('data_config', {}))
            self.preprocess(**kwargs.get('preprocessing_config', {}))
            self.engineer_features(**kwargs.get('feature_engineering_config', {}))
            self.train_models(**kwargs.get('training_config', {}))
            self.evaluate(**kwargs.get('evaluation_config', {}))
            self.save_artifacts()
            
            # Prepare results
            results = {
                'experiment_name': self.experiment_name,
                'best_model': min(self.metrics.items(), key=lambda x: x[1].get('rmse', float('inf')))[0] if self.task_type == 'regression' else max(self.metrics.items(), key=lambda x: x[1].get('accuracy', 0))[0],
                'metrics': self.metrics,
                'config': self.config,
                'output_dir': str(self.output_dir)
            }
            
            logger.info(f"Pipeline completed successfully. Best model: {results['best_model']}")
            
            return results
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise