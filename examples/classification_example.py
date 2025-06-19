"""
classification_example.py
Customer churn prediction using the Universal Data Science Toolkit
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Import toolkit components
from core import (DataPreprocessor, DataValidator, FeatureEngineer,
                  TabularDataLoader)
from evaluation import ModelEvaluator, ModelVisualizer, UncertaintyQuantifier
from models import AutoML, EnsembleModel, ModelStacker
from pipelines import ExperimentTracker, TrainingPipeline
from utils import ParallelProcessor, setup_logger

# Initialize logger
logger = setup_logger(level="INFO")


def basic_classification_example():
    """Basic classification example with minimal configuration."""
    print("\n=== Basic Classification Example ===")

    # 1. Load data
    loader = TabularDataLoader()
    data = loader.load("data/customer_churn.csv")

    # 2. Quick train with pipeline
    pipeline = TrainingPipeline(task_type="classification")
    results = pipeline.run(
        data=data,
        target_column="churn",
        test_size=0.2,
        models=["random_forest", "xgboost"],
        tune_hyperparameters=True,
        class_balance="auto",  # Handle imbalanced classes
    )

    # 3. Display results
    print(f"\nBest Model: {results['best_model']}")
    print(f"Test Accuracy: {results['test_metrics']['accuracy']:.4f}")
    print(f"Test F1 Score: {results['test_metrics']['f1']:.4f}")
    print(f"Test ROC-AUC: {results['test_metrics']['roc_auc']:.4f}")

    return results


def advanced_classification_example():
    """Advanced classification with custom preprocessing and model interpretation."""
    print("\n=== Advanced Classification Example ===")

    # 1. Initialize components
    loader = TabularDataLoader()
    preprocessor = DataPreprocessor()
    engineer = FeatureEngineer()
    validator = DataValidator()
    tracker = ExperimentTracker(experiment_name="churn_prediction")

    # 2. Load and profile data
    logger.info("Loading and profiling data...")
    data = loader.load("data/customer_churn.csv")

    # Generate data profile
    from core import DataProfiler

    profiler = DataProfiler()
    profile = profiler.profile(data)

    print("\n=== Data Profile ===")
    print(f"Shape: {profile['basic_info']['shape']}")
    print(f"Memory Usage: {profile['basic_info']['memory_usage']}")
    print(f"Missing Values: {profile['missing_values']['total_missing']}")
    print(f"Duplicate Rows: {profile['basic_info']['duplicates']}")

    # 3. Handle class imbalance
    logger.info("Checking class balance...")
    class_counts = data["churn"].value_counts()
    print(f"\nClass Distribution:")
    print(class_counts)

    imbalance_ratio = class_counts.min() / class_counts.max()
    if imbalance_ratio < 0.3:
        logger.warning(f"Severe class imbalance detected! Ratio: {imbalance_ratio:.2f}")

    # 4. Feature engineering
    logger.info("Engineering features...")

    # Create interaction features for important columns
    if all(col in data.columns for col in ["tenure", "monthly_charges"]):
        data["tenure_monthly_interaction"] = data["tenure"] * data["monthly_charges"]
        data["avg_charge_per_tenure"] = data["monthly_charges"] / (data["tenure"] + 1)

    # Create binned features
    if "age" in data.columns:
        data["age_group"] = pd.cut(
            data["age"],
            bins=[0, 25, 35, 50, 65, 100],
            labels=["<25", "25-35", "35-50", "50-65", "65+"],
        )

    # Create aggregate features
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    if "churn" in numeric_cols:
        numeric_cols.remove("churn")

    if len(numeric_cols) > 3:
        data["numeric_mean"] = data[numeric_cols].mean(axis=1)
        data["numeric_std"] = data[numeric_cols].std(axis=1)

    # 5. Preprocessing
    logger.info("Preprocessing data...")

    # Separate features and target
    target = "churn"
    features = [col for col in data.columns if col != target]

    # Handle missing values with different strategies
    data_clean = preprocessor.handle_missing_values(
        data,
        numeric_strategy="iterative",  # Advanced imputation
        categorical_strategy="mode",
        target_column=target,
    )

    # Encode categorical variables
    data_encoded = preprocessor.encode_categorical(
        data_clean,
        method="target",  # Target encoding for high cardinality
        target_column=target,
    )

    # Scale features
    data_scaled = preprocessor.scale_features(
        data_encoded, columns=features, method="standard"
    )

    # 6. Feature selection with multiple methods
    logger.info("Selecting features...")

    X = data_scaled[features]
    y = data_scaled[target]

    # Use multiple feature selection methods and combine
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.feature_selection import RFE, SelectKBest, f_classif

    # Method 1: Statistical tests
    selector_stats = SelectKBest(f_classif, k=20)
    selector_stats.fit(X, y)
    features_stats = X.columns[selector_stats.get_support()].tolist()

    # Method 2: Model-based (RFE)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    selector_rfe = RFE(rf, n_features_to_select=20)
    selector_rfe.fit(X, y)
    features_rfe = X.columns[selector_rfe.get_support()].tolist()

    # Method 3: Feature importance
    rf.fit(X, y)
    importances = pd.Series(rf.feature_importances_, index=X.columns)
    features_importance = importances.nlargest(20).index.tolist()

    # Combine features (union)
    selected_features = list(set(features_stats + features_rfe + features_importance))
    logger.info(
        f"Selected {len(selected_features)} features from {len(features)} original features"
    )

    X_selected = X[selected_features]

    # 7. Split data with stratification
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=0.2, random_state=42, stratify=y
    )

    # 8. Train multiple models with different approaches
    logger.info("Training models...")

    # Start experiment run
    with tracker.start_run(run_name="advanced_ensemble") as run:
        # Log parameters
        tracker.log_params(
            {
                "n_features": len(selected_features),
                "preprocessing": "iterative_imputation",
                "encoding": "target_encoding",
                "feature_selection": "multi_method_union",
            }
        )

        # Initialize models for ensemble
        from catboost import CatBoostClassifier
        from lightgbm import LGBMClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import SVC
        from xgboost import XGBClassifier

        base_models = [
            ("lr", LogisticRegression(class_weight="balanced", random_state=42)),
            ("svm", SVC(probability=True, class_weight="balanced", random_state=42)),
            (
                "lgb",
                LGBMClassifier(class_weight="balanced", random_state=42, verbose=-1),
            ),
            (
                "xgb",
                XGBClassifier(
                    scale_pos_weight=class_counts[0] / class_counts[1], random_state=42
                ),
            ),
            (
                "cat",
                CatBoostClassifier(
                    class_weights="Balanced", random_state=42, verbose=False
                ),
            ),
        ]

        # Create advanced stacking ensemble
        stacker = ModelStacker(
            base_models=base_models,
            meta_model=LogisticRegression(random_state=42),
            use_probabilities=True,  # Use predicted probabilities
            cv_folds=5,
            stack_method="predict_proba",
        )

        # Train with parallel processing
        processor = ParallelProcessor(n_jobs=-1)
        stacker.fit(X_train, y_train, parallel=processor)

        # 9. Make predictions with calibration
        logger.info("Making calibrated predictions...")

        from sklearn.calibration import CalibratedClassifierCV

        calibrated_stacker = CalibratedClassifierCV(stacker, method="isotonic", cv=3)
        calibrated_stacker.fit(X_train, y_train)

        # Get predictions
        y_pred = calibrated_stacker.predict(X_test)
        y_proba = calibrated_stacker.predict_proba(X_test)

        # 10. Comprehensive evaluation
        logger.info("Evaluating model...")
        evaluator = ModelEvaluator()

        # Calculate metrics
        metrics = evaluator.evaluate_classification(
            y_test,
            y_pred,
            y_proba,
            metrics=["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"],
            average="binary",
        )

        # Log metrics
        tracker.log_metrics(metrics)

        print("\n=== Model Performance ===")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")

        # Confusion matrix analysis
        from evaluation import ClassificationMetrics

        clf_metrics = ClassificationMetrics()
        cm_analysis = clf_metrics.confusion_matrix_analysis(
            y_test, y_pred, normalize="true"
        )

        print("\n=== Confusion Matrix Analysis ===")
        print(f"True Positive Rate (Recall): {cm_analysis['tpr']:.4f}")
        print(f"True Negative Rate: {cm_analysis['tnr']:.4f}")
        print(f"False Positive Rate: {cm_analysis['fpr']:.4f}")
        print(f"False Negative Rate: {cm_analysis['fnr']:.4f}")

        # 11. Model interpretation
        logger.info("Interpreting model...")
        visualizer = ModelVisualizer()

        # ROC curve with confidence intervals
        visualizer.plot_roc_curve(
            y_test, y_proba[:, 1], title="ROC Curve with 95% Confidence Interval"
        )

        # Precision-Recall curve
        visualizer.plot_precision_recall_curve(
            y_test, y_proba[:, 1], title="Precision-Recall Curve"
        )

        # Confusion matrix
        visualizer.plot_confusion_matrix(
            y_test,
            y_pred,
            labels=["No Churn", "Churn"],
            normalize="true",
            title="Normalized Confusion Matrix",
        )

        # Feature importance (averaged across all models)
        visualizer.plot_feature_importance(
            stacker,
            feature_names=selected_features,
            importance_type="permutation",
            top_n=15,
            title="Top 15 Features by Permutation Importance",
        )

        # SHAP analysis for model interpretation
        try:
            visualizer.plot_shap_analysis(
                calibrated_stacker,
                X_test.sample(min(1000, len(X_test))),  # Sample for speed
                plot_type="summary",
                max_display=20,
            )
        except:
            logger.warning("SHAP analysis not available for this model type")

        # 12. Threshold optimization
        logger.info("Optimizing decision threshold...")

        # Define cost matrix (example values)
        cost_matrix = {
            "true_positive": -50,  # Cost of retention campaign
            "false_positive": -50,  # Cost of unnecessary campaign
            "false_negative": -200,  # Cost of losing customer
            "true_negative": 0,  # No cost
        }

        from evaluation import ThresholdOptimizer

        optimizer = ThresholdOptimizer()

        optimal_threshold = optimizer.optimize_by_cost(
            y_test, y_proba[:, 1], cost_matrix
        )

        print(f"\n=== Threshold Optimization ===")
        print(f"Default Threshold: 0.5")
        print(f"Optimal Threshold: {optimal_threshold:.3f}")

        # Apply optimal threshold
        y_pred_optimal = (y_proba[:, 1] >= optimal_threshold).astype(int)
        metrics_optimal = evaluator.evaluate_classification(
            y_test, y_pred_optimal, metrics=["accuracy", "precision", "recall", "f1"]
        )

        print("\nMetrics with Optimal Threshold:")
        for metric, value in metrics_optimal.items():
            print(f"{metric}: {value:.4f}")

        # 13. Save model and artifacts
        logger.info("Saving model and artifacts...")

        # Log model
        tracker.log_model(calibrated_stacker, "calibrated_ensemble")

        # Save complete pipeline
        from utils import ModelSerializer

        serializer = ModelSerializer()

        pipeline_artifacts = {
            "model": calibrated_stacker,
            "preprocessor": preprocessor,
            "feature_selector": selected_features,
            "optimal_threshold": optimal_threshold,
            "cost_matrix": cost_matrix,
            "class_weights": dict(class_counts),
        }

        serializer.save(
            pipeline_artifacts,
            "models/churn_prediction_pipeline.pkl",
            metadata={
                "metrics": metrics,
                "metrics_optimal": metrics_optimal,
                "training_date": pd.Timestamp.now().isoformat(),
                "n_samples_train": len(X_train),
                "n_samples_test": len(X_test),
            },
        )

    return calibrated_stacker, metrics


def multiclass_classification_example():
    """Example with multiclass classification."""
    print("\n=== Multiclass Classification Example ===")

    # 1. Load data
    loader = TabularDataLoader()
    data = loader.load("data/product_categories.csv")

    # 2. Use AutoML for quick multiclass solution
    automl = AutoML(
        task_type="classification",
        optimization_metric="f1_macro",
        time_budget=600,  # 10 minutes
        ensemble=True,
    )

    # Prepare data
    X = data.drop("category", axis=1)
    y = data["category"]

    # Train
    automl.fit(X, y)

    # 3. Evaluate with multiclass metrics
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Retrain on train set
    automl.fit(X_train, y_train)
    predictions = automl.predict(X_test)
    probabilities = automl.predict_proba(X_test)

    # Evaluate
    evaluator = ModelEvaluator()
    metrics = evaluator.evaluate_classification(
        y_test,
        predictions,
        probabilities,
        metrics=["accuracy", "precision", "recall", "f1"],
        average="macro",  # For multiclass
    )

    print("\n=== Multiclass Performance ===")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    # 4. Visualize multiclass results
    visualizer = ModelVisualizer()

    # Confusion matrix
    visualizer.plot_confusion_matrix(
        y_test,
        predictions,
        labels=np.unique(y),
        normalize="true",
        title="Multiclass Confusion Matrix",
    )

    # Class-wise performance
    from sklearn.metrics import classification_report

    report = classification_report(y_test, predictions, output_dict=True)

    print("\n=== Per-Class Performance ===")
    for class_label in np.unique(y):
        if str(class_label) in report:
            class_metrics = report[str(class_label)]
            print(f"\nClass '{class_label}':")
            print(f"  Precision: {class_metrics['precision']:.4f}")
            print(f"  Recall: {class_metrics['recall']:.4f}")
            print(f"  F1-Score: {class_metrics['f1-score']:.4f}")

    return automl, metrics


if __name__ == "__main__":
    # Ensure data directory exists
    Path("data").mkdir(exist_ok=True)

    # Note: You'll need appropriate datasets in the data directory

    try:
        # Run basic example
        basic_results = basic_classification_example()

        print("\n" + "=" * 50 + "\n")

        # Run advanced example
        advanced_model, advanced_metrics = advanced_classification_example()

        print("\n" + "=" * 50 + "\n")

        # Run multiclass example
        multiclass_model, multiclass_metrics = multiclass_classification_example()

    except FileNotFoundError as e:
        print(f"\nNote: {e}")
        print("Please ensure you have the required datasets in the 'data' directory:")
        print("- customer_churn.csv (binary classification)")
        print("- product_categories.csv (multiclass classification)")
