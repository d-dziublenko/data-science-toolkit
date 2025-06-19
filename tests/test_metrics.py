"""
test_metrics.py
Unit tests for evaluation metrics and visualization
"""

import os
import tempfile
import unittest

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split

from evaluation.drift import (DataDriftDetector, DriftType,
                              StatisticalDriftDetector)
# Import components to test
from evaluation.metrics import (ClassificationMetrics, CustomMetric,
                                MetricCalculator, ModelEvaluator,
                                RegressionMetrics)
from evaluation.uncertainty import (BootstrapUncertainty, PredictionInterval,
                                    UncertaintyQuantifier)
from evaluation.visualization import (ModelVisualizer, PerformancePlotter,
                                      PlotConfig)


class TestRegressionMetrics(unittest.TestCase):
    """Test cases for regression metrics."""

    def setUp(self):
        """Set up test fixtures."""
        # Create perfect predictions and various error patterns
        self.y_true = np.array([1, 2, 3, 4, 5])
        self.y_perfect = np.array([1, 2, 3, 4, 5])
        self.y_biased = np.array([2, 3, 4, 5, 6])  # Constant bias
        self.y_noisy = np.array([1.1, 1.9, 3.2, 3.8, 5.1])  # Small noise

        self.metrics = RegressionMetrics()

    def test_rmse(self):
        """Test RMSE calculation."""
        # Perfect predictions
        rmse_perfect = self.metrics.rmse(self.y_true, self.y_perfect)
        self.assertEqual(rmse_perfect, 0.0)

        # Biased predictions
        rmse_biased = self.metrics.rmse(self.y_true, self.y_biased)
        self.assertAlmostEqual(rmse_biased, 1.0, places=5)

    def test_mae(self):
        """Test MAE calculation."""
        # Perfect predictions
        mae_perfect = self.metrics.mae(self.y_true, self.y_perfect)
        self.assertEqual(mae_perfect, 0.0)

        # Biased predictions
        mae_biased = self.metrics.mae(self.y_true, self.y_biased)
        self.assertEqual(mae_biased, 1.0)

    def test_r2(self):
        """Test R² calculation."""
        # Perfect predictions
        r2_perfect = self.metrics.r2(self.y_true, self.y_perfect)
        self.assertEqual(r2_perfect, 1.0)

        # Test negative R² for bad predictions
        y_bad = np.array([10, 10, 10, 10, 10])
        r2_bad = self.metrics.r2(self.y_true, y_bad)
        self.assertLess(r2_bad, 0)

    def test_mape(self):
        """Test MAPE calculation."""
        y_true = np.array([100, 200, 300, 400, 500])
        y_pred = np.array([110, 190, 310, 390, 510])

        mape = self.metrics.mape(y_true, y_pred)
        expected_mape = np.mean([10 / 100, 10 / 200, 10 / 300, 10 / 400, 10 / 500])
        self.assertAlmostEqual(mape, expected_mape, places=5)

    def test_calculate_all(self):
        """Test calculating all metrics at once."""
        all_metrics = self.metrics.calculate_all(self.y_true, self.y_noisy)

        # Check all expected metrics are present
        expected_metrics = [
            "rmse",
            "mae",
            "r2",
            "mape",
            "explained_variance",
            "max_error",
            "median_absolute_error",
        ]
        for metric in expected_metrics:
            self.assertIn(metric, all_metrics)

        # Check values are reasonable
        self.assertGreater(all_metrics["r2"], 0.9)  # Should be close
        self.assertLess(all_metrics["rmse"], 0.5)  # Small error


class TestClassificationMetrics(unittest.TestCase):
    """Test cases for classification metrics."""

    def setUp(self):
        """Set up test fixtures."""
        # Binary classification
        self.y_true_binary = np.array([0, 0, 1, 1, 0, 1, 0, 1])
        self.y_pred_binary = np.array([0, 1, 1, 1, 0, 0, 0, 1])
        self.y_proba_binary = np.array(
            [
                [0.9, 0.1],
                [0.4, 0.6],
                [0.2, 0.8],
                [0.1, 0.9],
                [0.8, 0.2],
                [0.6, 0.4],
                [0.7, 0.3],
                [0.3, 0.7],
            ]
        )

        # Multiclass classification
        self.y_true_multi = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
        self.y_pred_multi = np.array([0, 2, 2, 0, 1, 1, 0, 1, 2])

        self.metrics = ClassificationMetrics()

    def test_binary_metrics(self):
        """Test binary classification metrics."""
        results = self.metrics.evaluate_binary(
            self.y_true_binary, self.y_pred_binary, self.y_proba_binary
        )

        # Check expected metrics
        expected_metrics = [
            "accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "pr_auc",
            "specificity",
        ]
        for metric in expected_metrics:
            self.assertIn(metric, results)

        # Manually calculate accuracy
        correct = np.sum(self.y_true_binary == self.y_pred_binary)
        expected_acc = correct / len(self.y_true_binary)
        self.assertAlmostEqual(results["accuracy"], expected_acc, places=5)

    def test_multiclass_metrics(self):
        """Test multiclass classification metrics."""
        results = self.metrics.evaluate_multiclass(
            self.y_true_multi,
            self.y_pred_multi,
            None,  # No probabilities
            average="macro",
        )

        # Check expected metrics
        self.assertIn("accuracy", results)
        self.assertIn("precision", results)
        self.assertIn("recall", results)
        self.assertIn("f1", results)

        # Accuracy should be reasonable
        self.assertGreater(results["accuracy"], 0.5)

    def test_confusion_matrix_analysis(self):
        """Test confusion matrix analysis."""
        cm_analysis = self.metrics.confusion_matrix_analysis(
            self.y_true_binary, self.y_pred_binary, normalize="true"
        )

        # Check components
        self.assertIn("matrix", cm_analysis)
        self.assertIn("tpr", cm_analysis)  # True positive rate
        self.assertIn("tnr", cm_analysis)  # True negative rate
        self.assertIn("fpr", cm_analysis)  # False positive rate
        self.assertIn("fnr", cm_analysis)  # False negative rate

        # TPR + FNR should equal 1
        self.assertAlmostEqual(cm_analysis["tpr"] + cm_analysis["fnr"], 1.0, places=5)


class TestModelEvaluator(unittest.TestCase):
    """Test cases for ModelEvaluator."""

    def setUp(self):
        """Set up test fixtures."""
        self.evaluator = ModelEvaluator()

        # Create sample data
        X_reg, y_reg = make_regression(n_samples=100, n_features=10, random_state=42)
        X_clf, y_clf = make_classification(
            n_samples=100, n_features=10, random_state=42
        )

        # Split data
        self.X_train_reg, self.X_test_reg, self.y_train_reg, self.y_test_reg = (
            train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
        )

        self.X_train_clf, self.X_test_clf, self.y_train_clf, self.y_test_clf = (
            train_test_split(X_clf, y_clf, test_size=0.2, random_state=42)
        )

        # Train models
        self.reg_model = RandomForestRegressor(n_estimators=10, random_state=42)
        self.reg_model.fit(self.X_train_reg, self.y_train_reg)

        self.clf_model = RandomForestClassifier(n_estimators=10, random_state=42)
        self.clf_model.fit(self.X_train_clf, self.y_train_clf)

    def test_evaluate_regression(self):
        """Test regression evaluation."""
        y_pred = self.reg_model.predict(self.X_test_reg)

        metrics = self.evaluator.evaluate_regression(
            self.y_test_reg, y_pred, metrics=["rmse", "mae", "r2"]
        )

        # Check all requested metrics are present
        self.assertIn("rmse", metrics)
        self.assertIn("mae", metrics)
        self.assertIn("r2", metrics)

        # R² should be positive for decent model
        self.assertGreater(metrics["r2"], 0)

    def test_evaluate_classification(self):
        """Test classification evaluation."""
        y_pred = self.clf_model.predict(self.X_test_clf)
        y_proba = self.clf_model.predict_proba(self.X_test_clf)

        metrics = self.evaluator.evaluate_classification(
            self.y_test_clf, y_pred, y_proba, metrics=["accuracy", "f1", "roc_auc"]
        )

        # Check all requested metrics are present
        self.assertIn("accuracy", metrics)
        self.assertIn("f1", metrics)
        self.assertIn("roc_auc", metrics)

        # Should have reasonable performance
        self.assertGreater(metrics["accuracy"], 0.5)

    def test_cross_validate(self):
        """Test cross-validation evaluation."""
        cv_scores = self.evaluator.cross_validate(
            self.reg_model,
            self.X_train_reg,
            self.y_train_reg,
            cv=3,
            scoring=["r2", "neg_mean_squared_error"],
        )

        # Check structure
        self.assertIn("test_r2", cv_scores)
        self.assertIn("test_neg_mean_squared_error", cv_scores)
        self.assertIn("fit_time", cv_scores)
        self.assertIn("score_time", cv_scores)

        # Should have 3 scores (3-fold CV)
        self.assertEqual(len(cv_scores["test_r2"]), 3)

        # Mean R² should be positive
        self.assertGreater(np.mean(cv_scores["test_r2"]), 0)


class TestModelVisualizer(unittest.TestCase):
    """Test cases for ModelVisualizer."""

    def setUp(self):
        """Set up test fixtures."""
        self.visualizer = ModelVisualizer()
        self.temp_dir = tempfile.mkdtemp()

        # Create sample data and model
        X, y = make_regression(n_samples=100, n_features=5, random_state=42)
        self.feature_names = [f"feature_{i}" for i in range(5)]
        X = pd.DataFrame(X, columns=self.feature_names)

        self.model = RandomForestRegressor(n_estimators=10, random_state=42)
        self.model.fit(X, y)

        self.X_test = X.iloc[:20]
        self.y_true = y[:20]
        self.y_pred = self.model.predict(self.X_test)

    def tearDown(self):
        """Clean up."""
        plt.close("all")
        import shutil

        shutil.rmtree(self.temp_dir)

    def test_plot_predictions(self):
        """Test prediction plotting."""
        # Should not raise any errors
        self.visualizer.plot_predictions(self.y_true, self.y_pred, plot_type="scatter")

        # Check plot was created
        self.assertEqual(len(plt.get_fignums()), 1)
        plt.close("all")

    def test_plot_feature_importance(self):
        """Test feature importance plotting."""
        self.visualizer.plot_feature_importance(self.model, self.feature_names, top_n=3)

        # Check plot was created
        self.assertEqual(len(plt.get_fignums()), 1)
        plt.close("all")

    def test_plot_residuals(self):
        """Test residual plotting."""
        self.visualizer.plot_residuals(self.y_true, self.y_pred, plot_type="scatter")

        # Check plot was created
        self.assertEqual(len(plt.get_fignums()), 1)
        plt.close("all")

    def test_save_plots(self):
        """Test saving plots to file."""
        save_path = os.path.join(self.temp_dir, "test_plot.png")

        self.visualizer.plot_predictions(self.y_true, self.y_pred, save_path=save_path)

        # Check file was created
        self.assertTrue(os.path.exists(save_path))

        # Check file is not empty
        self.assertGreater(os.path.getsize(save_path), 0)


class TestUncertaintyQuantification(unittest.TestCase):
    """Test cases for uncertainty quantification."""

    def setUp(self):
        """Set up test fixtures."""
        self.uq = UncertaintyQuantifier(task_type="regression")

        # Create data
        X, y = make_regression(n_samples=100, n_features=5, random_state=42)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Train model
        self.model = RandomForestRegressor(n_estimators=10, random_state=42)
        self.model.fit(self.X_train, self.y_train)

    def test_bootstrap_uncertainty(self):
        """Test bootstrap uncertainty estimation."""
        results = self.uq.bootstrap_uncertainty(
            self.model,
            self.X_train,
            self.y_train,
            self.X_test,
            n_bootstrap=10,  # Small number for testing
            confidence_level=0.95,
        )

        # Check structure
        self.assertIn("predictions", results)
        self.assertIn("lower_bound", results)
        self.assertIn("upper_bound", results)
        self.assertIn("std", results)

        # Check shapes
        self.assertEqual(len(results["predictions"]), len(self.X_test))
        self.assertEqual(len(results["lower_bound"]), len(self.X_test))
        self.assertEqual(len(results["upper_bound"]), len(self.X_test))

        # Lower bound should be less than upper bound
        self.assertTrue(all(results["lower_bound"] < results["upper_bound"]))

    def test_ensemble_uncertainty(self):
        """Test ensemble-based uncertainty."""
        # Create multiple models
        models = [
            RandomForestRegressor(n_estimators=10, random_state=i) for i in range(5)
        ]

        # Train models
        for model in models:
            model.fit(self.X_train, self.y_train)

        # Get uncertainty
        results = self.uq.ensemble_uncertainty(
            models, self.X_test, confidence_level=0.95
        )

        # Check structure
        self.assertIn("predictions", results)
        self.assertIn("std", results)
        self.assertIn("confidence", results)

        # For regression, confidence intervals should be present
        self.assertIn("lower_bound", results)
        self.assertIn("upper_bound", results)


class TestDataDriftDetection(unittest.TestCase):
    """Test cases for data drift detection."""

    def setUp(self):
        """Set up test fixtures."""
        self.detector = DataDriftDetector()

        # Create reference and test data
        np.random.seed(42)
        self.reference_data = pd.DataFrame(
            {
                "feature1": np.random.normal(0, 1, 1000),
                "feature2": np.random.normal(5, 2, 1000),
                "category": np.random.choice(["A", "B", "C"], 1000),
            }
        )

        # Test data with no drift
        self.test_data_no_drift = pd.DataFrame(
            {
                "feature1": np.random.normal(0, 1, 200),
                "feature2": np.random.normal(5, 2, 200),
                "category": np.random.choice(["A", "B", "C"], 200),
            }
        )

        # Test data with drift
        self.test_data_drift = pd.DataFrame(
            {
                "feature1": np.random.normal(2, 1, 200),  # Mean shifted
                "feature2": np.random.normal(5, 4, 200),  # Variance increased
                "category": np.random.choice(
                    ["A", "B", "C"], 200, p=[0.6, 0.3, 0.1]
                ),  # Distribution changed
            }
        )

    def test_detect_drift_no_drift(self):
        """Test drift detection when there's no drift."""
        drift_results = self.detector.detect_drift(
            self.reference_data, self.test_data_no_drift
        )

        # Check structure
        self.assertIn("overall_drift", drift_results)
        self.assertIn("feature_drift", drift_results)
        self.assertIn("drift_scores", drift_results)

        # Should not detect drift
        self.assertFalse(drift_results["overall_drift"])

    def test_detect_drift_with_drift(self):
        """Test drift detection when there is drift."""
        drift_results = self.detector.detect_drift(
            self.reference_data, self.test_data_drift
        )

        # Should detect drift
        self.assertTrue(drift_results["overall_drift"])

        # Feature1 should show drift
        self.assertTrue(drift_results["feature_drift"]["feature1"]["drift_detected"])

    def test_statistical_tests(self):
        """Test individual statistical tests."""
        stat_detector = StatisticalDriftDetector()

        # Test numeric drift
        ref_numeric = self.reference_data["feature1"].values
        test_numeric_drift = self.test_data_drift["feature1"].values

        drift_score = stat_detector.test_numeric_drift(
            ref_numeric, test_numeric_drift, method="ks"
        )

        # Should have high drift score
        self.assertGreater(drift_score["statistic"], 0.1)
        self.assertLess(drift_score["p_value"], 0.05)

        # Test categorical drift
        ref_cat = self.reference_data["category"].values
        test_cat_drift = self.test_data_drift["category"].values

        cat_drift_score = stat_detector.test_categorical_drift(
            ref_cat, test_cat_drift, method="chi2"
        )

        # Should detect drift
        self.assertLess(cat_drift_score["p_value"], 0.05)


if __name__ == "__main__":
    unittest.main()
