"""
test_models.py
Unit tests for model components
"""

import unittest
import numpy as np
import pandas as pd
from sklearn.datasets import make_regression, make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score

# Import components to test
from models.base import BaseModel, AutoML, SimpleLinearModel
from models.ensemble import EnsembleModel, ModelStacker, ModelBlender
from models.neural import NeuralNetworkRegressor, NeuralNetworkClassifier
from models.transformers import (
    TargetTransformer, LogTransformer, BoxCoxTransformer,
    YeoJohnsonTransformer, AutoTargetTransformer
)


class TestBaseModel(unittest.TestCase):
    """Test cases for base model classes."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create regression data
        self.X_reg, self.y_reg = make_regression(
            n_samples=100, n_features=10, noise=0.1, random_state=42
        )
        self.X_reg = pd.DataFrame(self.X_reg, columns=[f'feat_{i}' for i in range(10)])
        self.y_reg = pd.Series(self.y_reg, name='target')
        
        # Create classification data
        self.X_clf, self.y_clf = make_classification(
            n_samples=100, n_features=10, n_classes=2, random_state=42
        )
        self.X_clf = pd.DataFrame(self.X_clf, columns=[f'feat_{i}' for i in range(10)])
        self.y_clf = pd.Series(self.y_clf, name='target')
    
    def test_simple_linear_model(self):
        """Test SimpleLinearModel."""
        model = SimpleLinearModel()
        
        # Test fitting
        model.fit(self.X_reg, self.y_reg)
        self.assertTrue(model.is_fitted)
        self.assertIsNotNone(model.coef_)
        self.assertIsNotNone(model.intercept_)
        
        # Test prediction
        predictions = model.predict(self.X_reg)
        self.assertEqual(len(predictions), len(self.y_reg))
        
        # Test scoring
        score = model.score(self.X_reg, self.y_reg)
        self.assertGreater(score, 0.5)  # Should have decent fit
    
    def test_automl_regression(self):
        """Test AutoML for regression."""
        automl = AutoML(task_type='regression')
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            self.X_reg, self.y_reg, test_size=0.2, random_state=42
        )
        
        # Fit model
        automl.fit(X_train, y_train)
        self.assertTrue(automl.is_fitted)
        self.assertIsNotNone(automl.best_model_)
        
        # Make predictions
        predictions = automl.predict(X_test)
        self.assertEqual(len(predictions), len(y_test))
        
        # Check performance
        r2 = r2_score(y_test, predictions)
        self.assertGreater(r2, 0.5)
        
        # Test search results
        results = automl.get_search_results()
        self.assertIsInstance(results, pd.DataFrame)
        self.assertGreater(len(results), 0)
    
    def test_automl_classification(self):
        """Test AutoML for classification."""
        automl = AutoML(task_type='classification')
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            self.X_clf, self.y_clf, test_size=0.2, random_state=42
        )
        
        # Fit model
        automl.fit(X_train, y_train)
        self.assertTrue(automl.is_fitted)
        self.assertIsNotNone(automl.best_model_)
        
        # Make predictions
        predictions = automl.predict(X_test)
        self.assertEqual(len(predictions), len(y_test))
        
        # Check performance
        acc = accuracy_score(y_test, predictions)
        self.assertGreater(acc, 0.5)
        
        # Test probability predictions
        proba = automl.predict_proba(X_test)
        self.assertEqual(proba.shape[0], len(y_test))
        self.assertEqual(proba.shape[1], 2)  # Binary classification
    
    def test_automl_auto_detect(self):
        """Test AutoML with automatic task detection."""
        # Should detect regression
        automl_reg = AutoML()
        automl_reg.fit(self.X_reg, self.y_reg)
        self.assertEqual(automl_reg.task_type, 'regression')
        
        # Should detect classification
        automl_clf = AutoML()
        automl_clf.fit(self.X_clf, self.y_clf)
        self.assertEqual(automl_clf.task_type, 'classification')


class TestEnsembleModels(unittest.TestCase):
    """Test cases for ensemble models."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create data
        self.X, self.y = make_regression(
            n_samples=200, n_features=10, noise=0.1, random_state=42
        )
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        
        # Create base models
        from sklearn.linear_model import LinearRegression, Ridge
        from sklearn.ensemble import RandomForestRegressor
        
        self.base_models = [
            ('lr', LinearRegression()),
            ('ridge', Ridge()),
            ('rf', RandomForestRegressor(n_estimators=10, random_state=42))
        ]
    
    def test_ensemble_model(self):
        """Test basic ensemble model."""
        ensemble = EnsembleModel(
            base_models=self.base_models,
            voting='soft',
            weights=None
        )
        
        # Fit ensemble
        ensemble.fit(self.X_train, self.y_train)
        self.assertTrue(ensemble.is_fitted)
        
        # Make predictions
        predictions = ensemble.predict(self.X_test)
        self.assertEqual(len(predictions), len(self.y_test))
        
        # Check performance
        r2 = r2_score(self.y_test, predictions)
        self.assertGreater(r2, 0.5)
    
    def test_model_stacker(self):
        from sklearn.linear_model import LinearRegression
        """Test model stacking."""
        stacker = ModelStacker(
            base_models=self.base_models,
            meta_model='linear',
            cv_folds=3
        )
        
        # Fit stacker
        stacker.fit(self.X_train, self.y_train)
        self.assertTrue(stacker.is_fitted)
        self.assertIsNotNone(stacker.meta_model_)
        
        # Make predictions
        predictions = stacker.predict(self.X_test)
        self.assertEqual(len(predictions), len(self.y_test))
        
        # Check that stacking improves performance
        # Compare with single model
        lr = LinearRegression()
        lr.fit(self.X_train, self.y_train)
        lr_predictions = lr.predict(self.X_test)
        
        stacker_r2 = r2_score(self.y_test, predictions)
        lr_r2 = r2_score(self.y_test, lr_predictions)
        
        # Stacking should perform at least as well
        self.assertGreaterEqual(stacker_r2, lr_r2 - 0.1)
    
    def test_model_blender(self):
        """Test model blending."""
        # Create validation set
        X_blend, X_val, y_blend, y_val = train_test_split(
            self.X_train, self.y_train, test_size=0.2, random_state=42
        )
        
        blender = ModelBlender(
            base_models=self.base_models,
            blend_method='weighted',
            optimization_metric='rmse'
        )
        
        # Fit blender
        blender.fit(X_blend, y_blend, X_val, y_val)
        self.assertTrue(blender.is_fitted)
        self.assertIsNotNone(blender.weights_)
        
        # Make predictions
        predictions = blender.predict(self.X_test)
        self.assertEqual(len(predictions), len(self.y_test))
        
        # Check weights sum to 1
        self.assertAlmostEqual(sum(blender.weights_), 1.0, places=5)


class TestNeuralNetworks(unittest.TestCase):
    """Test cases for neural network models."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Smaller dataset for faster testing
        self.X_reg, self.y_reg = make_regression(
            n_samples=100, n_features=5, noise=0.1, random_state=42
        )
        self.X_clf, self.y_clf = make_classification(
            n_samples=100, n_features=5, n_classes=2, random_state=42
        )
    
    def test_neural_network_regressor(self):
        """Test neural network for regression."""
        nn = NeuralNetworkRegressor(
            hidden_layers=[10, 5],
            activation='relu',
            epochs=10,
            batch_size=32,
            learning_rate=0.01,
            verbose=0
        )
        
        # Fit model
        nn.fit(self.X_reg, self.y_reg)
        self.assertTrue(nn.is_fitted)
        
        # Make predictions
        predictions = nn.predict(self.X_reg)
        self.assertEqual(len(predictions), len(self.y_reg))
        
        # Should have some predictive power
        r2 = r2_score(self.y_reg, predictions)
        self.assertGreater(r2, 0.0)
    
    def test_neural_network_classifier(self):
        """Test neural network for classification."""
        nn = NeuralNetworkClassifier(
            hidden_layers=[10, 5],
            activation='relu',
            epochs=10,
            batch_size=32,
            learning_rate=0.01,
            verbose=0
        )
        
        # Fit model
        nn.fit(self.X_clf, self.y_clf)
        self.assertTrue(nn.is_fitted)
        
        # Make predictions
        predictions = nn.predict(self.X_clf)
        self.assertEqual(len(predictions), len(self.y_clf))
        
        # Test probability predictions
        proba = nn.predict_proba(self.X_clf)
        self.assertEqual(proba.shape[0], len(self.y_clf))
        self.assertEqual(proba.shape[1], 2)
        
        # Probabilities should sum to 1
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-5)


class TestTargetTransformers(unittest.TestCase):
    """Test cases for target transformers."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create different distributions
        np.random.seed(42)
        self.y_normal = pd.Series(np.random.normal(100, 15, 1000))
        self.y_lognormal = pd.Series(np.random.lognormal(3, 1, 1000))
        self.y_skewed = pd.Series(np.random.gamma(2, 2, 1000))
        self.y_negative = pd.Series(np.random.normal(-50, 10, 1000))
    
    def test_log_transformer(self):
        """Test log transformation."""
        transformer = LogTransformer()
        
        # Fit and transform
        y_transformed = transformer.fit_transform(self.y_lognormal)
        
        # Should be more normal
        from scipy import stats
        _, p_original = stats.normaltest(self.y_lognormal)
        _, p_transformed = stats.normaltest(y_transformed)
        
        self.assertGreater(p_transformed, p_original)
        
        # Test inverse transform
        y_inverse = transformer.inverse_transform(y_transformed)
        np.testing.assert_allclose(y_inverse, self.y_lognormal, rtol=1e-10)
    
    def test_box_cox_transformer(self):
        """Test Box-Cox transformation."""
        transformer = BoxCoxTransformer()
        
        # Fit and transform (requires positive values)
        y_transformed = transformer.fit_transform(self.y_lognormal)
        
        # Should find optimal lambda
        self.assertIsNotNone(transformer.lambda_)
        
        # Test inverse transform
        y_inverse = transformer.inverse_transform(y_transformed)
        np.testing.assert_allclose(y_inverse, self.y_lognormal, rtol=1e-5)
    
    def test_yeo_johnson_transformer(self):
        """Test Yeo-Johnson transformation."""
        transformer = YeoJohnsonTransformer()
        
        # Works with negative values
        y_transformed = transformer.fit_transform(self.y_negative)
        
        # Should find optimal lambda
        self.assertIsNotNone(transformer.lambda_)
        
        # Test inverse transform
        y_inverse = transformer.inverse_transform(y_transformed)
        np.testing.assert_allclose(y_inverse, self.y_negative, rtol=1e-5)
    
    def test_auto_target_transformer(self):
        """Test automatic target transformation."""
        transformer = AutoTargetTransformer(test_normality=True)
        
        # Should automatically select best transformation
        y_transformed = transformer.fit_transform(self.y_skewed)
        
        # Should have selected a method
        self.assertIsNotNone(transformer.best_method_)
        self.assertIn(transformer.best_method_, 
                     ['none', 'log', 'sqrt', 'box-cox', 'yeo-johnson'])
        
        # Test inverse transform
        y_inverse = transformer.inverse_transform(y_transformed)
        np.testing.assert_allclose(y_inverse, self.y_skewed, rtol=1e-5)
    
    def test_target_transformer_integration(self):
        """Test target transformer with model pipeline."""
        from sklearn.pipeline import Pipeline
        from sklearn.linear_model import LinearRegression
        
        # Create pipeline with transformation
        transformer = TargetTransformer(method='box-cox')
        
        # Generate data where transformation helps
        X = np.random.randn(100, 5)
        y = np.exp(X[:, 0] + 0.5 * X[:, 1] + 0.1 * np.random.randn(100))
        
        # Fit with transformation
        y_transformed = transformer.fit_transform(y)
        
        model = LinearRegression()
        model.fit(X, y_transformed)
        
        # Predict and inverse transform
        y_pred_transformed = model.predict(X)
        y_pred = transformer.inverse_transform(y_pred_transformed)
        
        # Should have reasonable predictions
        self.assertEqual(len(y_pred), len(y))
        self.assertTrue(all(y_pred > 0))  # Exponential data is positive


if __name__ == '__main__':
    unittest.main()