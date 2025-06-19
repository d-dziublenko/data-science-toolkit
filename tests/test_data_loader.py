"""
test_data_loader.py
Unit tests for data loading functionality
"""

import json
import shutil
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# Handle optional dependencies gracefully
try:
    import geopandas as gpd
    from shapely.geometry import Point

    HAS_GEO = True
except ImportError:
    HAS_GEO = False
    warnings.warn("Geospatial dependencies not installed. Skipping geo tests.")

import os
# Import components to test
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.data_loader import DatasetSplitter, ModelLoader, TabularDataLoader

# Only import GeospatialDataLoader if dependencies are available
if HAS_GEO:
    from core.data_loader import GeospatialDataLoader


class TestTabularDataLoader(unittest.TestCase):
    """Test cases for TabularDataLoader class."""

    def setUp(self):
        """Set up test fixtures."""
        self.loader = TabularDataLoader()
        self.temp_dir = tempfile.mkdtemp()

        # Create sample data
        self.sample_data = pd.DataFrame(
            {
                "numeric1": np.random.randn(100),
                "numeric2": np.random.randn(100),
                "category": np.random.choice(["A", "B", "C"], 100),
                "date": pd.date_range("2024-01-01", periods=100, freq="D"),
                "target": np.random.randn(100),
            }
        )

    def tearDown(self):
        """Clean up test fixtures."""
        # Safely remove temporary directory
        try:
            shutil.rmtree(self.temp_dir)
        except Exception as e:
            warnings.warn(f"Could not remove temp directory: {e}")

    def test_csv_loading(self):
        """Test loading CSV files."""
        # Save sample data as CSV
        csv_path = Path(self.temp_dir) / "test.csv"
        self.sample_data.to_csv(csv_path, index=False)

        # Load and verify
        loaded_data = self.loader.load(csv_path)

        # Compare DataFrames (handle date parsing differences)
        pd.testing.assert_frame_equal(
            loaded_data.drop(columns=["date"]), self.sample_data.drop(columns=["date"])
        )

    def test_excel_loading(self):
        """Test loading Excel files."""
        # Skip if openpyxl not installed
        try:
            import openpyxl
        except ImportError:
            self.skipTest("openpyxl not installed")

        # Save sample data as Excel
        excel_path = Path(self.temp_dir) / "test.xlsx"
        self.sample_data.to_excel(excel_path, index=False)

        # Load and verify
        loaded_data = self.loader.load(excel_path)

        # Compare DataFrames (Excel may lose nanosecond precision in dates)
        pd.testing.assert_frame_equal(
            loaded_data.drop(columns=["date"]),
            self.sample_data.drop(columns=["date"]),
            check_dtype=False,  # Excel might change dtypes slightly
        )

    def test_parquet_loading(self):
        """Test loading Parquet files."""
        # Skip if pyarrow not installed
        try:
            import pyarrow
        except ImportError:
            self.skipTest("pyarrow not installed")

        # Save sample data as Parquet
        parquet_path = Path(self.temp_dir) / "test.parquet"
        self.sample_data.to_parquet(parquet_path)

        # Load and verify
        loaded_data = self.loader.load(parquet_path)
        pd.testing.assert_frame_equal(loaded_data, self.sample_data)

    def test_json_loading(self):
        """Test loading JSON files."""
        # Save sample data as JSON
        json_path = Path(self.temp_dir) / "test.json"
        # Convert dates to string for JSON serialization
        json_data = self.sample_data.copy()
        json_data["date"] = json_data["date"].dt.strftime("%Y-%m-%d")
        json_data.to_json(json_path, orient="records")

        # Load and verify
        loaded_data = self.loader.load(json_path)

        # Compare non-date columns
        pd.testing.assert_frame_equal(
            loaded_data.drop(columns=["date"]), self.sample_data.drop(columns=["date"])
        )

    def test_chunked_loading(self):
        """Test chunked loading for large files."""
        # Create larger dataset
        large_data = pd.DataFrame(
            {"col1": np.random.randn(10000), "col2": np.random.randn(10000)}
        )

        csv_path = Path(self.temp_dir) / "large.csv"
        large_data.to_csv(csv_path, index=False)

        # Load in chunks
        chunks = list(self.loader.load_chunked(csv_path, chunksize=1000))

        # Verify chunks
        self.assertEqual(len(chunks), 10)
        for chunk in chunks:
            self.assertLessEqual(len(chunk), 1000)

        # Verify complete data
        combined = pd.concat(chunks, ignore_index=True)
        pd.testing.assert_frame_equal(combined, large_data)

    def test_compression_handling(self):
        """Test loading compressed files."""
        # Save with gzip compression
        gz_path = Path(self.temp_dir) / "test.csv.gz"
        self.sample_data.to_csv(gz_path, index=False, compression="gzip")

        # Load and verify
        loaded_data = self.loader.load(gz_path)
        pd.testing.assert_frame_equal(
            loaded_data.drop(columns=["date"]), self.sample_data.drop(columns=["date"])
        )

    def test_column_selection(self):
        """Test loading with column selection."""
        csv_path = Path(self.temp_dir) / "test.csv"
        self.sample_data.to_csv(csv_path, index=False)

        # Load specific columns
        columns = ["numeric1", "category"]
        loaded_data = self.loader.load(csv_path, columns=columns)

        # Verify only selected columns are loaded
        self.assertEqual(list(loaded_data.columns), columns)
        pd.testing.assert_frame_equal(loaded_data, self.sample_data[columns])

    def test_parse_dates(self):
        """Test date parsing during load."""
        csv_path = Path(self.temp_dir) / "test.csv"
        self.sample_data.to_csv(csv_path, index=False)

        # Load with date parsing
        loaded_data = self.loader.load(csv_path, parse_dates=["date"])

        # Verify date column is datetime
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(loaded_data["date"]))

    def test_filters(self):
        """Test loading with filters."""
        csv_path = Path(self.temp_dir) / "test.csv"
        self.sample_data.to_csv(csv_path, index=False)

        # Load with filters
        filters = {"category": ["A", "B"]}
        loaded_data = self.loader.load(csv_path, filters=filters)

        # Verify filtering
        self.assertTrue(all(loaded_data["category"].isin(["A", "B"])))
        expected_len = len(
            self.sample_data[self.sample_data["category"].isin(["A", "B"])]
        )
        self.assertEqual(len(loaded_data), expected_len)

    def test_error_handling(self):
        """Test error handling for invalid files."""
        # Test non-existent file
        with self.assertRaises(FileNotFoundError):
            self.loader.load("non_existent.csv")

        # Test unsupported format
        with self.assertRaises(ValueError):
            self.loader.load("file.unsupported")


@unittest.skipIf(not HAS_GEO, "Geospatial dependencies not available")
class TestGeospatialDataLoader(unittest.TestCase):
    """Test cases for GeospatialDataLoader class."""

    def setUp(self):
        """Set up test fixtures."""
        self.loader = GeospatialDataLoader()
        self.temp_dir = tempfile.mkdtemp()

        # Create sample geodata
        self.sample_geodata = gpd.GeoDataFrame(
            {
                "id": range(10),
                "value": np.random.randn(10),
                "geometry": [
                    Point(x, y)
                    for x, y in zip(
                        np.random.uniform(-180, 180, 10), np.random.uniform(-90, 90, 10)
                    )
                ],
            },
            crs="EPSG:4326",
        )

    def tearDown(self):
        """Clean up test fixtures."""
        try:
            shutil.rmtree(self.temp_dir)
        except Exception as e:
            warnings.warn(f"Could not remove temp directory: {e}")

    def test_shapefile_loading(self):
        """Test loading shapefiles."""
        shp_path = Path(self.temp_dir) / "test.shp"
        self.sample_geodata.to_file(shp_path)

        # Load and verify
        loaded_data = self.loader.load(shp_path)
        self.assertIsInstance(loaded_data, gpd.GeoDataFrame)
        self.assertEqual(len(loaded_data), len(self.sample_geodata))
        self.assertIsNotNone(loaded_data.crs)

    def test_geojson_loading(self):
        """Test loading GeoJSON files."""
        json_path = Path(self.temp_dir) / "test.geojson"
        self.sample_geodata.to_file(json_path, driver="GeoJSON")

        # Load and verify
        loaded_data = self.loader.load(json_path)
        self.assertIsInstance(loaded_data, gpd.GeoDataFrame)
        self.assertEqual(len(loaded_data), len(self.sample_geodata))

    def test_crs_transformation(self):
        """Test CRS transformation during load."""
        shp_path = Path(self.temp_dir) / "test.shp"
        self.sample_geodata.to_file(shp_path)

        # Load with CRS transformation
        loaded_data = self.loader.load(shp_path, crs="EPSG:3857")

        # Verify CRS
        self.assertEqual(str(loaded_data.crs), "EPSG:3857")

    def test_bbox_filter(self):
        """Test loading with bounding box filter."""
        shp_path = Path(self.temp_dir) / "test.shp"
        self.sample_geodata.to_file(shp_path)

        # Define bounding box
        bbox = (-10, -10, 10, 10)

        # Load with bbox filter
        loaded_data = self.loader.load(shp_path, bbox=bbox)

        # Verify filtering
        bounds = loaded_data.total_bounds
        self.assertGreaterEqual(bounds[0], -10)
        self.assertLessEqual(bounds[2], 10)
        self.assertGreaterEqual(bounds[1], -10)
        self.assertLessEqual(bounds[3], 10)


class TestModelLoader(unittest.TestCase):
    """Test cases for ModelLoader class."""

    def setUp(self):
        """Set up test fixtures."""
        self.loader = ModelLoader()
        self.temp_dir = tempfile.mkdtemp()

        # Create sample model
        try:
            from sklearn.linear_model import LinearRegression

            self.sample_model = LinearRegression()
            X = np.random.randn(100, 5)
            y = np.random.randn(100)
            self.sample_model.fit(X, y)
            self.has_sklearn = True
        except ImportError:
            self.has_sklearn = False
            warnings.warn("scikit-learn not installed. Some tests will be skipped.")

    def tearDown(self):
        """Clean up test fixtures."""
        try:
            shutil.rmtree(self.temp_dir)
        except Exception as e:
            warnings.warn(f"Could not remove temp directory: {e}")

    @unittest.skipIf(
        not hasattr(unittest.TestCase, "has_sklearn")
        or not unittest.TestCase.has_sklearn,
        "scikit-learn not available",
    )
    def test_pickle_loading(self):
        """Test loading pickled models."""
        import pickle

        pkl_path = Path(self.temp_dir) / "model.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(self.sample_model, f)

        # Load and verify
        loaded_model = self.loader.load(pkl_path)
        self.assertIsNotNone(loaded_model.coef_)
        np.testing.assert_array_almost_equal(
            loaded_model.coef_, self.sample_model.coef_
        )

    @unittest.skipIf(
        not hasattr(unittest.TestCase, "has_sklearn")
        or not unittest.TestCase.has_sklearn,
        "scikit-learn not available",
    )
    def test_joblib_loading(self):
        """Test loading joblib models."""
        try:
            import joblib
        except ImportError:
            self.skipTest("joblib not installed")

        joblib_path = Path(self.temp_dir) / "model.joblib"
        joblib.dump(self.sample_model, joblib_path)

        # Load and verify
        loaded_model = self.loader.load(joblib_path)
        self.assertIsNotNone(loaded_model.coef_)
        np.testing.assert_array_almost_equal(
            loaded_model.coef_, self.sample_model.coef_
        )

    def test_metadata_loading(self):
        """Test loading models with metadata."""
        try:
            import joblib
        except ImportError:
            self.skipTest("joblib not installed")

        # Create dummy model if sklearn not available
        if not self.has_sklearn:
            model_data = {
                "model": {"type": "dummy", "params": {}},
                "metadata": {
                    "version": "1.0",
                    "training_date": "2024-01-01",
                    "metrics": {"r2": 0.85},
                },
            }
        else:
            model_data = {
                "model": self.sample_model,
                "metadata": {
                    "version": "1.0",
                    "training_date": "2024-01-01",
                    "metrics": {"r2": 0.85},
                },
            }

        joblib_path = Path(self.temp_dir) / "model_with_meta.joblib"
        joblib.dump(model_data, joblib_path)

        # Load and verify
        loaded_model, metadata = self.loader.load(joblib_path, return_metadata=True)

        self.assertIsNotNone(loaded_model)
        self.assertEqual(metadata["version"], "1.0")
        self.assertEqual(metadata["metrics"]["r2"], 0.85)


class TestDatasetSplitter(unittest.TestCase):
    """Test cases for DatasetSplitter class."""

    def setUp(self):
        """Set up test fixtures."""
        self.splitter = DatasetSplitter()

        # Create sample data
        np.random.seed(42)
        self.X = pd.DataFrame(
            {
                "feature1": np.random.randn(1000),
                "feature2": np.random.randn(1000),
                "feature3": np.random.choice(["A", "B", "C"], 1000),
            }
        )
        self.y = pd.Series(np.random.choice([0, 1], 1000), name="target")

        # Create imbalanced target
        self.y_imbalanced = pd.Series(
            np.random.choice([0, 1], 1000, p=[0.9, 0.1]), name="target"
        )

    def test_simple_split(self):
        """Test simple train/test split."""
        X_train, X_test, y_train, y_test = self.splitter.split(
            self.X, self.y, test_size=0.2
        )

        # Verify sizes
        self.assertEqual(len(X_train), 800)
        self.assertEqual(len(X_test), 200)
        self.assertEqual(len(y_train), 800)
        self.assertEqual(len(y_test), 200)

        # Verify no overlap
        train_indices = set(X_train.index)
        test_indices = set(X_test.index)
        self.assertEqual(len(train_indices & test_indices), 0)

    def test_stratified_split(self):
        """Test stratified split for classification."""
        X_train, X_test, y_train, y_test = self.splitter.split(
            self.X, self.y_imbalanced, test_size=0.2, stratify=True
        )

        # Verify class distribution is preserved
        train_dist = y_train.value_counts(normalize=True).sort_index()
        test_dist = y_test.value_counts(normalize=True).sort_index()

        # Check that distributions are similar (within 5% tolerance)
        for class_label in train_dist.index:
            self.assertAlmostEqual(
                train_dist[class_label], test_dist[class_label], delta=0.05
            )

    def test_time_series_split(self):
        """Test time series split."""
        # Add date column
        self.X["date"] = pd.date_range("2024-01-01", periods=1000, freq="D")

        X_train, X_test, y_train, y_test = self.splitter.split(
            self.X, self.y, test_size=0.2, time_series=True, date_column="date"
        )

        # Verify temporal order
        self.assertTrue(X_train["date"].max() < X_test["date"].min())

        # Verify sizes
        self.assertEqual(len(X_test), 200)
        self.assertEqual(len(X_train) + len(X_test), 1000)

    def test_validation_split(self):
        """Test train/val/test split."""
        splits = self.splitter.split_with_validation(
            self.X, self.y, test_size=0.2, val_size=0.2
        )

        X_train = splits["X_train"]
        X_val = splits["X_val"]
        X_test = splits["X_test"]
        y_train = splits["y_train"]
        y_val = splits["y_val"]
        y_test = splits["y_test"]

        # Verify sizes
        self.assertEqual(len(X_train), 600)  # 60%
        self.assertEqual(len(X_val), 200)  # 20%
        self.assertEqual(len(X_test), 200)  # 20%

        # Verify total equals original
        self.assertEqual(len(X_train) + len(X_val) + len(X_test), 1000)

        # Verify no overlap
        train_idx = set(X_train.index)
        val_idx = set(X_val.index)
        test_idx = set(X_test.index)

        self.assertEqual(len(train_idx & val_idx), 0)
        self.assertEqual(len(train_idx & test_idx), 0)
        self.assertEqual(len(val_idx & test_idx), 0)


if __name__ == "__main__":
    # Run tests with verbose output
    unittest.main(verbosity=2)
