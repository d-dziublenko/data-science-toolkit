"""
core/data_loader.py
Universal data loading utilities for various file formats and data sources.
"""

import pandas as pd
import numpy as np
import geopandas as gpd
from pathlib import Path
from typing import Union, Optional, List, Dict, Any, Tuple
import json
import pickle
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


class DataLoader(ABC):
    """
    Abstract base class for data loading operations.
    
    This class provides a common interface for loading different types of data
    and ensures consistent behavior across different data sources.
    """
    
    @abstractmethod
    def load(self, path: Union[str, Path], **kwargs) -> Any:
        """Load data from the specified path."""
        pass
    
    @abstractmethod
    def validate(self, data: Any) -> bool:
        """Validate the loaded data."""
        pass


class TabularDataLoader(DataLoader):
    """
    Loader for tabular data formats (CSV, Excel, Parquet, etc.).
    
    Supports automatic format detection and various preprocessing options
    like handling missing values, data type inference, and column selection.
    """
    
    SUPPORTED_FORMATS = {
        '.csv': pd.read_csv,
        '.xlsx': pd.read_excel,
        '.xls': pd.read_excel,
        '.parquet': pd.read_parquet,
        '.json': pd.read_json,
        '.feather': pd.read_feather,
        '.pkl': pd.read_pickle,
        '.pickle': pd.read_pickle
    }
    
    def __init__(self, 
                 handle_missing: bool = True,
                 missing_threshold: float = 0.5,
                 parse_dates: bool = True):
        """
        Initialize the TabularDataLoader.
        
        Args:
            handle_missing: Whether to automatically handle missing values
            missing_threshold: Maximum proportion of missing values allowed per column
            parse_dates: Whether to automatically parse date columns
        """
        self.handle_missing = handle_missing
        self.missing_threshold = missing_threshold
        self.parse_dates = parse_dates
    
    def load(self, 
             path: Union[str, Path], 
             columns: Optional[List[str]] = None,
             dtype: Optional[Dict[str, type]] = None,
             **kwargs) -> pd.DataFrame:
        """
        Load tabular data from file.
        
        Args:
            path: Path to the data file
            columns: List of columns to load (None loads all)
            dtype: Dictionary of column data types
            **kwargs: Additional arguments passed to pandas read functions
            
        Returns:
            pd.DataFrame: Loaded data
            
        Raises:
            ValueError: If file format is not supported
            FileNotFoundError: If file doesn't exist
        """
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        
        # Detect file format
        suffix = path.suffix.lower()
        if suffix not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported file format: {suffix}")
        
        # Load data
        read_func = self.SUPPORTED_FORMATS[suffix]
        
        # Handle specific format options
        if suffix == '.csv':
            kwargs.setdefault('low_memory', False)
            if self.parse_dates:
                kwargs.setdefault('parse_dates', True)
        
        logger.info(f"Loading data from {path}")
        data = read_func(path, **kwargs)
        
        # Select columns if specified
        if columns:
            missing_cols = set(columns) - set(data.columns)
            if missing_cols:
                logger.warning(f"Columns not found in data: {missing_cols}")
            data = data[[col for col in columns if col in data.columns]]
        
        # Apply dtype if specified
        if dtype:
            for col, col_dtype in dtype.items():
                if col in data.columns:
                    try:
                        data[col] = data[col].astype(col_dtype)
                    except Exception as e:
                        logger.warning(f"Could not convert {col} to {col_dtype}: {e}")
        
        # Handle missing values
        if self.handle_missing:
            data = self._handle_missing_values(data)
        
        # Validate data
        if not self.validate(data):
            raise ValueError("Data validation failed")
        
        logger.info(f"Successfully loaded data with shape {data.shape}")
        return data
    
    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        
        Removes columns with too many missing values and optionally
        fills remaining missing values.
        """
        # Calculate missing value proportions
        missing_props = data.isnull().sum() / len(data)
        
        # Remove columns with too many missing values
        cols_to_drop = missing_props[missing_props > self.missing_threshold].index
        if len(cols_to_drop) > 0:
            logger.warning(f"Dropping columns with >{self.missing_threshold*100}% missing: {list(cols_to_drop)}")
            data = data.drop(columns=cols_to_drop)
        
        return data
    
    def validate(self, data: pd.DataFrame) -> bool:
        """
        Validate the loaded DataFrame.
        
        Checks for basic requirements like non-empty data and valid shape.
        """
        if data.empty:
            logger.error("Loaded data is empty")
            return False
        
        if data.shape[1] == 0:
            logger.error("No columns in loaded data")
            return False
        
        return True


class GeospatialDataLoader(DataLoader):
    """
    Loader for geospatial data formats (Shapefile, GeoJSON, GeoPackage).
    
    Handles coordinate reference system (CRS) transformations and
    geometry validation.
    """
    
    SUPPORTED_FORMATS = ['.shp', '.geojson', '.gpkg', '.gdb']
    
    def __init__(self, target_crs: Optional[str] = None):
        """
        Initialize the GeospatialDataLoader.
        
        Args:
            target_crs: Target CRS to transform data to (e.g., 'EPSG:4326')
        """
        self.target_crs = target_crs
    
    def load(self, 
             path: Union[str, Path],
             layer: Optional[str] = None,
             bbox: Optional[Tuple[float, float, float, float]] = None,
             **kwargs) -> gpd.GeoDataFrame:
        """
        Load geospatial data from file.
        
        Args:
            path: Path to the geospatial file
            layer: Layer name for multi-layer formats
            bbox: Bounding box to filter data (minx, miny, maxx, maxy)
            **kwargs: Additional arguments passed to geopandas
            
        Returns:
            gpd.GeoDataFrame: Loaded geospatial data
        """
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        
        # Check if format is supported
        if not any(str(path).endswith(fmt) for fmt in self.SUPPORTED_FORMATS):
            raise ValueError(f"Unsupported geospatial format: {path.suffix}")
        
        logger.info(f"Loading geospatial data from {path}")
        
        # Load data
        if layer:
            gdf = gpd.read_file(path, layer=layer, **kwargs)
        else:
            gdf = gpd.read_file(path, **kwargs)
        
        # Filter by bounding box if specified
        if bbox:
            gdf = gdf.cx[bbox[0]:bbox[2], bbox[1]:bbox[3]]
        
        # Transform CRS if needed
        if self.target_crs and gdf.crs != self.target_crs:
            logger.info(f"Transforming CRS from {gdf.crs} to {self.target_crs}")
            gdf = gdf.to_crs(self.target_crs)
        
        # Validate
        if not self.validate(gdf):
            raise ValueError("Geospatial data validation failed")
        
        logger.info(f"Successfully loaded {len(gdf)} features")
        return gdf
    
    def validate(self, gdf: gpd.GeoDataFrame) -> bool:
        """Validate the loaded GeoDataFrame."""
        if gdf.empty:
            logger.error("Loaded geodata is empty")
            return False
        
        # Check for valid geometries
        invalid_geoms = gdf[~gdf.is_valid].index
        if len(invalid_geoms) > 0:
            logger.warning(f"Found {len(invalid_geoms)} invalid geometries")
            # Attempt to fix
            gdf.loc[invalid_geoms, 'geometry'] = gdf.loc[invalid_geoms].buffer(0)
        
        return True


class ModelLoader:
    """
    Loader for machine learning models.
    
    Supports various model formats including pickle, joblib, and framework-specific formats.
    """
    
    @staticmethod
    def load(path: Union[str, Path], framework: str = 'auto') -> Any:
        """
        Load a machine learning model from file.
        
        Args:
            path: Path to the model file
            framework: Model framework ('auto', 'sklearn', 'keras', 'torch', 'xgboost')
            
        Returns:
            Loaded model object
        """
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        
        if framework == 'auto':
            # Try to detect framework from file extension
            if path.suffix in ['.pkl', '.pickle']:
                framework = 'sklearn'
            elif path.suffix in ['.h5', '.keras']:
                framework = 'keras'
            elif path.suffix in ['.pt', '.pth']:
                framework = 'torch'
            elif path.suffix == '.json' and path.with_suffix('.bin').exists():
                framework = 'xgboost'
        
        logger.info(f"Loading {framework} model from {path}")
        
        if framework == 'sklearn':
            with open(path, 'rb') as f:
                model = pickle.load(f)
        
        elif framework == 'keras':
            import tensorflow as tf
            model = tf.keras.models.load_model(path)
        
        elif framework == 'torch':
            import torch
            model = torch.load(path, map_location='cpu')
        
        elif framework == 'xgboost':
            import xgboost as xgb
            model = xgb.Booster()
            model.load_model(path)
        
        else:
            raise ValueError(f"Unsupported framework: {framework}")
        
        logger.info("Model loaded successfully")
        return model


class DatasetSplitter:
    """
    Utility class for splitting datasets with various strategies.
    
    Supports random, stratified, temporal, and spatial splitting methods.
    """
    
    @staticmethod
    def split_tabular(data: pd.DataFrame,
                     target_col: str,
                     test_size: float = 0.2,
                     val_size: Optional[float] = None,
                     stratify: bool = False,
                     random_state: int = 42) -> Dict[str, pd.DataFrame]:
        """
        Split tabular data into train/test or train/val/test sets.
        
        Args:
            data: Input DataFrame
            target_col: Name of the target column
            test_size: Proportion of data for test set
            val_size: Proportion of data for validation set (optional)
            stratify: Whether to use stratified splitting
            random_state: Random seed for reproducibility
            
        Returns:
            Dictionary with 'train', 'test', and optionally 'val' DataFrames
        """
        from sklearn.model_selection import train_test_split
        
        X = data.drop(columns=[target_col])
        y = data[target_col]
        
        stratify_col = y if stratify else None
        
        if val_size:
            # First split: train+val vs test
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y, test_size=test_size, stratify=stratify_col, random_state=random_state
            )
            
            # Second split: train vs val
            val_proportion = val_size / (1 - test_size)
            stratify_temp = y_temp if stratify else None
            
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=val_proportion, 
                stratify=stratify_temp, random_state=random_state
            )
            
            return {
                'train': pd.concat([X_train, y_train], axis=1),
                'val': pd.concat([X_val, y_val], axis=1),
                'test': pd.concat([X_test, y_test], axis=1)
            }
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, stratify=stratify_col, random_state=random_state
            )
            
            return {
                'train': pd.concat([X_train, y_train], axis=1),
                'test': pd.concat([X_test, y_test], axis=1)
            }