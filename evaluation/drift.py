"""
evaluation/drift.py
Data drift detection methods for monitoring model performance over time.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
import warnings
import logging
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


class DriftType(Enum):
    """Enumeration of drift types."""
    NO_DRIFT = "no_drift"
    WARNING = "warning"
    DRIFT = "drift"


@dataclass
class DriftResult:
    """Container for drift detection results."""
    drift_score: float
    drift_type: DriftType
    p_value: Optional[float] = None
    threshold: Optional[float] = None
    feature_scores: Optional[Dict[str, float]] = None
    metadata: Optional[Dict[str, Any]] = None


class BaseDriftDetector(ABC):
    """Abstract base class for drift detectors."""
    
    @abstractmethod
    def detect_drift(self, *args, **kwargs) -> DriftResult:
        """Detect drift in data."""
        pass
    
    @abstractmethod
    def fit(self, reference_data: pd.DataFrame):
        """Fit the detector on reference data."""
        pass


class StatisticalDriftDetector(BaseDriftDetector):
    """
    Statistical tests-based drift detector.
    
    This class specializes in using various statistical tests to detect
    distribution changes between reference and current data.
    """
    
    def __init__(self, 
                 test_method: str = "ks",
                 confidence_level: float = 0.95,
                 correction_method: str = "bonferroni"):
        """
        Initialize the statistical drift detector.
        
        Args:
            test_method: Statistical test to use ('ks', 'chi2', 'anderson', 'cramervonmises')
            confidence_level: Confidence level for tests
            correction_method: Multiple testing correction ('bonferroni', 'fdr_bh', 'none')
        """
        self.test_method = test_method
        self.confidence_level = confidence_level
        self.correction_method = correction_method
        self.reference_data = None
        self.feature_types = {}
        self.is_fitted = False
    
    def fit(self, reference_data: pd.DataFrame, 
            feature_types: Optional[Dict[str, str]] = None):
        """
        Fit the detector on reference data.
        
        Args:
            reference_data: Reference dataset
            feature_types: Dictionary mapping column names to types ('numerical', 'categorical')
        """
        self.reference_data = reference_data.copy()
        
        # Infer feature types if not provided
        if feature_types is None:
            self.feature_types = self._infer_feature_types(reference_data)
        else:
            self.feature_types = feature_types
        
        # Compute reference statistics for efficiency
        self._compute_reference_stats()
        self.is_fitted = True
        logger.info(f"Fitted statistical drift detector with {len(self.reference_data.columns)} features")
    
    def _infer_feature_types(self, data: pd.DataFrame) -> Dict[str, str]:
        """Automatically infer feature types from data."""
        feature_types = {}
        for col in data.columns:
            if pd.api.types.is_numeric_dtype(data[col]):
                # Check if it's actually categorical (few unique values)
                if data[col].nunique() < 10:
                    feature_types[col] = 'categorical'
                else:
                    feature_types[col] = 'numerical'
            else:
                feature_types[col] = 'categorical'
        return feature_types
    
    def _compute_reference_stats(self):
        """Pre-compute reference statistics for efficiency."""
        self.reference_stats = {
            'numerical': {},
            'categorical': {}
        }
        
        for col, ftype in self.feature_types.items():
            if ftype == 'numerical':
                self.reference_stats['numerical'][col] = {
                    'mean': self.reference_data[col].mean(),
                    'std': self.reference_data[col].std(),
                    'values': self.reference_data[col].dropna().values
                }
            else:
                value_counts = self.reference_data[col].value_counts(normalize=True)
                self.reference_stats['categorical'][col] = value_counts.to_dict()
    
    def detect_drift(self, current_data: pd.DataFrame) -> DriftResult:
        """
        Detect drift between reference and current data.
        
        Args:
            current_data: Current dataset to test
            
        Returns:
            DriftResult with test results
        """
        if not self.is_fitted:
            raise ValueError("Detector must be fitted before detecting drift")
        
        # Run appropriate test based on method
        if self.test_method == "ks":
            return self._kolmogorov_smirnov_test(current_data)
        elif self.test_method == "chi2":
            return self._chi_squared_test(current_data)
        elif self.test_method == "anderson":
            return self._anderson_darling_test(current_data)
        elif self.test_method == "cramervonmises":
            return self._cramer_von_mises_test(current_data)
        else:
            raise ValueError(f"Unknown test method: {self.test_method}")
    
    def _kolmogorov_smirnov_test(self, current_data: pd.DataFrame) -> DriftResult:
        """Two-sample Kolmogorov-Smirnov test for numerical features."""
        p_values = {}
        statistics = {}
        
        for col in self.reference_data.columns:
            if self.feature_types.get(col) == 'numerical':
                ref_values = self.reference_stats['numerical'][col]['values']
                curr_values = current_data[col].dropna().values
                
                statistic, p_value = stats.ks_2samp(ref_values, curr_values)
                p_values[col] = p_value
                statistics[col] = statistic
        
        # Apply multiple testing correction
        if self.correction_method != 'none' and p_values:
            p_values = self._apply_correction(p_values)
        
        # Determine overall drift
        alpha = 1 - self.confidence_level
        drift_detected = any(p < alpha for p in p_values.values())
        
        if drift_detected:
            drift_type = DriftType.DRIFT
        elif any(p < alpha * 2 for p in p_values.values()):
            drift_type = DriftType.WARNING
        else:
            drift_type = DriftType.NO_DRIFT
        
        return DriftResult(
            drift_score=np.mean(list(statistics.values())) if statistics else 0.0,
            drift_type=drift_type,
            p_value=min(p_values.values()) if p_values else 1.0,
            threshold=alpha,
            feature_scores=statistics,
            metadata={'p_values': p_values, 'test': 'kolmogorov_smirnov'}
        )
    
    def _chi_squared_test(self, current_data: pd.DataFrame) -> DriftResult:
        """Chi-squared test for categorical features."""
        p_values = {}
        statistics = {}
        
        for col in self.reference_data.columns:
            if self.feature_types.get(col) == 'categorical':
                # Get frequency distributions
                ref_dist = self.reference_stats['categorical'][col]
                curr_counts = current_data[col].value_counts()
                
                # Align categories
                all_categories = set(ref_dist.keys()) | set(curr_counts.index)
                ref_freq = np.array([ref_dist.get(cat, 0) * len(self.reference_data) 
                                   for cat in all_categories])
                curr_freq = np.array([curr_counts.get(cat, 0) for cat in all_categories])
                
                # Chi-squared test
                if np.sum(ref_freq) > 0 and np.sum(curr_freq) > 0:
                    # Scale expected frequencies
                    expected = ref_freq * (np.sum(curr_freq) / np.sum(ref_freq))
                    statistic, p_value = stats.chisquare(curr_freq, f_exp=expected)
                    p_values[col] = p_value
                    statistics[col] = statistic
        
        # Apply correction and determine drift
        if self.correction_method != 'none' and p_values:
            p_values = self._apply_correction(p_values)
        
        alpha = 1 - self.confidence_level
        drift_detected = any(p < alpha for p in p_values.values()) if p_values else False
        
        if drift_detected:
            drift_type = DriftType.DRIFT
        elif p_values and any(p < alpha * 2 for p in p_values.values()):
            drift_type = DriftType.WARNING
        else:
            drift_type = DriftType.NO_DRIFT
        
        return DriftResult(
            drift_score=np.mean(list(statistics.values())) if statistics else 0.0,
            drift_type=drift_type,
            p_value=min(p_values.values()) if p_values else 1.0,
            threshold=alpha,
            feature_scores=statistics,
            metadata={'p_values': p_values, 'test': 'chi_squared'}
        )
    
    def _anderson_darling_test(self, current_data: pd.DataFrame) -> DriftResult:
        """Anderson-Darling test for distribution comparison."""
        statistics = {}
        critical_values = {}
        
        for col in self.reference_data.columns:
            if self.feature_types.get(col) == 'numerical':
                ref_values = self.reference_stats['numerical'][col]['values']
                curr_values = current_data[col].dropna().values
                
                # Anderson-Darling k-sample test
                try:
                    from scipy.stats import anderson_ksamp
                    result = anderson_ksamp([ref_values, curr_values])
                    statistics[col] = result.statistic
                    critical_values[col] = result.critical_values[2]  # 5% significance
                except ImportError:
                    logger.warning("anderson_ksamp not available, using KS test as fallback")
                    stat, _ = stats.ks_2samp(ref_values, curr_values)
                    statistics[col] = stat
                    critical_values[col] = 0.1  # Default threshold
        
        # Determine drift based on critical values
        drift_features = sum(1 for col, stat in statistics.items() 
                           if stat > critical_values.get(col, np.inf))
        
        drift_ratio = drift_features / len(statistics) if statistics else 0
        
        if drift_ratio > 0.3:
            drift_type = DriftType.DRIFT
        elif drift_ratio > 0.1:
            drift_type = DriftType.WARNING
        else:
            drift_type = DriftType.NO_DRIFT
        
        return DriftResult(
            drift_score=np.mean(list(statistics.values())) if statistics else 0.0,
            drift_type=drift_type,
            threshold=0.3,
            feature_scores=statistics,
            metadata={'critical_values': critical_values, 'test': 'anderson_darling'}
        )
    
    def _cramer_von_mises_test(self, current_data: pd.DataFrame) -> DriftResult:
        """Cramér-von Mises test for distribution comparison."""
        statistics = {}
        
        for col in self.reference_data.columns:
            if self.feature_types.get(col) == 'numerical':
                ref_values = self.reference_stats['numerical'][col]['values']
                curr_values = current_data[col].dropna().values
                
                # Compute Cramér-von Mises statistic
                statistic = self._compute_cvm_statistic(ref_values, curr_values)
                statistics[col] = statistic
        
        # Use empirical thresholds
        drift_threshold = 0.5
        warning_threshold = 0.25
        
        avg_statistic = np.mean(list(statistics.values())) if statistics else 0.0
        
        if avg_statistic > drift_threshold:
            drift_type = DriftType.DRIFT
        elif avg_statistic > warning_threshold:
            drift_type = DriftType.WARNING
        else:
            drift_type = DriftType.NO_DRIFT
        
        return DriftResult(
            drift_score=avg_statistic,
            drift_type=drift_type,
            threshold=drift_threshold,
            feature_scores=statistics,
            metadata={'test': 'cramer_von_mises'}
        )
    
    def _compute_cvm_statistic(self, x: np.ndarray, y: np.ndarray) -> float:
        """Compute two-sample Cramér-von Mises statistic."""
        n, m = len(x), len(y)
        
        # Combine and sort
        combined = np.concatenate([x, y])
        combined_sorted = np.sort(combined)
        
        # Compute empirical CDFs
        x_sorted = np.sort(x)
        y_sorted = np.sort(y)
        
        # Calculate statistic
        statistic = 0
        for value in combined_sorted:
            F_x = np.searchsorted(x_sorted, value, side='right') / n
            F_y = np.searchsorted(y_sorted, value, side='right') / m
            statistic += (F_x - F_y) ** 2
        
        statistic *= n * m / (n + m)
        return statistic
    
    def _apply_correction(self, p_values: Dict[str, float]) -> Dict[str, float]:
        """Apply multiple testing correction."""
        if self.correction_method == 'bonferroni':
            n_tests = len(p_values)
            return {k: min(v * n_tests, 1.0) for k, v in p_values.items()}
        elif self.correction_method == 'fdr_bh':
            # Benjamini-Hochberg procedure
            from statsmodels.stats.multitest import multipletests
            keys = list(p_values.keys())
            values = list(p_values.values())
            _, corrected_values, _, _ = multipletests(values, method='fdr_bh')
            return dict(zip(keys, corrected_values))
        else:
            return p_values
    
    def run_all_tests(self, current_data: pd.DataFrame) -> Dict[str, DriftResult]:
        """Run all available statistical tests."""
        results = {}
        original_method = self.test_method
        
        for method in ['ks', 'chi2', 'anderson', 'cramervonmises']:
            try:
                self.test_method = method
                results[method] = self.detect_drift(current_data)
            except Exception as e:
                logger.warning(f"Failed to run {method} test: {str(e)}")
        
        self.test_method = original_method
        return results


class DistanceDriftDetector(BaseDriftDetector):
    """
    Distance-based drift detector.
    
    This class uses various distance metrics to measure the difference
    between reference and current data distributions.
    """
    
    def __init__(self,
                 distance_metric: str = "wasserstein",
                 threshold: Optional[float] = None,
                 n_bins: int = 50):
        """
        Initialize the distance-based drift detector.
        
        Args:
            distance_metric: Distance metric to use ('wasserstein', 'kl', 'js', 'hellinger', 'bhattacharyya')
            threshold: Threshold for drift detection (auto-determined if None)
            n_bins: Number of bins for histogram-based metrics
        """
        self.distance_metric = distance_metric
        self.threshold = threshold
        self.n_bins = n_bins
        self.reference_data = None
        self.reference_histograms = {}
        self.is_fitted = False
    
    def fit(self, reference_data: pd.DataFrame):
        """
        Fit the detector on reference data.
        
        Args:
            reference_data: Reference dataset
        """
        self.reference_data = reference_data.copy()
        
        # Pre-compute histograms for efficiency
        self._compute_reference_histograms()
        
        # Auto-determine threshold if not provided
        if self.threshold is None:
            self.threshold = self._determine_threshold()
        
        self.is_fitted = True
        logger.info(f"Fitted distance drift detector with {self.distance_metric} metric")
    
    def _compute_reference_histograms(self):
        """Pre-compute reference histograms for histogram-based metrics."""
        if self.distance_metric in ['kl', 'js', 'hellinger', 'bhattacharyya']:
            for col in self.reference_data.select_dtypes(include=[np.number]).columns:
                values = self.reference_data[col].dropna()
                hist, bin_edges = np.histogram(values, bins=self.n_bins, density=True)
                self.reference_histograms[col] = {
                    'hist': hist,
                    'bin_edges': bin_edges,
                    'min': values.min(),
                    'max': values.max()
                }
    
    def _determine_threshold(self) -> float:
        """Auto-determine threshold based on metric and data characteristics."""
        thresholds = {
            'wasserstein': 0.1,
            'kl': 0.5,
            'js': 0.3,
            'hellinger': 0.3,
            'bhattacharyya': 0.3
        }
        return thresholds.get(self.distance_metric, 0.3)
    
    def detect_drift(self, current_data: pd.DataFrame) -> DriftResult:
        """
        Detect drift using distance metrics.
        
        Args:
            current_data: Current dataset to test
            
        Returns:
            DriftResult with distance measurements
        """
        if not self.is_fitted:
            raise ValueError("Detector must be fitted before detecting drift")
        
        # Calculate distances based on metric
        if self.distance_metric == "wasserstein":
            return self._wasserstein_distance(current_data)
        elif self.distance_metric == "kl":
            return self._kl_divergence(current_data)
        elif self.distance_metric == "js":
            return self._js_divergence(current_data)
        elif self.distance_metric == "hellinger":
            return self._hellinger_distance(current_data)
        elif self.distance_metric == "bhattacharyya":
            return self._bhattacharyya_distance(current_data)
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")
    
    def _wasserstein_distance(self, current_data: pd.DataFrame) -> DriftResult:
        """Calculate Wasserstein distance (Earth Mover's Distance)."""
        distances = {}
        
        for col in self.reference_data.select_dtypes(include=[np.number]).columns:
            if col in current_data.columns:
                ref_values = self.reference_data[col].dropna().values
                curr_values = current_data[col].dropna().values
                
                # Calculate 1D Wasserstein distance
                distance = stats.wasserstein_distance(ref_values, curr_values)
                
                # Normalize by standard deviation for scale invariance
                std = self.reference_data[col].std()
                if std > 0:
                    distance = distance / std
                
                distances[col] = distance
        
        # Overall score
        avg_distance = np.mean(list(distances.values())) if distances else 0.0
        
        # Determine drift type
        if avg_distance > self.threshold:
            drift_type = DriftType.DRIFT
        elif avg_distance > self.threshold * 0.7:
            drift_type = DriftType.WARNING
        else:
            drift_type = DriftType.NO_DRIFT
        
        return DriftResult(
            drift_score=avg_distance,
            drift_type=drift_type,
            threshold=self.threshold,
            feature_scores=distances,
            metadata={'metric': 'wasserstein', 'normalized': True}
        )
    
    def _kl_divergence(self, current_data: pd.DataFrame) -> DriftResult:
        """Calculate Kullback-Leibler divergence."""
        divergences = {}
        
        for col, ref_hist_data in self.reference_histograms.items():
            if col in current_data.columns:
                # Create histogram for current data with same bins
                curr_values = current_data[col].dropna()
                curr_hist, _ = np.histogram(
                    curr_values,
                    bins=ref_hist_data['bin_edges'],
                    density=True
                )
                
                # Add small epsilon to avoid log(0)
                epsilon = 1e-10
                ref_hist = ref_hist_data['hist'] + epsilon
                curr_hist = curr_hist + epsilon
                
                # Normalize
                ref_hist = ref_hist / ref_hist.sum()
                curr_hist = curr_hist / curr_hist.sum()
                
                # KL divergence
                kl_div = np.sum(ref_hist * np.log(ref_hist / curr_hist))
                divergences[col] = kl_div
        
        # Overall score
        avg_divergence = np.mean(list(divergences.values())) if divergences else 0.0
        
        # Determine drift type
        if avg_divergence > self.threshold:
            drift_type = DriftType.DRIFT
        elif avg_divergence > self.threshold * 0.5:
            drift_type = DriftType.WARNING
        else:
            drift_type = DriftType.NO_DRIFT
        
        return DriftResult(
            drift_score=avg_divergence,
            drift_type=drift_type,
            threshold=self.threshold,
            feature_scores=divergences,
            metadata={'metric': 'kl_divergence'}
        )
    
    def _js_divergence(self, current_data: pd.DataFrame) -> DriftResult:
        """Calculate Jensen-Shannon divergence."""
        divergences = {}
        
        for col, ref_hist_data in self.reference_histograms.items():
            if col in current_data.columns:
                # Create histogram for current data
                curr_values = current_data[col].dropna()
                curr_hist, _ = np.histogram(
                    curr_values,
                    bins=ref_hist_data['bin_edges'],
                    density=True
                )
                
                # Add epsilon and normalize
                epsilon = 1e-10
                ref_hist = ref_hist_data['hist'] + epsilon
                curr_hist = curr_hist + epsilon
                ref_hist = ref_hist / ref_hist.sum()
                curr_hist = curr_hist / curr_hist.sum()
                
                # JS divergence (symmetric version of KL)
                m = 0.5 * (ref_hist + curr_hist)
                js_div = 0.5 * np.sum(ref_hist * np.log(ref_hist / m)) + \
                         0.5 * np.sum(curr_hist * np.log(curr_hist / m))
                divergences[col] = js_div
        
        # Overall score
        avg_divergence = np.mean(list(divergences.values())) if divergences else 0.0
        
        # Determine drift type
        if avg_divergence > self.threshold:
            drift_type = DriftType.DRIFT
        elif avg_divergence > self.threshold * 0.6:
            drift_type = DriftType.WARNING
        else:
            drift_type = DriftType.NO_DRIFT
        
        return DriftResult(
            drift_score=avg_divergence,
            drift_type=drift_type,
            threshold=self.threshold,
            feature_scores=divergences,
            metadata={'metric': 'js_divergence'}
        )
    
    def _hellinger_distance(self, current_data: pd.DataFrame) -> DriftResult:
        """Calculate Hellinger distance."""
        distances = {}
        
        for col, ref_hist_data in self.reference_histograms.items():
            if col in current_data.columns:
                # Create histogram for current data
                curr_values = current_data[col].dropna()
                curr_hist, _ = np.histogram(
                    curr_values,
                    bins=ref_hist_data['bin_edges'],
                    density=True
                )
                
                # Normalize
                ref_hist = ref_hist_data['hist'] / ref_hist_data['hist'].sum()
                curr_hist = curr_hist / curr_hist.sum()
                
                # Hellinger distance
                h_dist = np.sqrt(0.5 * np.sum((np.sqrt(ref_hist) - np.sqrt(curr_hist))**2))
                distances[col] = h_dist
        
        # Overall score
        avg_distance = np.mean(list(distances.values())) if distances else 0.0
        
        # Determine drift type
        if avg_distance > self.threshold:
            drift_type = DriftType.DRIFT
        elif avg_distance > self.threshold * 0.6:
            drift_type = DriftType.WARNING
        else:
            drift_type = DriftType.NO_DRIFT
        
        return DriftResult(
            drift_score=avg_distance,
            drift_type=drift_type,
            threshold=self.threshold,
            feature_scores=distances,
            metadata={'metric': 'hellinger'}
        )
    
    def _bhattacharyya_distance(self, current_data: pd.DataFrame) -> DriftResult:
        """Calculate Bhattacharyya distance."""
        distances = {}
        
        for col, ref_hist_data in self.reference_histograms.items():
            if col in current_data.columns:
                # Create histogram for current data
                curr_values = current_data[col].dropna()
                curr_hist, _ = np.histogram(
                    curr_values,
                    bins=ref_hist_data['bin_edges'],
                    density=True
                )
                
                # Normalize
                ref_hist = ref_hist_data['hist'] / ref_hist_data['hist'].sum()
                curr_hist = curr_hist / curr_hist.sum()
                
                # Bhattacharyya coefficient
                bc = np.sum(np.sqrt(ref_hist * curr_hist))
                
                # Bhattacharyya distance
                b_dist = -np.log(bc) if bc > 0 else np.inf
                distances[col] = b_dist
        
        # Overall score
        avg_distance = np.mean([d for d in distances.values() if d != np.inf]) if distances else 0.0
        
        # Determine drift type
        if avg_distance > self.threshold:
            drift_type = DriftType.DRIFT
        elif avg_distance > self.threshold * 0.6:
            drift_type = DriftType.WARNING
        else:
            drift_type = DriftType.NO_DRIFT
        
        return DriftResult(
            drift_score=avg_distance,
            drift_type=drift_type,
            threshold=self.threshold,
            feature_scores=distances,
            metadata={'metric': 'bhattacharyya'}
        )
    
    def compare_distances(self, current_data: pd.DataFrame) -> pd.DataFrame:
        """Compare multiple distance metrics."""
        results = {}
        original_metric = self.distance_metric
        
        for metric in ['wasserstein', 'kl', 'js', 'hellinger', 'bhattacharyya']:
            try:
                self.distance_metric = metric
                result = self.detect_drift(current_data)
                results[metric] = {
                    'drift_score': result.drift_score,
                    'drift_type': result.drift_type.value,
                    'threshold': result.threshold
                }
            except Exception as e:
                logger.warning(f"Failed to compute {metric}: {str(e)}")
        
        self.distance_metric = original_metric
        return pd.DataFrame(results).T


class ModelPerformanceDriftDetector(BaseDriftDetector):
    """
    Model performance-based drift detector.
    
    This class monitors changes in model performance metrics to detect
    when a model's predictions are becoming less reliable.
    """
    
    def __init__(self,
                 model: Any,
                 metric: Union[str, Callable] = "auto",
                 threshold_drop: float = 0.1,
                 window_size: int = 100):
        """
        Initialize the performance drift detector.
        
        Args:
            model: Trained model to monitor
            metric: Performance metric ('accuracy', 'f1', 'rmse', 'r2', or callable)
            threshold_drop: Relative drop in performance to trigger drift alert
            window_size: Size of sliding window for performance calculation
        """
        self.model = model
        self.metric = metric
        self.threshold_drop = threshold_drop
        self.window_size = window_size
        self.reference_performance = None
        self.performance_history = []
        self.is_fitted = False
    
    def fit(self, X_reference: pd.DataFrame, y_reference: pd.Series):
        """
        Fit the detector by establishing baseline performance.
        
        Args:
            X_reference: Reference features
            y_reference: Reference targets
        """
        # Make predictions
        y_pred = self.model.predict(X_reference)
        
        # Calculate reference performance
        self.reference_performance = self._calculate_metric(y_reference, y_pred)
        
        # Determine task type if metric is 'auto'
        if self.metric == "auto":
            if hasattr(self.model, 'predict_proba'):
                self.metric = 'accuracy'
            else:
                self.metric = 'rmse'
        
        self.is_fitted = True
        logger.info(f"Fitted performance drift detector with reference {self.metric}: {self.reference_performance:.4f}")
    
    def _calculate_metric(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate the specified performance metric."""
        if callable(self.metric):
            return self.metric(y_true, y_pred)
        elif self.metric == 'accuracy':
            return accuracy_score(y_true, y_pred)
        elif self.metric == 'f1':
            return f1_score(y_true, y_pred, average='weighted')
        elif self.metric == 'rmse':
            return -mean_squared_error(y_true, y_pred, squared=False)  # Negative for consistency
        elif self.metric == 'r2':
            return r2_score(y_true, y_pred)
        else:
            raise ValueError(f"Unknown metric: {self.metric}")
    
    def detect_drift(self, X_current: pd.DataFrame, y_current: pd.Series) -> DriftResult:
        """
        Detect performance-based drift.
        
        Args:
            X_current: Current features
            y_current: Current true labels
            
        Returns:
            DriftResult with performance analysis
        """
        if not self.is_fitted:
            raise ValueError("Detector must be fitted before detecting drift")
        
        # Make predictions
        y_pred = self.model.predict(X_current)
        
        # Calculate current performance
        current_performance = self._calculate_metric(y_current, y_pred)
        
        # Add to history
        self.performance_history.append({
            'timestamp': len(self.performance_history),
            'performance': current_performance
        })
        
        # Calculate performance drop
        performance_drop = (self.reference_performance - current_performance) / abs(self.reference_performance)
        
        # Calculate moving average if enough history
        if len(self.performance_history) >= self.window_size:
            recent_performances = [h['performance'] for h in self.performance_history[-self.window_size:]]
            moving_avg = np.mean(recent_performances)
            moving_drop = (self.reference_performance - moving_avg) / abs(self.reference_performance)
        else:
            moving_drop = performance_drop
        
        # Determine drift type
        if moving_drop > self.threshold_drop:
            drift_type = DriftType.DRIFT
        elif moving_drop > self.threshold_drop * 0.5:
            drift_type = DriftType.WARNING
        else:
            drift_type = DriftType.NO_DRIFT
        
        return DriftResult(
            drift_score=moving_drop,
            drift_type=drift_type,
            threshold=self.threshold_drop,
            metadata={
                'current_performance': current_performance,
                'reference_performance': self.reference_performance,
                'performance_drop': performance_drop,
                'moving_drop': moving_drop,
                'metric': self.metric if isinstance(self.metric, str) else 'custom'
            }
        )
    
    def get_performance_trend(self) -> pd.DataFrame:
        """Get performance history as DataFrame."""
        if not self.performance_history:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.performance_history)
        df['performance_drop'] = (self.reference_performance - df['performance']) / abs(self.reference_performance)
        
        # Add moving average
        if len(df) >= self.window_size:
            df['moving_avg'] = df['performance'].rolling(window=self.window_size).mean()
            df['moving_drop'] = (self.reference_performance - df['moving_avg']) / abs(self.reference_performance)
        
        return df
    
    def plot_performance_trend(self, save_path: Optional[str] = None):
        """Plot performance trend over time."""
        trend_df = self.get_performance_trend()
        
        if trend_df.empty:
            logger.warning("No performance history to plot")
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        
        # Performance over time
        ax1.plot(trend_df['timestamp'], trend_df['performance'], 
                label='Current Performance', marker='o', alpha=0.7)
        ax1.axhline(y=self.reference_performance, color='green', 
                   linestyle='--', label='Reference Performance')
        
        if 'moving_avg' in trend_df.columns:
            ax1.plot(trend_df['timestamp'], trend_df['moving_avg'], 
                    label=f'Moving Average ({self.window_size})', linewidth=2)
        
        ax1.set_ylabel(f'Performance ({self.metric})')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Performance drop over time
        ax2.plot(trend_df['timestamp'], trend_df['performance_drop'] * 100, 
                label='Performance Drop', marker='o', alpha=0.7)
        
        if 'moving_drop' in trend_df.columns:
            ax2.plot(trend_df['timestamp'], trend_df['moving_drop'] * 100, 
                    label=f'Moving Drop ({self.window_size})', linewidth=2)
        
        ax2.axhline(y=self.threshold_drop * 100, color='red', 
                   linestyle='--', label='Drift Threshold')
        ax2.axhline(y=self.threshold_drop * 50, color='orange', 
                   linestyle='--', label='Warning Threshold')
        
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Performance Drop (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle('Model Performance Monitoring')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Performance trend plot saved to {save_path}")
        
        plt.show()
    
    def reset_history(self):
        """Reset performance history."""
        self.performance_history = []
        logger.info("Performance history reset")


class DataDriftDetector:
    """
    Comprehensive data drift detection for machine learning pipelines.
    
    This class provides multiple statistical and distance-based methods
    to detect distribution shifts between reference and current datasets.
    """
    
    def __init__(self, 
                 reference_data: pd.DataFrame,
                 feature_columns: Optional[List[str]] = None,
                 categorical_columns: Optional[List[str]] = None):
        """
        Initialize the DataDriftDetector.
        
        Args:
            reference_data: Reference dataset (e.g., training data)
            feature_columns: List of feature columns to monitor (None for all)
            categorical_columns: List of categorical columns
        """
        self.reference_data = reference_data
        self.feature_columns = feature_columns or reference_data.columns.tolist()
        self.categorical_columns = categorical_columns or []
        self.numerical_columns = [col for col in self.feature_columns 
                                 if col not in self.categorical_columns]
        
        # Preprocessing
        self._fit_preprocessors()
        
        # Store reference statistics
        self._compute_reference_statistics()
        
        logger.info(f"Initialized drift detector with {len(self.feature_columns)} features")
    
    def _fit_preprocessors(self):
        """Fit preprocessing transformers on reference data."""
        # Scaler for numerical features
        if self.numerical_columns:
            self.scaler = StandardScaler()
            self.scaler.fit(self.reference_data[self.numerical_columns])
        
        # Store categorical mappings
        self.categorical_mappings = {}
        for col in self.categorical_columns:
            unique_values = self.reference_data[col].unique()
            self.categorical_mappings[col] = {val: i for i, val in enumerate(unique_values)}
    
    def _compute_reference_statistics(self):
        """Compute and store reference data statistics."""
        self.reference_stats = {
            'mean': self.reference_data[self.numerical_columns].mean(),
            'std': self.reference_data[self.numerical_columns].std(),
            'median': self.reference_data[self.numerical_columns].median(),
            'quantiles': {}
        }
        
        # Compute quantiles
        for q in [0.25, 0.5, 0.75]:
            self.reference_stats['quantiles'][q] = self.reference_data[self.numerical_columns].quantile(q)
        
        # Categorical distributions
        self.reference_categorical_dist = {}
        for col in self.categorical_columns:
            value_counts = self.reference_data[col].value_counts(normalize=True)
            self.reference_categorical_dist[col] = value_counts.to_dict()
    
    def detect_drift(self,
                    current_data: pd.DataFrame,
                    method: str = 'ks',
                    confidence_level: float = 0.95,
                    return_feature_scores: bool = True) -> DriftResult:
        """
        Detect drift between reference and current data.
        
        Args:
            current_data: Current dataset to check for drift
            method: Detection method ('ks', 'chi2', 'mmd', 'psi', 'wasserstein')
            confidence_level: Confidence level for statistical tests
            return_feature_scores: Whether to return individual feature scores
            
        Returns:
            DriftResult object with detection results
        """
        logger.info(f"Detecting drift using {method} method")
        
        # Validate input
        self._validate_input(current_data)
        
        # Select detection method
        detection_methods = {
            'ks': self._kolmogorov_smirnov_test,
            'chi2': self._chi_squared_test,
            'mmd': self._maximum_mean_discrepancy,
            'psi': self._population_stability_index,
            'wasserstein': self._wasserstein_distance,
            'jensen_shannon': self._jensen_shannon_divergence,
            'hellinger': self._hellinger_distance
        }
        
        if method not in detection_methods:
            raise ValueError(f"Unknown method: {method}. Choose from {list(detection_methods.keys())}")
        
        # Run detection
        result = detection_methods[method](current_data, confidence_level, return_feature_scores)
        
        return result
    
    def _validate_input(self, current_data: pd.DataFrame):
        """Validate input data consistency."""
        # Check columns
        missing_cols = set(self.feature_columns) - set(current_data.columns)
        if missing_cols:
            raise ValueError(f"Missing columns in current data: {missing_cols}")
        
        # Check data types
        for col in self.numerical_columns:
            if not pd.api.types.is_numeric_dtype(current_data[col]):
                raise TypeError(f"Column {col} should be numeric")
    
    def _kolmogorov_smirnov_test(self,
                                 current_data: pd.DataFrame,
                                 confidence_level: float,
                                 return_feature_scores: bool) -> DriftResult:
        """
        Kolmogorov-Smirnov test for numerical features.
        
        The KS test compares the empirical distribution functions
        of two samples.
        """
        alpha = 1 - confidence_level
        p_values = {}
        ks_stats = {}
        
        # Test each numerical feature
        for col in self.numerical_columns:
            ref_values = self.reference_data[col].dropna()
            curr_values = current_data[col].dropna()
            
            # KS test
            ks_stat, p_value = stats.ks_2samp(ref_values, curr_values)
            
            p_values[col] = p_value
            ks_stats[col] = ks_stat
        
        # Overall drift score (using Bonferroni correction)
        adjusted_alpha = alpha / len(self.numerical_columns)
        drift_detected = any(p < adjusted_alpha for p in p_values.values())
        
        # Average KS statistic as overall score
        overall_score = np.mean(list(ks_stats.values()))
        
        # Determine drift type
        if drift_detected:
            drift_type = DriftType.DRIFT
        elif any(p < alpha for p in p_values.values()):
            drift_type = DriftType.WARNING
        else:
            drift_type = DriftType.NO_DRIFT
        
        return DriftResult(
            drift_score=overall_score,
            drift_type=drift_type,
            p_value=min(p_values.values()) if p_values else None,
            threshold=adjusted_alpha,
            feature_scores=ks_stats if return_feature_scores else None,
            metadata={'p_values': p_values}
        )
    
    def _chi_squared_test(self,
                         current_data: pd.DataFrame,
                         confidence_level: float,
                         return_feature_scores: bool) -> DriftResult:
        """
        Chi-squared test for categorical features.
        
        Tests if the observed frequencies differ significantly
        from expected frequencies.
        """
        if not self.categorical_columns:
            logger.warning("No categorical columns found for chi-squared test")
            return DriftResult(
                drift_score=0.0,
                drift_type=DriftType.NO_DRIFT,
                p_value=1.0
            )
        
        alpha = 1 - confidence_level
        p_values = {}
        chi2_stats = {}
        
        # Test each categorical feature
        for col in self.categorical_columns:
            # Get frequency tables
            ref_counts = self.reference_data[col].value_counts()
            curr_counts = current_data[col].value_counts()
            
            # Align categories
            all_categories = set(ref_counts.index) | set(curr_counts.index)
            ref_freq = np.array([ref_counts.get(cat, 0) for cat in all_categories])
            curr_freq = np.array([curr_counts.get(cat, 0) for cat in all_categories])
            
            # Chi-squared test
            chi2_stat, p_value = stats.chisquare(curr_freq, f_exp=ref_freq * (sum(curr_freq) / sum(ref_freq)))
            
            p_values[col] = p_value
            chi2_stats[col] = chi2_stat
        
        # Overall assessment
        adjusted_alpha = alpha / len(self.categorical_columns)
        drift_detected = any(p < adjusted_alpha for p in p_values.values())
        
        overall_score = np.mean(list(chi2_stats.values()))
        
        if drift_detected:
            drift_type = DriftType.DRIFT
        elif any(p < alpha for p in p_values.values()):
            drift_type = DriftType.WARNING
        else:
            drift_type = DriftType.NO_DRIFT
        
        return DriftResult(
            drift_score=overall_score,
            drift_type=drift_type,
            p_value=min(p_values.values()),
            threshold=adjusted_alpha,
            feature_scores=chi2_stats if return_feature_scores else None,
            metadata={'p_values': p_values}
        )
    
    def _population_stability_index(self,
                                   current_data: pd.DataFrame,
                                   confidence_level: float,
                                   return_feature_scores: bool) -> DriftResult:
        """
        Population Stability Index (PSI) calculation.
        
        PSI measures the shift in distributions, commonly used in 
        credit scoring and risk modeling.
        """
        psi_scores = {}
        
        # Calculate PSI for each feature
        for col in self.feature_columns:
            if col in self.numerical_columns:
                # For numerical, use binning
                ref_values = self.reference_data[col].dropna()
                curr_values = current_data[col].dropna()
                
                # Create bins based on reference data
                _, bin_edges = pd.qcut(ref_values, q=10, retbins=True, duplicates='drop')
                
                # Calculate distributions
                ref_dist = pd.cut(ref_values, bins=bin_edges, include_lowest=True).value_counts(normalize=True).sort_index()
                curr_dist = pd.cut(curr_values, bins=bin_edges, include_lowest=True).value_counts(normalize=True).sort_index()
                
                # Calculate PSI
                psi = 0
                for i in ref_dist.index:
                    ref_pct = ref_dist.get(i, 0) + 1e-10
                    curr_pct = curr_dist.get(i, 0) + 1e-10
                    psi += (curr_pct - ref_pct) * np.log(curr_pct / ref_pct)
                
                psi_scores[col] = psi
            else:
                # For categorical, use value counts
                ref_dist = self.reference_categorical_dist[col]
                curr_dist = current_data[col].value_counts(normalize=True).to_dict()
                
                # Calculate PSI
                psi = 0
                all_categories = set(ref_dist.keys()) | set(curr_dist.keys())
                for cat in all_categories:
                    ref_pct = ref_dist.get(cat, 0) + 1e-10
                    curr_pct = curr_dist.get(cat, 0) + 1e-10
                    psi += (curr_pct - ref_pct) * np.log(curr_pct / ref_pct)
                
                psi_scores[col] = psi
        
        # Overall PSI
        overall_psi = np.mean(list(psi_scores.values()))
        
        # PSI thresholds: <0.1 = no drift, 0.1-0.25 = warning, >0.25 = drift
        if overall_psi < 0.1:
            drift_type = DriftType.NO_DRIFT
        elif overall_psi < 0.25:
            drift_type = DriftType.WARNING
        else:
            drift_type = DriftType.DRIFT
        
        return DriftResult(
            drift_score=overall_psi,
            drift_type=drift_type,
            threshold=0.25,
            feature_scores=psi_scores if return_feature_scores else None,
            metadata={'interpretation': 'PSI < 0.1: No drift, 0.1-0.25: Warning, > 0.25: Drift'}
        )
    
    def _wasserstein_distance(self,
                             current_data: pd.DataFrame,
                             confidence_level: float,
                             return_feature_scores: bool) -> DriftResult:
        """
        Wasserstein distance (Earth Mover's Distance).
        
        Measures the minimum cost of transforming one distribution
        into another.
        """
        wasserstein_scores = {}
        
        # Calculate for each numerical feature
        for col in self.numerical_columns:
            ref_values = self.reference_data[col].dropna().values
            curr_values = current_data[col].dropna().values
            
            # Compute Wasserstein distance
            w_distance = stats.wasserstein_distance(ref_values, curr_values)
            
            # Normalize by standard deviation for interpretability
            std = self.reference_stats['std'][col]
            if std > 0:
                w_distance_normalized = w_distance / std
            else:
                w_distance_normalized = 0
            
            wasserstein_scores[col] = w_distance_normalized
        
        # Overall score
        overall_score = np.mean(list(wasserstein_scores.values()))
        
        # Threshold based on effect size
        # 0.1 = small, 0.3 = medium, 0.5 = large effect
        if overall_score < 0.1:
            drift_type = DriftType.NO_DRIFT
        elif overall_score < 0.3:
            drift_type = DriftType.WARNING
        else:
            drift_type = DriftType.DRIFT
        
        return DriftResult(
            drift_score=overall_score,
            drift_type=drift_type,
            threshold=0.3,
            feature_scores=wasserstein_scores if return_feature_scores else None,
            metadata={'normalized': True}
        )
    
    def _jensen_shannon_divergence(self,
                                  current_data: pd.DataFrame,
                                  confidence_level: float,
                                  return_feature_scores: bool) -> DriftResult:
        """
        Jensen-Shannon divergence between distributions.
        
        JS divergence is a symmetric version of KL divergence.
        """
        js_scores = {}
        
        # Calculate for each feature
        for col in self.numerical_columns:
            ref_values = self.reference_data[col].dropna()
            curr_values = current_data[col].dropna()
            
            # Create histogram bins
            n_bins = min(50, int(np.sqrt(len(ref_values))))
            min_val = min(ref_values.min(), curr_values.min())
            max_val = max(ref_values.max(), curr_values.max())
            bins = np.linspace(min_val, max_val, n_bins + 1)
            
            # Calculate histograms
            ref_hist, _ = np.histogram(ref_values, bins=bins)
            curr_hist, _ = np.histogram(curr_values, bins=bins)
            
            # Normalize
            ref_hist = ref_hist / ref_hist.sum()
            curr_hist = curr_hist / curr_hist.sum()
            
            # JS divergence
            m = 0.5 * (ref_hist + curr_hist)
            
            # Add small epsilon to avoid log(0)
            epsilon = 1e-10
            ref_hist = ref_hist + epsilon
            curr_hist = curr_hist + epsilon
            m = m + epsilon
            
            js_div = 0.5 * stats.entropy(ref_hist, m) + 0.5 * stats.entropy(curr_hist, m)
            js_scores[col] = js_div
        
        # Overall score
        overall_score = np.mean(list(js_scores.values()))
        
        # JS divergence is bounded between 0 and log(2)
        threshold = 0.3 * np.log(2)
        
        if overall_score < threshold * 0.5:
            drift_type = DriftType.NO_DRIFT
        elif overall_score < threshold:
            drift_type = DriftType.WARNING
        else:
            drift_type = DriftType.DRIFT
        
        return DriftResult(
            drift_score=overall_score,
            drift_type=drift_type,
            threshold=threshold,
            feature_scores=js_scores if return_feature_scores else None,
            metadata={'max_value': np.log(2)}
        )
    
    def _hellinger_distance(self,
                           current_data: pd.DataFrame,
                           confidence_level: float,
                           return_feature_scores: bool) -> DriftResult:
        """
        Hellinger distance between distributions.
        
        Hellinger distance is bounded between 0 and 1 and is
        symmetric.
        """
        hellinger_scores = {}
        
        # Calculate for each feature
        for col in self.numerical_columns:
            ref_values = self.reference_data[col].dropna()
            curr_values = current_data[col].dropna()
            
            # Create histogram bins
            n_bins = min(50, int(np.sqrt(len(ref_values))))
            min_val = min(ref_values.min(), curr_values.min())
            max_val = max(ref_values.max(), curr_values.max())
            bins = np.linspace(min_val, max_val, n_bins + 1)
            
            # Calculate histograms
            ref_hist, _ = np.histogram(ref_values, bins=bins)
            curr_hist, _ = np.histogram(curr_values, bins=bins)
            
            # Normalize
            ref_hist = ref_hist / ref_hist.sum()
            curr_hist = curr_hist / curr_hist.sum()
            
            # Hellinger distance
            h_dist = np.sqrt(0.5 * np.sum((np.sqrt(ref_hist) - np.sqrt(curr_hist))**2))
            
            hellinger_scores[col] = h_dist
        
        # Overall score
        overall_score = np.mean(list(hellinger_scores.values()))
        
        # Thresholds (Hellinger distance is bounded between 0 and 1)
        if overall_score < 0.1:
            drift_type = DriftType.NO_DRIFT
        elif overall_score < 0.3:
            drift_type = DriftType.WARNING
        else:
            drift_type = DriftType.DRIFT
        
        return DriftResult(
            drift_score=overall_score,
            drift_type=drift_type,
            threshold=0.3,
            feature_scores=hellinger_scores if return_feature_scores else None,
            metadata={'n_bins': n_bins}
        )
    
    def _maximum_mean_discrepancy(self,
                                 current_data: pd.DataFrame,
                                 confidence_level: float,
                                 return_feature_scores: bool) -> DriftResult:
        """
        Maximum Mean Discrepancy (MMD) test.
        
        MMD is a kernel-based method that measures the distance
        between the mean embeddings of two distributions.
        """
        # For efficiency, we'll use a simplified version with RBF kernel
        mmd_scores = {}
        
        for col in self.numerical_columns:
            ref_values = self.reference_data[col].dropna().values.reshape(-1, 1)
            curr_values = current_data[col].dropna().values.reshape(-1, 1)
            
            # Subsample for computational efficiency
            max_samples = 1000
            if len(ref_values) > max_samples:
                ref_idx = np.random.choice(len(ref_values), max_samples, replace=False)
                ref_values = ref_values[ref_idx]
            if len(curr_values) > max_samples:
                curr_idx = np.random.choice(len(curr_values), max_samples, replace=False)
                curr_values = curr_values[curr_idx]
            
            # Compute MMD with RBF kernel
            mmd = self._compute_mmd(ref_values, curr_values)
            mmd_scores[col] = mmd
        
        # Overall score
        overall_score = np.mean(list(mmd_scores.values()))
        
        # Threshold (empirically determined)
        threshold = 0.05
        
        if overall_score < threshold * 0.5:
            drift_type = DriftType.NO_DRIFT
        elif overall_score < threshold:
            drift_type = DriftType.WARNING
        else:
            drift_type = DriftType.DRIFT
        
        return DriftResult(
            drift_score=overall_score,
            drift_type=drift_type,
            threshold=threshold,
            feature_scores=mmd_scores if return_feature_scores else None,
            metadata={'kernel': 'rbf', 'max_samples': 1000}
        )
    
    def _compute_mmd(self, X: np.ndarray, Y: np.ndarray, gamma: float = None) -> float:
        """Compute MMD statistic with RBF kernel."""
        n, m = len(X), len(Y)
        
        # Set bandwidth using median heuristic
        if gamma is None:
            from sklearn.metrics.pairwise import pairwise_distances
            distances = pairwise_distances(np.vstack([X, Y]))
            gamma = 1.0 / (2.0 * np.median(distances)**2)
        
        # Compute kernel matrices
        from sklearn.metrics.pairwise import rbf_kernel
        K_XX = rbf_kernel(X, X, gamma=gamma)
        K_YY = rbf_kernel(Y, Y, gamma=gamma)
        K_XY = rbf_kernel(X, Y, gamma=gamma)
        
        # MMD statistic
        mmd = np.mean(K_XX) - 2 * np.mean(K_XY) + np.mean(K_YY)
        
        return mmd
    
    def monitor_drift_over_time(self,
                               data_batches: List[pd.DataFrame],
                               timestamps: Optional[List[Any]] = None,
                               method: str = 'ks',
                               window_size: Optional[int] = None) -> pd.DataFrame:
        """
        Monitor drift over multiple time periods.
        
        Args:
            data_batches: List of datasets over time
            timestamps: Optional timestamps for each batch
            method: Drift detection method
            window_size: Sliding window size (None for cumulative)
            
        Returns:
            DataFrame with drift scores over time
        """
        logger.info(f"Monitoring drift over {len(data_batches)} batches")
        
        if timestamps is None:
            timestamps = [f"Batch_{i}" for i in range(len(data_batches))]
        
        results = []
        
        for i, (batch, timestamp) in enumerate(zip(data_batches, timestamps)):
            # Determine reference data
            if window_size and i >= window_size:
                # Use sliding window
                ref_batches = data_batches[i-window_size:i]
                reference = pd.concat(ref_batches, ignore_index=True)
                
                # Update reference statistics
                self.reference_data = reference
                self._compute_reference_statistics()
            elif i == 0:
                # Skip first batch (it's the initial reference)
                continue
            
            # Detect drift
            drift_result = self.detect_drift(batch, method=method)
            
            results.append({
                'timestamp': timestamp,
                'drift_score': drift_result.drift_score,
                'drift_type': drift_result.drift_type.value,
                'p_value': drift_result.p_value
            })
        
        return pd.DataFrame(results)
    
    def get_drift_report(self,
                        current_data: pd.DataFrame,
                        methods: List[str] = ['ks', 'psi', 'wasserstein']) -> Dict[str, Any]:
        """
        Generate comprehensive drift report using multiple methods.
        
        Args:
            current_data: Current dataset
            methods: List of detection methods to use
            
        Returns:
            Dictionary with drift analysis results
        """
        logger.info(f"Generating drift report using methods: {methods}")
        
        report = {
            'summary': {},
            'method_results': {},
            'feature_analysis': {},
            'recommendations': []
        }
        
        # Run multiple detection methods
        for method in methods:
            try:
                result = self.detect_drift(current_data, method=method, return_feature_scores=True)
                report['method_results'][method] = {
                    'drift_score': result.drift_score,
                    'drift_type': result.drift_type.value,
                    'p_value': result.p_value,
                    'threshold': result.threshold
                }
                
                # Aggregate feature scores
                if result.feature_scores:
                    for feature, score in result.feature_scores.items():
                        if feature not in report['feature_analysis']:
                            report['feature_analysis'][feature] = {}
                        report['feature_analysis'][feature][method] = score
            except Exception as e:
                logger.error(f"Error running {method}: {str(e)}")
                report['method_results'][method] = {'error': str(e)}
        
        # Overall summary
        drift_votes = [r.get('drift_type', 'no_drift') 
                      for r in report['method_results'].values() 
                      if 'drift_type' in r]
        
        if drift_votes:
            most_common_drift = max(set(drift_votes), key=drift_votes.count)
            report['summary']['overall_drift_type'] = most_common_drift
            report['summary']['agreement_rate'] = drift_votes.count(most_common_drift) / len(drift_votes)
        
        # Feature-level summary
        for feature, scores in report['feature_analysis'].items():
            avg_score = np.mean(list(scores.values()))
            report['feature_analysis'][feature]['average_score'] = avg_score
        
        # Top drifting features
        if report['feature_analysis']:
            top_features = sorted(
                report['feature_analysis'].items(),
                key=lambda x: x[1].get('average_score', 0),
                reverse=True
            )[:5]
            report['summary']['top_drifting_features'] = [f[0] for f in top_features]
        
        # Recommendations
        if report['summary'].get('overall_drift_type') == 'drift':
            report['recommendations'].append("Significant drift detected. Consider retraining the model.")
            report['recommendations'].append("Investigate the top drifting features for root cause analysis.")
        elif report['summary'].get('overall_drift_type') == 'warning':
            report['recommendations'].append("Warning: Some drift detected. Monitor closely.")
            report['recommendations'].append("Consider collecting more data for verification.")
        else:
            report['recommendations'].append("No significant drift detected. Continue monitoring.")
        
        return report
    
    def visualize_drift(self,
                       current_data: pd.DataFrame,
                       features: Optional[List[str]] = None,
                       max_features: int = 6,
                       save_path: Optional[str] = None) -> plt.Figure:
        """
        Visualize drift by comparing distributions.
        
        Args:
            current_data: Current dataset
            features: Specific features to visualize (None for auto-selection)
            max_features: Maximum number of features to plot
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        # Select features to visualize
        if features is None:
            # Auto-select features with highest drift
            drift_result = self.detect_drift(current_data, return_feature_scores=True)
            if drift_result.feature_scores:
                sorted_features = sorted(
                    drift_result.feature_scores.items(),
                    key=lambda x: x[1],
                    reverse=True
                )
                features = [f[0] for f in sorted_features[:max_features]]
            else:
                features = self.feature_columns[:max_features]
        
        n_features = min(len(features), max_features)
        n_cols = min(3, n_features)
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
        if n_features == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        # Plot each feature
        for idx, feature in enumerate(features[:n_features]):
            ax = axes[idx]
            
            if feature in self.numerical_columns:
                # Density plots for numerical features
                ref_data = self.reference_data[feature].dropna()
                curr_data = current_data[feature].dropna()
                
                # Plot distributions
                ref_data.plot(kind='density', ax=ax, label='Reference', alpha=0.7)
                curr_data.plot(kind='density', ax=ax, label='Current', alpha=0.7)
                
                # Add KS statistic
                ks_stat, p_value = stats.ks_2samp(ref_data, curr_data)
                ax.text(0.05, 0.95, f'KS stat: {ks_stat:.3f}\np-value: {p_value:.3f}',
                       transform=ax.transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                
            else:
                # Bar plots for categorical features
                ref_counts = self.reference_data[feature].value_counts(normalize=True)
                curr_counts = current_data[feature].value_counts(normalize=True)
                
                # Align categories
                all_categories = sorted(set(ref_counts.index) | set(curr_counts.index))
                ref_props = [ref_counts.get(cat, 0) for cat in all_categories]
                curr_props = [curr_counts.get(cat, 0) for cat in all_categories]
                
                # Plot bars
                x = np.arange(len(all_categories))
                width = 0.35
                ax.bar(x - width/2, ref_props, width, label='Reference', alpha=0.7)
                ax.bar(x + width/2, curr_props, width, label='Current', alpha=0.7)
                ax.set_xticks(x)
                ax.set_xticklabels(all_categories, rotation=45, ha='right')
            
            ax.set_title(f'{feature}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Remove empty subplots
        for idx in range(n_features, len(axes)):
            fig.delaxes(axes[idx])
        
        plt.suptitle('Feature Distribution Comparison', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Drift visualization saved to {save_path}")
        
        return fig


# Utility functions for drift detection
def create_drift_detector(reference_data: pd.DataFrame,
                         feature_columns: Optional[List[str]] = None,
                         categorical_columns: Optional[List[str]] = None) -> DataDriftDetector:
    """
    Factory function to create a drift detector.
    
    Args:
        reference_data: Reference dataset
        feature_columns: Features to monitor
        categorical_columns: Categorical features
        
    Returns:
        Configured DataDriftDetector instance
    """
    return DataDriftDetector(
        reference_data=reference_data,
        feature_columns=feature_columns,
        categorical_columns=categorical_columns
    )


def detect_drift_simple(reference_data: pd.DataFrame,
                       current_data: pd.DataFrame,
                       method: str = 'ks',
                       confidence_level: float = 0.95) -> DriftResult:
    """
    Simple interface for drift detection.
    
    Args:
        reference_data: Reference dataset
        current_data: Current dataset
        method: Detection method
        confidence_level: Confidence level
        
    Returns:
        DriftResult object
    """
    detector = DataDriftDetector(reference_data)
    return detector.detect_drift(current_data, method, confidence_level)


# Example usage
def example_drift_detection():
    """Example of using drift detection."""
    from sklearn.datasets import make_classification
    
    # Generate synthetic data
    X_ref, y_ref = make_classification(n_samples=1000, n_features=10, n_informative=5, random_state=42)
    X_ref = pd.DataFrame(X_ref, columns=[f'feature_{i}' for i in range(10)])
    
    # Generate current data with drift
    X_curr, y_curr = make_classification(n_samples=1000, n_features=10, n_informative=5, 
                                       flip_y=0.1, random_state=123)
    X_curr = pd.DataFrame(X_curr, columns=[f'feature_{i}' for i in range(10)])
    
    # Add drift to some features
    X_curr['feature_0'] = X_curr['feature_0'] + 0.5  # Location shift
    X_curr['feature_1'] = X_curr['feature_1'] * 2    # Scale shift
    
    # Create detector
    detector = DataDriftDetector(X_ref)
    
    # Detect drift
    print("Kolmogorov-Smirnov Test:")
    ks_result = detector.detect_drift(X_curr, method='ks')
    print(f"  Drift Type: {ks_result.drift_type.value}")
    print(f"  Drift Score: {ks_result.drift_score:.4f}")
    
    print("\nPopulation Stability Index:")
    psi_result = detector.detect_drift(X_curr, method='psi')
    print(f"  Drift Type: {psi_result.drift_type.value}")
    print(f"  Drift Score: {psi_result.drift_score:.4f}")
    
    # Generate comprehensive report
    report = detector.get_drift_report(X_curr)
    print("\nDrift Report Summary:")
    print(f"  Overall Drift: {report['summary']['overall_drift_type']}")
    print(f"  Top Drifting Features: {report['summary']['top_drifting_features']}")
    print(f"  Recommendations: {report['recommendations'][0]}")
    
    # Visualize drift
    fig = detector.visualize_drift(X_curr, max_features=4)
    
    return detector, report, fig


def example_specialized_detectors():
    """Example of using specialized drift detectors."""
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    
    # Generate data
    X, y = make_classification(n_samples=2000, n_features=20, n_informative=15, 
                              n_redundant=5, random_state=42)
    X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(20)])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
    
    # Example 1: Statistical Drift Detector
    print("=== Statistical Drift Detector ===")
    stat_detector = StatisticalDriftDetector(test_method='ks')
    stat_detector.fit(X_train)
    
    # Add drift to test data
    X_drift = X_test.copy()
    X_drift['feature_0'] = X_drift['feature_0'] + 1.0
    
    stat_result = stat_detector.detect_drift(X_drift)
    print(f"Statistical Test Result: {stat_result.drift_type.value}")
    print(f"Drift Score: {stat_result.drift_score:.4f}")
    
    # Example 2: Distance Drift Detector
    print("\n=== Distance Drift Detector ===")
    dist_detector = DistanceDriftDetector(distance_metric='wasserstein')
    dist_detector.fit(X_train)
    
    dist_result = dist_detector.detect_drift(X_drift)
    print(f"Distance Metric Result: {dist_result.drift_type.value}")
    print(f"Drift Score: {dist_result.drift_score:.4f}")
    
    # Example 3: Model Performance Drift Detector
    print("\n=== Model Performance Drift Detector ===")
    
    # Train a model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Create performance detector
    perf_detector = ModelPerformanceDriftDetector(model, metric='accuracy')
    perf_detector.fit(X_train, y_train)
    
    # Simulate performance degradation
    y_test_noisy = y_test.copy()
    flip_indices = np.random.choice(len(y_test), size=int(0.2 * len(y_test)), replace=False)
    y_test_noisy[flip_indices] = 1 - y_test_noisy[flip_indices]
    
    perf_result = perf_detector.detect_drift(X_test, y_test_noisy)
    print(f"Performance Drift Result: {perf_result.drift_type.value}")
    print(f"Performance Drop: {perf_result.drift_score:.2%}")
    
    return stat_detector, dist_detector, perf_detector


if __name__ == "__main__":
    # Run basic example
    detector, report, fig = example_drift_detection()
    print("\nDrift detection example completed successfully!")
    
    # Run specialized detectors example
    print("\n" + "="*50 + "\n")
    stat_det, dist_det, perf_det = example_specialized_detectors()
    print("\nSpecialized detectors example completed successfully!")