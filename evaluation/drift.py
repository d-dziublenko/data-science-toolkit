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
import warnings
import logging
from dataclasses import dataclass
from enum import Enum

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
    
    def _maximum_mean_discrepancy(self,
                                 current_data: pd.DataFrame,
                                 confidence_level: float,
                                 return_feature_scores: bool) -> DriftResult:
        """
        Maximum Mean Discrepancy (MMD) test.
        
        MMD is a kernel-based method that measures the distance
        between the mean embeddings of two distributions.
        """
        # Scale numerical features
        ref_scaled = self.scaler.transform(self.reference_data[self.numerical_columns])
        curr_scaled = self.scaler.transform(current_data[self.numerical_columns])
        
        # Compute MMD
        mmd_score = self._compute_mmd(ref_scaled, curr_scaled)
        
        # Bootstrap threshold
        threshold = self._bootstrap_mmd_threshold(ref_scaled, confidence_level)
        
        # Feature-wise MMD if requested
        feature_scores = {}
        if return_feature_scores:
            for i, col in enumerate(self.numerical_columns):
                feature_mmd = self._compute_mmd(
                    ref_scaled[:, i:i+1],
                    curr_scaled[:, i:i+1]
                )
                feature_scores[col] = feature_mmd
        
        # Determine drift
        if mmd_score > threshold:
            drift_type = DriftType.DRIFT
        elif mmd_score > threshold * 0.8:  # Warning threshold
            drift_type = DriftType.WARNING
        else:
            drift_type = DriftType.NO_DRIFT
        
        return DriftResult(
            drift_score=mmd_score,
            drift_type=drift_type,
            threshold=threshold,
            feature_scores=feature_scores if return_feature_scores else None,
            metadata={'kernel': 'rbf'}
        )
    
    def _compute_mmd(self, X: np.ndarray, Y: np.ndarray, gamma: float = 1.0) -> float:
        """
        Compute MMD between two samples using RBF kernel.
        
        Args:
            X: First sample
            Y: Second sample
            gamma: RBF kernel parameter
            
        Returns:
            MMD score
        """
        m, n = len(X), len(Y)
        
        # Compute kernel matrices
        XX = self._rbf_kernel(X, X, gamma)
        YY = self._rbf_kernel(Y, Y, gamma)
        XY = self._rbf_kernel(X, Y, gamma)
        
        # MMD squared
        mmd2 = XX.sum() / (m * m) - 2 * XY.sum() / (m * n) + YY.sum() / (n * n)
        
        return np.sqrt(max(0, mmd2))
    
    def _rbf_kernel(self, X: np.ndarray, Y: np.ndarray, gamma: float) -> np.ndarray:
        """Compute RBF kernel matrix."""
        from scipy.spatial.distance import cdist
        sq_dists = cdist(X, Y, 'sqeuclidean')
        return np.exp(-gamma * sq_dists)
    
    def _bootstrap_mmd_threshold(self,
                               reference_scaled: np.ndarray,
                               confidence_level: float,
                               n_bootstrap: int = 100) -> float:
        """
        Estimate MMD threshold using bootstrap.
        
        Args:
            reference_scaled: Scaled reference data
            confidence_level: Confidence level
            n_bootstrap: Number of bootstrap iterations
            
        Returns:
            Threshold value
        """
        n_samples = len(reference_scaled)
        mmd_scores = []
        
        for _ in range(n_bootstrap):
            # Split reference data randomly
            idx = np.random.permutation(n_samples)
            split_point = n_samples // 2
            
            sample1 = reference_scaled[idx[:split_point]]
            sample2 = reference_scaled[idx[split_point:]]
            
            mmd_score = self._compute_mmd(sample1, sample2)
            mmd_scores.append(mmd_score)
        
        # Return percentile as threshold
        threshold = np.percentile(mmd_scores, confidence_level * 100)
        return threshold
    
    def _population_stability_index(self,
                                   current_data: pd.DataFrame,
                                   confidence_level: float,
                                   return_feature_scores: bool) -> DriftResult:
        """
        Population Stability Index (PSI) calculation.
        
        PSI measures the shift in the distribution of a variable
        by comparing the proportions across bins.
        """
        psi_scores = {}
        
        # Calculate PSI for each numerical feature
        for col in self.numerical_columns:
            ref_values = self.reference_data[col].dropna()
            curr_values = current_data[col].dropna()
            
            # Create bins based on reference data
            n_bins = min(10, int(np.sqrt(len(ref_values))))
            _, bin_edges = pd.qcut(ref_values, q=n_bins, retbins=True, duplicates='drop')
            
            # Calculate proportions
            ref_props = pd.cut(ref_values, bins=bin_edges, include_lowest=True).value_counts(normalize=True).sort_index()
            curr_props = pd.cut(curr_values, bins=bin_edges, include_lowest=True).value_counts(normalize=True).sort_index()
            
            # Align bins and calculate PSI
            psi = 0
            for bin_label in ref_props.index:
                ref_prop = ref_props[bin_label]
                curr_prop = curr_props.get(bin_label, 0.001)  # Small value to avoid log(0)
                
                # PSI formula
                psi += (curr_prop - ref_prop) * np.log(curr_prop / ref_prop)
            
            psi_scores[col] = psi
        
        # Overall PSI
        overall_psi = np.mean(list(psi_scores.values()))
        
        # PSI thresholds (industry standard)
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
            metadata={'psi_interpretation': {
                '<0.1': 'No significant change',
                '0.1-0.25': 'Moderate change',
                '>0.25': 'Significant change'
            }}
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
            js_div = 0.5 * self._kl_divergence(ref_hist, m) + 0.5 * self._kl_divergence(curr_hist, m)
            
            js_scores[col] = js_div
        
        # Overall score
        overall_score = np.mean(list(js_scores.values()))
        
        # Thresholds (JS divergence is bounded between 0 and 1)
        if overall_score < 0.05:
            drift_type = DriftType.NO_DRIFT
        elif overall_score < 0.15:
            drift_type = DriftType.WARNING
        else:
            drift_type = DriftType.DRIFT
        
        return DriftResult(
            drift_score=overall_score,
            drift_type=drift_type,
            threshold=0.15,
            feature_scores=js_scores if return_feature_scores else None,
            metadata={'n_bins': n_bins}
        )
    
    def _kl_divergence(self, p: np.ndarray, q: np.ndarray) -> float:
        """Calculate KL divergence KL(P||Q)."""
        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        p = p + epsilon
        q = q + epsilon
        
        return np.sum(p * np.log(p / q))
    
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
    
    def detect_multivariate_drift(self,
                                 current_data: pd.DataFrame,
                                 method: str = 'pca',
                                 confidence_level: float = 0.95) -> DriftResult:
        """
        Detect multivariate drift considering feature correlations.
        
        Args:
            current_data: Current dataset
            method: Multivariate method ('pca', 'autoencoder')
            confidence_level: Confidence level
            
        Returns:
            DriftResult object
        """
        logger.info(f"Detecting multivariate drift using {method}")
        
        if method == 'pca':
            return self._pca_drift_detection(current_data, confidence_level)
        elif method == 'nearest_neighbors':
            return self._nearest_neighbors_drift(current_data, confidence_level)
        else:
            raise ValueError(f"Unknown multivariate method: {method}")
    
    def _pca_drift_detection(self,
                            current_data: pd.DataFrame,
                            confidence_level: float) -> DriftResult:
        """
        PCA-based drift detection.
        
        Projects data onto principal components and detects
        drift in the reduced space.
        """
        # Fit PCA on reference data
        ref_scaled = self.scaler.transform(self.reference_data[self.numerical_columns])
        
        # Determine number of components to retain 95% variance
        pca = PCA(n_components=0.95)
        pca.fit(ref_scaled)
        
        # Transform both datasets
        ref_transformed = pca.transform(ref_scaled)
        curr_scaled = self.scaler.transform(current_data[self.numerical_columns])
        curr_transformed = pca.transform(curr_scaled)
        
        # Compute MMD in PCA space
        mmd_score = self._compute_mmd(ref_transformed, curr_transformed)
        
        # Bootstrap threshold
        threshold = self._bootstrap_mmd_threshold(ref_transformed, confidence_level)
        
        # Determine drift
        if mmd_score > threshold:
            drift_type = DriftType.DRIFT
        elif mmd_score > threshold * 0.8:
            drift_type = DriftType.WARNING
        else:
            drift_type = DriftType.NO_DRIFT
        
        return DriftResult(
            drift_score=mmd_score,
            drift_type=drift_type,
            threshold=threshold,
            metadata={
                'n_components': pca.n_components_,
                'explained_variance_ratio': pca.explained_variance_ratio_.tolist()
            }
        )
    
    def _nearest_neighbors_drift(self,
                               current_data: pd.DataFrame,
                               confidence_level: float,
                               k: int = 5) -> DriftResult:
        """
        Nearest neighbors-based drift detection.
        
        Compares the distribution of nearest neighbor distances.
        """
        # Scale data
        ref_scaled = self.scaler.transform(self.reference_data[self.numerical_columns])
        curr_scaled = self.scaler.transform(current_data[self.numerical_columns])
        
        # Fit nearest neighbors on reference data
        nn_model = NearestNeighbors(n_neighbors=k)
        nn_model.fit(ref_scaled)
        
        # Get distances for reference data (leave-one-out)
        ref_distances, _ = nn_model.kneighbors(ref_scaled, n_neighbors=k+1)
        ref_distances = ref_distances[:, 1:]  # Exclude self
        
        # Get distances for current data
        curr_distances, _ = nn_model.kneighbors(curr_scaled, n_neighbors=k)
        
        # Compare distance distributions
        ref_mean_distances = np.mean(ref_distances, axis=1)
        curr_mean_distances = np.mean(curr_distances, axis=1)
        
        # KS test on distance distributions
        ks_stat, p_value = stats.ks_2samp(ref_mean_distances, curr_mean_distances)
        
        # Determine drift
        alpha = 1 - confidence_level
        if p_value < alpha:
            drift_type = DriftType.DRIFT
        elif p_value < alpha * 2:
            drift_type = DriftType.WARNING
        else:
            drift_type = DriftType.NO_DRIFT
        
        return DriftResult(
            drift_score=ks_stat,
            drift_type=drift_type,
            p_value=p_value,
            threshold=alpha,
            metadata={
                'k_neighbors': k,
                'mean_ref_distance': np.mean(ref_mean_distances),
                'mean_curr_distance': np.mean(curr_mean_distances)
            }
        )
    
    def monitor_drift_over_time(self,
                               data_batches: List[pd.DataFrame],
                               timestamps: Optional[List[str]] = None,
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
            report['recommendations'].append("Moderate drift detected. Monitor closely and prepare for potential retraining.")
            report['recommendations'].append("Consider collecting more recent data for validation.")
        else:
            report['recommendations'].append("No significant drift detected. Continue regular monitoring.")
        
        return report
    
    def visualize_drift(self,
                       current_data: pd.DataFrame,
                       features_to_plot: Optional[List[str]] = None,
                       max_features: int = 6):
        """
        Visualize drift for selected features.
        
        Args:
            current_data: Current dataset
            features_to_plot: Specific features to plot (None for top drifting)
            max_features: Maximum number of features to plot
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Determine features to plot
        if features_to_plot is None:
            # Get drift scores for all features
            result = self.detect_drift(current_data, return_feature_scores=True)
            if result.feature_scores:
                # Sort by drift score
                sorted_features = sorted(
                    result.feature_scores.items(),
                    key=lambda x: x[1],
                    reverse=True
                )
                features_to_plot = [f[0] for f in sorted_features[:max_features]]
            else:
                features_to_plot = self.numerical_columns[:max_features]
        
        # Create subplots
        n_features = len(features_to_plot)
        n_cols = min(3, n_features)
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        if n_features == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        # Plot each feature
        for idx, feature in enumerate(features_to_plot):
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


if __name__ == "__main__":
    detector, report, fig = example_drift_detection()
    print("\nDrift detection example completed successfully!")