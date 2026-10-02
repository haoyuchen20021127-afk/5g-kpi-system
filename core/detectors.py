"""Four univariate GOTSF detectors with a separate calibration split.

Fitting is available only to the offline command. Web pages load frozen models.
"""
from dataclasses import dataclass
import math
import warnings
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.exceptions import ConvergenceWarning

METHODS = ('three_sigma', 'iqr', 'isolation_forest', 'one_class_svm')


def finite_quantile(values, alpha=.05):
    values = np.sort(np.asarray(values, dtype=float))
    values = values[np.isfinite(values)]
    if not len(values):
        return float('nan')
    rank = math.ceil((len(values) + 1) * (1 - alpha))
    return float(values[rank - 1]) if rank <= len(values) else float('inf')


def compare_score(score, threshold, method):
    # A numerical guard, not a probability or a solver error guarantee.
    guard = 1e-6 if method == 'one_class_svm' else 1e-10 * max(1., abs(threshold))
    score = np.asarray(score, dtype=float)
    valid = np.isfinite(score) & np.isfinite(threshold)
    flags = np.full(score.shape, -1, dtype=np.int8)
    flags[valid] = (score[valid] > threshold + guard).astype(np.int8)
    return flags, guard


@dataclass
class Detector:
    method: str
    status: str
    center: float = 0.
    scale: float = 1.
    lower: float = float('nan')
    upper: float = float('nan')
    threshold: float = float('nan')
    model: object = None
    metadata: dict = None

    def score(self, values):
        values = np.asarray(values, dtype=float)
        out = np.full(values.shape, np.nan)
        valid = np.isfinite(values)
        if self.status != 'ok' or not valid.any():
            return out
        x = values[valid]
        if self.method in METHODS[:2]:
            out[valid] = np.maximum(self.lower - x, x - self.upper) / self.scale
        else:
            # Compute once per distinct value without dropping repeated records.
            unique, inverse = np.unique((x - self.center) / self.scale, return_inverse=True)
            out[valid] = -self.model.decision_function(unique.reshape(-1, 1))[inverse]
        return out

    def comparison_method(self):
        return self.method

    def flag(self, values):
        return compare_score(self.score(values), self.threshold, self.method)[0]


def fit_detector(method, fit, calibration, observed_fit=None, observed_calibration=None):
    """Refit prepared GOTSF residuals; calibration and test data never train trees."""
    if method not in METHODS:
        raise ValueError('未知检测方法')
    def finite(x):
        x = np.asarray(x, dtype=float)
        return x[np.isfinite(x)]
    fit, calibration = finite(fit), finite(calibration)
    observed_fit = finite(fit if observed_fit is None else observed_fit)
    observed_calibration = finite(calibration if observed_calibration is None else observed_calibration)
    detector = Detector(method, 'ok', metadata={'alpha': .05, 'fit_n': len(fit), 'calibration_n': len(calibration)})
    if len(observed_fit) < 100 or len(observed_calibration) < 50 or not len(fit) or not len(calibration):
        detector.status = 'insufficient_reference_or_calibration'
        return detector
    cardinality = len(np.unique(observed_fit))
    if cardinality < 2 and method != 'iqr':
        detector.status = 'constant_reference'
        return detector
    q1, median, q3 = np.quantile(fit, [.25, .5, .75])
    std = fit.std(ddof=0)
    detector.center = float(median)
    detector.scale = 1. if method == 'iqr' else float(q3-q1) if q3 > q1 else float(std)
    if not np.isfinite(detector.scale) or detector.scale <= 0:
        detector.status = 'invalid_scale'
        return detector
    if method == 'three_sigma':
        detector.lower, detector.upper = fit.mean()-3*std, fit.mean()+3*std
        detector.scale = float(std)
    elif method == 'iqr':
        detector.lower, detector.upper = q1-1.5*(q3-q1), q3+1.5*(q3-q1)
        detector.metadata['zero_iqr'] = bool(q1 == q3)
    else:
        if cardinality < 5:
            detector.status = 'low_cardinality_reference'
            return detector
        rng = np.random.default_rng(42)
        chosen = rng.choice(len(fit), 6000, replace=False) if len(fit) > 6000 else np.arange(len(fit))
        x = ((fit[chosen]-detector.center)/detector.scale).reshape(-1, 1)
        if len(np.unique(x)) < 5:
            detector.status = 'low_cardinality_fit_sample'
            return detector
        if method == 'isolation_forest':
            detector.model = IsolationForest(n_estimators=200, max_samples=min(256, len(x)),
                                            contamination='auto', random_state=42, n_jobs=2)
        else:
            detector.model = OneClassSVM(kernel='rbf', nu=.05, gamma='scale', tol=1e-7,
                                        max_iter=100000, cache_size=256)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('error', ConvergenceWarning)
                detector.model.fit(x)
        except ConvergenceWarning:
            detector.status, detector.model = 'solver_not_converged', None
            return detector
        detector.metadata['ml_fit_n'] = len(x)
    threshold = finite_quantile(detector.score(calibration))
    detector.threshold = max(0., threshold) if method in METHODS[:2] else threshold
    return detector
