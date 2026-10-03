"""Univariate reference detectors with separate empirical calibration.

Missing observations are unscored. Degenerate IQR references retain warnings;
inapplicable models remain explicit. Numerical guards and saturation are audited.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import warnings
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.exceptions import ConvergenceWarning

METHODS = ('three_sigma','iqr','isolation_forest','one_class_svm')
CONFIG = dict(alpha=0.05, min_fit=100, min_cal=50, min_model_unique=5,
              max_ml_fit=6000, seed=42, if_trees=200, svm_nu=.05,
              svm_tol=1e-7, svm_max_iter=100000, iqr_score_policy='native_fence_distance')
LOW_CARDINALITY_FEATURES = frozenset({'cqi', 'pcmax'})


def finite_quantile(values, alpha=.05):
    a = np.sort(np.asarray(values,dtype=float))
    a = a[np.isfinite(a)]
    if not len(a):
        return float('nan')
    k = math.ceil((len(a)+1)*(1-alpha))
    return float(a[k-1]) if k<=len(a) else float('inf')


def compare_score(score, threshold, method):
    # SVM solver tolerance is not a mathematical score-error bound. This fixed
    # protection is operational, reported alongside raw margins, not optimized.
    guard = 1e-6 if method == 'one_class_svm' else 1e-10*max(1.,abs(threshold))
    s = np.asarray(score,dtype=float)
    valid = np.isfinite(s) & np.isfinite(threshold)
    flag = np.full(s.shape,-1,dtype=np.int8)
    flag[valid] = (s[valid] > threshold + guard).astype(np.int8)
    return flag, guard


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
        values = np.asarray(values,dtype=float)
        out = np.full(values.shape,np.nan)
        good = np.isfinite(values)
        if self.status != 'ok' or not good.any():
            return out
        x = values[good]
        if self.metadata and self.metadata.get('decision_origin') == 'constant_reference_consistency':
            out[good] = np.abs(x-self.metadata['reference_value'])
        elif self.method in METHODS[:2]:
            out[good] = np.maximum(self.lower-x, x-self.upper)/self.scale
        else:
            # Exact same-value reuse preserves every row and its frequency.
            unique, inverse = np.unique((x-self.center)/self.scale,return_inverse=True)
            scores = -self.model.decision_function(unique.reshape(-1,1))
            out[good] = scores[inverse]
        return out

    def flag(self, values):
        return compare_score(self.score(values),self.threshold,self.comparison_method())[0]

    def comparison_method(self):
        if self.metadata and self.metadata.get('decision_origin') == 'constant_reference_consistency':
            return 'constant_reference_consistency'
        return self.method


def _category_preserving_indices(values, budget, rng):
    """Keep every observed value without creating or duplicating observations.

    Reserve one actual row per value, then sample the remaining budget uniformly
    without replacement. This is only used for named discrete KPI exceptions;
    its fitted frequencies are saved because rare values can be overrepresented
    relative to an unrestricted random subsample. None means the budget cannot
    represent every distinct value, rather than silently dropping categories.
    """
    if len(values) <= budget:
        return np.arange(len(values))
    _, representatives = np.unique(values, return_index=True)
    if len(representatives) > budget:
        return None
    available = np.ones(len(values), dtype=bool)
    available[representatives] = False
    additional = rng.choice(np.flatnonzero(available), budget-len(representatives), replace=False)
    chosen = np.concatenate([representatives, additional])
    rng.shuffle(chosen)
    return chosen


def _finish_calibration(d, calibration, config, med, std, q1, q3):
    """Calibrate scores without any evaluation data or forced anomaly quota."""
    method, meta = d.method, d.metadata
    comparison_method = d.comparison_method()
    rule = comparison_method == 'constant_reference_consistency'
    score=d.score(calibration)
    threshold=finite_quantile(score,config['alpha'])
    d.threshold=max(0.,threshold) if method in METHODS[:2] or rule else threshold
    flags,guard=compare_score(score,d.threshold,comparison_method)
    meta.update(calibration_flag_n=int((flags==1).sum()),calibration_flag_fraction=float((flags==1).mean()),
        raw_calibration_quantile=float(threshold),threshold=float(d.threshold),numerical_guard=float(guard),
        threshold_tie_fraction=float((np.abs(score-d.threshold)<=guard).mean()),
        threshold_at_max_score=bool(d.threshold>=np.max(score)-guard),
        training_median=float(med),training_std=float(std),training_iqr=float(q3-q1),
        base_lower=float(d.lower) if np.isfinite(d.lower) else None,
        base_upper=float(d.upper) if np.isfinite(d.upper) else None)
    if method in METHODS[:2] or rule:
        meta['effective_lower']=float(d.lower-(d.threshold+guard)*d.scale)
        meta['effective_upper']=float(d.upper+(d.threshold+guard)*d.scale)
        base=compare_score(score,0.,comparison_method)[0]
        meta['base_calibration_flag_n']=int((base==1).sum())
        meta['base_calibration_flag_fraction']=float((base==1).mean())
    if rule:
        meta['rule_tolerance']=float(d.threshold+guard)
    return d


def fit_detector(method, fit, calibration, config=None, feature=None,
                 observed_fit=None, observed_calibration=None):
    """Fit/calibrate on supplied values; qualify references on actual observations.

    Optional observed arrays must contain the pre-imputation reference values.
    They prevent filled rows or newly introduced median values from satisfying
    minimum observed sample-count or observed-cardinality requirements.
    """
    config = dict(CONFIG, **(config or {}))
    fit = np.asarray(fit,dtype=float)
    calibration = np.asarray(calibration,dtype=float)
    fit = fit[np.isfinite(fit)]
    calibration = calibration[np.isfinite(calibration)]
    unique = np.unique(fit)
    observed_fit = fit if observed_fit is None else np.asarray(observed_fit,dtype=float)
    observed_calibration = calibration if observed_calibration is None else np.asarray(observed_calibration,dtype=float)
    observed_fit = observed_fit[np.isfinite(observed_fit)]
    observed_calibration = observed_calibration[np.isfinite(observed_calibration)]
    observed_unique = np.unique(observed_fit)
    feature_name = str(feature).lower() if feature is not None else None
    discrete_exception = feature_name in LOW_CARDINALITY_FEATURES
    minimum_unique = 2 if discrete_exception else config['min_model_unique']
    meta = dict(fit_n=len(fit),calibration_n=len(calibration),fit_unique_n=len(unique),
                alpha=config['alpha'],fit_min=float(fit.min()) if len(fit) else None,
                fit_max=float(fit.max()) if len(fit) else None,
                reference_observed_fit_n=len(observed_fit),reference_observed_calibration_n=len(observed_calibration),
                reference_observed_unique_n=len(observed_unique),
                feature=feature_name,low_cardinality=bool(len(observed_unique)<config['min_model_unique']),
                model_min_unique_required=int(minimum_unique),
                low_cardinality_override=bool(method in METHODS[2:] and discrete_exception and 2<=len(observed_unique)<config['min_model_unique']),
                decision_origin='statistical_reference' if method in METHODS[:2] else 'not_fitted',model_fitted=False)
    d = Detector(method,'ok',metadata=meta)
    if len(observed_fit)<config['min_fit'] or len(observed_calibration)<config['min_cal'] or not len(fit) or not len(calibration):
        d.status='insufficient_reference_or_calibration'
        return d
    if len(observed_unique)<2 and method!='iqr':
        if feature_name == 'ri' and method in ('three_sigma','isolation_forest','one_class_svm'):
            reference = float(observed_unique[0])
            d.center,d.lower,d.upper,d.scale = reference,reference,reference,1.
            meta.update(decision_origin='constant_reference_consistency',model_fitted=False,
                        reference_value=reference,reference_warning='constant_reference',
                        constant_reference=True,score_units='native_feature_units',
                        interpretation='reference_deviation_not_confirmed_fault')
            return _finish_calibration(d,calibration,config,reference,0.,reference,reference)
        d.status='constant_reference'
        return d
    q1,med,q3 = np.quantile(fit,[.25,.5,.75])
    std = fit.std(ddof=0)
    d.center = float(med)
    d.scale = 1.0 if method=='iqr' else float(q3-q1) if q3>q1 else float(std)
    if not np.isfinite(d.scale) or d.scale<=0:
        d.status='invalid_scale'
        return d
    if method == 'three_sigma':
        mean = float(fit.mean())
        d.lower,d.upper = mean-3*std,mean+3*std
        d.scale=float(std)
    elif method == 'iqr':
        d.lower,d.upper = float(q1-1.5*(q3-q1)),float(q3+1.5*(q3-q1))
        # Tukey fences do not require division by IQR. A zero-width fence is
        # computable, but is explicitly reported as a degenerate reference.
        d.scale=1.0
        meta.update(score_units='native_feature_units',zero_iqr=bool(q3==q1),
                    constant_reference=bool(len(unique)==1),
                    reference_warning='constant_reference' if len(unique)==1 else 'zero_iqr' if q3==q1 else 'none',
                    interpretation='reference_deviation_not_confirmed_fault',
                    q1=float(q1),q3=float(q3))
    else:
        if len(observed_unique)<minimum_unique:
            d.status='low_cardinality_reference'
            return d
        rng = np.random.default_rng(config['seed'])
        # Named CQI series can also contain continuous temporal averages. Only
        # truly low-cardinality references need category-preserving sampling.
        if discrete_exception and 2 <= len(observed_unique) <= 4:
            chosen = _category_preserving_indices(fit, config['max_ml_fit'], rng)
            if chosen is None:
                d.status='insufficient_sampling_budget_for_unique_values'
                return d
            meta['ml_fit_sampling']='all_rows' if len(fit)<=config['max_ml_fit'] else 'category_preserving_without_replacement'
            meta['ml_fit_value_counts']=[dict(value=float(value),reference_n=int((fit==value).sum()),
                                             observed_reference_n=int((observed_fit==value).sum()),
                                             fitted_n=int((fit[chosen]==value).sum())) for value in unique]
        else:
            chosen = rng.choice(len(fit),config['max_ml_fit'],replace=False) if len(fit)>config['max_ml_fit'] else np.arange(len(fit))
            meta['ml_fit_sampling']='all_rows' if len(fit)<=config['max_ml_fit'] else 'random_without_replacement'
        x = ((fit[chosen]-d.center)/d.scale).reshape(-1,1)
        meta['ml_fit_n']=len(x)
        meta['ml_fit_unique_n']=len(np.unique(x))
        if discrete_exception:
            meta['ml_fit_observed_unique_n']=len(np.intersect1d(np.unique(fit[chosen]),observed_unique))
            if meta['ml_fit_observed_unique_n']<2:
                d.status='low_cardinality_fit_sample'
                return d
        if meta['ml_fit_unique_n']<minimum_unique:
            d.status='low_cardinality_fit_sample'
            return d
        if method == 'isolation_forest':
            d.model=IsolationForest(n_estimators=config['if_trees'],max_samples=min(256,len(x)),
                contamination='auto',random_state=config['seed'],n_jobs=2)
        else:
            d.model=OneClassSVM(kernel='rbf',nu=config['svm_nu'],gamma='scale',
                tol=config['svm_tol'],max_iter=config['svm_max_iter'],cache_size=256)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('error',ConvergenceWarning)
                d.model.fit(x)
        except ConvergenceWarning:
            d.status='solver_not_converged'
            d.model=None
            return d
        meta.update(decision_origin='trained_model',model_fitted=True)
    return _finish_calibration(d,calibration,config,med,std,q1,q3)
