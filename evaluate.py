"""Shared metric module used by every training/eval step.

Headline metrics: F1, MCC, PR-AUC (never accuracy). Confidence intervals resample
TEST REPOSITORIES, not rows, so a CI reflects variation across unseen projects.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, matthews_corrcoef, precision_score, recall_score


def binary_metrics(y_true, y_score=None, threshold: float | None = None, y_pred=None) -> dict:
    """F1 / MCC at a threshold (or given hard predictions) and PR-AUC from scores.

    Hard 0/1 predictions (the rule baseline) give PR-AUC as a single operating point;
    `pr_auc_single_point` flags that so it is never compared as if it were a curve.
    """
    y_true = np.asarray(y_true).astype(int)
    if y_pred is None:
        if y_score is None or threshold is None:
            raise ValueError("pass y_pred, or y_score with a frozen threshold")
        y_pred = (np.asarray(y_score) >= threshold).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    scores = np.asarray(y_score) if y_score is not None else y_pred
    single_point = y_score is None or len(np.unique(scores)) <= 2
    has_both = 0 < y_true.sum() < len(y_true)
    return {
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)) if has_both else float("nan"),
        "pr_auc": float(average_precision_score(y_true, scores)) if y_true.sum() > 0 else float("nan"),
        "pr_auc_single_point": bool(single_point),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "threshold": threshold,
        "n": int(len(y_true)),
        "n_pos": int(y_true.sum()),
        "n_pred_pos": int(y_pred.sum()),
    }


def _group_index(groups) -> tuple[np.ndarray, list[np.ndarray]]:
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    return uniq, [np.flatnonzero(groups == g) for g in uniq]


def bootstrap_ci(y_true, y_pred, y_score, groups, metric: str, n_boot: int = 2000,
                 alpha: float = 0.05, seed: int = 42) -> dict:
    """Percentile CI for one metric, resampling groups (repositories) with replacement."""
    y_true, y_pred = np.asarray(y_true).astype(int), np.asarray(y_pred).astype(int)
    y_score = np.asarray(y_score) if y_score is not None else y_pred
    _, idx = _group_index(groups)
    rng = np.random.default_rng(seed)
    fn = _metric_fn(metric)
    vals = []
    for _ in range(n_boot):
        pick = np.concatenate([idx[i] for i in rng.integers(0, len(idx), len(idx))])
        v = fn(y_true[pick], y_pred[pick], y_score[pick])
        if not np.isnan(v):
            vals.append(v)
    point = fn(y_true, y_pred, y_score)
    lo, hi = (np.quantile(vals, [alpha / 2, 1 - alpha / 2]) if vals else (np.nan, np.nan))
    return {"metric": metric, "point": float(point), "ci_low": float(lo), "ci_high": float(hi),
            "n_boot_valid": len(vals), "n_groups": len(idx)}


def paired_bootstrap(y_true, pred_a, score_a, pred_b, score_b, groups, metric: str,
                     n_boot: int = 2000, alpha: float = 0.05, seed: int = 42) -> dict:
    """CI of metric(A) - metric(B) on the same resampled repositories; share of resamples where A <= B."""
    y_true = np.asarray(y_true).astype(int)
    pred_a, pred_b = np.asarray(pred_a).astype(int), np.asarray(pred_b).astype(int)
    score_a = np.asarray(score_a) if score_a is not None else pred_a
    score_b = np.asarray(score_b) if score_b is not None else pred_b
    _, idx = _group_index(groups)
    rng = np.random.default_rng(seed)
    fn = _metric_fn(metric)
    diffs = []
    for _ in range(n_boot):
        pick = np.concatenate([idx[i] for i in rng.integers(0, len(idx), len(idx))])
        d = fn(y_true[pick], pred_a[pick], score_a[pick]) - fn(y_true[pick], pred_b[pick], score_b[pick])
        if not np.isnan(d):
            diffs.append(d)
    diffs = np.asarray(diffs)
    point = fn(y_true, pred_a, score_a) - fn(y_true, pred_b, score_b)
    return {"metric": metric, "diff_point": float(point),
            "ci_low": float(np.quantile(diffs, alpha / 2)) if len(diffs) else float("nan"),
            "ci_high": float(np.quantile(diffs, 1 - alpha / 2)) if len(diffs) else float("nan"),
            "p_a_not_better": float((diffs <= 0).mean()) if len(diffs) else float("nan"),
            "n_boot_valid": int(len(diffs))}


def _metric_fn(metric: str) -> Callable[[np.ndarray, np.ndarray, np.ndarray], float]:
    def f1(y, p, _s):
        return f1_score(y, p, zero_division=0)

    def mcc(y, p, _s):
        return matthews_corrcoef(y, p) if 0 < y.sum() < len(y) else np.nan

    def pr_auc(y, _p, s):
        return average_precision_score(y, s) if y.sum() > 0 else np.nan

    fns = {"f1": f1, "mcc": mcc, "pr_auc": pr_auc}
    if metric not in fns:
        raise ValueError(f"unknown metric {metric!r}; headline metrics are {list(fns)}")
    return fns[metric]
