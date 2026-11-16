"""Step 12 safeguards on synthetic data: cross-fitted calibration never uses a fold's own labels, thresholds
come from calibrated OOF scores only, isotonic is monotone in [0, 1], paired tests resample repositories,
rule arm not calibrated, contrast folds and locked refits, reproduction check, per-repository counts."""
import copy

import numpy as np
import pandas as pd
import pytest

from common import load_config
from phase_d_results import final_eval as fe


def _cfg():
    cfg = copy.deepcopy(load_config())
    cfg["metrics_models"]["seeds"] = [1, 2]
    mc = cfg["metrics_models"]
    mc["features"], mc["negative_is_missing"], mc["missing_indicators"] = ["ck_loc", "ck_wmc"], [], []
    mc["log1p_features"] = ["ck_loc", "ck_wmc"]
    cfg["evaluation"]["bootstrap_n"] = 30
    return cfg


def _data(n_repos=10, per_repo=30, seed=0):
    rng = np.random.default_rng(seed)
    n = n_repos * per_repo
    y = rng.integers(0, 2, n) * (rng.uniform(size=n) < 0.6)
    oof = np.clip(0.3 * y + rng.uniform(0, 0.7, n), 0, 1)
    fid = np.repeat(np.arange(5), n // 5)
    rng.shuffle(fid)
    return y.astype(int), oof, fid


def _rows(n, seed=1, n_repos=6):
    rng = np.random.default_rng(seed)
    y = (rng.uniform(size=n) < 0.3).astype(int)
    df = pd.DataFrame({"sample_id": [str(5000 + i) for i in range(n)], "repository": [f"t{i % n_repos}" for i in range(n)],
                       "y_blob": pd.array(y, dtype="Int64"), "atfd": rng.integers(0, 20, n), "ck_wmc": rng.integers(0, 120, n),
                       "ck_tcc": rng.uniform(0, 1, n)})
    df["y_blob_major_critical_only"] = df["y_blob"]
    return df


def test_crossfit_isotonic_never_uses_the_folds_own_labels():
    y, oof, fid = _data()
    a = fe.crossfit_isotonic(oof, y, fid)
    y2 = y.copy()
    y2[fid == 0] = 1 - y2[fid == 0]
    b = fe.crossfit_isotonic(oof, y2, fid)
    assert np.array_equal(a[fid == 0], b[fid == 0])
    y3 = y.copy()
    y3[fid == 1] = 1 - y3[fid == 1]
    assert not np.array_equal(a[fid == 0], fe.crossfit_isotonic(oof, y3, fid)[fid == 0])


def test_calibration_is_monotone_and_bounded_and_threshold_ignores_test():
    y, oof, fid = _data()
    test = np.random.default_rng(3).uniform(-0.2, 1.2, 200)          # out of the training range on purpose
    c = fe.calibrate(oof, y, fid, test)
    order = np.argsort(test)
    assert np.all(np.diff(c["cal_test"][order]) >= -1e-12)
    assert c["cal_test"].min() >= 0 and c["cal_test"].max() <= 1
    c2 = fe.calibrate(oof, y, fid, test[::-1] * 0.5)
    assert c2["threshold"] == c["threshold"]


def test_evaluate_learned_applies_frozen_thresholds():
    cfg = _cfg()
    y, oof, fid = _data()
    rows = _rows(120)
    rng = np.random.default_rng(5)
    test = np.clip(0.3 * rows["y_blob"].to_numpy(int) + rng.uniform(0, 0.7, (2, 120)), 0, 1)
    r = fe.evaluate_learned(np.array([oof, oof * 0.9]), test, y, fid, rows, "y_blob", cfg, [1, 2])
    assert r["test"]["metrics"]["threshold"] == r["threshold"]["value"]
    assert [s["primary"]["threshold"] for s in r["per_seed"]] == [s["threshold"] for s in r["per_seed"]]
    assert np.array_equal(r["_"]["pred"], (r["_"]["score"] >= r["threshold"]["value"]).astype(int))
    assert set(r["calibration"]) >= {"test_brier", "test_ece"}
    assert "major_critical_only" in r["sensitivity_report_only"]


def test_rule_arm_is_not_calibrated_or_thresholded():
    cfg = _cfg()
    rows = _rows(90)
    r = fe.evaluate_rule(rows, "blob", cfg)
    exp, _ = fe.RULES["god_class"](rows, cfg["rule_baseline"]["god_class"])
    assert r["threshold"]["value"] is None and r["_"]["score"] is None
    assert np.array_equal(r["_"]["pred"], exp)


def test_paired_tests_identical_arms_and_rule_pairs():
    cfg = _cfg()
    rows = _rows(90)
    y = rows["y_blob"].to_numpy(int)
    s = np.random.default_rng(0).uniform(size=90)
    learned = {"pred": (s > 0.5).astype(int), "score": s}
    rule = {"pred": (s > 0.7).astype(int), "score": None}
    arms = {a: {"_": learned} for a in fe.LEARNED} | {"rule": {"_": rule}}
    out = fe.paired_tests(arms, y, rows["repository"].to_numpy(), cfg)
    same = [r for r in out if r["a"] == "fusion" and r["b"] == "metrics"]
    assert {r["metric"] for r in same} == {"f1", "mcc", "pr_auc"}
    assert all(r["diff_point"] == 0 and r["ci_low"] == 0 and r["ci_high"] == 0 for r in same)
    assert {r["metric"] for r in out if r["b"] == "rule"} == {"f1", "mcc"}
    assert len(out) == 5 * 3 + 4 * 2


def test_row_folds_are_stratified_and_refuse_other_protocols():
    cfg = _cfg()
    y = np.array([1] * 50 + [0] * 200)
    fid = fe.row_folds(y, cfg)
    assert set(fid) == set(range(5)) and (fid >= 0).all()
    assert all(y[fid == k].sum() == 10 for k in range(5))
    cfg["evaluation"]["contrast"]["refit"] = "retune"
    with pytest.raises(ValueError):
        fe.row_folds(y, cfg)


def test_refit_oof_of_a_fold_ignores_its_labels():
    cfg = _cfg()
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame({"ck_loc": rng.integers(5, 500, n), "ck_wmc": rng.integers(0, 80, n)})
    X = {"metrics": fe.tm.feature_frame(df, cfg["metrics_models"])}
    y = (np.log(df["ck_loc"]) + rng.normal(0, 1, n) > 4.5).astype(int).to_numpy()
    fid = fe.row_folds(y, cfg)
    spec = {"kind": "metrics", "model": "logistic_regression", "params": {"C": 0.1}}
    a, ta = fe.refit_scores(spec, cfg, 1, X, y, fid, X)
    y2 = y.copy()
    y2[fid == 0] = 1 - y2[fid == 0]
    b, _ = fe.refit_scores(spec, cfg, 1, X, y2, fid, X)
    assert np.allclose(a[fid == 0], b[fid == 0], atol=1e-12)
    assert ta.shape == (n,)


def test_check_reproduces_aborts_on_mismatch():
    res = {"uncalibrated_reference": {"f1": 0.5, "mcc": 0.3, "pr_auc": 0.6}}
    good = {"test": {"primary": {"seed_averaged_scores": {"metrics": {"f1": 0.5, "mcc": 0.3, "pr_auc": 0.6}}}}}
    assert fe.check_reproduces("metrics", res, good)["max_abs_diff"] == 0
    bad = copy.deepcopy(good)
    bad["test"]["primary"]["seed_averaged_scores"]["metrics"]["mcc"] = 0.31
    with pytest.raises(SystemExit, match="differ"):
        fe.check_reproduces("metrics", res, bad)


def test_ece_values():
    y = np.array([0, 0, 1, 1])
    assert fe.ece(y, np.array([0.0, 0.0, 1.0, 1.0]), 10) == 0
    assert fe.ece(y, np.array([0.5, 0.5, 0.5, 0.5]), 10) == pytest.approx(0.0)
    assert fe.ece(np.array([1, 1]), np.array([0.25, 0.25]), 10) == pytest.approx(0.75)


def test_per_repo_counts_add_up():
    rows = _rows(60)
    y = rows["y_blob"].to_numpy(int)
    pred = np.random.default_rng(1).integers(0, 2, 60)
    tab = fe.per_repo({"metrics": {"_": {"pred": pred}}}, rows, "y_blob", "blob")
    assert tab["n"].sum() == 60 and tab["n_pos"].sum() == y.sum()
    assert tab["metrics_tp"].sum() == ((pred == 1) & (y == 1)).sum()
    assert (tab[["metrics_tp", "metrics_fp", "metrics_fn", "metrics_tn"]].sum(axis=1) == tab["n"]).all()
    s = fe.per_repo_summary(tab, ["metrics"])
    assert s["n_repos"] == 6 and s["arms"]["metrics"]["errors_total"] == int((pred != y).sum())
