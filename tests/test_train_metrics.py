"""Step 8 safeguards: split/fold checks, repo-based fold mapping, row-wise features, per-fit
scale_pos_weight, in-fold imputation, OOF threshold, determinism."""
import copy

import numpy as np
import pandas as pd
import pytest

from common import load_config, p
from phase_b_metrics import train_metrics as tm


def _cfg(seeds=(1, 2)):
    cfg = copy.deepcopy(load_config())
    mc = cfg["metrics_models"]
    mc["seeds"] = list(seeds)
    mc["n_jobs"] = 1
    mc["permutation_importance_repeats"] = 2
    mc["logistic_regression"]["grid"] = {"C": [0.1, 1]}
    mc["random_forest"]["n_estimators"] = 20
    mc["random_forest"]["grid"] = {"max_depth": [3], "min_samples_leaf": [1], "max_features": ["sqrt"]}
    mc["xgboost"]["grid"] = {"max_depth": [2], "learning_rate": [0.1], "n_estimators": [20], "min_child_weight": [1]}
    return cfg


def _folds(repos_by_fold):
    allr = [r for rs in repos_by_fold for r in rs]
    return {"folds": [{"fold": i, "val_repos": rs, "train_repos": [r for r in allr if r not in rs]}
                      for i, rs in enumerate(repos_by_fold)]}


def _synthetic(n_repos=10, per_repo=20, seed=0):
    rng = np.random.default_rng(seed)
    cfg = load_config()
    rows = []
    for r in range(n_repos):
        for _ in range(per_repo):
            wmc = rng.gamma(2, 15)
            y = int(wmc + rng.normal(0, 10) > 40)
            row = {f: float(rng.integers(0, 20)) for f in cfg["metrics_models"]["features"]}
            row.update(ck_wmc=wmc, ck_tcc=rng.choice([-1.0, np.nan, rng.random()]), ck_lcc=rng.random(),
                       repository=f"r{r}", y_blob=y)
            rows.append(row)
    df = pd.DataFrame(rows)
    df["sample_id"] = range(len(df))
    df["y_blob"] = df["y_blob"].astype("Int64")
    return df


def test_check_splits_catches_leaks():
    split = {"train_repos": ["a", "b", "c", "d"], "test_repos": ["t"]}
    good = _folds([["a", "b"], ["c", "d"]])
    assert len(tm.check_splits(split, good)) == 4
    with pytest.raises(AssertionError, match="both sides of the primary"):
        tm.check_splits({**split, "test_repos": ["a"]}, good)
    bad = copy.deepcopy(good)
    bad["folds"][0]["train_repos"].append("a")         # val repo also in train
    with pytest.raises(AssertionError, match="CV fold 0"):
        tm.check_splits(split, bad)
    bad = copy.deepcopy(good)
    bad["folds"][1]["val_repos"].append("t")           # test repo inside a fold
    with pytest.raises(AssertionError, match="test repository"):
        tm.check_splits(split, bad)
    uncovered = {"folds": [{"fold": 0, "val_repos": ["a"], "train_repos": ["b", "c", "d"]},
                           {"fold": 1, "val_repos": ["c", "d"], "train_repos": ["a", "b"]}]}   # "b" never validated
    with pytest.raises(AssertionError, match="partition"):
        tm.check_splits(split, uncovered)


def test_fold_ids_follow_repository():
    folds = _folds([["a"], ["b", "c"]])
    ids = tm.fold_ids(pd.Series(["a", "b", "c", "a"]), folds)
    assert ids.tolist() == [0, 1, 1, 0]
    with pytest.raises(AssertionError):
        tm.fold_ids(pd.Series(["a", "zzz"]), folds)


def test_feature_frame_negative_and_blank_tcc_become_missing_with_flags():
    mc = load_config()["metrics_models"]
    df = pd.DataFrame({f: [1.0, 2.0, 3.0] for f in mc["features"]})
    df["ck_tcc"] = [-1.0, np.nan, 0.5]
    df["ck_lcc"] = [-1.0, 0.2, 0.5]
    X = tm.feature_frame(df, mc)
    assert X["ck_tcc"].isna().tolist() == [True, True, False]
    assert X["ck_tcc_missing"].tolist() == [1.0, 1.0, 0.0]
    assert X["ck_lcc_missing"].tolist() == [1.0, 0.0, 0.0]
    assert X["ck_lcom*_missing"].tolist() == [0.0, 0.0, 0.0]
    assert len(X.columns) == len(mc["features"]) + len(mc["missing_indicators"])


def test_xgboost_scale_pos_weight_from_its_own_training_rows():
    mc = load_config()["metrics_models"]
    m = tm.make_model("xgboost", {"max_depth": 2}, mc, 1, np.array([1, 0, 0, 0, 1, 0, 0, 0]), ["a"])
    assert m.get_params()["scale_pos_weight"] == 3.0


def test_logreg_imputation_fitted_on_training_rows_only():
    mc = load_config()["metrics_models"]
    df = _synthetic(n_repos=2, per_repo=30)
    X, y = tm.feature_frame(df, mc), df["y_blob"].astype(int).to_numpy()
    tr = np.arange(len(y)) < 30
    m = tm.make_model("logistic_regression", {"C": 1}, mc, 1, y[tr], list(X.columns)).fit(X[tr], y[tr])
    lin = m[0].named_transformers_["lin"][0]
    cols = [c for c in X.columns if c not in mc["log1p_features"]]
    assert np.allclose(lin.statistics_, X[tr][cols].median().to_numpy())


def test_max_f1_threshold():
    y = np.array([0, 0, 1, 1, 0, 1])
    s = np.array([0.1, 0.2, 0.8, 0.9, 0.7, 0.6])
    thr, f1 = tm.max_f1_threshold(y, s)
    assert thr == 0.6 and f1 == pytest.approx(6 / 7)


@pytest.mark.parametrize("name", ["logistic_regression", "random_forest", "xgboost"])
def test_train_one_is_deterministic_and_uses_folds(name):
    cfg = _cfg()
    df = _synthetic()
    folds = _folds([[f"r{i}", f"r{i + 5}"] for i in range(5)])
    a = tm.train_one(name, "blob", df, folds, cfg)
    b = tm.train_one(name, "blob", df, folds, cfg)
    assert [s["threshold"] for s in a["per_seed"]] == [s["threshold"] for s in b["per_seed"]]
    assert np.array_equal(a["_oof_mean"], b["_oof_mean"])
    assert len(a["tuning"]["grid"]) == len(list(__import__("sklearn.model_selection").model_selection
                                                .ParameterGrid(cfg["metrics_models"][name]["grid"])))
    assert all(len(g["cv_pr_auc_folds"]) == 5 for g in a["tuning"]["grid"])
    assert a["n_train"] == len(df)


def test_score_test_uses_frozen_thresholds_and_reports_sensitivity():
    cfg = _cfg()
    cfg["evaluation"]["bootstrap_n"] = 20
    df = _synthetic()
    df["y_blob_major_critical_only"] = df["y_blob"]
    train, test = df[df["repository"].isin([f"r{i}" for i in range(8)])], df[df["repository"].isin(["r8", "r9"])]
    folds = _folds([[f"r{i}", f"r{i + 4}"] for i in range(4)])
    r = tm.train_one("logistic_regression", "blob", train, folds, cfg)
    thresholds = [s["threshold"] for s in r["per_seed"]]
    tm.score_test(r, test, cfg)
    assert [s["threshold"] for s in r["per_seed"]] == thresholds
    assert [e["primary"]["threshold"] for e in r["test"]["per_seed"]] == thresholds
    sens = r["test"]["sensitivity_report_only"]["major_critical_only"]
    assert sens["seed_averaged_scores"]["metrics"] == r["test"]["primary"]["seed_averaged_scores"]["metrics"]


def test_real_splits_and_training_rows_have_no_test_repositories():
    cfg = load_config()
    if not p(cfg["paths"]["class_dataset"]).exists() or not p(cfg["paths"]["labels_class"]).exists():
        pytest.skip("local data not present")
    split, folds = tm.load_splits(cfg)
    tm.check_splits(split, folds)
    train = tm.load_train(cfg, split)
    assert not set(train["repository"]) & set(split["test_repos"])
    assert len(train) == split["n_train_samples"]
    for smell, (target, _) in tm.SMELLS.items():
        rows = train[train[target].notna()]
        tm.fold_ids(rows["repository"], folds)
