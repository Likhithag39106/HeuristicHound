"""Step 11 safeguards on synthetic data: early fusion fits PCA/scalers inside the training folds, output
dimensions, nested late fusion never uses a fold's own labels, base refits reproduce Steps 8/10, input
mismatches abort, no non-LogReg stacker, frozen thresholds on test, Step 8 predictions schema."""
import copy

import numpy as np
import pandas as pd
import pytest

from common import load_config
from phase_d_results import fusion as fu

METRICS = ["ck_loc", "ck_wmc", "ck_tcc"]


def _cfg():
    cfg = copy.deepcopy(load_config())
    mc = cfg["metrics_models"]
    mc["seeds"] = [1, 2]
    mc["features"], mc["negative_is_missing"], mc["missing_indicators"] = METRICS, ["ck_tcc"], ["ck_tcc"]
    mc["log1p_features"] = ["ck_loc", "ck_wmc"]
    mc["xgboost"]["grid"] = {"max_depth": [2], "learning_rate": [0.1], "n_estimators": [20], "min_child_weight": [1]}
    mc["n_jobs"] = 1
    fe = cfg["fusion"]["early"]
    fe["embedding_pca_dims"] = 3
    fe["logistic_regression"]["grid"] = {"C": [0.1, 1]}
    cfg["evaluation"]["bootstrap_n"] = 20
    return cfg


def _folds(repos_by_fold):
    allr = [r for rs in repos_by_fold for r in rs]
    return {"folds": [{"fold": i, "val_repos": rs, "train_repos": [r for r in allr if r not in rs]}
                      for i, rs in enumerate(repos_by_fold)]}


def _synthetic(n_repos=10, per_repo=20, dim=8, seed=0):
    rng = np.random.default_rng(seed)
    n = n_repos * per_repo
    vec = rng.normal(size=(n, dim)).astype(np.float32)
    loc = rng.integers(5, 500, n)
    y = (vec[:, 0] + np.log(loc) / 3 + rng.normal(0, 0.7, n) > 2.2).astype(int)
    df = pd.DataFrame({"sample_id": [str(1000 + i) for i in range(n)],
                       "repository": [f"r{i // per_repo}" for i in range(n)],
                       "y_blob": pd.array(y, dtype="Int64"), "ck_loc": loc,
                       "ck_wmc": rng.integers(0, 80, n), "ck_tcc": rng.uniform(-1, 1, n)})
    df["y_blob_major_critical_only"] = df["y_blob"]
    rowmap = {1000 + i: i for i in range(n)}
    return df, (rowmap, vec, {"vectors_file": "synthetic"})


def _spec():
    return {"metrics": {"model": "logistic_regression", "params": {"C": 0.1}},
            "embedding": {"strategy": "truncation", "head": "logistic_regression", "params": {"C": 0.1}}}


FOLDS5 = [[f"r{i}", f"r{i + 5}"] for i in range(5)]


def _xy(cfg, df, vectors):
    rows = df.reset_index(drop=True)
    X, mcols, ecols = fu.early_frame(rows, cfg, vectors)
    return rows, X, mcols, ecols, rows["y_blob"].astype(int).to_numpy(), fu.tm.fold_ids(rows["repository"], _folds(FOLDS5))


@pytest.mark.parametrize("clf", ["logistic_regression", "xgboost"])
def test_early_oof_score_ignores_its_own_validation_fold(clf):
    """Scalers, imputer and PCA are fitted on the other folds only: changing the other rows of row r's
    validation fold must not move r's OOF score; changing a training-fold row must."""
    cfg = _cfg()
    df, vectors = _synthetic()
    _, X, mc, ec, y, fid = _xy(cfg, df, vectors)
    params = {"C": 1} if clf == "logistic_regression" else {"max_depth": 2, "learning_rate": 0.1, "n_estimators": 20,
                                                             "min_child_weight": 1}
    base, _, _ = fu.early_oof(clf, params, cfg, 1, X, y, fid, mc, ec)
    r = int(np.flatnonzero(fid == 0)[0])
    same = np.flatnonzero(fid == 0)
    X2 = X.copy()
    X2.loc[same[same != r], ec] *= 50
    moved, _, _ = fu.early_oof(clf, params, cfg, 1, X2, y, fid, mc, ec)
    assert moved[r] == pytest.approx(base[r], abs=1e-12)
    X3 = X.copy()
    X3.loc[np.flatnonzero(fid == 1), ec] += 3
    X3.loc[np.flatnonzero(fid == 1), "ck_loc"] *= 7
    shifted, _, _ = fu.early_oof(clf, params, cfg, 1, X3, y, fid, mc, ec)
    if clf == "logistic_regression":
        assert shifted[r] != pytest.approx(base[r], abs=1e-6)


def test_early_dimensions_are_metrics_plus_pca():
    cfg = _cfg()
    df, vectors = _synthetic()
    _, X, mc, ec, y, _ = _xy(cfg, df, vectors)
    m = fu.make_early("logistic_regression", {"C": 1}, cfg, 1, y, mc, ec).fit(X, y)
    assert m[0].transform(X).shape[1] == len(mc) + cfg["fusion"]["early"]["embedding_pca_dims"]
    assert len(mc) == len(METRICS) + 1                         # + ck_tcc_missing


def test_nested_late_never_uses_the_scored_folds_labels():
    cfg = _cfg()
    df, vectors = _synthetic()
    rows = df.reset_index(drop=True)
    X, y = fu.base_inputs(rows, cfg, vectors), rows["y_blob"].astype(int).to_numpy()
    fid = fu.tm.fold_ids(rows["repository"], _folds(FOLDS5))
    a = fu.nested_late(_spec(), cfg, 1, X, y, fid)
    y2 = y.copy()
    k0 = fid == 0
    y2[k0] = 1 - y2[k0]                                         # flip every label in fold 0
    b = fu.nested_late(_spec(), cfg, 1, X, y2, fid)
    assert np.allclose(a["stack_oof"][k0], b["stack_oof"][k0], atol=1e-12)
    assert not np.allclose(a["stack_oof"][~k0], b["stack_oof"][~k0], atol=1e-6)


def test_nested_base_oof_reproduces_step8_and_step10_code():
    cfg = _cfg()
    df, vectors = _synthetic()
    rows = df.reset_index(drop=True)
    X, y = fu.base_inputs(rows, cfg, vectors), rows["y_blob"].astype(int).to_numpy()
    fid = fu.tm.fold_ids(rows["repository"], _folds(FOLDS5))
    nl = fu.nested_late(_spec(), cfg, 1, X, y, fid)
    m8, _ = fu.tm.oof_scores("logistic_regression", {"C": 0.1}, cfg["metrics_models"], 1, X["metrics"], y, fid)
    m10, _ = fu.te.oof_scores("logistic_regression", {"C": 0.1}, cfg["embedding_models"], 1, X["embedding"], y, fid, {})
    assert np.allclose(nl["base_oof"]["metrics"], m8, atol=1e-12)
    assert np.allclose(nl["base_oof"]["embedding"], m10, atol=1e-12)


def _saved_frames(cfg, df, vectors, part="train_oof"):
    """Saved-parquet stand-ins built with the Step 8 / 10 code itself."""
    rows = df.reset_index(drop=True)
    X, y = fu.base_inputs(rows, cfg, vectors), rows["y_blob"].astype(int).to_numpy()
    fid = fu.tm.fold_ids(rows["repository"], _folds(FOLDS5))
    out = {}
    for arm, fn in (("metrics", lambda s: fu.tm.oof_scores("logistic_regression", {"C": 0.1}, cfg["metrics_models"], s,
                                                            X["metrics"], y, fid)[0]),
                    ("embedding", lambda s: fu.te.oof_scores("logistic_regression", {"C": 0.1}, cfg["embedding_models"],
                                                              s, X["embedding"], y, fid, {})[0])):
        f = rows[["sample_id", "repository"]].assign(y=y)
        for s in cfg["metrics_models"]["seeds"]:
            f[f"score_seed{s}"] = fn(s)
        out[arm] = f.sample(frac=1, random_state=0)            # order must not matter
    return out


def test_train_late_reproduction_check_aborts_on_a_different_base():
    cfg = _cfg()
    df, vectors = _synthetic()
    saved = _saved_frames(cfg, df, vectors)
    r = fu.train_late("blob", df, vectors, _folds(FOLDS5), cfg, _spec(), saved)
    assert max(r["base_reproduction_max_abs_diff"].values()) < 1e-9
    assert r["stacker"]["features"] == ["logit(metrics score)", "logit(embedding score)"]
    saved["metrics"]["score_seed1"] += 1e-3
    with pytest.raises(SystemExit, match="differ from the saved parquet"):
        fu.train_late("blob", df, vectors, _folds(FOLDS5), cfg, _spec(), saved)


def test_align_base_aborts_on_mismatch():
    df, _ = _synthetic()
    f = df[["sample_id", "repository"]].assign(y=df["y_blob"].astype(int), score_seed1=0.5)
    assert fu.align_base(df, "y_blob", f.iloc[::-1], [1]).shape == (1, len(df))
    bad = f.copy()
    bad.loc[0, "y"] = 1 - bad.loc[0, "y"]
    with pytest.raises(SystemExit, match="different labels"):
        fu.align_base(df, "y_blob", bad, [1])
    with pytest.raises(SystemExit, match="exactly the rows"):
        fu.align_base(df, "y_blob", f.iloc[1:], [1])


def test_locked_selection_must_match_summaries():
    cfg = load_config()
    s8 = {"selected": {"blob": "logistic_regression", "data_class": "xgboost"}}
    s10 = {"selected": {"blob": "truncation_logistic_regression", "data_class": "hierarchical_mean_logistic_regression"}}
    spec = fu.check_locked(cfg, s8, s10)
    assert spec["data_class"]["embedding"]["strategy"] == "hierarchical_mean"
    with pytest.raises(SystemExit, match="Step 10 summary"):
        fu.check_locked(cfg, s8, {"selected": {**s10["selected"], "blob": "truncation_mlp"}})
    with pytest.raises(SystemExit, match="Step 8 summary"):
        fu.check_locked(cfg, {"selected": {**s8["selected"], "blob": "xgboost"}}, s10)


def test_stacker_must_be_logistic_regression():
    cfg = _cfg()
    cfg["fusion"]["late"]["stacker"] = "max"
    with pytest.raises(ValueError, match="never a max"):
        fu.make_stacker(cfg, 1)


def test_test_scoring_uses_frozen_thresholds_and_step8_schema():
    cfg = _cfg()
    df, vectors = _synthetic()
    train_repos, test_repos = [f"r{i}" for i in range(8)], ["r8", "r9"]
    train, test = df[df.repository.isin(train_repos)], df[df.repository.isin(test_repos)]
    folds = _folds([[f"r{i}", f"r{i + 4}"] for i in range(4)])
    r = fu.train_early("truncation", "logistic_regression", "blob", train, vectors, folds, cfg)
    thr = [s["threshold"] for s in r["per_seed"]]
    fu.score_early_test(r, test, vectors, cfg)
    assert [e["primary"]["threshold"] for e in r["test"]["per_seed"]] == thr
    pred = fu.te.predictions_frame(r)
    assert list(pred.columns) == ["sample_id", "repository", "y", "part", "score_seed1", "score_seed2", "score_mean"]
    assert not set(pred.loc[pred.part == "train_oof", "repository"]) & set(test_repos)
    assert len(pred) == len(df)

    # late: saved OOF on train repos, saved test scores from base models refitted on all train repos
    rows = train.reset_index(drop=True)
    X, y = fu.base_inputs(rows, cfg, vectors), rows["y_blob"].astype(int).to_numpy()
    fid = fu.tm.fold_ids(rows["repository"], folds)
    trows = test.reset_index(drop=True)
    Xt = fu.base_inputs(trows, cfg, vectors)
    saved, saved_test = {}, {}
    for arm in fu.ARMS:
        spec = _spec()[arm]
        f = rows[["sample_id", "repository"]].assign(y=y)
        ft = trows[["sample_id", "repository"]].assign(y=trows["y_blob"].astype(int))
        for s in cfg["metrics_models"]["seeds"]:
            f[f"score_seed{s}"] = fu.nested_late(_spec(), cfg, s, X, y, fid)["base_oof"][arm]
            ft[f"score_seed{s}"] = fu.fit_base(arm, spec, cfg, s, X[arm], y).predict_proba(Xt[arm])[:, 1]
        saved[arm], saved_test[arm] = f, ft
    late = fu.train_late("blob", train, vectors, folds, cfg, _spec(), saved)
    lthr = [s["threshold"] for s in late["per_seed"]]
    fu.score_late_test(late, test, saved_test, cfg)
    assert [e["primary"]["threshold"] for e in late["test"]["per_seed"]] == lthr
    assert late["test"]["n_test"] == len(test)
    assert 0 <= late["_test_mean"].min() and late["_test_mean"].max() <= 1
