"""Step 10 safeguards on synthetic vectors: missing vectors abort, in-fold scaling (no leakage across
folds), balanced MLP weights, determinism per seed, frozen thresholds on test, Step 8 predictions schema."""
import copy

import numpy as np
import pandas as pd
import pytest
from sklearn.neural_network import MLPClassifier

from common import load_config
from phase_c_embeddings import train_embeddings as te


def _cfg():
    cfg = copy.deepcopy(load_config())
    cfg["metrics_models"]["seeds"] = [1, 2]
    ec = cfg["embedding_models"]
    ec["logistic_regression"]["grid"] = {"C": [0.01, 1]}
    ec["mlp"]["grid"] = {"hidden_layer_sizes": [[8]], "alpha": [0.01]}
    ec["mlp"]["max_iter"] = 50
    cfg["evaluation"]["bootstrap_n"] = 20
    return cfg


def _folds(repos_by_fold):
    allr = [r for rs in repos_by_fold for r in rs]
    return {"folds": [{"fold": i, "val_repos": rs, "train_repos": [r for r in allr if r not in rs]}
                      for i, rs in enumerate(repos_by_fold)]}


def _synthetic(n_repos=10, per_repo=20, dim=6, seed=0):
    rng = np.random.default_rng(seed)
    n = n_repos * per_repo
    vec = rng.normal(size=(n, dim)).astype(np.float32)
    y = (vec[:, 0] + rng.normal(0, 0.7, n) > 0.6).astype(int)
    df = pd.DataFrame({"sample_id": [str(1000 + i) for i in range(n)],          # text, as Step 5 stores it
                       "repository": [f"r{i // per_repo}" for i in range(n)],
                       "y_blob": pd.array(y, dtype="Int64"), "ck_loc": rng.integers(5, 500, n)})
    df["y_blob_major_critical_only"] = df["y_blob"]
    rowmap = {1000 + i: i for i in range(n)}
    return df, (rowmap, vec, {"vectors_file": "synthetic"})


def test_missing_vector_aborts():
    df, (rowmap, vec, _) = _synthetic()
    del rowmap[1005]
    with pytest.raises(SystemExit, match="no Step 9 vector"):
        te.attach(df, rowmap, vec)


def test_attach_keeps_row_order_with_text_ids():
    df, (rowmap, vec, _) = _synthetic()
    sub = df.iloc[[7, 3, 0]]
    assert np.array_equal(te.attach(sub, rowmap, vec), vec[[7, 3, 0]])


def test_oof_score_of_a_row_ignores_its_own_validation_fold():
    """Scaler and model are fitted on the other folds only: changing the other rows of row r's
    validation fold must not change r's out-of-fold score; changing a training row must."""
    cfg = _cfg()
    df, (rowmap, vec, _) = _synthetic()
    X, y = vec.astype(float), df["y_blob"].astype(int).to_numpy()
    folds = _folds([[f"r{i}", f"r{i + 5}"] for i in range(5)])
    fid = te.tm.fold_ids(df["repository"], folds)
    ec = cfg["embedding_models"]
    base, _ = te.oof_scores("logistic_regression", {"C": 1}, ec, 1, X, y, fid, {})
    r = int(np.flatnonzero(fid == 0)[0])
    X2 = X.copy()
    same_fold = np.flatnonzero(fid == 0)
    X2[same_fold[same_fold != r]] *= 50
    moved, _ = te.oof_scores("logistic_regression", {"C": 1}, ec, 1, X2, y, fid, {})
    assert moved[r] == pytest.approx(base[r], abs=1e-12)
    X3 = X.copy()
    X3[np.flatnonzero(fid == 1)] += 3
    shifted, _ = te.oof_scores("logistic_regression", {"C": 1}, ec, 1, X3, y, fid, {})
    assert shifted[r] != pytest.approx(base[r], abs=1e-6)


def test_mlp_gets_balanced_sample_weights(monkeypatch):
    seen = {}
    real_fit = MLPClassifier.fit

    def spy(self, X, y, sample_weight=None):
        seen["w"], seen["y"] = sample_weight, np.asarray(y)
        return real_fit(self, X, y, sample_weight=sample_weight)

    monkeypatch.setattr(MLPClassifier, "fit", spy)
    cfg = _cfg()
    y = np.array([1] * 10 + [0] * 30)
    X = np.random.default_rng(0).normal(size=(40, 4))
    stats = {}
    te.fit_head(te.make_head("mlp", {"hidden_layer_sizes": [4], "alpha": 0.01}, cfg["embedding_models"], 1),
                "mlp", cfg["embedding_models"], X, y, stats)
    w = seen["w"]
    assert w[y == 1].sum() == pytest.approx(w[y == 0].sum())
    assert stats["fits"] == 1 and "hit_max_iter" in stats


def test_early_stopping_is_refused():
    cfg = _cfg()
    cfg["embedding_models"]["mlp"]["early_stopping"] = True
    with pytest.raises(ValueError, match="early stopping"):
        te.make_head("mlp", {"hidden_layer_sizes": [4], "alpha": 0.01}, cfg["embedding_models"], 1)


@pytest.mark.parametrize("head", ["logistic_regression", "mlp"])
def test_train_one_is_deterministic_and_complete(head):
    cfg = _cfg()
    df, vectors = _synthetic()
    folds = _folds([[f"r{i}", f"r{i + 5}"] for i in range(5)])
    a = te.train_one("truncation", head, "blob", df, vectors, folds, cfg)
    b = te.train_one("truncation", head, "blob", df, vectors, folds, cfg)
    assert np.array_equal(a["_oof_mean"], b["_oof_mean"])
    assert [s["threshold"] for s in a["per_seed"]] == [s["threshold"] for s in b["per_seed"]]
    assert a["n_train"] == len(df) and a["dim"] == 6
    assert all(len(g["cv_pr_auc_folds"]) == 5 for g in a["tuning"]["grid"])
    sd = a["size_diagnostic"]["spearman_oof_vs_log_loc"]
    assert set(sd) == {"all", "within_positive", "within_negative"}


def test_grid_edges():
    grid = {"C": [0.01, 0.1, 1], "alpha": [0.5]}
    assert te.grid_edges({"C": 0.01, "alpha": 0.5}, grid) == {"C": True}
    assert te.grid_edges({"C": 0.1, "alpha": 0.5}, grid) == {"C": False}
    assert te.grid_edges({"hidden_layer_sizes": [64]}, {"hidden_layer_sizes": [[64], [256]]}) == {"hidden_layer_sizes": True}


def test_score_test_uses_frozen_thresholds_and_step8_schema():
    cfg = _cfg()
    df, vectors = _synthetic()
    train, test = df[df["repository"].isin([f"r{i}" for i in range(8)])], df[df["repository"].isin(["r8", "r9"])]
    folds = _folds([[f"r{i}", f"r{i + 4}"] for i in range(4)])
    r = te.train_one("hierarchical_mean", "logistic_regression", "blob", train, vectors, folds, cfg)
    thresholds = [s["threshold"] for s in r["per_seed"]]
    te.score_test(r, test, vectors, cfg)
    assert [e["primary"]["threshold"] for e in r["test"]["per_seed"]] == thresholds
    assert r["test"]["n_test"] == len(test)
    pred = te.predictions_frame(r)
    assert list(pred.columns) == ["sample_id", "repository", "y", "part", "score_seed1", "score_seed2", "score_mean"]
    assert set(pred["part"]) == {"train_oof", "test"} and len(pred) == len(df)
    assert not set(pred.loc[pred.part == "train_oof", "repository"]) & set(test["repository"])


def test_loc_only_reference_uses_the_folds():
    df, _ = _synthetic()
    folds = _folds([[f"r{i}", f"r{i + 5}"] for i in range(5)])
    ref = te.loc_only_reference(df["y_blob"].astype(int).to_numpy(), np.log1p(df["ck_loc"].to_numpy(float)),
                                te.tm.fold_ids(df["repository"], folds), 1)
    assert len(ref["cv_pr_auc_folds"]) == 5 and 0 <= ref["cv_pr_auc_mean"] <= 1
