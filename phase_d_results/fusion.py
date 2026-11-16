"""Step 11 — fusion of the metrics arm and the embedding arm, per smell (Blob, Data Class).

Protocol = Step 8's (decisions 2026-11-15, config.yaml `fusion`); Step 8 / 10 helpers are reused unchanged:
  * Rows: primary-split TRAIN repos, primary labels, the same rows as Steps 8 and 10 (asserted).
  * Early fusion: Step 8's metric features (+ _missing flags) concatenated with
    StandardScaler -> PCA(fusion.early.embedding_pca_dims) -> StandardScaler of the Step 9 vectors, for each
    embedding strategy in fusion.early.embedding_strategies. Every scaler / imputer / PCA sits inside the
    Pipeline, so it is fitted on each fit's own training rows (each CV fold, then the full-train refit).
    Classifiers: LogReg (Step 8 preprocessing for the metric block) and XGBoost (raw metrics, native NaN),
    tuned on the 5 fixed folds with the first seed, as in Step 8.
  * Late fusion: logistic-regression stacker on [logit(metrics score), logit(embedding score)], the base
    models being the locked selections (metrics_models.locked_selection, embedding_models.locked_selection).
    The deployed stacker (per seed) is fitted on the saved Step 8 / 10 out-of-fold scores and applied to the
    saved test scores of the base models refitted on all training repos.
    Its own out-of-fold scores (CV PR-AUC, frozen threshold) come from NESTED cross-fitting: for outer fold k,
    the base models (locked hyperparameters) get inner out-of-fold scores on the other 4 folds, a stacker is
    fitted on those, and it scores fold k through base models fitted on those 4 folds. Fold k's labels
    therefore never reach the stacker that scores it. Remaining limitation: the base hyperparameters and the
    arm selections were tuned on CV over all 5 folds. Never a per-sample max of the two scores.
    Before any of this, the refitted base out-of-fold scores must reproduce the saved parquets
    (fusion.late.reproduce_tol), or the run aborts.
  * Per seed: OOF scores -> F1-maximizing threshold (frozen) -> refit / deploy. Seed-averaged threshold from
    the seed-averaged OOF scores. Selection: one fusion model per smell by CV PR-AUC (tuning seed) across
    the early candidates and the late stacker — fixed before test is read. Test is loaded once, after that,
    and every candidate is scored once. Sensitivity labels report-only. Paired comparisons with the arms
    belong to Step 12.
  * Steps 7-10 outputs are only read: their sha256 is checked before and after the run.

Outputs:
  eval/results/step11_early_<strategy>_<clf>_<smell>_<ts>.json, step11_late_<smell>_<ts>.json,
  eval/results/step11_summary_<ts>.json
  <paths.predictions>/fusion/<model>_<smell>_<ts>.parquet          (Step 8 schema)
  <fusion.models_dir>/<ts>/<model>_<smell>_seed<k>.joblib

    python -m phase_d_results.fusion     (requires both gates and complete Steps 8-10)
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import time

import joblib
import numpy as np
import pandas as pd
import sklearn
from scipy.special import logit
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from common import load_config, p, portable_path, read_json, timestamp, write_results
from evaluate import binary_metrics
from phase_b_metrics import train_metrics as tm
from phase_c_embeddings import train_embeddings as te

ARMS = ("metrics", "embedding")
EPS = 1e-7


# ---------------------------------------------------------------------------
# Locked base models and their saved predictions
# ---------------------------------------------------------------------------

def check_locked(cfg: dict, step8_summary: dict, step10_summary: dict) -> dict:
    """Base-model spec per smell from config, checked against the Step 8 / 10 summaries."""
    m_lock, e_lock = cfg["metrics_models"]["locked_selection"], cfg["embedding_models"]["locked_selection"]
    spec = {}
    for smell in tm.SMELLS:
        m, e = m_lock[smell], e_lock[smell]
        if step8_summary["selected"][smell] != m["model"]:
            sys.exit(f"{smell}: metrics_models.locked_selection {m['model']!r} != Step 8 summary "
                     f"{step8_summary['selected'][smell]!r}")
        e_name = f"{e['strategy']}_{e['head']}"
        if step10_summary["selected"][smell] != e_name:
            sys.exit(f"{smell}: embedding_models.locked_selection {e_name!r} != Step 10 summary "
                     f"{step10_summary['selected'][smell]!r}")
        spec[smell] = {"metrics": {"model": m["model"], "params": dict(m["params"])},
                       "embedding": {"strategy": e["strategy"], "head": e["head"], "params": dict(e["params"])}}
    return spec


def _one(files: list, what: str):
    if len(files) != 1:
        sys.exit(f"expected exactly one {what}, found {[f.name for f in files]}")
    return files[0]


def base_prediction_files(cfg: dict, spec: dict) -> dict:
    m_dir = p(cfg["metrics_models"]["locked_selection"]["predictions_dir"])
    el = cfg["embedding_models"]["locked_selection"]
    e_dir, run = p(el["predictions_dir"]), el["run_id"]
    out = {}
    for smell, s in spec.items():
        mm, ee = s["metrics"], s["embedding"]
        out[smell] = {"metrics": _one(sorted(m_dir.glob(f"{mm['model']}_{smell}_*.parquet")), f"Step 8 {mm['model']} {smell} parquet"),
                      "embedding": _one(sorted(e_dir.glob(f"{ee['strategy']}_{ee['head']}_{smell}_{run}.parquet")),
                                        f"Step 10 {ee['strategy']}_{ee['head']} {smell} parquet")}
    return out


def load_base_part(files: dict, part: str) -> dict:
    """{arm: DataFrame} of the saved predictions for one part ('train_oof' or 'test' — the latter only
    after every selection is fixed)."""
    out = {}
    for arm, f in files.items():
        d = pd.read_parquet(f)
        out[arm] = d[d["part"] == part].drop(columns="part").reset_index(drop=True)
    return out


def align_base(rows: pd.DataFrame, target: str, frame: pd.DataFrame, seeds: list[int]) -> np.ndarray:
    """(n_seeds, n_rows) saved scores in the order of `rows`; aborts on any id / label / repository mismatch."""
    f = frame.set_index("sample_id")
    ids = rows["sample_id"].astype(str)
    if set(ids) != set(f.index) or not f.index.is_unique:
        sys.exit("saved base predictions do not cover exactly the rows being fused")
    f = f.loc[ids]
    if not np.array_equal(f["y"].to_numpy(int), rows[target].astype(int).to_numpy()):
        sys.exit("saved base predictions carry different labels from the rows being fused")
    if not np.array_equal(f["repository"].to_numpy(), rows["repository"].to_numpy()):
        sys.exit("saved base predictions carry different repositories from the rows being fused")
    return np.array([f[f"score_seed{s}"].to_numpy(float) for s in seeds])


def protected_hashes(cfg: dict, extra: list) -> dict:
    """sha256 of every Step 7-10 output this step might touch: results files, predictions, Step 10 models."""
    files = []
    rd = p(cfg["paths"]["results_dir"])
    for pat in ("step7_*.json", "step8_*.json", "step9_*.json", "step10_*.json"):
        files += sorted(rd.glob(pat))
    pd_ = p(cfg["paths"]["predictions"])
    files += sorted((pd_ / "metrics").glob("*.parquet")) + sorted((pd_ / "embeddings").glob("*.parquet"))
    files += sorted(p(cfg["embedding_models"]["models_dir"]).rglob("*.joblib"))
    files += list(extra)
    out = {}
    for f in files:
        h = hashlib.sha256()
        with open(f, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        out[portable_path(f)] = h.hexdigest()
    return out


# ---------------------------------------------------------------------------
# Early fusion
# ---------------------------------------------------------------------------

def early_frame(rows: pd.DataFrame, cfg: dict, vectors: tuple) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Row-wise: metric features (Step 8's feature_frame) next to the raw embedding columns."""
    Xm = tm.feature_frame(rows, cfg["metrics_models"])
    E = te.attach(rows, vectors[0], vectors[1]).astype(float)
    emb_cols = [f"emb_{i}" for i in range(E.shape[1])]
    X = pd.concat([Xm, pd.DataFrame(E, columns=emb_cols, index=Xm.index)], axis=1)
    return X, list(Xm.columns), emb_cols


def _xgb_settings(cfg: dict) -> dict:
    if cfg["fusion"]["early"]["xgboost"] != "metrics_models":
        raise ValueError("fusion.early.xgboost must be 'metrics_models' (Step 8's XGBoost settings and grid)")
    return cfg["metrics_models"]["xgboost"]


def early_grid(clf: str, cfg: dict) -> dict:
    if clf == "logistic_regression":
        return cfg["fusion"]["early"]["logistic_regression"]["grid"]
    if clf == "xgboost":
        return _xgb_settings(cfg)["grid"]
    raise ValueError(f"unknown early-fusion classifier {clf!r}")


def make_early(clf: str, params: dict, cfg: dict, seed: int, y_train: np.ndarray, metric_cols: list[str],
               emb_cols: list[str]):
    fc, mc = cfg["fusion"]["early"], cfg["metrics_models"]
    if not fc["scale_metrics"]:
        raise ValueError("only fusion.early.scale_metrics: true is implemented")
    emb_steps = [StandardScaler(), PCA(n_components=fc["embedding_pca_dims"], svd_solver="full")]
    if fc["pca_rescale"]:
        emb_steps.append(StandardScaler())
    emb = make_pipeline(*emb_steps)
    if clf == "logistic_regression":
        c = fc["logistic_regression"]
        log_cols = [col for col in metric_cols if col in mc["log1p_features"]]
        lin_cols = [col for col in metric_cols if col not in log_cols]
        parts = []
        if log_cols:
            parts.append(("log", make_pipeline(SimpleImputer(strategy="median"),
                                               FunctionTransformer(np.log1p, feature_names_out="one-to-one"),
                                               StandardScaler()), log_cols))
        if lin_cols:
            parts.append(("lin", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), lin_cols))
        parts.append(("emb", emb, emb_cols))
        return make_pipeline(ColumnTransformer(parts), LogisticRegression(
            l1_ratio=0, class_weight=c["class_weight"], max_iter=c["max_iter"], random_state=seed, **params))
    if clf == "xgboost":
        _xgb_settings(cfg)
        pre = ColumnTransformer([("metrics", "passthrough", metric_cols), ("emb", emb, emb_cols)])
        return make_pipeline(pre, tm.make_model("xgboost", params, mc, seed, y_train, metric_cols))
    raise ValueError(f"unknown early-fusion classifier {clf!r}")


def early_oof(clf, params, cfg, seed, X, y, fid, metric_cols, emb_cols) -> tuple[np.ndarray, list[float], list[float]]:
    oof, fold_ap, pca_var = np.full(len(y), np.nan), [], []
    for k in np.unique(fid):
        tr, va = fid != k, fid == k
        m = make_early(clf, params, cfg, seed, y[tr], metric_cols, emb_cols).fit(X[tr], y[tr])
        oof[va] = m.predict_proba(X[va])[:, 1]
        fold_ap.append(float(average_precision_score(y[va], oof[va])))
        pca_var.append(float(m[0].named_transformers_["emb"][1].explained_variance_ratio_.sum()))
    assert not np.isnan(oof).any()
    return oof, fold_ap, pca_var


def train_early(strategy: str, clf: str, smell: str, train: pd.DataFrame, vectors: tuple, folds: dict,
                cfg: dict) -> dict:
    mc = cfg["metrics_models"]
    target = tm.SMELLS[smell][0]
    rows = train[train[target].notna()].reset_index(drop=True)
    X, metric_cols, emb_cols = early_frame(rows, cfg, vectors)
    y, fid = rows[target].astype(int).to_numpy(), tm.fold_ids(rows["repository"], folds)
    seeds, t0, grid_def = mc["seeds"], time.time(), early_grid(clf, cfg)
    grid = []
    for params in ParameterGrid(grid_def):
        _, ap, _ = early_oof(clf, params, cfg, seeds[0], X, y, fid, metric_cols, emb_cols)
        grid.append({"params": params, "cv_pr_auc_mean": float(np.mean(ap)), "cv_pr_auc_std": float(np.std(ap)),
                     "cv_pr_auc_folds": ap})
    best = max(grid, key=lambda g: g["cv_pr_auc_mean"])          # first in grid order on ties
    params = best["params"]
    per_seed, oofs, fitted, pca_var = [], [], [], None
    for seed in seeds:
        oof, ap, var = early_oof(clf, params, cfg, seed, X, y, fid, metric_cols, emb_cols)
        pca_var = pca_var or var
        thr, oof_f1 = tm.max_f1_threshold(y, oof)
        fitted.append(make_early(clf, params, cfg, seed, y, metric_cols, emb_cols).fit(X, y))
        per_seed.append({"seed": seed, "threshold": thr, "oof_f1_at_threshold": oof_f1,
                         "oof_pr_auc_pooled": float(average_precision_score(y, oof)), "cv_pr_auc_folds": ap})
        oofs.append(oof)
    oof_mean = np.mean(oofs, axis=0)
    thr_mean, f1_mean = tm.max_f1_threshold(y, oof_mean)
    fc = cfg["fusion"]["early"]
    out = {
        "fusion": "early", "strategy": strategy, "classifier": clf, "model": f"early_{strategy}_{clf}",
        "smell": smell, "target": target, "inputs": vectors[2],
        "n_train": int(len(y)), "n_train_pos": int(y.sum()), "n_train_repos": int(rows["repository"].nunique()),
        "features": {"metrics": metric_cols, "embedding_pca_dims": fc["embedding_pca_dims"],
                     "embedding_dim_raw": len(emb_cols), "pca_rescale": fc["pca_rescale"],
                     "dim_after_preprocessing": len(metric_cols) + fc["embedding_pca_dims"]},
        "pca_explained_variance_per_fold": pca_var,
        "tuning": {"tuning_seed": seeds[0], "metric": mc["tuning_metric"], "best_params": params,
                   "best_cv_pr_auc_mean": best["cv_pr_auc_mean"], "best_cv_pr_auc_std": best["cv_pr_auc_std"],
                   "best_on_grid_edge": te.grid_edges(params, grid_def), "grid": grid},
        "per_seed": per_seed,
        "seed_averaged": {"threshold": thr_mean, "oof_f1_at_threshold": f1_mean,
                          "oof_pr_auc_pooled": float(average_precision_score(y, oof_mean))},
        "train_seconds": round(time.time() - t0, 1),
    }
    if clf == "logistic_regression":
        lr, names = fitted[0][-1], fitted[0][0].get_feature_names_out()
        coef = dict(zip([n.split("__", 1)[1] for n in names], lr.coef_[0].tolist()))
        out["coefficients_seed0"] = {"metrics": {c: coef[c] for c in metric_cols},
                                     "abs_sum_metrics": float(sum(abs(coef[c]) for c in metric_cols)),
                                     "abs_sum_pca": float(sum(abs(v) for k, v in coef.items() if k not in metric_cols))}
    if cfg["embedding_models"]["size_diagnostic"]:
        out["size_diagnostic"] = te.size_diagnostic(oof_mean, y, np.log1p(rows["ck_loc"].astype(float).to_numpy()))
    out["_fitted"], out["_oof"], out["_oof_mean"] = fitted, oofs, oof_mean
    out["_rows"] = rows[["sample_id", "repository"]].assign(y=y)
    return out


# ---------------------------------------------------------------------------
# Late fusion
# ---------------------------------------------------------------------------

def base_inputs(rows: pd.DataFrame, cfg: dict, vectors: tuple) -> dict:
    return {"metrics": tm.feature_frame(rows, cfg["metrics_models"]),
            "embedding": te.attach(rows, vectors[0], vectors[1])}


def fit_base(arm: str, spec: dict, cfg: dict, seed: int, X, y: np.ndarray):
    """A base model with its locked hyperparameters, exactly as Step 8 / Step 10 fit it."""
    if arm == "metrics":
        return tm.make_model(spec["model"], spec["params"], cfg["metrics_models"], seed, y, list(X.columns)).fit(X, y)
    ec = cfg["embedding_models"]
    return te.fit_head(te.make_head(spec["head"], spec["params"], ec, seed), spec["head"], ec, X, y, {})


def make_stacker(cfg: dict, seed: int) -> LogisticRegression:
    lc = cfg["fusion"]["late"]
    if lc["stacker"] != "logistic_regression":
        raise ValueError("the late-fusion stacker must be logistic_regression (never a max of the two scores)")
    if lc["score_transform"] != "logit":
        raise ValueError("only fusion.late.score_transform: logit is implemented")
    return LogisticRegression(l1_ratio=0, C=lc["C"], class_weight=lc["class_weight"], max_iter=lc["max_iter"],
                              random_state=seed)


def stack_features(s_metrics: np.ndarray, s_embedding: np.ndarray) -> np.ndarray:
    return np.column_stack([logit(np.clip(s_metrics, EPS, 1 - EPS)), logit(np.clip(s_embedding, EPS, 1 - EPS))])


def nested_late(spec: dict, cfg: dict, seed: int, X: dict, y: np.ndarray, fid: np.ndarray) -> dict:
    """Nested cross-fitting for one seed. Returns the stacker OOF scores, the (non-nested) base OOF scores
    produced on the way (used for the reproduction check), per-fold PR-AUC and the fold stackers' weights."""
    n = len(y)
    stack_oof = np.full(n, np.nan)
    base_oof = {a: np.full(n, np.nan) for a in ARMS}
    fold_ap, coefs = [], []
    for k in np.unique(fid):
        tr, va = fid != k, fid == k
        outer = {}
        for a in ARMS:
            m = fit_base(a, spec[a], cfg, seed, X[a][tr], y[tr])
            outer[a] = m.predict_proba(X[a][va])[:, 1]
            base_oof[a][va] = outer[a]
        inner = {a: np.full(n, np.nan) for a in ARMS}
        for j in np.unique(fid[tr]):
            itr, iva = tr & (fid != j), fid == j
            for a in ARMS:
                inner[a][iva] = fit_base(a, spec[a], cfg, seed, X[a][itr], y[itr]).predict_proba(X[a][iva])[:, 1]
        assert not np.isnan(inner["metrics"][tr]).any() and np.isnan(inner["metrics"][va]).all()
        st = make_stacker(cfg, seed).fit(stack_features(inner["metrics"][tr], inner["embedding"][tr]), y[tr])
        stack_oof[va] = st.predict_proba(stack_features(outer["metrics"], outer["embedding"]))[:, 1]
        fold_ap.append(float(average_precision_score(y[va], stack_oof[va])))
        coefs.append({"fold": int(k), "w_metrics": float(st.coef_[0][0]), "w_embedding": float(st.coef_[0][1]),
                      "intercept": float(st.intercept_[0])})
    assert not np.isnan(stack_oof).any()
    return {"stack_oof": stack_oof, "base_oof": base_oof, "fold_ap": fold_ap, "fold_coefs": coefs}


def train_late(smell: str, train: pd.DataFrame, vectors: tuple, folds: dict, cfg: dict, spec: dict,
               saved_oof: dict) -> dict:
    """spec: this smell's base-model spec; saved_oof: {arm: saved train_oof predictions frame}."""
    mc, lc = cfg["metrics_models"], cfg["fusion"]["late"]
    target = tm.SMELLS[smell][0]
    rows = train[train[target].notna()].reset_index(drop=True)
    X, y = base_inputs(rows, cfg, vectors), rows[target].astype(int).to_numpy()
    fid, seeds, t0 = tm.fold_ids(rows["repository"], folds), mc["seeds"], time.time()
    saved = {a: align_base(rows, target, saved_oof[a], seeds) for a in ARMS}
    per_seed, oofs, fitted, reproduction = [], [], [], {a: [] for a in ARMS}
    for i, seed in enumerate(seeds):
        nl = nested_late(spec, cfg, seed, X, y, fid)
        for a in ARMS:
            d = float(np.max(np.abs(nl["base_oof"][a] - saved[a][i])))
            reproduction[a].append(d)
            if d > lc["reproduce_tol"]:
                sys.exit(f"{smell} seed {seed}: refitted {a} base OOF scores differ from the saved parquet by {d:.2e} "
                         f"(> {lc['reproduce_tol']}) — not the same base model; aborting.")
        oof = nl["stack_oof"]
        thr, oof_f1 = tm.max_f1_threshold(y, oof)
        st = make_stacker(cfg, seed).fit(stack_features(saved["metrics"][i], saved["embedding"][i]), y)
        fitted.append(st)
        per_seed.append({"seed": seed, "threshold": thr, "oof_f1_at_threshold": oof_f1,
                         "oof_pr_auc_pooled": float(average_precision_score(y, oof)), "cv_pr_auc_folds": nl["fold_ap"],
                         "deployed_stacker": {"w_metrics": float(st.coef_[0][0]), "w_embedding": float(st.coef_[0][1]),
                                              "intercept": float(st.intercept_[0])},
                         "nested_fold_stackers": nl["fold_coefs"],
                         "base_cv_pr_auc_folds": {a: [float(average_precision_score(y[fid == k], nl["base_oof"][a][fid == k]))
                                                      for k in np.unique(fid)] for a in ARMS}})
        oofs.append(oof)
    oof_mean = np.mean(oofs, axis=0)
    thr_mean, f1_mean = tm.max_f1_threshold(y, oof_mean)
    ap0 = per_seed[0]["cv_pr_auc_folds"]
    negative = [s["seed"] for s in per_seed if min(s["deployed_stacker"]["w_metrics"], s["deployed_stacker"]["w_embedding"]) < 0]
    out = {
        "fusion": "late", "model": "late_stacker", "smell": smell, "target": target,
        "base_models": spec, "inputs": {"embedding_vectors": vectors[2]},
        "n_train": int(len(y)), "n_train_pos": int(y.sum()), "n_train_repos": int(rows["repository"].nunique()),
        "stacker": {"model": lc["stacker"], "features": ["logit(metrics score)", "logit(embedding score)"],
                    "C": lc["C"], "class_weight": lc["class_weight"], "cross_fit": lc["cross_fit"],
                    "negative_weight_seeds": negative},
        "base_reproduction_max_abs_diff": {a: max(v) for a, v in reproduction.items()},
        "tuning": {"tuning_seed": seeds[0], "metric": mc["tuning_metric"], "best_params": {"C": lc["C"]},
                   "best_cv_pr_auc_mean": float(np.mean(ap0)), "best_cv_pr_auc_std": float(np.std(ap0)),
                   "note": "no tuning: fixed C; CV PR-AUC from the nested stacker OOF scores of the tuning seed"},
        "per_seed": per_seed,
        "seed_averaged": {"threshold": thr_mean, "oof_f1_at_threshold": f1_mean,
                          "oof_pr_auc_pooled": float(average_precision_score(y, oof_mean))},
        "limitation": "base hyperparameters and arm selections were tuned on CV over all 5 folds (incl. each outer "
                      "fold), so the nested stacker OOF scores remain slightly optimistic",
        "train_seconds": round(time.time() - t0, 1),
    }
    if cfg["embedding_models"]["size_diagnostic"]:
        out["size_diagnostic"] = te.size_diagnostic(oof_mean, y, np.log1p(rows["ck_loc"].astype(float).to_numpy()))
    out["_fitted"], out["_oof"], out["_oof_mean"] = fitted, oofs, oof_mean
    out["_rows"] = rows[["sample_id", "repository"]].assign(y=y)
    return out


# ---------------------------------------------------------------------------
# Test scoring (once, after training and selection are frozen)
# ---------------------------------------------------------------------------

def finish_test(res: dict, rows: pd.DataFrame, scores: np.ndarray, cfg: dict) -> None:
    """Fill res['test'] from (n_seeds, n_rows) test scores with the frozen thresholds (as Step 10's score_test)."""
    ev, target = cfg["evaluation"], res["target"]
    y, groups = rows[target].astype(int).to_numpy(), rows["repository"].to_numpy()
    smean = scores.mean(axis=0)
    sens_rules = cfg["labels"].get("sensitivity_severity_rules") or []
    per_seed = []
    for s, sc in zip(res["per_seed"], scores):
        entry = {"seed": s["seed"], "primary": binary_metrics(y, y_score=sc, threshold=s["threshold"])}
        for rule in sens_rules:
            ok = rows[f"{target}_{rule}"].notna().to_numpy()
            entry[rule] = binary_metrics(rows.loc[ok, f"{target}_{rule}"].astype(int), y_score=sc[ok],
                                         threshold=s["threshold"])
        per_seed.append(entry)
    thr = res["seed_averaged"]["threshold"]
    res["test"] = {
        "n_test": int(len(y)), "n_test_pos": int(y.sum()), "n_test_repos": int(rows["repository"].nunique()),
        "primary": {"mean_std_over_seeds": tm._mean_std(per_seed, "primary"),
                    "seed_averaged_scores": tm._metrics_with_ci(y, smean, thr, groups, ev, cfg["seed"])},
        "per_seed": per_seed, "sensitivity_report_only": {}}
    for rule in sens_rules:
        col = f"{target}_{rule}"
        ok = rows[col].notna().to_numpy()
        res["test"]["sensitivity_report_only"][rule] = {
            "label_column": col,
            "note": "primary-trained models and primary OOF thresholds; these labels drive no training or selection",
            "mean_std_over_seeds": tm._mean_std(per_seed, rule),
            "seed_averaged_scores": tm._metrics_with_ci(rows.loc[ok, col].astype(int).to_numpy(), smean[ok], thr,
                                                        groups[ok], ev, cfg["seed"])}
    res["_test_rows"] = rows.assign(y=y)
    res["_test_scores"], res["_test_mean"] = scores, smean


def score_early_test(res: dict, test: pd.DataFrame, vectors: tuple, cfg: dict) -> None:
    rows = test[test[res["target"]].notna()].reset_index(drop=True)
    X, _, _ = early_frame(rows, cfg, vectors)
    finish_test(res, rows, np.array([m.predict_proba(X)[:, 1] for m in res["_fitted"]]), cfg)


def score_late_test(res: dict, test: pd.DataFrame, saved_test: dict, cfg: dict) -> None:
    """Deployed stackers on the saved test scores of the base models refitted on all training repos."""
    rows = test[test[res["target"]].notna()].reset_index(drop=True)
    seeds = [s["seed"] for s in res["per_seed"]]
    base = {a: align_base(rows, res["target"], saved_test[a], seeds) for a in ARMS}
    scores = np.array([st.predict_proba(stack_features(base["metrics"][i], base["embedding"][i]))[:, 1]
                       for i, st in enumerate(res["_fitted"])])
    finish_test(res, rows, scores, cfg)


def save_outputs(res: dict, cfg: dict, ts: str) -> None:
    d = p(cfg["paths"]["predictions"]) / "fusion"
    d.mkdir(parents=True, exist_ok=True)
    f = d / f"{res['model']}_{res['smell']}_{ts}.parquet"
    te.predictions_frame(res).to_parquet(f, index=False)
    res["predictions_file"] = portable_path(f)
    if cfg["fusion"]["save_models"]:
        md = p(cfg["fusion"]["models_dir"]) / ts
        md.mkdir(parents=True, exist_ok=True)
        files = []
        for s, m in zip(res["per_seed"], res["_fitted"]):
            mf = md / f"{res['model']}_{res['smell']}_seed{s['seed']}.joblib"
            joblib.dump(m, mf)
            files.append(portable_path(mf))
        res["model_files"] = {"files": files, "sklearn": sklearn.__version__}


# ---------------------------------------------------------------------------

def _latest(cfg: dict, pattern: str):
    return _one(sorted(p(cfg["paths"]["results_dir"]).glob(pattern)), pattern)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.parse_args(argv)
    cfg = load_config()
    if not (cfg["gates"]["join_class_verified"] and cfg["gates"]["metrics_beat_rules"]):
        sys.exit("Steps 3 and 8 gates must both have passed before fusion.")
    fc, mc = cfg["fusion"], cfg["metrics_models"]
    make_stacker(cfg, 0)                                   # refuse a non-LogReg stacker before any work
    split, folds = tm.load_splits(cfg)
    checks = tm.check_splits(split, folds)
    train = tm.load_train(cfg, split)
    checks.append("training data holds training repositories only")
    split_file = p(cfg["paths"]["splits_dir"]) / "class_repo_split.json"

    s8 = read_json(_latest(cfg, "step8_summary_*.json"))
    s10_file = p(cfg["embedding_models"]["locked_selection"]["summary_file"])
    spec = check_locked(cfg, s8, read_json(s10_file))
    checks.append("locked base models in config equal the Step 8 and Step 10 summaries' CV selections")
    files = base_prediction_files(cfg, spec)
    strategies = sorted(set(fc["early"]["embedding_strategies"]) | {s["embedding"]["strategy"] for s in spec.values()})
    vectors = {s: te.load_vectors(cfg, s) for s in strategies}
    vec_files = [p(v[2]["vectors_file"]) for v in vectors.values()]
    hashes_before = protected_hashes(cfg, vec_files)
    step8 = te.step8_row_counts(cfg)
    saved_oof = {smell: load_base_part(files[smell], "train_oof") for smell in tm.SMELLS}

    # 1) everything on training repositories
    results: dict[str, list[dict]] = {s: [] for s in tm.SMELLS}
    for smell, (target, _) in tm.SMELLS.items():
        n_rows = int(train[target].notna().sum())
        assert n_rows == step8[smell]["n_train"], f"{smell}: {n_rows} training rows != Step 8's {step8[smell]['n_train']}"
        for strategy in fc["early"]["embedding_strategies"]:
            for clf in fc["early"]["classifiers"]:
                r = train_early(strategy, clf, smell, train, vectors[strategy], folds, cfg)
                results[smell].append(r)
                print(f"[train] {smell:<10} {r['model']:<40} CV PR-AUC {r['tuning']['best_cv_pr_auc_mean']:.3f}"
                      f"±{r['tuning']['best_cv_pr_auc_std']:.3f}  best {r['tuning']['best_params']}  "
                      f"PCA var {np.mean(r['pca_explained_variance_per_fold']):.2f}  ({r['train_seconds']}s)", flush=True)
        r = train_late(smell, train, vectors[spec[smell]["embedding"]["strategy"]], folds, cfg, spec[smell],
                       saved_oof[smell])
        results[smell].append(r)
        w = r["per_seed"][0]["deployed_stacker"]
        print(f"[train] {smell:<10} {'late_stacker':<40} CV PR-AUC {r['tuning']['best_cv_pr_auc_mean']:.3f}"
              f"±{r['tuning']['best_cv_pr_auc_std']:.3f}  w_metrics {w['w_metrics']:.2f} w_emb {w['w_embedding']:.2f}  "
              f"reproduction {r['base_reproduction_max_abs_diff']}  ({r['train_seconds']}s)", flush=True)
    checks += [f"training row counts equal Step 8's ({ {s: v['n_train'] for s, v in step8.items()} })",
               "early fusion: imputers, scalers and PCA fitted inside each fit (Pipeline), never on validation or test rows",
               "late fusion: stacker OOF scores from nested cross-fitting (fold k's labels never reach the stacker scoring fold k)",
               f"late fusion: refitted base OOF scores reproduce the saved Step 8/10 parquets (tol {fc['late']['reproduce_tol']})"]

    # 2) selection on CV PR-AUC — frozen before test is read
    cv = lambda r: r["tuning"]["best_cv_pr_auc_mean"]
    selected = {s: max(rs, key=cv)["model"] for s, rs in results.items()}
    best_early = {s: max((r for r in rs if r["fusion"] == "early"), key=cv)["model"] for s, rs in results.items()}
    print(f"[select] by CV PR-AUC: {selected}  (best early: {best_early})", flush=True)

    # 3) test, once
    test = tm.load_test(cfg, split)
    saved_test = {smell: load_base_part(files[smell], "test") for smell in tm.SMELLS}
    checks.append("test data and saved test base scores loaded only after tuning, thresholds and selection were fixed")
    ts = timestamp()
    for smell, rs in results.items():
        for r in rs:
            if r["fusion"] == "early":
                score_early_test(r, test, vectors[r["strategy"]], cfg)
            else:
                score_late_test(r, test, saved_test[smell], cfg)
            assert r["test"]["n_test"] == step8[smell]["n_test"], f"{smell}: test rows != Step 8's"
            save_outputs(r, cfg, ts)
            name = f"step11_{r['model']}_{smell}"
            payload = {"step": 11, "arm": "fusion", "leakage_checks": checks,
                       "threshold_policy": mc["threshold_policy"], "selected_for_smell": selected[smell] == r["model"],
                       **{k: v for k, v in r.items() if not k.startswith("_")}}
            out = write_results(cfg, name, payload, split_file)
            t = r["test"]["primary"]["mean_std_over_seeds"]
            print(f"[test]  {smell:<10} {r['model']:<40} F1 {t['f1']['mean']:.3f}±{t['f1']['std']:.3f}  "
                  f"MCC {t['mcc']['mean']:.3f}±{t['mcc']['std']:.3f}  PR-AUC {t['pr_auc']['mean']:.3f}±{t['pr_auc']['std']:.3f}"
                  f"  -> {out.name}", flush=True)

    hashes_after = protected_hashes(cfg, vec_files)
    unchanged = hashes_after == hashes_before
    assert unchanged, "a Step 7-10 output changed during the Step 11 run"
    s10 = read_json(s10_file)
    summary = {
        "step": 11, "leakage_checks": checks,
        "selection_metric": "CV PR-AUC (tuning seed, mean over 5 folds); late = nested stacker OOF",
        "cv_pr_auc": {s: {r["model"]: cv(r) for r in rs} for s, rs in results.items()},
        "selected": selected, "best_early": best_early,
        "test_mean_over_seeds": {s: {r["model"]: r["test"]["primary"]["mean_std_over_seeds"] for r in rs}
                                 for s, rs in results.items()},
        "late_stacker": {s: {"deployed_weights_per_seed": [x["deployed_stacker"] for x in rs[-1]["per_seed"]],
                             "negative_weight_seeds": rs[-1]["stacker"]["negative_weight_seeds"],
                             "base_reproduction_max_abs_diff": rs[-1]["base_reproduction_max_abs_diff"]}
                         for s, rs in results.items()},
        "size_diagnostic": {s: {r["model"]: r.get("size_diagnostic", {}).get("spearman_oof_vs_log_loc") for r in rs}
                            for s, rs in results.items()},
        "cv_references_training_folds": {
            s: {"metrics_arm_locked": mc["locked_selection"][s]["cv_pr_auc"],
                "embedding_arm_locked": cfg["embedding_models"]["locked_selection"][s]["cv_pr_auc"],
                "loc_only": (s10.get("cv_references_training_folds", {}).get(s, {}).get("loc_only") or {}).get("cv_pr_auc_mean")}
            for s in results},
        "base_models": spec, "base_prediction_files": {s: {a: portable_path(f) for a, f in v.items()} for s, v in files.items()},
        "protected_outputs_unchanged": {"n_files": len(hashes_before), "unchanged": unchanged, "sha256": hashes_before},
        "step8_row_counts": step8,
        "note": "Selection fixed before test was read; every candidate scored on test once. Paired comparisons with "
                "the arms and calibration are Step 12. fusion.locked_selection is recorded by the user after review.",
    }
    out = write_results(cfg, "step11_summary", summary, split_file)
    print(f"selected {selected} -> {out}  (Step 7-10 outputs unchanged: {len(hashes_before)} files)", flush=True)


if __name__ == "__main__":
    main()
