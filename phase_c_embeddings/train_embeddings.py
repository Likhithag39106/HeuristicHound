"""Step 10 — classifier heads on frozen CodeBERT vectors: (truncation | hierarchical_mean) x
(LogReg | shallow MLP) x (Blob | Data Class).

Protocol = Step 8's (phase_b_metrics.train_metrics, whose helpers are reused unchanged):
  * Rows: primary-split TRAIN repos, primary labels; rows not rated for a smell dropped for that smell.
    Every labelled row must have a Step 9 vector (a missing one aborts: silently dropping it would make
    the comparison with Step 8 unfair), and the row counts must equal Step 8's.
  * Inputs: embeddings.token_pooling vectors from Step 9. StandardScaler inside each Pipeline, so it is
    fitted on each fit's own training rows. No PCA or feature selection here.
  * Tuning: embedding_models grids, mean PR-AUC over the 5 fixed CV folds, first seed.
  * Per seed: OOF scores -> F1-maximizing threshold (frozen) -> refit on all train repos.
  * Selection: best (strategy, head) per smell by CV PR-AUC, also the best head per strategy — fixed
    before test is read. Test is loaded once, after that, and scored once.
  * Size diagnostic (training folds only): Spearman of seed-averaged OOF scores vs log1p(LOC), overall
    and within each class, plus a LOC-only LogReg CV PR-AUC reference per smell.
  * Sensitivity labels scored on test, report-only, as in Step 8. No gate here; comparisons with the
    metrics arm and the rule baseline belong to Steps 11-12.

Outputs:
  eval/results/step10_<strategy>_<head>_<smell>_<timestamp>.json, eval/results/step10_summary_<timestamp>.json
  <paths.predictions>/embeddings/<strategy>_<head>_<smell>_<timestamp>.parquet   (Step 8 schema)
  <embedding_models.models_dir>/<timestamp>/<strategy>_<head>_<smell>_seed<k>.joblib

    python -m phase_c_embeddings.train_embeddings     (requires both gates so far and a complete Step 9)
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings

import joblib
import numpy as np
import pandas as pd
import sklearn
from scipy.stats import spearmanr
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import ParameterGrid
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from common import load_config, p, portable_path, read_json, timestamp, write_results
from evaluate import binary_metrics
from phase_b_metrics import train_metrics as tm
from phase_c_embeddings.encode import out_dir as embeddings_dir, sha256_file


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_vectors(cfg: dict, strategy: str) -> tuple[dict[int, int], np.ndarray, dict]:
    """(sample_id -> row, vectors, provenance) for one Step 9 strategy; refuses an unfinished one."""
    root = embeddings_dir(cfg)
    d = root / strategy
    if not (read_json(d / "manifest.json") or {}).get("finalized"):
        sys.exit(f"Step 9 output {d} is not finalized — run phase_c_embeddings.encode first.")
    f = d / f"vectors_{cfg['embeddings']['token_pooling']}.npy"
    vec = np.load(f)
    idx = pd.read_parquet(d / "index.parquet")
    assert (idx["row"] == np.arange(len(idx))).all() and len(idx) == len(vec)
    fp = read_json(root / "fingerprint.json") or {}
    return ({int(s): int(r) for s, r in zip(idx["sample_id"], idx["row"])}, vec,
            {"vectors_file": portable_path(f), "vectors_sha256": sha256_file(f),
             "step9_fingerprint_id": fp.get("id"), "token_pooling": cfg["embeddings"]["token_pooling"]})


def attach(rows: pd.DataFrame, rowmap: dict[int, int], vec: np.ndarray) -> np.ndarray:
    """Vectors for these rows, in their order. Aborts if any row has no vector."""
    ids = rows["sample_id"].astype("int64")
    missing = sorted(set(ids) - set(rowmap))
    if missing:
        sys.exit(f"{len(missing)} labelled samples have no Step 9 vector (e.g. {missing[:5]}) — not training.")
    return vec[ids.map(rowmap).to_numpy()]


def step8_row_counts(cfg: dict) -> dict:
    """n_train / n_test per smell from the Step 8 results of the locked models (the comparison target)."""
    out = {}
    for smell, sel in cfg["metrics_models"]["locked_selection"].items():
        if not isinstance(sel, dict) or "model" not in sel:
            continue
        files = sorted(p(cfg["paths"]["results_dir"]).glob(f"step8_{sel['model']}_{smell}_*.json"))
        if files:
            r = read_json(files[-1])
            out[smell] = {"n_train": r["n_train"], "n_test": r["test"]["n_test"], "file": files[-1].name}
    return out


# ---------------------------------------------------------------------------
# Heads
# ---------------------------------------------------------------------------

def make_head(head: str, params: dict, ec: dict, seed: int):
    c = ec[head]
    if head == "logistic_regression":
        # l1_ratio=0 is sklearn >= 1.8's spelling of penalty="l2" (as in Step 8)
        return make_pipeline(StandardScaler(), LogisticRegression(
            l1_ratio=0, class_weight=c["class_weight"], max_iter=c["max_iter"], random_state=seed, **params))
    if head == "mlp":
        if c["early_stopping"]:
            raise ValueError("early stopping holds out random rows across repositories; keep it off")
        return make_pipeline(StandardScaler(), MLPClassifier(
            hidden_layer_sizes=tuple(params["hidden_layer_sizes"]), alpha=params["alpha"], solver=c["solver"],
            learning_rate_init=c["learning_rate_init"], max_iter=c["max_iter"], early_stopping=False,
            random_state=seed))
    raise ValueError(f"unknown head {head!r}")


def fit_head(model, head: str, ec: dict, X: np.ndarray, y: np.ndarray, stats: dict):
    """Fit; the MLP gets balanced per-row weights. Counts MLP fits that stopped at max_iter."""
    if head == "mlp":
        if ec["mlp"]["sample_weight"] != "balanced":
            raise ValueError("embedding_models.mlp.sample_weight must be balanced")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model.fit(X, y, mlpclassifier__sample_weight=compute_sample_weight("balanced", y))
        stats["fits"] = stats.get("fits", 0) + 1
        stats["hit_max_iter"] = stats.get("hit_max_iter", 0) + int(model[-1].n_iter_ >= model[-1].max_iter)
    else:
        model.fit(X, y)
    return model


def oof_scores(head, params, ec, seed, X, y, fid, stats) -> tuple[np.ndarray, list[float]]:
    oof = np.full(len(y), np.nan)
    fold_ap = []
    for k in np.unique(fid):
        tr, va = fid != k, fid == k
        m = fit_head(make_head(head, params, ec, seed), head, ec, X[tr], y[tr], stats)
        oof[va] = m.predict_proba(X[va])[:, 1]
        fold_ap.append(float(average_precision_score(y[va], oof[va])))
    assert not np.isnan(oof).any()
    return oof, fold_ap


def grid_edges(params: dict, grid: dict) -> dict:
    """True for each tuned parameter whose chosen value is the first or last grid value."""
    out = {}
    for k, values in grid.items():
        if len(values) > 1:
            out[k] = params[k] in (values[0], values[-1])
    return out


def size_diagnostic(oof: np.ndarray, y: np.ndarray, log_loc: np.ndarray) -> dict:
    def rho(a, b):
        return float(spearmanr(a, b).statistic) if len(a) > 2 else float("nan")
    return {"spearman_oof_vs_log_loc": {"all": rho(oof, log_loc), "within_positive": rho(oof[y == 1], log_loc[y == 1]),
                                        "within_negative": rho(oof[y == 0], log_loc[y == 0])},
            "note": "seed-averaged out-of-fold scores, training repositories only"}


def loc_only_reference(y: np.ndarray, log_loc: np.ndarray, fid: np.ndarray, seed: int) -> dict:
    """CV PR-AUC of LogReg on log1p(LOC) alone, same folds — how far size alone gets."""
    ap = []
    for k in np.unique(fid):
        tr, va = fid != k, fid == k
        m = make_pipeline(StandardScaler(), LogisticRegression(l1_ratio=0, class_weight="balanced", random_state=seed))
        m.fit(log_loc[tr].reshape(-1, 1), y[tr])
        ap.append(float(average_precision_score(y[va], m.predict_proba(log_loc[va].reshape(-1, 1))[:, 1])))
    return {"cv_pr_auc_mean": float(np.mean(ap)), "cv_pr_auc_std": float(np.std(ap)), "cv_pr_auc_folds": ap}


def train_one(strategy: str, head: str, smell: str, train: pd.DataFrame, vectors: tuple, folds: dict,
              cfg: dict) -> dict:
    """Everything on training repos: tuning, per-seed OOF thresholds, refits, size diagnostic."""
    ec, mc = cfg["embedding_models"], cfg["metrics_models"]
    target = tm.SMELLS[smell][0]
    rows = train[train[target].notna()].reset_index(drop=True)
    rowmap, vec, prov = vectors
    X, y = attach(rows, rowmap, vec), rows[target].astype(int).to_numpy()
    fid = tm.fold_ids(rows["repository"], folds)
    seeds, stats, t0 = mc["seeds"], {}, time.time()
    grid = []
    for params in ParameterGrid(ec[head]["grid"]):
        _, ap = oof_scores(head, params, ec, seeds[0], X, y, fid, stats)
        grid.append({"params": params, "cv_pr_auc_mean": float(np.mean(ap)), "cv_pr_auc_std": float(np.std(ap)),
                     "cv_pr_auc_folds": ap})
    best = max(grid, key=lambda g: g["cv_pr_auc_mean"])          # first in grid order on ties
    params = best["params"]
    per_seed, oofs, fitted = [], [], []
    for seed in seeds:
        oof, ap = oof_scores(head, params, ec, seed, X, y, fid, stats)
        thr, oof_f1 = tm.max_f1_threshold(y, oof)
        fitted.append(fit_head(make_head(head, params, ec, seed), head, ec, X, y, stats))
        per_seed.append({"seed": seed, "threshold": thr, "oof_f1_at_threshold": oof_f1,
                         "oof_pr_auc_pooled": float(average_precision_score(y, oof)), "cv_pr_auc_folds": ap})
        oofs.append(oof)
    oof_mean = np.mean(oofs, axis=0)
    thr_mean, f1_mean = tm.max_f1_threshold(y, oof_mean)
    out = {
        "strategy": strategy, "head": head, "model": f"{strategy}_{head}", "smell": smell, "target": target,
        "inputs": prov, "n_train": int(len(y)), "n_train_pos": int(y.sum()),
        "n_train_repos": int(rows["repository"].nunique()), "dim": int(X.shape[1]),
        "tuning": {"tuning_seed": seeds[0], "metric": mc["tuning_metric"], "best_params": params,
                   "best_cv_pr_auc_mean": best["cv_pr_auc_mean"], "best_cv_pr_auc_std": best["cv_pr_auc_std"],
                   "best_on_grid_edge": grid_edges(params, ec[head]["grid"]), "grid": grid},
        "per_seed": per_seed,
        "seed_averaged": {"threshold": thr_mean, "oof_f1_at_threshold": f1_mean,
                          "oof_pr_auc_pooled": float(average_precision_score(y, oof_mean))},
        "train_seconds": round(time.time() - t0, 1),
    }
    if stats:
        out["mlp_convergence"] = {**stats, "note": "fits that stopped at max_iter (no early stopping by design)"}
    if ec["size_diagnostic"]:
        out["size_diagnostic"] = size_diagnostic(oof_mean, y, np.log1p(rows["ck_loc"].astype(float).to_numpy()))
    out["_fitted"], out["_oof"], out["_oof_mean"] = fitted, oofs, oof_mean
    out["_rows"] = rows[["sample_id", "repository"]].assign(y=y)
    return out


# ---------------------------------------------------------------------------
# Test scoring (once, after training and selection are frozen)
# ---------------------------------------------------------------------------

def score_test(res: dict, test: pd.DataFrame, vectors: tuple, cfg: dict) -> None:
    """Fills res['test'] like Step 8's score_test, with the frozen per-seed and seed-averaged thresholds."""
    ev = cfg["evaluation"]
    target = res["target"]
    rows = test[test[target].notna()].reset_index(drop=True)
    X, y, groups = attach(rows, vectors[0], vectors[1]), rows[target].astype(int).to_numpy(), rows["repository"].to_numpy()
    scores = np.array([m.predict_proba(X)[:, 1] for m in res["_fitted"]])
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


def predictions_frame(res: dict) -> pd.DataFrame:
    """OOF + test scores in the Step 8 predictions schema."""
    seeds = [s["seed"] for s in res["per_seed"]]
    oof = res["_rows"].assign(part="train_oof")
    for seed, o in zip(seeds, res["_oof"]):
        oof[f"score_seed{seed}"] = o
    oof["score_mean"] = res["_oof_mean"]
    te = res["_test_rows"][["sample_id", "repository", "y"]].assign(part="test")
    for seed, s in zip(seeds, res["_test_scores"]):
        te[f"score_seed{seed}"] = s
    te["score_mean"] = res["_test_mean"]
    return pd.concat([oof, te], ignore_index=True)


def save_outputs(res: dict, cfg: dict, ts: str) -> None:
    d = p(cfg["paths"]["predictions"]) / "embeddings"
    d.mkdir(parents=True, exist_ok=True)
    f = d / f"{res['model']}_{res['smell']}_{ts}.parquet"
    predictions_frame(res).to_parquet(f, index=False)
    res["predictions_file"] = portable_path(f)
    if cfg["embedding_models"]["save_models"]:
        md = p(cfg["embedding_models"]["models_dir"]) / ts
        md.mkdir(parents=True, exist_ok=True)
        files = []
        for s, m in zip(res["per_seed"], res["_fitted"]):
            mf = md / f"{res['model']}_{res['smell']}_seed{s['seed']}.joblib"
            joblib.dump(m, mf)
            files.append(portable_path(mf))
        res["model_files"] = {"files": files, "sklearn": sklearn.__version__}


# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.parse_args(argv)
    cfg = load_config()
    if not (cfg["gates"]["join_class_verified"] and cfg["gates"]["metrics_beat_rules"]):
        sys.exit("Steps 3 and 8 gates must both have passed before the embedding arm.")
    ec, mc = cfg["embedding_models"], cfg["metrics_models"]
    split, folds = tm.load_splits(cfg)
    checks = tm.check_splits(split, folds)
    train = tm.load_train(cfg, split)
    checks.append("training data holds training repositories only")
    split_file = p(cfg["paths"]["splits_dir"]) / "class_repo_split.json"
    vectors = {s: load_vectors(cfg, s) for s in ec["strategies"]}
    step8 = step8_row_counts(cfg)

    # 1) everything on training repositories
    results: dict[str, list[dict]] = {s: [] for s in tm.SMELLS}
    loc_ref = {}
    for smell, (target, _) in tm.SMELLS.items():
        rows = train[train[target].notna()]
        if smell in step8:
            assert len(rows) == step8[smell]["n_train"], f"{smell}: {len(rows)} training rows != Step 8's {step8[smell]['n_train']}"
        if ec["size_diagnostic"]:
            loc_ref[smell] = loc_only_reference(rows[target].astype(int).to_numpy(),
                                                np.log1p(rows["ck_loc"].astype(float).to_numpy()),
                                                tm.fold_ids(rows["repository"], folds), mc["seeds"][0])
        for strategy in ec["strategies"]:
            for head in ec["heads"]:
                r = train_one(strategy, head, smell, train, vectors[strategy], folds, cfg)
                results[smell].append(r)
                sd = r.get("size_diagnostic", {}).get("spearman_oof_vs_log_loc", {})
                print(f"[train] {smell:<10} {strategy:<17} {head:<19} CV PR-AUC {r['tuning']['best_cv_pr_auc_mean']:.3f}"
                      f"±{r['tuning']['best_cv_pr_auc_std']:.3f}  best {r['tuning']['best_params']}  "
                      f"rho(LOC) {sd.get('all', float('nan')):.2f}  ({r['train_seconds']}s)", flush=True)
    checks.append(f"training row counts equal Step 8's ({ {s: v['n_train'] for s, v in step8.items()} })")

    # 2) selection on CV PR-AUC — frozen before test is read
    cv = lambda r: r["tuning"]["best_cv_pr_auc_mean"]
    selected = {s: max(rs, key=cv)["model"] for s, rs in results.items()}
    best_per_strategy = {s: {st: max((r for r in rs if r["strategy"] == st), key=cv)["model"] for st in ec["strategies"]}
                         for s, rs in results.items()}
    print(f"[select] by CV PR-AUC: {selected}", flush=True)

    # 3) test, once
    test = tm.load_test(cfg, split)
    checks.append("test data loaded only after tuning, thresholds and model selection were fixed")
    ts = timestamp()
    for smell, rs in results.items():
        for r in rs:
            score_test(r, test, vectors[r["strategy"]], cfg)
            if smell in step8:
                assert r["test"]["n_test"] == step8[smell]["n_test"], f"{smell}: test rows != Step 8's"
            save_outputs(r, cfg, ts)
            payload = {"step": 10, "arm": "embedding", "leakage_checks": checks,
                       "threshold_policy": mc["threshold_policy"], "selected_for_smell": selected[smell] == r["model"],
                       **{k: v for k, v in r.items() if not k.startswith("_")}}
            out = write_results(cfg, f"step10_{r['strategy']}_{r['head']}_{smell}", payload, split_file)
            t = r["test"]["primary"]["mean_std_over_seeds"]
            print(f"[test]  {smell:<10} {r['strategy']:<17} {r['head']:<19} F1 {t['f1']['mean']:.3f}±{t['f1']['std']:.3f}  "
                  f"MCC {t['mcc']['mean']:.3f}±{t['mcc']['std']:.3f}  PR-AUC {t['pr_auc']['mean']:.3f}±{t['pr_auc']['std']:.3f}"
                  f"  -> {out.name}", flush=True)

    summary = {
        "step": 10, "leakage_checks": checks,
        "selection_metric": "CV PR-AUC (tuning seed, mean over 5 folds)",
        "cv_pr_auc": {s: {r["model"]: cv(r) for r in rs} for s, rs in results.items()},
        "selected": selected, "best_head_per_strategy": best_per_strategy,
        "test_mean_over_seeds": {s: {r["model"]: r["test"]["primary"]["mean_std_over_seeds"] for r in rs}
                                 for s, rs in results.items()},
        "size_diagnostic": {s: {r["model"]: r.get("size_diagnostic", {}).get("spearman_oof_vs_log_loc") for r in rs}
                            for s, rs in results.items()},
        "cv_references_training_folds": {
            s: {"loc_only": loc_ref.get(s),
                "metrics_arm_locked": mc["locked_selection"].get(s, {}).get("cv_pr_auc")} for s in results},
        "step8_row_counts": step8,
        "note": "No gate in Step 10. Test comparisons with the metrics arm and the rule baseline are Steps 11-12.",
    }
    out = write_results(cfg, "step10_summary", summary, split_file)
    print(f"selected {selected} -> {out}")


if __name__ == "__main__":
    main()
