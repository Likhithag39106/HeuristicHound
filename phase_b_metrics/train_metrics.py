"""Step 8 [GATE] — CK-metrics classifiers per smell: LogReg -> RandomForest -> XGBoost.

Protocol (PRD Step 8; decisions 2026-10-28, see config.yaml `metrics_models`):
  * Training rows: primary-split TRAIN repos only, primary labels (y_blob, y_data_class); rows not
    rated for a smell are dropped for that smell. Test rows are loaded only in `score_test`, after
    every model, threshold and selection is fixed.
  * Features: metrics_models.features; CK's -1 for TCC/LCC -> missing; <feature>_missing flags.
    These are row-wise (nothing fitted). LogReg: median impute, log1p on skewed counts, scale —
    fitted inside each fold (Pipeline). RF / XGBoost: raw features, missing values handled natively.
  * Tuning: grid in config, scored by mean PR-AUC over the 5 fixed CV folds (class_cv_folds.json,
    folds mapped by repository), first seed only.
  * Per seed: out-of-fold scores with the tuned parameters -> F1-maximizing threshold (frozen) ->
    refit on all train repos -> score test once. Report mean +/- std over seeds; bootstrap CIs over
    test repositories on the seed-averaged scores (threshold from the seed-averaged OOF scores).
  * Best model per smell = highest CV PR-AUC (never test). Gate: its mean test F1 and MCC must both
    exceed the rule baseline's, for both smells. This script reports the gate; it never edits
    config.yaml (gates.metrics_beat_rules is flipped by the user only).
  * Sensitivity labels (labels.sensitivity_severity_rules) are scored on test report-only, with
    the same models and thresholds; they are never used for training, thresholds or selection.

Outputs:
  eval/results/step8_<model>_<smell>_<timestamp>.json    one per model and smell
  eval/results/step8_summary_<timestamp>.json            selection + gate comparison vs Step 7
  <paths.predictions>/metrics/<model>_<smell>_<timestamp>.parquet   OOF + test scores (Steps 11-12)

    python -m phase_b_metrics.train_metrics      (requires gates.join_class_verified and Steps 5-6)
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from xgboost import XGBClassifier

from common import load_config, p, portable_path, read_json, timestamp, write_results
from evaluate import binary_metrics, bootstrap_ci, paired_bootstrap
from phase_b_metrics.rule_baseline import RULES

# smell key -> (primary label column, rule_baseline rule name)
SMELLS = {"blob": ("y_blob", "god_class"), "data_class": ("y_data_class", "data_class")}
HEADLINE = ("f1", "mcc", "pr_auc")


# ---------------------------------------------------------------------------
# Data and leakage safeguards
# ---------------------------------------------------------------------------

def load_splits(cfg: dict) -> tuple[dict, dict]:
    d = p(cfg["paths"]["splits_dir"])
    split, folds = read_json(d / "class_repo_split.json"), read_json(d / "class_cv_folds.json")
    if split is None or folds is None:
        sys.exit("eval/splits/class_repo_split.json or class_cv_folds.json missing — run Step 6 first.")
    return split, folds


def check_splits(split: dict, folds: dict) -> list[str]:
    """Assert the split files are internally consistent; return the checks that ran."""
    train, test = set(split["train_repos"]), set(split["test_repos"])
    assert not train & test, "repository on both sides of the primary split"
    seen: set[str] = set()
    for f in folds["folds"]:
        tr, va = set(f["train_repos"]), set(f["val_repos"])
        assert not tr & va, f"repository on both sides of CV fold {f['fold']}"
        assert not (tr | va) & test, f"test repository inside CV fold {f['fold']}"
        assert tr == train - va, f"CV fold {f['fold']} train repos != train repos minus its val repos"
        assert not va & seen, f"repository in more than one CV validation fold ({f['fold']})"
        seen |= va
    assert seen == train, "CV validation folds do not partition the training repos"
    return ["primary train/test repositories disjoint",
            "each CV fold: train/val repositories disjoint",
            "no test repository in any CV fold",
            "CV validation folds partition the training repositories exactly"]


def _joined(cfg: dict, repos: list[str]) -> pd.DataFrame:
    labels = pd.read_parquet(p(cfg["paths"]["labels_class"]))
    ds = pd.read_parquet(p(cfg["paths"]["class_dataset"]))
    rows = labels[labels["repository"].isin(repos)]
    out = rows.merge(ds.drop(columns=["repository", "repo_key"], errors="ignore"), on="sample_id", how="inner")
    assert out["sample_id"].is_unique
    return out.reset_index(drop=True)


def load_train(cfg: dict, split: dict) -> pd.DataFrame:
    df = _joined(cfg, split["train_repos"])
    assert not set(df["repository"]) & set(split["test_repos"]), "test repository in training data"
    return df


def load_test(cfg: dict, split: dict) -> pd.DataFrame:
    """Only called from score_test, after training, thresholds and selection are frozen."""
    df = _joined(cfg, split["test_repos"])
    assert set(df["repository"]) <= set(split["test_repos"])
    return df


def fold_ids(repos: pd.Series, folds: dict) -> np.ndarray:
    """CV fold of each training row, by its repository (the fold whose val_repos holds it)."""
    repo_to_fold = {r: f["fold"] for f in folds["folds"] for r in f["val_repos"]}
    ids = repos.map(repo_to_fold)
    assert ids.notna().all(), "training row whose repository is in no CV validation fold"
    return ids.to_numpy(dtype=int)


def feature_frame(df: pd.DataFrame, mc: dict) -> pd.DataFrame:
    """Row-wise feature preparation (no statistic is fitted here, so no fold can leak into another)."""
    X = df[mc["features"]].apply(pd.to_numeric, errors="coerce").astype(float)
    for c in mc["negative_is_missing"]:
        X.loc[X[c] < 0, c] = np.nan
    for c in mc["missing_indicators"]:
        X[f"{c}_missing"] = X[c].isna().astype(float)
    return X


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def make_model(name: str, params: dict, mc: dict, seed: int, y_train: np.ndarray, columns: list[str]):
    c = mc[name]
    if name == "logistic_regression":
        if c["penalty"] != "l2":
            raise ValueError("only an L2 penalty is implemented")
        log_cols = [col for col in columns if col in mc["log1p_features"]]
        lin_cols = [col for col in columns if col not in log_cols]
        pre = ColumnTransformer([
            ("log", make_pipeline(SimpleImputer(strategy="median"), FunctionTransformer(np.log1p, feature_names_out="one-to-one"),
                                  StandardScaler()), log_cols),
            ("lin", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), lin_cols)])
        # l1_ratio=0 is sklearn >= 1.8's spelling of penalty="l2"
        return make_pipeline(pre, LogisticRegression(l1_ratio=0, class_weight=c["class_weight"],
                                                     max_iter=c["max_iter"], random_state=seed, **params))
    if name == "random_forest":
        return RandomForestClassifier(n_estimators=c["n_estimators"], class_weight=c["class_weight"],
                                      random_state=seed, n_jobs=mc["n_jobs"], **params)
    if name == "xgboost":
        if c["scale_pos_weight"] != "auto":
            raise ValueError("metrics_models.xgboost.scale_pos_weight must be auto")
        return XGBClassifier(scale_pos_weight=scale_pos_weight(y_train), subsample=c["subsample"],
                             colsample_bytree=c["colsample_bytree"], tree_method=c["tree_method"],
                             importance_type="gain", random_state=seed, n_jobs=mc["n_jobs"], **params)
    raise ValueError(f"unknown model {name!r}")


def scale_pos_weight(y: np.ndarray) -> float:
    """negatives / positives of the rows this model is fitted on (each CV fold computes its own)."""
    pos = int(np.sum(y == 1))
    return float((len(y) - pos) / pos)


def oof_scores(name: str, params: dict, mc: dict, seed: int, X: pd.DataFrame, y: np.ndarray,
               fid: np.ndarray) -> tuple[np.ndarray, list[float]]:
    oof = np.full(len(y), np.nan)
    fold_ap = []
    for k in np.unique(fid):
        tr, va = fid != k, fid == k
        m = make_model(name, params, mc, seed, y[tr], list(X.columns))
        m.fit(X[tr], y[tr])
        oof[va] = m.predict_proba(X[va])[:, 1]
        fold_ap.append(float(average_precision_score(y[va], oof[va])))
    assert not np.isnan(oof).any()
    return oof, fold_ap


def max_f1_threshold(y: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    """Score threshold maximizing F1 (predict positive when score >= threshold) and that F1."""
    prec, rec, thr = precision_recall_curve(y, scores)
    prec, rec = prec[:-1], rec[:-1]          # the last point has no threshold
    denom = prec + rec
    f1 = np.divide(2 * prec * rec, denom, out=np.zeros_like(denom), where=denom > 0)
    i = int(np.argmax(f1))
    return float(thr[i]), float(f1[i])


def tune(name: str, mc: dict, seed: int, X: pd.DataFrame, y: np.ndarray, fid: np.ndarray) -> dict:
    grid = []
    for params in ParameterGrid(mc[name]["grid"]):
        _, ap = oof_scores(name, params, mc, seed, X, y, fid)
        grid.append({"params": params, "cv_pr_auc_mean": float(np.mean(ap)),
                     "cv_pr_auc_std": float(np.std(ap)), "cv_pr_auc_folds": ap})
    best = max(grid, key=lambda g: g["cv_pr_auc_mean"])   # first in grid order on ties
    return {"tuning_seed": seed, "metric": mc["tuning_metric"], "best_params": best["params"],
            "best_cv_pr_auc_mean": best["cv_pr_auc_mean"], "best_cv_pr_auc_std": best["cv_pr_auc_std"],
            "grid": grid}


def importances(name: str, fitted: list, params: dict, mc: dict, seed: int, X: pd.DataFrame,
                y: np.ndarray, fid: np.ndarray) -> dict:
    cols = list(X.columns)
    if name == "logistic_regression":
        pipe = fitted[0]                                  # deterministic: every seed gives the same model
        names = [n.split("__", 1)[1] for n in pipe[0].get_feature_names_out()]
        coef = dict(zip(names, pipe[-1].coef_[0].tolist()))
        return {"standardized_coefficients": {c: coef[c] for c in cols},
                "note": "coefficients on median-imputed, log1p (counts) and standardized features"}
    kind = "impurity" if name == "random_forest" else "gain"
    imp = np.mean([m.feature_importances_ for m in fitted], axis=0)
    perm = []
    for k in np.unique(fid):                              # CV validation folds only, never test
        tr, va = fid != k, fid == k
        m = make_model(name, params, mc, seed, y[tr], cols).fit(X[tr], y[tr])
        r = permutation_importance(m, X[va], y[va], scoring="average_precision",
                                   n_repeats=mc["permutation_importance_repeats"], random_state=seed,
                                   n_jobs=1)
        perm.append(r.importances_mean)
    perm = np.asarray(perm)
    return {f"{kind}_importance_mean_over_seeds": dict(zip(cols, imp.tolist())),
            "permutation_importance_pr_auc_drop": {
                c: {"mean_over_folds": float(perm[:, i].mean()), "std_over_folds": float(perm[:, i].std())}
                for i, c in enumerate(cols)},
            "permutation_note": f"PR-AUC drop on each CV validation fold (model fitted on the other folds, "
                                f"seed {seed}), {mc['permutation_importance_repeats']} repeats, averaged over folds"}


def train_one(name: str, smell: str, train: pd.DataFrame, folds: dict, cfg: dict) -> dict:
    """Everything that uses training repos only: tuning, per-seed OOF thresholds, refits, importance."""
    mc = cfg["metrics_models"]
    target = SMELLS[smell][0]
    rows = train[train[target].notna()].reset_index(drop=True)
    X, y = feature_frame(rows, mc), rows[target].astype(int).to_numpy()
    fid = fold_ids(rows["repository"], folds)
    seeds = mc["seeds"]
    t0 = time.time()
    tuning = tune(name, mc, seeds[0], X, y, fid)
    params = tuning["best_params"]
    per_seed, oofs, fitted = [], [], []
    for seed in seeds:
        oof, ap = oof_scores(name, params, mc, seed, X, y, fid)
        thr, oof_f1 = max_f1_threshold(y, oof)
        model = make_model(name, params, mc, seed, y, list(X.columns)).fit(X, y)
        per_seed.append({"seed": seed, "threshold": thr, "oof_f1_at_threshold": oof_f1,
                         "oof_pr_auc_pooled": float(average_precision_score(y, oof)),
                         "cv_pr_auc_folds": ap})
        oofs.append(oof)
        fitted.append(model)
    oof_mean = np.mean(oofs, axis=0)
    thr_mean, f1_mean = max_f1_threshold(y, oof_mean)
    out = {
        "model": name, "smell": smell, "target": target,
        "n_train": int(len(y)), "n_train_pos": int(y.sum()), "n_train_repos": int(rows["repository"].nunique()),
        "features": list(X.columns), "tuning": tuning, "per_seed": per_seed,
        "seed_averaged": {"threshold": thr_mean, "oof_f1_at_threshold": f1_mean,
                          "oof_pr_auc_pooled": float(average_precision_score(y, oof_mean))},
        "importance": importances(name, fitted, params, mc, seeds[0], X, y, fid),
        "train_seconds": round(time.time() - t0, 1),
    }
    if name == "xgboost":
        out["scale_pos_weight_full_train"] = scale_pos_weight(y)
    # kept in memory for test scoring and the predictions file; dropped before JSON
    out["_fitted"], out["_oof"], out["_oof_mean"], out["_rows"] = fitted, oofs, oof_mean, rows[["sample_id", "repository"]].assign(y=y)
    return out


# ---------------------------------------------------------------------------
# Test scoring (once, after everything above is frozen)
# ---------------------------------------------------------------------------

def _metrics_with_ci(y, scores, thr, groups, ev, seed) -> dict:
    pred = (scores >= thr).astype(int)
    return {"metrics": binary_metrics(y, y_score=scores, threshold=thr),
            "ci": {m: bootstrap_ci(y, pred, scores, groups, m, ev["bootstrap_n"], ev["bootstrap_alpha"], seed)
                   for m in HEADLINE}}


def _mean_std(per_seed: list[dict], key: str) -> dict:
    out = {}
    for m in HEADLINE:
        v = np.array([s[key][m] for s in per_seed], dtype=float)
        out[m] = {"mean": float(v.mean()), "std": float(v.std(ddof=0))}
    return out


def score_test(res: dict, test: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Score the frozen models on test; fills res['test']; returns the rows scored (for the gate)."""
    mc, ev = cfg["metrics_models"], cfg["evaluation"]
    target = res["target"]
    rows = test[test[target].notna()].reset_index(drop=True)
    X, y, groups = feature_frame(rows, mc), rows[target].astype(int).to_numpy(), rows["repository"].to_numpy()
    scores = np.array([m.predict_proba(X)[:, 1] for m in res["_fitted"]])
    smean = scores.mean(axis=0)
    sens_rules = cfg["labels"].get("sensitivity_severity_rules") or []
    per_seed = []
    for s, sc in zip(res["per_seed"], scores):
        entry = {"seed": s["seed"], "primary": binary_metrics(y, y_score=sc, threshold=s["threshold"])}
        for rule in sens_rules:
            col = f"{target}_{rule}"
            ok = rows[col].notna().to_numpy()
            entry[rule] = binary_metrics(rows.loc[ok, col].astype(int), y_score=sc[ok], threshold=s["threshold"])
        per_seed.append(entry)
    thr = res["seed_averaged"]["threshold"]
    test_out = {
        "n_test": int(len(y)), "n_test_pos": int(y.sum()), "n_test_repos": int(rows["repository"].nunique()),
        "primary": {"mean_std_over_seeds": _mean_std(per_seed, "primary"),
                    "seed_averaged_scores": _metrics_with_ci(y, smean, thr, groups, ev, cfg["seed"])},
        "per_seed": per_seed,
    }
    test_out["sensitivity_report_only"] = {}
    for rule in sens_rules:
        col = f"{target}_{rule}"
        ok = rows[col].notna().to_numpy()
        test_out["sensitivity_report_only"][rule] = {
            "label_column": col,
            "note": "primary-trained models and primary OOF thresholds; these labels drive no training or selection",
            "mean_std_over_seeds": _mean_std(per_seed, rule),
            "seed_averaged_scores": _metrics_with_ci(rows.loc[ok, col].astype(int).to_numpy(), smean[ok], thr,
                                                     groups[ok], ev, cfg["seed"])}
    res["test"] = test_out
    res["_test_rows"] = rows.assign(y=y)
    res["_test_scores"], res["_test_mean"] = scores, smean
    return rows


def save_predictions(res: dict, cfg: dict, ts: str) -> str:
    seeds = [s["seed"] for s in res["per_seed"]]
    oof = res["_rows"].assign(part="train_oof")
    for seed, o in zip(seeds, res["_oof"]):
        oof[f"score_seed{seed}"] = o
    oof["score_mean"] = res["_oof_mean"]
    te = res["_test_rows"][["sample_id", "repository", "y"]].assign(part="test")
    for seed, s in zip(seeds, res["_test_scores"]):
        te[f"score_seed{seed}"] = s
    te["score_mean"] = res["_test_mean"]
    d = p(cfg["paths"]["predictions"]) / "metrics"
    d.mkdir(parents=True, exist_ok=True)
    out = d / f"{res['model']}_{res['smell']}_{ts}.parquet"
    pd.concat([oof, te], ignore_index=True).to_parquet(out, index=False)
    return portable_path(out)


def rule_comparison(smell: str, best: dict, test_rows: pd.DataFrame, cfg: dict) -> dict:
    """Gate check for one smell: best model (mean over seeds) vs the rule on the same test rows."""
    ev = cfg["evaluation"]
    target, rule_name = SMELLS[smell]
    rule_pred, _ = RULES[rule_name](test_rows, cfg["rule_baseline"][rule_name])
    y, groups = test_rows[target].astype(int).to_numpy(), test_rows["repository"].to_numpy()
    rule_m = binary_metrics(y, y_pred=rule_pred)
    model = best["test"]["primary"]
    model_pred = (best["_test_mean"] >= best["seed_averaged"]["threshold"]).astype(int)
    paired = {m: paired_bootstrap(y, model_pred, best["_test_mean"], rule_pred, None, groups, m,
                                  ev["bootstrap_n"], ev["bootstrap_alpha"], cfg["seed"]) for m in ("f1", "mcc")}
    beats = {m: model["mean_std_over_seeds"][m]["mean"] > rule_m[m] for m in ("f1", "mcc")}
    return {"rule": rule_name, "rule_metrics": rule_m, "best_model": best["model"],
            "best_model_mean_over_seeds": model["mean_std_over_seeds"],
            "best_model_seed_averaged": model["seed_averaged_scores"]["metrics"],
            "paired_bootstrap_seed_averaged_minus_rule": paired,
            "beats_rule": beats, "passes": bool(all(beats.values()))}


# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.parse_args(argv)
    cfg = load_config()
    if not cfg["gates"]["join_class_verified"]:
        sys.exit("Step 3 gate not passed: set gates.join_class_verified after checking join_manual_check.csv.")
    split, folds = load_splits(cfg)
    checks = check_splits(split, folds)
    train = load_train(cfg, split)
    checks.append("training data holds training repositories only")
    split_file = p(cfg["paths"]["splits_dir"]) / "class_repo_split.json"
    mc = cfg["metrics_models"]

    # 1) everything on training repositories
    results: dict[str, dict[str, dict]] = {s: {} for s in SMELLS}
    for smell in SMELLS:
        for name in mc["order"]:
            r = train_one(name, smell, train, folds, cfg)
            results[smell][name] = r
            print(f"[train] {smell:<10} {name:<19} CV PR-AUC {r['tuning']['best_cv_pr_auc_mean']:.3f} "
                  f"best {r['tuning']['best_params']}  ({r['train_seconds']}s)")

    # 2) selection on CV PR-AUC — frozen before test is read
    selected = {s: max(results[s].values(), key=lambda r: r["tuning"]["best_cv_pr_auc_mean"])["model"]
                for s in SMELLS}
    print(f"[select] by CV PR-AUC: {selected}")

    # 3) test, once
    test = load_test(cfg, split)
    checks.append("test data loaded only after tuning, thresholds and model selection were fixed")
    ts = timestamp()
    gate, test_rows = {}, {}
    for smell in SMELLS:
        for name, r in results[smell].items():
            test_rows[smell] = score_test(r, test, cfg)
            r["predictions_file"] = save_predictions(r, cfg, ts)
            payload = {"step": 8, "arm": "metrics", "leakage_checks": checks, "threshold_policy": mc["threshold_policy"],
                       "selected_for_smell": selected[smell] == name,
                       **{k: v for k, v in r.items() if not k.startswith("_")}}
            out = write_results(cfg, f"step8_{name}_{smell}", payload, split_file)
            t = r["test"]["primary"]["mean_std_over_seeds"]
            print(f"[test]  {smell:<10} {name:<19} F1 {t['f1']['mean']:.3f}±{t['f1']['std']:.3f}  "
                  f"MCC {t['mcc']['mean']:.3f}±{t['mcc']['std']:.3f}  PR-AUC {t['pr_auc']['mean']:.3f}±"
                  f"{t['pr_auc']['std']:.3f}  -> {out.name}")
        gate[smell] = rule_comparison(smell, results[smell][selected[smell]], test_rows[smell], cfg)

    passed = all(g["passes"] for g in gate.values())
    summary = {"step": 8, "leakage_checks": checks, "selection_metric": "CV PR-AUC (tuning seed, mean over 5 folds)",
               "cv_pr_auc": {s: {n: r["tuning"]["best_cv_pr_auc_mean"] for n, r in results[s].items()} for s in SMELLS},
               "selected": selected, "gate": gate, "gate_passed": passed,
               "gate_rule": "selected model's mean test F1 AND MCC (over seeds) > rule baseline, for both smells",
               "note": "gates.metrics_beat_rules is not modified by this script; the user flips it."}
    out = write_results(cfg, "step8_summary", summary, split_file)
    for smell, g in gate.items():
        b, rm = g["best_model_mean_over_seeds"], g["rule_metrics"]
        print(f"[gate]  {smell:<10} {g['best_model']}: F1 {b['f1']['mean']:.3f} vs rule {rm['f1']:.3f}, "
              f"MCC {b['mcc']['mean']:.3f} vs rule {rm['mcc']:.3f} -> {'PASS' if g['passes'] else 'FAIL'}")
    print(f"gate {'PASSED' if passed else 'FAILED'} (config not modified) -> {out}")


if __name__ == "__main__":
    main()
