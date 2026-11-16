"""Step 12 [GATE] — calibration, frozen thresholds, final class-level evaluation, row-level contrast.

Plan approved by the user 2026-11-16 (config.yaml `evaluation`); nothing is re-selected:
  * Arms (evaluation.arms): the rule baseline (Step 7, hard 0/1: not calibrated or thresholded), the locked
    metrics model (Step 8), the CV-best head per embedding strategy (Step 10), the locked fusion model (Step 11).
  * Calibration: isotonic, cross-fitted on the saved out-of-fold scores of the training repositories — fold k
    is calibrated by an isotonic fit on the other 4 folds, so no fold's labels calibrate its own scores. The
    deployed calibrator is fitted on all OOF scores and applied to the saved test scores. The final prediction
    of an arm is its seed-averaged (5-seed ensemble) score; per-seed results are reported as mean +/- std.
  * Threshold: one global threshold per smell and arm, maximizing F1 on the calibrated OOF scores, frozen.
  * Test: every final configuration scored once — F1, MCC, PR-AUC with bootstrap 95% CIs over test
    repositories; paired bootstrap for every pair in evaluation.paired_comparisons (no correction, number of
    comparisons stated); per-repository breakdown; sensitivity labels report-only; Krippendorff's alpha.
    Earlier test reads (Steps 7, 8, 10, 11) are listed in `test_use_history`.
  * Sanity: the uncalibrated seed-averaged test metrics recomputed here must equal the Step 8 / 10 / 11 results
    files, and the rule metrics Step 7's, or the run aborts.
  * Row-level contrast (eval/splits/class_rowlevel_contrast.json) — CONTRAST_ONLY, kept for research
    transparency in its own results section (`row_level_contrast`) and predictions folder (final/contrast/);
    not part of the primary evaluation, which is repository-level only (user decision 2026-11-16). The same pipeline with the
    locked models and hyperparameters refitted on the contrast training rows (row-level StratifiedKFold inside
    them for OOF scores, the nested fusion stacker and calibration), contrast test scored once. Reported next to
    the repository-level numbers with the delta.
  * Steps 7-11 outputs are only read: their sha256 is checked before and after the run.

Outputs:
  eval/results/class_level_final_<ts>.json, eval/results/class_level_final_per_repo_<ts>.csv
  <paths.predictions>/final/<arm>_<smell>_<ts>.parquet, <paths.predictions>/final/contrast/<arm>_<smell>_<ts>.parquet

    python -m phase_d_results.final_eval     (requires both gates and complete Steps 7-11)
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import StratifiedKFold

from common import load_config, p, portable_path, read_json, timestamp, write_results
from evaluate import binary_metrics, paired_bootstrap
from phase_b_metrics import train_metrics as tm
from phase_b_metrics.rule_baseline import RULES
from phase_c_embeddings import train_embeddings as te
from phase_d_results import fusion as fu

HEADLINE = ("f1", "mcc", "pr_auc")
LEARNED = ("metrics", "embedding_truncation", "embedding_hierarchical_mean", "fusion")
REPRO_TOL = 1e-9


# ---------------------------------------------------------------------------
# Arms: locked selections and their saved predictions
# ---------------------------------------------------------------------------

def _one(files: list, what: str):
    if len(files) != 1:
        sys.exit(f"expected exactly one {what}, found {[getattr(f, 'name', f) for f in files]}")
    return files[0]


def arm_specs(cfg: dict) -> dict:
    """{smell: {arm: spec}} for the learned arms: saved predictions, the earlier results file and what is
    needed to refit the arm (contrast run). Checked against the Step 8 / 10 / 11 summaries."""
    rd = p(cfg["paths"]["results_dir"])
    s8 = read_json(_one(sorted(rd.glob("step8_summary_*.json")), "Step 8 summary"))
    el, fl = cfg["embedding_models"]["locked_selection"], cfg["fusion"]["locked_selection"]
    s10, s11 = read_json(p(el["summary_file"])), read_json(p(fl["summary_file"]))
    base = fu.check_locked(cfg, s8, s10)
    m_dir, e_dir, f_dir = (p(cfg["metrics_models"]["locked_selection"]["predictions_dir"]), p(el["predictions_dir"]),
                           p(fl["predictions_dir"]))
    out = {}
    for smell in tm.SMELLS:
        m = base[smell]["metrics"]
        specs = {"metrics": {"kind": "metrics", "model": m["model"], "params": m["params"],
                             "parquet": _one(sorted(m_dir.glob(f"{m['model']}_{smell}_*.parquet")), f"{m['model']} {smell} parquet"),
                             "results": _one(sorted(rd.glob(f"step8_{m['model']}_{smell}_*.json")), f"step8 {m['model']} {smell}")}}
        for strategy in cfg["embedding_models"]["strategies"]:
            head = s10["best_head_per_strategy"][smell][strategy]
            head = head[len(strategy) + 1:] if head.startswith(strategy + "_") else head
            rf = _one(sorted(rd.glob(f"step10_{strategy}_{head}_{smell}_*.json")), f"step10 {strategy} {head} {smell}")
            specs[f"embedding_{strategy}"] = {
                "kind": "embedding", "strategy": strategy, "head": head,
                "params": read_json(rf)["tuning"]["best_params"],
                "parquet": _one(sorted(e_dir.glob(f"{strategy}_{head}_{smell}_{el['run_id']}.parquet")), f"{strategy} {head} {smell} parquet"),
                "results": rf}
        fm = fl[smell]["model"]
        if s11["selected"][smell] != fm:
            sys.exit(f"{smell}: fusion.locked_selection {fm!r} != Step 11 summary {s11['selected'][smell]!r}")
        specs["fusion"] = {"kind": "fusion", "model": fm, "base": base[smell],
                           "parquet": _one(sorted(f_dir.glob(f"{fm}_{smell}_{fl['run_id']}.parquet")), f"{fm} {smell} parquet"),
                           "results": _one(sorted(rd.glob(f"step11_{fm}_{smell}_*.json")), f"step11 {fm} {smell}")}
        out[smell] = specs
    return out


def protected_hashes(cfg: dict) -> dict:
    rd, pr = p(cfg["paths"]["results_dir"]), p(cfg["paths"]["predictions"])
    files = []
    for pat in ("step7_*.json", "step8_*.json", "step9_*.json", "step10_*.json", "step11_*.json"):
        files += sorted(rd.glob(pat))
    for sub in ("metrics", "embeddings", "fusion"):
        files += sorted((pr / sub).glob("*.parquet"))
    for d in (cfg["embedding_models"]["models_dir"], cfg["fusion"]["models_dir"]):
        files += sorted(p(d).rglob("*.joblib"))
    out = {}
    for f in files:
        h = hashlib.sha256()
        with open(f, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        out[portable_path(f)] = h.hexdigest()
    return out


def aligned_scores(frame: pd.DataFrame, rows: pd.DataFrame, target: str, seeds: list[int]) -> np.ndarray:
    """(n_seeds, n_rows) saved scores in the order of `rows` (fusion.align_base: aborts on any mismatch)."""
    return fu.align_base(rows, target, frame, seeds)


# ---------------------------------------------------------------------------
# Calibration, thresholds, metrics
# ---------------------------------------------------------------------------

def _iso() -> IsotonicRegression:
    return IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")


def crossfit_isotonic(oof: np.ndarray, y: np.ndarray, fid: np.ndarray) -> np.ndarray:
    """Calibrated OOF scores: fold k mapped by an isotonic fit on the other folds only."""
    cal = np.full(len(y), np.nan)
    for k in np.unique(fid):
        tr, va = fid != k, fid == k
        cal[va] = _iso().fit(oof[tr], y[tr]).predict(oof[va])
    assert not np.isnan(cal).any()
    return cal


def calibrate(oof: np.ndarray, y: np.ndarray, fid: np.ndarray, test: np.ndarray) -> dict:
    """Cross-fitted calibrated OOF -> frozen max-F1 threshold; deployed calibrator (all OOF) -> test."""
    cal_oof = crossfit_isotonic(oof, y, fid)
    thr, f1 = tm.max_f1_threshold(y, cal_oof)
    iso = _iso().fit(oof, y)
    return {"cal_oof": cal_oof, "cal_test": iso.predict(test), "threshold": thr, "oof_f1_at_threshold": f1}


def ece(y: np.ndarray, prob: np.ndarray, bins: int) -> float:
    """Expected calibration error, equal-width bins."""
    idx = np.minimum((prob * bins).astype(int), bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            total += m.sum() / len(y) * abs(y[m].mean() - prob[m].mean())
    return float(total)


def _sensitivity(rows: pd.DataFrame, target: str, cfg: dict, scores=None, thr=None, pred=None) -> dict:
    ev, out = cfg["evaluation"], {}
    for rule in cfg["labels"].get("sensitivity_severity_rules") or []:
        col = f"{target}_{rule}"
        ok = rows[col].notna().to_numpy()
        yy, g = rows.loc[ok, col].astype(int).to_numpy(), rows["repository"].to_numpy()[ok]
        if pred is not None:
            out[rule] = {"label_column": col, "metrics": binary_metrics(yy, y_pred=pred[ok])}
        else:
            out[rule] = {"label_column": col, **tm._metrics_with_ci(yy, scores[ok], thr, g, ev, cfg["seed"])}
        out[rule]["note"] = "primary-trained models and primary thresholds; report-only"
    return out


def evaluate_learned(oof: np.ndarray, test: np.ndarray, y_tr: np.ndarray, fid: np.ndarray,
                     rows_te: pd.DataFrame, target: str, cfg: dict, seeds: list[int]) -> dict:
    """oof: (n_seeds, n_train) and test: (n_seeds, n_test) raw scores. Returns the arm's final evaluation and,
    under '_', the arrays for the paired tests and the predictions file."""
    ev = cfg["evaluation"]
    y_te, groups = rows_te[target].astype(int).to_numpy(), rows_te["repository"].to_numpy()
    oof_mean, test_mean = oof.mean(axis=0), test.mean(axis=0)
    final = calibrate(oof_mean, y_tr, fid, test_mean)
    pred = (final["cal_test"] >= final["threshold"]).astype(int)
    per_seed = []
    for s, o, t in zip(seeds, oof, test):
        c = calibrate(o, y_tr, fid, t)
        per_seed.append({"seed": s, "threshold": c["threshold"],
                         "primary": binary_metrics(y_te, y_score=c["cal_test"], threshold=c["threshold"])})
    raw_thr, _ = tm.max_f1_threshold(y_tr, oof_mean)
    bins = ev["ece_bins"]
    out = {
        "n_train": int(len(y_tr)), "n_test": int(len(y_te)), "n_test_pos": int(y_te.sum()),
        "n_test_repos": int(len(np.unique(groups))),
        "threshold": {"value": final["threshold"], "oof_f1_at_threshold": final["oof_f1_at_threshold"],
                      "how": "max F1 on cross-fitted isotonic-calibrated OOF scores of the seed-averaged score "
                             "(training repositories), frozen before test"},
        "test": tm._metrics_with_ci(y_te, final["cal_test"], final["threshold"], groups, ev, cfg["seed"]),
        "test_mean_std_over_seeds": tm._mean_std(per_seed, "primary"),
        "per_seed": per_seed,
        "uncalibrated_reference": binary_metrics(y_te, y_score=test_mean, threshold=raw_thr),
        "calibration": {
            "method": f"isotonic, {ev['calibration_mode']}",
            "test_brier": {"raw": float(brier_score_loss(y_te, test_mean)),
                           "calibrated": float(brier_score_loss(y_te, final["cal_test"]))},
            "test_ece": {"raw": ece(y_te, test_mean, bins), "calibrated": ece(y_te, final["cal_test"], bins)},
            "oof_brier": {"raw": float(brier_score_loss(y_tr, oof_mean)),
                          "calibrated_crossfit": float(brier_score_loss(y_tr, final["cal_oof"]))},
            "oof_ece": {"raw": ece(y_tr, oof_mean, bins), "calibrated_crossfit": ece(y_tr, final["cal_oof"], bins)},
            "ece_bins": bins},
        "sensitivity_report_only": _sensitivity(rows_te, target, cfg, scores=final["cal_test"], thr=final["threshold"]),
    }
    out["_"] = {"pred": pred, "score": final["cal_test"], "raw_test": test_mean, "raw_oof": oof_mean,
                "cal_oof": final["cal_oof"]}
    return out


def evaluate_rule(rows_te: pd.DataFrame, smell: str, cfg: dict) -> dict:
    target, rule = tm.SMELLS[smell]
    pred, missing = RULES[rule](rows_te, cfg["rule_baseline"][rule])
    y, groups, ev = rows_te[target].astype(int).to_numpy(), rows_te["repository"].to_numpy(), cfg["evaluation"]
    from evaluate import bootstrap_ci
    out = {"rule": rule, "n_test": int(len(y)),
           "threshold": {"value": None, "how": "fixed Lanza & Marinescu rule (hard 0/1): not calibrated or thresholded"},
           "test": {"metrics": binary_metrics(y, y_pred=pred),
                    "ci": {m: bootstrap_ci(y, pred, None, groups, m, ev["bootstrap_n"], ev["bootstrap_alpha"], cfg["seed"])
                           for m in HEADLINE}},
           "pr_auc_note": "hard 0/1 rule: PR-AUC is a single operating point, not a curve",
           "n_missing_rule_inputs_predicted_negative": int(missing.sum()),
           "sensitivity_report_only": _sensitivity(rows_te, target, cfg, pred=pred)}
    out["_"] = {"pred": pred, "score": None}
    return out


def paired_tests(arms: dict, y: np.ndarray, groups: np.ndarray, cfg: dict) -> list[dict]:
    ev, out = cfg["evaluation"], []
    for a, b in ev["paired_comparisons"]:
        metrics = ("f1", "mcc") if "rule" in (a, b) else HEADLINE
        for m in metrics:
            r = paired_bootstrap(y, arms[a]["_"]["pred"], arms[a]["_"]["score"], arms[b]["_"]["pred"],
                                 arms[b]["_"]["score"], groups, m, ev["bootstrap_n"], ev["bootstrap_alpha"], cfg["seed"])
            out.append({"a": a, "b": b, **r, "ci_excludes_0": bool(r["ci_low"] > 0 or r["ci_high"] < 0)})
    return out


def per_repo(arms: dict, rows: pd.DataFrame, target: str, smell: str) -> pd.DataFrame:
    y = rows[target].astype(int).to_numpy()
    df = pd.DataFrame({"smell": smell, "repository": rows["repository"].to_numpy(), "y": y})
    for a, r in arms.items():
        pr = r["_"]["pred"]
        df[f"{a}_tp"], df[f"{a}_fp"] = (pr == 1) & (y == 1), (pr == 1) & (y == 0)
        df[f"{a}_fn"], df[f"{a}_tn"] = (pr == 0) & (y == 1), (pr == 0) & (y == 0)
    g = df.groupby(["smell", "repository"], sort=True)
    out = g.agg(n=("y", "size"), n_pos=("y", "sum"),
                **{c: (c, "sum") for c in df.columns if c.endswith(("_tp", "_fp", "_fn", "_tn"))}).reset_index()
    for a in arms:
        tp, fp, fn = out[f"{a}_tp"], out[f"{a}_fp"], out[f"{a}_fn"]
        denom = 2 * tp + fp + fn
        out[f"{a}_f1"] = np.where(denom > 0, 2 * tp / denom.where(denom > 0, 1), np.nan)
    return out


def per_repo_summary(tab: pd.DataFrame, arms: list[str]) -> dict:
    out = {"n_repos": int(len(tab)), "n_repos_with_positives": int((tab["n_pos"] > 0).sum()),
           "samples_per_repo": {"median": float(tab["n"].median()), "max": int(tab["n"].max())}, "arms": {}}
    for a in arms:
        err = tab[f"{a}_fp"] + tab[f"{a}_fn"]
        top = tab.assign(err=err).sort_values(["err", "n"], ascending=False).head(5)
        rho = spearmanr(tab["n"], err / tab["n"]).statistic if len(tab) > 2 else float("nan")
        out["arms"][a] = {"errors_total": int(err.sum()),
                          "share_of_errors_in_top5_repos": float(top["err"].sum() / err.sum()) if err.sum() else 0.0,
                          "top5_repos_by_errors": [{"repository": str(t.repository), "n": int(t.n), "n_pos": int(t.n_pos),
                                                    "errors": int(t.err)} for t in top.itertuples()],
                          "repos_with_f1_defined": int(tab[f"{a}_f1"].notna().sum()),
                          "spearman_repo_size_vs_error_rate": float(rho)}
    return out


def check_reproduces(arm: str, res: dict, earlier: dict) -> dict:
    """Uncalibrated seed-averaged test metrics here == the earlier step's results file."""
    ref = earlier["test"]["primary"]["seed_averaged_scores"]["metrics"] if arm != "rule" else earlier["metrics"]
    mine = res["uncalibrated_reference"] if arm != "rule" else res["test"]["metrics"]
    diff = {m: abs(mine[m] - ref[m]) for m in HEADLINE}
    if max(diff.values()) > REPRO_TOL:
        sys.exit(f"{arm}: recomputed test metrics differ from the earlier results file ({diff}) — aborting.")
    return {"max_abs_diff": max(diff.values())}


# ---------------------------------------------------------------------------
# Row-level contrast
# ---------------------------------------------------------------------------

def contrast_frames(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    c = read_json(p(cfg["paths"]["splits_dir"]) / "class_rowlevel_contrast.json")
    if c is None or c.get("kind") != "CONTRAST_ONLY":
        sys.exit("eval/splits/class_rowlevel_contrast.json missing or not CONTRAST_ONLY")
    tr_ids, te_ids = set(map(str, c["train_sample_ids"])), set(map(str, c["test_sample_ids"]))
    assert not tr_ids & te_ids, "sample on both sides of the contrast split"
    labels = pd.read_parquet(p(cfg["paths"]["labels_class"]))
    ds = pd.read_parquet(p(cfg["paths"]["class_dataset"]))
    allr = labels.merge(ds.drop(columns=["repository", "repo_key"], errors="ignore"), on="sample_id", how="inner")
    ids = allr["sample_id"].astype(str)
    return (allr[ids.isin(tr_ids)].reset_index(drop=True), allr[ids.isin(te_ids)].reset_index(drop=True),
            {"n_train": len(tr_ids), "n_test": len(te_ids), "n_repos_on_both_sides": c["n_repos_on_both_sides"]})


def row_folds(y: np.ndarray, cfg: dict) -> np.ndarray:
    cc = cfg["evaluation"]["contrast"]
    if cc["cv_strategy"] != "StratifiedKFold" or cc["refit"] != "locked_params":
        raise ValueError("only contrast cv_strategy StratifiedKFold with refit locked_params is implemented")
    fid = np.full(len(y), -1)
    skf = StratifiedKFold(cc["cv_folds"], shuffle=True, random_state=cc["fold_seed"])
    for k, (_, va) in enumerate(skf.split(np.zeros(len(y)), y)):
        fid[va] = k
    return fid


def refit_scores(spec: dict, cfg: dict, seed: int, Xtr: dict, y: np.ndarray, fid: np.ndarray, Xte: dict
                 ) -> tuple[np.ndarray, np.ndarray]:
    """OOF (by fid) and test scores of one arm refitted with its locked parameters."""
    if spec["kind"] == "fusion":
        if spec["model"] != "late_stacker":
            raise ValueError("contrast refit is implemented for the late stacker only")
        nl = fu.nested_late(spec["base"], cfg, seed, Xtr, y, fid)
        st = fu.make_stacker(cfg, seed).fit(fu.stack_features(nl["base_oof"]["metrics"], nl["base_oof"]["embedding"]), y)
        base_te = {a: fu.fit_base(a, spec["base"][a], cfg, seed, Xtr[a], y).predict_proba(Xte[a])[:, 1] for a in fu.ARMS}
        return nl["stack_oof"], st.predict_proba(fu.stack_features(base_te["metrics"], base_te["embedding"]))[:, 1]
    arm = "metrics" if spec["kind"] == "metrics" else "embedding"
    key = "metrics" if arm == "metrics" else spec["strategy"]
    X, Xt = Xtr[key], Xte[key]
    oof = np.full(len(y), np.nan)
    for k in np.unique(fid):
        tr, va = fid != k, fid == k
        oof[va] = fu.fit_base(arm, spec, cfg, seed, X[tr], y[tr]).predict_proba(X[va])[:, 1]
    return oof, fu.fit_base(arm, spec, cfg, seed, X, y).predict_proba(Xt)[:, 1]


def run_contrast(cfg: dict, specs: dict, vectors: dict, primary: dict) -> dict:
    tr_all, te_all, info = contrast_frames(cfg)
    seeds = cfg["metrics_models"]["seeds"]
    out = {"split": "eval/splits/class_rowlevel_contrast.json", **info, "smells": {}}
    preds = {}
    for smell, (target, _) in tm.SMELLS.items():
        t0 = time.time()
        tr = tr_all[tr_all[target].notna()].reset_index(drop=True)
        tst = te_all[te_all[target].notna()].reset_index(drop=True)
        y = tr[target].astype(int).to_numpy()
        fid = row_folds(y, cfg)
        Xtr = {"metrics": tm.feature_frame(tr, cfg["metrics_models"]),
               **{s: te.attach(tr, vectors[s][0], vectors[s][1]) for s in vectors}}
        Xte = {"metrics": tm.feature_frame(tst, cfg["metrics_models"]),
               **{s: te.attach(tst, vectors[s][0], vectors[s][1]) for s in vectors}}
        arms = {"rule": evaluate_rule(tst, smell, cfg)}
        for arm in LEARNED:
            spec = specs[smell][arm]
            if spec["kind"] == "fusion":
                Xf = {"metrics": Xtr["metrics"], "embedding": Xtr[spec["base"]["embedding"]["strategy"]]}
                Xft = {"metrics": Xte["metrics"], "embedding": Xte[spec["base"]["embedding"]["strategy"]]}
                pairs = [refit_scores(spec, cfg, s, Xf, y, fid, Xft) for s in seeds]
            else:
                pairs = [refit_scores(spec, cfg, s, Xtr, y, fid, Xte) for s in seeds]
            oof, test = np.array([a for a, _ in pairs]), np.array([b for _, b in pairs])
            arms[arm] = evaluate_learned(oof, test, y, fid, tst, target, cfg, seeds)
            print(f"[contrast] {smell:<10} {arm:<28} F1 {arms[arm]['test']['metrics']['f1']:.3f}  "
                  f"MCC {arms[arm]['test']['metrics']['mcc']:.3f}  PR-AUC {arms[arm]['test']['metrics']['pr_auc']:.3f}",
                  flush=True)
        delta = {a: {m: arms[a]["test"]["metrics"][m] - primary[smell][a]["test"]["metrics"][m]
                     for m in (("f1", "mcc") if a == "rule" else HEADLINE)} for a in arms}
        out["smells"][smell] = {
            "n_train": int(len(y)), "n_test": int(len(tst)), "n_test_pos": int(tst[target].astype(int).sum()),
            "arms": {a: {k: v for k, v in r.items() if k != "_"} for a, r in arms.items()},
            "delta_row_level_minus_repository_level": delta,
            "note": "different test sets: no paired CI; each side has its own CI (contrast CI also resamples repositories)",
            "seconds": round(time.time() - t0, 1)}
        preds[smell] = (arms, tr, tst, y)
    out["_preds"] = preds
    return out


# ---------------------------------------------------------------------------

def save_predictions(arms: dict, rows_tr: pd.DataFrame | None, rows_te: pd.DataFrame, target: str, smell: str,
                     d, ts: str) -> dict:
    d.mkdir(parents=True, exist_ok=True)
    files = {}
    for a, r in arms.items():
        te_part = rows_te[["sample_id", "repository"]].assign(y=rows_te[target].astype(int).to_numpy(), part="test",
                                                              pred=r["_"]["pred"])
        if r["_"]["score"] is not None:
            te_part = te_part.assign(score_raw=r["_"]["raw_test"], score_calibrated=r["_"]["score"])
            if rows_tr is not None:
                oof = rows_tr[["sample_id", "repository", "y"]].assign(part="train_oof", score_raw=r["_"]["raw_oof"],
                                                                       score_calibrated=r["_"]["cal_oof"])
                te_part = pd.concat([oof, te_part], ignore_index=True)
        f = d / f"{a}_{smell}_{ts}.parquet"
        te_part.to_parquet(f, index=False)
        files[a] = portable_path(f)
    return files


def test_use_history(cfg: dict) -> list[dict]:
    rd = p(cfg["paths"]["results_dir"])
    rows = [("step 7", "rule baseline scored on test", "step7_*.json"),
            ("step 8", "metrics models scored on test after CV selection", "step8_*.json"),
            ("step 10", "embedding heads scored on test after CV selection", "step10_*.json"),
            ("step 11", "fusion candidates scored on test after CV selection", "step11_*.json")]
    return [{"step": s, "what": w, "files": [f.name for f in sorted(rd.glob(g))]} for s, w, g in rows] + [
        {"step": "step 12", "what": "each calibrated final configuration scored on test once (this file); selections "
                                    "and thresholds were fixed on training repositories before test was read; none "
                                    "was changed after any earlier test read"}]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.parse_args(argv)
    cfg = load_config()
    if not (cfg["gates"]["join_class_verified"] and cfg["gates"]["metrics_beat_rules"]):
        sys.exit("Steps 3 and 8 gates must both have passed before Step 12.")
    ev = cfg["evaluation"]
    if ev["calibration_method"] != "isotonic" or ev["calibration_mode"] != "crossfit_oof" \
            or ev["calibration_scores"] != "seed_averaged":
        sys.exit("only isotonic / crossfit_oof / seed_averaged calibration is implemented")
    if list(ev["arms"]) != ["rule", *LEARNED]:
        sys.exit(f"evaluation.arms must be {['rule', *LEARNED]}")
    t0 = time.time()
    split, folds = tm.load_splits(cfg)
    checks = tm.check_splits(split, folds)
    split_file = p(cfg["paths"]["splits_dir"]) / "class_repo_split.json"
    specs = arm_specs(cfg)
    checks.append("locked selections in config equal the Step 8, 10 and 11 summaries")
    hashes_before = protected_hashes(cfg)
    seeds = cfg["metrics_models"]["seeds"]

    # 1) calibration + thresholds from training-repo OOF scores only (saved predictions, train part)
    train_parts = {s: {a: fu.load_base_part({a: sp["parquet"]}, "train_oof")[a] for a, sp in specs[s].items()}
                   for s in tm.SMELLS}
    # 2) test: rows loaded once; saved test scores per arm
    test = tm.load_test(cfg, split)
    checks.append("test rows and saved test scores read after every model, calibrator input and selection was fixed")
    ts = timestamp()
    primary, paired, tables, pred_files = {}, {}, [], {}
    for smell, (target, _) in tm.SMELLS.items():
        rows_te = test[test[target].notna()].reset_index(drop=True)
        ref_tr = train_parts[smell]["metrics"].reset_index(drop=True)
        for a, f in train_parts[smell].items():         # every arm's OOF rows identical (ids, labels, repos)
            fu.align_base(ref_tr.rename(columns={"y": target}), target, f, seeds)
        y_tr = ref_tr["y"].to_numpy(int)
        fid = tm.fold_ids(ref_tr["repository"], folds)
        arms = {"rule": evaluate_rule(rows_te, smell, cfg)}
        s7 = read_json(_one(sorted(p(cfg["paths"]["results_dir"]).glob(f"step7_rule_{tm.SMELLS[smell][1]}_*.json")), "step7 file"))
        arms["rule"]["reproduces_earlier_step"] = check_reproduces("rule", arms["rule"], s7)
        test_parts = {a: fu.load_base_part({a: sp["parquet"]}, "test")[a] for a, sp in specs[smell].items()}
        for a in LEARNED:
            tr_rows = ref_tr.rename(columns={"y": target})
            oof = aligned_scores(train_parts[smell][a], tr_rows, target, seeds)
            tst = aligned_scores(test_parts[a], rows_te, target, seeds)
            r = evaluate_learned(oof, tst, y_tr, fid, rows_te, target, cfg, seeds)
            r["reproduces_earlier_step"] = check_reproduces(a, r, read_json(specs[smell][a]["results"]))
            r["source"] = {k: (portable_path(v) if hasattr(v, "exists") else v) for k, v in specs[smell][a].items()
                           if k not in ("base",)}
            arms[a] = r
            m = r["test"]["metrics"]
            print(f"[final] {smell:<10} {a:<28} thr {r['threshold']['value']:.3f}  F1 {m['f1']:.3f}  MCC {m['mcc']:.3f}  "
                  f"PR-AUC {m['pr_auc']:.3f}  (uncalibrated F1 {r['uncalibrated_reference']['f1']:.3f})", flush=True)
        primary[smell] = arms
        y_te, groups = rows_te[target].astype(int).to_numpy(), rows_te["repository"].to_numpy()
        paired[smell] = paired_tests(arms, y_te, groups, cfg)
        tables.append(per_repo(arms, rows_te, target, smell))
        pred_files[smell] = save_predictions(arms, ref_tr, rows_te, target, smell, p(cfg["paths"]["predictions"]) / "final", ts)
    checks += ["calibration: isotonic cross-fitted over the 5 CV folds (fold k never calibrates itself); thresholds "
               "from calibrated OOF scores only", "uncalibrated test metrics reproduce the Step 7/8/10/11 results files"]
    n_comp = sum(len(v) for v in paired.values())
    print(f"[paired] {n_comp} comparisons ({', '.join(f'{s}: {len(v)}' for s, v in paired.items())})", flush=True)

    # 3) row-level contrast (refit with locked parameters)
    vectors = {s: te.load_vectors(cfg, s) for s in cfg["embedding_models"]["strategies"]}
    contrast = run_contrast(cfg, specs, vectors, primary)
    cpreds = contrast.pop("_preds")
    for smell, (arms, tr, tst, y) in cpreds.items():
        target = tm.SMELLS[smell][0]
        contrast["smells"][smell]["predictions_files"] = save_predictions(
            arms, tr[["sample_id", "repository"]].assign(y=y), tst, target, smell,
            p(cfg["paths"]["predictions"]) / "final" / "contrast", ts)
    checks.append("contrast: contrast train/test sample ids disjoint; row-level folds used only inside the contrast run")

    hashes_after = protected_hashes(cfg)
    unchanged = hashes_after == hashes_before
    assert unchanged, "a Step 7-11 output changed during the Step 12 run"
    per_repo_tab = pd.concat(tables, ignore_index=True)
    csv = None
    if ev["per_repo_csv"]:
        csv = p(cfg["paths"]["results_dir"]) / f"class_level_final_per_repo_{ts}.csv"
        per_repo_tab.to_csv(csv, index=False)

    agreement_file = _one(sorted(p(cfg["paths"]["results_dir"]).glob("agreement_*.json")), "agreement file")
    ag = read_json(agreement_file)["agreement"]
    names = {"blob": "blob", "data_class": "data class"}
    payload = {
        "step": 12, "level": "class", "leakage_checks": checks,
        "headline": {s: {a: {**{m: r["test"]["metrics"][m] for m in HEADLINE},
                             **{f"{m}_ci": [r["test"]["ci"][m]["ci_low"], r["test"]["ci"][m]["ci_high"]] for m in HEADLINE},
                             "threshold": r["threshold"]["value"]} for a, r in arms.items()}
                     for s, arms in primary.items()},
        "agreement_krippendorff_alpha": {
            s: {"alpha_binary": ag["smells"][names[s]]["alpha_binary"], "alpha_ordinal": ag["smells"][names[s]]["alpha_ordinal"],
                "alpha_binary_major_critical_only": ag["sensitivity"]["major_critical_only"]["smells"][names[s]]["alpha_binary"],
                "n_multi_reviewed": ag["smells"][names[s]]["n_multi_reviewed"], "file": agreement_file.name} for s in tm.SMELLS},
        "arms": {s: {a: {k: v for k, v in r.items() if k != "_"} for a, r in arms.items()} for s, arms in primary.items()},
        "paired_bootstrap": paired,
        "multiple_comparisons": {"n_comparisons": n_comp, "correction": ev["multiple_comparison_correction"],
                                 "note": "every comparison reported; read CIs that exclude 0 with this number in mind"},
        "per_repository": {"summary": {s: per_repo_summary(per_repo_tab[per_repo_tab.smell == s], list(primary[s]))
                                       for s in primary},
                           "csv": portable_path(csv) if csv else None},
        "row_level_contrast": contrast,
        "test_use_history": test_use_history(cfg),
        "predictions_files": pred_files,
        "ood": {"status": "deferred (D-9, user 2026-11-16)", "dataset": ev.get("ood_dataset")},
        "protected_outputs_unchanged": {"n_files": len(hashes_before), "unchanged": unchanged, "sha256": hashes_before},
        "seconds": round(time.time() - t0, 1),
        "note": "GATE: the user reviews this file before method level or anything else proceeds.",
    }
    out = write_results(cfg, "class_level_final", payload, split_file)
    for s, arms in primary.items():
        for a, r in arms.items():
            m, c = r["test"]["metrics"], r["test"]["ci"]
            print(f"[headline] {s:<10} {a:<28} F1 {m['f1']:.3f} [{c['f1']['ci_low']:.2f},{c['f1']['ci_high']:.2f}]  "
                  f"MCC {m['mcc']:.3f} [{c['mcc']['ci_low']:.2f},{c['mcc']['ci_high']:.2f}]  PR-AUC {m['pr_auc']:.3f}", flush=True)
    print(f"done -> {out}  (Step 7-11 outputs unchanged: {len(hashes_before)} files; {round(time.time() - t0)}s)", flush=True)


if __name__ == "__main__":
    main()
