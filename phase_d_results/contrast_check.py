"""CONTRAST_ONLY — Step 12 follow-up (user request 2026-11-16): does the row-level split leak? Check 1, from saved
predictions only. Kept for research transparency. NOT part of the primary evaluation: every reported result,
selection, threshold and calibrator of the project is repository-level (user decision 2026-11-16: the paper
focuses on the repository-level evaluation).

A. Seen vs unseen: in the row-level (CONTRAST_ONLY) test set, a sample is "seen" when its repository also has
   labelled rows in the contrast training set for that smell. Metrics of the row-level models (calibrated,
   frozen thresholds from Step 12) on seen vs unseen samples; difference CI by resampling repositories
   independently within each group. Leakage predicts seen > unseen.
B. Paired, same samples: samples in BOTH the row-level and the repository-level test sets. Row-level model vs
   repository-level model on those samples (paired bootstrap over repositories), split by whether the
   row-level model saw the sample's repository. Leakage predicts row > repository on the seen part only.

No model is fitted; nothing from Steps 7-12 is modified.

    python -m phase_d_results.contrast_check --run <Step 12 predictions timestamp>
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from common import load_config, p, read_json, write_results
from evaluate import _metric_fn, paired_bootstrap
from phase_b_metrics import train_metrics as tm

ARMS = ["rule", "metrics", "embedding_truncation", "embedding_hierarchical_mean", "fusion"]


def _metrics(frame: pd.DataFrame, arm: str) -> dict:
    y, pr = frame["y"].to_numpy(int), frame["pred"].to_numpy(int)
    s = frame["score_calibrated"].to_numpy(float) if "score_calibrated" in frame else pr
    out = {m: float(_metric_fn(m)(y, pr, s)) for m in ("f1", "mcc", "pr_auc")}
    if arm == "rule":
        out.pop("pr_auc")
    return out


def two_group_bootstrap(a: pd.DataFrame, b: pd.DataFrame, metric: str, n_boot: int, alpha: float, seed: int) -> dict:
    """CI of metric(a) - metric(b), resampling repositories independently within each group."""
    fn, rng = _metric_fn(metric), np.random.default_rng(seed)

    def prep(f):
        idx = [np.flatnonzero(f["repository"].to_numpy() == r) for r in f["repository"].unique()]
        s = f["score_calibrated"].to_numpy(float) if "score_calibrated" in f else f["pred"].to_numpy(float)
        return f["y"].to_numpy(int), f["pred"].to_numpy(int), s, idx
    ya, pa, sa, ia = prep(a)
    yb, pb, sb, ib = prep(b)
    diffs = []
    for _ in range(n_boot):
        ka = np.concatenate([ia[i] for i in rng.integers(0, len(ia), len(ia))])
        kb = np.concatenate([ib[i] for i in rng.integers(0, len(ib), len(ib))])
        d = fn(ya[ka], pa[ka], sa[ka]) - fn(yb[kb], pb[kb], sb[kb])
        if not np.isnan(d):
            diffs.append(d)
    diffs = np.asarray(diffs)
    point = fn(ya, pa, sa) - fn(yb, pb, sb)
    return {"metric": metric, "diff_point": float(point), "ci_low": float(np.quantile(diffs, alpha / 2)),
            "ci_high": float(np.quantile(diffs, 1 - alpha / 2)), "n_boot_valid": int(len(diffs)),
            "ci_excludes_0": bool(np.quantile(diffs, alpha / 2) > 0 or np.quantile(diffs, 1 - alpha / 2) < 0)}


def _group_info(f: pd.DataFrame) -> dict:
    per = f.groupby("repository").size()
    return {"n": int(len(f)), "n_pos": int(f["y"].sum()), "pos_rate": float(f["y"].mean()),
            "n_repos": int(f["repository"].nunique()), "median_samples_per_repo": float(per.median())}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="Step 12 predictions timestamp (predictions/final/*_<run>.parquet)")
    args = ap.parse_args(argv)
    cfg = load_config()
    ev = cfg["evaluation"]
    nb, al, seed = ev["bootstrap_n"], ev["bootstrap_alpha"], cfg["seed"]
    contrast = read_json(p(cfg["paths"]["splits_dir"]) / "class_rowlevel_contrast.json")
    assert contrast["kind"] == "CONTRAST_ONLY"
    labels = pd.read_parquet(p(cfg["paths"]["labels_class"]))
    labels["sample_id"] = labels["sample_id"].astype(str)
    tr_ids = set(map(str, contrast["train_sample_ids"]))
    pdir = p(cfg["paths"]["predictions"]) / "final"
    out = {"step": "12-followup", "check": "contrast seen vs unseen repositories (check 1)", "run": args.run, "smells": {}}
    for smell, (target, _) in tm.SMELLS.items():
        tr = labels[labels["sample_id"].isin(tr_ids) & labels[target].notna()]
        seen_repos = set(tr["repository"])
        res = {"n_contrast_train_repos": len(seen_repos), "A_seen_vs_unseen": {}, "B_paired_same_samples": {}}
        for arm in ARMS:
            c = pd.read_parquet(pdir / "contrast" / f"{arm}_{smell}_{args.run}.parquet")
            c = c[c["part"] == "test"].copy()
            c["seen"] = c["repository"].isin(seen_repos)
            seen, unseen = c[c["seen"]], c[~c["seen"]]
            metrics = ("f1", "mcc") if arm == "rule" else ("f1", "mcc", "pr_auc")
            res["A_seen_vs_unseen"][arm] = {
                "seen": {**_group_info(seen), **_metrics(seen, arm)},
                "unseen": {**_group_info(unseen), **_metrics(unseen, arm)},
                "seen_minus_unseen": [two_group_bootstrap(seen, unseen, m, nb, al, seed) for m in metrics]}

            prim = pd.read_parquet(pdir / f"{arm}_{smell}_{args.run}.parquet")
            prim = prim[prim["part"] == "test"]
            both = c.merge(prim, on=["sample_id", "repository", "y"], suffixes=("_row", "_repo"))
            b = {"n_shared_samples": int(len(both))}
            for name, sub in (("seen_by_row_model", both[both["seen"]]), ("unseen_by_row_model", both[~both["seen"]]),
                              ("all_shared", both)):
                if len(sub) == 0 or sub["y"].sum() == 0:
                    b[name] = {"n": int(len(sub)), "skipped": "no positives"}
                    continue
                sa = sub["score_calibrated_row"].to_numpy() if arm != "rule" else None
                sb = sub["score_calibrated_repo"].to_numpy() if arm != "rule" else None
                b[name] = {**_group_info(sub), "row_minus_repository": [
                    {**(r := paired_bootstrap(sub["y"].to_numpy(int), sub["pred_row"].to_numpy(int), sa,
                                              sub["pred_repo"].to_numpy(int), sb, sub["repository"].to_numpy(), m, nb, al, seed)),
                     "ci_excludes_0": bool(r["ci_low"] > 0 or r["ci_high"] < 0)} for m in metrics]}
            res["B_paired_same_samples"][arm] = b
            a = res["A_seen_vs_unseen"][arm]
            bs = b.get("seen_by_row_model", {}).get("row_minus_repository", [])
            print(f"{smell:<10} {arm:<28} A seen-unseen: " + "  ".join(
                f"{d['metric']} {d['diff_point']:+.3f} [{d['ci_low']:+.2f},{d['ci_high']:+.2f}]" for d in a["seen_minus_unseen"])
                + "   B(seen) row-repo: " + "  ".join(
                f"{d['metric']} {d['diff_point']:+.3f} [{d['ci_low']:+.2f},{d['ci_high']:+.2f}]" for d in bs), flush=True)
        out["smells"][smell] = res
    out["notes"] = ["Groups in A differ in composition (seen repositories are the larger ones), so A mixes leakage with "
                    "repository differences; B compares two models on the same samples.",
                    "Repository-level model = Step 12 final (calibrated) model, which never saw any test repository."]
    f = write_results(cfg, "contrast_check_seen_repos", out, p(cfg["paths"]["splits_dir"]) / "class_rowlevel_contrast.json")
    print(f"-> {f}")


if __name__ == "__main__":
    main()
