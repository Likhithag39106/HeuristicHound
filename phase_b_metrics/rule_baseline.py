"""Step 7 — Lanza & Marinescu detection strategies (God Class, Data Class). No learning.

Scored once on the PRIMARY test split. Hard 0/1 predictions, so PR-AUC is a single
operating point (flagged in the results). Constants come from config.yaml (D-4).

  God Class:  ATFD > atfd_gt AND WMC >= wmc_gte AND TCC < tcc_lt
  Data Class: WOC < woc_lt AND ((NOPA+NOAM > nopa_noam_gt_low AND WMC < wmc_lt_low)
                             OR (NOPA+NOAM > nopa_noam_gt_high AND WMC < wmc_lt_high))
A sample whose rule inputs are missing is predicted negative and counted in the results.

    python -m phase_b_metrics.rule_baseline     (requires gates.join_class_verified)
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from common import load_config, p, read_json, write_results
from evaluate import binary_metrics, bootstrap_ci

SMELLS = {"god_class": "y_blob", "data_class": "y_data_class"}


def god_class(df: pd.DataFrame, c: dict) -> tuple[np.ndarray, np.ndarray]:
    inputs = df[["atfd", "ck_wmc", "ck_tcc"]].apply(pd.to_numeric, errors="coerce")
    missing = inputs.isna().any(axis=1).to_numpy()
    pred = ((inputs["atfd"] > c["atfd_gt"]) & (inputs["ck_wmc"] >= c["wmc_gte"])
            & (inputs["ck_tcc"] < c["tcc_lt"])).to_numpy()
    return pred.astype(int), missing


def data_class(df: pd.DataFrame, c: dict) -> tuple[np.ndarray, np.ndarray]:
    inputs = df[["woc", "nopa", "noam", "ck_wmc"]].apply(pd.to_numeric, errors="coerce")
    missing = inputs.isna().any(axis=1).to_numpy()
    exposed = inputs["nopa"] + inputs["noam"]
    pred = ((inputs["woc"] < c["woc_lt"])
            & (((exposed > c["nopa_noam_gt_low"]) & (inputs["ck_wmc"] < c["wmc_lt_low"]))
               | ((exposed > c["nopa_noam_gt_high"]) & (inputs["ck_wmc"] < c["wmc_lt_high"])))).to_numpy()
    return pred.astype(int), missing


RULES = {"god_class": god_class, "data_class": data_class}


def test_frame(cfg: dict) -> tuple[pd.DataFrame, dict]:
    split = read_json(p(cfg["paths"]["splits_dir"]) / "class_repo_split.json")
    if split is None:
        sys.exit("eval/splits/class_repo_split.json missing — run Step 6 first.")
    labels = pd.read_parquet(p(cfg["paths"]["labels_class"]))
    ds = pd.read_parquet(p(cfg["paths"]["class_dataset"]))
    test = labels[labels["repository"].isin(split["test_repos"])]
    merged = test.merge(ds.drop(columns=["repository", "repo_key"], errors="ignore"), on="sample_id", how="left",
                        indicator=True)
    info = {"n_test_samples": int(len(test)),
            "n_test_unmatched_in_join": int((merged["_merge"] == "left_only").sum())}
    return merged[merged["_merge"] == "both"].drop(columns="_merge"), info


def evaluate_rule(name: str, df: pd.DataFrame, cfg: dict) -> dict:
    target = SMELLS[name]
    consts = cfg["rule_baseline"][name]
    nulls = [k for k, v in consts.items() if v is None]
    if nulls:
        return {"skipped": True, "reason": f"D-4 open: constants {nulls} are null in config.yaml"}
    rated = df[df[target].notna()]
    pred, missing = RULES[name](rated, consts)
    y = rated[target].astype(int).to_numpy()
    groups = rated["repository"].to_numpy()
    ev = cfg["evaluation"]
    return {
        "skipped": False, "constants": consts,
        "metrics": binary_metrics(y, y_pred=pred),
        "ci": {m: bootstrap_ci(y, pred, None, groups, m, ev["bootstrap_n"], ev["bootstrap_alpha"], cfg["seed"])
               for m in ("f1", "mcc")},
        "pr_auc_note": "hard 0/1 rule: PR-AUC is a single operating point, not a curve",
        "n_scored": int(len(rated)), "n_missing_rule_inputs_predicted_negative": int(missing.sum()),
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.parse_args(argv)
    cfg = load_config()
    if not cfg["gates"]["join_class_verified"]:
        sys.exit("Step 3 gate not passed: set gates.join_class_verified after checking join_manual_check.csv.")
    df, info = test_frame(cfg)
    split_file = p(cfg["paths"]["splits_dir"]) / "class_repo_split.json"
    for name in RULES:
        res = evaluate_rule(name, df, cfg)
        if res["skipped"]:
            print(f"{name}: SKIPPED — {res['reason']}")
            continue
        out = write_results(cfg, f"step7_rule_{name}", {"step": 7, "arm": "rule_baseline", "smell": name,
                                                       **info, **res}, split_file)
        m = res["metrics"]
        print(f"{name}: F1={m['f1']:.3f} [{res['ci']['f1']['ci_low']:.3f}, {res['ci']['f1']['ci_high']:.3f}]  "
              f"MCC={m['mcc']:.3f} [{res['ci']['mcc']['ci_low']:.3f}, {res['ci']['mcc']['ci_high']:.3f}]  "
              f"PR-AUC(point)={m['pr_auc']:.3f}  -> {out}")


if __name__ == "__main__":
    main()
