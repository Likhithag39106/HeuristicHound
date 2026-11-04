"""Step 5 — class-level labels and inter-rater agreement.

Outputs:
  data/processed/labels_class.parquet   one row per class sample_id:
      sev_<smell>, y_<smell> (primary), n_reviews_<smell>,
      y_<smell>_annot_filtered (ratings from reviewers who say they can't recognize the smell dropped),
      high_conf_<smell> (multi-reviewed AND all reviewers agree on the binary label);
      the same y / annot_filtered / high_conf columns again for each rule in
      labels.sensitivity_severity_rules, suffixed _<rule> (e.g. y_blob_major_critical_only)
  eval/results/agreement_<timestamp>.json   Krippendorff's alpha per smell, binary + ordinal,
      for the primary rule (top level) and each sensitivity rule (under "sensitivity")

Models train on the primary labels (labels.minor_severity_rule) only; variants and sensitivity
rules are for analysis.

    python -m phase_a_data.labels           (requires gates.join_class_verified)
"""
from __future__ import annotations

import argparse
import sys

import krippendorff
import numpy as np
import pandas as pd

from common import SEVERITY_ORDER, load_config, load_samples, p, write_results

SMELLS = {"blob": "blob", "data class": "data_class"}
# Survey answer text for "Which of the following code smells do you believe you can recognize?"
SURVEY_NAMES = {"blob": "Blob / God Class", "data class": "Data Class"}
SURVEY_RECOGNIZE_PREFIX = "Which of the following code smells do you believe you can recognize"


def binarize(sev: pd.Series, rule: str) -> pd.Series:
    if rule == "major_critical_only":
        return (sev >= SEVERITY_ORDER["major"]).astype(int)
    if rule == "minor_plus":
        return (sev >= SEVERITY_ORDER["minor"]).astype(int)
    raise ValueError(f"labels.minor_severity_rule must be major_critical_only or minor_plus, got {rule!r}")


def class_ratings(cfg: dict) -> pd.DataFrame:
    """One rating per (sample, smell, reviewer); the 2 duplicated reviewer ratings keep the max."""
    df = load_samples(cfg)
    df = df[df["type"] == "class"].copy()
    df["sev"] = df["severity"].map(SEVERITY_ORDER)
    if df["sev"].isna().any():
        raise ValueError(f"unknown severities: {df.loc[df['sev'].isna(), 'severity'].unique()}")
    return (df.groupby(["sample_id", "smell", "reviewer_id", "repository", "repo_key"], as_index=False)["sev"].max())


def recognizers(cfg: dict) -> dict[str, set[str]]:
    """smell -> reviewer_ids who said they CANNOT recognize it. Reviewers absent from the survey are kept."""
    sv = pd.read_csv(p(cfg["paths"]["survey_csv"]), dtype=str, keep_default_na=False)
    col = next(c for c in sv.columns if c.startswith(SURVEY_RECOGNIZE_PREFIX))
    out = {}
    for smell, name in SURVEY_NAMES.items():
        out[smell] = {r for r, ans in zip(sv["reviewer_id"], sv[col]) if name not in ans}
    return out


def alpha(ratings: pd.DataFrame, value_col: str, level: str) -> float:
    mat = ratings.pivot_table(index="reviewer_id", columns="sample_id", values=value_col, aggfunc="max")
    data = mat.to_numpy(dtype=float)
    if np.sum((~np.isnan(data)).sum(axis=0) >= 2) == 0:
        return float("nan")
    return float(krippendorff.alpha(reliability_data=data, level_of_measurement=level))


def _labels_for_rule(r: pd.DataFrame, base: pd.DataFrame, cannot: dict[str, set[str]],
                     rule: str, suffix: str) -> dict:
    """Add y / high_conf / annot_filtered columns for one binarization rule to `base` (in place),
    with `suffix` appended to each column name; return that rule's agreement block."""
    r = r.assign(y=binarize(r["sev"], rule))
    out = {"smells": {}}
    for smell, short in SMELLS.items():
        rs = r[r["smell"] == smell]
        agg = rs.groupby("sample_id").agg(sev=("sev", "max"), n_reviews=("sev", "size"),
                                          y_min=("y", "min"), y_max=("y", "max"))
        y_col, hc_col = f"y_{short}{suffix}", f"high_conf_{short}{suffix}"
        # Int64: a sample not rated for this smell (6 Blob, 2 Data Class) stays <NA>, never 0.
        base[y_col] = binarize(agg["sev"], rule).astype("Int64")
        multi = agg["n_reviews"] >= 2
        base[hc_col] = (multi & (agg["y_min"] == agg["y_max"])).where(multi)

        kept = rs[~rs["reviewer_id"].isin(cannot[smell])]
        filt = kept.groupby("sample_id")["sev"].max()
        af_col = f"y_{short}_annot_filtered{suffix}"
        base[af_col] = binarize(filt, rule).astype("Int64").reindex(base.index)  # <NA> = no rating left

        out["smells"][smell] = {
            "alpha_binary": alpha(rs, "y", "nominal"),
            "alpha_ordinal": alpha(rs, "sev", "ordinal"),   # rule-independent (4 severity levels)
            "alpha_binary_annot_filtered": alpha(kept, "y", "nominal"),
            "n_samples": int(len(agg)),
            "n_unrated": int(base[y_col].isna().sum()),
            "n_multi_reviewed": int(multi.sum()),
            "n_positive": int(base[y_col].sum()),
            "positive_rate": round(float(base[y_col].mean()), 4),
            "n_high_conf": int(base[hc_col].fillna(False).astype(bool).sum()),
            "n_positive_annot_filtered": int(base[af_col].sum()),
            "n_unrated_annot_filtered": int(base[af_col].isna().sum()),
            "reviewers_cannot_recognize": sorted(cannot[smell], key=int),
            "n_ratings_dropped_by_annot_filter": int(len(rs) - len(kept)),
        }
    out["n_positive_both"] = int(((base[f"y_blob{suffix}"] == 1) & (base[f"y_data_class{suffix}"] == 1)).sum())
    return out


def build_labels(cfg: dict) -> tuple[pd.DataFrame, dict]:
    """Primary labels (labels.minor_severity_rule) keep the plain column names (y_blob, ...); each
    rule in labels.sensitivity_severity_rules adds the same columns suffixed `_<rule>`
    (y_blob_major_critical_only, ...) and an entry under agreement["sensitivity"]."""
    rule = cfg["labels"]["minor_severity_rule"]
    sens_rules = list(cfg["labels"].get("sensitivity_severity_rules") or [])
    if cfg["labels"]["reviewer_aggregation"] != "max_severity":
        raise ValueError("only max_severity aggregation is implemented")
    if rule in sens_rules:
        raise ValueError(f"primary rule {rule!r} also listed in labels.sensitivity_severity_rules")
    r = class_ratings(cfg)
    cannot = recognizers(cfg)

    base = r[["sample_id", "repository", "repo_key"]].drop_duplicates("sample_id").set_index("sample_id")
    for smell, short in SMELLS.items():
        agg = r[r["smell"] == smell].groupby("sample_id")["sev"].agg(["max", "size"])
        base[f"sev_{short}"] = agg["max"]
        base[f"n_reviews_{short}"] = agg["size"]

    agreement = {"minor_severity_rule": rule, **_labels_for_rule(r, base, cannot, rule, "")}
    agreement["n_samples"] = int(len(base))
    agreement["sensitivity"] = {s: _labels_for_rule(r, base, cannot, s, f"_{s}") for s in sens_rules}
    return base.reset_index(), agreement


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.parse_args(argv)
    cfg = load_config()
    if not cfg["gates"]["join_class_verified"]:
        sys.exit("Step 3 gate not passed: set gates.join_class_verified after checking join_manual_check.csv.")
    labels, agreement = build_labels(cfg)
    out = p(cfg["paths"]["labels_class"])
    labels.to_parquet(out, index=False)
    res = write_results(cfg, "agreement", {"agreement": agreement})
    blocks = {f"{agreement['minor_severity_rule']} (primary)": agreement, **agreement["sensitivity"]}
    for name, block in blocks.items():
        print(f"[{name}]  positive for both: {block['n_positive_both']}")
        for smell, a in block["smells"].items():
            print(f"  {smell:<11} pos={a['n_positive']:<4} alpha_bin={a['alpha_binary']:.3f} "
                  f"alpha_ord={a['alpha_ordinal']:.3f}  n={a['n_samples']} multi-reviewed={a['n_multi_reviewed']} "
                  f"annot_filtered_pos={a['n_positive_annot_filtered']} high_conf={a['n_high_conf']}")
    print(f"wrote {out}\n      {res}")


if __name__ == "__main__":
    main()
