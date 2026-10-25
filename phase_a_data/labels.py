"""Step 5 — class-level labels and inter-rater agreement.

Outputs:
  data/processed/labels_class.parquet   one row per class sample_id:
      sev_<smell>, y_<smell> (primary), n_reviews_<smell>,
      y_<smell>_annot_filtered (ratings from reviewers who say they can't recognize the smell dropped),
      high_conf_<smell> (multi-reviewed AND all reviewers agree on the binary label)
  eval/results/agreement_<timestamp>.json   Krippendorff's alpha per smell, binary + ordinal

Models train on the primary labels only; variants are for analysis.

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


def build_labels(cfg: dict) -> tuple[pd.DataFrame, dict]:
    rule = cfg["labels"]["minor_severity_rule"]
    if cfg["labels"]["reviewer_aggregation"] != "max_severity":
        raise ValueError("only max_severity aggregation is implemented")
    r = class_ratings(cfg)
    r["y"] = binarize(r["sev"], rule)
    cannot = recognizers(cfg)

    base = r[["sample_id", "repository", "repo_key"]].drop_duplicates("sample_id").set_index("sample_id")
    agreement = {"minor_severity_rule": rule, "smells": {}}
    for smell, short in SMELLS.items():
        rs = r[r["smell"] == smell]
        agg = rs.groupby("sample_id").agg(sev=("sev", "max"), n_reviews=("sev", "size"),
                                          y_min=("y", "min"), y_max=("y", "max"))
        base[f"sev_{short}"] = agg["sev"]
        # Int64: a sample not rated for this smell (6 Blob, 2 Data Class) stays <NA>, never 0.
        base[f"y_{short}"] = binarize(agg["sev"], rule).astype("Int64")
        base[f"n_reviews_{short}"] = agg["n_reviews"]
        multi = agg["n_reviews"] >= 2
        base[f"high_conf_{short}"] = (multi & (agg["y_min"] == agg["y_max"])).where(multi)

        kept = rs[~rs["reviewer_id"].isin(cannot[smell])]
        filt = kept.groupby("sample_id")["sev"].max()
        base[f"y_{short}_annot_filtered"] = binarize(filt, rule).astype("Int64").reindex(base.index)  # <NA> = no rating left

        agreement["smells"][smell] = {
            "alpha_binary": alpha(rs, "y", "nominal"),
            "alpha_ordinal": alpha(rs, "sev", "ordinal"),
            "alpha_binary_annot_filtered": alpha(kept, "y", "nominal"),
            "n_samples": int(len(agg)),
            "n_unrated": int(base[f"y_{short}"].isna().sum()),
            "n_multi_reviewed": int(multi.sum()),
            "n_positive": int(base[f"y_{short}"].sum()),
            "positive_rate": round(float(base[f"y_{short}"].mean()), 4),
            "n_high_conf": int(base[f"high_conf_{short}"].fillna(False).astype(bool).sum()),
            "reviewers_cannot_recognize": sorted(cannot[smell], key=int),
            "n_ratings_dropped_by_annot_filter": int(len(rs) - len(kept)),
        }
    agreement["n_samples"] = int(len(base))
    agreement["n_positive_both"] = int(((base["y_blob"] == 1) & (base["y_data_class"] == 1)).sum())
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
    for smell, a in agreement["smells"].items():
        print(f"{smell:<11} pos={a['n_positive']:<4} alpha_bin={a['alpha_binary']:.3f} "
              f"alpha_ord={a['alpha_ordinal']:.3f}  multi-reviewed={a['n_multi_reviewed']}")
    print(f"wrote {out}\n      {res}")


if __name__ == "__main__":
    main()
