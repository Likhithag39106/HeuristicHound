"""Step 6 — class-level splits. Every arm reads the same files.

  eval/splits/class_repo_split.json         PRIMARY: train/test by repository (GroupShuffleSplit, 80/20)
  eval/splits/class_cv_folds.json           StratifiedGroupKFold(5, groups=repository) on TRAIN repos only
  eval/splits/class_rowlevel_contrast.json  CONTRAST_ONLY: stratified row split, same test size; Step 12 only

Splits are built on all 2,340 labelled class samples (437 class repos), so they do not depend on which samples
the Step 3 join matched; unmatched samples are dropped at training time, never re-split.
The test positive rate must be within `splits.positive_rate_tolerance_pp` of the overall rate
for both smells; if a seed misses, the next seed is tried and the seed used is recorded
(labels only — no model output is involved).

    python -m phase_a_data.make_splits      (requires gates.join_class_verified and Step 5 labels)
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold, train_test_split

from common import load_config, p, write_json

TARGETS = ["y_blob", "y_data_class"]


def _rates(df: pd.DataFrame) -> dict:
    """Positive rate among samples rated for that smell."""
    return {t: round(float(df[t].astype(float).mean()), 4) for t in TARGETS}


def _strat(df: pd.DataFrame) -> pd.Series:
    """Joint label so both smells stay balanced; unrated (<NA>) counts as 0 for stratification only."""
    return df["y_blob"].fillna(0).astype(int) * 2 + df["y_data_class"].fillna(0).astype(int)


def repo_split(labels: pd.DataFrame, cfg: dict) -> dict:
    sc = cfg["splits"]
    tol = sc["positive_rate_tolerance_pp"] / 100
    overall = _rates(labels)
    for attempt in range(sc["max_seed_attempts"]):
        seed = cfg["seed"] + attempt
        gss = GroupShuffleSplit(n_splits=1, test_size=sc["test_size"], random_state=seed)
        tr, te = next(gss.split(labels, groups=labels[sc["group_column"]]))
        test = labels.iloc[te]
        if all(abs(_rates(test)[t] - overall[t]) <= tol for t in TARGETS):
            train = labels.iloc[tr]
            train_repos = sorted(train[sc["group_column"]].unique())
            test_repos = sorted(test[sc["group_column"]].unique())
            assert not set(train_repos) & set(test_repos), "repository on both sides of the primary split"
            return {"kind": "PRIMARY_REPOSITORY_LEVEL", "group_column": sc["group_column"],
                    "seed_used": seed, "attempts": attempt + 1, "tolerance_pp": sc["positive_rate_tolerance_pp"],
                    "positive_rate_overall": overall, "positive_rate_train": _rates(train),
                    "positive_rate_test": _rates(test),
                    "n_train_samples": int(len(train)), "n_test_samples": int(len(test)),
                    "n_train_repos": len(train_repos), "n_test_repos": len(test_repos),
                    "train_repos": train_repos, "test_repos": test_repos}
    raise RuntimeError(f"no seed within {sc['max_seed_attempts']} attempts met the ±{tol*100}pp constraint")


def cv_folds(labels: pd.DataFrame, train_repos: list[str], cfg: dict) -> dict:
    sc = cfg["splits"]
    g = sc["group_column"]
    train = labels[labels[g].isin(train_repos)].reset_index(drop=True)
    strat = _strat(train)
    sgkf = StratifiedGroupKFold(n_splits=sc["cv_folds"], shuffle=True, random_state=cfg["seed"])
    folds = []
    for i, (tr, va) in enumerate(sgkf.split(train, strat, groups=train[g])):
        tr_repos, va_repos = sorted(train.iloc[tr][g].unique()), sorted(train.iloc[va][g].unique())
        assert not set(tr_repos) & set(va_repos), f"repository on both sides of CV fold {i}"
        folds.append({"fold": i, "train_repos": tr_repos, "val_repos": va_repos,
                      "n_train_samples": int(len(tr)), "n_val_samples": int(len(va)),
                      "positive_rate_val": _rates(train.iloc[va])})
    covered = sorted(r for f in folds for r in f["val_repos"])
    assert covered == sorted(train_repos), "CV validation folds must partition the training repos"
    return {"kind": "CV_FOLDS_TRAIN_REPOS_ONLY", "strategy": sc["cv_strategy"], "n_folds": sc["cv_folds"],
            "seed": cfg["seed"], "folds": folds}


def rowlevel_contrast(labels: pd.DataFrame, n_test: int, cfg: dict) -> dict:
    strat = _strat(labels)
    tr, te = train_test_split(labels.index, test_size=n_test, stratify=strat, random_state=cfg["seed"])
    shared = set(labels.loc[tr, cfg["splits"]["group_column"]]) & set(labels.loc[te, cfg["splits"]["group_column"]])
    return {"kind": "CONTRAST_ONLY",
            "warning": "Row-level split for the Step 12 leakage contrast ONLY. Never tune, select or report on it otherwise.",
            "seed": cfg["seed"], "n_train_samples": int(len(tr)), "n_test_samples": int(len(te)),
            "n_repos_on_both_sides": len(shared),
            "positive_rate_test": _rates(labels.loc[te]),
            "train_sample_ids": sorted(labels.loc[tr, "sample_id"]),
            "test_sample_ids": sorted(labels.loc[te, "sample_id"])}


def build_splits(labels: pd.DataFrame, cfg: dict) -> tuple[dict, dict, dict]:
    primary = repo_split(labels, cfg)
    folds = cv_folds(labels, primary["train_repos"], cfg)
    contrast = rowlevel_contrast(labels, primary["n_test_samples"], cfg)
    return primary, folds, contrast


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.parse_args(argv)
    cfg = load_config()
    if not cfg["gates"]["join_class_verified"]:
        sys.exit("Step 3 gate not passed: set gates.join_class_verified after checking join_manual_check.csv.")
    labels = pd.read_parquet(p(cfg["paths"]["labels_class"]))
    primary, folds, contrast = build_splits(labels, cfg)
    d = p(cfg["paths"]["splits_dir"])
    for name, obj in (("class_repo_split.json", primary), ("class_cv_folds.json", folds),
                      ("class_rowlevel_contrast.json", contrast)):
        if (d / name).exists():
            sys.exit(f"{d / name} exists. Splits are fixed once written; delete it deliberately to regenerate.")
        write_json(d / name, obj)
    print(f"primary: {primary['n_train_repos']} train repos / {primary['n_test_repos']} test repos, "
          f"{primary['n_test_samples']} test samples, seed {primary['seed_used']}")
    print(f"  test positive rate {primary['positive_rate_test']} vs overall {primary['positive_rate_overall']}")
    print(f"row-level contrast: {contrast['n_repos_on_both_sides']} repos appear on both sides (the leakage)")
    print(f"wrote {d}")


if __name__ == "__main__":
    main()
