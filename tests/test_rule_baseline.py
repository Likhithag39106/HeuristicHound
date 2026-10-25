import copy

import pandas as pd

from common import load_config
from phase_b_metrics.rule_baseline import data_class, evaluate_rule, god_class


def test_god_class_rule():
    cfg = copy.deepcopy(load_config())
    df = pd.DataFrame({
        "atfd": [6, 6, 2, None], "ck_wmc": [50, 50, 80, 90], "ck_tcc": [0.1, 0.5, 0.1, 0.1],
        "y_blob": pd.array([1, 0, 1, 1], dtype="Int64"), "repository": ["a", "a", "b", "c"],
    })
    pred, missing = god_class(df, cfg["rule_baseline"]["god_class"])
    assert pred.tolist() == [1, 0, 0, 0] and missing.tolist() == [False, False, False, True]
    cfg["evaluation"]["bootstrap_n"] = 50
    res = evaluate_rule("god_class", df, cfg)
    assert res["metrics"]["f1"] == 0.5 and res["n_missing_rule_inputs_predicted_negative"] == 1


def test_data_class_rule_with_pmd_constants():
    c = {"woc_lt": 1 / 3, "nopa_noam_gt_low": 3, "wmc_lt_low": 31, "nopa_noam_gt_high": 5, "wmc_lt_high": 47}
    df = pd.DataFrame({
        # exposed=4,wmc=10 -> low branch; exposed=6,wmc=40 -> high branch; woc too high; nothing exposed
        "woc": [0.1, 0.2, 0.5, 0.1], "nopa": [2, 3, 5, 0], "noam": [2, 3, 5, 1], "ck_wmc": [10, 40, 5, 5],
    })
    pred, missing = data_class(df, c)
    assert pred.tolist() == [1, 1, 0, 0] and not missing.any()


def test_rule_skipped_while_constants_null():
    cfg = copy.deepcopy(load_config())
    cfg["rule_baseline"]["data_class"]["woc_lt"] = None
    df = pd.DataFrame({"y_data_class": pd.array([1], dtype="Int64"), "repository": ["a"]})
    assert evaluate_rule("data_class", df, cfg)["skipped"]
