"""Step 5 acceptance: positive counts reproduce PRD §4.4 under major_critical_only."""
import copy

from common import load_config
from phase_a_data.labels import build_labels


def test_positive_counts_reproduce_prd():
    cfg = copy.deepcopy(load_config())
    cfg["labels"]["minor_severity_rule"] = "major_critical_only"
    labels, agr = build_labels(cfg)
    assert len(labels) == 2340
    assert labels["y_blob"].sum() == 286 and labels["y_data_class"].sum() == 311
    assert agr["n_positive_both"] == 63
    assert labels["y_blob"].isna().sum() == 6 and labels["y_data_class"].isna().sum() == 2
    cfg["labels"]["minor_severity_rule"] = "minor_plus"
    labels, _ = build_labels(cfg)
    assert labels["y_blob"].sum() == 506 and labels["y_data_class"].sum() == 485
