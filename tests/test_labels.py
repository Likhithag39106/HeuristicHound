"""Step 5 acceptance: positive counts and alpha reproduce PRD §4.4 under the primary rule
(minor_plus) and the sensitivity rule (major_critical_only)."""
import copy

import pytest

from common import load_config
from phase_a_data.labels import build_labels

SENS = "major_critical_only"


@pytest.fixture(scope="module")
def built():
    cfg = copy.deepcopy(load_config())
    cfg["labels"]["minor_severity_rule"] = "minor_plus"
    cfg["labels"]["sensitivity_severity_rules"] = [SENS]
    return build_labels(cfg)


def test_positive_counts_reproduce_prd(built):
    labels, agr = built
    assert len(labels) == 2340
    # primary (minor_plus): plain column names, what the models train on
    assert labels["y_blob"].sum() == 506 and labels["y_data_class"].sum() == 485
    assert agr["n_positive_both"] == 221
    # sensitivity (major_critical_only): suffixed columns
    assert labels[f"y_blob_{SENS}"].sum() == 286 and labels[f"y_data_class_{SENS}"].sum() == 311
    assert agr["sensitivity"][SENS]["n_positive_both"] == 63
    for col in ("y_blob", f"y_blob_{SENS}"):
        assert labels[col].isna().sum() == 6
    for col in ("y_data_class", f"y_data_class_{SENS}"):
        assert labels[col].isna().sum() == 2


def test_alpha_reproduces_prd(built):
    _, agr = built
    prim, sens = agr["smells"], agr["sensitivity"][SENS]["smells"]
    assert prim["blob"]["alpha_binary"] == pytest.approx(0.359, abs=5e-4)
    assert prim["data class"]["alpha_binary"] == pytest.approx(0.431, abs=5e-4)
    assert sens["blob"]["alpha_binary"] == pytest.approx(0.273, abs=5e-4)
    assert sens["data class"]["alpha_binary"] == pytest.approx(0.339, abs=5e-4)
    for smell in ("blob", "data class"):
        # ordinal alpha uses the 4 severity levels, so it must not depend on the binarization rule
        assert prim[smell]["alpha_ordinal"] == sens[smell]["alpha_ordinal"]
        assert prim[smell]["n_multi_reviewed"] == sens[smell]["n_multi_reviewed"]


def test_sensitivity_matches_running_it_as_primary(built):
    """The suffixed columns are exactly what the rule would give as the primary rule."""
    labels, agr = built
    cfg = copy.deepcopy(load_config())
    cfg["labels"]["minor_severity_rule"] = SENS
    cfg["labels"]["sensitivity_severity_rules"] = []
    alone, agr_alone = build_labels(cfg)
    a, b = labels.set_index("sample_id"), alone.set_index("sample_id").loc[labels["sample_id"]]
    for short in ("blob", "data_class"):
        for base_col in (f"y_{short}", f"y_{short}_annot_filtered", f"high_conf_{short}"):
            assert a[f"{base_col}_{SENS}"].equals(b[base_col]), base_col
    assert agr["sensitivity"][SENS]["smells"] == agr_alone["smells"]
    assert agr_alone["sensitivity"] == {}
    assert not any(c.endswith(SENS) for c in alone.columns)


def test_sensitivity_positives_are_subset_of_primary(built):
    labels, _ = built
    for short in ("blob", "data_class"):
        sens = labels[f"y_{short}_{SENS}"].fillna(0).astype(bool)
        prim = labels[f"y_{short}"].fillna(0).astype(bool)
        assert not (sens & ~prim).any()


def test_primary_rule_listed_as_sensitivity_is_rejected():
    cfg = copy.deepcopy(load_config())
    cfg["labels"]["minor_severity_rule"] = "minor_plus"
    cfg["labels"]["sensitivity_severity_rules"] = ["minor_plus"]
    with pytest.raises(ValueError, match="also listed"):
        build_labels(cfg)
