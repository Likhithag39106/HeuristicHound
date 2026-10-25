"""Step 6 acceptance: zero repository overlap; rate constraint; folds partition train repos."""
from common import load_config
from phase_a_data.labels import build_labels
from phase_a_data.make_splits import build_splits


def test_splits():
    cfg = load_config()
    labels, _ = build_labels(cfg)
    primary, folds, contrast = build_splits(labels, cfg)
    assert not set(primary["train_repos"]) & set(primary["test_repos"])
    assert len(primary["train_repos"]) + len(primary["test_repos"]) == 437
    for t in ("y_blob", "y_data_class"):
        assert abs(primary["positive_rate_test"][t] - primary["positive_rate_overall"][t]) <= 0.02
    for f in folds["folds"]:
        assert not set(f["train_repos"]) & set(f["val_repos"])
        assert set(f["val_repos"]) <= set(primary["train_repos"])
    assert contrast["n_test_samples"] == primary["n_test_samples"]
    assert contrast["n_repos_on_both_sides"] > 0
