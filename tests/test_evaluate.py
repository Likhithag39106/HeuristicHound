import numpy as np

from evaluate import binary_metrics, bootstrap_ci, paired_bootstrap


def test_metrics_and_bootstrap():
    rng = np.random.default_rng(0)
    y = rng.random(400) < 0.12
    score = y * 0.5 + rng.random(400) * 0.6
    groups = rng.integers(0, 40, 400)
    m = binary_metrics(y, score, threshold=0.5)
    assert 0 < m["f1"] <= 1 and m["n_pos"] == y.sum() and not m["pr_auc_single_point"]
    rule = binary_metrics(y, y_pred=score > 0.7)
    assert rule["pr_auc_single_point"]
    ci = bootstrap_ci(y, score >= 0.5, score, groups, "mcc", n_boot=200)
    assert ci["ci_low"] <= ci["point"] <= ci["ci_high"] and ci["n_groups"] == 40
    d = paired_bootstrap(y, score >= 0.5, score, score >= 0.9, None, groups, "f1", n_boot=200)
    assert d["diff_point"] > 0
