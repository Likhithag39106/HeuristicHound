"""Results record machine-independent paths (no user name / drive), and p() resolves them back."""
import copy
import json
from pathlib import Path

from common import ROOT, load_config, p, portable_path, write_results


def test_portable_path_round_trips():
    inside = ROOT / "eval" / "splits" / "class_repo_split.json"
    assert portable_path(inside) == "eval/splits/class_repo_split.json"
    home = Path.home() / "HeuristicHound_local" / "data" / "x.parquet"
    assert portable_path(home) == "~/HeuristicHound_local/data/x.parquet"
    for path in (inside, home):
        assert p(portable_path(path)) == path


def test_write_results_records_portable_split_file(tmp_path):
    cfg = copy.deepcopy(load_config())
    cfg["paths"]["results_dir"] = str(tmp_path)
    split = ROOT / "eval" / "splits" / "class_repo_split.json"
    out = write_results(cfg, "probe", {"x": 1}, split)
    meta = json.loads(out.read_text(encoding="utf-8"))["meta"]
    assert meta["split_file"] == "eval/splits/class_repo_split.json"
    assert Path.home().name not in json.dumps(meta["split_file"])
