"""Targeted recovery: SHA-checked fetch, scoped-log update, relational row replacement (no network)."""
import copy
import json

import pandas as pd
import pytest

from common import load_config
from phase_a_data import recover_extraction as rx
from phase_a_data.relational import CLASS_COLUMNS, METHOD_COLUMNS


def fake_github(files: dict[str, bytes], corrupt: str | None = None):
    def get(url, timeout_s):
        if url.startswith("https://api.github.com/"):
            path = url.split("/contents/")[1].split("?")[0]
            entries = [{"type": "file", "path": f, "name": f.split("/")[-1], "sha": rx.git_blob_sha(b),
                        "download_url": f"https://raw/{f}"} for f, b in files.items() if f.startswith(path)]
            return json.dumps(entries[0] if path in files else entries).encode()
        f = url[len("https://raw/"):]
        return files[f] + (b"x" if f == corrupt else b"")
    return get


def test_fetch_tree_verifies_blob_sha(tmp_path):
    files = {"src/a/A.java": b"class A {}", "src/a/B.java": b"class B {}"}
    got = rx.fetch_tree("o/r", "/src/a", "c0", tmp_path, get=fake_github(files))
    assert sorted(f.name for f in got) == ["A.java", "B.java"]
    assert (tmp_path / "src/a/A.java").read_bytes() == b"class A {}"
    with pytest.raises(ValueError, match="SHA mismatch"):
        rx.fetch_tree("o/r", "/src/a", "c0", tmp_path / "2", get=fake_github(files, corrupt="src/a/B.java"))


def test_mark_unit_recovered_keeps_previous_failure():
    scoped = {"k": {"units": [{"kind": "module", "path": "/x"},
                              {"kind": "failed", "path": "/y", "reason": "ck_exit_1"}]}}
    assert rx.mark_unit_recovered(scoped, "k", "/y", {"excluded_files": ["Z.java"]})
    u = scoped["k"]["units"][1]
    assert u["kind"] == "folder_recovered" and u["previous_failure"] == "ck_exit_1" and u["excluded_files"] == ["Z.java"]
    # re-running the recovery keeps the original failure reason
    assert rx.mark_unit_recovered(scoped, "k", "/y", {"excluded_files": ["Z.java"]})
    assert scoped["k"]["units"][1]["previous_failure"] == "ck_exit_1"
    assert not rx.mark_unit_recovered(scoped, "k", "/x", {})


def test_recover_relational_file_replaces_only_that_file(tmp_path, monkeypatch):
    cfg = copy.deepcopy(load_config())
    cfg["paths"]["relational"] = str(tmp_path / "rel")
    cfg["extraction"]["recovery_dir"] = str(tmp_path / "recovery")
    key, path = "apache__groovy", "/src/main/java/p/G.java"
    (tmp_path / "rel").mkdir()
    other = {**{c: "" for c in CLASS_COLUMNS}, "file": "/src/O.java", "class_fqn": "p.O", "atfd": "3", "parse_error": ""}
    broken = {**{c: "" for c in CLASS_COLUMNS}, "file": path, "parse_error": "JavaSyntaxError: Expected ';'"}
    pd.DataFrame([other, broken], columns=CLASS_COLUMNS).to_csv(tmp_path / "rel" / f"{key}.csv", index=False)
    pd.DataFrame([], columns=METHOD_COLUMNS).to_csv(tmp_path / "rel" / f"{key}_method.csv", index=False)
    src = b"package p;\npublic class G { Object[] f(java.util.List<X> xs) { return xs.stream().map(x -> x.getY()).toArray(Object[]::new); } }\n"
    monkeypatch.setattr(rx, "repo_info", lambda cfg, repo: (key, "apache/groovy", "c0"))
    monkeypatch.setattr(rx, "_get", fake_github({path.lstrip("/"): src}))
    log = []
    rx.recover_relational_file(cfg, {"repo": "apache/groovy", "path": path}, log)
    rel = pd.read_csv(tmp_path / "rel" / f"{key}.csv", dtype=str, keep_default_na=False)
    assert log[0]["ok"] and rel.loc[rel["file"] == "/src/O.java", "atfd"].tolist() == ["3"]
    g = rel[rel["file"] == path]
    assert g["class_fqn"].tolist() == ["p.G"] and g["parse_error"].tolist() == [""] and g["atfd"].tolist() == ["1"]
    assert list((tmp_path / "recovery" / key).glob("relational_backup_*/apache__groovy.csv"))
