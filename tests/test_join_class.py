"""Join logic on a fake CK output: nested '$' names, duplicate FQNs, simple-name fallback."""
import copy

import pandas as pd

from common import load_config, write_json
from phase_a_data import join_class

REPO = "spring-projects__spring-retry"
P1 = "/src/main/java/org/springframework/retry/backoff/ExponentialRandomBackOffPolicy.java"
P2 = "/src/main/java/org/springframework/retry/stats/DefaultStatisticsRepository.java"


def test_join(tmp_path):
    cfg = copy.deepcopy(load_config())
    for k, v in {"ck_output": "ck", "relational": "rel", "snippets": "snip", "class_dataset": "ds.parquet",
                 "join_report": "report.json", "join_manual_check": "check.csv"}.items():
        cfg["paths"][k] = str(tmp_path / v)
    cfg["fetch"]["manifest_file"] = str(tmp_path / "manifest.json")
    cfg["extraction"]["scoped_repos_log"] = str(tmp_path / "scoped.json")
    cfg["join"]["check_links"] = False  # no network in tests
    write_json(tmp_path / "manifest.json", {REPO: {"status": "extracted"}})
    (tmp_path / "ck" / REPO).mkdir(parents=True)
    pd.DataFrame([
        # nested class in CK binary-name format
        {"rel_path": P1, "class": "org.springframework.retry.backoff.ExponentialRandomBackOffPolicy$ExponentialRandomBackOffContext",
         "type": "innerclass", "wmc": 3, "loc": 15, "tcc": 0.5},
        # simple-name fallback (binding not resolved) -> matched by path suffix
        {"rel_path": P2, "class": "DefaultStatisticsRepository", "type": "class", "wmc": 12, "loc": 48, "tcc": 0.2},
    ]).to_csv(tmp_path / "ck" / REPO / "class.csv", index=False)
    (tmp_path / "rel").mkdir()
    pd.DataFrame([
        {"file": P1, "class_fqn": "org.springframework.retry.backoff.ExponentialRandomBackOffPolicy.ExponentialRandomBackOffContext",
         "simple_name": "ExponentialRandomBackOffContext", "atfd": 2, "fdp": 1, "parse_error": None},
        {"file": P2, "class_fqn": "org.springframework.retry.stats.DefaultStatisticsRepository",
         "simple_name": "DefaultStatisticsRepository", "atfd": 0, "fdp": 0, "parse_error": None},
    ]).to_csv(tmp_path / "rel" / f"{REPO}.csv", index=False)

    join_class.main(cfg)
    ds = pd.read_parquet(tmp_path / "ds.parquet").set_index("sample_id")
    assert ds.loc["7558669", "join_key"] == "name" and ds.loc["7558669", "atfd"] == 2
    assert ds.loc["7558740", "join_key"] == "path_suffix" and ds.loc["7558740", "ck_wmc"] == 12
    report = pd.read_json(tmp_path / "report.json", typ="series")
    assert report["n_matched"] == 2
    assert report["unmatched_by_reason"]["repo_not_yet_extracted"] == 2340 - 2
    assert ds.loc["7558669", "working_link"] == (
        "https://github.com/spring-projects/spring-retry/blob/e538399d7507145a4d171160be2d7afab294d4e3"
        "/src/main/java/org/springframework/retry/backoff/ExponentialRandomBackOffPolicy.java#L61-L78")
    check = pd.read_csv(tmp_path / "check.csv", dtype={"sample_id": str})
    assert "working_link" in check.columns

    # Re-run: the previously checked rows are kept, and the old file is backed up.
    join_class.main(cfg)
    again = pd.read_csv(tmp_path / "check.csv", dtype={"sample_id": str})
    assert sorted(again["sample_id"]) == sorted(check["sample_id"])
    assert list(tmp_path.glob("check.prev_*.csv"))


def test_duplicate_fqn_disambiguated_by_path():
    ck = pd.DataFrame([
        {"class_norm": "a.B", "simple": "B", "rel_path": "/m1/a/B.java", "class": "a.B"},
        {"class_norm": "a.B", "simple": "B", "rel_path": "/m2/a/B.java", "class": "a.B"},
    ])
    s = pd.Series({"code_name": "a.B", "path": "/m2/a/B.java"})
    row, key, reason = join_class.match_ck(s, ck)
    assert key == "name+path" and row["rel_path"] == "/m2/a/B.java" and reason is None
    s2 = pd.Series({"code_name": "a.B", "path": "/m3/a/B.java"})
    assert join_class.match_ck(s2, ck)[2] == "ambiguous"


def test_raw_name_fallback_for_dollar_in_top_level_name():
    ck = pd.DataFrame([{"class": "o.a.InitMethod$JAXB", "class_norm": "o.a.InitMethod.JAXB", "simple": "JAXB",
                        "rel_path": "/src/o/a/InitMethod$JAXB.java"}])
    s = pd.Series({"code_name": "o.a.InitMethod$JAXB", "path": "/src/o/a/InitMethod$JAXB.java"})
    row, key, reason = join_class.match_ck(s, ck)
    assert key == "raw_name" and row["class"] == "o.a.InitMethod$JAXB" and reason is None


def test_ck_failed_unit():
    units = [{"kind": "failed", "path": "/src/jdk.compiler/share/classes/com/sun/tools/javac/comp", "reason": "ck_exit_1"}]
    hit = join_class.ck_failed_unit("/src/jdk.compiler/share/classes/com/sun/tools/javac/comp/Resolve.java", units)
    assert hit is units[0]
    # a sibling folder sharing the prefix is not inside the unit
    assert join_class.ck_failed_unit("/src/jdk.compiler/share/classes/com/sun/tools/javac/compX/A.java", units) is None


def test_working_link():
    fetch = {"repo_aliases": {"eclipse/jgit": "eclipse-jgit/jgit"}, "software_heritage_repos": ["apache/wss4j"]}

    def s(repo, link):
        return pd.Series({"repository": f"git@github.com:{repo}.git", "commit_hash": "abc123",
                          "path": "/m/src/A.java", "start_line": 5, "end_line": 9, "link": link})

    plain = s("apache/tez", "https://github.com/apache/tez/blob/abc123/m/src/A.java/#L5-L9")
    assert join_class.working_link(plain, fetch) == "https://github.com/apache/tez/blob/abc123/m/src/A.java#L5-L9"
    moved = s("eclipse/jgit", "https://github.com/eclipse/jgit/blob/abc123/m/src/A.java/#L5-L9")
    assert join_class.working_link(moved, fetch) == "https://github.com/eclipse-jgit/jgit/blob/abc123/m/src/A.java#L5-L9"
    spaced = s("eclipse/jgit", "https://github.com/eclipse/jgit/blob/abc123/Eclipse UI/A$B.java/#L5-L9")
    assert join_class.working_link(spaced, fetch) == "https://github.com/eclipse-jgit/jgit/blob/abc123/Eclipse%20UI/A$B.java#L5-L9"
    swh = s("apache/wss4j", "https://github.com/apache/wss4j/blob/abc123/m/src/A.java/#L5-L9")
    assert join_class.working_link(swh, fetch) ==         "https://archive.softwareheritage.org/browse/revision/abc123/?path=m/src/A.java#L5-L9"


def test_pick_manual_check_keeps_previous_ids():
    ds = pd.DataFrame({"sample_id": [str(i) for i in range(50)]})
    picked = join_class.pick_manual_check(ds, ["3", "7", "999"], n=5, seed=0)
    assert len(picked) == 5 and {"3", "7"} <= set(picked["sample_id"]) and "999" not in set(picked["sample_id"])
