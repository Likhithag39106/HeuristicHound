from common import load_config
from phase_a_data.fetch_sources import build_plan, fetch_url


def test_aliases_change_url_not_identity():
    cfg = load_config()
    plan = {x["repo_key"]: x for x in build_plan(cfg)}
    jgit = plan["eclipse__jgit"]
    assert jgit["url"] == "https://github.com/eclipse-jgit/jgit.git"
    assert jgit["repository"] == "git@github.com:eclipse/jgit.git"   # MLCQ identity kept
    assert jgit["fetched_from"] == "eclipse-jgit/jgit"
    assert plan["apache__tez"]["fetched_from"] is None if "apache__tez" in plan else True
    assert fetch_url("git@github.com:a/b.git", {}) == "https://github.com/a/b.git"
    assert len(plan) == 522 - len(cfg["fetch"]["excluded_repos"] or [])
