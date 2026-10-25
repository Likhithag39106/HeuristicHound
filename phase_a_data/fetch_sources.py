"""Step 1 — resolve MLCQ (repository, commit_hash) pairs to local clones. Resumable.

This module owns the fetch plan, the git fetch itself and the progress manifest.
It does not bulk-clone on its own: single-pass extraction (Step 2, extract_all.py) drives it
one repo at a time so a clone is extracted and deleted before the next one lands on disk.

    python -m phase_a_data.fetch_sources --plan      # print fetch order and counts
    python -m phase_a_data.fetch_sources --status    # summarize the manifest
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import os
import random
import shutil
import stat
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

from common import https_url, load_config, load_samples, long_path, owner_name, p, read_json, write_json

STATUSES = ("pending", "cloned", "extracted", "failed")


# ---------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------

def build_plan(cfg: dict) -> list[dict]:
    """One entry per (repository, commit_hash): class-level repos first, each group shuffled
    with a fixed seed so the first `pilot_repo_count` entries are a representative pilot."""
    df = load_samples(cfg)
    counts = (df.drop_duplicates("sample_id")
                .groupby(["repo_key", "repository", "commit_hash", "type"]).size()
                .unstack(fill_value=0).reset_index())
    for col in ("class", "function"):
        if col not in counts:
            counts[col] = 0
    aliases = cfg["fetch"].get("repo_aliases") or {}
    swh = set(cfg["fetch"].get("software_heritage_repos") or [])
    plan = [{"repo_key": r.repo_key, "repository": r.repository,
             "url": fetch_url(r.repository, aliases),
             "fetched_from": ("software_heritage" if owner_name(r.repository) in swh
                              else aliases.get(owner_name(r.repository))),
             "commit": r.commit_hash, "n_class": int(r["class"]), "n_function": int(r["function"])}
            for _, r in counts.iterrows()]
    rng = random.Random(cfg["fetch"]["order_seed"])
    class_repos = sorted([x for x in plan if x["n_class"] > 0], key=lambda x: x["repo_key"])
    other_repos = sorted([x for x in plan if x["n_class"] == 0], key=lambda x: x["repo_key"])
    rng.shuffle(class_repos)
    rng.shuffle(other_repos)
    scope = cfg["fetch"]["repo_scope"]
    if scope == "class_only":
        return class_repos
    if scope != "all":
        raise ValueError(f"fetch.repo_scope must be 'all' or 'class_only', got {scope!r}")
    return class_repos + other_repos


def fetch_url(repository: str, aliases: dict[str, str]) -> str:
    """https URL to fetch from: the moved-to location when the repo has an alias."""
    new = aliases.get(owner_name(repository))
    return f"https://github.com/{new}.git" if new else https_url(repository)


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

class Manifest:
    """data/raw/_progress_manifest.json — per-repo status, reason, timings, disk."""

    def __init__(self, cfg: dict):
        self.path = p(cfg["fetch"]["manifest_file"])
        self.unresolved_path = p(cfg["fetch"]["unresolved_file"])
        self.data: dict[str, dict] = read_json(self.path, default={})

    def get(self, key: str) -> dict:
        return self.data.get(key, {"status": "pending"})

    def update(self, key: str, **fields) -> None:
        entry = self.data.setdefault(key, {"status": "pending"})
        if "status" in fields and fields["status"] not in STATUSES:
            raise ValueError(fields["status"])
        entry.update(fields)
        entry["updated"] = datetime.now().isoformat(timespec="seconds")
        write_json(self.path, self.data)
        self._write_unresolved()

    def _write_unresolved(self) -> None:
        unresolved = [{"repo_key": k, "repository": v.get("repository"), "commit": v.get("commit"),
                       "stage": v.get("failed_stage"), "reason": v.get("reason"),
                       "detail": v.get("detail")}
                      for k, v in sorted(self.data.items())
                      if v.get("status") == "failed" and v.get("failed_stage") == "fetch"]
        write_json(self.unresolved_path, unresolved)

    def summary(self) -> dict:
        out = {s: 0 for s in STATUSES}
        reasons: dict[str, int] = {}
        for v in self.data.values():
            out[v.get("status", "pending")] += 1
            if v.get("status") == "failed":
                r = f'{v.get("failed_stage")}:{v.get("reason")}'
                reasons[r] = reasons.get(r, 0) + 1
        out["failure_reasons"] = reasons
        return out


# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------

def _git(args: list[str], cwd: Path, timeout: int) -> subprocess.CompletedProcess:
    env = dict(os.environ, GIT_TERMINAL_PROMPT="0", GCM_INTERACTIVE="never")
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=timeout, env=env)


def classify_git_error(stderr: str) -> str:
    s = stderr.lower()
    if ("not our ref" in s or "couldn't find remote ref" in s or "unadvertised object" in s
            or "no such remote ref" in s or "upload-pack: not our ref" in s):
        return "commit_gone"
    if "repository not found" in s or "could not read username" in s:
        return "repo_gone"
    if "filename too long" in s:
        return "path_too_long"
    if "invalid path" in s or "unable to create file" in s:
        return "invalid_path_on_windows"
    if "could not resolve host" in s or "unable to access" in s or "connection" in s:
        return "network"
    return "git_error"


# ---------------------------------------------------------------------------
# Software Heritage (repos whose pinned commit is no longer on GitHub)
# ---------------------------------------------------------------------------

SWH_API = "https://archive.softwareheritage.org/api/1"


def _swh(url: str, method: str = "GET") -> dict:
    """One API call; sleeps through the anonymous rate limit (120/hour) instead of failing."""
    for _ in range(5):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, method=method,
                                        headers={"User-Agent": "heuristic-hound-research"}), timeout=60) as r:
                if int(r.headers.get("X-RateLimit-Remaining", 99)) < 3:
                    time.sleep(max(0, int(r.headers.get("X-RateLimit-Reset", time.time())) - time.time()) + 5)
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code != 429:
                raise
            time.sleep(max(60, int(e.headers.get("X-RateLimit-Reset", time.time() + 600)) - time.time() + 5))
    raise RuntimeError(f"Software Heritage rate limit not clearing for {url}")


def swh_request_cooking(commit: str) -> dict:
    """Ask the vault to package the commit's source tree. Idempotent; returns the cooking task."""
    directory = _swh(f"{SWH_API}/revision/{commit}/")["directory"]
    return _swh(f"{SWH_API}/vault/flat/swh:1:dir:{directory}/", "POST")


def fetch_from_swh(entry: dict, dest: Path, cfg: dict) -> tuple[bool, str | None, str | None]:
    """Vault 'flat' tarball of the pinned commit's tree, unpacked with the sparse-checkout filter.
    Same content as a git checkout of that commit (Software Heritage keys commits by git hash)."""
    fc = cfg["fetch"]
    if dest.exists():
        remove_tree(dest)
    dest.mkdir(parents=True)
    try:
        task = swh_request_cooking(entry["commit"])
        deadline = time.time() + int(fc["swh_timeout_s"])
        while task["status"] in ("new", "pending"):
            if time.time() > deadline:
                return False, "swh_timeout", f"vault still {task['status']} after {fc['swh_timeout_s']}s"
            time.sleep(int(fc["swh_poll_s"]))
            task = _swh(f"{SWH_API}/vault/flat/{task['swhid']}/")
        if task["status"] != "done":
            return False, "swh_cooking_failed", str(task.get("progress_message"))[:500]
        tar_path = dest.parent / f"{dest.name}.tar.gz"
        with urllib.request.urlopen(urllib.request.Request(task["fetch_url"],
                                    headers={"User-Agent": "heuristic-hound-research"}), timeout=3600) as r,                 open(tar_path, "wb") as fh:
            shutil.copyfileobj(r, fh)
    except (urllib.error.URLError, OSError, KeyError, RuntimeError) as e:
        return False, "swh_error", f"{type(e).__name__}: {e}"[:800]

    patterns = fc.get("sparse_checkout_patterns") or ["*"]
    root = long_path(dest)
    try:
        with tarfile.open(tar_path) as tar:
            for mem in tar:
                parts = mem.name.split("/")[1:]          # drop 'swh:1:dir:<id>' (':' is illegal on NTFS)
                if not mem.isfile() or not parts or ".." in parts:
                    continue
                if not any(fnmatch.fnmatch(parts[-1], pat) for pat in patterns):
                    continue
                target = root.joinpath(*parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(mem) as src, open(target, "wb") as out:
                    shutil.copyfileobj(src, out)
    except (tarfile.TarError, OSError) as e:
        return False, "swh_unpack_error", f"{type(e).__name__}: {e}"[:800]
    finally:
        tar_path.unlink(missing_ok=True)
    return True, None, None


def fetch_repo(entry: dict, dest: Path, cfg: dict) -> tuple[bool, str | None, str | None]:
    """git init -> remote add -> sparse-checkout -> fetch --depth 1 <sha> -> checkout FETCH_HEAD.
    Repos listed in fetch.software_heritage_repos come from the Software Heritage vault instead.
    Returns (ok, reason, detail)."""
    if entry.get("fetched_from") == "software_heritage":
        return fetch_from_swh(entry, dest, cfg)
    fc = cfg["fetch"]
    timeout = int(fc["git_timeout_s"])
    if dest.exists():
        remove_tree(dest)
    dest.mkdir(parents=True)
    fetch_args = ["fetch", "--depth", str(fc["clone_depth"])]
    if fc.get("partial_clone_filter"):
        fetch_args.append(f"--filter={fc['partial_clone_filter']}")
    fetch_args += ["origin", entry["commit"]]
    steps = [
        ["init", "-q"],
        ["config", "core.longpaths", "true"],
        ["config", "core.autocrlf", "false"],
        ["config", "core.symlinks", "false"],
        ["remote", "add", "origin", entry["url"]],
    ]
    if fc.get("sparse_checkout_patterns"):
        steps.append(["sparse-checkout", "set", "--no-cone", *fc["sparse_checkout_patterns"]])
    steps += [fetch_args, ["checkout", "-q", "FETCH_HEAD"]]
    for args in steps:
        try:
            res = _git(args, dest, timeout)
        except subprocess.TimeoutExpired:
            return False, "timeout", f"git {args[0]} exceeded {timeout}s"
        if res.returncode != 0 and args[0] == "checkout" and classify_git_error(res.stderr) == "invalid_path_on_windows":
            # Non-Java files with names illegal on NTFS (e.g. '?') abort the whole checkout even
            # though sparse checkout never writes them. Retry without git's NTFS name check;
            # only *.java and build files are materialized.
            try:
                res = _git(["-c", "core.protectNTFS=false", *args], dest, timeout)
            except subprocess.TimeoutExpired:
                return False, "timeout", f"git {args[0]} exceeded {timeout}s"
        if res.returncode != 0:
            return False, classify_git_error(res.stderr), f"git {' '.join(args[:2])}: {res.stderr.strip()[-800:]}"
    return True, None, None


def remove_tree(path: Path) -> None:
    """rmtree that copes with read-only git object files on Windows."""
    def _onexc(func, target, _exc):
        os.chmod(target, stat.S_IWRITE)
        func(target)
    path = long_path(path)
    if not path.exists():
        return
    if sys.version_info >= (3, 12):
        shutil.rmtree(path, onexc=_onexc)
    else:
        shutil.rmtree(path, onerror=lambda f, t, e: _onexc(f, t, e))


def clone_dir(cfg: dict, key: str) -> Path:
    return p(cfg["paths"]["raw_clones"]) / key


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", action="store_true", help="print fetch order")
    ap.add_argument("--status", action="store_true", help="summarize the progress manifest")
    args = ap.parse_args(argv)
    cfg = load_config()
    if args.plan:
        plan = build_plan(cfg)
        pilot = cfg["fetch"]["pilot_repo_count"]
        print(f"{len(plan)} repos ({sum(x['n_class'] > 0 for x in plan)} with class samples); "
              f"first {pilot} = pilot")
        for i, x in enumerate(plan[: max(pilot, 40)]):
            print(f"{i:4d}  {x['repo_key']:<60} class={x['n_class']:<3} function={x['n_function']}")
    if args.status or not args.plan:
        m = Manifest(cfg)
        print(m.summary())


if __name__ == "__main__":
    main()
