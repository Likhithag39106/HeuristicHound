"""Targeted re-extraction for the few Step 2 outputs that failed, without re-cloning a repo.

Each recovery is listed in config.yaml (`extraction.recoveries`) with the evidence for it. The
sources are fetched at the pinned commit from the GitHub contents API, and every file's git
blob SHA is checked against the API listing before it is used.

  ck_units         a scoped CK unit that crashed (`_scoped_repos.json`, kind `failed`). CK is
                   re-run on that folder without the files named in `exclude`. Output goes to
                   ck/<repo_key>/class_recovered.csv (+ method_recovered.csv); the unit is
                   re-recorded as kind `folder_recovered` with the excluded files and the
                   original failure reason.
  relational_files a file whose relational row is a parse error. It is re-analysed with the
                   current relational.py, and its rows in relational/<repo_key>.csv
                   (+ _method.csv) are replaced. Backups go to extraction.recovery_dir.

Every action is appended to extraction.recovery_log. Nothing is invented: if CK or javalang
still fails, the original failure stays in place and the log says why.

    python -m phase_a_data.recover_extraction [--only ck_units|relational_files]
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import time
import urllib.request
from datetime import datetime
from pathlib import Path

import pandas as pd

from common import (load_config, load_samples, long_path, owner_name, p, read_json, repo_key, timestamp,
                    write_json)
from phase_a_data.extract_all import add_rel_paths, read_source, run_ck
from phase_a_data.relational import CLASS_COLUMNS, METHOD_COLUMNS, analyze_source

API = "https://api.github.com/repos/{repo}/contents/{path}?ref={commit}"


def git_blob_sha(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def _get(url: str, timeout_s: float) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "HeuristicHound-recovery"})
    with urllib.request.urlopen(req, timeout=timeout_s) as r:
        return r.read()


def fetch_tree(repo: str, path: str, commit: str, dest_root: Path, timeout_s: float = 60,
               get=None) -> list[Path]:
    """Download `path` (a file or a folder, recursively) at `commit` under dest_root/<path>.
    Raises if any file's blob SHA differs from the listing."""
    get = get or _get
    listing = json.loads(get(API.format(repo=repo, path=path.strip("/"), commit=commit), timeout_s))
    entries = listing if isinstance(listing, list) else [listing]
    files = []
    for e in entries:
        if e["type"] == "dir":
            files += fetch_tree(repo, "/" + e["path"], commit, dest_root, timeout_s, get)
            continue
        if e["type"] != "file":
            continue
        data = get(e["download_url"], timeout_s)
        if git_blob_sha(data) != e["sha"]:
            raise ValueError(f"blob SHA mismatch for {e['path']} at {commit}")
        out = dest_root / e["path"]
        long_path(out.parent).mkdir(parents=True, exist_ok=True)  # deep repo paths pass MAX_PATH
        long_path(out).write_bytes(data)
        files.append(out)
        time.sleep(0.2)
    return files


def repo_info(cfg: dict, repo: str) -> tuple[str, str, str]:
    """(repo_key, GitHub owner/name to fetch from, pinned commit) for an MLCQ owner/name."""
    if repo in (cfg["fetch"].get("software_heritage_repos") or []):
        raise ValueError(f"{repo} is fetched from Software Heritage; recovery supports GitHub only")
    samples = load_samples(cfg)
    commits = samples.loc[samples["repository"].map(owner_name) == repo, "commit_hash"].unique()
    if len(commits) != 1:
        raise ValueError(f"{repo}: expected one pinned commit, found {list(commits)}")
    fetch_from = (cfg["fetch"].get("repo_aliases") or {}).get(repo, repo)
    return repo_key(f"git@github.com:{repo}.git"), fetch_from, commits[0]


def mark_unit_recovered(scoped: dict, key: str, unit_path: str, record: dict) -> bool:
    """Replace the `failed` (or an earlier `folder_recovered`) unit for unit_path in place."""
    for i, u in enumerate(scoped.get(key, {}).get("units", [])):
        if u["path"] == unit_path and u["kind"] in ("failed", "folder_recovered"):
            previous = u.get("previous_failure") or u.get("reason")
            scoped[key]["units"][i] = {"kind": "folder_recovered", "path": unit_path,
                                       "previous_failure": previous, **record}
            return True
    return False


def recover_ck_unit(cfg: dict, spec: dict, log: list) -> None:
    key, fetch_from, commit = repo_info(cfg, spec["repo"])
    unit = spec["path"].rstrip("/")
    work = p(cfg["extraction"]["recovery_dir"]) / key
    tree = work / "tree"
    files = fetch_tree(fetch_from, unit, commit, tree)
    excluded = set(spec.get("exclude") or [])
    unexpected = excluded - {f.name for f in files}
    if unexpected:
        raise ValueError(f"exclude lists files not in {unit}: {sorted(unexpected)}")
    src = work / "ck_input"
    if src.exists():
        shutil.rmtree(long_path(src))
    for f in files:
        if f.name not in excluded and f.suffix == ".java":
            dest = src / f.relative_to(tree)
            long_path(dest.parent).mkdir(parents=True, exist_ok=True)
            shutil.copy2(long_path(f), long_path(dest))
    ck_dir = work / "ck_out"
    ok, reason = run_ck(src / unit.lstrip("/"), ck_dir, cfg)
    entry = {"type": "ck_unit", "repo_key": key, "unit": unit, "commit": commit,
             "fetched_from": f"github:{fetch_from}", "n_files": len(files),
             "excluded_files": sorted(excluded), "why": spec.get("why"), "ok": ok, "ck_error": reason}
    if not ok:
        entry["note"] = "CK still fails; the unit stays `failed`"
        log.append(entry)
        print(f"{key} {unit}: CK still fails ({reason}); unit left as failed")
        return
    out = p(cfg["paths"]["ck_output"]) / key
    cls = add_rel_paths(ck_dir / "class.csv", src)
    cls.to_csv(out / "class_recovered.csv", index=False)
    if (ck_dir / "method.csv").exists():
        add_rel_paths(ck_dir / "method.csv", src).to_csv(out / "method_recovered.csv", index=False)
    scoped_path = p(cfg["extraction"]["scoped_repos_log"])
    scoped = read_json(scoped_path, default={})
    if not mark_unit_recovered(scoped, key, unit, {"excluded_files": sorted(excluded), "why": spec.get("why"),
                                                   "recovered": datetime.now().isoformat(timespec="seconds")}):
        raise ValueError(f"{key}: no failed unit {unit} in {scoped_path}")
    write_json(scoped_path, scoped)
    entry["n_classes"] = int(len(cls))
    log.append(entry)
    print(f"{key} {unit}: CK ok without {sorted(excluded)} -> {len(cls)} classes in class_recovered.csv")


def _replace_rows(csv_path: Path, file: str, new_rows: list[dict], columns: list[str]) -> None:
    """Swap the rows of one file in a relational CSV, leaving every other row's text untouched."""
    old = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    buf = io.StringIO()
    pd.DataFrame(new_rows, columns=columns).to_csv(buf, index=False)
    buf.seek(0)
    new = pd.read_csv(buf, dtype=str, keep_default_na=False)
    pd.concat([old[old["file"] != file], new], ignore_index=True).to_csv(csv_path, index=False)


def recover_relational_file(cfg: dict, spec: dict, log: list) -> None:
    ex = cfg["extraction"]
    key, fetch_from, commit = repo_info(cfg, spec["repo"])
    work = p(ex["recovery_dir"]) / key
    path = spec["path"]
    (f,) = fetch_tree(fetch_from, path, commit, work / "tree")
    text, enc = read_source(long_path(f))
    classes, methods = analyze_source(text, path, ex["atfd_ignored_receiver_types"], ex["relational_method_level"])
    entry = {"type": "relational_file", "repo_key": key, "file": path, "commit": commit,
             "fetched_from": f"github:{fetch_from}", "encoding": enc, "why": spec.get("why")}
    errors = [c["parse_error"] for c in classes if c["parse_error"]]
    if errors:
        entry.update(ok=False, parse_error=errors[0], note="still fails to parse; rows left unchanged")
        log.append(entry)
        print(f"{key} {path}: still fails to parse ({errors[0]}); left unchanged")
        return
    rel_dir = p(cfg["paths"]["relational"])
    backup = work / f"relational_backup_{timestamp()}"
    backup.mkdir(parents=True, exist_ok=True)
    targets = [(rel_dir / f"{key}.csv", classes, CLASS_COLUMNS)]
    if ex["relational_method_level"] and (rel_dir / f"{key}_method.csv").exists():
        targets.append((rel_dir / f"{key}_method.csv", methods, METHOD_COLUMNS))
    for csv_path, rows, cols in targets:
        shutil.copy2(csv_path, backup / csv_path.name)
        _replace_rows(csv_path, path, rows, cols)
    entry.update(ok=True, n_class_rows=len(classes), n_method_rows=len(methods), backup=str(backup),
                 classes={c["class_fqn"]: {k: c[k] for k in ("atfd", "fdp", "nopa", "noam", "woc")}
                          for c in classes})
    log.append(entry)
    print(f"{key} {path}: re-analysed -> {len(classes)} class rows, {len(methods)} method rows")


def main(cfg: dict | None = None, only: str | None = None) -> None:
    cfg = cfg or load_config()
    rec = cfg["extraction"].get("recoveries") or {}
    log_path = p(cfg["extraction"]["recovery_log"])
    log = read_json(log_path, default=[])
    run: list[dict] = []
    if only in (None, "ck_units"):
        for spec in rec.get("ck_units") or []:
            recover_ck_unit(cfg, spec, run)
    if only in (None, "relational_files"):
        for spec in rec.get("relational_files") or []:
            recover_relational_file(cfg, spec, run)
    stamp = datetime.now().isoformat(timespec="seconds")
    write_json(log_path, log + [{**e, "run": stamp} for e in run])
    print(f"logged {len(run)} recoveries to {log_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["ck_units", "relational_files"])
    main(only=ap.parse_args().only)
