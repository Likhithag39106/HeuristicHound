"""Step 2 — single-pass extraction: fetch -> CK + snippets + relational -> delete clone.

One repository at a time, in the Step 1 plan order:
  1. fetch at the pinned commit (fetch_sources.fetch_repo)
  2. CK class.csv + method.csv          -> data/processed/ck/<repo_key>/
  3. snippets for every MLCQ sample     -> data/processed/snippets/<sample_id>.java
  4. relational features (javalang)     -> data/processed/relational/<repo_key>.csv (+ _method.csv)
  5. manifest = extracted, THEN delete the clone.
A clone is never deleted unless all three outputs exist and are non-empty.

    python -m phase_a_data.extract_all --pilot          # first fetch.pilot_repo_count repos, then report
    python -m phase_a_data.extract_all                  # full run (resumable; skips extracted repos)
    python -m phase_a_data.extract_all --repos apache__tez --retry-failed

Pause safely: create the file fetch.stop_file (an empty file named STOP); the run finishes the
current repo and exits. Resume: run the same command again; extracted repos are skipped.
"""
from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import os
import sys
import time
from pathlib import Path

import pandas as pd

from common import (RunLock, dir_size_bytes, load_config, load_samples, long_path, p, plain_path,
                    read_json, unique_samples, write_json)
from phase_a_data.fetch_sources import Manifest, build_plan, clone_dir, fetch_repo, remove_tree
from phase_a_data.relational import CLASS_COLUMNS, METHOD_COLUMNS, analyze_source

BUILD_FILES = ("pom.xml", "build.gradle", "build.gradle.kts")


# ---------------------------------------------------------------------------
# Source reading
# ---------------------------------------------------------------------------

def read_source(path: Path) -> tuple[str, str]:
    """Return (text, encoding). UTF-8 (BOM tolerated), falling back to latin-1."""
    raw = path.read_bytes()
    try:
        return raw.decode("utf-8-sig"), "utf-8"
    except UnicodeDecodeError:
        return raw.decode("latin-1"), "latin-1"


def sample_file(clone: Path, mlcq_path: str) -> Path:
    return clone / mlcq_path.lstrip("/")


def rel_path(clone: Path, file: str) -> str:
    """CK writes absolute paths; store '/repo/relative/path.java' to match MLCQ `path`."""
    try:
        rel = Path(os.path.abspath(plain_path(file))).relative_to(Path(os.path.abspath(plain_path(clone))))
    except ValueError:
        return str(file)
    return "/" + rel.as_posix()


# ---------------------------------------------------------------------------
# CK
# ---------------------------------------------------------------------------

def java_path(cfg: dict) -> str | None:
    """extraction.java_executable: a path (relative to the project root) or a name on PATH."""
    exe = cfg["extraction"]["java_executable"]
    if "/" in exe or "\\" in exe:
        path = p(exe)
        return str(path) if path.exists() else None
    return shutil.which(exe)


def check_java(cfg: dict) -> str:
    ex = cfg["extraction"]
    jar = p(ex["ck_jar_path"]) if ex.get("ck_jar_path") else None
    if jar is None or not jar.exists():
        sys.exit(f"CK jar not found at {jar}. Set extraction.ck_jar_path in config.yaml.")
    java = java_path(cfg)
    if java is None:
        sys.exit(f"Java not found at '{ex['java_executable']}'. Install a JDK (17+) or fix extraction.java_executable.")
    res = subprocess.run([java, "-version"], capture_output=True, text=True)
    return (res.stderr or res.stdout).splitlines()[0]


def run_ck(src_dir: Path, out_dir: Path, cfg: dict) -> tuple[bool, str | None]:
    """java -jar ck.jar <dir> false <maxFiles> <varsAndFields> <out_dir/>  (no jars).
    variable.csv / field.csv are deleted afterwards; only class.csv and method.csv are kept."""
    ex = cfg["extraction"]
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [java_path(cfg), f"-Xmx{ex['java_xmx']}", "-jar", str(p(ex["ck_jar_path"])),
           str(plain_path(src_dir)), "false", str(ex["ck_max_files_per_partition"]),
           str(bool(ex["ck_variables_and_fields"])).lower(),
           str(out_dir) + os.sep]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
                             timeout=int(ex["ck_timeout_s"]))
    except subprocess.TimeoutExpired:
        return False, "ck_timeout"
    (out_dir / "ck_stderr.log").write_text(res.stderr[-200_000:], encoding="utf-8")
    if res.returncode != 0:
        return False, f"ck_exit_{res.returncode}"
    for extra in ("variable.csv", "field.csv"):
        (out_dir / extra).unlink(missing_ok=True)
    if not (out_dir / "class.csv").exists():
        return False, "ck_no_output"
    return True, None


def add_rel_paths(csv_path: Path, clone: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, dtype={"file": str, "class": str}, keep_default_na=False,
                     na_values=["NaN"], quoting=csv.QUOTE_MINIMAL, low_memory=False)
    df.insert(1, "rel_path", df["file"].map(lambda f: rel_path(clone, f)))
    df = df.drop(columns=["file"])
    df.to_csv(csv_path, index=False)
    return df


def clear_dir(path: Path, attempts: int = 5) -> None:
    """Empty an output dir from a previous attempt. Files only, with retries: the project sits in
    OneDrive, which briefly locks synced files and refuses deletion of watched folders."""
    if not path.exists():
        return
    for f in sorted(path.rglob("*"), key=lambda x: len(x.parts), reverse=True):
        for i in range(attempts):
            try:
                if f.is_file():
                    f.unlink()
                else:
                    f.rmdir()
                break
            except FileNotFoundError:
                break
            except PermissionError:
                if f.is_dir():
                    break  # an empty leftover subfolder is harmless
                if i == attempts - 1:
                    raise
                time.sleep(0.5 * (i + 1))


def module_root(clone: Path, file: Path) -> Path:
    """Nearest ancestor directory holding a Maven/Gradle build file; the file's dir otherwise."""
    cur = file.parent
    while True:
        if any((cur / b).exists() for b in BUILD_FILES):
            return cur
        if cur == clone or clone not in cur.parents:
            return file.parent
        cur = cur.parent


def ck_scoped(clone: Path, out: Path, key: str, files: list[Path], cfg: dict, log: dict,
              why: str) -> tuple[bool, str | None]:
    """Run CK per Maven/Gradle module holding a sample file; where a module still crashes CK,
    per folder of each sample file. Writes combined class.csv / method.csv (with rel_path) and
    records every unit in _scoped_repos.json. Metrics that need the whole project (DIT, NOC,
    fan-in) only see the scoped unit."""
    clear_dir(out)
    parts_c, parts_m, units = [], [], []

    def collect(sub: Path, kind: str, unit: Path) -> None:
        parts_c.append(add_rel_paths(sub / "class.csv", clone))
        if (sub / "method.csv").exists():
            parts_m.append(add_rel_paths(sub / "method.csv", clone))
        units.append({"kind": kind, "path": rel_path(clone, str(unit))})

    roots: dict[Path, list[Path]] = {}
    for f in files:
        roots.setdefault(module_root(clone, f), []).append(f)
    for i, (root, fs) in enumerate(sorted(roots.items())):
        sub = out / "scoped" / f"module_{i}"
        ok, reason = run_ck(root, sub, cfg) if root != clone else (False, "module is repo root")
        if ok:
            collect(sub, "module", root)
            continue
        for j, d in enumerate(sorted({x.parent for x in fs})):
            sub = out / "scoped" / f"module_{i}_dir_{j}"
            ok, reason = run_ck(d, sub, cfg)
            if ok:
                collect(sub, "folder", d)
            else:
                units.append({"kind": "failed", "path": rel_path(clone, str(d)), "reason": reason})
    log["ck_scoped_to"] = units
    scoped = read_json(p(cfg["extraction"]["scoped_repos_log"]), default={})
    scoped[key] = {"why": why, "java_mb": log.get("java_mb"), "units": units}
    write_json(p(cfg["extraction"]["scoped_repos_log"]), scoped)
    if not parts_c:
        return False, "ck_failed_even_scoped"
    pd.concat(parts_c).to_csv(out / "class.csv", index=False)
    if parts_m:
        pd.concat(parts_m).to_csv(out / "method.csv", index=False)
    return True, None


def ck_for_repo(clone: Path, key: str, samples: pd.DataFrame, cfg: dict, log: dict) -> tuple[bool, str | None]:
    """`clone` is the long_path() form; run_ck strips it for java."""
    ex = cfg["extraction"]
    out = p(cfg["paths"]["ck_output"]) / key
    clear_dir(out)
    threshold = ex.get("large_repo_size_mb_threshold")
    java_mb = sum(os.path.getsize(os.path.join(r, f)) for r, _d, fs in os.walk(clone)
                  for f in fs if f.endswith(".java")) / 1e6
    log["java_mb"] = round(java_mb, 2)

    present = [sample_file(clone, s) for s in samples["path"].unique()]
    present = [f for f in present if f.exists()]

    if threshold is not None and java_mb > threshold:
        # D-5: scope CK to the modules containing sample files.
        ok, reason = ck_scoped(clone, out, key, present, cfg, log, why=f"java_mb {java_mb:.0f} > {threshold}")
        if not ok:
            return False, reason
    else:
        ok, reason = run_ck(clone, out, cfg)
        if not ok:
            # CK aborts the whole repo when JDT chokes on one file (seen: SapMachine NPE,
            # reactor-core AIOOBE). Fall back to the modules / folders holding the samples.
            ok, reason2 = ck_scoped(clone, out, key, present, cfg, log, why=f"full-repo CK failed: {reason}")
            if not ok:
                return False, f"{reason};{reason2}"
    if "ck_scoped_to" not in log:
        cls = add_rel_paths(out / "class.csv", clone)
        if (out / "method.csv").exists():
            add_rel_paths(out / "method.csv", clone)

        # CK keeps one result per class name; a duplicate FQN elsewhere in the repo can
        # overwrite a sample's class. Rerun CK on the modules of sample files CK lost.
        if ex.get("rerun_ck_for_overwritten_classes"):
            covered = set(cls["rel_path"])
            lost = [f for f in present if rel_path(clone, str(f)) not in covered]
            log["files_missing_from_ck"] = len(lost)
            if lost:
                roots = sorted({module_root(clone, f) for f in lost})
                parts_c, parts_m = [], []
                for i, r in enumerate(roots):
                    sub = out / "supplement" / f"module_{i}"
                    ok, reason = run_ck(r, sub, cfg)
                    if ok:
                        parts_c.append(add_rel_paths(sub / "class.csv", clone))
                        if (sub / "method.csv").exists():
                            parts_m.append(add_rel_paths(sub / "method.csv", clone))
                if parts_c:
                    pd.concat(parts_c).to_csv(out / "class_supplement.csv", index=False)
                if parts_m:
                    pd.concat(parts_m).to_csv(out / "method_supplement.csv", index=False)
                log["supplement_modules"] = [rel_path(clone, str(r)) for r in roots]

    if (out / "class.csv").stat().st_size == 0 or len(pd.read_csv(out / "class.csv", usecols=[0])) == 0:
        return False, "ck_empty"
    return True, None


# ---------------------------------------------------------------------------
# Snippets + relational
# ---------------------------------------------------------------------------

def snippets_for_repo(clone: Path, samples: pd.DataFrame, cfg: dict, log: dict) -> int:
    out_dir = p(cfg["paths"]["snippets"])
    out_dir.mkdir(parents=True, exist_ok=True)
    issues = []
    written = 0
    cache: dict[str, tuple[list[str], str]] = {}
    for s in samples.itertuples():
        f = sample_file(clone, s.path)
        if not f.exists():
            issues.append({"sample_id": s.sample_id, "issue": "file_missing", "path": s.path})
            continue
        if s.path not in cache:
            text, enc = read_source(f)
            cache[s.path] = (text.splitlines(keepends=True), enc)
        lines, enc = cache[s.path]
        if enc != "utf-8":
            issues.append({"sample_id": s.sample_id, "issue": f"decoded_{enc}", "path": s.path})
        if s.end_line > len(lines) or s.start_line < 1:
            issues.append({"sample_id": s.sample_id, "issue": "line_range_out_of_bounds",
                           "n_lines": len(lines), "start": s.start_line, "end": s.end_line})
        chunk = "".join(lines[s.start_line - 1: s.end_line])
        if not chunk.strip():
            issues.append({"sample_id": s.sample_id, "issue": "empty_snippet"})
            continue
        with open(out_dir / f"{s.sample_id}.java", "w", encoding="utf-8", newline="") as fh:
            fh.write(chunk)
        written += 1
    log["snippets_written"] = written
    log["snippet_issues"] = issues
    return written


def relational_for_repo(clone: Path, key: str, samples: pd.DataFrame, cfg: dict, log: dict) -> bool:
    ex = cfg["extraction"]
    out_dir = p(cfg["paths"]["relational"])
    out_dir.mkdir(parents=True, exist_ok=True)
    class_rows, method_rows = [], []
    for path in sorted(samples["path"].unique()):
        f = sample_file(clone, path)
        if not f.exists():
            continue
        text, _ = read_source(f)
        c, m = analyze_source(text, path, ex["atfd_ignored_receiver_types"], ex["relational_method_level"])
        class_rows += c
        method_rows += m
    if not class_rows:
        return False
    pd.DataFrame(class_rows, columns=CLASS_COLUMNS).to_csv(out_dir / f"{key}.csv", index=False)
    if ex["relational_method_level"]:
        pd.DataFrame(method_rows, columns=METHOD_COLUMNS).to_csv(out_dir / f"{key}_method.csv", index=False)
    log["relational_parse_errors"] = sum(1 for r in class_rows if r["parse_error"])
    return True


def outputs_ok(key: str, samples: pd.DataFrame, cfg: dict) -> list[str]:
    """Names of missing/empty outputs; empty list means safe to delete the clone."""
    missing = []
    ck = p(cfg["paths"]["ck_output"]) / key / "class.csv"
    if not ck.exists() or ck.stat().st_size == 0:
        missing.append("ck")
    rel = p(cfg["paths"]["relational"]) / f"{key}.csv"
    if not rel.exists() or rel.stat().st_size == 0:
        missing.append("relational")
    snip_dir = p(cfg["paths"]["snippets"])
    if not any((snip_dir / f"{sid}.java").exists() for sid in samples["sample_id"]):
        missing.append("snippets")
    return missing


# ---------------------------------------------------------------------------
# Per-repo driver
# ---------------------------------------------------------------------------

def process_repo(entry: dict, samples: pd.DataFrame, cfg: dict, manifest: Manifest) -> dict:
    key = entry["repo_key"]
    dest = clone_dir(cfg, key)
    rec = {"repository": entry["repository"], "commit": entry["commit"],
           "fetched_from": entry.get("fetched_from"),
           "n_class": entry["n_class"], "n_function": entry["n_function"]}
    t0 = time.time()
    state = manifest.get(key)
    if state.get("status") == "cloned" and dest.exists():
        rec["fetch_s"] = state.get("fetch_s")
    else:
        ok, reason, detail = fetch_repo(entry, dest, cfg)
        rec["fetch_s"] = round(time.time() - t0, 1)
        if not ok:
            manifest.update(key, status="failed", failed_stage="fetch", reason=reason, detail=detail, **rec)
            remove_tree(dest)
            return manifest.get(key)
    lp = long_path(dest)  # deep repos exceed MAX_PATH; Python file ops use the long form
    rec["clone_mb"] = round(dir_size_bytes(lp) / 1e6, 1)
    manifest.update(key, status="cloned", **rec)

    log: dict = {}
    t1 = time.time()
    ok, reason = ck_for_repo(lp, key, samples, cfg, log)
    rec["ck_s"] = round(time.time() - t1, 1)
    if ok:
        t2 = time.time()
        snippets_for_repo(lp, samples, cfg, log)
        rel_ok = relational_for_repo(lp, key, samples, cfg, log)
        rec["snippets_relational_s"] = round(time.time() - t2, 1)
        missing = outputs_ok(key, samples, cfg)
        if missing:
            ok, reason = False, "missing_outputs:" + ",".join(missing)
        elif not rel_ok:
            ok, reason = False, "relational_empty"
    log_dir = p(cfg["paths"]["ck_output"]) / key
    log_dir.mkdir(parents=True, exist_ok=True)
    write_json(log_dir / "_extraction.json", log)

    rec.update({"snippets_written": log.get("snippets_written"),
                "n_snippet_issues": len(log.get("snippet_issues", [])),
                "files_missing_from_ck": log.get("files_missing_from_ck"),
                "relational_parse_errors": log.get("relational_parse_errors"),
                "total_s": round(time.time() - t0, 1)})
    if not ok:
        manifest.update(key, status="failed", failed_stage="extract", reason=reason, detail=None, **rec)
        if not cfg["extraction"]["keep_failed_clones"]:
            remove_tree(dest)
        return manifest.get(key)
    manifest.update(key, status="extracted", reason=None, detail=None, failed_stage=None, **rec)
    remove_tree(dest)  # only after every output exists
    return manifest.get(key)


def pilot_report(cfg: dict, keys: list[str], manifest: Manifest, plan: list[dict], wall_s: float) -> dict:
    rows = [manifest.get(k) | {"repo_key": k} for k in keys]
    df = pd.DataFrame(rows)
    done = df[df["status"] == "extracted"]
    n_total = len(plan)
    per_repo_s = float(df["total_s"].dropna().mean()) if "total_s" in df else float("nan")
    processed_mb = sum(dir_size_bytes(p(cfg["paths"][k])) for k in ("ck_output", "snippets", "relational")) / 1e6
    report = {
        "n_pilot": len(keys),
        "n_extracted": int(len(done)),
        "n_failed": int((df["status"] == "failed").sum()),
        "failure_rate": round(float((df["status"] == "failed").mean()), 3),
        "failures": df[df["status"] == "failed"][["repo_key", "failed_stage", "reason"]].to_dict("records"),
        "wall_clock_s": round(wall_s, 1),
        "per_repo_s_mean": round(per_repo_s, 1),
        "clone_mb": {"mean": round(float(df["clone_mb"].mean()), 1) if "clone_mb" in df else None,
                     "max": round(float(df["clone_mb"].max()), 1) if "clone_mb" in df else None},
        "peak_disk_note": "clones are deleted one at a time, so peak clone disk ~= max clone_mb",
        "processed_outputs_mb_so_far": round(processed_mb, 1),
        "projected_full_run_hours": round(per_repo_s * n_total / 3600, 1),
        "projected_processed_outputs_mb": round(processed_mb / max(len(done), 1) * n_total, 0),
        "snippet_issues": int(df["n_snippet_issues"].fillna(0).sum()) if "n_snippet_issues" in df else None,
        "files_missing_from_ck": int(df["files_missing_from_ck"].fillna(0).sum()) if "files_missing_from_ck" in df else None,
        "relational_parse_errors": int(df["relational_parse_errors"].fillna(0).sum()) if "relational_parse_errors" in df else None,
        "per_repo": df.to_dict("records"),
    }
    write_json(p(cfg["fetch"]["pilot_report_file"]), report)
    return report


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pilot", action="store_true", help="process the first fetch.pilot_repo_count repos, then stop and report")
    ap.add_argument("--limit", type=int, help="process at most N not-yet-extracted repos")
    ap.add_argument("--repos", nargs="*", help="only these repo_keys")
    ap.add_argument("--retry-failed", action="store_true", help="retry repos marked failed")
    args = ap.parse_args(argv)

    cfg = load_config()
    with RunLock(p(cfg["fetch"]["lock_file"])):
        run(args, cfg)


def run(args: argparse.Namespace, cfg: dict) -> None:
    print(f"java: {check_java(cfg)}")
    plan = build_plan(cfg)
    samples = unique_samples(load_samples(cfg))
    by_repo = {k: g for k, g in samples.groupby("repo_key")}
    manifest = Manifest(cfg)

    selected = plan
    if args.pilot:
        selected = plan[: cfg["fetch"]["pilot_repo_count"]]
    if args.repos:
        wanted = set(args.repos)
        selected = [x for x in plan if x["repo_key"] in wanted]
    todo = []
    for x in selected:
        st = manifest.get(x["repo_key"]).get("status")
        if st == "extracted" or (st == "failed" and not args.retry_failed):
            continue
        todo.append(x)
    if args.limit is not None:
        todo = todo[: args.limit]
    print(f"{len(todo)} repos to process ({len(selected) - len(todo)} skipped as already extracted/failed)")

    t0 = time.time()
    stop_file = p(cfg["fetch"]["stop_file"])
    stop_file.unlink(missing_ok=True)  # a STOP left from a previous pause must not end this run
    for i, entry in enumerate(todo, 1):
        if stop_file.exists():
            print(f"\nSTOP file found ({stop_file}): paused cleanly after {i - 1} repos. "
                  "Re-run the same command to resume.", flush=True)
            stop_file.unlink(missing_ok=True)
            break
        try:
            res = process_repo(entry, by_repo[entry["repo_key"]], cfg, manifest)
        except Exception as e:  # one bad repo must not stop the run; the clone is kept for retry
            manifest.update(entry["repo_key"], status="failed", failed_stage="extract",
                            reason=f"exception:{type(e).__name__}", detail=str(e)[-800:],
                            repository=entry["repository"], commit=entry["commit"])
            res = manifest.get(entry["repo_key"])
        print(f"[{i}/{len(todo)}] {entry['repo_key']}: {res['status']}"
              f"{' (' + str(res.get('failed_stage')) + ':' + str(res.get('reason')) + ')' if res['status'] == 'failed' else ''}"
              f"  {res.get('total_s', '?')}s  clone={res.get('clone_mb', '?')}MB", flush=True)

    if args.pilot:
        rep = pilot_report(cfg, [x["repo_key"] for x in selected], manifest, plan, time.time() - t0)
        print("\nPILOT REPORT  ->", p(cfg["fetch"]["pilot_report_file"]))
        for k in ("n_pilot", "n_extracted", "n_failed", "failure_rate", "wall_clock_s", "per_repo_s_mean",
                  "clone_mb", "processed_outputs_mb_so_far", "projected_full_run_hours",
                  "projected_processed_outputs_mb", "snippet_issues", "files_missing_from_ck",
                  "relational_parse_errors"):
            print(f"  {k}: {rep[k]}")
        print("  failures:", rep["failures"])
        print("\nSTOP: review the pilot, set extraction.large_repo_size_mb_threshold (D-5) if needed, "
              "then authorize the full run.")
    print("\nmanifest:", manifest.summary())
    print(f"REMINDER: back up {p(cfg['paths']['snippets'])} — snippets cannot be regenerated without re-cloning.")


if __name__ == "__main__":
    main()
