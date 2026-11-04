"""Shared helpers: config loading, MLCQ loading, repo keys, JSON I/O, run metadata.

Every phase script imports from here so paths and parameters come from config.yaml only.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config.yaml"

SEVERITY_ORDER = {"none": 0, "minor": 1, "major": 2, "critical": 3}


def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def p(rel: str | os.PathLike) -> Path:
    """Resolve a config path (relative to the project root, `~` allowed) to an absolute Path."""
    path = Path(rel).expanduser()
    return path if path.is_absolute() else ROOT / path


_LONG_PREFIX = "\\\\?\\"


def long_path(path: Path) -> Path:
    r"""Windows extended-length form (\\?\C:\...) so Python file calls work past MAX_PATH (260).
    Use it for Python file operations only; pass plain_path() to git/java subprocesses."""
    if os.name != "nt":
        return path
    s = str(path)
    return path if s.startswith(_LONG_PREFIX) else Path(_LONG_PREFIX + os.path.abspath(s))


def plain_path(path: Path | str) -> Path:
    s = str(path)
    return Path(s[len(_LONG_PREFIX):]) if s.startswith(_LONG_PREFIX) else Path(s)


# ---------------------------------------------------------------------------
# MLCQ
# ---------------------------------------------------------------------------

def load_samples(cfg: dict, include_excluded: bool = False) -> pd.DataFrame:
    """MLCQ annotation rows (one reviewer x one smell x one sample). Semicolon-delimited.

    Rows from fetch.excluded_repos (source unavailable) are dropped unless include_excluded,
    so every step works on the same set of repositories."""
    df = pd.read_csv(p(cfg["paths"]["samples_csv"]), sep=";", dtype=str, keep_default_na=False)
    if not include_excluded:
        excluded = set(cfg["fetch"].get("excluded_repos") or [])
        df = df[~df["repository"].map(owner_name).isin(excluded)].reset_index(drop=True)
    df["start_line"] = df["start_line"].astype(int)
    df["end_line"] = df["end_line"].astype(int)
    df["repo_key"] = df["repository"].map(repo_key)
    return df


def unique_samples(df: pd.DataFrame) -> pd.DataFrame:
    """One row per sample_id with its code address (addresses are constant per sample_id)."""
    cols = ["sample_id", "type", "code_name", "repository", "repo_key", "commit_hash",
            "path", "start_line", "end_line", "link"]
    return df[cols].drop_duplicates("sample_id").reset_index(drop=True)


def owner_name(repository: str) -> str:
    """git@github.com:owner/name.git -> owner/name"""
    s = repository.strip()
    for prefix in ("git@github.com:", "https://github.com/", "http://github.com/"):
        if s.startswith(prefix):
            s = s[len(prefix):]
            break
    return s[:-4] if s.endswith(".git") else s


def https_url(repository: str) -> str:
    return f"https://github.com/{owner_name(repository)}.git"


def repo_key(repository: str) -> str:
    """Filesystem-safe owner__name. MLCQ has exactly one commit per repository, so this is unique."""
    return owner_name(repository).replace("/", "__")


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: Any) -> None:
    """Atomic write: a crash mid-write never leaves a truncated manifest."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)
    os.replace(tmp, path)


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def git_commit_hash() -> str | None:
    """Read-only: the project's own HEAD, recorded in every results file."""
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                             text=True, timeout=30)
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def file_sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def portable_path(path: Path | str) -> str:
    """Machine-independent form for paths recorded in results: project-relative (POSIX) inside the repo,
    `~/...` under the home directory, absolute otherwise. `p()` resolves it back."""
    path = Path(path).absolute()
    for base, prefix in ((ROOT, ""), (Path.home(), "~/")):
        try:
            return prefix + path.relative_to(base).as_posix()
        except ValueError:
            continue
    return str(path)


def write_results(cfg: dict, name: str, payload: dict, split_file: Path | None = None) -> Path:
    """Write eval/results/<name>_<timestamp>.json with run metadata. Never overwrites."""
    results_dir = p(cfg["paths"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    base = f"{name}_{timestamp()}"
    out = results_dir / f"{base}.json"
    n = 1
    while out.exists():
        out = results_dir / f"{base}_{n}.json"
        n += 1
    meta = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "git_commit": git_commit_hash(),
        "seed": cfg.get("seed"),
        "split_file": portable_path(split_file) if split_file else None,
        "split_file_sha256": file_sha256(split_file) if split_file else None,
        "config": copy.deepcopy(cfg),
    }
    write_json(out, {"meta": meta, **payload})
    return out


def pid_alive(pid: int) -> bool:
    """True if a process with this PID is running. (os.kill(pid, 0) would terminate it on Windows.)"""
    if os.name == "nt":
        import ctypes
        k32 = ctypes.windll.kernel32
        h = k32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return False
        code = ctypes.c_ulong()
        ok = k32.GetExitCodeProcess(h, ctypes.byref(code))
        k32.CloseHandle(h)
        return bool(ok) and code.value == 259  # STILL_ACTIVE
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


class RunLock:
    """One extraction run at a time: a second run on the same manifest would fight over clones
    and overwrite each other's progress. A lock left by a killed run (dead PID) is taken over."""

    def __init__(self, path: Path):
        self.path = path

    def __enter__(self):
        held = read_json(self.path, default=None)
        if held and held.get("pid") != os.getpid() and pid_alive(int(held["pid"])):
            raise SystemExit(f"Another extraction run is already going (PID {held['pid']}, started "
                             f"{held.get('started')}). Watch its log instead, or pause it with the STOP file.")
        write_json(self.path, {"pid": os.getpid(), "started": datetime.now().isoformat(timespec="seconds")})
        return self

    def __exit__(self, *exc):
        held = read_json(self.path, default=None)
        if held and held.get("pid") == os.getpid():
            self.path.unlink(missing_ok=True)
        return False


def dir_size_bytes(path: Path) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.lstat(os.path.join(root, name)).st_size
            except OSError:
                pass
    return total
