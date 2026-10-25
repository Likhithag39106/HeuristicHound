"""Step 3 [GATE] — join CK class metrics + ATFD to the 2,340 MLCQ class samples.

Keys, in order:
  1. name:         CK class (with '$' -> '.') == code_name, within the sample's repo
                   (repo implies commit: MLCQ has one commit per repository).
                   Several CK rows with that name -> keep the one whose file == MLCQ path.
  2. raw_name:     CK class as written (no '$' normalization) == code_name, for top-level classes
                   whose own name contains '$' (tomee `InitMethod$JAXB`).
  3. path_suffix:  CK file path ends with MLCQ path AND CK simple class name == last segment
                   of code_name (covers CK's simple-name fallback when bindings don't resolve).
Every sample appears in the report: matched (with the key used) or unmatched with a reason.
A sample whose file sits under a unit CK crashed on (`_scoped_repos.json`, kind `failed`) is
reported as `ck_failed`, not `file_missing`.

`working_link` is a browsable link to the exact code the reviewers saw (MLCQ's `link` with the
`/#L` typo fixed, the new address for moved repos, Software Heritage for archived ones). The
manual-check rows get an HTTP status for it (`join.check_links`), and keep the same sample_ids
as the previous check file so rows already verified by hand stay in the check.

    python -m phase_a_data.join_class
Then STOP: the user checks data/processed/join_manual_check.csv by hand.
"""
from __future__ import annotations

import shutil
import time
import urllib.error
import urllib.request
from urllib.parse import quote, urlsplit, urlunsplit

import pandas as pd
from scipy.stats import pearsonr, spearmanr

from common import load_config, load_samples, owner_name, p, read_json, timestamp, unique_samples, write_json

REL_FEATURES = ["atfd", "fdp", "nopa", "noam", "n_public_methods", "woc"]
ID_COLS = ["sample_id", "code_name", "repository", "repo_key", "commit_hash", "path",
           "start_line", "end_line", "link"]
SWH_BROWSE = "https://archive.softwareheritage.org/browse/revision/{commit}/?path={path}#L{start}-L{end}"


def working_link(s, fetch_cfg: dict) -> str:
    """A link to the sample's lines at the pinned commit that still resolves."""
    repo = owner_name(s.repository)
    if repo in (fetch_cfg.get("software_heritage_repos") or []):
        return SWH_BROWSE.format(commit=s.commit_hash, path=quote(s.path.lstrip("/")),
                                 start=s.start_line, end=s.end_line)
    link = s.link.replace("/#L", "#L")
    new = (fetch_cfg.get("repo_aliases") or {}).get(repo)
    if new:
        link = link.replace(f"github.com/{repo}/", f"github.com/{new}/", 1)
    u = urlsplit(link)  # MLCQ links keep raw spaces in paths (e.g. ".../Eclipse UI/...")
    return urlunsplit(u._replace(path=quote(u.path, safe="/%$")))


def http_status(url: str, timeout_s: float, method: str = "GET") -> int | str:
    """Status code of a request (the fragment is not sent), or the error name if no response."""
    req = urllib.request.Request(url, method=method, headers={"User-Agent": "HeuristicHound-join-check"})
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as r:
            return r.status
    except urllib.error.HTTPError as e:
        return e.code
    except (urllib.error.URLError, OSError, ValueError) as e:  # ValueError: http.client.InvalidURL
        return type(e).__name__


RETRY_STATUSES = {429, 500, 502, 503, 504}


def check_one(url: str, lc: dict, sleep=time.sleep, status=None) -> int | str:
    """HEAD (GET if HEAD is refused), retried with backoff on 429 / 5xx / no response."""
    status = status or http_status
    st: int | str = "not_checked"
    for attempt in range(lc["max_retries"] + 1):
        st = status(url, lc["timeout_s"], method="HEAD")
        if st == 405:  # HEAD not allowed: fall back to GET
            st = status(url, lc["timeout_s"], method="GET")
        if isinstance(st, int) and st not in RETRY_STATUSES:
            return st
        sleep(lc["backoff_s"] * (2 ** attempt))
    return st


def failed_units(cfg: dict) -> dict[str, list[dict]]:
    scoped = read_json(p(cfg["extraction"]["scoped_repos_log"]), default={}) or {}
    return {k: [u for u in v.get("units", []) if u.get("kind") == "failed"] for k, v in scoped.items()}


def ck_failed_unit(path: str, units: list[dict]) -> dict | None:
    """The failed CK unit (folder/module) that contains this file, if any."""
    for u in units:
        if path.startswith(u["path"].rstrip("/") + "/"):
            return u
    return None


def pick_manual_check(ds: pd.DataFrame, previous_ids: list[str], n: int, seed: int) -> pd.DataFrame:
    """Keep previously checked sample_ids (still matched); top up at random to n."""
    keep = ds[ds["sample_id"].isin(previous_ids)]
    rest = ds[~ds["sample_id"].isin(previous_ids)]
    extra = rest.sample(min(max(n - len(keep), 0), len(rest)), random_state=seed)
    return pd.concat([keep, extra])


def load_ck_classes(cfg: dict, keys: list[str]) -> pd.DataFrame:
    root = p(cfg["paths"]["ck_output"])
    parts = []
    for k in keys:
        for name in ("class.csv", "class_supplement.csv", "class_recovered.csv"):
            f = root / k / name
            if f.exists() and f.stat().st_size > 0:
                df = pd.read_csv(f, dtype={"class": str, "rel_path": str}, keep_default_na=False,
                                 na_values=["NaN"], low_memory=False)
                df["repo_key"] = k
                df["ck_source"] = name
                parts.append(df)
    if not parts:
        return pd.DataFrame(columns=["repo_key", "class", "rel_path", "type"])
    ck = pd.concat(parts, ignore_index=True)
    # The supplement re-measures files CK lost to duplicate-FQN overwrites; it never duplicates a
    # (class, file) pair that the main run kept.
    ck = ck.drop_duplicates(["repo_key", "class", "rel_path"], keep="first")
    text_cols = {"repo_key", "class", "rel_path", "type", "ck_source"}
    for c in ck.columns.difference(list(text_cols)):
        ck[c] = pd.to_numeric(ck[c], errors="coerce")  # CK leaves some cells blank (e.g. lcom*)
    ck["class_norm"] = ck["class"].str.replace("$", ".", regex=False)
    ck["simple"] = ck["class_norm"].str.split(".").str[-1]
    return ck


def load_relational(cfg: dict, keys: list[str]) -> pd.DataFrame:
    root = p(cfg["paths"]["relational"])
    parts = []
    for k in keys:
        f = root / f"{k}.csv"
        if f.exists() and f.stat().st_size > 0:
            df = pd.read_csv(f, dtype={"class_fqn": str, "file": str})
            df["repo_key"] = k
            parts.append(df)
    if not parts:
        return pd.DataFrame(columns=["repo_key", "file", "class_fqn", "simple_name", "atfd", "fdp", "parse_error"])
    return pd.concat(parts, ignore_index=True)


def match_ck(s, ck_repo: pd.DataFrame) -> tuple[pd.Series | None, str | None, str | None]:
    """Return (ck_row, key_used, unmatched_reason)."""
    by_name = ck_repo[ck_repo["class_norm"] == s.code_name]
    if len(by_name) > 1:
        on_path = by_name[by_name["rel_path"] == s.path]
        if len(on_path) == 1:
            return on_path.iloc[0], "name+path", None
        return None, None, "ambiguous"
    if len(by_name) == 1:
        row = by_name.iloc[0]
        return row, ("name" if row["rel_path"] == s.path else "name_path_differs"), None

    # '$' in a top-level class's own name: the normalization above broke it, so try CK's raw name.
    by_raw = ck_repo[ck_repo["class"] == s.code_name]
    if len(by_raw) == 1:
        return by_raw.iloc[0], "raw_name", None
    if len(by_raw) > 1:
        on_path = by_raw[by_raw["rel_path"] == s.path]
        return (on_path.iloc[0], "raw_name+path", None) if len(on_path) == 1 else (None, None, "ambiguous")

    simple = s.code_name.split(".")[-1]
    by_path = ck_repo[ck_repo["rel_path"].str.endswith(s.path) & (ck_repo["simple"] == simple)]
    if len(by_path) == 1:
        return by_path.iloc[0], "path_suffix", None
    if len(by_path) > 1:
        return None, None, "ambiguous"
    if not ck_repo["rel_path"].str.endswith(s.path).any():
        return None, None, "file_missing"
    return None, None, "name_mismatch"


def match_relational(s, rel_repo: pd.DataFrame) -> pd.Series | None:
    in_file = rel_repo[rel_repo["file"] == s.path]
    hit = in_file[in_file["class_fqn"] == s.code_name]
    if len(hit) == 1:
        return hit.iloc[0]
    hit = in_file[in_file["simple_name"] == s.code_name.split(".")[-1]]
    return hit.iloc[0] if len(hit) == 1 else None


def main(cfg: dict | None = None) -> None:
    cfg = cfg or load_config()
    samples = unique_samples(load_samples(cfg))
    samples = samples[samples["type"] == "class"].reset_index(drop=True)
    manifest = read_json(p(cfg["fetch"]["manifest_file"]), default={})
    extracted = sorted(k for k, v in manifest.items() if v.get("status") == "extracted")
    ck = load_ck_classes(cfg, extracted)
    rel = load_relational(cfg, extracted)
    ck_by_repo = {k: g for k, g in ck.groupby("repo_key")}
    rel_by_repo = {k: g for k, g in rel.groupby("repo_key")}
    failed_by_repo = failed_units(cfg)
    snip_dir = p(cfg["paths"]["snippets"])
    metric_cols = [c for c in ck.columns if c not in ("repo_key", "class", "rel_path", "class_norm",
                                                       "simple", "ck_source")]

    rows, unmatched = [], []
    for s in samples.itertuples():
        status = manifest.get(s.repo_key, {}).get("status", "pending")
        if status != "extracted":
            unmatched.append({"sample_id": s.sample_id, "repo_key": s.repo_key,
                              "reason": "repo_unresolved" if status == "failed" else "repo_not_yet_extracted",
                              "detail": manifest.get(s.repo_key, {}).get("reason")})
            continue
        ck_row, key, reason = match_ck(s, ck_by_repo.get(s.repo_key, ck.iloc[0:0]))
        if ck_row is None:
            detail = None
            if reason == "file_missing":
                unit = ck_failed_unit(s.path, failed_by_repo.get(s.repo_key, []))
                if unit is not None:
                    reason, detail = "ck_failed", f"CK {unit.get('reason')} on {unit['path']}"
            unmatched.append({"sample_id": s.sample_id, "repo_key": s.repo_key, "reason": reason,
                              "detail": detail, "code_name": s.code_name, "path": s.path})
            continue
        r = match_relational(s, rel_by_repo.get(s.repo_key, rel.iloc[0:0]))
        row = {c: getattr(s, c) for c in ID_COLS}
        row.update({"working_link": working_link(s, cfg["fetch"]),
                    "join_key": key, "ck_class": ck_row["class"], "ck_rel_path": ck_row["rel_path"],
                    "ck_source": ck_row["ck_source"],
                    "snippet_lines": s.end_line - s.start_line + 1,
                    "snippet_exists": (snip_dir / f"{s.sample_id}.java").exists()})
        row.update({f"ck_{c}": ck_row[c] for c in metric_cols})
        row.update({c: (r[c] if r is not None and c in r else None) for c in REL_FEATURES})
        row.update({
                    "relational_parse_error": r["parse_error"] if r is not None else "no_relational_row"})
        rows.append(row)

    ds = pd.DataFrame(rows)
    um = pd.DataFrame(unmatched, columns=["sample_id", "repo_key", "reason", "detail", "code_name", "path"])
    out = p(cfg["paths"]["class_dataset"])
    out.parent.mkdir(parents=True, exist_ok=True)
    ds.to_parquet(out, index=False)

    n = len(samples)
    n_eligible = n - int(um["reason"].isin(["repo_unresolved", "repo_not_yet_extracted"]).sum())
    report = {
        "n_class_samples": n,
        "n_matched": len(ds),
        "match_rate_overall": round(len(ds) / n, 4) if n else None,
        "match_rate_on_extracted_repos": round(len(ds) / n_eligible, 4) if n_eligible else None,
        "n_repos_extracted": len(extracted),
        "join_key_counts": ds["join_key"].value_counts().to_dict() if len(ds) else {},
        "unmatched_by_reason": um["reason"].value_counts().to_dict(),
        "atfd_missing": int(ds["atfd"].isna().sum()) if len(ds) else 0,
        "snippet_missing": int((~ds["snippet_exists"]).sum()) if len(ds) else 0,
        "ck_nested_name_format_examples": ck.loc[ck["class"].str.contains("$", regex=False), "class"].head(5).tolist(),
        "unmatched": um.to_dict("records"),
    }
    if len(ds) and {"ck_totalMethodsQty", "ck_tcc"} <= set(ds.columns):
        # Cohesion sanity: TCC stuck at 0 for multi-method classes means CK ran without field usage.
        multi = ds[pd.to_numeric(ds["ck_totalMethodsQty"], errors="coerce") >= 3]
        tcc = pd.to_numeric(multi["ck_tcc"], errors="coerce")
        report["tcc_sanity"] = {"n_classes_3plus_methods": int(len(multi)),
                                "share_tcc_zero": round(float((tcc == 0).mean()), 3) if len(multi) else None,
                                "share_tcc_missing_or_neg": round(float(((tcc < 0) | tcc.isna()).mean()), 3) if len(multi) else None,
                                "median_tcc": float(tcc[tcc >= 0].median()) if (tcc >= 0).any() else None}
    if len(ds) > 2:
        loc = pd.to_numeric(ds["ck_loc"], errors="coerce")
        ok = loc.notna()
        report["loc_vs_snippet_lines"] = {
            "spearman": round(float(spearmanr(loc[ok], ds.loc[ok, "snippet_lines"]).statistic), 4),
            "pearson": round(float(pearsonr(loc[ok], ds.loc[ok, "snippet_lines"]).statistic), 4),
            "median_ratio_loc_over_lines": round(float((loc[ok] / ds.loc[ok, "snippet_lines"]).median()), 3),
        }
        per_repo = (samples.assign(matched=samples["sample_id"].isin(ds["sample_id"]))
                    .groupby("repo_key")["matched"].agg(["sum", "count"]))
        report["repos_below_full_match"] = {k: f"{int(v['sum'])}/{int(v['count'])}"
                                            for k, v in per_repo.iterrows()
                                            if v["sum"] < v["count"] and k in extracted}
    jc = cfg["join"]
    lc_file = p(jc["link_check"]["file"]) if "link_check" in jc else None
    if lc_file is not None and lc_file.exists() and len(ds):
        lc = pd.read_csv(lc_file, dtype=str).drop_duplicates("sample_id", keep="last")
        cur = ds[["sample_id", "working_link"]].merge(lc, on=["sample_id", "working_link"], how="left")
        report["all_links_checked"] = int(cur["status"].notna().sum())
        report["all_links_ok"] = f"{int((cur['status'] == '200').sum())}/{len(cur)}"
        report["all_links_failed"] = cur[cur["status"].notna() & (cur["status"] != "200")] \
            [["sample_id", "working_link", "status"]].to_dict("records")
    write_json(p(cfg["paths"]["join_report"]), report)

    if len(ds):
        check_path = p(cfg["paths"]["join_manual_check"])
        previous = []
        if check_path.exists():
            previous = pd.read_csv(check_path, dtype={"sample_id": str})["sample_id"].tolist()
            shutil.copy2(check_path, check_path.with_name(f"{check_path.stem}.prev_{timestamp()}.csv"))
        check = pick_manual_check(ds, previous, jc["manual_check_rows"], jc["manual_check_seed"]).copy()
        if jc.get("check_links", False):
            check["link_http_status"] = [check_one(u, jc["link_check"]) for u in check["working_link"]]
            ok = check["link_http_status"].astype(str) == "200"
            report["manual_check_links_ok"] = f"{int(ok.sum())}/{len(check)}"
            report["manual_check_links_failed"] = check.loc[~ok, ["sample_id", "working_link",
                                                                  "link_http_status"]].to_dict("records")
            write_json(p(cfg["paths"]["join_report"]), report)
        cols = ["sample_id", "code_name", "ck_class", "path", "ck_rel_path", "join_key", "ck_loc",
                "snippet_lines", "start_line", "end_line", "atfd", "link", "working_link"]
        check[cols + (["link_http_status"] if "link_http_status" in check else [])] \
            .to_csv(check_path, index=False)
        print(f"manual check: {int(check['sample_id'].isin(previous).sum())} rows kept from the "
              f"previous check file, {len(check)} total")
        if "manual_check_links_ok" in report:
            print("working_link HTTP 200:", report["manual_check_links_ok"])

    print(f"matched {len(ds)}/{n} class samples "
          f"({report['match_rate_on_extracted_repos']} of samples in extracted repos)")
    print("join keys:", report["join_key_counts"])
    print("unmatched:", report["unmatched_by_reason"])
    if "loc_vs_snippet_lines" in report:
        print("CK loc vs snippet lines:", report["loc_vs_snippet_lines"])
    print(f"wrote {out}\n      {p(cfg['paths']['join_report'])}\n      {p(cfg['paths']['join_manual_check'])}")
    print("\nGATE (Step 3): STOP. Check the 30 rows in join_manual_check.csv against each `link`. "
          "No class-level modelling until you confirm the join is correct.")


if __name__ == "__main__":
    main()
