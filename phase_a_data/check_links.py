"""Check every `working_link` in class_dataset.parquet (Step 3 output) and record the HTTP status.

HEAD requests (GitHub and Software Heritage answer HEAD like GET: 200 for a file at the commit,
404 for a missing one), one host at a time with a pause between requests. 429/5xx/no-response
are retried with exponential backoff (join.link_check.backoff_s). Resumable: links already recorded as 200 are
skipped. A 200 means the file exists at that commit; the #L line range itself is not checked.

    python -m phase_a_data.check_links
Writes join.link_check.file; the next join run adds the summary to join_report.json.
"""
from __future__ import annotations

import time
from datetime import datetime
from urllib.parse import urlparse

import pandas as pd

from common import load_config, p
from phase_a_data.join_class import check_one

def main(cfg: dict | None = None) -> None:
    cfg = cfg or load_config()
    lc = cfg["join"]["link_check"]
    ds = pd.read_parquet(p(cfg["paths"]["class_dataset"]), columns=["sample_id", "working_link"])
    out = p(lc["file"])
    done = pd.read_csv(out, dtype=str) if out.exists() else pd.DataFrame(columns=["sample_id", "working_link", "status", "checked"])
    ok = set(done.loc[done["status"] == "200", "working_link"])
    todo = ds[~ds["working_link"].isin(ok)]
    print(f"{len(ds)} links, {len(ds) - len(todo)} already 200, checking {len(todo)}")
    rows = done[done["working_link"].isin(ok)].to_dict("records")
    for i, s in enumerate(todo.itertuples(), 1):
        host = urlparse(s.working_link).netloc
        st = check_one(s.working_link, lc)
        rows.append({"sample_id": s.sample_id, "working_link": s.working_link, "status": str(st),
                     "checked": datetime.now().isoformat(timespec="seconds")})
        if i % 100 == 0 or i == len(todo):
            pd.DataFrame(rows).to_csv(out, index=False)  # checkpoint
            print(f"  {i}/{len(todo)} checked", flush=True)
        time.sleep(lc["delay_s"].get(host, lc["delay_s"]["default"]))
    res = pd.DataFrame(rows, columns=["sample_id", "working_link", "status", "checked"])
    res.to_csv(out, index=False)
    print("status counts:", res["status"].value_counts().to_dict())
    bad = res[res["status"] != "200"]
    if len(bad):
        print(bad.to_string(index=False))


if __name__ == "__main__":
    main()
