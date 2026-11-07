"""Step 9 — encode cached class snippets with the frozen encoder (CodeBERT, D-3), two strategies.

  truncation         whole snippet -> <s> + first max_tokens-2 tokens + </s> -> one vector
  hierarchical_mean  snippet -> methods (phase_c_embeddings.methods, D-12) -> each method truncated the
                     same way -> one vector each -> mean over the class's methods. A class with no
                     extractable method (parse error, boundary mismatch, no methods) takes its truncation
                     vector unchanged and is flagged `fallback`.
A token window becomes a vector by the masked mean of the last layer over non-padding tokens (D-11);
the <s> vector is kept alongside (`vectors_cls.npy`). Both come from the same forward pass.

Inputs are the cached snippets only (paths.snippets) — no repository or network access: the model is
loaded from embeddings.hf_cache_dir with HF_HUB_OFFLINE=1, after its weights' sha256 is checked.
Encoder frozen (eval mode, no gradients), CPU, batches of embeddings.batch_size, inputs sorted by token
length. Labels are not read for encoding; they are read only for the descriptive truncation-rate table.

Resumable: every embeddings.checkpoint_every_batches batches a shard is written and the manifest
updated; a rerun skips encoded units. The output directory carries a fingerprint (encoder, revision,
weights and tokenizer hashes, max_tokens, method extractor settings, torch / transformers versions);
a run with a different fingerprint, or with changed inputs, is refused. Final files are written once
and never rewritten.

Outputs, under <paths.embeddings>/<default_encoder>/<revision[:12]>/:
  fingerprint.json, encode_log.jsonl, method_review.{csv,txt}
  truncation/         manifest.json, _shards/, vectors_mean.npy, vectors_cls.npy, index.parquet
  hierarchical_mean/  manifest.json, _shards/, vectors_mean.npy, vectors_cls.npy, index.parquet,
                      method_vectors_mean.npy, method_vectors_cls.npy, method_index.parquet
  Row i of every vectors_*.npy is row i of the index.parquet beside it (sorted by sample_id).
  eval/results/step9_encode_<timestamp>.json   counts, truncation rates, method stats, checks, timings

    python -m phase_c_embeddings.encode --review-methods     # extraction review, no model needed
    python -m phase_c_embeddings.encode                      # both strategies (resumable)
    python -m phase_c_embeddings.encode --limit-shards 2     # stop after 2 new shards (smoke run)

Pause safely: create embeddings.stop_file (an empty file named STOP); the run finishes the current
shard and exits. Resume: run the same command again.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"          # before transformers / huggingface_hub are imported

import numpy as np
import pandas as pd

from common import RunLock, load_config, load_samples, p, portable_path, read_json, unique_samples, write_json, write_results
from phase_c_embeddings.methods import EXTRACTOR_VERSION, PARSE_ERRORS, extract_methods

STRATEGIES = ("truncation", "hierarchical_mean")
TOKENIZER_FILES = ("config.json", "vocab.json", "merges.txt", "tokenizer_config.json", "special_tokens_map.json")
SMELL_LABELS = {"blob": "y_blob", "data_class": "y_data_class"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def snapshot_dir(cfg: dict) -> Path:
    """The pinned revision's folder in the local HF cache (layout models--<org>--<name>/snapshots/<rev>).
    Only the PyTorch weights and tokenizer files were downloaded, so huggingface_hub's offline
    snapshot_download would call the snapshot incomplete; the needed files are checked here instead."""
    e = cfg["embeddings"]
    d = (p(e["hf_cache_dir"]) / f"models--{e['encoder_hf_id'].replace('/', '--')}" / "snapshots"
         / e["encoder_revision"])
    missing = [f for f in ("pytorch_model.bin", *TOKENIZER_FILES) if not (d / f).exists()]
    if missing:
        sys.exit(f"encoder files missing from {d}: {missing} — download the pinned revision first.")
    return d


def out_dir(cfg: dict) -> Path:
    e = cfg["embeddings"]
    return p(cfg["paths"]["embeddings"]) / e["default_encoder"] / e["encoder_revision"][:12]


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def class_samples(cfg: dict) -> pd.DataFrame:
    """One row per class sample (MLCQ), sorted by sample_id; must match the Step 5 labels exactly."""
    s = unique_samples(load_samples(cfg))
    s = s[s["type"] == "class"].copy()
    s["sample_id"] = s["sample_id"].astype("int64")
    labels = pd.read_parquet(p(cfg["paths"]["labels_class"]), columns=["sample_id"])
    if set(s["sample_id"]) != set(labels["sample_id"].astype("int64")):
        sys.exit("class samples in MLCQ and labels_class.parquet differ — rerun Step 5 before encoding.")
    return s.sort_values("sample_id").reset_index(drop=True)


def read_snippets(cfg: dict, ids) -> tuple[dict[int, str], pd.DataFrame]:
    """Texts of usable snippets and a status row per sample: ok | missing | unreadable | empty.
    Snippets under embeddings.tiny_snippet_bytes are encoded as usual and only flagged."""
    snip, tiny = p(cfg["paths"]["snippets"]), cfg["embeddings"]["tiny_snippet_bytes"]
    texts, rows = {}, []
    for sid in ids:
        f = snip / f"{sid}.java"
        row = {"sample_id": int(sid), "status": "ok", "n_bytes": None, "tiny": False}
        if not f.exists():
            row["status"] = "missing"
        else:
            row["n_bytes"] = f.stat().st_size
            try:
                text = f.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                row["status"] = "unreadable"
            else:
                if not text.strip():
                    row["status"] = "empty"
                else:
                    texts[int(sid)] = text
                    row["tiny"] = row["n_bytes"] < tiny
        rows.append(row)
    return texts, pd.DataFrame(rows)


def split_all_methods(texts: dict[int, str], with_javadoc: bool) -> tuple[dict[int, list], dict[int, str]]:
    """Methods per class; classes that cannot be split map to their reason instead."""
    methods, reasons = {}, {}
    for sid, text in texts.items():
        try:
            ms = extract_methods(text, with_javadoc=with_javadoc)
        except PARSE_ERRORS as e:
            reasons[sid] = f"parse_error:{type(e).__name__}"
            continue
        except ValueError as e:
            reasons[sid] = f"boundary:{str(e)[:120]}"
            continue
        if ms:
            methods[sid] = ms
        else:
            reasons[sid] = "no_methods"
    return methods, reasons


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class Encoder:
    """Frozen encoder + tokenizer. `encode` returns (masked-mean, <s>) vectors in input order."""

    def __init__(self, model, tokenizer, max_tokens: int, batch_size: int):
        import torch
        self.torch = torch
        self.model = model.eval()
        self.model.requires_grad_(False)
        self.tok = tokenizer
        self.max_tokens, self.batch_size = max_tokens, batch_size
        self.bos, self.eos = tokenizer.cls_token_id, tokenizer.sep_token_id
        self.dim = model.config.hidden_size

    @classmethod
    def from_config(cls, cfg: dict) -> tuple["Encoder", dict]:
        """Load the pinned encoder from the local cache only; abort on any integrity problem."""
        import torch
        import transformers
        from transformers import AutoModel, AutoTokenizer
        transformers.logging.set_verbosity_error()   # silences "sequence longer than 512" on full-text token counts
        e = cfg["embeddings"]
        d = snapshot_dir(cfg)
        weights_sha = sha256_file(d / "pytorch_model.bin")
        if weights_sha != e["encoder_weights_sha256"]:
            sys.exit(f"encoder weights sha256 {weights_sha} != embeddings.encoder_weights_sha256 — not encoding.")
        tok = AutoTokenizer.from_pretrained(d)
        model, info = AutoModel.from_pretrained(d, output_loading_info=True)
        bad = {k: v for k, v in info.items() if v}
        if bad:
            sys.exit(f"encoder weights did not load cleanly: {bad}")
        torch.manual_seed(cfg["seed"])
        enc = cls(model, tok, e["max_tokens"], e["batch_size"])
        probe = "class A { int f() { return 1; } }" * 200
        if enc.token_ids(probe)[0] != tok(probe, truncation=True, max_length=e["max_tokens"])["input_ids"]:
            sys.exit("token_ids() disagrees with the tokenizer's own truncation — not encoding.")
        prov = {"snapshot_dir": portable_path(d), "weights_sha256": weights_sha,
                "tokenizer_files_sha256": {f: sha256_file(d / f) for f in TOKENIZER_FILES if (d / f).exists()},
                "model_class": type(model).__name__, "hidden_size": enc.dim,
                "torch": torch.__version__, "transformers": transformers.__version__,
                "torch_num_threads": torch.get_num_threads()}
        return enc, prov

    def token_ids(self, text: str) -> tuple[list[int], int]:
        """(<s> + first max_tokens-2 tokens + </s>, token count of the whole text incl. the 2 specials)."""
        ids = self.tok(text, add_special_tokens=False)["input_ids"]
        return [self.bos] + ids[: self.max_tokens - 2] + [self.eos], len(ids) + 2

    def encode(self, seqs: list[list[int]]) -> tuple[np.ndarray, np.ndarray]:
        order = sorted(range(len(seqs)), key=lambda i: (len(seqs[i]), i))
        mean = np.empty((len(seqs), self.dim), np.float32)
        cls = np.empty_like(mean)
        with self.torch.inference_mode():
            for k in range(0, len(order), self.batch_size):
                idx = order[k:k + self.batch_size]
                b = self.tok.pad({"input_ids": [seqs[i] for i in idx]}, return_tensors="pt")
                h = self.model(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).last_hidden_state
                m = b["attention_mask"].unsqueeze(-1).to(h.dtype)
                mean[idx] = ((h * m).sum(1) / m.sum(1)).numpy()
                cls[idx] = h[:, 0].numpy()
        return mean, cls


# ---------------------------------------------------------------------------
# Units, shards, manifest
# ---------------------------------------------------------------------------

@dataclass
class Unit:
    sample_id: int
    ordinal: int          # method ordinal; -1 for a whole-snippet (truncation) unit
    ids: list
    n_tokens_full: int
    text_sha1: str


def make_units(enc: Encoder, items: list[tuple[int, int, str]]) -> list[Unit]:
    units = []
    for sid, ordinal, text in items:
        ids, n_full = enc.token_ids(text)
        units.append(Unit(sid, ordinal, ids, n_full, hashlib.sha1(text.encode("utf-8")).hexdigest()))
    return units


def unit_table_sha(units: list[Unit]) -> str:
    rows = sorted((u.sample_id, u.ordinal, u.text_sha1) for u in units)
    return hashlib.sha256(json.dumps(rows).encode()).hexdigest()


def fingerprint(cfg: dict, prov: dict) -> dict:
    e = cfg["embeddings"]
    fp = {"encoder_hf_id": e["encoder_hf_id"], "encoder_revision": e["encoder_revision"],
          "weights_sha256": prov["weights_sha256"], "tokenizer_files_sha256": prov["tokenizer_files_sha256"],
          "max_tokens": e["max_tokens"], "hierarchical_members": e["hierarchical_members"],
          "method_text": e["method_text"], "method_extractor_version": EXTRACTOR_VERSION,
          "torch": prov["torch"], "transformers": prov["transformers"]}
    fp["id"] = hashlib.sha256(json.dumps(fp, sort_keys=True).encode()).hexdigest()[:16]
    return fp


def check_fingerprint(root: Path, fp: dict) -> None:
    """First run writes fingerprint.json; any later run must match it exactly."""
    f = root / "fingerprint.json"
    held = read_json(f)
    if held is None:
        write_json(f, fp)
    elif held != fp:
        diff = {k: (held.get(k), fp.get(k)) for k in set(held) | set(fp) if held.get(k) != fp.get(k)}
        sys.exit(f"{f} was written with different settings {diff}. Not mixing vectors: move that directory "
                 f"aside (nothing is deleted automatically) or restore the settings.")


class Log:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, event: str, **kw) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"t": datetime.now().isoformat(timespec="seconds"), "event": event, **kw},
                               default=str) + "\n")


def load_shards(sdir: Path, manifest: dict) -> dict[str, np.ndarray]:
    parts = [np.load(sdir / "_shards" / s["file"]) for s in manifest["shards"]]
    keys = ("sample_id", "ordinal", "n_tokens_full", "mean", "cls")
    if not parts:
        return {"sample_id": np.zeros(0, np.int64), "ordinal": np.zeros(0, np.int32),
                "n_tokens_full": np.zeros(0, np.int32), "mean": np.zeros((0, 0), np.float32),
                "cls": np.zeros((0, 0), np.float32)}
    return {k: np.concatenate([q[k] for q in parts]) for k in keys}


def encode_units(name: str, units: list[Unit], enc: Encoder, root: Path, cfg: dict, log: Log,
                 budget: dict) -> bool:
    """Encode every unit not yet in a shard. True when all units are encoded."""
    sdir = root / name
    (sdir / "_shards").mkdir(parents=True, exist_ok=True)
    mpath = sdir / "manifest.json"
    table = unit_table_sha(units)
    manifest = read_json(mpath) or {"strategy": name, "unit_table_sha256": table, "units_total": len(units),
                                    "shards": [], "finalized": False}
    if manifest["unit_table_sha256"] != table:
        sys.exit(f"{mpath}: the inputs (snippets or extracted methods) changed since this strategy started. "
                 f"Not mixing vectors: move {sdir} aside or restore the inputs.")
    for s in manifest["shards"]:
        if sha256_file(sdir / "_shards" / s["file"]) != s["sha256"]:
            sys.exit(f"shard {s['file']} does not match its manifest checksum — not resuming.")
    done = load_shards(sdir, manifest)
    done_keys = set(zip(done["sample_id"].tolist(), done["ordinal"].tolist()))
    pending = sorted((u for u in units if (u.sample_id, u.ordinal) not in done_keys),
                     key=lambda u: (len(u.ids), u.sample_id, u.ordinal))
    if not pending:
        return True
    shard_n = cfg["embeddings"]["checkpoint_every_batches"] * enc.batch_size
    stop = p(cfg["embeddings"]["stop_file"])
    t0, n0 = time.time(), len(done_keys)
    print(f"[{name}] {len(pending)} of {len(units)} units to encode ({n0} already done)", flush=True)
    log("strategy_start", strategy=name, pending=len(pending), done=n0)
    for k in range(0, len(pending), shard_n):
        if stop.exists():
            stop.unlink(missing_ok=True)
            print(f"[{name}] STOP file found: paused after {len(done_keys) + k - n0} new units. "
                  f"Run the same command to resume.", flush=True)
            log("stopped", strategy=name)
            return False
        if budget["shards"] is not None and budget["shards"] <= 0:
            print(f"[{name}] --limit-shards reached; run again to continue.", flush=True)
            return False
        chunk = pending[k:k + shard_n]
        mean, cls = enc.encode([u.ids for u in chunk])
        fname = f"part_{len(manifest['shards']) + 1:05d}.npz"
        tmp = sdir / "_shards" / (fname + ".tmp.npz")
        np.savez(tmp, sample_id=np.array([u.sample_id for u in chunk], np.int64),
                 ordinal=np.array([u.ordinal for u in chunk], np.int32),
                 n_tokens_full=np.array([u.n_tokens_full for u in chunk], np.int32), mean=mean, cls=cls)
        os.replace(tmp, sdir / "_shards" / fname)
        manifest["shards"].append({"file": fname, "n": len(chunk), "sha256": sha256_file(sdir / "_shards" / fname),
                                   "written": datetime.now().isoformat(timespec="seconds")})
        write_json(mpath, manifest)
        if budget["shards"] is not None:
            budget["shards"] -= 1
        done_now = k + len(chunk)
        rate = done_now / (time.time() - t0)
        print(f"[{name}] {n0 + done_now}/{len(units)}  {rate:.1f} units/s  "
              f"ETA {(len(pending) - done_now) / rate / 60:.0f} min", flush=True)
        log("shard", strategy=name, file=fname, n=len(chunk))
    budget["seconds"][name] = budget["seconds"].get(name, 0) + round(time.time() - t0, 1)
    return True


def _write_once(path: Path, save) -> None:
    if path.exists():
        raise SystemExit(f"{path} already exists — final files are never rewritten.")
    tmp = path.with_name(path.name + ".tmp" + path.suffix)
    save(tmp)
    os.replace(tmp, path)


def _save_npy(path: Path, arr: np.ndarray) -> None:
    _write_once(path, lambda t: np.save(t, arr))


def _save_parquet(path: Path, df: pd.DataFrame) -> None:
    _write_once(path, lambda t: df.to_parquet(t, index=False))


def finalize_truncation(root: Path, units: list[Unit], cfg: dict, status: pd.DataFrame) -> None:
    sdir = root / "truncation"
    manifest = read_json(sdir / "manifest.json")
    d = load_shards(sdir, manifest)
    order = np.argsort(d["sample_id"], kind="stable")
    sid = d["sample_id"][order]
    assert len(sid) == len(units) and len(np.unique(sid)) == len(sid), "truncation shards: missing or duplicate samples"
    max_t = cfg["embeddings"]["max_tokens"]
    n_full = d["n_tokens_full"][order]
    idx = pd.DataFrame({"row": np.arange(len(sid)), "sample_id": sid, "n_tokens_full": n_full,
                        "n_tokens_used": np.minimum(n_full, max_t), "truncated": n_full > max_t})
    idx = idx.merge(status[["sample_id", "tiny"]], on="sample_id", how="left")
    _save_npy(sdir / "vectors_mean.npy", d["mean"][order])
    if cfg["embeddings"]["store_cls"]:
        _save_npy(sdir / "vectors_cls.npy", d["cls"][order])
    _save_parquet(sdir / "index.parquet", idx)
    manifest["finalized"] = True
    write_json(sdir / "manifest.json", manifest)


def finalize_hierarchical(root: Path, units: list[Unit], methods: dict[int, list], reasons: dict[int, str],
                          cfg: dict) -> None:
    sdir = root / "hierarchical_mean"
    manifest = read_json(sdir / "manifest.json")
    d = load_shards(sdir, manifest)
    order = np.lexsort((d["ordinal"], d["sample_id"]))
    m_sid, m_ord = d["sample_id"][order], d["ordinal"][order]
    m_mean, m_cls, m_full = d["mean"][order], d["cls"][order], d["n_tokens_full"][order]
    assert len(m_sid) == len(units) and len(set(zip(m_sid.tolist(), m_ord.tolist()))) == len(m_sid), \
        "hierarchical shards: missing or duplicate methods"
    max_t = cfg["embeddings"]["max_tokens"]
    meta = {(sid, m.ordinal): m for sid, ms in methods.items() for m in ms}
    mrows = []
    for i, (sid, o) in enumerate(zip(m_sid.tolist(), m_ord.tolist())):
        m = meta[(sid, o)]
        mrows.append({"row": i, "sample_id": sid, "ordinal": o, "name": m.name, "kind": m.kind,
                      "nested_depth": m.nested_depth, "owner": m.owner, "start_line": m.start_line,
                      "end_line": m.end_line, "has_javadoc": m.has_javadoc, "empty_body": m.empty_body,
                      "n_tokens_full": int(m_full[i]), "truncated": bool(m_full[i] > max_t)})
    mindex = pd.DataFrame(mrows)

    tdir = root / "truncation"
    t_idx = pd.read_parquet(tdir / "index.parquet")
    t_mean = np.load(tdir / "vectors_mean.npy")
    t_cls = np.load(tdir / "vectors_cls.npy") if cfg["embeddings"]["store_cls"] else None
    starts = {}
    for i, sid in enumerate(m_sid.tolist()):
        starts.setdefault(sid, [i, i])[1] = i + 1
    c_mean = np.empty_like(t_mean)
    c_cls = np.empty_like(t_mean) if t_cls is not None else None
    rows = []
    for r in t_idx.itertuples():
        sid = int(r.sample_id)
        if sid in starts:
            a, b = starts[sid]
            c_mean[r.row] = m_mean[a:b].mean(axis=0)
            if c_cls is not None:
                c_cls[r.row] = m_cls[a:b].mean(axis=0)
            g = mindex.iloc[a:b]
            rows.append({"row": r.row, "sample_id": sid, "n_methods": b - a,
                         "n_methods_nested": int((g.nested_depth > 0).sum()),
                         "n_methods_truncated": int(g.truncated.sum()), "n_methods_javadoc": int(g.has_javadoc.sum()),
                         "fallback": False, "fallback_reason": None})
        else:
            c_mean[r.row] = t_mean[r.row]
            if c_cls is not None:
                c_cls[r.row] = t_cls[r.row]
            rows.append({"row": r.row, "sample_id": sid, "n_methods": 0, "n_methods_nested": 0,
                         "n_methods_truncated": 0, "n_methods_javadoc": 0, "fallback": True,
                         "fallback_reason": reasons.get(sid, "no_methods")})
    _save_npy(sdir / "vectors_mean.npy", c_mean)
    if c_cls is not None:
        _save_npy(sdir / "vectors_cls.npy", c_cls)
    _save_parquet(sdir / "index.parquet", pd.DataFrame(rows))
    if cfg["embeddings"]["store_method_vectors"]:
        _save_npy(sdir / "method_vectors_mean.npy", m_mean)
        if cfg["embeddings"]["store_cls"]:
            _save_npy(sdir / "method_vectors_cls.npy", m_cls)
        _save_parquet(sdir / "method_index.parquet", mindex)
    manifest["finalized"] = True
    write_json(sdir / "manifest.json", manifest)


# ---------------------------------------------------------------------------
# Validation and report
# ---------------------------------------------------------------------------

def validate(root: Path, enc: Encoder, trunc_units: list[Unit], meth_units: list[Unit], cfg: dict) -> list[str]:
    """Post-run checks; raises AssertionError on the first failure. Returns the checks that passed."""
    e = cfg["embeddings"]
    checks = []
    t_idx = pd.read_parquet(root / "truncation" / "index.parquet")
    h_idx = pd.read_parquet(root / "hierarchical_mean" / "index.parquet")
    for name, idx in (("truncation", t_idx), ("hierarchical_mean", h_idx)):
        for kind in ("mean", "cls") if e["store_cls"] else ("mean",):
            v = np.load(root / name / f"vectors_{kind}.npy")
            assert v.shape == (len(idx), enc.dim), f"{name} {kind}: shape {v.shape}"
            assert np.isfinite(v).all(), f"{name} {kind}: non-finite values"
            assert (np.abs(v).sum(axis=1) > 0).all(), f"{name} {kind}: all-zero vector"
        assert idx["sample_id"].is_unique and (idx["row"] == np.arange(len(idx))).all()
        assert list(idx["sample_id"]) == sorted(u.sample_id for u in trunc_units), f"{name}: sample set"
    checks += ["one vector per encodable sample per strategy, rows aligned with index.parquet (sorted by sample_id)",
               f"vectors are {enc.dim}-dim, finite, none all-zero"]
    tv, hv = np.load(root / "truncation" / "vectors_mean.npy"), np.load(root / "hierarchical_mean" / "vectors_mean.npy")
    fb = h_idx["fallback"].to_numpy()
    assert np.array_equal(tv[fb], hv[fb]), "fallback rows differ from truncation rows"
    checks.append(f"{int(fb.sum())} hierarchical fallback rows equal their truncation rows exactly")
    rng = np.random.default_rng(cfg["seed"])
    n, tol = e["determinism_check_n"], e["determinism_tol"]
    worst = 0.0
    for name, units, vec, key_rows in (
            ("truncation", trunc_units, tv, {int(s): i for i, s in enumerate(t_idx["sample_id"])}),
            ("hierarchical_mean", meth_units,
             np.load(root / "hierarchical_mean" / "method_vectors_mean.npy") if e["store_method_vectors"] else None,
             None)):
        if vec is None or not units:
            continue
        if key_rows is None:
            mi = pd.read_parquet(root / "hierarchical_mean" / "method_index.parquet")
            key_rows = {(int(s), int(o)): int(r) for s, o, r in zip(mi.sample_id, mi.ordinal, mi.row)}
        pick = rng.choice(len(units), size=min(n, len(units)), replace=False)
        for i in pick:
            u = units[i]
            again, _ = enc.encode([u.ids])
            row = key_rows[u.sample_id] if name == "truncation" else key_rows[(u.sample_id, u.ordinal)]
            worst = max(worst, float(np.abs(again[0] - vec[row]).max()))
    assert worst <= tol, f"re-encoding one at a time differs by {worst} > {tol}"
    checks.append(f"{n} truncation + {n} method units re-encoded one at a time: max |diff| {worst:.2e} <= {tol}")
    return checks


def truncation_rates(cfg: dict, t_idx: pd.DataFrame) -> dict:
    """Share of class snippets over max_tokens, per smell x positive/negative, for the primary and the
    sensitivity label rules, over all samples and per split side. Descriptive only (drives nothing)."""
    labels = pd.read_parquet(p(cfg["paths"]["labels_class"]))
    split = read_json(p(cfg["paths"]["splits_dir"]) / "class_repo_split.json")
    df = labels.merge(t_idx[["sample_id", "truncated", "n_tokens_full"]], on="sample_id", how="inner")
    side = np.where(df["repository"].isin(split["train_repos"]), "train", "test")
    rules = ["primary"] + list(cfg["labels"].get("sensitivity_severity_rules") or [])
    out = {"overall": {"n": int(len(df)), "rate": round(float(df["truncated"].mean()), 4)}}
    for smell, col in SMELL_LABELS.items():
        for rule in rules:
            c = col if rule == "primary" else f"{col}_{rule}"
            for part, mask in (("all", np.ones(len(df), bool)), ("train", side == "train"), ("test", side == "test")):
                g = df[mask & df[c].notna().to_numpy()]
                for y, nm in ((1, "positive"), (0, "negative")):
                    h = g[g[c] == y]
                    out.setdefault(smell, {}).setdefault(rule, {}).setdefault(part, {})[nm] = {
                        "n": int(len(h)), "over_max_tokens": int(h["truncated"].sum()),
                        "rate": round(float(h["truncated"].mean()), 4) if len(h) else None,
                        "median_tokens": int(h["n_tokens_full"].median()) if len(h) else None}
    return out


def method_stats(root: Path, reasons: dict[int, str]) -> dict:
    mi = pd.read_parquet(root / "hierarchical_mean" / "method_index.parquet")
    hi = pd.read_parquet(root / "hierarchical_mean" / "index.parquet")
    per = hi.loc[~hi["fallback"], "n_methods"]
    return {"methods": int(len(mi)), "nested": int((mi.nested_depth > 0).sum()),
            "by_kind": mi["kind"].value_counts().to_dict(), "with_javadoc": int(mi.has_javadoc.sum()),
            "empty_body": int(mi.empty_body.sum()), "truncated": int(mi.truncated.sum()),
            "tokens_quantiles": {str(q): int(mi.n_tokens_full.quantile(q)) for q in (0.5, 0.9, 0.99, 1.0)},
            "per_class_median": float(per.median()) if len(per) else None, "per_class_max": int(per.max()) if len(per) else None,
            "fallback_classes": int(hi["fallback"].sum()),
            "fallback_reasons": pd.Series(list(reasons.values())).value_counts().to_dict() if reasons else {},
            "fallback_sample_ids": sorted(int(s) for s in reasons)}


# ---------------------------------------------------------------------------
# Method extraction review (before the full run; no model)
# ---------------------------------------------------------------------------

def write_review(cfg: dict) -> None:
    samples = class_samples(cfg)
    texts, status = read_snippets(cfg, samples["sample_id"])
    methods, reasons = split_all_methods(texts, cfg["embeddings"]["method_text"] == "with_javadoc")
    flat = [(sid, m) for sid in sorted(methods) for m in methods[sid]]
    rng = np.random.default_rng(cfg["seed"])
    pick = list(rng.choice(len(flat), size=min(cfg["embeddings"]["method_review_n"], len(flat)), replace=False))
    cases = {"nested type": lambda m: m.nested_depth > 0, "javadoc": lambda m: m.has_javadoc,
             "abstract / interface": lambda m: m.kind == "abstract", "constructor": lambda m: m.kind == "constructor",
             "empty body": lambda m: m.empty_body,
             "annotation on its own line": lambda m: m.text.lstrip().startswith("@") and "\n" in m.text.split("(")[0],
             "anonymous class inside": lambda m: "new " in m.text and "() {" in m.text}
    why = {int(i): "random" for i in pick}
    for case, test in cases.items():
        if any(test(flat[i][1]) for i in pick):
            continue
        cand = [i for i, (_s, m) in enumerate(flat) if test(m)]
        if cand:
            j = int(rng.choice(cand))
            why.setdefault(j, f"edge case: {case}")
    meta = samples.set_index("sample_id")
    rows, lines_out = [], []
    for k, i in enumerate(sorted(why, key=lambda i: (flat[i][0], flat[i][1].ordinal))):
        sid, m = flat[i]
        base = int(meta.loc[sid, "start_line"])
        link = meta.loc[sid, "link"].split("#")[0].rstrip("/") + f"#L{base + m.start_line - 1}-L{base + m.end_line - 1}"
        rows.append({"review_no": k + 1, "selected_as": why[i], "sample_id": sid, "ordinal": m.ordinal,
                     "owner": m.owner, "name": m.name, "kind": m.kind, "nested_depth": m.nested_depth,
                     "has_javadoc": m.has_javadoc, "file_lines": f"{base + m.start_line - 1}-{base + m.end_line - 1}",
                     "link": link, "extracted_ok": ""})
        src = texts[sid].split("\n")
        lo, hi = max(1, m.start_line - 2), min(len(src), m.end_line + 1)
        lines_out += [f"=== #{k + 1}  sample {sid}  {m.owner}.{m.name} ({m.kind}, depth {m.nested_depth})  [{why[i]}]",
                      f"    {link}", "--- extracted text:", m.text,
                      f"--- snippet lines {lo}-{hi} (> marks the extracted range):"]
        lines_out += [f"{'>' if m.start_line <= n <= m.end_line else ' '} {n:5d} | {src[n - 1]}" for n in range(lo, hi + 1)]
        lines_out.append("")
    root = out_dir(cfg)
    root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(root / "method_review.csv", index=False)
    (root / "method_review.txt").write_text("\n".join(lines_out), encoding="utf-8")
    print(f"{len(flat)} methods in {len(methods)} classes; {len(reasons)} classes without methods "
          f"({pd.Series(list(reasons.values())).value_counts().to_dict()}); snippet status "
          f"{status['status'].value_counts().to_dict()}")
    print(f"review: {len(rows)} methods -> {root / 'method_review.txt'} and method_review.csv "
          f"(fill extracted_ok with y/n)")


# ---------------------------------------------------------------------------

def run(args: argparse.Namespace, cfg: dict) -> None:
    e = cfg["embeddings"]
    if e["hierarchical_members"] != "all_with_nested" or e["method_text"] not in ("with_javadoc", "code_only") \
            or e["token_pooling"] not in ("mean", "cls"):
        sys.exit("unsupported embeddings.hierarchical_members / method_text / token_pooling value")
    root = out_dir(cfg)
    log = Log(root / "encode_log.jsonl")
    enc, prov = Encoder.from_config(cfg)
    fp = fingerprint(cfg, prov)
    check_fingerprint(root, fp)
    p(e["stop_file"]).unlink(missing_ok=True)   # a STOP left from an earlier pause must not end this run
    log("run_start", args=vars(args), fingerprint=fp["id"])

    samples = class_samples(cfg)
    texts, status = read_snippets(cfg, samples["sample_id"])
    for r in status[status["status"] != "ok"].itertuples():
        log("snippet_unusable", sample_id=r.sample_id, status=r.status)
    t0 = time.time()
    trunc_units = make_units(enc, [(sid, -1, texts[sid]) for sid in sorted(texts)])
    methods, reasons = split_all_methods(texts, e["method_text"] == "with_javadoc")
    for sid, why in reasons.items():
        log("hierarchical_fallback", sample_id=sid, reason=why)
    meth_units = make_units(enc, [(sid, m.ordinal, m.text) for sid in sorted(methods) for m in methods[sid]])
    prep_s = round(time.time() - t0, 1)
    print(f"{len(texts)} usable snippets ({len(samples) - len(texts)} unusable), {len(meth_units)} methods, "
          f"{len(reasons)} hierarchical fallbacks; tokenized in {prep_s}s", flush=True)

    budget = {"shards": args.limit_shards, "seconds": {}}
    wanted = STRATEGIES if args.strategy == "all" else (args.strategy,)
    finalized_now = []
    for name in wanted:
        units = trunc_units if name == "truncation" else meth_units
        manifest = read_json(root / name / "manifest.json") or {}
        if manifest.get("finalized"):
            print(f"[{name}] already complete — nothing to do (final files are never rewritten).")
            continue
        if not encode_units(name, units, enc, root, cfg, log, budget):
            return
        if name == "truncation":
            finalize_truncation(root, units, cfg, status)
        elif not (read_json(root / "truncation" / "manifest.json") or {}).get("finalized"):
            print("[hierarchical_mean] methods encoded; finalizing waits for truncation (fallback rows).")
            continue
        else:
            finalize_hierarchical(root, units, methods, reasons, cfg)
        finalized_now.append(name)
        log("finalized", strategy=name)
        print(f"[{name}] finalized -> {root / name}", flush=True)

    if not finalized_now or not all((read_json(root / s / "manifest.json") or {}).get("finalized") for s in STRATEGIES):
        return
    checks = validate(root, enc, trunc_units, meth_units, cfg)
    t_idx = pd.read_parquet(root / "truncation" / "index.parquet")
    files = {f"{s}/{f.name}": sha256_file(f) for s in STRATEGIES for f in sorted((root / s).glob("*.*"))
             if f.suffix in (".npy", ".parquet")}
    payload = {"step": 9, "strategies": list(STRATEGIES), "output_dir": portable_path(root),
               "fingerprint": fp, "provenance": prov,
               "settings": {k: e[k] for k in ("max_tokens", "batch_size", "token_pooling", "store_cls",
                                              "hierarchical_members", "method_text", "store_method_vectors",
                                              "checkpoint_every_batches")},
               "snippets": {"class_samples": int(len(samples)), "status": status["status"].value_counts().to_dict(),
                            "unusable_sample_ids": status.loc[status["status"] != "ok", "sample_id"].tolist(),
                            "tiny_sample_ids": status.loc[status["tiny"], "sample_id"].tolist()},
               "truncation_rate": truncation_rates(cfg, t_idx),
               "methods": method_stats(root, reasons),
               "checks": checks + ["inputs: cached snippets only; model loaded offline after a weights sha256 check",
                                   "encoder frozen: eval mode, requires_grad False, torch.inference_mode"],
               "seconds": {"tokenize_and_split": prep_s, **budget["seconds"]},
               "files_sha256": files}
    out = write_results(cfg, "step9_encode", payload, p(cfg["paths"]["splits_dir"]) / "class_repo_split.json")
    log("report", file=str(out))
    print(f"checks passed ({len(checks)}) -> {out}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--review-methods", action="store_true",
                    help="write the method-extraction review (method_review.txt/.csv) and stop; no model")
    ap.add_argument("--strategy", choices=[*STRATEGIES, "all"], default="all")
    ap.add_argument("--limit-shards", type=int, help="stop after N new shards (smoke run)")
    args = ap.parse_args(argv)
    cfg = load_config()
    if not cfg["gates"]["join_class_verified"]:
        sys.exit("Step 3 gate not passed: set gates.join_class_verified after checking join_manual_check.csv.")
    if args.review_methods:
        write_review(cfg)
        return
    root = out_dir(cfg)
    root.mkdir(parents=True, exist_ok=True)
    with RunLock(root / "_run.lock"):
        run(args, cfg)


if __name__ == "__main__":
    main()
