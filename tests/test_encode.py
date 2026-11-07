"""Step 9 encoding safeguards, on a tiny random RoBERTa + word-level tokenizer built in memory (no
download, no CodeBERT): truncation, padding/ordering invariance, snippet statuses, resume, refusal to
mix or overwrite, finalize + validation, weights checksum."""
import copy

import numpy as np
import pandas as pd
import pytest
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast, RobertaConfig, RobertaModel

from common import load_config
from phase_c_embeddings import encode as E
from phase_c_embeddings.methods import extract_methods

MAX = 8
WORDS = "class A B int f g return x y z { } ( ) ; = 1 2 public void new".split()


@pytest.fixture(scope="module")
def enc():
    import torch
    torch.manual_seed(0)
    vocab = {"<s>": 0, "<pad>": 1, "</s>": 2, "<unk>": 3, **{w: i + 4 for i, w in enumerate(WORDS)}}
    t = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    t.pre_tokenizer = pre_tokenizers.Whitespace()
    tok = PreTrainedTokenizerFast(tokenizer_object=t, cls_token="<s>", sep_token="</s>", pad_token="<pad>",
                                  unk_token="<unk>")
    model = RobertaModel(RobertaConfig(vocab_size=len(vocab), hidden_size=16, num_hidden_layers=2,
                                       num_attention_heads=2, intermediate_size=32, max_position_embeddings=MAX + 2,
                                       pad_token_id=1, type_vocab_size=1))
    return E.Encoder(model, tok, max_tokens=MAX, batch_size=2)


@pytest.fixture
def cfg(tmp_path):
    c = copy.deepcopy(load_config())
    e = c["embeddings"]
    e.update(max_tokens=MAX, batch_size=2, checkpoint_every_batches=1, determinism_check_n=3,
             stop_file=str(tmp_path / "STOP"), store_cls=True, store_method_vectors=True)
    c["paths"]["snippets"] = str(tmp_path / "snippets")
    return c


TEXTS = {11: "class A { int f ( ) { return 1 ; } }",
         12: "class B { }",
         13: "class A { int f ( ) { return x ; } void g ( ) { y = 2 ; } public int z ( ) { return 1 ; } }",
         14: "class B { int x ; }"}


def test_truncation_keeps_max_tokens_including_specials(enc):
    ids, n_full = enc.token_ids("class A { int f ( ) { return 1 ; } }")
    assert len(ids) == MAX and ids[0] == enc.bos and ids[-1] == enc.eos
    assert n_full == 13 + 2
    ids, n_full = enc.token_ids("class A")
    assert len(ids) == 4 and n_full == 4


def test_padding_and_batch_order_do_not_change_vectors(enc):
    seqs = [enc.token_ids(t)[0] for t in ("class A", "class A { int f ( ) }", "x", "return 1 ; }")]
    mean, cls = enc.encode(seqs)
    for i, s in enumerate(seqs):
        m1, c1 = enc.encode([s])
        np.testing.assert_allclose(mean[i], m1[0], atol=1e-5)
        np.testing.assert_allclose(cls[i], c1[0], atol=1e-5)
    rev_mean, _ = enc.encode(seqs[::-1])
    np.testing.assert_allclose(rev_mean[::-1], mean, atol=1e-5)


def test_snippet_statuses(cfg, tmp_path):
    d = tmp_path / "snippets"
    d.mkdir()
    (d / "1.java").write_text("public class X {\n}", encoding="utf-8")
    (d / "2.java").write_text("  \n\t", encoding="utf-8")
    (d / "3.java").write_bytes(b"class \xff {}")
    (d / "5.java").write_text("public class Big {" + " int a;" * 20 + "}", encoding="utf-8")
    texts, st = E.read_snippets(cfg, [1, 2, 3, 4, 5])
    assert dict(zip(st.sample_id, st.status)) == {1: "ok", 2: "empty", 3: "unreadable", 4: "missing", 5: "ok"}
    assert dict(zip(st.sample_id, st.tiny)) == {1: True, 2: False, 3: False, 4: False, 5: False}
    assert set(texts) == {1, 5}


def _units(enc):
    return E.make_units(enc, [(sid, -1, t) for sid, t in sorted(TEXTS.items())])


def test_resume_gives_the_same_vectors_as_one_run(enc, cfg, tmp_path):
    log = E.Log(tmp_path / "log.jsonl")
    units = _units(enc)
    budget = {"shards": 1, "seconds": {}}
    assert not E.encode_units("truncation", units, enc, tmp_path / "a", cfg, log, budget)   # paused after 1 shard
    assert E.encode_units("truncation", units, enc, tmp_path / "a", cfg, log, {"shards": None, "seconds": {}})
    assert E.encode_units("truncation", units, enc, tmp_path / "a", cfg, log, {"shards": None, "seconds": {}})  # no-op
    m = E.read_json(tmp_path / "a" / "truncation" / "manifest.json")
    d = E.load_shards(tmp_path / "a" / "truncation", m)
    assert sorted(d["sample_id"].tolist()) == sorted(TEXTS)
    one, _ = enc.encode([u.ids for u in units])
    by = {int(s): i for i, s in enumerate(d["sample_id"])}
    for i, u in enumerate(units):
        np.testing.assert_allclose(d["mean"][by[u.sample_id]], one[i], atol=1e-5)


def test_changed_inputs_and_tampered_shards_are_refused(enc, cfg, tmp_path):
    log = E.Log(tmp_path / "log.jsonl")
    units = _units(enc)
    E.encode_units("truncation", units, enc, tmp_path, cfg, log, {"shards": 1, "seconds": {}})
    changed = E.make_units(enc, [(11, -1, "class B")] + [(s, -1, t) for s, t in sorted(TEXTS.items()) if s != 11])
    with pytest.raises(SystemExit, match="inputs .* changed"):
        E.encode_units("truncation", changed, enc, tmp_path, cfg, log, {"shards": None, "seconds": {}})
    shard = next((tmp_path / "truncation" / "_shards").glob("part_*.npz"))
    shard.write_bytes(shard.read_bytes() + b"x")
    with pytest.raises(SystemExit, match="checksum"):
        E.encode_units("truncation", units, enc, tmp_path, cfg, log, {"shards": None, "seconds": {}})


def test_stop_file_pauses_cleanly(enc, cfg, tmp_path):
    (tmp_path / "STOP").write_text("")
    assert not E.encode_units("truncation", _units(enc), enc, tmp_path, cfg, E.Log(tmp_path / "l"), {"shards": None, "seconds": {}})
    assert not (tmp_path / "STOP").exists()


def test_fingerprint_mismatch_is_refused(tmp_path):
    E.check_fingerprint(tmp_path, {"id": "a", "max_tokens": 512})
    E.check_fingerprint(tmp_path, {"id": "a", "max_tokens": 512})
    with pytest.raises(SystemExit, match="different settings"):
        E.check_fingerprint(tmp_path, {"id": "b", "max_tokens": 256})


def test_final_files_are_never_rewritten(tmp_path):
    E._save_npy(tmp_path / "v.npy", np.zeros(3))
    with pytest.raises(SystemExit, match="never rewritten"):
        E._save_npy(tmp_path / "v.npy", np.ones(3))


def test_finalize_and_validate_end_to_end(enc, cfg, tmp_path):
    log = E.Log(tmp_path / "log.jsonl")
    texts = {11: "class A { int f() { return 1; } }", 12: "class B { }",
             13: "class A { int f() { return x; } void g() { y = 2; } static class B { int z() { return 1; } } }"}
    status = pd.DataFrame({"sample_id": list(texts), "status": "ok", "n_bytes": 10, "tiny": [False, True, False]})
    tu = E.make_units(enc, [(s, -1, t) for s, t in sorted(texts.items())])
    methods, reasons = E.split_all_methods(texts, with_javadoc=True)
    assert reasons == {12: "no_methods"}
    mu = E.make_units(enc, [(s, m.ordinal, m.text) for s in sorted(methods) for m in methods[s]])
    budget = {"shards": None, "seconds": {}}
    assert E.encode_units("truncation", tu, enc, tmp_path, cfg, log, budget)
    E.finalize_truncation(tmp_path, tu, cfg, status)
    assert E.encode_units("hierarchical_mean", mu, enc, tmp_path, cfg, log, budget)
    E.finalize_hierarchical(tmp_path, mu, methods, reasons, cfg)
    checks = E.validate(tmp_path, enc, tu, mu, cfg)
    assert any("fallback rows equal" in c for c in checks)
    hi = pd.read_parquet(tmp_path / "hierarchical_mean" / "index.parquet")
    assert hi.set_index("sample_id").loc[13, ["n_methods", "n_methods_nested"]].tolist() == [3, 1]
    assert hi.set_index("sample_id").loc[12, "fallback_reason"] == "no_methods"
    mv = np.load(tmp_path / "hierarchical_mean" / "method_vectors_mean.npy")
    hv = np.load(tmp_path / "hierarchical_mean" / "vectors_mean.npy")
    mi = pd.read_parquet(tmp_path / "hierarchical_mean" / "method_index.parquet")
    row13 = int(hi.loc[hi.sample_id == 13, "row"].iloc[0])
    np.testing.assert_allclose(hv[row13], mv[mi.sample_id == 13].mean(axis=0), atol=1e-6)
    ti = pd.read_parquet(tmp_path / "truncation" / "index.parquet")
    assert ti["truncated"].tolist() == [n > MAX for n in ti["n_tokens_full"]]
    with pytest.raises(SystemExit, match="never rewritten"):
        E.finalize_truncation(tmp_path, tu, cfg, status)


def test_weights_checksum_mismatch_aborts(cfg, tmp_path):
    e = cfg["embeddings"]
    e["hf_cache_dir"] = str(tmp_path / "hf")
    with pytest.raises(SystemExit, match="missing"):
        E.Encoder.from_config(cfg)
    d = tmp_path / "hf" / f"models--{e['encoder_hf_id'].replace('/', '--')}" / "snapshots" / e["encoder_revision"]
    d.mkdir(parents=True)
    for f in ("pytorch_model.bin", *E.TOKENIZER_FILES):
        (d / f).write_bytes(b"not the pinned files")
    with pytest.raises(SystemExit, match="sha256"):
        E.Encoder.from_config(cfg)


def test_method_units_follow_extraction():
    (m,) = extract_methods("class C { /** d */ int f() { return 1; } }")
    assert m.text == "/** d */ int f() { return 1; }"
