# Copyright (c) 2025 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

#!/usr/bin/env python3
"""Behavioral-equivalence DB check (v2) — "is this the SAME DB", not byte-identical.

Different vendors may legitimately rebuild the vector DB: HTML files and passages
in a different order, and numerically different embeddings (as long as the SAME
embedding MODEL is used). Those DBs are considered equivalent. What must match:

  * embedding model dimension + FAISS index params/algorithm (same index config)
  * the CORPUS SET: same HTML + chunking + parsing => same set of passage texts,
    regardless of order (order-independent set hash). Catches parser/chunking
    drift (e.g. shifted chunk boundaries, injected markup) but NOT reordering.
  * TOP-K RETRIEVAL behaviour against reference queries, within a tolerance
    (mean top-K URL set-overlap), since different embeddings shuffle exact ranks.

This tool does NOT check stored-vector cosine or an order-dependent corpus hash:
a regenerated DB is not byte-identical, and that is allowed by design.

    # Reference vendor writes a manifest from the reference DB:
    python3 -m ingestion.db_manifest_v2 write --db reference.db --output ref.json.gz

    # Any vendor verifies their independently-built DB against it:
    python3 -m ingestion.db_manifest_v2 verify --db vendor.db --manifest ref.json.gz

    # Or compare two DBs directly on disk (no manifest):
    python3 -m ingestion.db_manifest_v2 compare --ref reference.db --db vendor.db

Reuses DB-loading / probe / query helpers from db_manifest.py; does not modify it.
"""

import argparse
import gzip
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List

# Reuse the v1 helpers verbatim — no duplication, no changes to v1.
from ingestion.db_manifest import (
    _gather_top_k,
    _load_db,
    _load_probe_queries,
    _open_manifest,
)

MANIFEST_VERSION = "v2-behavioral"
NUM_REFERENCE_QUERIES = 50
PROBE_TOP_K = 10
# Mean fraction of the reference top-K URLs that must also appear in the
# candidate top-K (order-independent). Different embeddings shuffle exact ranks,
# so we require set-overlap, not identical lists.
DEFAULT_OVERLAP_THRESHOLD = 0.95
DEFAULT_MODEL = "intfloat_e5-base-v2/e5-base-v2"


def _resolve_model(args, manifest=None):
    """Accept either --embedding_model or --retriever_model, falling back to
    the manifest's embedding_model / retriever_model key, then DEFAULT_MODEL.
    The two names are interchangeable (same underlying model)."""
    model = getattr(args, "embedding_model", None) or getattr(args, "retriever_model", None)
    if model:
        return model
    if manifest is not None:
        model = manifest.get("embedding_model") or manifest.get("retriever_model")
        if model:
            return model
    return DEFAULT_MODEL


def _add_model_args(parser):
    """--embedding_model / --retriever_model as interchangeable aliases."""
    parser.add_argument("--embedding_model", "--retriever_model", dest="embedding_model",
                        default=None)


# ---------------------------------------------------------------------------
# Corpus-set fingerprint (order-independent)
# ---------------------------------------------------------------------------
def _corpus_set_sha256(db) -> str:
    """SHA256 over the SORTED set of per-passage text hashes.

    Order-independent: reordering HTML files/passages yields the same value.
    Sensitive to parsing/chunking: any changed passage text (whitespace,
    boundary shift, injected markup) changes exactly one member hash and thus
    the overall fingerprint. Text is hashed RAW (no normalization) so that
    parser whitespace differences are treated as real differences.
    """
    n = len(db._vector_store.index_to_docstore_id)
    per_passage = []
    for i in range(n):
        doc_id = db._vector_store.index_to_docstore_id[i]
        doc = db._vector_store.docstore.search(doc_id)
        per_passage.append(
            hashlib.sha256(doc.page_content.encode("utf-8", errors="replace")).hexdigest()
        )
    h = hashlib.sha256()
    for ph in sorted(per_passage):
        h.update(ph.encode("ascii"))
        h.update(b"\x00")
    return h.hexdigest()


def _passage_hash_set(db) -> set:
    """Set of per-passage raw-text SHA256 hashes (for overlap diagnostics)."""
    n = len(db._vector_store.index_to_docstore_id)
    out = set()
    for i in range(n):
        doc_id = db._vector_store.index_to_docstore_id[i]
        doc = db._vector_store.docstore.search(doc_id)
        out.add(hashlib.sha256(doc.page_content.encode("utf-8", errors="replace")).hexdigest())
    return out


# ---------------------------------------------------------------------------
# FAISS index params / algorithm
# ---------------------------------------------------------------------------
def _index_params(db) -> Dict:
    """Extract index type / metric / HNSW build params for equivalence check."""
    index = db._vector_store.index
    try:
        import faiss
        base = faiss.downcast_index(index) if hasattr(faiss, "downcast_index") else index
    except Exception:
        base = index

    params = {
        "class": type(base).__name__,
        "dim": int(getattr(base, "d", 0)),
        "metric_type": int(getattr(base, "metric_type", -1)),
    }
    hnsw = getattr(base, "hnsw", None)
    if hnsw is not None:
        params["efConstruction"] = int(hnsw.efConstruction)
        params["efSearch"] = int(hnsw.efSearch)
        try:
            # HNSW stores up to 2*M neighbors at level 0.
            params["M"] = int(hnsw.nb_neighbors(0)) // 2
        except Exception:
            pass
    return params


# ---------------------------------------------------------------------------
# Retrieval overlap (behavioral equivalence)
# ---------------------------------------------------------------------------
def _norm_url(u: str) -> str:
    """Normalize a doc URL so equivalent DBs match despite metadata-format
    differences, e.g. 'https://en.wikipedia.org/wiki/James_Cameron#Filmography'
    and 'en.wikipedia.org_wiki_James_Cameron#Filmography.html' -> the same key.
    Compares the underlying article (anchors dropped), not the storage format."""
    u = u.lower()
    for pre in ("https://", "http://"):
        if u.startswith(pre):
            u = u[len(pre):]
    if u.endswith(".html"):
        u = u[:-5]
    u = u.replace("en.wikipedia.org/wiki/", "").replace("en.wikipedia.org_wiki_", "")
    u = u.split("#")[0]
    return u.replace("/", "_").strip("_")


def _overlap_vs_reference(cand_top: List[Dict], ref_top_map: Dict[int, List[str]],
                          top_k: int):
    """Return (mean_overlap, top1_rate, n). Overlap = fraction of reference
    top-K URLs also present in the candidate top-K (order-independent). URLs are
    normalized so differing metadata formats don't cause false mismatches."""
    overlaps, top1 = [], 0
    n = 0
    for entry in cand_top:
        ref_urls = ref_top_map.get(entry["index"])
        if ref_urls is None:
            continue
        n += 1
        cand_urls = [_norm_url(u) for u in entry["top_k_urls"][:top_k]]
        ref_urls = [_norm_url(u) for u in ref_urls[:top_k]]
        sr, sc = set(ref_urls), set(cand_urls)
        overlaps.append(len(sr & sc) / (len(sr) or 1))
        if ref_urls and cand_urls and ref_urls[0] == cand_urls[0]:
            top1 += 1
    mean_ov = sum(overlaps) / (len(overlaps) or 1)
    return mean_ov, (top1 / n if n else 0.0), n


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------
def cmd_write(args):
    model = _resolve_model(args)
    db = _load_db(args.db, model)
    n = len(db._vector_store.index_to_docstore_id)
    print(f"[manifest-v2] DB has {n} passages, dim={db._embedding_dimension}")

    ref_queries = _load_probe_queries(args.dataset, args.num_queries)
    ref_top = _gather_top_k(db, ref_queries, PROBE_TOP_K)

    manifest = {
        "version": MANIFEST_VERSION,
        # Write both names so the manifest is portable across repos that use
        # either "embedding_model" or "retriever_model".
        "embedding_model": model,
        "retriever_model": model,
        "total_passages": n,
        "embedding_dim": db._embedding_dimension,
        "index_params": _index_params(db),
        "corpus_set_sha256": _corpus_set_sha256(db),
        "probe_top_k": PROBE_TOP_K,
        "reference_queries": ref_queries,
        "reference_top_k": ref_top,
    }
    with _open_manifest(args.output, "wt") as f:
        json.dump(manifest, f)
    print(f"[manifest-v2] wrote {args.output} "
          f"({len(ref_queries)} reference queries, top-{PROBE_TOP_K})")


def verify_manifest(db_path, manifest_path, embedding_model=None,
                    retrieval_threshold=DEFAULT_OVERLAP_THRESHOLD, **_ignored):
    """Verify a DB against a v2 behavioral manifest, returning a result dict.

    Importable counterpart to `cmd_verify` — used by
    evaluation/accuracy_eval_ingestion.py, which needs the metrics (in
    particular `retrieval_accuracy`, the headline number written to
    accuracy.txt) rather than a process exit code.

    Extra keyword arguments are accepted and ignored so callers written against
    the v1 signature (cosine_threshold, top_k_depth, retriever_model) keep
    working; v2 gates on top-K retrieval overlap, not sample-embedding cosine.

    Returns:
        dict with "passed" (bool), "failures" (list[str]), "metrics" (dict).
        On a manifest-version mismatch, "error" is set and "passed" is False.
    """
    if embedding_model is None:
        embedding_model = _ignored.get("retriever_model")

    with _open_manifest(manifest_path, "rt") as f:
        manifest = json.load(f)
    if manifest.get("version") != MANIFEST_VERSION:
        return {
            "passed": False,
            "error": (f"not a {MANIFEST_VERSION} manifest "
                      f"(got {manifest.get('version')!r})"),
            "failures": [f"manifest version {manifest.get('version')!r} "
                         f"!= {MANIFEST_VERSION}"],
            "metrics": {},
            "manifest_path": manifest_path,
        }

    # A None model falls back to the manifest's own model, then DEFAULT_MODEL.
    model = _resolve_model(argparse.Namespace(embedding_model=embedding_model),
                           manifest)
    db = _load_db(db_path, model)
    n = len(db._vector_store.index_to_docstore_id)
    failures = []

    # 1. Structural: passage count + embedding dim.
    if n != manifest["total_passages"]:
        failures.append(f"total_passages: local={n} manifest={manifest['total_passages']}")
    if db._embedding_dimension != manifest["embedding_dim"]:
        failures.append(f"embedding_dim: local={db._embedding_dimension} "
                        f"manifest={manifest['embedding_dim']}")

    # 2. Index params / algorithm.
    local_params = _index_params(db)
    if local_params != manifest["index_params"]:
        failures.append(f"index_params differ:\n"
                        f"    local    = {local_params}\n"
                        f"    manifest = {manifest['index_params']}")
    print(f"[verify-v2] index params: {local_params}")

    # 3. Corpus set (order-independent). Exact gate; overlap % on mismatch.
    local_set_sha = _corpus_set_sha256(db)
    corpus_match = local_set_sha == manifest["corpus_set_sha256"]
    if corpus_match:
        print("[verify-v2] corpus set: MATCH (same HTML + chunking + parsing)")
    else:
        local_hashes = _passage_hash_set(db)
        # The manifest doesn't store the full hash set, so we can only report
        # the local side here; compare mode gives the full overlap.
        failures.append(
            f"corpus set sha256 differs (parsing/chunking/HTML changed):\n"
            f"    local    = {local_set_sha}\n"
            f"    manifest = {manifest['corpus_set_sha256']}\n"
            f"    (local has {len(local_hashes)} distinct passages; run "
            f"`compare` against the reference DB for a passage-overlap breakdown)")

    # 4. Top-K retrieval overlap vs reference queries.
    cand_top = _gather_top_k(db, manifest["reference_queries"], manifest["probe_top_k"])
    ref_map = {r["index"]: r["top_k_urls"] for r in manifest["reference_top_k"]}
    mean_ov, top1, nq = _overlap_vs_reference(cand_top, ref_map, manifest["probe_top_k"])
    print(f"[verify-v2] retrieval vs reference ({nq} queries, top-{manifest['probe_top_k']}): "
          f"mean overlap={mean_ov:.3f} (threshold {retrieval_threshold}), "
          f"top-1 match={top1:.3f} [reported]")
    if mean_ov < retrieval_threshold:
        failures.append(f"retrieval overlap {mean_ov:.3f} < threshold "
                        f"{retrieval_threshold} — retrieval behaviour diverges "
                        f"from reference")

    return {
        "passed": not failures,
        "failures": failures,
        "metrics": {
            # Headline metric reported in accuracy.txt.
            "retrieval_accuracy": mean_ov,
            "top1_match_rate": top1,
            "num_probe_queries": nq,
            "probe_top_k": manifest["probe_top_k"],
            "total_passages": n,
            "embedding_dim": db._embedding_dimension,
            "corpus_set_match": corpus_match,
        },
        "manifest_path": manifest_path,
        "database_path": db_path,
        "embedding_model": model,
        "manifest_version": MANIFEST_VERSION,
    }


def cmd_verify(args):
    result = verify_manifest(
        args.db, args.manifest,
        embedding_model=getattr(args, "embedding_model", None),
        retrieval_threshold=args.overlap_threshold,
    )
    if result.get("error"):
        print(f"[verify-v2] ERROR: {result['error']}", file=sys.stderr)
        sys.exit(2)
    if not result["passed"]:
        print("\n[verify-v2] FAILED:")
        for f in result["failures"]:
            print(f"  - {f}")
        sys.exit(1)
    print("\n[verify-v2] OK — DB is behaviourally equivalent to the reference")


def cmd_compare(args):
    """Direct DB-vs-DB behavioral comparison, no manifest."""
    model = _resolve_model(args)
    ref = _load_db(args.ref, model)
    cand = _load_db(args.db, model)
    n_ref = len(ref._vector_store.index_to_docstore_id)
    n_cand = len(cand._vector_store.index_to_docstore_id)
    print(f"[compare] REF  {Path(args.ref).name}: {n_ref} passages")
    print(f"[compare] CAND {Path(args.db).name}: {n_cand} passages")

    failures = []

    # Structural + index params.
    if n_cand != n_ref:
        failures.append(f"passage count: REF={n_ref} CAND={n_cand}")
    rp, cp = _index_params(ref), _index_params(cand)
    if rp != cp:
        failures.append(f"index params differ:\n    REF ={rp}\n    CAND={cp}")
    print(f"[compare] index params REF ={rp}")
    print(f"[compare] index params CAND={cp}")

    # Corpus set overlap (order-independent).
    rh, ch = _passage_hash_set(ref), _passage_hash_set(cand)
    common = rh & ch
    ov_ref = len(common) / (len(rh) or 1)
    print(f"\n[compare] corpus set: {len(common)} common passages; "
          f"{100 * ov_ref:.2f}% of REF also in CAND "
          f"({len(rh - ch)} only-REF, {len(ch - rh)} only-CAND)")
    if rh != ch:
        failures.append(f"corpus set differs: only {100 * ov_ref:.2f}% of REF "
                        f"passages present in CAND (parsing/chunking/HTML changed)")

    # Retrieval overlap.
    queries = _load_probe_queries(args.dataset, args.num_queries)
    ref_top = _gather_top_k(ref, queries, args.probe_k)
    cand_top = _gather_top_k(cand, queries, args.probe_k)
    ref_map = {r["index"]: r["top_k_urls"] for r in ref_top}
    mean_ov, top1, nq = _overlap_vs_reference(cand_top, ref_map, args.probe_k)
    print(f"\n[compare] retrieval vs REF ({nq} queries, top-{args.probe_k}): "
          f"mean overlap={mean_ov:.3f} (threshold {args.overlap_threshold}), "
          f"top-1 match={top1:.3f} [reported]")
    if mean_ov < args.overlap_threshold:
        failures.append(f"retrieval overlap {mean_ov:.3f} < {args.overlap_threshold}")

    if failures:
        print("\n[compare] NOT EQUIVALENT:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print("\n[compare] OK — CAND is behaviourally equivalent to REF")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    pw = sub.add_parser("write", help="Write a behavioral-equivalence manifest.")
    pw.add_argument("--db", required=True)
    _add_model_args(pw)
    pw.add_argument("--dataset", default="data/frames_dataset.tsv")
    pw.add_argument("--num-queries", type=int, default=NUM_REFERENCE_QUERIES)
    pw.add_argument("--output", required=True)
    pw.set_defaults(func=cmd_write)

    pv = sub.add_parser("verify", help="Verify a DB against a behavioral manifest.")
    pv.add_argument("--db", required=True)
    pv.add_argument("--manifest", required=True)
    _add_model_args(pv)
    pv.add_argument("--overlap-threshold", type=float, default=DEFAULT_OVERLAP_THRESHOLD)
    pv.set_defaults(func=cmd_verify)

    pc = sub.add_parser("compare", help="Directly compare two DBs (no manifest).")
    pc.add_argument("--ref", required=True)
    pc.add_argument("--db", required=True)
    _add_model_args(pc)
    pc.add_argument("--dataset", default="data/frames_dataset.tsv")
    pc.add_argument("--num-queries", type=int, default=NUM_REFERENCE_QUERIES)
    pc.add_argument("--probe-k", type=int, default=PROBE_TOP_K)
    pc.add_argument("--overlap-threshold", type=float, default=DEFAULT_OVERLAP_THRESHOLD)
    pc.set_defaults(func=cmd_compare)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
