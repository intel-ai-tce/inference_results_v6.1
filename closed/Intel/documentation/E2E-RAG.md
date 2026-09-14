# Intel E2E-RAG submission

Two E2E-RAG benchmarks on a single 1-node 2-socket Xeon 6787P system:

- **e2e-rag-db** (Offline) -- data setup: parse the frozen Wikipedia HTML corpus,
  chunk it, embed passages with e5-base-v2, build a FAISS-HNSW index. CPU only.
- **e2e-rag-qna** (Offline) -- multi-hop question answering over that index:
  iterative retrieve -> grade -> check-sufficiency -> answer with query
  decomposition. gpt-oss-120b on 4x Intel Arc Pro B70; gpt-oss-20b grader,
  embedding and reranking on CPU.

See `src/<benchmark>/README.md` for each implementation and
`results/1-node-2S-Xeon6787P/<benchmark>/Offline/README.md` for reproduction steps.
