# ragdoc

A retrieval-augmented generation (RAG) pipeline for question answering over your own documents — built from first principles, with hybrid retrieval, source citations, an evaluation harness, and a clean seam between every stage.

```
$ ragdoc ingest docs/
Indexed 142 chunks -> .ragdoc_index

$ ragdoc ask "What does error XK-99 mean?"
Error code XK-99 indicates a failed temperature sensor [1]. Power-cycle the
device and contact support if the error persists.

  [1] docs/product_faq.md (chunk 0, via bm25+dense)
```

## Why this project is interesting

Most RAG demos are a thin wrapper around a vector database SDK. This one implements the retrieval stack itself, because the interesting engineering decisions in RAG live *inside* retrieval:

1. **Hybrid retrieval, not dense-only.** Dense embeddings handle paraphrase ("How do I cancel?" matches "terminating your subscription") but fail on exact tokens — error codes, SKUs, function names, section numbers. BM25 is the mirror image. ragdoc runs both and fuses them.
2. **Reciprocal Rank Fusion (RRF)** to combine the two rankings. Cosine scores and BM25 scores live on incomparable scales, so weighted sums need per-corpus calibration; RRF uses only ranks (`score = Σ 1/(k + rank)`), needs zero tuning, and naturally promotes chunks found by *both* retrievers.
3. **Sentence-aware chunking with overlap.** Fixed character windows cut sentences in half, hurting both embedding quality and the readability of context shown to the LLM. The chunker packs whole sentences and carries a configurable overlap so facts straddling a boundary survive in at least one chunk. Pathological inputs (1,000-character "sentences" from OCR or tables) are hard-split so packing always terminates.
4. **Grounded generation with citations and refusal.** The LLM is instructed to answer only from the numbered context, cite chunk numbers inline, and say "the context doesn't contain this" rather than guess — the standard anti-hallucination posture. Every answer carries machine-readable source attribution.
5. **Retrieval evaluated separately from generation.** `scripts/evaluate_retrieval.py` measures hit-rate@k and MRR against a labeled QA set, and CI fails if retrieval regresses. If the right chunk never reaches the LLM, no prompt can save the answer — so retrieval gets its own metric.
6. **Graceful degradation everywhere.** No `ANTHROPIC_API_KEY`? Answers fall back to an extractive baseline. No `sentence-transformers`? A hashed n-gram embedder keeps the pipeline (and the entire test suite) running offline with zero model downloads.

## Architecture

```
 documents (txt / md / pdf)
        │
        ▼
 ┌──────────────┐    ┌─────────────────┐
 │   Loader     │───▶│    Chunker      │   sentence-aware, overlapping
 └──────────────┘    └────────┬────────┘
                              │ chunks
              ┌───────────────┴───────────────┐
              ▼                               ▼
     ┌─────────────────┐             ┌────────────────┐
     │  VectorStore    │             │   BM25Index    │
     │ (dense, cosine) │             │   (lexical)    │
     └────────┬────────┘             └───────┬────────┘
              │ top-k candidates             │ top-k candidates
              └───────────────┬──────────────┘
                              ▼
                ┌──────────────────────────┐
                │ Reciprocal Rank Fusion   │
                └────────────┬─────────────┘
                             ▼
                ┌──────────────────────────┐
                │  Answerer (Claude API    │
                │  or extractive fallback) │──▶ answer + citations
                └──────────────────────────┘
```

Every stage is behind an interface (`Embedder` is a `Protocol`; the answerer is duck-typed), so each is independently swappable and testable. The test suite injects a deterministic offline embedder and never touches the network.

## Quick start

```bash
git clone https://github.com/vermastha/ragdoc && cd ragdoc
pip install -e ".[all]"          # or just `pip install -e .` for the minimal offline core

# 1. Index some documents
ragdoc ingest examples/sample_docs

# 2. Ask questions (uses Claude if ANTHROPIC_API_KEY is set, extractive otherwise)
export ANTHROPIC_API_KEY=sk-ant-...
ragdoc ask "What is the refund policy?"
ragdoc chat                       # interactive loop

# 3. Or run it as a REST API
uvicorn ragdoc.api:app --reload
curl -F "file=@notes.pdf" localhost:8000/ingest
curl -X POST localhost:8000/ask -H "Content-Type: application/json" \
     -d '{"question": "What is the refund policy?"}'
```

## Evaluation

```bash
python scripts/evaluate_retrieval.py examples/sample_docs examples/eval_set.json
# {"cases": 6, "hit_rate@4": 1.0, "mrr": 1.0}
```

The eval set is a JSON list of `{question, expected_source, expected_substring}` triples. CI runs it on every push and fails on regression, which turns retrieval quality into a guarded invariant rather than a vibe.

## Design decisions & tradeoffs

| Decision | Why | What I'd change at scale |
|---|---|---|
| Brute-force NumPy vector search | Exact, fast, zero deps for corpora that fit in RAM (~10⁵ chunks) | Swap `VectorStore` for FAISS HNSW or pgvector behind the same interface |
| BM25 implemented from scratch (~60 lines) | The algorithm is tiny; owning it makes scoring inspectable | Fine as-is; an inverted index for the candidate scan if corpora get large |
| RRF over weighted score fusion | No calibration needed across incomparable score scales | Learned re-ranker (cross-encoder) as a third stage |
| Character-budget, sentence-packed chunks | Token-exact budgeting requires a tokenizer dependency; characters are a good proxy | Tokenizer-aware budgets; structure-aware splitting on Markdown headings |
| BM25 rebuilt from chunk text on index load | One persisted artifact, can't drift out of sync | Persist it once rebuild time matters |
| Single-process API state | Keeps the demo honest and simple | Move index to a shared store (pgvector/Redis) for multi-worker deployments |

## Known limitations

- No incremental re-indexing or deduplication — re-ingesting a file adds duplicate chunks.
- PDF extraction is text-layer only; scanned PDFs need OCR upstream.
- No conversational memory in `chat` — each question is independent (multi-turn query rewriting is the natural next step).
- The extractive fallback is a baseline, not a product: it returns relevant sentences, it does not reason.

## Project layout

```
src/ragdoc/
├── ingestion/    loader.py (txt/md/pdf), chunker.py (sentence packing + overlap)
├── embeddings/   base.py (Protocol), sentence_transformer.py, hashing.py (offline fallback)
├── retrieval/    vector_store.py, bm25.py, hybrid.py (RRF)
├── generation/   answerer.py (Claude + extractive, grounding prompt)
├── pipeline.py   facade: ingest / save / load / ask
├── api.py        FastAPI: /ingest, /ask, /health
└── cli.py        ragdoc ingest | ask | chat
tests/            23 tests, fully offline (deterministic embedder via conftest fixture)
scripts/          evaluate_retrieval.py (hit-rate@k, MRR)
```

## Running the tests

```bash
pip install -e ".[api,dev]"
pytest -v          # 23 tests, < 1s, no network or model downloads
ruff check src tests scripts
```

## License

MIT
