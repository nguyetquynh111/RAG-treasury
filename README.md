# RAG Treasury

Local retrieval-augmented QA pipelines for U.S. Treasury Bulletin text files and
OfficeQA-style question/answer rows.

## Setup

```bash
conda create -n rag_treasury python=3.12
conda activate rag_treasury
pip install -r requirements.txt
```

Add your DeepInfra API key to `.env` before running QA:

```bash
DEEPINFRA_API_KEY="..."
```

## 1. Baseline RAG

Baseline is a simple RAG system

```text
Treasury text files
        ↓
Load selected years: 2022–2025
        ↓
Fixed-size chunking
chunk_size = 512
chunk_overlap = 50
        ↓
Embedding chunks
backend = DeepInfra embeddings API
model = nvidia/llama-nemotron-embed-vl-1b-v2
        ↓
Build FAISS vector index
        ↓
For each question
        ↓
Vector retrieval top_k = 5
        ↓
Create retrieved context [S1] [S2] ...
        ↓
Send question + retrieved context to LLM
backend = DeepInfra OpenAI-compatible API
model = nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning
        ↓
Generate grounded answer with citations
        ↓
Save prediction + retrieved evidence
```

Build the baseline index:

```bash
python -m baseline.index --config config/baseline.yaml
```

Generate baseline predictions:

```bash
python -m baseline.qa --config config/baseline.yaml
```

Baseline outputs are written to `outputs/baseline/`:

```text
index.faiss
embeddings.npy
chunks.jsonl
manifest.json
predictions.csv
```

`predictions.csv` keeps the generated answer and the RAG evidence:

```text
predicted_answer
retrieved_sources
retrieved_context_ids
retrieved_context
retrieval_method
model_config
```

## 2. Engineered RAG

Engineered RAG improves the retrieval stage by using section/table-aware chunks, date-aware query handling, hybrid FAISS + BM25 retrieval, rank fusion, and optional reranking before sending the final retrieved context to DeepInfra's OpenAI-compatible API (`nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning` by default).

```text
Treasury text files
        ↓
Load selected years: 2022–2025
        ↓
Section/table-aware chunking
chunk_size = 768
chunk_overlap = 100
        ↓
Embedding chunks
backend = DeepInfra embeddings API
model = nvidia/llama-nemotron-embed-vl-1b-v2
        ↓
Build FAISS vector index
        ↓
For each question
        ↓
Detect year/month from query
        ↓
Run vector retrieval
vector_top_k = 20
        ↓
Run BM25 retrieval
bm25_top_k = 20
        ↓
Merge candidates
        ↓
Rank fusion
method = RRF
        ↓
Optional reranking
cross-encoder if available,
deterministic fallback if unavailable
        ↓
Select final_top_k = 5 chunks
        ↓
Create retrieved context [S1] [S2] ...
        ↓
Send question + retrieved context to LLM
backend = DeepInfra OpenAI-compatible API
model = nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning
        ↓
Generate grounded answer with citations
        ↓
Save prediction + retrieved evidence + retrieval logs
```

Build the engineered index:

```bash
python -m engineered.index --config config/engineered.yaml
```

Generate engineered predictions:

```bash
python -m engineered.qa --config config/engineered.yaml
```

Engineered outputs are written to `outputs/engineered/`:

```text
index.faiss
embeddings.npy
chunks.jsonl
manifest.json
predictions.csv
retrieval_logs.jsonl
```

`predictions.csv` keeps:

```text
predicted_answer
selected_years
detected_year
detected_month
retrieved_sources
retrieved_context_ids
retrieved_context
retrieval_method
model_config
```

## Comparison

Both systems share the same high-level generation flow:

```text
Question
→ retrieve evidence
→ pass evidence to DeepInfra
→ generate cited answer
→ save prediction + retrieved_context
```

They differ mainly in retrieval quality:

| Part          | Baseline RAG        | Engineered RAG              |
| ------------- | ------------------- | --------------------------- |
| Chunking      | fixed-size chunks   | section/table-aware chunks  |
| Chunk size    | 512                 | 768                         |
| Overlap       | 50                  | 100                         |
| Retrieval     | FAISS vector only   | FAISS + BM25                |
| Query parsing | answer-key date filter | detect year/month + fallback |
| Fusion        | none                | RRF                         |
| Reranking     | no                  | optional cross-encoder      |
| Final context | top 5 vector chunks | top 5 fused/reranked chunks |
| LLM           | DeepInfra `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning` | DeepInfra `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning` |


## Evaluation

The evaluation step reads an existing `predictions.csv` and `chunks.jsonl`, then computes the six required metrics:

* `hit_rate@5`: Measures whether at least one relevant chunk appears in the top 5 retrieved chunks.
* `mrr`: Measures how highly the first relevant chunk is ranked using the reciprocal of its rank (`1/rank`).
* `recall`: Measures the percentage of expected relevant snippets retrieved in the top 5 results.
* `factual_accuracy`: Measures whether the generated answer matches the gold answer from `officeqa_full.csv` (±1% tolerance for numeric answers).
* `groundedness`: Measures the percentage of factual claims supported by the retrieved chunks, using the configured DeepInfra model as an LLM judge.
* `hallucination_rate`: Measures the percentage of factual claims not supported by the retrieved chunks, using the configured DeepInfra model as an LLM judge.

Evaluate baseline outputs:

```bash
python -m evaluate --config config/baseline.yaml
```

Evaluate engineered outputs:

```bash
python -m evaluate --config config/engineered.yaml
```

If the configured DeepInfra judge call fails because the API key, endpoint, or request is invalid, retrieval and factual metrics are still computed. Judge-based metrics are written as `null`, and the failure details are saved to `judge_error.json`.

## Tests

```bash
pytest
```
