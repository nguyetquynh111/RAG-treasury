# RAG Treasury

Local RAG for U.S. Treasury Bulletin text files and OfficeQA-style questions.

## Setup

```bash
conda create -n rag_treasury python=3.12
conda activate rag_treasury
pip install -r requirements.txt
export DEEPINFRA_API_KEY="..."
```

Edit `config/config.yaml` when needed:

```yaml
selected_years: [2020, 2021, 2022, 2023, 2024, 2025, 2026]
document_years: [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026]
index_dir: outputs/index
runs:
  baseline:
    output_dir: outputs/baseline
    metadata_enabled: false
  engineered:
    output_dir: outputs/engineered
    metadata_enabled: true
```

## Run
Build the shared index once:

```bash
python -m common.index --config config/config.yaml
```

Run QA with the shared QA entrypoint:

```bash
python -m common.qa --config config/config.yaml --mode baseline
python -m common.qa --config config/config.yaml --mode engineered
```

Evaluate with the same evaluator:

```bash
python -m evaluate --config config/config.yaml --mode baseline
python -m evaluate --config config/config.yaml --mode engineered
```

## Difference between the two runs

Both modes use the same documents, chunker, chunks, embeddings, FAISS index, generator, and evaluator.

```text
baseline   = vector top-k over the full shared index
engineered = vector top-k over the same index + year/month/source filters
```

## Metrics

Evaluation writes the same six metrics for both modes at `K=5`:

```text
hit_rate@5           1 if any correct source document appears in top 5, else 0, averaged over questions
mrr                  average reciprocal rank of the first correct source document in top 5
recall               relevant chunks found in top 5 / total relevant chunks
factual_accuracy     correct answers / total questions using LLM judge
groundedness         supported claims / total answer claims using LLM judge
hallucination_rate   fabricated claims / total answer claims using LLM judge
```

The evaluator always calls the LLM judge for answer metrics. If the judge API fails or returns invalid JSON, evaluation fails loudly instead of writing partial answer metrics.
