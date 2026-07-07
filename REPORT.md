# Treasury RAG Report

Name: Quynh Nguyen | Recent Years Used: 2020, 2021, 2022, 2023, 2024, 2025, 2026

Github Link to the work: https://github.com/nguyetquynh111/RAG-treasury

## Part 1: The Scorecard

| Metric | Baseline (Simple) | Engineered (Improved) |
| --- | ---: | ---: |
| Hit Rate (K=5) | 60.00% | 100.00% |
| MRR | 0.38 | 1.00 |
| Groundedness | 58.33% | 85.71% |
| Factual Accuracy | 0.00% | 6.67% |
| Hallucination Rate | 41.67% | 14.29% |

## Part 2: Engineering Reflection

### 1. The Bottleneck

The main baseline failure was in finding the right data, so the bottleneck was primarily the retriever. The clearest evidence is Hit Rate@5: the baseline only found a correct source document for 60.00% of questions, with an MRR of 0.38. That means even before generation, many questions did not have the right Treasury Bulletin source in the top five chunks. Generation also struggled, especially with 0.00% factual accuracy, but the first and most measurable failure was retrieval.

### 2. The Metadata Fix

Adding year/month metadata filters made the biggest improvement in retrieval. Hit Rate@5 increased from 60.00% to 100.00%, and MRR increased from 0.38 to 1.00. This happened because the engineered retriever filtered the shared FAISS index by source year/month and source date pairs before returning the vector top-k results.

Generation metrics improved too, but less dramatically. Groundedness increased from 58.33% to 85.71%, factual accuracy rose from 0.00% to 6.67%, and hallucination rate dropped from 41.67% to 14.29%. This suggests metadata filtering gave the generator better evidence, but exact numerical reasoning and answer synthesis remained difficult.

### 3. Scaling Insight

If this pipeline scaled from the current recent-year subset to the full 1939-2025 archive, the first component likely to become too slow or fragile would be indexing and retrieval over the much larger chunk set. The current FAISS index has 7,486 chunks across 66 documents; an 80-year archive would multiply both storage and retrieval candidates substantially. Embedding all chunks would take much longer and cost more API calls, and retrieval-time filtering would become more important to avoid searching too much irrelevant text. The next likely bottleneck would be generation context quality, because many Treasury tables repeat similar headings across years and months, making near-duplicate chunks harder for the generator to disambiguate.
