import unittest

import numpy as np

from common.retrieval_utils import ranked_vector_search
from common.vector_index import build_inner_product_index
from engineered.retrieval import (
    BM25Index,
    distribution_based_scores,
    keyword_overlap,
    normalize_scores,
    reciprocal_rank_fusion_score,
    tokenize,
)


class EngineeredRetrievalTests(unittest.TestCase):
    def test_bm25_scores_allowed_indices_only(self):
        documents = [
            tokenize("federal debt increased in march"),
            tokenize("international capital tables"),
            tokenize("federal budget receipts and outlays"),
        ]
        index = BM25Index(documents)

        scored = index.score(tokenize("federal budget"), allowed_indices=[1, 2])

        self.assertEqual(scored[0][0], 2)
        self.assertTrue(all(document_index in {1, 2} for document_index, _ in scored))

    def test_normalize_scores_keeps_missing_scores_zero(self):
        self.assertEqual(normalize_scores([None, 2.0, 4.0]), [0.0, 0.0, 1.0])
        self.assertEqual(normalize_scores([None, None]), [0.0, 0.0])
        self.assertEqual(normalize_scores([3.0, 3.0, None]), [1.0, 1.0, 0.0])

    def test_keyword_overlap_returns_question_term_coverage(self):
        self.assertEqual(keyword_overlap({"federal", "debt"}, {"federal", "receipts"}), 0.5)
        self.assertEqual(keyword_overlap(set(), {"federal"}), 0.0)

    def test_reciprocal_rank_fusion_scores_ranked_results(self):
        self.assertGreater(reciprocal_rank_fusion_score(1, k=60), reciprocal_rank_fusion_score(10, k=60))
        self.assertEqual(reciprocal_rank_fusion_score(None, k=60), 0.0)

    def test_distribution_based_scores_handles_missing_scores(self):
        scores = distribution_based_scores([None, 10.0, 20.0, 1000.0])

        self.assertEqual(scores[0], 0.0)
        self.assertEqual(max(scores), 1.0)
        self.assertTrue(all(0.0 <= score <= 1.0 for score in scores))

    def test_ranked_vector_search_handles_filtered_indices(self):
        embeddings = np.asarray(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.8, 0.2],
            ],
            dtype="float32",
        )
        index = build_inner_product_index(embeddings)

        ranked = ranked_vector_search(
            index=index,
            embeddings=embeddings,
            query=np.asarray([1.0, 0.0], dtype="float32"),
            allowed_indices=[1, 2],
            top_k=2,
        )

        self.assertEqual([index for index, _ in ranked], [2, 1])


if __name__ == "__main__":
    unittest.main()
