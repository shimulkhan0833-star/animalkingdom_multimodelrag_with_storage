"""Check evaluation metrics without model calls."""
import unittest
from evaluation.evaluate_rag import score_case, summarize


class EvaluationTests(unittest.TestCase):
    def test_rank_and_recall(self):
        case = {"question": "frog", "answerable": True, "expected_pages": [12, 13]}
        retrieval = {"question": "frog", "results": [{"record_id": "x", "page_number": 1}, {"record_id": "frog", "page_number": 12}]}
        metrics = score_case(case, retrieval)
        self.assertTrue(metrics["hit_at_k"])
        self.assertEqual(metrics["reciprocal_rank"], 0.5)
        self.assertEqual(metrics["expected_page_recall"], 0.5)

    def test_unsupported_and_error_denominators(self):
        metrics = score_case({"question": "weather", "answerable": False},
                             {"question": "weather", "results": []},
                             {"question": "weather", "answer": "No evidence", "sources": [], "images": []})
        self.assertTrue(metrics["unsupported_no_sources_or_images"])
        self.assertNotIn("hit_at_k", metrics)
        summary = summarize([{"status": "completed", "metrics": {"hit_at_k": True}}, {"status": "error"}])
        self.assertEqual(summary["errors"], 1)
        self.assertEqual(summary["metrics"]["hit_at_k"]["evaluated_cases"], 1)


if __name__ == "__main__":
    unittest.main()
