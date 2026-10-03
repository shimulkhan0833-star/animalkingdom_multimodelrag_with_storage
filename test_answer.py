"""Validate answer citations and image attachments without APIs."""
import json
import unittest
from pathlib import Path

from answer_question import build_messages, validate_answer


class AnswerTests(unittest.TestCase):
    def test_citations_and_image_selection(self):
        source = {"record_id": "frog", "citation": "animals.pdf, PDF page 2", "paths": {"image_path": "frog.png"}}
        valid = json.dumps({"answer": "Frog [S1]", "source_ids": ["S1"], "image_source_ids": ["S1"]})
        result = validate_answer(valid, {"S1": source}, {"S1"})
        self.assertEqual(result["images"][0]["image_path"], "frog.png")
        with self.assertRaises(ValueError):
            validate_answer(valid, {"S1": source}, set())
        with self.assertRaises(ValueError):
            validate_answer(json.dumps({"answer": "Frog [S9]", "source_ids": ["S1"], "image_source_ids": []}), {"S1": source}, set())

    def test_image_budget(self):
        # Reuse locally saved retrieval results rather than generating new PDFs.
        path = Path("storage/animal_kingdom_5eed592711d3/frog_search_results.json")
        if not path.exists():
            self.skipTest("Local retrieval fixture unavailable")
        results = json.loads(path.read_text(encoding="utf-8"))["results"]
        messages, sources, attached = build_messages("frog", results)
        self.assertLessEqual(len(attached), 3)
        self.assertEqual(len(sources), len(results))
        self.assertEqual(messages[0]["role"], "system")


if __name__ == "__main__":
    unittest.main()
