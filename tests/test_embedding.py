"""Verify embedding/upload checkpoints without calling APIs."""
import tempfile
import unittest
from pathlib import Path

from ingestion.embed_documents import process_records, checked_vector


class EmbeddingTests(unittest.TestCase):
    def test_resume_after_failed_upload(self):
        calls = []
        records = [{"record_id": "frog"}]
        state = {"config": {"dimensions": 2}, "records": {}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.json"

            def embed(record):
                calls.append(record["record_id"])
                return [0.5, 0.5]

            def fail(record, values):
                raise RuntimeError("Upload failed")

            with self.assertRaises(RuntimeError):
                process_records(records, state, path, embed, fail, interval=0)
            self.assertFalse(state["records"]["frog"]["uploaded"])
            process_records(records, state, path, embed, lambda r, v: None, interval=0)
            process_records(records, state, path, embed, fail, interval=0)
            self.assertEqual(calls, ["frog"])

    def test_invalid_vector(self):
        with self.assertRaises(ValueError):
            checked_vector([float("nan")], 1)
        with self.assertRaises(ValueError):
            checked_vector([0.5], 1536)


if __name__ == "__main__":
    unittest.main()
