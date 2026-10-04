"""Check quota retries and pacing without API calls or real waits."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import httpx

from ingestion.generate_captions import generate_captions


class QuotaError(Exception):
    code = 429


class CaptionTests(unittest.TestCase):
    def test_dns_failure_retries(self):
        calls = []

        def captioner(path, record):
            calls.append(record["record_id"])
            if len(calls) == 1:
                raise httpx.ConnectError("getaddrinfo failed")
            return "A frog"

        with tempfile.TemporaryDirectory() as directory, patch("ingestion.generate_captions.time.sleep"):
            root = Path(directory)
            result = generate_captions([{"record_id": "frog", "image_path": "frog.png"}],
                                       root, root / "captions.json", captioner, interval=0)
            self.assertEqual(result, {"frog": "A frog"})
            self.assertEqual(len(calls), 2)

    def test_pacing_retry_and_resume(self):
        clock = [0.0]
        requests = []

        def sleep(seconds):
            clock[0] += seconds

        def captioner(path, record):
            requests.append(clock[0])
            if len(requests) == 1:
                raise QuotaError("Please retry in 37.78s")
            return "A frog"

        records = [{"record_id": str(i), "image_path": "frog.png"} for i in range(2)]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("ingestion.generate_captions.time.sleep", side_effect=sleep), patch(
                    "ingestion.generate_captions.time.monotonic", side_effect=lambda: clock[0]):
                result = generate_captions(records, root, root / "captions.json", captioner)
                self.assertEqual(len(result), 2)
                self.assertGreaterEqual(requests[1] - requests[0], 37.78)
                self.assertGreaterEqual(requests[2] - requests[1], 15)
                generate_captions(records, root, root / "captions.json", captioner)
                self.assertEqual(len(requests), 3)


if __name__ == "__main__":
    unittest.main()
