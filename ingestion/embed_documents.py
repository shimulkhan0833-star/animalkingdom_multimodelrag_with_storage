"""Embed prepared PDF records with Gemini and upsert dense vectors to Pinecone."""
import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path

from ingestion.generate_captions import retry_delay, safe_error
from ingestion.prepare_documents import evidence_path


def save_checkpoint(path, state):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def checked_vector(values, dimensions):
    if len(values) != dimensions or not all(math.isfinite(v) for v in values):
        raise ValueError("Embedding dimensions or values are invalid")
    return values


def process_records(records, state, checkpoint, embed, upload, interval=15, limit=None):
    """Cache vectors before upload; mark complete only after successful upsert."""
    pending = [r for r in records if not state["records"].get(r["record_id"], {}).get("uploaded")]
    if limit:
        pending = pending[:limit]
    last_request = None
    for record in pending:
        record_id = record["record_id"]
        cached = state["records"].get(record_id)
        if cached is None:
            if last_request is not None:
                time.sleep(max(0, interval - (time.monotonic() - last_request)))
            last_request = time.monotonic()
            values = checked_vector(embed(record), state["config"]["dimensions"])
            cached = {"values": values, "uploaded": False}
            state["records"][record_id] = cached
            save_checkpoint(checkpoint, state)
        upload(record, cached["values"])
        cached["uploaded"] = True
        save_checkpoint(checkpoint, state)
        completed = sum(r["uploaded"] for r in state["records"].values())
        print(f"Uploaded {record_id} ({completed}/{len(records)})", flush=True)


def retry(operation):
    """Retry transient failures, but stop rather than sleep for hours."""
    import httpx
    for attempt in range(3):
        try:
            return operation()
        except Exception as exc:
            response = getattr(exc, "response", None)
            code = getattr(exc, "status_code", None) or getattr(exc, "code", None) or (response.status_code if response is not None else None)
            network_error = isinstance(exc, httpx.TransportError) or type(exc).__name__ in ("APIConnectionError", "APITimeoutError")
            if attempt == 2 or (code not in (429, 500, 502, 503, 504) and not network_error):
                raise RuntimeError(safe_error(exc)) from None
            wait = retry_delay(exc, attempt)
            if wait > 120:
                raise RuntimeError(f"Quota retry delay is {wait:.0f}s. Resume later; checkpoint retained.") from None
            print(f"Retrying in {wait:.1f}s", flush=True)
            time.sleep(wait)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prepared", nargs="?", type=Path)
    parser.add_argument("--evidence-root", type=Path)
    parser.add_argument("--index", default=None)
    parser.add_argument("--namespace", default=None)
    parser.add_argument("--dimensions", type=int, choices=(768, 1536, 3072), default=1536)
    parser.add_argument("--interval", type=float, default=15)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        if args.interval < 0 or (args.limit is not None and args.limit < 1):
            raise ValueError("Interval must be nonnegative; limit must be positive")
        if args.prepared is None:
            matches = list((Path(__file__).resolve().parents[1] / "storage").glob("*/prepared_documents_captioned.jsonl"))
            if len(matches) != 1:
                raise ValueError("Specify one captioned prepared JSONL file")
            args.prepared = matches[0]
        prepared = args.prepared.resolve()
        root = args.evidence_root.resolve() if args.evidence_root else prepared.parent
        records = [json.loads(line) for line in prepared.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not records or len({r["record_id"] for r in records}) != len(records):
            raise ValueError("Records must be nonempty and have unique IDs")
        source_hash = hashlib.sha256(prepared.read_bytes())
        for record in records:
            item = record["embedding_input"]
            if "image_path" in item:
                image = root / evidence_path(root, item["image_path"])
                if image.stat().st_size > 14_000_000:
                    raise ValueError("Image exceeds inline request limit")
                source_hash.update(image.read_bytes())
            elif not item.get("text", "").strip():
                raise ValueError("Empty text embedding input")
        if args.dry_run:
            print(f"Validated {len(records)} records; model gemini-embedding-2, dimensions {args.dimensions}. No API calls.")
            return
        import httpx
        from dotenv import load_dotenv
        from google import genai
        from google.genai import types
        load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)
        google_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        pinecone_key = os.getenv("PINECONE_API_KEY")
        if not google_key or not pinecone_key:
            raise ValueError("Set GOOGLE_API_KEY and PINECONE_API_KEY in .env")
        index_name = args.index or os.getenv("PINECONE_INDEX", "pdf-multimodal")
        namespace = args.namespace or os.getenv("PINECONE_NAMESPACE", "gemini-embedding-2-1536" if args.dimensions == 1536 else f"gemini-embedding-2-{args.dimensions}")
        config = {"model": "gemini-embedding-2", "dimensions": args.dimensions,
                  "index": index_name, "namespace": namespace, "source_hash": source_hash.hexdigest(),
                  "account_key_hash": hashlib.sha256(pinecone_key.encode()).hexdigest()}
        checkpoint = args.checkpoint.resolve() if args.checkpoint else root / "embedding_checkpoint.json"
        if checkpoint == prepared or checkpoint == root / "manifest.json":
            raise ValueError("Checkpoint cannot overwrite source data")
        state = json.loads(checkpoint.read_text(encoding="utf-8")) if checkpoint.exists() else {"config": config, "records": {}}
        if {k: v for k, v in state["config"].items() if k != "host"} != config:
            raise ValueError("Checkpoint inputs or target differ; use a new --checkpoint")
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        with httpx.Client(timeout=60) as http:
            headers = {"Api-Key": pinecone_key, "X-Pinecone-API-Version": "2025-10"}

            def request(method, url, body=None):
                response = http.request(method, url, headers=headers, json=body)
                response.raise_for_status()
                return response.json()

            url = f"https://api.pinecone.io/indexes/{index_name}"
            response = http.get(url, headers=headers)
            if response.status_code == 404:
                retry(lambda: request("POST", "https://api.pinecone.io/indexes", {
                    "name": index_name, "dimension": args.dimensions, "metric": "cosine",
                    "spec": {"serverless": {"cloud": "aws", "region": "us-east-1"}}}))
            else:
                response.raise_for_status()
            deadline = time.monotonic() + 120
            while True:
                description = retry(lambda: request("GET", url))
                if description["dimension"] != args.dimensions or description["metric"] != "cosine":
                    raise ValueError("Pinecone index must match dimensions and cosine metric")
                if description["status"]["ready"]:
                    break
                if time.monotonic() > deadline:
                    raise ValueError("Index is not ready; rerun later")
                time.sleep(5)
            host = description["host"]
            # Checkpoint identifies the index host to detect deletion/recreation.
            if "host" in state["config"] and state["config"]["host"] != host:
                raise ValueError("Index host changed; use a new checkpoint")
            state["config"]["host"] = host
            with genai.Client(api_key=google_key, http_options=types.HttpOptions(timeout=60000)) as client:
                def embed(record):
                    item = record["embedding_input"]
                    parts = []
                    if "image_path" in item:
                        parts.append(types.Part.from_bytes(data=(root / item["image_path"]).read_bytes(), mime_type="image/png"))
                    text = item.get("text", "")
                    if text:
                        if not parts:
                            text = f"title: {record['source_filename']} | text: {text}"
                        parts.append(types.Part.from_text(text=text))
                    result = retry(lambda: client.models.embed_content(
                        model=config["model"], contents=types.Content(parts=parts),
                        config=types.EmbedContentConfig(output_dimensionality=args.dimensions)))
                    if not result.embeddings or len(result.embeddings) != 1:
                        raise ValueError("Expected one combined embedding per record")
                    return result.embeddings[0].values

                def upload(record, values):
                    metadata = {k: record[k] for k in ("document_id", "source_filename", "page_number", "content_type")}
                    metadata.update(embedding_model=config["model"], dimensions=args.dimensions)
                    for key in ("image_path", "table_path", "csv_path", "text_path", "page_image_path"):
                        if key in record:
                            metadata[key] = record[key]
                    metadata["text"] = record["embedding_input"].get("text", "")[:6000]
                    retry(lambda: request("POST", f"https://{host}/vectors/upsert", {
                        "namespace": namespace, "vectors": [{"id": record["record_id"], "values": values, "metadata": metadata}]}))

                print(f"Target: {index_name}/{namespace}", flush=True)
                process_records(records, state, checkpoint, embed, upload, args.interval, args.limit)
        print(f"Checkpoint saved: {checkpoint}")
    except KeyboardInterrupt:
        parser.exit(1, "Stopped. Saved embeddings and uploads will resume next run.\n")
    except Exception as exc:
        parser.exit(1, f"Embedding failed: {safe_error(exc)}\nCheckpoint retained.\n")


if __name__ == "__main__":
    main()
