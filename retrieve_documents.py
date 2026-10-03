"""Search embedded PDF records and return original local evidence."""
import argparse
import json
import os
from pathlib import Path

from embed_documents import checked_vector, retry
from generate_captions import safe_error
from prepare_documents import evidence_path


def build_filter(document_id, content_type=None):
    """Scope searches to this PDF; optionally restrict the evidence type."""
    filters = {"document_id": {"$eq": document_id}}
    if content_type:
        filters["content_type"] = {"$eq": content_type}
    return filters


def resolve_matches(matches, records, root, min_score=None):
    """Join vector hits to full local records, preserving scores and evidence paths."""
    results = []
    for match in matches:
        score = match["score"]
        if min_score is not None and score < min_score:
            continue
        record = records.get(match["id"])
        if record is None:
            raise ValueError(f"Pinecone record missing from prepared file: {match['id']}")
        paths = {}
        for key in ("image_path", "table_path", "csv_path", "text_path", "page_image_path"):
            if key in record:
                paths[key] = str(root / evidence_path(root, record[key]))
        results.append({"record_id": match["id"], "score": score,
                        "content_type": record["content_type"],
                        "source_filename": record["source_filename"],
                        "page_number": record["page_number"],
                        "citation": f"{record['source_filename']}, PDF page {record['page_number']}",
                        "text": record.get("text", record["embedding_input"].get("text", "")),
                        "caption": record.get("caption"),
                        "structured_table": record.get("structured_table"),
                        "paths": paths})
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question")
    parser.add_argument("--prepared", type=Path)
    parser.add_argument("--evidence-root", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--type", choices=("text", "image", "table"))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--min-score", type=float, help="Optional calibrated cosine cutoff; not a confidence probability")
    parser.add_argument("--output", type=Path, help="Save full results as JSON; refuses existing files")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        if not args.question.strip() or not 1 <= args.top_k <= 100:
            raise ValueError("Provide a nonempty question and top-k between 1 and 100")
        if args.min_score is not None and not -1 <= args.min_score <= 1:
            raise ValueError("Cosine cutoff must be between -1 and 1")
        if args.output and args.output.exists():
            raise ValueError("Output file already exists; choose a new path")
        if args.prepared is None:
            matches = list((Path(__file__).resolve().parent / "storage").glob("*/prepared_documents_captioned.jsonl"))
            if len(matches) != 1:
                raise ValueError("Specify --prepared when zero or multiple files exist")
            args.prepared = matches[0]
        prepared = args.prepared.resolve()
        root = args.evidence_root.resolve() if args.evidence_root else prepared.parent
        rows = [json.loads(line) for line in prepared.read_text(encoding="utf-8").splitlines() if line.strip()]
        records = {r["record_id"]: r for r in rows}
        if not rows or len(records) != len(rows) or len({r["document_id"] for r in rows}) != 1:
            raise ValueError("Prepared file must contain one PDF and unique record IDs")
        checkpoint = args.checkpoint.resolve() if args.checkpoint else root / "embedding_checkpoint.json"
        state = json.loads(checkpoint.read_text(encoding="utf-8"))
        config = state["config"]
        if config["model"] != "gemini-embedding-2":
            raise ValueError("Expected Gemini Embedding 2 checkpoint")
        filters = build_filter(rows[0]["document_id"], args.type)
        # Validate paths locally before spending an embedding request.
        for row in rows:
            for key in ("image_path", "table_path", "csv_path", "text_path", "page_image_path"):
                if key in row:
                    evidence_path(root, row[key])
        if args.dry_run:
            print(f"Validated {len(rows)} records. Target: {config['index']}/{config['namespace']}\nFilter: {filters}\nNo API calls.")
            return
        import httpx
        from dotenv import load_dotenv
        from google import genai
        from google.genai import types
        load_dotenv(Path(__file__).resolve().parent / ".env", override=False)
        google_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        pinecone_key = os.getenv("PINECONE_API_KEY")
        if not google_key or not pinecone_key:
            raise ValueError("Set GOOGLE_API_KEY and PINECONE_API_KEY in .env")
        with httpx.Client(timeout=60) as http:
            headers = {"Api-Key": pinecone_key, "X-Pinecone-API-Version": "2025-10"}

            def request(method, url, body=None):
                response = http.request(method, url, headers=headers, json=body)
                response.raise_for_status()
                return response.json()

            description = retry(lambda: request("GET", f"https://api.pinecone.io/indexes/{config['index']}"))
            if description["dimension"] != config["dimensions"] or description["metric"] != "cosine":
                raise ValueError("Index dimension/metric differs from checkpoint")
            if description["host"] != config["host"]:
                raise ValueError("Index was recreated; re-upload before retrieving")
            with genai.Client(api_key=google_key, http_options=types.HttpOptions(timeout=60000)) as client:
                embedded = retry(lambda: client.models.embed_content(
                    model=config["model"], contents=f"task: search result | query: {args.question}",
                    config=types.EmbedContentConfig(output_dimensionality=config["dimensions"])))
            if not embedded.embeddings or len(embedded.embeddings) != 1:
                raise ValueError("Expected one query embedding")
            vector = checked_vector(embedded.embeddings[0].values, config["dimensions"])
            response = retry(lambda: request("POST", f"https://{description['host']}/query", {
                "namespace": config["namespace"], "vector": vector, "topK": args.top_k,
                "includeMetadata": True, "includeValues": False, "filter": filters}))
        results = resolve_matches(response.get("matches", []), records, root, args.min_score)
        payload = {"question": args.question, "index": config["index"], "namespace": config["namespace"],
                   "results": results, "note": "Similarity matches require review; scores are not confidence probabilities."}
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8") as stream:
                json.dump(payload, stream, ensure_ascii=False, indent=2)
        if not results:
            print("No results matched the filters and score cutoff.")
        for index, result in enumerate(results, 1):
            print(f"\n{index}. {result['content_type']} | score {result['score']:.4f} | {result['citation']}")
            print(result["text"][:700])
            for kind, path in result["paths"].items():
                print(f"{kind}: {path}")
        if args.output:
            print(f"\nFull results saved to {args.output.resolve()}")
    except KeyboardInterrupt:
        parser.exit(1, "Search stopped.\n")
    except Exception as exc:
        parser.exit(1, f"Retrieval failed: {safe_error(exc)}\n")


if __name__ == "__main__":
    main()
