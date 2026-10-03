"""Prepare extracted PDF evidence for embedding, without calling external APIs."""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def evidence_path(root, value):
    """Resolve manifest paths and reject missing files or paths outside the extraction."""
    path = (root / value).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Evidence path escapes extraction directory: {value}")
    if not path.is_file():
        raise FileNotFoundError(f"Missing evidence: {path}")
    return path.relative_to(root).as_posix()


def text_chunks(blocks, chunk_words, overlap_words):
    """Merge blocks into bounded word chunks, retaining contributing element IDs."""
    words = []
    for block in blocks:
        words.extend((word, block["element_id"]) for word in block["text"].split())
    start = 0
    while start < len(words):
        stop = min(start + chunk_words, len(words))
        window = words[start:stop]
        yield " ".join(word for word, _ in window), list(dict.fromkeys(i for _, i in window))
        if stop == len(words):
            break
        start = stop - overlap_words


def prepare_documents(manifest_path, output=None, chunk_words=400, overlap_words=50, captions=None):
    """Write one JSONL record per text chunk, image, or table.

    File paths are relative to the source manifest directory. Image captions
    are optional verified input; nearby text remains explicitly unverified.
    """
    if chunk_words < 1 or not 0 <= overlap_words < chunk_words:
        raise ValueError("Require chunk_words > 0 and 0 <= overlap_words < chunk_words")
    manifest_path = Path(manifest_path).resolve()
    root = manifest_path.parent
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("Expected extraction manifest schema_version 1")
    captions = captions or {}
    if not isinstance(captions, dict) or not all(isinstance(v, str) for v in captions.values()):
        raise ValueError("Captions must map element IDs to strings")
    elements = manifest["elements"]
    image_ids = {e["element_id"] for e in elements if e["content_type"] == "image"}
    if set(captions) - image_ids:
        raise ValueError("Caption file contains unknown image element IDs")
    # Validate every evidence link before writing an output file.
    for element in elements:
        for key in ("file_path", "image_path", "csv_path", "page_image_path"):
            if key in element:
                evidence_path(root, element[key])
    records = []

    def base(element, record_id, source_ids):
        return {"schema_version": 1, "record_id": record_id,
                "document_id": manifest["document_id"],
                "source_filename": manifest["source_filename"],
                "page_number": element["page_number"],
                "content_type": element["content_type"],
                "source_element_ids": source_ids,
                "page_image_path": element["page_image_path"]}

    # Keep chunks within a page so citations always identify one source page.
    pages = defaultdict(list)
    for element in elements:
        if element["content_type"] == "text":
            pages[element["page_number"]].append(element)
    for number, blocks in sorted(pages.items()):
        for index, (text, ids) in enumerate(text_chunks(blocks, chunk_words, overlap_words), 1):
            record = base(blocks[0], f"{manifest['document_id']}_page_{number:04d}_chunk_{index:04d}", ids)
            record.update(text=text, embedding_input={"text": text},
                          text_path=blocks[0]["file_path"])
            records.append(record)

    for element in elements:
        kind = element["content_type"]
        if kind == "text":
            continue
        if kind not in ("image", "table"):
            raise ValueError(f"Unsupported content type: {kind}")
        record = base(element, element["element_id"], [element["element_id"]])
        record["bbox"] = element["bbox"]
        nearby = element.get("nearby_text", [])
        record["nearby_text"] = nearby
        if kind == "image":
            caption = captions.get(element["element_id"], element.get("caption"))
            caption = caption.strip() if caption else None
            record.update(image_path=element["file_path"], caption=caption,
                          caption_status="provided" if caption else "needs_review",
                          extraction_method=element.get("extraction_method"))
            # An image and its context will form ONE multimodal embedding input.
            text = f"Caption: {caption}" if caption else ""
            if nearby:
                text += "\nNearby PDF text (unverified context): " + "\n".join(nearby)
        else:
            table = json.loads((root / element["file_path"]).read_text(encoding="utf-8"))
            headers = table.get("header_names", [])
            record.update(image_path=element["image_path"], table_path=element["file_path"],
                          csv_path=element["csv_path"], header_names=headers,
                          structured_table=table, review_status="needs_review")
            # Keep exact cells in structured_table; retrieve using screenshot and headers.
            text = "Table headers: " + " | ".join(str(h or "") for h in headers)
            if nearby:
                text += "\nNearby PDF text (unverified context): " + "\n".join(nearby)
        record["embedding_input"] = {"image_path": record["image_path"], "text": text.strip()}
        records.append(record)
    records.sort(key=lambda r: (r["page_number"], r["content_type"], r["record_id"]))
    output = Path(output).resolve() if output else root / "prepared_documents.jsonl"
    if output == manifest_path:
        raise ValueError("Output cannot overwrite the extraction manifest")
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation preserves any previously prepared records.
    with output.open("x", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    return output, dict(Counter(r["content_type"] for r in records))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", nargs="?", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--chunk-words", type=int, default=400)
    parser.add_argument("--overlap-words", type=int, default=50)
    parser.add_argument("--captions", type=Path, help="JSON mapping image element IDs to verified captions")
    args = parser.parse_args()
    try:
        if args.manifest is None:
            matches = list((Path(__file__).resolve().parent / "storage").glob("*/manifest.json"))
            if len(matches) != 1:
                raise ValueError("Specify a manifest path when storage contains zero or multiple manifests")
            args.manifest = matches[0]
        captions = json.loads(args.captions.read_text(encoding="utf-8")) if args.captions else None
        output, counts = prepare_documents(args.manifest, args.output, args.chunk_words,
                                           args.overlap_words, captions)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f"Preparation failed: {exc}\n")
    print(f"Source manifest: {args.manifest.resolve()}\nSaved: {output}\nRecords: {counts}")


if __name__ == "__main__":
    main()
