"""Extract local PDF evidence for multimodal RAG. No API keys required."""
import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

import pymupdf


def write_json(path, value):
    """Save readable JSON while preserving non-English PDF text."""
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def nearby_text(rect, blocks):
    """Return up to three nearby text blocks as unverified caption candidates."""
    candidates = []
    for block in blocks:
        box = pymupdf.Rect(block[:4])
        # Prefer text above/below the image that overlaps it horizontally.
        # Distances are measured in PDF points (72 points per inch).
        overlap = min(rect.x1, box.x1) - max(rect.x0, box.x0)
        gap = max(box.y0 - rect.y1, rect.y0 - box.y1, 0)
        if overlap > 0 and gap <= 60 and not rect.intersects(box):
            candidates.append((gap, block[4].strip()))
    return [text for _, text in sorted(candidates)[:3]]


def extract_pdf(source, output_root, dpi=150, ocr=False, language="eng", table_strategy="lines"):
    """Save PDF evidence and return (output directory, manifest).

    Manifest paths are relative to the output directory. Bounding boxes retain
    page coordinates so retrieved content can be traced to its source region.
    OCR is optional and applies only to pages without native text.
    """
    source = Path(source).resolve()
    # Content-based IDs distinguish different PDFs even with the same filename.
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    slug = re.sub(r"[^\w-]+", "_", source.stem).strip("_") or "document"
    destination = Path(output_root).resolve() / f"{slug}_{digest[:12]}"
    # Refuse overwrites so an interrupted run cannot silently mix old and new evidence.
    destination.mkdir(parents=True, exist_ok=False)
    for folder in ("text", "images", "tables", "pages"):
        (destination / folder).mkdir()
    manifest = {"schema_version": 1, "document_id": digest, "source_filename": source.name,
                "source_sha256": digest, "dpi": dpi, "pages": [], "elements": [], "warnings": []}
    with pymupdf.open(source) as document:
        if document.needs_pass:
            raise ValueError("Password-protected PDFs must be unlocked before extraction.")
        manifest["page_count"] = len(document)
        for page in document:
            # PyMuPDF uses zero-based indices; citations use one-based PDF pages.
            number = page.number + 1
            prefix = f"page_{number:04d}"
            page_path = f"pages/{prefix}.png"
            # Full-page evidence preserves vector drawings and surrounding labels.
            page.get_pixmap(dpi=dpi, alpha=False).save(destination / page_path)
            textpage = None
            native = page.get_text().strip()
            # Avoid OCR cost and recognition errors when selectable text exists.
            if not native and ocr:
                textpage = page.get_textpage_ocr(language=language, dpi=dpi, full=True)
            if not native and not ocr:
                manifest["warnings"].append(f"Page {number}: no native text; use --ocr with Tesseract installed.")
            # Block tuples contain bbox, text, block number, and block type.
            # Type 0 means text; sort=True applies coordinate-based ordering.
            blocks = [b for b in page.get_text("blocks", textpage=textpage, sort=True)
                      if b[6] == 0 and b[4].strip()]
            full_text = "\n\n".join(b[4].strip() for b in blocks)
            (destination / "text" / f"{prefix}.txt").write_text(full_text, encoding="utf-8")
            manifest["pages"].append({"page_number": number, "file_path": page_path,
                                      "text_path": f"text/{prefix}.txt", "ocr_used": textpage is not None})

            def record(kind, index, rect, **extra):
                """Attach common source metadata to one retrievable element."""
                item = {"element_id": f"{digest[:12]}_{prefix}_{kind}_{index:04d}",
                        "document_id": digest, "source_filename": source.name,
                        "page_number": number, "content_type": kind,
                        "bbox": list(rect), "page_image_path": page_path, **extra}
                manifest["elements"].append(item)
                return item

            for index, block in enumerate(blocks, 1):
                # Store each block's text inline; file_path references the whole page.
                record("text", index, pymupdf.Rect(block[:4]), text=block[4].strip(),
                       file_path=f"text/{prefix}.txt")
            seen = set()
            for info in page.get_image_info():
                # Clip to the visible page and deduplicate equal occurrence regions.
                # '&' intersects rectangles; rounding tolerates tiny coordinate noise.
                rect = pymupdf.Rect(info["bbox"]) & page.rect
                key = tuple(round(v, 2) for v in rect)
                # PDFs often contain hundreds of tiny raster decorations/glyphs.
                if (rect.is_empty or key in seen or rect.width < 12 or rect.height < 12
                        or info["width"] < 16 or info["height"] < 16):
                    continue
                seen.add(key)
                image_path = f"images/{prefix}_image_{len(seen):04d}.png"
                # Render the occurrence to preserve masks, rotation, and visible appearance.
                page.get_pixmap(clip=rect, dpi=dpi, alpha=False).save(destination / image_path)
                record("image", len(seen), rect, file_path=image_path,
                       nearby_text=nearby_text(rect, blocks), caption=None,
                       extraction_method="rendered_raster_image_region")
            try:
                # Table detection is heuristic; its strategy is configurable in the CLI.
                tables = page.find_tables(strategy=table_strategy).tables
                for index, table in enumerate(tables, 1):
                    rect = pymupdf.Rect(table.bbox)
                    stem = f"tables/{prefix}_table_{index:04d}"
                    rows = table.extract()
                    # JSON preserves header metadata and null cells; CSV is convenient
                    # for inspection but does not preserve all structural information.
                    structured = {"rows": rows, "header_names": table.header.names,
                                  "header_external": table.header.external,
                                  "row_count": table.row_count, "column_count": table.col_count}
                    write_json(destination / f"{stem}.json", structured)
                    with (destination / f"{stem}.csv").open("w", encoding="utf-8", newline="") as stream:
                        csv.writer(stream).writerows(rows)
                    screenshot_rect = rect | pymupdf.Rect(table.header.bbox)
                    # '|' unions rectangles to include headers outside the table body.
                    page.get_pixmap(clip=screenshot_rect & page.rect, dpi=dpi, alpha=False).save(destination / f"{stem}.png")
                    record("table", index, rect, file_path=f"{stem}.json",
                           csv_path=f"{stem}.csv", image_path=f"{stem}.png",
                           nearby_text=nearby_text(rect, blocks))
            except Exception as exc:
                # Preserve other page evidence even when table processing fails.
                manifest["warnings"].append(f"Page {number}: table extraction failed: {exc}")
            print(f"Extracted page {number}/{len(document)}", flush=True)
    manifest["warnings"].append("Vector illustrations and scanned-page sub-images are not automatically segmented; use page screenshots as fallback. Nearby text is context, not a verified caption. Table detection is heuristic and requires review.")
    # Write the manifest last, after all recorded evidence has been saved.
    manifest["counts"] = {kind: sum(e["content_type"] == kind for e in manifest["elements"])
                          for kind in ("text", "image", "table")}
    write_json(destination / "manifest.json", manifest)
    return destination, manifest


def main():
    """Parse CLI options, validate input, and report extraction results."""
    parser = argparse.ArgumentParser(description=__doc__)
    # With no filename, load the sample PDF located next to this script.
    parser.add_argument("pdf", type=Path, nargs="?",
                        default=Path(__file__).resolve().parent / "animal_kingdom.pdf",
                        help="PDF to extract (default: animal_kingdom.pdf next to this script)")
    parser.add_argument("--output", type=Path, default=Path("storage"))
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--ocr", action="store_true", help="OCR pages without native text; requires Tesseract")
    parser.add_argument("--language", default="eng")
    parser.add_argument("--table-strategy", choices=("lines", "lines_strict", "text"), default="lines")
    args = parser.parse_args()
    if not args.pdf.is_file() or args.pdf.suffix.lower() != ".pdf":
        parser.error("Provide an existing PDF file.")
    if not 72 <= args.dpi <= 600:
        # Bound rendering resolution to avoid unexpectedly large page images.
        parser.error("--dpi must be between 72 and 600.")
    try:
        destination, manifest = extract_pdf(args.pdf, args.output, args.dpi, args.ocr,
                                            args.language, args.table_strategy)
    except Exception as exc:
        # A failed run may have saved files; do not reuse its output directory.
        parser.exit(1, f"Extraction failed: {exc}\nUse a new --output directory if retrying a partial run.\n")
    print(f"Saved to {destination}\nCounts: {manifest['counts']}")


if __name__ == "__main__":
    main()
