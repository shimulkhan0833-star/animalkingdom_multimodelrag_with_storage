"""Integration check for text, raster images, tables, and evidence paths."""
import json
import tempfile
import unittest
from pathlib import Path

import pymupdf

from extract_pdf import extract_pdf


class ExtractionTest(unittest.TestCase):
    def test_evidence_and_table_cells(self):
        """Check saved evidence, cell values, caption context, and overwrite refusal."""
        # Use a controlled fixture so checks do not depend on a particular user PDF.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "fixture.pdf"
            with pymupdf.open() as document:
                page = document.new_page()
                page.insert_text((40, 40), "Animal reference")
                # A raster image plus nearby text exercises image/context extraction.
                pixmap = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 40, 40), False)
                pixmap.clear_with(100)
                page.insert_image(pymupdf.Rect(40, 60, 140, 160), pixmap=pixmap)
                page.insert_text((40, 180), "Figure 1: Frog")
                # Draw a two-column table with known cell values for exact assertions.
                for y in (220, 250, 280):
                    page.draw_line((40, y), (240, y))
                for x in (40, 140, 240):
                    page.draw_line((x, 220), (x, 280))
                for x, y, text in ((50, 240, "Animal"), (150, 240, "Class"),
                                   (50, 270, "Frog"), (150, 270, "Amphibian")):
                    page.insert_text((x, y), text)
                document.save(source)
            output, manifest = extract_pdf(source, root / "output")
            self.assertEqual(manifest["counts"]["image"], 1)
            self.assertEqual(manifest["counts"]["table"], 1)
            image = next(e for e in manifest["elements"] if e["content_type"] == "image")
            self.assertTrue(any("Frog" in t for t in image["nearby_text"]))
            table = next(e for e in manifest["elements"] if e["content_type"] == "table")
            data = json.loads((output / table["file_path"]).read_text(encoding="utf-8"))
            self.assertEqual(data["rows"][1], ["Frog", "Amphibian"])
            # Every evidence path must resolve relative to the manifest directory.
            for element in manifest["elements"]:
                for key in ("file_path", "image_path", "csv_path", "page_image_path"):
                    if key in element:
                        self.assertTrue((output / element[key]).is_file())
            with self.assertRaises(FileExistsError):
                # Reruns must not mix existing evidence with a new extraction.
                extract_pdf(source, root / "output")


if __name__ == "__main__":
    unittest.main()
