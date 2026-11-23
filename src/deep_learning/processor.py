"""Bounded local multimodal extraction and dataset profiling.

Optional readers are imported lazily. This module never writes to Vector RAG,
never trains a model, and never sends source content to a remote service.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import re
from pathlib import Path
from typing import Any

MAX_PDF_PAGES = 200
MAX_TABLE_ROWS = 20000
MAX_COLUMNS = 256
MAX_TEXT_CHARS = 5_000_000
MAX_IMAGE_PIXELS = 40_000_000


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _clean_text(value: str) -> str:
    value = (value or "").replace("\x00", " ")
    value = re.sub(r"[ \t]+", " ", value)
    return re.sub(r"\n{3,}", "\n\n", value).strip()


def _safe_cell(value: Any) -> Any:
    """Normalize spreadsheet cell values into strict JSON-compatible scalars."""
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            pass
    return str(value)


def _table_profile(rows: list[list[Any]]) -> dict[str, Any]:
    if not rows:
        return {"rows": 0, "columns": 0, "headers": [], "column_profiles": []}
    width = min(MAX_COLUMNS, max(len(row) for row in rows))
    headers = [str(rows[0][i]).strip() if i < len(rows[0]) else f"column_{i+1}" for i in range(width)]
    profiles = []
    for col in range(width):
        values = [row[col] if col < len(row) else None for row in rows[1:MAX_TABLE_ROWS+1]]
        present = [v for v in values if v not in (None, "")]
        numeric = []
        for value in present:
            try:
                number = float(str(value).replace(",", ""))
                if math.isfinite(number):
                    numeric.append(number)
            except (TypeError, ValueError):
                pass
        if present and len(numeric) == len(present):
            kind = "numeric"
        elif present and len({str(v) for v in present}) <= min(20, max(2, len(present)//5)):
            kind = "categorical_candidate"
        else:
            kind = "text"
        profiles.append({
            "name": headers[col], "type": kind,
            "non_empty": len(present), "missing": len(values)-len(present),
            "unique": len({str(v) for v in present}),
            "minimum": min(numeric) if numeric else None,
            "maximum": max(numeric) if numeric else None,
        })
    return {"rows": min(len(rows)-1, MAX_TABLE_ROWS), "columns": width,
            "headers": headers, "column_profiles": profiles}


def _parse_delimited(data: bytes, delimiter: str | None = None) -> list[list[Any]]:
    text = data.decode("utf-8-sig", errors="replace")
    sample = text[:8192]
    if delimiter is None:
        try:
            delimiter = csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
        except csv.Error:
            delimiter = ","
    rows = []
    for row in csv.reader(io.StringIO(text), delimiter=delimiter):
        if len(rows) >= MAX_TABLE_ROWS + 1:
            break
        rows.append([cell[:10000] for cell in row[:MAX_COLUMNS]])
    return rows


def _extract_pdf(path: Path, warnings: list[str], assets_dir: Path | None = None) -> tuple[list[dict[str, Any]], list[list[Any]], list[dict[str, Any]]]:
    pages: list[dict[str, Any]] = []
    tables: list[list[Any]] = []
    images: list[dict[str, Any]] = []
    try:
        from pypdf import PdfReader
        reader = PdfReader(str(path), strict=False)
        for index, page in enumerate(reader.pages[:MAX_PDF_PAGES]):
            text = _clean_text(page.extract_text() or "")
            pages.append({"page": index+1, "text": text, "extraction": "text" if text else "empty"})
        if len(reader.pages) > MAX_PDF_PAGES:
            warnings.append(f"PDF page limit reached ({MAX_PDF_PAGES}).")
    except Exception as exc:
        warnings.append(f"PDF text extraction unavailable: {type(exc).__name__}")
    try:
        import pdfplumber
        with pdfplumber.open(str(path)) as pdf:
            for page in pdf.pages[:MAX_PDF_PAGES]:
                for table in page.extract_tables() or []:
                    if table and len(tables) < MAX_TABLE_ROWS:
                        tables.append(table[:MAX_TABLE_ROWS])
    except ImportError:
        warnings.append("PDF table extraction unavailable: pdfplumber is not installed.")
    except Exception as exc:
        warnings.append(f"PDF table extraction warning: {type(exc).__name__}")

    try:
        import fitz
        document = fitz.open(str(path))
        if not pages:
            pages = [{"page": index+1, "text": "", "extraction": "empty"}
                     for index in range(min(len(document), MAX_PDF_PAGES))]
        if any(not item["text"] for item in pages):
            try:
                from PIL import Image
                import pytesseract
                for index, page in enumerate(document[:MAX_PDF_PAGES]):
                    if pages[index]["text"]:
                        continue
                    pix = page.get_pixmap(matrix=fitz.Matrix(1.4, 1.4), alpha=False)
                    if pix.width * pix.height > MAX_IMAGE_PIXELS:
                        warnings.append(f"Skipped OCR on PDF page {index+1}: pixel limit.")
                        continue
                    image = Image.open(io.BytesIO(pix.tobytes("png")))
                    ocr = _clean_text(pytesseract.image_to_string(image))
                    if ocr:
                        pages[index] = {"page": index+1, "text": ocr, "extraction": "ocr"}
            except ImportError:
                warnings.append("Scanned PDF OCR unavailable: install Pillow and pytesseract with local Tesseract.")
        if assets_dir is not None:
            assets_dir.mkdir(parents=True, exist_ok=True)
            total_bytes = 0
            for page_index, page in enumerate(document[:MAX_PDF_PAGES]):
                for image_index, image_info in enumerate(page.get_images(full=True)):
                    if len(images) >= 100:
                        warnings.append("PDF embedded-image limit reached (100).")
                        break
                    xref = image_info[0]
                    extracted = document.extract_image(xref)
                    image_bytes = extracted.get("image", b"")
                    if not image_bytes or len(image_bytes) > 20 * 1024 * 1024 or total_bytes + len(image_bytes) > 100 * 1024 * 1024:
                        warnings.append(f"Skipped oversized PDF image on page {page_index+1}.")
                        continue
                    extension = re.sub(r"[^a-z0-9]", "", str(extracted.get("ext") or "png").lower())[:8] or "png"
                    width, height = int(extracted.get("width", 0)), int(extracted.get("height", 0))
                    if width <= 0 or height <= 0 or width * height > MAX_IMAGE_PIXELS:
                        warnings.append(f"Skipped invalid/oversized PDF image on page {page_index+1}.")
                        continue
                    asset_name = f"page_{page_index+1:03d}_image_{image_index+1:03d}.{extension}"
                    (assets_dir / asset_name).write_bytes(image_bytes)
                    total_bytes += len(image_bytes)
                    images.append({"page": page_index+1, "asset_name": asset_name,
                                   "width": width, "height": height, "format": extension,
                                   "source": "embedded_pdf_image"})
    except ImportError:
        if assets_dir is not None:
            warnings.append("PDF embedded-image extraction unavailable: install PyMuPDF.")
        if any(not item["text"] for item in pages):
            warnings.append("Scanned PDF OCR unavailable: install PyMuPDF, Pillow and pytesseract with local Tesseract.")
    except Exception as exc:
        warnings.append(f"PDF image/OCR extraction warning: {type(exc).__name__}")
    return pages, tables, images


def process_dataset(path: str | Path, *, assets_dir: str | Path | None = None) -> dict[str, Any]:
    source = Path(path).resolve(strict=True)
    if not source.is_file():
        raise ValueError("Dataset source must be a regular file")
    suffix = source.suffix.lower()
    warnings: list[str] = []
    text_parts: list[dict[str, Any]] = []
    tables: list[list[Any]] = []
    images: list[dict[str, Any]] = []

    if suffix == ".pdf":
        text_parts, tables, images = _extract_pdf(source, warnings, Path(assets_dir) if assets_dir else None)
    elif suffix in {".csv"}:
        tables = [_parse_delimited(source.read_bytes())]
    elif suffix in {".xlsx", ".xls"}:
        if suffix == ".xls":
            warnings.append("Legacy .xls is not parsed by the XLSX reader; convert to .xlsx or CSV.")
        else:
            try:
                from openpyxl import load_workbook
                workbook = load_workbook(source, read_only=True, data_only=True)
                for sheet in workbook.worksheets[:100]:
                    rows = []
                    for row in sheet.iter_rows(values_only=True):
                        if len(rows) >= MAX_TABLE_ROWS + 1:
                            warnings.append(f"Sheet {sheet.title}: row limit reached.")
                            break
                        rows.append([_safe_cell(value) for value in row[:MAX_COLUMNS]])
                    tables.append([{"sheet": sheet.title, "rows": rows}])
            except Exception as exc:
                warnings.append(f"Workbook extraction failed: {type(exc).__name__}")
    elif suffix in {".txt", ".md", ".json", ".jsonl"}:
        raw = source.read_text(encoding="utf-8-sig", errors="replace")[:MAX_TEXT_CHARS]
        if suffix == ".json":
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, list) and parsed and all(isinstance(x, dict) for x in parsed[:100]):
                    headers = list(dict.fromkeys(k for item in parsed[:1000] for k in item.keys()))[:MAX_COLUMNS]
                    tables = [[headers] + [[item.get(key) for key in headers] for item in parsed[:MAX_TABLE_ROWS]]]
                else:
                    text_parts.append({"page": 1, "text": json.dumps(parsed, ensure_ascii=False, indent=2), "extraction": "structured_json"})
            except json.JSONDecodeError:
                warnings.append("JSON parse failed; retained as text.")
                text_parts.append({"page": 1, "text": _clean_text(raw), "extraction": "text"})
        elif suffix == ".jsonl":
            parsed_rows = []
            for line in raw.splitlines()[:MAX_TABLE_ROWS]:
                try:
                    value = json.loads(line)
                    parsed_rows.append(value if isinstance(value, dict) else {"value": value})
                except json.JSONDecodeError:
                    warnings.append("Some JSONL lines were invalid.")
            if parsed_rows:
                headers = list(dict.fromkeys(k for item in parsed_rows for k in item.keys()))[:MAX_COLUMNS]
                tables = [[headers] + [[item.get(key) for key in headers] for item in parsed_rows]]
        else:
            text_parts.append({"page": 1, "text": _clean_text(raw), "extraction": "text"})
    elif suffix in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}:
        try:
            from PIL import Image
            with Image.open(source) as image:
                width, height = image.size
                if width * height > MAX_IMAGE_PIXELS:
                    raise ValueError("Image exceeds pixel limit")
                images.append({"width": width, "height": height, "format": image.format,
                               "mode": image.mode, "ocr_status": "pending"})
                try:
                    import pytesseract
                    ocr = _clean_text(pytesseract.image_to_string(image))
                    if ocr:
                        text_parts.append({"page": 1, "text": ocr, "extraction": "ocr"})
                    images[-1]["ocr_status"] = "completed" if ocr else "no_text_found"
                except ImportError:
                    warnings.append("Image OCR unavailable: install pytesseract and local Tesseract.")
        except ImportError:
            warnings.append("Image metadata unavailable: Pillow is not installed.")
    elif suffix == ".docx":
        try:
            from docx import Document
            doc = Document(str(source))
            text_parts.append({"page": 1, "text": _clean_text("\n".join(p.text for p in doc.paragraphs)), "extraction": "docx"})
            for table in doc.tables:
                table_rows = [[cell.text for cell in row.cells[:MAX_COLUMNS]] for row in table.rows[:MAX_TABLE_ROWS]]
                if table_rows:
                    tables.append(table_rows)
        except Exception as exc:
            warnings.append(f"DOCX extraction failed: {type(exc).__name__}")
    elif suffix == ".pptx":
        try:
            from pptx import Presentation
            presentation = Presentation(str(source))
            for index, slide in enumerate(presentation.slides[:MAX_PDF_PAGES]):
                chunks = []
                for shape in slide.shapes:
                    if getattr(shape, "has_text_frame", False):
                        chunks.append(shape.text)
                    if getattr(shape, "has_table", False):
                        slide_table = [[cell.text for cell in row.cells[:MAX_COLUMNS]] for row in shape.table.rows[:MAX_TABLE_ROWS]]
                        if slide_table:
                            tables.append(slide_table)
                text_parts.append({"page": index+1, "text": _clean_text("\n".join(chunks)), "extraction": "pptx"})
        except Exception as exc:
            warnings.append(f"PPTX extraction failed: {type(exc).__name__}")

    total_chars = 0
    bounded_parts = []
    for part in text_parts:
        text = part.get("text", "")
        remaining = max(0, MAX_TEXT_CHARS-total_chars)
        text = text[:remaining]
        total_chars += len(text)
        bounded_parts.append({**part, "text": text})
        if total_chars >= MAX_TEXT_CHARS:
            warnings.append("Extracted text character limit reached.")
            break

    # Normalize XLSX sheet containers into one profile per sheet.
    table_profiles = []
    for table in tables:
        if len(table) == 1 and isinstance(table[0], dict) and "rows" in table[0]:
            item = table[0]
            table_profiles.append({"sheet": item["sheet"], **_table_profile(item["rows"])})
        else:
            table_profiles.append(_table_profile(table))

    combined_text = "\n\n".join(item["text"] for item in bounded_parts if item["text"])
    return {
        "schema_version": 1, "source_name": source.name, "source_type": suffix,
        "source_sha256": sha256_file(source), "text_pages": bounded_parts,
        "tables": tables, "images": images,
        "profile": {
            "text_characters": len(combined_text), "pages_with_text": sum(bool(x["text"]) for x in bounded_parts),
            "table_count": len(table_profiles), "tables": table_profiles,
            "image_count": len(images), "warnings": warnings,
        },
        "labeling": {"status": "not_reviewed", "suggestions": [],
                     "supervised_training_ready": False},
    }
