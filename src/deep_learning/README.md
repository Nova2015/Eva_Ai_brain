# EVA Persistent Multimodal Deep Learning Engine

This package is an independent learning subsystem. It does not replace or mutate Vector RAG, the PDF Knowledge Base, or the chat/Speed Priority execution path.

## Runtime policy

Configure with environment variables:

- `EVA_DL_TRAINING_MODE=automatic`
- `EVA_DL_ENVIRONMENT=local-cloud`
- `EVA_DL_COMPUTE=auto`
- `EVA_DL_CLOUD_ENABLED=false` (safe default; no cloud worker is implemented yet)
- `EVA_DL_REGISTRY_DB` (optional SQLite registry path)
- `EVA_DL_DATASET_DIR` (optional persistent dataset root)
- `EVA_DL_MAX_UPLOAD_BYTES` (default 100 MiB)

Automatic compute selects CUDA when PyTorch is installed and CUDA is available; otherwise it selects CPU. Explicit GPU mode fails closed when CUDA is unavailable. A later resource-aware scheduler should also consider free VRAM, active EVA workloads and configured limits before dispatching jobs.

## Data lifecycle

1. Upload through `POST /api/training/datasets`. The file is stored locally and registered with the existing Data Governance system. Its Data ID, source hash, owner, workspace, classification and version are retained.
2. Prepare through `POST /api/training/datasets/{dataset_id}/prepare`. Extraction and profiling are local and bounded. Prepared artifacts are stored under the configured dataset root; the public API returns profiles, not local paths or extracted private content.
3. Review labels and data quality. The processor intentionally marks labels as unreviewed and supervised training as not ready.
4. Explicitly approve the governed Data ID through `POST /api/training/datasets/{dataset_id}/approve`. The job is then queued for a compatible worker.
5. Check `GET /api/training/status/{job_id}`.

## Current extraction coverage

- PDF: text extraction, table extraction, scanned-page OCR when PyMuPDF, Pillow, pytesseract and the local Tesseract binary are available; embedded PDF images are saved as derived assets when PyMuPDF is available.
- Images: dimensions/format metadata and optional local OCR.
- CSV/JSON/JSONL: tabular parsing and column profiling.
- XLSX: worksheet extraction and per-sheet profiles.
- DOCX/PPTX: text and table extraction.

Extraction limits are defined in `processor.py`. Optional reader failures are reported as warnings; they do not silently mark a dataset as training-ready.

## Model selection

`model_selector.py` recommends families by modality and task: CNN/ViT or detectors for vision, XGBoost/MLP for tabular data, Transformers for text, temporal models for time-series, and fusion models for paired multimodal data. Supervised tasks without labels return `needs_labels`. Selection is a recommendation, not a trained model.

## Not yet implemented

The persistent queue is metadata only until a background worker is installed. Actual CNN/ViT, XGBoost/MLP, Transformer and temporal training adapters; evaluation; safe artifact loading; activation/rollback; prediction; and cloud GPU execution remain future stages. Do not interpret a queued job as a completed training run. Cloud transfer is disabled by default and private project data must not be sent to cloud workers.
