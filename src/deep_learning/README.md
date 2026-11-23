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
- `EVA_DL_MIN_FREE_VRAM_MB` (default 1024)
- `EVA_DL_MIN_AVAILABLE_RAM_MB` (default 2048)

Automatic compute selects CUDA only when PyTorch/CUDA telemetry is available and free VRAM plus available system RAM meet the configured thresholds; otherwise it selects CPU. Explicit GPU mode fails closed when CUDA or the resource thresholds are unavailable. The current policy does not yet account for all active EVA workloads or reserve GPU memory for concurrent chat/RAG inference; the training worker must enforce those limits before execution.

## Data lifecycle

1. Upload through `POST /api/training/datasets`. The file is stored locally and registered with the existing Data Governance system. Its Data ID, source hash, owner, workspace, classification and version are retained.
2. Prepare through `POST /api/training/datasets/{dataset_id}/prepare`. Extraction and profiling are local and bounded. Prepared artifacts are stored under the configured dataset root; the public API returns profiles, not local paths or extracted private content.
3. Review supervised target and feature columns through `POST /api/training/datasets/{dataset_id}/review` with `confirm_labels=true`. This worker release accepts tabular classification and regression only.
4. Explicitly approve the governed Data ID through `POST /api/training/datasets/{dataset_id}/approve`. Governance approval is checked separately from label review; the approved job is dispatched to a local FastAPI background worker.
5. Check `GET /api/training/status/{job_id}`.

## Current extraction coverage

- PDF: text extraction, table extraction, scanned-page OCR when PyMuPDF, Pillow, pytesseract and the local Tesseract binary are available; embedded PDF images are saved as derived assets when PyMuPDF is available.
- Images: dimensions/format metadata and optional local OCR.
- CSV/JSON/JSONL: tabular parsing and column profiling.
- XLSX: worksheet extraction and per-sheet profiles.
- DOCX/PPTX: text and table extraction.

Extraction limits are defined in `processor.py`. Optional reader failures are reported as warnings; they do not silently mark a dataset as training-ready.

## Executable training adapter

The current worker implements a local XGBoost adapter for reviewed CSV/JSON/JSONL/XLSX tabular classification and regression. It verifies owner/workspace scope, Data Governance approval, source SHA-256, source version and prepared-artifact hash before training. Jobs are atomically claimed and serialized to one active local training job at a time, protecting shared system/GPU resources and draining queued work sequentially. Training uses a deterministic 80/20 holdout split and persists evaluation metrics plus model artifacts as candidates. GPU jobs recheck CUDA, XGBoost CUDA support and free VRAM immediately before training; CPU jobs use one training thread.

CNN/ViT, detection, Transformer, LSTM/time-series and multimodal fusion remain model-family recommendations only; their executable training adapters are not yet implemented. Model activation, rollback, inference/prediction APIs, full restart queue dispatch and GPU reservation against concurrent Chat/RAG workloads remain future work. A queued job is not proof of a successful model; inspect its terminal status and metrics.

## Model selection

`model_selector.py` recommends families by modality and task: CNN/ViT or detectors for vision, XGBoost/MLP for tabular data, Transformers for text, temporal models for time-series, and fusion models for paired multimodal data. Supervised tasks without labels return `needs_labels`. Selection is a recommendation, not a trained model.

## Not yet implemented

Additional modality-specific training adapters, robust model activation/rollback, safe artifact loading, prediction APIs, restart-time queued-job dispatch, GPU reservation against concurrent Chat/RAG inference, and cloud GPU execution remain future stages. The current XGBoost worker evaluates a holdout set but does not automatically activate a candidate. Cloud transfer is disabled by default and private project data must not be sent to cloud workers.
