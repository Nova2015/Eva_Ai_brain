"""Authenticated FastAPI surface for EVA's independent Deep Learning Engine."""
from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile

from .config import get_runtime_policy
from .registry import registry

try:
    from src.data_governance import register_import, get_record, set_training_approval
except ImportError:
    from data_governance import register_import, get_record, set_training_approval

MAX_UPLOAD_BYTES = int(os.getenv("EVA_DL_MAX_UPLOAD_BYTES", str(100 * 1024 * 1024)))
ALLOWED_EXTENSIONS = {
    ".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp",
    ".csv", ".xlsx", ".xls", ".txt", ".md", ".docx", ".pptx", ".json", ".jsonl"
}
ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = Path(os.getenv("EVA_DL_DATASET_DIR", str(ROOT / "data" / "deep_learning" / "datasets"))).expanduser()


def build_router(auth_dependency) -> APIRouter:
    router = APIRouter(tags=["EVA Deep Learning"])

    @router.get("/training/policy")
    async def training_policy(_: str = Depends(auth_dependency)):
        return {"ok": True, "policy": get_runtime_policy()}

    @router.post("/training/datasets")
    async def upload_dataset(
        file: UploadFile = File(...),
        workspace_id: str = Form("default"),
        classification: str = Form("restricted"),
        project_id: str | None = Form(None),
        component_id: str | None = Form(None),
        user_id: str = Depends(auth_dependency),
    ):
        if classification not in {"public", "internal", "confidential", "restricted"}:
            raise HTTPException(status_code=400, detail="Invalid data classification")
        if not workspace_id.strip():
            raise HTTPException(status_code=400, detail="workspace_id is required")
        original_name = Path(file.filename or "dataset").name
        suffix = Path(original_name).suffix.lower()
        if suffix not in ALLOWED_EXTENSIONS:
            raise HTTPException(status_code=415, detail="Unsupported dataset file type")
        payload = await file.read(MAX_UPLOAD_BYTES + 1)
        if not payload:
            raise HTTPException(status_code=400, detail="Dataset file is empty")
        if len(payload) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="Dataset exceeds configured upload limit")

        DATASET_ROOT.mkdir(parents=True, exist_ok=True)
        stored_name = f"{uuid.uuid4().hex}{suffix}"
        target = DATASET_ROOT / stored_name
        target.write_bytes(payload)
        governed = None
        try:
            governed = register_import(
                target, source="deep_learning_dataset_upload", owner_id=user_id,
                workspace_id=workspace_id, actor_id=user_id,
                classification=classification, project_id=project_id,
                component_id=component_id,
            )
            dataset = registry.create_dataset(
                data_id=governed["data_id"], owner_id=user_id,
                workspace_id=workspace_id, file_name=original_name,
                file_path=str(target), file_type=suffix,
                size_bytes=len(payload), sha256=governed["sha256"],
                classification=classification, source_version=governed["version"],
                metadata={"label_status": "not_reviewed", "ingestion_status": "stored"},
            )
        except Exception:
            # Never delete bytes after Governance has registered them: that
            # would leave a persistent Data ID pointing to a missing source.
            if governed is None:
                try:
                    target.unlink(missing_ok=True)
                except Exception:
                    pass
            raise
        return {
            "ok": True, "dataset": dataset,
            "training": "awaiting_explicit_governance_approval",
            "note": "Upload is stored locally. Automatic training is queued only after this dataset is approved for training.",
        }

    @router.get("/training/datasets")
    async def list_datasets(
        workspace_id: str = Query("default"),
        user_id: str = Depends(auth_dependency),
    ):
        return {"ok": True, "datasets": registry.list_datasets(owner_id=user_id, workspace_id=workspace_id)}

    @router.post("/training/datasets/{dataset_id}/prepare")
    async def prepare_dataset(
        dataset_id: str,
        workspace_id: str = Query("default"),
        user_id: str = Depends(auth_dependency),
    ):
        dataset = registry.get_dataset(dataset_id, owner_id=user_id, workspace_id=workspace_id)
        if dataset is None:
            raise HTTPException(status_code=404, detail="Dataset not found")
        governed = get_record(dataset["data_id"])
        if not governed or governed.get("owner_id") != user_id or governed.get("workspace_id") != workspace_id:
            raise HTTPException(status_code=404, detail="Governed dataset not found")
        if governed.get("sha256") != dataset["sha256"] or int(governed.get("version", 0)) != int(dataset["source_version"]):
            raise HTTPException(status_code=409, detail="Dataset source revision changed; revalidate before preparation")
        source_path = Path(dataset["file_path"])
        if not source_path.is_file():
            raise HTTPException(status_code=409, detail="Dataset source file is missing")
        from .processor import process_dataset, sha256_file
        if sha256_file(source_path) != dataset["sha256"]:
            raise HTTPException(status_code=409, detail="Dataset file integrity check failed")
        try:
            prepared = process_dataset(source_path)
        except Exception as exc:
            raise HTTPException(status_code=422, detail=f"Dataset preparation failed: {type(exc).__name__}")
        if prepared["source_sha256"] != dataset["sha256"]:
            raise HTTPException(status_code=409, detail="Dataset changed during preparation")
        artifact_dir = DATASET_ROOT / "processed" / dataset_id
        artifact_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = artifact_dir / "prepared.json"
        temporary_path = artifact_dir / "prepared.json.tmp"
        import json
        temporary_path.write_text(json.dumps(prepared, ensure_ascii=False), encoding="utf-8")
        temporary_path.replace(artifact_path)
        summary = prepared["profile"]
        registry.update_dataset_metadata(
            dataset_id, owner_id=user_id, workspace_id=workspace_id,
            updates={
                "preparation_status": "completed",
                "profile": summary,
                "label_status": "not_reviewed",
                "prepared_source_sha256": prepared["source_sha256"],
            },
            status="prepared",
        )
        return {
            "ok": True, "dataset_id": dataset_id, "data_id": dataset["data_id"],
            "preparation_status": "completed", "profile": summary,
            "labeling": prepared["labeling"],
            "training_ready": False,
            "note": "Extraction and profiling completed locally. Labels still require review; no model was trained.",
        }

    @router.get("/training/datasets/{dataset_id}/profile")
    async def dataset_profile(
        dataset_id: str,
        workspace_id: str = Query("default"),
        user_id: str = Depends(auth_dependency),
    ):
        dataset = registry.get_dataset(dataset_id, owner_id=user_id, workspace_id=workspace_id)
        if dataset is None:
            raise HTTPException(status_code=404, detail="Dataset not found")
        metadata = json.loads(dataset.get("metadata_json") or "{}")
        if not metadata.get("profile"):
            raise HTTPException(status_code=404, detail="Dataset has not been prepared")
        return {"ok": True, "dataset_id": dataset_id, "data_id": dataset["data_id"],
                "preparation_status": metadata.get("preparation_status"),
                "profile": metadata["profile"], "label_status": metadata.get("label_status", "not_reviewed")}

    @router.post("/training/datasets/{dataset_id}/approve")
    async def approve_dataset(
        dataset_id: str,
        workspace_id: str = Query("default"),
        user_id: str = Depends(auth_dependency),
    ):
        dataset = registry.get_dataset(dataset_id, owner_id=user_id, workspace_id=workspace_id)
        if dataset is None:
            raise HTTPException(status_code=404, detail="Dataset not found")
        governed = get_record(dataset["data_id"])
        if not governed or governed.get("owner_id") != user_id or governed.get("workspace_id") != workspace_id:
            raise HTTPException(status_code=404, detail="Governed dataset not found")
        if governed.get("sha256") != dataset["sha256"] or int(governed.get("version", 0)) != int(dataset["source_version"]):
            raise HTTPException(status_code=409, detail="Dataset source revision changed; revalidate before training")
        if not Path(dataset["file_path"]).is_file():
            raise HTTPException(status_code=409, detail="Dataset source file is missing")
        try:
            approval = set_training_approval(dataset["data_id"], "approved", actor_id=user_id)
            policy = get_runtime_policy()
            job = registry.approve_dataset(
                dataset_id, owner_id=user_id, workspace_id=workspace_id,
                device=policy["selected_device"], environment="local",
            )
        except (KeyError, ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=409, detail=str(exc))
        return {
            "ok": True, "dataset_id": dataset_id, "data_id": dataset["data_id"],
            "training_approval": approval.get("training_approval", "approved"),
            "job": job, "policy": policy,
            "note": "Dataset approved and automatically queued. A compatible training worker must be available to execute the queued job.",
        }

    @router.post("/training/start")
    async def start_training(
        dataset_id: str = Form(...),
        workspace_id: str = Form("default"),
        user_id: str = Depends(auth_dependency),
    ):
        dataset = registry.get_dataset(dataset_id, owner_id=user_id, workspace_id=workspace_id)
        if dataset is None:
            raise HTTPException(status_code=404, detail="Dataset not found")
        governed = get_record(dataset["data_id"])
        if not governed or governed.get("owner_id") != user_id or governed.get("workspace_id") != workspace_id:
            raise HTTPException(status_code=404, detail="Governed dataset not found")
        if governed.get("training_approval") != "approved" or dataset.get("status") != "approved":
            raise HTTPException(status_code=403, detail="Explicit Data Governance training approval is required")
        if governed.get("sha256") != dataset["sha256"] or int(governed.get("version", 0)) != int(dataset["source_version"]):
            raise HTTPException(status_code=409, detail="Dataset source revision changed; revalidate before training")
        if not Path(dataset["file_path"]).is_file():
            raise HTTPException(status_code=409, detail="Dataset source file is missing")
        try:
            policy = get_runtime_policy()
            job = registry.approve_dataset(
                dataset_id, owner_id=user_id, workspace_id=workspace_id,
                device=policy["selected_device"], environment="local",
            )
        except (KeyError, ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=409, detail=str(exc))
        return {"ok": True, "job": job, "policy": policy}

    @router.get("/training/status/{job_id}")
    async def training_status(
        job_id: str,
        workspace_id: str = Query("default"),
        user_id: str = Depends(auth_dependency),
    ):
        job = registry.get_job(job_id, owner_id=user_id, workspace_id=workspace_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Training job not found")
        return {"ok": True, "job": job}

    @router.get("/models")
    async def list_models(
        workspace_id: str = Query("default"),
        user_id: str = Depends(auth_dependency),
    ):
        return {"ok": True, "models": registry.list_models(owner_id=user_id, workspace_id=workspace_id)}

    return router
