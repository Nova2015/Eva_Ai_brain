"""Task-aware model family selection; deliberately does not train models."""
from __future__ import annotations

from typing import Any

SUPPORTED_MODALITIES = {"image", "video", "tabular", "text", "time_series", "multimodal"}
SUPPORTED_TASKS = {"classification", "regression", "detection", "forecasting", "analysis"}


def select_model(modality: str, task_type: str, *, sample_count: int | None = None,
                 has_labels: bool = False, paired_modalities: bool = False) -> dict[str, Any]:
    modality = (modality or "").strip().lower()
    task_type = (task_type or "").strip().lower()
    if modality not in SUPPORTED_MODALITIES:
        raise ValueError(f"Unsupported modality: {modality}")
    if task_type not in SUPPORTED_TASKS:
        raise ValueError(f"Unsupported task type: {task_type}")

    if task_type in {"classification", "regression", "detection", "forecasting"} and not has_labels:
        return {
            "status": "needs_labels", "modality": modality, "task_type": task_type,
            "recommended_family": None,
            "reason": "Supervised training requires verified labels or target values.",
        }

    if modality == "image":
        if task_type == "detection":
            family, candidates = "object_detection", ["YOLO-family detector", "Faster R-CNN"]
        else:
            family, candidates = "vision", ["pretrained ResNet/EfficientNet CNN", "pretrained Vision Transformer"]
    elif modality == "video":
        family, candidates = "video_vision", ["frame-wise detector plus temporal aggregation", "video transformer"]
    elif modality == "tabular":
        family, candidates = "tabular", ["XGBoost", "MLP"]
    elif modality == "text":
        family, candidates = "text_transformer", ["pretrained Transformer encoder", "task-specific Transformer"]
    elif modality == "time_series":
        family, candidates = "temporal", ["LSTM/GRU", "Temporal Convolutional Network", "Transformer time-series model"]
    else:
        family, candidates = "multimodal_fusion", ["late-fusion encoders", "cross-modal Transformer"]

    if task_type == "analysis":
        return {
            "status": "analysis_only", "modality": modality, "task_type": task_type,
            "recommended_family": family, "candidates": candidates,
            "reason": "Analysis/profiling can run without supervised labels; it does not create a trained model.",
        }
    if sample_count is not None and sample_count < 20:
        warning = "Very small dataset: prefer evaluation/augmentation or classical baselines before deep training."
    else:
        warning = None
    return {
        "status": "candidate_selected", "modality": modality, "task_type": task_type,
        "recommended_family": family, "candidates": candidates,
        "transfer_learning_preferred": modality in {"image", "video", "text"},
        "warning": warning,
        "note": "Selection is a recommendation; benchmark and validation must determine the final model.",
    }
