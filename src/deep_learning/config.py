"""Runtime policy for EVA's independent Deep Learning Engine.

Cloud execution is deliberately disabled until an approved, secure cloud worker
adapter is configured. Private governed data is never sent to a cloud worker.
"""
from __future__ import annotations

import os
from typing import Any


def get_runtime_policy() -> dict[str, Any]:
    training_mode = os.getenv("EVA_DL_TRAINING_MODE", "automatic").strip().lower()
    environment = os.getenv("EVA_DL_ENVIRONMENT", "local-cloud").strip().lower()
    compute = os.getenv("EVA_DL_COMPUTE", "auto").strip().lower()
    cloud_enabled = os.getenv("EVA_DL_CLOUD_ENABLED", "false").strip().lower() == "true"
    if training_mode not in {"automatic", "manual"}:
        training_mode = "automatic"
    if environment not in {"local", "cloud", "local-cloud"}:
        environment = "local-cloud"
    if compute not in {"auto", "gpu", "cpu"}:
        compute = "auto"

    device = "cpu"
    cuda_available = False
    try:
        import torch
        cuda_available = bool(torch.cuda.is_available())
        if compute == "gpu":
            if not cuda_available:
                raise RuntimeError("GPU priority requested but CUDA is unavailable")
            device = "cuda"
        elif compute == "auto" and cuda_available:
            device = "cuda"
    except ImportError:
        if compute == "gpu":
            raise RuntimeError("GPU priority requested but PyTorch is not installed")

    return {
        "training_mode": training_mode,
        "environment": environment,
        "compute": compute,
        "selected_device": device,
        "cuda_available": cuda_available,
        "cloud_enabled": cloud_enabled,
        "cloud_ready": False,
        "cloud_policy": "public data only; explicit per-job approval required",
        "private_data_cloud_transfer": False,
        "training_requires_governance_approval": True,
    }
