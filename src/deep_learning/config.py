"""Runtime policy for EVA's independent Deep Learning Engine.

Cloud execution is deliberately disabled until an approved, secure cloud worker
adapter is configured. Private governed data is never sent to a cloud worker.
"""
from __future__ import annotations

import os
from typing import Any


def choose_compute_device(
    compute: str,
    *,
    cuda_available: bool,
    free_vram_mb: int | None,
    available_ram_mb: int | None,
    min_free_vram_mb: int = 1024,
    min_available_ram_mb: int = 2048,
) -> tuple[str, str]:
    """Choose a device conservatively from observed free resources."""
    if compute not in {"auto", "gpu", "cpu"}:
        raise ValueError("Unsupported compute policy")
    if compute == "cpu":
        return "cpu", "CPU-only policy selected"
    ram_ok = available_ram_mb is None or available_ram_mb >= min_available_ram_mb
    gpu_ok = cuda_available and free_vram_mb is not None and free_vram_mb >= min_free_vram_mb and ram_ok
    if compute == "gpu":
        if not cuda_available:
            raise RuntimeError("GPU priority requested but CUDA is unavailable")
        if free_vram_mb is None or free_vram_mb < min_free_vram_mb:
            raise RuntimeError("GPU priority requested but free VRAM is below the configured safety threshold")
        if not ram_ok:
            raise RuntimeError("GPU priority requested but available system RAM is below the configured safety threshold")
        return "cuda", "GPU priority selected; free VRAM and system RAM satisfy safety thresholds"
    if gpu_ok:
        return "cuda", "Auto selected GPU; free VRAM and system RAM satisfy safety thresholds"
    return "cpu", "Auto selected CPU because CUDA/resource checks did not satisfy safety thresholds"


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

    cuda_available = False
    free_vram_mb = None
    try:
        import torch
        cuda_available = bool(torch.cuda.is_available())
        if cuda_available:
            free_bytes, _total_bytes = torch.cuda.mem_get_info()
            free_vram_mb = int(free_bytes / (1024 * 1024))
    except ImportError:
        pass
    except Exception:
        # A failed telemetry query must not make Auto overcommit the GPU.
        free_vram_mb = None

    available_ram_mb = None
    try:
        import psutil
        available_ram_mb = int(psutil.virtual_memory().available / (1024 * 1024))
    except Exception:
        pass

    # The first executable worker uses XGBoost. Do not select CUDA for that
    # adapter unless both the CUDA runtime and this XGBoost build support it.
    xgboost_cuda_available = False
    try:
        import xgboost
        build_info = xgboost.build_info()
        raw_cuda_flag = build_info.get("USE_CUDA", False)
        xgboost_cuda_available = raw_cuda_flag is True or str(raw_cuda_flag).strip().lower() in {"1", "true", "on", "yes"}
    except Exception:
        pass
    worker_cuda_available = cuda_available and xgboost_cuda_available

    min_vram = int(os.getenv("EVA_DL_MIN_FREE_VRAM_MB", "1024"))
    min_ram = int(os.getenv("EVA_DL_MIN_AVAILABLE_RAM_MB", "2048"))
    device, selection_reason = choose_compute_device(
        compute, cuda_available=worker_cuda_available, free_vram_mb=free_vram_mb,
        available_ram_mb=available_ram_mb, min_free_vram_mb=min_vram,
        min_available_ram_mb=min_ram,
    )
    return {
        "training_mode": training_mode,
        "environment": environment,
        "compute": compute,
        "selected_device": device,
        "selection_reason": selection_reason,
        "cuda_available": cuda_available,
        "xgboost_cuda_available": xgboost_cuda_available,
        "worker_cuda_available": worker_cuda_available,
        "free_vram_mb": free_vram_mb,
        "available_ram_mb": available_ram_mb,
        "min_free_vram_mb": min_vram,
        "min_available_ram_mb": min_ram,
        "cloud_enabled": cloud_enabled,
        "cloud_ready": False,
        "cloud_policy": "public data only; explicit per-job approval required",
        "private_data_cloud_transfer": False,
        "training_requires_governance_approval": True,
    }
