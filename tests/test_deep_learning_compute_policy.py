import pytest

from src.deep_learning.config import choose_compute_device


def test_auto_prefers_gpu_when_free_resources_are_sufficient():
    device, reason = choose_compute_device(
        "auto", cuda_available=True, free_vram_mb=4096, available_ram_mb=8192,
    )
    assert device == "cuda"
    assert "Auto selected GPU" in reason


def test_auto_falls_back_to_cpu_when_vram_is_low():
    device, reason = choose_compute_device(
        "auto", cuda_available=True, free_vram_mb=512, available_ram_mb=8192,
    )
    assert device == "cpu"
    assert "CPU" in reason


def test_gpu_priority_fails_closed_when_cuda_is_missing():
    with pytest.raises(RuntimeError):
        choose_compute_device(
            "gpu", cuda_available=False, free_vram_mb=None, available_ram_mb=8192,
        )


def test_gpu_priority_fails_when_system_ram_is_low():
    with pytest.raises(RuntimeError):
        choose_compute_device(
            "gpu", cuda_available=True, free_vram_mb=4096, available_ram_mb=512,
        )


def test_auto_falls_back_to_cpu_when_system_ram_is_low():
    device, reason = choose_compute_device(
        "auto", cuda_available=True, free_vram_mb=4096, available_ram_mb=512,
    )
    assert device == "cpu"
    assert "CPU" in reason


def test_gpu_priority_fails_when_vram_is_low():
    with pytest.raises(RuntimeError, match="VRAM"):
        choose_compute_device(
            "gpu", cuda_available=True, free_vram_mb=512, available_ram_mb=8192,
        )


def test_cpu_policy_never_selects_gpu():
    device, reason = choose_compute_device(
        "cpu", cuda_available=True, free_vram_mb=4096, available_ram_mb=8192,
    )
    assert device == "cpu"
    assert "CPU-only" in reason


def test_invalid_compute_policy_is_rejected():
    with pytest.raises(ValueError, match="Unsupported compute policy"):
        choose_compute_device(
            "tpu", cuda_available=True, free_vram_mb=4096, available_ram_mb=8192,
        )


def test_auto_falls_back_when_vram_telemetry_is_unavailable():
    device, reason = choose_compute_device(
        "auto", cuda_available=True, free_vram_mb=None, available_ram_mb=8192,
    )
    assert device == "cpu"
    assert "CPU" in reason


def test_auto_allows_unknown_ram_telemetry_when_gpu_vram_is_sufficient():
    device, reason = choose_compute_device(
        "auto", cuda_available=True, free_vram_mb=4096, available_ram_mb=None,
    )
    assert device == "cuda"


def test_gpu_selection_accepts_exact_resource_thresholds():
    device, reason = choose_compute_device(
        "gpu",
        cuda_available=True,
        free_vram_mb=1024,
        available_ram_mb=2048,
        min_free_vram_mb=1024,
        min_available_ram_mb=2048,
    )
    assert device == "cuda"


def _patch_runtime_without_xgboost_cuda(monkeypatch, compute="auto"):
    import sys
    from types import SimpleNamespace
    from src.deep_learning import config

    monkeypatch.setenv("EVA_DL_COMPUTE", compute)
    monkeypatch.setenv("EVA_DL_MIN_FREE_VRAM_MB", "1024")
    monkeypatch.setenv("EVA_DL_MIN_AVAILABLE_RAM_MB", "2048")

    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: True,
            mem_get_info=lambda: (4 * 1024**3, 8 * 1024**3),
        )
    )
    fake_psutil = SimpleNamespace(
        virtual_memory=lambda: SimpleNamespace(available=8 * 1024**3)
    )
    fake_xgboost = SimpleNamespace(
        build_info=lambda: {"USE_CUDA": False}
    )

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)
    monkeypatch.setitem(sys.modules, "xgboost", fake_xgboost)

    return config


def test_runtime_policy_falls_back_when_xgboost_cuda_build_is_unavailable(monkeypatch):
    config = _patch_runtime_without_xgboost_cuda(monkeypatch, "auto")

    policy = config.get_runtime_policy()

    assert policy["cuda_available"] is True
    assert policy["xgboost_cuda_available"] is False
    assert policy["worker_cuda_available"] is False
    assert policy["selected_device"] == "cpu"


def test_runtime_policy_rejects_gpu_when_xgboost_cuda_build_is_unavailable(monkeypatch):
    config = _patch_runtime_without_xgboost_cuda(monkeypatch, "gpu")

    with pytest.raises(RuntimeError, match="CUDA is unavailable"):
        config.get_runtime_policy()


def test_dataset_root_uses_environment_override(monkeypatch, tmp_path):
    from src.deep_learning.config import get_dataset_root

    expected = tmp_path / "custom_datasets"
    monkeypatch.setenv("EVA_DL_DATASET_DIR", str(expected))

    assert get_dataset_root() == expected


def test_registry_db_path_uses_environment_override(monkeypatch, tmp_path):
    from src.deep_learning.config import get_registry_db_path

    expected = tmp_path / "custom_registry.sqlite3"
    monkeypatch.setenv("EVA_DL_REGISTRY_DB", str(expected))

    assert get_registry_db_path() == expected


def test_storage_paths_use_local_app_data_default(monkeypatch, tmp_path):
    from src.deep_learning.config import (
        get_deep_learning_storage_root,
        get_dataset_root,
        get_registry_db_path,
    )

    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    monkeypatch.delenv("EVA_DL_DATASET_DIR", raising=False)
    monkeypatch.delenv("EVA_DL_REGISTRY_DB", raising=False)

    storage_root = tmp_path / "EVA" / "DeepLearning"

    assert get_deep_learning_storage_root() == storage_root
    assert get_dataset_root() == storage_root / "datasets"
    assert get_registry_db_path() == (
        storage_root / "registry" / "eva_deep_learning.db"
    )


def test_storage_path_helpers_do_not_create_directories(monkeypatch, tmp_path):
    from src.deep_learning.config import get_dataset_root, get_registry_db_path

    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    monkeypatch.delenv("EVA_DL_DATASET_DIR", raising=False)
    monkeypatch.delenv("EVA_DL_REGISTRY_DB", raising=False)

    get_dataset_root()
    get_registry_db_path()

    assert list(tmp_path.iterdir()) == []
