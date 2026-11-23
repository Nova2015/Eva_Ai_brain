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
