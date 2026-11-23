import pytest

from src.deep_learning.model_selector import select_model


def test_images_select_transfer_learning_family():
    result = select_model("image", "classification", has_labels=True, sample_count=500)
    assert result["recommended_family"] == "vision"
    assert result["transfer_learning_preferred"] is True


def test_tabular_uses_tabular_models_not_cnn():
    result = select_model("tabular", "regression", has_labels=True, sample_count=1000)
    assert result["recommended_family"] == "tabular"
    assert "XGBoost" in result["candidates"]
    assert not any("CNN" in name for name in result["candidates"])


def test_unlabelled_supervised_dataset_is_not_trainable():
    result = select_model("image", "detection", has_labels=False)
    assert result["status"] == "needs_labels"
    assert result["recommended_family"] is None


def test_time_series_selects_temporal_models():
    result = select_model("time_series", "forecasting", has_labels=True)
    assert result["recommended_family"] == "temporal"


def test_unknown_modality_is_rejected():
    with pytest.raises(ValueError):
        select_model("spreadsheet", "classification", has_labels=True)
