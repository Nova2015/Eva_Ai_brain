from src.deep_learning.worker import _tabular_rows


def test_worker_reads_csv_style_prepared_table():
    headers, records = _tabular_rows({
        "tables": [[["part", "defect"], ["A-1", "no"], ["A-2", "yes"]]]
    })
    assert headers == ["part", "defect"]
    assert records == [{"part": "A-1", "defect": "no"}, {"part": "A-2", "defect": "yes"}]


def test_worker_reads_first_excel_sheet_container():
    headers, records = _tabular_rows({
        "tables": [[{"sheet": "Sheet1", "rows": [["x", "y"], [1, 2], [3, 4]]}]]
    })
    assert headers == ["x", "y"]
    assert len(records) == 2


def test_tabular_xgboost_worker_trains_and_evaluates_cpu_candidate(tmp_path):
    from src.deep_learning.worker import _train_tabular

    rows = [["measurement", "defect"]]
    for index in range(40):
        rows.append([float(index), "yes" if index >= 20 else "no"])
    model, result = _train_tabular(
        {"tables": [rows]},
        {"target_column": "defect", "feature_columns": ["measurement"], "task_type": "classification"},
        "cpu",
    )
    import xgboost as xgb
    assert isinstance(model, xgb.Booster)
    assert 0.0 <= result["metrics"]["accuracy"] <= 1.0
    assert result["metrics"]["train_rows"] + result["metrics"]["validation_rows"] == 40
    assert result["model_family"] == "XGBoost"

    # Native Booster serialization must not depend on sklearn estimator tags.
    artifact = tmp_path / "classifier.json"
    model.save_model(str(artifact))
    reloaded = xgb.Booster()
    reloaded.load_model(str(artifact))
    sample = xgb.DMatrix([[10.0]])
    prediction = reloaded.predict(sample)
    assert prediction.shape == (1,)
    assert 0.0 <= float(prediction[0]) <= 1.0


def test_tabular_xgboost_worker_trains_native_cpu_regressor():
    import xgboost as xgb
    from src.deep_learning.worker import _train_tabular

    rows = [["measurement", "target"]]
    for index in range(40):
        rows.append([float(index), float(index * 3 + 2)])
    model, result = _train_tabular(
        {"tables": [rows]},
        {"target_column": "target", "feature_columns": ["measurement"], "task_type": "regression"},
        "cpu",
    )
    assert isinstance(model, xgb.Booster)
    assert result["metrics"]["rmse"] >= 0.0
    assert result["metrics"]["mae"] >= 0.0
    assert result["metrics"]["train_rows"] + result["metrics"]["validation_rows"] == 40
    assert result["model_family"] == "XGBoost"


def test_safe_worker_persists_failure_after_job_is_claimed(tmp_path, monkeypatch):
    import json

    from src.deep_learning import worker
    from src.deep_learning.registry import DeepLearningRegistry

    test_registry = DeepLearningRegistry(tmp_path / "worker-failure.sqlite3")
    dataset = test_registry.create_dataset(
        data_id="EVA-DATA-WORKER-FAIL", owner_id="alice", workspace_id="team-a",
        file_name="synthetic.csv", file_path=str(tmp_path / "synthetic.csv"),
        file_type=".csv", size_bytes=12, sha256="4" * 64,
        classification="internal",
    )
    job = test_registry.approve_dataset(
        dataset["dataset_id"], owner_id="alice", workspace_id="team-a",
        device="cpu", environment="local",
    )
    monkeypatch.setattr(worker, "registry", test_registry)

    def fail_after_claim(job_id, *, owner_id, workspace_id, already_claimed=False):
        assert test_registry.claim_job(
            job_id, owner_id=owner_id, workspace_id=workspace_id
        )
        raise RuntimeError("synthetic worker failure")

    monkeypatch.setattr(worker, "run_training_job", fail_after_claim)
    worker._execute_job_safely(
        job["job_id"], owner_id="alice", workspace_id="team-a"
    )

    persisted = test_registry.get_job(
        job["job_id"], owner_id="alice", workspace_id="team-a"
    )
    assert persisted["status"] == "failed"
    assert persisted["stage"] == "failed"
    assert persisted["progress"] == 0
    assert persisted["error_code"] == "RuntimeError"

    with test_registry.connect() as conn:
        row = conn.execute(
            "SELECT metadata_json FROM dl_training_jobs WHERE job_id=?",
            (job["job_id"],),
        ).fetchone()
    metadata = json.loads(row["metadata_json"])
    assert metadata["failure_message"] == "synthetic worker failure"


def test_tabular_xgboost_worker_trains_native_cpu_multiclass_with_missing_values(tmp_path):
    import numpy as np
    import xgboost as xgb
    from src.deep_learning.worker import _train_tabular

    rows = [["measurement", "machine_type", "defect_class"]]
    classes = ["low", "medium", "high"]
    for index in range(90):
        measurement = None if index % 11 == 0 else float(index % 30)
        machine_type = ["A", "B", "C"][index % 3]
        rows.append([measurement, machine_type, classes[index % 3]])

    model, result = _train_tabular(
        {"tables": [rows]},
        {
            "target_column": "defect_class",
            "feature_columns": ["measurement", "machine_type"],
            "task_type": "classification",
        },
        "cpu",
    )

    assert isinstance(model, xgb.Booster)
    assert 0.0 <= result["metrics"]["accuracy"] <= 1.0
    assert result["metrics"]["train_rows"] + result["metrics"]["validation_rows"] == 90
    preprocessing = result["preprocessing"]
    assert preprocessing["label_mapping"] == {"high": 0, "low": 1, "medium": 2}
    assert preprocessing["feature_kinds"]["machine_type"] == "categorical"
    assert preprocessing["missing_value_medians"]["measurement"] >= 0.0

    artifact = tmp_path / "multiclass.json"
    model.save_model(str(artifact))
    reloaded = xgb.Booster()
    reloaded.load_model(str(artifact))
    sample = xgb.DMatrix(np.asarray([[5.0, 0.0]], dtype=np.float32))
    probabilities = reloaded.predict(sample)
    assert probabilities.shape == (1, 3)
    assert np.isclose(float(probabilities.sum()), 1.0, atol=1e-5)



def test_tabular_xgboost_worker_records_actual_cuda_execution():
    import pytest
    torch = pytest.importorskip("torch", reason="CUDA execution test requires PyTorch")
    import xgboost as xgb
    from src.deep_learning.worker import _train_tabular

    cuda_flag = xgb.build_info().get("USE_CUDA", False)
    cuda_build = cuda_flag is True or str(cuda_flag).strip().lower() in {
        "1", "true", "on", "yes"
    }
    if not torch.cuda.is_available() or not cuda_build:
        pytest.skip("CUDA-enabled PyTorch and XGBoost are required")

    rows = [["measurement", "temperature", "defect"]]
    for index in range(600):
        measurement = float(index % 97)
        temperature = float((index * 7) % 53)
        defect = "yes" if measurement + temperature > 75 else "no"
        rows.append([measurement, temperature, defect])

    model, result = _train_tabular(
        {"tables": [rows]},
        {
            "target_column": "defect",
            "feature_columns": ["measurement", "temperature"],
            "task_type": "classification",
        },
        "cuda",
    )

    assert isinstance(model, xgb.Booster)
    execution = result["execution"]
    assert execution["requested_device"] == "cuda"
    assert "cuda" in execution["actual_device"].lower()
    assert "grow_gpu_hist" in execution["actual_updater"].lower()
    assert execution["xgboost_cuda_build"] is True
    assert execution["gpu_name"] == torch.cuda.get_device_name(0)
