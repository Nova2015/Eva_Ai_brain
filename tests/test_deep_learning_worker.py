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
    torch = pytest.importorskip(
        "torch", reason="PyTorch is required only for the optional CUDA execution test"
    )
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

def _build_worker_failure_scenario(tmp_path, monkeypatch):
    import json

    from src.deep_learning import worker
    from src.deep_learning.processor import sha256_file
    from src.deep_learning.registry import DeepLearningRegistry

    registry = DeepLearningRegistry(tmp_path / "failure-scenarios.sqlite3")
    monkeypatch.setattr(worker, "registry", registry)

    source = tmp_path / "synthetic.csv"
    source.write_text("measurement,defect\n1,no\n2,yes\n", encoding="utf-8")
    source_hash = sha256_file(source)

    dataset = registry.create_dataset(
        data_id="EVA-DATA-FAILURE-SCENARIO",
        owner_id="alice",
        workspace_id="team-a",
        file_name=source.name,
        file_path=str(source),
        file_type=".csv",
        size_bytes=source.stat().st_size,
        sha256=source_hash,
        classification="internal",
        metadata={
            "label_review": {
                "status": "verified",
                "modality": "tabular",
                "target_column": "defect",
                "feature_columns": ["measurement"],
                "task_type": "classification",
            }
        },
    )

    job = registry.approve_dataset(
        dataset["dataset_id"],
        owner_id="alice",
        workspace_id="team-a",
        device="cpu",
        environment="local",
    )

    data_root = tmp_path / "datasets"
    artifact_dir = data_root / "processed" / dataset["dataset_id"]
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "prepared.json").write_text(
        json.dumps({
            "source_sha256": source_hash,
            "tables": [[
                ["measurement", "defect"],
                [1, "no"],
                [2, "yes"],
            ]],
        }),
        encoding="utf-8",
    )
    monkeypatch.setenv("EVA_DL_DATASET_DIR", str(data_root))

    from src import data_governance

    monkeypatch.setattr(
        data_governance,
        "get_record",
        lambda data_id: {
            "data_id": data_id,
            "owner_id": "alice",
            "workspace_id": "team-a",
            "training_approval": "approved",
            "sha256": source_hash,
            "version": 1,
        },
    )

    class FakeModel:
        def save_model(self, path):
            with open(path, "w", encoding="utf-8") as file:
                file.write('{"fake_model": true}')

    monkeypatch.setattr(
        worker,
        "_train_tabular",
        lambda prepared, review, device: (
            FakeModel(),
            {
                "model_family": "XGBoost",
                "metrics": {"accuracy": 0.9},
                "preprocessing": {"target_column": "defect"},
                "execution": {
                    "requested_device": "cpu",
                    "actual_device": "cpu",
                },
            },
        ),
    )

    return worker, registry, job, artifact_dir


def test_worker_persists_failure_when_model_artifact_save_fails(tmp_path, monkeypatch):
    import pytest

    worker, registry, job, artifact_dir = _build_worker_failure_scenario(
        tmp_path, monkeypatch
    )

    class SaveFailureModel:
        def save_model(self, path):
            raise OSError("synthetic artifact save failure")

    monkeypatch.setattr(
        worker,
        "_train_tabular",
        lambda prepared, review, device: (
            SaveFailureModel(),
            {
                "model_family": "XGBoost",
                "metrics": {"accuracy": 0.9},
                "preprocessing": {},
                "execution": {"requested_device": "cpu", "actual_device": "cpu"},
            },
        ),
    )

    with pytest.raises(OSError, match="synthetic artifact save failure"):
        worker.run_training_job(
            job["job_id"], owner_id="alice", workspace_id="team-a"
        )

    persisted = registry.get_job(
        job["job_id"], owner_id="alice", workspace_id="team-a"
    )
    assert persisted["status"] == "failed"
    assert persisted["error_code"] == "OSError"
    assert registry.list_models(
        owner_id="alice", workspace_id="team-a"
    ) == []
    assert list(artifact_dir.glob("EVA-MODEL-*_v1.json")) == []


def test_worker_persists_failure_when_model_registration_fails(tmp_path, monkeypatch):
    import pytest

    worker, registry, job, artifact_dir = _build_worker_failure_scenario(
        tmp_path, monkeypatch
    )

    def fail_registration(**kwargs):
        raise RuntimeError("synthetic model registration failure")

    monkeypatch.setattr(registry, "create_model", fail_registration)

    with pytest.raises(RuntimeError, match="synthetic model registration failure"):
        worker.run_training_job(
            job["job_id"], owner_id="alice", workspace_id="team-a"
        )

    persisted = registry.get_job(
        job["job_id"], owner_id="alice", workspace_id="team-a"
    )
    assert persisted["status"] == "failed"
    assert persisted["error_code"] == "RuntimeError"
    assert registry.list_models(
        owner_id="alice", workspace_id="team-a"
    ) == []

    
    # Failed model registration must not leave an orphan artifact.
    artifacts = list(artifact_dir.glob("EVA-MODEL-*_v1.json"))
    assert len(artifacts) == 0


def test_safe_worker_persists_failure_when_dataset_validation_fails(
    tmp_path, monkeypatch
):
    worker, registry, job, artifact_dir = _build_worker_failure_scenario(
        tmp_path, monkeypatch
    )

    # Simulate dataset disappearing after the job has been claimed.
    monkeypatch.setattr(
        registry,
        "get_dataset",
        lambda *args, **kwargs: None,
    )

    # Safe wrapper must catch the validation exception and persist failure.
    worker._execute_job_safely(
        job["job_id"],
        owner_id="alice",
        workspace_id="team-a",
    )

    persisted = registry.get_job(
        job["job_id"],
        owner_id="alice",
        workspace_id="team-a",
    )

    assert persisted["status"] == "failed"
    assert persisted["error_code"] == "KeyError"
    import json

    with registry.connect() as conn:
        row = conn.execute(
            "SELECT metadata_json FROM dl_training_jobs WHERE job_id = ?",
            (job["job_id"],),
        ).fetchone()

    persisted_metadata = json.loads(row["metadata_json"])

    assert "Training dataset not found" in (
        persisted_metadata.get("failure_message", "")
    )
    

    assert registry.list_models(
        owner_id="alice",
        workspace_id="team-a",
    ) == []

    artifacts = list(artifact_dir.glob("EVA-MODEL-*_v1.json"))
    assert artifacts == []


def test_direct_worker_validation_failure_does_not_leave_job_running(
    tmp_path, monkeypatch
):
    import pytest

    worker, registry, job, artifact_dir = _build_worker_failure_scenario(
        tmp_path, monkeypatch
    )

    monkeypatch.setattr(
        registry,
        "get_dataset",
        lambda *args, **kwargs: None,
    )

    with pytest.raises(KeyError, match="Training dataset not found"):
        worker.run_training_job(
            job["job_id"],
            owner_id="alice",
            workspace_id="team-a",
        )

    persisted = registry.get_job(
        job["job_id"],
        owner_id="alice",
        workspace_id="team-a",
    )

    # This assertion should expose the current pre-try validation gap.
    assert persisted["status"] == "failed"


def test_worker_drains_next_queued_job_after_previous_job_fails(
    tmp_path, monkeypatch
):
    from src.deep_learning import worker
    from src.deep_learning.registry import DeepLearningRegistry

    test_registry = DeepLearningRegistry(
        tmp_path / "worker-queue-drain.sqlite3"
    )

    jobs = []
    for index in range(2):
        dataset = test_registry.create_dataset(
            data_id=f"EVA-DATA-QUEUE-{index}",
            owner_id="alice",
            workspace_id="team-a",
            file_name=f"synthetic-{index}.csv",
            file_path=str(tmp_path / f"synthetic-{index}.csv"),
            file_type=".csv",
            size_bytes=12,
            sha256=str(index + 1) * 64,
            classification="internal",
        )
        jobs.append(
            test_registry.approve_dataset(
                dataset["dataset_id"],
                owner_id="alice",
                workspace_id="team-a",
                device="cpu",
                environment="local",
            )
        )

    monkeypatch.setattr(worker, "registry", test_registry)

    executed = []

    def fake_run_training_job(
        job_id, *, owner_id, workspace_id, already_claimed=False
    ):
        executed.append(job_id)

        if job_id == jobs[0]["job_id"]:
            assert test_registry.claim_job(
                job_id,
                owner_id=owner_id,
                workspace_id=workspace_id,
            )
            raise RuntimeError("synthetic first-job failure")

        assert already_claimed is True
        return test_registry.update_job(
            job_id,
            owner_id=owner_id,
            workspace_id=workspace_id,
            status="completed",
            stage="completed",
            progress=100,
        )

    monkeypatch.setattr(
        worker, "run_training_job", fake_run_training_job
    )

    worker.run_training_job_safely(
        jobs[0]["job_id"],
        owner_id="alice",
        workspace_id="team-a",
    )

    first = test_registry.get_job(
        jobs[0]["job_id"],
        owner_id="alice",
        workspace_id="team-a",
    )
    second = test_registry.get_job(
        jobs[1]["job_id"],
        owner_id="alice",
        workspace_id="team-a",
    )

    assert first["status"] == "failed"
    assert second["status"] == "completed"
    assert executed == [
        jobs[0]["job_id"],
        jobs[1]["job_id"],
    ]
