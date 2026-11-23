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


def test_tabular_xgboost_worker_trains_and_evaluates_cpu_candidate():
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
