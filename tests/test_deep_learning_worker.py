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
    assert model is not None
    assert 0.0 <= result["metrics"]["accuracy"] <= 1.0
    assert result["metrics"]["train_rows"] + result["metrics"]["validation_rows"] == 40
    assert result["model_family"] == "XGBoost"
