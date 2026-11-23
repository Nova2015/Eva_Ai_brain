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
