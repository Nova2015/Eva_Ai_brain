from pathlib import Path

from src.deep_learning.processor import process_dataset


def test_csv_processing_profiles_numeric_and_missing_values(tmp_path):
    source = tmp_path / "measurements.csv"
    source.write_text("part_id,diameter,defect\nA-1,12.5,no\nA-2,13.0,yes\nA-3,,no\n", encoding="utf-8")
    result = process_dataset(source)
    assert result["source_type"] == ".csv"
    assert result["profile"]["table_count"] == 1
    profiles = result["profile"]["tables"][0]["column_profiles"]
    assert profiles[0]["name"] == "part_id"
    assert profiles[1]["type"] == "numeric"
    assert profiles[1]["missing"] == 1
    assert result["labeling"]["supervised_training_ready"] is False


def test_text_processing_is_bounded_and_local(tmp_path):
    source = tmp_path / "notes.txt"
    source.write_text("Inspection notes\nCrack found near weld.", encoding="utf-8")
    result = process_dataset(source)
    assert "Crack found" in result["text_pages"][0]["text"]
    assert result["profile"]["text_characters"] > 0
    assert result["labeling"]["status"] == "not_reviewed"


def test_json_records_become_tabular_data(tmp_path):
    source = tmp_path / "records.json"
    source.write_text('[{"temperature":20,"ok":true},{"temperature":25,"ok":false}]', encoding="utf-8")
    result = process_dataset(source)
    assert result["profile"]["table_count"] == 1
    assert result["profile"]["tables"][0]["rows"] == 2
    assert result["profile"]["tables"][0]["columns"] == 2
