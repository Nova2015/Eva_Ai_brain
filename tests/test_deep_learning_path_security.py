from pathlib import Path

import pytest

from src.deep_learning.path_security import PathSecurityError, validate_path


def test_validate_path_accepts_existing_regular_file(tmp_path):
    source = tmp_path / "dataset.csv"
    source.write_text("x,y\n1,2\n", encoding="utf-8")

    assert validate_path(source, must_exist=True, regular_file=True) == source


def test_validate_path_rejects_missing_required_path(tmp_path):
    with pytest.raises(PathSecurityError):
        validate_path(tmp_path / "missing" / "dataset.csv", must_exist=True)


def test_validate_path_rejects_traversal_components(tmp_path):
    unsafe = str(tmp_path) + "/../outside.csv"

    with pytest.raises(PathSecurityError):
        validate_path(unsafe)


def test_validate_path_rejects_symlink_source(tmp_path):
    source = tmp_path / "source.csv"
    source.write_text("x\n1\n", encoding="utf-8")
    link = tmp_path / "linked.csv"
    try:
        link.symlink_to(source)
    except (OSError, NotImplementedError):
        pytest.skip("Symlink creation is unavailable")
    with pytest.raises(PathSecurityError):
        validate_path(link, must_exist=True, regular_file=True)


def test_validate_path_rejects_file_as_parent(tmp_path):
    parent_file = tmp_path / "not-a-directory"
    parent_file.write_text("x", encoding="utf-8")

    with pytest.raises(PathSecurityError):
        validate_path(parent_file / "child.json")
