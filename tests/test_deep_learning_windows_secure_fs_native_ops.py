"""Windows native file I/O, rename and deletion integration tests."""

import os
import uuid
from pathlib import Path

import pytest

from src.deep_learning.secure_fs import (
    IdentityMismatchError,
    SecureFilesystemError,
)
from src.deep_learning.windows_secure_fs import WindowsSecureDirectory


pytestmark = pytest.mark.skipif(
    os.name != "nt",
    reason="Windows native backend tests require Windows.",
)


def _storage_root() -> Path:
    local = os.environ.get("LOCALAPPDATA")
    if not local:
        pytest.skip("LOCALAPPDATA is unavailable.")

    root = Path(local) / "EVA" / "DeepLearning"
    if not root.is_dir():
        pytest.skip("Existing EVA DeepLearning storage root is unavailable.")
    return root


def _name(prefix: str) -> str:
    return f".pytest-{prefix}-{uuid.uuid4().hex}.bin"


def _open_secure():
    return WindowsSecureDirectory(_storage_root())


def test_native_write_flush_and_read_round_trip():
    payload = b"EVA Windows native I/O\x00payload" * 4096
    with _open_secure() as secure:
        name = _name("roundtrip")
        writer = secure.create_exclusive(name)
        try:
            assert secure.write_file(writer, payload) == len(payload)
            secure.flush_file(writer)
        finally:
            secure.close_file(writer)

        reader = secure.open_read(name)
        try:
            assert secure.read_file(reader, len(payload) + 32) == payload
            assert secure.read_file(reader, 1) == b""
        finally:
            secure.close_file(reader)

        secure.delete_file(name)


def test_read_only_handle_rejects_write_and_flush():
    with _open_secure() as secure:
        name = _name("readonly")
        writer = secure.create_exclusive(name)
        secure.close_file(writer)

        reader = secure.open_read(name)
        try:
            with pytest.raises(SecureFilesystemError, match="does not grant write access"):
                secure.write_file(reader, b"no")
            with pytest.raises(SecureFilesystemError, match="does not grant flush access"):
                secure.flush_file(reader)
        finally:
            secure.close_file(reader)
            secure.delete_file(name)


def test_write_only_handle_rejects_read():
    with _open_secure() as secure:
        name = _name("writeonly")
        writer = secure.create_exclusive(name)
        try:
            with pytest.raises(SecureFilesystemError, match="does not grant read access"):
                secure.read_file(writer, 1)
        finally:
            secure.close_file(writer)
            secure.delete_file(name)


def test_rename_moves_file_and_preserves_identity():
    with _open_secure() as secure:
        source = _name("rename-source")
        destination = _name("rename-destination")
        writer = secure.create_exclusive(source)
        try:
            secure.write_file(writer, b"rename-payload")
            secure.flush_file(writer)
            original_identity = secure.get_file_identity(writer)
        finally:
            secure.close_file(writer)

        moved_identity = secure.rename(
            source,
            destination,
            expected_identity=original_identity,
        )
        assert moved_identity == original_identity

        reader = secure.open_read(destination)
        try:
            assert secure.read_file(reader, 128) == b"rename-payload"
        finally:
            secure.close_file(reader)

        with pytest.raises(FileNotFoundError):
            secure.open_read(source)
        secure.delete_file(destination, expected_identity=original_identity)


def test_rename_without_replace_rejects_existing_destination():
    with _open_secure() as secure:
        source = _name("no-replace-source")
        destination = _name("no-replace-destination")
        first = secure.create_exclusive(source)
        second = secure.create_exclusive(destination)
        secure.close_file(first)
        secure.close_file(second)

        try:
            with pytest.raises((FileExistsError, SecureFilesystemError)):
                secure.rename(source, destination, replace=False)
            # The source still exists; verify it remains openable.
            handle = secure.open_read(source)
            secure.close_file(handle)
        finally:
            secure.delete_file(source)
            secure.delete_file(destination)


def test_rename_replace_replaces_destination_atomically():
    with _open_secure() as secure:
        source = _name("replace-source")
        destination = _name("replace-destination")

        source_handle = secure.create_exclusive(source)
        try:
            secure.write_file(source_handle, b"new-content")
            secure.flush_file(source_handle)
            source_identity = secure.get_file_identity(source_handle)
        finally:
            secure.close_file(source_handle)

        destination_handle = secure.create_exclusive(destination)
        try:
            secure.write_file(destination_handle, b"old-content")
            secure.flush_file(destination_handle)
        finally:
            secure.close_file(destination_handle)

        result = secure.rename(
            source,
            destination,
            replace=True,
            expected_identity=source_identity,
        )
        assert result == source_identity

        reader = secure.open_read(destination)
        try:
            assert secure.read_file(reader, 128) == b"new-content"
            assert secure.get_file_identity(reader) == source_identity
        finally:
            secure.close_file(reader)

        secure.delete_file(destination, expected_identity=source_identity)


def test_delete_file_checks_expected_identity():
    with _open_secure() as secure:
        name = _name("delete-identity")
        handle = secure.create_exclusive(name)
        try:
            identity = secure.get_file_identity(handle)
        finally:
            secure.close_file(handle)

        wrong_identity = type(identity)(
            backend=identity.backend,
            volume_id=identity.volume_id,
            file_id=identity.file_id + 1,
            object_type=identity.object_type,
        )
        with pytest.raises(IdentityMismatchError):
            secure.delete_file(name, expected_identity=wrong_identity)

        secure.delete_file(name, expected_identity=identity)
        with pytest.raises(FileNotFoundError):
            secure.open_read(name)


def test_delete_directory_removes_only_empty_directory():
    with _open_secure() as secure:
        name = f".pytest-empty-dir-{uuid.uuid4().hex}"
        directory_path = _storage_root() / name
        os.mkdir(directory_path)

        # Pin the child directory to capture the identity through the same
        # native handle inspection used by the backend.
        child = secure._nt_open(
            name,
            secure._handles[-1],
            directory=True,
        )
        try:
            secure._inspect_directory(child)
            identity = secure._identity(child, object_type="directory")
        finally:
            secure._close_handles([child], "test directory handle")

        secure.delete_directory(name, expected_identity=identity)
        assert not directory_path.exists()


def test_close_retries_failed_handles_and_attempts_all_handles():
    import ctypes
    import threading
    from types import SimpleNamespace
    from ctypes import wintypes

    from src.deep_learning.windows_secure_fs import (
        WindowsSecureDirectory,
        SecureFilesystemError,
    )

    secure = WindowsSecureDirectory.__new__(WindowsSecureDirectory)
    secure._lock = threading.RLock()
    secure._closed = False

    file_handle = 101
    orphan_handle = wintypes.HANDLE(202)
    directory_failed = wintypes.HANDLE(303)
    directory_ok = wintypes.HANDLE(404)

    secure._file_handles = {file_handle: object()}
    secure._file_handle_access = {file_handle: 3}
    secure._orphan_handles = [orphan_handle]
    secure._handles = [directory_failed, directory_ok]

    outcomes = {
        101: -1,
        202: 0,
        303: -1,
        404: 0,
    }
    attempted = []

    def fake_nt_close(handle):
        value = handle.value if hasattr(handle, "value") else int(handle)
        attempted.append(value)
        return outcomes[value]

    secure._ntdll = SimpleNamespace(NtClose=fake_nt_close)

    with pytest.raises(SecureFilesystemError):
        secure.close()

    assert attempted == [101, 202, 404, 303]
    assert 101 in secure._file_handles
    assert secure._file_handle_access[101] == 3
    assert secure._orphan_handles == []
    assert secure._handles == [directory_failed]

    outcomes[101] = 0
    outcomes[303] = 0
    attempted.clear()

    secure.close()

    assert attempted == [101, 303]
    assert secure._file_handles == {}
    assert secure._file_handle_access == {}
    assert secure._orphan_handles == []
    assert secure._handles == []

    secure.close()

def test_temporary_handle_cleanup_retains_failures_for_close_retry():
    import threading
    from types import SimpleNamespace
    from ctypes import wintypes

    from src.deep_learning.windows_secure_fs import (
        WindowsSecureDirectory,
        SecureFilesystemError,
    )

    secure = WindowsSecureDirectory.__new__(WindowsSecureDirectory)
    secure._lock = threading.RLock()
    secure._closed = False
    secure._file_handles = {}
    secure._file_handle_access = {}
    secure._orphan_handles = []
    secure._handles = []

    failed_handle = wintypes.HANDLE(501)
    successful_handle = wintypes.HANDLE(502)

    outcomes = {501: -1, 502: 0}
    attempted = []

    def fake_nt_close(handle):
        value = handle.value if hasattr(handle, "value") else int(handle)
        attempted.append(value)
        return outcomes[value]

    secure._ntdll = SimpleNamespace(NtClose=fake_nt_close)

    with pytest.raises(SecureFilesystemError):
        secure._close_handles(
            [failed_handle, successful_handle],
            "temporary directory handle",
        )

    assert attempted == [501, 502]
    assert secure._orphan_handles == [failed_handle]

    outcomes[501] = 0
    attempted.clear()

    secure.close()

    assert attempted == [501]
    assert secure._orphan_handles == []
    assert secure._handles == []

    secure.close()