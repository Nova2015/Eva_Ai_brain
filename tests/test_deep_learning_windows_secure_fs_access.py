"""Cross-platform unit tests for Windows file-handle access tracking."""

import ctypes
import threading
from ctypes import wintypes
from types import SimpleNamespace

import pytest

from src.deep_learning.secure_fs import SecureFilesystemError
from src.deep_learning.windows_secure_fs import WindowsSecureDirectory


READ = WindowsSecureDirectory._FILE_READ_DATA
WRITE = WindowsSecureDirectory._FILE_WRITE_DATA


def _secure_with_handle(access):
    secure = WindowsSecureDirectory.__new__(WindowsSecureDirectory)
    secure._lock = threading.RLock()
    secure._closed = False
    secure._MAX_READ_SIZE = WindowsSecureDirectory._MAX_READ_SIZE
    secure._file_handles = {101: object()}
    secure._file_handle_access = {101: access}
    secure.get_file_identity = lambda handle: secure._file_handles[handle]
    secure._kernel32 = SimpleNamespace(
        ReadFile=lambda *args: (_ for _ in ()).throw(
            AssertionError("ReadFile must not be called for a denied handle")
        ),
        WriteFile=lambda *args: (_ for _ in ()).throw(
            AssertionError("WriteFile must not be called for a denied handle")
        ),
        FlushFileBuffers=lambda *args: (_ for _ in ()).throw(
            AssertionError("FlushFileBuffers must not be called for a denied handle")
        ),
    )
    return secure


def test_read_file_rejects_write_only_handle_before_native_io():
    secure = _secure_with_handle(WRITE)

    with pytest.raises(SecureFilesystemError, match="does not grant read access"):
        secure.read_file(101, 1)


def test_write_file_rejects_read_only_handle_before_native_io():
    secure = _secure_with_handle(READ)

    with pytest.raises(SecureFilesystemError, match="does not grant write access"):
        secure.write_file(101, b"payload")


def test_flush_file_rejects_read_only_handle_before_native_io():
    secure = _secure_with_handle(READ)

    with pytest.raises(SecureFilesystemError, match="does not grant flush access"):
        secure.flush_file(101)


def test_combined_handle_rights_allow_both_access_checks():
    secure = _secure_with_handle(READ | WRITE)

    secure._require_file_access(101, READ, "read")
    secure._require_file_access(101, WRITE, "write")


def test_missing_access_metadata_fails_closed():
    secure = _secure_with_handle(READ)
    del secure._file_handle_access[101]

    with pytest.raises(SecureFilesystemError, match="access metadata is unavailable"):
        secure._require_file_access(101, READ, "read")


def test_close_file_removes_identity_and_access_metadata():
    secure = _secure_with_handle(READ | WRITE)
    secure._ntdll = SimpleNamespace(NtClose=lambda handle: 0)

    secure.close_file(101)

    assert 101 not in secure._file_handles
    assert 101 not in secure._file_handle_access


def test_failed_close_preserves_access_metadata_for_retry():
    secure = _secure_with_handle(READ | WRITE)
    secure._ntdll = SimpleNamespace(NtClose=lambda handle: -1)

    with pytest.raises(SecureFilesystemError, match="NtClose failed"):
        secure.close_file(101)

    assert 101 in secure._file_handles
    assert secure._file_handle_access[101] == READ | WRITE
