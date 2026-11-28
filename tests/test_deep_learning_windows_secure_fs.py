import os
import ctypes
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from ctypes import wintypes

import pytest

from src.deep_learning.secure_fs import (
    IdentityMismatchError,
    SecureFilesystemError,
    SecureFilesystemUnavailable,
)
from src.deep_learning.windows_secure_fs import WindowsSecureDirectory


pytestmark = pytest.mark.skipif(
    os.name != "nt",
    reason="Windows native backend tests require Windows.",
)


def _storage_root():
    local = os.environ.get("LOCALAPPDATA")
    if not local:
        pytest.skip("LOCALAPPDATA is unavailable.")

    root = Path(local) / "EVA" / "DeepLearning"
    if not root.is_dir():
        pytest.skip("Existing EVA DeepLearning storage root is unavailable.")

    return root


def test_existing_storage_root_opens_and_identity_verifies():
    with WindowsSecureDirectory(_storage_root()) as secure:
        identity = secure.get_identity()

        assert identity.backend == "windows"
        assert identity.object_type == "directory"
        assert identity.volume_id > 0
        assert identity.file_id > 0

        secure.verify_identity(identity)


def test_closed_directory_rejects_operations():
    secure = WindowsSecureDirectory(_storage_root())
    secure.close()

    with pytest.raises(SecureFilesystemError):
        secure.get_identity()


def test_close_is_idempotent():
    secure = WindowsSecureDirectory(_storage_root())

    secure.close()
    secure.close()

    assert secure._closed is True
    assert secure._handles == []


@pytest.mark.parametrize(
    "path",
    [
        "",
        "relative\\path",
        "C:\\folder\\..\\escape",
        "C:\\folder\\.\\child",
        "C:\\folder\\\\child",
        "\\\\server\\share",
        "\\\\?\\C:\\folder",
        "\\\\.\\C:\\folder",
    ],
)
def test_invalid_root_paths_are_rejected(path):
    with pytest.raises(SecureFilesystemError):
        WindowsSecureDirectory(path)


def test_identity_mismatch_is_rejected():
    secure = WindowsSecureDirectory(_storage_root())

    try:
        actual = secure.get_identity()
        wrong = type(actual)(
            backend=actual.backend,
            volume_id=actual.volume_id,
            file_id=actual.file_id + 1,
            object_type=actual.object_type,
        )

        with pytest.raises(IdentityMismatchError):
            secure.verify_identity(wrong)
    finally:
        secure.close()


def test_current_user_sid_returns_valid_windows_sid():
    secure = WindowsSecureDirectory(_storage_root())

    try:
        sid = secure._current_user_sid()

        assert isinstance(sid, str)
        assert sid.startswith("S-1-")
        assert len(sid) >= 5
    finally:
        secure.close()



def _token_type(secure, token):
    token_type = __import__("ctypes").wintypes.DWORD()
    required = __import__("ctypes").wintypes.DWORD()

    ok = secure._advapi32.GetTokenInformation(
        token,
        8,  # TokenType
        __import__("ctypes").byref(token_type),
        __import__("ctypes").sizeof(token_type),
        __import__("ctypes").byref(required),
    )
    assert ok, "GetTokenInformation(TokenType) failed"
    return token_type.value


def _assert_token_closed(secure, token):
    import ctypes

    token_type = ctypes.wintypes.DWORD()
    required = ctypes.wintypes.DWORD()

    ok = secure._advapi32.GetTokenInformation(
        token,
        8,
        ctypes.byref(token_type),
        ctypes.sizeof(token_type),
        ctypes.byref(required),
    )
    assert not ok, "Token handle remained usable after context exit"


def test_impersonation_token_has_expected_type_and_is_closed():
    secure = WindowsSecureDirectory(_storage_root())

    try:
        with secure._impersonation_token() as token:
            assert token.value
            assert _token_type(secure, token) == 2  # TokenImpersonation

        _assert_token_closed(secure, token)
    finally:
        secure.close()


def test_impersonation_token_is_closed_after_body_exception():
    secure = WindowsSecureDirectory(_storage_root())
    captured = []

    try:
        with pytest.raises(RuntimeError, match="test exception"):
            with secure._impersonation_token() as token:
                captured.append(token.value)
                raise RuntimeError("test exception")

        assert len(captured) == 1
        assert captured[0]
        _assert_token_closed(secure, __import__("ctypes").wintypes.HANDLE(captured[0]))
    finally:
        secure.close()




def test_local_free_failure_is_raised_without_primary_exception(monkeypatch):
    secure = WindowsSecureDirectory.__new__(WindowsSecureDirectory)
    secure._kernel32 = SimpleNamespace(LocalFree=lambda pointer: 1)
    monkeypatch.setattr(ctypes, "get_last_error", lambda: 5)

    with pytest.raises(SecureFilesystemUnavailable, match="WinError 5"):
        secure._local_free(ctypes.c_void_p(123), "test allocation")


def test_local_free_failure_preserves_primary_exception(monkeypatch):
    secure = WindowsSecureDirectory.__new__(WindowsSecureDirectory)
    secure._kernel32 = SimpleNamespace(LocalFree=lambda pointer: 1)
    monkeypatch.setattr(ctypes, "get_last_error", lambda: 5)

    with pytest.raises(RuntimeError, match="original operation"):
        try:
            raise RuntimeError("original operation")
        except RuntimeError as primary:
            secure._local_free(ctypes.c_void_p(123), "test allocation")
            assert len(primary._eva_cleanup_errors) == 1
            assert "WinError 5" in str(primary._eva_cleanup_errors[0])
            raise


def test_both_token_close_failures_preserve_body_exception(monkeypatch):
    secure = WindowsSecureDirectory.__new__(WindowsSecureDirectory)
    closed_handles = []

    def assign_handle(output_pointer, value):
        ctypes.cast(
            output_pointer, ctypes.POINTER(wintypes.HANDLE)
        ).contents.value = value

    def open_process_token(process, access, output_pointer):
        assign_handle(output_pointer, 101)
        return True

    def duplicate_token(source, access, attributes, level, token_type,
                        output_pointer):
        assign_handle(output_pointer, 202)
        return True

    def close_handle(handle):
        closed_handles.append(handle.value)
        return False

    secure._kernel32 = SimpleNamespace(
        GetCurrentProcess=lambda: wintypes.HANDLE(999),
        CloseHandle=close_handle,
    )
    secure._advapi32 = SimpleNamespace(
        OpenProcessToken=open_process_token,
        DuplicateTokenEx=duplicate_token,
    )
    monkeypatch.setattr(ctypes, "get_last_error", lambda: 6)

    with pytest.raises(RuntimeError, match="body operation failed") as caught:
        with secure._impersonation_token():
            raise RuntimeError("body operation failed")

    assert closed_handles == [202, 101]

    cleanup_errors = caught.value._eva_cleanup_errors
    assert len(cleanup_errors) == 2
    assert "impersonation token" in str(cleanup_errors[0])
    assert "source process token" in str(cleanup_errors[1])
    assert all("WinError 6" in str(error) for error in cleanup_errors)

def test_cleanup_failure_is_raised_without_primary_exception():
    with pytest.raises(SecureFilesystemUnavailable):
        WindowsSecureDirectory._report_cleanup_failure(
            None, "test cleanup failure"
        )


def test_cleanup_failure_is_attached_to_primary_exception():
    primary = RuntimeError("original operation failed")

    WindowsSecureDirectory._report_cleanup_failure(
        primary, "test handle close failed"
    )

    assert str(primary) == "original operation failed"
    errors = getattr(primary, "_eva_cleanup_errors")
    assert len(errors) == 1
    assert "test handle close failed" in str(errors[0])


def _accesscheck_test_instance(monkeypatch, access_check):
    secure = WindowsSecureDirectory.__new__(WindowsSecureDirectory)
    secure._advapi32 = SimpleNamespace(AccessCheck=access_check)

    @contextmanager
    def fake_impersonation_token():
        yield wintypes.HANDLE(123)

    monkeypatch.setattr(
        secure, "_impersonation_token", fake_impersonation_token
    )
    return secure


def test_accesscheck_allows_required_directory_access(monkeypatch):
    def fake_access_check(*args):
        granted_access = ctypes.cast(
            args[6], ctypes.POINTER(wintypes.DWORD)
        )
        access_status = ctypes.cast(
            args[7], ctypes.POINTER(wintypes.BOOL)
        )
        granted_access.contents.value = args[2]
        access_status.contents.value = True
        return 1

    secure = _accesscheck_test_instance(monkeypatch, fake_access_check)

    secure._check_effective_directory_access(ctypes.c_void_p(456))


def test_accesscheck_rejects_incomplete_granted_access(monkeypatch):
    def fake_access_check(*args):
        granted_access = ctypes.cast(
            args[6], ctypes.POINTER(wintypes.DWORD)
        )
        access_status = ctypes.cast(
            args[7], ctypes.POINTER(wintypes.BOOL)
        )
        granted_access.contents.value = 0
        access_status.contents.value = True
        return 1

    secure = _accesscheck_test_instance(monkeypatch, fake_access_check)

    with pytest.raises(SecureFilesystemError, match="lacks required"):
        secure._check_effective_directory_access(ctypes.c_void_p(456))


def test_inherit_only_ace_does_not_apply_to_current_object():
    assert not WindowsSecureDirectory._ace_applies_to_object(0x08)


def test_inheritable_ace_without_inherit_only_applies_to_current_object():
    assert WindowsSecureDirectory._ace_applies_to_object(0x03)


def test_accesscheck_rejects_missing_directory_access(monkeypatch):
    def fake_access_check(*args):
        access_status = ctypes.cast(
            args[7], ctypes.POINTER(wintypes.BOOL)
        )
        access_status.contents.value = False
        return 1

    secure = _accesscheck_test_instance(monkeypatch, fake_access_check)

    with pytest.raises(SecureFilesystemError, match="lacks required"):
        secure._check_effective_directory_access(ctypes.c_void_p(456))


def test_accesscheck_api_failure_fails_closed(monkeypatch):
    def fake_access_check(*args):
        return 0

    secure = _accesscheck_test_instance(monkeypatch, fake_access_check)
    monkeypatch.setattr(ctypes, "get_last_error", lambda: 5)

    with pytest.raises(
        SecureFilesystemUnavailable, match="AccessCheck API failed"
    ):
        secure._check_effective_directory_access(ctypes.c_void_p(456))

def test_constructor_cleanup_preserves_primary_error_and_retries_failed_handle(
    monkeypatch,
):
    import ctypes
    from types import SimpleNamespace
    from ctypes import wintypes

    from src.deep_learning.windows_secure_fs import WindowsSecureDirectory

    first_handle = wintypes.HANDLE(601)
    second_handle = wintypes.HANDLE(602)

    handles = iter([first_handle, second_handle])
    close_results = {601: 0, 602: -1}
    close_attempts = []
    inspect_count = 0

    def fake_nt_close(handle):
        value = handle.value
        close_attempts.append(value)
        return close_results[value]

    fake_dll = SimpleNamespace(NtClose=fake_nt_close)

    monkeypatch.setattr(
        ctypes,
        "WinDLL",
        lambda *args, **kwargs: fake_dll,
    )
    monkeypatch.setattr(
        WindowsSecureDirectory,
        "_configure_apis",
        lambda self: None,
    )
    monkeypatch.setattr(
        WindowsSecureDirectory,
        "_parse_root",
        staticmethod(lambda root: ("storage",)),
    )
    monkeypatch.setattr(
        WindowsSecureDirectory,
        "_nt_open",
        lambda self, *args, **kwargs: next(handles),
    )

    def fake_inspect(self, handle):
        nonlocal inspect_count
        inspect_count += 1
        if inspect_count == 2:
            raise RuntimeError("original initialization failure")

    monkeypatch.setattr(
        WindowsSecureDirectory,
        "_inspect_directory",
        fake_inspect,
    )

    secure = WindowsSecureDirectory.__new__(WindowsSecureDirectory)

    with pytest.raises(
        RuntimeError,
        match="original initialization failure",
    ) as caught:
        secure.__init__("C:\\storage")

    assert close_attempts == [602, 601]
    assert len(caught.value._eva_cleanup_errors) == 1
    assert "Constructor handle cleanup failed" in str(
        caught.value._eva_cleanup_errors[0]
    )
    assert secure._handles == [second_handle]

    close_results[602] = 0
    close_attempts.clear()

    secure.close()

    assert close_attempts == [602]
    assert secure._handles == []
    assert secure._orphan_handles == []