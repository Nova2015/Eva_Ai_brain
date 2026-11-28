"""Handle-relative Windows filesystem backend.

Uses pinned directory handles and native relative opens. File I/O,
rename, replacement and deletion are performed through native handles.
Security checks remain conservative and fail closed where unsupported.
"""
from __future__ import annotations

import ctypes
import os
import sys
import threading
from contextlib import contextmanager
from ctypes import wintypes
from typing import Iterator

from .secure_fs import (
    FileIdentity,
    IdentityMismatchError,
    SecureFilesystemError,
    SecureFilesystemUnavailable,
)


class _UNICODE_STRING(ctypes.Structure):
    _fields_ = [
        ("Length", wintypes.USHORT),
        ("MaximumLength", wintypes.USHORT),
        ("Buffer", wintypes.LPWSTR),
    ]


class _SID_AND_ATTRIBUTES(ctypes.Structure):
    _fields_ = [
        ("Sid", ctypes.c_void_p),
        ("Attributes", wintypes.DWORD),
    ]


class _TOKEN_USER(ctypes.Structure):
    _fields_ = [
        ("User", _SID_AND_ATTRIBUTES),
    ]


class _OBJECT_ATTRIBUTES(ctypes.Structure):
    _fields_ = [
        ("Length", wintypes.ULONG),
        ("RootDirectory", wintypes.HANDLE),
        ("ObjectName", ctypes.POINTER(_UNICODE_STRING)),
        ("Attributes", wintypes.ULONG),
        ("SecurityDescriptor", ctypes.c_void_p),
        ("SecurityQualityOfService", ctypes.c_void_p),
    ]


class _IO_STATUS_UNION(ctypes.Union):
    _fields_ = [
        ("Status", ctypes.c_long),
        ("Pointer", ctypes.c_void_p),
    ]


class _IO_STATUS_BLOCK(ctypes.Structure):
    _fields_ = [
        ("u", _IO_STATUS_UNION),
        ("Information", ctypes.c_size_t),
    ]


class _FILE_ATTRIBUTE_TAG_INFO(ctypes.Structure):
    _fields_ = [
        ("FileAttributes", wintypes.DWORD),
        ("ReparseTag", wintypes.DWORD),
    ]


class _FILE_ID_INFO(ctypes.Structure):
    _fields_ = [
        ("VolumeSerialNumber", ctypes.c_ulonglong),
        ("FileId", ctypes.c_ubyte * 16),
    ]


class _ACL_SIZE_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("AceCount", wintypes.DWORD),
        ("AclBytesInUse", wintypes.DWORD),
        ("AclBytesFree", wintypes.DWORD),
    ]


class _GENERIC_MAPPING(ctypes.Structure):
    _fields_ = [
        ("GenericRead", wintypes.DWORD),
        ("GenericWrite", wintypes.DWORD),
        ("GenericExecute", wintypes.DWORD),
        ("GenericAll", wintypes.DWORD),
    ]


class _LUID(ctypes.Structure):
    _fields_ = [
        ("LowPart", wintypes.DWORD),
        ("HighPart", ctypes.c_long),
    ]


class _LUID_AND_ATTRIBUTES(ctypes.Structure):
    _fields_ = [
        ("Luid", _LUID),
        ("Attributes", wintypes.DWORD),
    ]


class _FILE_RENAME_INFO(ctypes.Structure):
    _fields_ = [
        ("ReplaceIfExists", ctypes.c_ubyte),
        ("RootDirectory", wintypes.HANDLE),
        ("FileNameLength", wintypes.DWORD),
        ("FileName", ctypes.c_wchar * 1),
    ]


class _FILE_DISPOSITION_INFO(ctypes.Structure):
    _fields_ = [
        ("DeleteFile", ctypes.c_ubyte),
    ]


class _PRIVILEGE_SET(ctypes.Structure):
    _fields_ = [
        ("PrivilegeCount", wintypes.DWORD),
        ("Control", wintypes.DWORD),
        ("Privilege", _LUID_AND_ATTRIBUTES * 1),
    ]


class WindowsSecureDirectory:
    """Pinned Windows directory handles with conservative root ACL checks."""

    _OBJ_CASE_INSENSITIVE = 0x40
    _FILE_READ_ATTRIBUTES = 0x80
    _FILE_READ_DATA = 0x00000001
    _FILE_WRITE_DATA = 0x00000002
    _READ_CONTROL = 0x00020000
    _SYNCHRONIZE = 0x00100000
    _SHARE_ALL = 0x7
    _FILE_OPEN = 1
    _FILE_DIRECTORY_FILE = 0x1
    _FILE_SYNCHRONOUS_IO_NONALERT = 0x20
    _FILE_OPEN_REPARSE_POINT = 0x00200000

    # SetFileInformationByHandle information classes.
    _FILE_RENAME_INFO_CLASS = 10
    _FILE_DISPOSITION_INFO_CLASS = 4
    _FILE_DELETE_ACCESS = 0x00010000

    _IO_CHUNK_SIZE = 1024 * 1024
    _MAX_READ_SIZE = 64 * 1024 * 1024

    _FILE_ATTRIBUTE_DIRECTORY = 0x10
    _FILE_ATTRIBUTE_REPARSE_POINT = 0x400

    _FILE_INFO_ATTRIBUTE_TAG = 9
    _FILE_INFO_ID = 18

    _SE_FILE_OBJECT = 1
    _OWNER_SECURITY_INFORMATION = 0x1
    _DACL_SECURITY_INFORMATION = 0x4
    _ACL_SIZE_INFORMATION_CLASS = 2

    # Directory rights required by the secure artifact workflow.
    _FILE_LIST_DIRECTORY = 0x00000001
    _FILE_ADD_FILE = 0x00000002
    _FILE_ADD_SUBDIRECTORY = 0x00000004
    _FILE_TRAVERSE = 0x00000020

    _ROOT_REQUIRED_ACCESS = (
        _FILE_LIST_DIRECTORY
        | _FILE_ADD_FILE
        | _FILE_ADD_SUBDIRECTORY
        | _FILE_READ_ATTRIBUTES
        | _FILE_TRAVERSE
    )

    # Windows file-object GENERIC_MAPPING.
    _FILE_GENERIC_MAPPING = _GENERIC_MAPPING(
        0x00120089,  # GENERIC_READ
        0x00120116,  # GENERIC_WRITE
        0x001200A0,  # GENERIC_EXECUTE
        0x001F01FF,  # GENERIC_ALL
    )

    # Principals that must not have broad write rights on the trusted root.
    _BROAD_SIDS = {
        "S-1-1-0",       # Everyone
        "S-1-5-11",      # Authenticated Users
        "S-1-5-32-545",  # BUILTIN\\Users
    }

    # File/directory mutation and security-descriptor write rights.
    _WRITE_MASK = (
        0x00000002  # FILE_ADD_FILE / FILE_WRITE_DATA
        | 0x00000004  # FILE_ADD_SUBDIRECTORY / FILE_APPEND_DATA
        | 0x00000010  # FILE_WRITE_EA
        | 0x00000040  # FILE_DELETE_CHILD
        | 0x00000100  # FILE_WRITE_ATTRIBUTES
        | 0x00010000  # DELETE
        | 0x00040000  # WRITE_DAC
        | 0x00080000  # WRITE_OWNER
        | 0x40000000  # GENERIC_WRITE
        | 0x10000000  # GENERIC_ALL
    )

    def __init__(self, root: str | os.PathLike[str]) -> None:
        if os.name != "nt":
            raise SecureFilesystemUnavailable(
                "WindowsSecureDirectory is available only on Windows."
            )

        self._kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self._advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)
        self._ntdll = ctypes.WinDLL("ntdll")

        self._configure_apis()
        self._handles: list[int] = []
        self._file_handles: dict[int, FileIdentity] = {}
        # Granted data-operation rights are tracked separately from identity.
        # This prevents an owned read-only handle from being used for writes.
        self._file_handle_access: dict[int, int] = {}
        self._orphan_handles = []
        self._closed = False
        self._lock = threading.RLock()

        try:
            components = self._parse_root(root)
            drive = os.fspath(root).replace("/", "\\")[:2]

            current = self._nt_open(
                "\\??\\" + drive + "\\",
                directory_access=(
                    self._FILE_ADD_FILE if not components else 0
                ),
            )
            self._handles.append(current)
            self._inspect_directory(current)

            for index, component in enumerate(components):
                is_storage_root = index == len(components) - 1
                child = self._nt_open(
                    component,
                    current,
                    directory=True,
                    directory_access=(
                        self._FILE_ADD_FILE if is_storage_root else 0
                    ),
                )
                self._handles.append(child)
                self._inspect_directory(child)
                current = child

            # Validate the DACL on the actual pinned storage-root object.
            self._validate_root_dacl(current)
            self._root_identity = self._identity(current)

        except BaseException as primary:
            try:
                self.close()
            except BaseException as cleanup:
                self._report_cleanup_failure(
                    primary,
                    f"Constructor handle cleanup failed: {cleanup}",
                )
            raise

    @staticmethod
    def _parse_root(root: str | os.PathLike[str]) -> tuple[str, ...]:
        try:
            raw = os.fspath(root)
        except TypeError as exc:
            raise SecureFilesystemError("Invalid Windows root path.") from exc

        if not isinstance(raw, str) or not raw or "\x00" in raw:
            raise SecureFilesystemError("Invalid Windows root path.")

        raw = raw.replace("/", "\\")

        if raw.startswith("\\\\") or raw.startswith("\\?\\") or raw.startswith("\\.\\"):
            raise SecureFilesystemError(
                "UNC and device paths are not supported by this backend."
            )

        if len(raw) < 3 or raw[1:3] != ":\\" or not raw[0].isalpha():
            raise SecureFilesystemError(
                "An absolute drive-letter path is required."
            )

        tail = raw[3:].rstrip("\\")
        if not tail:
            return ()

        parts = tuple(tail.split("\\"))
        if any(part in ("", ".", "..") for part in parts):
            raise SecureFilesystemError(
                "Empty or traversal path component rejected."
            )

        return parts

    def _configure_apis(self) -> None:
        k = self._kernel32
        a = self._advapi32
        n = self._ntdll

        k.GetCurrentProcess.argtypes = []
        k.GetCurrentProcess.restype = wintypes.HANDLE
        k.CloseHandle.argtypes = [wintypes.HANDLE]
        k.CloseHandle.restype = wintypes.BOOL

        k.ReadFile.argtypes = [
            wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD,
            ctypes.POINTER(wintypes.DWORD), ctypes.c_void_p,
        ]
        k.ReadFile.restype = wintypes.BOOL

        k.WriteFile.argtypes = [
            wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD,
            ctypes.POINTER(wintypes.DWORD), ctypes.c_void_p,
        ]
        k.WriteFile.restype = wintypes.BOOL

        k.FlushFileBuffers.argtypes = [wintypes.HANDLE]
        k.FlushFileBuffers.restype = wintypes.BOOL

        k.SetFileInformationByHandle.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            ctypes.c_void_p,
            wintypes.DWORD,
        ]
        k.SetFileInformationByHandle.restype = wintypes.BOOL

        a.OpenProcessToken.argtypes = [
            wintypes.HANDLE,
            wintypes.DWORD,
            ctypes.POINTER(wintypes.HANDLE),
        ]
        a.OpenProcessToken.restype = wintypes.BOOL
        a.GetTokenInformation.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            ctypes.c_void_p,
            wintypes.DWORD,
            ctypes.POINTER(wintypes.DWORD),
        ]
        a.GetTokenInformation.restype = wintypes.BOOL

        a.DuplicateTokenEx.argtypes = [
            wintypes.HANDLE,
            wintypes.DWORD,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(wintypes.HANDLE),
        ]
        a.DuplicateTokenEx.restype = wintypes.BOOL

        a.AccessCheck.argtypes = [
            ctypes.c_void_p,
            wintypes.HANDLE,
            wintypes.DWORD,
            ctypes.POINTER(_GENERIC_MAPPING),
            ctypes.c_void_p,
            ctypes.POINTER(wintypes.DWORD),
            ctypes.POINTER(wintypes.DWORD),
            ctypes.POINTER(wintypes.BOOL),
        ]
        a.AccessCheck.restype = wintypes.BOOL

        n.NtCreateFile.restype = ctypes.c_long
        n.NtCreateFile.argtypes = [
            ctypes.POINTER(wintypes.HANDLE),
            wintypes.ULONG,
            ctypes.POINTER(_OBJECT_ATTRIBUTES),
            ctypes.POINTER(_IO_STATUS_BLOCK),
            ctypes.c_void_p,
            wintypes.ULONG,
            wintypes.ULONG,
            wintypes.ULONG,
            wintypes.ULONG,
            ctypes.c_void_p,
            wintypes.ULONG,
        ]

        n.NtSetInformationFile.restype = ctypes.c_long
        n.NtSetInformationFile.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(_IO_STATUS_BLOCK),
            ctypes.c_void_p,
            wintypes.ULONG,
            ctypes.c_int,
        ]

        n.NtClose.argtypes = [wintypes.HANDLE]
        n.NtClose.restype = ctypes.c_long

        k.GetFileInformationByHandleEx.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            ctypes.c_void_p,
            wintypes.DWORD,
        ]
        k.GetFileInformationByHandleEx.restype = wintypes.BOOL

        a.GetSecurityInfo.argtypes = [
            wintypes.HANDLE, ctypes.c_int, wintypes.DWORD,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
        ]
        a.GetSecurityInfo.restype = wintypes.DWORD

        a.GetAclInformation.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p,
            wintypes.DWORD, ctypes.c_int,
        ]
        a.GetAclInformation.restype = wintypes.BOOL

        a.GetAce.argtypes = [
            ctypes.c_void_p, wintypes.DWORD,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        a.GetAce.restype = wintypes.BOOL

        a.ConvertSidToStringSidW.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(wintypes.LPWSTR)
        ]
        a.ConvertSidToStringSidW.restype = wintypes.BOOL

        a.IsValidSid.argtypes = [ctypes.c_void_p]
        a.IsValidSid.restype = wintypes.BOOL

        k.LocalFree.argtypes = [ctypes.c_void_p]
        k.LocalFree.restype = ctypes.c_void_p

    def _nt_open(
        self,
        name: str,
        parent=None,
        *,
        directory: bool = True,
        create: bool = False,
        directory_access: int = 0,
        file_access: int | None = None,
    ):
        name_buffer = ctypes.create_unicode_buffer(name)
        encoded_length = len(name.encode("utf-16-le"))

        unicode_name = _UNICODE_STRING(
            encoded_length,
            encoded_length + 2,
            ctypes.cast(name_buffer, wintypes.LPWSTR),
        )

        attributes = _OBJECT_ATTRIBUTES(
            ctypes.sizeof(_OBJECT_ATTRIBUTES),
            parent,
            ctypes.pointer(unicode_name),
            self._OBJ_CASE_INSENSITIVE,
            None,
            None,
        )

        io_status = _IO_STATUS_BLOCK()
        handle = wintypes.HANDLE()

        if directory:
            desired_access = (
                self._FILE_READ_ATTRIBUTES
                | self._READ_CONTROL
                | self._SYNCHRONIZE
                | directory_access
            )
            options = (
                self._FILE_DIRECTORY_FILE
                | self._FILE_SYNCHRONOUS_IO_NONALERT
                | self._FILE_OPEN_REPARSE_POINT
            )
        else:
            if file_access is None:
                file_access = (
                    self._FILE_WRITE_DATA
                    if create else self._FILE_READ_DATA
                )
            desired_access = (
                file_access
                | self._FILE_READ_ATTRIBUTES
                | self._READ_CONTROL
                | self._SYNCHRONIZE
            )
            options = (
                0x00000040  # FILE_NON_DIRECTORY_FILE
                | self._FILE_SYNCHRONOUS_IO_NONALERT
                | self._FILE_OPEN_REPARSE_POINT
            )

        status = self._ntdll.NtCreateFile(
            ctypes.byref(handle),
            desired_access,
            ctypes.byref(attributes),
            ctypes.byref(io_status),
            None,
            0x00000080 if create else 0,  # FILE_ATTRIBUTE_NORMAL
            self._SHARE_ALL,
            2 if create else self._FILE_OPEN,  # FILE_CREATE / FILE_OPEN
            options,
            None,
            0,
        )

        if status < 0:
            ntstatus = status & 0xFFFFFFFF
            message = (
                f"NtCreateFile failed for {name!r}: "
                f"NTSTATUS=0x{ntstatus:08X}"
            )

            if ntstatus == 0xC0000035:  # STATUS_OBJECT_NAME_COLLISION
                raise FileExistsError(message)

            if ntstatus in (
                0xC0000034,  # STATUS_OBJECT_NAME_NOT_FOUND
                0xC000003A,  # STATUS_OBJECT_PATH_NOT_FOUND
            ):
                raise FileNotFoundError(message)

            if ntstatus == 0xC0000022:  # STATUS_ACCESS_DENIED
                raise PermissionError(message)

            raise SecureFilesystemUnavailable(message)

        return handle

    @staticmethod
    def _relative_parts(value) -> tuple[str, ...]:
        try:
            raw = os.fspath(value)
        except TypeError as exc:
            raise SecureFilesystemError("Invalid relative file path.") from exc

        if not isinstance(raw, str) or not raw or "\x00" in raw:
            raise SecureFilesystemError("Invalid relative file path.")

        if raw.startswith("/") or "\\" in raw:
            raise SecureFilesystemError(
                "Absolute or non-portable file path rejected."
            )

        parts = tuple(raw.split("/"))
        if any(part in ("", ".", "..") for part in parts):
            raise SecureFilesystemError(
                "Empty or traversal path component rejected."
            )

        return parts

    def _nt_close_status(self, handle, resource: str) -> str | None:
        """Close one native handle and return a diagnostic on failure."""
        status = self._ntdll.NtClose(handle)
        if status < 0:
            return (
                f"{resource}: NTSTATUS=0x"
                f"{status & 0xFFFFFFFF:08X}"
            )
        return None

    def _close_handles(self, handles, resource: str) -> None:
        """Attempt every close and preserve any active primary exception."""
        failures = []
        failed_handles = []

        for handle in handles:
            failure = self._nt_close_status(handle, resource)
            if failure:
                failures.append(failure)
                failed_handles.append(handle)

        if failed_handles:
            self._orphan_handles.extend(failed_handles)

        if failures:
            self._report_cleanup_failure(
                sys.exc_info()[1],
                "Windows handle cleanup failed: " + "; ".join(failures),
            )

    def _open_file(
        self,
        relative_path,
        *,
        create: bool,
        file_access: int | None = None,
    ):
        parts = self._relative_parts(relative_path)
        temporary_handles = []

        with self._lock:
            self._ensure_open()
            parent = self._handles[-1]

            try:
                for index, component in enumerate(parts[:-1]):
                    is_final_parent = index == len(parts) - 2
                    child = self._nt_open(
                        component,
                        parent,
                        directory=True,
                        directory_access=(
                            self._FILE_ADD_FILE
                            if create and is_final_parent
                            else 0
                        ),
                    )
                    temporary_handles.append(child)
                    self._inspect_directory(child)
                    self._validate_root_dacl(child)
                    parent = child

                file_handle = self._nt_open(
                    parts[-1],
                    parent,
                    directory=False,
                    create=create,
                    file_access=file_access,
                )

                try:
                    self._inspect_regular_file(file_handle)
                    identity = self._identity(file_handle, object_type="file")
                    handle_value = int(file_handle.value)

                    if handle_value in self._file_handles:
                        raise SecureFilesystemError(
                            "Windows returned an already-registered handle."
                        )

                    effective_file_access = (
                        file_access
                        if file_access is not None
                        else (
                            self._FILE_WRITE_DATA
                            if create
                            else self._FILE_READ_DATA
                        )
                    )
                    self._file_handles[handle_value] = identity
                    self._file_handle_access[handle_value] = effective_file_access
                    return handle_value

                except BaseException:
                    self._close_handles([file_handle], "file handle")
                    raise

            finally:
                self._close_handles(
                    reversed(temporary_handles),
                    "temporary directory handle",
                )

    def open_read(self, relative_path) -> int:
        """Open an existing regular file relative to the pinned root."""
        return self._open_file(
            relative_path,
            create=False,
            file_access=self._FILE_READ_DATA,
        )

    def create_exclusive(self, relative_path) -> int:
        """Create a new regular file without overwriting an existing entry."""
        return self._open_file(
            relative_path,
            create=True,
            file_access=self._FILE_WRITE_DATA,
        )

    def close_file(self, handle: int) -> None:
        """Close only a file HANDLE owned by this directory instance."""
        if not isinstance(handle, int) or handle <= 0:
            raise SecureFilesystemError("Invalid Windows file handle.")

        with self._lock:
            self._ensure_open()

            if handle not in self._file_handles:
                raise SecureFilesystemError(
                    "File handle is not owned by this directory instance."
                )

            status = self._ntdll.NtClose(wintypes.HANDLE(handle))
            if status < 0:
                raise SecureFilesystemError(
                    f"NtClose failed: NTSTATUS=0x{status & 0xFFFFFFFF:08X}"
                )

            del self._file_handles[handle]
            self._file_handle_access.pop(handle, None)

    def _require_file_access(
        self,
        handle: int,
        required_access: int,
        operation: str,
    ) -> None:
        """Require the requested data right on an owned open file handle."""
        granted_access = self._file_handle_access.get(handle)
        if granted_access is None:
            raise SecureFilesystemError(
                "File handle access metadata is unavailable."
            )
        if granted_access & required_access != required_access:
            raise SecureFilesystemError(
                f"File handle does not grant {operation} access."
            )

    def read_file(self, handle: int, size: int) -> bytes:
        """Read up to size bytes from an owned read handle."""
        if not isinstance(handle, int) or isinstance(handle, bool) or handle <= 0:
            raise SecureFilesystemError("Invalid Windows file handle.")
        if not isinstance(size, int) or isinstance(size, bool):
            raise SecureFilesystemError("Read size must be an integer.")
        if size < 0 or size > self._MAX_READ_SIZE:
            raise SecureFilesystemError(
                f"Read size must be between 0 and {self._MAX_READ_SIZE} bytes."
            )

        with self._lock:
            self._ensure_open()
            self.get_file_identity(handle)
            self._require_file_access(
                handle, self._FILE_READ_DATA, "read"
            )

            if size == 0:
                return b""

            result = bytearray()
            remaining = size

            while remaining:
                amount = min(remaining, self._IO_CHUNK_SIZE)
                buffer = ctypes.create_string_buffer(amount)
                bytes_read = wintypes.DWORD()

                ok = self._kernel32.ReadFile(
                    wintypes.HANDLE(handle),
                    ctypes.cast(buffer, ctypes.c_void_p),
                    amount,
                    ctypes.byref(bytes_read),
                    None,
                )
                if not ok:
                    error = ctypes.get_last_error()
                    if error == 38:  # ERROR_HANDLE_EOF
                        break
                    raise SecureFilesystemUnavailable(
                        f"ReadFile failed: {ctypes.WinError(error)}"
                    )

                count = int(bytes_read.value)
                if count > amount:
                    raise SecureFilesystemUnavailable(
                        "ReadFile returned an invalid byte count."
                    )
                if count == 0:
                    break

                result.extend(buffer.raw[:count])
                remaining -= count

            return bytes(result)

    def write_file(self, handle: int, data) -> int:
        """Write all bytes to an owned writable file handle."""
        if not isinstance(handle, int) or isinstance(handle, bool) or handle <= 0:
            raise SecureFilesystemError("Invalid Windows file handle.")

        try:
            view = memoryview(data)
            if not view.contiguous:
                raise SecureFilesystemError(
                    "Write data must use a contiguous bytes-like buffer."
                )
            view = view.cast("B")
        except (TypeError, ValueError) as exc:
            raise SecureFilesystemError(
                "Write data must be a contiguous bytes-like buffer."
            ) from exc

        with self._lock:
            self._ensure_open()
            self.get_file_identity(handle)
            self._require_file_access(
                handle, self._FILE_WRITE_DATA, "write"
            )

            total = len(view)
            offset = 0

            while offset < total:
                end = min(offset + self._IO_CHUNK_SIZE, total)
                chunk = bytes(view[offset:end])
                buffer = ctypes.create_string_buffer(chunk, len(chunk))
                bytes_written = wintypes.DWORD()

                ok = self._kernel32.WriteFile(
                    wintypes.HANDLE(handle),
                    ctypes.cast(buffer, ctypes.c_void_p),
                    len(chunk),
                    ctypes.byref(bytes_written),
                    None,
                )
                if not ok:
                    error = ctypes.get_last_error()
                    raise SecureFilesystemUnavailable(
                        f"WriteFile failed: {ctypes.WinError(error)}"
                    )

                count = int(bytes_written.value)
                if count <= 0 or count > len(chunk):
                    raise SecureFilesystemUnavailable(
                        "WriteFile made no progress or returned an invalid count."
                    )

                offset += count

            return offset

    def flush_file(self, handle: int) -> None:
        """Flush buffered data for an owned writable file handle."""
        if not isinstance(handle, int) or isinstance(handle, bool) or handle <= 0:
            raise SecureFilesystemError("Invalid Windows file handle.")

        with self._lock:
            self._ensure_open()
            self.get_file_identity(handle)
            self._require_file_access(
                handle, self._FILE_WRITE_DATA, "flush"
            )

            if not self._kernel32.FlushFileBuffers(wintypes.HANDLE(handle)):
                error = ctypes.get_last_error()
                raise SecureFilesystemUnavailable(
                    f"FlushFileBuffers failed: {ctypes.WinError(error)}"
                )

    def _query_info(self, handle, info_class: int, structure):
        ok = self._kernel32.GetFileInformationByHandleEx(
            handle, info_class, ctypes.byref(structure), ctypes.sizeof(structure)
        )
        if not ok:
            raise SecureFilesystemUnavailable(
                f"GetFileInformationByHandleEx failed: "
                f"{ctypes.WinError(ctypes.get_last_error())}"
            )
        return structure

    def _inspect_directory(self, handle) -> None:
        info = self._query_info(
            handle, self._FILE_INFO_ATTRIBUTE_TAG, _FILE_ATTRIBUTE_TAG_INFO()
        )

        if not info.FileAttributes & self._FILE_ATTRIBUTE_DIRECTORY:
            raise SecureFilesystemError("A path component is not a directory.")

        if info.FileAttributes & self._FILE_ATTRIBUTE_REPARSE_POINT:
            raise SecureFilesystemError(
                "Reparse-point directory rejected."
            )

    def _identity(self, handle, object_type="directory") -> FileIdentity:
        info = self._query_info(
            handle, self._FILE_INFO_ID, _FILE_ID_INFO()
        )
        return FileIdentity(
            backend="windows",
            volume_id=int(info.VolumeSerialNumber),
            file_id=int.from_bytes(bytes(info.FileId), "little"),
            object_type=object_type,
        )

    def _inspect_regular_file(self, handle) -> None:
        info = self._query_info(
            handle, self._FILE_INFO_ATTRIBUTE_TAG, _FILE_ATTRIBUTE_TAG_INFO()
        )

        if info.FileAttributes & self._FILE_ATTRIBUTE_REPARSE_POINT:
            raise SecureFilesystemError("Reparse-point file rejected.")

        if info.FileAttributes & self._FILE_ATTRIBUTE_DIRECTORY:
            raise SecureFilesystemError("Directory supplied where file expected.")

        if info.FileAttributes & 0x40:  # FILE_ATTRIBUTE_DEVICE
            raise SecureFilesystemError("Device entry rejected.")

    def get_file_identity(self, handle: int) -> FileIdentity:
        """Verify ownership and return identity for an open file handle."""
        if not isinstance(handle, int) or handle <= 0:
            raise SecureFilesystemError("Invalid Windows file handle.")

        with self._lock:
            self._ensure_open()

            expected = self._file_handles.get(handle)
            if expected is None:
                raise SecureFilesystemError(
                    "File handle is not owned by this directory instance."
                )

            wrapped = wintypes.HANDLE(handle)
            self._inspect_regular_file(wrapped)
            actual = self._identity(wrapped, object_type="file")

            if actual != expected:
                raise IdentityMismatchError(
                    "Open file handle identity changed unexpectedly."
                )

            return actual

    @staticmethod
    def _report_cleanup_failure(active_exception, message: str) -> None:
        """Preserve the primary exception and retain cleanup diagnostics."""
        cleanup_error = SecureFilesystemUnavailable(message)

        if active_exception is None:
            raise cleanup_error

        errors = getattr(active_exception, "_eva_cleanup_errors", None)
        if errors is None:
            errors = []
            try:
                setattr(active_exception, "_eva_cleanup_errors", errors)
            except Exception:
                errors = None

        if errors is not None:
            errors.append(cleanup_error)

        add_note = getattr(active_exception, "add_note", None)
        if callable(add_note):
            add_note(f"Additional cleanup failure: {message}")

    def _local_free(self, pointer, resource: str) -> None:
        """Free LocalAlloc memory and check the LocalFree result."""
        if not pointer:
            return

        if self._kernel32.LocalFree(pointer):
            error = ctypes.get_last_error()
            self._report_cleanup_failure(
                sys.exc_info()[1],
                f"Could not free {resource}: WinError {error}",
            )

    def _sid_string(self, sid_pointer) -> str:
        if not sid_pointer:
            raise SecureFilesystemError("Empty SID pointer rejected.")

        if not self._advapi32.IsValidSid(sid_pointer):
            raise SecureFilesystemError("Invalid Windows SID rejected.")

        output = wintypes.LPWSTR()
        if not self._advapi32.ConvertSidToStringSidW(
            sid_pointer, ctypes.byref(output)
        ):
            raise SecureFilesystemUnavailable(
                "Could not convert an ACL SID to text."
            )
        try:
            return output.value
        finally:
            self._local_free(
                ctypes.cast(output, ctypes.c_void_p),
                "converted SID string",
            )

    @contextmanager
    def _impersonation_token(self) -> Iterator[wintypes.HANDLE]:
        """Yield a query-capable impersonation token and close both handles."""
        source = wintypes.HANDLE()
        duplicate = wintypes.HANDLE()

        # TOKEN_QUERY | TOKEN_DUPLICATE
        if not self._advapi32.OpenProcessToken(
            self._kernel32.GetCurrentProcess(),
            0x0008 | 0x0002,
            ctypes.byref(source),
        ):
            raise SecureFilesystemUnavailable(
                "Could not open process token for duplication: "
                f"WinError {ctypes.get_last_error()}"
            )

        try:
            # TOKEN_QUERY, SecurityImpersonation, TokenImpersonation
            if not self._advapi32.DuplicateTokenEx(
                source,
                0x0008,
                None,
                2,
                2,
                ctypes.byref(duplicate),
            ):
                raise SecureFilesystemUnavailable(
                    "Could not create impersonation token: "
                    f"WinError {ctypes.get_last_error()}"
                )

            try:
                yield duplicate
            finally:
                active_exception = sys.exc_info()[1]
                if duplicate.value and not self._kernel32.CloseHandle(duplicate):
                    error = ctypes.get_last_error()
                    self._report_cleanup_failure(
                        active_exception,
                        "Could not close impersonation token: "
                        f"WinError {error}",
                    )
        finally:
            active_exception = sys.exc_info()[1]
            if source.value and not self._kernel32.CloseHandle(source):
                error = ctypes.get_last_error()
                self._report_cleanup_failure(
                    active_exception,
                    "Could not close source process token: "
                    f"WinError {error}",
                )

    def _current_user_sid(self) -> str:
        """Return the current process token user SID without changing ACLs."""
        token = wintypes.HANDLE()

        if not self._advapi32.OpenProcessToken(
            self._kernel32.GetCurrentProcess(),
            0x0008,  # TOKEN_QUERY
            ctypes.byref(token),
        ):
            raise SecureFilesystemUnavailable(
                "Could not open current process token: "
                f"WinError {ctypes.get_last_error()}"
            )

        try:
            required = wintypes.DWORD()
            self._advapi32.GetTokenInformation(
                token, 1, None, 0, ctypes.byref(required)
            )

            if not required.value:
                raise SecureFilesystemUnavailable(
                    "Could not determine token user buffer size: "
                    f"WinError {ctypes.get_last_error()}"
                )

            buffer = ctypes.create_string_buffer(required.value)
            if not self._advapi32.GetTokenInformation(
                token,
                1,  # TokenUser
                ctypes.cast(buffer, ctypes.c_void_p),
                required.value,
                ctypes.byref(required),
            ):
                raise SecureFilesystemUnavailable(
                    "Could not read current process token user: "
                    f"WinError {ctypes.get_last_error()}"
                )

            token_user = ctypes.cast(
                buffer, ctypes.POINTER(_TOKEN_USER)
            ).contents
            if not token_user.User.Sid:
                raise SecureFilesystemUnavailable(
                    "Current process token returned an empty user SID."
                )

            return self._sid_string(token_user.User.Sid)
        finally:
            active_exception = sys.exc_info()[1]
            if not self._kernel32.CloseHandle(token):
                error = ctypes.get_last_error()
                self._report_cleanup_failure(
                    active_exception,
                    "Could not close current process token handle: "
                    f"WinError {error}",
                )

    def _check_effective_directory_access(self, descriptor) -> None:
        """Check required directory DACL rights for the current process token."""
        desired = self._ROOT_REQUIRED_ACCESS
        mapping = self._FILE_GENERIC_MAPPING

        with self._impersonation_token() as token:
            buffer_size = 1024

            for attempt in range(3):
                privilege_buffer = ctypes.create_string_buffer(buffer_size)
                privilege_length = wintypes.DWORD(buffer_size)
                granted_access = wintypes.DWORD()
                access_status = wintypes.BOOL()

                ok = self._advapi32.AccessCheck(
                    descriptor,
                    token,
                    desired,
                    ctypes.byref(mapping),
                    ctypes.cast(privilege_buffer, ctypes.c_void_p),
                    ctypes.byref(privilege_length),
                    ctypes.byref(granted_access),
                    ctypes.byref(access_status),
                )

                if ok:
                    if (
                        not access_status.value
                        or (granted_access.value & desired) != desired
                    ):
                        raise SecureFilesystemError(
                            "Current process token lacks required storage-root "
                            f"directory access: desired=0x{desired:08X}, "
                            f"granted=0x{granted_access.value:08X}"
                        )
                    return

                error = ctypes.get_last_error()

                # AccessCheck reports the required privilege-buffer size.
                if error == 122 and privilege_length.value > buffer_size:
                    if privilege_length.value > 65536:
                        raise SecureFilesystemUnavailable(
                            "AccessCheck requested an unexpectedly large "
                            "privilege buffer."
                        )
                    buffer_size = privilege_length.value
                    continue

                raise SecureFilesystemUnavailable(
                    f"AccessCheck API failed: WinError {error}"
                )

            raise SecureFilesystemUnavailable(
                "AccessCheck could not complete after privilege-buffer retries."
            )

    @staticmethod
    def _ace_applies_to_object(ace_flags: int) -> bool:
        """Return whether an ACE applies to the object carrying its DACL."""
        return not bool(ace_flags & 0x08)  # INHERIT_ONLY_ACE

    def _validate_root_dacl(self, handle) -> None:
        owner = ctypes.c_void_p()
        dacl = ctypes.c_void_p()
        descriptor = ctypes.c_void_p()

        try:
            status = self._advapi32.GetSecurityInfo(
                handle,
                self._SE_FILE_OBJECT,
                self._OWNER_SECURITY_INFORMATION | self._DACL_SECURITY_INFORMATION,
                ctypes.byref(owner),
                None,
                ctypes.byref(dacl),
                None,
                ctypes.byref(descriptor),
            )

            if status:
                raise SecureFilesystemUnavailable(
                    f"GetSecurityInfo failed: WinError {status}"
                )

            if not owner.value:
                raise SecureFilesystemError(
                    "Storage root has no readable owner SID."
                )

            # Validate owner SID, but do not require it to equal the
            # current user: SYSTEM and Administrators may be valid owners.
            self._owner_sid = self._sid_string(owner)

            # A NULL DACL grants unrestricted access: fail closed.
            if not dacl.value:
                raise SecureFilesystemError(
                    "Storage root has a NULL DACL; unrestricted access rejected."
                )

            size_info = _ACL_SIZE_INFORMATION()
            if not self._advapi32.GetAclInformation(
                dacl,
                ctypes.byref(size_info),
                ctypes.sizeof(size_info),
                self._ACL_SIZE_INFORMATION_CLASS,
            ):
                raise SecureFilesystemUnavailable(
                    "Could not inspect storage-root ACL entries."
                )

            for index in range(size_info.AceCount):
                ace = ctypes.c_void_p()
                if not self._advapi32.GetAce(
                    dacl, index, ctypes.byref(ace)
                ):
                    raise SecureFilesystemUnavailable(
                        "Could not read an ACL entry."
                    )

                address = ace.value
                ace_type = ctypes.c_ubyte.from_address(address).value
                ace_flags = ctypes.c_ubyte.from_address(address + 1).value

                # Standard ACCESS_ALLOWED_ACE and ACCESS_DENIED_ACE only.
                # Unknown/object/callback ACEs fail closed until explicitly
                # supported and tested.
                if ace_type not in (0, 1):
                    raise SecureFilesystemUnavailable(
                        f"Unsupported ACL ACE type: {ace_type}"
                    )

                if ace_type == 1:  # Deny ACE does not grant access.
                    continue

                # INHERIT_ONLY_ACE (0x08) does not apply to this object.
                if not self._ace_applies_to_object(ace_flags):
                    continue

                mask = ctypes.c_uint32.from_address(address + 4).value
                sid = ctypes.c_void_p(address + 8)
                sid_text = self._sid_string(sid)

                if (
                    sid_text in self._BROAD_SIDS
                    and mask & self._WRITE_MASK
                ):
                    raise SecureFilesystemError(
                        "Storage root grants write/security-modification "
                        f"rights to a broad principal: {sid_text}, "
                        f"mask=0x{mask:08X}"
                    )

            self._check_effective_directory_access(descriptor)

        finally:
            if descriptor.value:
                self._local_free(descriptor, "security descriptor")

    def _ensure_open(self) -> None:
        if self._closed:
            raise SecureFilesystemError("Directory handle is closed.")

    @contextmanager
    def _parent_handle(self, relative_path, *, directory_access=0):
        """Yield a leaf name and its pinned parent directory handle."""
        parts = self._relative_parts(relative_path)
        temporary_handles = []

        with self._lock:
            self._ensure_open()
            parent = self._handles[-1]

            try:
                for index, component in enumerate(parts[:-1]):
                    is_final_parent = index == len(parts) - 2
                    child = self._nt_open(
                        component,
                        parent,
                        directory=True,
                        directory_access=(
                            directory_access if is_final_parent else 0
                        ),
                    )
                    temporary_handles.append(child)
                    self._inspect_directory(child)
                    self._validate_root_dacl(child)
                    parent = child

                yield parts[-1], parent

            finally:
                self._close_handles(
                    reversed(temporary_handles),
                    "temporary parent directory handle",
                )

    def _close_owned_file(self, handle: int) -> None:
        """Close an owned file handle without losing retry information."""
        with self._lock:
            if handle not in self._file_handles:
                return

            status = self._ntdll.NtClose(wintypes.HANDLE(handle))
            if status < 0:
                self._report_cleanup_failure(
                    sys.exc_info()[1],
                    "File handle close failed: "
                    f"NTSTATUS=0x{status & 0xFFFFFFFF:08X}",
                )
                return

            del self._file_handles[handle]
            self._file_handle_access.pop(handle, None)

    def _raise_file_info_error(self, operation: str, error: int) -> None:
        message = f"{operation} failed: {ctypes.WinError(error)}"

        if error in (80, 183):
            raise FileExistsError(message)
        if error in (2, 3):
            raise FileNotFoundError(message)
        if error == 5:
            raise PermissionError(message)

        raise SecureFilesystemUnavailable(message)

    def rename(
        self,
        source_path,
        destination_path,
        *,
        replace: bool = False,
        expected_identity: FileIdentity | None = None,
    ) -> FileIdentity:
        """Atomically rename an owned regular file using native handles."""
        if not isinstance(replace, bool):
            raise SecureFilesystemError("replace must be a boolean.")

        source_handle = self._open_file(
            source_path,
            create=False,
            file_access=(
                self._FILE_READ_DATA | self._FILE_DELETE_ACCESS
            ),
        )

        try:
            source_identity = self.get_file_identity(source_handle)

            if expected_identity is not None:
                if source_identity != expected_identity:
                    raise IdentityMismatchError(
                        "Source file identity does not match expected identity."
                    )

            with self._parent_handle(
                destination_path,
                directory_access=self._FILE_ADD_FILE,
            ) as (destination_name, destination_parent):

                parent_identity = self._identity(
                    destination_parent,
                    object_type="directory",
                )

                if source_identity.volume_id != parent_identity.volume_id:
                    raise SecureFilesystemUnavailable(
                        "Cross-volume rename is not supported."
                    )

                if replace:
                    existing_handle = None
                    try:
                        existing_handle = self._nt_open(
                            destination_name,
                            destination_parent,
                            directory=False,
                            file_access=(
                                self._FILE_READ_DATA
                                | self._FILE_DELETE_ACCESS
                            ),
                        )
                    except FileNotFoundError:
                        pass
                    else:
                        try:
                            self._inspect_regular_file(existing_handle)
                            existing_identity = self._identity(
                                existing_handle,
                                object_type="file",
                            )
                            if existing_identity == source_identity:
                                raise SecureFilesystemError(
                                    "Rename source and destination "
                                    "refer to the same file."
                                )
                        finally:
                            self._close_handles(
                                [existing_handle],
                                "replacement target validation handle",
                            )

                encoded_name = destination_name.encode("utf-16-le")
                name_offset = _FILE_RENAME_INFO.FileName.offset

                buffer_size = (
                    ctypes.sizeof(_FILE_RENAME_INFO)
                    + len(encoded_name)
                )
                buffer = ctypes.create_string_buffer(buffer_size)

                info = ctypes.cast(
                    buffer,
                    ctypes.POINTER(_FILE_RENAME_INFO),
                ).contents
                info.ReplaceIfExists = 1 if replace else 0
                info.RootDirectory = destination_parent
                info.FileNameLength = len(encoded_name)

                ctypes.memmove(
                    ctypes.addressof(buffer) + name_offset,
                    encoded_name,
                    len(encoded_name),
                )

                io_status = _IO_STATUS_BLOCK()

                status = self._ntdll.NtSetInformationFile(
                    wintypes.HANDLE(source_handle),
                    ctypes.byref(io_status),
                    ctypes.cast(buffer, ctypes.c_void_p),
                    buffer_size,
                    self._FILE_RENAME_INFO_CLASS,
                )

                if status < 0:
                    ntstatus = status & 0xFFFFFFFF
                    message = (
                        "Native handle-based rename failed: "
                        f"NTSTATUS=0x{ntstatus:08X}"
                    )

                    if ntstatus == 0xC0000035:
                        raise FileExistsError(message)

                    if ntstatus in (
                        0xC0000034,
                        0xC000003A,
                    ):
                        raise FileNotFoundError(message)

                    if ntstatus == 0xC0000022:
                        raise PermissionError(message)

                    raise SecureFilesystemUnavailable(message)


                # Verify the destination entry resolves to the opened source.
                published_handle = self._nt_open(
                    destination_name,
                    destination_parent,
                    directory=False,
                    file_access=self._FILE_READ_DATA,
                )
                try:
                    self._inspect_regular_file(published_handle)
                    published_identity = self._identity(
                        published_handle,
                        object_type="file",
                    )
                    if published_identity != source_identity:
                        raise IdentityMismatchError(
                            "Published file identity differs from source."
                        )
                finally:
                    self._close_handles(
                        [published_handle],
                        "rename verification handle",
                    )

                return published_identity

        finally:
            self._close_owned_file(source_handle)

    def _delete_entry(
        self,
        relative_path,
        *,
        directory: bool,
        expected_identity: FileIdentity | None = None,
    ) -> None:
        """Delete one file or empty directory through its native handle."""
        with self._parent_handle(relative_path) as (name, parent):
            target_handle = self._nt_open(
                name,
                parent,
                directory=directory,
                directory_access=(
                    self._FILE_DELETE_ACCESS if directory else 0
                ),
                file_access=(
                    None if directory else self._FILE_DELETE_ACCESS
                ),
            )

            try:
                if directory:
                    self._inspect_directory(target_handle)
                    identity = self._identity(
                        target_handle,
                        object_type="directory",
                    )
                else:
                    self._inspect_regular_file(target_handle)
                    identity = self._identity(
                        target_handle,
                        object_type="file",
                    )

                if (
                    expected_identity is not None
                    and identity != expected_identity
                ):
                    raise IdentityMismatchError(
                        "Target identity does not match expected identity."
                    )

                disposition = _FILE_DISPOSITION_INFO(1)
                ok = self._kernel32.SetFileInformationByHandle(
                    target_handle,
                    self._FILE_DISPOSITION_INFO_CLASS,
                    ctypes.byref(disposition),
                    ctypes.sizeof(disposition),
                )

                if not ok:
                    self._raise_file_info_error(
                        "Handle-based deletion",
                        ctypes.get_last_error(),
                    )

            finally:
                self._close_handles(
                    [target_handle],
                    "delete target handle",
                )

    def delete_file(
        self,
        relative_path,
        *,
        expected_identity: FileIdentity | None = None,
    ) -> None:
        """Delete one regular file; never follows a reparse point."""
        self._delete_entry(
            relative_path,
            directory=False,
            expected_identity=expected_identity,
        )

    def delete_directory(
        self,
        relative_path,
        *,
        expected_identity: FileIdentity | None = None,
    ) -> None:
        """Delete one empty directory; recursive deletion is not supported."""
        self._delete_entry(
            relative_path,
            directory=True,
            expected_identity=expected_identity,
        )

    def get_identity(self) -> FileIdentity:
        with self._lock:
            self._ensure_open()
            current = self._identity(self._handles[-1])
            if current != self._root_identity:
                raise IdentityMismatchError(
                    "Pinned storage-root identity unexpectedly changed."
                )
            return current

    def verify_identity(self, expected: FileIdentity) -> None:
        actual = self.get_identity()
        if actual != expected:
            raise IdentityMismatchError(
                "Storage-root identity does not match the expected identity."
            )

    def close(self) -> None:
        with getattr(self, "_lock", threading.RLock()):
            if (
                getattr(self, "_closed", True)
                and not self._file_handles
                and not self._orphan_handles
                and not self._handles
            ):
                return

            # Block new operations, but allow subsequent close() retries.
            self._closed = True
            failures = []

            for handle in list(self._file_handles):
                status = self._ntdll.NtClose(wintypes.HANDLE(handle))
                if status < 0:
                    failures.append(
                        f"file handle 0x{handle:X}: "
                        f"NTSTATUS=0x{status & 0xFFFFFFFF:08X}"
                    )
                else:
                    del self._file_handles[handle]
                    self._file_handle_access.pop(handle, None)

            remaining_orphans = []
            for handle in self._orphan_handles:
                status = self._ntdll.NtClose(handle)
                if status < 0:
                    value = int(handle.value or 0)
                    failures.append(
                        f"orphan handle 0x{value:X}: "
                        f"NTSTATUS=0x{status & 0xFFFFFFFF:08X}"
                    )
                    remaining_orphans.append(handle)
            self._orphan_handles = remaining_orphans

            remaining_directories = []
            for handle in reversed(self._handles):
                status = self._ntdll.NtClose(handle)
                if status < 0:
                    value = int(handle.value or 0)
                    failures.append(
                        f"directory handle 0x{value:X}: "
                        f"NTSTATUS=0x{status & 0xFFFFFFFF:08X}"
                    )
                    remaining_directories.append(handle)

            self._handles = list(reversed(remaining_directories))

            if failures:
                raise SecureFilesystemError(
                    "One or more Windows handles could not be closed: "
                    + "; ".join(failures)
                )

    def __enter__(self) -> "WindowsSecureDirectory":
        self._ensure_open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
