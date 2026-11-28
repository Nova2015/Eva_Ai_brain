"""POSIX handle-relative filesystem operations for Deep Learning artifacts.

This module deliberately fails closed on Windows until a native,
handle-relative Windows backend is implemented.

Security model:
- The configured root is opened component-by-component without following links.
- All operations below that root are relative to pinned directory descriptors.
- Write operations require private directories owned by the current user.
- Exclusive creation never overwrites an existing entry.
- Atomic publication uses same-directory operations.
"""
from __future__ import annotations

import errno
import ctypes
from dataclasses import dataclass
import os
import secrets
import stat
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


class SecureFilesystemError(RuntimeError):
    """A filesystem operation could not meet the security contract."""


class SecureFilesystemUnavailable(SecureFilesystemError):
    """The platform cannot provide the required secure filesystem primitives."""


def _require_posix() -> None:
    if os.name != "posix":
        raise SecureFilesystemUnavailable(
            "SecureDirectory currently requires the POSIX backend; "
            "no path-based fallback is permitted."
        )

    required = ("O_DIRECTORY", "O_NOFOLLOW")
    if any(not hasattr(os, item) for item in required):
        raise SecureFilesystemUnavailable(
            "This POSIX platform lacks required directory/no-follow flags."
        )

    if not os.open in getattr(os, "supports_dir_fd", set()):
        raise SecureFilesystemUnavailable(
            "This Python build does not support directory-relative os.open."
        )


def _parts(value: str | os.PathLike[str]) -> tuple[str, ...]:
    try:
        raw = os.fspath(value)
    except TypeError as exc:
        raise SecureFilesystemError("Invalid relative path.") from exc

    if not isinstance(raw, str) or not raw or "\x00" in raw:
        raise SecureFilesystemError("Invalid relative path.")

    # Use a portable relative-path contract, even on POSIX.
    if raw.startswith("/") or "\\" in raw:
        raise SecureFilesystemError("Absolute or non-portable path rejected.")

    items = tuple(raw.split("/"))
    if any(item in ("", ".", "..") for item in items):
        raise SecureFilesystemError("Empty or traversal path component rejected.")

    return items


def _check_private_directory(fd: int) -> None:
    info = os.fstat(fd)
    if not stat.S_ISDIR(info.st_mode):
        raise SecureFilesystemError("Expected a directory.")

    if info.st_uid != os.geteuid():
        raise SecureFilesystemError("Write directory is not owned by this user.")

    if stat.S_IMODE(info.st_mode) & 0o022:
        raise SecureFilesystemError(
            "Write directory is group-writable or world-writable."
        )


@dataclass(frozen=True)
class FileIdentity:
    """Stable filesystem identity for a file or directory."""

    backend: str
    volume_id: int
    file_id: int
    object_type: str


class IdentityMismatchError(SecureFilesystemError):
    """Raised when a filesystem object differs from its expected identity."""


class SecureDirectory:
    """Pinned directory root with descriptor-relative operations.

    The root must already exist and be owned by the effective user.
    Its mode must not permit group/other writes (normally 0700).
    """

    def __init__(self, root: str | os.PathLike[str]) -> None:
        _require_posix()

        candidate = Path(root)
        if not candidate.is_absolute():
            raise SecureFilesystemError("Trusted root must be absolute.")

        if ".." in candidate.parts:
            raise SecureFilesystemError("Trusted root contains traversal.")

        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
        fd = os.open("/", flags)

        try:
            for component in candidate.parts[1:]:
                next_fd = os.open(component, flags, dir_fd=fd)
                os.close(fd)
                fd = next_fd

            _check_private_directory(fd)
            self._root_fd = fd
            self._lock = threading.RLock()
            self._closed = False
        except Exception:
            os.close(fd)
            raise

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                os.close(self._root_fd)
                self._closed = True

    def __enter__(self) -> "SecureDirectory":
        with self._lock:
            self._ensure_open()
            return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _ensure_open(self) -> None:
        if self._closed:
            raise SecureFilesystemError("SecureDirectory is closed.")

    @contextmanager
    def _parent(self, relative_path) -> Iterator[tuple[int, str]]:
        components = _parts(relative_path)

        with self._lock:
            self._ensure_open()
            current_fd = os.dup(self._root_fd)

            try:
                for component in components[:-1]:
                    next_fd = os.open(
                        component,
                        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                        dir_fd=current_fd,
                    )
                    os.close(current_fd)
                    current_fd = next_fd

                check_fd = os.dup(self._root_fd)
                try:
                    for component in components[:-1]:
                        next_fd = os.open(
                            component,
                            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                            dir_fd=check_fd,
                        )
                        os.close(check_fd)
                        check_fd = next_fd

                    first = os.fstat(current_fd)
                    second = os.fstat(check_fd)

                    if (first.st_dev, first.st_ino) != (
                        second.st_dev, second.st_ino
                    ):
                        raise SecureFilesystemError(
                            "Parent directory identity changed during resolution."
                        )
                finally:
                    os.close(check_fd)

                yield current_fd, components[-1]

            except OSError as exc:
                if exc.errno in (errno.ELOOP, errno.ENOTDIR):
                    raise SecureFilesystemError(
                        "A path component is a link or is not a directory."
                    ) from exc
                raise
            finally:
                os.close(current_fd)

    def open_read(self, relative_path) -> int:
        """Open a regular file read-only; caller owns and must close the fd."""
        with self._parent(relative_path) as (parent_fd, name):
            flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC
            try:
                fd = os.open(name, flags, dir_fd=parent_fd)
            except OSError as exc:
                raise SecureFilesystemError("Could not securely open file.") from exc

            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode):
                os.close(fd)
                raise SecureFilesystemError("Target is not a regular file.")

            return fd

    def create_exclusive(self, relative_path, mode: int = 0o600) -> int:
        """Create a new regular file without overwriting; caller owns the fd."""
        with self._parent(relative_path) as (parent_fd, name):
            _check_private_directory(parent_fd)
            flags = (
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | os.O_NOFOLLOW
                | os.O_CLOEXEC
            )
            try:
                fd = os.open(name, flags, mode, dir_fd=parent_fd)
            except FileExistsError:
                raise
            except OSError as exc:
                raise SecureFilesystemError("Secure exclusive creation failed.") from exc

            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode):
                os.close(fd)
                raise SecureFilesystemError("Created entry is not a regular file.")

            return fd

    def mkdir_exclusive(self, relative_path, mode: int = 0o700) -> None:
        """Create exactly one new directory below a verified private parent."""
        with self._parent(relative_path) as (parent_fd, name):
            _check_private_directory(parent_fd)
            try:
                os.mkdir(name, mode=mode, dir_fd=parent_fd)
            except FileExistsError:
                raise
            except OSError as exc:
                raise SecureFilesystemError("Secure directory creation failed.") from exc

            child_fd = os.open(
                name,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                dir_fd=parent_fd,
            )
            try:
                _check_private_directory(child_fd)
            finally:
                os.close(child_fd)
            os.fsync(parent_fd)

    def atomic_write(
        self,
        relative_path,
        data: bytes,
        *,
        mode: int = 0o600,
        replace: bool = False,
    ) -> None:
        """Write bytes to a same-directory temp file and atomically publish.

        By default publication is no-clobber: an existing destination is an
        error. replace=True explicitly permits atomic replacement.
        """
        if not isinstance(data, bytes):
            raise TypeError("atomic_write data must be bytes.")

        with self._parent(relative_path) as (parent_fd, name):
            _check_private_directory(parent_fd)

            temp_name = ".secure-" + secrets.token_hex(16) + ".tmp"
            flags = (
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | os.O_NOFOLLOW
                | os.O_CLOEXEC
            )
            temp_fd = os.open(temp_name, flags, mode, dir_fd=parent_fd)

            try:
                view = memoryview(data)
                while view:
                    written = os.write(temp_fd, view)
                    if written <= 0:
                        raise SecureFilesystemError("Short write during publication.")
                    view = view[written:]

                os.fsync(temp_fd)

                if replace:
                    os.replace(
                        temp_name,
                        name,
                        src_dir_fd=parent_fd,
                        dst_dir_fd=parent_fd,
                    )
                else:
                    # link() fails if the destination already exists.
                    os.link(
                        temp_name,
                        name,
                        src_dir_fd=parent_fd,
                        dst_dir_fd=parent_fd,
                        follow_symlinks=False,
                    )
                    os.unlink(temp_name, dir_fd=parent_fd)

                os.fsync(parent_fd)
            finally:
                os.close(temp_fd)
                try:
                    os.unlink(temp_name, dir_fd=parent_fd)
                except FileNotFoundError:
                    pass


    @staticmethod
    def _identity_from_stat(info: os.stat_result) -> FileIdentity:
        if stat.S_ISREG(info.st_mode):
            object_type = "file"
        elif stat.S_ISDIR(info.st_mode):
            object_type = "directory"
        else:
            raise SecureFilesystemError(
                "Only regular files and directories are supported."
            )

        return FileIdentity(
            backend="posix",
            volume_id=info.st_dev,
            file_id=info.st_ino,
            object_type=object_type,
        )

    def get_identity(self, relative_path: str) -> FileIdentity:
        """Return identity after opening the object without following symlinks."""
        with self._parent(relative_path) as (parent_fd, name):
            entry = os.stat(
                name,
                dir_fd=parent_fd,
                follow_symlinks=False,
            )

            if stat.S_ISLNK(entry.st_mode):
                raise SecureFilesystemError("Symlinks are not permitted.")

            flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC
            if stat.S_ISDIR(entry.st_mode):
                flags |= os.O_DIRECTORY
            elif not stat.S_ISREG(entry.st_mode):
                raise SecureFilesystemError("Unsupported filesystem object.")

            fd = os.open(name, flags, dir_fd=parent_fd)
            try:
                opened = os.fstat(fd)
                identity = self._identity_from_stat(opened)
                entry_identity = self._identity_from_stat(entry)

                if identity != entry_identity:
                    raise IdentityMismatchError(
                        "Filesystem object changed during identity lookup."
                    )

                return identity
            finally:
                os.close(fd)

    def verify_identity(
        self,
        relative_path: str,
        expected: FileIdentity,
    ) -> bool:
        """Verify that a path still identifies the expected filesystem object."""
        actual = self.get_identity(relative_path)
        if actual != expected:
            raise IdentityMismatchError(
                f"Filesystem identity mismatch for {relative_path!r}."
            )
        return True

    def delete_file(
        self,
        relative_path: str,
        expected_identity: FileIdentity | None = None,
    ) -> None:
        """Delete one regular file; never follow a symlink or recurse."""
        with self._parent(relative_path) as (parent_fd, name):
            fd = os.open(
                name,
                os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC,
                dir_fd=parent_fd,
            )
            try:
                opened = os.fstat(fd)
                identity = self._identity_from_stat(opened)

                if identity.object_type != "file":
                    raise SecureFilesystemError(
                        "delete_file only accepts regular files."
                    )

                if expected_identity is not None and identity != expected_identity:
                    raise IdentityMismatchError(
                        "File identity does not match the expected identity."
                    )

                current = os.stat(
                    name,
                    dir_fd=parent_fd,
                    follow_symlinks=False,
                )
                if self._identity_from_stat(current) != identity:
                    raise IdentityMismatchError(
                        "File changed before deletion."
                    )

                os.unlink(name, dir_fd=parent_fd)
                os.fsync(parent_fd)
            finally:
                os.close(fd)

    def delete_directory(
        self,
        relative_path: str,
        expected_identity: FileIdentity | None = None,
    ) -> None:
        """Delete one empty directory; recursive deletion is never performed."""
        with self._parent(relative_path) as (parent_fd, name):
            fd = os.open(
                name,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                dir_fd=parent_fd,
            )
            try:
                opened = os.fstat(fd)
                identity = self._identity_from_stat(opened)

                if identity.object_type != "directory":
                    raise SecureFilesystemError(
                        "delete_directory requires a directory."
                    )

                if expected_identity is not None and identity != expected_identity:
                    raise IdentityMismatchError(
                        "Directory identity does not match the expected identity."
                    )

                current = os.stat(
                    name,
                    dir_fd=parent_fd,
                    follow_symlinks=False,
                )
                if self._identity_from_stat(current) != identity:
                    raise IdentityMismatchError(
                        "Directory changed before deletion."
                    )

                # rmdir is intentionally non-recursive and fails if not empty.
                os.rmdir(name, dir_fd=parent_fd)
                os.fsync(parent_fd)
            finally:
                os.close(fd)


    def rename(
        self,
        source_path: str,
        destination_path: str,
        *,
        replace: bool = False,
        expected_identity: FileIdentity | None = None,
    ) -> FileIdentity:
        """Atomically rename a regular file within this secure directory.

        replace=False uses Linux renameat2(RENAME_NOREPLACE).
        replace=True uses descriptor-relative os.rename.
        Directory and cross-filesystem moves are intentionally unsupported.
        """
        with self._parent(source_path) as (source_parent_fd, source_name):
            with self._parent(destination_path) as (
                destination_parent_fd,
                destination_name,
            ):
                _check_private_directory(source_parent_fd)
                _check_private_directory(destination_parent_fd)

                source_fd = os.open(
                    source_name,
                    os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC,
                    dir_fd=source_parent_fd,
                )
                try:
                    source_stat = os.fstat(source_fd)
                    source_identity = self._identity_from_stat(source_stat)

                    if source_identity.object_type != "file":
                        raise SecureFilesystemError(
                            "rename only supports regular files."
                        )

                    current_source = os.stat(
                        source_name,
                        dir_fd=source_parent_fd,
                        follow_symlinks=False,
                    )
                    if self._identity_from_stat(current_source) != source_identity:
                        raise IdentityMismatchError(
                            "Source changed during rename validation."
                        )

                    if (
                        expected_identity is not None
                        and source_identity != expected_identity
                    ):
                        raise IdentityMismatchError(
                            "Source identity does not match expected identity."
                        )

                    if replace:
                        # POSIX rename is atomic and replaces the destination
                        # directory entry itself; it does not follow symlinks.
                        os.rename(
                            source_name,
                            destination_name,
                            src_dir_fd=source_parent_fd,
                            dst_dir_fd=destination_parent_fd,
                        )
                    else:
                        # Fail closed if the platform lacks atomic no-clobber
                        # rename. Never emulate this with link()+unlink().
                        libc = ctypes.CDLL(None, use_errno=True)
                        renameat2 = getattr(libc, "renameat2", None)
                        if renameat2 is None:
                            raise SecureFilesystemUnavailable(
                                "Atomic no-clobber renameat2 is unavailable."
                            )

                        renameat2.argtypes = [
                            ctypes.c_int,
                            ctypes.c_char_p,
                            ctypes.c_int,
                            ctypes.c_char_p,
                            ctypes.c_uint,
                        ]
                        renameat2.restype = ctypes.c_int

                        result = renameat2(
                            source_parent_fd,
                            os.fsencode(source_name),
                            destination_parent_fd,
                            os.fsencode(destination_name),
                            1,  # RENAME_NOREPLACE
                        )
                        if result != 0:
                            error_number = ctypes.get_errno()
                            if error_number in (errno.ENOSYS, errno.EINVAL):
                                raise SecureFilesystemUnavailable(
                                    "Filesystem does not support atomic "
                                    "no-clobber rename."
                                )
                            raise OSError(
                                error_number,
                                os.strerror(error_number),
                                destination_name,
                            )

                    # Confirm the published entry identifies the opened source.
                    published_stat = os.stat(
                        destination_name,
                        dir_fd=destination_parent_fd,
                        follow_symlinks=False,
                    )
                    published_identity = self._identity_from_stat(published_stat)

                    if published_identity != source_identity:
                        raise IdentityMismatchError(
                            "Published file identity differs from source."
                        )

                    os.fsync(source_parent_fd)
                    if destination_parent_fd != source_parent_fd:
                        os.fsync(destination_parent_fd)

                    return published_identity
                finally:
                    os.close(source_fd)
