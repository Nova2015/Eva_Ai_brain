"""Filesystem path validation for Deep Learning data and artifacts.

Checks reject symbolic links and Windows reparse points before reads/writes.
These are preflight checks; filesystem TOCTOU races remain a residual risk.
"""
from __future__ import annotations

import os
import re
import stat
from pathlib import Path


class PathSecurityError(ValueError):
    """Raised when a filesystem path cannot be verified as safe."""


def _is_unsafe_path_entry(path: Path) -> bool:
    """Return True for symlinks, junctions, reparse points, or unverifiable entries."""
    try:
        os.lstat(path)
    except FileNotFoundError:
        pass
    except (OSError, RuntimeError, ValueError):
        return True
    try:
        if path.is_symlink():
            return True
        is_junction = getattr(path, "is_junction", None)
        if is_junction is not None and is_junction():
            return True
        # Python 3.11 Windows fallback for junctions and other reparse points.
        if os.name == "nt":
            import ctypes
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            get_attributes = kernel32.GetFileAttributesW
            get_attributes.argtypes = [ctypes.c_wchar_p]
            get_attributes.restype = ctypes.c_uint32
            attributes = get_attributes(str(path))
            if attributes == 0xFFFFFFFF:
                error_code = ctypes.get_last_error()
                if error_code in (2, 3):  # file/path not found
                    return False
                return True
            if attributes & 0x0400:  # FILE_ATTRIBUTE_REPARSE_POINT
                return True
        return False
    except (OSError, RuntimeError, ValueError):
        return True


def validate_path(value, *, must_exist=False, regular_file=False, directory=False) -> Path:
    """Return an absolute lexical path after checking all existing components.

    Does not call resolve() before validation, because that would hide link
    components. Missing targets are allowed for creation unless must_exist=True.
    """
    try:
        raw = os.fspath(value)
        if not isinstance(raw, str) or not raw.strip():
            raise PathSecurityError("Filesystem path is empty or invalid")
        if any(part in {".", ".."} for part in re.split(r"[\\/]", raw)):
            raise PathSecurityError("Filesystem path contains traversal components")
        candidate = Path(raw).expanduser()
        if not candidate.is_absolute():
            candidate = Path.cwd() / candidate
        parts = candidate.parts
        if not parts:
            raise PathSecurityError("Filesystem path is invalid")
        current = Path(parts[0])
        for index, part in enumerate(parts[1:], start=1):
            current = current / part
            final = index == len(parts) - 1
            if _is_unsafe_path_entry(current):
                raise PathSecurityError("Filesystem path contains an unsafe link or reparse point")
            try:
                info = os.lstat(current)
            except FileNotFoundError:
                if must_exist:
                    raise PathSecurityError("Required filesystem path does not exist")
                break
            except (OSError, RuntimeError, ValueError) as exc:
                raise PathSecurityError("Filesystem path cannot be safely inspected") from exc
            if not final and not stat.S_ISDIR(info.st_mode):
                raise PathSecurityError("Filesystem path has a non-directory parent")
            if final:
                if regular_file and not stat.S_ISREG(info.st_mode):
                    raise PathSecurityError("Filesystem path is not a regular file")
                if directory and not stat.S_ISDIR(info.st_mode):
                    raise PathSecurityError("Filesystem path is not a directory")
        if len(parts) == 1:
            try:
                info = os.lstat(current)
            except FileNotFoundError:
                if must_exist:
                    raise PathSecurityError("Required filesystem path does not exist")
            except (OSError, RuntimeError, ValueError) as exc:
                raise PathSecurityError("Filesystem path cannot be safely inspected") from exc
            else:
                if _is_unsafe_path_entry(current):
                    raise PathSecurityError("Filesystem path is unsafe")
                if regular_file and not stat.S_ISREG(info.st_mode):
                    raise PathSecurityError("Filesystem path is not a regular file")
                if directory and not stat.S_ISDIR(info.st_mode):
                    raise PathSecurityError("Filesystem path is not a directory")
        return candidate
    except PathSecurityError:
        raise
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise PathSecurityError("Filesystem path cannot be safely validated") from exc
