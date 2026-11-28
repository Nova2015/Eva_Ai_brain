import os
from pathlib import Path

import pytest

from src.deep_learning.secure_fs import (
    SecureDirectory,
    SecureFilesystemError,
    SecureFilesystemUnavailable,
)


POSIX = os.name == "posix"


@pytest.fixture
def secure_root(tmp_path):
    root = tmp_path / "secure-root"
    root.mkdir(mode=0o700)
    root.chmod(0o700)
    return root


@pytest.mark.skipif(not POSIX, reason="POSIX backend test")
def test_create_exclusive_and_read_regular_file(secure_root):
    with SecureDirectory(secure_root) as secure:
        secure.mkdir_exclusive("data")

        fd = secure.create_exclusive("data/input.csv")
        try:
            os.write(fd, b"x,y\n1,2\n")
        finally:
            os.close(fd)

        read_fd = secure.open_read("data/input.csv")
        try:
            assert os.read(read_fd, 100) == b"x,y\n1,2\n"
        finally:
            os.close(read_fd)


@pytest.mark.skipif(not POSIX, reason="POSIX backend test")
def test_exclusive_create_does_not_overwrite(secure_root):
    with SecureDirectory(secure_root) as secure:
        fd = secure.create_exclusive("data.txt")
        os.write(fd, b"original")
        os.close(fd)

        with pytest.raises(FileExistsError):
            secure.create_exclusive("data.txt")

        read_fd = secure.open_read("data.txt")
        try:
            assert os.read(read_fd, 100) == b"original"
        finally:
            os.close(read_fd)


@pytest.mark.skipif(not POSIX, reason="POSIX backend test")
@pytest.mark.parametrize("unsafe", ["../outside.txt", "/absolute.txt", "a/../b", "a//b", "a\\b"])
def test_rejects_unsafe_relative_paths(secure_root, unsafe):
    with SecureDirectory(secure_root) as secure:
        with pytest.raises(SecureFilesystemError):
            secure.open_read(unsafe)


@pytest.mark.skipif(not POSIX, reason="POSIX backend test")
def test_rejects_symlink_file(secure_root, tmp_path):
    outside = tmp_path / "outside.txt"
    outside.write_text("outside", encoding="utf-8")
    (secure_root / "linked.txt").symlink_to(outside)

    with SecureDirectory(secure_root) as secure:
        with pytest.raises(SecureFilesystemError):
            secure.open_read("linked.txt")


@pytest.mark.skipif(not POSIX, reason="POSIX backend test")
def test_rejects_symlink_directory(secure_root, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("secret", encoding="utf-8")
    (secure_root / "linked-dir").symlink_to(outside, target_is_directory=True)

    with SecureDirectory(secure_root) as secure:
        with pytest.raises(SecureFilesystemError):
            secure.open_read("linked-dir/secret.txt")


@pytest.mark.skipif(not POSIX, reason="POSIX backend test")
def test_atomic_write_no_clobber_and_explicit_replace(secure_root):
    with SecureDirectory(secure_root) as secure:
        secure.atomic_write("result.json", b'{"version":1}')

        with pytest.raises(FileExistsError):
            secure.atomic_write("result.json", b'{"version":2}')

        secure.atomic_write(
            "result.json",
            b'{"version":2}',
            replace=True,
        )

        fd = secure.open_read("result.json")
        try:
            assert os.read(fd, 100) == b'{"version":2}'
        finally:
            os.close(fd)


@pytest.mark.skipif(not POSIX, reason="POSIX backend test")
def test_atomic_write_does_not_follow_destination_symlink(secure_root, tmp_path):
    outside = tmp_path / "outside.txt"
    outside.write_text("untouched", encoding="utf-8")
    (secure_root / "linked.txt").symlink_to(outside)

    with SecureDirectory(secure_root) as secure:
        with pytest.raises(OSError):
            secure.atomic_write("linked.txt", b"replacement")

        secure.atomic_write("linked.txt", b"replacement", replace=True)

    assert outside.read_text(encoding="utf-8") == "untouched"
    assert (secure_root / "linked.txt").read_bytes() == b"replacement"


@pytest.mark.skipif(not POSIX, reason="POSIX backend test")
def test_rejects_group_or_world_writable_root(tmp_path):
    root = tmp_path / "insecure-root"
    root.mkdir(mode=0o700)
    root.chmod(0o777)

    try:
        with pytest.raises(SecureFilesystemError):
            SecureDirectory(root)
    finally:
        root.chmod(0o700)


def test_windows_backend_fails_closed():
    if os.name != "nt":
        pytest.skip("Windows-specific platform test")

    with pytest.raises(SecureFilesystemUnavailable):
        SecureDirectory(Path("C:/secure-root"))


@pytest.mark.skipif(not POSIX, reason="POSIX backend test")
def test_concurrent_exclusive_creation_has_single_winner(secure_root):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    worker_count = 12
    barrier = Barrier(worker_count)

    with SecureDirectory(secure_root) as secure:
        def create_file(worker_id):
            barrier.wait()
            try:
                fd = secure.create_exclusive("shared.bin")
            except FileExistsError:
                return "exists"

            try:
                os.write(fd, str(worker_id).encode())
            finally:
                os.close(fd)

            return "created"

        with ThreadPoolExecutor(max_workers=worker_count) as pool:
            results = list(pool.map(create_file, range(worker_count)))

        assert results.count("created") == 1
        assert results.count("exists") == worker_count - 1

        fd = secure.open_read("shared.bin")
        try:
            content = os.read(fd, 100)
        finally:
            os.close(fd)

        assert content in [str(i).encode() for i in range(worker_count)]


@pytest.mark.skipif(not POSIX, reason="POSIX backend test")
def test_parent_directory_replacement_race_does_not_escape_root(
    secure_root, tmp_path, monkeypatch
):
    import src.deep_learning.secure_fs as secure_fs_module

    data_dir = secure_root / "data"
    data_dir.mkdir(mode=0o700)

    outside_dir = tmp_path / "outside"
    outside_dir.mkdir(mode=0o700)

    moved_dir = tmp_path / "moved-data"

    original_open = os.open
    race_triggered = False

    with SecureDirectory(secure_root) as secure:

        def racing_open(path, flags, *args, **kwargs):
            nonlocal race_triggered

            fd = original_open(path, flags, *args, **kwargs)

            if (
                path == "data"
                and kwargs.get("dir_fd") is not None
                and not race_triggered
            ):
                race_triggered = True
                os.rename(data_dir, moved_dir)
                os.symlink(outside_dir, data_dir)

            return fd

        monkeypatch.setattr(secure_fs_module.os, "open", racing_open)

        try:
            fd = secure.create_exclusive("data/escaped.txt")
        except SecureFilesystemError:
            pass
        else:
            os.close(fd)
            pytest.fail(
                "Operation succeeded after parent directory was moved "
                "outside the trusted root."
            )

    assert race_triggered
    assert not (moved_dir / "escaped.txt").exists()
    assert not (outside_dir / "escaped.txt").exists()


@pytest.mark.skipif(not POSIX, reason="POSIX backend test")
def test_ancestor_symlink_is_rejected(tmp_path):
    from src.deep_learning.secure_fs import SecureDirectory

    real_parent = tmp_path / "real-parent"
    real_parent.mkdir(mode=0o700)
    root = real_parent / "secure-root"
    root.mkdir(mode=0o700)

    alias = tmp_path / "alias"
    alias.symlink_to(real_parent, target_is_directory=True)

    with pytest.raises(OSError):
        SecureDirectory(alias / "secure-root")


@pytest.mark.skipif(not POSIX, reason="POSIX backend test")
def test_insecure_parent_directory_permissions_are_rejected(tmp_path):
    from src.deep_learning.secure_fs import (
        SecureDirectory,
        SecureFilesystemError,
    )

    root = tmp_path / "secure-root"
    root.mkdir(mode=0o700)

    data_dir = root / "data"
    data_dir.mkdir(mode=0o700)
    data_dir.chmod(0o777)

    secure = SecureDirectory(root)
    try:
        with pytest.raises(SecureFilesystemError):
            secure.create_exclusive("data/blocked.bin")
    finally:
        secure.close()


@pytest.mark.skipif(not POSIX, reason="POSIX backend test")
def test_concurrent_atomic_publication_has_single_winner(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    from src.deep_learning.secure_fs import SecureDirectory

    root = tmp_path / "secure-root"
    root.mkdir(mode=0o700)

    secure = SecureDirectory(root)
    workers = 12
    barrier = Barrier(workers)

    def publish(index):
        payload = f"worker-{index}".encode()
        barrier.wait()
        try:
            secure.atomic_write(
                "published.bin",
                payload,
                replace=False,
            )
            return ("success", payload)
        except FileExistsError:
            return ("exists", payload)

    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(publish, range(workers)))

        successes = [r for r in results if r[0] == "success"]
        assert len(successes) == 1

        fd = secure.open_read("published.bin")
        try:
            actual = os.read(fd, 4096)
        finally:
            os.close(fd)

        assert actual == successes[0][1]
    finally:
        secure.close()


@pytest.mark.skipif(not POSIX, reason="POSIX backend test")
def test_get_and_verify_file_identity(secure_root):
    from src.deep_learning.secure_fs import IdentityMismatchError

    with SecureDirectory(secure_root) as secure:
        fd = secure.create_exclusive("identity.txt")
        os.write(fd, b"identity")
        os.close(fd)

        identity = secure.get_identity("identity.txt")
        assert identity.backend == "posix"
        assert identity.object_type == "file"
        assert secure.verify_identity("identity.txt", identity) is True

        fd = secure.create_exclusive("replacement.txt")
        os.close(fd)
        os.replace(secure_root / "replacement.txt", secure_root / "identity.txt")

        with pytest.raises(IdentityMismatchError):
            secure.verify_identity("identity.txt", identity)


@pytest.mark.skipif(not POSIX, reason="POSIX backend test")
def test_delete_file_requires_matching_identity(secure_root):
    from src.deep_learning.secure_fs import IdentityMismatchError

    with SecureDirectory(secure_root) as secure:
        fd = secure.create_exclusive("delete-me.txt")
        os.write(fd, b"payload")
        os.close(fd)

        identity = secure.get_identity("delete-me.txt")
        secure.delete_file("delete-me.txt", identity)

        with pytest.raises(FileNotFoundError):
            secure.get_identity("delete-me.txt")

        fd = secure.create_exclusive("keep.txt")
        os.write(fd, b"keep")
        os.close(fd)

        wrong_identity = secure.get_identity("keep.txt")
        wrong_identity = type(wrong_identity)(
            backend=wrong_identity.backend,
            volume_id=wrong_identity.volume_id,
            file_id=wrong_identity.file_id + 1,
            object_type=wrong_identity.object_type,
        )

        with pytest.raises(IdentityMismatchError):
            secure.delete_file("keep.txt", wrong_identity)

        read_fd = secure.open_read("keep.txt")
        try:
            assert os.read(read_fd, 100) == b"keep"
        finally:
            os.close(read_fd)


@pytest.mark.skipif(not POSIX, reason="POSIX backend test")
def test_delete_directory_is_non_recursive(secure_root):
    with SecureDirectory(secure_root) as secure:
        secure.mkdir_exclusive("empty-dir")
        identity = secure.get_identity("empty-dir")
        secure.delete_directory("empty-dir", identity)

        with pytest.raises(FileNotFoundError):
            secure.get_identity("empty-dir")

        secure.mkdir_exclusive("non-empty")
        fd = secure.create_exclusive("non-empty/keep.txt")
        os.close(fd)

        with pytest.raises(OSError):
            secure.delete_directory("non-empty")

        assert secure.get_identity("non-empty").object_type == "directory"


@pytest.mark.skipif(not POSIX, reason="POSIX backend test")
def test_delete_file_rejects_symlink(secure_root, tmp_path):
    outside = tmp_path / "outside.txt"
    outside.write_text("untouched", encoding="utf-8")
    (secure_root / "linked.txt").symlink_to(outside)

    with SecureDirectory(secure_root) as secure:
        with pytest.raises(SecureFilesystemError):
            secure.delete_file("linked.txt")

    assert outside.read_text(encoding="utf-8") == "untouched"


@pytest.mark.skipif(not POSIX, reason="POSIX backend test")
def test_rename_no_clobber_and_identity(secure_root):
    from src.deep_learning.secure_fs import IdentityMismatchError

    with SecureDirectory(secure_root) as secure:
        fd = secure.create_exclusive("source.txt")
        os.write(fd, b"source-data")
        os.close(fd)

        identity = secure.get_identity("source.txt")
        moved_identity = secure.rename(
            "source.txt",
            "destination.txt",
            expected_identity=identity,
        )

        assert moved_identity == identity
        assert secure.get_identity("destination.txt") == identity

        with pytest.raises(FileNotFoundError):
            secure.get_identity("source.txt")

        fd = secure.create_exclusive("another.txt")
        os.write(fd, b"another")
        os.close(fd)

        with pytest.raises(FileExistsError):
            secure.rename("another.txt", "destination.txt")

        assert secure.get_identity("another.txt").object_type == "file"

        another_identity = secure.get_identity("another.txt")
        wrong_identity = type(another_identity)(
            backend=another_identity.backend,
            volume_id=another_identity.volume_id,
            file_id=another_identity.file_id + 1,
            object_type=another_identity.object_type,
        )
        with pytest.raises(IdentityMismatchError):
            secure.rename(
                "another.txt",
                "new-destination.txt",
                expected_identity=wrong_identity,
            )


@pytest.mark.skipif(not POSIX, reason="POSIX backend test")
def test_rename_explicit_replace_is_atomic_and_does_not_follow_symlink(
    secure_root, tmp_path
):
    outside = tmp_path / "outside.txt"
    outside.write_text("outside-data", encoding="utf-8")

    with SecureDirectory(secure_root) as secure:
        fd = secure.create_exclusive("source.txt")
        os.write(fd, b"new-data")
        os.close(fd)

        (secure_root / "destination.txt").symlink_to(outside)

        identity = secure.rename(
            "source.txt",
            "destination.txt",
            replace=True,
        )

        assert identity == secure.get_identity("destination.txt")
        assert (secure_root / "destination.txt").read_bytes() == b"new-data"
        assert outside.read_text(encoding="utf-8") == "outside-data"


@pytest.mark.skipif(not POSIX, reason="POSIX backend test")
def test_rename_rejects_symlink_source(secure_root, tmp_path):
    outside = tmp_path / "outside.txt"
    outside.write_text("outside-data", encoding="utf-8")
    (secure_root / "source-link.txt").symlink_to(outside)

    with SecureDirectory(secure_root) as secure:
        with pytest.raises(SecureFilesystemError):
            secure.rename("source-link.txt", "destination.txt")

    assert outside.read_text(encoding="utf-8") == "outside-data"
    assert not (secure_root / "destination.txt").exists()


@pytest.mark.skipif(not POSIX, reason="POSIX backend test")
def test_concurrent_no_clobber_rename_has_single_winner(secure_root):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    worker_count = 8
    barrier = Barrier(worker_count)

    with SecureDirectory(secure_root) as secure:
        for index in range(worker_count):
            fd = secure.create_exclusive(f"source-{index}.txt")
            os.write(fd, f"payload-{index}".encode())
            os.close(fd)

        def publish(index):
            barrier.wait()
            try:
                secure.rename(
                    f"source-{index}.txt",
                    "published.txt",
                )
                return "renamed"
            except FileExistsError:
                return "exists"

        with ThreadPoolExecutor(max_workers=worker_count) as pool:
            results = list(pool.map(publish, range(worker_count)))

        assert results.count("renamed") == 1
        assert results.count("exists") == worker_count - 1

        fd = secure.open_read("published.txt")
        try:
            content = os.read(fd, 100)
        finally:
            os.close(fd)

        assert content.startswith(b"payload-")
