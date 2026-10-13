import asyncio
import os
from pathlib import Path

from app.storage.base import ObjectNotFoundError, StorageError


class LocalStorage:
    """Files on local disk. Development only: container disks on free hosts are ephemeral."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()

    def _path(self, key: str) -> Path:
        path = (self.root / key).resolve()
        # Defence in depth: keys are generated internally, but never escape the root.
        if not path.is_relative_to(self.root):
            raise StorageError(f"invalid storage key: {key!r}")
        return path

    async def put(self, key: str, data: bytes, content_type: str) -> None:
        path = self._path(key)

        def write() -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(path.suffix + ".part")
            tmp.write_bytes(data)
            os.replace(tmp, path)  # atomic: readers never see a half-written file

        try:
            await asyncio.to_thread(write)
        except OSError as exc:
            raise StorageError(f"could not write {key}: {exc}") from exc

    async def get(self, key: str) -> bytes:
        path = self._path(key)
        try:
            return await asyncio.to_thread(path.read_bytes)
        except FileNotFoundError as exc:
            raise ObjectNotFoundError(key) from exc
        except OSError as exc:
            raise StorageError(f"could not read {key}: {exc}") from exc

    async def delete(self, key: str) -> None:
        path = self._path(key)

        def remove() -> None:
            path.unlink(missing_ok=True)
            # Tidy the now-empty per-document directory.
            try:
                path.parent.rmdir()
            except OSError:
                pass

        try:
            await asyncio.to_thread(remove)
        except OSError as exc:
            raise StorageError(f"could not delete {key}: {exc}") from exc
