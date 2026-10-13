from pathlib import Path

import httpx
import pytest

from app.storage import LocalStorage, ObjectNotFoundError, StorageError, SupabaseStorage

KEY = "documents/abc/original.pdf"


async def test_local_roundtrip_and_idempotent_delete(tmp_path: Path) -> None:
    storage = LocalStorage(tmp_path)
    await storage.put(KEY, b"%PDF-data", "application/pdf")
    assert await storage.get(KEY) == b"%PDF-data"
    await storage.delete(KEY)
    await storage.delete(KEY)  # deleting again is fine
    with pytest.raises(ObjectNotFoundError):
        await storage.get(KEY)
    assert not (tmp_path / "documents" / "abc").exists()  # empty folder tidied up


async def test_local_rejects_keys_escaping_root(tmp_path: Path) -> None:
    storage = LocalStorage(tmp_path / "root")
    with pytest.raises(StorageError):
        await storage.put("../outside.pdf", b"x", "application/pdf")


def supabase(handler) -> SupabaseStorage:  # type: ignore[no-untyped-def]
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return SupabaseStorage("https://proj.supabase.co/", "secret-key", "documents", client=client)


async def test_supabase_upload_request_shape() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"Key": KEY})

    await supabase(handler).put(KEY, b"bytes", "application/pdf")
    request = seen[0]
    assert request.method == "POST"
    assert str(request.url) == f"https://proj.supabase.co/storage/v1/object/documents/{KEY}"
    assert request.headers["apikey"] == "secret-key"
    assert request.headers["authorization"] == "Bearer secret-key"
    assert request.headers["content-type"] == "application/pdf"
    assert request.headers["x-upsert"] == "true"
    assert request.content == b"bytes"


async def test_supabase_download() -> None:
    storage = supabase(lambda request: httpx.Response(200, content=b"file-bytes"))
    assert await storage.get(KEY) == b"file-bytes"


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(404, json={"error": "not found"}),
        httpx.Response(
            400, json={"statusCode": "404", "error": "not_found", "message": "Object not found"}
        ),
    ],
)
async def test_supabase_missing_object(response: httpx.Response) -> None:
    storage = supabase(lambda request: response)
    with pytest.raises(ObjectNotFoundError):
        await storage.get(KEY)
    await storage.delete(KEY)  # deleting a missing object is not an error


async def test_supabase_server_error() -> None:
    storage = supabase(lambda request: httpx.Response(500, text="boom"))
    with pytest.raises(StorageError):
        await storage.put(KEY, b"x", "application/pdf")
    with pytest.raises(StorageError):
        await storage.delete(KEY)


async def test_supabase_unreachable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route")

    with pytest.raises(StorageError):
        await supabase(handler).get(KEY)
