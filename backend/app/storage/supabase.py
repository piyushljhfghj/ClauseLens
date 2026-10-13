"""Supabase Storage over its REST API (no SDK needed).

Uses a *private* bucket and the server-side secret key, which must never reach the browser:
files are always streamed to the client through our API.
"""

import httpx

from app.storage.base import ObjectNotFoundError, StorageError


class SupabaseStorage:
    def __init__(
        self,
        url: str,
        service_key: str,
        bucket: str,
        *,
        client: httpx.AsyncClient | None = None,
        timeout_s: float = 60.0,
    ) -> None:
        self._base = f"{url.rstrip('/')}/storage/v1/object/{bucket}"
        self._headers = {"apikey": service_key, "Authorization": f"Bearer {service_key}"}
        self._client = client or httpx.AsyncClient(timeout=timeout_s)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _request(
        self,
        method: str,
        key: str,
        *,
        content: bytes | None = None,
        headers: dict[str, str] | None = None,
    ) -> httpx.Response:
        try:
            return await self._client.request(
                method,
                f"{self._base}/{key}",
                content=content,
                headers={**self._headers, **(headers or {})},
            )
        except httpx.HTTPError as exc:
            raise StorageError(f"Supabase Storage unreachable: {type(exc).__name__}") from exc

    @staticmethod
    def _is_not_found(response: httpx.Response) -> bool:
        # Supabase reports a missing object as 404, or as 400 with a "not_found" body.
        return response.status_code == 404 or (
            response.status_code == 400 and "not_found" in response.text.lower()
        )

    async def put(self, key: str, data: bytes, content_type: str) -> None:
        response = await self._request(
            "POST", key, content=data, headers={"Content-Type": content_type, "x-upsert": "true"}
        )
        if response.status_code >= 400:
            raise StorageError(
                f"Supabase upload failed ({response.status_code}): {response.text[:200]}"
            )

    async def get(self, key: str) -> bytes:
        response = await self._request("GET", key)
        if self._is_not_found(response):
            raise ObjectNotFoundError(key)
        if response.status_code >= 400:
            raise StorageError(f"Supabase download failed ({response.status_code})")
        return response.content

    async def delete(self, key: str) -> None:
        response = await self._request("DELETE", key)
        if response.status_code >= 400 and not self._is_not_found(response):
            raise StorageError(f"Supabase delete failed ({response.status_code})")
