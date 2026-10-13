from app.core.config import Settings
from app.storage.base import ObjectNotFoundError, Storage, StorageError, document_key
from app.storage.local import LocalStorage
from app.storage.supabase import SupabaseStorage

__all__ = [
    "LocalStorage",
    "ObjectNotFoundError",
    "Storage",
    "StorageError",
    "SupabaseStorage",
    "build_storage",
    "document_key",
]


def build_storage(settings: Settings) -> Storage:
    if settings.storage_backend == "supabase":
        assert settings.supabase_url and settings.supabase_service_key  # checked by Settings
        return SupabaseStorage(
            settings.supabase_url,
            settings.supabase_service_key.get_secret_value(),
            settings.supabase_storage_bucket,
        )
    return LocalStorage(settings.storage_local_dir)
