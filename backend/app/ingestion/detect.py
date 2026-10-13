"""Upload-time file validation: decide from the file's *contents* (magic bytes), cross-checked
against its extension, whether we accept it. Runs synchronously in the upload request so a bad
file is rejected immediately, before anything is stored.
"""

import io
import re
import unicodedata
import zipfile
from dataclasses import dataclass
from pathlib import PurePath
from typing import Literal

from app.core.errors import AppError

PDF_MAGIC = b"%PDF-"
ZIP_MAGIC = b"PK\x03\x04"
# Compound File Binary (OLE2): legacy .doc, and also *password-protected* .docx files.
OLE_MAGIC = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"

# Guard against zip bombs: a DOCX is a zip, and python-docx inflates document.xml in memory.
MAX_DOCX_UNCOMPRESSED_BYTES = 200 * 1024 * 1024

ACCEPTED_DESCRIPTION = "PDF (.pdf) and Word (.docx)"


class UploadRejectedError(AppError):
    status_code = 415
    code = "UNSUPPORTED_FILE_TYPE"


@dataclass(frozen=True)
class DetectedFile:
    file_type: Literal["pdf", "docx"]
    mime_type: str


MIME_TYPES = {
    "pdf": "application/pdf",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}


def _zip_names_and_size(data: bytes) -> tuple[set[str], int] | None:
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            infos = archive.infolist()
            return {info.filename for info in infos}, sum(info.file_size for info in infos)
    except (zipfile.BadZipFile, ValueError, OSError):
        return None


def detect_file_type(filename: str, data: bytes) -> DetectedFile:
    extension = PurePath(filename).suffix.lower()

    if not data:
        raise UploadRejectedError("The file is empty.", code="EMPTY_FILE", status_code=400)

    # Some PDF producers put junk before the header; the spec tolerates it within 1 KB.
    if PDF_MAGIC in data[:1024]:
        if extension == ".pdf":
            return DetectedFile("pdf", MIME_TYPES["pdf"])
        raise UploadRejectedError(
            f"This file contains a PDF but is named '{extension or 'without an extension'}'. "
            "Rename it to .pdf and upload it again.",
            code="FILE_TYPE_MISMATCH",
        )

    if data.startswith(OLE_MAGIC):
        if extension == ".docx":
            raise UploadRejectedError(
                "This Word document is password-protected. Remove the password in Word "
                "(File > Info > Protect Document) and upload it again.",
                code="ENCRYPTED_DOCUMENT",
            )
        if extension == ".doc":
            raise UploadRejectedError(
                "Legacy Word .doc files are not supported. Open the file in Word and use "
                "Save As > Word Document (.docx), then upload that.",
                code="LEGACY_DOC_UNSUPPORTED",
            )

    if data.startswith(ZIP_MAGIC) and extension == ".docx":
        listing = _zip_names_and_size(data)
        if listing is None or "word/document.xml" not in listing[0]:
            raise UploadRejectedError(
                "This file is named .docx but is not a valid Word document.",
                code="FILE_TYPE_MISMATCH",
            )
        if listing[1] > MAX_DOCX_UNCOMPRESSED_BYTES:
            raise UploadRejectedError(
                "This Word document is too large to process once decompressed.",
                code="FILE_TOO_LARGE",
                status_code=413,
            )
        return DetectedFile("docx", MIME_TYPES["docx"])

    if extension in (".pdf", ".docx"):
        kind = "a PDF" if extension == ".pdf" else "a Word document"
        raise UploadRejectedError(
            f"This file is named {extension} but its contents are not {kind}.",
            code="FILE_TYPE_MISMATCH",
        )
    if extension == ".doc":
        raise UploadRejectedError(
            "Legacy Word .doc files are not supported. Save the file as .docx and upload that.",
            code="LEGACY_DOC_UNSUPPORTED",
        )
    raise UploadRejectedError(
        f"Unsupported file type{f' ({extension})' if extension else ''}. "
        f"Only {ACCEPTED_DESCRIPTION} files are supported."
    )


_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]")
_WHITESPACE = re.compile(r"\s+")


def sanitize_filename(raw: str | None, fallback_extension: str) -> str:
    """A display name safe to store, log and echo back. Never used as a storage path."""
    name = (raw or "").replace("\\", "/").rsplit("/", 1)[-1]  # browsers may send full paths
    name = unicodedata.normalize("NFC", name)
    name = _WHITESPACE.sub(" ", _CONTROL_CHARS.sub("", name)).strip(" .")
    if not name:
        return f"document.{fallback_extension}"
    if len(name) > 255:
        stem, dot, ext = name.rpartition(".")
        name = (stem[: 250 - len(ext)] + dot + ext) if dot and len(ext) <= 10 else name[:255]
    return name
