import pytest

from app.ingestion import detect
from app.ingestion.detect import UploadRejectedError, detect_file_type, sanitize_filename
from tests.fixtures import factory


def rejected(filename: str, data: bytes) -> UploadRejectedError:
    with pytest.raises(UploadRejectedError) as info:
        detect_file_type(filename, data)
    return info.value


def test_accepts_pdf() -> None:
    assert detect_file_type("Contract.PDF", factory.text_pdf(1)).file_type == "pdf"


def test_accepts_docx() -> None:
    detected = detect_file_type("contract.docx", factory.docx_file())
    assert detected.file_type == "docx"
    assert detected.mime_type.endswith("wordprocessingml.document")


def test_pdf_with_leading_junk_is_accepted() -> None:
    assert detect_file_type("a.pdf", b"\n\r junk" + factory.text_pdf(1)).file_type == "pdf"


@pytest.mark.parametrize(
    ("filename", "data", "code", "status"),
    [
        ("notes.txt", b"hello world", "UNSUPPORTED_FILE_TYPE", 415),
        ("image.png", b"\x89PNG\r\n\x1a\n....", "UNSUPPORTED_FILE_TYPE", 415),
        ("noextension", b"hello", "UNSUPPORTED_FILE_TYPE", 415),
        ("old.doc", factory.OLE_HEADER, "LEGACY_DOC_UNSUPPORTED", 415),
        ("old.doc", b"not even ole", "LEGACY_DOC_UNSUPPORTED", 415),
        ("locked.docx", factory.OLE_HEADER, "ENCRYPTED_DOCUMENT", 415),
        ("fake.pdf", b"just text pretending", "FILE_TYPE_MISMATCH", 415),
        ("fake.docx", factory.plain_zip(), "FILE_TYPE_MISMATCH", 415),
        ("fake.docx", b"plain text", "FILE_TYPE_MISMATCH", 415),
        ("empty.pdf", b"", "EMPTY_FILE", 400),
    ],
)
def test_rejections(filename: str, data: bytes, code: str, status: int) -> None:
    error = rejected(filename, data)
    assert (error.code, error.status_code) == (code, status)
    assert error.message  # always a human-readable explanation


def test_pdf_content_with_wrong_extension_is_a_mismatch() -> None:
    error = rejected("contract.docx", factory.text_pdf(1))
    assert error.code == "FILE_TYPE_MISMATCH"
    assert "PDF" in error.message


def test_docx_zip_bomb_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(detect, "MAX_DOCX_UNCOMPRESSED_BYTES", 100)
    error = rejected("big.docx", factory.docx_file())
    assert (error.code, error.status_code) == ("FILE_TOO_LARGE", 413)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("contract.pdf", "contract.pdf"),
        ("C:\\Users\\me\\Desktop\\contract.pdf", "contract.pdf"),
        ("../../etc/passwd.pdf", "passwd.pdf"),
        ("bad\x00name\n.pdf", "badname.pdf"),
        ("  spaced   out  name.pdf ", "spaced out name.pdf"),
        ("Vertrag_über_Lieferung.pdf", "Vertrag_über_Lieferung.pdf"),
        ("", "document.pdf"),
        (None, "document.pdf"),
        ("...", "document.pdf"),
    ],
)
def test_sanitize_filename(raw: str | None, expected: str) -> None:
    assert sanitize_filename(raw, "pdf") == expected


def test_sanitize_truncates_but_keeps_extension() -> None:
    name = sanitize_filename("a" * 400 + ".docx", "docx")
    assert len(name) <= 255
    assert name.endswith(".docx")
