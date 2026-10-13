"""Generate test documents in code, so tests never depend on files lying around.

Every generator returns bytes. PDFs come from PyMuPDF, DOCX files from python-docx.
"""

import io
import zipfile

import docx
import pymupdf

CLAUSE = (
    "The Company may terminate this Agreement upon thirty (30) days written notice to the "
    "Supplier. Termination shall not affect any rights accrued before the termination date."
)


def text_pdf(pages: int = 3, text: str = CLAUSE) -> bytes:
    document = pymupdf.open()
    for number in range(1, pages + 1):
        page = document.new_page()
        page.insert_textbox(pymupdf.Rect(72, 72, 520, 760), f"{number}. {text}", fontsize=11)
    return document.tobytes()


def _image_only_page(document: pymupdf.Document, text: str) -> None:
    """A page that *looks* like text but is only an image, like a scan."""
    source = pymupdf.open()
    source.new_page().insert_textbox(pymupdf.Rect(72, 72, 520, 760), text, fontsize=11)
    pixmap = source[0].get_pixmap(dpi=72)
    page = document.new_page()
    page.insert_image(page.rect, pixmap=pixmap)


def scanned_pdf(pages: int = 2) -> bytes:
    document = pymupdf.open()
    for _ in range(pages):
        _image_only_page(document, CLAUSE)
    return document.tobytes()


def mixed_pdf() -> bytes:
    """One text page followed by one scanned page."""
    document = pymupdf.open(stream=text_pdf(pages=1), filetype="pdf")
    _image_only_page(document, CLAUSE)
    return document.tobytes()


def encrypted_pdf(user_password: str = "secret") -> bytes:
    document = pymupdf.open(stream=text_pdf(pages=1), filetype="pdf")
    return document.tobytes(
        encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw=user_password, owner_pw="owner"
    )


def owner_restricted_pdf() -> bytes:
    """Encrypted with an owner password only: opens without a password (common for contracts)."""
    document = pymupdf.open(stream=text_pdf(pages=1), filetype="pdf")
    return document.tobytes(encryption=pymupdf.PDF_ENCRYPT_AES_256, owner_pw="owner", permissions=0)


def corrupt_pdf() -> bytes:
    """Cut off before any page objects: unrecoverable."""
    return text_pdf(pages=2)[:300]


def damaged_pdf() -> bytes:
    """Cut in half: MuPDF repairs it, recovering only some page content."""
    good = text_pdf(pages=2)
    return good[: len(good) // 2]


def docx_file(paragraphs: list[str] | None = None, table: list[list[str]] | None = None) -> bytes:
    document = docx.Document()
    for text in paragraphs if paragraphs is not None else [CLAUSE]:
        document.add_paragraph(text)
    if table:
        grid = document.add_table(rows=len(table), cols=len(table[0]))
        for r, row in enumerate(table):
            for c, value in enumerate(row):
                grid.cell(r, c).text = value
    buffer = io.BytesIO()
    document.save(buffer)
    return buffer.getvalue()


def corrupt_docx() -> bytes:
    """A zip with the right part names but broken XML inside."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("[Content_Types].xml", "<Types")
        archive.writestr("word/document.xml", "<w:document><not-closed>")
    return buffer.getvalue()


def plain_zip() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("readme.txt", "hello")
    return buffer.getvalue()


OLE_HEADER = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"\x00" * 504
