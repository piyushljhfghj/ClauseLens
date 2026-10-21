"""Grounded answering end to end: real PostgreSQL/pgvector retrieval, deterministic concept
embeddings, and a scripted LLM whose output we control (so verification is what is tested)."""

import re
import uuid
from collections.abc import AsyncIterator, Callable
from datetime import datetime

import httpx
import pytest
from fastapi import FastAPI
from pydantic import SecretStr
from sqlalchemy import delete, event, select, update
from sqlalchemy import text as text_interval
from sqlalchemy.exc import OperationalError

from app.answering.prompt import INSUFFICIENT_ANSWER
from app.answering.service import AnswerService
from app.core.config import Settings
from app.db.models import (
    Document,
    DocumentBlock,
    DocumentChunk,
    DocumentPage,
    DocumentSection,
    DocumentText,
)
from app.db.session import get_engine, get_sessionmaker
from app.ingestion import pdf_extract
from app.ingestion.model import SourceBlock, SourceLine, SourceWord
from app.ingestion.ocr import OcrResult, tesseract_version
from app.llm import ChatMessage, LLMResult, LLMUnavailableError, OpenAICompatibleClient
from app.main import create_app
from app.retrieval.service import RetrievalService
from tests.conftest import truncate_all
from tests.fakes import ConceptEmbedder, FakeLLM, answer_output, evidence_in
from tests.fixtures import factory
from tests.test_retrieval import CONTRACT_A, CONTRACT_B, contract_pdf

ANSWERS = "/api/answers"
needs_tesseract = pytest.mark.skipif(tesseract_version() is None, reason="Tesseract not installed")
MISSING = "00000000-0000-0000-0000-000000000001"
QUOTE_A = "shall not exceed AED 100,000"
QUOTE_B = "shall not exceed AED 1,000,000"


@pytest.fixture
def llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
async def answer_app(db_settings: Settings, llm: FakeLLM) -> AsyncIterator[FastAPI]:
    app = create_app(db_settings, embedder=ConceptEmbedder(), llm=llm)
    async with app.router.lifespan_context(app):
        await truncate_all()
        yield app


@pytest.fixture
async def api(answer_app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    transport = httpx.ASGITransport(app=answer_app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


async def upload(api: httpx.AsyncClient, app: FastAPI, name: str, data: bytes) -> str:
    response = await api.post("/api/documents", files={"file": (name, data)})
    assert response.status_code == 202, response.text
    while await app.state.worker.run_once():
        pass
    return response.json()["id"]


async def ask(api: httpx.AsyncClient, question: str, *ids: str) -> dict:  # type: ignore[type-arg]
    response = await api.post(ANSWERS, json={"question": question, "document_ids": list(ids)})
    assert response.status_code == 200, response.text
    return response.json()


def find(evidence: dict[str, str], needle: str) -> str:
    """Evidence id whose passage contains `needle` (whitespace-insensitive)."""
    for evidence_id, text in evidence.items():
        if needle in re.sub(r"\s+", " ", text):
            return evidence_id
    raise AssertionError(f"{needle!r} not in evidence {list(evidence)}")


async def canonical(document_id: str) -> str:
    async with get_sessionmaker()() as session:
        text = await session.get(DocumentText, uuid.UUID(document_id))
        assert text is not None
        return text.canonical_text


@pytest.fixture
async def contract(api: httpx.AsyncClient, answer_app: FastAPI) -> str:
    return await upload(api, answer_app, "contract-a.pdf", contract_pdf(CONTRACT_A))


# Unrelated operational annexes: none of the concept words of the fake embedder, and none of
# the questions' keywords. Appended after Contract A, they keep its page numbers (1-6) and make
# the document too large for small-document mode, so answers use the excerpt selection.
_ANNEX_TOPICS = [
    ("Site Access", "Visitors sign the register at reception and wear a badge at all times"),
    ("Parking", "Vehicles use the marked bays behind the warehouse between seven and seven"),
    ("Canteen", "Hot meals are served from noon and trays are returned to the racks"),
    ("Fire Drills", "Staff assemble at the north gate when the alarm sounds twice"),
    ("Equipment", "Laptops are collected from the service desk with a signed slip"),
    ("Recycling", "Paper, glass and cans are sorted into the coloured bins on each floor"),
    ("Meeting Rooms", "Rooms are reserved through the booking board one day ahead"),
    ("Holidays", "The office calendar lists public holidays for the coming year"),
    ("Dress Code", "Safety shoes are worn on the loading dock and in the workshop"),
    ("Travel", "Train tickets are booked by the travel desk after manager approval"),
    ("Stationery", "Notebooks and pens are requested from the supplies cupboard"),
    ("Visitor Wifi", "Guests receive a daily wifi voucher printed at reception"),
    ("Cleaning", "Desks are cleared each Friday evening for the cleaning crew"),
    ("First Aid", "Trained first aiders are listed on the board beside each lift"),
]
LARGE_CONTRACT = CONTRACT_A + [
    f"Annex {i} - {title}\n\n"
    + " ".join(f"{sentence}, as noted in section {i}.{n}." for n in range(12))
    for i, (title, sentence) in enumerate(_ANNEX_TOPICS, start=1)
]


@pytest.fixture
async def large_contract(api: httpx.AsyncClient, answer_app: FastAPI) -> str:
    """Contract A plus unrelated annexes: too large for small-document mode."""
    return await upload(api, answer_app, "contract-large.pdf", contract_pdf(LARGE_CONTRACT))


# --- grounded answers with verified quotations ----------------------------------------------------

TERMINATION = "Either party may terminate this Agreement by giving thirty (30) days' written notice"


async def test_grounded_answer_with_verified_quote(
    api: httpx.AsyncClient, contract: str, llm: FakeLLM
) -> None:
    def script(messages: list[ChatMessage]) -> str:
        e = find(evidence_in(messages), "terminate this Agreement")
        return answer_output(
            "Either party can end it with 30 days' written notice [1].",
            # The quotation contains the "30" the answer states (unverified figures are flagged).
            [{"n": 1, "evidence": e, "quote": TERMINATION}],
        )

    llm.script = script
    result = await ask(api, "How can the agreement be terminated?", contract)
    assert result["grounding"] == "grounded"
    assert result["answer"] == "Either party can end it with 30 days' written notice [1]."
    citation = result["citations"][0]
    assert citation["status"] == "verified"
    text = await canonical(contract)
    assert text[citation["char_start"] : citation["char_end"]] == citation["quote"]
    assert citation["quote"] == TERMINATION
    assert citation["page_start"] == citation["page_end"] == 2  # termination clause is on page 2
    assert citation["document_filename"] == "contract-a.pdf"
    # The contract is small: the model was given its full text, and the notes say so.
    assert any("based on the full text" in n for n in result["notes"])


async def test_quote_with_formatting_differences_returns_canonical_text(
    api: httpx.AsyncClient, contract: str, llm: FakeLLM
) -> None:
    def script(messages: list[ChatMessage]) -> str:
        e = find(evidence_in(messages), "terminate this Agreement")
        quote = "\u201cby giving THIRTY (30) days\u2019   written notice\u201d"
        return answer_output("Thirty days' notice [1].", [{"n": 1, "evidence": e, "quote": quote}])

    llm.script = script
    citation = (await ask(api, "What notice period applies to termination?", contract))[
        "citations"
    ][0]
    assert citation["status"] == "verified"
    assert re.sub(r"\s+", " ", citation["quote"]) == "by giving thirty (30) days' written notice"
    assert (
        citation["quote"]
        == (await canonical(contract))[citation["char_start"] : citation["char_end"]]
    )


async def test_llm_page_numbers_are_ignored(
    api: httpx.AsyncClient, contract: str, llm: FakeLLM
) -> None:
    def script(messages: list[ChatMessage]) -> str:
        e = find(evidence_in(messages), "AED 100,000")
        return answer_output(
            "Capped at AED 100,000 [1].",
            [{"n": 1, "evidence": e, "quote": "shall not exceed AED 100,000", "page": 99}],
        )

    llm.script = script
    citation = (await ask(api, "What is the liability cap?", contract))["citations"][0]
    assert citation["status"] == "verified" and citation["page_start"] == 3  # from storage


# --- hallucinated, partial, mislabelled ---------------------------------------------------------


async def test_hallucinated_quote_is_never_presented_as_an_answer(
    api: httpx.AsyncClient, contract: str, llm: FakeLLM
) -> None:
    def script(messages: list[ChatMessage]) -> str:
        e = find(evidence_in(messages), "terminate this Agreement")
        return answer_output(
            "Termination requires 90 days' notice [1].",
            [{"n": 1, "evidence": e, "quote": "Either party may terminate on ninety days notice"}],
        )

    llm.script = script
    result = await ask(api, "How can the agreement be terminated?", contract)
    assert result["grounding"] == "ungrounded"
    assert result["answer"] == INSUFFICIENT_ANSWER
    assert result["unsupported_draft"] == "Termination requires 90 days' notice [1]."
    assert result["citations"][0]["status"] == "unverified"
    assert result["citations"][0]["quote"] is None and result["citations"][0]["char_start"] is None


async def test_partially_supported_answer_flags_the_unsupported_part(
    api: httpx.AsyncClient, contract: str, llm: FakeLLM
) -> None:
    def script(messages: list[ChatMessage]) -> str:
        e = find(evidence_in(messages), "terminate this Agreement")
        return answer_output(
            "Either party may terminate [1], and a break fee applies [2].",
            [
                {"n": 1, "evidence": e, "quote": "Either party may terminate this Agreement"},
                {"n": 2, "evidence": e, "quote": "a break fee of AED 50,000 applies"},
            ],
        )

    llm.script = script
    result = await ask(api, "How can the agreement be terminated?", contract)
    assert result["grounding"] == "partially_grounded"
    assert [c["status"] for c in result["citations"]] == ["verified", "unverified"]
    assert any("[2] could not be verified" in n for n in result["notes"])


async def test_mislabelled_evidence_in_the_same_document_is_relocated(
    api: httpx.AsyncClient, contract: str, llm: FakeLLM
) -> None:
    def script(messages: list[ChatMessage]) -> str:
        evidence = evidence_in(messages)
        right = find(evidence, "AED 100,000")
        wrong = next(e for e in evidence if e != right)
        return answer_output(
            "Capped at AED 100,000 [1].",
            [{"n": 1, "evidence": wrong, "quote": "shall not exceed AED 100,000"}],
        )

    llm.script = script
    citation = (await ask(api, "What is the liability cap and the governing law?", contract))[
        "citations"
    ][0]
    assert citation["status"] == "verified"
    assert citation["evidence_id"] != citation["cited_evidence_id"]
    assert "Found in" in citation["reason"]


async def test_quote_from_another_document_is_rejected(
    api: httpx.AsyncClient, answer_app: FastAPI, contract: str, llm: FakeLLM
) -> None:
    other = await upload(api, answer_app, "contract-b.pdf", contract_pdf(CONTRACT_B))

    def script(messages: list[ChatMessage]) -> str:
        evidence = evidence_in(messages)
        a_evidence = find(evidence, "AED 100,000")  # document A's clause
        return answer_output(
            "The cap is AED 1,000,000 [1].",
            # Text that exists only in document B, attributed to document A's passage.
            [{"n": 1, "evidence": a_evidence, "quote": "shall not exceed AED 1,000,000"}],
        )

    llm.script = script
    result = await ask(api, "What is the liability cap?", contract, other)
    assert result["citations"][0]["status"] == "unverified"
    assert result["grounding"] == "ungrounded"


async def test_multi_document_answer_cites_each_document(
    api: httpx.AsyncClient, answer_app: FastAPI, contract: str, llm: FakeLLM
) -> None:
    other = await upload(api, answer_app, "contract-b.pdf", contract_pdf(CONTRACT_B))

    def script(messages: list[ChatMessage]) -> str:
        evidence = evidence_in(messages)
        return answer_output(
            "Contract A caps liability at AED 100,000 [1]; contract B at AED 1,000,000 [2].",
            [
                {"n": 1, "evidence": find(evidence, "AED 100,000"), "quote": QUOTE_A},
                {"n": 2, "evidence": find(evidence, "AED 1,000,000"), "quote": QUOTE_B},
            ],
        )

    llm.script = script
    result = await ask(api, "Compare the liability caps.", contract, other)
    assert result["grounding"] == "grounded"
    by_n = {c["n"]: c for c in result["citations"]}
    assert by_n[1]["document_id"] == contract and by_n[2]["document_id"] == other
    assert {e["document_id"] for e in result["evidence"]} == {contract, other}
    assert by_n[2]["quote"] == (await canonical(other))[by_n[2]["char_start"] : by_n[2]["char_end"]]


# --- insufficient evidence, weak tails, safety ----------------------------------------------------


async def test_model_saying_insufficient_is_not_found(
    api: httpx.AsyncClient, contract: str, llm: FakeLLM
) -> None:
    llm.script = lambda messages: f"<answer>{INSUFFICIENT_ANSWER}</answer>"
    result = await ask(api, "What is the liability cap?", contract)
    assert (result["grounding"], result["answer"], result["citations"]) == (
        "not_found",
        INSUFFICIENT_ANSWER,
        [],
    )


async def test_no_relevant_evidence_means_no_llm_call(
    api: httpx.AsyncClient, contract: str, llm: FakeLLM
) -> None:
    for question in ["chocolate cake recipe", "what is the"]:
        result = await ask(api, question, contract)
        assert result["grounding"] == "not_found" and result["answer"] == INSUFFICIENT_ANSWER
    assert llm.calls == []


async def test_weak_tail_is_not_shown_to_the_model(
    api: httpx.AsyncClient, large_contract: str, llm: FakeLLM
) -> None:
    # Too large for small-document mode: the excerpt selection runs and drops the weak tail.
    llm.script = lambda messages: f"<answer>{INSUFFICIENT_ANSWER}</answer>"
    result = await ask(
        api, "Which law governs the agreement and the term of the order form?", large_contract
    )
    shown = evidence_in(llm.calls[0])
    assert len(shown) == len(result["evidence"])
    weak = [e for e in result["excluded"] if e["reason"].startswith("weak")]
    assert weak, result["excluded"]  # the weak tail exists and was dropped
    excluded_ids = {e["chunk_id"] for e in result["excluded"]}
    assert not excluded_ids & {e["chunk_id"] for e in result["evidence"]}
    assert any(f"{len(weak)} weakly related passage(s)" in n for n in result["notes"])
    # The excerpt path's reading note (the full-text note is asserted elsewhere).
    assert any(
        f"based on {len(result['evidence'])} excerpt(s)" in n and "not on a full reading" in n
        for n in result["notes"]
    )
    assert not any("full text" in n for n in result["notes"])


async def test_context_budget_is_respected(db_settings: Settings, llm: FakeLLM) -> None:
    app = create_app(
        db_settings.model_copy(update={"answer_context_token_budget": 100}),
        embedder=ConceptEmbedder(),
        llm=llm,
    )
    async with app.router.lifespan_context(app):
        await truncate_all()
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as api:
            doc = await upload(api, app, "contract-a.pdf", contract_pdf(CONTRACT_A))
            llm.script = lambda messages: f"<answer>{INSUFFICIENT_ANSWER}</answer>"
            result = await ask(api, "What is the liability cap and the governing law?", doc)
    assert len(result["evidence"]) >= 1
    assert sum(len(e["text"]) // 4 + 30 for e in result["evidence"]) <= 100
    assert any(e["reason"] == "context budget reached" for e in result["excluded"])


async def test_malformed_model_output_is_handled(
    api: httpx.AsyncClient, contract: str, llm: FakeLLM
) -> None:
    llm.script = lambda messages: "The cap is AED 100,000.\n<citations>\nnot json at all\n"
    result = await ask(api, "What is the liability cap?", contract)
    assert result["grounding"] == "ungrounded" and result["answer"] == INSUFFICIENT_ANSWER
    assert any("did not use the answer format" in n for n in result["notes"])
    assert all("<answer>" not in n for n in result["notes"])  # notes never show markup
    assert any("malformed citation" in n for n in result["notes"])


async def test_low_quality_ocr_evidence_is_labelled_and_warned(
    api: httpx.AsyncClient, answer_app: FastAPI, llm: FakeLLM, monkeypatch: pytest.MonkeyPatch
) -> None:
    def low_confidence_ocr(page, *, dpi, language, page_index):  # type: ignore[no-untyped-def]
        words = [SourceWord(w, 72 + 80 * i, 100, 140 + 80 * i, 112) for i, w in
                 enumerate("the indemnity obligations survive termination".split())]  # fmt: skip
        return OcrResult([SourceBlock([SourceLine(words)], page_index=page_index)], 0.31)

    monkeypatch.setattr(pdf_extract, "ocr_page", low_confidence_ocr)
    monkeypatch.setattr(pdf_extract, "tesseract_version", lambda cmd=None: "5.0")
    doc = await upload(api, answer_app, "scan.pdf", factory.scanned_pdf(pages=1))

    def script(messages: list[ChatMessage]) -> str:
        e = find(evidence_in(messages), "indemnity obligations")
        return answer_output(
            "Indemnity obligations survive termination [1] (low-quality scan).",
            [{"n": 1, "evidence": e, "quote": "the indemnity obligations survive termination"}],
        )

    llm.script = script
    result = await ask(api, "Do the indemnity obligations survive termination?", doc)
    assert "LOW-QUALITY OCR (pages 1)" in llm.calls[0][-1].content
    citation = result["citations"][0]
    assert citation["status"] == "verified"
    assert citation["quality"]["level"] == "low_quality_ocr"
    notes = " ".join(result["notes"])
    assert "Quote [1] comes from a low-quality OCR page" in notes
    assert "relevant content on those pages may have been missed" in notes


async def test_llm_unavailable_and_not_configured(
    api: httpx.AsyncClient, answer_app: FastAPI, contract: str, llm: FakeLLM, db_settings: Settings
) -> None:
    def failing(messages: list[ChatMessage]) -> str:
        raise LLMUnavailableError("The answering service is temporarily unavailable.")

    llm.script = failing
    response = await api.post(
        ANSWERS, json={"question": "liability cap", "document_ids": [contract]}
    )
    assert response.status_code == 502 and response.json()["error"]["code"] == "LLM_UNAVAILABLE"

    unconfigured = create_app(db_settings, embedder=ConceptEmbedder())  # no LLM settings
    async with unconfigured.router.lifespan_context(unconfigured):
        transport = httpx.ASGITransport(app=unconfigured)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                ANSWERS, json={"question": "liability cap", "document_ids": [contract]}
            )
    assert response.status_code == 503 and response.json()["error"]["code"] == "LLM_NOT_CONFIGURED"


@pytest.mark.parametrize(
    ("body", "status", "code"),
    [
        ({"question": "  ", "document_ids": [MISSING]}, 422, "VALIDATION_ERROR"),
        ({"question": "cap", "document_ids": []}, 422, "VALIDATION_ERROR"),
        ({"question": "cap", "document_ids": [MISSING]}, 404, "DOCUMENT_NOT_FOUND"),
        ({"question": "x" * 1001, "document_ids": [MISSING]}, 422, "QUERY_TOO_LONG"),
    ],
)
async def test_malformed_requests(
    api: httpx.AsyncClient,
    body: dict,
    status: int,
    code: str,  # type: ignore[type-arg]
) -> None:
    response = await api.post(ANSWERS, json=body)
    assert response.status_code == status and response.json()["error"]["code"] == code


async def test_stale_index_is_refused(api: httpx.AsyncClient, contract: str) -> None:
    async with get_sessionmaker()() as session:
        await session.execute(update(Document).values(index_version=2))
        await session.commit()
    response = await api.post(ANSWERS, json={"question": "cap", "document_ids": [contract]})
    assert response.status_code == 409


async def test_answers_are_deterministic(
    api: httpx.AsyncClient, contract: str, llm: FakeLLM
) -> None:
    def script(messages: list[ChatMessage]) -> str:
        e = find(evidence_in(messages), "terminate this Agreement")
        return answer_output(
            "Notice [1].", [{"n": 1, "evidence": e, "quote": "may terminate this Agreement"}]
        )

    llm.script = script
    first = await ask(api, "How can the agreement be terminated?", contract)
    assert await ask(api, "How can the agreement be terminated?", contract) == first
    assert llm.calls[0] == llm.calls[1]  # identical prompt, so a deterministic model agrees too


async def test_no_database_connection_is_held_during_the_llm_call(
    api: httpx.AsyncClient, contract: str, llm: FakeLLM
) -> None:
    checked_out: list[int] = []
    llm.on_call = lambda: checked_out.append(get_engine().pool.checkedout())
    llm.script = lambda messages: f"<answer>{INSUFFICIENT_ANSWER}</answer>"
    await ask(api, "What is the liability cap?", contract)
    assert checked_out == [0]


# --- answer integrity (regression: "Which book inspired the student the most, and why?") ---------

LDQUO, RDQUO, A_CIRC, EURO = chr(0x201C), chr(0x201D), chr(0xE2), chr(0x20AC)
TITLE = f"{LDQUO}Wings of Fire{RDQUO}"
BOOK_QUESTION = "Which book inspired the student the most, and why?"
QUOTE_TITLE = f"{TITLE} by Dr. A.P.J. Abdul Kalam has inspired the student the most."
QUOTE_WHY = "It shows how persistence, discipline and a clear purpose can overcome failures."
QUOTE_UNICODE = (
    f"The book costs {EURO}10 at the caf{chr(0xE9)} {chr(0x2014)} a bargain{chr(0x2026)}"
)


@pytest.fixture
async def essay(api: httpx.AsyncClient, answer_app: FastAPI) -> str:
    paragraphs = [
        "My Favourite Book",
        f"{QUOTE_TITLE} {QUOTE_WHY} {QUOTE_UNICODE}",
        "The student also enjoys football and music on weekends.",
    ]
    return await upload(api, answer_app, "essay.docx", factory.docx_file(paragraphs))


def book_script(
    answer: str, citations: list[tuple[int, str]]
) -> Callable[[list[ChatMessage]], str]:
    def script(messages: list[ChatMessage]) -> str:
        e = find(evidence_in(messages), "Wings of Fire")
        return answer_output(answer, [{"n": n, "evidence": e, "quote": q} for n, q in citations])

    return script


async def test_valid_answer_with_multiple_verified_citations_and_unicode(
    api: httpx.AsyncClient, essay: str, llm: FakeLLM
) -> None:
    answer = (
        f"The student was most inspired by {TITLE} by Dr. A.P.J. Abdul Kalam [1]. It shows that "
        f"persistence and discipline overcome failures [2]. It costs {EURO}10 {chr(0x2014)} a "
        f"bargain{chr(0x2026)} [3]"
    )
    llm.script = book_script(answer, [(1, QUOTE_TITLE), (2, QUOTE_WHY), (3, QUOTE_UNICODE)])
    response = await api.post(ANSWERS, json={"question": BOOK_QUESTION, "document_ids": [essay]})
    assert response.status_code == 200
    # Declared UTF-8, and the bytes are real UTF-8 (a client cannot mis-decode them as Latin-1).
    assert response.headers["content-type"] == "application/json; charset=utf-8"
    assert TITLE.encode("utf-8") in response.content
    result = response.json()
    assert result["grounding"] == "grounded"
    assert result["answer"] == answer  # complete, legitimate Unicode unchanged
    assert [c["n"] for c in result["citations"]] == [1, 2, 3]
    text = await canonical(essay)
    for citation, quote in zip(
        result["citations"], (QUOTE_TITLE, QUOTE_WHY, QUOTE_UNICODE), strict=True
    ):
        assert citation["status"] == "verified"
        assert citation["quote"] == quote  # including the source's own curly quotes
        assert text[citation["char_start"] : citation["char_end"]] == quote


@pytest.mark.parametrize(
    "answer",
    [
        f"The book is {A_CIRC}{EURO}{chr(0x153)}Wings of Fire{A_CIRC}{EURO} by Dr. Kalam [1].",
        f"The book that inspired the student the most is {A_CIRC}Wings of Fire{A_CIRC}Wings of "
        f"Fire{A_CIRC}Wings of Fire{A_CIRC} [1].",
    ],
    ids=["mojibake", "repetition"],
)
async def test_corrupted_answer_is_refused_even_with_a_verified_quote(
    api: httpx.AsyncClient, essay: str, llm: FakeLLM, answer: str
) -> None:
    llm.script = book_script(answer, [(1, QUOTE_TITLE)])
    response = await api.post(ANSWERS, json={"question": BOOK_QUESTION, "document_ids": [essay]})
    assert response.status_code == 502
    error = response.json()["error"]
    assert error["code"] == "LLM_OUTPUT_INVALID" and "Wings" not in error["message"]


async def test_truncated_output_is_refused(
    api: httpx.AsyncClient, essay: str, llm: FakeLLM
) -> None:
    def script(messages: list[ChatMessage]) -> str:
        e = find(evidence_in(messages), "Wings of Fire")
        full = answer_output(
            "The book is Wings of Fire [1]. It teaches persistence and discipline [2].",
            [
                {"n": 1, "evidence": e, "quote": QUOTE_TITLE},
                {"n": 2, "evidence": e, "quote": QUOTE_WHY},
            ],
        )
        return full[: full.index(QUOTE_WHY) + 20]  # cut off inside the second citation

    llm.script = script
    response = await api.post(ANSWERS, json={"question": BOOK_QUESTION, "document_ids": [essay]})
    assert response.status_code == 502
    assert response.json()["error"]["code"] == "LLM_OUTPUT_INVALID"
    assert "cut off" in response.json()["error"]["message"]


async def test_insufficient_reply_without_closing_tag_is_not_found_not_an_error(
    api: httpx.AsyncClient, essay: str, llm: FakeLLM
) -> None:
    llm.script = lambda messages: f"<answer>{INSUFFICIENT_ANSWER}"
    result = await ask(api, BOOK_QUESTION, essay)
    assert result["grounding"] == "not_found" and result["answer"] == INSUFFICIENT_ANSWER


async def test_non_matching_quote_is_never_verified(
    api: httpx.AsyncClient, essay: str, llm: FakeLLM
) -> None:
    llm.script = book_script(
        "The student was inspired by The Alchemist [1].",
        [(1, "The Alchemist by Paulo Coelho has inspired the student the most.")],
    )
    result = await ask(api, BOOK_QUESTION, essay)
    assert result["grounding"] == "ungrounded" and result["answer"] == INSUFFICIENT_ANSWER
    assert result["citations"][0]["status"] == "unverified"


async def test_markers_without_citations_are_not_grounded(
    api: httpx.AsyncClient, essay: str, llm: FakeLLM
) -> None:
    llm.script = book_script(
        f"The book is {TITLE} [1]. The student read it twelve times last year [3].",
        [(1, QUOTE_TITLE)],
    )
    result = await ask(api, BOOK_QUESTION, essay)
    assert result["grounding"] == "partially_grounded"
    assert any("[3]" in n and "no citation" in n for n in result["notes"])


async def test_uncited_claim_is_not_grounded(
    api: httpx.AsyncClient, essay: str, llm: FakeLLM
) -> None:
    llm.script = book_script(
        f"The book is {TITLE} [1]. The student plans to become a rocket scientist like Kalam.",
        [(1, QUOTE_TITLE)],
    )
    result = await ask(api, BOOK_QUESTION, essay)
    assert result["grounding"] == "partially_grounded"
    assert any("no citation" in n and "rocket scientist" in n for n in result["notes"])


async def test_provider_error_response_never_contains_the_api_key(
    db_settings: Settings, essay: str
) -> None:
    secret = "sk-test-SECRET-987654"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": {"message": f"Incorrect API key: {secret}"}})

    configured = db_settings.model_copy(
        update={
            "llm_base_url": "https://llm.example/v1",
            "llm_api_key": SecretStr(secret),
            "llm_model": "cheap-model",
        }
    )
    client = OpenAICompatibleClient(
        configured, http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
    )
    app = create_app(configured, embedder=ConceptEmbedder(), llm=client)
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as api_client:
            response = await api_client.post(
                ANSWERS, json={"question": BOOK_QUESTION, "document_ids": [essay]}
            )
    assert response.status_code == 502
    assert response.json()["error"]["code"] == "LLM_UNAVAILABLE"
    assert secret not in response.text


# --- whole-document verification (Phase 6, task 1) ------------------------------------------------

GOVERNING_LAW = "This Agreement is governed by the laws of the United Arab Emirates."


async def test_quote_outside_the_evidence_is_verified_in_its_own_document(
    api: httpx.AsyncClient, large_contract: str, llm: FakeLLM
) -> None:
    def script(messages: list[ChatMessage]) -> str:
        evidence = evidence_in(messages)
        assert not any("United Arab Emirates" in text for text in evidence.values())
        e = find(evidence, "terminate this Agreement")
        return answer_output(
            "Either party may terminate [1]. The governing law is that of the UAE [2].",
            [
                {"n": 1, "evidence": e, "quote": "Either party may terminate this Agreement"},
                {"n": 2, "evidence": e, "quote": GOVERNING_LAW},
            ],
        )

    llm.script = script
    result = await ask(api, "How can the agreement be terminated?", large_contract)
    first, second = result["citations"]
    assert first["status"] == second["status"] == "verified"
    assert first["evidence_id"] == first["cited_evidence_id"] and first["reason"] is None
    # Found by searching the whole document: exact location, but not attributed to the evidence.
    text = await canonical(large_contract)
    assert second["evidence_id"] is None
    assert "outside the passages the model was given" in second["reason"]
    assert second["quote"] == GOVERNING_LAW
    assert text[second["char_start"] : second["char_end"]] == GOVERNING_LAW
    assert second["page_start"] == second["page_end"] == 4
    assert second["document_id"] == large_contract and second["quality"]["level"] == "text"


async def test_partial_word_quote_is_not_verified_end_to_end(
    api: httpx.AsyncClient, contract: str, llm: FakeLLM
) -> None:
    def script(messages: list[ChatMessage]) -> str:
        e = find(evidence_in(messages), "terminate this Agreement")
        return answer_output(
            "Either party may terminate [1].",
            [{"n": 1, "evidence": e, "quote": "ither party may terminate this Agreement"}],
        )

    llm.script = script
    result = await ask(api, "How can the agreement be terminated?", contract)
    assert result["grounding"] == "ungrounded" and result["answer"] == INSUFFICIENT_ANSWER
    assert "middle of a word" in result["citations"][0]["reason"]


# --- repeated occurrences (Phase 6, task 2) -------------------------------------------------------

PRICING_QUOTE = "Pricing applies as set out in the Order Form agreed by the parties."


async def test_quote_repeated_outside_the_evidence_is_flagged_ambiguous(
    api: httpx.AsyncClient, large_contract: str, llm: FakeLLM
) -> None:
    # Contract A repeats its pricing page (pages 5 and 6); the model is asked about termination,
    # so neither pricing page is in its evidence.
    def script(messages: list[ChatMessage]) -> str:
        evidence = evidence_in(messages)
        assert not any("Pricing applies" in text for text in evidence.values())
        e = find(evidence, "terminate this Agreement")
        return answer_output(
            "Either party may terminate [1]. Pricing follows the Order Form [2].",
            [
                {"n": 1, "evidence": e, "quote": "Either party may terminate this Agreement"},
                {"n": 2, "evidence": e, "quote": PRICING_QUOTE},
            ],
        )

    llm.script = script
    result = await ask(api, "How can the agreement be terminated?", large_contract)
    first, second = result["citations"]
    assert first["occurrence_count"] == 1 and first["location_ambiguous"] is False
    assert second["status"] == "verified" and second["evidence_id"] is None
    assert second["occurrence_count"] == 2 and second["location_ambiguous"] is True
    assert "It occurs 2 times outside those passages" in second["reason"]
    text = await canonical(large_contract)
    assert text[second["char_start"] : second["char_end"]] == PRICING_QUOTE
    assert second["page_start"] == 5  # the first of the two


async def test_unverified_citation_has_no_occurrence_data(
    api: httpx.AsyncClient, contract: str, llm: FakeLLM
) -> None:
    def script(messages: list[ChatMessage]) -> str:
        e = find(evidence_in(messages), "terminate this Agreement")
        return answer_output(
            "Termination needs 90 days [1].",
            [{"n": 1, "evidence": e, "quote": "Either party may terminate on ninety days notice"}],
        )

    llm.script = script
    citation = (await ask(api, "How can the agreement be terminated?", contract))["citations"][0]
    assert citation["status"] == "unverified"
    assert citation["occurrence_count"] is None and citation["location_ambiguous"] is False


# --- extraction differences (Phase 6, task 3) -----------------------------------------------------

NON_COMPETE_PAGES = [
    "1. Definitions. Agreement means this services agreement.",
    "9. Restrictions\n\n9.1 The Supplier accepts a non-\ncompete restriction for twelve months "
    "after termination of this Agreement.",
]


async def test_line_break_hyphen_quote_is_verified_end_to_end(
    api: httpx.AsyncClient, answer_app: FastAPI, llm: FakeLLM
) -> None:
    document = await upload(api, answer_app, "restrictions.pdf", contract_pdf(NON_COMPETE_PAGES))
    text = await canonical(document)
    assert "non-\ncompete" in text  # the extraction keeps the line break after the hyphen

    def script(messages: list[ChatMessage]) -> str:
        e = find(evidence_in(messages), "restriction for twelve months")
        quote = "The Supplier accepts a non-compete restriction for twelve months"
        return answer_output(
            "A twelve-month non-compete applies [1].", [{"n": 1, "evidence": e, "quote": quote}]
        )

    llm.script = script
    citation = (await ask(api, "Is there a non-compete restriction on the Supplier?", document))[
        "citations"
    ][0]
    assert citation["status"] == "verified" and citation["reason"] is None
    assert citation["quote"] == "The Supplier accepts a non-\ncompete restriction for twelve months"
    assert text[citation["char_start"] : citation["char_end"]] == citation["quote"]
    assert citation["page_start"] == citation["page_end"] == 2


async def test_split_word_quote_is_verified_and_flagged_end_to_end(
    api: httpx.AsyncClient, answer_app: FastAPI, llm: FakeLLM
) -> None:
    paragraphs = [
        "1. This Agreement is made between the parties named below.",
        "2. Each party shall sign this Agree ment before the Effective Date of the services.",
    ]
    document = await upload(api, answer_app, "signing.docx", factory.docx_file(paragraphs))

    def script(messages: list[ChatMessage]) -> str:
        e = find(evidence_in(messages), "Effective Date")
        quote = "Each party shall sign this Agreement before the Effective Date"
        return answer_output(
            "Both sign before the Effective Date [1].", [{"n": 1, "evidence": e, "quote": quote}]
        )

    llm.script = script
    citation = (await ask(api, "When must each party sign the Agreement?", document))["citations"][
        0
    ]
    assert citation["status"] == "verified"
    assert citation["quote"] == "Each party shall sign this Agree ment before the Effective Date"
    assert "spacing differences" in citation["reason"]
    text = await canonical(document)
    assert text[citation["char_start"] : citation["char_end"]] == citation["quote"]
    assert citation["quality"]["level"] == "not_applicable"  # DOCX: no pages


# --- PDF highlight rectangles (Phase 6, task 4) ---------------------------------------------------


async def stored_page(document_id: str, page: int) -> DocumentPage:
    async with get_sessionmaker()() as session:
        row = await session.get(DocumentPage, (uuid.UUID(document_id), page))
        assert row is not None
        return row


def covering(page: DocumentPage, start: int, end: int) -> list[list[float]]:
    return [w for w in page.words if w[0] < end and start < w[1]]


async def test_verified_pdf_quote_gets_a_line_rectangle_from_stored_word_boxes(
    api: httpx.AsyncClient, contract: str, llm: FakeLLM
) -> None:
    def script(messages: list[ChatMessage]) -> str:
        e = find(evidence_in(messages), "terminate this Agreement")
        quote = "Either party may terminate this Agreement"
        # The model's page claim is ignored: the rectangle comes from the verified offsets.
        return answer_output(
            "Either party may terminate [1].",
            [{"n": 1, "evidence": e, "quote": quote, "page": 1}],
        )

    llm.script = script
    citation = (await ask(api, "How can the agreement be terminated?", contract))["citations"][0]
    assert citation["status"] == "verified" and citation["reason"] is None
    page = await stored_page(contract, 2)
    words = covering(page, citation["char_start"], citation["char_end"])
    assert len(words) == 6  # Either party may terminate this Agreement
    assert citation["highlights"] == [
        {
            "page": 2,
            "page_width": page.width,
            "page_height": page.height,
            "x0": min(w[2] for w in words),
            "y0": min(w[3] for w in words),
            "x1": max(w[4] for w in words),
            "y1": max(w[5] for w in words),
        }
    ]


async def test_quote_over_lines_and_a_page_break_gets_rectangles_per_line_and_page(
    api: httpx.AsyncClient, answer_app: FastAPI, llm: FakeLLM
) -> None:
    document = await upload(api, answer_app, "delivery.pdf", factory.furniture_pdf(pages=3))
    text = await canonical(document)
    quote = (
        "on all matters. The Supplier shall deliver the goods within ten (10) business days of "
        "the order. Clause 2 text"
    )

    def script(messages: list[ChatMessage]) -> str:
        e = find(evidence_in(messages), "Clause 2 text")
        return answer_output(
            "Delivery is due within ten business days [1].",
            [{"n": 1, "evidence": e, "quote": quote}],
        )

    llm.script = script
    citation = (await ask(api, "When must the Supplier deliver the goods?", document))["citations"][
        0
    ]
    assert citation["status"] == "verified"
    assert (citation["page_start"], citation["page_end"]) == (1, 2)
    highlights = citation["highlights"]
    assert {h["page"] for h in highlights} == {1, 2}
    # One rectangle per line of the quote: count the line breaks it crosses on each page.
    matched = text[citation["char_start"] : citation["char_end"]]
    before_break, after_break = matched.split("\n\n")
    assert len([h for h in highlights if h["page"] == 1]) == before_break.count("\n") + 1
    assert len([h for h in highlights if h["page"] == 2]) == after_break.count("\n") + 1
    for h in highlights:
        page = await stored_page(document, h["page"])
        assert (h["page_width"], h["page_height"]) == (page.width, page.height)
        assert 0 <= h["x0"] < h["x1"] <= page.width and 0 <= h["y0"] < h["y1"] <= page.height
    # The running header was removed as page furniture: it is not text, so never highlighted.
    assert "ACME CORP" not in text


async def test_running_header_quote_is_never_verified_or_highlighted(
    api: httpx.AsyncClient, answer_app: FastAPI, llm: FakeLLM
) -> None:
    document = await upload(api, answer_app, "header.pdf", factory.furniture_pdf(pages=3))

    def script(messages: list[ChatMessage]) -> str:
        e = find(evidence_in(messages), "deliver the goods")
        return answer_output(
            "The document is confidential [1].",
            [{"n": 1, "evidence": e, "quote": "ACME CORP - CONFIDENTIAL"}],
        )

    llm.script = script
    citation = (await ask(api, "When must the Supplier deliver the goods?", document))["citations"][
        0
    ]
    assert citation["status"] == "unverified" and citation["highlights"] == []


async def test_repeated_text_is_highlighted_where_the_verifier_located_it(
    api: httpx.AsyncClient, contract: str, llm: FakeLLM
) -> None:
    # The pricing sentence is on pages 5 and 6; outside the evidence, the first is reported
    # (and flagged ambiguous), so only page 5 is highlighted.
    def script(messages: list[ChatMessage]) -> str:
        e = find(evidence_in(messages), "terminate this Agreement")
        return answer_output(
            "Pricing follows the Order Form [1].",
            [{"n": 1, "evidence": e, "quote": PRICING_QUOTE}],
        )

    llm.script = script
    citation = (await ask(api, "How can the agreement be terminated?", contract))["citations"][0]
    assert citation["location_ambiguous"] and citation["page_start"] == 5
    assert citation["highlights"] and {h["page"] for h in citation["highlights"]} == {5}
    page = await stored_page(contract, 5)
    words = covering(page, citation["char_start"], citation["char_end"])
    assert min(w[3] for w in words) == min(h["y0"] for h in citation["highlights"])


async def test_unverified_quotes_and_word_documents_get_no_rectangles(
    api: httpx.AsyncClient, contract: str, essay: str, llm: FakeLLM
) -> None:
    def script(messages: list[ChatMessage]) -> str:
        e = find(evidence_in(messages), "terminate this Agreement")
        return answer_output(
            "Termination needs 90 days [1].",
            [{"n": 1, "evidence": e, "quote": "Either party may terminate on ninety days notice"}],
        )

    llm.script = script
    citation = (await ask(api, "How can the agreement be terminated?", contract))["citations"][0]
    assert citation["status"] == "unverified" and citation["highlights"] == []

    llm.script = book_script(f"The book is {TITLE} [1].", [(1, QUOTE_TITLE)])
    docx = (await ask(api, BOOK_QUESTION, essay))["citations"][0]
    assert docx["status"] == "verified" and docx["highlights"] == []
    assert docx["reason"] is None  # no pages: nothing to highlight, nothing to explain


async def test_missing_word_geometry_keeps_the_quote_verified_without_rectangles(
    api: httpx.AsyncClient, contract: str, llm: FakeLLM
) -> None:
    async with get_sessionmaker()() as session:
        await session.execute(
            update(DocumentPage)
            .where(DocumentPage.document_id == uuid.UUID(contract), DocumentPage.page_number == 2)
            .values(words=[])
        )
        await session.commit()

    def script(messages: list[ChatMessage]) -> str:
        e = find(evidence_in(messages), "terminate this Agreement")
        return answer_output(
            "Either party may terminate [1].",
            [{"n": 1, "evidence": e, "quote": "Either party may terminate this Agreement"}],
        )

    llm.script = script
    citation = (await ask(api, "How can the agreement be terminated?", contract))["citations"][0]
    assert citation["status"] == "verified" and citation["page_start"] == 2
    assert citation["highlights"] == []
    assert "could not be determined from the extracted layout" in citation["reason"]


async def test_document_reprocessed_during_the_answer_gets_no_rectangles(
    api: httpx.AsyncClient, contract: str, llm: FakeLLM
) -> None:
    def script(messages: list[ChatMessage]) -> str:
        e = find(evidence_in(messages), "terminate this Agreement")
        return answer_output(
            "Either party may terminate [1].",
            [{"n": 1, "evidence": e, "quote": "Either party may terminate this Agreement"}],
        )

    llm.script = script
    complete = llm.complete

    async def reprocessed_meanwhile(messages: list[ChatMessage]) -> LLMResult:
        async with get_sessionmaker()() as session:  # as if a re-index committed mid-answer
            await session.execute(
                update(Document)
                .where(Document.id == uuid.UUID(contract))
                .values(processed_at=text_interval("now() + interval '1 minute'"))
            )
            await session.commit()
        return await complete(messages)

    llm.complete = reprocessed_meanwhile  # type: ignore[method-assign]
    citation = (await ask(api, "How can the agreement be terminated?", contract))["citations"][0]
    assert citation["status"] == "verified" and citation["highlights"] == []
    assert "could not be determined" in citation["reason"]


@needs_tesseract
async def test_ocr_page_quote_is_highlighted_in_the_same_coordinates(
    api: httpx.AsyncClient, answer_app: FastAPI, llm: FakeLLM
) -> None:
    document = await upload(api, answer_app, "scan.pdf", factory.scanned_pdf(pages=1))
    page = await stored_page(document, 1)
    assert page.extraction == "ocr" and page.words
    text = await canonical(document)
    quote = " ".join(text.split()[:6])  # the OCR text itself, so the quote verifies

    def script(messages: list[ChatMessage]) -> str:
        e = next(iter(evidence_in(messages)))
        return answer_output("The clause says so [1].", [{"n": 1, "evidence": e, "quote": quote}])

    llm.script = script
    result = await ask(api, " ".join(text.split()[:4]), document)
    citation = result["citations"][0]
    assert citation["status"] == "verified"
    words = covering(page, citation["char_start"], citation["char_end"])
    highlights = citation["highlights"]
    assert highlights and {h["page"] for h in highlights} == {1}
    assert all((h["page_width"], h["page_height"]) == (page.width, page.height) for h in highlights)
    assert min(h["x0"] for h in highlights) == min(w[2] for w in words)
    assert max(h["x1"] for h in highlights) == max(w[4] for w in words)


async def test_only_the_cited_pages_geometry_is_loaded_for_a_large_document(
    api: httpx.AsyncClient, answer_app: FastAPI, contract: str, llm: FakeLLM
) -> None:
    large = await upload(api, answer_app, "large.pdf", factory.large_pdf_with_needle())
    statements: list[tuple[str, object]] = []

    def capture(conn, cursor, statement, parameters, context, executemany):  # type: ignore[no-untyped-def]
        if "document_pages.words" in statement and "documents.processed_at" in statement:
            statements.append((statement, parameters))

    def script(messages: list[ChatMessage]) -> str:
        e = find(evidence_in(messages), "Escrow Agent")
        return answer_output(
            "The deposit is released within seven days [1].",
            [{"n": 1, "evidence": e, "quote": factory.NEEDLE}],
        )

    llm.script = script
    engine = get_engine().sync_engine
    event.listen(engine, "before_cursor_execute", capture)
    try:
        result = await ask(api, "When does the Escrow Agent release the Deposit?", large, contract)
    finally:
        event.remove(engine, "before_cursor_execute", capture)
    citation = result["citations"][0]
    assert citation["status"] == "verified" and citation["page_start"] == 137
    assert {h["page"] for h in citation["highlights"]} == {137}
    # One geometry query, for the cited document's page only: not the other 149 pages, and
    # nothing from the other selected document.
    assert len(statements) == 1
    parameters = [str(p) for p in statements[0][1]]  # type: ignore[union-attr]
    assert large in parameters and contract not in parameters
    assert "137" in parameters and len(parameters) == 2


def fail_geometry_query(error: Exception):  # type: ignore[no-untyped-def]
    """A cursor hook that raises `error` for the highlight geometry query only."""

    def hook(conn, cursor, statement, parameters, context, executemany):  # type: ignore[no-untyped-def]
        if "document_pages.words" in statement and "documents.processed_at" in statement:
            raise error

    return hook


async def test_geometry_read_failure_keeps_the_verified_answer(
    api: httpx.AsyncClient, contract: str, llm: FakeLLM
) -> None:
    def script(messages: list[ChatMessage]) -> str:
        e = find(evidence_in(messages), "terminate this Agreement")
        return answer_output(
            "Either party may terminate [1].",
            [{"n": 1, "evidence": e, "quote": "Either party may terminate this Agreement"}],
        )

    llm.script = script
    hook = fail_geometry_query(OperationalError("SELECT ...", {}, OSError("connection lost")))
    engine = get_engine().sync_engine
    event.listen(engine, "before_cursor_execute", hook)
    try:
        response = await api.post(
            ANSWERS,
            json={"question": "How can the agreement be terminated?", "document_ids": [contract]},
        )
    finally:
        event.remove(engine, "before_cursor_execute", hook)
    assert response.status_code == 200
    result = response.json()
    assert result["grounding"] == "grounded"
    assert result["answer"] == "Either party may terminate [1]."
    citation = result["citations"][0]
    assert citation["status"] == "verified" and citation["page_start"] == 2
    assert citation["quote"] == "Either party may terminate this Agreement"
    text = await canonical(contract)
    assert text[citation["char_start"] : citation["char_end"]] == citation["quote"]
    assert citation["highlights"] == []  # nothing fabricated
    assert citation["reason"] == (
        "Its position on the page could not be loaded just now, so it is not highlighted."
    )
    assert get_engine().pool.checkedout() == 0  # the failed read released its connection


async def test_unexpected_errors_in_the_geometry_step_are_not_hidden(
    api: httpx.AsyncClient, contract: str, llm: FakeLLM
) -> None:
    def script(messages: list[ChatMessage]) -> str:
        e = find(evidence_in(messages), "terminate this Agreement")
        return answer_output(
            "Either party may terminate [1].",
            [{"n": 1, "evidence": e, "quote": "Either party may terminate this Agreement"}],
        )

    llm.script = script
    hook = fail_geometry_query(RuntimeError("a bug, not a database failure"))
    engine = get_engine().sync_engine
    event.listen(engine, "before_cursor_execute", hook)
    try:
        response = await api.post(
            ANSWERS,
            json={"question": "How can the agreement be terminated?", "document_ids": [contract]},
        )
    finally:
        event.remove(engine, "before_cursor_execute", hook)
    assert response.status_code == 500
    assert response.json()["error"]["code"] == "INTERNAL_ERROR"


# --- Word citation block ranges (Phase 6, task 5) -------------------------------------------------

TERMINATION_CLAUSE = (
    "2.1 Either party may terminate this Agreement on thirty (30) days written notice."
)


async def stored_blocks(document_id: str) -> list[DocumentBlock]:
    async with get_sessionmaker()() as session:
        rows = await session.execute(
            select(DocumentBlock)
            .where(DocumentBlock.document_id == uuid.UUID(document_id))
            .order_by(DocumentBlock.ordinal)
        )
        return list(rows.scalars())


def quoting(anchor: str, *quotes: str) -> Callable[[list[ChatMessage]], str]:
    """A model that cites the evidence passage containing `anchor` for each quote."""

    def script(messages: list[ChatMessage]) -> str:
        e = find(evidence_in(messages), anchor)
        markers = " ".join(f"Point [{n}]." for n in range(1, len(quotes) + 1))
        return answer_output(
            markers, [{"n": n, "evidence": e, "quote": q} for n, q in enumerate(quotes, 1)]
        )

    return script


async def assert_block_ranges_match(document_id: str, citation: dict) -> None:  # type: ignore[type-arg]
    """Every range lies in its stored block, and together they spell the verified quote."""
    text = await canonical(document_id)
    blocks = {b.ordinal: b for b in await stored_blocks(document_id)}
    pieces = []
    for r in citation["block_ranges"]:
        block = blocks[r["block"]]
        assert (r["kind"], r["block_char_start"], r["block_char_end"]) == (
            block.kind,
            block.char_start,
            block.char_end,
        )
        block_text = text[block.char_start : block.char_end]
        assert 0 <= r["start"] < r["end"] <= len(block_text)
        pieces.append(block_text[r["start"] : r["end"]])
    assert "\n\n".join(pieces) == citation["quote"]
    assert citation["highlights"] == []  # Word documents have no page rectangles


@pytest.fixture
async def numbered(api: httpx.AsyncClient, answer_app: FastAPI) -> str:
    return await upload(api, answer_app, "msa.docx", factory.numbered_docx())


async def test_docx_quote_in_a_numbered_clause_gets_its_block_range(
    api: httpx.AsyncClient, numbered: str, llm: FakeLLM
) -> None:
    llm.script = quoting("terminate this Agreement", TERMINATION_CLAUSE)
    citation = (await ask(api, "How can the agreement be terminated?", numbered))["citations"][0]
    assert citation["status"] == "verified" and citation["reason"] is None
    (r,) = citation["block_ranges"]
    assert r["kind"] == "paragraph" and r["start"] == 0  # the "2.1" label is part of the block
    await assert_block_ranges_match(numbered, citation)


async def test_docx_quote_across_paragraphs_and_a_table_row(
    api: httpx.AsyncClient, numbered: str, llm: FakeLLM
) -> None:
    text = await canonical(numbered)
    across = text[text.index("becomes insolvent.") : text.index("LIABILITY") + len("LIABILITY")]
    row = "Fee\nAED 100,000"
    assert across.count("\n\n") == 1 and row in text
    llm.script = quoting("terminate this Agreement", across, row)
    result = await ask(api, "How can the agreement be terminated?", numbered)
    first, second = result["citations"]
    assert first["status"] == second["status"] == "verified"
    # "3. LIABILITY" is stored as a heading block by structure detection.
    assert [r["kind"] for r in first["block_ranges"]] == ["paragraph", "heading"]
    assert [r["kind"] for r in second["block_ranges"]] == ["table_row"]
    await assert_block_ranges_match(numbered, first)
    await assert_block_ranges_match(numbered, second)


async def test_docx_manual_line_break_stays_in_one_block(
    api: httpx.AsyncClient, answer_app: FastAPI, llm: FakeLLM
) -> None:
    paragraphs = ["Notices must be given in writing.\nEmail counts as writing for notices."]
    document = await upload(api, answer_app, "notices.docx", factory.docx_file(paragraphs))
    assert "in writing.\nEmail counts" in await canonical(document)
    llm.script = quoting("Email counts", "given in writing.\nEmail counts as writing")
    citation = (await ask(api, "How must notices be given in writing?", document))["citations"][0]
    assert citation["status"] == "verified" and len(citation["block_ranges"]) == 1
    await assert_block_ranges_match(document, citation)


async def test_docx_repeated_paragraph_uses_the_verifiers_occurrence(
    api: httpx.AsyncClient, answer_app: FastAPI, llm: FakeLLM
) -> None:
    repeated = "Pricing follows the Order Form agreed by the parties."
    paragraphs = [
        "1. Fees are payable monthly in advance by the Customer.",
        repeated,
        "2. Payment terms are thirty days from the date of each invoice.",
        repeated,
    ]
    document = await upload(api, answer_app, "pricing.docx", factory.docx_file(paragraphs))
    llm.script = quoting("Payment terms", repeated)
    citation = (await ask(api, "What are the payment terms for each invoice?", document))[
        "citations"
    ][0]
    assert citation["status"] == "verified" and citation["occurrence_count"] == 2
    (r,) = citation["block_ranges"]
    # The block is the one holding the verifier's chosen range, never another occurrence.
    assert r["block_char_start"] <= citation["char_start"] < r["block_char_end"]
    assert citation["char_start"] - r["block_char_start"] == r["start"]
    await assert_block_ranges_match(document, citation)


async def test_unverified_docx_quote_gets_no_block_ranges(
    api: httpx.AsyncClient, numbered: str, llm: FakeLLM
) -> None:
    llm.script = quoting(
        "terminate this Agreement", "Either party may terminate this Agreement at any time"
    )
    citation = (await ask(api, "How can the agreement be terminated?", numbered))["citations"][0]
    assert citation["status"] == "unverified" and citation["block_ranges"] == []


async def test_pdf_citations_use_highlights_not_block_ranges(
    api: httpx.AsyncClient, contract: str, llm: FakeLLM
) -> None:
    llm.script = quoting("terminate this Agreement", "Either party may terminate this Agreement")
    citation = (await ask(api, "How can the agreement be terminated?", contract))["citations"][0]
    assert citation["status"] == "verified" and citation["highlights"]
    assert citation["block_ranges"] == [] and citation["reason"] is None


async def test_verified_pdf_quote_on_no_stored_page_is_explained(
    api: httpx.AsyncClient, contract: str, llm: FakeLLM
) -> None:
    # Page records that no longer cover the quote: it stays verified, is not presented as a
    # Word citation, and says why it is not highlighted.
    async with get_sessionmaker()() as session:
        await session.execute(
            delete(DocumentPage).where(
                DocumentPage.document_id == uuid.UUID(contract), DocumentPage.page_number == 2
            )
        )
        await session.commit()
    llm.script = quoting("terminate this Agreement", "Either party may terminate this Agreement")
    citation = (await ask(api, "How can the agreement be terminated?", contract))["citations"][0]
    assert citation["status"] == "verified" and citation["page_start"] is None
    assert citation["highlights"] == [] and citation["block_ranges"] == []
    assert "could not be determined from the extracted layout" in citation["reason"]


async def test_docx_reprocessed_during_the_answer_gets_no_block_ranges(
    api: httpx.AsyncClient, numbered: str, llm: FakeLLM
) -> None:
    llm.script = quoting("terminate this Agreement", TERMINATION_CLAUSE)
    complete = llm.complete

    async def reprocessed_meanwhile(messages: list[ChatMessage]) -> LLMResult:
        async with get_sessionmaker()() as session:
            await session.execute(
                update(Document)
                .where(Document.id == uuid.UUID(numbered))
                .values(processed_at=text_interval("now() + interval '1 minute'"))
            )
            await session.commit()
        return await complete(messages)

    llm.complete = reprocessed_meanwhile  # type: ignore[method-assign]
    citation = (await ask(api, "How can the agreement be terminated?", numbered))["citations"][0]
    assert citation["status"] == "verified" and citation["block_ranges"] == []
    assert "could not be determined from the extracted text structure" in citation["reason"]


def fail_block_query(error: Exception):  # type: ignore[no-untyped-def]
    def hook(conn, cursor, statement, parameters, context, executemany):  # type: ignore[no-untyped-def]
        if "FROM blocks" in statement and "documents.processed_at" in statement:
            raise error

    return hook


async def test_docx_block_query_failure_keeps_the_verified_answer(
    api: httpx.AsyncClient, numbered: str, llm: FakeLLM
) -> None:
    llm.script = quoting("terminate this Agreement", TERMINATION_CLAUSE)
    hook = fail_block_query(OperationalError("SELECT ...", {}, OSError("connection lost")))
    engine = get_engine().sync_engine
    event.listen(engine, "before_cursor_execute", hook)
    try:
        response = await api.post(
            ANSWERS,
            json={"question": "How can the agreement be terminated?", "document_ids": [numbered]},
        )
    finally:
        event.remove(engine, "before_cursor_execute", hook)
    assert response.status_code == 200
    result = response.json()
    assert result["grounding"] == "grounded"
    citation = result["citations"][0]
    assert citation["status"] == "verified" and citation["quote"] == TERMINATION_CLAUSE
    text = await canonical(numbered)
    assert text[citation["char_start"] : citation["char_end"]] == citation["quote"]
    assert citation["block_ranges"] == []
    assert citation["reason"] == (
        "Its location in the document could not be loaded just now, so it is not highlighted."
    )
    assert get_engine().pool.checkedout() == 0


async def test_unexpected_errors_in_the_block_step_are_not_hidden(
    api: httpx.AsyncClient, numbered: str, llm: FakeLLM
) -> None:
    llm.script = quoting("terminate this Agreement", TERMINATION_CLAUSE)
    hook = fail_block_query(RuntimeError("a bug, not a database failure"))
    engine = get_engine().sync_engine
    event.listen(engine, "before_cursor_execute", hook)
    try:
        response = await api.post(
            ANSWERS,
            json={"question": "How can the agreement be terminated?", "document_ids": [numbered]},
        )
    finally:
        event.remove(engine, "before_cursor_execute", hook)
    assert response.status_code == 500


async def test_only_blocks_overlapping_the_cited_range_are_queried(
    api: httpx.AsyncClient, answer_app: FastAPI, numbered: str, llm: FakeLLM
) -> None:
    other = await upload(
        api, answer_app, "other.docx", factory.docx_file(["Either party may terminate on notice."])
    )
    statements: list[list[str]] = []

    def capture(conn, cursor, statement, parameters, context, executemany):  # type: ignore[no-untyped-def]
        if "FROM blocks" in statement and "documents.processed_at" in statement:
            statements.append([str(p) for p in parameters])

    llm.script = quoting("2.1 Either party", TERMINATION_CLAUSE)
    engine = get_engine().sync_engine
    event.listen(engine, "before_cursor_execute", capture)
    try:
        result = await ask(api, "How can the agreement be terminated?", numbered, other)
    finally:
        event.remove(engine, "before_cursor_execute", capture)
    citation = result["citations"][0]
    assert citation["status"] == "verified" and citation["document_id"] == numbered
    # One query: the cited document and the cited range only (document id, range end, start).
    assert len(statements) == 1
    assert statements[0] == [numbered, str(citation["char_end"]), str(citation["char_start"])]
    assert len(citation["block_ranges"]) == 1


# --- document versions: which version of each document the answer was produced from -------------


async def document_version(api: httpx.AsyncClient, document_id: str) -> str | None:
    """The document's current version, exactly as the documents API serializes it."""
    response = await api.get(f"/api/documents/{document_id}")
    assert response.status_code == 200, response.text
    return response.json()["processed_at"]


async def set_version(document_id: str, value: str | None = None) -> None:
    """Change `processed_at` as a re-index would (a minute later), or to `value` (SQL literal)."""
    async with get_sessionmaker()() as session:
        await session.execute(
            update(Document)
            .where(Document.id == uuid.UUID(document_id))
            .values(processed_at=text_interval(value or "now() + interval '1 minute'"))
        )
        await session.commit()


async def test_single_document_answer_reports_its_version(
    api: httpx.AsyncClient, contract: str, llm: FakeLLM
) -> None:
    llm.script = quoting("terminate this Agreement", "Either party may terminate this Agreement")
    result = await ask(api, "How can the agreement be terminated?", contract)
    assert result["citations"][0]["status"] == "verified"
    version = await document_version(api, contract)
    assert result["documents"] == [{"id": contract, "file_type": "pdf", "processed_at": version}]
    async with get_sessionmaker()() as session:
        stored = await session.get(Document, uuid.UUID(contract))
        assert stored is not None and version is not None
        # Full precision and the same instant as stored (microseconds, timezone-aware).
        assert datetime.fromisoformat(version) == stored.processed_at


async def test_multi_document_answer_reports_every_selected_document_in_order(
    api: httpx.AsyncClient, answer_app: FastAPI, contract: str, numbered: str, llm: FakeLLM
) -> None:
    other = await upload(api, answer_app, "contract-b.pdf", contract_pdf(CONTRACT_B))
    llm.script = quoting("terminate this Agreement", "Either party may terminate this Agreement")
    # Selected twice: reported once. `other` supplies no cited quote but is still reported.
    result = await ask(
        api, "How can the agreement be terminated?", numbered, contract, other, numbered
    )
    assert [d["id"] for d in result["documents"]] == [numbered, contract, other]
    assert [d["file_type"] for d in result["documents"]] == ["docx", "pdf", "pdf"]
    for d in result["documents"]:
        assert d["processed_at"] is not None
        assert d["processed_at"] == await document_version(api, d["id"])
    # Every cited document is among the reported ones.
    reported = {d["id"] for d in result["documents"]}
    assert {c["document_id"] for c in result["citations"] if c["document_id"]} <= reported


async def test_versions_are_reported_when_nothing_is_relevant(
    api: httpx.AsyncClient, contract: str, llm: FakeLLM
) -> None:
    result = await ask(api, "chocolate cake recipe", contract)
    assert result["grounding"] == "not_found" and llm.calls == []
    assert result["documents"] == [
        {"id": contract, "file_type": "pdf", "processed_at": await document_version(api, contract)}
    ]


async def test_versions_serialize_exactly_as_the_documents_api(
    api: httpx.AsyncClient, contract: str, llm: FakeLLM
) -> None:
    llm.script = lambda messages: f"<answer>{INSUFFICIENT_ANSWER}</answer>"
    # Whole seconds (no fractional part), and an instant given in another time zone.
    for value in ["'2026-10-01 10:00:00+00'", "'2026-10-01 12:00:00.000001+02'"]:
        await set_version(contract, value)
        result = await ask(api, "What is the liability cap?", contract)
        assert result["documents"][0]["processed_at"] == await document_version(api, contract)


async def test_document_reprocessed_during_the_llm_call_reports_the_answers_version(
    api: httpx.AsyncClient, contract: str, llm: FakeLLM
) -> None:
    llm.script = quoting("terminate this Agreement", "Either party may terminate this Agreement")
    before = await document_version(api, contract)
    complete = llm.complete

    async def reprocessed_meanwhile(messages: list[ChatMessage]) -> LLMResult:
        await set_version(contract)
        return await complete(messages)

    llm.complete = reprocessed_meanwhile  # type: ignore[method-assign]
    result = await ask(api, "How can the agreement be terminated?", contract)
    after = await document_version(api, contract)
    assert before != after
    # The version the answer was produced from, never the newer one; the location is dropped.
    assert result["documents"][0]["processed_at"] == before
    assert result["citations"][0]["status"] == "verified"
    assert result["citations"][0]["highlights"] == []


async def test_version_is_read_in_the_retrieval_snapshot(
    api: httpx.AsyncClient, contract: str, llm: FakeLLM, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A re-index committing right after retrieval took its snapshot (before the versions and
    # the text used for verification are read) must not be reported as the answer's version.
    llm.script = quoting("terminate this Agreement", "Either party may terminate this Agreement")
    before = await document_version(api, contract)
    search = RetrievalService.search

    async def search_then_reprocess(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        response = await search(self, *args, **kwargs)
        await set_version(contract)
        return response

    monkeypatch.setattr(RetrievalService, "search", search_then_reprocess)
    result = await ask(api, "How can the agreement be terminated?", contract)
    assert result["documents"][0]["processed_at"] == before != await document_version(api, contract)
    citation = result["citations"][0]
    assert citation["status"] == "verified" and citation["highlights"] == []


async def test_missing_version_is_reported_as_null(
    api: httpx.AsyncClient, contract: str, llm: FakeLLM
) -> None:
    # Not produced by processing (a ready document always has a version), but the column allows
    # it: the answer still works and reports the version as unknown.
    await set_version(contract, "NULL")
    llm.script = quoting("terminate this Agreement", "Either party may terminate this Agreement")
    result = await ask(api, "How can the agreement be terminated?", contract)
    assert result["documents"] == [{"id": contract, "file_type": "pdf", "processed_at": None}]
    assert result["citations"][0]["status"] == "verified"


# --- P0: output-format tags, abstention, marker/citation consistency ----------------------------


async def test_nested_insufficient_reply_is_not_found_not_an_unsupported_draft(
    api: httpx.AsyncClient, contract: str, llm: FakeLLM
) -> None:
    # Observed: this reply was shown as "not supported" with the draft "<answer>I couldn't ...".
    llm.script = lambda messages: f"<answer>\n<answer>{INSUFFICIENT_ANSWER}</answer>\n</answer>"
    result = await ask(api, "What is the liability cap?", contract)
    assert result["grounding"] == "not_found"
    assert result["answer"] == INSUFFICIENT_ANSWER
    assert result["unsupported_draft"] is None and result["citations"] == []


async def test_unverifiable_reply_never_shows_protocol_tags(
    api: httpx.AsyncClient, contract: str, llm: FakeLLM
) -> None:
    llm.script = lambda messages: (
        "<answer><answer>The liability cap is AED 5,000.</answer></answer>"
    )
    result = await ask(api, "What is the liability cap?", contract)
    assert result["grounding"] == "ungrounded"
    assert result["unsupported_draft"] == "The liability cap is AED 5,000."
    assert all("<answer>" not in note for note in result["notes"])
    # A "not stated" reply is shown in its own words: still without the tags.
    llm.script = lambda messages: (
        "<answer><answer>The provided passages do not show a CGPA.</answer></answer>"
    )
    result = await ask(api, "What is the liability cap?", contract)
    assert result["grounding"] == "not_found"
    assert result["answer"] == "The provided passages do not show a CGPA."


async def test_nested_tags_around_a_cited_answer_are_removed(
    api: httpx.AsyncClient, contract: str, llm: FakeLLM
) -> None:
    def script(messages: list[ChatMessage]) -> str:
        e = find(evidence_in(messages), "terminate this Agreement")
        raw = answer_output(
            "Either party may terminate [1].",
            [{"n": 1, "evidence": e, "quote": "Either party may terminate this Agreement"}],
        )
        return raw.replace("<answer>", "<answer><answer>", 1)

    llm.script = script
    result = await ask(api, "How can the agreement be terminated?", contract)
    assert result["grounding"] == "grounded"
    assert result["answer"] == "Either party may terminate [1]."


async def test_mismatched_marker_numbers_are_flagged_and_not_matched_up(
    api: httpx.AsyncClient, contract: str, llm: FakeLLM
) -> None:
    # Observed: markers [1] and [3] in the text, citations numbered 1 and 2.
    def script(messages: list[ChatMessage]) -> str:
        e = find(evidence_in(messages), "terminate this Agreement")
        return answer_output(
            "Either party may terminate [1]. Notice must be written [3].",
            [
                {"n": 1, "evidence": e, "quote": "Either party may terminate this Agreement"},
                {"n": 2, "evidence": e, "quote": "written notice"},
            ],
        )

    llm.script = script
    result = await ask(api, "How can the agreement be terminated?", contract)
    assert result["grounding"] == "partially_grounded"
    by_n = {c["n"]: c for c in result["citations"]}
    assert by_n[1]["referenced"] is True and by_n[2]["referenced"] is False
    assert by_n[2]["status"] == "verified"  # verified, but supports no statement
    notes = " ".join(result["notes"])
    assert "[3] have no citation" in notes and "[2] are not referenced" in notes
    assert "do not match" in notes


async def test_extra_unreferenced_citation_is_noted_but_does_not_downgrade(
    api: httpx.AsyncClient, contract: str, llm: FakeLLM
) -> None:
    def script(messages: list[ChatMessage]) -> str:
        e = find(evidence_in(messages), "terminate this Agreement")
        return answer_output(
            "Either party may terminate [1].",
            [
                {"n": 1, "evidence": e, "quote": "Either party may terminate this Agreement"},
                {"n": 2, "evidence": e, "quote": "written notice"},
            ],
        )

    llm.script = script
    result = await ask(api, "How can the agreement be terminated?", contract)
    assert result["grounding"] == "grounded"
    assert [c["referenced"] for c in result["citations"]] == [True, False]
    assert any("[2] are not referenced" in note for note in result["notes"])


# --- P1 Step 1a: passages without substantive content ------------------------------------------


async def test_headings_only_passages_are_excluded_from_answer_evidence(
    api: httpx.AsyncClient, answer_app: FastAPI, llm: FakeLLM
) -> None:
    from tests.fixtures.journal import journal_docx

    journal = await upload(api, answer_app, "journal.docx", journal_docx())
    llm.script = lambda messages: f"<answer>{INSUFFICIENT_ANSWER}</answer>"
    result = await ask(api, "Tell me about yourself and your learning experiences.", journal)
    no_content = [e for e in result["excluded"] if e["reason"].startswith("no substantive content")]
    assert no_content, result["excluded"]
    shown = evidence_in(llm.calls[0]) if llm.calls else {}
    assert all(
        text.strip() not in {"About Yourself", "My Learning Experiences"} for text in shown.values()
    )


# --- P1 Step 1b: small-document mode ------------------------------------------------------------


async def meaningful(*document_ids: str) -> tuple[int, int]:
    """(context cost, number) of the passages of the documents other than bare headings,
    computed here in Python from the stored rows, independently of the code under test."""
    ids = [uuid.UUID(d) for d in document_ids]
    async with get_sessionmaker()() as session:
        chunks = (
            await session.execute(
                select(
                    DocumentChunk.document_id,
                    DocumentChunk.block_start,
                    DocumentChunk.block_end,
                    DocumentChunk.text,
                ).where(DocumentChunk.document_id.in_(ids))
            )
        ).all()
        blocks = (
            await session.execute(
                select(
                    DocumentBlock.document_id,
                    DocumentBlock.ordinal,
                    DocumentBlock.kind,
                    DocumentSection.kind,
                )
                .outerjoin(DocumentSection, DocumentSection.id == DocumentBlock.section_id)
                .where(DocumentBlock.document_id.in_(ids))
            )
        ).all()
        await session.rollback()
    stored = {(b[0], b[1]) for b in blocks}
    bare = {(b[0], b[1]) for b in blocks if b[2] == "heading" and b[3] == "heading"}
    kept = []
    for doc, first, last, text in chunks:
        present = [(doc, o) for o in range(first, last + 1) if (doc, o) in stored]
        if not (present and all(key in bare for key in present)):
            kept.append(len(text) // 4 + 30)
    return sum(kept), len(kept)


async def meaningful_tokens(*document_ids: str) -> int:
    return (await meaningful(*document_ids))[0]


async def evidence_for(budget: int, question: str, *document_ids: str) -> tuple[bool, int]:
    """(small-document mode used, passages selected) for the answer service with `budget`."""
    settings = Settings(_env_file=None, answer_context_token_budget=budget)  # type: ignore[call-arg]
    async with get_sessionmaker()() as session:
        service = AnswerService(session, ConceptEmbedder(), None, settings)
        _, selection = await service.select_evidence(question, [uuid.UUID(d) for d in document_ids])
        await session.rollback()
    return selection.whole_documents, len(selection.selected)


async def test_small_document_is_given_whole_in_document_order(
    api: httpx.AsyncClient, answer_app: FastAPI, llm: FakeLLM
) -> None:
    from tests.fixtures.journal import journal_docx

    journal = await upload(api, answer_app, "journal.docx", journal_docx())
    llm.script = lambda messages: f"<answer>{INSUFFICIENT_ANSWER}</answer>"
    result = await ask(api, "What are my three strongest qualities?", journal)
    async with get_sessionmaker()() as session:
        chunks = (
            await session.execute(
                select(DocumentChunk.ordinal, DocumentChunk.text)
                .where(DocumentChunk.document_id == uuid.UUID(journal))
                .order_by(DocumentChunk.ordinal)
            )
        ).all()
    bare = {"About Yourself", "My Learning Experiences"}
    expected = [text for _, text in chunks if text.strip() not in bare]
    assert [e["text"] for e in result["evidence"]] == expected  # all, in order, no headings
    assert any(e["retrieval_rank"] == 0 for e in result["evidence"])  # not all were retrieved
    assert any("full text" in n for n in result["notes"])
    assert sum(len(e["text"]) // 4 + 30 for e in result["evidence"]) <= 3000


async def test_small_document_mode_boundary(api: httpx.AsyncClient, answer_app: FastAPI) -> None:
    from tests.fixtures.journal import journal_docx

    journal = await upload(api, answer_app, "journal.docx", journal_docx())
    total, count = await meaningful(journal)
    question = "What are my three strongest qualities?"
    assert await evidence_for(total, question, journal) == (True, count)  # exactly at the budget
    whole, selected = await evidence_for(total - 1, question, journal)
    assert not whole and selected < count  # one token over: the usual selection, not truncated


async def previous_selection(question: str, *document_ids: str):  # type: ignore[no-untyped-def]
    """The answer service's evidence next to what the previous path (retrieval, then
    `select_context` on its passages) selects for the same search."""
    from app.answering.context import select_context

    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    async with get_sessionmaker()() as session:
        service = AnswerService(session, ConceptEmbedder(), None, settings)
        retrieval, selection = await service.select_evidence(
            question, [uuid.UUID(d) for d in document_ids]
        )
        await session.rollback()
    previous = select_context(retrieval.passages, settings, tuple(retrieval.analysis.terms))
    return selection, previous


def _ids(selection) -> tuple[list[int], list[tuple[int, str]]]:  # type: ignore[no-untyped-def]
    return (
        [p.chunk_id for p in selection.selected],
        [(p.chunk_id, reason) for p, reason in selection.excluded],
    )


async def test_over_budget_document_uses_exactly_the_previous_selection(
    api: httpx.AsyncClient, answer_app: FastAPI, large_contract: str
) -> None:
    from tests.fixtures.journal import journal_docx

    padded = await upload(api, answer_app, "journal.docx", journal_docx(padded=True))
    assert await meaningful_tokens(large_contract) > 3000
    assert await meaningful_tokens(padded) > 3000
    for question, document in [
        ("How can the agreement be terminated?", large_contract),
        ("Which law governs the agreement and the term of the order form?", large_contract),
        ("What are my three strongest qualities?", padded),
    ]:
        selection, previous = await previous_selection(question, document)
        assert not selection.whole_documents
        assert _ids(selection) == _ids(previous)  # same passages, order and exclusions
    # A controlled retrieval result: the termination clause is selected, as before Step 1b.
    selection, _ = await previous_selection("How can the agreement be terminated?", large_contract)
    assert any("terminate this Agreement" in p.text for p in selection.selected)
    assert 0 < len(selection.selected) <= 8


async def test_database_cost_estimate_equals_the_estimator(
    api: httpx.AsyncClient, answer_app: FastAPI, large_contract: str
) -> None:
    from app.answering.context import CHARS_PER_TOKEN, HEADER_TOKENS
    from app.retrieval.service import RetrievalService as Retrieval
    from tests.fixtures.journal import journal_docx

    journal = await upload(api, answer_app, "journal.docx", journal_docx())
    padded = await upload(api, answer_app, "padded.docx", journal_docx(padded=True))
    for ids in [(journal,), (padded,), (large_contract,), (journal, large_contract)]:
        async with get_sessionmaker()() as session:
            retriever = Retrieval(session, ConceptEmbedder(), Settings(_env_file=None))  # type: ignore[call-arg]
            cost = await retriever.meaningful_passage_cost(
                [uuid.UUID(d) for d in ids],
                chars_per_token=CHARS_PER_TOKEN,
                header_tokens=HEADER_TOKENS,
            )
            await session.rollback()
        assert cost == (await meaningful(*ids))[0]


async def test_over_budget_documents_are_rejected_without_reading_passage_text(
    api: httpx.AsyncClient, answer_app: FastAPI, large_contract: str
) -> None:
    statements: list[tuple[str, int]] = []

    def capture(conn, cursor, statement, parameters, context, executemany):  # type: ignore[no-untyped-def]
        statements.append((statement, len(parameters or ())))

    engine = get_engine().sync_engine
    event.listen(engine, "before_cursor_execute", capture)
    try:
        selection, _ = await previous_selection(
            "How can the agreement be terminated?", large_contract
        )
    finally:
        event.remove(engine, "before_cursor_execute", capture)
    assert not selection.whole_documents
    # The fit check is one aggregate over stored offsets; no statement lists passage ids beyond
    # the search's own top-k, and none reads the text of every passage.
    aggregate = [s for s, _ in statements if "sum(" in s and "char_end" in s]
    assert len(aggregate) == 1 and "chunks.text" not in aggregate[0]
    assert all(count <= 60 for _, count in statements)
    assert not any("length(chunks.text)" in s for s, _ in statements)


async def test_small_document_mode_needs_the_whole_selection_to_fit(
    api: httpx.AsyncClient, answer_app: FastAPI
) -> None:
    from tests.fixtures.journal import journal_docx

    journal = await upload(api, answer_app, "journal.docx", journal_docx())
    contract = await upload(api, answer_app, "contract-a.pdf", contract_pdf(CONTRACT_A))
    question = "What are my strengths and how can the agreement be terminated?"
    both = await meaningful_tokens(journal, contract)
    each = max(await meaningful_tokens(journal), await meaningful_tokens(contract))
    assert (await evidence_for(both, question, journal, contract))[0] is True
    # Each document fits alone, but not together: the usual selection for both.
    assert (await evidence_for(each, question, journal, contract))[0] is False


async def test_off_topic_question_still_never_reaches_the_model(
    api: httpx.AsyncClient, answer_app: FastAPI, llm: FakeLLM
) -> None:
    from tests.fixtures.journal import journal_docx

    journal = await upload(api, answer_app, "journal.docx", journal_docx())
    result = await ask(api, "chocolate cake recipe", journal)
    assert result["grounding"] == "not_found" and llm.calls == []


async def test_context_over_budget_is_never_sent(
    api: httpx.AsyncClient, contract: str, llm: FakeLLM, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The boundary check before the prompt is built holds whatever chose the evidence.
    import app.answering.service as service
    from app.answering.context import Selection

    def oversized(passages, retrieved, settings):  # type: ignore[no-untyped-def]
        big = [p.model_copy(update={"text": "x" * 20000}) for p in passages]
        return Selection(selected=big, excluded=[], tokens_used=0, whole_documents=True)

    monkeypatch.setattr(service, "whole_document_selection", oversized)
    response = await api.post(
        ANSWERS,
        json={"question": "How can the agreement be terminated?", "document_ids": [contract]},
    )
    assert response.status_code == 500 and llm.calls == []


# --- Pass A review: citation numbers, combined markers, quoted text, notes ----------------------


@pytest.fixture
async def journal(api: httpx.AsyncClient, answer_app: FastAPI) -> str:
    from tests.fixtures.journal import journal_docx

    return await upload(api, answer_app, "journal.docx", journal_docx())


def _cite(n: int, evidence: str, quote: str) -> dict[str, object]:
    return {"n": n, "evidence": evidence, "quote": quote}


async def test_a04_repeated_number_is_never_presented_as_verified(
    api: httpx.AsyncClient, journal: str, llm: FakeLLM
) -> None:
    from tests.fixtures.journal import LONG_TERM, SHORT_TERM

    def script(messages: list[ChatMessage]) -> str:
        e = find(evidence_in(messages), "Short term:")
        return answer_output(
            f'The short-term goals are "{SHORT_TERM[12:60]}" [4]. '
            f'The long-term goals are "{LONG_TERM[11:60]}" [4].',
            [_cite(4, e, SHORT_TERM), _cite(4, e, LONG_TERM)],
        )

    llm.script = script
    result = await ask(api, "What are my short-term and long-term goals?", journal)
    # Before: [4] kept the short-term quote only, the long-term sentence pointed at it, and the
    # answer showed as fully verified. Now: both quotations kept, neither matched to a sentence.
    assert result["grounding"] == "ungrounded"
    assert result["answer"] == INSUFFICIENT_ANSWER and result["unsupported_draft"]
    assert [c["marker_conflict"] for c in result["citations"]] == [True, True]
    assert all(c["status"] == "verified" for c in result["citations"])
    assert any("used for more than one quotation" in n for n in result["notes"])


async def test_a05_quote_marked_with_another_quotations_number_is_reported(
    api: httpx.AsyncClient, journal: str, llm: FakeLLM
) -> None:
    from tests.fixtures.journal import COURSE_GROWTH, IMPROVE_COMMUNICATION

    def script(messages: list[ChatMessage]) -> str:
        ev = evidence_in(messages)
        strengths, improve = find(ev, "Strengths:"), find(ev, "need to improve")
        growth = find(ev, "Through presentations")
        return answer_output(
            'The student states under strengths: "I collaborate well in group projects" [3], and '
            'under weaknesses: "I hesitate to speak in front of large audiences" [3]. '
            f'Later, the student notes: "{COURSE_GROWTH} It strengthened my curiosity and my '
            'teamwork" [12]. Finally: "I need to improve my communication and public speaking '
            'skills" [9]. These statements do not truly contradict each other and can all be true.',
            [
                _cite(3, strengths, "I collaborate well in group projects"),
                _cite(9, improve, IMPROVE_COMMUNICATION[:61]),
                _cite(12, growth, COURSE_GROWTH + " It strengthened my curiosity and my teamwork."),
            ],
        )

    llm.script = script
    result = await ask(api, "Do my statements about communication conflict?", journal)
    assert result["grounding"] == "partially_grounded"
    notes = " ".join(result["notes"])
    assert "quote text that is not in the quotation they cite" in notes
    assert "The student states under strengths" in notes
    # The quotation containing a full stop no longer produces a false "uncited" statement.
    assert "1 statement(s) in the answer have no citation" in notes
    assert "Later, the student notes" not in notes.split("have no citation")[1].split("|")[0]


async def test_a06_combined_marker_is_parsed_and_checked(
    api: httpx.AsyncClient, journal: str, llm: FakeLLM
) -> None:
    from tests.fixtures.journal import IMPROVE_COMMUNICATION, WEAKNESSES

    def script(messages: list[ChatMessage]) -> str:
        ev = evidence_in(messages)
        return answer_output(
            f'The weaknesses are "{WEAKNESSES[12:95]}" [3]. The student will "manage stress '
            'before exams" [9]. Hesitating to speak relates directly to communication [3, 9].',
            [
                _cite(3, find(ev, "Weaknesses:"), WEAKNESSES),
                _cite(9, find(ev, "need to improve"), IMPROVE_COMMUNICATION),
            ],
        )

    llm.script = script
    result = await ask(api, "Which weaknesses could affect communication?", journal)
    assert result["grounding"] == "grounded"  # [3, 9] cites the last sentence
    assert not any("have no citation" in n for n in result["notes"])
    assert all(c["referenced"] and not c["marker_conflict"] for c in result["citations"])


async def test_a15_reused_quote_for_a_paraphrase_is_a_known_limit(
    api: httpx.AsyncClient, journal: str, llm: FakeLLM
) -> None:
    from tests.fixtures.journal import SHORT_TERM

    def script(messages: list[ChatMessage]) -> str:
        e = find(evidence_in(messages), "Short term:")
        return answer_output(
            "The short-term goals are to learn machine learning and finish an internship [1]. "
            "The long-term goals are to build assistive technology [1].",
            [_cite(1, e, SHORT_TERM)],
        )

    llm.script = script
    result = await ask(api, "What are my short-term and long-term goals?", journal)
    # Not detectable without checking meaning: the label stays "grounded". Documented, so a
    # future claim-level check has a test to change.
    assert result["grounding"] == "grounded"


async def test_whole_document_answers_drop_the_search_only_omission_note(
    api: httpx.AsyncClient, contract: str, large_contract: str, llm: FakeLLM
) -> None:
    llm.script = lambda messages: f"<answer>{INSUFFICIENT_ANSWER}</answer>"
    whole = await ask(api, "What does the agreement say about pricing?", contract)
    assert any("full text" in n for n in whole["notes"])
    assert not any("near-duplicate" in n for n in whole["notes"])  # both pages were sent
    excerpt = await ask(api, "What does the agreement say about pricing?", large_contract)
    assert any("near-duplicate passage(s) were omitted" in n for n in excerpt["notes"])


# --- replies that follow the revised prompt, through the unchanged verification -----------------


async def test_c02_partial_answer_is_presented_with_the_missing_ranking_flagged(
    api: httpx.AsyncClient, journal: str, llm: FakeLLM
) -> None:
    def script(messages: list[ChatMessage]) -> str:
        e = find(evidence_in(messages), "Strengths:")
        quotes = [
            "I am curious about how software and machine learning systems work",
            "I pick up new programming tools quickly",
            "I stay composed when deadlines are tight",
            "I collaborate well in group projects",
        ]
        return answer_output(
            "The journal lists these strengths: curiosity about software and machine learning "
            "systems [1], picking up new programming tools quickly [2], staying composed under "
            "tight deadlines [3] and collaborating well in group projects [4]. The passages "
            "provided do not rank these qualities, so they do not state which three are strongest.",
            [_cite(n, e, q) for n, q in enumerate(quotes, start=1)],
        )

    llm.script = script
    result = await ask(api, "What are my three strongest qualities?", journal)
    assert result["answer"] != INSUFFICIENT_ANSWER  # answered, not refused
    assert result["grounding"] == "partially_grounded"  # the missing-ranking sentence is uncited
    assert [c["status"] for c in result["citations"]] == ["verified"] * 4
    assert not any(c["marker_conflict"] for c in result["citations"])
    assert any("do not rank these qualities" in n for n in result["notes"])


async def test_c03_labelled_inference_with_its_quotations_is_grounded(
    api: httpx.AsyncClient, journal: str, llm: FakeLLM
) -> None:
    def script(messages: list[ChatMessage]) -> str:
        ev = evidence_in(messages)
        return answer_output(
            'You describe yourself as "curious about how software and machine learning systems '
            'work" [1]. Your short-term goal is to "Learn the foundations of machine learning, '
            'statistics and Python" [2]. Inference: that curiosity is consistent with, and '
            "could help with, your short-term machine learning goal [1][2].",
            [
                _cite(1, find(ev, "Strengths:"), "I am curious about how software and machine "
                      "learning systems work"),
                _cite(2, find(ev, "Short term:"), "Learn the foundations of machine learning, "
                      "statistics and Python"),
            ],
        )  # fmt: skip

    llm.script = script
    result = await ask(api, "How could my strengths help my AI/ML goals?", journal)
    assert result["grounding"] == "grounded"
    assert "Inference:" in result["answer"]
    assert not any("quote text that is not in the quotation" in n for n in result["notes"])


async def test_c05_premise_correction_with_related_evidence_is_presented(
    api: httpx.AsyncClient, journal: str, llm: FakeLLM
) -> None:
    from tests.fixtures.journal import ACHIEVEMENTS

    listed = ACHIEVEMENTS.removeprefix("Achievements: ").rstrip(".")

    def script(messages: list[ChatMessage]) -> str:
        e = find(evidence_in(messages), "Achievements:")
        return answer_output(
            "The passages provided do not support the assumption of an AI competition or prize. "
            f'The achievements they list are "{listed}" [1].',
            [_cite(1, e, listed)],
        )

    llm.script = script
    result = await ask(api, "Which AI competition is among my semester achievements?", journal)
    assert result["answer"] != INSUFFICIENT_ANSWER
    # The correction only says what the passages do not support: it needs no marker (P1.3).
    assert result["grounding"] == "grounded"
    assert result["citations"][0]["status"] == "verified"


async def test_c06_premise_corrected_with_contradicting_quotations(
    api: httpx.AsyncClient, journal: str, llm: FakeLLM
) -> None:
    def script(messages: list[ChatMessage]) -> str:
        ev = evidence_in(messages)
        return answer_output(
            "The passages provided do not support the assumption that public speaking is your "
            'greatest strength. They describe it as an area to improve: "I hesitate to speak in '
            'front of large audiences" [1] and "I need to improve my communication and public '
            'speaking skills" [2].',
            [
                _cite(1, find(ev, "Weaknesses:"),
                      "I hesitate to speak in front of large audiences"),
                _cite(2, find(ev, "need to improve"), "I need to improve my communication and "
                      "public speaking skills"),
            ],
        )  # fmt: skip

    llm.script = script
    result = await ask(
        api, "Since public speaking is my greatest strength, how did I show it?", journal
    )
    assert result["answer"] != INSUFFICIENT_ANSWER
    assert result["grounding"] == "grounded"  # the correction sentence needs no marker (P1.3)
    assert all(c["status"] == "verified" for c in result["citations"])
    assert not any("quote text that is not in the quotation" in n for n in result["notes"])


async def test_reply_that_only_says_what_is_not_stated_is_not_found_in_its_own_words(
    api: httpx.AsyncClient, journal: str, llm: FakeLLM
) -> None:
    # Pass A, A08/A09/A10 (2026-10-20): an honest "not stated" reply claims nothing a quotation
    # could prove. It is "not found" (never verified), and a corrected premise stays visible.
    reply = (
        "The passages provided do not support that assumption.\n"
        "The passages provided do not mention any AI competition."
    )
    llm.script = lambda messages: f"<answer>{reply}</answer>\n<citations>\n</citations>"
    result = await ask(api, "Which AI competition is among my semester achievements?", journal)
    assert result["grounding"] == "not_found" and result["answer"] == reply
    assert result["citations"] == [] and result["unsupported_draft"] is None
    assert any("nothing in it is verified" in n for n in result["notes"])


async def test_a_factual_sentence_without_a_quotation_is_still_not_presented(
    api: httpx.AsyncClient, journal: str, llm: FakeLLM
) -> None:
    # The exception covers only plain "not stated" sentences: any other claim keeps the rule.
    reply = (
        "The passages provided do not mention any AI competition. "
        "The student won a hackathon instead."
    )
    llm.script = lambda messages: f"<answer>{reply}</answer>\n<citations>\n</citations>"
    result = await ask(api, "Which AI competition is among my semester achievements?", journal)
    assert result["grounding"] == "ungrounded" and result["answer"] == INSUFFICIENT_ANSWER
    assert result["unsupported_draft"] == reply


# --- P1.3 absence statements and P1.4 figures, end to end ---------------------------------------


async def test_a_number_the_quotation_does_not_contain_is_flagged(
    api: httpx.AsyncClient, contract: str, llm: FakeLLM
) -> None:
    def script(messages: list[ChatMessage]) -> str:
        e = find(evidence_in(messages), "terminate this Agreement")
        return answer_output(
            "Either party can end it with 60 days' written notice [1].", [_cite(1, e, TERMINATION)]
        )

    llm.script = script
    result = await ask(api, "How can the agreement be terminated?", contract)
    assert result["citations"][0]["status"] == "verified"  # the quotation itself is genuine
    assert result["grounding"] == "partially_grounded"
    assert any("contain a number that is not in the quotation" in n for n in result["notes"])


async def test_honest_gap_is_not_penalised_but_a_quotation_cited_for_it_is(
    api: httpx.AsyncClient, journal: str, llm: FakeLLM
) -> None:
    gap = "The passages provided do not state how they differ from the long-term goals."

    def honest(messages: list[ChatMessage]) -> str:
        e = find(evidence_in(messages), "Short term:")
        return answer_output(
            f"The short-term goals include a summer internship in data analytics [1]. {gap}",
            [_cite(1, e, "finish a summer internship in data analytics")],
        )

    llm.script = honest
    result = await ask(api, "How do my short-term and long-term goals differ?", journal)
    assert result["grounding"] == "grounded"

    def decorated(messages: list[ChatMessage]) -> str:
        e = find(evidence_in(messages), "Short term:")
        return answer_output(
            "The short-term goals include a summer internship in data analytics [1]. "
            f"{gap[:-1]} [2].",
            [
                _cite(1, e, "finish a summer internship in data analytics"),
                _cite(2, e, "Learn the foundations of machine learning"),
            ],
        )

    llm.script = decorated
    result = await ask(api, "How do my short-term and long-term goals differ?", journal)
    assert result["grounding"] == "partially_grounded"
    assert any("cannot show that something is missing" in n for n in result["notes"])
