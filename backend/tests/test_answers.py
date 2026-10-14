"""Grounded answering end to end: real PostgreSQL/pgvector retrieval, deterministic concept
embeddings, and a scripted LLM whose output we control (so verification is what is tested)."""

import re
import uuid
from collections.abc import AsyncIterator, Callable

import httpx
import pytest
from fastapi import FastAPI
from pydantic import SecretStr
from sqlalchemy import update

from app.answering.prompt import INSUFFICIENT_ANSWER
from app.core.config import Settings
from app.db.models import Document, DocumentText
from app.db.session import get_engine, get_sessionmaker
from app.ingestion import pdf_extract
from app.ingestion.model import SourceBlock, SourceLine, SourceWord
from app.ingestion.ocr import OcrResult
from app.llm import ChatMessage, LLMUnavailableError, OpenAICompatibleClient
from app.main import create_app
from tests.conftest import truncate_all
from tests.fakes import ConceptEmbedder, FakeLLM, answer_output, evidence_in
from tests.fixtures import factory
from tests.test_retrieval import CONTRACT_A, CONTRACT_B, contract_pdf

ANSWERS = "/api/answers"
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


# --- grounded answers with verified quotations ----------------------------------------------------


async def test_grounded_answer_with_verified_quote(
    api: httpx.AsyncClient, contract: str, llm: FakeLLM
) -> None:
    def script(messages: list[ChatMessage]) -> str:
        e = find(evidence_in(messages), "terminate this Agreement")
        return answer_output(
            "Either party can end it with 30 days' written notice [1].",
            [{"n": 1, "evidence": e, "quote": "Either party may terminate this Agreement"}],
        )

    llm.script = script
    result = await ask(api, "How can the agreement be terminated?", contract)
    assert result["grounding"] == "grounded"
    assert result["answer"] == "Either party can end it with 30 days' written notice [1]."
    citation = result["citations"][0]
    assert citation["status"] == "verified"
    text = await canonical(contract)
    assert text[citation["char_start"] : citation["char_end"]] == citation["quote"]
    assert citation["quote"] == "Either party may terminate this Agreement"
    assert citation["page_start"] == citation["page_end"] == 2  # termination clause is on page 2
    assert citation["document_filename"] == "contract-a.pdf"
    assert any("based on" in n and "excerpt" in n for n in result["notes"])


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
    api: httpx.AsyncClient, contract: str, llm: FakeLLM
) -> None:
    llm.script = lambda messages: f"<answer>{INSUFFICIENT_ANSWER}</answer>"
    result = await ask(
        api, "Which law governs the agreement and the term of the order form?", contract
    )
    shown = evidence_in(llm.calls[0])
    assert len(shown) == len(result["evidence"])
    excluded_ids = {e["chunk_id"] for e in result["excluded"]}
    assert not excluded_ids & {e["chunk_id"] for e in result["evidence"]}
    for item in result["excluded"]:
        assert item["reason"]


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
    assert any("<answer> format" in n for n in result["notes"])
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
