"""Answer-quality evaluation with the configured model (opt-in; never runs by default).

What a run does, per tests/eval/answer_cases.py: ingests the synthetic documents into the
`_test` database, and for each run records retrieval coverage (through the answer service's own
evidence selection, no model call), then asks the question through the API with the configured
model, then scores the response automatically. Reports (JSON, Markdown and a CSV review sheet)
are written outside the repository.

Safety gates, all checked before any document is written or any model is called:
1. Opt-in: CLAUSELENS_LLM_EVAL=1 and CLAUSELENS_LLM_EVAL_PASS=A or B. Anything else fails.
2. Database identity: the database actually connected to must be the `_test` database derived
   for the test suite (name ends in "_test", equals the expected name, differs from the
   development database). Checked with a direct connection and again through the app's own
   engine. If it cannot be verified, the run fails without doing any work.
3. Reports outside the repository: a report directory inside the repository fails the run.
4. Model-call budget: Pass A 15 calls, Pass B 14; a call beyond the budget is refused before
   it reaches the provider, and the run fails.
No secret is read into the report or printed: the model client is built from the configured
settings and only its replies are recorded.

Scoring is automatic only where the rubric is mechanical (required facts, forbidden content,
citation location, abstention, labels, the ambiguity rule). It never declares a pass: whether a
quote supports its statement, and each case's review points, stay "PENDING" until a reviewer
judges them. Retrieval failures (expected evidence never given to the model) are reported as
such, separately from model answers.
"""

import csv
import json
import os
import re
import shutil
import sys
import tempfile
import uuid
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI
from sqlalchemy import text
from sqlalchemy.engine import make_url

from app.answering.integrity import statements
from app.answering.parser import MARKER, marker_numbers
from app.answering.prompt import INSUFFICIENT_ANSWER
from app.answering.service import AnswerService
from app.core.config import Settings, get_settings
from app.db.models import DocumentText
from app.db.session import create_engine, get_sessionmaker
from app.embeddings import FastEmbedEmbedder
from app.llm import ChatMessage, LLMClient, LLMResult, OpenAICompatibleClient
from app.main import create_app
from tests.conftest import BACKEND_DIR, truncate_all
from tests.eval.answer_cases import (
    PASS_A,
    PASS_A_MODEL_CALLS,
    PASS_B,
    PASS_B_MODEL_CALLS,
    PRICING,
    RUNS,
    AnswerRun,
)
from tests.eval.cases import CASES, PARAPHRASES
from tests.fixtures.journal import journal_docx
from tests.test_retrieval import CONTRACT_A, contract_pdf, upload

OPT_IN = "CLAUSELENS_LLM_EVAL"
PASS_VAR = "CLAUSELENS_LLM_EVAL_PASS"
REPORT_VAR = "CLAUSELENS_EVAL_DIR"
REPO_ROOT = BACKEND_DIR.parent
PENDING = "PENDING"

# Frozen definitions: changing either file changes these checksums and fails the tests below.
CASES_SHA256 = "33b611b7ef2ebe6e8f16f5d95aa8eba503bbf0baaf00ced93da96f03a0b330fb"
ANSWER_CASES_SHA256 = "047c5c423057578add242bc3162e6f0b4ea00cdcbdcc3cd8355ce5e50045c2e6"


# --- safety gates ---------------------------------------------------------------------------------


class UnsafeEvaluation(Exception):
    """A safety gate failed: the evaluation must not do any work."""


def eval_request(environ: Mapping[str, str]) -> str | None:
    """The pass to run ("A" or "B"), None when not opted in. An opt-in without a valid pass is an
    error rather than a default, so a run is never started by accident."""
    if environ.get(OPT_IN) != "1":
        return None
    requested = environ.get(PASS_VAR, "")
    if requested not in {"A", "B"}:
        raise UnsafeEvaluation(f"{PASS_VAR} must be A or B when {OPT_IN}=1.")
    return requested


def check_database_identity(
    actual: str | None, expected: str | None, development: str | None
) -> None:
    """Fail closed unless `actual` is verifiably the suite's `_test` database."""
    if not actual or not expected:
        raise UnsafeEvaluation("The database identity could not be verified.")
    if actual != expected:
        raise UnsafeEvaluation("Connected to an unexpected database.")
    if not actual.endswith("_test"):
        raise UnsafeEvaluation("The evaluation database is not a _test database.")
    if development is not None and actual == development:
        raise UnsafeEvaluation("The evaluation database is the development database.")


def report_dir(environ: Mapping[str, str], repo_root: Path = REPO_ROOT) -> Path:
    """The report directory, which must lie outside the repository."""
    base = environ.get(REPORT_VAR) or str(Path(tempfile.gettempdir()) / "clauselens-eval")
    path = Path(base).resolve()
    if path == repo_root.resolve() or path.is_relative_to(repo_root.resolve()):
        raise UnsafeEvaluation("Reports must be written outside the repository.")
    return path


class BudgetExceeded(Exception):
    pass


@dataclass
class BudgetedLLM:
    """The configured client behind a hard call budget; records each reply for the report."""

    inner: LLMClient
    budget: int
    replies: list[str] = field(default_factory=list)
    refused: int = 0

    @property
    def model_name(self) -> str:
        return self.inner.model_name

    async def complete(self, messages: list[ChatMessage]) -> LLMResult:
        if len(self.replies) >= self.budget:
            self.refused += 1
            raise BudgetExceeded("Model-call budget exhausted.")
        self.replies.append("")  # counted before the call: a failed call still used the budget
        result = await self.inner.complete(messages)
        self.replies[-1] = result.text
        return result


# --- retrieval coverage (no model call) ---------------------------------------------------------


def _squash(value: str) -> str:
    return " ".join(value.split())


@dataclass
class Coverage:
    evidence: int
    in_candidates: int
    selected: int
    whole_documents: bool
    selected_passages: int


async def measure_coverage(
    run: AnswerRun, document_id: uuid.UUID, settings: Settings, embedder: FastEmbedEmbedder
) -> Coverage:
    """What the answer service would give the model, through its own evidence selection."""
    async with get_sessionmaker()() as session:
        service = AnswerService(session, embedder, None, settings)
        retrieval, selection = await service.select_evidence(run.question, [document_id])
        await session.rollback()
    found = sum(
        any(_squash(s) in _squash(p.text) for p in retrieval.passages) for s in run.evidence
    )
    chosen = sum(
        any(_squash(s) in _squash(p.text) for p in selection.selected) for s in run.evidence
    )
    return Coverage(
        len(run.evidence), found, chosen, selection.whole_documents, len(selection.selected)
    )


# --- automatic scoring (never a pass on its own) ------------------------------------------------

_MARKER = re.compile(r"\[\d{1,3}\]")


def _plain(answer: str) -> str:
    return _MARKER.sub("", answer)


def evidence_spans(canonical: str, sentences: tuple[str, ...]) -> list[tuple[int, int]]:
    """Every occurrence of each evidence sentence in the canonical text (whitespace-insensitive),
    as [start, end) code-point offsets."""
    spans = []
    for sentence in sentences:
        pattern = r"\s+".join(re.escape(word) for word in sentence.split())
        spans += [(m.start(), m.end()) for m in re.finditer(pattern, canonical)]
    return spans


@dataclass
class CitationScore:
    n: int
    status: str
    referenced: bool
    quote: str
    on_evidence: bool | None  # automatic: overlaps an expected evidence sentence
    location_ambiguous: bool
    occurrence_count: int | None
    marker_conflict: bool = False  # its number is also used for a different quotation
    support: str = PENDING  # reviewer judgement: supports / partly / does not


@dataclass
class AutoScore:
    stage: str  # "retrieval failure" | "no model call" | "model answer" | "error"
    model_called: bool
    grounding: str | None
    abstained: bool
    required_facts: dict[str, bool]
    forbidden_in_answer: list[str]
    forbidden_in_draft: list[str]
    citations: list[CitationScore]
    checks: dict[str, str]  # automatic checks: "pass" | "fail" | "partial" | "review"
    findings: list[str]
    review_points: dict[str, str]  # each review point -> PENDING
    verdict: str  # "retrieval failure" | "fail" | "pending review" | "error"


def score_response(
    run: AnswerRun,
    response: Mapping[str, object] | None,
    coverage: Coverage,
    canonical: str,
    model_called: bool,
) -> AutoScore:
    checks: dict[str, str] = {}
    findings: list[str] = []
    review = {point: PENDING for point in run.review_points}
    if response is None:
        return AutoScore("error", model_called, None, False, {}, [], [], [], {}, ["no response"],
                         review, "error")  # fmt: skip

    grounding = str(response.get("grounding"))
    answer = str(response.get("answer") or "")
    draft = str(response.get("unsupported_draft") or "")
    abstained = grounding == "not_found" or answer.strip() == INSUFFICIENT_ANSWER
    plain = _plain(answer).lower()
    required = {
        " / ".join(group): any(alt.lower() in plain for alt in group)
        for group in run.required_facts
    }
    forbidden_answer = [p for p in run.forbidden if re.search(p, _plain(answer), re.I)]
    forbidden_draft = [p for p in run.forbidden if re.search(p, _plain(draft), re.I)]

    spans = evidence_spans(canonical, run.evidence)
    citations = []
    for c in response.get("citations") or []:  # type: ignore[attr-defined]
        start, end = c.get("char_start"), c.get("char_end")
        on = None
        if c.get("status") == "verified" and start is not None and end is not None:
            on = any(start < s_end and s_start < end for s_start, s_end in spans)
        citations.append(
            CitationScore(
                n=c.get("n"),
                status=c.get("status"),
                referenced=bool(c.get("referenced", True)),
                quote=c.get("quote") or c.get("proposed_quote") or "",
                on_evidence=on,
                location_ambiguous=bool(c.get("location_ambiguous")),
                occurrence_count=c.get("occurrence_count"),
                marker_conflict=bool(c.get("marker_conflict")),
            )
        )

    # Retrieval separately from the model: expected evidence never given to the model.
    if run.evidence and coverage.selected == 0 and not model_called:
        stage = "retrieval failure"
    elif not model_called:
        stage = "no model call"
    else:
        stage = "model answer"
    if run.evidence:
        checks["retrieval: all evidence selected"] = (
            "pass" if coverage.selected == coverage.evidence else "fail"
        )

    checks["no forbidden content in the answer"] = "fail" if forbidden_answer else "pass"
    if forbidden_draft:
        findings.append("forbidden content in the unverified draft (not shown as the answer)")

    if run.expect in {"answer", "answer_with_inference"}:
        if abstained:
            checks["answers (does not abstain)"] = (
                "fail" if coverage.selected else "n/a (evidence never reached the model)"
            )
        else:
            checks["answers (does not abstain)"] = "pass"
            missing = [k for k, ok in required.items() if not ok]
            checks["required facts present"] = "partial" if missing else "pass"
            if missing:
                findings.append(f"missing required facts: {missing}")
            supporting = [
                c for c in citations if c.status == "verified" and c.referenced and c.on_evidence
            ]
            checks["a verified, referenced citation on expected evidence"] = (
                "pass" if supporting else "fail"
            )
    elif run.expect == "abstain":
        if abstained:
            checks["abstains"] = "pass"
        else:
            checks["abstains"] = "review"
            findings.append("answered: check it only says the information is not there")
        if grounding == "ungrounded":
            findings.append("abstention labelled 'No verified quotations' (label error)")
    elif run.expect == "correct_premise":
        if abstained:
            checks["explicit premise correction"] = "partial"
            findings.append("bare 'not found': safe but does not correct the premise")
        else:
            checks["explicit premise correction"] = "review"

    if run.ambiguity_quote is not None:
        checks["ambiguous quote handled"] = ambiguity_check(run.ambiguity_quote, response)

    unreferenced = [c.n for c in citations if c.status == "verified" and not c.referenced]
    if unreferenced:
        findings.append(f"verified citations no marker refers to: {unreferenced}")

    # Citation integrity: each a hard failure, whatever else the answer gets right.
    notes = [str(n) for n in response.get("notes") or []]  # type: ignore[attr-defined]
    conflict = any(c.marker_conflict for c in citations) or any(CONFLICT_NOTE in n for n in notes)
    checks["no citation-number conflicts"] = "fail" if conflict else "pass"
    misattributed = any(MISATTRIBUTION_NOTE in n for n in notes)
    checks["no misattributed quoted text"] = "fail" if misattributed else "pass"
    unparsed = unparsed_markers(answer)
    checks["no unparsed citation markers"] = "fail" if unparsed else "pass"
    if unparsed:
        findings.append(f"bracketed markers the parser does not read: {unparsed}")
    for n, count in shared_quotations(answer).items():
        # Not detectable automatically (a paraphrase may reuse one quote): a review hint.
        findings.append(f"quotation [{n}] is cited by {count} statements: check it supports each")

    if stage == "retrieval failure":
        verdict = "retrieval failure"
    elif "fail" in checks.values():
        verdict = "fail"
    else:
        verdict = "pending review"  # never "pass" without the reviewer
    return AutoScore(
        stage, model_called, grounding, abstained, required, forbidden_answer, forbidden_draft,
        citations, checks, findings, review, verdict,
    )  # fmt: skip


CONFLICT_NOTE = "used for more than one quotation"
MISATTRIBUTION_NOTE = "quote text that is not in the quotation they cite"
_BRACKETED = re.compile(r"\[[^\[\]\n]{1,20}\]")


def unparsed_markers(answer: str) -> list[str]:
    """Bracketed text that looks like a citation (contains a digit) but is not a marker the
    parser reads, e.g. "[E3]" or "[3-5]"."""
    return [
        m.group(0)
        for m in _BRACKETED.finditer(answer)
        if re.search(r"\d", m.group(0)) and not MARKER.fullmatch(m.group(0))
    ]


def shared_quotations(answer: str) -> dict[int, int]:
    """Citation numbers used by more than one statement, with the number of statements."""
    counts: dict[int, int] = {}
    for statement in statements(answer):
        for n in set(marker_numbers(statement)):
            counts[n] = counts.get(n, 0) + 1
    return {n: k for n, k in sorted(counts.items()) if k > 1}


def ambiguity_check(quote: str, response: Mapping[str, object]) -> str:
    """X01: the repeated quote must be verified, recorded as occurring twice, and either placed
    in the passage the model cited or flagged ambiguous (so the viewer never opens it)."""
    matching = [
        c
        for c in response.get("citations") or []  # type: ignore[attr-defined]
        if c.get("status") == "verified" and _squash(c.get("quote") or "") == _squash(quote)
    ]
    if not matching:
        return "fail"
    for c in matching:
        if c.get("occurrence_count") != 2:
            return "fail"
        if not (c.get("location_ambiguous") or c.get("evidence_id") == c.get("cited_evidence_id")):
            return "fail"
    return "pass"


# --- reports -----------------------------------------------------------------------------------


def write_reports(
    out: Path, pass_name: str, rows: list[dict[str, object]], usage: dict[str, int]
) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    base = out / f"answers-pass{pass_name}-{stamp}"
    base.with_suffix(".json").write_text(
        json.dumps({"pass": pass_name, "usage": usage, "runs": rows}, indent=2), "utf-8"
    )
    lines = [
        f"# Answer-quality evaluation, pass {pass_name}",
        "",
        "Automatic checks only. Citation support and review points are PENDING until judged;",
        "no run is a pass until then.",
        "",
        "| Run | Case | Source | Stage | Grounding | Evidence selected | Automatic verdict |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        cov, score = r["coverage"], r["score"]
        lines.append(
            f"| {r['run_id']} | {r['case_id']} | {r['source']} | {score['stage']} | "  # type: ignore[index]
            f"{score['grounding']} | {cov['selected']}/{cov['evidence']} | {score['verdict']} |"  # type: ignore[index]
        )
    lines += ["", f"Model usage: {usage}", ""]
    base.with_suffix(".md").write_text("\n".join(lines), "utf-8")
    with base.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["run_id", "attempt", "case_id", "item", "detail", "automatic", "judgement", "notes"]
        )
        for r in rows:
            score = r["score"]
            for c in score["citations"]:  # type: ignore[index]
                writer.writerow(
                    [
                        r["run_id"],
                        r["attempt"],
                        r["case_id"],
                        f"citation [{c['n']}]",
                        c["quote"],
                        f"on evidence: {c['on_evidence']}",
                        PENDING,
                        "",
                    ]
                )
            for point in score["review_points"]:  # type: ignore[index]
                writer.writerow([r["run_id"], r["attempt"], r["case_id"], "review point",
                                 point, "", PENDING, ""])  # fmt: skip
    return base.with_suffix(".md")


# --- the model-backed evaluation (opt-in) --------------------------------------------------------


async def _current_database(settings: Settings) -> str | None:
    engine = create_engine(settings)
    try:
        async with engine.connect() as connection:
            return (await connection.execute(text("SELECT current_database()"))).scalar_one()
    except Exception:
        return None
    finally:
        await engine.dispose()


class EvaluationStopped(Exception):
    """The run was stopped part-way (budget exhausted); no report is written."""


STORAGE_PREFIX = "clauselens-eval-storage-"


async def _database_name_via_app() -> str | None:
    """The database the app's engine is connected to, or None if it cannot be read."""
    try:
        async with get_sessionmaker()() as session:
            name = (await session.execute(text("SELECT current_database()"))).scalar_one()
            await session.rollback()
            return str(name)
    except Exception:
        return None


async def _clean_test_database(
    expected: str | None, development: str | None
) -> UnsafeEvaluation | None:
    """Remove the evaluation's synthetic data from `_test`, after verifying again that the app
    is connected to it. A problem is returned, not raised, so that the app still shuts down;
    if the database cannot be verified, nothing is deleted."""
    try:
        check_database_identity(await _database_name_via_app(), expected, development)
    except UnsafeEvaluation as unsafe:
        return UnsafeEvaluation(f"Cleanup refused, nothing deleted: {unsafe}")
    try:
        await truncate_all()
    except Exception as exc:
        return UnsafeEvaluation(f"Cleanup of the _test database failed: {type(exc).__name__}")
    return None


def _storage_parent(parent: Path | None) -> Path:
    resolved = Path(parent or tempfile.gettempdir()).resolve()
    if resolved == REPO_ROOT.resolve() or resolved.is_relative_to(REPO_ROOT.resolve()):
        raise UnsafeEvaluation("Evaluation storage must be outside the repository.")
    return resolved


def remove_run_storage(storage: Path, parent: Path) -> None:
    """Remove this run's storage folder and nothing else: it must be the folder `mkdtemp`
    created for the run, directly inside `parent`, outside the repository."""
    resolved = storage.resolve()
    if (
        resolved.parent != parent.resolve()
        or not resolved.name.startswith(STORAGE_PREFIX)
        or resolved.is_relative_to(REPO_ROOT.resolve())
    ):
        raise UnsafeEvaluation("Refusing to remove a folder this run did not create.")
    if resolved.exists():
        shutil.rmtree(resolved)


async def run_evaluation(
    *,
    pass_name: str,
    runs: tuple[AnswerRun, ...],
    settings: Settings,
    llm: BudgetedLLM,
    embedder: FastEmbedEmbedder,
    out: Path,
    expected_database: str | None,
    development_database: str | None,
    storage_parent: Path | None = None,
) -> tuple[Path, list[dict[str, object]], dict[str, int]]:
    """Every gate, then the runs, then the reports. The model client is always the injected,
    budgeted `llm`: the app never builds its own.

    Cleanup, on success and on failure alike: the run's own storage folder (created here with
    `mkdtemp`) is removed; the `_test` database is emptied after its identity is verified again
    (if it cannot be verified, nothing is deleted and the run fails); and the app shuts down
    normally (worker stopped, storage closed, database engine disposed). The original error is
    raised after cleanup; a report is written only for a complete run."""
    if not isinstance(llm, BudgetedLLM):
        raise UnsafeEvaluation("The model client must be injected behind the call budget.")
    out = report_dir({REPORT_VAR: str(out)})
    parent = _storage_parent(storage_parent)
    check_database_identity(
        await _current_database(settings), expected_database, development_database
    )
    # Nothing is created before the gates above: from here on, everything is cleaned up.
    storage = Path(tempfile.mkdtemp(prefix=STORAGE_PREFIX, dir=parent))
    settings = settings.model_copy(
        update={"storage_backend": "local", "storage_local_dir": str(storage)}
    )
    rows: list[dict[str, object]] = []
    usage = {"prompt_tokens": 0, "completion_tokens": 0}
    failure: Exception | None = None
    cleanup_problem: UnsafeEvaluation | None = None
    try:
        app = create_app(settings, embedder=embedder, llm=llm)
        async with app.router.lifespan_context(app):
            try:
                await _execute(
                    app, runs, settings, llm, embedder, expected_database,
                    development_database, rows, usage,
                )  # fmt: skip
            except Exception as exc:  # cleaned up below, then raised again
                failure = exc
            finally:
                cleanup_problem = await _clean_test_database(
                    expected_database, development_database
                )
        # The lifespan exited normally even after a failure, so the app's shutdown ran.
    finally:
        remove_run_storage(storage, parent)
    if cleanup_problem is not None:
        raise cleanup_problem from failure
    if failure is not None:
        raise failure
    return write_reports(out, pass_name, rows, usage), rows, usage


async def _execute(
    app: FastAPI,
    runs: tuple[AnswerRun, ...],
    settings: Settings,
    llm: BudgetedLLM,
    embedder: FastEmbedEmbedder,
    expected_database: str | None,
    development_database: str | None,
    rows: list[dict[str, object]],
    usage: dict[str, int],
) -> None:
    if app.state.llm is not llm:
        raise UnsafeEvaluation("The app is not using the injected model client.")
    # The app's own engine, before any write.
    check_database_identity(await _database_name_via_app(), expected_database, development_database)
    await truncate_all()
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://test", timeout=120) as api:
        sources = {
            "compact": ("journal.docx", lambda: journal_docx()),
            "padded": ("journal-padded.docx", lambda: journal_docx(padded=True)),
            "contract_a": ("contract-a.pdf", lambda: contract_pdf(CONTRACT_A)),
        }
        documents = {
            source: await upload(api, app, sources[source][0], sources[source][1]())
            for source in dict.fromkeys(r.source for r in runs)
        }
        canonical = {}
        async with get_sessionmaker()() as session:
            for source, document_id in documents.items():
                row = await session.get(DocumentText, uuid.UUID(document_id))
                assert row is not None
                canonical[source] = row.canonical_text
            await session.rollback()

        for attempt, run in enumerate(runs, start=1):
            document_id = uuid.UUID(documents[run.source])
            coverage = await measure_coverage(run, document_id, settings, embedder)
            before = len(llm.replies)
            reply = await api.post(
                "/api/answers",
                json={"question": run.question, "document_ids": [str(document_id)]},
            )
            if llm.refused:
                raise EvaluationStopped("Model-call budget exhausted: evaluation stopped.")
            model_called = len(llm.replies) > before
            body = (
                reply.json()
                if reply.headers.get("content-type", "").startswith("application/json")
                else None
            )
            response = body if reply.status_code == 200 else None
            if response:
                for key in usage:
                    usage[key] += int((response.get("usage") or {}).get(key, 0))
            score = score_response(run, response, coverage, canonical[run.source], model_called)
            rows.append(
                {
                    "run_id": run.run_id,
                    "attempt": attempt,
                    "case_id": run.case_id,
                    "source": run.source,
                    "question": run.question,
                    "expect": run.expect,
                    "coverage": asdict(coverage),
                    "http_status": reply.status_code,
                    "error": None if response else (body or {}).get("error"),
                    "raw_model_reply": llm.replies[-1] if model_called else None,
                    "response": response,
                    "score": asdict(score),
                }
            )


@pytest.mark.skipif(os.environ.get(OPT_IN) != "1", reason="model-backed: opt in explicitly")
async def test_answer_quality_evaluation(test_database_url: str, tmp_path: Path) -> None:
    try:
        pass_name = eval_request(os.environ)
        assert pass_name is not None
        out = report_dir(os.environ)
        configured = get_settings()
        if not configured.llm_configured:
            raise UnsafeEvaluation("No model is configured.")
        settings = configured.model_copy(
            update={
                "app_env": "test",
                "log_level": "WARNING",
                "database_url": test_database_url,
                "worker_enabled": False,
            }
        )
        runs = PASS_A if pass_name == "A" else PASS_B
        budget = PASS_A_MODEL_CALLS if pass_name == "A" else PASS_B_MODEL_CALLS
        llm = BudgetedLLM(OpenAICompatibleClient(settings), budget)
        report, _, _ = await run_evaluation(
            pass_name=pass_name,
            runs=runs,
            settings=settings,
            llm=llm,
            embedder=FastEmbedEmbedder(settings.embedding_model, cache_dir=".cache/fastembed"),
            out=out,
            expected_database=make_url(test_database_url).database,
            development_database=make_url(configured.database_url).database,
        )
    except (UnsafeEvaluation, EvaluationStopped) as stopped:
        pytest.fail(f"Evaluation refused or stopped: {stopped}")
    print(f"\nreport: {report}\nmodel calls: {len(llm.replies)} of {budget}")


# --- tests that need no model --------------------------------------------------------------------


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_case_definitions_are_frozen() -> None:
    assert _sha256(BACKEND_DIR / "tests" / "eval" / "cases.py") == CASES_SHA256
    assert _sha256(BACKEND_DIR / "tests" / "eval" / "answer_cases.py") == ANSWER_CASES_SHA256


def test_journal_runs_reuse_the_frozen_cases_verbatim() -> None:
    by_id = {c.id: c for c in CASES}
    for run in RUNS:
        if run.case_id == "X01":
            continue
        case = by_id[run.case_id]
        assert (run.expect, run.evidence, run.required_facts, run.forbidden) == (
            case.expect, case.evidence, case.required_facts, case.forbidden,
        )  # fmt: skip
        assert run.question in {case.question, *PARAPHRASES.get(run.case_id, ())}
    a12 = next(r for r in RUNS if r.run_id == "A12")
    assert a12.question == "How do I behave when a deadline is close?"
    assert not a12.model_call_expected


def test_run_plan_and_budgets() -> None:
    assert len(RUNS) == 16 and len({r.run_id for r in RUNS}) == 16
    assert PASS_A_MODEL_CALLS == 15  # the known retrieval miss (A12) is not a model call
    repeated = {r.case_id for r in RUNS if r.repeat_in_pass_b}
    assert repeated == {"C02", "C03", "C04", "C05", "C06", "C09", "X01"}
    assert len(PASS_B) == PASS_B_MODEL_CALLS == 14
    assert {r.source for r in RUNS} == {"compact", "padded", "contract_a"}


def test_x01_quote_occurs_twice_in_contract_a() -> None:
    assert sum(page.count(PRICING) for page in CONTRACT_A) == 2


def test_opt_in_requires_an_explicit_pass() -> None:
    assert eval_request({}) is None
    assert eval_request({OPT_IN: "yes", PASS_VAR: "A"}) is None
    assert eval_request({OPT_IN: "1", PASS_VAR: "A"}) == "A"
    assert eval_request({OPT_IN: "1", PASS_VAR: "B"}) == "B"
    for bad in ({OPT_IN: "1"}, {OPT_IN: "1", PASS_VAR: "a"}, {OPT_IN: "1", PASS_VAR: "AB"}):
        with pytest.raises(UnsafeEvaluation):
            eval_request(bad)


@pytest.mark.parametrize(
    ("actual", "expected", "development"),
    [
        (None, "clauselens_test", "clauselens"),  # identity could not be read
        ("clauselens_test", None, "clauselens"),  # nothing to compare with
        ("clauselens", "clauselens_test", "clauselens"),  # the development database
        ("other_test", "clauselens_test", "clauselens"),  # a different database
        ("clauselens", "clauselens", "clauselens"),  # expected name is not a _test database
        ("clauselens_test", "clauselens_test", "clauselens_test"),  # same as development
    ],
)
def test_database_identity_fails_closed(
    actual: str | None, expected: str | None, development: str | None
) -> None:
    with pytest.raises(UnsafeEvaluation):
        check_database_identity(actual, expected, development)


def test_database_identity_accepts_only_the_expected_test_database() -> None:
    check_database_identity("clauselens_test", "clauselens_test", "clauselens")


def test_reports_must_be_written_outside_the_repository(tmp_path: Path) -> None:
    with pytest.raises(UnsafeEvaluation):
        report_dir({REPORT_VAR: str(REPO_ROOT / "backend" / "reports")})
    with pytest.raises(UnsafeEvaluation):
        report_dir({REPORT_VAR: str(REPO_ROOT)})
    assert report_dir({REPORT_VAR: str(tmp_path)}) == tmp_path.resolve()
    assert not report_dir({}).is_relative_to(REPO_ROOT.resolve())


async def test_budget_refuses_calls_before_they_reach_the_provider() -> None:
    class Provider:
        model_name = "fake"
        calls = 0

        async def complete(self, messages: list[ChatMessage]) -> LLMResult:
            Provider.calls += 1
            return LLMResult(text=f"reply {Provider.calls}", model="fake")

    llm = BudgetedLLM(Provider(), budget=2)
    await llm.complete([])
    await llm.complete([])
    with pytest.raises(BudgetExceeded):
        await llm.complete([])
    assert Provider.calls == 2 and llm.refused == 1 and llm.replies == ["reply 1", "reply 2"]


def test_the_model_backed_evaluation_is_skipped_without_opt_in() -> None:
    marks = [m for m in test_answer_quality_evaluation.pytestmark if m.name == "skipif"]  # type: ignore[attr-defined]
    assert marks and marks[0].args[0] == (os.environ.get(OPT_IN) != "1")


# --- automatic scoring --------------------------------------------------------------------------

_CANONICAL = "Intro. Strengths: I am curious and I stay composed. Other text here."
_STRENGTH = "Strengths: I am curious and I stay composed."


def _run(**overrides: object) -> AnswerRun:
    base = dict(
        run_id="T01", case_id="T", source="compact", question="q", expect="answer",
        evidence=(_STRENGTH,), required_facts=(("curious",), ("composed",)), forbidden=(),
        repeat_in_pass_b=False, model_call_expected=True, review_points=("support",),
    )  # fmt: skip
    base.update(overrides)
    return AnswerRun(**base)  # type: ignore[arg-type]


def _covered(selected: int = 1) -> Coverage:
    return Coverage(evidence=1, in_candidates=1, selected=selected, whole_documents=True,
                    selected_passages=3)  # fmt: skip


def _citation(start: int, end: int, **extra: object) -> dict[str, object]:
    return {"n": 1, "status": "verified", "referenced": True, "quote": _CANONICAL[start:end],
            "char_start": start, "char_end": end, "location_ambiguous": False,
            "occurrence_count": 1, **extra}  # fmt: skip


def test_scoring_never_passes_without_review() -> None:
    start = _CANONICAL.index("I am curious")
    response = {"grounding": "grounded", "answer": "Curious [1] and composed [1].",
                "citations": [_citation(start, start + 12)]}  # fmt: skip
    score = score_response(_run(), response, _covered(), _CANONICAL, model_called=True)
    assert score.verdict == "pending review"
    assert score.checks["a verified, referenced citation on expected evidence"] == "pass"
    assert all(c.support == PENDING for c in score.citations)
    assert score.review_points == {"support": PENDING}


def test_citation_off_the_expected_evidence_fails() -> None:
    response = {"grounding": "grounded", "answer": "Curious and composed [1].",
                "citations": [_citation(0, 6)]}  # "Intro."  # fmt: skip
    score = score_response(_run(), response, _covered(), _CANONICAL, model_called=True)
    assert score.citations[0].on_evidence is False and score.verdict == "fail"


def test_abstaining_with_the_evidence_selected_is_a_model_failure() -> None:
    response = {"grounding": "not_found", "answer": INSUFFICIENT_ANSWER, "citations": []}
    score = score_response(_run(), response, _covered(), _CANONICAL, model_called=True)
    assert score.stage == "model answer" and score.verdict == "fail"


def test_evidence_never_given_to_the_model_is_a_retrieval_failure() -> None:
    response = {"grounding": "not_found", "answer": INSUFFICIENT_ANSWER, "citations": []}
    score = score_response(_run(), response, _covered(selected=0), _CANONICAL, model_called=False)
    assert score.stage == "retrieval failure" and score.verdict == "retrieval failure"
    assert score.checks["retrieval: all evidence selected"] == "fail"


def test_forbidden_content_ignores_citation_markers() -> None:
    run = _run(expect="abstain", evidence=(), forbidden=(r"\b\d+(?:[.,]\d+)?\b",))
    clean = {"grounding": "not_found", "answer": "Not stated [1] [2].", "citations": []}
    assert score_response(run, clean, _covered(0), _CANONICAL, True).forbidden_in_answer == []
    invented = {"grounding": "grounded", "answer": "Your CGPA was 8.4 [1].", "citations": []}
    score = score_response(run, invented, _covered(0), _CANONICAL, True)
    assert score.forbidden_in_answer and score.verdict == "fail"


def test_bare_not_found_on_a_false_premise_is_partial() -> None:
    run = _run(expect="correct_premise", forbidden=(r"\bwon\b",))
    response = {"grounding": "not_found", "answer": INSUFFICIENT_ANSWER, "citations": []}
    score = score_response(run, response, _covered(), _CANONICAL, True)
    assert score.checks["explicit premise correction"] == "partial"
    assert score.verdict == "pending review"


def test_ambiguity_rule() -> None:
    def response(**citation: object) -> dict[str, object]:
        c = {
            "n": 1,
            "status": "verified",
            "quote": PRICING,
            "occurrence_count": 2,
            "location_ambiguous": False,
            "evidence_id": "E5",
            "cited_evidence_id": "E5",
        }
        c.update(citation)
        return {"citations": [c]}

    assert ambiguity_check(PRICING, response()) == "pass"  # placed where the model cited it
    assert ambiguity_check(PRICING, response(evidence_id="E6", location_ambiguous=True)) == "pass"
    assert ambiguity_check(PRICING, response(evidence_id="E6")) == "fail"  # silently moved
    assert ambiguity_check(PRICING, response(occurrence_count=1)) == "fail"
    assert ambiguity_check(PRICING, {"citations": []}) == "fail"


# --- Pass 0: the whole runner with a scripted model (no external call) ---------------------------
#
# The scripted model is injected explicitly. The settings are built without the .env file, so no
# provider is configured and the app could not build a real client; the real client's `complete`
# is also replaced by a hard failure for the duration of each test. A sentinel "key" is set in the
# settings so the reports can be checked for leaks.

SENTINEL_KEY = "pass0-sentinel-not-a-real-key-7f3c"
INSUFFICIENT_REPLY = f"<answer>{INSUFFICIENT_ANSWER}</answer>"


def _find(evidence: dict[str, str], needle: str) -> str:
    for evidence_id, passage in evidence.items():
        if needle in _squash(passage):
            return evidence_id
    raise AssertionError(f"{needle!r} not in the evidence")


def _scripts() -> dict[str, object]:
    """Raw replies by run id: deterministic, chosen to exercise each scoring path."""
    from tests.fakes import answer_output

    return {
        # A normal, cited answer.
        "A01": lambda ev: answer_output(
            "You are curious, pick up new tools quickly, stay composed when deadlines are tight "
            "and collaborate well in group projects [1].",
            [
                {
                    "n": 1,
                    "evidence": _find(ev, "Strengths:"),
                    "quote": "I stay composed when deadlines are tight",
                }
            ],
        ),
        # Missing markers: a citation, but no [n] in the text.
        "A03": lambda ev: answer_output(
            "You stay composed when deadlines are tight.",
            [
                {
                    "n": 1,
                    "evidence": _find(ev, "Strengths:"),
                    "quote": "I stay composed when deadlines are tight",
                }
            ],
        ),
        # Mismatched markers: [1] and [3] in the text, citations numbered 1 and 2.
        "A04": lambda ev: answer_output(
            "Short term: an internship in data analytics [1]. Long term: assistive technology [3].",
            [
                {
                    "n": 1,
                    "evidence": _find(ev, "Short term:"),
                    "quote": "finish a summer internship in data analytics",
                },
                {
                    "n": 2,
                    "evidence": _find(ev, "Long term:"),
                    "quote": "Build assistive technology for people with visual impairments",
                },
            ],
        ),
        # A malformed citation line next to a valid one.
        "A05": lambda ev: (
            "<answer>\nYou hesitate to speak to large audiences [1].\n</answer>\n<citations>\n"
            + json.dumps(
                {
                    "n": 1,
                    "evidence": _find(ev, "Weaknesses:"),
                    "quote": "I hesitate to speak in front of large audiences",
                }
            )
            + "\n{n: 2, evidence: E1}\n</citations>"
        ),
        # Abstention, plain and wrapped in a nested tag (the P0 parsing fix).
        "A08": lambda ev: INSUFFICIENT_REPLY,
        "A09": lambda ev: f"<answer>\n<answer>{INSUFFICIENT_ANSWER}</answer>\n</answer>",
        # An uncited reply to a false premise: never presented as an answer.
        "A10": lambda ev: "<answer>The provided passages do not mention any competition.</answer>",
        # The repeated pricing quote, cited against a passage that does not contain it: found in
        # the other evidence, twice, so it must be flagged ambiguous.
        "A16": lambda ev: answer_output(
            "Pricing follows the Order Form agreed by the parties [1].",
            [{"n": 1, "evidence": _find(ev, "terminate this Agreement"), "quote": PRICING}],
        ),
    }


@dataclass
class ScriptedLLM:
    """A deterministic fake model: replies are looked up by question, never generated."""

    replies_by_question: dict[str, object]
    model_name: str = "pass0-scripted"
    questions: list[str] = field(default_factory=list)

    async def complete(self, messages: list[ChatMessage]) -> LLMResult:
        from tests.fakes import evidence_in

        question = messages[-1].content.rsplit("Question: ", 1)[1].strip()
        self.questions.append(question)
        script = self.replies_by_question.get(question, lambda ev: INSUFFICIENT_REPLY)
        reply = script(evidence_in(messages))  # type: ignore[operator]
        usage = {
            "prompt_tokens": len(messages[-1].content) // 4,
            "completion_tokens": len(reply) // 4,
        }
        return LLMResult(text=reply, model=self.model_name, usage=usage)


def _pass0_settings(database_url: str) -> Settings:
    settings = Settings(  # type: ignore[call-arg]
        _env_file=None,  # never the .env file: no provider is configured
        app_env="test",
        log_level="WARNING",
        database_url=database_url,
        worker_enabled=False,
        llm_api_key=SENTINEL_KEY,
    )
    assert not settings.llm_configured  # no base URL or model: no real client can exist
    return settings


@pytest.fixture
def no_real_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    async def refuse(*args: object, **kwargs: object) -> LLMResult:
        raise AssertionError("Pass 0 must never reach a real provider.")

    monkeypatch.setattr(OpenAICompatibleClient, "complete", refuse)


def _scripted_llm(budget: int) -> tuple[BudgetedLLM, ScriptedLLM]:
    by_question = {
        r.question: script for r in RUNS for rid, script in _scripts().items() if r.run_id == rid
    }
    scripted = ScriptedLLM(by_question)
    return BudgetedLLM(scripted, budget), scripted


@pytest.fixture(scope="module")
def local_embedder() -> FastEmbedEmbedder:
    return FastEmbedEmbedder("BAAI/bge-small-en-v1.5", cache_dir=".cache/fastembed")


async def _test_database(test_database_url: str, statement: str) -> int | None:
    """Run `statement` on the `_test` database through a dedicated engine, after checking its
    name; returns the first value for a query."""
    engine = create_engine(Settings(_env_file=None, database_url=test_database_url))  # type: ignore[call-arg]
    try:
        async with engine.begin() as connection:
            name = (await connection.execute(text("SELECT current_database()"))).scalar_one()
            assert name == make_url(test_database_url).database and name.endswith("_test")
            result = await connection.execute(text(statement))
            return result.scalar_one() if result.returns_rows else None
    finally:
        await engine.dispose()


def _leftover_storage(parent: Path) -> list[Path]:
    """Run storage folders still present in `parent`."""
    return list(parent.glob(f"{STORAGE_PREFIX}*"))


async def _assert_cleaned_up(test_database_url: str, tmp_path: Path) -> None:
    """After a run, success or failure: no synthetic documents in `_test`, no run storage
    folder left, and the app's database engine disposed by its normal shutdown."""
    from app.db import session as db_session

    assert await _test_database(test_database_url, "SELECT count(*) FROM documents") == 0
    assert not _leftover_storage(tmp_path)
    assert db_session._engine is None


def _common(test_database_url: str, tmp_path: Path) -> dict[str, object]:
    expected = make_url(test_database_url).database
    assert expected and expected.endswith("_test")
    return {
        "pass_name": "0",
        "settings": _pass0_settings(test_database_url),
        "out": tmp_path / "reports",
        "expected_database": expected,
        "development_database": expected.removesuffix("_test"),
        "storage_parent": tmp_path,
    }


async def test_pass0_dry_run_exercises_the_whole_runner(
    test_database_url: str,
    tmp_path: Path,
    no_real_provider: None,
    local_embedder: FastEmbedEmbedder,
) -> None:
    llm, scripted = _scripted_llm(PASS_A_MODEL_CALLS)
    report, rows, usage = await run_evaluation(
        runs=PASS_A, llm=llm, embedder=local_embedder, **_common(test_database_url, tmp_path)
    )  # type: ignore[arg-type]
    await _assert_cleaned_up(test_database_url, tmp_path)  # success: everything cleaned up

    # Calls: one per run except the known retrieval miss, all to the scripted model.
    assert len(scripted.questions) == len(llm.replies) == PASS_A_MODEL_CALLS == 15
    assert llm.refused == 0
    by_run = {r["run_id"]: r for r in rows}
    assert len(by_run) == 16
    a12 = by_run["A12"]["score"]
    assert a12["stage"] == "retrieval failure" and a12["model_called"] is False
    assert by_run["A12"]["raw_model_reply"] is None
    assert PARAPHRASES["C08"][1] not in scripted.questions

    # Scoring paths, from the scripted replies.
    a01 = by_run["A01"]["score"]
    assert a01["grounding"] == "grounded" and a01["citations"][0]["on_evidence"] is True
    assert a01["checks"]["a verified, referenced citation on expected evidence"] == "pass"
    a03 = by_run["A03"]["response"]
    assert a03["grounding"] == "partially_grounded"
    assert any("does not mark which statements" in n for n in a03["notes"])
    a04 = by_run["A04"]
    assert any("do not match" in n for n in a04["response"]["notes"])
    assert [c["referenced"] for c in a04["response"]["citations"]] == [True, False]
    assert any("no marker refers to" in f for f in a04["score"]["findings"])
    assert any("malformed citation" in n for n in by_run["A05"]["response"]["notes"])
    for rid in ("A08", "A09"):
        assert by_run[rid]["score"]["grounding"] == "not_found"
        assert by_run[rid]["score"]["checks"]["abstains"] == "pass"
        assert "<answer" not in json.dumps(by_run[rid]["response"])
    # A reply that only says what is not stated is "not found", shown in its own words.
    assert by_run["A10"]["score"]["grounding"] == "not_found"
    assert by_run["A10"]["response"]["answer"].startswith("The provided passages")
    a16 = by_run["A16"]
    assert a16["coverage"]["whole_documents"] is True
    assert a16["score"]["checks"]["ambiguous quote handled"] == "pass"
    pricing = a16["response"]["citations"][0]
    assert pricing["location_ambiguous"] is True and pricing["occurrence_count"] == 2

    # No automatic pass; every human judgement pending.
    for row in rows:
        score = row["score"]
        assert score["verdict"] in {"retrieval failure", "fail", "pending review", "error"}
        assert all(c["support"] == PENDING for c in score["citations"])
        assert set(score["review_points"].values()) <= {PENDING}

    # Reports: schema, pending judgements, no secret, outside the repository.
    data = json.loads(report.with_suffix(".json").read_text("utf-8"))
    assert data["pass"] == "0" and data["usage"] == usage and len(data["runs"]) == 16
    expected_keys = {
        "run_id",
        "attempt",
        "case_id",
        "source",
        "question",
        "expect",
        "coverage",
        "http_status",
        "error",
        "raw_model_reply",
        "response",
        "score",
    }
    assert all(set(r) == expected_keys for r in data["runs"])
    score_keys = {
        "stage",
        "model_called",
        "grounding",
        "abstained",
        "required_facts",
        "forbidden_in_answer",
        "forbidden_in_draft",
        "citations",
        "checks",
        "findings",
        "review_points",
        "verdict",
    }
    assert all(set(r["score"]) == score_keys for r in data["runs"])
    with report.with_suffix(".csv").open(encoding="utf-8") as handle:
        sheet = list(csv.DictReader(handle))
    assert sheet and {row["judgement"] for row in sheet} == {PENDING}
    assert "PENDING until judged" in report.read_text("utf-8")
    out = tmp_path / "reports"
    artifacts = sorted(out.iterdir())
    assert {a.suffix for a in artifacts} == {".json", ".md", ".csv"}
    for artifact in artifacts:
        assert SENTINEL_KEY not in artifact.read_text("utf-8")
        assert not artifact.resolve().is_relative_to(REPO_ROOT.resolve())

    # The report files are this test's own output: remove exactly them.
    for artifact in artifacts:
        artifact.unlink()
    out.rmdir()
    print(
        f"\nPass 0: {len(scripted.questions)} scripted calls; removed {[a.name for a in artifacts]}"
    )


async def test_cleanup_after_the_budget_stops_the_run(
    test_database_url: str,
    tmp_path: Path,
    no_real_provider: None,
    local_embedder: FastEmbedEmbedder,
) -> None:
    llm, scripted = _scripted_llm(budget=2)
    first_three = tuple(r for r in PASS_A if r.model_call_expected)[:3]
    with pytest.raises(EvaluationStopped):
        await run_evaluation(
            runs=first_three,
            llm=llm,
            embedder=local_embedder,
            **_common(test_database_url, tmp_path),
        )  # type: ignore[arg-type]
    assert len(scripted.questions) == 2 and llm.refused == 1  # the third never reached it
    assert not (tmp_path / "reports").exists()  # no report for an incomplete run
    await _assert_cleaned_up(test_database_url, tmp_path)


async def test_cleanup_after_an_upload_fails(
    test_database_url: str,
    tmp_path: Path,
    no_real_provider: None,
    local_embedder: FastEmbedEmbedder,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    uploaded: list[str] = []
    real_upload = upload

    async def first_then_fail(api, app, name, data):  # type: ignore[no-untyped-def]
        if uploaded:
            raise RuntimeError("simulated upload failure")
        uploaded.append(await real_upload(api, app, name, data))
        return uploaded[-1]

    monkeypatch.setattr(sys.modules[__name__], "upload", first_then_fail)
    llm, scripted = _scripted_llm(budget=15)
    runs = (PASS_A[0], PASS_A[-1])  # two sources: the second upload fails
    with pytest.raises(RuntimeError, match="simulated upload failure"):
        await run_evaluation(
            runs=runs, llm=llm, embedder=local_embedder, **_common(test_database_url, tmp_path)
        )  # type: ignore[arg-type]
    assert len(uploaded) == 1 and scripted.questions == []
    await _assert_cleaned_up(test_database_url, tmp_path)  # the first document was removed too


async def test_cleanup_after_an_assertion_fails_mid_run(
    test_database_url: str,
    tmp_path: Path,
    no_real_provider: None,
    local_embedder: FastEmbedEmbedder,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def broken_scoring(*args: object, **kwargs: object) -> AutoScore:
        raise AssertionError("simulated assertion failure")

    monkeypatch.setattr(sys.modules[__name__], "score_response", broken_scoring)
    llm, scripted = _scripted_llm(budget=15)
    with pytest.raises(AssertionError, match="simulated assertion failure"):
        await run_evaluation(
            runs=(PASS_A[0],),
            llm=llm,
            embedder=local_embedder,
            **_common(test_database_url, tmp_path),
        )  # type: ignore[arg-type]
    assert len(scripted.questions) == 1
    await _assert_cleaned_up(test_database_url, tmp_path)


async def test_a_failed_model_request_is_recorded_and_cleaned_up(
    test_database_url: str,
    tmp_path: Path,
    no_real_provider: None,
    local_embedder: FastEmbedEmbedder,
) -> None:
    def provider_failure(evidence: dict[str, str]) -> str:
        raise RuntimeError("simulated provider failure")

    scripted = ScriptedLLM({PASS_A[0].question: provider_failure})
    llm = BudgetedLLM(scripted, budget=15)
    _, rows, _ = await run_evaluation(
        runs=(PASS_A[0],), llm=llm, embedder=local_embedder, **_common(test_database_url, tmp_path)
    )  # type: ignore[arg-type]
    assert len(scripted.questions) == 1 and len(llm.replies) == 1  # the failed call counted
    assert rows[0]["http_status"] == 500 and rows[0]["response"] is None
    assert rows[0]["score"]["verdict"] == "error"
    await _assert_cleaned_up(test_database_url, tmp_path)
    shutil.rmtree(tmp_path / "reports")  # this test's own report output


async def test_cleanup_refuses_to_delete_when_the_database_cannot_be_verified_again(
    test_database_url: str,
    tmp_path: Path,
    no_real_provider: None,
    local_embedder: FastEmbedEmbedder,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.db import session as db_session

    module = sys.modules[__name__]
    real = module._database_name_via_app
    calls: list[str | None] = []

    async def identity_lost_before_cleanup() -> str | None:
        calls.append(await real() if not calls else "postgres")  # second call: cleanup
        return calls[-1]

    monkeypatch.setattr(module, "_database_name_via_app", identity_lost_before_cleanup)
    llm, _ = _scripted_llm(budget=15)
    with pytest.raises(UnsafeEvaluation, match="Cleanup refused, nothing deleted"):
        await run_evaluation(
            runs=(PASS_A[0],),
            llm=llm,
            embedder=local_embedder,
            **_common(test_database_url, tmp_path),
        )  # type: ignore[arg-type]
    assert len(calls) == 2  # verified before the work and again before cleanup
    # Fail closed: the synthetic document was NOT deleted, but storage and the app were.
    assert await _test_database(test_database_url, "SELECT count(*) FROM documents") == 1
    assert not _leftover_storage(tmp_path) and db_session._engine is None
    await _test_database(test_database_url, "TRUNCATE documents, jobs CASCADE")  # tidy _test


@pytest.mark.parametrize("target", ["missing_database", "maintenance_database"])
async def test_pass0_database_guard_fails_closed_before_any_work(
    test_database_url: str,
    tmp_path: Path,
    no_real_provider: None,
    local_embedder: FastEmbedEmbedder,
    target: str,
) -> None:
    database = "clauselens_pass0_missing_test" if target == "missing_database" else "postgres"
    url = make_url(test_database_url).set(database=database).render_as_string(hide_password=False)
    common = _common(test_database_url, tmp_path)
    common["settings"] = _pass0_settings(url)
    llm, scripted = _scripted_llm(budget=15)
    with pytest.raises(UnsafeEvaluation):
        await run_evaluation(runs=PASS_A, llm=llm, embedder=local_embedder, **common)  # type: ignore[arg-type]
    assert scripted.questions == [] and not (tmp_path / "reports").exists()
    assert not _leftover_storage(tmp_path)  # nothing was created


async def test_pass0_refuses_an_uninjected_client(tmp_path: Path) -> None:
    with pytest.raises(UnsafeEvaluation):
        await run_evaluation(
            pass_name="0",
            runs=PASS_A,
            settings=_pass0_settings("postgresql+asyncpg://x/y_test"),
            llm=ScriptedLLM({}),  # type: ignore[arg-type]
            embedder=None,  # type: ignore[arg-type]
            out=tmp_path,
            expected_database="y_test",
            development_database="y",
            storage_parent=tmp_path,
        )
    assert not _leftover_storage(tmp_path)


async def test_pass0_refuses_a_report_directory_inside_the_repository(tmp_path: Path) -> None:
    llm, scripted = _scripted_llm(budget=15)
    with pytest.raises(UnsafeEvaluation):
        await run_evaluation(
            pass_name="0",
            runs=PASS_A,
            settings=_pass0_settings("postgresql+asyncpg://x/y_test"),
            llm=llm,
            embedder=None,  # type: ignore[arg-type]
            out=REPO_ROOT / "backend" / "pass0-reports",
            expected_database="y_test",
            development_database="y",
            storage_parent=tmp_path,
        )
    assert scripted.questions == [] and not (REPO_ROOT / "backend" / "pass0-reports").exists()


def test_run_storage_must_be_outside_the_repository() -> None:
    with pytest.raises(UnsafeEvaluation):
        _storage_parent(REPO_ROOT / "backend")


def test_only_the_runs_own_storage_folder_can_be_removed(tmp_path: Path) -> None:
    own = Path(tempfile.mkdtemp(prefix=STORAGE_PREFIX, dir=tmp_path))
    (own / "documents").mkdir()
    keep = tmp_path / "unrelated"
    keep.mkdir()
    for wrong in (
        keep,
        tmp_path,
        REPO_ROOT,
        REPO_ROOT / "s",
        tmp_path / "x" / f"{STORAGE_PREFIX}1",
    ):
        with pytest.raises(UnsafeEvaluation):
            remove_run_storage(wrong, tmp_path)
    remove_run_storage(own, tmp_path)
    assert not own.exists() and keep.exists() and (REPO_ROOT / "s").exists()


# --- stricter scoring from the Pass A review ------------------------------------------------------


def test_a04_conflicting_citation_numbers_fail_automatically() -> None:
    start = _CANONICAL.index("I am curious")
    response = {
        "grounding": "partially_grounded",
        "answer": "Curious and composed [1].",
        "citations": [_citation(start, start + 12, marker_conflict=True)],
        "notes": ["Citation number(s) [1] are used for more than one quotation, so ..."],
    }
    score = score_response(_run(), response, _covered(), _CANONICAL, True)
    assert score.checks["no citation-number conflicts"] == "fail" and score.verdict == "fail"
    assert score.citations[0].marker_conflict is True


def test_a05_misattributed_quoted_text_fails_automatically() -> None:
    start = _CANONICAL.index("I am curious")
    response = {
        "grounding": "partially_grounded",
        "answer": "Curious and composed [1].",
        "citations": [_citation(start, start + 12)],
        "notes": ["1 statement(s) quote text that is not in the quotation they cite, so ..."],
    }
    score = score_response(_run(), response, _covered(), _CANONICAL, True)
    assert score.checks["no misattributed quoted text"] == "fail" and score.verdict == "fail"


@pytest.mark.parametrize(
    ("answer", "unparsed"),
    [
        ("Curious and composed [3, 9].", []),  # combined markers are read
        ("Curious and composed [3,9][12].", []),
        ("Curious and composed [E3].", ["[E3]"]),
        ("Curious and composed [3-5].", ["[3-5]"]),
        ("Curious [1]; see [note] and [a, b].", []),  # no digits: not a citation lookalike
    ],
)
def test_unparsed_bracket_markers(answer: str, unparsed: list[str]) -> None:
    assert unparsed_markers(answer) == unparsed
    response = {"grounding": "grounded", "answer": answer, "citations": [], "notes": []}
    score = score_response(_run(), response, _covered(), _CANONICAL, True)
    assert score.checks["no unparsed citation markers"] == ("fail" if unparsed else "pass")


def test_a15_shared_quotation_is_a_review_hint_not_a_pass_or_fail() -> None:
    start = _CANONICAL.index("I am curious")
    answer = "You are curious [1]. You also stay composed [1]."
    response = {
        "grounding": "grounded",
        "answer": answer,
        "citations": [_citation(start, start + 12)],
        "notes": [],
    }
    score = score_response(_run(), response, _covered(), _CANONICAL, True)
    assert shared_quotations(answer) == {1: 2}
    assert any("cited by 2 statements" in f for f in score.findings)
    assert score.verdict == "pending review"
