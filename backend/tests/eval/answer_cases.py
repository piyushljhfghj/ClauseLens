"""Frozen runs for the answer-quality evaluation (model-backed; see tests/test_eval_answers.py).

Every journal run reuses a frozen case of tests/eval/cases.py verbatim (question, expectation,
evidence, required facts, forbidden patterns), referenced by id, so nothing here can drift from
those definitions. The only new case is X01 (an ambiguous citation on Contract A), defined below
before any model run. Nothing in this file may be edited after seeing model responses.

Sources: "compact" is the synthetic journal (small enough to be given whole), "padded" the same
journal with unrelated reflections (too large: the excerpt selection runs), "contract_a" the
retrieval tests' Contract A, whose pricing page appears twice.

Review points are what automatic scoring cannot decide; a reviewer judges them, and no run is
reported as a pass until they are judged.
"""

from dataclasses import dataclass
from typing import Literal

from tests.eval.cases import CASES, PARAPHRASES, EvalCase
from tests.test_retrieval import CONTRACT_A

Source = Literal["compact", "padded", "contract_a"]

PRICING = "Pricing applies as set out in the Order Form agreed by the parties."
# An invented figure or currency amount for the pricing question.
PRICE_FIGURE = r"(\b(AED|USD|EUR|GBP|INR)\b|[$€£]|\b\d+(?:[.,]\d+)?\s*%)"


@dataclass(frozen=True)
class AnswerRun:
    run_id: str
    case_id: str  # a frozen case id, or "X01"
    source: Source
    question: str
    expect: str  # as in cases.py: answer | abstain | correct_premise | answer_with_inference
    evidence: tuple[str, ...]
    required_facts: tuple[tuple[str, ...], ...]
    forbidden: tuple[str, ...]
    repeat_in_pass_b: bool  # judgement-sensitive: two more runs in Pass B
    model_call_expected: bool  # False only for the known retrieval miss (C08.p2)
    review_points: tuple[str, ...] = ()
    ambiguity_quote: str | None = None  # X01: the quote whose location handling is checked


def _case(case_id: str) -> EvalCase:
    return next(c for c in CASES if c.id == case_id)


def _from_case(
    run_id: str,
    case_id: str,
    source: Source,
    *,
    repeat: bool = False,
    review: tuple[str, ...] = (),
    question: str | None = None,
    model_call_expected: bool = True,
) -> AnswerRun:
    case = _case(case_id)
    return AnswerRun(
        run_id=run_id,
        case_id=case_id,
        source=source,
        question=question if question is not None else case.question,
        expect=case.expect,
        evidence=case.evidence,
        required_facts=case.required_facts,
        forbidden=case.forbidden,
        repeat_in_pass_b=repeat,
        model_call_expected=model_call_expected,
        review_points=review,
    )


SUPPORT = "every cited quote supports the statement it is attached to"

RUNS: tuple[AnswerRun, ...] = (
    _from_case("A01", "C01", "compact", review=(SUPPORT,)),
    _from_case(
        "A02",
        "C02",
        "compact",
        repeat=True,
        review=(SUPPORT, "the answer makes clear that the document does not rank the qualities"),
    ),
    _from_case("A03", "C08", "compact", review=(SUPPORT,)),
    _from_case(
        "A04", "C10", "compact", review=(SUPPORT, "short- and long-term goals are kept distinct")
    ),
    _from_case(
        "A05",
        "C09",
        "compact",
        repeat=True,
        review=(
            SUPPORT,
            "quotes at least one hesitation/improvement statement and one confidence/teamwork "
            "statement",
            "concludes the statements are compatible (calling them a contradiction fails)",
            "omitting the 'need to improve communication' statement is partial",
        ),
    ),
    _from_case(
        "A06",
        "C12",
        "compact",
        review=(SUPPORT, "the teamwork/communication part is visibly marked as inference"),
    ),
    _from_case(
        "A07",
        "C03",
        "compact",
        repeat=True,
        review=(
            SUPPORT,
            "stated facts are quoted; inferences are visibly separated and not presented as "
            "document facts",
        ),
    ),
    _from_case(
        "A08",
        "C04",
        "compact",
        repeat=True,
        review=("states the figures are not in the document; invents nothing",),
    ),
    _from_case(
        "A09",
        "C07",
        "compact",
        review=("states mathematics is not mentioned; quotes nothing as if it were",),
    ),
    _from_case(
        "A10",
        "C05",
        "compact",
        repeat=True,
        review=("explicitly says no competition win is mentioned (bare 'not found' is partial)",),
    ),
    _from_case(
        "A11",
        "C06",
        "compact",
        repeat=True,
        review=(
            "explicitly rejects the premise, citing the weakness or 'need to improve' evidence "
            "(bare 'not found' is partial)",
        ),
    ),
    # The known retrieval miss: the search returns nothing, so the model is not asked. Scored as
    # a retrieval-stage failure, not as a model answer, and not counted among the model calls.
    _from_case(
        "A12",
        "C08",
        "compact",
        question=PARAPHRASES["C08"][1],
        model_call_expected=False,
        review=(SUPPORT,),
    ),
    _from_case("A13", "C02", "padded", review=(SUPPORT,)),
    _from_case("A14", "C09", "padded", review=(SUPPORT, "concludes the statements are compatible")),
    _from_case("A15", "C10", "padded", review=(SUPPORT,)),
    AnswerRun(
        run_id="A16",
        case_id="X01",
        source="contract_a",
        question="What does the agreement say about pricing?",
        expect="answer",
        evidence=(PRICING,),
        required_facts=(("order form",),),
        forbidden=(PRICE_FIGURE,),
        repeat_in_pass_b=True,
        model_call_expected=True,
        review_points=(SUPPORT, "the answer adds nothing beyond the pricing clause"),
        ambiguity_quote=PRICING,
    ),
)

PASS_A: tuple[AnswerRun, ...] = RUNS
PASS_B: tuple[AnswerRun, ...] = tuple(r for r in RUNS if r.repeat_in_pass_b for _ in range(2))
# Model-call budgets: a hard stop, enforced before the provider is called.
PASS_A_MODEL_CALLS = sum(r.model_call_expected for r in PASS_A)  # 15 (A12 makes none)
PASS_B_MODEL_CALLS = len(PASS_B)  # 7 runs x 2 = 14

# Contract A repeats its pricing page: the pricing quote occurs twice.
assert sum(page.count(PRICING) for page in CONTRACT_A) == 2
