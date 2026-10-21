"""Evaluation cases for answer quality over the synthetic journal (tests/fixtures/journal.py).

Expected behaviour, evidence and facts are defined from the document text alone, before any
measurement, and must not be edited to match what the system happens to do.

Scoring rules
-------------
Retrieval (deterministic; no model call):
- An evidence sentence is *in the candidates* if one retrieved passage contains it (whitespace-
  insensitive), and *selected* if one passage given to the model contains it.
- Case evidence recall = evidence sentences found / evidence sentences of the case, measured
  separately for candidates and for the selection. A case is *fully selected* when all of its
  evidence is selected. Aggregates are micro-averaged over (case, sentence) pairs.
- A missed sentence is attributed to the stage that lost it: "not retrieved" (no candidate
  contains it) or the selection's exclusion reason for the passage that does.
- A passage is *heading-only* when its text is nothing but its own section heading.
- Cases without evidence (absent facts) have no recall; only what was selected is recorded.

Answers (model evaluation, later; recorded per run):
- answer: grounding is grounded or partially_grounded, every required fact group is present
  (any alternative matches, case-insensitive), no forbidden pattern matches.
- abstain: the answer gives none of the absent facts (no forbidden pattern) and says they were
  not found; grounding not_found is the expected label.
- correct_premise: the false premise is not affirmed (no forbidden pattern); ideal is an explicit
  correction citing the contradicting evidence; a bare "not found" is safe but incomplete.
- answer_with_inference: as `answer`, and inferences are visibly separated from stated facts
  (judged by a reviewer until the answer format has an inference section).
- Citation location: a verified, referenced citation is on-evidence if its character range
  overlaps one of the case's evidence sentences. Whether a quote *supports* its statement is a
  reviewer's judgement, recorded separately.
"""

from dataclasses import dataclass
from typing import Literal

from tests.fixtures.journal import (
    ACHIEVEMENTS,
    COURSE_GROWTH,
    IMPROVE_COMMUNICATION,
    LONG_TERM,
    SHORT_TERM,
    STRENGTHS,
    WEAKNESSES,
)

Expectation = Literal["answer", "abstain", "correct_premise", "answer_with_inference"]

# Any number in the answer (citation markers removed first) would be an invented figure.
ANY_NUMBER = r"\b\d+(?:[.,]\d+)?\b"


@dataclass(frozen=True)
class EvalCase:
    id: str
    category: str
    question: str
    expect: Expectation
    evidence: tuple[str, ...]  # exact sentences from the document that the answer needs
    required_facts: tuple[tuple[str, ...], ...] = ()  # each group: any alternative must appear
    forbidden: tuple[str, ...] = ()  # regexes that must not match the answer
    note: str = ""
    observed: str = ""  # the failure seen on the real journal, if any


CASES: tuple[EvalCase, ...] = (
    EvalCase(
        id="C01",
        category="explicit fact",
        question="What are the strengths mentioned in the journal?",
        expect="answer",
        evidence=(STRENGTHS,),
        required_facts=(("curious",), ("tools",), ("composed", "deadline"), ("collaborat",)),
    ),
    EvalCase(
        id="C02",
        category="paraphrase (ranking not stated)",
        question=(
            "What are my three strongest qualities, and what exact evidence in the document "
            "supports each one?"
        ),
        expect="answer",
        evidence=(STRENGTHS,),
        required_facts=(("curious",), ("tools",), ("composed", "deadline"), ("collaborat",)),
        note="The document lists strengths but ranks none; a good answer says so.",
        observed="Real journal: 'Not found' - the strengths passage was excluded at selection.",
    ),
    EvalCase(
        id="C03",
        category="inference requested",
        question=(
            "Based on my stated strengths and short-term goals, explain how my current strengths "
            "could help me achieve my AI/ML career goals. Separate what the document explicitly "
            "states from your own inferences."
        ),
        expect="answer_with_inference",
        evidence=(STRENGTHS, SHORT_TERM),
        required_facts=(("curious",), ("machine learning",)),
        observed="Real journal: 'Not found' although both passages were given to the model.",
    ),
    EvalCase(
        id="C04",
        category="absent numeric facts",
        question=(
            "What was my exact CGPA in the previous semester, what was my highest ML exam score, "
            "and which company offered me the highest salary? Give exact numbers and cite each one."
        ),
        expect="abstain",
        evidence=(),
        forbidden=(ANY_NUMBER,),
        note="Only 'keep a good CGPA' appears: no figure, exam score, salary or employer.",
        observed="Real journal: correct abstention, shown as 'Not supported' with raw markup.",
    ),
    EvalCase(
        id="C05",
        category="false premise (absent achievement)",
        question=(
            "The document says I won an AI competition. Which competition was it, and what did "
            "I win?"
        ),
        expect="correct_premise",
        evidence=(ACHIEVEMENTS,),
        forbidden=(r"\bwon\b(?!'t)", r"\bwinner\b", r"\bprize\b"),
        note="No competition is mentioned; the achievements are a course, mentoring and a quiz.",
    ),
    EvalCase(
        id="C06",
        category="false premise (contradicted)",
        question=(
            "Since I said public speaking is my greatest strength, explain how I demonstrated it "
            "and cite the evidence."
        ),
        expect="correct_premise",
        evidence=(WEAKNESSES, IMPROVE_COMMUNICATION),
        note="Public speaking is described as something to improve, twice.",
        observed="Real journal: 'Not found' although contradicting passages were selected.",
    ),
    EvalCase(
        id="C07",
        category="absent weakness",
        question="Where do I say that I struggle with mathematics? Quote the sentence.",
        expect="abstain",
        evidence=(),
        forbidden=(r"struggle[sd]? with (mathematics|maths?)",),
        note="Mathematics is never mentioned.",
    ),
    EvalCase(
        id="C08",
        category="paraphrase (different wording)",
        question="How do I cope with tight deadlines and stressful situations?",
        expect="answer",
        evidence=(STRENGTHS,),
        required_facts=(("composed", "calm"),),
        note="'stay composed when deadlines are tight'; 'manage stress before exams' is a goal.",
    ),
    EvalCase(
        id="C09",
        category="apparent contradiction",
        question=(
            "Find any statements in the document that appear to conflict about my confidence, "
            "teamwork, or communication. Quote both sides, identify their sections, and explain "
            "whether they truly contradict each other or can both be true."
        ),
        expect="answer",
        evidence=(STRENGTHS, WEAKNESSES, IMPROVE_COMMUNICATION, COURSE_GROWTH),
        required_facts=(("public speaking", "speak"), ("confident",)),
        note=(
            "Compatible: hesitation and a wish to improve, then growing confidence after the "
            "course."
        ),
        observed=(
            "Real journal: right conclusion; the 'need to improve communication' passage was "
            "excluded."
        ),
    ),
    EvalCase(
        id="C10",
        category="distinguish related facts",
        question="What are my short-term goals and how are they different from my long-term goals?",
        expect="answer",
        evidence=(SHORT_TERM, LONG_TERM),
        required_facts=(("internship",), ("assistive", "visual")),
    ),
    EvalCase(
        id="C11",
        category="paraphrase (multi-part)",
        question=(
            "What does my document suggest about how I respond under pressure, adapt to "
            "unfamiliar technology, and collaborate with other people?"
        ),
        expect="answer",
        evidence=(STRENGTHS,),
        required_facts=(("composed", "calm"), ("tools",), ("collaborat", "group")),
    ),
    EvalCase(
        id="C12",
        category="multi-part with inference",
        question=(
            "What weaknesses have I acknowledged, what actions did I say I was taking to improve "
            "them, and which of those weaknesses could affect teamwork or communication?"
        ),
        expect="answer_with_inference",
        evidence=(WEAKNESSES, IMPROVE_COMMUNICATION),
        required_facts=(("postpone",), ("speak",), ("planning",)),
        note="'Which could affect teamwork or communication' is an inference.",
        observed="Real journal: the second weakness passage was selected but not used.",
    ),
)


# Blind paraphrase probe: four rewordings of each case that has expected evidence, written
# before any of them was measured and frozen since. Each shares its case's evidence and
# expectation. They must not be edited to make a measurement pass or fail.
PARAPHRASES: dict[str, tuple[str, ...]] = {
    "C01": (
        "Which personal strengths does the author describe?",
        "What does the writer say they are good at?",
        "List the positive qualities I wrote about myself.",
        "According to the journal, where do my abilities lie?",
    ),
    "C02": (
        "What would you say are my best traits, with supporting quotes?",
        "Name my top three qualities and back each one with evidence from the text.",
        "Which of my characteristics stand out most positively, and where is that shown?",
        "What am I best at, according to what I wrote? Cite the lines.",
    ),
    "C03": (
        "How might the strengths I listed help me reach my machine learning goals? Mark which "
        "parts are your inference.",
        "Connect my stated abilities to my short-term aims in AI, separating facts from your "
        "interpretation.",
        "Do my personal qualities fit my plans for a career in artificial intelligence? "
        "Distinguish what is written from what you infer.",
        "Using only what I wrote about my strengths and near-term goals, explain how one "
        "supports the other, labelling any inference.",
    ),
    "C05": (
        "Which AI hackathon did I win, according to the journal?",
        "Tell me about the artificial intelligence contest I won last semester.",
        "What prize did I receive in the machine learning competition?",
        "The journal mentions my first place in an AI challenge. What was it?",
    ),
    "C06": (
        "Public speaking is my best skill, right? Show where I demonstrated it.",
        "Given that I'm a confident public speaker, quote the evidence of that.",
        "Where does the journal show that addressing large audiences is my strongest ability?",
        "Explain how I proved my excellence at presentations, citing the document.",
    ),
    "C08": (
        "What do I say about working under time pressure?",
        "How do I behave when a deadline is close?",
        "Am I calm in stressful situations, according to the journal?",
        "How do I deal with pressure?",
    ),
    "C09": (
        "Are there any inconsistencies in what I say about my confidence and communication?",
        "Does the journal contradict itself about teamwork or speaking up? Quote both sides.",
        "Compare what I wrote about my communication skills in different sections and say "
        "whether they conflict.",
        "Do my comments on self-confidence in one part clash with another part of the journal?",
    ),
    "C10": (
        "What are my immediate goals compared with my long-range ambitions?",
        "How do my near-term plans differ from where I want to be eventually?",
        "List my goals for the next year and my goals for later in life.",
        "What do I want to achieve soon, and what do I hope to achieve in the long run?",
    ),
    "C11": (
        "How do I handle stress, new software and group work?",
        "What does the journal reveal about my composure, my learning of new technologies, and "
        "my teamwork?",
        "Describe my behaviour under pressure, with unfamiliar tools, and in teams.",
        "What kind of team member am I, and how do I cope with new technology and stress?",
    ),
    "C12": (
        "What are my weak points and what am I doing about them?",
        "Which shortcomings did I admit to, and how am I trying to fix them?",
        "List the areas I said I need to improve and the steps I am taking.",
        "What flaws do I acknowledge, what is my plan to address them, and could any hurt how I "
        "communicate with others?",
    ),
}
