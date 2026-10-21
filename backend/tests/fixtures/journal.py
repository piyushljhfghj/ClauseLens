"""A synthetic, fictional reflective journal for the answer-quality evaluation.

It reproduces the *structure* of a real student journal that exposed retrieval failures (bold
"Normal" paragraphs as headings, heading-only sections, two flat numbered question lists, a
question heading too long to be a title, one ending in a full stop instead of "?"), with
invented content only. Nothing here comes from a real person's document.

The EVIDENCE_* constants are the exact sentences the evaluation expects to find in the
evidence for each question; they are defined here, with the text, before any measurement.
"""

import io

import docx

# --- the evidence sentences (exact text as it appears in the document) ------------------------

STRENGTHS = (
    "Strengths: I am curious about how software and machine learning systems work, I pick up new "
    "programming tools quickly, I stay composed when deadlines are tight, and I collaborate well "
    "in group projects."
)
WEAKNESSES = (
    "Weaknesses: I tend to postpone tasks that I find dull, and I hesitate to speak in front of "
    "large audiences. I am working on planning my week in advance and on becoming more "
    "confident at public speaking."
)
SHORT_TERM = (
    "Short term: Learn the foundations of machine learning, statistics and Python, keep a good "
    "CGPA, and finish a summer internship in data analytics."
)
LONG_TERM = (
    "Long term: Build assistive technology for people with visual impairments and, in time, "
    "lead a small research team of my own."
)
IMPROVE_COMMUNICATION = (
    "I need to improve my communication and public speaking skills, manage stress before "
    "exams, and stop waiting for motivation before I start work."
)
COURSE_GROWTH = (
    "Through presentations and group discussions I became more organised with my time and more "
    "confident when sharing my opinions."
)
COURSE_TEAMWORK = "It strengthened my curiosity and my teamwork."
TEAM_ACTIVITY = (
    "Learning Outcome: I learned the value of listening, dividing roles and trusting teammates."
)
ACHIEVEMENTS = (
    "Achievements: Completed an online course in Python, volunteered as a mentor in the coding "
    "club, and helped organise the department quiz."
)

# --- the document --------------------------------------------------------------------------------

_LONG_QUESTION = (
    "1. Which qualities of your personality do you think still need to be developed further so "
    "that you can become a better, calmer and more balanced version of yourself over the coming "
    "years?"
)

# (text, bold): bold "Normal" paragraphs are how the real journal marks its headings.
_PARAGRAPHS: list[tuple[str, bool]] = [
    ("Personality and Human Values", True),
    ("REFLECTIVE JOURNAL", True),
    ("Personal Information", True),
    ("Name of the student: Test Student", False),
    ("Programme: B.Tech Computer Science, Semester V", False),
    ("About Yourself", True),
    ("1. What are your strengths and weaknesses?", False),
    (STRENGTHS, False),
    (WEAKNESSES, False),
    ("2. Your goals?", False),
    (SHORT_TERM, False),
    (LONG_TERM, False),
    ("3. Your role model (state specific reason for the same)?", False),
    (
        "My role model is my grandmother, a retired school teacher who taught evening classes for "
        "adults. I admire her patience and her belief that anyone can keep learning.",
        False,
    ),
    ("4. A book that has inspired you the most and why?", False),
    (
        "A biography of Marie Curie inspired me the most. It shows how persistence and careful "
        "work can overcome a lack of resources.",
        False,
    ),
    ("5. What do you expect from your course/programme?", False),
    (
        "I expect the programme to give me strong technical foundations, practical experience "
        "through projects and labs, and the chance to develop communication, teamwork and "
        "leadership skills.",
        False,
    ),
    ("Learnings from any five Classroom Activities", True),
    (
        "Activity 1: Learning Outcome: I learned that honesty and respect are the base of every "
        "lasting relationship, and that small promises matter.",
        False,
    ),
    ("Activity 2: Team Building Exercise (group task with limited materials)", False),
    (TEAM_ACTIVITY, False),
    (
        "Activity 3: Learning Outcome: I learned to plan a weekly timetable and to split large "
        "tasks into smaller steps.",
        False,
    ),
    (
        "Activity 4: Learning Outcome: I learned that leadership means taking initiative and "
        "responsibility in team work.",
        False,
    ),
    (
        "Activity 5: Learning Outcome: I learned to respect different opinions during debates and "
        "to stay calm when others disagree.",
        False,
    ),
    ("My Learning Experiences", True),
    (_LONG_QUESTION, False),
    (IMPROVE_COMMUNICATION, False),
    ("2. Are human values and success connected?", False),
    (
        "Yes. Values such as honesty, discipline and empathy build trust, and trust is what turns "
        "short-term results into long-term success.",
        False,
    ),
    ("3. How will you contribute to your national growth and development?", False),
    (
        "By building technology that is useful and accessible, paying taxes honestly and "
        "volunteering to teach basic computer skills.",
        False,
    ),
    ("4. How did this course help you grow? Explain in the context of your strengths.", False),
    (COURSE_GROWTH + " " + COURSE_TEAMWORK, False),
    ("5. List your semester achievements and learnings.", False),
    (ACHIEVEMENTS, False),
]

# Filler for the over-budget variant: on-topic for a journal, irrelevant to every question.
_PADDING_TOPICS = [
    "campus sustainability drive",
    "inter-college sports week",
    "library reading circle",
    "photography workshop",
    "blood donation camp",
    "cultural festival",
    "visit to a water treatment plant",
    "guest lecture on urban planning",
    "first-aid training",
    "tree plantation day",
    "alumni networking evening",
    "debate on renewable energy",
    "science exhibition for school children",
    "hostel cleanliness week",
    "road safety awareness rally",
    "career fair",
    "music club concert",
    "visit to a textile museum",
    "chess tournament",
    "orientation for new students",
    "talk on financial literacy",
    "yoga day session",
    "visit to an old age home",
    "poster design contest",
    "cycling day",
    "talk on digital privacy",
    "heritage walk",
    "book donation drive",
]


def _padding() -> list[tuple[str, bool]]:
    paragraphs: list[tuple[str, bool]] = [("Additional Reflections", True)]
    for number, topic in enumerate(_PADDING_TOPICS, start=1):
        paragraphs.append((f"{number}. Describe the {topic}.", False))
        paragraphs.append(
            (
                f"The {topic} was organised by student volunteers over two days. I helped with "
                "registrations, arranged the schedule with the coordinators and wrote a short "
                "report for the notice board. The event showed how much careful preparation, "
                "clear signage and friendly volunteers matter when many people take part, and "
                "the organisers collected feedback forms to improve the next edition.",
                False,
            )
        )
    return paragraphs


def journal_docx(*, padded: bool = False) -> bytes:
    """The synthetic journal; `padded` appends 28 unrelated reflections so the text is well
    beyond a 3,000-token context budget (about 5,000 estimated tokens), which keeps the
    retrieval-and-selection path in use."""
    document = docx.Document()
    for text, bold in _PARAGRAPHS + (_padding() if padded else []):
        document.add_paragraph().add_run(text).bold = bold or None
    buffer = io.BytesIO()
    document.save(buffer)
    return buffer.getvalue()
