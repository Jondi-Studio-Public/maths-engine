"""`senior-identify-the-dependent-and-independent-variable` (ACMEM140, Essential Mathematics).

Scope: given a described pair of variables from an experiment or study context,
identify which one is the dependent variable.

Answer space widened: the pair alone gives only ONE
distractor (the independent variable), which fails the >=2-distractor floor. The skill's
own wording ("from an experiment or study context") licenses naming one or two further
quantities from the same context that are neither the dependent nor independent variable:
a controlled (held-constant) quantity, and/or a second recorded-but-irrelevant quantity.

Every draw states, for one of fifteen (IV, DV, controlled, other, exam_trap) scenarios:
- the independent variable (IV) and dependent variable (DV) of the pair,
- one or two further named quantities that are neither.

`answer: mcq`, options are the plain-text quantity names themselves (never lettered).
The word "dependent" appears exactly once, in the question sentence; "independent"
never appears at all (the independent-variable option is never labelled as such --
it is offered only by its plain quantity name, so the prompt can never be answered by
spotting the word). No form of the word "depend" (the giveaway root) ever appears in the
stimulus -- relations are phrased with "affects" / "is affected by" instead.

1. Dropped "X depends on Y" stimulus wording (named the key); replaced with "is affected
   by" / "affects", neither of which contains the "depend" root.
2. Every per-case "for each/every ___" phrase now uses the scenario's own `unit` word
   (plant, student, participant, patient, person, musician, store, seedling, week)
   instead of a hard-coded "participant" that used to leak onto a plant scenario.
3. Every exam-tier held-constant clause now either (a) states a genuinely fixed
   condition ("kept the same", `trap_style="condition"`) or (b) states that subjects
   were SELECTED to already match on a baseline attribute ("matched across every ___",
   `trap_style="selection"`) -- never "a measured outcome was kept the same", which is
   incoherent.
4. Every "other" quantity is now a genuine per-case variable (varies case by case) --
   dropped the whole-study quantities ("the number of students in the class", "the
   number of patients in the trial") that used to be offered as "recorded for each case".
5. The exam-tier stimulus is two sentences; the second sentence's leading variable name
   is now capitalised (`_cap`) so every sentence starts with a capital letter.
6. Widened the scenario pool from ten to fifteen so a handful of seeds rarely repeat.
"""

from __future__ import annotations

from ..core import check, generator, latex_answer

# Fifteen (IV, DV, controlled, other, exam_trap) scenarios. `other` is a recorded-but-not-
# part-of-the-relationship quantity that genuinely varies case by case (used at medium/hard);
# `exam_trap` is a baseline/held-constant quantity in the SAME domain as the DV, offered only
# at the exam tier, that shares surface language with the DV. `trap_style` says how the
# exam-tier stimulus may honestly claim it is fixed:
#   "condition" -- a genuinely controllable condition, "kept the same for every <unit>".
#   "selection" -- subjects were selected/matched on a baseline attribute in advance
#                  ("<unit>s were chosen so that <trap> matched across every <unit>"),
#                  never phrased as forcibly "keeping a measurement the same".
# `unit` (and its plural `unit_plural`) is the per-case noun that fits the context, used
# everywhere the stimulus says "for each ___" / "for every ___".
PAIRS = (
    dict(
        iv="the amount of fertiliser applied to each plant",
        dv="the height of the plant after 4 weeks",
        controlled="the amount of water each plant received",
        other="the colour of the plant pot",
        exam_trap="the plants' starting height",
        trap_style="selection",
        unit="plant",
        unit_plural="plants",
    ),
    dict(
        iv="the number of hours a student studies each week",
        dv="the student's score on the end-of-term exam",
        controlled="the difficulty of the exam",
        other="the student's favourite subject",
        exam_trap="the room used for the exam",
        trap_style="condition",
        unit="student",
        unit_plural="students",
    ),
    dict(
        iv="the duration of daily exercise",
        dv="a participant's resting heart rate after 6 weeks",
        controlled="the room temperature during testing",
        other="the participant's age",
        exam_trap="the participants' resting heart rate before the study began",
        trap_style="selection",
        unit="participant",
        unit_plural="participants",
    ),
    dict(
        iv="the brightness of a classroom's screen display",
        dv="how many words per minute students read from the screen",
        controlled="the font size used on the screen",
        other="the noise level measured at each student's desk",
        exam_trap="the students' reading speed on paper before the trial",
        trap_style="selection",
        unit="student",
        unit_plural="students",
    ),
    dict(
        iv="the dosage of a new medication",
        dv="a patient's recovery time in days",
        controlled="the patients' age group",
        other="the patient's blood type",
        exam_trap="the patients' illness severity before treatment began",
        trap_style="selection",
        unit="patient",
        unit_plural="patients",
    ),
    dict(
        iv="how often a plant is watered each week",
        dv="the plant's growth in centimetres over a month",
        controlled="the type of soil used",
        other="the colour of the plant pot",
        exam_trap="the plants' starting height",
        trap_style="selection",
        unit="plant",
        unit_plural="plants",
    ),
    dict(
        iv="the amount of daily typing practice",
        dv="a participant's typing speed in words per minute",
        controlled="the keyboard layout used",
        other="the time of day the test was run",
        exam_trap="the participants' typing speed before practice began",
        trap_style="selection",
        unit="participant",
        unit_plural="participants",
    ),
    dict(
        iv="the number of hours a person sleeps",
        dv="a person's score on a memory test",
        controlled="the difficulty of the memory test",
        other="the room temperature when each person was tested",
        exam_trap="the people's memory scores from a test taken earlier",
        trap_style="selection",
        unit="person",
        unit_plural="people",
    ),
    dict(
        iv="the price charged for a product",
        dv="the number of units sold each week",
        controlled="the store's location",
        other="the day of the week",
        exam_trap="the store's opening hours",
        trap_style="condition",
        unit="week",
        unit_plural="weeks",
    ),
    dict(
        iv="the amount of daily practice on a musical instrument",
        dv="a musician's score in a skills assessment",
        controlled="the instrument used",
        other="the time of day the assessment was run",
        exam_trap="the musicians' scores in a previous assessment",
        trap_style="selection",
        unit="musician",
        unit_plural="musicians",
    ),
    dict(
        iv="the amount of sunlight a plant receives each day",
        dv="the plant's leaf count after 3 weeks",
        controlled="the type of soil used",
        other="the colour of the plant pot",
        exam_trap="the plants' starting height",
        trap_style="selection",
        unit="plant",
        unit_plural="plants",
    ),
    dict(
        iv="the temperature of the water given to each seedling",
        dv="the seedlings' growth in millimetres per week",
        controlled="the amount of light each seedling receives",
        other="the size of the pot used",
        exam_trap="the seedlings' starting height",
        trap_style="selection",
        unit="seedling",
        unit_plural="seedlings",
    ),
    dict(
        iv="the number of practice quizzes a student completes each week",
        dv="the student's final exam score",
        controlled="the format of the practice quizzes",
        other="the student's favourite subject",
        exam_trap="the students' scores on a similar quiz taken earlier",
        trap_style="selection",
        unit="student",
        unit_plural="students",
    ),
    dict(
        iv="the amount of daily screen time",
        dv="a person's score on a memory test",
        controlled="the time of day the test was given",
        other="the person's age",
        exam_trap="the people's memory scores from a test taken earlier",
        trap_style="selection",
        unit="person",
        unit_plural="people",
    ),
    dict(
        iv="the number of customers visiting a store each day",
        dv="the store's daily revenue",
        controlled="the store's opening hours",
        other="the day of the week",
        exam_trap="the checkout staffing level",
        trap_style="condition",
        unit="store",
        unit_plural="stores",
    ),
)

_EASY_VERBS = ("changing", "varying", "setting different levels of")
_MEDIUM_AIMS = ("find out whether", "investigate whether", "explore whether")
_HARD_RECORD_VERBS = ("recorded", "noted", "logged")
_EXAM_AIMS = ("hypothesise that", "predict that", "want to find out whether")

_QUESTION = "Which quantity is the dependent variable in this {noun}?"


def _cap(s: str) -> str:
    """Capitalise the first letter of a sentence-initial phrase without touching the rest."""
    return s[0].upper() + s[1:] if s else s


def _verify_no_answer_leak(text: str, dv: str, iv: str, extras: tuple[str, ...]) -> None:
    """In-generator check: 'dependent' appears exactly once (the question sentence,
    not the stimulus), 'independent' never appears at all, no form of the root 'depend'
    appears outside that one question occurrence, every named quantity is present
    verbatim and all named quantities are pairwise distinct."""
    check(text.count("dependent") == 1, "the word 'dependent' must appear exactly once (the question)")
    check("independent" not in text, "the word 'independent' must never appear in the rendered text")
    stimulus = text.rsplit(" Which quantity", 1)[0]
    check("depend" not in stimulus.lower(), "no form of 'depend' may appear in the stimulus")
    check(dv in text, "the dependent variable's own text must appear verbatim in the stimulus")
    check(iv in text, "the independent variable's own text must appear verbatim in the stimulus")
    for extra in extras:
        check(extra in text, f"named quantity {extra!r} must appear verbatim in the stimulus")
    # Plain-text quantity names, never mathematics: compare by string equality,
    # never `check_pairwise_distinct` (which sympifies -- a word containing an
    # apostrophe, e.g. "a participant's typing speed", is not valid sympy input).
    values = (dv, iv, *extras)
    check(len(set(values)) == len(values), "every named quantity must be textually distinct")


def _easy(rng, pair):
    verb = rng.choice(_EASY_VERBS)
    dep_first = rng.choice([True, False])
    if dep_first:
        lead = (
            f"A researcher measures {pair['dv']} while {verb} {pair['iv']}, "
            f"keeping {pair['controlled']} the same throughout the trial."
        )
    else:
        lead = (
            f"A researcher spends a trial {verb} {pair['iv']} and measures {pair['dv']}, "
            f"keeping {pair['controlled']} the same throughout."
        )
    text = f"{lead} {_QUESTION.format(noun='experiment')}"
    extras = (pair["controlled"],)
    return text, extras


def _medium(rng, pair):
    aim = rng.choice(_MEDIUM_AIMS)
    dep_first = rng.choice([True, False])
    if dep_first:
        lead = (
            f"Researchers want to {aim} {pair['dv']} is affected by {pair['iv']}. "
            f"In an observational study, they record {pair['dv']}, {pair['iv']} and {pair['other']} "
            f"for each case without changing anything."
        )
    else:
        lead = (
            f"Researchers want to {aim} {pair['iv']} affects {pair['dv']}. "
            f"In an observational study, they record {pair['iv']}, {pair['dv']} and {pair['other']} "
            f"for each case without changing anything."
        )
    text = f"{lead} {_QUESTION.format(noun='study')}"
    extras = (pair["other"],)
    return text, extras


def _hard(rng, pair):
    verb = rng.choice(_HARD_RECORD_VERBS)
    dep_first = rng.choice([True, False])
    order = (pair["dv"], pair["iv"]) if dep_first else (pair["iv"], pair["dv"])
    unit = pair["unit"]
    lead = (
        f"As part of a study, {order[0]}, {order[1]}, {pair['controlled']} and {pair['other']} "
        f"were all {verb} for each {unit}. The aim was to see whether "
        f"{pair['dv']} changes according to {pair['iv']}; {pair['controlled']} was kept the same "
        f"for every {unit}, and {pair['other']} was {verb} but is not part of this relationship."
    )
    text = f"{lead} {_QUESTION.format(noun='study')}"
    extras = (pair["controlled"], pair["other"])
    return text, extras


def _exam(rng, pair):
    aim = rng.choice(_EXAM_AIMS)
    dep_first = rng.choice([True, False])
    order = (pair["dv"], pair["iv"]) if dep_first else (pair["iv"], pair["dv"])
    if dep_first:
        aim_clause = f"Researchers {aim} {pair['dv']} is affected by {pair['iv']}."
    else:
        aim_clause = f"Researchers {aim} {pair['iv']} affects {pair['dv']}."
    unit = pair["unit"]
    trap = pair["exam_trap"]
    order_sentence = f"{_cap(order[0])} and {order[1]} were both recorded for each {unit}."
    if pair["trap_style"] == "selection":
        trap_clause = (
            f"{_cap(pair['unit_plural'])} were chosen so that {trap} matched "
            f"across every {unit} at the start of the study."
        )
    else:
        trap_clause = f"The study kept {trap} the same for every {unit} throughout."
    lead = f"{aim_clause} {order_sentence} {trap_clause}"
    text = f"{lead} {_QUESTION.format(noun='study')}"
    extras = (trap,)
    return text, extras


@generator(
    "p19b_dependent_variable",
    level=12,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
        "exam": dict(tier="exam"),
    },
)
def p19b_dependent_variable(rng, tier="easy"):
    if tier not in ("easy", "medium", "hard", "exam"):
        raise ValueError(f"tier must be easy/medium/hard/exam; got {tier!r}")

    pair = rng.choice(PAIRS)
    builder = {"easy": _easy, "medium": _medium, "hard": _hard, "exam": _exam}[tier]
    text, extras = builder(rng, pair)

    dv, iv = pair["dv"], pair["iv"]
    _verify_no_answer_leak(text, dv, iv, extras)

    distractors = [iv, *extras]
    check(len(distractors) >= 2, "must offer at least two distractors")

    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=dv,
        answer_text=dv,
        answer_latex=latex_answer(dv, plain=True),
        working=[f"The researcher's aim is to see how {dv} responds to {iv}, so {dv} is the dependent variable."],
        distractors=distractors,
    )
