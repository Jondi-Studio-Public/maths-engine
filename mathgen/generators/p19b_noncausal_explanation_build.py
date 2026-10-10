"""`senior-identify-possible-non-causal-explanations-for-an-association` (ACMGM065, General
Mathematics).

Scope: given a description of an observed association between two variables
together with a plausible confounding or coincidental circumstance, classify the most
plausible non-causal explanation as coincidence, confounding due to a common response to
another variable, or a common cause.

Design rules:
1. The scenario text never states a category's definitional phrase (see BANNED_PHRASES);
   it states plain FACTS about the world (what a third variable does, whether X could
   plausibly affect Y). The three definitions are stated ONCE, after the scenario, as a
   fixed block; the MCQ options are the three short labels only.
2. Every scenario opens by stating the observed association between X and Y itself.
3. Every (category x X x Y x Z) combination is a hand-checked coherent real-world claim
   (see the pool comments below); no pair asserts a false direction.
4. Every scenario starts with a capital letter (the opening lead-in sentence).
5. The scenario/N/subject/lead-in pools are wide enough that 60 seeds rarely repeat.
6. MCQ options are the short labels (`Coincidence`, `Common cause`, `Confounding`).
7. The hard-tier irrelevant extra is a plausible study-meta quantity, never the linking
   variable.

`answer: mcq`, three options always (the three short labels), so always 2 distractors.
The category is drawn uniformly first, then the scenario is built to realise it
(construct-to-target).
"""

from __future__ import annotations

import random

from ..core import check, generator, latex_answer

# ---------------------------------------------------------------------------
# The three categories are defined ONCE per question, after the scenario. The
# MCQ options are the short labels only (never the full definition text).
# ---------------------------------------------------------------------------
LABEL_COINCIDENCE = "Coincidence"
LABEL_COMMON_CAUSE = "Common cause"
LABEL_CONFOUNDING = "Confounding"
OPTIONS = (LABEL_COINCIDENCE, LABEL_COMMON_CAUSE, LABEL_CONFOUNDING)

CATEGORIES = ("coincidence", "common_cause", "confounding")
_LABEL_FOR_CATEGORY = {
    "coincidence": LABEL_COINCIDENCE,
    "common_cause": LABEL_COMMON_CAUSE,
    "confounding": LABEL_CONFOUNDING,
}

_DEFINITION_COINCIDENCE = "Coincidence: the association arose by chance; no variable links the two."
_DEFINITION_COMMON_CAUSE = (
    "Common cause: a third variable directly drives both variables, and neither variable affects the other."
)
_DEFINITION_CONFOUNDING = (
    "Confounding (a common response to another variable): the explanatory variable may affect the "
    "response, but another variable that also affects the response changes along with it, so the two "
    "effects cannot be separated."
)
_DEFINITIONS_MARKER = "The three possible explanations are defined as follows."
_DEFINITIONS_BLOCK = (
    f"{_DEFINITIONS_MARKER} {_DEFINITION_COINCIDENCE} {_DEFINITION_COMMON_CAUSE} {_DEFINITION_CONFOUNDING}"
)
_QUESTION = "Based on the scenario above, which is the most plausible explanation for the observed association?"

# Scenario text (before `_DEFINITIONS_MARKER`) must never contain a category's own
# definitional phrase -- a learner should reason from facts, not string-match wording.
BANNED_PHRASES = (
    "independently",
    "neither",
    "cannot be cleanly separated",
    "cannot be separated",
    "no plausible variable",
    "by chance",
    "coincidence",
    "confound",
    "common cause",
    "common response",
)

# The fixed marker phrase (case-insensitive) that names each category's mechanism in
# the scenario text, used both to require the correct one and to forbid the others
# (which would make the scenario ambiguous).
_MARKER_PHRASE = {
    "coincidence": "no known factor has been found that links",
    "common_cause": "drives increases in both",
    "confounding": "is also known to affect",
}

# (X, Y) pairs with genuinely no plausible link -- classic spurious-correlation examples,
# framed as two series that happen to move together over time with no real mechanism.
COINCIDENCE_PAIRS = (
    (
        "the number of people who drowned by falling into a swimming pool",
        "the number of films a particular actor appeared in that year",
    ),
    (
        "national per-capita cheese consumption",
        "the number of people who died after becoming tangled in their bedsheets",
    ),
    ("the number of chess tournaments held in a small town", "the number of rainy days recorded in a distant country"),
    ("the number of new smartphone apps released each year", "the number of ice-cream flavours launched that year"),
    ("the number of pages in that year's almanac", "the average shoe size of graduating students"),
    (
        "the number of new mathematics papers published each year",
        "the number of umbrellas sold in a country with no rain that year",
    ),
    (
        "the number of tourists visiting a distant island each year",
        "the number of new bakeries opened in an unrelated country that year",
    ),
    ("the number of crossword puzzles published each year", "the number of new bicycle models released that year"),
)
N_YEARS_POOL = (8, 10, 12, 15, 20)

# (X, Y, Z, when_clause, subjects): Z is a shared trigger that drives increases in both
# X and Y; there is no direct causal path between X and Y themselves (checked by hand --
# an ice-cream buyer does not cause a swimmer, a sunburn does not cause an ice-lolly sale,
# a beach drowning does not cause a sunscreen sale, a heater sale does not cause a flu
# case, school attendance does not cause allergy-medication sales, an umbrella-carrier
# does not cause a bus delay, an office fan does not cause a cold-drink sale, a coat sale
# does not cause a hot-drink sale).
COMMON_CAUSE_TRIPLES = (
    (
        "ice-cream sales",
        "the number of people who go swimming",
        "hot summer weather",
        "during hot summer weather",
        "days",
    ),
    (
        "the number of sunburn cases",
        "the number of ice lollies sold",
        "hours of sunshine that day",
        "on days with more hours of sunshine",
        "days",
    ),
    ("the number of beach drownings", "sunscreen sales", "hot weather that week", "during hot weather", "weeks"),
    (
        "heater sales",
        "the number of flu cases reported in a suburb",
        "the arrival of winter",
        "when winter arrives",
        "weeks",
    ),
    (
        "school attendance rates",
        "sales of seasonal allergy medication",
        "the change of season from winter to spring",
        "as the season changes from winter to spring",
        "weeks",
    ),
    (
        "the number of people carrying umbrellas",
        "the number of bus delays",
        "heavy rainfall that day",
        "on days with heavy rainfall",
        "days",
    ),
    (
        "the number of fans running in an office",
        "the number of cold drinks sold in the office cafeteria",
        "a heatwave that week",
        "during a heatwave that week",
        "weeks",
    ),
    (
        "the number of coats sold in a department store",
        "the number of hot drinks sold in the same store",
        "cold weather that month",
        "during cold weather that month",
        "months",
    ),
)
N_POOL = (150, 200, 300, 500, 1000)

# (X, Y, Z, xy_clause, xz_clause, subjects): X may plausibly affect Y; Z moves alongside
# X and separately affects Y. `xy_clause` and `xz_clause` are hand-written to state the
# TRUE real-world direction (checked by hand):
# - exercise/cholesterol/age: more exercise -> lower cholesterol; more exercise -> younger.
# - coffee/heart disease/smoking: more coffee -> higher risk (classic confound); more coffee
#   -> more smoking.
# - homework/exam score/study environment: more homework -> higher score; more homework
#   -> better study environment.
# - income/life expectancy/healthcare access: higher income -> longer life expectancy;
#   higher income -> better healthcare access.
# - sleep/productivity/stress: more sleep -> higher productivity; more sleep -> lower stress.
# - books/reading scores/parents' education: more books -> higher scores; more books ->
#   more educated parents.
# - screen time/obesity risk/snacking: more screen time -> higher obesity risk (sedentary
#   behaviour); more screen time -> more snacking while on screens (which itself also raises
#   obesity risk -- both Z's path and X's own path push Y the SAME direction, so Z is a
#   genuine confound, not a suppressor working against the observed association).
# - advertising/sales/shoppers in market: more advertising -> higher sales; more
#   advertising -> more shoppers actively in the market that month.
#
# Every triple above was checked so that Z's own effect on Y pushes Y in the SAME direction
# as X's own (possible) effect on Y -- a Z that pushes Y the opposite way (a suppressor) is
# not a valid explanation for why X and Y appear associated, so no such triple is included
# (a hours-outdoors/vitamin-D/sunscreen-use triple was rejected for exactly this reason: more
# sunscreen use REDUCES vitamin D, working against, not for, the observed association).
CONFOUNDING_TRIPLES = (
    (
        "the amount of exercise a person does each week",
        "a person's cholesterol level",
        "the person's age",
        "adults who exercise more each week tend to have lower cholesterol levels",
        "adults who exercise more each week tend to be younger",
        "adults",
    ),
    (
        "coffee consumption",
        "a person's risk of heart disease",
        "the amount the person smokes",
        "people who drink more coffee tend to have a higher risk of heart disease",
        "people who drink more coffee tend to smoke more",
        "adults",
    ),
    (
        "the number of hours spent on homework",
        "a student's exam score",
        "the quality of the student's study environment at home",
        "students who spend more hours on homework tend to get higher exam scores",
        "students who spend more hours on homework tend to have a higher-quality study environment at home",
        "students",
    ),
    (
        "a neighbourhood's average income",
        "residents' life expectancy",
        "the neighbourhood's access to healthcare services",
        "neighbourhoods with higher average income tend to have residents with a longer life expectancy",
        "neighbourhoods with higher average income tend to have better access to healthcare services",
        "neighbourhoods",
    ),
    (
        "the amount of sleep a worker gets",
        "a worker's productivity score",
        "the worker's overall stress level",
        "workers who sleep more tend to have higher productivity scores",
        "workers who sleep more tend to have lower stress levels",
        "workers",
    ),
    (
        "the number of books in a household",
        "children's reading test scores",
        "the parents' own level of education",
        "households with more books tend to have children with higher reading test scores",
        "households with more books tend to have parents with a higher level of education",
        "households",
    ),
    (
        "the number of hours a person spends on screens each day",
        "a person's risk of obesity",
        "how much the person snacks while on screens",
        "people who spend more hours on screens each day tend to have a higher risk of obesity",
        "people who spend more hours on screens each day tend to snack more while on screens",
        "adults",
    ),
    (
        "a company's advertising spending",
        "the company's monthly sales",
        "the number of shoppers actively looking to buy that month",
        "months when a company spends more on advertising tend to be months with higher sales",
        "months when a company spends more on advertising tend to have more shoppers actively looking to buy",
        "months of retail data",
    ),
)

# Plausible study-meta quantities a researcher would genuinely record, never the linking
# variable for any triple above. Split by study frame: a study of named people/groups can
# genuinely record "the number of participants" or "the number of survey questions", but a
# time-series study of days/weeks/months/years of records has no "participants" to count --
# only the generic extras below fit every frame.
_IRRELEVANT_EXTRAS_GENERIC = (
    "the month the data collection began",
    "the researchers' funding source",
    "the year the study was published",
    "the region where the study was conducted",
)
_IRRELEVANT_EXTRAS_SUBJECTS = _IRRELEVANT_EXTRAS_GENERIC + (
    "the number of participants included in the study",
    "the number of survey questions used in the study",
)
# Subject types that name a group of people/entities rather than a run of time periods.
_PEOPLE_SUBJECTS = {"adults", "students", "neighbourhoods", "workers", "households"}

_LEAD_INS = (
    "A study of {N} {subjects} found that {clause}.",
    "Researchers studied {N} {subjects} and found that {clause}.",
    "Data collected from {N} {subjects} showed that {clause}.",
)
_LEAD_INS_TIME = (
    "A record covering {N} years found that {clause}.",
    "Data collected over {N} years showed that {clause}.",
    "Researchers tracked {N} years of records and found that {clause}.",
)


def _cap(s: str) -> str:
    return s[0:1].upper() + s[1:]


def _draw_scenario(rng, category: str):
    if category == "coincidence":
        X, Y = rng.choice(COINCIDENCE_PAIRS)
        return dict(
            X=X,
            Y=Y,
            Z=None,
            when_clause=None,
            xy_clause=None,
            xz_clause=None,
            N=rng.choice(N_YEARS_POOL),
            subjects=None,
        )
    if category == "common_cause":
        X, Y, Z, when_clause, subjects = rng.choice(COMMON_CAUSE_TRIPLES)
        return dict(
            X=X,
            Y=Y,
            Z=Z,
            when_clause=when_clause,
            xy_clause=None,
            xz_clause=None,
            N=rng.choice(N_POOL),
            subjects=subjects,
        )
    X, Y, Z, xy_clause, xz_clause, subjects = rng.choice(CONFOUNDING_TRIPLES)
    return dict(
        X=X,
        Y=Y,
        Z=Z,
        when_clause=None,
        xy_clause=xy_clause,
        xz_clause=xz_clause,
        N=rng.choice(N_POOL),
        subjects=subjects,
    )


def _opening(rng, category: str, s: dict) -> str:
    if category == "coincidence":
        lead = rng.choice(_LEAD_INS_TIME)
        clause = f"{s['X']} and {s['Y']} have tended to rise and fall together"
        return lead.format(N=s["N"], clause=clause)
    lead = rng.choice(_LEAD_INS)
    if category == "common_cause":
        clause = f"both {s['X']} and {s['Y']} increase {s['when_clause']}"
    else:
        clause = s["xy_clause"]
    return lead.format(N=s["N"], subjects=s["subjects"], clause=clause)


def _fact(category: str, s: dict) -> str:
    """Lowercase-first fact sentence stating the mechanism the category needs,
    never the category's own definitional wording."""
    if category == "coincidence":
        return f"no known factor has been found that links {s['X']} and {s['Y']}."
    if category == "common_cause":
        return f"{s['Z']} drives increases in both {s['X']} and {s['Y']}."
    return f"{s['Z']} is also known to affect {s['Y']} directly, since {s['xz_clause']}."


def _decision(category: str, s: dict) -> str:
    if category == "coincidence":
        return f"Researchers also search for a third factor connecting {s['X']} and {s['Y']}, without success."
    if category == "common_cause":
        return f"On reflection, {s['X']} does not act on {s['Y']} directly, and {s['Y']} does not act on {s['X']} directly."
    return f"{_cap(s['X'])} could plausibly affect {s['Y']} directly."


def _verify_scenario(text: str, category: str) -> None:
    """In-generator check: the scenario text (this string, before the definitions
    block is appended) never states a category's own definitional phrase, states the
    required marker phrase for the drawn category, and states no OTHER category's
    marker phrase (which would make the scenario ambiguous)."""
    lower = text.lower()
    for phrase in BANNED_PHRASES:
        check(phrase not in lower, f"banned phrase {phrase!r} found in scenario text: {text!r}")
    own_phrase = _MARKER_PHRASE[category]
    check(own_phrase in lower, f"required marker phrase {own_phrase!r} missing for category {category}: {text!r}")
    for other_cat, other_phrase in _MARKER_PHRASE.items():
        if other_cat == category:
            continue
        check(other_phrase not in lower, f"ambiguous scenario: also matches marker phrase for {other_cat!r}: {text!r}")


def _direct_effect(category, s) -> str:
    """Confounding needs the explanatory variable to plausibly affect the response. Without saying so, "Z affects
    Y, and Z goes with X" fits Common cause as well. The exam tier states it through `_decision`."""
    return f" {_decision(category, s)}" if category == "confounding" else ""


def _easy(rng, category, s):
    opening = _opening(rng, category, s)
    fact = _cap(_fact(category, s))
    return f"{opening}{_direct_effect(category, s)} {fact}"


def _medium(rng, category, s):
    opening = _opening(rng, category, s)
    fact = _fact(category, s)
    return f"{opening}{_direct_effect(category, s)} Separately, {fact}"


def _hard(rng, category, s):
    base = _medium(rng, category, s)
    pool = _IRRELEVANT_EXTRAS_SUBJECTS if s["subjects"] in _PEOPLE_SUBJECTS else _IRRELEVANT_EXTRAS_GENERIC
    W = _cap(rng.choice(pool))
    named = [x for x in (s["X"], s["Y"], s["Z"]) if x]
    others = " or ".join(named) if len(named) <= 2 else ", ".join(named[:-1]) + f", or {named[-1]}"
    return f"{base} {W} was also recorded during the study but does not relate to {others}."


def _exam(rng, category, s):
    opening = _opening(rng, category, s)
    decision = _decision(category, s)
    fact = _cap(_fact(category, s))
    return (
        f"{opening} Researchers first ask whether {s['X']} could plausibly have a "
        f"direct effect on {s['Y']}. {decision} {fact}"
    )


@generator(
    "p19b_noncausal_explanation",
    level=12,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
        "exam": dict(tier="exam"),
    },
)
def p19b_noncausal_explanation(rng: random.Random, tier="easy"):
    if tier not in ("easy", "medium", "hard", "exam"):
        raise ValueError(f"tier must be easy/medium/hard/exam; got {tier!r}")

    category = rng.choice(CATEGORIES)
    s = _draw_scenario(rng, category)

    builder = {"easy": _easy, "medium": _medium, "hard": _hard, "exam": _exam}[tier]
    scenario_text = builder(rng, category, s)

    _verify_scenario(scenario_text, category)

    text = f"{scenario_text} {_DEFINITIONS_BLOCK} {_QUESTION}"

    answer = _LABEL_FOR_CATEGORY[category]
    check(answer in OPTIONS, "answer must be one of the three fixed labels")
    distractors = [opt for opt in OPTIONS if opt != answer]
    check(len(distractors) == 2, "must offer exactly two distractors")

    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=answer,
        answer_text=answer,
        answer_latex=latex_answer(answer, plain=True),
        working=[f"The scenario states: {_cap(_fact(category, s))}"],
        distractors=distractors,
    )
