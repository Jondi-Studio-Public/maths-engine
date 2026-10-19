"""Generates `pv0c_choose_display` questions: choose the data display best suited to a described dataset.

Scope: given a short description of a dataset (the kind of variable
and how many variables are recorded), select from four labelled display types the one best suited to it.
"Best" is made decisive by variable type and number of variables ALONE -- never by taste -- via the fixed
compatibility matrix below.

Six mutually exclusive data classes, each with exactly one correct display
(`key`). Each class also names an "also acceptable" set of displays that
must NEVER be offered as an option for that class (offering one would make
the choice a matter of taste, not decision), and a "wrong" pool the three
distractors are drawn from.

    C1 one categorical variable                        -> column graph
    C2 one numerical variable, continuous, many values  -> histogram
    C3 two numerical variables, same individuals        -> scatterplot
    C4 one numerical variable, regular time points       -> line graph
    C5 one numerical variable, two groups compared        -> back-to-back stem-and-leaf plot
    C6 two categorical variables                          -> segmented column graph

The class is drawn FIRST, uniformly over all six, on every tier (construct-
to-target): six classes keep the top answer share near 1/6 on
every tier, comfortably under the 25% cap, and every option set is exactly
[key] + 3 distractors sampled from that class's own wrong pool, so every
draw has exactly three distractors by construction.

Tiers change the DECISION, not the numbers:
  - easy: the variable type and count are NAMED in the prompt.
  - medium: the type is not named; the learner infers it from a plain
    description.
  - hard: the surface form misleads -- numeric LABELS (postcodes, jersey
    numbers) that are actually categorical, or two numeric quantities where
    the learner must notice whether one is a time axis (C4) or both are
    measured on the same individuals with no time order (C3).
  - exam: an investigation purpose is described and the learner must extract
    the variable(s) and their roles before choosing.
"""

from __future__ import annotations

import random

from ..core import generator, check

TIERS = ("easy", "medium", "hard", "exam")

#: Exact option phrases -- lower case, plain text, as printed in the
#: compatibility matrix. Never edited per-class; only assembled per-class.
CLASSES = ("C1", "C2", "C3", "C4", "C5", "C6")

CLASS_INFO: dict[str, dict] = {
    "C1": dict(
        key="column graph",
        wrong=["histogram", "scatterplot", "line graph", "back-to-back stem-and-leaf plot"],
    ),
    "C2": dict(
        key="histogram",
        wrong=["column graph", "pie chart", "scatterplot", "line graph", "segmented column graph"],
    ),
    "C3": dict(
        key="scatterplot",
        wrong=[
            "histogram",
            "column graph",
            "pie chart",
            "line graph",
            "segmented column graph",
            "back-to-back stem-and-leaf plot",
        ],
    ),
    "C4": dict(
        key="line graph",
        wrong=["histogram", "pie chart", "back-to-back stem-and-leaf plot", "segmented column graph", "dot plot"],
    ),
    "C5": dict(
        key="back-to-back stem-and-leaf plot",
        wrong=["scatterplot", "pie chart", "line graph", "segmented column graph"],
    ),
    "C6": dict(
        key="segmented column graph",
        wrong=["histogram", "scatterplot", "line graph", "back-to-back stem-and-leaf plot"],
    ),
}

#: The full display universe -- used only to prove the matrix is internally
#: consistent (a class's key/wrong never collides) in the generator's own
#: check below, never to invent a new option at draw time.
_ALL_DISPLAYS = {
    "column graph",
    "pie chart",
    "dot plot",
    "segmented column graph",
    "histogram",
    "scatterplot",
    "line graph",
    "back-to-back stem-and-leaf plot",
}
for _cls, _info in CLASS_INFO.items():
    check(_info["key"] in _ALL_DISPLAYS, f"choose-display: unknown key for {_cls}")
    check(_info["key"] not in _info["wrong"], f"choose-display: {_cls} wrong pool contains its own key")
    check(len(set(_info["wrong"])) == len(_info["wrong"]), f"choose-display: {_cls} wrong pool has duplicates")
    check(len(_info["wrong"]) >= 3, f"choose-display: {_cls} wrong pool too small")

_WHY = {
    "column graph": "it compares the sizes of separate categories.",
    "histogram": "it shows how one measured quantity is spread across equal-width intervals.",
    "scatterplot": "it shows how two measured quantities vary together.",
    "line graph": "it shows how a quantity changes over time.",
    "back-to-back stem-and-leaf plot": "it compares the spread of two small data sets side by side.",
    "segmented column graph": "it compares how each category is split into parts across several groups.",
}

INSTRUCTION = "Which type of graph is best suited to display this data?"

# ---------------------------------------------------------------------------
# Scenario banks -- (subject-noun-phrase) tuples, one small bank per
# class per tier, chosen with the generator's own `rng` for variety.

_C1_SUBJECTS = [
    ("favourite sport", "students"),
    ("favourite music genre", "students"),
    ("blood type", "hospital patients"),
    ("preferred holiday destination", "survey respondents"),
    ("home suburb", "commuters"),
    ("t-shirt size ordered", "customers"),
]
_C1_HARD = [
    # Numeric LABELS that genuinely repeat across individuals: postcodes, route/gate/flight numbers and school house numbers
    # are shared by many people, unlike a unique identifier (room number,
    # jersey number, customer ID) which a segmented/column graph could not
    # sensibly tabulate.
    ("postcode", "customers who placed an online order"),
    ("flight number", "passengers who checked in this morning"),
    ("bus route number", "commuters using the city network this week"),
    ("gate number", "flights scheduled to depart this morning"),
]

_C2_SUBJECTS = [
    ("running time", "minutes", "movies"),
    ("battery life", "hours", "phones"),
    ("delivery time", "minutes", "parcels"),
    ("seedling height", "cm", "seedlings"),
    ("resting heart rate", "beats per minute", "patients"),
    ("commute time", "minutes", "workers"),
]
_C2_HARD = [
    # C2 is continuous measurements only: "length of
    # hospital stay, in days" was whole-day counts with few values, where a
    # column graph is defensible -- replaced with a genuine
    # continuous measurement.
    ("length of each customer service call", "minutes", "calls logged last week"),
    ("distance travelled to work", "km", "employees surveyed"),
    ("volume of fuel used per trip", "litres", "delivery vans logged over the past month"),
]

_C3_SUBJECTS = [
    ("height", "cm", "weight", "kg", "adults"),
    ("study time", "hours", "test score", "marks", "students"),
    ("advertising spend", "$", "weekly revenue", "$", "stores"),
    ("arm span", "cm", "height", "cm", "gymnasts"),
    ("floor area", "m²", "sale price", "$", "houses"),
]
_C3_HARD = [
    ("house price", "$", "floor area", "m²", "houses sold in a suburb last year"),
    ("annual income", "$", "years of education", "years", "adults surveyed"),
    ("study hours per week", "hours", "final exam mark", "marks", "students"),
]

_C4_SUBJECTS = [
    ("average daily temperature", "°C", "for each day of March"),
    # unit left empty -- "number of visitors
    # (people)" and "number of website hits (hits)" echoed the variable
    # name; `_paren` below omits the parenthetical for an empty unit.
    ("number of visitors", "", "for each month of last year"),
    ("closing share price", "$", "for each trading day over the past month"),
    ("monthly rainfall", "mm", "for each month from 2020 to 2024"),
    ("number of website hits", "", "for each week of the campaign"),
]
_C4_HARD = [
    ("closing share price", "$", "at the end of each trading day for the past {n} days"),
    ("daily maximum temperature", "°C", "at the same weather station for each of the past {n} days"),
    # "number of daily active users (users)" echoed;
    # unit left empty, see `_paren`.
    ("number of daily active users", "", "on an app for each of the past {n} days"),
]


def _paren(unit: str) -> str:
    """`" (unit)"` for a real unit, `""` for the empty-unit sentinel used
    where naming a unit would echo the variable name."""
    return f" ({unit})" if unit else ""


_C5_SUBJECTS = [
    ("reaction time", "ms", "left-handed students", "right-handed students"),
    ("resting heart rate", "beats per minute", "smokers", "non-smokers"),
    ("commute time", "minutes", "city residents", "suburban residents"),
    ("sleep duration", "hours", "shift workers", "day workers"),
]
_C5_HARD = [
    # replaced discrete counts ("number of sick days taken
    # (days)" etc.) with continuous measurements -- continuous
    # measures are preferred for C5, and the count phrasing echoed its
    # own unit after "number of".
    ("time taken to finish a 5 km fun run", "minutes", "runners under 30", "runners over 30"),
    ("mass of newborn babies", "kg", "babies born in public hospitals", "babies born in private hospitals"),
    ("length of nails grown in a month", "mm", "participants taking the supplement", "participants taking the placebo"),
]

_C6_SUBJECTS = [
    ("eye colour", "hair colour", "students"),
    ("car brand", "fuel type", "vehicles"),
    ("preferred subject", "preferred sport", "students"),
    ("playing position", "dominant foot", "football players"),
    ("phone brand", "operating system", "customers"),
]
_C6_HARD = [
    # unique-identifier pairs ("jersey number and playing
    # position", "room number and room type", "customer ID number and
    # membership tier") each had near-one-member-per-code columns -- a
    # segmented column graph would be degenerate. Replaced with numeric
    # CODES that genuinely repeat across individuals, paired with a second
    # categorical variable.
    ("bus route number", "ticket type", "passengers who boarded during peak hour"),
    ("flight number", "ticket class", "passengers who checked in this morning"),
    ("gate number", "airline", "flights scheduled to depart this morning"),
    ("school house number", "sport chosen for the athletics carnival", "students"),
]

_C1_EXAM = [
    "A cafe owner wants to know which drink is most popular. She records the drink each of {n} customers orders over a week.",
    "A council wants to know which bus route is used most. It records the route each of {n} commuters boards during peak hour.",
    "A city library wants to know which genre is borrowed most. It records the genre of each of {n} books borrowed last week.",
    "A gym manager wants to know which class time slot is most popular. She records the class time slot chosen by each of {n} members last month.",
]
_C2_EXAM = [
    "A quality-control officer wants to know how consistent a factory's bottling line is. She records the volume, in millilitres, of the liquid in each of {n} bottles sampled from the line.",
    "A nutritionist wants to know how much sleep university students get. She records the sleep duration, in hours, of {n} students over one night.",
    "An engineer wants to know how consistent a 3D printer's output is. She records the diameter, in millimetres, of {n} printed parts sampled from a production run.",
    "A dietitian wants to know how much water people drink. She records the daily water intake, in millilitres, of {n} participants over one day.",
]
_C3_EXAM = [
    "A sports scientist wants to know whether taller players jump higher. She records the height (cm) and the vertical jump (cm) of {n} basketball players.",
    "An economist wants to know whether higher advertising spend increases revenue. She records the advertising spend ($) and the weekly revenue ($) of {n} stores.",
    "A biologist wants to know whether taller trees have thicker trunks. She measures the height (m) and the trunk diameter (cm) of {n} trees in a forest plot.",
    "A researcher wants to know whether more training hours are linked to higher fitness test scores. She records the training hours per week and the fitness test score of {n} athletes.",
]
_C4_EXAM = [
    "An analyst wants to know how a company's share price has moved. She records the company's closing share price at the end of each trading day for the past {n} days.",
    "A climatologist wants to know how rainfall has changed over time. She records the total rainfall (mm) for each month over the past {n} months.",
    "A hydrologist wants to know how a river's flow has changed. She records the river's flow rate, in megalitres per day, for each day over the past {n} days.",
    "A retailer wants to know how weekly sales have trended. She records the store's total weekly sales, in dollars, for each week over the past {n} weeks.",
]
_C5_EXAM = [
    "A researcher wants to know whether left-handed students have faster reaction times than right-handed students. She records the reaction time, in milliseconds, of {n} left-handed students and {n} right-handed students.",
    "A doctor wants to know whether smokers have higher resting heart rates than non-smokers. She records the resting heart rate, in beats per minute, of {n} smokers and {n} non-smokers.",
    "A physiotherapist wants to know whether a stretching program improves flexibility more than a strength program. She records the flexibility test score, in cm, of {n} clients following the stretching program and {n} clients following the strength program.",
    "A researcher wants to know whether city runners are faster than regional runners in a fun run. She records the finishing time, in minutes, of {n} city runners and {n} regional runners.",
]
_C6_EXAM = [
    "A marketer wants to know whether product preference differs by age group. She records the age group and the preferred product of each of {n} survey respondents.",
    "A researcher wants to know whether car brand choice is linked to fuel type. She records the car brand and the fuel type of each of {n} vehicles registered this year.",
    "A council wants to know whether recycling participation differs by suburb. It records the suburb and whether each of {n} households recycles.",
    "An airline wants to know whether ticket class is linked to the check-in method used. It records the ticket class and the check-in method used by each of {n} passengers.",
]


def _describe(rng: random.Random, class_id: str, tier: str) -> str:
    """Build the dataset-description sentence (no trailing instruction) for
    `class_id` at `tier`. Every class appears on every tier."""
    if tier == "easy":
        if class_id == "C1":
            var, subject = rng.choice(_C1_SUBJECTS)
            n = rng.randint(20, 200)
            return f"One categorical variable: the {var} of each of {n} {subject}."
        if class_id == "C2":
            var, unit, subject = rng.choice(_C2_SUBJECTS)
            n = rng.randint(40, 400)
            return f"One numerical variable, continuous measurements, many values: the {var} ({unit}) of {n} {subject}."
        if class_id == "C3":
            v1, u1, v2, u2, subject = rng.choice(_C3_SUBJECTS)
            n = rng.randint(20, 200)
            return (
                f"Two numerical variables measured on the same individuals: "
                f"the {v1} ({u1}) and {v2} ({u2}) of {n} {subject}."
            )
        if class_id == "C4":
            var, unit, timeframe = rng.choice(_C4_SUBJECTS)
            return f"One numerical variable recorded at regular time points: the {var}{_paren(unit)} {timeframe}."
        if class_id == "C5":
            var, unit, g1, g2 = rng.choice(_C5_SUBJECTS)
            return f"One numerical variable compared between two groups: the {var} ({unit}) of {g1} and {g2}."
        # C6
        v1, v2, subject = rng.choice(_C6_SUBJECTS)
        n = rng.randint(20, 200)
        return f"Two categorical variables: the {v1} and {v2} of {n} {subject}."

    if tier == "medium":
        if class_id == "C1":
            var, subject = rng.choice(_C1_SUBJECTS)
            n = rng.randint(20, 200)
            return f"The {var} of each of {n} {subject}."
        if class_id == "C2":
            var, unit, subject = rng.choice(_C2_SUBJECTS)
            n = rng.randint(40, 400)
            return f"The {var}, in {unit}, measured for each of {n} {subject}."
        if class_id == "C3":
            v1, u1, v2, u2, subject = rng.choice(_C3_SUBJECTS)
            n = rng.randint(20, 200)
            return f"The {v1} ({u1}) and the {v2} ({u2}) recorded for {n} {subject}."
        if class_id == "C4":
            var, unit, timeframe = rng.choice(_C4_SUBJECTS)
            return f"The {var} recorded {timeframe}."
        if class_id == "C5":
            var, unit, g1, g2 = rng.choice(_C5_SUBJECTS)
            return f"The {var} ({unit}) of {g1}, compared with {g2}."
        v1, v2, subject = rng.choice(_C6_SUBJECTS)
        n = rng.randint(20, 200)
        return f"The {v1} and the {v2} recorded for {n} {subject}."

    if tier == "hard":
        if class_id == "C1":
            var, subject = rng.choice(_C1_HARD)
            n = rng.randint(20, 200)
            return f"The {var} of each of {n} {subject}."
        if class_id == "C2":
            var, unit, subject = rng.choice(_C2_HARD)
            n = rng.randint(40, 400)
            return f"The {var}, in {unit}, recorded for {n} {subject}."
        if class_id == "C3":
            v1, u1, v2, u2, subject = rng.choice(_C3_HARD)
            n = rng.randint(20, 200)
            return f"The {v1} ({u1}) and the {v2} ({u2}) recorded for {n} {subject}."
        if class_id == "C4":
            var, unit, timeframe_tpl = rng.choice(_C4_HARD)
            n = rng.randint(30, 200)
            return f"The {var}{_paren(unit)} recorded {timeframe_tpl.format(n=n)}."
        if class_id == "C5":
            var, unit, g1, g2 = rng.choice(_C5_HARD)
            return f"The {var} ({unit}) of {g1}, compared with {g2}."
        v1, v2, subject = rng.choice(_C6_HARD)
        n = rng.randint(20, 200)
        return f"The {v1} and the {v2} recorded for each of {n} {subject}."

    # exam
    n = rng.randint(20, 200)
    bank = {
        "C1": _C1_EXAM,
        "C2": _C2_EXAM,
        "C3": _C3_EXAM,
        "C4": _C4_EXAM,
        "C5": _C5_EXAM,
        "C6": _C6_EXAM,
    }[class_id]
    template = rng.choice(bank)
    return template.format(n=n)


def _pick_options(rng: random.Random, class_id: str) -> tuple[str, list[str]]:
    info = CLASS_INFO[class_id]
    key = info["key"]
    distractors = rng.sample(info["wrong"], 3)
    check(key not in distractors, "choose-display: distractor collides with the key")
    check(len(set(distractors)) == 3, "choose-display: distractors are not pairwise distinct")
    check(len(set([key, *distractors])) == 4, "choose-display: an option repeats")
    return key, distractors


@generator(
    "pv0c_choose_display",
    level=12,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
        "exam": dict(tier="exam"),
    },
)
def pv0c_choose_display(rng, tier: str = "easy"):
    if tier not in TIERS:
        raise ValueError(f"unknown tier {tier!r}")

    # Construct-to-target: draw the class FIRST, uniformly.
    class_id = CLASSES[rng.randrange(len(CLASSES))]
    key, distractors = _pick_options(rng, class_id)

    description = _describe(rng, class_id, tier)
    text = f"{description} {INSTRUCTION}"

    working = [f"The data described fits a {key}: {_WHY[key]}"]

    return dict(
        text=text,
        answer=key,
        answer_text=key,
        answer_latex=f"plain:{key}",
        working=working,
        distractors=distractors,
    )
