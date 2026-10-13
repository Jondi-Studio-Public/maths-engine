"""Generates `ir2a-question-to-data-needed` questions: from a statistical question to the data that answers it.

Every scenario poses a question that compares a measurement across groups ("do students who play a sport sleep
more than students who do not?"). The learner works out which variables the question needs, what type the
measurement is, who the data should come from and how many values that makes:

  * the variables needed, picked by number from a list of things that could be recorded, the one measured on each
    member first and the one that sorts members into groups second (an ordered list);
  * the type of the measured variable (multiple choice, five options);
  * the plan that collects the right data from the right population (multiple choice, five options);
  * the size of a sample taken as a percentage of the population, and the number of values recorded.

Ladder (reasoning escalation, not size):
  easy:   four things that could be recorded; list the two the question needs, in the stated order.
  medium: five things that could be recorded, three of them unrelated to the question; list the two needed, then
          give the type of the measured variable (nominal, ordinal, discrete or continuous).
  hard:   the learner chooses among five data-collection plans, each of which gets one thing wrong (a variable
          missing, an unrelated variable, one group only, the wrong population), then works out the sample size
          and the total number of values recorded.
  exam:   the list, the type, the sample size and the total number of values, each part using the one before.
"""

from __future__ import annotations

import random

from mathgen.core import check, generator

from ._ir2a_kit import NAMES, chain, mcq_part, number_part, ordered_part, single

_TIERS = ("easy", "medium", "hard")

NOMINAL = "Categorical (nominal)"
ORDINAL = "Categorical (ordinal)"
DISCRETE = "Numerical (discrete)"
CONTINUOUS = "Numerical (continuous)"
VAGUE = "Impossible to tell until the data are collected"
TYPES = (NOMINAL, ORDINAL, DISCRETE, CONTINUOUS)

# question, population noun (plural), unit noun, population size choices, response (text, type), grouping
# (text, type), decoys [(text, type)] x 3, narrow population, wrong population
_SCENARIOS = (
    dict(
        q="Do Year 9 students who play a sport sleep for longer each night than Year 9 students who do not?",
        pl="Year 9 students at the school",
        unit="student",
        sizes=[160, 200, 240, 280],
        R=("the number of hours the student slept last night", CONTINUOUS),
        G=("whether the student plays a sport", NOMINAL),
        decoys=[
            ("the student's favourite subject", NOMINAL),
            ("the student's height in centimetres", CONTINUOUS),
            ("the number of siblings the student has", DISCRETE),
        ],
        narrow="the Year 9 students who play a sport",
        other="the teachers at the school",
    ),
    dict(
        q="Do students who catch the bus to school arrive earlier than students who walk?",
        pl="students at the school",
        unit="student",
        sizes=[800, 900, 1000, 1200],
        R=("the number of minutes before the bell the student arrived", CONTINUOUS),
        G=("how the student travels to school", NOMINAL),
        decoys=[
            ("the student's year level", ORDINAL),
            ("the number of students in the student's class", DISCRETE),
            ("the student's favourite subject", NOMINAL),
        ],
        narrow="the students who catch the bus",
        other="the staff at the school",
    ),
    dict(
        q="Do households in the northern suburbs own more pets than households in the southern suburbs?",
        pl="households in the Shire of Marlow",
        unit="household",
        sizes=[2000, 2400, 3000, 4000],
        R=("the number of pets the household owns", DISCRETE),
        G=("the suburb the household lives in", NOMINAL),
        decoys=[
            ("the weekly rent the household pays (in dollars)", CONTINUOUS),
            ("the household's preferred type of pet food", NOMINAL),
            ("how satisfied the household is with the council (from very low to very high)", ORDINAL),
        ],
        narrow="the households in the northern suburbs",
        other="the businesses in the Shire of Marlow",
    ),
    dict(
        q="Do tomato plants given fertiliser grow taller than tomato plants that are not?",
        pl="tomato plants in the garden",
        unit="plant",
        sizes=[120, 160, 200, 240],
        R=("the height of the plant after six weeks (in centimetres)", CONTINUOUS),
        G=("whether the plant was given fertiliser", NOMINAL),
        decoys=[
            ("the number of tomatoes on the plant", DISCRETE),
            ("the colour of the plant's pot", NOMINAL),
            ("the number of leaves on the plant", DISCRETE),
        ],
        narrow="the plants that are given fertiliser",
        other="the weeds in the garden",
    ),
    dict(
        q="Do Year 10 students jump further in the long jump than Year 7 students?",
        pl="Year 7 and Year 10 students at the school",
        unit="student",
        sizes=[300, 360, 420, 480],
        R=("the length of the student's best jump (in metres)", CONTINUOUS),
        G=("the student's year level", ORDINAL),
        decoys=[
            ("the student's house colour", NOMINAL),
            ("the number of jumps the student takes in training each week", DISCRETE),
            ("the student's favourite sport", NOMINAL),
        ],
        narrow="the Year 10 students",
        other="the students at a different school",
    ),
    dict(
        q="Does the brand of battery change how many hours a torch stays lit?",
        pl="batteries in the store's stockroom",
        unit="battery",
        sizes=[400, 500, 600, 800],
        R=("the number of hours the torch stays lit on the battery", CONTINUOUS),
        G=("the brand of the battery", NOMINAL),
        decoys=[
            ("the price of the battery (in dollars)", CONTINUOUS),
            ("the number of batteries in the packet it came in", DISCRETE),
            ("the colour of the battery's label", NOMINAL),
        ],
        narrow="the batteries of one brand",
        other="the torches in the store",
    ),
    dict(
        q="Do households with a higher income band own more cars?",
        pl="households in the Shire of Marlow",
        unit="household",
        sizes=[2000, 2400, 3000, 4000],
        R=("the number of cars the household owns", DISCRETE),
        G=("the household's income band (from low to high)", ORDINAL),
        decoys=[
            ("the household's street name", NOMINAL),
            ("the number of people living in the household", DISCRETE),
            ("the distance from the household to the nearest station (in kilometres)", CONTINUOUS),
        ],
        narrow="the households in the highest income band",
        other="the businesses in the Shire of Marlow",
    ),
    dict(
        q="Do members who train with a coach run faster than members who train alone?",
        pl="members of the Wattle Park Running Club",
        unit="member",
        sizes=[200, 240, 300, 400],
        R=("the member's time for a 5 km run (in minutes)", CONTINUOUS),
        G=("whether the member trains with a coach", NOMINAL),
        decoys=[
            ("the number of runs the member does each week", DISCRETE),
            ("the member's favourite brand of shoes", NOMINAL),
            ("how enjoyable the member rates training (from not at all to very)", ORDINAL),
        ],
        narrow="the members who train with a coach",
        other="the club's volunteers",
    ),
)

_WRONG_TYPE_TAG = {
    (NOMINAL, ORDINAL): "ir2a-confuses-nominal-and-ordinal",
    (ORDINAL, NOMINAL): "ir2a-confuses-nominal-and-ordinal",
    (DISCRETE, CONTINUOUS): "ir2a-confuses-discrete-and-continuous",
    (CONTINUOUS, DISCRETE): "ir2a-confuses-discrete-and-continuous",
}


def _type_tag(key: str, wrong: str) -> str:
    cat = {NOMINAL, ORDINAL}
    if (key, wrong) in _WRONG_TYPE_TAG:
        return _WRONG_TYPE_TAG[(key, wrong)]
    if wrong == VAGUE:
        return "ir2a-says-the-type-cannot-be-known"
    return "ir2a-treats-a-number-as-a-category" if wrong in cat else "ir2a-treats-a-category-as-a-number"


def _units(sc) -> str:
    unit = sc["unit"]
    return unit[:-1] + "ies" if unit.endswith("y") else unit + "s"


def _type_part(sc, marks=1):
    key = sc["R"][1]
    order = [key] + [t for t in (CONTINUOUS, DISCRETE, NOMINAL, ORDINAL) if t != key] + [VAGUE]
    wrongs = [(t, _type_tag(key, t)) for t in order[1:]]
    # the confusion closest to the key comes first
    wrongs.sort(key=lambda w: 0 if (key, w[0]) in _WRONG_TYPE_TAG else 1)
    return mcq_part(
        f"What type of variable is {sc['R'][0]}?",
        key,
        wrongs,
        marks=marks,
        working=[f"{sc['R'][0].capitalize()} is {key.lower()}."],
        keep=4,
    )


def _candidates(rng, sc, extra_decoys: int):
    """(numbered candidate lines, number of R, number of G)."""
    items = [("R", sc["R"][0]), ("G", sc["G"][0])] + [("D", d[0]) for d in sc["decoys"][:extra_decoys]]
    rng.shuffle(items)
    numbers = {tag: i + 1 for i, (tag, _) in enumerate(items) if tag in ("R", "G")}
    lines = " ".join(f"({i + 1}) {text};" for i, (_, text) in enumerate(items))
    decoy_numbers = [i + 1 for i, (tag, _) in enumerate(items) if tag == "D"]
    return lines.rstrip(";") + ".", numbers["R"], numbers["G"], decoy_numbers


def _order_part(sc, r, g, decoys, marks):
    key = [str(r), str(g)]
    wrongs = [([str(g), str(r)], "ir2a-lists-the-groups-before-the-measurement")]
    for d in decoys:
        wrongs.append(([str(r), str(d)], "ir2a-picks-an-unrelated-variable"))
        wrongs.append(([str(d), str(g)], "ir2a-picks-an-unrelated-variable"))
    return ordered_part(
        "Which two things must be recorded? Write their numbers separated by a comma, the measurement first and "
        f"then the one that sorts the {_units(sc)} into groups.",
        key,
        wrongs,
        marks=marks,
        working=[f"The measurement is ({r}) and the grouping is ({g})."],
    )


def _plans(sc):
    pl = sc["pl"]
    r, g = sc["R"][0], sc["G"][0]
    d = sc["decoys"][0][0]
    key = f"Take a random sample of the {pl} and record {r} and {g} for each."
    wrongs = [
        (f"Take a random sample of the {pl} and record {r} for each.", "ir2a-collects-one-variable-only"),
        (f"Take a random sample of the {pl} and record {d} and {g} for each.", "ir2a-collects-an-unrelated-variable"),
        (f"Take only {sc['narrow']} and record {r} and {g} for each.", "ir2a-collects-from-one-group-only"),
        (f"Take a random sample of {sc['other']} and record {r} and {g} for each.", "ir2a-asks-the-wrong-population"),
    ]
    return key, wrongs


def _sample_parts(rng, sc, n_label="b", marks_b=2, marks_c=2):
    N = rng.choice(sc["sizes"])
    pct = rng.choice([5, 10, 15, 20, 25])
    check(N * pct % 100 == 0, "sample size must be whole")
    n = N * pct // 100
    b = number_part(
        prompt=f"A random sample of {pct}% of the {N:,} {_units(sc)} is taken. How many {_units(sc)} are in the sample?",
        exact=n,
        places=0,
        marks=marks_b,
        working=[f"{pct}% of {N:,} = {n}."],
        wrong=[
            (pct, "ir2a-reports-the-percentage-as-the-count"),
            (N - n, "ir2a-gives-those-left-out"),
            (N // 2, "ir2a-takes-half-the-population"),
        ],
        sensible_range=(0, N),
    )
    c = number_part(
        prompt=f"Both variables are recorded for every {sc['unit']} in the sample. How many values are recorded altogether?",
        exact=2 * n,
        places=0,
        marks=marks_c,
        working=[f"{n} x 2 = {2 * n}."],
        wrong=[
            (n, "ir2a-counts-one-variable-only"),
            (n + 2, "ir2a-adds-the-variable-count"),
            (N * 2, "ir2a-uses-the-whole-population"),
        ],
        follow_through=f"2*part_{n_label}",
        sensible_range=(0, None),
    )
    return N, n, b, c


def _stem(rng, sc, name, lines):
    return (
        f"{name} wants to answer this question about the {sc['pl']}: “{sc['q']}” "
        f"On each {sc['unit']} {name} could record: {lines}"
    )


def _easy(rng):
    sc = rng.choice(_SCENARIOS)
    name = rng.choice(NAMES)
    lines, r, g, decoys = _candidates(rng, sc, 2)
    return single(_stem(rng, sc, name, lines), _order_part(sc, r, g, decoys, 2))


def _medium(rng):
    sc = rng.choice(_SCENARIOS)
    name = rng.choice(NAMES)
    lines, r, g, decoys = _candidates(rng, sc, 3)
    return chain(_stem(rng, sc, name, lines), _order_part(sc, r, g, decoys, 2), _type_part(sc, 1))


def _hard(rng):
    sc = rng.choice(_SCENARIOS)
    name = rng.choice(NAMES)
    key, wrongs = _plans(sc)
    stem = f"{name} wants to answer this question about the {sc['pl']}: “{sc['q']}”"
    a = mcq_part(
        "Which plan collects the data needed to answer the question?",
        key,
        wrongs,
        marks=2,
        working=["The plan must record the measurement and the grouping, for a random sample of the right population."],
        keep=4,
        bare=True,
    )
    N, n, b, c = _sample_parts(rng, sc)
    return chain(stem, a, b, c)


@generator("ir2a_question_to_data_needed", level=11, presets={t: dict(tier=t) for t in _TIERS})
def ir2a_question_to_data_needed(rng: random.Random, tier: str = "easy") -> dict:
    check(tier in _TIERS, f"unknown tier {tier!r}")
    return {"easy": _easy, "medium": _medium, "hard": _hard}[tier](rng)


@generator("exd_x_ir2a_question_to_data_needed", level=12, presets={"exam": dict(tier="exam")})
def exd_x_ir2a_question_to_data_needed(rng: random.Random, tier: str = "exam") -> dict:
    """The variables needed, the type of the measurement, the sample size and the number of values recorded."""
    check(tier == "exam", "only the exam tier exists")
    sc = rng.choice(_SCENARIOS)
    name = rng.choice(NAMES)
    lines, r, g, decoys = _candidates(rng, sc, 3)
    N, n, b, c = _sample_parts(rng, sc, n_label="c")
    stem = _stem(rng, sc, name, lines)
    return chain(stem, _order_part(sc, r, g, decoys, 2), _type_part(sc, 1), b, c)
