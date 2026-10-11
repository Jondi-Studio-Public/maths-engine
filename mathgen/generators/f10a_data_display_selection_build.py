"""Generates `f10-data-display-selection` questions (AC9M9ST04, Year 9): name the display that suits a
described data set and a stated purpose, `answer: mcq`.

Scope: a short scenario describes what was recorded and what the reader wants to see. The learner names the one
display, from a four-option list, that fits both the data and the purpose. Every question is text only: no display is
drawn, and no display is read or built. Justifying the choice and interpreting a finished display are outside this
skill.

Each scenario class has exactly one defensible display. A display that a careful person could also defend for that
class (for example a pie chart when the purpose is to compare category sizes, or a dot plot for a small data set)
is never offered as an option, so the choice is a decision and not a matter of taste. Every distractor is a named
misconception, recorded in `option_notes`.

    A  compare how many fall in each category                       -> column graph
    B  show each category's share of a whole                        -> pie chart
    C  show the shape of one continuous numerical variable          -> histogram
    D  show how one quantity changed over time                      -> line graph
    E  see whether two numerical variables measured together relate -> scatterplot
    F  compare two groups value by value                            -> back-to-back stem-and-leaf plot

The class is drawn first, uniformly, so no display is the key more than about a fifth of the time.

Ladder (reasoning escalation, not size):
  easy:   the scenario names the kind of variable outright ("The variable is categorical"), so the learner only
          has to match a named data type and purpose to a display.
  medium: the kind of variable is not named; the learner reads it from what was recorded.
  hard:   the surface form points the wrong way. A numeric label (a bib, a postcode, a room number) is really a
          category; data recorded day by day is wanted only for its spread; a year column is a time axis and not
          an ordinary variable; whole-number hours are a measured variable, not a time sequence.
"""

from __future__ import annotations

import random

from mathgen.core import check, generator, latex_answer, phrase

TIERS = ("easy", "medium", "hard")

COLUMN = "column graph"
PIE = "pie chart"
HIST = "histogram"
LINE = "line graph"
SCATTER = "scatterplot"
BACK = "back-to-back stem-and-leaf plot"

#: Misconception id for each wrong display, per class: [(display, misconception id), ...]. The displays listed are
#: the only ones ever offered for that class; a display that would also be defensible is left out on purpose.
WRONG = {
    "A": [
        (LINE, "line-graph-for-categories"),
        (HIST, "histogram-for-categories"),
        (SCATTER, "scatterplot-for-one-variable"),
        (BACK, "two-group-display-for-one-group"),
    ],
    "B": [
        (LINE, "line-graph-for-categories"),
        (HIST, "histogram-for-categories"),
        (SCATTER, "scatterplot-for-one-variable"),
        (BACK, "two-group-display-for-one-group"),
    ],
    "C": [
        (PIE, "pie-chart-for-numerical-data"),
        (LINE, "line-graph-without-a-time-axis"),
        (SCATTER, "scatterplot-for-one-variable"),
        (BACK, "two-group-display-for-one-group"),
    ],
    "D": [
        (PIE, "pie-chart-for-numerical-data"),
        (HIST, "distribution-display-ignores-time"),
        (BACK, "two-group-display-for-one-group"),
    ],
    "E": [
        (PIE, "pie-chart-for-numerical-data"),
        (LINE, "line-graph-without-a-time-axis"),
        (HIST, "histogram-for-two-variables"),
        (COLUMN, "column-graph-for-numerical-pairs"),
        (BACK, "two-group-display-for-one-group"),
    ],
    "F": [
        (PIE, "pie-chart-for-numerical-data"),
        (SCATTER, "scatterplot-for-groups"),
        (LINE, "line-graph-without-a-time-axis"),
        (COLUMN, "column-graph-hides-individual-values"),
    ],
    # Hard tier. Each class is a scenario whose surface form points at a different display.
    "H1": [
        (HIST, "numeric-label-treated-as-measurement"),
        (LINE, "line-graph-for-categories"),
        (SCATTER, "scatterplot-for-one-variable"),
        (BACK, "two-group-display-for-one-group"),
    ],
    "H2": [
        (LINE, "line-graph-because-data-has-dates"),
        (PIE, "pie-chart-for-numerical-data"),
        (SCATTER, "scatterplot-for-one-variable"),
        (BACK, "two-group-display-for-one-group"),
    ],
    "H3": [
        (SCATTER, "time-treated-as-ordinary-variable"),
        (HIST, "distribution-display-ignores-time"),
        (PIE, "pie-chart-for-numerical-data"),
        (BACK, "two-group-display-for-one-group"),
    ],
    "H4": [
        (LINE, "line-graph-for-ordered-pairs-not-time"),
        (COLUMN, "column-graph-for-numerical-pairs"),
        (HIST, "histogram-for-two-variables"),
        (PIE, "pie-chart-for-numerical-data"),
    ],
}

KEY = {
    "A": COLUMN,
    "B": PIE,
    "C": HIST,
    "D": LINE,
    "E": SCATTER,
    "F": BACK,
    "H1": COLUMN,
    "H2": HIST,
    "H3": LINE,
    "H4": SCATTER,
}

# What each class says about itself, used only by the easy tier ("The variable is ...").
KIND_LINE = {
    "A": "The variable is categorical: each person falls in exactly one category.",
    "B": "The variable is categorical: each item falls in exactly one category.",
    "C": "The variable is numerical and continuous: each value is a measurement.",
    "D": "The variable is numerical and was recorded at regular points in time.",
    "E": "There are two numerical variables, both measured on every individual.",
    "F": "The variable is numerical, measured separately in two groups.",
}

# Scenario banks: (setup, purpose). `{n}` is filled with a seeded count. The purpose is the only sentence that says
# what the reader wants; the setup says what was recorded.
SCENARIOS = {
    "A": [
        (
            "A school asks each of {n} students for their favourite lunch (pizza, wrap, salad, sushi or pasta).",
            "The aim is to compare how many students chose each lunch.",
        ),
        (
            "A library records which of six genres each of {n} borrowed books belongs to.",
            "The aim is to compare how many books were borrowed in each genre.",
        ),
        (
            "A council surveys {n} residents on how they usually travel to work (car, bus, train, bike or on foot).",
            "The aim is to compare the number of residents using each way of travelling.",
        ),
        (
            "A vet clinic records the type of pet (dog, cat, bird, rabbit or other) for each of {n} visits.",
            "The aim is to compare how many visits there were for each type of pet.",
        ),
    ],
    "B": [
        (
            "A household splits its monthly budget between rent, food, transport, savings and other costs.",
            "The aim is to show what share of the whole budget each part takes up.",
        ),
        (
            "In a survey of {n} students, each student ticked exactly one of four ways of getting to school.",
            "The aim is to show what part of all the students each way of getting to school makes up.",
        ),
        (
            "A canteen sold {n} items, each one a sandwich, a drink, a piece of fruit or a snack.",
            "The aim is to show what share of the total sales each type of item made up.",
        ),
        (
            "Each of {n} students belongs to exactly one of four school houses.",
            "The aim is to show what part of the whole group each house makes up.",
        ),
    ],
    "C": [
        (
            "A gardener measures the height, to the nearest 0.1 cm, of each of {n} sunflower seedlings.",
            "The aim is to show the shape of the heights: where most lie and how spread out they are.",
        ),
        (
            "A coach times each of {n} students over 100 metres, to the nearest 0.01 second.",
            "The aim is to show the shape of the times: where most lie and how spread out they are.",
        ),
        (
            "A scientist weighs {n} eggs, to the nearest 0.1 gram.",
            "The aim is to show how the masses are distributed, including where they bunch up.",
        ),
        (
            "A bakery records the baking time, in minutes to one decimal place, of {n} loaves.",
            "The aim is to show the shape of the baking times and how spread out they are.",
        ),
    ],
    "D": [
        (
            "A town records its total rainfall, in millimetres, for each month of the year.",
            "The aim is to show how the rainfall changed through the year.",
        ),
        (
            "A shop records its sales, in dollars, on each day of a four-week period.",
            "The aim is to show how the daily sales changed over the four weeks.",
        ),
        (
            "A school records the number of students enrolled in each year from 2015 to 2024.",
            "The aim is to show the trend in enrolments over those years.",
        ),
        (
            "The closing price of a company share is recorded at the end of each week for 20 weeks.",
            "The aim is to show how the share price moved from week to week.",
        ),
    ],
    "E": [
        (
            "A teacher measures the height and the arm span of each of {n} students.",
            "The aim is to see whether taller students tend to have a longer arm span.",
        ),
        (
            "A cafe records the maximum temperature and the number of cold drinks sold on each of {n} different days.",
            "The aim is to see whether hotter days tend to sell more cold drinks.",
        ),
        (
            "A coach records the weekly training hours and the 5 km run time of each of {n} runners.",
            "The aim is to see whether more training tends to go with a faster time.",
        ),
        (
            "A researcher records the age and the reaction time, in milliseconds, of each of {n} adults.",
            "The aim is to see whether reaction time tends to change with age.",
        ),
    ],
    "F": [
        (
            "Each of {n} students in Class A and {n} students in Class B is timed solving a puzzle, to the nearest "
            "second.",
            "The aim is to compare how the individual times are spread in the two classes, value by value, and not "
            "only to compare totals.",
        ),
        (
            "The reaction times of {n} athletes and {n} non-athletes are recorded.",
            "The aim is to compare how the individual times are spread in the two groups, value by value.",
        ),
        (
            "The battery life, in hours, of {n} phones of brand X and {n} phones of brand Y is recorded.",
            "The aim is to compare the spread of the individual battery lives of the two brands, value by value.",
        ),
        (
            "The test marks of {n} Year 7 students and {n} Year 9 students on the same test are recorded.",
            "The aim is to compare how the individual marks are spread in the two year levels, value by value.",
        ),
    ],
    # Hard tier: the surface form misleads.
    "H1": [
        (
            "A netball club has {n} players, and each wears a numbered bib between 1 and 15.",
            "The aim is to compare how many players wear each bib number.",
        ),
        (
            "A delivery firm records the four-digit postcode of each of {n} parcels (only five postcodes appear).",
            "The aim is to compare how many parcels went to each postcode.",
        ),
        (
            "A school records the room number (101 to 106) where each of {n} detentions was served.",
            "The aim is to compare how many detentions were served in each room.",
        ),
    ],
    "H2": [
        (
            "A town records its maximum temperature, to the nearest 0.1 degree, on each of {n} days in a row.",
            "The aim is to show how the temperatures are spread, whatever day each one happened on.",
        ),
        (
            "A call centre records how long, to the nearest 0.1 minute, each of {n} callers waited, in the order the "
            "calls arrived.",
            "The aim is to show the shape of the waiting times, not when each call arrived.",
        ),
        (
            "A fitness tracker records the distance walked, in kilometres to two decimal places, on each of {n} days.",
            "The aim is to show how the distances are distributed, not how they changed from day to day.",
        ),
    ],
    "H3": [
        (
            "A zoo records two numerical variables: the year (2010 to 2024) and the number of visitors that year.",
            "The aim is to show whether visitor numbers have been rising or falling.",
        ),
        (
            "A farm records two numerical variables: the year (2012 to 2024) and its total milk output that year.",
            "The aim is to show how the milk output has changed from year to year.",
        ),
        (
            "A city records two numerical variables: the year (2005 to 2024) and its population at the start of the "
            "year.",
            "The aim is to show the trend in the population.",
        ),
    ],
    "H4": [
        (
            "A teacher records, for each of {n} students, the whole number of hours studied (0 to 10) and the test "
            "mark.",
            "The aim is to see whether more study tends to go with a higher mark.",
        ),
        (
            "A gym records, for each of {n} members, the number of visits last month (0 to 30) and the change in "
            "their fitness score.",
            "The aim is to see whether more visits tend to go with a bigger improvement.",
        ),
        (
            "A farmer records, for each of {n} paddocks, the number of sheep (1 to 40) and the weight of wool "
            "collected.",
            "The aim is to see whether paddocks with more sheep tend to yield more wool.",
        ),
    ],
}

HARD_CLASSES = ("H1", "H2", "H3", "H4")
BASE_CLASSES = ("A", "B", "C", "D", "E", "F")

STEMS = [
    "Which display is the most appropriate?",
    "Which of these displays best suits the data and the aim?",
    "Which display should be drawn?",
]

REASON = {
    "A": "Counts for separate categories are compared by the height of separate columns.",
    "B": "Parts of one whole, each category's share of the total, are shown by sectors of a circle.",
    "C": "A continuous measurement is grouped into intervals and shown by touching bars, which gives the shape.",
    "D": "A quantity followed over time is shown by points joined in time order, so the trend is visible.",
    "E": "Two variables measured on the same individuals are shown as points, one per individual, so any relationship is visible.",
    "F": "Two groups of individual values are shown side by side on one set of stems, so their spreads can be compared.",
    "H1": "A bib number, postcode or room number is a label, so the variable is categorical and a column graph compares the counts.",
    "H2": "The aim ignores order in time, so the values are grouped into intervals and the shape is shown by a histogram.",
    "H3": "The year is a time axis, so a quantity followed over the years is shown by a line graph.",
    "H4": "Hours or visits are measured values, not a time order, so each individual is one point on a scatterplot.",
}


def _pick_distractors(rng: random.Random, cls: str) -> list[tuple[str, str]]:
    pool = list(WRONG[cls])
    long_in = [p for p in pool if p[0] == BACK]
    rest = [p for p in pool if p[0] != BACK]
    if long_in and len(pool) > 3 and rng.random() < 0.6:
        # Offer the longest-named display often, so the longest option is not a giveaway.
        chosen = long_in + rng.sample(rest, 2)
    elif len(rest) >= 3:
        chosen = rng.sample(rest, 3)
    else:
        chosen = rng.sample(pool, 3)
    rng.shuffle(chosen)
    return chosen


@generator(
    "f10a_data_display_selection",
    level=9,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def f10a_data_display_selection(rng: random.Random, tier: str = "easy") -> dict:
    if tier not in TIERS:
        raise ValueError(f"unknown tier {tier!r}")
    cls = rng.choice(HARD_CLASSES if tier == "hard" else BASE_CLASSES)
    setup, purpose = rng.choice(SCENARIOS[cls])
    n = rng.randrange(40, 200, 10) if cls not in ("F",) else rng.randrange(20, 60, 5)
    setup = setup.format(n=n)

    key = KEY[cls]
    picked = _pick_distractors(rng, cls)
    distractors = [d for d, _ in picked]
    options = [key, *distractors]
    check(len(set(options)) == 4, "f10-data-display-selection: options must be pairwise distinct")
    check(len(distractors) == 3, "f10-data-display-selection: exactly 3 distractors required")
    # Independent check: the key is the only option the class's own table does not list as a wrong display.
    check(
        [o for o in options if o not in {d for d, _ in WRONG[cls]}] == [key],
        "f10-data-display-selection: a defensible rival display was offered, or the key is listed as wrong",
    )

    parts = [setup]
    if tier == "easy":
        parts.append(KIND_LINE[cls])
    parts.append(purpose)
    parts.append(phrase(rng, STEMS))
    text = " ".join(parts)

    working = [
        f"Aim and data: {purpose[0].lower() + purpose[1:]}",
        REASON[cls],
        f"The most appropriate display is a {key}.",
    ]
    for d, m in picked:
        working.append(f"A {d} would not suit: {m.replace('-', ' ')}.")
    return dict(
        text=text,
        answer=key,
        answer_text=key,
        answer_latex=latex_answer(key, plain=True),
        working=working,
        distractors=distractors,
        option_notes={d: m for d, m in picked},
    )
