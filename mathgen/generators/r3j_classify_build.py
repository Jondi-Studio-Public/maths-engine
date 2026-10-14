"""Four-way classification of data variables: `r3j_classify_variable` (medium, hard) and the exam chain
`exd_x_r3j_classify_variables`, appended to the existing `classify-data` skill.

A pool of named variables, each tagged nominal, ordinal, discrete numerical or continuous numerical.

* medium: classify one variable whose type is clear from what is recorded, four options.
* hard: a variable whose wording invites a mistake (numerals used as labels, ranks and ratings, a measurement
  rounded to whole units, an age in completed years), or the reverse question: which of four variables is of a
  stated type.
* exam (``exd_x_r3j_classify_variables``): six variables listed; how many are categorical, how many are continuous,
  how many are discrete (following from the first two), and which display suits one named variable.
"""

from __future__ import annotations

from ..core import attempts, check, generator
from ._exd_distinct_parts import distinct_part_answers
from ._r3j_kit import choice_part, choice_question, number_part

NOMINAL = "Nominal categorical"
ORDINAL = "Ordinal categorical"
DISCRETE = "Discrete numerical"
CONTINUOUS = "Continuous numerical"
CLASSES = [NOMINAL, ORDINAL, DISCRETE, CONTINUOUS]

# (phrase, class, twist). A twist is a variable whose wording invites a wrong classification.
POOL = [
    # nominal
    ("the colour of each car in a car park", NOMINAL, False),
    ("each student's favourite sport", NOMINAL, False),
    ("the make of each car sold at a dealership", NOMINAL, False),
    ("the brand of cereal each shopper buys", NOMINAL, False),
    ("the country of birth of each resident", NOMINAL, False),
    ("the type of pet owned by each household", NOMINAL, False),
    ("the blood group of each blood donor", NOMINAL, False),
    ("the main language spoken at home by each student", NOMINAL, False),
    ("the suburb each student lives in", NOMINAL, False),
    ("the flavour of ice cream chosen by each customer", NOMINAL, False),
    ("the postcode of each customer's address", NOMINAL, True),
    ("the mobile phone number of each club member", NOMINAL, True),
    ("the jersey number worn by each netball player", NOMINAL, True),
    ("the bus route number each commuter catches", NOMINAL, True),
    ("the student ID number of each student", NOMINAL, True),
    # ordinal
    ("a satisfaction rating (very unhappy, unhappy, neutral, happy, very happy) from each customer", ORDINAL, False),
    ("the size of each drink ordered (small, medium or large)", ORDINAL, False),
    ("the grade (A, B, C, D or E) awarded to each student", ORDINAL, False),
    ("the bushfire danger rating (low, moderate, high or extreme) for each day", ORDINAL, False),
    ("the T-shirt size (S, M, L or XL) ordered by each customer", ORDINAL, False),
    ("the highest level of schooling completed (Year 10, Year 12, diploma or degree) by each adult", ORDINAL, False),
    ("the level of agreement (strongly disagree to strongly agree) with a statement", ORDINAL, False),
    ("the finishing place (1st, 2nd, 3rd and so on) of each runner in a race", ORDINAL, True),
    ("the star rating (1 to 5 stars) given to each restaurant", ORDINAL, True),
    ("the pain level (0 for none to 10 for the worst) reported by each patient", ORDINAL, True),
    ("the year level (Year 7 to Year 12) of each student in a school", ORDINAL, True),
    ("the difficulty level (1 to 5) of each puzzle in a magazine", ORDINAL, True),
    # discrete
    ("the number of siblings each student has", DISCRETE, False),
    ("the number of goals scored in each football match", DISCRETE, False),
    ("the number of text messages each student sends in a day", DISCRETE, False),
    ("the number of cars passing a gate each minute", DISCRETE, False),
    ("the number of rooms in each house", DISCRETE, False),
    ("the number of students absent from school each day", DISCRETE, False),
    ("the number of books borrowed from a library each week", DISCRETE, False),
    ("the number of faulty items in each box of 50", DISCRETE, False),
    ("the number of eggs laid by each hen in a week", DISCRETE, False),
    ("the number of correct answers (out of 20) on a quiz", DISCRETE, False),
    ("the number of coins in each person's pocket", DISCRETE, False),
    ("the shoe size (in whole and half sizes) of each student", DISCRETE, True),
    ("each person's age in completed years", DISCRETE, True),
    ("the number of customers in a shop at 1 pm each day", DISCRETE, True),
    ("the number of pets in each household in a street", DISCRETE, True),
    # continuous
    ("the height of each student", CONTINUOUS, False),
    ("the time each runner takes to complete a 100 metre sprint", CONTINUOUS, False),
    ("the mass of each parcel at a post office", CONTINUOUS, False),
    ("the volume of petrol in each car's tank", CONTINUOUS, False),
    ("the temperature at noon on each day", CONTINUOUS, False),
    ("the length of each fish caught in a competition", CONTINUOUS, False),
    ("the time each customer waits in a queue", CONTINUOUS, False),
    ("the distance each student travels to school", CONTINUOUS, False),
    ("the rainfall in each month", CONTINUOUS, False),
    ("the speed of each car recorded by a speed camera", CONTINUOUS, False),
    ("the diameter of each ball bearing made by a machine", CONTINUOUS, False),
    ("the battery life of each phone", CONTINUOUS, False),
    ("the mass of each newborn baby", CONTINUOUS, False),
    ("the area of each block of land", CONTINUOUS, False),
    ("the reaction time of each driver", CONTINUOUS, False),
    ("the height of each student, recorded to the nearest centimetre", CONTINUOUS, True),
    ("the time each student takes to complete a puzzle, recorded to the nearest minute", CONTINUOUS, True),
    ("the mass of each parcel, recorded to the nearest kilogram", CONTINUOUS, True),
    ("each person's exact age, including years, months and days", CONTINUOUS, True),
    ("the amount of sugar in a can of drink, recorded to the nearest gram", CONTINUOUS, True),
    ("the distance each student can throw a cricket ball, recorded to the nearest metre", CONTINUOUS, True),
]
check(len(POOL) >= 60, "the pool should hold at least 60 variables")
check(len({p[0] for p in POOL}) == len(POOL), "pool phrases must be distinct")

_WRONG_TAG = {
    (NOMINAL, ORDINAL): "r3j-classify-order-assumed",
    (NOMINAL, DISCRETE): "r3j-classify-numerals-as-numbers",
    (NOMINAL, CONTINUOUS): "r3j-classify-numerals-as-numbers",
    (ORDINAL, NOMINAL): "r3j-classify-order-ignored",
    (ORDINAL, DISCRETE): "r3j-classify-ranks-as-counts",
    (ORDINAL, CONTINUOUS): "r3j-classify-ranks-as-counts",
    (DISCRETE, CONTINUOUS): "r3j-classify-count-as-continuous",
    (DISCRETE, NOMINAL): "r3j-classify-number-as-category",
    (DISCRETE, ORDINAL): "r3j-classify-number-as-category",
    (CONTINUOUS, DISCRETE): "r3j-classify-measure-as-discrete",
    (CONTINUOUS, NOMINAL): "r3j-classify-number-as-category",
    (CONTINUOUS, ORDINAL): "r3j-classify-number-as-category",
}
_WHY = {
    NOMINAL: "its values are names or labels with no natural order",
    ORDINAL: "its values are categories that can be put in a natural order, but the gaps between them are not measurements",
    DISCRETE: "its values are counted, so they are separate whole-number steps",
    CONTINUOUS: "its values are measured, so any value on a scale is possible, even when it is recorded to a whole unit",
}
_CATEGORICAL = (NOMINAL, ORDINAL)


def _classify(rng, twist):
    phrase, cls, _ = rng.choice([p for p in POOL if p[2] == twist])
    leads = [
        "A survey records {p}. How is this variable best classified?",
        "In a study, the variable is {p}. What type of variable is it?",
        "Data are collected on {p}. Classify the variable.",
    ]
    text = rng.choice(leads).format(p=phrase)
    slips = [(c, _WRONG_TAG[(cls, c)]) for c in CLASSES if c != cls]
    return choice_question(text, cls, [f"The variable is {cls.lower()}: {_WHY[cls]}."], slips)


def _cap(text):
    return text[0].upper() + text[1:]


def _which(rng):
    """Which of four variables is of a stated type?"""
    asked = rng.choice(CLASSES)
    for _ in attempts(200, "four variables of four different types"):
        chosen = {
            c: rng.choice([p for p in POOL if p[1] == c and p[2] == (c != asked or rng.random() < 0.5)])
            for c in CLASSES
        }
        if len({v[0] for v in chosen.values()}) == 4:
            break
    key = _cap(chosen[asked][0])
    slips = [(_cap(chosen[c][0]), _WRONG_TAG[(c, asked)]) for c in CLASSES if c != asked]
    article = "an" if asked == ORDINAL else "a"
    text = f"Which of these variables is {article} {asked.lower()} variable?"
    return choice_question(
        text,
        key,
        [f"{key} is {asked.lower()}: {_WHY[asked]}."],
        slips,
    )


@generator(
    "r3j_classify_variable",
    level=8,
    presets={"medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def r3j_classify_variable(rng, tier="medium"):
    if tier == "medium":
        return _classify(rng, False)
    check(tier == "hard", f"unknown tier {tier!r}")
    return _classify(rng, True) if rng.random() < 0.5 else _which(rng)


# -- exam ---------------------------------------------------------------------------------------------------------

_DISPLAYS = {
    CONTINUOUS: (
        "Histogram",
        [
            ("Sector (pie) graph", "r3j-classify-display-for-categories"),
            ("Bar chart with a gap between the bars", "r3j-classify-display-for-categories"),
            ("Scatterplot", "r3j-classify-display-needs-two-variables"),
        ],
    ),
    NOMINAL: (
        "Bar chart",
        [
            ("Histogram", "r3j-classify-histogram-for-categories"),
            ("Box plot", "r3j-classify-box-plot-for-categories"),
            ("Scatterplot", "r3j-classify-display-needs-two-variables"),
        ],
    ),
}


@generator("exd_x_r3j_classify_variables", level=12, presets={"exam": dict(tier="exam")})
@distinct_part_answers
def exd_x_r3j_classify_variables(rng, tier="exam"):
    check(tier == "exam", f"unknown tier {tier!r}")
    n_cat, n_cont, n_disc = rng.sample([1, 2, 3], 3)
    chosen = []
    for _ in range(n_cat):
        chosen.append(rng.choice([p for p in POOL if p[1] in _CATEGORICAL and p not in chosen]))
    for _ in range(n_cont):
        chosen.append(rng.choice([p for p in POOL if p[1] == CONTINUOUS and p not in chosen]))
    for _ in range(n_disc):
        chosen.append(rng.choice([p for p in POOL if p[1] == DISCRETE and p not in chosen]))
    rng.shuffle(chosen)
    letters = "ABCDEF"
    listing = "; ".join(f"{letters[i]}: {v[0]}" for i, v in enumerate(chosen))
    text = f"A researcher records six variables. {listing}."
    # the display part names one continuous variable or one nominal variable
    target_class = rng.choice([CONTINUOUS, NOMINAL])
    targets = [i for i, v in enumerate(chosen) if v[1] == target_class]
    if not targets:
        target_class = CONTINUOUS
        targets = [i for i, v in enumerate(chosen) if v[1] == CONTINUOUS]
    target = rng.choice(targets)
    display_key, display_slips = _DISPLAYS[target_class]
    # the display is only unambiguous if exactly one variable in the six is of the target's kind or the wording names it
    check(sum(1 for v in chosen if v[1] in _CATEGORICAL) == n_cat, "categorical count disagrees")
    parts = [
        _count_part(
            "How many of the six variables are categorical?",
            n_cat,
            1,
            [
                (n_cat + n_disc, "r3j-classify-count-as-category"),
                (6 - n_cat, "r3j-classify-numerical-count-given"),
                (n_cat + n_cont, "r3j-classify-measured-as-category"),
            ],
            "A categorical variable has values that are names or ordered categories (nominal or ordinal).",
        ),
        _count_part(
            "How many of the six variables are continuous numerical?",
            n_cont,
            1,
            [
                (n_cont + n_disc, "r3j-classify-numerical-count-given"),
                (n_disc, "r3j-classify-measure-as-discrete"),
                (n_cont + n_cat, "r3j-classify-numeral-labels-as-measured"),
            ],
            "A continuous variable is measured, so any value on a scale is possible, even when recorded to a whole unit.",
        ),
        _count_part(
            "How many of the six variables are discrete numerical?",
            n_disc,
            2,
            [
                (n_cont, "r3j-classify-count-as-continuous"),
                (n_disc + n_cat, "r3j-classify-count-as-category"),
                (6 - n_cat, "r3j-classify-numerical-count-given"),
            ],
            "The variables that are neither categorical nor continuous are the counted ones.",
        ),
        choice_part(
            f"Which display is most suitable for showing the distribution of variable {letters[target]}?",
            display_key,
            1,
            [f"Variable {letters[target]} is {target_class.lower()}, so {display_key.lower()} suits it."],
            display_slips,
        ),
    ]
    parts[2]["follow_through"] = "6 - part_a - part_b"
    return dict(text=text, parts=parts)


def _count_part(prompt, value, marks, slips, why):
    # `slips` may repeat the key or each other; number_part drops those
    slips = list(slips) + [
        (value + 1, "r3j-classify-miscounted-one-out"),
        (value - 1, "r3j-classify-miscounted-one-out"),
    ]
    return number_part(prompt, value, 0, marks, [why, f"The count is {value}."], slips, sensible_range=(0, 6))
