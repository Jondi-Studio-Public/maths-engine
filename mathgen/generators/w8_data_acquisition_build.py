"""Generates `w8_data_acquisition` questions (easy, medium, hard) and the exam chain `exd_x_w8_data_acquisition`.

``acquire-and-check-data`` (``statistics-probability``, answer ``mcq``): before any graph is drawn, work out what to
collect to answer a question, how to ask it, and whether the data that came back can be trusted. Nothing is drawn.

* easy: given a question a group wants answered, choose what to record for each person or event. The wrong options
  record a different quantity, or record the right thing in a form that cannot answer the question (a yes/no where a
  count is needed, a count where a category is needed).
* medium: choose the survey question that collects exactly that data. The wrong options are the yes/no version, a vague
  version and a two-questions-in-one version.
* hard: a list of recorded entries is given with one that cannot be right (a count with a decimal part, a value in the
  wrong unit, a score above the maximum, or an entry from the wrong category). Choose the entry to check. The options
  are entries from the list, including its largest and smallest valid values.
* exam: one recorded list of counts with one impossible entry: which to check (mcq), what type of data it is (mcq), the
  total after the entry is corrected, and the mean after it is corrected (follow-through from the total).

Every wrong option is wrong by construction: it records a different thing, the right thing in the wrong form, or is an
entry that is valid on its face, and each carries a misconception id.
"""

from __future__ import annotations

from ..answers import Distractor
from ..core import attempts, check, generator, latex_answer, phrase, pick_distractors
from ._exd_x_w12a_kit import mcq_part, number_part
from ._w6a_kit import TooFewDistractors, retry_draw

TOPIC = "w8_data_acquisition"
EXAM_TOPIC = "exd_x_w8_data_acquisition"
TIERS = ("easy", "medium", "hard")
AREA = "acquire-and-check-data"

MISCONCEPTIONS = [
    (
        "dq-different-variable",
        "Chooses something that can be measured about the same people, but that cannot answer the question asked.",
        AREA,
    ),
    (
        "dq-count-for-category",
        "Records how many of something each person has where the question is about which kind they have.",
        AREA,
    ),
    ("dq-category-for-count", "Records which kind each person has where the question asks how many.", AREA),
    (
        "dq-yes-no-too-little",
        "Records only a yes or no, which loses the count or the category the question needs.",
        AREA,
    ),
    ("dq-vague-question", "Chooses a question that is too open to give data that can be sorted or counted.", AREA),
    (
        "dq-two-questions-in-one",
        "Chooses a question that asks two things at once, so one answer cannot be sorted into one group.",
        AREA,
    ),
    (
        "dq-flags-largest-valid",
        "Flags the largest entry just because it is the largest, although it is a possible value.",
        AREA,
    ),
    (
        "dq-flags-smallest-valid",
        "Flags the smallest entry just because it is the smallest, although it is a possible value.",
        AREA,
    ),
    (
        "dq-flags-repeated-entry",
        "Flags an entry that appears more than once, although a repeated value is not an error.",
        AREA,
    ),
    ("dq-flags-other-valid", "Flags an entry that is a possible value and fits the others.", AREA),
    ("dq-count-type-confused", "Calls a count of whole things continuous or categorical.", AREA),
    ("dq-total-keeps-error", "Adds up the entries without correcting the one that was wrong.", AREA),
    (
        "dq-total-drops-entry",
        "Leaves the wrong entry out of the total instead of replacing it with the corrected value.",
        AREA,
    ),
    ("dq-total-adds-error-twice", "Adds the corrected value to a total that still contains the wrong entry.", AREA),
    (
        "dq-mean-uses-wrong-count",
        "Divides the total by the number of different values, or by one fewer than the number of people.",
        AREA,
    ),
    ("dq-total-ignores-sign", "Adds the negative entry as if it were positive when finding the corrected total.", AREA),
    ("dq-arithmetic-slip", "Makes a small slip of one or two when adding or dividing.", AREA),
    ("dq-mean-uses-error-total", "Divides the uncorrected total by the number of people.", AREA),
]

# ---------------------------------------------------------------------------------------------------------------------
# the purposes: a question, what to record, a clear survey question and its flawed versions
# ---------------------------------------------------------------------------------------------------------------------
# (question being answered, record, [(wrong record, id)], survey question, yes/no version, vague version, two-in-one version)
_PURPOSES = (
    (
        "which pet is the most common among {g}",
        "The type of pet each person has",
        [
            ("The number of pets each person has", "dq-count-for-category"),
            ("The age of each person's pet", "dq-different-variable"),
            ("The amount each person spends on pet food", "dq-different-variable"),
            ("Whether each person has a pet", "dq-yes-no-too-little"),
        ],
        "What type of pet do you have?",
        "Do you have a dog?",
        "What do you think about pets?",
        "What type of pet do you have and how old is it?",
    ),
    (
        "how many brothers and sisters {g} have",
        "The number of brothers and sisters each person has",
        [
            ("The names of each person's brothers and sisters", "dq-different-variable"),
            ("The age of each person's oldest brother or sister", "dq-different-variable"),
            ("Whether each person has any brothers or sisters", "dq-yes-no-too-little"),
            ("The type of home each person lives in", "dq-different-variable"),
        ],
        "How many brothers and sisters do you have?",
        "Do you have a brother or a sister?",
        "Tell me about your family.",
        "How many brothers do you have and how many sisters are older than you?",
    ),
    (
        "how {g} travel to school",
        "The way each person travels to school",
        [
            ("How far each person lives from school", "dq-different-variable"),
            ("How long each person takes to get to school", "dq-different-variable"),
            ("The number of ways each person could travel to school", "dq-count-for-category"),
            ("Whether each person walks to school", "dq-yes-no-too-little"),
        ],
        "How do you usually travel to school?",
        "Do you walk to school?",
        "How is your trip to school?",
        "How do you travel to school and how long does it take?",
    ),
    (
        "how many goals a team scores in a game",
        "The number of goals scored in each game",
        [
            ("The name of the player who scored in each game", "dq-different-variable"),
            ("Whether the team won, drew or lost each game", "dq-yes-no-too-little"),
            ("The number of players on the field in each game", "dq-different-variable"),
            ("The ground each game was played on", "dq-category-for-count"),
        ],
        "How many goals did the team score in this game?",
        "Did the team score a goal in this game?",
        "How did the team play in this game?",
        "How many goals did the team score and who kicked the first one?",
    ),
    (
        "which lunch item {g} buy most often",
        "The lunch item each person buys most often",
        [
            ("How much each person spends at the canteen", "dq-different-variable"),
            ("The number of times each person buys lunch in a week", "dq-count-for-category"),
            ("The time each person lines up at the canteen", "dq-different-variable"),
            ("Whether each person buys lunch at the canteen", "dq-yes-no-too-little"),
        ],
        "What do you buy most often at the canteen?",
        "Do you buy a sandwich at the canteen?",
        "What do you think of the canteen?",
        "What do you buy at the canteen and how much does it cost?",
    ),
    (
        "how many books {g} read in a month",
        "The number of books each person reads in a month",
        [
            ("The favourite book of each person", "dq-different-variable"),
            ("The number of pages in the book each person is reading", "dq-different-variable"),
            ("Whether each person reads fiction or non-fiction", "dq-category-for-count"),
            ("Whether each person has read a book this month", "dq-yes-no-too-little"),
        ],
        "How many books did you read last month?",
        "Did you read a book last month?",
        "What do you like to read?",
        "How many books did you read last month and which one was best?",
    ),
    (
        "in which month most birthdays fall among {g}",
        "The birth month of each person",
        [
            ("The age of each person", "dq-different-variable"),
            ("The number of birthdays each person has had", "dq-count-for-category"),
            ("Whether each person has had a birthday this year", "dq-yes-no-too-little"),
            ("The day of the week each person was born on", "dq-different-variable"),
        ],
        "In which month is your birthday?",
        "Is your birthday in January?",
        "When were you born?",
        "In which month is your birthday and what age will you turn?",
    ),
    (
        "how many students are absent each day",
        "The number of students absent on each day",
        [
            ("The reason given for each absence", "dq-different-variable"),
            ("The total number of students enrolled at the school", "dq-different-variable"),
            ("Whether any student was absent on each day", "dq-yes-no-too-little"),
            ("The name of the teacher who took the roll each day", "dq-different-variable"),
        ],
        "How many students were absent today?",
        "Was anyone absent today?",
        "How was attendance today?",
        "How many students were absent today and why?",
    ),
    (
        "which colour of car is the most common",
        "The colour of each car that passes",
        [
            ("The number of cars that pass in each hour", "dq-count-for-category"),
            ("The speed of each car that passes", "dq-different-variable"),
            ("The number of people in each car that passes", "dq-different-variable"),
            ("Whether each car that passes is red", "dq-yes-no-too-little"),
        ],
        "What colour is this car?",
        "Is this car red?",
        "What does this car look like?",
        "What colour is this car and what make is it?",
    ),
    (
        "how many cars pass the school gate in each hour",
        "The number of cars that pass in each hour",
        [
            ("The colour of each car that passes", "dq-category-for-count"),
            ("The make of each car that passes", "dq-category-for-count"),
            ("The speed of each car that passes", "dq-different-variable"),
            ("Whether any car passes in each hour", "dq-yes-no-too-little"),
        ],
        "How many cars passed the gate in the last hour?",
        "Did a car pass the gate in the last hour?",
        "How busy is the road outside the gate?",
        "How many cars passed the gate in the last hour and how fast were they?",
    ),
    (
        "which fruit {g} prefer",
        "The fruit each person prefers",
        [
            ("The number of pieces of fruit each person eats in a week", "dq-count-for-category"),
            ("The mass of fruit each person eats in a day", "dq-different-variable"),
            ("The year level of each person", "dq-different-variable"),
            ("Whether each person likes apples", "dq-yes-no-too-little"),
        ],
        "Which fruit do you prefer?",
        "Do you like apples?",
        "How do you feel about fruit?",
        "Which fruit do you prefer and how often do you eat it?",
    ),
    (
        "how many pets {g} have",
        "The number of pets each person has",
        [
            ("The type of pet each person has", "dq-category-for-count"),
            ("The name of each person's pet", "dq-different-variable"),
            ("Whether each person has a pet", "dq-yes-no-too-little"),
            ("The amount each person spends on pet food", "dq-different-variable"),
        ],
        "How many pets do you have?",
        "Do you have a pet?",
        "Tell me about your pets.",
        "How many pets do you have and what are their names?",
    ),
)

_GROUPS = (
    "the students in the class",
    "the students in Year 4",
    "the students in Year 5",
    "the members of the school choir",
    "the students at the school",
    "the children at an after-school club",
)


def _tagged(rng, key: str, wrong, want: int = 3) -> list[Distractor]:
    """``want`` tagged wrong options in a random order, with the key left out."""
    pool = list(wrong)
    rng.shuffle(pool)
    out = pick_distractors(key, [Distractor(t, tag, t) for t, tag in pool], want)
    if len(out) < want:
        raise TooFewDistractors(f"{len(out)} options for {key}")
    return out


def _mcq(text, key: str, distractors, working) -> dict:
    return dict(
        text=text,
        answer=key,
        answer_text=key,
        answer_latex=latex_answer(key, plain=True),
        working=working,
        distractors=distractors,
    )


def _easy(rng):
    question, record, wrong, *_ = rng.choice(_PURPOSES)
    group = rng.choice(_GROUPS)
    text = phrase(
        rng,
        (
            "A group wants to find out {q}. What should they record?",
            "To find out {q}, what data should be collected?",
        ),
        q=question.format(g=group),
    )
    distractors = _tagged(rng, record, wrong)
    return _mcq(
        text,
        record,
        distractors,
        [
            f"The question is about {question.format(g=group)}.",
            f"So the data to collect is: {record[0].lower() + record[1:]}.",
        ],
    )


def _medium(rng):
    question, record, _wrong, ask, yes_no, vague, double = rng.choice(_PURPOSES)
    group = rng.choice(_GROUPS)
    text = phrase(
        rng,
        (
            "A group wants to find out {q}. Which survey question will collect the data they need?",
            "Which of these survey questions would give data to find out {q}?",
        ),
        q=question.format(g=group),
    )
    wrong = [(yes_no, "dq-yes-no-too-little"), (vague, "dq-vague-question"), (double, "dq-two-questions-in-one")]
    distractors = _tagged(rng, ask, wrong, 3)
    return _mcq(
        text,
        ask,
        distractors,
        [
            f"The data needed is: {record[0].lower() + record[1:]}.",
            "A good survey question asks for exactly one thing and gives answers that can be sorted or counted.",
        ],
    )


# -- hard: spot the entry to check ----------------------------------------------------------------------------------


def _fmt(x) -> str:
    return str(int(x)) if float(x) == int(x) else f"{x:g}"


_COUNTS = (
    ("the number of pets each of {n} students has", "pets", 7),
    ("the number of brothers and sisters each of {n} students has", "brothers and sisters", 7),
    ("the number of books each of {n} students read last month", "books", 12),
    ("the number of goals scored by each of {n} players this season", "goals", 10),
)
_CATEGORIES = (
    (
        "how each of {n} students travels to school",
        ("walk", "bike", "bus", "car", "train", "scooter"),
        ("Tuesday", "Blue", "Seven", "Pizza"),
    ),
    (
        "the favourite colour of each of {n} students",
        ("red", "blue", "green", "yellow", "purple", "orange"),
        ("Monday", "Soccer", "Twelve", "Bus"),
    ),
    (
        "the type of pet each of {n} students has",
        ("dog", "cat", "bird", "fish", "rabbit", "lizard"),
        ("Winter", "Red", "Nine", "Library"),
    ),
)


def _entry_list(rng, kind):
    """``(description, entries, key entry, valid entries, why)`` for one kind of impossible entry."""
    n = rng.randint(7, 10)
    if kind == "negative-count":
        what, unit, cap = rng.choice(_COUNTS)
        hi = rng.randint(4, min(cap, 6))
        valid = [rng.randint(0, hi) for _ in range(n - 1)]
        valid[0], valid[1] = 0, hi
        bad = -rng.randint(1, hi)
        why = f"A count of {unit} cannot be negative, so {_fmt(bad)} is not possible."
        return what.format(n=n), valid, bad, why
    if kind == "unit-slip":
        valid = [f"{rng.randint(125, 160) / 100:.2f}" for _ in range(n - 1)]
        bad = f"{rng.randint(125, 160) / 10:.2f}"
        why = f"{bad} metres is far taller than the rest, which are all about 1.5 metres: the decimal point is in the wrong place."
        return f"the height, in metres, of each of {n} students", valid, bad, why
    if kind == "over-maximum":
        top = rng.choice((10, 20, 25, 50))
        valid = [rng.randint(top // 2, top) for _ in range(n - 1)]
        valid[0] = top
        bad = top + rng.choice((5, 10, 15)) if top >= 20 else top + rng.randint(2, 5)
        why = f"{_fmt(bad)} is more than the highest possible mark of {top}."
        return f"the mark out of {top} gained by each of {n} students in a test", valid, bad, why
    what, cats, bads = rng.choice(_CATEGORIES)
    valid = [rng.choice(cats) for _ in range(n - 1)]
    bad = rng.choice(bads)
    why = f"{bad} does not answer the question that was asked."
    return what.format(n=n), valid, bad, why


def _hard(rng):
    kind = rng.choice(("negative-count", "unit-slip", "over-maximum", "wrong-category"))
    for _ in attempts(500, "an entry list"):
        what, valid, bad, why = _entry_list(rng, kind)
        distinct = list(dict.fromkeys(valid))
        if len(distinct) >= 4:
            break
    entries = valid[:]
    entries.insert(rng.randint(1, len(entries) - 1), bad)
    shown = ", ".join(_fmt(e) if not isinstance(e, str) else e for e in entries)
    key = _fmt(bad) if not isinstance(bad, str) else bad
    pool = []
    if kind != "wrong-category":
        show = lambda v: v if isinstance(v, str) else _fmt(v)  # noqa: E731
        pool += [
            (show(max(valid, key=float)), "dq-flags-largest-valid"),
            (show(min(valid, key=float)), "dq-flags-smallest-valid"),
        ]
    repeated = [v for v in dict.fromkeys(valid) if valid.count(v) > 1]
    if repeated:
        r = repeated[0]
        pool.append((r if isinstance(r, str) else _fmt(r), "dq-flags-repeated-entry"))
    pool += [(v if isinstance(v, str) else _fmt(v), "dq-flags-other-valid") for v in dict.fromkeys(valid)]
    distractors = _tagged(rng, key, pool)
    text = f"Students recorded {what}. The entries were: {shown}. One entry cannot be right and should be checked. Which one?"
    return _mcq(text, key, distractors, [why, "The other entries are all possible."])


_POOL = {"easy": _easy, "medium": _medium, "hard": _hard}


@generator(TOPIC, level=5, presets={t: {"tier": t} for t in TIERS})
def w8_data_acquisition(rng, tier="easy"):
    """What to record, how to ask it and which recorded entry to check; the tiers are described in the module docstring."""
    if tier not in _POOL:
        raise ValueError(f"unknown tier {tier!r}")
    return retry_draw(rng, _POOL[tier])


# -- exam -------------------------------------------------------------------------------------------------------------


@generator(EXAM_TOPIC, level=5, presets={"exam": {"tier": "exam"}})
def exd_x_w8_data_acquisition(rng, tier="exam"):
    """Exam chain on one list of counts with one impossible entry: the entry, the type of data, the corrected total and mean."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")

    def build(rng):
        what, unit, cap = rng.choice(_COUNTS)
        n = rng.randint(6, 10)
        hi = rng.randint(5, cap)
        for _ in attempts(4000, "counts with a whole-number mean"):
            valid = [rng.randint(0, hi) for _ in range(n - 1)]
            true_value = rng.randint(1, hi - 1)
            total = sum(valid) + true_value
            if total % n == 0 and total // n >= 1 and max(valid) == hi and len(set(valid)) >= 4:
                break
        bad = -rng.randint(1, hi)
        entries = valid[:]
        entries.insert(rng.randint(1, n - 1), bad)
        shown = ", ".join(_fmt(e) for e in entries)
        check(sum(entries) - bad + true_value == total, "total differs")
        mean = total // n
        pool = [(_fmt(max(valid)), "dq-flags-largest-valid"), (_fmt(min(valid)), "dq-flags-smallest-valid")]
        pool += [(_fmt(v), "dq-flags-other-valid") for v in dict.fromkeys(valid)]
        part_a = mcq_part(
            "Which entry should be checked?",
            _fmt(bad),
            1,
            [f"A count of {unit} cannot be negative, so {_fmt(bad)} is not possible."],
            pool,
        )
        part_b = mcq_part(
            "What type of data is this?",
            "Discrete numerical",
            1,
            ["It is a count of whole things, so it is numerical, and it takes separate values."],
            [
                ("Continuous numerical", "dq-count-type-confused"),
                ("Categorical", "dq-count-type-confused"),
                ("Ordinal categorical", "dq-count-type-confused"),
            ],
        )
        part_c = number_part(
            f"The entry that was wrong should have been {true_value}. What is the total number of {unit} after it is corrected?",
            total,
            2,
            [f"Replace {_fmt(bad)} by {true_value}: the total is {sum(valid)} + {true_value} = {total}."],
            [
                (sum(entries), "dq-total-keeps-error"),
                (sum(valid), "dq-total-drops-entry"),
                (sum(entries) + true_value, "dq-total-adds-error-twice"),
                (sum(valid) + abs(bad), "dq-total-ignores-sign"),
                (total + 1, "dq-arithmetic-slip"),
                (total - 1, "dq-arithmetic-slip"),
                (total + 2, "dq-arithmetic-slip"),
            ],
        )
        part_d = number_part(
            "What is the mean of the entries after the wrong one is corrected?",
            mean,
            2,
            [f"{total} divided by {n} is {mean}."],
            [
                (round(sum(entries) / n), "dq-mean-uses-error-total"),
                (round(total / (n - 1)), "dq-mean-uses-wrong-count"),
                (round(total / len(set(entries))), "dq-mean-uses-wrong-count"),
                (round(sum(valid) / n), "dq-mean-uses-error-total"),
                (mean + 1, "dq-arithmetic-slip"),
                (mean - 1, "dq-arithmetic-slip"),
                (mean + 2, "dq-arithmetic-slip"),
            ],
            follow=f"part_c/{n}",
        )
        if None in (part_a, part_b, part_c, part_d):
            raise TooFewDistractors("exam part")
        return dict(
            text=f"Students recorded {what.format(n=n)}. The entries were: {shown}.",
            parts=[part_a, part_b, part_c, part_d],
        )

    return retry_draw(rng, build)
