"""Early-years maths generators: concrete, low-reading, tap-friendly questions.

Every topic returns four stable multiple-choice possibilities.  The skill
catalogue decides that these are graded as MCQ, so the same questions work in
any practice setting.
"""

from .. import figures
from ..core import generator, latex_answer, pick_distractors
from ..figure_kinds.early import counters, counters_total, scatter_dots
from ..slots import NAMES
from ..figure_kinds import measure


DOTS = ("●", "◆", "★", "■")


def _number_question(text, answer, candidates, *, latex=None, working=None):
    answer_text = str(answer)
    return dict(
        text=text,
        latex=latex or rf"\text{{{text}}}",
        answer=answer,
        answer_text=answer_text,
        answer_latex=answer_text,
        working=working or [],
        distractors=pick_distractors(answer_text, candidates),
    )


def _choice_question(text, answer, candidates, *, latex=None, working=None):
    return dict(
        text=text,
        latex=latex or rf"\text{{{text}}}",
        answer=answer,
        answer_text=answer,
        answer_latex=latex_answer(answer, plain=True),
        working=working or [],
        distractors=pick_distractors(answer, candidates),
    )


def _nearby(n, lo=0, hi=100):
    values = [n - 1, n + 1, n + 2, n - 2, n + 5]
    return [value for value in values if lo <= value <= hi]


_SHAPE_OF = {
    "●": ("circle", "circles"),
    "◆": ("diamond", "diamonds"),
    "★": ("star", "stars"),
    "■": ("square", "squares"),
}
_COUNT_TEXT = (
    "How many {w} are there?",
    "{n} counts the {w}. How many are there?",
    "Count the {w}. How many?",
)


@generator("count_objects", level=1)
def count_objects(rng, lo=1, hi=5):
    count = rng.randint(lo, hi)
    shape, plural = _SHAPE_OF[rng.choice(DOTS)]
    fig = counters(
        layout="scatter",
        alt="a group of shapes to count",
        shape=shape,
        dots=scatter_dots(rng, count, arrangement="tidy"),
    )
    figures.assert_no_leak(fig, [str(count)])
    text = rng.choice(_COUNT_TEXT).format(n=rng.choice(NAMES), w=plural)
    q = _number_question(
        text,
        count,
        _nearby(count, 0, max(10, hi + 2)),
        latex=f"plain:{text}",
        working=[f"Touch each shape once while counting: {count}."],
    )
    q["figure"] = fig
    return q


figures.register_reader("count_objects", lambda q: str(counters_total(q.figure)))


@generator("recognise_numeral", level=1)
def recognise_numeral(rng, lo=0, hi=10):
    target = rng.randint(lo, hi)
    return _number_question(
        f"Tap the number {target}",
        target,
        _nearby(target, lo, hi),
        latex=rf"\text{{Tap the number }} {target}",
    )


@generator("compare_quantities", level=1)
def compare_quantities(rng, hi=10):
    left = rng.randint(1, hi)
    right = rng.randint(1, hi)
    if rng.random() < 0.2:
        right = left
    answer = "Same" if left == right else ("Left" if left > right else "Right")
    text = f"Which side has more?   {'● ' * left}  |  {'● ' * right}"
    return _choice_question(text, answer, ["Left", "Right", "Same"], latex=f"plain:{text}")


@generator("number_sequence", level=1)
def number_sequence(rng, hi=10):
    start = rng.randint(0, hi - 3)
    answer = start + 3
    return _number_question(
        f"What comes next?  {start}, {start + 1}, {start + 2}, __",
        answer,
        _nearby(answer, 0, hi + 2),
        latex=rf"{start},\ {start + 1},\ {start + 2},\ \square",
    )


@generator("before_after", level=1)
def before_after(rng, hi=10):
    number = rng.randint(1, hi - 1)
    ask_after = rng.choice([True, False])
    answer = number + 1 if ask_after else number - 1
    word = "after" if ask_after else "before"
    return _number_question(
        f"Which number comes {word} {number}?", answer, _nearby(answer, 0, hi), latex=rf"\text{{{word} }} {number}"
    )


@generator("number_bond", level=1)
def number_bond(rng, total=5):
    part = rng.randint(0, total)
    missing = total - part
    return _number_question(
        f"{part} and what make {total}?",
        missing,
        _nearby(missing, 0, total),
        latex=rf"{part} + \square = {total}",
        working=[f"Start at {part} and count up to {total}: {missing} more."],
    )


@generator("early_addition", level=1)
def early_addition(rng, total_hi=10):
    a = rng.randint(0, total_hi)
    b = rng.randint(0, total_hi - a)
    answer = a + b
    return _number_question(
        f"{a} + {b} = ?",
        answer,
        _nearby(answer, 0, total_hi),
        latex=rf"{a} + {b} = \square",
        working=[f"Put the groups together: {a} and {b} make {answer}."],
    )


@generator("early_subtraction", level=1)
def early_subtraction(rng, total_hi=10):
    a = rng.randint(0, total_hi)
    b = rng.randint(0, a)
    answer = a - b
    return _number_question(
        f"{a} - {b} = ?",
        answer,
        _nearby(answer, 0, total_hi),
        latex=rf"{a} - {b} = \square",
        working=[f"Start with {a}, take away {b}, and {answer} remain."],
    )


@generator("teen_place_value", level=1)
def teen_place_value(rng, hi=99):
    number = rng.randint(10, hi)
    tens, ones = divmod(number, 10)
    if rng.choice([True, False]):
        answer = tens
        text = f"How many tens are in {number}?"
        cands = [ones, tens + 1, max(0, tens - 1), number]
    else:
        answer = ones
        text = f"How many ones are in {number}?"
        cands = [tens, (ones + 1) % 10, max(0, ones - 1), number]
    return _number_question(text, answer, cands, latex=rf"{number}= {tens}\text{{ tens}}+{ones}\text{{ ones}}")


@generator("shape_names", level=1)
def shape_names(rng):
    shapes = [("●", "Circle"), ("▲", "Triangle"), ("■", "Square"), ("▭", "Rectangle")]
    symbol, answer = rng.choice(shapes)
    text = f"What shape is this?   {symbol}"
    return _choice_question(text, answer, [name for _symbol, name in shapes], latex=f"plain:{text}")


@generator("repeating_pattern", level=1)
def repeating_pattern(rng):
    patterns = [
        (["🔵", "🟡"], "🔵"),
        (["▲", "■"], "▲"),
        (["★", "●", "●"], "★"),
        (["🍎", "🍐"], "🍎"),
    ]
    unit, answer = rng.choice(patterns)
    # Complete repeats mean the next item is the first item in the unit.
    shown = unit * (3 if len(unit) == 2 else 2)
    text = f"What comes next?  {'  '.join(shown)}  __"
    return _choice_question(text, answer, ["🔵", "🟡", "▲", "■", "★", "●", "🍎", "🍐"], latex=f"plain:{text}")


@generator("compare_length", level=1)
def compare_length(rng):
    long_side = rng.choice(["Left", "Right"])
    left = "━━━━━━" if long_side == "Left" else "━━━"
    right = "━━━━━━" if long_side == "Right" else "━━━"
    text = f"Which line is longer?  {left}  |  {right}"
    return _choice_question(text, long_side, ["Left", "Right", "Same"], latex=f"plain:{text}")


_CLOCK_OPENERS = (
    "Here is a clock.",
    "Look at the clock on the wall.",
    "This clock is in the classroom.",
    "A clock hangs in the kitchen.",
    "Look at the clock by the door.",
    "Ben looks at the clock in his room.",
    "The clock at the bus stop is shown.",
)
_CLOCK_QUESTIONS = (
    "What time is it?",
    "What time does the clock show?",
    "Which time is on the clock?",
    "Choose the time the clock shows.",
)


@generator("clock_hours", level=1)
def clock_hours(rng):
    answer = rng.randint(1, 12)
    fig = measure.clock(hour=answer, minute=0, alt="an analogue clock", numerals="all", minute_ticks=False)
    text = f"{rng.choice(_CLOCK_OPENERS)} {rng.choice(_CLOCK_QUESTIONS)}"

    def hour(h):
        return f"{(h - 1) % 12 + 1} o'clock"

    # Named misreadings: the hour numeral either side of the one the short hand points at (always offered), then one
    # of: the 12 the long hand points at (reading the wrong hand), or the numeral two or three away (a slip in
    # counting round the face). Which one varies, so no single option recurs.
    rest = [hour(12), hour(answer + 2), hour(answer - 2), hour(answer + 3)]
    rng.shuffle(rest)
    distractors = pick_distractors(hour(answer), [hour(answer + 1), hour(answer - 1), *rest])
    figures.assert_no_leak(fig, [hour(answer), *distractors])
    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=hour(answer),
        answer_text=hour(answer),
        answer_latex=latex_answer(hour(answer), plain=True),
        working=["The long hand points at the 12, so it is an o'clock time; the short hand points at the hour."],
        distractors=distractors,
        figure=fig,
    )


def _read_clock_hours(question) -> str:
    hour, minute = measure.read_clock(question.figure)
    if minute != 0:
        raise ValueError("an o'clock clock must have the long hand at the 12")
    return f"{hour} o'clock"


figures.register_reader("clock_hours", _read_clock_hours)


@generator("australian_coins", level=1)
def australian_coins(rng):
    values = ["5 cents", "10 cents", "20 cents", "50 cents", "1 dollar", "2 dollars"]
    answer = rng.choice(values)
    label = answer.replace(" cents", "c").replace("1 dollar", "$1").replace("2 dollars", "$2")
    text = f"What is this Australian coin worth?  {label}"
    return _choice_question(text, answer, values, latex=f"plain:{text}")


@generator("skip_count", level=1)
def skip_count(rng, steps=(2, 5, 10), hi=100):
    step = rng.choice(list(steps))
    start = rng.randint(0, max(0, (hi // step) - 4)) * step
    answer = start + 3 * step
    return _number_question(
        f"What comes next?  {start}, {start + step}, {start + 2 * step}, __",
        answer,
        [answer - step, answer + step, answer + 1, answer - 1],
        latex=rf"{start},\ {start + step},\ {start + 2 * step},\ \square",
    )


@generator("equal_groups", level=2)
def equal_groups(rng, groups_hi=5, each_hi=5):
    groups = rng.randint(2, groups_hi)
    each = rng.randint(1, each_hi)
    answer = groups * each
    return _number_question(
        f"There are {groups} groups with {each} in each group. How many altogether?",
        answer,
        [groups + each, answer - each, answer + each, answer - 1],
        latex=rf"{groups}\times {each}=\square",
        working=[f"Count {each}, {groups} times: {answer}."],
    )


@generator("fair_sharing", level=2)
def fair_sharing(rng, groups_hi=5, each_hi=5):
    groups = rng.randint(2, groups_hi)
    each = rng.randint(1, each_hi)
    total = groups * each
    return _number_question(
        f"Share {total} things equally between {groups} children. How many each?",
        each,
        [groups, total, each + 1, max(0, each - 1)],
        latex=rf"{total}\div {groups}=\square",
        working=[f"Make {groups} equal groups. Each group gets {each}."],
    )
