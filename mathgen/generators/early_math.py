"""Early-years maths generators: concrete, low-reading, tap-friendly questions.

Every topic returns four stable multiple-choice possibilities.  The skill
catalogue decides that these are graded as MCQ, so the same questions work in
any practice setting.
"""

from ..core import generator, latex_answer, pick_distractors


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


@generator("count_objects", level=1)
def count_objects(rng, lo=1, hi=5):
    count = rng.randint(lo, hi)
    symbol = rng.choice(DOTS)
    objects = "  ".join([symbol] * count)
    text = f"How many?   {objects}"
    return _number_question(
        text,
        count,
        _nearby(count, 0, max(10, hi + 2)),
        latex=f"plain:{text}",
        working=[f"Touch each shape once while counting: {count}."],
    )


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


@generator("clock_hours", level=1)
def clock_hours(rng):
    hours = [
        ("🕐", 1),
        ("🕑", 2),
        ("🕒", 3),
        ("🕓", 4),
        ("🕔", 5),
        ("🕕", 6),
        ("🕖", 7),
        ("🕗", 8),
        ("🕘", 9),
        ("🕙", 10),
        ("🕚", 11),
        ("🕛", 12),
    ]
    clock, answer = rng.choice(hours)
    text = f"What time is it?  {clock}"
    return _choice_question(
        text, f"{answer} o'clock", [f"{h} o'clock" for _c, h in hours if h != answer], latex=f"plain:{text}"
    )


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
