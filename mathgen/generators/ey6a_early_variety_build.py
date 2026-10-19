"""Extra early-years sources with leak-free stems and larger pools.

These sit beside the original early-years generators in ``early_math`` and ``p4a_compare_attributes_build``,
which are left as they are. Each topic here is a second (or a replacement) source for an existing skill:

- ``ey6a_recognise_numeral`` -- choose the numeral for a number word, or for a row of shapes. The stem never
  shows the glyph that is also among the options.
- ``ey6a_compare_length`` -- three or four lines, or three coloured ribbons: pick the longest or the shortest.
  Many lengths and wordings instead of two fixed pictures.
- ``ey6a_shape_names`` -- name a shape from a drawing in one of several glyph styles, from its description, or
  pick the drawing that matches a name.
- ``ey6a_repeating_pattern`` -- continue, or fill the gap in, a repeating pattern of five kinds of unit.
- ``ey6a_compare_attributes`` -- compare two objects or events by length, mass, capacity or duration. The stem
  describes what was observed (ends lined up, a balance tipping, water poured across, who finished first); the
  options state the comparison, so no option is already written in the stem.
- ``ey6a_picture_graph_one_to_one`` -- a picture graph where one symbol is one thing (key 1), small counts. Not yet
  wired into a skill: it is the source a Year 1 picture-graph skill will use.
"""

from __future__ import annotations

from mathgen import figures
from mathgen.core import attempts, check, generator, latex_answer, phrase, pick_distractors
from mathgen.generators._stat_display_common import leaks, noun, number_fields, pick_number_distractors
from mathgen.slots import NAMES

_WORDS = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten")


def _mcq(text, answer, candidates, *, working=None):
    """A multiple-choice question whose answer and options are strings."""
    answer = str(answer)
    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=answer,
        answer_text=answer,
        answer_latex=latex_answer(answer, plain=True),
        working=working or [],
        distractors=pick_distractors(answer, [str(c) for c in candidates]),
    )


# -- numerals --------------------------------------------------------------------------------------------------

_NUMERAL_WORD_STEMS = (
    "Which numeral says the word {w}?",
    "Tap the numeral for the number word {w}.",
    "{n} writes the number word {w}. Which numeral matches it?",
)
_NUMERAL_COUNT_STEMS = (
    "Count the shapes. Tap the numeral that tells how many.",
    "How many shapes are there? Tap the numeral.",
    "{n} counts the shapes. Which numeral shows how many there are?",
)


def _near_first(rng, target, lo, hi):
    """Other numerals in range: the neighbours first (the easy ones to confuse), then the rest in a random order."""
    near = [v for v in (target + 1, target - 1, target + 2, target - 2) if lo <= v <= hi]
    rest = [v for v in range(lo, hi + 1) if v != target and v not in near]
    rng.shuffle(rest)
    rng.shuffle(near)
    return near[:2] + rest + near[2:]


@generator("ey6a_recognise_numeral", level=1)
def ey6a_recognise_numeral(rng, lo=0, hi=10):
    target = rng.randint(lo, hi)
    others = _near_first(rng, target, lo, hi)
    if rng.random() < 0.5:
        text = phrase(rng, _NUMERAL_WORD_STEMS, w=_WORDS[target], n=rng.choice(NAMES))
        q = _mcq(text, target, others, working=[f"The word {_WORDS[target]} is written {target}."])
        q["answer"] = target
        return q
    if target == 0:
        target = max(lo, 1)
        others = _near_first(rng, target, lo, hi)
    dot = rng.choice(("●", "★", "■", "▲"))
    text = f"{phrase(rng, _NUMERAL_COUNT_STEMS, n=rng.choice(NAMES))}   {'  '.join([dot] * target)}"
    return _mcq(text, target, others, working=[f"Touch each shape once while counting: {target}."]) | {"answer": target}


# -- comparing lengths -----------------------------------------------------------------------------------------

_COLOURS = ("red", "blue", "green", "yellow", "purple", "orange")
_BAR = "━"
_LENGTH_STEMS = {
    "longest": ("Which line is the longest?", "Tap the longest line.", "Which one is the longest?"),
    "shortest": ("Which line is the shortest?", "Tap the shortest line.", "Which one is the shortest?"),
}


@generator("ey6a_compare_length", level=1)
def ey6a_compare_length(rng, lines=3):
    if lines not in (3, 4):
        raise ValueError("lines must be 3 or 4")
    lengths = rng.sample(range(2, 11), lines)
    labelled_by_colour = rng.random() < 0.5
    labels = rng.sample(_COLOURS, lines) if labelled_by_colour else list("ABCD"[:lines])
    ask = rng.choice(("longest", "shortest"))
    pick = max(lengths) if ask == "longest" else min(lengths)
    answer = labels[lengths.index(pick)]
    drawing = "   ".join(f"{label} {_BAR * length}" for label, length in zip(labels, lengths, strict=True))
    text = f"{phrase(rng, _LENGTH_STEMS[ask])}   {drawing}"
    extras = ["They are the same"] if lines == 3 else []
    check(len({*lengths}) == lines, "lengths must all differ")
    return _mcq(
        text,
        answer,
        [label for label in labels if label != answer] + extras,
        working=[f"Compare how far each line reaches: the {ask} line is {answer}."],
    )


# -- shapes ----------------------------------------------------------------------------------------------------

_SHAPE_GLYPHS = {
    "Circle": ("●", "○", "◯"),
    "Triangle": ("▲", "△"),
    "Square": ("■", "□", "◼", "◻"),
    "Rectangle": ("▭", "▬", "▮", "▯"),
}
_SHAPE_CLUES = {
    "Circle": ("It is round. It has no corners.", "It is round and it has no straight sides."),
    "Triangle": ("It has 3 straight sides and 3 corners.", "It has 3 corners."),
    "Square": (
        "It has 4 straight sides that are all the same length.",
        "It has 4 corners and 4 sides the same length.",
    ),
    "Rectangle": (
        "It has 4 corners. Two sides are long and two sides are short.",
        "It has 4 straight sides. Two are long and two are short.",
    ),
}
_SHAPE_NAMES = tuple(_SHAPE_GLYPHS)


@generator("ey6a_shape_names", level=1)
def ey6a_shape_names(rng):
    name = rng.choice(_SHAPE_NAMES)
    others = [n for n in _SHAPE_NAMES if n != name]
    mode = rng.choice(("glyph", "clue", "match"))
    if mode == "glyph":
        glyph = rng.choice(_SHAPE_GLYPHS[name])
        text = phrase(
            rng,
            ("What shape is this?   {g}", "Name this shape.   {g}", "{n} draws this shape.   {g}   What is it called?"),
            g=glyph,
            n=rng.choice(NAMES),
        )
        return _mcq(text, name, others)
    if mode == "clue":
        clue = rng.choice(_SHAPE_CLUES[name])
        text = phrase(
            rng,
            (
                "{c} What shape is it?",
                "Read the clue. {c} Which shape is it?",
                "{n} thinks of a shape. {c} What is it?",
            ),
            c=clue,
            n=rng.choice(NAMES),
        )
        return _mcq(text, name, others)
    answer = rng.choice(_SHAPE_GLYPHS[name])
    distract = [rng.choice(_SHAPE_GLYPHS[o]) for o in others]
    text = phrase(rng, ("Which one is a {s}?", "Tap the {s}.", "Find the {s}."), s=name.lower())
    return _mcq(text, answer, distract)


# -- repeating patterns ----------------------------------------------------------------------------------------

_ITEM_SETS = (
    ("🔵", "🟡", "🔴", "🟢"),
    ("▲", "■", "●", "★"),
    ("🍎", "🍐", "🍌", "🍇"),
    ("🐶", "🐱", "🐟", "🐦"),
    ("☀", "☁", "☂", "★"),
)
#: unit written as indices into three chosen items, all primitive (no unit is a repeat of a shorter one).
_UNIT_SHAPES = ((0, 1), (0, 0, 1), (0, 1, 1), (0, 1, 2), (0, 0, 1, 1))
_PATTERN_STEMS = ("What comes next?", "Which one comes next?", "Continue the pattern. What comes next?")
_GAP_STEMS = ("Which one is missing?", "What goes in the gap?", "Fill the gap. Which one is missing?")


@generator("ey6a_repeating_pattern", level=1)
def ey6a_repeating_pattern(rng, gap=None):
    items = list(rng.choice(_ITEM_SETS))
    rng.shuffle(items)
    shape = rng.choice(_UNIT_SHAPES)
    unit = [items[i] for i in shape]
    size = len(unit)
    use_gap = (rng.random() < 0.4) if gap is None else gap
    for _ in attempts(100, "a pattern with a clear next item"):
        shown_n = rng.randint(2 * size + 1, 2 * size + size)
        if shown_n <= 9:
            break
    sequence = [unit[i % size] for i in range(shown_n + 1)]
    if use_gap:
        hole = rng.randint(size, shown_n - 1)
        answer = sequence[hole]
        drawn = [("__" if i == hole else v) for i, v in enumerate(sequence[:shown_n])]
        text = f"{phrase(rng, _GAP_STEMS)}  {'  '.join(drawn)}"
    else:
        answer = sequence[shown_n]
        text = f"{phrase(rng, _PATTERN_STEMS)}  {'  '.join(sequence[:shown_n])}  __"
    check(answer == unit[(shown_n if not use_gap else hole) % size], "pattern continuation mismatch")
    return _mcq(
        text, answer, [i for i in items if i != answer], working=["Find the part that repeats, then carry it on."]
    )


# -- comparing measurement attributes ---------------------------------------------------------------------------

_ATTRS = ("length", "mass", "capacity", "duration")
_KINDS = {
    "length": ("ribbon", "stick", "rope", "strip of paper", "scarf"),
    "mass": ("box", "bag of sand", "ball of clay", "bucket", "backpack"),
    "capacity": ("cup", "jug", "bottle", "bowl", "tub"),
}
_OBSERVED = {
    "length": (
        "{n} lines up the {a} and the {b} at the start. The end of the {a} reaches past the end of the {b}.",
        "{n} puts the {a} and the {b} side by side with the left ends together. The {b} stops short of the {a}.",
    ),
    "mass": (
        "{n} puts the {a} on one side of a balance and the {b} on the other. The side with the {a} goes down.",
        "{n} holds the {a} in one hand and the {b} in the other. {n}'s hand with the {b} goes up.",
    ),
    "capacity": (
        "{n} fills the {a} with water and pours all of it into the {b}. The water spills over the top.",
        "{n} fills the {a} with water and pours all of it into the {b}. The {b} is only half full.",
    ),
    "duration": (
        "{a} and {b} start drawing a picture at the same time. {a} finishes when {b} is still drawing.",
        "{a} and {b} start a puzzle together. {b} puts in the last piece first.",
    ),
}
_OBSERVED_EQUAL = {
    "length": "{n} lines up the {a} and the {b} at the start. Their ends meet at exactly the same place.",
    "mass": "{n} puts the {a} on one side of a balance and the {b} on the other. The balance stays level.",
    "capacity": "{n} fills the {a} with water and pours all of it into the {b}. The water fills the {b} exactly to the top.",
    "duration": "{a} and {b} start drawing at the same time. They both finish at the same moment.",
}
_COMPARATIVE = {"length": "is longer", "mass": "is heavier", "capacity": "holds more", "duration": "took longer"}
_SAME = {
    "length": "They are the same length.",
    "mass": "They have the same mass.",
    "capacity": "They hold the same amount.",
    "duration": "They took the same time.",
}


#: For each observation wording, whether the first item named (``a``) is the greater one.
_FIRST_IS_GREATER = {
    ("length", 0): True,
    ("length", 1): True,
    ("mass", 0): True,
    ("mass", 1): True,
    ("capacity", 0): True,  # a's water spills over b, so a holds more
    ("capacity", 1): False,  # a's water only half fills b, so b holds more
    ("duration", 0): False,
    ("duration", 1): True,
}


@generator(
    "ey6a_compare_attributes",
    level=1,
    presets={
        "easy": dict(attr_pool=("length", "mass"), equal_fraction=0.0),
        "medium": dict(attr_pool=("capacity", "duration"), equal_fraction=0.0),
        "hard": dict(attr_pool=_ATTRS, equal_fraction=1 / 3),
    },
)
def ey6a_compare_attributes(rng, attr_pool=_ATTRS, equal_fraction=0.0):
    unknown = [a for a in attr_pool if a not in _ATTRS]
    if unknown or not attr_pool:
        raise ValueError(f"attr_pool must be a non-empty subset of {_ATTRS}")
    attr = rng.choice(attr_pool)
    name = rng.choice(NAMES)
    if attr == "duration":
        a, b = rng.sample([n for n in NAMES if n != name], 2)
    else:
        kind = rng.choice(_KINDS[attr])
        c1, c2 = rng.sample(_COLOURS, 2)
        a, b = f"{c1} {kind}", f"{c2} {kind}"

    def say(label):
        sentence = f"{label} {_COMPARATIVE[attr]}." if attr == "duration" else f"The {label} {_COMPARATIVE[attr]}."
        return sentence

    equal = equal_fraction > 0 and rng.random() < equal_fraction
    if equal:
        observed = _OBSERVED_EQUAL[attr].format(n=name, a=a, b=b)
        answer = _SAME[attr]
        wrong = [say(a), say(b)]
    else:
        variant = rng.randrange(2)
        observed = _OBSERVED[attr][variant].format(n=name, a=a, b=b)
        greater = a if _FIRST_IS_GREATER[(attr, variant)] else b
        lesser = b if greater == a else a
        answer = say(greater)
        wrong = [say(lesser), _SAME[attr]]
    return _mcq(
        f"{observed} What can we say?",
        answer,
        wrong,
        working=[f"From what happened: {answer}"],
    )


# -- one-to-one picture graph ----------------------------------------------------------------------------------

_GRAPH_CONTEXTS = (
    (
        "children",
        "child",
        "chose",
        "Each child in a class chose a favourite pet.",
        ("Dog", "Cat", "Bird", "Fish", "Rabbit"),
    ),
    (
        "children",
        "child",
        "chose",
        "Children voted for their favourite fruit.",
        ("Apple", "Banana", "Orange", "Grapes", "Pear"),
    ),
    (
        "children",
        "child",
        "chose",
        "Children voted for their favourite colour.",
        ("Red", "Blue", "Green", "Yellow", "Purple"),
    ),
    (
        "days",
        "day",
        "were",
        "A class wrote down the weather each day for two weeks.",
        ("Sunny", "Cloudy", "Rainy", "Windy", "Foggy"),
    ),
)
_GRAPH_SYMBOLS = ("circle", "square", "triangle", "star", "heart", "diamond")
_GRAPH_TOPIC = "ey6a_picture_graph_one_to_one"
_COUNT_ASK = (
    "How many {unit} {verb} {c}?",
    "Use the graph. How many {unit} {verb} {c}?",
    "Count the symbols in the {c} row. How many {unit}?",
)
_MORE_ASK = ("How many more {unit} {verb} {a} than {b}?", "Find how many more {unit} {verb} {a} than {b}.")
_TOTAL_ASK = ("How many {unit} are on the graph altogether?", "How many {unit} are there in total?")


@generator(_GRAPH_TOPIC, level=2)
def ey6a_picture_graph_one_to_one(rng, ask=None):
    ask = ask or rng.choice(("count", "count", "more", "total"))
    if ask not in ("count", "more", "total"):
        raise ValueError(f"unknown ask {ask!r}")
    unit, _one, verb, lead, cats = rng.choice(_GRAPH_CONTEXTS)
    rows = rng.randint(3, 5)
    chosen = rng.sample(cats, rows)
    for _ in attempts(100, "a one-to-one picture graph"):
        counts = [rng.randint(1, 8) for _ in chosen]
        if len(set(counts)) == rows:
            break
    fig = figures.check(
        {
            "kind": "picture_graph",
            "alt": "a picture graph",
            "symbol": rng.choice(_GRAPH_SYMBOLS),
            "key": {"each": 1, "unit": _one},
            "rows": [{"label": c, "symbols": n} for c, n in zip(chosen, counts, strict=True)],
        }
    )
    i = rng.randrange(rows)
    if ask == "count":
        value = counts[i]
        text = f"{lead} {phrase(rng, _COUNT_ASK, unit=unit, verb=verb, c=chosen[i])}"
        candidates = [counts[(i + 1) % rows], counts[(i - 1) % rows], value + 1, value - 1, sum(counts)]
        working = [f"The {chosen[i]} row has {value} {noun(value, 'symbol')} and each symbol is 1, so {value}."]
    elif ask == "more":
        order = sorted(range(rows), key=lambda r: counts[r])
        hi_i, lo_i = order[-1], order[0]
        value = counts[hi_i] - counts[lo_i]
        text = f"{lead} {phrase(rng, _MORE_ASK, unit=unit, verb=verb, a=chosen[hi_i], b=chosen[lo_i])}"
        candidates = [counts[hi_i] + counts[lo_i], counts[hi_i], counts[lo_i], value + 1, value - 1]
        working = [f"{counts[hi_i]} - {counts[lo_i]} = {value}."]
    else:
        value = sum(counts)
        text = f"{lead} {phrase(rng, _TOTAL_ASK, unit=unit)}"
        candidates = [value + 1, value - 1, max(counts), value + 2, value - 2]
        working = [f"Add the rows: {' + '.join(str(c) for c in counts)} = {value}."]
    answer, answer_text, answer_latex = number_fields(value)
    distractors = pick_number_distractors(_GRAPH_TOPIC, value, candidates)
    check(not leaks(fig, [answer_text, *distractors]), "answer shown in the figure labels")
    return dict(
        text=text,
        answer=answer,
        answer_text=answer_text,
        answer_latex=answer_latex,
        distractors=distractors,
        working=working,
        figure=fig,
    )


def _graph_read(question) -> str:
    fig, text = question.figure, question.text
    rows = {r["label"]: r["symbols"] for r in fig["rows"]}
    named = [label for label in rows if label in text]
    low = text.lower()
    if "altogether" in low or "in total" in low:
        return str(sum(rows.values()))
    if "more" in low and len(named) == 2:
        return str(abs(rows[named[0]] - rows[named[1]]))
    check(len(named) == 1, "the prompt must name exactly one row")
    return str(rows[named[0]])


figures.register_reader(_GRAPH_TOPIC, _graph_read)
