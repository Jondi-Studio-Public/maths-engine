"""Descriptions for the early figure kinds: money, the hundred chart, fraction models, counters and base-ten blocks.

Each describes what the picture draws, in reading order, and works nothing out: no coin total, no fraction
numeral, no missing chart number, no place-value total.
"""

from __future__ import annotations

from mathgen.describe import count, join, num, register_describer, sentence


_WORDS = "zero one two three four five six seven eight nine ten eleven twelve".split()


def _word(n: int, noun: str) -> str:
    """``Two shapes``-style opening: the count in words, capitalised."""
    text = count(n, noun)
    first, _, rest = text.partition(" ")
    return f"{_WORDS[n].capitalize()} {rest}" if n < len(_WORDS) else text


def _runs(tokens):
    """[(token, run length)] for consecutive equal tokens."""
    out: list[list] = []
    for t in tokens:
        if out and out[-1][0] == t:
            out[-1][1] += 1
        else:
            out.append([t, 1])
    return [(t, n) for t, n in out]


def _run_text(tokens) -> str:
    """Run-length phrasing of ``(singular, plural)`` tokens: ``2 solid circles, then 3 empty boxes``."""
    parts = [count(n, t[0], t[1]) for t, n in _runs(tokens)]
    return ", then ".join(parts)


# ---------------------------------------------------------------- money


def _describe_money(spec: dict) -> str:
    groups = spec["groups"]
    out = []
    if len(groups) == 1:
        out.append("A collection of Australian coins and notes, drawn left to right.")
    else:
        out.append(f"{_word(len(groups), 'collection')} of Australian coins and notes, each drawn in its own box.")
    for g in groups:
        items = [f"a {i['denom']} {i['type']}" for i in g["items"]]
        head = f"Collection {g['label']} contains" if g["label"] else "The collection contains"
        out.append(f"{head}, from left to right: {join(items)}.")
    out.append("Each coin is a round disc and each note a wider rectangle, with its value printed in the middle.")
    return " ".join(out)


# ---------------------------------------------------------------- hundred chart


def _describe_hundred_chart(spec: dict) -> str:
    start = spec["start"]
    hidden, shaded, marked = set(spec["hidden"]), set(spec["shaded"]), set(spec["marked"])
    last = start + 99
    out = [
        f"A number chart of 100 squares in 10 rows of 10, with the numbers running left to right and then "
        f"top to bottom, from {start} at the top left to {last} at the bottom right."
    ]
    notes = []
    if hidden:
        notes.append("Some squares are empty, with no numeral printed.")
    if shaded:
        notes.append("Shaded squares are filled with colour.")
    if marked:
        notes.append("Ringed squares have a heavy ring around them.")
    out.extend(notes)
    for r in range(10):
        cells = []
        for c in range(10):
            n = start + r * 10 + c
            text = "empty" if n in hidden else num(n)
            flags = [w for w, s in (("shaded", shaded), ("ringed", marked)) if n in s]
            if flags:
                text += f" ({' and '.join(flags)})"
            cells.append(text)
        out.append(f"Row {r + 1}: {', '.join(cells)}.")
    return " ".join(out)


# ---------------------------------------------------------------- fraction models


def _shade_runs(flags) -> str:
    return ", then ".join(f"{n} {'shaded' if t else 'empty'}" for t, n in _runs(flags))


def _describe_model(m: dict) -> str:
    shape, parts = m["shape"], m["parts"]
    shaded = set(m["shaded"])
    flags = [i in shaded for i in range(parts)]
    head = f"Shape {m['label']}: " if m["label"] else ""
    if shape == "bar":
        return (
            f"{head}{'a' if head else 'A'} horizontal bar cut into {parts} equal parts by vertical lines. "
            f"From the left: {_shade_runs(flags)}."
        )
    if shape == "circle":
        return (
            f"{head}{'a' if head else 'A'} circle cut into {parts} equal sectors by lines from the centre. "
            f"Starting at the top (12 o'clock) and going clockwise: {_shade_runs(flags)}."
        )
    rows, cols = m["rows"], m["cols"]
    lines = []
    for r in range(rows):
        lines.append(f"Row {r + 1}: {_shade_runs(flags[r * cols : (r + 1) * cols])}.")
    return f"{head}{'a' if head else 'A'} rectangle cut into {rows} rows of {cols} equal cells. " + " ".join(lines)


def _describe_fraction_model(spec: dict) -> str:
    models = spec["models"]
    n = len(models)
    if n == 1:
        intro = "A shape cut into equal parts, with some parts shaded. Shaded parts are filled with a hatch pattern and the others are plain."
    else:
        intro = (
            f"{_word(n, 'shape')}, each the same size and cut into equal parts, "
            f"with some parts shaded. Shaded parts are filled with a hatch pattern and the others are plain. No fraction is written."
        )
    return " ".join([intro, *(_describe_model(m) for m in models)])


# ---------------------------------------------------------------- counters

_COLOUR_WORD = {1: "solid", 2: "ringed"}
_PLURAL = {"star": "stars"}


def _counter_token(shape: str, colour: int):
    word = f"{_COLOUR_WORD[colour]} {shape}"
    return word, word + "s"


_EMPTY_BOX = ("empty box", "empty boxes")

_DIE_PATTERN = {
    1: "one dot in the centre",
    2: "two dots, at the top left and the bottom right",
    3: "three dots along the diagonal from the top left to the bottom right",
    4: "four dots, one in each corner",
    5: "five dots, one in each corner and one in the centre",
    6: "six dots in two columns of three",
}


def _kind_note(shape: str, colours) -> str:
    notes = []
    if 1 in colours:
        notes.append(f"a solid {shape} is filled in")
    if 2 in colours:
        notes.append(f"a ringed {shape} is hollow with a thick outline")
    return sentence(join(notes)) if notes else ""


def _describe_ten_frame(spec: dict) -> str:
    shape, frames = spec["shape"], spec["frames"]
    used = {c for f in frames for c in f if c}
    out = [
        "A ten frame is a box of 2 rows of 5 boxes. One ten frame is drawn."
        if len(frames) == 1
        else "Two ten frames are drawn side by side, each a box of 2 rows of 5 boxes."
    ]
    note = _kind_note(shape, used)
    if note:
        out.append(note)
    for fi, f in enumerate(frames):
        who = "" if len(frames) == 1 else ("Left frame, " if fi == 0 else "Right frame, ")
        for ri, name in enumerate(("top", "bottom")):
            row = f[ri * 5 : ri * 5 + 5]
            toks = [_EMPTY_BOX if c == 0 else _counter_token(shape, c) for c in row]
            lead = f"{who}{name} row, left to right" if who else f"{name.capitalize()} row, left to right"
            out.append(f"{lead}: {_run_text(toks)}.")
    return " ".join(out)


def _describe_array(spec: dict) -> str:
    shape, rows, cols = spec["shape"], spec["rows"], spec["cols"]
    colours = spec.get("colours")
    if not colours:
        return (
            f"An array of counters in {count(rows, 'row')} and {count(cols, 'column')}, every position holding "
            f"a counter. Each row has {count(cols, 'solid ' + shape)}."
        )
    used = {v for r in colours for v in r}
    out = [
        f"An array of counters in {count(rows, 'row')} and {count(cols, 'column')}, every position holding a counter."
    ]
    note = _kind_note(shape, used)
    if note:
        out.append(note)
    for ri, r in enumerate(colours):
        out.append(f"Row {ri + 1}, left to right: {_run_text([_counter_token(shape, v) for v in r])}.")
    return " ".join(out)


def _describe_groups(spec: dict) -> str:
    shape, groups = spec["shape"], spec["groups"]
    n = len(groups)
    per_row = n if n <= 3 else 2 if n == 4 else 3
    wrap = f", {per_row} to a row" if n > per_row else ""
    out = [f"{_word(n, 'ring')} drawn left to right{wrap}, each ring holding some solid {shape}s."]
    k = 1
    for g, run in _runs(groups):
        span = f"Ring {k}" if run == 1 else f"Rings {k} to {k + run - 1}"
        verb = "is" if run == 1 else "are each"
        what = "empty" if g == 0 else f"holding {count(g, 'solid ' + shape)}"
        out.append(f"{span} {verb} {what}.")
        k += run
    return " ".join(out)


def _describe_dice(spec: dict) -> str:
    faces = spec["faces"]
    out = [
        f"{_word(len(faces), 'die face')} drawn side by side, each a rounded square with round dots."
        if len(faces) > 1
        else "One die face, a rounded square with round dots."
    ]
    used = {f["colour"] for f in faces}
    if used == {1, 2}:
        out.append("Solid dots are filled in and ringed dots are hollow with a thick outline.")
    for i, f in enumerate(faces):
        who = f"Die {i + 1}" if len(faces) > 1 else "The die"
        kind = "ringed" if f["colour"] == 2 else "solid"
        out.append(f"{who} has {kind} dots: {_DIE_PATTERN[f['n']]}.")
    return " ".join(out)


def _describe_scatter(spec: dict) -> str:
    shape, dots = spec["shape"], spec["dots"]
    out = [
        f"Counters scattered in a box. Each position is given as (across, down): percent of the way across from "
        f"the left edge and down from the top edge. They are listed from top "
        "to bottom and then left to right."
    ]
    order = sorted(dots, key=lambda d: (round(d["at"][1] * 100), round(d["at"][0] * 100)))
    for colour in (1, 2):
        pts = [d for d in order if d["colour"] == colour]
        if pts:
            where = join([f"({num(round(d['at'][0] * 100))}, {num(round(d['at'][1] * 100))})" for d in pts])
            out.append(f"{_COLOUR_WORD[colour].capitalize()} {shape}s at {where}.")
    return " ".join(out)


def _describe_counters(spec: dict) -> str:
    return {
        "ten_frame": _describe_ten_frame,
        "array": _describe_array,
        "groups": _describe_groups,
        "dice": _describe_dice,
        "scatter": _describe_scatter,
    }[spec["layout"]](spec)


# ---------------------------------------------------------------- base ten


def _describe_base_ten(spec: dict) -> str:
    h, t, o = spec["hundreds"], spec["tens"], spec["ones"]
    out = [
        "Base-ten blocks drawn left to right: flats, then rods, then loose cubes. A flat is a large square made "
        "of 10 rows of 10 small squares, a rod is a column of 10 small squares and a cube is one small square."
    ]
    out.append(f"{count(h, 'flat')}." if h else "There are no flats.")
    out.append(f"{count(t, 'rod')}." if t else "There are no rods.")
    out.append(f"{count(o, 'loose cube')}." if o else "There are no loose cubes.")
    return " ".join(out)


register_describer("money", _describe_money)
register_describer("hundred_chart", _describe_hundred_chart)
register_describer("fraction_model", _describe_fraction_model)
register_describer("counters", _describe_counters)
register_describer("base_ten", _describe_base_ten)
