"""Two wider-pool generators for the early and middle primary years (task w2).

- ``halving_scene`` -- the same halving idea as ``fraction_halving`` (which calls it for ``scenes=True``) (halve a fraction once or twice, or
  undo one or two halvings), told about a pizza, a ribbon or a chocolate bar, with the starting piece
  shaded on a picture of the whole when there is a piece to show. The older topic had two or three
  questions per tier; this one has dozens.
- ``w2_median_even`` -- the median of an even-sized data set (the middle two values averaged), shown
  unordered, which the odd-sized ``calculate-median`` tier does not cover.

Both build the answer first and check it a second way.
"""

from __future__ import annotations

from fractions import Fraction

from .. import figures
from ..core import attempts, check, generator, pick_distractors
from ..figure_kinds.early import fraction_model, fraction_shape
from ..slots import NAMES

# (item, shape the picture uses)
_ITEMS = (
    ("pizza", "circle"),
    ("pie", "circle"),
    ("sandwich", "rectangle"),
    ("chocolate bar", "rectangle"),
    ("ribbon", "bar"),
    ("paper strip", "bar"),
    ("garden bed", "rectangle"),
    ("rope", "bar"),
)
_STEP_WORDS = {1: "once", 2: "twice"}
_PART_WORD = {2: "half", 4: "quarter"}


def _frac_text(f: Fraction) -> str:
    return str(f.numerator) if f.denominator == 1 else f"{f.numerator}/{f.denominator}"


def _picture(item, shape, parts):
    rows, cols = (1, parts) if parts == 2 else (2, parts // 2)
    spec = (
        fraction_shape(shape, parts, [0], rows=rows, cols=cols)
        if shape == "rectangle"
        else fraction_shape(shape, parts, [0])
    )
    return fraction_model(models=[spec], alt=f"a {item} cut into {parts} equal parts with one part shaded")


def _distractors(start, steps, answer, inverse):
    """The three slips the older topic offered: wrong direction, one step out, adding a half each time."""
    if inverse:
        wrong_direction = start * Fraction(1, 2) ** steps
        off_by_one = start * (2 ** max(0, steps - 1))
        adds = start + steps * Fraction(1, 2)
    else:
        wrong_direction = Fraction(1, max(1, start.denominator // (2**steps)))
        off_by_one = start * Fraction(1, 2) ** (steps + 1)
        adds = start + steps * Fraction(1, 2)
    return [_frac_text(c) for c in (wrong_direction, off_by_one, adds)]


def _forward(rng, steps, starts):
    name = rng.choice(NAMES)
    item, shape = rng.choice(_ITEMS)
    start = Fraction(1, rng.choice(starts))
    answer = start
    for _ in range(steps):
        answer *= Fraction(1, 2)
    check(answer == Fraction(1, start.denominator * 2**steps), "halving key wrong")  # second route
    cut = "in half" if steps == 1 else "in half, then in half again"
    smallest = "new piece" if steps == 1 else "smallest piece"
    fig = None
    if start == 1:
        text = f"{name} cuts a whole {item} {cut}. What fraction of the whole is each {smallest}?"
    else:
        text = (
            f"{name} cuts the shaded {_PART_WORD[start.denominator]} {cut}. "
            f"What fraction of the whole {item} is each {smallest}?"
        )
        fig = _picture(item, shape, start.denominator)
    answer_text = _frac_text(answer)
    out = dict(
        text=text,
        latex=f"plain:{text}",
        answer={"p": answer.numerator, "q": answer.denominator},
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[f"Halving {_frac_text(start)} {_STEP_WORDS[steps]} gives {answer_text}."],
        distractors=pick_distractors(answer_text, _distractors(start, steps, answer, False)),
    )
    if fig is not None:
        figures.assert_no_leak(fig, [answer_text])
        out["figure"] = fig
    return out


def _inverse(rng):
    name = rng.choice(NAMES)
    item, _shape = rng.choice(_ITEMS)
    steps = rng.choice((1, 2))
    target = Fraction(1, rng.choice((4, 8)))
    original = target * 2**steps
    simulated = original
    for _ in range(steps):
        simulated *= Fraction(1, 2)
    check(simulated == target, "inverse halving key wrong")  # second route
    cut = "in half" if steps == 1 else "in half, then in half again"
    what = "a" if original == 1 else "a piece of the"
    if original == 1:
        ask = f"What fraction of the whole {item} did {name} start with?"
    else:
        ask = "What fraction was the first piece?"
    text = f"{name} cuts {what} {item} {cut}. Each smallest piece is {_frac_text(target)} of the whole {item}. {ask}"
    answer_text = _frac_text(original)
    return dict(
        text=text,
        latex=f"plain:{text}",
        answer={"p": original.numerator, "q": original.denominator},
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[f"Undoing {_STEP_WORDS[steps]} halving from {_frac_text(target)} gives {answer_text}."],
        distractors=pick_distractors(answer_text, _distractors(target, steps, original, True)),
    )


def halving_scene(rng, variant="forward", n_steps=1):
    """Halve a fraction of a pizza, ribbon or bar once or twice, or undo the halving."""
    if variant == "inverse":
        return _inverse(rng)
    return _forward(rng, n_steps, (1, 2, 4) if n_steps == 1 else (1, 2))


@generator("w2_median_even", level=7)
def w2_median_even(rng, count=6, hi=30):
    """The median of an even number of values: the mean of the middle two, from data shown out of order."""
    n = count if count % 2 == 0 else count + 1
    for _ in attempts(500, "an even-sized set with different middle values"):
        values = sorted(rng.randint(1, hi) for _ in range(n))
        low, high = values[n // 2 - 1], values[n // 2]
        if low != high:
            break
    answer = Fraction(low + high, 2)
    # Second route: drop the largest and smallest values until two remain, then average.
    trimmed = values
    while len(trimmed) > 2:
        trimmed = trimmed[1:-1]
    check(Fraction(sum(trimmed), 2) == answer, "even median key wrong")
    shown = values[:]
    for _ in attempts(100, "an unordered display"):
        rng.shuffle(shown)
        if shown != values:
            break
    whole = answer.denominator == 1
    answer_text = str(int(answer)) if whole else f"{float(answer):.1f}"
    mean = Fraction(sum(values), n)
    candidates = [low, high, mean, answer + Fraction(1, 2), sum(values), answer - 1, answer + 1, answer + 2]
    # A distractor shows the same number of decimal places as the key: whole numbers beside a whole key,
    # one decimal place beside a half.
    clear = [
        c
        for c in candidates
        if abs(float(c) - float(answer)) > 0.06 and (Fraction(c).denominator == 1) == whole and c > 0
    ]
    texts = [str(int(c)) if whole else f"{float(c):.1f}" for c in clear]
    return dict(
        text=f"Find the median of: {', '.join(map(str, shown))}.",
        answer=float(answer) if answer.denominator != 1 else int(answer),
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[
            f"In order: {', '.join(map(str, values))}.",
            f"The middle two values are {low} and {high}; their mean is {answer_text}.",
        ],
        distractors=pick_distractors(answer_text, texts),
    )
