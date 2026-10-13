"""Generates `fr2b_mixed_numerals_number_line` questions for `f10-mixed-numerals-number-line` (AC9M4N04). `answer: fraction`.

Scope: a drawn number line with equal steps between the whole numbers. The learner reads a marked point as a mixed
numeral, or the distance between two marked points. The figure is the `number_line` kind with `tick = 1/n` and no
tick captions, and the whole numbers it needs are dotted and captioned, so the steps are visible and the fractions
are never printed on the line.

Ladder (reasoning escalation, not size):
  easy:   halves or quarters, every whole number captioned: the point is read from the whole number to its left.
  medium: thirds or fifths, only the two ends captioned: the learner counts the whole numbers across the line as
          well as the steps inside one.
  hard:   two points P and Q, the distance from P to Q as a mixed numeral, where the fraction at Q is smaller than
          the fraction at P, so the subtraction has to regroup.

The key is built from exact Fractions. A plausibility reader (`figures.register_reader`) re-derives it from the
figure alone: the marks and the tick step, never the key.
"""

from __future__ import annotations

from fractions import Fraction

from .. import figures
from ..answers import Distractor, exact_value
from ..core import attempts, check, generator, phrase
from ._fr2b_kit import build, mixed_text, tagged

_TIERS = ("easy", "medium", "hard")


def _mixed_dist(value: Fraction, misconception: str) -> Distractor:
    return Distractor(mixed_text(value), misconception, exact_value(value))


def _figure(first, last, n, points, labelled):
    """The number line from `first` to `last` with 1/n steps, whole numbers in `labelled` captioned and dotted, and
    each of `points` (label, value) dotted and captioned with its letter."""
    marks = [{"at": w, "label": str(w)} for w in labelled]
    marks += [{"at": float(v), "label": name} for name, v in points]
    names = " and ".join(name for name, _ in points)
    alt = (
        f"A number line from {first} to {last} with equal steps between the whole numbers. "
        f"{'Point' if len(points) == 1 else 'Points'} {names} {'is' if len(points) == 1 else 'are'} marked on it."
    )
    return figures.number_line(range_=(first, last), alt=alt, tick=1 / n, marks=marks, tick_labels=False)


def _read_figure(fig) -> list[tuple[str, Fraction]]:
    """(label, value) of each captioned non-whole mark, read from the figure: its tick step gives n."""
    n = round(1 / fig["tick"])
    out = []
    for m in fig["marks"]:
        label = m.get("label", "")
        if label.isalpha():
            value = Fraction(m["at"]).limit_denominator(12)
            check(value * n == int(value * n), "a mark is not on a tick")
            out.append((label, value))
    return out


def _position_candidates(whole: int, k: int, n: int, first: int):
    value = whole + Fraction(k, n)
    cands = [
        _mixed_dist(whole + Fraction(k, n + 1), "fr2b-counts-marks-not-steps"),
        _mixed_dist(whole + 1 + Fraction(k, n), "fr2b-reads-the-whole-number-after-the-point"),
        _mixed_dist(whole + Fraction(n - k, n), "fr2b-counts-steps-from-the-next-whole-number"),
    ]
    if whole > 0:
        cands.append(_mixed_dist(whole - 1 + Fraction(k, n), "fr2b-reads-the-whole-number-before-the-point"))
    if first > 0:
        cands.append(_mixed_dist(value - first, "fr2b-reads-from-the-left-end-as-zero"))
    return value, cands


def _single(rng, tier):
    for _ in attempts(500, "a point with enough distractors"):
        if tier == "easy":
            n = rng.choice((2, 4, 4))
            first = rng.randint(0, 6)
            labelled = list(range(first, first + 4))
        else:
            n = rng.choice((3, 5, 5))
            first = rng.randint(2, 18)
            labelled = [first, first + 3]
        last = first + 3
        whole = first + rng.randint(0, 2)
        k = rng.randint(1, n - 1)
        if whole + k / n < 1:
            continue
        value, cands = _position_candidates(whole, k, n, first)
        distinct = {c.text for c in cands} - {mixed_text(value)}
        if len(distinct) >= 3:
            break
    fig = _figure(first, last, n, [("P", value)], labelled)
    ((_name, read),) = _read_figure(fig)
    check(read == value, "the figure does not show the value")
    answer_text = mixed_text(value)
    templates = (
        [
            "Point P is marked on the number line. Write the number at P as a mixed numeral.",
            "What number is marked by P on the number line? Write it as a mixed numeral.",
            "A dot labelled P sits on the number line. Give its value as a mixed numeral.",
        ]
        if tier == "easy"
        else [
            "Only the two ends of the number line are labelled. Write the number at P as a mixed numeral.",
            "The number line is labelled at its two ends only. What number is marked by P? Write it as a mixed numeral.",
            "Just the ends of the number line carry labels. P is marked between them. What is the value of P, as a mixed numeral?",
        ]
    )
    text = phrase(rng, templates)
    working = [
        f"Between each pair of whole numbers there are {n} equal steps, so each step is 1/{n}.",
        f"P is {k} {'step' if k == 1 else 'steps'} past {whole}, so P is {answer_text}.",
    ]
    return _result(text, value, answer_text, tagged(answer_text, cands), working, fig)


def _pair_candidates(p: Fraction, q: Fraction):
    whole_p, frac_p = divmod(p, 1)
    whole_q, frac_q = divmod(q, 1)
    diff = q - p
    return [
        _mixed_dist((whole_q - whole_p) + abs(frac_q - frac_p), "fr2b-subtracts-parts-without-regrouping"),
        _mixed_dist(q, "fr2b-gives-the-position-not-the-distance"),
        _mixed_dist(Fraction(int(whole_q - whole_p)), "fr2b-ignores-the-fractions"),
        _mixed_dist(p + q, "fr2b-adds-the-two-readings"),
    ], diff


def _hard(rng):
    for _ in attempts(1000, "two points where the subtraction regroups"):
        n = rng.choice((2, 3, 4, 5))
        first = rng.randint(1, 15)
        last = first + 4
        kp, kq = rng.randint(1, n - 1), rng.randint(1, n - 1)
        if kq >= kp:
            continue
        wp = first + rng.randint(0, 1)
        wq = wp + rng.randint(1, 3)
        p, q = wp + Fraction(kp, n), wq + Fraction(kq, n)
        if q >= last:
            continue
        cands, diff = _pair_candidates(p, q)
        if diff.denominator == 1 or diff < 1:
            continue
        if len({c.text for c in cands} - {mixed_text(diff)}) >= 3:
            break
    fig = _figure(first, last, n, [("P", p), ("Q", q)], [first, last])
    read = dict(_read_figure(fig))
    check(read["Q"] - read["P"] == diff, "the figure does not show the distance")
    answer_text = mixed_text(diff)
    text = phrase(
        rng,
        [
            "Points P and Q are marked on the number line. How far is it from P to Q? Write the distance as a mixed numeral.",
            "P and Q are marked on the number line. Write the distance from P to Q as a mixed numeral.",
            "Two points, P and Q, sit on the number line. Find the length of the gap between them as a mixed numeral.",
        ],
    )
    working = [
        f"P is {mixed_text(p)} and Q is {mixed_text(q)}.",
        f"Q - P = {mixed_text(q)} - {mixed_text(p)} = {answer_text}.",
    ]
    return _result(text, diff, answer_text, tagged(answer_text, cands), working, fig)


def _result(text, value, answer_text, distractors, working, fig):
    return dict(
        text=text,
        latex="",
        figure=fig,
        answer_kind="fraction",
        answer=exact_value(value),
        answer_text=answer_text,
        answer_latex=answer_text,
        working=working,
        distractors=distractors,
    )


def _read_answer(question) -> str:
    points = dict(_read_figure(question.figure))
    value = points["Q"] - points["P"] if "Q" in points else points["P"]
    return mixed_text(value)


figures.register_reader("fr2b_mixed_numerals_number_line", _read_answer)


@generator("fr2b_mixed_numerals_number_line", level=4, presets={tier: dict(tier=tier) for tier in _TIERS})
def fr2b_mixed_numerals_number_line(rng, tier="easy"):
    """A mixed numeral read from a drawn number line; tiers in the module docstring."""
    check(tier in _TIERS, f"unknown tier {tier!r}")
    return build(_hard if tier == "hard" else (lambda r: _single(r, tier)), rng, tier)
