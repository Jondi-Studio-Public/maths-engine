"""Helpers shared by the generators that read a statistical display (picture graph, column graph, dot plot,
histogram, box plot, pie chart, line graph).

Not a generator module (the leading underscore keeps ``mathgen.generators`` from importing it as one).
"""

from __future__ import annotations

from fractions import Fraction

from mathgen.core import Question, check, rational_text


def number_fields(value) -> tuple:
    """``(answer, answer_text, answer_latex)`` for a whole or half-integer reading."""
    f = Fraction(value).limit_denominator(1000)
    if f.denominator == 1:
        return int(f), str(int(f)), str(int(f))
    text = rational_text(f) if f.denominator > 2 else f"{float(f):g}"
    if f.denominator == 2:
        text = f"{float(f):g}"
    return float(f), text, text


def number_text(value) -> str:
    return number_fields(value)[1]


def stub_question(topic: str, value, text: str = "") -> Question:
    answer, answer_text, answer_latex = number_fields(value)
    return Question(
        topic=topic,
        level=1,
        seed=0,
        text=text,
        latex=text,
        answer=answer,
        answer_text=answer_text,
        answer_latex=answer_latex,
    )


def graded_wrong(topic: str, key, candidate) -> bool:
    """True when the live ``number`` grader rejects ``candidate`` (typed as text) for an answer ``key``."""
    from mathgen.grading import grade

    q = stub_question(topic, key)
    return not grade("number", q, number_text(candidate)).correct


def pick_number_distractors(topic: str, key, candidates, n: int = 3) -> list[str]:
    """First ``n`` candidate values (in misconception priority order) that are non-negative, distinct from each
    other and from ``key``, and that the real grader marks wrong. ``candidates`` are numbers, no strings."""
    out: list[str] = []
    seen = {number_text(key)}
    half = (
        Fraction(key).limit_denominator(1000).denominator == 2
    )  # a distractor must look like the answer: decimal iff it is
    slips = [key + d for d in (1, -1, 2, -2, 3)]  # a miscount by a few: always available as a last resort
    for c in [*candidates, *slips]:
        if (
            c is None
            or c <= 0
            or Fraction(c).limit_denominator(1000).denominator not in (1, 2)
            or (Fraction(c).limit_denominator(1000).denominator == 2) != half
        ):
            continue  # only whole and half-step values read as plausible options
        s = number_text(c)
        if s in seen or not graded_wrong(topic, key, c):
            continue
        seen.add(s)
        out.append(s)
        if len(out) == n:
            break
    check(len(out) == n, f"{topic}: fewer than {n} distractors that grade wrong")
    return out


def verify_wrong_everywhere(topic: str, key, distractors) -> None:
    for d in distractors:
        check(graded_wrong(topic, key, Fraction(d)), f"{topic}: distractor {d} grades correct")


def mentioned_in_order(names, text: str) -> list[str]:
    """The ``names`` that occur in ``text`` as whole words, in the order they first appear."""
    import re

    found = []
    for name in names:
        m = re.search(rf"(?<![A-Za-z0-9]){re.escape(name)}(?![A-Za-z0-9])", text)
        if m:
            found.append((m.start(), name))
    return [n for _, n in sorted(found)]


def leaks(fig: dict, texts) -> bool:
    """True when an answer text appears in the figure's labels (see ``figures.assert_no_leak``)."""
    from mathgen import figures

    try:
        figures.assert_no_leak(fig, texts)
    except AssertionError:
        return True
    return False


def noun(count, singular: str, plural: str | None = None) -> str:
    """The noun to follow ``count``: the singular for exactly one ("1 slice"), else the plural ("2 slices")."""
    return singular if count == 1 else (plural or singular + "s")
