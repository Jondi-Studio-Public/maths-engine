"""Shared pieces for the statistical-investigation step skills (`ir2a_*` and `exd_x_ir2a_*`).

Not a generator module (the leading underscore keeps `mathgen.generators` from importing it as one). It builds the
answer parts these skills mix: a multiple-choice part with at least four options, an ordered list of lettered
steps, an unordered list of numbered statements, and the number and fraction parts of the shared exam-chain
support module. Every part is returned as a plain part dict whose wrong answers are tagged with misconception ids
and have been run through the real grader, so none of them can be accepted by accident.
"""

from __future__ import annotations

from fractions import Fraction
from types import SimpleNamespace

from mathgen.answers import Distractor
from mathgen.core import check
from mathgen.grading import typed

from .exd_x_r2j_support import fraction_part, number_part, shown, text_of  # noqa: F401  (re-exported)

NAMES = (
    "Priya",
    "Marcus",
    "Aiko",
    "Tom",
    "Leila",
    "Sam",
    "Jess",
    "Kofi",
    "Mia",
    "Dev",
    "Hannah",
    "Liam",
    "Zara",
    "Noah",
    "Isla",
    "Ravi",
)

LETTERS = "ABCDEFGH"


def tidy(text: str) -> str:
    """Collapse runs of spaces left by optional pieces of a template, and start the sentence with a capital."""
    text = " ".join(text.split())
    return text[:1].upper() + text[1:]


def mcq_part(
    prompt: str,
    key: str,
    wrongs: list[tuple[str, str]],
    *,
    marks: int = 1,
    working: list[str] | None = None,
    keep: int = 4,
    bare: bool = False,
) -> dict:
    """A multiple-choice part. `bare` drops a closing full stop from every option, for a part that is not last. `wrongs` lists `(option text, catalogue id)` in priority order; the part keeps up
    to `keep` of them, so it offers `keep + 1` options at most and never fewer than four."""
    if bare:
        key = key.rstrip(".")
        wrongs = [(t.rstrip("."), g) for t, g in wrongs]
    seen = {key}
    options = []
    for text, tag in wrongs:
        if text in seen:
            continue
        seen.add(text)
        options.append(Distractor(text, tag))
        if len(options) == keep:
            break
    check(len(options) >= 3, f"fewer than three wrong options for {prompt!r}")
    return dict(
        prompt=prompt,
        answer_kind="mcq",
        answer=key,
        answer_text=key,
        answer_latex="plain:" + key,
        marks=marks,
        working=list(working or []),
        distractors=options,
    )


def _reference(kind: str, prompt: str, answer, text: str):
    return SimpleNamespace(
        answer=answer, answer_text=text, text=prompt, answer_kind=kind, answer_alt=None, answer_form=None
    )


def ordered_part(
    prompt: str,
    key: list[str],
    wrongs: list[tuple[list[str], str]],
    *,
    marks: int = 1,
    working: list[str] | None = None,
    keep: int = 3,
) -> dict:
    """An ordered-list part (positional comparison). `key` and each wrong order are lists of short strings."""
    text = ", ".join(key)
    ref = _reference("ordered_list", prompt, list(key), text)
    options, seen = [], {text}
    for order, tag in wrongs:
        shown_text = ", ".join(order)
        if shown_text in seen or typed.grade("ordered_list", ref, shown_text).correct:
            continue
        seen.add(shown_text)
        options.append(Distractor(shown_text, tag))
        if len(options) == keep:
            break
    check(len(options) >= 2, f"fewer than two wrong orders for {prompt!r}")
    return dict(
        prompt=prompt,
        answer_kind="ordered_list",
        answer=list(key),
        answer_text=text,
        answer_latex="plain:" + text,
        marks=marks,
        working=list(working or []),
        distractors=options,
    )


def list_part(
    prompt: str,
    key: list[int],
    wrongs: list[tuple[list[int], str]],
    *,
    marks: int = 1,
    working: list[str] | None = None,
    keep: int = 3,
) -> dict:
    """An unordered list of whole numbers (the numbers of the statements that apply)."""
    key = sorted(key)
    text = ", ".join(map(str, key))
    ref = _reference("list", prompt, key, text)
    options, seen = [], {text}
    for values, tag in wrongs:
        shown_text = ", ".join(map(str, sorted(values)))
        if not values or shown_text in seen or typed.grade("list", ref, shown_text).correct:
            continue
        seen.add(shown_text)
        options.append(Distractor(shown_text, tag))
        if len(options) == keep:
            break
    check(len(options) >= 2, f"fewer than two wrong lists for {prompt!r}")
    return dict(
        prompt=prompt,
        answer_kind="list",
        answer=key,
        answer_text=text,
        answer_latex="plain:" + text,
        marks=marks,
        working=list(working or []),
        distractors=options,
    )


def single(stem: str, part: dict) -> dict:
    """A one-answer question from `part`: the stem followed by the part's prompt, and the part's answer fields."""
    out = {k: v for k, v in part.items() if k not in ("prompt", "marks", "follow_through", "sensible_range")}
    out["text"] = tidy(f"{stem} {part['prompt']}")
    return out


def chain(stem: str, *parts: dict) -> dict:
    """A multi-part question: the stem and its parts."""
    return dict(text=tidy(stem), parts=list(parts))


def frac_str(value) -> str:
    value = Fraction(value)
    return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"


def percent_text(value, places: int = 1) -> str:
    return text_of(value, places)


def join_and(items: list[str]) -> str:
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def lettered(rng, items: list[str]) -> tuple[list[str], dict[str, str]]:
    """Shuffle `items` and letter them A, B, C...: returns (shown lines, {item: letter})."""
    order = list(items)
    rng.shuffle(order)
    letter_of = {item: LETTERS[i] for i, item in enumerate(order)}
    lines = [f"{LETTERS[i]}: {item}" for i, item in enumerate(order)]
    return lines, letter_of


def median_of(values: list) -> Fraction:
    xs = sorted(values)
    n = len(xs)
    mid = n // 2
    return Fraction(xs[mid]) if n % 2 else (Fraction(xs[mid - 1]) + Fraction(xs[mid])) / 2


def quartiles(values: list) -> tuple[Fraction, Fraction, Fraction]:
    """(Q1, median, Q3): the medians of the lower and upper halves, the middle value left out of both when n is odd."""
    xs = sorted(values)
    n = len(xs)
    lower, upper = xs[: n // 2], xs[(n + 1) // 2 :]
    return median_of(lower), median_of(xs), median_of(upper)
