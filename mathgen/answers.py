"""The typed answer contract: what kind of thing a question's answer is, and what each wrong option stands for.

A question's answer has always been four loose attributes (`answer`, `answer_text`, `answer_latex`, `answer_form`)
whose kind was known only to the skill that served it, and its distractors were bare strings. This module names the
contract those attributes already follow and adds the two things they could not say:

* `answer_kind`: the grader kind of the value (`number`, `fraction`, `list`, `expression`, ...). `answer_spec(q)`
  reads the attributes back as one frozen `AnswerSpec`.
* `Distractor`: a wrong option that records the misconception it stands for (an id in `mathgen.misconceptions`), the
  structured value a learner who made that error would reach, and optionally the share of the marks that error still
  earns (`credit`, for a right method with a slip). A generator returns them in `distractors`; `make()` flattens them
  onto the question, so everything that reads `distractors` as strings is unchanged.

Everything here is read by attribute name, never by type, so any object with the same attributes (a part of a
multi-part question, say) is graded and diagnosed the same way. See `docs/design/typed-answer-contract.md`.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Any

#: The grader kinds (`mathgen.grading.typed.GRADERS`). Kept here, not imported, because `mathgen.core` imports this
#: module and the graders import `mathgen.core`; a test keeps the two lists equal.
ANSWER_KINDS = frozenset(
    {
        "number",
        "fraction",
        "pair",
        "list",
        "ordered_list",
        "set",
        "expression",
        "mcq",
        "table",
        "points",
        "polyline",
        "line",
    }
)

#: The shapes a grader can require of a reply (`Question.answer_form`). Order of items is not a form: it is the
#: difference between the `list`/`set` kinds and `ordered_list`.
ANSWER_FORMS = frozenset(
    {"expanded", "factored", "simplified_surd", "rationalised", "single_fraction", "line_standard_form"}
)

# (skill kind, value kind) pairs where a skill's grader reads a value of another kind. Any value can be offered as a
# multiple-choice option, so `mcq` accepts every kind.
_COMPATIBLE_KINDS = frozenset({("expression", "number"), ("expression", "fraction"), ("number", "fraction")})


def kind_accepts(skill_kind: str, value_kind: str) -> bool:
    """Whether a skill graded as `skill_kind` can grade an answer whose value is of `value_kind`."""
    return skill_kind == value_kind or skill_kind == "mcq" or (skill_kind, value_kind) in _COMPATIBLE_KINDS


@dataclass(frozen=True)
class AnswerSpec:
    """One answer, typed. A read-only view over a question's answer attributes; nothing stores it."""

    kind: str | None  # grader kind of `value`; None when neither the question nor the caller says
    value: Any  # the structured, JSON-serialisable answer (`Question.answer`)
    text: str  # the canonical spelling (`Question.answer_text`)
    latex: str = ""
    form: str | None = None  # one of ANSWER_FORMS, or None for "any equivalent form"
    alt: Any = None  # a numeric alternative a grader may also accept (`Question.answer_alt`)


def answer_spec(obj, kind: str | None = None) -> AnswerSpec:
    """The typed answer of `obj` (a Question, or anything with its answer attributes).

    `kind` is the fallback when `obj` has no `answer_kind` of its own: pass the skill's `answer`.
    """
    return AnswerSpec(
        kind=getattr(obj, "answer_kind", None) or kind,
        value=obj.answer,
        text=obj.answer_text,
        latex=getattr(obj, "answer_latex", "") or "",
        form=getattr(obj, "answer_form", None),
        alt=getattr(obj, "answer_alt", None),
    )


@dataclass(frozen=True)
class Distractor:
    """A wrong option and the error it stands for.

    `text` is spelled like `answer_text`. `misconception` is an id in `mathgen.misconceptions.CATALOGUE` (or None for
    a filler that stands for no particular error). `value` is what `answer` would be had this been the right answer:
    with it, a typed reply is matched to the distractor by the grader itself, so "x=-3" matches "x = -3"; without it,
    only the option's own spelling matches. `credit` is the share of the marks the error still earns, from 0 up to
    but not including 1.
    """

    text: str
    misconception: str | None = None
    value: Any = None
    credit: Fraction | None = None

    def __post_init__(self):
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError(f"a distractor needs its text, got {self.text!r}")
        if self.credit is not None:
            credit = Fraction(self.credit)
            if not 0 <= credit < 1:
                raise ValueError(f"distractor credit must be at least 0 and below 1, got {self.credit!r}")
            object.__setattr__(self, "credit", credit)

    def __str__(self) -> str:
        return self.text


def exact_value(number) -> Any:
    """`number` (an int, Fraction or SymPy Rational) as a JSON value the number and fraction graders read: an int when
    it is whole, else ``{"p": numerator, "q": denominator}``."""
    value = Fraction(int(number.p), int(number.q)) if hasattr(number, "p") else Fraction(number)
    return value.numerator if value.denominator == 1 else {"p": value.numerator, "q": value.denominator}


def flatten_distractors(question: dict) -> None:
    """Rewrite a generator's returned dict in place so `distractors` holds strings only.

    Each `Distractor` becomes its text in `distractors`, with its misconception in `option_notes`, its value in
    `distractor_values` and its credit in `distractor_credit`, all keyed by the text. A dict whose distractors are all
    strings is left exactly as it was. Raises `ValueError` for a misconception id the catalogue does not know, and for
    an option tagged twice with different ids.
    """
    distractors = question.get("distractors") or []
    if not any(isinstance(d, Distractor) for d in distractors):
        return
    from .misconceptions import CATALOGUE

    texts: list[str] = []
    notes = dict(question.get("option_notes") or {})
    values = dict(question.get("distractor_values") or {})
    credit = dict(question.get("distractor_credit") or {})
    for d in distractors:
        if not isinstance(d, Distractor):
            texts.append(str(d))
            continue
        texts.append(d.text)
        if d.misconception is not None:
            if d.misconception not in CATALOGUE:
                raise ValueError(f"distractor {d.text!r}: misconception {d.misconception!r} is not in the catalogue")
            if notes.get(d.text, d.misconception) != d.misconception:
                raise ValueError(f"distractor {d.text!r} is tagged both {notes[d.text]!r} and {d.misconception!r}")
            notes[d.text] = d.misconception
        if d.value is not None:
            values[d.text] = d.value
        if d.credit is not None:
            credit[d.text] = str(d.credit)
    question["distractors"] = texts
    for key, mapping in (("option_notes", notes), ("distractor_values", values), ("distractor_credit", credit)):
        if mapping:
            question[key] = mapping


def tagged_distractors(obj) -> list[Distractor]:
    """`obj`'s distractors as `Distractor`s, reassembled from the flat attributes `make()` stores."""
    notes = getattr(obj, "option_notes", None) or {}
    values = getattr(obj, "distractor_values", None) or {}
    credit = getattr(obj, "distractor_credit", None) or {}
    return [
        Distractor(
            text=text,
            misconception=notes.get(text),
            value=values.get(text),
            credit=Fraction(credit[text]) if text in credit else None,
        )
        for text in getattr(obj, "distractors", None) or []
    ]
