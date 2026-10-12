"""Marking a multi-part question: each part with its own grader, with follow-through marks.

A multi-part question (`Question.parts`, see `mathgen.core.Part` and `docs/design/multi-part-questions.md`) is
marked part by part, in order. A part earns its marks when its reply is right; or, when the part has a
`follow_through` expression, when its reply is right for the learner's OWN replies to the earlier parts the
expression names (an error carried forward from a wrong earlier part, or a correct earlier answer that was
rounded as asked and then used); or a share of them when its reply matches a distractor that carries partial
credit. `grade_parts` returns a `GradeResult` whose `marks`, `max_marks` and `parts` say how.

The reply is a dict of replies keyed by part label, a JSON object string of the same, or text labelled
"(a) ... (b) ...", labels in any order and any left out. Text with no labels is split by ";" or line breaks
when it has exactly one piece per part.
"""

from __future__ import annotations

import dataclasses
import json
import re
from fractions import Fraction

import sympy
from sympy.core.function import AppliedUndef

from mathgen.core import VerificationError, round_half_up, stated_precision_string, numeric_answer, rational_text

from . import typed
from .safe_parse import sympify_safely

#: Answer kinds whose reply reads as a single value, so a later part's `follow_through` can name them.
VALUE_KINDS = frozenset({"number", "fraction", "expression"})

_SYMBOL_PREFIX = "part_"
_NUMBER_LITERAL_RE = re.compile(r"-?\d+(?:\.\d+)?")
#: Significant digits a follow-through value is worked out to before it is rounded to the part's precision.
_DIGITS = 30
#: Largest follow-through value a number part states. A learner's earlier reply can drive a follow-through
#: expression to any size (z = 999999 puts erfc(z/sqrt(2)) near 10**-217000000000), and writing such a value out
#: exactly takes a number with that many digits, so a value outside this bound has no follow-through answer.
_MAX_MAGNITUDE = sympy.Integer(10) ** 30


def symbol_name(label: str) -> str:
    """The name a `follow_through` expression uses for part `label`'s value: "a" -> "part_a"."""
    return _SYMBOL_PREFIX + label


# -- reading the reply --------------------------------------------------------------------------------------


def _label_marker(label: str) -> re.Pattern:
    """ "(a)", "a)" or "a:" at the start of the reply or after a space, ";" or "," (any case)."""
    escaped = re.escape(label)
    return re.compile(rf"(?:^|(?<=[\s;,]))(?:\(\s*{escaped}\s*\)|{escaped}\s*[):])\s*", re.IGNORECASE)


def split_reply(parts, given) -> dict[str, str]:
    """`given` as one reply per part label. A label with no reply gets "", which no part accepts."""
    labels = [p.label for p in parts]
    if isinstance(given, dict):
        return {label: str(given.get(label, "") or "") for label in labels}
    text = str(given or "").strip()
    if text.startswith("{"):
        try:
            loaded = json.loads(text)
        except ValueError:
            loaded = None
        if isinstance(loaded, dict):
            return {label: str(loaded.get(label, "") or "") for label in labels}
    # Labelled: each label's first marker, in any order. A label with no marker is a part left unanswered, so
    # "(a) 2.8 (b) 17.6" still marks (a) and (b). Text must open with a marker to be read this way.
    spans = []
    for label in labels:
        match = _label_marker(label).search(text)
        if match is not None:
            spans.append((match.start(), match.end(), label))
    spans.sort()
    if spans and spans[0][0] == 0:
        replies = {label: "" for label in labels}
        for i, (_start, end, label) in enumerate(spans):
            stop = spans[i + 1][0] if i + 1 < len(spans) else len(text)
            replies[label] = text[end:stop].strip().rstrip(";,").strip()
        return replies
    # Unlabelled: one piece per part, split by ";" or line breaks.
    for separator in (";", "\n"):
        pieces = [piece.strip() for piece in text.split(separator)]
        if len(pieces) == len(labels):
            return dict(zip(labels, pieces))
    if len(labels) == 1:
        return {labels[0]: text}
    return {label: "" for label in labels}


def reply_value(kind: str, reply: str):
    """The single value a reply to a part of `kind` states, as a SymPy object, or None when it states none.

    This is what the learner's earlier answer is taken to be when a later part follows through from it."""
    reply = typed.normalise_reply(kind, reply or "")
    if not reply.strip():
        return None
    try:
        if kind == "number":
            value = typed._typed_number_value(reply)
            return None if value is None else sympy.Rational(value.numerator, value.denominator)
        if kind == "fraction":
            value = typed._parse_typed_fraction(typed._strip_label(reply))
            if value is None:
                value = typed._typed_number_value(reply)
            return None if value is None else sympy.Rational(value.numerator, value.denominator)
        if kind == "expression":
            value = typed._parse_expression(typed._strip_label(reply))
            return value if isinstance(value, sympy.Basic) else None
    except Exception:  # noqa: BLE001 - an unreadable reply simply has no value to follow through from
        return None
    return None


# -- follow-through ------------------------------------------------------------------------------------------


#: Functions a `follow_through` expression may use beyond the safe parser's own (which reads them as
#: undefined functions): `erf` and `erfc` for normal probabilities, and `Subs(expr, (x, y), (2, 3))` to
#: evaluate an earlier part's expression at a point.
_FUNCTIONS = {
    "erf": sympy.erf,
    "erfc": sympy.erfc,
    "Subs": lambda expr, names, values: expr.subs(dict(zip(names, values))),
}


def follow_through_expression(part) -> sympy.Expr:
    """`part.follow_through` parsed (by the engine's safe parser), with `pi` and `E` as the constants."""
    expr = sympify_safely(part.follow_through)
    return expr.xreplace({sympy.Symbol("pi"): sympy.pi, sympy.Symbol("E"): sympy.E})


def _apply_functions(expr):
    """`expr` with the `_FUNCTIONS` applications made real (and `Subs` evaluated)."""
    return expr.replace(
        lambda e: isinstance(e, AppliedUndef) and e.func.__name__ in _FUNCTIONS,
        lambda e: _FUNCTIONS[e.func.__name__](*e.args),
    )


def named_parts(part) -> list[str]:
    """Labels of the earlier parts `part.follow_through` names, in name order."""
    expr = follow_through_expression(part)
    return sorted(s.name[len(_SYMBOL_PREFIX) :] for s in expr.free_symbols if s.name.startswith(_SYMBOL_PREFIX))


def retarget(part, value):
    """A copy of `part` whose canonical answer is `value`, stated the way the part states its own answer, or
    None when `value` cannot be stated that way (not a real number for a number part, not rational for a
    fraction part, or a number part whose `answer_text` does not hold exactly one number)."""
    kind = part.answer_kind
    common = dict(answer_alt=None, distractors=[], option_notes={}, distractor_values={}, distractor_credit={})
    if kind == "expression":
        return dataclasses.replace(part, answer=value, answer_text=sympy.sstr(value), **common)
    if not value.is_number:
        return None
    value = sympy.nsimplify(value) if kind == "fraction" else value
    if kind == "fraction":
        if not value.is_Rational:
            return None
        return dataclasses.replace(
            part, answer={"p": int(value.p), "q": int(value.q)}, answer_text=rational_text(value), **common
        )
    if kind != "number" or not value.is_real:
        return None
    literals = _NUMBER_LITERAL_RE.findall(part.answer_text or "")
    if len(literals) != 1:
        return None
    places = len(literals[0].split(".")[1]) if "." in literals[0] else 0
    approx = sympy.N(value, _DIGITS)
    if not approx.is_finite or abs(approx) >= _MAX_MAGNITUDE:
        return None
    # Under a hundredth of the part's last place, the value states as zero; its exact digits are never needed.
    exact = Fraction(0) if abs(approx) < sympy.Rational(1, 10 ** (places + 2)) else Fraction(str(approx))
    rounded = round_half_up(exact, places) if exact >= 0 else -round_half_up(-exact, places)
    stated = stated_precision_string(rounded, places) if places else str(int(rounded))
    answer_text = _NUMBER_LITERAL_RE.sub(stated, part.answer_text, count=1)
    return dataclasses.replace(part, answer=numeric_answer(rounded), answer_text=answer_text, **common)


def follow_through_target(part, values: dict[str, object]):
    """`part` retargeted to its follow-through answer from earlier parts' `values` (label -> SymPy value), or
    None when a named part has no value or the result cannot be stated as this part's answer."""
    value = follow_through_value(part, values)
    if value is None:
        return None
    try:
        return retarget(part, value)
    except (TypeError, ValueError, ZeroDivisionError, ArithmeticError):
        return None


def follow_through_value(part, values: dict[str, object]):
    """`part.follow_through` worked out exactly at earlier parts' `values`, before it is stated to the part's
    precision, or None when a named part has no value or it cannot be worked out."""
    expr = follow_through_expression(part)
    substitutions = {}
    for symbol in expr.free_symbols:
        if not symbol.name.startswith(_SYMBOL_PREFIX):
            continue
        value = values.get(symbol.name[len(_SYMBOL_PREFIX) :])
        if value is None:
            return None
        substitutions[symbol] = value
    try:
        return _apply_functions(expr.xreplace(substitutions))
    except (TypeError, ValueError, ZeroDivisionError, ArithmeticError):
        return None


def verify_follow_through(parts) -> None:
    """Raise `VerificationError` unless every `follow_through` names only earlier parts of a single-value kind,
    and, worked out from the earlier parts' canonical answers, reproduces its own part's canonical answer (a
    number part to within one unit of its last stated place, which allows for the earlier answer having been
    rounded). Called by `make()` on every multi-part question."""
    from .typed import GRADERS

    earlier, values = {}, {}
    for part in parts:
        if part.answer_kind not in GRADERS:
            raise VerificationError(f"part {part.label!r}: answer kind {part.answer_kind!r} has no grader")
        if part.sensible_range is not None:
            _verify_range(part)
        if part.follow_through:
            names = named_parts(part)
            if not names:
                raise VerificationError(f"part {part.label!r}: follow_through names no earlier part")
            for name in names:
                if name not in earlier:
                    raise VerificationError(f"part {part.label!r}: follow_through names {name!r}, not an earlier part")
                if earlier[name].answer_kind not in VALUE_KINDS:
                    raise VerificationError(
                        f"part {part.label!r}: follow_through names part {name!r}, "
                        f"whose {earlier[name].answer_kind!r} reply is not a single value"
                    )
            target = follow_through_target(part, values)
            if target is None:
                raise VerificationError(f"part {part.label!r}: follow_through cannot be worked out from the answers")
            if not typed.grade(part.answer_kind, target, part.answer_text).correct and not _within_one_place(
                part, target
            ):
                raise VerificationError(
                    f"part {part.label!r}: follow_through gives {target.answer_text!r}, "
                    f"not the part's answer {part.answer_text!r}"
                )
        earlier[part.label] = part
        if part.answer_kind in VALUE_KINDS and any(p.follow_through for p in parts):
            values[part.label] = reply_value(part.answer_kind, part.answer_text)


def _verify_range(part) -> None:
    bounds = part.sensible_range
    if (
        not isinstance(bounds, (tuple, list))
        or len(bounds) != 2
        or not all(b is None or (isinstance(b, (int, float, Fraction)) and not isinstance(b, bool)) for b in bounds)
        or (None not in bounds and bounds[0] >= bounds[1])
    ):
        raise VerificationError(f"part {part.label!r}: sensible_range must be (low, high), either end a number or None")
    if part.answer_kind not in ("number", "fraction"):
        raise VerificationError(f"part {part.label!r}: sensible_range needs a number or fraction part")
    value = reply_value(part.answer_kind, part.answer_text)
    if value is None or not _in_range(bounds, value):
        raise VerificationError(f"part {part.label!r}: its answer {part.answer_text!r} is outside sensible_range")


def _within_one_place(part, target) -> bool:
    if part.answer_kind != "number":
        return False
    mine = _NUMBER_LITERAL_RE.findall(part.answer_text or "")
    theirs = _NUMBER_LITERAL_RE.findall(target.answer_text or "")
    if len(mine) != 1 or len(theirs) != 1:
        return False
    places = len(mine[0].split(".")[1]) if "." in mine[0] else 0
    return abs(Fraction(mine[0]) - Fraction(theirs[0])) <= Fraction(1, 10**places)


# -- marking -------------------------------------------------------------------------------------------------


def grade_parts(question, given, *, skill=None) -> typed.GradeResult:
    """Mark every part of a multi-part `question` against `given` (see the module docstring)."""
    if isinstance(given, str) and len(given) > typed.MAX_REPLY_LENGTH:
        given = ""
    replies = split_reply(question.parts, given)
    results: dict[str, typed.GradeResult] = {}
    values: dict[str, object] = {}
    # Labels whose reply is not the canonical answer, whether or not it earned marks: a later part that follows
    # through from one of them is marked `follow_through`, so a whole chain carried from a wrong (a) says so.
    off_canonical: set[str] = set()
    for part in question.parts:
        reply = replies.get(part.label, "")
        if len(reply) > typed.MAX_REPLY_LENGTH:
            reply = ""
        result = typed.grade(part.answer_kind, part, reply, skill=skill)
        earned = Fraction(part.marks) if result.correct else Fraction(0)
        if not result.correct:
            off_canonical.add(part.label)
        if not result.correct and part.follow_through and _carries(question.parts, part, values):
            target = follow_through_target(part, values)
            if (
                target is not None
                and _not_trivial(question.parts, part, target, values)
                and typed.grade(part.answer_kind, target, reply).correct
            ):
                earned = Fraction(part.marks)
                result.correct = True
                result.note = result.note_kind = None
                for attr in ("misconception", "matched_distractor"):
                    if hasattr(result, attr):
                        setattr(result, attr, None)
                result.follow_through = bool(off_canonical & set(named_parts(part)))
        if earned == 0:
            credit = _distractor_credit(part, result)
            if credit:
                earned = Fraction(part.marks) * credit
        result.marks, result.max_marks = earned, part.marks
        results[part.label] = result
        if part.answer_kind in VALUE_KINDS:
            values[part.label] = reply_value(part.answer_kind, reply)
    total = sum((r.marks for r in results.values()), Fraction(0))
    out_of = sum(p.marks for p in question.parts)
    first_note = next((r for r in results.values() if r.note), None)
    return typed.GradeResult(
        correct=total == out_of,
        canonical=question.answer_text,
        given_normalised="; ".join(
            f"({label}) {r.given_normalised}" for label, r in results.items() if r.given_normalised is not None
        )
        or None,
        note=first_note.note if first_note else None,
        note_kind=first_note.note_kind if first_note else None,
        marks=total,
        max_marks=out_of,
        parts=results,
    )


def _carries(parts, part, values) -> bool:
    """Whether the learner's replies to the parts `part` follows through from can carry marks into it. Not
    when one is zero, or rounds to zero at its part's own precision, where its canonical answer is not (zero
    makes most later work trivial), nor when one lies outside its part's `sensible_range` (a negative length,
    a probability above 1)."""
    by_label = {p.label: p for p in parts}
    for name in named_parts(part):
        value, source = values.get(name), by_label[name]
        if value is None or not getattr(value, "is_number", False):
            continue
        if _states_zero(source, value) and not _same(_canonical_value(source), 0):
            return False
        if not _in_range(source.sensible_range, value):
            return False
    return True


def _not_trivial(parts, part, target, values) -> bool:
    """Whether a follow-through answer still asks for real work: not when a tiny earlier answer rounds it to
    zero while the canonical answer is not zero (an answer that is exactly zero, such as a tangent through the
    origin, still counts), nor when it is the very value of an earlier reply it uses while the canonical
    answers differ. Writing the same number again is not following through
    (a constant dy/dx is its own gradient at P; "1" for every part matches an arc of 1 radian)."""
    value = reply_value(part.answer_kind, target.answer_text)
    if value is None or not getattr(value, "is_number", False):
        return True
    canonical = _canonical_value(part)
    exact = follow_through_value(part, values)
    if _same(value, 0) and not _same(canonical, 0) and exact is not None and not _same(exact, 0):
        return False  # a tiny earlier answer that rounds this part's answer away to nothing
    by_label = {p.label: p for p in parts}
    for name in named_parts(part):
        earlier = values.get(name)
        if earlier is not None and _same(earlier, value) and not _same(_canonical_value(by_label[name]), canonical):
            return False
    return True


def _states_zero(part, value) -> bool:
    """Whether `value` is zero as `part` states its answer: 0.001 is 0.0 for a part asked to 1 decimal place."""
    if _same(value, 0):
        return True
    try:
        stated = retarget(part, value)
    except (TypeError, ValueError, ZeroDivisionError, ArithmeticError):
        return False
    return stated is not None and _same(reply_value(part.answer_kind, stated.answer_text), 0)


def _same(left, right) -> bool:
    """Equal as numbers, a float reply included: SymPy does not count Float(0.5) equal to Rational(1, 2)."""
    if left is None or right is None:
        return left is right
    try:
        return bool((left - right).is_zero)
    except TypeError:
        return False


def _canonical_value(part):
    return reply_value(part.answer_kind, part.answer_text) if part.answer_kind in VALUE_KINDS else None


def _in_range(bounds, value) -> bool:
    if bounds is None or not getattr(value, "is_real", False):
        return True
    low, high = (None if b is None else sympy.Rational(Fraction(b).numerator, Fraction(b).denominator) for b in bounds)
    return (low is None or value > low) and (high is None or value < high)


def _distractor_credit(part, result) -> Fraction:
    """The share of `part`'s marks a reply matching one of its distractors earns: `part.distractor_credit`
    (option text -> a fraction from 0 to 1) at the distractor the grader matched (`result.matched_distractor`).
    Both come from the typed answer contract; without them a wrong reply earns nothing."""
    matched = getattr(result, "matched_distractor", None)
    credits = getattr(part, "distractor_credit", None) or {}
    if matched is None or matched not in credits:
        return Fraction(0)
    credit = Fraction(str(credits[matched]))
    return credit if 0 < credit <= 1 else Fraction(0)
