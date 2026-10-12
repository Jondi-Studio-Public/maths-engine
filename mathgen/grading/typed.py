"""Answer-kind grading registry.

Twelve kinds, one grader function each, registered in ``GRADERS``. ``number`` does **not** delegate to
``check_answer`` for a plain ``int``/ ``float`` answer: that path used ``check_answer``'s spoken-transcript
tolerance (``_TOLERANCE = 0.051``) for typed answers too, which is coarser than the hundredths place several
decimal-answer skills teach -- see ``grade_number``'s own docstring for the typed-answer comparison this was
replaced with, and ``_check_answer_kind``/``check_answer`` are kept only as ``grade_number``'s fallback for a
non-scalar ``answer`` shape and as the unchanged path for spoken transcripts. ``expression`` gets its own
sympy parse (accepting ``^`` for power and implicit multiplication, falling back to the spoken-word parser)
since it needs a parsed expression back, not just a bool. ``ordered_list`` does its own delimiter-tolerant
splitting instead of delegating to ``check_answer``/``extract_values``, because that pair's list branch grades
an unordered multiset and its tokenizer would split a comma-grouped numeral -- see the function's docstring
for both the splitting policy and why.
"""

from __future__ import annotations

import copy
import functools
import json
import logging
import random
import math
import re
from tokenize import TokenError
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
from typing import Callable

import sympy
from sympy.parsing.sympy_parser import (
    implicit_multiplication_application,
    standard_transformations,
)

from mathgen.grading.safe_parse import (
    MAX_LENGTH as MAX_ANSWER_LENGTH,
    TooCostly,
    check_expression_cost,
    parse_expression_safely,
    sympify_safely,
    too_costly,
)
from mathgen.grading.spoken import NUMBER_WORDS, check_answer, extract_values, spoken_to_expr
from mathgen.grading.units import unit_is_acceptable
from mathgen.core import mcq_options, symbolically_equal  # noqa: F401
from mathgen.misconceptions import describe as describe_misconception
from mathgen.text_quality import requested_rounding

logger = logging.getLogger("mathgen.grading")

_TRANSFORMATIONS = standard_transformations + (implicit_multiplication_application,)

_LABEL_RE = re.compile(r"^\s*[^\W\d]\w*(\s*\([^)]*\))?\s*=\s*")  # a Greek label ("λ = 3") too

# "1 1/2" (mixed number) isn't a literal fractions.Fraction can parse on its own, so it gets its own
# regex; plain fractions ("3/4") and exact decimals ("0.75") are read by `_exact_fraction`.
_MIXED_FRACTION_RE = re.compile(r"^\s*(-?)\s*(\d+)\s+(\d+)\s*/\s*(\d+)\s*$")


def _strip_label(given: str) -> str:
    return _LABEL_RE.sub("", given, count=1)


def _label_name(text: str | None) -> str | None:
    """The variable a reply or canonical is labelled with ("x" in "x = 3"), or None when it has no label."""
    match = _LABEL_RE.match(text or "")
    return match.group(0).split("=")[0].strip() if match else None


_VARIABLE_LABEL_RE = re.compile(r"[^\W\d_]\d?")  # "x", "λ", "x1": a variable, not a word or a function


def _label_fits(question, canonical: str | None, label: str | None) -> bool:
    """Whether a reply's label may be dropped: it is the canonical's own label, or the canonical has none and the
    label is a single variable the question's text names ("x" in "Solve 3^x = 27")."""
    if label is None:
        return False
    canonical_label = _label_name(canonical)
    if canonical_label is not None:
        return label == canonical_label
    if not _VARIABLE_LABEL_RE.fullmatch(label):
        return False
    text = getattr(question, "text", None) or ""
    return re.search(rf"(?<![^\W\d_]){re.escape(label)}(?![^\W\d_])", text) is not None


_RATIO_RE = re.compile(r"^(-?)\s*(\d+)\s*:\s*(\d+)$")


#: A number as a learner types it into a fraction or a list: an integer, a decimal ("0.75", ".5", "5.") or
#: "a/b" (spaces and tabs around the slash are fine: "7 / 13"), with an optional sign. No exponent and no
#: underscore: `Fraction("1e9999999")` works out 10**9999999, which takes seconds, and `1e99999999` minutes, for
#: ten characters of reply.
_EXACT_NUMBER_RE = re.compile(r"[-+]?(?:\d+[ \t]*/[ \t]*\d+|\d+\.?\d*|\.\d+)")
#: Longest text read as one exact number. The longest in the library is a dozen characters.
_MAX_EXACT_NUMBER_LENGTH = 100


def _exact_fraction(text: str) -> Fraction | None:
    """`text` as an exact `Fraction`, or None when it is not a plain number (see `_EXACT_NUMBER_RE`).

    The one place a reply's text becomes a `Fraction`: `Fraction(str)` itself reads exponents (and
    underscores), so it is only ever called here, on text that has already matched the pattern above.
    """
    text = (text or "").strip()
    if len(text) > _MAX_EXACT_NUMBER_LENGTH or not _EXACT_NUMBER_RE.fullmatch(text):
        return None
    try:
        return Fraction(text.replace(" ", "").replace("\t", ""))  # Fraction itself refuses "7 / 13"
    except (ValueError, ZeroDivisionError):
        return None


# "½", "2¾": the vulgar fraction characters, read as the fraction they show.
_VULGAR_FRACTIONS = {
    "½": "1/2", "⅓": "1/3", "⅔": "2/3", "¼": "1/4", "¾": "3/4", "⅕": "1/5", "⅖": "2/5", "⅗": "3/5", "⅘": "4/5",
    "⅙": "1/6", "⅚": "5/6", "⅛": "1/8", "⅜": "3/8", "⅝": "5/8", "⅞": "7/8",
}  # fmt: skip
_VULGAR_FRACTION_RE = re.compile("(\\d?)[ \t]*([" + "".join(_VULGAR_FRACTIONS) + "])")
# "\frac{11}{19}", "\dfrac{11}{19}", "\frac11{19}": LaTeX's fraction, as a learner copies it from a rendered page.
_LATEX_FRACTION_RE = re.compile(r"\\[dt]?frac[ \t]*\{?[ \t]*(\d+)[ \t]*\}?[ \t]*\{[ \t]*(\d+)[ \t]*\}")
# "(11/19)" and "(-3)/5": brackets around a whole fraction, or around its numerator.
_BRACKETED_FRACTION_RE = re.compile(r"^\(\s*([^()]*)\s*\)$|^\(\s*(-?\s*\d+)\s*\)\s*/\s*(\d+)$")
# The fraction slash, the division slash and the division sign are all a fraction bar here.
_FRACTION_BARS = str.maketrans({"⁄": "/", "∕": "/", "÷": "/"})


def _vulgar_fraction(match: re.Match) -> str:
    whole, fraction = match.groups()
    return f"{whole} {_VULGAR_FRACTIONS[fraction]}" if whole else _VULGAR_FRACTIONS[fraction]


def _tidy_fraction_text(given: str) -> str:
    """A typed fraction in the plain "a/b" or "w a/b" form `_parse_typed_fraction` reads: LaTeX's \\frac, the
    vulgar fraction characters, other fraction bars, a unicode minus and one pair of brackets are rewritten."""
    text = (given or "").strip().translate(_MINUS_SIGNS).translate(_FRACTION_BARS)
    text = _LATEX_FRACTION_RE.sub(r"\1/\2", text)
    text = _VULGAR_FRACTION_RE.sub(_vulgar_fraction, text)
    bracketed = _BRACKETED_FRACTION_RE.match(text)
    if bracketed:
        inner, numerator, denominator = bracketed.groups()
        text = inner if inner is not None else f"{numerator.replace(' ', '')}/{denominator}"
    return text.strip()


def _parse_typed_fraction(given: str) -> Fraction | None:
    text = (given or "").strip()
    if not text or len(text) > _MAX_EXACT_NUMBER_LENGTH:
        return None
    text = _tidy_fraction_text(text)
    m = _MIXED_FRACTION_RE.match(text)
    if m:
        sign, whole, num, den = m.groups()
        try:
            if int(den) == 0:
                return None
            value = Fraction(int(whole)) + Fraction(int(num), int(den))
        except ValueError:
            return None
        return -value if sign == "-" else value
    # Ratio notation. ratio-rate-proportion states its canonical answer as
    # "12:5" and is graded by grade_fraction, which could not parse a colon at
    # all: every learner who typed the answer they were shown was marked wrong
    # (120 false rejects of 120 questions). a:b is the value a/b here, matching
    # this grader's by-value contract, so "24:10" and "12/5" are accepted for
    # "12:5" exactly as "24/10" already is.
    ratio = _RATIO_RE.match(text)
    if ratio:
        sign, left, right = ratio.groups()
        try:
            if int(right) == 0:
                return None
            value = Fraction(int(left), int(right))
        except ValueError:
            return None
        return -value if sign == "-" else value
    return _exact_fraction(text)


@dataclass
class GradeResult:
    correct: bool
    canonical: str
    given_normalised: str | None = None
    note: str | None = None  # Resolved misconception/near-miss text
    note_kind: str | None = None  # "misconception" or "near_miss"; None when `note` is None
    cells: dict[str, bool] | None = None  # table: per-blank verdict, keyed "r,c"
    misconception: str | None = None  # id of the tagged distractor a wrong reply matched (mathgen/misconceptions.py)
    matched_distractor: str | None = None  # that distractor's option text, the key for its `distractor_credit`
    # Multi-part questions (`mathgen.grading.parts`). On the question's result: `marks` earned (a Fraction),
    # `max_marks`, and `parts`, each part's own result keyed by label. On a part's result: its `marks` and
    # `max_marks`, and `follow_through` when its marks were earned from the learner's own wrong earlier answer.
    # All None/False for a single-answer question.
    marks: Fraction | None = None
    max_marks: int | None = None
    parts: dict[str, GradeResult] | None = None
    follow_through: bool = False


def _multi_part_aware(grader: Callable) -> Callable:
    """Every grader, handed a multi-part question, grades it part by part with each part's own grader,
    whatever kind it was called as (see `mathgen.grading.parts`). Applied to each grader where it is defined,
    so `GRADERS[kind]` and the module-level name stay the same object."""

    @functools.wraps(grader)
    def graded(question, given):
        if getattr(question, "parts", None):
            from .parts import grade_parts

            return grade_parts(question, given)
        return grader(question, given)

    return graded


def _normalised_values(given: str) -> str | None:
    try:
        values = extract_values(given)
    except Exception:  # noqa: BLE001 - normalisation is best-effort
        logger.warning("extract_values failed for given=%.200r", given, exc_info=True)
        return None
    if not values:
        return None
    return ", ".join(str(v) for v in values)


def _check_answer_kind(question, given: str) -> GradeResult:
    correct = bool(check_answer(question, given))
    return GradeResult(correct=correct, canonical=question.answer_text, given_normalised=_normalised_values(given))


# -- number (typed-answer path) ------------------------------
#
# A typed number is not held to the flat +/-0.051 that spoken transcripts get (`mathgen.grading.spoken` still
# uses that, unchanged). Its precision is derived from what `question.answer_text` itself *displays*, not
# from a fixed constant and not from `repr(question.answer)`, which carries float noise
# (0.30000000000000004) that the learner was never shown.

_NUMERIC_LITERAL_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _decimal_places(literal: str) -> int:
    """Count of digits after the decimal point *as written* in `literal`
    (no trailing-zero stripping -- "14.10" states hundredths, "14.1" states
    tenths; if a generator's own formatting already collapsed "14.10" to
    "14.1", that precision is genuinely lost before grading.py ever sees it)."""
    if "." not in literal:
        return 0
    return len(literal.split(".", 1)[1])


def _typed_number_tolerance(places: int) -> Fraction:
    """Half a unit in the last stated decimal place -- this is the **comparison** tolerance: how close a
    learner's *typed* answer must be to the trusted target to count as correct. An integer literal (`places ==
    0`, no decimal point at all) states no decimal slack -- the obvious formula
    `Fraction(1, 2 * 10**places)` evaluates to `Fraction(1, 2)` at `places == 0`, which
    would wrongly accept e.g. "42.3" for a canonical "42" (a regression test rejects
    "42.03" for "42").

    Do **not** reuse this function for the sanity check inside
    `_typed_number_target` (whether an extracted literal is a plausible
    rounding of `question.answer`, and therefore safe to trust as the
    comparison target at all) -- that is a different question with a
    different answer at `places == 0`, and collapsing the two caused a real
    regression (`decimal-arithmetic` seed 494, `answer=159.0001,
    answer_text="159"`: the zero-decimal-place literal "159" is a perfectly
    plausible rounding of 159.0001, but sharing this function's `places <= 0
    -> Fraction(0)` special case for the sanity check made the literal
    untrustable unless it matched the float *exactly*, which sent grading
    into the fallback path and then rejected the very literal the learner
    was shown as the canonical answer). See `_plausible_rounding_tolerance`
    for the sanity-check threshold instead."""
    if places <= 0:
        return Fraction(0)
    return Fraction(1, 2 * 10**places)


def _plausible_rounding_tolerance(places: int) -> Fraction:
    """Half a unit in `places` decimal places, with **no** special-casing at
    `places == 0` -- this is the **sanity-check** tolerance used only inside
    `_typed_number_target` to decide whether an extracted `answer_text`
    literal is a plausible correctly-rounded display of the underlying
    float `question.answer`, and therefore safe to trust as the grading
    target/precision source.

    This is deliberately a different function from `_typed_number_tolerance`
    (the **comparison** tolerance applied once a literal is trusted, or in
    the fallback path). The two answer different questions: "is this literal
    a plausible rounding of the float" (this function -- ordinary
    round-to-nearest semantics, `Fraction(1, 2)` at zero decimal places, the
    same as rounding 159.0001 to the nearest integer landing on 159) versus
    "how much slack should a learner's typed answer get once we trust the
    literal" (`_typed_number_tolerance` -- deliberately `Fraction(0)` at zero
    decimal places, an integer canonical requires an exact typed match).
    Sharing one function for both was the fix's original bug: it made the sanity check reject exact-integer literals
    that were still perfectly good roundings of a float with hidden
    fractional noise, e.g. "159" for the true value 159.0001."""
    return Fraction(1, 2 * 10**places)


_FLOAT_NOISE = Fraction(1, 10**9)


def _is_exact_display(literal_value: Fraction, answer) -> bool:
    """Whether the displayed literal is the answer itself rather than a rounding of it.

    An `int` answer is compared exactly. A `float` answer carries binary noise (7.7 - 5.1 is
    2.6000000000000005), so it counts as exact when it is within a relative 1e-9 of the literal: far
    below any precision a question displays, and far above double-precision noise."""
    if isinstance(answer, int):
        return literal_value == answer
    return abs(Fraction(repr(float(answer))) - literal_value) <= _FLOAT_NOISE * max(1, abs(literal_value))


@dataclass
class _TypedNumberTarget:
    target: Fraction
    tolerance: Fraction
    source: str  # "literal" (derived from answer_text) or "fallback" (repr(float(answer)))


def _typed_number_target(question, answer=None) -> _TypedNumberTarget:
    """Derive the exact comparison target and tolerance for a scalar
    `int`/`float` answer (`question.answer`, or `answer` when the caller has read it from some other form),
    from what `question.answer_text` itself displays.

    Extracts numeric literals from `answer_text` with a plain decimal regex (no exponent form). If exactly one
    is found, it is sanity-checked against the fallback target described below using
    `_plausible_rounding_tolerance` -- **not** `_typed_number_tolerance` -- at its own stated decimal places.
    Once trusted, the literal's *comparison* tolerance still comes from `_typed_number_tolerance` (computed
    separately, below) -- only the trust decision uses the more lenient threshold, not the eventual grading
    strictness. If zero or more than one literal is found, or the sanity check fails (the literal disagrees
    with `answer` by more than its own plausible-rounding tolerance -- a wrong regex extraction, or an
    unrelated number in `answer_text`), falls back to `repr(float(question.answer))` itself as both the target
    and the precision source -- **except** when `question.answer` is a plain `int` (not merely an int-*valued*
    `float`): `repr(float(some_int))` always ends `.0` (one stated decimal place, e.g. `"30.0"` for `30`),
    which would give an `int` answer a `0.05` tolerance in the fallback path -- the same defect shape this
    code exists to fix, just narrower than the old `0.051`. A genuine Python `int` answer states whole-unit
    precision with no such ambiguity, so it gets an exact-match fallback instead."""
    if answer is None:
        answer = question.answer
    if isinstance(answer, int):
        fallback_target = Fraction(answer)
        fallback_tolerance = Fraction(0)
    else:
        fallback_literal = repr(float(answer))
        fallback_target = Fraction(fallback_literal)
        fallback_tolerance = _typed_number_tolerance(_decimal_places(fallback_literal))

    literals = _NUMERIC_LITERAL_RE.findall(question.answer_text or "")
    if len(literals) == 1:
        literal = literals[0]
        try:
            literal_value = Fraction(literal)
        except (ValueError, ZeroDivisionError):
            literal_value = None
        if literal_value is not None:
            places = _decimal_places(literal)
            # Sanity check (is this literal a plausible rounding of the
            # float?) uses the lenient, non-special-cased threshold. The
            # comparison tolerance actually returned/used for grading
            # (`tolerance`, below) still comes from `_typed_number_tolerance`.
            sanity_tolerance = _plausible_rounding_tolerance(places)
            tolerance = _typed_number_tolerance(places)
            if abs(literal_value - fallback_target) <= sanity_tolerance:
                if _is_exact_display(literal_value, answer):
                    # The literal is the answer itself, not a rounding of it ("2.6" for 7.7 - 5.1), so
                    # there is nothing to round and no slack to give: 2.64 is a different number.
                    tolerance = Fraction(0)
                return _TypedNumberTarget(target=literal_value, tolerance=tolerance, source="literal")

    return _TypedNumberTarget(target=fallback_target, tolerance=fallback_tolerance, source="fallback")


def _value_side(text: str) -> str:
    """The side of a labelled answer that carries the value.

    Most labels hold no number ("F = 60 N"), but a function-application label
    holds its own argument: `extract_values` reads "f(0) = 1" as [0, 1], so a
    learner whose value was wrong was accepted whenever the argument happened
    to equal the expected value. Four skills built on function_evaluation
    accepted "f(0) = 1" for "f(0) = 0" and "f(2) = 2" for "f(2) = 10".

    Taking the side after the last "=" removes the argument from
    consideration. A learner who types the bare value with no label is
    unaffected, since there is then nothing to split.
    """
    if "=" not in (text or ""):
        return text or ""
    return text.rsplit("=", 1)[1]


def _is_rounded_as_asked(question, given: str) -> bool:
    """Whether a typed number has been rounded the way the question asked.

    "Round 20.26 to the nearest tenth" has canonical 20.3, and the unrounded
    20.26 sits 0.04 away -- inside the half-a-tenth comparison tolerance -- so
    the skill accepted the one answer it exists to reject (80 false accepts).
    An answer with more decimal places than were asked for is only acceptable
    when the extra places do not change it: "20.30" is the same number stated
    redundantly, "20.26" has not been rounded.
    """
    places = requested_rounding(getattr(question, "text", ""))
    if places is None:
        return True
    literal = _NUMERIC_LITERAL_RE.search(_number_text(given))
    if literal is None:
        return True
    token = literal.group(0)
    if _decimal_places(token) <= places:
        return True
    try:
        value = Fraction(token)
    except (ValueError, ZeroDivisionError):
        return True
    scale = 10**places
    return Fraction(round(value * scale), scale) == value


# -- number: exactly one value per reply ------------------------------------
#
# `extract_values` is the spoken-transcript normaliser: it returns EVERY number in the text, and the
# grader used to accept the reply when ANY of them matched. So "5 9", "9 or 7", "not 9" and a counted
# run "0 1 2 ... 29 9" were all correct for 9, and anyone (or any script) could pass a number skill by
# listing numbers. A typed number is a single value, so a reply is read as follows.
#
#   * Digits and number words are read as before ("9", "9.0", "nine", "1,000", "$9", "9 apples",
#     "9 cm", "x = 9", "f(2) = 9", "minus nine", "-9"), and so is a decimal written without its
#     leading zero (".5", "-.5").
#   * A minus that is not a subtraction is a sign, whatever stands between it and the number: "-9",
#     "- 9", "-$542.00", "- $ 542". It is a subtraction when a word character, a digit, "." or ")" stands
#     right before it ("5-9", "x-9"). The unicode minus and the dash variants are a minus sign too.
#   * An exponent written on a unit ("9 cm^2", "9 m2", "9 m/s^2", "9 s^-1") is part of the unit, not a
#     second value, and is dropped before the count. Only what follows the first number can be a unit
#     ("AUD9", "Rs9" and "minus5" are the number 9 and 5).
#   * A precision note ("2 dp", "to 1 decimal place", "3 s.f.", "2 significant figures") and a bracketed
#     remark with no digit in it ("(no units)", "(approx.)") say nothing about the value and are dropped.
#   * Exactly one number must remain. Two or more ("5 9", "5-9", "2 + 2") make the reply wrong, even
#     when one of them is the answer; none ("x") is wrong too.
#   * A reply that negates or hedges between values is wrong: "not 9", "isn't 9", "9 or -9", "+-9",
#     "9 != 9" (a comparison sign, a "plus or minus" and a "not equal" all say something other than
#     "the answer is this number"). "No" is not one of these words: "9 (no units)", "No. of sides = 9"
#     and "no, it's 9" all say 9. A "+-" inside a sum ("y=2x+-3") is a plus and a minus, not "plus or minus".
#
# What was said about the label side of "=" is unchanged: only the side after the LAST "=" is the
# value, so "area = 3 x 3 = 9" still reads as 9. Thousands grouped by a space ("12 345") and the school form
# of scientific notation ("1.2 × 10^3") are one value, and so is a mixed number ("9 1/2", "9½"). Calculator
# notation ("1e3") is still more than one value.

_NEGATION_RE = re.compile(
    r"\b(?:not|never|neither|nor|cannot)\b"  # a negating word
    r"|n['’]t\b"  # isn't, don't, can't, won't ...
    r"|[≠≤≥]|!=|<(?!-)|(?<![-=])>"  # not-equal and ordering signs (but not an arrow)
    r"|[±∓]|\+\s*/\s*-"  # plus-or-minus: two values at once
    r"|(?<![\w)])(?:\+-|-\+)"  # "+-9" is, "y=2x+-3" is not
    r"|\bplus[\s-]+(?:or[\s-]+)?minus\b",
    re.IGNORECASE,
)
# The minus sign and the dashes people type or paste in its place.
_MINUS_SIGNS = str.maketrans({c: "-" for c in "−‒–—―﹣－"})
# The exponent of a unit: one or two digits right after a letter, directly or through ^ / **
# ("cm^2", "m2", "s^-1", "m^(-2)"); a bracket counts only after the ^, so "prime(10**9)" is two numbers. The
# letter may be the end of a unit written in LaTeX ("\text{cm}^2"): the "}" it closes must be a text macro's
# (see `_closes_a_unit_name`), so "{5}^2" is still two numbers.
# Anchored on the digit through a lookbehind so that a long run of letters cannot make the match quadratic.
_UNIT_EXPONENT_RE = re.compile(
    r"(?<=[^\W\d_]|\})(?:[ \t]*(?:\^|\*\*)[ \t]*(?:\(-?\d{1,2}\)|-?\d{1,2})|-?\d{1,2})(?!\d|\.\d)"
)
_UNIT_NAME_MACROS = ("\\text", "\\textrm", "\\mathrm", "\\mbox", "\\operatorname")
# "2 dp", "to 1 d.p.", "3 s.f.", "two decimal places", "2 significant figures", "3 sig figs".
_COUNT_WORDS = "one|two|three|four|five|six|seven|eight|nine|ten"
_PRECISION_NOTE_RE = re.compile(
    r"(?<![\w.])(?:\d+|" + _COUNT_WORDS + r")[ \t]*"
    r"(?:d\.?[ \t]?p\.?|decimal[ \t]+(?:places?|points?)|s\.?[ \t]?f\.?|sig(?:nificant|\.)?[ \t]*(?:fig(?:ure)?s?\.?|digits?))"
    r"(?![A-Za-z])",
    re.IGNORECASE,
)
# A bracketed remark with no digit in it: "(no units)", "(approx.)", "(rounded)", and what a precision note leaves behind.
_DIGITLESS_BRACKET_RE = re.compile(r"\([^()\d]*\)")
# A minus that is a sign: not right after a word character, a digit, "." or ")", and followed by the number
# directly or through a space and/or a currency sign.
_SIGN_MINUS_RE = re.compile(r"(?<![\w.)])-[ \t]*(?:[$£€¥][ \t]*)?(?=\.?\d)")
# ".5" is 0.5, not 5.
_LEADING_POINT_RE = re.compile(r"(?<![\w.])\.(?=\d)")
_NUMBER_OR_WORD_RE = re.compile(r"\d+(?:\.\d+)?|[A-Za-z']+")
_COUNTING_WORDS = frozenset(NUMBER_WORDS) - {"point"}


def _first_number_end(text: str) -> int:
    """Where the first number in `text` (digits or a number word) ends, or its length when there is none."""
    for match in _NUMBER_OR_WORD_RE.finditer(text):
        word = match.group(0)
        if word[0].isdigit() or word.lower() in _COUNTING_WORDS:
            return match.end()
    return len(text)


def _closes_a_unit_name(text: str, end: int) -> bool:
    """Whether the "}" just before `end` closes a text macro's argument: "\\text{cm}", "\\mathrm{ m }".
    Looks back a few dozen characters at most, so a reply of nothing but braces costs a bounded step each."""
    window = text[max(0, end - 41) : end - 1]
    opening = window.rfind("{")
    return opening >= 0 and "}" not in window[opening:] and window[:opening].rstrip().endswith(_UNIT_NAME_MACROS)


def _drop_unit_exponent(match: re.Match, first_number_end: int) -> str:
    text, start = match.string, match.start()
    if start < first_number_end:
        return match.group(0)  # "AUD9", "Rs9": the digits are the number, not the exponent of a unit
    # "1e3" and "2.5E-4" are scientific notation, which this grader does not read: leave the exponent
    # in place so the reply counts as two numbers instead of quietly becoming 1 or 2.5.
    if text[start - 1] in "eE" and start >= 2 and text[start - 2].isdigit():
        return match.group(0)
    if text[start - 1] == "}" and not _closes_a_unit_name(text, start):
        return match.group(0)  # "{5}^2" is not a unit
    return ""


# "12 345" and "1 234 567.5": thousands grouped by a space, the way the SI and the Australian style manual
# write them. Groups after the first are exactly three digits, and nothing numeric touches either end, so
# "9 1/2", "3 45" and "12 3456" are still two values.
_SPACE_GROUP_RE = re.compile(r"(?<![\d.,/])\d{1,3}(?:[ \u00a0\u2009\u202f]\d{3})+(?:\.\d+)?(?![\d/]|\.\d)")
_SUPERSCRIPT_DIGITS = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻", "0123456789+-")
# "1.2 × 10^3", "1.2 x 10^-4", "1.2*10**3", "1.2 × 10⁻⁴": the school form of scientific notation. The power
# is at most two digits, so a reply cannot ask for an enormous number. Calculator "1.2e3" is not this form
# and stays two values (see `_drop_unit_exponent`).
_TIMES_TEN_RE = re.compile(
    r"(?<![\w.])(-?\d+(?:\.\d+)?)[ \t]*[×xX*·⋅][ \t]*10[ \t]*"
    r"(?:(?:\^|\*\*)[ \t]*(?:\([ \t]*([+-]?\d{1,2})[ \t]*\)|\{[ \t]*([+-]?\d{1,2})[ \t]*\}|([+-]?\d{1,2}))"
    r"|([⁺⁻]?[⁰¹²³⁴⁵⁶⁷⁸⁹]{1,2}))(?![\d⁰¹²³⁴⁵⁶⁷⁸⁹.])"
)


def _times_ten_as_decimal(match: re.Match) -> str:
    """The plain decimal a "a × 10^n" match stands for, written out in full: "1.2 × 10^-3" is "0.0012"."""
    power = next(group for group in match.groups()[1:] if group is not None)
    value = Decimal(match.group(1)).scaleb(int(power.translate(_SUPERSCRIPT_DIGITS)))
    return format(value, "f")


# "2 1/2" and "-2 1/2": a mixed number, whose fraction is proper. "2 5/3" and "9 1/2 7" are not one.
_MIXED_NUMBER_RE = re.compile(r"(?<![\w./)])(-[ \t]*)?(\d+)[ \t]+(\d+)[ \t]*/[ \t]*(\d+)(?![\w/.])")


def _mixed_number_as_fraction(match: re.Match) -> str:
    sign, whole, numerator, denominator = match.groups()
    if not 0 < int(numerator) < int(denominator):
        return match.group(0)
    value = int(whole) + Fraction(int(numerator), int(denominator))
    return f"{'-' if sign else ''}{value.numerator}/{value.denominator}"


def _number_text(given: str) -> str:
    """The part of a typed number reply that states the value, tidied for `extract_values`: the side after
    the last "=", no thousands commas or spaces, "a × 10^n" written out, one minus sign, no precision note, a
    sign attached to its number and a leading zero on a decimal that has none."""
    text = _strip_comma_groups(_value_side(given)).translate(_MINUS_SIGNS)
    text = _VULGAR_FRACTION_RE.sub(_vulgar_fraction, text)
    text = _MIXED_NUMBER_RE.sub(_mixed_number_as_fraction, text)
    text = _SPACE_GROUP_RE.sub(lambda match: re.sub(r"\s", "", match.group(0)), text)
    text = _TIMES_TEN_RE.sub(_times_ten_as_decimal, text)
    text = _PRECISION_NOTE_RE.sub(" ", text)
    text = _SIGN_MINUS_RE.sub("-", text)
    return _LEADING_POINT_RE.sub("0.", text)


def _read_values(text: str) -> list:
    """Every number in tidied text, unit exponents aside."""
    first_end = _first_number_end(text)
    text = _UNIT_EXPONENT_RE.sub(lambda match: _drop_unit_exponent(match, first_end), text)
    return extract_values(text) if text.strip() else []


def _typed_number_value(given: str) -> Fraction | None:
    """The single number a typed reply states, or None when it states none or more than one value
    (or negates or hedges it). See the block comment above for the reading."""
    if _NEGATION_RE.search(given or ""):
        return None
    text = _number_text(given)
    stripped = _DIGITLESS_BRACKET_RE.sub(" ", text)
    # A remark with no digit is dropped, unless that leaves nothing: "(nine)" is the number nine.
    for candidate in (stripped, text) if stripped != text else (text,):
        try:
            values = _read_values(candidate)
        except Exception:  # noqa: BLE001 - normalisation is best-effort
            logger.warning("extract_values failed for given=%.200r", given, exc_info=True)
            return None
        if values:
            return values[0] if len(values) == 1 else None
    return None


def _reply_negates_the_answer(question, given: str) -> bool:
    """Whether the reply says "not" (or "no", "isn't", "!=", "+-" ...) and the answer itself does not.

    The pair and list graders count the numbers in a reply, which cannot tell "(28, 24)" from "not (28, 24)":
    both hold exactly the right values. A reply that negates is not the answer. When the canonical text
    uses such a word or sign itself ("no real roots", "x > 2") the check stands down.
    """
    return bool(_NEGATION_RE.search(given or "")) and not _NEGATION_RE.search(
        getattr(question, "answer_text", "") or ""
    )


@_multi_part_aware
def grade_number(question, given: str) -> GradeResult:
    """Typed numeric answers are graded against a target/tolerance derived
    from `question.answer_text`'s own displayed precision (see
    `_typed_number_target`), never by float `==`/subtraction (`0.1 + 0.2 !=
    0.3`) -- comparison is entirely in `Fraction` arithmetic, the same
    discipline `grade_fraction` uses. This only applies when `question.answer`
    is a plain `int`/`float`; anything else (e.g. a dict/list-shaped answer
    reachable via this kind for a generator this fix hasn't audited) falls
    through to the pre-existing `check_answer` delegation unchanged."""
    canonical = question.answer_text
    answer = _as_exact_fraction(question.answer)
    if isinstance(answer, sympy.Integer):
        answer = int(answer)
    if isinstance(answer, dict) and {"p", "q"} <= set(answer):
        # A `number` skill whose generator answers with an exact fraction
        # (definite-integral's "16/21"). Falling through to `check_answer`
        # graded it with that function's ~0.05 transcript slack, so "16/22"
        # -- a different number, 0.035 away -- was accepted for an answer that
        # is exact. Exact values are graded exactly, as `fraction` already is.
        return grade_fraction(question, given)
    if isinstance(answer, str) and _NUMERIC_LITERAL_RE.fullmatch(answer.strip()):
        # A generator that writes its numeric answer as text ("10", "3.5": 22 questions across eight senior
        # skills) was graded by `check_answer`'s spoken reading, where "not 10" is correct. It is a number
        # like any other, so it takes the number path below.
        # The target is worked out from the number the text spells; the question itself is left as it is.
        text = answer.strip()
        answer = float(text) if "." in text else int(text)
    if isinstance(answer, bool) or not isinstance(answer, (int, float)):
        return _check_answer_kind(question, given)

    given_normalised = _normalised_values(given)
    resolved = _typed_number_target(question, answer)
    # One value per reply, and it must be the answer: see `_typed_number_value`.
    value = _typed_number_value(given)
    correct = value is not None and (value == resolved.target or abs(value - resolved.target) <= resolved.tolerance)
    if correct and not _is_rounded_as_asked(question, given):
        correct = False
    if correct and not unit_is_acceptable(question, given):
        correct = False  # "30 kg" for "Convert 3 cm to mm": see mathgen.grading.units
    return GradeResult(correct=correct, canonical=canonical, given_normalised=given_normalised)


def _ratio_form_is_acceptable(canonical: str, given: str) -> bool:
    """Whether a ratio-notation answer is written in the form the answer asks for.

    Ratio notation carries form as well as value. "Simplify the ratio 96:40"
    has canonical "12:5" and offers the unsimplified "96:40" as a distractor;
    graded purely by value those are the same number, so the distractor was
    accepted on every question of the skill.

    The by-value contract for fractions is
    untouched: this only applies when BOTH the canonical and the learner's
    answer are written a:b, which among active skills is ratio-rate-proportion
    alone. A learner who answers "12/5" is still accepted by value.
    """
    canonical_ratio = _RATIO_RE.match((canonical or "").strip())
    given_ratio = _RATIO_RE.match((given or "").strip())
    if canonical_ratio is None or given_ratio is None:
        return True
    if math.gcd(int(canonical_ratio.group(2)), int(canonical_ratio.group(3))) != 1:
        return True  # the canonical is not in lowest terms; nothing to require
    return math.gcd(int(given_ratio.group(2)), int(given_ratio.group(3))) == 1


_WRITTEN_FRACTION_RE = re.compile(r"(-?\d+)\s*/\s*(\d+)")


def _as_exact_fraction(answer):
    """An exact rational answer (a sympy Rational, a `fractions.Fraction`) as the {"p", "q"} form the fraction
    path reads. Left as it was, it fell through to the spoken reading, which takes neither, and every reply was
    marked wrong, its own canonical included."""
    if isinstance(answer, Fraction) or (isinstance(answer, sympy.Rational) and not isinstance(answer, sympy.Integer)):
        return {"p": int(answer.numerator), "q": int(answer.denominator)}
    return answer


@_multi_part_aware
def grade_fraction(question, given: str) -> GradeResult:
    """Typed fraction answers are graded *exactly*: "0.8" is wrong for 3/4 even though it's within the ~0.05
    tolerance `mathgen.grading.spoken.check_answer` accepts for spoken transcripts. That loose tolerance is
    unchanged for the transcript path -- callers grade a transcript with `mathgen.grading.spoken.check_answer`
    directly and never reach this function for one."""
    canonical = question.answer_text
    answer = question.answer
    if (
        isinstance(answer, str)
        and (written := _WRITTEN_FRACTION_RE.fullmatch(answer.strip()))
        and int(written.group(2)) != 0
    ):
        # A generator that writes its fraction as text ("3/20") was graded by `check_answer`'s spoken
        # reading, where "not 3/20" is correct. It is the same exact fraction as the {"p", "q"} form.
        answer = {"p": int(written.group(1)), "q": int(written.group(2))}
    answer = _as_exact_fraction(answer)
    if isinstance(answer, dict) and {"p", "q"} <= set(answer):
        given_normalised = _normalised_values(given)
        parsed = _parse_typed_fraction(given)
        if parsed is None and _label_fits(question, canonical, _label_name(given)):
            # "x = 50/7" for a canonical shown as "x = 50/7", or "x = 3" for a bare "3" when the question is about
            # x ("Solve 3^x = 27"). "y = 50/7" is wrong for x, and so is a label the question never names.
            given = _strip_label(given)
            parsed = _parse_typed_fraction(given)
        if parsed is None:
            return GradeResult(correct=False, canonical=canonical, given_normalised=given_normalised)
        expected = Fraction(answer["p"], answer["q"])
        correct = (
            parsed == expected
            and _ratio_form_is_acceptable(canonical, given)
            and _in_lowest_terms_if_asked(question, given)
        )
        return GradeResult(correct=correct, canonical=canonical, given_normalised=given_normalised)
    return _check_answer_kind(question, given)


# "in lowest terms", "in simplest form", "simplify", "fully simplified": the question asks for the fraction
# itself to be reduced, so "6/8" is not an answer to "Write 6/8 in lowest terms" even though it equals 3/4.
_LOWEST_TERMS_RE = re.compile(
    r"\b(?:lowest\s+terms|simplest\s+form|simplif(?:y|ied)|fully\s+simplified|reduced?\s+form)\b", re.IGNORECASE
)
_WRITTEN_FRACTION_PART_RE = re.compile(r"(\d+)\s*/\s*(\d+)\s*$")


def _in_lowest_terms_if_asked(question, given: str) -> bool:
    """Whether a fraction reply is reduced, when the question asks for lowest terms. A reply that writes no a/b
    (a whole number, a decimal) is judged by value alone, as before."""
    if not _LOWEST_TERMS_RE.search(getattr(question, "text", "") or ""):
        return True
    written = _WRITTEN_FRACTION_PART_RE.search(_tidy_fraction_text(_strip_label(given)))
    if written is None:
        return True
    numerator, denominator = int(written.group(1)), int(written.group(2))
    return denominator != 0 and math.gcd(numerator, denominator) == 1


_PAIR_TOLERANCE = 0.051  # mathgen.grading.spoken's transcript tolerance: now only the fallback


# -- pair and list: per-component stated precision ---------------------------
#
# Both graders compared every component with the flat ±0.051 the spoken-
# transcript path uses, whatever precision the question asked for, so
# "volume = 12.39" was accepted for "volume = 12.34" and a proportion interval
# could be five percentage points out.
#
# A typed component is now graded the way `grade_number` grades a typed
# number: against the precision its own canonical displays. The canonical is
# split into one segment per component, and each segment's single numeric
# literal sets that component's target and tolerance -- half a unit in its
# last decimal place, exact for an integer or an a/b fraction. The literal is
# trusted only when it is a plausible rounding of the component's value; a
# segment with no literal, several literals, or an implausible one keeps the
# old ±0.051, so the tolerance can only ever tighten.

_COMPONENT_LITERAL_RE = re.compile(r"-?\d+(?:\.\d+)?(?:\s*/\s*\d+)?")
_LIST_SEPARATOR_RE = re.compile(r"\s*(?:,|;|\bor\b|\band\b)\s*")


@dataclass(frozen=True)
class _ComponentTarget:
    value: object  # the component's value, as the generator computed it
    target: Fraction | None  # the displayed literal, when trusted
    tolerance: Fraction | float

    def matches(self, given) -> bool:
        try:
            if given == self.value:
                return True
            if self.target is None:
                return abs(float(given) - float(self.value)) <= self.tolerance
            return abs(Fraction(given) - self.target) <= self.tolerance
        except (TypeError, ValueError, OverflowError):
            return given == self.value


def _as_fraction(value) -> Fraction | None:
    if isinstance(value, bool):
        return None
    try:
        return Fraction(str(value))
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def _component_target(segment: str | None, value) -> _ComponentTarget:
    """The target and tolerance for one component, from its canonical segment."""
    fallback = _ComponentTarget(value=value, target=None, tolerance=_PAIR_TOLERANCE)
    exact_value = _as_fraction(value)
    if segment is None or exact_value is None:
        return fallback
    literals = _COMPONENT_LITERAL_RE.findall(_strip_comma_groups(_value_side(segment)))
    if len(literals) != 1:
        return fallback
    literal = literals[0]
    if "/" in literal:
        numerator, denominator = (part.strip() for part in literal.split("/"))
        try:
            shown = Fraction(numerator) / Fraction(denominator)
        except (ValueError, ZeroDivisionError):
            return fallback
        if shown != exact_value:
            return fallback
        return _ComponentTarget(value=value, target=shown, tolerance=Fraction(0))
    shown = Fraction(literal)
    places = _decimal_places(literal)
    if abs(shown - exact_value) > _plausible_rounding_tolerance(places):
        return fallback
    return _ComponentTarget(value=value, target=shown, tolerance=_typed_number_tolerance(places))


def _canonical_segments(canonical: str, count: int) -> list[str | None]:
    """Split a pair or list canonical into one segment per component.

    Labelled ("x = 3, y = -6"), bracketed ("(63, 81)", "{6}") and worded
    ("x = -3 or x = 3") canonicals all split here. A count that does not match
    the answer's returns all None, which sends every component to the fallback.
    """
    text = _strip_outer_brackets(_strip_comma_groups(canonical or ""))
    segments = [seg for seg in _LIST_SEPARATOR_RE.split(text) if seg.strip()]
    if len(segments) != count:
        return [None] * count
    return segments


def _component_targets(canonical: str, values: list) -> list[_ComponentTarget]:
    segments = _canonical_segments(canonical, len(values))
    return [_component_target(seg, value) for seg, value in zip(segments, values)]


_EQUATION_ANSWER_RE = re.compile(r"^\s*[A-Za-z]\w*\s*=\s*[^,=]+$")


def _equation_pair_matches(canonical: str, given: str):
    """Compare two single-equation answers symbolically, or None if not applicable.

    A pair answer that is written as an equation ("y = -x + 7" for
    {"m": -1, "b": 7}) cannot be graded through `extract_values`: that is the
    spoken-transcript normaliser, and it reads "y = -x + 7" as [7]. It loses
    the implicit coefficient on x, so a learner was marked wrong on every
    question whose gradient was 1 or -1 (69 false rejects), and it loses the
    sign of the constant, so "y = -9x + 4" was accepted for "y = -9x - 4"
    (51 false accepts).

    Only single equations qualify -- the pattern excludes anything with a
    comma, so the component-wise pair answers this grader was written for
    ("x = 3, y = -6") take the unchanged path below.
    """
    if not _EQUATION_ANSWER_RE.match(canonical or ""):
        return None
    expected = _parse_expression(_strip_label(canonical))
    if expected is None:
        return None
    given = given or ""
    parts = given.split("=")
    if len(parts) == 2 and not _LABEL_RE.match(given):
        # Not "label = value": a rearrangement ("y - 3 = 0", "3 = y").
        canonical_label = canonical.split("=", 1)[0].strip()
        equivalent = _constant_line_equivalent(canonical_label, expected, parts)
        if equivalent is not None:
            return equivalent
    if not expected.free_symbols and _is_line_key_label(canonical):
        # A horizontal or vertical line ("y = 3", "x = -2") is an equation, not a number: the reply has to
        # be written as that equation. A bare "3" or the wrong axis ("x = 3" for "y = 3") is wrong.
        # Other single-equation keys ("lambda = 17") keep the label-stripped comparison below.
        reply_label = _LABEL_RE.match(given)
        if (
            not reply_label
            or reply_label.group(0).split("=")[0].strip().lower() != canonical.split("=", 1)[0].strip().lower()
        ):
            return False
    # Any other bare value ("17" for "lambda = 17") and any "label = value" stay accepted (the label is
    # stripped; see tests/test_grading_line_key_fixes.py).
    supplied = _resolve_constants(_parse_expression(_strip_label(given)), expected)
    if supplied is None:
        return False
    try:
        return symbolically_equal(expected, supplied)
    except (TypeError, ValueError):
        return None


def _is_line_key_label(canonical: str) -> bool:
    """True for a key labelled with an axis letter ("y = 3", "x = -2")."""
    return canonical.split("=", 1)[0].strip().lower() in ("x", "y")


def _constant_line_equivalent(label: str, expected, parts: list[str]) -> bool | None:
    """Rearrangements of a constant line key (`y = 3`): `y - 3 = 0`, `3 = y`.

    Equal up to sign only, never a scaled multiple (`2y = 6`). Only keys whose
    value side is a pure number take this path; any other key returns None
    (not applicable).
    """
    if expected.free_symbols:
        return None
    lhs, rhs = _parse_expression(parts[0]), _parse_expression(parts[1])
    if lhs is None or rhs is None:
        # A side the parser cannot read (a Python keyword such as `lambda` as the label,
        # or stray punctuation): this branch has no verdict, so the original path decides.
        return None
    try:
        key = sympy.Symbol(label) - expected
        diff = lhs - rhs
        return bool(symbolically_equal(diff, key) or symbolically_equal(diff, -key))
    except (TypeError, ValueError):
        return False


# -- pair: labelled-order independence -----------------------
#
# `grade_pair`'s positional comparison below is right for an UNLABELLED pair
# ("(-5, 5)" for quadratic-vertex) -- position is the only thing that carries
# meaning there. But most active pair skills answer with two LABELLED
# segments ("x = -2, y = 5"), where the labels carry the meaning and order
# does not: "y = 5, x = -2" is the same answer, marked wrong. This block
# detects that shape and matches by label name instead; everything that
# isn't this shape (either side) falls through to the unchanged positional
# comparison, unmodified, below.

_LABEL_PREFIX_RE = re.compile(r"^\s*([^\W\d]\w*)\s*=")


def _split_labelled_segments(text: str) -> list[tuple[str, str]] | None:
    """Split a comma-separated pair answer into `(label, raw segment)` pairs,
    or None if `text` is not made entirely of `label = value` segments.

    Unicode-aware: a label is any leading identifier that is not a digit
    (`[^\\W\\d]\\w*`), not an ASCII-only `[A-Za-z_]` class, so a Greek-letter
    label ("δx = 4") is recognised -- see
    senior-use-the-leibniz-notation-delta-x-and-delta, whose canonical answer
    is literally "δx = 4, δy = -32". A wholly bare answer ("-2, 5")
    -- which starts with a digit or sign, not a label -- returns None so the
    caller falls back to the positional comparison unchanged.

    `_strip_comma_groups` runs first so a thousands-grouped numeral in a
    segment's value ("R_A = 12,345 kN") is never mistaken for an extra,
    unlabelled segment.
    """
    text = _strip_comma_groups(text or "")
    if not text:
        return None
    segments = [s.strip() for s in text.split(",")]
    if not segments or any(not seg for seg in segments):
        return None
    parsed: list[tuple[str, str]] = []
    for seg in segments:
        m = _LABEL_PREFIX_RE.match(seg)
        if not m:
            return None
        parsed.append((m.group(1), seg))
    return parsed


def _label_matches(label: str, wanted: str) -> bool:
    """Whether a reply's label can stand for the canonical's: the same name in another case, without its
    subscript ("R" or "RA" for "R_A"), or the name spelled out or shortened ("height" for "h")."""
    given, target = label.casefold(), wanted.casefold()
    return (
        given == target
        or given == target.replace("_", "")
        or given == target.split("_")[0]
        or given.startswith(target)
        or target.startswith(given)
    )


def _labelled_pair_verdict(canonical: str, given: str, expected_values: list) -> bool | None:
    """Order-independent verdict for a LABELLED pair answer, or None if this
    shape doesn't apply (the caller then uses the unchanged positional path).

    Matches the learner's segments to the canonical's **by label name**,
    never by position and never by bare value -- the pinned
    `test_grade_pair_rejects_swapped_values` requires "x = 3, y = -6" to stay
    wrong for {"x": -6, "y": 3} (same label order, values swapped between
    labels), which a values-only/multiset comparison would wrongly accept
    (`{3, -6} == {-6, 3}`), so every comparison here is label-to-value.

    Returns None when the canonical itself doesn't parse into exactly
    `len(expected_values)` labelled segments -- an unlabelled/coordinate pair
    (quadratic-vertex's "(-5, 5)"), or a single-equation pair already routed
    through `_equation_pair_matches` before this is ever called. That is what
    keeps a positional swap of an unlabelled pair rejected: this whole branch
    never engages for it.

    Also returns None when the canonical *is* labelled but the learner's
    `given` is not (e.g. the learner dropped the labels and typed bare values
    in the canonical's order) -- the caller falls back to positional, which
    is what keeps that already-working case working unchanged.

    This is decisive (True or False) in exactly one case: `given`'s label
    set, with NO repeated label on either side, is exactly equal (as a set)
    to the canonical's. That is the shape a learner types when they reorder the pair ("y = 5, x = -2"
    for canonical "x = -2, y = 5") and is compared label-to-value.

    Every other shape returns None so the caller falls back to the
    unmodified positional comparison below, exactly as before this
    function existed -- including a `given` with a REPEATED label (e.g.
    "x = -2, x = 5", or beam-reactions-point-load's dropped-subscript
    "R = 7.5 kN, R = 22.5 kN" against canonical "R_A = 7.5 kN, R_B = 22.5
    kN") and a `given` whose (duplicate-free) label set merely *differs*
    from the canonical's (different names, case, a missing underscore, a
    symbol used for a spelled-out name, or a segment count mismatch because a
    semicolon/newline was used instead of a comma).

    An earlier revision of this function hard-rejected the repeated-label
    case outright (no positional fallback), reasoning that a positional
    comparison could luck into accepting a repeated-but-wrong labelling
    whenever it happened to preserve the canonical's value order. That
    reasoning is superseded: it created a real false-reject regression on a
    shipping skill (beam-reactions-point-load's dropped-subscript case
    above, which is accepted at HEAD and must stay accepted), and falling
    back to positional for a repeated label cannot introduce any NEW verdict
    relative to HEAD -- it is the exact same positional comparison HEAD
    already ran on that input before this function existed. So the only new
    verdict this function can produce relative to HEAD is the
    exact-label-match accept/reject described above; every other input,
    duplicate-labelled or mismatched, falls through unchanged.
    """
    canonical_segments = _split_labelled_segments(canonical)
    if canonical_segments is None or len(canonical_segments) != len(expected_values):
        return None
    label_order = [label for label, _ in canonical_segments]
    if len(set(label_order)) != len(label_order):
        return None  # malformed/duplicate-labelled canonical: not this shape

    given_segments = _split_labelled_segments(given) if given else None
    if given_segments is None:
        return None  # learner dropped the labels -- fall back to positional

    given_labels = [label for label, _ in given_segments]
    if any(not any(_label_matches(label, wanted) for wanted in label_order) for label in given_labels):
        return False  # "a = 3, b = -6" for "x = 3, y = -6": a label the answer does not have
    if len(set(given_labels)) != len(given_labels) or set(given_labels) != set(label_order):
        return None  # a repeated label on the learner's side, or a label set
        # that merely differs from the canonical's -- either way, fall back to the unmodified
        # positional comparison below. This is EXACTLY what HEAD did for both shapes of input, so
        # falling back here can never produce a new false accept or a new false reject relative to
        # HEAD -- see beam-reactions-point-load's "R =..., R =..." (dropped-subscript duplicate) for
        # the regression this supersedes.

    # This zip is safe specifically because the CANONICAL text itself is
    # already known to be graded correct against `answer` by the existing
    # positional comparison below (the canonical always is) -- so
    # canonical-text label order and `answer`-dict value order are already
    # known to correspond; this reads that correspondence off by name rather
    # than inventing a new one.
    expected_by_label = dict(zip(label_order, _component_targets(canonical, expected_values)))
    for label, segment in given_segments:
        try:
            segment_values = extract_values(_value_side(segment))
        except Exception:  # noqa: BLE001
            logger.warning("extract_values failed for segment=%.200r", segment, exc_info=True)
            segment_values = []
        if len(segment_values) != 1:
            return False
        if not expected_by_label[label].matches(segment_values[0]):
            return False
    return True


# A vector in component form, "-4i - 3j + 0k" or "i - 2j": each unit vector at most once, in any order, a
# missing one being 0. Read as the tuple of its components. Without this a spaced reply was read as a
# subtraction ("- 3j" lost its sign), while the same reply typed without spaces was accepted.
_COMPONENT_TERM_RE = re.compile(r"\s*([+-]?)\s*(\d+(?:\.\d+)?(?:/\d+)?)?\s*\*?\s*([ijk])(?![A-Za-z])")


def _component_form(given: str, count: int) -> str | None:
    text = (given or "").strip()
    # Each term is matched where the last one ended, so the reading is linear; a vector is a few dozen characters.
    if not text or count not in (2, 3) or len(text) > 200 or not any(unit in text for unit in "ijk"):
        return None
    coefficients: dict[str, str] = {}
    position = 0
    while position < len(text):
        match = _COMPONENT_TERM_RE.match(text, position)
        if match is None or (position and not match.group(1)) or match.group(3) in coefficients:
            return None
        sign, number, unit = match.groups()
        coefficients[unit] = (sign if sign == "-" else "") + (number or "1")
        position = match.end()
        if not text[position:].strip():
            break
    if not coefficients:
        return None
    units = "ijk"[:count]
    if any(unit not in units and coefficients[unit].lstrip("-") not in ("0", "0.0") for unit in coefficients):
        return None
    return "(" + ", ".join(coefficients.get(unit, "0") for unit in units) + ")"


@_multi_part_aware
def grade_pair(question, given: str) -> GradeResult:
    """Unlike `check_answer`'s dict branch (which accepts the two values in
    either order, since it only checks presence), a pair answer kind
    compares positionally in dict order -- "x = 3, y = -6" is wrong for
    {"x": -6, "y": 3} even though both magnitudes appear somewhere in the
    text, because component order carries meaning for an UNLABELLED pair.

    For a LABELLED pair, order carries no meaning at all -- only the labels
    do -- so `_labelled_pair_verdict` is tried first and, when
    decisive, used instead of the positional comparison below."""
    canonical = question.answer_text
    answer = question.answer
    if not isinstance(answer, dict):
        return _check_answer_kind(question, given)
    expected_values = list(answer.values())
    given = _component_form(given, len(expected_values)) or given
    given_values = _normalised_values(given)
    if _reply_negates_the_answer(question, given):
        return GradeResult(correct=False, canonical=canonical, given_normalised=given_values)
    equation_verdict = _equation_pair_matches(canonical, given)
    if equation_verdict is not None:
        return GradeResult(correct=equation_verdict, canonical=canonical, given_normalised=given_values)
    labelled_verdict = _labelled_pair_verdict(canonical, given, expected_values)
    if labelled_verdict is not None:
        return GradeResult(correct=labelled_verdict, canonical=canonical, given_normalised=given_values)
    try:
        values = extract_values(_strip_comma_groups(given)) if given else []
        if given and len(values) != len(expected_values):
            # "316,156" for a pair is two values typed without a space, not
            # one grouped numeral: the pair's own length decides.
            values = extract_values(given)
    except Exception:  # noqa: BLE001
        logger.warning("extract_values failed for given=%.200r", given, exc_info=True)
        values = []
    targets = _component_targets(canonical, expected_values)
    correct = len(values) == len(expected_values) and all(target.matches(v) for v, target in zip(values, targets))
    return GradeResult(correct=correct, canonical=canonical, given_normalised=given_values)


@_multi_part_aware
def grade_list(question, given: str) -> GradeResult:
    """Order-insensitive, but the set of values must match exactly.

    Delegating to `check_answer` accepted any SUPERSET of the answer, because
    that function only checks each expected value is *present* and never looks
    for extras: "1, 17, 18" graded correct for "1, 17", and "{1, 2, 6}" for
    "{1, 6}". 145 false accepts across list-factors and finite-set-operations,
    where naming a value that is not a factor, or not in the set, is precisely
    the error the skill assesses.

    Every active list skill answers with a list of numbers, so values are
    compared numerically and count-sensitively (a repeated value must be
    repeated), each to the precision its canonical displays (see
    `_component_targets`), as the pair grader does.
    """
    expected = question.answer
    if not isinstance(expected, (list, tuple)):
        return _check_answer_kind(question, given)
    if _reply_negates_the_answer(question, given):
        return GradeResult(correct=False, canonical=question.answer_text, given_normalised=_normalised_values(given))
    try:
        values = extract_values(given) if given else []
    except Exception:  # noqa: BLE001
        logger.warning("extract_values failed for given=%.200r", given, exc_info=True)
        values = []
    canonical = question.answer_text
    given_normalised = _normalised_values(given)
    if len(values) != len(expected):
        return GradeResult(correct=False, canonical=canonical, given_normalised=given_normalised)

    remaining = _component_targets(canonical, list(expected))
    for value in values:
        for index, candidate in enumerate(remaining):
            if candidate.matches(value):
                del remaining[index]
                break
        else:
            return GradeResult(correct=False, canonical=canonical, given_normalised=given_normalised)
    return GradeResult(correct=not remaining, canonical=canonical, given_normalised=given_normalised)


# -- ordered_list -------------------------------------------------------
#
# Deliberately does not reuse `check_answer`/`extract_values`: `grade_list` above is correct for its
# 12 existing (order-insensitive) callers and must not change; and `extract_values`' tokenizer
# splits on every comma as a value separator, which would misparse a single comma-grouped numeral
# ("12,345") into two values. This grader does its own splitting instead.

_ARROW_RE = re.compile(r"\s*(?:->|→)\s*")
# Bounded on both sides, so "316,1564" (no group of exactly three) and the
# tail of "12,34,567" are not read as grouped numerals.
_COMMA_GROUP_RE = re.compile(r"(?<![\d,.])\d{1,3}(?:,\d{3})+(?:\.\d+)?(?!\d)")
_TOP_DELIM_RE = re.compile(r"[,;\n]+")
# Deliberately does NOT include ">"/"＞": a descending chain is a different claim from what an
# ascending-order skill asks for, so it stays unrecognised and rejected; see
# `_split_ordered_items`'s docstring for the walkthrough.
_LT_RE = re.compile(r"\s*[<＜]\s*")


def _strip_comma_groups(text: str) -> str:
    """Remove the internal commas from any thousands-grouped numeral in
    `text`, leaving every other comma untouched.

    Shared by `grade_number` and `grade_pair` as well as
    `_split_ordered_items` below, so all three response kinds treat "1,234"
    the same way. Only a `_COMMA_GROUP_RE` match loses its commas -- a comma
    that is not part of an exactly-three-digit group (an item separator in
    "1, 234", or the malformed grouping in "12,34,567") is left alone, so
    `grade_pair`'s item-separating commas and `_COMMA_GROUP_RE`'s own
    strictness are both preserved.
    """
    return _COMMA_GROUP_RE.sub(lambda m: m.group(0).replace(",", ""), text or "")


# A leading ordinal/numbering marker: "1. ", "1) ", "1 - " (a trailing space
# is required after the punctuation, so "1-2" -- a hyphenated value, not a
# numbered item -- is never mistaken for one).
_ORDINAL_PREFIX_RE = re.compile(r"^\s*\d+\s*(?:[.)]|-)\s+")
_PLACEHOLDER_RE = re.compile("\0(\\d+)\0")


_BRACKET_PAIRS = (("(", ")"), ("[", "]"), ("{", "}"), ("⟨", "⟩"))  # ⟨1, 2, 3⟩ is a vector too


def _strip_outer_brackets(given: str) -> str:
    """Drop one matching bracket pair wrapping the whole answer.

    A sequence answer is often shown bracketed -- permutation-composition's
    canonical is "(1, 2, 3)" -- and without this the splitter yields "(1" and
    "3)", which match no expected item, so every learner who typed the answer
    on screen was marked wrong.
    """
    text = (given or "").strip()
    for opening, closing in _BRACKET_PAIRS:
        if len(text) >= 2 and text.startswith(opening) and text.endswith(closing):
            return text[1:-1].strip()
    return text


def _split_ordered_items(given: str) -> list[str]:
    """Split a typed ordered-list answer into its component items.

    One matching bracket pair around the whole answer is removed first, so a
    sequence may be written "(1, 2, 3)" as well as "1, 2, 3".

    Delimiter policy: a comma, semicolon, newline, or arrow (``->``/``→``) between
    items is always a separator. A run of internal whitespace is *only*
    treated as a separator when the whole answer contains no comma/newline/
    arrow anywhere (e.g. "Monday Tuesday Wednesday" typed with no commas) --
    once any comma/newline/arrow is present, whitespace inside an item is
    left alone, so "1 - Monday, 2 - Tuesday" keeps "1 - Monday" as one item
    rather than splitting it into three tokens on the internal spaces.

    "<"/"＜" is also a separator -- a learner may write an
    ordering the way mathematics does ("0.3 < 0.31 < 0.4") instead of with
    commas -- but ONLY when it is the SOLE top-level delimiter in the
    answer. If the (already comma-group-protected) text contains BOTH a
    "<"/"＜" and a top-level comma/newline (the latter including one
    produced by the arrow substitution above), the two notations are mixed
    and the answer is not a coherent ordering statement at all -- "1 < 2, 3"
    and "1 -> 2 < 3" are rejected outright (this returns `[]`, which can
    never accidentally length-match a real >=1-item expected answer, so
    `grade_ordered_list`'s existing length check does the rejecting) rather
    than being guessed at. Deliberately does NOT recognise ">"/"＞" at all --
    see `_LT_RE`'s comment for why -- so "1 < 2 > 3" needs no dedicated
    mixed-delimiter handling: it has no comma, so "<" is substituted for
    "\n" same as arrow, giving "1\n2 > 3", which splits into the two items
    "1" and "2 > 3" and fails length/parse the ordinary way.

    This assumes each item is a single numeral or a single word/short
    phrase with no comma or ambiguous internal delimiter of its own -- it
    does not attempt to distinguish a comma that is part of a multi-word
    item's label from a comma that separates two items. That is sufficient
    for a numeral sequence or a short-label sequence (this function's intended
    users); a future skill needing comma-bearing multi-word items would
    need a different splitting rule.

    A comma that is part of a thousands-grouped numeral ("12,345") is
    protected before delimiter-splitting so it is never mistaken for an
    item separator -- this is exactly the bug `mathgen.grading.spoken`'s tokenizer
    would introduce if it were reused here (see module docstring).

    KNOWN LIMITATION: the thousands-grouping protection is purely
    syntactic (any run of ", ddd" groups looks like one grouped numeral),
    so a *sequence of separate 3-digit items typed with no space after the
    comma* is swallowed as a single value -- "100,200,300" (intended as
    three items, 100/200/300, e.g. a skip-counting-by-100s answer) parses
    as the one number 100200300, not three items. "100, 200, 300" (a space
    after each comma) is unaffected and splits correctly, because the
    grouping regex requires the digits to immediately follow the comma. A
    skill whose expected items are exactly-3-digit numerals should word its
    question/placeholder to encourage a space after each comma, or avoid
    this grader, until/unless this is worth solving generally."""
    given = _strip_outer_brackets(given)
    text = _ARROW_RE.sub("\n", given or "")
    protected: list[str] = []

    def _protect(match: re.Match) -> str:
        protected.append(match.group(0))
        return f"\0{len(protected) - 1}\0"

    text = _COMMA_GROUP_RE.sub(_protect, text)

    # "<"/"＜" is a delimiter only when it is the SOLE top-level delimiter -- see the docstring
    # above. Checked here, after outer-bracket stripping, arrow-to-newline conversion and
    # thousands-grouping protection all already ran, so a top-level comma/newline at this point is
    # genuinely item-separating (never a protected numeral's internal comma, and never an arrow that
    # hasn't already become a newline).
    if _LT_RE.search(text):
        if _TOP_DELIM_RE.search(text):
            return []  # mixed "<"/comma-or-newline chain: not an ordering
        text = _LT_RE.sub("\n", text)

    if _TOP_DELIM_RE.search(text):
        raw_items = _TOP_DELIM_RE.split(text)
    else:
        raw_items = text.split()

    def _restore(match: re.Match) -> str:
        return protected[int(match.group(1))]

    items = []
    for raw in raw_items:
        raw = raw.strip()
        if not raw:
            continue
        items.append(_PLACEHOLDER_RE.sub(_restore, raw))
    return items


def _strip_ordinal_prefixes(items: list[str]) -> list[str]:
    """Strip a leading "1. "/"1) "/"1 - " numbering marker from every item,
    but only when *every* item carries one -- so a bare numeric answer like
    "1" (the intended value for a numeric-sequence question, not a numbered
    label) is never stripped just because it happens to look like the start
    of a numbered item."""
    matches = [_ORDINAL_PREFIX_RE.match(item) for item in items]
    if items and all(matches):
        return [item[m.end() :].strip() for item, m in zip(items, matches)]
    return items


def _parse_numeric_item(item: str) -> Fraction | None:
    # Thousands-grouping commas are stripped here too (belt-and-braces --
    # normally already resolved by `_split_ordered_items`'s protection, but
    # cheap to tolerate again in case an item arrives with one some other
    # way) before parsing as an exact Fraction, so "03" == 3 and "12,345"
    # == 12345 the same way this file's other numeric comparisons tolerate
    # leading zeros/formatting.
    return _exact_fraction(item.strip().replace(",", ""))


def _numeric_items_equal(given_value: Fraction, expected) -> bool:
    if isinstance(expected, (int, Fraction)):
        return given_value == expected
    return abs(float(given_value) - float(expected)) < 1e-9


@_multi_part_aware
def grade_ordered_list(question, given: str) -> GradeResult:
    """Order-sensitive sequence grader. Unlike `grade_list`, which matches the multiset of values in
    any order, this compares positionally and requires an exact length match --
    any transposition is wrong, with no partial credit. (`grade_list` used to
    delegate to `check_answer`; that accepted supersets and was replaced.)

    Per-item comparison rule: a numeric expected item (int/float/Fraction)
    compares by parsed value (`_parse_numeric_item`, exact `Fraction`
    equality for int/Fraction, tiny-epsilon float equality otherwise); a
    string expected item compares case-insensitively after stripping
    whitespace, with *no* abbreviation table -- "Mon" does not match
    "Monday". A future skill needing common abbreviations should either
    spell the full word out in its question text, or this grader should
    grow a configurable alias list later; none is invented speculatively
    here."""
    canonical = question.answer_text
    expected = question.answer
    if not isinstance(expected, list):
        return GradeResult(correct=False, canonical=canonical, given_normalised=None)
    stripped = (given or "").strip()
    if stripped.startswith("<") and stripped.endswith(">"):
        given = "(" + stripped[1:-1] + ")"  # "<5, 6, -5>" is a vector, not an ordering sign
    if _reply_negates_the_answer(question, given):
        return GradeResult(correct=False, canonical=canonical, given_normalised=None)
    if all(isinstance(item, (int, float, Fraction)) for item in expected):
        # A vector answer may be typed in component form, "5i + 6j - 5k".
        given = _component_form(given, len(expected)) or given

    items = _strip_ordinal_prefixes(_split_ordered_items(given))
    given_normalised = ", ".join(items) if items else None

    if len(items) != len(expected):
        return GradeResult(correct=False, canonical=canonical, given_normalised=given_normalised)

    correct = True
    for given_item, expected_item in zip(items, expected):
        if isinstance(expected_item, (int, float, Fraction)):
            given_value = _parse_numeric_item(given_item)
            if given_value is None or not _numeric_items_equal(given_value, expected_item):
                correct = False
                break
        elif given_item.strip().lower() != str(expected_item).strip().lower():
            correct = False
            break
    return GradeResult(correct=correct, canonical=canonical, given_normalised=given_normalised)


# -- surd notation normalisation ---------------------------------
#
# Two spellings a learner will reasonably type reach sympy's own parser and
# are silently MIS-parsed -- no error, just a wrong expression compared
# against the canonical:
#
#   "√50"    -- sympy reads the unicode radical (U+221A) as a Python
#               identifier character, so it becomes a SYMBOL named "√", not a
#               call to `sqrt`: "√50" parses to "50*√", not "5*sqrt(2)".
#   "5sqrt2" -- `implicit_multiplication_application`'s identifier-splitting
#               only recognises "sqrt" as the function name when it is
#               immediately followed by "("; "sqrt2" with no parens is read
#               as an unrecognised identifier and split into single-letter
#               symbols instead, so "5sqrt2" parses to "10*q*r*s*t".
#
# Both are fixed the same way: rewrite the surd spelling to an explicit,
# already-parenthesised `sqrt(...)` call *before* it ever reaches sympy's
# parser -- at which point sympy's own (already-correct) handling of
# "5*sqrt(2)"-shaped text and implicit multiplication takes over unchanged
# (measured: "2sqrt(3)+sqrt(8)" already parses correctly once the argument
# has explicit parens; only the bare, unparenthesised spelling is the trap).
# Chosen over "make it fail to parse" (the other option)
# because the rewrite is unambiguous for every case considered
# and is a strict improvement: no existing generator emits either spelling
# (this module's own answer_text is always ascii "sqrt(...)", never "√",
# grepped repo-wide), so there is nothing for
# this rewrite to break -- it can only turn a silent wrong-parse into a
# correct one.
#
# This is a RAW-TEXT rewrite of the string sympy is about to see, run before
# any parsing -- shared by `_parse_expression` (below) and the raw-text
# `simplified_surd`/`rationalised` form checks in `grade_expression`, so a
# form check can never be fooled by a spelling `_parse_expression` itself
# already normalises.
_RADICAL_PAREN_RE = re.compile(r"√\s*\(")
# "√n" / "√x" -- bind the radical to the single immediate token (a digit run, or exactly one letter
# NOT itself followed by another letter) that follows it. The single-letter alternative's negative
# lookahead is deliberate: it stops this from reaching into the first letter of a longer identifier
# or function name -- without it, "√sin(x)" matched "√s" and rewrote to "sqrt(s)in(x)" (a new, worse
# mis-parse this change would have introduced, found while probing the neighbourhood). With the
# lookahead, "√sin(x)" and "√xy" are simply left un-normalised: they fall through to the behaviour they
# always had (the "√" becomes a stray Symbol) instead of being silently read as something new.
_RADICAL_TOKEN_RE = re.compile(r"√\s*(\d+(?:\.\d+)?|[A-Za-z](?![A-Za-z]))")
# Bare ASCII "sqrt" immediately followed by digits with no "(" at all
# ("5sqrt2", "sqrt50") -- NOT matched when "(" already follows (optionally
# after whitespace), so an already-well-formed "sqrt(50)"/"sqrt (50)" is
# left untouched.
_BARE_SQRT_DIGIT_RE = re.compile(r"sqrt(?!\s*\()\s*(\d+(?:\.\d+)?)")


def _normalize_surd_notation(text: str) -> str:
    """Rewrite the surd spellings above to explicit `sqrt(...)` calls.

    Order matters: the paren pass must run before the token pass, since
    `_RADICAL_TOKEN_RE` requires a digit/letter immediately after "√" and
    would not match "√(" at all -- but running the paren pass first means
    it's already gone by the time the token pass runs, so the two passes
    never fight over the same text. The bare-ASCII-"sqrt" pass is
    independent of both (it only ever fires on the literal word "sqrt", which
    the radical passes never produce without also adding the "(" that makes
    it a no-op)."""
    text = _RADICAL_PAREN_RE.sub("sqrt(", text)
    text = _RADICAL_TOKEN_RE.sub(lambda m: f"sqrt({m.group(1)})", text)
    text = _BARE_SQRT_DIGIT_RE.sub(lambda m: f"sqrt({m.group(1)})", text)
    return text


# -- expression: answer_form "expanded" -- redundant enclosing brackets -----. `"(" in stripped` used to reject ANY bracket at all, so a
# correct answer typed back wrapped in one redundant pair ("(x - 5)" for a
# canonical "x - 5") was marked wrong. Strip one FULLY-ENCLOSING bracket pair
# before that check -- "fully-enclosing" must be balance-aware, not
# `startswith("(") and endswith(")")`: "(x^2 - 25)/(x + 5)" also starts and
# ends that way, but its first "(" closes at the "/", well before the final
# character, so it is NOT one pair wrapping the whole string and must still
# trip the bracket check.


def _strip_one_enclosing_bracket(text: str) -> str:
    """Strip one fully-enclosing "(" ... ")" pair from `text`, or return it
    unchanged if the leading "(" (if any) does not close at the final index.

    Tracks paren depth from the left; the pair opened at index 0 only
    encloses the whole string if depth first returns to 0 at the last
    character. `(x + 1)(x + 2)` starts and ends with brackets too, but the
    first "(" closes well before the end (after "(x + 1)"), so this leaves
    it untouched -- the caller's `while` loop below is what turns "one pair
    at a time" into "every redundant nested pair", not this function trying
    to be clever about more than one pair itself. An unbalanced/malformed
    string (`"(x - 5"`, no closing ")" at all) also falls through unchanged
    rather than raising -- depth never returns to 0, the loop just ends.
    """
    if len(text) < 2 or text[0] != "(" or text[-1] != ")":
        return text
    depth = 0
    for i, ch in enumerate(text):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return text[1:-1] if i == len(text) - 1 else text
    return text


def _strip_redundant_enclosing_brackets(text: str) -> str:
    """Loop `_strip_one_enclosing_bracket` until nothing more strips, so a
    nested redundant pair ("((x - 5))") is peeled down to bare text, not just
    one layer of it. Whitespace is stripped each pass -- `stripped` (the
    caller's raw learner text) is not itself whitespace-trimmed, and neither
    is the text left over after peeling one pair, e.g. " (x - 5)" or the
    inner text of "( (x - 5) )"."""
    candidate = text.strip()
    while True:
        unwrapped = _strip_one_enclosing_bracket(candidate).strip()
        if unwrapped == candidate:
            return candidate
        candidate = unwrapped


# -- expression: learner notation ------------------------------------------
#
# Three spellings a learner types by hand and the library never emits, so no
# sweep fed them to the grader: a multiplication sign ("2·x", "2×x"), absolute
# value bars ("ln|x| + C") and the constants as written on paper ("e^(2x)",
# "6πi"). Each used to parse to something else -- "2·x" to 2*x*·, "e^(2x)"
# with `e` a free symbol -- and bars were worst of all: the parse failed and
# the spoken-word fallback silently turned "ln|x| + C" into `x`.
#
# The multiplication signs and bars are rewritten as text here, before
# parsing. The constants cannot be: whether `e` and `i` are Euler's number and
# the imaginary unit depends on the question, so `_resolve_constants` does it
# after parsing, against the expected expression. `π` is never anything but
# pi, so `_parse_expression` maps it unconditionally.

_MULTIPLICATION_SIGNS = str.maketrans({"·": "*", "×": "*", "⋅": "*"})
# Square brackets and braces group like round ones on paper ("2[x + 3(x - 1)]", "e^{2x}"). Unread, a brace sent
# the reply to the spoken-word fallback, which dropped what it could not say: "e^{2x}" was read as 2x and
# "\sqrt{12}" as 12. LaTeX (a backslash) is not rewritten; it is not read at all.
_ROUND_BRACKETS = str.maketrans("[]{}", "()()")


def _round_brackets(text: str) -> str:
    return text if "\\" in text else text.translate(_ROUND_BRACKETS)


_PI_SYMBOL = sympy.Symbol("π")
_E_SYMBOL = sympy.Symbol("e")
_I_SYMBOL = sympy.Symbol("i")


def _bars_to_abs(text: str) -> str | None:
    """Rewrite "|...|" as "(Abs(...))", or None if the bars do not pair up.

    The extra bracket keeps "ln|x|" a function application, "ln(Abs(x))":
    a bare "lnAbs(x)" is split into the product l*n*A*b*s*x.

    A bar closes the innermost open bar when it follows an operand (a letter,
    digit, "." or closing bracket, or a bar that just closed); otherwise it
    opens one. That reads "||x| - 1|", "|x| + |y|", "|x||y|" and "2|x|" the
    way they are written. Unpaired bars return None, so the caller fails
    closed rather than guessing.
    """
    if "|" not in text:
        return text
    out: list[str] = []
    depth = 0
    prev = ""
    for ch in text:
        if ch == "|":
            if depth and (prev.isalnum() or prev in ".)|"):
                out.append("))")
                depth -= 1
                prev = "|"
            else:
                out.append("(Abs(")
                depth += 1
                prev = "("
            continue
        out.append(ch)
        if not ch.isspace():
            prev = ch
    if depth:
        return None
    return "".join(out)


# -- learner notation the parser misread -------------------------------------
#
# Before this normalisation:
# "x²" and "sin²x" parsed as x (the "²" failed the parse and the spoken-word
# fallback dropped it), "cos(theta)²" as ²*cos(theta), "cosec(x)" as
# c**2*e*o*s*x, "sinx" as i*n*s*x, "sin^-1(x)" as 1/sin(x), and
# "2 sin x cos x" as 2*sin(x*cos(x)). `_normalize_learner_notation` rewrites
# the spellings with one reading and returns None for the ones without, so the
# caller fails closed (wrong, with no reading) instead of grading a
# different expression.

_SUPERSCRIPT_DIGITS = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻", "0123456789+-")
_SUPERSCRIPT_RUN_RE = re.compile("[⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻]+")
# Longest first, so "arcsin", "asin" and "sinh" are never read as "sin", and
# "cosec" never as "cos" + "ec".
_FUNCTION_NAMES = (
    "arcsin",
    "arccos",
    "arctan",
    "cosec",
    "asin",
    "acos",
    "atan",
    "sinh",
    "cosh",
    "tanh",
    "sech",
    "csch",
    "coth",
    "sin",
    "cos",
    "tan",
    "sec",
    "csc",
    "cot",
    "exp",
    "log",
    "ln",
)
_FUNCTION_NAME_RE = re.compile("|".join(_FUNCTION_NAMES))
# sympy spells these differently; "arcsin(x)" otherwise parses as a*c*i*n*r*s*x.
_FUNCTION_ALIASES = {"cosec": "csc", "arcsin": "asin", "arccos": "acos", "arctan": "atan"}
_TRIG_NAMES = r"(?:arcsin|arccos|arctan|asin|acos|atan|sinh|cosh|tanh|sech|csch|coth|sin|cos|tan|sec|csc|cot)"
# "sin^-1 x" and "sin⁻¹x" mean arcsin to a learner and 1/sin(x) to sympy.
_INVERSE_POWER_RE = re.compile(r"\b" + _TRIG_NAMES + r"\s*\^\s*\(?\s*-")
# A function's own power: "sin^2 x", "sin^(2) x".
_OWN_POWER_RE = re.compile(r"\s*\^\s*(?:\([^()]*\)|[\d.]+)")
# Two operands separated only by space: "x cos", "2 x".
_SPACED_OPERANDS_RE = re.compile(r"[\w.)\]]\s+[\w(\[]")


def _normalize_learner_notation(text: str) -> str | None:
    """Rewrite learner spellings sympy misreads, or None when there is no single reading.

    - superscript runs become powers: "x²" -> "x^(2)", "x⁻¹" -> "x^(-1)";
    - "cosec" and "arcsin"/"arccos"/"arctan" become sympy's names, and a
      function name glued to its neighbours is spaced out: "sinx" -> " sin x",
      "3sinθ" -> "3 sin θ";
    - a trig function raised to a negative power is refused (arcsin or 1/sin?);
    - an unbracketed function whose argument runs on into another function or a
      second space-separated factor is refused: "2 sin x cos x" and
      "sin 2x cos x" have no reading every learner shares, and neither does a
      function with no argument before an operator ("sin -x" parsed as x), or
      one whose argument carries a power ("sin x^2": sin(x²) or (sin x)²?).
      "sin 2x", "sin x/2" and "x sin x" are read as before.
    """
    text = _SUPERSCRIPT_RUN_RE.sub(lambda m: "^(" + m.group().translate(_SUPERSCRIPT_DIGITS) + ")", text)
    text = _FUNCTION_NAME_RE.sub(lambda m: f" {_FUNCTION_ALIASES.get(m.group(), m.group())} ", text)
    if _INVERSE_POWER_RE.search(text):
        return None
    for m in _FUNCTION_NAME_RE.finditer(text):
        rest = text[m.end() :]
        own_power = _OWN_POWER_RE.match(rest)
        if own_power:
            rest = rest[own_power.end() :]
        rest = rest.lstrip()
        if rest[:1] == "(":
            continue
        if not rest or rest[0] in "+-*/^=,)":
            return None  # "sin -x" parsed as x; "sin" alone has no argument
        depth = 0
        end = len(rest)
        for i, ch in enumerate(rest):
            if ch in "([":
                depth += 1
            elif ch in ")]":
                if depth == 0:
                    end = i
                    break
                depth -= 1
            elif depth == 0 and ch in "+-=,":
                end = i
                break
        argument = rest[:end]
        if _FUNCTION_NAME_RE.search(argument) or _SPACED_OPERANDS_RE.search(argument):
            return None
        if "^" in argument or "**" in argument:
            return None  # "sin x^2": sin(x²) to sympy, (sin x)² to many learners
    return text


# Greek letters a learner types as the character or spells out. `_resolve_constants`
# reads either as the expected expression's spelling when only that one is used.
_GREEK_NAMES = {
    "α": "alpha",
    "β": "beta",
    "γ": "gamma",
    "δ": "delta",
    "ε": "epsilon",
    "ζ": "zeta",
    "η": "eta",
    "θ": "theta",
    "ι": "iota",
    "κ": "kappa",
    "λ": "lambda",
    "μ": "mu",
    "ν": "nu",
    "ξ": "xi",
    "ρ": "rho",
    "σ": "sigma",
    "τ": "tau",
    "υ": "upsilon",
    "φ": "phi",
    "ϕ": "phi",
    "χ": "chi",
    "ψ": "psi",
    "ω": "omega",
}


def _greek_swaps(expr, expected_free) -> dict:
    swaps = {}
    for sym in expr.free_symbols:
        if sym in expected_free or not isinstance(sym, sympy.Symbol):
            continue
        name = _GREEK_NAMES.get(sym.name)
        if name and sympy.Symbol(name) in expected_free:
            swaps[sym] = sympy.Symbol(name)
            continue
        for char, spelled in _GREEK_NAMES.items():
            if spelled == sym.name and sympy.Symbol(char) in expected_free:
                swaps[sym] = sympy.Symbol(char)
                break
    return swaps


def _resolve_constants(expr, expected):
    """Read a learner's `e` and `i` as Euler's number and the imaginary unit
    whenever the question does not itself use them as variables.

    `e` becomes E unless the expected expression has a free symbol `e`; `i`
    becomes I only when the expected expression is complex (contains I) and
    has no free symbol `i`, so a vector's "3i + 4j" or a summation index is
    left alone.

    Returns None when reading `e` as Euler's number makes the expression too costly to compare: `cot(e)^1000`
    was measured as a power of a variable (a thousand terms) and is a power of a constant (it takes `simplify`
    minutes) once `e` is a number, and the cost check ran before this.
    """
    if expr is None or expected is None:
        return expr
    try:
        free = expected.free_symbols
        swaps = {}
        if _E_SYMBOL in expr.free_symbols and _E_SYMBOL not in free:
            swaps[_E_SYMBOL] = sympy.E
        if _I_SYMBOL in expr.free_symbols and _I_SYMBOL not in free and expected.has(sympy.I):
            swaps[_I_SYMBOL] = sympy.I
        swaps.update(_greek_swaps(expr, free))
        if not swaps:
            return expr
        resolved = expr.xreplace(swaps)
        if sympy.E in swaps.values() or sympy.I in swaps.values():
            check_expression_cost(resolved)
        return resolved
    except (TooCostly, ValueError):  # ValueError: `Max(e*I, 2)` cannot be built once `e` is a number
        return None
    except (AttributeError, TypeError):
        return expr


# sympy evaluates integer powers and factorials while it parses, and compares by substituting numbers and
# simplifying, so a short reply such as "2^2^2^2^2^2", "9^9999999", "99999!", "(9!)!", "x^(10^10)",
# "2^-2^-2^-...", or "ln(ln(ln(...)))" 30 deep can hang the request.
# `safe_parse` measures the shape of the expression -- unevaluated before the
# parse, and again on the parsed result -- and refuses one SymPy could spend seconds on, so such a reply
# is marked wrong unread. No answer in the library comes near any of its limits.


def _too_costly_to_parse(candidate: str) -> bool:
    """Whether evaluating `candidate` could take SymPy seconds (see `safe_parse.too_costly`)."""
    return too_costly(candidate, transformations=_TRANSFORMATIONS)


def _sympy_text(text: str) -> str | None:
    """`text` rewritten for sympy's parser, or None when the learner notation has no single reading."""
    candidate = _normalize_learner_notation(_round_brackets(text).translate(_MULTIPLICATION_SIGNS))
    if candidate is None:
        return None  # no single reading: never guess, never hand it to the spoken fallback
    candidate = _bars_to_abs(candidate)
    if candidate is None:
        return None  # unpaired "|": never hand bars to the spoken fallback
    return _normalize_surd_notation(candidate).replace("^", "**")


def _parse_expression(given: str):
    text = given.strip()
    if not text:
        return None
    if len(text) > MAX_ANSWER_LENGTH:
        return None  # never run the normalisers or the spoken fallback over a pasted essay
    candidate = _sympy_text(text)
    if candidate is None:
        return None
    try:
        # Measured before it is evaluated and again once parsed: an expression SymPy could spend seconds
        # on is not graded (and never reaches the spoken-word fallback, which would read it differently).
        parsed = parse_expression_safely(candidate, transformations=_TRANSFORMATIONS)
        if isinstance(parsed, sympy.Basic) and _PI_SYMBOL in parsed.free_symbols:
            parsed = parsed.xreplace({_PI_SYMBOL: sympy.pi})
        return parsed
    except TooCostly:
        return None
    # sympy tokenises with Python's own tokenizer, and an unclosed bracket or a stray apostrophe
    # ("(x+1", "f'", "don't know") raises tokenize.TokenError, which is NOT a SyntaxError.
    #
    # A stray *closing* bracket ("x - 5)", ")", "2)") is a different failure shape -- sympy's
    # implicit-multiplication pass pops an empty parenthesis-tracking stack and raises IndexError,
    # not any of the types above. Same defect, second exception type: enumerating "the types known
    # so far" is exactly what let this one back in, so `grade` itself now also has a total backstop
    # (see the `except Exception` in `grade`) -- this tuple should keep growing for anything sympy's
    # parser can throw that is cheap to recognise here, but it is no longer the only line of
    # defence.
    except (sympy.SympifyError, SyntaxError, TokenError, TypeError, ValueError, IndexError):
        pass
    if "|" in text or "\\" in text or any(b in text for b in "[]{}") or not text.isascii():
        # The spoken-word fallback drops the bars and whatever they touch:
        # it read "ln|x| + C" as `x`, and "x²" as `x`. Marking an unreadable
        # answer wrong is bad; grading it as a different expression is worse.
        # Spoken transcripts are plain ASCII words, with no LaTeX and no brackets
        # other than round ones.
        return None
    return spoken_to_expr(text)


# -- expression: answer_form "simplified_surd" -------------------
#
# MUST be a raw-text check on `stripped`, never on `given_expr` -- the same
# trap `answer_form="factored"` already avoids (see the comment above
# `answer_form` in `grade_expression`). Measured: `sympify("sqrt(50)")`
# returns `5*sqrt(2)` -- sympy extracts the perfect-square factor on parse,
# so by the time a parsed expression exists the evidence that the learner
# typed an unsimplified radicand is already gone.


def _sqrt_call_radicands(text: str) -> list[str]:
    """The raw text inside every `sqrt(...)` call in `text` (balanced-paren
    extraction, so a nested call like `sqrt(2*(3+4))` still yields its whole
    argument, not a truncation at the first inner ")"). `text` must already
    have every √/bare-"sqrtN" spelling rewritten to this ascii "sqrt(" form
    (`_normalize_surd_notation`) -- this function only ever looks for the
    literal substring "sqrt(".
    """
    radicands = []
    for m in re.finditer(r"sqrt\(", text):
        start = m.end()
        depth = 1
        i = start
        n = len(text)
        while i < n and depth:
            if text[i] == "(":
                depth += 1
            elif text[i] == ")":
                depth -= 1
            i += 1
        radicands.append(text[start : i - 1])
    return radicands


#: The largest radicand `_simplified_surd_ok` searches for a square factor (trial division is O(sqrt n)).
_MAX_RADICAND = 10**12


def _has_perfect_square_factor(n: int) -> bool:
    if n < 2:
        return False
    i = 2
    while i * i <= n:
        if n % (i * i) == 0:
            return True
        i += 1
    return False


def _simplified_surd_ok(stripped: str) -> bool:
    """False if any radicand typed under a `sqrt(...)` in `stripped` has an
    integer value with a perfect-square factor greater than 1 -- e.g.
    `sqrt(50)` (50 = 25*2) or `sqrt(18) + sqrt(8)` (18 = 9*2, 8 = 4*2).

    Runs on `_normalize_surd_notation(stripped)` -- the same √/bare-"sqrtN"
    rewrite `_parse_expression` applies -- so a learner who typed "√50" can't
    walk this check in through the back door just because it only recognises
    the literal ascii spelling `sqrt(`.

    A radicand is only checked when it is (or evaluates to) a plain integer
    literal (`sqrt(25*2)` is checked via a parse of "25*2", not a string
    match on "50"); a radicand containing a free symbol (`sqrt(x + 1)`) is
    left alone -- there is no general text-level notion of "simplified" for
    a symbolic radicand, and this form is only ever declared for a purely
    numeric simplify-surds question.
    """
    normalized = _normalize_surd_notation(stripped)
    for radicand in _sqrt_call_radicands(normalized):
        try:
            # "^" -> "**" first, same as `_parse_expression`'s own candidate
            # text -- a radicand written "sqrt(5^2*2)" would otherwise hit
            # Python's "^" (xor) operator and raise, silently skipping the
            # check rather than evaluating it.
            value = parse_expression_safely(radicand.strip().replace("^", "**"), transformations=_TRANSFORMATIONS)
        # Deliberately NOT adding IndexError here too. `radicand` comes from
        # `_sqrt_call_radicands`'s depth-counted extraction, which only ever stops at the ")" that
        # brings depth back to 0 -- so every ")" inside the extracted text was already matched by a
        # "(" also inside it. An unmatched *closing* bracket cannot occur in this substring by
        # construction; an unmatched *opening* one can, but that leaves sympy's parenthesis stack
        # non-empty rather than popping it, so it doesn't raise IndexError either. Checked
        # empirically against 15 adversarial radicands (unclosed, doubly-closed, nested-unbalanced)
        # with none raising.
        except (sympy.SympifyError, SyntaxError, TokenError, TypeError, ValueError):
            continue
        if not getattr(value, "is_Integer", False):
            continue
        n = int(value)
        if n > _MAX_RADICAND:
            return False  # not a simplified surd, and too large to search for square factors
        if n > 1 and _has_perfect_square_factor(n):
            return False
    return True


# -- expression: answer_form "rationalised" ----------------------
#
# Measured: sympy does NOT rationalise "1/(2+sqrt(3))" on parse --
# `sympy.fraction()[1]` stays `sqrt(3) + 2` -- so a PARSED check (reject a
# `Pow` with a half-integer exponent in the denominator) is sound here,
# unlike `simplified_surd` above. Confined to binomial denominators (this
# skill's own scope): measured, "1/sqrt(2)" auto-parses to "sqrt(2)/2", so for a
# monomial denominator sympy has already done the rationalising work itself
# and the "rationalised"/"not rationalised" forms are literally the same
# expression -- no check, parsed or raw, can tell them apart, so this form
# must never be declared for a monomial-denominator question.
#
# A raw-text backstop runs alongside the parsed check, for two measured
# reasons: (1) the same √-spelling back door as `simplified_surd` above, and
# (2) a case the parsed check provably cannot see at all -- see
# `_rationalised_denominator_flagged_raw`'s own docstring.


def _denominator_operand_text(text: str, start: int) -> str:
    """The raw text of the single operand immediately to the right of a "/"
    at `text[start - 1]` (`start` is the index just after the "/") -- a
    balanced `(...)` group if the next non-space character opens one,
    otherwise a run of characters up to the next top-level `+`, `-`, `*` or
    `/` (division binds only as far as the next same/lower-precedence
    operator; a following `*` is a SEPARATE factor, not part of the
    denominator: "1/3*sqrt(3)" is `(1/3)*sqrt(3)`, not `1/(3*sqrt(3))`)."""
    n = len(text)
    i = start
    while i < n and text[i].isspace():
        i += 1
    if i < n and text[i] == "(":
        depth = 1
        j = i + 1
        while j < n and depth:
            if text[j] == "(":
                depth += 1
            elif text[j] == ")":
                depth -= 1
            j += 1
        return text[i:j]
    depth = 0
    j = i
    while j < n:
        ch = text[j]
        if ch == "(":
            depth += 1
        elif ch == ")":
            if depth == 0:
                break
            depth -= 1
        elif ch in "+-*/" and depth == 0:
            break
        j += 1
    return text[i:j]


def _rationalised_denominator_flagged_raw(stripped: str) -> bool:
    """Raw-text backstop for `rationalised`.

    Needed for a case the parsed check in `grade_expression` provably cannot
    see: `1/(sqrt(2)+sqrt(8))`. sympy simplifies `sqrt(8)` to `2*sqrt(2)` on
    parse, collapsing the denominator to the MONOMIAL `3*sqrt(2)`, which
    sympy then auto-rationalises to `sqrt(2)/6` -- by the time the parsed
    check runs, the denominator is the plain integer `6`, with no `Pow` left
    in it at all (measured: `sympy.fraction(parse_expr("1/(sqrt(2)+sqrt(8))"
    ))` gives denominator `6`). The raw text still shows two literal
    `sqrt(...)` terms right after the "/", so this still catches it.

    Runs on `_normalize_surd_notation(stripped)` so the √ spelling is caught
    too, same as `simplified_surd`.

    Confined to the operand immediately after each "/" (`_denominator_operand_
    text`), not everything to the right of it, so a genuinely rationalised
    answer written as "2/3 - 1/3*sqrt(3)" -- whose only "/"'s immediate
    denominator operand is the plain integer "3" -- is not falsely flagged
    just because a sqrt appears later in the same expression (a bare
    `rsplit("/", 1)` would flag this; measured false reject).
    """
    normalized = _normalize_surd_notation(stripped)
    for m in re.finditer(r"/", normalized):
        operand = _denominator_operand_text(normalized, m.end())
        if "sqrt(" in operand:
            return True
    return False


def _rationalised_denominator_flagged_parsed(given_expr) -> bool:
    """True if `sympy.fraction(given_expr)`'s denominator contains a `Pow`
    with a half-integer (non-integer rational) exponent -- i.e. still
    carries an unrationalised radical. Deliberately checks "rational, not
    integer" rather than "exponent == 1/2" specifically, so a cube-root (or
    any other root) denominator is caught the same way; A4 confines the
    *skill* to binomial square-root denominators, but nothing here assumes
    the exponent is exactly 1/2.
    """
    try:
        _, denominator = sympy.fraction(given_expr)
    except (TypeError, ValueError, AttributeError):
        return False
    try:
        nodes = sympy.preorder_traversal(denominator)
    except (TypeError, ValueError, AttributeError):
        return False
    for node in nodes:
        if isinstance(node, sympy.Pow):
            exponent = node.exp
            if getattr(exponent, "is_Rational", False) and not getattr(exponent, "is_Integer", False):
                return True
    return False


# -- expression: answer_form "line_standard_form" ----------------
#
# Sanctioned, narrow addition to `grade_expression`'s closed `answer_form` set --:
# `simplified_surd`/`rationalised` were themselves added to this exact set, the established
# precedent for one new elif branch on an already-open extension point. `ax + by + c = 0` (integer
# coefficients, `a > 0`, `gcd(|a|,|b|,|c|) == 1`) is a NORMALISED form -- `2x + 3y - 6 = 0`, `-2x -
# 3y + 6 = 0` and `4x + 6y - 12 = 0` are all the same line (the same VALUE) and only one is the
# required answer, exactly the shape of problem `factored`/`expanded` already solve for other forms.
#
# Unlike `factored`/`expanded` (raw-text checks, because sympy would silently fix the very property
# being tested), this property IS recoverable from a parse: sympy only loses which side of "="
# things were written on, and `a > 0` is exactly the freedom that is meant to remove -- so this form
# is checked on the PARSED equation.
#
# `_parse_expression` cannot parse a literal "=" -- it is built for a single expression, not an `Eq`
# -- and measured directly, `_parse_expression("2x + 3y - 6 = 0")` returns `None` even for the
# CANONICAL text. `grade_expression`'s shared `if given_expr is None: return ...` early-return (just
# below) would therefore reject every candidate for this form, including the correct one, before the
# `answer_form` dispatch chain is ever reached -- so this form is graded on its own path, split out
# BEFORE that early-return, rather than as a literal `elif` inside the existing chain.
# `_parse_expression` itself is never changed: it is used by every other `answer_form` too, and the
# split-on-"=" happens locally, here, instead (`_split_line_equation`).


def _split_line_equation(text: str):
    """Split `text` on a single "=" into `(lhs, rhs)` PARSED expressions, or
    `None` if `text` is not exactly one equation (zero, or more than one,
    "=" sign) or either side fails to parse. Reuses `_parse_expression` for
    each side so both halves get the same surd/spoken-word normalisation as
    every other `answer_form` -- `_parse_expression` itself is untouched.
    """
    parts = text.split("=")
    if len(parts) != 2:
        return None
    lhs = _parse_expression(parts[0])
    rhs = _parse_expression(parts[1])
    if lhs is None or rhs is None:
        return None
    return lhs, rhs


_LINE_X, _LINE_Y = sympy.symbols("x y")


def _line_standard_form_coeffs(text: str):
    """Normalise a line written as an equation (`"2x + 3y - 6 = 0"`) to
    `(a, b, c)` -- integer coefficients, `a > 0`,
    `gcd(|a|, |b|, |c|) == 1` -- or `None` if `text` is not exactly one
    equation, is not linear in exactly `x` and `y` (an extra free symbol, or
    an `x**2`/`x*y` term), carries a non-integer coefficient, is degenerate
    (`a == 0`), has the wrong sign (`a < 0`), or is not already fully
    reduced. This is a REJECT/ACCEPT check on text already claimed to be in
    the required form -- it never reduces a candidate for the caller, it
    only confirms one already is.
    """
    split = _split_line_equation(text)
    if split is None:
        return None
    lhs, rhs = split
    try:
        diff = sympy.expand(lhs - rhs)
        coeffs = diff.as_coefficients_dict()
    except (sympy.SympifyError, TypeError, ValueError, AttributeError):
        return None
    allowed = {_LINE_X, _LINE_Y, sympy.S.One}
    if any(term not in allowed for term in coeffs):
        return None  # not linear in exactly x and y
    try:
        a_r = sympy.Rational(coeffs.get(_LINE_X, 0))
        b_r = sympy.Rational(coeffs.get(_LINE_Y, 0))
        c_r = sympy.Rational(coeffs.get(sympy.S.One, 0))
    except (TypeError, ValueError):
        return None
    if a_r.q != 1 or b_r.q != 1 or c_r.q != 1:
        return None  # non-integer coefficient
    a, b, c = int(a_r), int(b_r), int(c_r)
    if a <= 0:
        return None  # a == 0 is degenerate; a < 0 is the wrong sign
    if math.gcd(math.gcd(abs(a), abs(b)), abs(c)) != 1:
        return None  # not fully reduced
    return (a, b, c)


# -- expression: answer_form "single_fraction" -------------------
#
# Two things were checked before this form got a grader of its own:
#
# 1. Whether `expanded` could stand in for it. It cannot: a `Question` with `answer_form="expanded"` and
#    `answer_text="(21*x - 1)/6"`, graded against that same text, is rejected. `expanded`'s bracket check only
#    strips a bracket that spans the whole string, and "(21*x - 1)/6" ends in "6", not ")", so the leading "("
#    is never stripped and the "(" left in the unwrapped text rejects the canonical answer itself. `expanded`
#    is therefore unusable for any "combine into one fraction" answer whose own canonical text needs a
#    bracket that does not enclose the whole of it, around a numerator of two or more terms.
#
# 2. Whether a check on the parsed expression could tell the forms apart. It cannot: SymPy parses
#    "(5*x+1)/2 + (3*x-2)/3" and "(21*x-1)/6" to the identical object (both become "7*x/2 - 1/6"), so nothing
#    on the parsed expression distinguishes "typed as one combined fraction" from "typed as an unmerged sum of
#    two fractions". Only a raw-text, bracket-depth-aware scan can. This is therefore a raw-text-only branch,
#    like `_rationalised_denominator_flagged_raw`'s own unparseable-by-value case above, and there is
#    deliberately no parsed-expression companion check to pair it with.


def _single_fraction_flagged_raw(stripped: str) -> bool:
    """True if `stripped` (after `_normalize_surd_notation`, same
    normalisation `rationalised`/`simplified_surd` already run, so a
    learner's √ spelling cannot dodge this check) carries a BINARY "+" or "-"
    at bracket depth 0.

    "Binary" means: not the first non-space character in `stripped` (a
    learner's raw text is not itself whitespace-trimmed -- same caveat
    `_strip_redundant_enclosing_brackets` documents -- so a LEADING sign may
    still be preceded by spaces at index > 0), and the immediately preceding
    non-space character is not one of "( * / ^ + -" (a leading/unary sign,
    and a sign immediately after an operator -- e.g. the "-" in "x^-2" -- is
    never mistaken for a binary operator). Depth is tracked left to right the
    same way `_denominator_operand_text` above does ("(" -> +1, ")" -> -1);
    only a "+"/"-" seen while depth == 0 is a candidate.

    A depth-0 binary "+"/"-" is the signature of an UNMERGED sum of two
    fraction terms: each fraction's own internal "+"/"-" sits inside its
    numerator's bracket, at depth >= 1, and is never seen at depth 0. See the
    two probes recorded above this function for why this must be a raw-text
    check and why no parsed check can ever substitute for it.
    """
    normalized = _normalize_surd_notation(stripped)
    depth = 0
    prev_non_space = None  # None until the first non-space character is seen
    # a "+"/"-" with no preceding non-space char is by definition leading/unary, however many spaces
    # come before it.
    for i, ch in enumerate(normalized):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch in "+-":
            if depth == 0 and prev_non_space is not None and prev_non_space not in "(*/^+-":
                return True
        if not ch.isspace():
            prev_non_space = ch
    return False


def _expected_expression(question):
    """The expression a learner's answer is compared against.

    ``question.answer`` is the generator's internal value; ``answer_text`` is
    the canonical the learner is actually shown. For an indefinite integral the
    two differ by the constant of integration -- ``answer_text`` carries "+ C"
    and ``answer`` does not -- so a learner who typed the canonical they were
    shown was marked wrong on every question of four skills (480 false rejects
    across integrate-polynomial, integrate-trig, integration-by-substitution
    and solve-ode-separable-poly), while a learner who omitted the constant was
    accepted.

    Where the displayed canonical differs from the internal answer by exactly
    one free symbol, the displayed one is authoritative. Every other divergence
    keeps ``question.answer``: ``answer_text`` is a display string and does not
    always parse to the same expression. The deferred solve-ode-exponential
    shows an answer as "y = C.e^(-3x)", which parses to nonsense rather than
    failing outright, so "use answer_text whenever it parses" would silently
    grade against garbage.
    """
    try:
        internal = sympify_safely(question.answer)
    except (sympy.SympifyError, TypeError, ValueError):
        internal = None

    displayed = _parse_expression(_strip_label(question.answer_text or ""))
    if displayed is None:
        return internal
    if internal is None:
        return displayed

    try:
        extra = sympy.simplify(displayed - internal)
    except (TypeError, ValueError):
        return internal
    # Exactly one free symbol and nothing else: the constant of integration.
    if extra.free_symbols and extra in extra.free_symbols:
        return displayed
    return internal


# The constant of integration is written "+ C" in every canonical, and on paper as often "+ c" or "+ K". Read as a
# different symbol, every such reply was marked wrong. When the expected answer ends in a lone "+ C" and the reply
# has no C but one of these letters, unknown to the expected answer, the letter is read as C.
_CONSTANT_SYMBOL = sympy.Symbol("C")
_CONSTANT_SPELLINGS = frozenset(sympy.Symbol(name) for name in ("c", "k", "K"))


def _match_constant_of_integration(given_expr, expected):
    try:
        if _CONSTANT_SYMBOL not in sympy.Add.make_args(expected) or _CONSTANT_SYMBOL in given_expr.free_symbols:
            return given_expr
        stand_ins = (given_expr.free_symbols - expected.free_symbols) & _CONSTANT_SPELLINGS
    except (AttributeError, TypeError):
        return given_expr
    if len(stand_ins) != 1:
        return given_expr
    return given_expr.xreplace({stand_ins.pop(): _CONSTANT_SYMBOL})


_DECIMAL_LITERAL_RE = re.compile(r"(?<![\w.])(\d+)?\.(\d+)?(?![\d.])")


def _decimal_as_quotient(match: re.Match) -> str:
    whole, places = match.group(1) or "", match.group(2) or ""
    if not whole and not places:
        return match.group(0)
    return f"({int(whole + places or '0')}/1{'0' * len(places)})"


def _exact_decimals(given_expr, expected, text: str):
    """The reply re-read with each typed decimal as the exact number it spells ("x^0.5" as x^(5/10)), when the
    expected answer has no decimal of its own. sympy keeps a typed decimal as a float, so "x^0.5" and "0.5x" never
    simplified to match sqrt(x) and x/2: a right answer marked wrong. The decimals are rewritten in the text, before
    sympy folds them: "x/3.0" is x/3, where rationalising the parsed float 0.333... afterwards would not be. A decimal
    that is not the exact value stays wrong."""
    try:
        if expected.atoms(sympy.Float) or not given_expr.atoms(sympy.Float):
            return given_expr
    except (AttributeError, TypeError):
        return given_expr
    reread = _parse_expression(_DECIMAL_LITERAL_RE.sub(_decimal_as_quotient, text))
    if reread is None:
        return given_expr
    reread = _resolve_constants(reread, expected)
    if reread is None:
        return given_expr
    return _match_constant_of_integration(reread, expected)


# -- expression: answer_form "factored" ------------------------------------
#
# A bracket in the reply is not enough: "(6x + 9)", "1(6x + 9)" and the question's own "2(x - 3)^2 - 7(x - 3) - 30"
# all carry one. The reply is read as written (an unevaluated parse keeps the product the learner typed) and must
# be a product of at least two factors, a number other than 1 counting as one; no polynomial factor may keep a
# common whole-number factor ("3(4x + 6)" for 6(2x + 3)) unless the canonical keeps one too; and it must have at least
# as many non-constant factors as the canonical ("(x^2 - 1)(x^2 + 1)" for (x - 1)(x + 1)(x^2 + 1)). Factoring
# further than the canonical is fine.
# Text the unevaluated parse cannot read keeps the bracket check alone.


def _written_factors(text: str) -> tuple[object, list] | None:
    """(the product of the constant factors, the non-constant factors with repeats) of `text` as written."""
    candidate = _sympy_text(text.strip()) if text and len(text) <= MAX_ANSWER_LENGTH else None
    if candidate is None:
        return None
    try:
        tree = parse_expression_safely(candidate, transformations=_TRANSFORMATIONS, evaluate=False)
    except Exception:  # noqa: BLE001 - unreadable as written: the caller keeps its simpler check
        return None
    if not isinstance(tree, sympy.Basic):
        return None
    constant, factors, stack = sympy.Integer(1), [], [tree]
    while stack:
        node = stack.pop()
        if isinstance(node, sympy.Mul):
            stack.extend(node.args)
        elif not node.free_symbols:
            constant *= node
        elif isinstance(node, sympy.Pow) and node.exp.is_Integer and 1 < node.exp <= 20:
            stack.extend([node.base] * int(node.exp))
        else:
            factors.append(node)
    return constant, factors


def _keeps_a_common_factor(factor) -> bool:
    try:
        poly = sympy.Poly(factor)
    except (sympy.PolynomialError, sympy.GeneratorsNeeded, TypeError, ValueError):
        return False
    return poly.domain.is_ZZ and abs(poly.content()) != 1


def _factorised_as_far_as(stripped: str, question) -> bool:
    given = _written_factors(stripped)
    if given is None:
        return True
    constant, factors = given
    try:
        trivial_constant = bool(abs(constant).equals(1))
    except (TypeError, ValueError):
        trivial_constant = False
    if len(factors) + (0 if trivial_constant else 1) < 2:
        return False
    expected = _written_factors(_strip_label(question.answer_text or "")) or _written_factors(str(question.answer))
    # A canonical that leaves a common factor inside a bracket, "(3x - 3)(x^2 + 9)", answers a question about the
    # split into factors, not the common factor: there a reply may leave one too.
    if (expected is None or not any(map(_keeps_a_common_factor, expected[1]))) and any(
        map(_keeps_a_common_factor, factors)
    ):
        return False
    return expected is None or len(factors) >= len(expected[1])


_DEGREE_SUFFIX_RE = re.compile(r"(?<=[\d)])\s*(?:[°º˚]|deg(?:rees?|s)?\.?)\s*$", re.IGNORECASE)


_INFINITY_REPLY_RE = re.compile(r"^\s*([+-]?)\s*(?:∞|infinity|inf|oo)\s*$", re.I)


# A label naming a derivative or just saying "answer": "f'(x) =", "y'' =", "f′(x) =", "dy/dx =", "d²y/dx² =",
# "d^2y/dx^2 =", "d/dx =", "Answer:". `_LABEL_RE` takes "y =" and "f(x) =" but not these.
_DERIVATIVE_LABEL_RE = re.compile(
    r"^\s*(?:"
    r"[^\W\d_]\w*\s*(?:'+|′+|″|‴)\s*(?:\([^()=]*\))?"  # f'(x), y'', g′(t)
    r"|d\s*(?:\^\s*\d|[²³])?\s*(?:[^\W\d_]\w*)?\s*/\s*d\s*[^\W\d_]\w*?\s*(?:\^\s*\d|[²³])?"  # dy/dx, d²y/dx², d/dx
    r")\s*=\s*"
    r"|^\s*answer\s*[:=]\s*",
    re.IGNORECASE,
)


def _strip_derivative_label(question, given: str) -> str:
    """`given` without a leading derivative or "answer" label, unless the canonical is itself written with one."""
    if not isinstance(given, str) or _DERIVATIVE_LABEL_RE.match(getattr(question, "answer_text", "") or ""):
        return given
    return _DERIVATIVE_LABEL_RE.sub("", given, count=1)


@_multi_part_aware
def grade_expression(question, given: str) -> GradeResult:
    canonical = question.answer_text
    if isinstance(given, str):
        given = _strip_derivative_label(question, given).replace("÷", "/")
    stripped = _round_brackets(_strip_label(given))

    # answer_form "line_standard_form" -- graded on its own path,
    # split out here rather than as an `elif` further down: see the banner
    # comment above `_line_standard_form_coeffs` for why it must run before
    # the `given_expr is None` early-return just below, which would
    # otherwise reject this form's canonical answer along with everything
    # else (`_parse_expression` cannot parse the literal "=" this form's
    # `answer_text` always contains). Two-part check, both required: (1)
    # `_line_standard_form_coeffs` normalises AND validates the given text
    # (integer coefficients, a > 0, gcd 1) -- reject outright if it fails;
    # (2) an EXACT tuple match to the canonical's own normalised (a, b, c)
    # -- both sides are already normalised the same way, so this is not a
    # proportionality check.
    if getattr(question, "answer_form", None) == "line_standard_form":
        given_coeffs = _line_standard_form_coeffs(stripped)
        canonical_coeffs = _line_standard_form_coeffs(canonical or "")
        correct = given_coeffs is not None and canonical_coeffs is not None and given_coeffs == canonical_coeffs
        return GradeResult(
            correct=correct,
            canonical=canonical,
            given_normalised=str(given_coeffs) if given_coeffs is not None else None,
        )

    try:
        expected = _expected_expression(question)
    except (sympy.SympifyError, TypeError, ValueError):
        expected = None
    if getattr(expected, "is_number", False) and not expected.has(sympy.pi):
        # An angle typed with its unit: "30°", "30 degrees". Only when the answer is a plain number, so
        # nothing else a learner writes is read differently, and never for an answer in radians ("9π/4°").
        stripped = _DEGREE_SUFFIX_RE.sub("", stripped)
    if "∞" in (canonical or ""):
        # "infinity", "inf" or "oo" for an answer shown as ∞: the reply on its own, so nothing else is read differently.
        stripped = _INFINITY_REPLY_RE.sub(r"\1∞", stripped)
    given_expr = _parse_expression(stripped)
    if given_expr is None:
        return GradeResult(correct=False, canonical=canonical, given_normalised=None)
    given_expr = _resolve_constants(given_expr, expected)
    if given_expr is None:  # `e` as Euler's number made it too costly to compare
        return GradeResult(correct=False, canonical=canonical, given_normalised=None)
    given_normalised = str(given_expr)

    # A numeric answer within 0.005 of the question's `answer_alt` (set by a
    # generator that keeps a surd/exact canonical answer but also offers a
    # rounded decimal, e.g. vector magnitude) is accepted outright, before
    # the exact-equivalence/form checks below (which a bare decimal like
    # "10.72" could never satisfy against a canonical of "sqrt(115)").
    answer_alt = getattr(question, "answer_alt", None)
    if answer_alt is not None:
        try:
            given_value = float(given_expr)
        except (TypeError, ValueError, OverflowError):  # OverflowError: an integer past the range of a float
            given_value = None
        # Only a reply that is itself a plain number: "sqrt(12)" is within 0.005 of 2*sqrt(3)'s decimal too,
        # and must still meet the question's answer_form below.
        if not given_expr.is_Number:
            given_value = None
        if given_value is not None and abs(given_value - float(answer_alt)) <= 0.005:
            return GradeResult(correct=True, canonical=canonical, given_normalised=given_normalised)

    # Plain sympy equivalence accepts any algebraically equal form -- which
    # would wrongly accept the unfactored expression for a factoring question
    # (e.g. "6x + 9" for a "3(2x + 3)" answer) or the question copied verbatim
    # for an expansion question (e.g. "-2(2x + 1)" for a "-4x - 2" answer).
    # The required form is a property of what the QUESTION asks, not of how
    # the canonical answer happens to be spelled -- a generator
    # declares it explicitly via `answer_form`, because sympy auto-distributes
    # Number * Add, so neither the canonical nor the learner's raw text can be
    # parsed to recover it (a factored "6*(1 - 6*x)" and a learner's factored
    # "7(2x - 3)" both parse to an already-expanded expression).
    answer_form = getattr(question, "answer_form", None)
    if answer_form == "factored":
        # The learner's RAW TEXT must carry a bracket -- the parsed expression
        # cannot show this (see above), so this is a string check on `stripped`,
        # never on `given_expr`. A pair around the whole reply factors nothing:
        # "(6x + 9)" is the expression the question gave, so it is peeled first.
        if "(" not in _strip_redundant_enclosing_brackets(stripped) or not _factorised_as_far_as(stripped, question):
            return GradeResult(correct=False, canonical=canonical, given_normalised=given_normalised)
    elif answer_form == "expanded":
        # Peel off one (or, looped, several nested) fully-enclosing redundant bracket pair before
        # the raw "(" check -- see `_strip_redundant_enclosing_brackets` above. A bracket that is
        # NOT fully-enclosing (mid-expression, or closes before the string ends) survives this and
        # still trips the check below, unchanged from before this fix.
        unwrapped = _strip_redundant_enclosing_brackets(stripped)
        rejected = "(" in unwrapped
        if not rejected:
            try:
                rejected = sympy.expand(given_expr) != given_expr
            except (sympy.SympifyError, TypeError, ValueError, AttributeError):
                rejected = False  # can't raise out of grade(); fall through to equivalence
        if rejected:
            return GradeResult(correct=False, canonical=canonical, given_normalised=given_normalised)
    elif answer_form == "simplified_surd":
        # RAW-TEXT check on `stripped` -- see `_simplified_surd_ok`.
        if not _simplified_surd_ok(stripped):
            return GradeResult(correct=False, canonical=canonical, given_normalised=given_normalised)
    elif answer_form == "rationalised":
        # A PARSED check plus a raw-text backstop -- see
        # `_rationalised_denominator_flagged_parsed`/`_flagged_raw`. Confined
        # to binomial denominators; never declare this form for a question
        # whose rationalised denominator is a single sqrt term.
        if _rationalised_denominator_flagged_raw(stripped) or _rationalised_denominator_flagged_parsed(given_expr):
            return GradeResult(correct=False, canonical=canonical, given_normalised=given_normalised)
    elif answer_form == "single_fraction":
        # A RAW-TEXT-ONLY check -- see `_single_fraction_flagged_
        # raw`'s docstring for why no parsed-expression companion check is
        # possible here (sympy collapses an unmerged sum and its single-
        # fraction combination to the identical parsed object).
        if _single_fraction_flagged_raw(stripped):
            return GradeResult(correct=False, canonical=canonical, given_normalised=given_normalised)
    try:
        # symbolically_equal, not simplify: identical result, but a numeric probe
        # settles the common "these differ" case in microseconds instead of the
        # ~35 ms simplify spends exhausting its rewrites first. It can only ever
        # short-circuit to False, never to True.
        if expected is not None:
            given_expr = _exact_decimals(_match_constant_of_integration(given_expr, expected), expected, stripped)
        correct = expected is not None and symbolically_equal(expected, given_expr)
    except (sympy.SympifyError, TypeError, ValueError):
        correct = False
    return GradeResult(correct=correct, canonical=canonical, given_normalised=given_normalised)


_INDEX_RE = re.compile(r"-?\d{1,9}")
# "B", "b)", "(B)", "B.": an option's letter, A for the first option.
_OPTION_LETTER_RE = re.compile(r"^\(?\s*([A-Za-z])\s*[).:]?$")


def _mcq_choice_by_text(options: list, given: str) -> int | None:
    """The option a reply names by its text or its letter, or None to read it as an index.

    A reply of digits only is always an index, as documented, even when an option's text is the same digits.
    Option text is matched whole, ignoring case and spacing. A letter is read only when no option is itself a
    single letter, so "x" is never mistaken for option X's position.
    """
    reply = " ".join(given.split()).casefold()
    if not reply or reply.lstrip("-").isdigit():
        return None
    texts = [" ".join(str(option).split()).casefold() for option in options]
    if reply in texts:
        return texts.index(reply)
    letter = _OPTION_LETTER_RE.match(reply)
    if letter and not any(len(text) == 1 and text.isalpha() for text in texts):
        index = ord(letter.group(1).lower()) - ord("a")
        if index < len(options):
            return index
    return None


@_multi_part_aware
def grade_mcq(question, given: str) -> GradeResult:
    options = mcq_options(question)
    canonical = question.answer_text
    if isinstance(given, int) and not isinstance(given, bool):
        # The documented reply is the option's 0-based index, and a library caller passes it as an int; the regex
        # below takes text, so an int raised and every such reply was marked wrong.
        given = str(given)
    if not isinstance(given, str):
        return GradeResult(correct=False, canonical=canonical, given_normalised=None)
    chosen = _mcq_choice_by_text(options, given)
    if chosen is not None:
        return GradeResult(correct=options[chosen] == canonical, canonical=canonical, given_normalised=options[chosen])
    match = _INDEX_RE.search(given)
    if not match:
        return GradeResult(correct=False, canonical=canonical, given_normalised=None)
    idx = int(match.group())
    if not (0 <= idx < len(options)):
        return GradeResult(correct=False, canonical=canonical, given_normalised=None)
    correct = options[idx] == question.answer_text
    return GradeResult(correct=correct, canonical=canonical, given_normalised=options[idx])


# -- table ---------------
#
# A grid with some cells left blank for the learner. `Question.answer` is
#
#     {"columns": [corner, header, ...],
#      "rows": [[row label, cell, ...], ...],      # None marks a blank
#      "blanks": {"r,c": value, ...}}              # 0-based into "rows"
#
# Only blanks are graded, and every blank must be right. A blank's value is an
# int, a Fraction, a numeric string ("12.50", whose written places set the
# precision, as a pair component's canonical does) or a text label. The client
# posts the filled blanks as JSON ({"r,c": "18"}); typed text is also read, as
# "label = value" per blank in any order or as bare values in reading order.

_TABLE_SEPARATOR_RE = re.compile(r"\s*[,;\n]\s*")


def _table_key(key: str) -> tuple[int, int]:
    r, c = key.split(",")
    return int(r), int(c)


def _blank_label(answer: dict, key: str) -> str:
    r, c = _table_key(key)
    return f"{answer['rows'][r][0]} {answer['columns'][c]}".strip()


def _label_key(text: str) -> str:
    return re.sub(r"\s+", "", str(text)).casefold()


def _is_numeric_cell(value) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float, Fraction)):
        return True
    return isinstance(value, str) and _COMPONENT_LITERAL_RE.fullmatch(value.strip()) is not None


def _cell_text(value) -> str:
    return str(value)


def table_answer_text(answer: dict) -> str:
    """The canonical for a table question: "label = value" per blank, in
    reading order. Generators call this so the canonical and the grader's
    labels cannot drift apart."""
    keys = sorted(answer["blanks"], key=_table_key)
    labels = [_label_key(_blank_label(answer, k)) for k in keys]
    if len(set(labels)) != len(labels):
        # Typed "label = value" answers are matched by label, so two blanks
        # sharing a row label and column header could never both be read.
        raise ValueError("table blanks need distinct row-label + column-header pairs")
    return ", ".join(f"{_blank_label(answer, k)} = {_cell_text(answer['blanks'][k])}" for k in keys)


def table_prompt(answer: dict) -> dict:
    """The grid the client renders: headers and given cells, blanks as None.
    Masks every blank position even if a generator left its value in "rows",
    so the answer can never reach the client."""
    blanks = {_table_key(k) for k in answer["blanks"]}
    rows = [[None if (r, c) in blanks else cell for c, cell in enumerate(row)] for r, row in enumerate(answer["rows"])]
    return {"columns": list(answer["columns"]), "rows": rows, "blanks": sorted(answer["blanks"], key=_table_key)}


def _table_given_cells(answer: dict, given: str) -> dict[str, str] | None:
    """The learner's text for each blank, or None if it cannot be read."""
    keys = sorted(answer["blanks"], key=_table_key)
    text = (given or "").strip()
    if text.startswith("{"):
        try:
            data = json.loads(text)
        except ValueError:
            return None
        if not isinstance(data, dict):
            return None
        return {k: str(data[k]) for k in keys if data.get(k) is not None}
    segments = [seg for seg in _TABLE_SEPARATOR_RE.split(_strip_comma_groups(text)) if seg]
    if segments and all("=" in seg for seg in segments):
        by_label = {_label_key(_blank_label(answer, k)): k for k in keys}
        cells: dict[str, str] = {}
        for seg in segments:
            label, value = seg.rsplit("=", 1)
            key = by_label.get(_label_key(label))
            if key is None or key in cells:
                return None
            cells[key] = value.strip()
        return cells
    if len(segments) != len(keys):
        return None
    return dict(zip(keys, segments))


def _table_cell_correct(expected, given: str) -> bool:
    if _is_numeric_cell(expected):
        try:
            values = extract_values(_strip_comma_groups(given))
        except Exception:  # noqa: BLE001 - normalisation is best-effort
            logger.warning("extract_values failed for table cell %.200r", given, exc_info=True)
            return False
        if len(values) != 1:
            return False
        value = _as_fraction(expected.replace(" ", "")) if isinstance(expected, str) else expected
        return _component_target(_cell_text(expected), value).matches(values[0])
    return _label_key(given) == _label_key(expected)


@_multi_part_aware
def grade_table(question, given: str) -> GradeResult:
    answer = question.answer
    canonical = question.answer_text
    if not isinstance(answer, dict) or not answer.get("blanks"):
        return GradeResult(correct=False, canonical=canonical)
    cells = _table_given_cells(answer, given)
    if cells is None:
        return GradeResult(correct=False, canonical=canonical, given_normalised=None)
    verdicts = {
        key: key in cells and _table_cell_correct(expected, cells[key]) for key, expected in answer["blanks"].items()
    }
    given_normalised = ", ".join(f"{_blank_label(answer, k)} = {cells[k]}" for k in sorted(cells, key=_table_key))
    return GradeResult(
        correct=all(verdicts.values()), canonical=canonical, given_normalised=given_normalised, cells=verdicts
    )


# -- set -----------------
#
# An unordered collection whose items are numbers, short labels or tuples of
# numbers: `Question.answer` is a list of them. Graded as an exact multiset --
# nothing missing, nothing extra -- with numbers to their displayed precision
# as in a pair component (a numeric string such as "0.40" sets its own places)
# and labels ignoring case and spacing. Braces, commas, semicolons, new lines,
# "and" and "or" separate items; a comma inside brackets belongs to a tuple.

_SET_WORD_SEPARATOR_RE = re.compile(r"\s+(?:and|or)\s+", re.IGNORECASE)
# The empty set as it is written on paper: "{}" already read as no items, "∅" (and the "Ø" keyboards offer for it)
# and the words did not.
_EMPTY_SET_RE = re.compile(r"\s*(?:[∅Øø]|(?:the\s+)?empty\s+set|\{\s*\})\s*", re.IGNORECASE)


def _split_set_items(text: str) -> list[str]:
    if _EMPTY_SET_RE.fullmatch(text or ""):
        return []
    text = _SET_WORD_SEPARATOR_RE.sub(",", _strip_comma_groups(text or "").strip())
    if text.startswith("{") and text.endswith("}"):
        text = text[1:-1]
    items, depth, current = [], 0, []
    for ch in text:
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        if depth == 0 and ch in ",;\n":
            items.append("".join(current).strip())
            current = []
        else:
            current.append(ch)
    items.append("".join(current).strip())
    return [item for item in items if item]


def _set_item_text(item) -> str:
    if isinstance(item, (list, tuple)):
        return "(" + ", ".join(str(part) for part in item) + ")"
    return str(item)


def set_answer_text(items) -> str:
    """The canonical for a set question, "{a, b, c}". Generators call this."""
    return "{" + ", ".join(_set_item_text(item) for item in items) + "}"


def _set_item_matches(expected, given: str) -> bool:
    if isinstance(expected, (list, tuple)):
        inner = given.strip()
        if len(inner) >= 2 and inner[0] in "([" and inner[-1] in ")]":
            inner = inner[1:-1]
        parts = [part.strip() for part in inner.split(",")]
        return len(parts) == len(expected) and all(_set_item_matches(e, part) for e, part in zip(expected, parts))
    if _is_numeric_cell(expected):
        return _table_cell_correct(expected, given)
    return _label_key(given) == _label_key(expected)


@_multi_part_aware
def grade_set(question, given: str) -> GradeResult:
    expected = question.answer
    canonical = question.answer_text
    if not isinstance(expected, (list, tuple)):
        return GradeResult(correct=False, canonical=canonical)
    items = _split_set_items(given)
    given_normalised = "{" + ", ".join(items) + "}"
    if len(items) != len(expected):
        return GradeResult(correct=False, canonical=canonical, given_normalised=given_normalised)
    remaining = list(expected)
    for item in items:
        for index, candidate in enumerate(remaining):
            if _set_item_matches(candidate, item):
                del remaining[index]
                break
        else:
            return GradeResult(correct=False, canonical=canonical, given_normalised=given_normalised)
    return GradeResult(correct=True, canonical=canonical, given_normalised=given_normalised)


# -- points, polyline, line ---
#
# Drawn answers on a coordinate grid. The learner's clicks snap to grid
# points in the browser, so every coordinate is exact and these graders
# compare exactly, with no tolerance. The client posts canonical JSON; the
# typed fallback is the points written out, "(1, 2) (3, -4)", which is also
# what the regrade sweep feeds. Coordinates may be integers, decimals or a/b
# and are compared as Fractions, so "0.5" and "1/2" are the same grid point.
#
#   points    `Question.answer` is a list of [x, y], or {"points": [...],
#             "ordered": true}. Unordered: the same multiset of points.
#             Ordered: the same sequence.
#   polyline  `Question.answer` is {"pieces": [piece, ...]} (or one piece),
#             a piece being {"vertices": [[x, y], ...], "ends": [start, end]}
#             with "open" or "closed" ends for step graphs ("ends" omitted:
#             ends are not graded). Each piece must match one expected piece
#             vertex for vertex, drawn either way round; pieces in any order.
#             A piece is graded by its shape, not its clicks: repeated points
#             and vertices partway along a straight run are dropped from both
#             sides first. A line graph whose every vertex is a data value
#             sets "keep_vertices": true, so a collinear value still counts.
#   line      `Question.answer` is two distinct points [[x1, y1], [x2, y2]]
#             on the target line. The learner's two points pass when they
#             are distinct and both lie on it.

# A point, with the end mark before it if any. Linear in the text: a coordinate starts and ends on a character that
# is not a space, so nothing here can be matched in more than one way, and the leading `\s*` belongs to the mark
# (a run of spaces that is not followed by a bracket is not tried from every one of its characters).
_POINT_COORDINATE = r"([^(),\s](?:[^(),]*[^(),\s])?)"
_POINT_RE = re.compile(
    r"(?:(\*|o)\s*)?\(\s*" + _POINT_COORDINATE + r"\s*,\s*" + _POINT_COORDINATE + r"\s*\)", re.IGNORECASE
)
_POINT_FILLER_RE = re.compile(r"^(?:[\s,;]|\band\b|\bthen\b|->|→)*$", re.IGNORECASE)
_END_MARKS = {"o": "open", "*": "closed"}


def _coordinate(value) -> Fraction | None:
    if isinstance(value, bool) or value is None:
        return None
    text = str(value).strip().replace("−", "-")
    if not re.fullmatch(r"-?\d+(?:\.\d+)?(?:/\d+)?", text):
        return None
    value = _as_fraction(text)
    # The grid posts coordinates rounded to 6 dp, so a third-step grid sends
    # 0.333333 for 1/3: both sides are compared at that resolution.
    return None if value is None else Fraction(round(value * 10**6), 10**6)


def _point(value) -> tuple[Fraction, Fraction] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    x, y = _coordinate(value[0]), _coordinate(value[1])
    return None if x is None or y is None else (x, y)


def _points(values) -> list[tuple[Fraction, Fraction]] | None:
    if not isinstance(values, (list, tuple)):
        return None
    points = [_point(v) for v in values]
    return None if any(p is None for p in points) else points


def _typed_points(text: str) -> list[tuple[str | None, tuple[Fraction, Fraction]]] | None:
    """The points typed in `text` with any end mark ("o" open, "*" closed)
    before each, or None if anything else is in the text."""
    found = []
    for match in _POINT_RE.finditer(text):
        x, y = _coordinate(match.group(2)), _coordinate(match.group(3))
        if x is None or y is None:
            return None
        mark = match.group(1)
        found.append((_END_MARKS[mark.lower()] if mark else None, (x, y)))
    if not found or not _POINT_FILLER_RE.fullmatch(_POINT_RE.sub(" ", text)):
        return None
    return found


def _given_points(given: str) -> list[tuple[Fraction, Fraction]] | None:
    text = (given or "").strip()
    if text.startswith("["):
        try:
            return _points(json.loads(text))
        except ValueError:
            return None
    typed = _typed_points(text)
    return None if typed is None else [point for _, point in typed]


def _point_text(point) -> str:
    return f"({point[0]}, {point[1]})"


def points_answer_text(points) -> str:
    """The canonical for a points question, "(1, 2) (3, 4)". Generators call this."""
    return " ".join(_point_text(p) for p in _points(points))


@_multi_part_aware
def grade_points(question, given: str) -> GradeResult:
    answer = question.answer
    canonical = question.answer_text
    ordered = isinstance(answer, dict) and bool(answer.get("ordered"))
    expected = _points(answer.get("points") if isinstance(answer, dict) else answer)
    if not expected:
        return GradeResult(correct=False, canonical=canonical)
    points = _given_points(given)
    if points is None:
        return GradeResult(correct=False, canonical=canonical, given_normalised=None)
    correct = points == expected if ordered else sorted(points) == sorted(expected)
    return GradeResult(correct=correct, canonical=canonical, given_normalised=" ".join(_point_text(p) for p in points))


def _polyline_piece(value) -> tuple[list, tuple | None] | None:
    """(vertices, ends) for one piece, from a dict or a bare vertex list."""
    if isinstance(value, dict):
        vertices, ends = _points(value.get("vertices")), value.get("ends")
    else:
        vertices, ends = _points(value), None
    if not vertices or len(vertices) < 2:
        return None
    if ends is not None:
        if not isinstance(ends, (list, tuple)) or len(ends) != 2 or any(e not in ("open", "closed") for e in ends):
            return None
        ends = tuple(ends)
    return vertices, ends


def _simplified(vertices: list, keep_vertices: bool = False) -> list:
    """The vertices with repeated clicks removed and, unless every vertex is a
    data value, points partway along a straight run too."""
    out: list = []
    for v in vertices:
        if out and out[-1] == v:
            continue
        if len(out) >= 2 and not keep_vertices:
            (x0, y0), (x1, y1) = out[-2], out[-1]
            same_line = (x1 - x0) * (v[1] - y1) == (y1 - y0) * (v[0] - x1)
            same_way = (x1 - x0) * (v[0] - x1) + (y1 - y0) * (v[1] - y1) > 0
            if same_line and same_way:
                out[-1] = v
                continue
        out.append(v)
    return out


def _polyline_pieces(value, keep_vertices: bool = False) -> list | None:
    if isinstance(value, dict) and "pieces" in value:
        value = value["pieces"]
    elif isinstance(value, dict) or (isinstance(value, (list, tuple)) and value and _point(value[0]) is not None):
        value = [value]
    if not isinstance(value, (list, tuple)) or not value:
        return None
    pieces = [_polyline_piece(piece) for piece in value]
    if any(p is None for p in pieces):
        return None
    pieces = [(_simplified(vertices, keep_vertices), ends) for vertices, ends in pieces]
    return None if any(len(vertices) < 2 for vertices, _ in pieces) else pieces


def _given_polyline(given: str, keep_vertices: bool) -> list | None:
    text = (given or "").strip()
    if text.startswith(("[", "{")):
        try:
            return _polyline_pieces(json.loads(text), keep_vertices)
        except ValueError:
            return None
    pieces = []
    for chunk in re.split(r"\s*[;\n]\s*", text):
        if not chunk:
            continue
        typed = _typed_points(chunk)
        if typed is None or len(typed) < 2:
            return None
        ends = (typed[0][0] or "closed", typed[-1][0] or "closed")
        vertices = _simplified([point for _, point in typed], keep_vertices)
        if len(vertices) < 2:
            return None
        pieces.append((vertices, ends))
    return pieces or None


def _piece_matches(expected, given) -> bool:
    (want, want_ends), (got, got_ends) = expected, given
    if got == want:
        return want_ends is None or (got_ends or ("closed", "closed")) == want_ends
    if got[::-1] == want:
        return want_ends is None or (got_ends or ("closed", "closed"))[::-1] == want_ends
    return False


def _piece_text(piece) -> str:
    vertices, ends = piece
    marks = ["", ""] if ends is None else ["o " if e == "open" else "" for e in ends]
    texts = [_point_text(v) for v in vertices]
    texts[0], texts[-1] = marks[0] + texts[0], marks[1] + texts[-1]
    return " ".join(texts)


def polyline_answer_text(answer) -> str:
    """The canonical for a polyline question: each piece's vertices in order,
    "o " before an open end, pieces separated by "; ". Generators call this."""
    keep_vertices = isinstance(answer, dict) and bool(answer.get("keep_vertices"))
    return "; ".join(_piece_text(piece) for piece in _polyline_pieces(answer, keep_vertices))


@_multi_part_aware
def grade_polyline(question, given: str) -> GradeResult:
    canonical = question.answer_text
    keep_vertices = isinstance(question.answer, dict) and bool(question.answer.get("keep_vertices"))
    expected = _polyline_pieces(question.answer, keep_vertices)
    if not expected:
        return GradeResult(correct=False, canonical=canonical)
    pieces = _given_polyline(given, keep_vertices)
    if pieces is None:
        return GradeResult(correct=False, canonical=canonical, given_normalised=None)
    given_normalised = "; ".join(_piece_text(piece) for piece in pieces)
    if len(pieces) != len(expected):
        return GradeResult(correct=False, canonical=canonical, given_normalised=given_normalised)
    remaining = list(expected)
    for piece in pieces:
        match = next((i for i, want in enumerate(remaining) if _piece_matches(want, piece)), None)
        if match is None:
            return GradeResult(correct=False, canonical=canonical, given_normalised=given_normalised)
        del remaining[match]
    return GradeResult(correct=True, canonical=canonical, given_normalised=given_normalised)


def line_answer_text(points) -> str:
    """The canonical for a line question: the two points the generator chose."""
    return points_answer_text(points)


@_multi_part_aware
def grade_line(question, given: str) -> GradeResult:
    canonical = question.answer_text
    target = _points(question.answer)
    if not target or len(target) != 2 or target[0] == target[1]:
        return GradeResult(correct=False, canonical=canonical)
    points = _given_points(given)
    if points is None:
        return GradeResult(correct=False, canonical=canonical, given_normalised=None)
    given_normalised = " ".join(_point_text(p) for p in points)
    (x1, y1), (x2, y2) = target
    a, b = y2 - y1, x1 - x2
    c = a * x1 + b * y1
    correct = len(points) == 2 and points[0] != points[1] and all(a * x + b * y == c for x, y in points)
    return GradeResult(correct=correct, canonical=canonical, given_normalised=given_normalised)


GRADERS: dict[str, Callable] = {
    "number": grade_number,
    "fraction": grade_fraction,
    "pair": grade_pair,
    "list": grade_list,
    "ordered_list": grade_ordered_list,
    "expression": grade_expression,
    "mcq": grade_mcq,
    "table": grade_table,
    "set": grade_set,
    "points": grade_points,
    "polyline": grade_polyline,
    "line": grade_line,
}


# -- option_notes resolution --------------------------------
#
# `Question.option_notes` (mathgen/core.py) maps an option's exact text to a
# misconception id: an id in the shared catalogue (mathgen/misconceptions.py),
# or one in the *skill's* own `misconceptions` or `near_misses` list -- the
# question doesn't carry the note text itself, since the same wrong-thinking
# id can be shared across every question a generator makes. Resolution only
# ever runs when the answer graded incorrect. A reply is matched to an option
# by its spelling first, then -- for an option whose structured value the
# generator recorded in `distractor_values` -- by grading the reply as though
# that option were the answer, so "x=-3" matches the option "x = -3". A match
# fills `misconception` and `matched_distractor`; `note` is filled only when
# the id resolves (the skill's wording first, then the catalogue's), and a
# miss leaves all four at their `None` default -- never guess a note.


def _matched_option_text(kind: str, question, given: str, given_normalised: str | None) -> str | None:
    """The exact `option_notes` key the learner's answer hits, or None.

    For `mcq`, `given` is the selected option's *index* (see `grade_mcq`),
    not its text, so the match has to go through `given_normalised` -- which
    `grade_mcq` already sets to the selected option's own text on a valid
    index. Falling back to the raw `given` for `mcq` would be wrong: it is a
    digit string, and a coincidentally-matching option_notes key (an option
    whose text happens to look like a valid index) would fire a note nobody
    intended.

    For every other kind, `given` is free text typed by the learner; a hit
    requires it to match an `option_notes` key verbatim once stripped.
    """
    option_notes = getattr(question, "option_notes", None)
    if not option_notes:
        return None
    if kind == "mcq":
        return given_normalised if given_normalised in option_notes else None
    text = (given or "").strip()
    if text in option_notes:
        return text
    if given_normalised is not None and given_normalised in option_notes:
        return given_normalised
    return None


# The answer attributes a grader reads, cleared on the stand-in question a reply is graded against when it is matched
# to a distractor: the distractor has no alternative answer, no required form and no options of its own.
_STAND_IN_CLEARED = {
    "answer_latex": "",
    "answer_alt": None,
    "answer_form": None,
    "distractors": [],
    "option_notes": {},
    "distractor_values": {},
    "distractor_credit": {},
}


def _graded_option_match(kind: str, question, given: str) -> str | None:
    """The first tagged option the reply is a correct answer to, or None.

    Each such option is made the answer of a shallow copy of `question` and the reply graded against it with the
    same grader, so the match is exactly as forgiving as grading is: spacing, labels, order where order does not
    count, equivalent forms. Duck-typed, so a question part is matched the same way.
    """
    option_notes = getattr(question, "option_notes", None) or {}
    values = getattr(question, "distractor_values", None) or {}
    folded = " ".join((given or "").split()).casefold()
    for text in getattr(question, "distractors", None) or ():
        if text not in option_notes:
            continue
        if values.get(text) is None:
            # No value to grade against ("undefined", "does not exist"): the spelling, in any case and spacing.
            if folded and folded == " ".join(text.split()).casefold():
                return text
            continue
        stand_in = copy.copy(question)
        for attr, value in {**_STAND_IN_CLEARED, "answer": values[text], "answer_text": text}.items():
            setattr(stand_in, attr, value)
        try:
            if GRADERS[kind](stand_in, given).correct:
                return text
        except Exception:  # noqa: BLE001 - a diagnosis that cannot be made is no diagnosis, never a crash
            logger.warning("matching %r to option %r failed", given, text, exc_info=True)
    return None


def _resolve_note(kind: str, question, given: str, result: GradeResult, skill) -> None:
    """Fill `result`'s `misconception`, `matched_distractor`, `note` and `note_kind` for a graded-wrong reply.

    The matched option's id resolves against `skill.misconceptions`, then
    `skill.near_misses` (`validate()` in mathgen/skills.py rejects an id that
    collides between the two, so this order never matters for a valid skill),
    then the shared catalogue. `skill` is duck-typed rather than imported, so
    this module never depends on `mathgen.skills` -- that module already
    imports this one.
    """
    matched = _matched_option_text(kind, question, given, result.given_normalised)
    if matched is None and kind != "mcq":
        matched = _graded_option_match(kind, question, given)
    if matched is None:
        return
    note_id = question.option_notes[matched]
    result.misconception, result.matched_distractor = note_id, matched
    resolved = describe_misconception(note_id, skill)
    if resolved is not None:
        result.note, result.note_kind = resolved


# -- the minus sign ----------------------------------------------------------------------
#
# Every grader reads "-" as a minus and nothing else: the typographic minus U+2212, which a phone keyboard,
# a word processor or a pasted worksheet puts in its place, was read as no sign at all ("x = 3, y = \u22126"
# was accepted for y = 6, "{\u22121, 2, 3}" for {1, 2, 3}) or as an unparseable symbol ("\u2212x", "x \u2212 2").
# `grade` therefore spells the minus the one way before any grader reads the reply:
#
#   * U+2212 (minus), U+FE63 (small hyphen-minus) and U+FF0D (fullwidth hyphen-minus) always become "-".
#   * U+2012-U+2015 (figure dash, en dash, em dash, horizontal bar) become "-" in a number or an expression,
#     where a range ("5\u20139") is not a thing a reply says, and in the other kinds only where they are a sign
#     (not right after a word character, a digit, "." or a closing bracket, and right before a number): a
#     range written with a dash stays a range. No answer text in the library holds one of these dashes.
#   * A multiple-choice reply is an option's text or index, so it is not changed.
#
# The spoken-transcript readers (`mathgen.grading.spoken`) are not called from here and do not change.
_ASCII_MINUS = str.maketrans({c: "-" for c in "\u2212\ufe63\uff0d"})
_DASHES = "\u2012\u2013\u2014\u2015"
_ALL_DASHES = str.maketrans({c: "-" for c in _DASHES})
_SIGN_DASH_RE = re.compile(r"(?<![\w.)\]}])[\u2012-\u2015](?=[ \t]*[\d.$\u00a3\u20ac\u00a5])")
_DASH_AS_MINUS_KINDS = frozenset({"number", "expression"})
_MINUS_FREE_KINDS = frozenset({"mcq"})


def _spell_minus_as_ascii(kind: str, given: str) -> str:
    """`given` with the minus sign and its lookalikes spelled "-" (see the block comment above)."""
    if kind in _MINUS_FREE_KINDS:
        return given
    given = given.translate(_ASCII_MINUS)
    if kind in _DASH_AS_MINUS_KINDS:
        return given.translate(_ALL_DASHES)
    return _SIGN_DASH_RE.sub("-", given)


# A minus that is a sign, with a space between it and its number ("y = - 6", "(3, - 6)", "{- 1, 2, 3}", "3, - 6",
# "x = 3 or x = - 3"), is a negative number in the kinds that read values. The rule is the number grader's: a
# minus that does not follow a value (a word character, ".", a closing bracket) is a sign, looking back over one
# space, so "5 - 6" and "x - 6" are still a subtraction; "or" and "and" are separators, not values. Done here and
# not in `_spell_minus_as_ascii`, which spells the characters and nothing else.
_SIGNED_VALUE_KINDS = frozenset(
    {"pair", "list", "set", "ordered_list", "points", "polyline", "line", "table", "fraction"}
)
_SPACED_SIGN_RE = re.compile(
    r"(?:(?<![\w.)\]}])(?<![\w.)\]}][ \t])|(?<=\bor[ \t])|(?<=\band[ \t]))-[ \t]+(?=\.?\d)", re.IGNORECASE
)


def _attach_spaced_sign(kind: str, given: str) -> str:
    return _SPACED_SIGN_RE.sub("-", given) if kind in _SIGNED_VALUE_KINDS else given


# Two minus signs on one number: "--3", "- -3", "(3, --6)". The value readers took the second sign and dropped the
# first, so "--3" was accepted for -3, though it is 3. No answer is written that way, so in the kinds that read
# values such a reply is wrong; `expression` reads it correctly (as 3) and is left to do so.
_DOUBLE_SIGN_KINDS = _SIGNED_VALUE_KINDS | {"number"}
_DOUBLE_SIGN_RE = re.compile(r"-[ \t]*-[ \t]*\.?\d")


# The question pasted back as the reply. The value readers take the number after the last "=", so the question
# itself could grade correct: "Solve 2x/2 = 4" for x = 4, or "... satisfy p·u + q·w = i - 2j ... Find q" for -2.
# A reply is a copy when a stretch of prose from the question makes it up (twelve characters or more, with a
# letter), or when what stands before its last "=" is an expression with a variable (a digit and a letter) that
# the question shows. Compared without spaces or case. The kinds whose answers never repeat the question's text.
_COPY_CHECKED_KINDS = frozenset({"number", "fraction", "pair"})


def _squeezed(text: str) -> str:
    return re.sub(r"\s+", "", (text or "").translate(_MINUS_SIGNS)).lower()


def _copies_the_question(question, given: str) -> bool:
    asked = _squeezed(getattr(question, "text", ""))
    reply = _squeezed(given)
    if not asked or not reply:
        return False
    if len(reply) >= 12 and re.search(r"[a-z]", reply) and reply in asked:
        return True
    label = reply.rpartition("=")[0]
    return bool(label and re.search(r"[a-z]", label) and re.search(r"\d", label) and label + "=" in asked)


#: Longest reply `grade` reads at all. The longest answer in the library, table answers included, is a few hundred
#: characters; this is a bound on the work (about 0.3 s at the limit), not a judgement on any real reply. A
#: reply shorter than this still reaches its grader, which truncates what it logs (see test_logged_reply_is_truncated).
MAX_REPLY_LENGTH = 100_000
_WHITESPACE_RUN_RE = re.compile(r"\s+")


def _one_gap(match: re.Match) -> str:
    return "\n" if "\n" in match.group() else " "


def normalise_reply(kind: str, given: str) -> str:
    r"""`given` as every grader of `kind` reads it.

    The parsers are not all linear in a long run of whitespace (a separator written `\s*,\s*` is tried from
    every space of the run, and several together are quartic), so no run reaches them longer than one
    character: a run that holds a line break is that one line break, any other run one space. Nothing a
    parser reads is in the width of a gap."""
    return _attach_spaced_sign(kind, _spell_minus_as_ascii(kind, _WHITESPACE_RUN_RE.sub(_one_gap, given)))


def grade(kind: str, question, given, *, skill=None) -> GradeResult:
    """`skill` is optional and keyword-only: every pre-existing
    call site is unaffected. A wrong reply that matches a tagged distractor
    gets `misconception` and `matched_distractor` either way; its `note` comes
    from the skill's own `misconceptions`/`near_misses` when `skill` is passed
    and lists the id, else from the shared catalogue (mathgen/misconceptions.py).

    A multi-part question (`question.parts`) is graded part by part, each with its own grader, whatever
    `kind` is; `given` may then be a dict of replies keyed by part label (see `mathgen.grading.parts`)."""
    if kind not in GRADERS:
        raise KeyError(f"unknown answer kind {kind!r}; known: {', '.join(sorted(GRADERS))}")
    if getattr(question, "parts", None):
        from .parts import grade_parts

        return grade_parts(question, given, skill=skill)
    if isinstance(given, str) and len(given) > MAX_REPLY_LENGTH:
        # A megabyte of text is seconds of work for nothing: no answer in the library, in any spelling, is a
        # fraction of this long.
        return GradeResult(correct=False, canonical=getattr(question, "answer_text", ""))
    if isinstance(given, str):
        given = normalise_reply(kind, given)
    if isinstance(given, str) and kind in _DOUBLE_SIGN_KINDS and _DOUBLE_SIGN_RE.search(given):
        return GradeResult(correct=False, canonical=getattr(question, "answer_text", ""))
    if isinstance(given, str) and kind in _COPY_CHECKED_KINDS and _copies_the_question(question, given):
        return GradeResult(correct=False, canonical=getattr(question, "answer_text", ""))
    try:
        result = GRADERS[kind](question, given)
    except Exception:
        # The narrow fix (a known exception type caught where the input is parsed, e.g.
        # `_parse_expression`'s except tuple) is still the right place to handle a *known* input
        # shape; this is the backstop for the shape nobody has hit yet. `logging.exception` is not
        # optional here: a silent `except Exception: pass` would turn a future generator/grader bug
        # into "every answer is wrong" with nothing in the logs, which is worse than the crash it
        # replaces -- this log line is what would surface that instead of hiding it.
        logger.exception("grade(kind=%r, given=%.200r) raised; grading incorrect", kind, given)
        return GradeResult(correct=False, canonical=getattr(question, "answer_text", ""))
    if not result.correct:
        _resolve_note(kind, question, given, result, skill)
    return result
