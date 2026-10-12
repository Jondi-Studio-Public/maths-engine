"""Learner-facing text checks shared by the build-topic test and the render script.

The regrade sweep proves a generator's *grading* is right; nothing in it reads the *prompt*, so a question
can grade perfectly and still read badly. Examples are ``Simplify: 5**3 × 5**3`` (Python exponent notation),
``Estimate 192 * 544`` (a bare ``*`` operator) and ``1 more parts are used`` (a count of one with a plural
noun). These patterns are the mechanical subset of that check: cheap, unambiguous and checkable on
every seed. Wording that merely reads badly still needs the rendered output and a human skim.

Used by the render script's ``flags`` column and exit status
(``scripts/render_skill_questions.py``) and by any test that checks the
learner-facing text of build topics. Keep them in step by editing this
module, never by copying the patterns.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

from mathgen.slots import CONTEXTS, NAMES, OBJECTS, UNITS

# Words that legitimately follow "1 " and end in "s" without being a plural
# noun: verbs, prepositions and operator words. Extend when a false positive
# is proven with a seed, never to silence a real plural.
NOT_PLURAL_NOUNS = frozenset(
    "is plus minus times has was does means equals gives less this as us its "
    "thus versus makes needs takes goes comes shows leaves costs holds contains "
    "yes across always gets uses scores lists wins loses moves starts ends "
    "receives lands draws rolls picks chooses spends earns pays saves sells buys "
    "reads writes says adds subtracts multiplies divides".split()
)

# A capitalised label before the 1 ("Player 1 gets", "Year 1 students",
# "Table 1 lists") names a thing numbered one, not a count of one. Only
# these words count as labels, so a sentence-initial verb ("Write 1
# hundredths") is still checked.
LABEL_WORDS = frozenset(
    "Player Team Year Grade Group Step Stage Level Round Table Figure Question "
    "Part Option Card Box Bag Jar Row Column Line Page Day Week Term Unit Set "
    "Trial Game Sample Version Method Rule Case Item Task Section Chapter Class "
    "Room Bus Train Track Lane Door Shelf Bin Tray Batch Lot Ticket Seat Machine "
    "Robot Car Boat Plane Route Flight Path Lap Heat Race Match Test Quiz".split()
)

_SINGULAR_COUNT_RE = re.compile(r"(?<![\d.,])1 (?:more |extra |fewer |other |less )?([a-z]+s)\b")
_LABEL_BEFORE_RE = re.compile(r"([A-Z][a-z]+) $")


def _singular_count_plural(text: str) -> str | None:
    for match in _SINGULAR_COUNT_RE.finditer(text):
        if match.group(1) in NOT_PLURAL_NOUNS:
            continue
        label = _LABEL_BEFORE_RE.search(text[: match.start()])
        if label and label.group(1) in LABEL_WORDS:
            continue
        return match.group(0)
    return None


def _regex(pattern: str) -> Callable[[str], str | None]:
    compiled = re.compile(pattern)

    def find(text: str) -> str | None:
        match = compiled.search(text)
        return match.group(0) if match else None

    return find


@dataclass(frozen=True)
class TextPattern:
    name: str
    #: "prompt" checks only ``Question.text``; "learner" also checks
    #: ``answer_text`` and every distractor (what the feedback screen shows).
    scope: str
    find: Callable[[str], str | None]
    why: str


BAD_TEXT_PATTERNS: tuple[TextPattern, ...] = (
    TextPattern(
        "double_star",
        "learner",
        _regex(r"\*\*"),
        "Python exponent notation; write the power in words, with a caret, or in the latex field",
    ),
    TextPattern(
        "star_operator",
        "prompt",
        _regex(r"(?<![*\w])\*(?![*\w])|\d\*[\dxa-z(]"),
        "an asterisk as the multiplication sign; use the multiplication sign or words",
    ),
    TextPattern(
        "singular_count_plural",
        "prompt",
        _singular_count_plural,
        "a count of one followed by a plural noun",
    ),
    TextPattern(
        "signed_after_operator",
        "learner",
        _regex(r"[+\u2212-]\s+-\s*\d"),
        'a negative value written straight after an operator, as "17 + -14". Use '
        'core.join_terms where the expression should simplify ("17 - 14"), or '
        "core.paren_negative where the operator is the thing being taught "
        '("17 + (-14)")',
    ),
    TextPattern(
        "double_punctuation",
        "learner",
        _regex(r"(?<!\.)\.\.(?!\.)|(?<!\.)\.\s*;|,\s*,"),
        'two sentence marks in a row, as "cost $9.." or "per hour.;" -- an ellipsis is '
        "not one. Usually a "
        "list whose items already end in a full stop joined with a separator that "
        "adds another -- fix the join, or drop the item punctuation",
    ),
    TextPattern(
        "newline_in_text",
        "learner",
        _regex(r"[\r\n]"),
        "a line break the learner never sees: question text is rendered into a span "
        "with no white-space rule and no newline-to-<br> step "
        "(the front-end renderer), so it collapses to nothing. Write prose that "
        "reads correctly joined, or put the structure in the latex field",
    ),
)


def learner_facing_fields(question) -> list[tuple[str, str]]:
    """``[(field_name, value), ...]`` for every string a learner can see."""
    fields = [("text", question.text), ("answer_text", question.answer_text)]
    fields.extend((f"distractor[{i}]", d) for i, d in enumerate(question.distractors))
    return [(name, value) for name, value in fields if isinstance(value, str)]


def working_fields(question) -> list[tuple[str, str]]:
    """``[("working[i]", line), ...]`` for the worked solution the learner reads.

    Separate from:func:`learner_facing_fields` because ``working`` was outside every check until recently
    and the two are being brought together under a ratchet rather than in one step: a worked line is prose the
    learner reads, so the prompt bar is the right bar.
    """
    working = getattr(question, "working", None) or []
    return [(f"working[{i}]", line) for i, line in enumerate(working) if isinstance(line, str)]


def find_text_defects(question, *, include_working: bool = False) -> list[str]:
    """Every defect in one question as ``"<pattern> in <field>: <match>"``.

    Empty when the question is clean. Deterministic: the same question always
    yields the same list in the same order.

    ``include_working`` additionally checks the worked-solution lines. It
    defaults to False so this function's result is unchanged for callers that
    only check the prompt and answer.
    """
    defects: list[str] = []
    for pattern in BAD_TEXT_PATTERNS:
        for field_name, value in learner_facing_fields(question):
            if pattern.scope == "prompt" and field_name != "text":
                continue
            match = pattern.find(value)
            if match is not None:
                defects.append(f"{pattern.name} in {field_name}: {match!r}")
        if include_working:
            for field_name, value in working_fields(question):
                match = pattern.find(value)
                if match is not None:
                    defects.append(f"{pattern.name} in {field_name}: {match!r}")
    return defects


# --- rounding instructions -------------------------------------------------
#
# A question that asks the learner to round states the precision its answer
# must carry. Both `mathgen/grading/typed.py` (which refuses an answer that has not been
# rounded as asked) and any test that requires a generator's own answer_text to
# carry that precision need to read the instruction, and
# they must read it identically -- a grader stricter than the generator marks
# correct answers wrong, and a generator stricter than the grader is untested.
# Keep the two in step by editing this function, never by copying the patterns.

# "2 decimal places", "two decimal places", and the abbreviations "(1 d.p.)", "to 2dp", "3 dp". A count
# that is the upper end of a range ("1-2 dp") or part of a larger number ("0.5 dp") is not a count. An
# abbreviation is read only when the text has no other rounding instruction: questions use it for the
# precision of the data they give ("the ratios (each rounded to 4 d.p.) are ...") as well as for the answer.
_DECIMAL_PLACES_RE = re.compile(r"(?<![\d.\-\u2013/])(\d+|one|two|three|four)\s*decimal\s+place", re.I)
_DP_ABBREVIATION_RE = re.compile(r"(?<![\d.\-\u2013/])(\d+|one|two|three|four)\s*d\.?\s?p\b", re.I)
_PLACE_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4}
_NEAREST_PLACES = {
    "whole": 0,
    "integer": 0,
    "unit": 0,
    "tenth": 1,
    "hundredth": 2,
    "thousandth": 3,
    # Money to the cent is two places; the rest are a whole number of that unit. Only units a
    # generator's question text really uses (plus metre) are listed: "ten", "hundred", "half",
    # "quarter", "multiple", "lower" and so on are not place counts and stay out.
    "cent": 2,
    "dollar": 0,
    "degree": 0,
    "newton": 0,
    "minute": 0,
    "metre": 0,
    "kilometre": 0,
    "kilojoule": 0,
    "nautical": 0,
}
_NEAREST_RE = re.compile(
    r"nearest\s+(whole|integer|unit|tenth|hundredth|thousandth|cent|dollar|degree|newton|minute|metre|kilometre|"
    r"kilojoule|nautical)\b",
    re.I,
)


def requested_rounding(text: str) -> int | None:
    """Decimal places the question asks the learner to round to, or None.

    Covers the phrasings the library uses: "to 2 decimal places" / "to two
    decimal places", the abbreviations "(1 d.p.)" / "2dp", "to the nearest
    hundredth", "to the nearest cent" (2 places) and "to the nearest degree" and
    other whole-unit phrasings (0 places). Missing the "nearest hundredth" form
    first hid a real defect -- "Round 16.201 to the nearest
    hundredth" answered "16.2" instead of "16.20".

    Returns None when the text states MORE THAN ONE DISTINCT decimal-place count. A single-value answer (the
    only kind `mathgen/grading/typed.py:_is_rounded_as_asked` ever checks this against -- `grade_number`,
    never `grade_pair`/`grade_list`) can only ever have ONE precision asked of it; a text with two DIFFERENT
    counts is unambiguous only per-component, which this whole-text-level function cannot resolve, so it
    correctly declines rather than picking one count and silently misapplying it to every number in the
    answer. Counts from "nearest ..." phrases and from spelled-out decimal-place phrases are
    pooled; a "d.p." abbreviation counts only when neither is present, because a question also uses it
    for the precision of the values it gives. A repeated STATEMENT of the SAME count ("x and y, each rounded to 1 decimal place") is not
    ambiguous and still resolves normally.
    """
    counts = {_PLACE_WORDS.get(tok.lower()) or int(tok) for tok in _DECIMAL_PLACES_RE.findall(text or "")}
    counts |= {_NEAREST_PLACES[unit.lower()] for unit in _NEAREST_RE.findall(text or "")}
    if not counts:
        counts = {_PLACE_WORDS.get(tok.lower()) or int(tok) for tok in _DP_ABBREVIATION_RE.findall(text or "")}
    if len(counts) == 1:
        return next(iter(counts))
    return None


# --- prompt_shape ------------------------------------------------------

#: A signed numeric token: an optional leading `-` directly against the
#: digits (never with a space before the digits, so "17 - 14"'s operator
#: `-` is untouched but "(-5, 0, 1)"'s sign is part of the value), then a
#: run of digits, optionally with a decimal point. Mirrors
#: `mathgen.core._VALUE_TOKEN_RE`'s own `-?\d+(?:\.\d+)?` prefix -- so this
#: agrees with how `core._distractor_signature` tokenises a number -- but
#: deliberately WITHOUT that regex's fraction-combining suffix, which
#: swallows "1/2" as a single token; this needs "1" and "2" separately to
#: tell "1/2" apart from "3/4" (see `_mask_digits` below). The leading sign
#: has to be included: an EARLIER version of this regex left it out, and a
#: vector-addition question's `(-5, 0, 1)` versus `(4, 4, -2)` then differed
#: only in WHICH position carried a bare `-`, which is exactly the kind of
#: value variation (which component happened to be drawn negative) this
#: mask exists to hide -- `vector-addition` alone showed ~17 "distinct
#: shapes" in 20 seeds before this fix, all sign-pattern noise.
_DIGIT_RUN_RE = re.compile(r"-?\d+(?:\.\d+)?")

# A literal `1`, not itself part of a longer digit run (`(?<![\d.])`), that
#: is the NUMERATOR of a written fraction -- immediately followed by
#: optional whitespace, `/`, optional whitespace, then a digit. Deliberately
#: this narrow: an EARLIER version of this mask preserved every bare literal
#: `1` anywhere in the text, on the theory that `1` is a structural marker
#: the way it already is in this module's `NOT_PLURAL_NOUNS`/
#: `_SINGULAR_COUNT_RE`. Measured against the full library
#:, that was far too broad: `vector-addition`'s
#: easy tier alone showed 20 "distinct shapes" in 20 seeds, because vector
#: components are small integers drawn from a range that routinely includes
#: 1 -- whether a *component* happens to be 1 is exactly the kind of bare
#: magnitude variation `_mask_digits` exists to hide, not a structural
#: signal. A fraction numerator of 1 is different in kind: "1/r" (root
#: only) versus "m/r" with m != 1 (root then power) is a difference in what
#: OPERATION the question asks for, not in which number was drawn -- and
#: that is only true when the `1` sits in a written fraction. So only that
#: context is protected.
_UNIT_FRACTION_NUMERATOR_RE = re.compile(r"(?<![\d.])1(?=\s*/\s*\d)")

#: A private-use sentinel `_mask_digits` uses to shield a protected `1`
#: through the blanket digit substitution below, then restores. Never
#: appears in real question text (no generator emits U+E000), so it cannot
#: collide with genuine content.
_PROTECT_SENTINEL = ""


def _mask_digits(text: str) -> str:
    """Replace every digit run with `#`, except a fraction numerator of `1`
    ("1/2", "1 / 3") -- see `_UNIT_FRACTION_NUMERATOR_RE` above for why only
    that narrow context is protected and not every bare `1`.

    Validated against the depth-and-cleanup note's own named false-outcome:
    `senior-use-radicals-and-convert-to-and-from-fractional` (topic `radicals_fractional_indices`) renders
    "Evaluate 25^(1/2)." at its easy tier and "Evaluate 81^(3/4)." at its hard tier -- a single-step root
    extraction (exponent always exactly `1/r`) versus a root-then-power (exponent `m/r` with `m` drawn from
    2..r-1, so `m` is never 1). Proved against this skill's actual renders (and against the over-broad first
    attempt's regression on `vector-addition` et al.) in `tests/test_exd_instrument.py`.
    """
    protected = _UNIT_FRACTION_NUMERATOR_RE.sub(_PROTECT_SENTINEL, text)
    masked = _DIGIT_RUN_RE.sub("#", protected)
    return masked.replace(_PROTECT_SENTINEL, "1")


def _bank_pattern(values) -> re.Pattern:
    """Compile one alternation over `values`, longest-first so a multi-word
    phrase ("a bakery") is tried before any word inside it could be, whole-
    word (`\\b`) for anything with an actual word character, and WITHOUT a
    word boundary for a bare symbol (`$`) -- `\\b` is defined at a
    word/non-word transition, and a symbol with non-word characters on both
    sides never has one, so a `\\b`-wrapped `\\$` would simply never match.
    """
    parts = []
    for value in sorted(set(values), key=len, reverse=True):
        escaped = re.escape(value)
        parts.append(rf"\b{escaped}\b" if re.search(r"\w", value) else escaped)
    return re.compile("|".join(parts), re.IGNORECASE)


_NAME_RE = _bank_pattern(NAMES)
_OBJECT_RE = _bank_pattern(word for pair in OBJECTS for word in pair)
_UNIT_RE = _bank_pattern(UNITS)
_CONTEXT_RE = _bank_pattern(CONTEXTS)


def prompt_shape(question) -> str:
    """The canonical "shape" of `question`'s prompt: digits and every
    `mathgen.slots` bank value masked, whitespace collapsed.

    Uses `.text`, never `.latex`/`.answer_text`: the prompt is what a
    learner reads BEFORE answering, which is the same reader's-first-read
    surface `learner_facing_fields`/`find_text_defects` already treat as
    primary (`.latex` is a rendering of the same prose for KaTeX, not a
    second independent surface; `.answer_text` is read only after the
    learner has already committed to an answer, so it cannot be what
    decided whether the question FELT different going in).

    Order: digits first (`_mask_digits`), then every slot-bank category
    (name, object singular/plural, unit, context) case-insensitively to its
    own placeholder so a genuine sentence-structure difference survives
    while a name/object/unit/context swap does not, then whitespace
    collapsed to single spaces.

    ACCEPTED LIMITATION, by construction: a property that is invisible in
    the rendered text but present in the underlying arithmetic -- whether
    `multidigit_multiplication`'s hard tier forces a carry the easy tier
    does not -- cannot be seen here, because "47 x 3" and "530 x 12" both
    render as the same masked shape regardless of whether multiplying them
    out requires carrying. That is not a bug to chase inside this function:
    it is a genuinely different question class of defect (an arithmetic
    property, not a textual one) from the digit-run mask fixed above, whose
    defect WAS visible in the text (a literal `1` versus not).
    """
    text = _mask_digits(question.text or "")
    text = _NAME_RE.sub("§NAME§", text)
    text = _OBJECT_RE.sub("§OBJ§", text)
    text = _UNIT_RE.sub("§UNIT§", text)
    text = _CONTEXT_RE.sub("§CTX§", text)
    return " ".join(text.split())


# --- Python notation, checked on what a learner is shown ---------------------------------------------------------
#
# Generators keep building strings in Python's spelling (`x**2`, `2*x`, `sqrt(21)`, `exp(-x)`, `pi`) and graders
# and tests read that spelling. `core.display_question` rewrites it as it is on paper (`core.paper_notation`), and
# these patterns are what that rewrite must leave none of. They are separate from `BAD_TEXT_PATTERNS` because they
# apply to the displayed strings, not the stored ones.

PYTHON_NOTATION_PATTERNS: tuple[TextPattern, ...] = (
    TextPattern("double_star", "learner", _regex(r"\*\*"), "Python exponent notation"),
    TextPattern("star_operator", "learner", _regex(r"(?<![*\w])\*(?![*\w])|\d\*[\dxa-z(]"), "an asterisk as times"),
    TextPattern("sqrt_call", "learner", _regex(r"(?<![A-Za-z])sqrt\s*\("), "Python's sqrt( )"),
    TextPattern("exp_call", "learner", _regex(r"(?<![A-Za-z])exp\s*\("), "Python's exp( )"),
    TextPattern("spelled_pi", "learner", _regex(r"(?<![A-Za-z])pi(?![A-Za-z])"), "the word pi for the symbol"),
)


def find_python_notation(shown: dict) -> list[str]:
    """Every Python spelling left in a `core.display_question` result, as ``"<pattern> in <field>: <match>"``."""
    fields = [("text", shown["text"]), ("answer_text", shown["answer_text"])]
    fields += [(f"option[{i}]", o) for i, o in enumerate(shown["options"])]
    fields += [(f"working[{i}]", w) for i, w in enumerate(shown["working"])]
    for part in shown.get("parts", ()):
        fields += [(f"part {part['label']} {name}", value) for name, value in part.items() if isinstance(value, str)]
    found = []
    for pattern in PYTHON_NOTATION_PATTERNS:
        for name, value in fields:
            match = pattern.find(value)
            if match is not None:
                found.append(f"{pattern.name} in {name}: {match!r}")
    return found
