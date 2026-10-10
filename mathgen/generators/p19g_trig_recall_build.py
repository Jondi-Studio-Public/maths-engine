"""Exact trig-value recall and unit-circle coordinate statement, both ``acara-senior-v8-4``, domain
``trigonometry``.

Two generators, a deliberate BOUNDARY pair (see each docstring):

- ``p19g_exact_trig_value``
  (``senior-recognise-the-exact-values-of-cos-theta-sin``, ACMMM035) --
  given theta as a stated integer multiple of pi/6 or pi/4, and which of
  cos(theta), sin(theta) or tan(theta) is asked for, state its exact value
  in simplified surd form. States a SINGLE named value. `answer: mcq`.
- ``p19g_unit_circle_coordinates``
  (``senior-understand-the-unit-circle-definition-of-cos-theta``, ACMMM029)
  -- given theta as a stated standard angle, state the FULL coordinate
  PAIR (cos theta, sin theta) together. `answer: expression` (a
  sympy-parseable Tuple string).

Neither asks about a plotted terminal point -- that is the shipping, already-live sibling
``senior-understand-the-unit-circle-definition-of-cos-theta-acmmm034`` (ACMMM034), whose kernel DERIVES tan
from a supplied terminal point (the REVERSE direction: point given, ratio computed).

**GRADER-KIND DECISIONS (load-bearing, probed directly against
``mathgen/grading/typed.py`` -- do not deviate):**

- Skill 1 is ``answer: mcq``, never ``expression`` -- typing "undefined" or
  "zoo" for tan(theta) at an odd multiple of pi/2 cannot be graded through
  ``grade_expression``: ``_parse_expression`` reads "undefined" as a
  product of six single-letter symbols, never a keyword; "zoo" parses
  correctly on both sides but ``symbolically_equal`` compares via
  ``simplify(zoo - zoo)``, which is ``nan``, not ``0`` -- infinity minus
  infinity is undefined in sympy's own arithmetic, so the expression grader
  can never confirm ``zoo == zoo``. Same design as the shipping ACMMM034
  skill (``p7a_unit_circle_aux_angle_build.py::unit_circle_tan_evaluate``):
  a plain mcq option list, index-graded by ``grade_mcq``, "undefined" never
  parsed as an expression.
- Skill 2 is ``answer: expression``, with ``answer_text`` a Tuple-shaped
  string like ``"(sqrt(3)/2, 1/2)"`` -- ``grade_expression`` parses a
  parenthesised comma pair into a ``sympy.Tuple`` and ``symbolically_equal``'s
  ``_as_sequence``/``pairs_equal`` branch grades it exact-surd-strict
  (rejects a decimal approximation) and spacing-tolerant (accepts
  "(sqrt(3)/2,1/2)", "sqrt(3)/2, 1/2"). Never ``answer: pair``
  (``grade_pair`` routes every component through ``extract_values``, numbers
  only -- a surd component like ``sqrt(3)/2`` never extracts) and never
  ``answer: ordered_list`` (``grade_ordered_list``'s per-item comparison for
  a non-numeric item is an exact case-insensitive STRING match with no sympy
  normalisation -- "sqrt(3) / 2, 1/2" is rejected against "sqrt(3)/2, 1/2").

Both generators share one hand-built exact-value table (``_TABLE``, keyed by
``k`` where ``theta = k*pi/12``, for the 16 standard angles in ``[0, 2*pi)``
that are an integer multiple of pi/6 OR pi/4) -- verified ONCE at import
time against a wholly independent sympy route (``sympy.cos``/``sympy.sin``/
``sympy.tan`` evaluated directly, catching ``zoo`` as the undefined case),
never per-question. The test file's oracle re-derives the same way,
independently, per QUESTION (parsing only the rendered text, never this
module), over >= 1000 seeds for skill 1's mcq.
"""

from __future__ import annotations

import functools
import itertools
from fractions import Fraction

from sympy import cos as _scos, pi as _spi, sin as _ssin, sympify as _sympify, tan as _stan, zoo as _szoo

from ..core import (
    check,
    check_equivalent,
    format_pi_multiple,
    generator,
    latex_answer,
    pick_distractors,
)

# ===========================================================================
# Shared exact-value table
# ===========================================================================

# theta = k*pi/12 for k in _KVALS is exactly the set of integer multiples of
# pi/6 (k a multiple of 2) or pi/4 (k a multiple of 3) in [0, 2*pi).
_KVALS: tuple[int, ...] = (0, 2, 3, 4, 6, 8, 9, 10, 12, 14, 15, 16, 18, 20, 21, 22)

# k -> (cos_str, sin_str, tan_str_or_None); tan is None exactly where
# cos(theta) = 0 (k = 6, 18 -- theta = pi/2, 3*pi/2). Hand-built, matching
# the values a student would memorise; verified below against a wholly
# separate sympy route, once, at import time -- never re-derived from this
# table at question-construction time.
_TABLE: dict[int, tuple[str, str, str | None]] = {
    0: ("1", "0", "0"),
    2: ("sqrt(3)/2", "1/2", "sqrt(3)/3"),
    3: ("sqrt(2)/2", "sqrt(2)/2", "1"),
    4: ("1/2", "sqrt(3)/2", "sqrt(3)"),
    6: ("0", "1", None),
    8: ("-1/2", "sqrt(3)/2", "-sqrt(3)"),
    9: ("-sqrt(2)/2", "sqrt(2)/2", "-1"),
    10: ("-sqrt(3)/2", "1/2", "-sqrt(3)/3"),
    12: ("-1", "0", "0"),
    14: ("-sqrt(3)/2", "-1/2", "sqrt(3)/3"),
    15: ("-sqrt(2)/2", "-sqrt(2)/2", "1"),
    16: ("-1/2", "-sqrt(3)/2", "sqrt(3)"),
    18: ("0", "-1", None),
    20: ("1/2", "-sqrt(3)/2", "-sqrt(3)"),
    21: ("sqrt(2)/2", "-sqrt(2)/2", "-1"),
    22: ("sqrt(3)/2", "-1/2", "-sqrt(3)/3"),
}


def _verify_table() -> None:
    """Import-time proof, one wholly independent sympy route: for every k,
    ``sympy.cos(k*pi/12)`` / ``sin`` / ``tan`` (never ``_TABLE`` itself,
    never any other generator-internal helper) must equal the hand-built
    entry -- and ``tan`` must be exactly ``zoo`` (ComplexInfinity) where
    ``_TABLE`` says undefined, never anywhere else."""
    for k, (cos_str, sin_str, tan_str) in _TABLE.items():
        theta = Fraction(k, 12) * _spi
        check_equivalent(_sympify(cos_str), _scos(theta), f"cos table entry wrong at k={k}")
        check_equivalent(_sympify(sin_str), _ssin(theta), f"sin table entry wrong at k={k}")
        tan_value = _stan(theta)
        if tan_str is None:
            check(tan_value == _szoo, f"tan should be undefined (zoo) at k={k}")
        else:
            check(tan_value != _szoo, f"tan should be defined at k={k}")
            check_equivalent(_sympify(tan_str), tan_value, f"tan table entry wrong at k={k}")


_verify_table()


def _quadrant_of(k: int) -> int:
    """1-4 for a genuinely-in-a-quadrant k (mod 24), 0 for a quadrantal k (a
    multiple of 6 -- theta a multiple of pi/2)."""
    k = k % 24
    if k % 6 == 0:
        return 0
    return k // 6 + 1


def _theta_str(k: int) -> str:
    return format_pi_multiple(Fraction(k, 12))


def _compound_str(k: int) -> str | None:
    """A sum/difference-of-two-standard-angles rendering of theta, for the
    exam tier's kind-escalation (e.g. "pi - pi/6" for 5*pi/6) -- ``None``
    when ``k`` has no such decomposition in this family (quadrant I or a
    quadrantal k, where a compound form adds nothing genuinely new over the
    bare ``_theta_str``)."""
    kk = k % 24
    quadrant = _quadrant_of(kk)
    if quadrant == 2:
        return f"pi - {format_pi_multiple(Fraction(12 - kk, 12))}"
    if quadrant == 3:
        return f"pi + {format_pi_multiple(Fraction(kk - 12, 12))}"
    if quadrant == 4:
        return f"2pi - {format_pi_multiple(Fraction(24 - kk, 12))}"
    return None


def _degrees_str(k: int) -> str:
    return f"{k * 15} degrees"


def _flip_sign(value_str: str) -> str:
    if value_str.startswith("-"):
        return value_str[1:]
    if value_str == "0":
        return value_str
    return f"-{value_str}"


# ===========================================================================
# 1. p19g_exact_trig_value (answer: mcq)
# ===========================================================================

_COS_SIN_DISTRACTOR_POOL = ["0", "1", "-1", "1/2", "-1/2", "sqrt(2)/2", "-sqrt(2)/2", "sqrt(3)/2", "-sqrt(3)/2"]
_TAN_DISTRACTOR_POOL = ["0", "1", "-1", "sqrt(3)", "-sqrt(3)", "sqrt(3)/3", "-sqrt(3)/3"]

# Three paraphrases of the same instruction. Every template keeps "theta = {theta}," as
# its own opening clause (never reworded) so both this module's own working steps and the test
# file's independent oracle can always extract theta the same way regardless of which paraphrase was
# drawn -- and keeps "value of {fn}(theta)" verbatim so the asked-for function is always extractable
# too.
_VALUE_TEMPLATES = (
    "Given theta = {theta}, state the exact value of {fn}(theta), in simplified surd form.",
    "For theta = {theta}, find the exact value of {fn}(theta), in simplified surd form.",
    "Theta = {theta}, a standard angle. State the exact value of {fn}(theta), in simplified surd form.",
)


# The easy tier's second phrasing family, drawn from IN ADDITION to `_VALUE_TEMPLATES` (same "theta
# = {theta}," clause and "value of {fn}(theta)" text, so every oracle still extracts both the same
# way).
_EASY_EXTRA_TEMPLATES = (
    "When theta = {theta}, what is the exact value of {fn}(theta)? Give it in simplified surd form.",
    "Suppose theta = {theta}, a standard angle. What is the exact value of {fn}(theta), in simplified surd form?",
    "Recall the standard-angle values: theta = {theta}, so write down the exact value of {fn}(theta) in simplified surd form.",
)


def _exact_value_question(rng, k_lookup: int, fn: str, theta_display: str, templates=_VALUE_TEMPLATES) -> dict:
    """``k_lookup`` is the value actually used to index ``_TABLE`` (always
    in ``_KVALS``); ``theta_display`` is what the question STATES (may be a
    coterminal-shifted, degrees, or compound rendering of the same angle) --
    kept as two separate parameters so a hard/exam draw can display a
    different-looking theta while the exact value is always looked up from
    the same table entry."""
    cos_str, sin_str, tan_str = _TABLE[k_lookup]
    tan_display = tan_str if tan_str is not None else "undefined"
    if fn == "cos":
        answer_text = cos_str
        other = sin_str
    elif fn == "sin":
        answer_text = sin_str
        other = cos_str
    else:
        answer_text = tan_display
        other = sin_str

    text = rng.choice(templates).format(theta=theta_display, fn=fn)

    if fn == "tan":
        # Only values tan genuinely takes at a standard angle (never +-1/2,
        # which are sin/cos values), plus "undefined".
        tan_pool = list(_TAN_DISTRACTOR_POOL)
        rng.shuffle(tan_pool)
        if answer_text == "undefined":
            candidates = tan_pool
        else:
            candidates = [_flip_sign(answer_text), "undefined"] + tan_pool
    else:
        # sin/cos are defined everywhere: "undefined" is never offered.
        primary = [other, _flip_sign(answer_text), _flip_sign(other)]
        if tan_display != "undefined":
            primary.append(tan_display)
        candidates = primary + _COS_SIN_DISTRACTOR_POOL

    distractors = pick_distractors(answer_text, candidates, n=3)
    check(len(distractors) == 3, "p19g_exact_trig_value: fewer than 3 distractors survived")

    # The hint names an option, so it is only shown when that option is
    # actually on screen for this draw.
    if fn == "tan" and "undefined" in [answer_text] + distractors:
        text += ' Select "undefined" if it does not exist.'

    working = [
        f"theta = {theta_display}",
        f"cos(theta) = {cos_str}, sin(theta) = {sin_str}, tan(theta) = {tan_display}",
        f"{fn}(theta) = {answer_text}",
    ]
    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=answer_text,
        answer_text=answer_text,
        answer_latex=latex_answer(answer_text),
        working=working,
        distractors=distractors,
    )


@generator(
    "p19g_exact_trig_value",
    level=19,
    presets={
        "easy": dict(variant="easy"),
        "medium": dict(variant="medium"),
        "hard": dict(variant="hard"),
        "exam": dict(variant="exam"),
    },
)
def p19g_exact_trig_value(rng, variant="easy"):
    """Given theta as a stated integer multiple of pi/6 or pi/4, and which
    of cos(theta), sin(theta) or tan(theta) is asked for, state its exact
    value in simplified surd form, as an mcq option (never a free-typed
    expression -- see the module docstring's grader-kind decision).

    Scope: "Given theta as a
    stated integer multiple of pi/6 or pi/4, and which of cos(theta),
    sin(theta) or tan(theta) is asked for, state its exact value in
    simplified surd form." States a SINGLE named value -- never the full
    coordinate pair (that is the sibling skill
    ``p19g_unit_circle_coordinates``, below) and never derived from a
    supplied plotted terminal point (that is the shipping
    ``unit_circle_tan_evaluate``, ACMMM034)."""
    if variant == "easy":
        # Standard first-quadrant-anchored multiple of pi/6 or pi/4, cos or sin only -- pure recall
        # of one memorised exact value, no sign or tan-ratio reasoning. The five first-quadrant
        # standard angles (0, pi/6, pi/4, pi/3, pi/2) and two phrasing families: 10 (angle,
        # function) pairs x 6 templates, 5 distinct key texts.
        k = rng.choice([0, 2, 3, 4, 6])
        fn = rng.choice(["cos", "sin"])
        return _exact_value_question(rng, k, fn, _theta_str(k), templates=_VALUE_TEMPLATES + _EASY_EXTRA_TEMPLATES)

    if variant == "medium":
        # Any quadrant (sign genuinely varies now); tan may be asked but
        # only where it is genuinely a ratio decision -- never at a
        # quadrantal angle (0, pi/2, pi, 3*pi/2), where tan is either
        # undefined (held back for hard/exam) or a boring 0 with no sign
        # reasoning at all; redraw cos/sin there instead. Quadrantal k's are
        # drawn at half the weight of a genuinely-in-a-quadrant k -- they
        # are still real, reachable draws (the axis-angle 0/1/-1 values are
        # part of this tier's contract), just not over-represented, so the
        # answer-space concentration stays comfortably under the 15%
        # ceiling rather than riding its edge on sampling noise.
        k = rng.choices(_KVALS, weights=[1 if _quadrant_of(kk) == 0 else 2 for kk in _KVALS])[0]
        fn = rng.choice(["cos", "sin", "tan"])
        if fn == "tan" and _quadrant_of(k) == 0:
            fn = rng.choice(["cos", "sin"])
        return _exact_value_question(rng, k, fn, _theta_str(k))

    if variant == "hard":
        # Coterminal reduction required (theta stated outside [0, 2*pi), a
        # full turn either way) AND tan's undefined case is a real possible
        # key (any function may be asked, including tan at k = 6 or 18).
        k = rng.choice(_KVALS)
        shift = rng.choice([-24, 24])
        theta_display = _theta_str(k + shift)
        fn = rng.choice(["cos", "sin", "tan"])
        return _exact_value_question(rng, k, fn, theta_display)

    if variant == "exam":
        # Genuine kind-escalation: theta stated in degrees (needs
        # conversion) or as a compound sum/difference of two standard
        # angles (needs simplification) before the standard-angle lookup
        # applies.
        k = rng.choice(_KVALS)
        fn = rng.choice(["cos", "sin", "tan"])
        representation = rng.choice(["degrees", "compound"])
        if representation == "compound" and _compound_str(k) is None:
            representation = "degrees"
        theta_display = _degrees_str(k) if representation == "degrees" else _compound_str(k)
        return _exact_value_question(rng, k, fn, theta_display)

    raise ValueError(f"unknown variant {variant!r}")


# ===========================================================================
# 2. p19g_unit_circle_coordinates (answer: expression, Tuple-shaped)
# ===========================================================================

#: Every table pair, in fixed `_KVALS` order, as `(k, "(cos, sin)")` --
#: reused as a "wrong standard angle" distractor pool (skill 2 only), a
#: guaranteed-diverse fallback beyond the swap/sign-flip candidates so the
#: quadrantal draws (where a coordinate is 0) still surface >= 3 distinct
#: distractors.
_ALL_PAIRS: tuple[tuple[int, str], ...] = tuple((k, f"({_TABLE[k][0]}, {_TABLE[k][1]})") for k in _KVALS)


# Three paraphrases of the same instruction, same "theta = {theta}," opening-clause
# discipline as `_VALUE_TEMPLATES` above.
_COORDINATE_TEMPLATES = (
    "Given theta = {theta}, a stated standard angle, state the exact coordinates "
    "(cos theta, sin theta) of the corresponding point on the unit circle, as an "
    "ordered pair in simplified surd form.",
    "For theta = {theta}, a stated standard angle, find the exact coordinates "
    "(cos theta, sin theta) of the point on the unit circle, as an ordered pair "
    "in simplified surd form.",
    "Theta = {theta}, a standard angle. State the exact coordinates (cos theta, "
    "sin theta) of the corresponding point on the unit circle, as an ordered "
    "pair in simplified surd form.",
)


# The coordinate skill's second phrasing family, drawn from IN ADDITION to `_COORDINATE_TEMPLATES`
# on the easy and medium tiers (same "theta = {theta}," opening clause, so every oracle extracts
# theta the same way, and no "value of").
_COORDINATE_EXTRA_TEMPLATES = (
    "When theta = {theta}, what are the exact coordinates (cos theta, sin theta) "
    "of the point on the unit circle? Give them as an ordered pair in simplified surd form.",
    "Suppose theta = {theta}, a standard angle. What ordered pair (cos theta, sin theta) "
    "gives the corresponding point on the unit circle, in simplified surd form?",
    "Recall the standard-angle values: theta = {theta}, so write down the exact "
    "coordinates (cos theta, sin theta) of its point on the unit circle, in simplified surd form.",
)


def _consensus_scores(options: list[str]) -> list[int]:
    """Per option, how many of its coordinate strings also appear in the other
    options -- the signal a shared-coordinate 'consensus' attacker reads."""
    comps = [set(o.strip("()").split(", ")) for o in options]
    return [
        sum(sum(1 for c in other if c in own) for j, other in enumerate(comps) if j != i) for i, own in enumerate(comps)
    ]


def _coordinate_distractor_families(k_lookup: int) -> dict[str, list[str]]:
    """Independent error families for the pair at ``k_lookup``, each
    a list of exact-surd Tuple strings. They are separate misconceptions, not
    edits of one another, so the three picked never form a shared-coordinate
    hub around the key."""
    cos_str, sin_str, tan_str = _TABLE[k_lookup]
    fam: dict[str, list[str]] = {
        # swaps cos and sin
        "swap": [f"({sin_str}, {cos_str})"],
        # a quarter-turn confusion: -sin/cos or sin/-cos
        "rotate": [f"({_flip_sign(sin_str)}, {cos_str})", f"({sin_str}, {_flip_sign(cos_str)})"],
        # the sign pattern of another quadrant
        "quadrant": [
            f"({_flip_sign(cos_str)}, {sin_str})",
            f"({cos_str}, {_flip_sign(sin_str)})",
            f"({_flip_sign(cos_str)}, {_flip_sign(sin_str)})",
        ],
        # a different table angle altogether (wrong reference angle)
        "angle": [pair for k, pair in _ALL_PAIRS if k != (k_lookup % 24)],
    }
    key = f"({cos_str}, {sin_str})"
    return {name: [c for c in dict.fromkeys(cands) if c != key] for name, cands in fam.items()}


def _shared_pairs(triple: list[str]) -> int:
    """How many of the three distractor pairs share a coordinate string."""
    comps = [set(o.strip("()").split(", ")) for o in triple]
    return sum(1 for i in range(3) for j in range(i + 1, 3) if comps[i] & comps[j])


@functools.lru_cache(maxsize=None)
def _distractor_triples(answer_text: str) -> tuple[tuple, tuple, tuple]:
    """All triples of DISTINCT table points (never the key; no two the same
    coordinate set, so no swap or mirror twins) with at most one distractor
    pair sharing a coordinate, in three classes by the key's shared-coordinate
    score against the distractors' scores: strictly between lowest and highest
    (safe from a max and a min attacker, ties to the first option), strictly
    below the highest, strictly above the lowest."""
    pool = [pair for _k, pair in _ALL_PAIRS if pair != answer_text]
    between, below, above = [], [], []
    for triple in itertools.combinations(pool, 3):
        if len({frozenset(o.strip("()").split(", ")) for o in triple}) < 3:
            continue
        if _shared_pairs(list(triple)) > 1:
            continue
        scores = _consensus_scores([answer_text, *triple])
        lo, hi = min(scores[1:]), max(scores[1:])
        if lo < scores[0] < hi:
            between.append(triple)
        if scores[0] < hi:
            below.append(triple)
        if scores[0] > lo:
            above.append(triple)
    return tuple(between), tuple(below), tuple(above)


def _pick_independent_distractors(rng, k_lookup: int, answer_text: str) -> list[str]:
    """Three distractors, every one a real table point on the unit circle (no tan routes, so none is off the circle or outside [-1, 1]), pairwise
    distinct and with at most one pair sharing a coordinate. Where the pool
    allows, the key sits strictly between the lowest and highest
    shared-coordinate score, so neither a max nor a min consensus attacker (ties
    to the first option) finds it; the quadrantal keys have no such triple, so
    they alternate at random between 'below the top' and 'above the bottom'."""
    between, below, above = _distractor_triples(answer_text)
    if between:
        pool = between
    else:
        pool = below if rng.random() < 0.5 else above
    check(bool(pool), "p19g_unit_circle_coordinates: no independent distractor triple")
    triple = list(rng.choice(pool))
    rng.shuffle(triple)
    return triple


def _coordinate_pair_question(rng, k_lookup: int, theta_display: str, extra_templates: bool = False) -> dict:
    cos_str, sin_str, _tan_str = _TABLE[k_lookup]
    answer_text = f"({cos_str}, {sin_str})"
    templates = _COORDINATE_TEMPLATES + (_COORDINATE_EXTRA_TEMPLATES if extra_templates else ())
    text = rng.choice(templates).format(theta=theta_display)

    distractors = _pick_independent_distractors(rng, k_lookup, answer_text)
    check(
        len(distractors) == 3 and answer_text not in distractors,
        "p19g_unit_circle_coordinates: fewer than 3 distractors survived",
    )

    working = [
        f"theta = {theta_display}",
        f"cos(theta) = {cos_str}, sin(theta) = {sin_str}",
        f"(cos theta, sin theta) = {answer_text}",
    ]
    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=answer_text,
        answer_text=answer_text,
        answer_latex=latex_answer(answer_text),
        working=working,
        distractors=distractors,
    )


@generator(
    "p19g_unit_circle_coordinates",
    level=19,
    presets={
        "easy": dict(variant="easy"),
        "medium": dict(variant="medium"),
        "hard": dict(variant="hard"),
        "exam": dict(variant="exam"),
    },
)
def p19g_unit_circle_coordinates(rng, variant="easy"):
    """Given theta as a stated standard angle (an integer multiple of pi/6
    or pi/4), state the coordinates (cos theta, sin theta) of the
    corresponding point on the unit circle, as one typed ordered pair.

    Scope: "Given theta as a
    stated standard angle (an integer multiple of pi/6 or pi/4), state the
    coordinates (cos theta, sin theta) of the corresponding point on the
    unit circle." States the FULL coordinate PAIR together -- never a single
    named value (that is the sibling skill ``p19g_exact_trig_value``,
    above), and theta is never given as a plotted terminal point (that is
    the shipping ``unit_circle_tan_evaluate``, ACMMM034, whose kernel
    DERIVES tan from a supplied point -- the reverse direction)."""
    if variant == "easy":
        # Quadrant I only (theta in [0, pi/2]), all-positive coordinates --
        # pure lookup, no sign reasoning.
        k = rng.choice([0, 2, 3, 4, 6])
        return _coordinate_pair_question(rng, k, _theta_str(k), extra_templates=True)

    if variant == "medium":
        # Any quadrant, or a quadrantal axis angle (0/1/-1 component) --
        # sign reasoning is now genuinely required.
        k = rng.choice(_KVALS)
        return _coordinate_pair_question(rng, k, _theta_str(k), extra_templates=True)

    if variant == "hard":
        # Coterminal reduction required: theta stated outside [0, 2*pi), a
        # full turn either way, before the standard-angle lookup applies.
        k = rng.choice(_KVALS)
        shift = rng.choice([-24, 24])
        return _coordinate_pair_question(rng, k, _theta_str(k + shift))

    if variant == "exam":
        # Genuine kind-escalation: theta stated in degrees, or as a
        # compound sum/difference of two standard angles that must be
        # simplified before the lookup applies.
        k = rng.choice(_KVALS)
        representation = rng.choice(["degrees", "compound"])
        if representation == "compound" and _compound_str(k) is None:
            representation = "degrees"
        theta_display = _degrees_str(k) if representation == "degrees" else _compound_str(k)
        return _coordinate_pair_question(rng, k, theta_display)

    raise ValueError(f"unknown variant {variant!r}")
