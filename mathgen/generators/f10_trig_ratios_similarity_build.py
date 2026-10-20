"""Constant right-triangle ratios and the unknown side they give.

- ``f10_trig_ratios_similarity`` (``f10-trig-ratios-similarity``, AC9M9SP01,
  domain ``trigonometry``, answer: ``number``) -- given one acute angle and
  one known side of a right-angled triangle, use the constant sine, cosine or
  tangent ratio for that angle to find a different, unknown side. Never an
  unknown angle (that is the prerequisite ``right-triangle-angle``'s own
  graded operation).

Three tiers, each a different reasoning shape:

- easy: no calculator. A reference right triangle gives all three sides
  (a Pythagorean triple); a larger triangle with the SAME acute angle gives
  one side. The learner finds another side of the larger triangle by
  keeping the ratio of the two sides constant (similarity). Integer answer.
- medium: the angle and one side are stated; the three ratios for the angle
  are printed, each rounded to 4 decimal places, and the answer is wanted to
  1 decimal place. The learner picks the ratio from the side NAMES
  (opposite, adjacent, hypotenuse) and decides whether to multiply or divide.
- hard: the triangle is named by vertex letters (right angle at one of
  them), the stated angle sits at a named vertex, the sides are named by
  their end letters, and the printed table holds the ratios for BOTH acute
  angles. The learner works out which side is opposite, adjacent or the
  hypotenuse from the letters, reads the right row and answers to 2 decimal
  places.

Rounding is explicit in the stem, and the printed 4-decimal ratios and the
calculator's exact ratios must round to the SAME final answer, with a safe
margin from a rounding boundary, so a learner who uses either is graded fairly.

The generator re-derives its answer from the RENDERED text (a regex parse,
then SymPy's own sine/cosine/tangent at 30 digits), independently of how
construction built the question.
"""

from __future__ import annotations

import math
import re
from functools import lru_cache
from fractions import Fraction

from sympy import N, cos, rad, sin, tan

from ..core import (
    check,
    check_pairwise_distinct,
    generator,
    latex_answer,
    numeric_answer,
    pick_distractors,
    round_half_up,
    stated_precision_string,
)

_ROLES = ("opposite", "adjacent", "hypotenuse")

# Primitive Pythagorean triples, written (leg, leg, hypotenuse).
_TRIPLES = ((3, 4, 5), (5, 12, 13), (8, 15, 17), (7, 24, 25), (20, 21, 29), (9, 40, 41))

# -- misconception ids (must match the skill YAML) ---------------------------------
M_ADDITIVE = "adds-the-difference"
M_THIRD_SIDE = "scales-the-wrong-side"
M_NO_SCALE = "forgets-to-scale"
M_ADD_FACTOR = "adds-the-scale-factor"
M_SWAP = "swaps-opposite-and-adjacent"
M_INVERT = "multiplies-instead-of-divides"
M_AS_HYP = "treats-the-known-side-as-the-hypotenuse"
M_DIVIDE_UNKNOWN = "divides-by-the-unknown-sides-ratio"

_NOTE_EASY = {
    M_ADDITIVE: "Added the same amount to the reference side that the known side grew by, instead of scaling.",
    M_THIRD_SIDE: "Scaled the third side of the reference triangle instead of the side asked for.",
    M_NO_SCALE: "Reported the reference triangle's side without scaling it up.",
    M_ADD_FACTOR: "Added the scale factor to the reference side instead of multiplying by it.",
}


# -- exact trig values, independent of math.sin ------------------------------------


@lru_cache(maxsize=None)
def _ratio_exact(name: str, degrees: int) -> Fraction:
    """The exact (30-digit) ratio as a Fraction, via SymPy -- a second route from the
    math.* used in construction."""
    fn = {"sin": sin, "cos": cos, "tan": tan}[name]
    return Fraction(str(N(fn(rad(degrees)), 30)))


def _ratio_table(degrees: int) -> dict:
    return {name: round_half_up(_ratio_exact(name, degrees), 4) for name in ("sin", "cos", "tan")}


def _coefficients(angle: int, table: dict | None = None) -> dict:
    """Side length as a multiple of the hypotenuse, for each role, at ``angle``.
    ``table`` swaps the exact values for printed ones."""
    if table is None:
        s, c = _ratio_exact("sin", angle), _ratio_exact("cos", angle)
    else:
        s, c = table["sin"], table["cos"]
    return {"hypotenuse": Fraction(1), "opposite": s, "adjacent": c}


def _solve(known_role: str, unknown_role: str, known: Fraction, coeffs: dict) -> Fraction:
    return known * coeffs[unknown_role] / coeffs[known_role]


# -- easy: similar triangles, exact -----------------------------------------------

_EASY_CLOSERS = (
    "Give the length of the {role} of the second triangle, in cm.",
    "Find the length of the {role} of the second triangle, in cm.",
    "How long is the {role} of the second triangle, in cm?",
)

_PHRASE_ROLE = {"opposite side": "opposite", "adjacent side": "adjacent", "hypotenuse": "hypotenuse"}
_ROLE_PHRASE = {v: k for k, v in _PHRASE_ROLE.items()}


def _easy(rng):
    for _ in range(100):
        a, b, h = rng.choice(_TRIPLES)
        if rng.random() < 0.5:
            a, b = b, a
        ref = {"opposite": a, "adjacent": b, "hypotenuse": h}
        k = rng.randint(2, 8)
        known_role, unknown_role = rng.sample(_ROLES, 2)
        third = next(r for r in _ROLES if r not in (known_role, unknown_role))
        answer = k * ref[unknown_role]
        cands = {
            M_ADDITIVE: ref[unknown_role] + (k * ref[known_role] - ref[known_role]),
            M_THIRD_SIDE: k * ref[third],
            M_NO_SCALE: ref[unknown_role],
            M_ADD_FACTOR: ref[unknown_role] + k,
        }
        survivors = {m: v for m, v in cands.items() if v != answer}
        distinct = {}
        for m, v in survivors.items():
            if v not in distinct.values():
                distinct[m] = v
        if len(distinct) >= 3:
            break
    else:  # pragma: no cover
        raise RuntimeError("f10_trig_ratios_similarity easy: no draw with three distinct distractors")

    text = (
        f"In a right-angled triangle with an acute angle marked x, the side opposite x is {ref['opposite']} cm, "
        f"its adjacent side is {ref['adjacent']} cm and its hypotenuse is {ref['hypotenuse']} cm. "
        f"A second right-angled triangle has the same angle x. In the second triangle the "
        f"{_ROLE_PHRASE[known_role]} is {k * ref[known_role]} cm. "
        + rng.choice(_EASY_CLOSERS).format(role=_ROLE_PHRASE[unknown_role])
    )
    answer_text = str(answer)
    _check_easy(text, answer_text)

    distractors = pick_distractors(answer_text, [str(v) for v in distinct.values()])
    notes = {str(v): m for m, v in distinct.items() if str(v) in distractors}
    working = [
        f"Both triangles have the same angle x, so they are similar and the ratio of "
        f"the {_ROLE_PHRASE[unknown_role]} to the {_ROLE_PHRASE[known_role]} is the same in both.",
        f"{ref[unknown_role]} : {ref[known_role]} = ? : {k * ref[known_role]}, so the scale factor is {k}.",
        f"{k} x {ref[unknown_role]} = {answer} cm.",
    ]
    return text, answer_text, answer, distractors, notes, working


def _check_easy(text: str, answer_text: str) -> None:
    m = re.search(
        r"side opposite x is (\d+) cm, its adjacent side is (\d+) cm and its hypotenuse is (\d+) cm\. .*? "
        r"In the second triangle the (opposite side|adjacent side|hypotenuse) is (\d+) cm\. "
        r"(?:(?:Give|Find) the length of the (opposite side|adjacent side|hypotenuse) of the second triangle, in cm\.|"
        r"How long is the (opposite side|adjacent side|hypotenuse) of the second triangle, in cm\?)",
        text,
    )
    check(m is not None, "f10_trig_ratios_similarity: could not parse the easy question text")
    ref = {"opposite": int(m.group(1)), "adjacent": int(m.group(2)), "hypotenuse": int(m.group(3))}
    known_role, big = _PHRASE_ROLE[m.group(4)], int(m.group(5))
    unknown_role = _PHRASE_ROLE[m.group(6) or m.group(7)]
    check(
        ref["opposite"] ** 2 + ref["adjacent"] ** 2 == ref["hypotenuse"] ** 2,
        "f10_trig_ratios_similarity: reference triangle is not right-angled",
    )
    value = Fraction(big * ref[unknown_role], ref[known_role])
    check(
        value.denominator == 1 and int(value) == int(answer_text),
        "f10_trig_ratios_similarity: similarity re-derivation disagrees with answer_text",
    )


# -- medium / hard: trig ratios --------------------------------------------------

_MEDIUM_ANGLES = tuple(a for a in range(14, 77) if a != 45 and abs(a - 45) > 3)
_HARD_ANGLES = _MEDIUM_ANGLES
_LETTER_SETS = (("A", "B", "C"), ("P", "Q", "R"), ("D", "E", "F"), ("X", "Y", "Z"), ("K", "L", "M"))

_MEDIUM_CLOSERS = (
    "Find the length of the {unknown}, correct to {dp_word}.",
    "Calculate the length of the {unknown}, correct to {dp_word}.",
    "What is the length of the {unknown}, correct to {dp_word}?",
)
_DP_WORD = {1: "1 decimal place", 2: "2 decimal places"}


def _distractor_values(angle, known_role, unknown_role, known, dp):
    """Wrong answers from specific errors, each rounded to ``dp`` places from EXACT ratios."""
    exact = _coefficients(angle)
    swapped = dict(exact, opposite=exact["adjacent"], adjacent=exact["opposite"])
    cands = {
        M_SWAP: _solve(known_role, unknown_role, known, swapped),
        M_INVERT: known * exact[known_role] / exact[unknown_role],
        M_AS_HYP: known * exact[unknown_role],
        M_DIVIDE_UNKNOWN: known / exact[unknown_role],
    }
    # Two misconceptions can coincide for a given pair of roles (swap and invert are the same number for
    # opposite/adjacent); keep the first of each value so every listed note names a distinct option.
    out, seen = {}, []
    for m, v in cands.items():
        r = round_half_up(v, dp)
        if r not in seen:
            seen.append(r)
            out[m] = r
    return out


def _fmt_table(angle: int, table: dict) -> str:
    return ", ".join(f"{n} {angle} degrees = {stated_precision_string(table[n], 4)}" for n in ("sin", "cos", "tan"))


def _draw_angle_and_side(rng, hard: bool):
    angle = rng.choice(_HARD_ANGLES if hard else _MEDIUM_ANGLES)
    if hard:
        known = Fraction(rng.randint(80, 600), 10)  # one decimal place
    else:
        known = Fraction(rng.randint(5, 45))
    known_role, unknown_role = rng.sample(_ROLES, 2)
    return angle, known, known_role, unknown_role


def _answer_is_safe(angle, known_role, unknown_role, known, dp) -> Fraction | None:
    """The rounded answer, or None when the exact and the printed-ratio routes disagree or either sits near a
    rounding boundary (so a learner using either is graded the same)."""
    exact_v = _solve(known_role, unknown_role, known, _coefficients(angle))
    printed_v = _solve(known_role, unknown_role, known, _coefficients(angle, _ratio_table(angle)))
    rounded = round_half_up(exact_v, dp)
    if round_half_up(printed_v, dp) != rounded:
        return None
    unit = Fraction(1, 10**dp)
    for v in (exact_v, printed_v):
        # distance to the nearest rounding boundary (a half-unit point)
        frac = (v / unit) - math.floor(v / unit)
        if abs(frac - Fraction(1, 2)) < Fraction(1, 5):
            return None
    return rounded


def _medium(rng):
    for _ in range(200):
        angle, known, known_role, unknown_role = _draw_angle_and_side(rng, hard=False)
        dp = 1
        rounded = _answer_is_safe(angle, known_role, unknown_role, known, dp)
        if rounded is None:
            continue
        dist = _distractor_values(angle, known_role, unknown_role, known, dp)
        if _enough(rounded, dist, dp):
            break
    else:  # pragma: no cover
        raise RuntimeError("f10_trig_ratios_similarity medium: no safe draw")

    table = _ratio_table(angle)
    text = (
        f"In a right-angled triangle, one acute angle is {angle} degrees. "
        f"The {_PHRASE[known_role]} is {known} cm. "
        f"For {angle} degrees, the ratios from a four-figure table are: {_fmt_table(angle, table)}. "
        + rng.choice(_MEDIUM_CLOSERS).format(unknown=_PHRASE[unknown_role], dp_word=_DP_WORD[dp])
    )
    answer_text = stated_precision_string(rounded, dp)
    _check_ratio_text(text, answer_text, dp)
    return _finish(rounded, dist, dp, angle, known_role, unknown_role, known, text, lead=[])


def _enough(rounded, dist, dp) -> bool:
    unit = Fraction(1, 10**dp)
    kept = []
    for v in dist.values():
        if abs(v - rounded) <= unit:
            continue
        if any(abs(v - k) <= unit for k in kept):
            continue
        kept.append(v)
    return len(kept) >= 3


def _finish(rounded, dist, dp, angle, known_role, unknown_role, known, text, *, lead):
    answer_text = stated_precision_string(rounded, dp)
    unit = Fraction(1, 10**dp)
    kept = {m: v for m, v in dist.items() if abs(v - rounded) > unit}
    check_pairwise_distinct(rounded, *kept.values(), msg="f10_trig_ratios_similarity: distractor collision")
    strings = {m: stated_precision_string(v, dp) for m, v in kept.items()}
    distractors = pick_distractors(answer_text, list(strings.values()), tolerance=unit)
    notes = {s: m for m, s in strings.items() if s in distractors}
    ratio = _RATIO_NAME[frozenset((known_role, unknown_role))]
    defn = {"sin": "opposite / hypotenuse", "cos": "adjacent / hypotenuse", "tan": "opposite / adjacent"}[ratio]
    denominator_role = {"sin": "hypotenuse", "cos": "hypotenuse", "tan": "adjacent"}[ratio]
    printed = stated_precision_string(_ratio_table(angle)[ratio], 4)
    known_text = str(known) if known.denominator == 1 else stated_precision_string(known, 1)
    operation = f"{known_text} / {printed}" if unknown_role == denominator_role else f"{known_text} x {printed}"
    working = [
        *lead,
        f"The sides involved are the {known_role} and the {unknown_role}, so use {ratio} {angle} = {defn}.",
        f"{ratio} {angle} degrees = {printed}.",
        f"Unknown side = {operation} = {answer_text} cm (to {dp} decimal place{'s' if dp > 1 else ''}).",
    ]
    return text, answer_text, numeric_answer(rounded), distractors, notes, working


_RATIO_NAME = {
    frozenset(("opposite", "hypotenuse")): "sin",
    frozenset(("adjacent", "hypotenuse")): "cos",
    frozenset(("opposite", "adjacent")): "tan",
}
_PHRASE = {"opposite": "side opposite the angle", "adjacent": "leg adjacent to the angle", "hypotenuse": "hypotenuse"}
_PHRASE_BACK = {v: k for k, v in _PHRASE.items()}
_PHRASE_ALT = "|".join(re.escape(v) for v in _PHRASE.values())


# Rendered-text oracle (medium) --------------------------------------------------

_MED_RE = re.compile(
    r"one acute angle is (\d+) degrees\. The (" + _PHRASE_ALT + r") is ([\d.]+) cm\. "
    r"For \d+ degrees, the ratios from a four-figure table are: "
    r"sin \d+ degrees = ([\d.]+), cos \d+ degrees = ([\d.]+), tan \d+ degrees = ([\d.]+)\. "
    r".*? length of the (" + _PHRASE_ALT + r"), correct to (\d) decimal place"
)


def _check_ratio_text(text: str, answer_text: str, dp: int) -> None:
    m = _MED_RE.search(text)
    check(m is not None, "f10_trig_ratios_similarity: could not parse the medium question text")
    check(int(m.group(8)) == dp, "f10_trig_ratios_similarity: stated rounding disagrees with the answer's")
    _recheck(int(m.group(1)), _PHRASE_BACK[m.group(2)], _PHRASE_BACK[m.group(7)], Fraction(m.group(3)), answer_text, dp)


def _recheck(angle, known_role, unknown_role, known, answer_text, dp) -> None:
    """Re-derive via SymPy at 30 digits, cross-checked against plain floats."""
    exact = _solve(known_role, unknown_role, known, _coefficients(angle))
    check(
        stated_precision_string(round_half_up(exact, dp), dp) == answer_text,
        "f10_trig_ratios_similarity: trig re-derivation disagrees with answer_text",
    )
    r = math.radians(angle)
    c = {"hypotenuse": 1.0, "opposite": math.sin(r), "adjacent": math.cos(r)}
    as_float = float(known) * c[unknown_role] / c[known_role]
    check(abs(float(exact) - as_float) < 1e-9, "f10_trig_ratios_similarity: float and exact routes disagree")


# -- hard: named vertices ---------------------------------------------------------


def _hard(rng):
    for _ in range(300):
        angle, known, known_role, unknown_role = _draw_angle_and_side(rng, hard=True)
        dp = 2
        rounded = _answer_is_safe(angle, known_role, unknown_role, known, dp)
        if rounded is None:
            continue
        dist = _distractor_values(angle, known_role, unknown_role, known, dp)
        if _enough(rounded, dist, dp):
            break
    else:  # pragma: no cover
        raise RuntimeError("f10_trig_ratios_similarity hard: no safe draw")

    v1, v2, vr = rng.choice(_LETTER_SETS)  # vr is the right-angle vertex
    if rng.random() < 0.5:
        v1, v2 = v2, v1
    at = v1  # the stated angle sits at v1; the other acute vertex is v2
    names = {
        "hypotenuse": f"{v1}{v2}",
        "opposite": f"{v2}{vr}",  # does not touch the angle's vertex
        "adjacent": f"{v1}{vr}",
    }
    comp = 90 - angle
    t1, t2 = _ratio_table(angle), _ratio_table(comp)
    text = (
        f"Triangle {v1}{v2}{vr} has a right angle at {vr}, and the angle at {at} is {angle} degrees. "
        f"Side {names[known_role]} is {stated_precision_string(known, 1)} cm. "
        f"Ratios from a four-figure table: {_fmt_table(angle, t1)}; {_fmt_table(comp, t2)}. "
        f"Find the length of side {names[unknown_role]}, correct to {_DP_WORD[dp]}."
    )
    answer_text = stated_precision_string(rounded, dp)
    _check_hard_text(text, answer_text, dp)
    lead = [
        f"The right angle is at {vr}, so {names['hypotenuse']} is the hypotenuse. "
        f"Relative to the angle at {at}, {names['opposite']} is opposite and {names['adjacent']} is adjacent."
    ]
    return _finish(rounded, dist, dp, angle, known_role, unknown_role, known, text, lead=lead)


_HARD_RE = re.compile(
    r"Triangle (\w)(\w)(\w) has a right angle at (\w), and the angle at (\w) is (\d+) degrees\. "
    r"Side (\w\w) is ([\d.]+) cm\. "
    r"Ratios from a four-figure table: .*? "
    r"Find the length of side (\w\w), correct to (\d) decimal places\."
)


def _check_hard_text(text: str, answer_text: str, dp: int) -> None:
    m = _HARD_RE.search(text)
    check(m is not None, "f10_trig_ratios_similarity: could not parse the hard question text")
    letters, right, at = (m.group(1), m.group(2), m.group(3)), m.group(4), m.group(5)
    angle = int(m.group(6))
    known_name, known = m.group(7), Fraction(m.group(8))
    unknown_name = m.group(9)
    check(right in letters and at in letters and at != right, "f10_trig_ratios_similarity: bad vertex letters")
    other = next(x for x in letters if x not in (right, at))

    def role(name: str) -> str:
        pair = set(name)
        if pair == {at, other}:
            return "hypotenuse"
        if pair == {at, right}:
            return "adjacent"
        check(pair == {other, right}, "f10_trig_ratios_similarity: side name is not a side of the triangle")
        return "opposite"

    _recheck(angle, role(known_name), role(unknown_name), known, answer_text, dp)


@generator(
    "f10_trig_ratios_similarity",
    level=9,
    presets={
        "easy": dict(variant="similar"),
        "medium": dict(variant="roles"),
        "hard": dict(variant="vertices"),
    },
)
def f10_trig_ratios_similarity(rng, variant="similar"):
    if variant == "similar":
        text, answer_text, answer, distractors, notes, working = _easy(rng)
    elif variant == "roles":
        text, answer_text, answer, distractors, notes, working = _medium(rng)
    elif variant == "vertices":
        text, answer_text, answer, distractors, notes, working = _hard(rng)
    else:
        raise ValueError(f"unknown variant {variant!r}")
    return {
        "text": text,
        "answer": answer,
        "answer_text": answer_text,
        "answer_latex": latex_answer(answer_text),
        "distractors": distractors,
        "option_notes": notes,
        "working": working,
    }
