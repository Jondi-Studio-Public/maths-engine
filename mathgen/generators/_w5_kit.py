"""Small helpers shared by the `w5_*` builds (inequalities, circles, functions, algebra foundations).

Not a generator module (the leading underscore keeps the package loader from importing it). It holds the answer
dictionaries and tagged distractors for the kinds these builds use that `_w6a_kit` does not cover: solution sets
(`inequality`), lists of numbers (`list`, `ordered_list`) and option texts (`mcq`), each filtered through the real
grader so that no distractor is accepted as the answer.
"""

from __future__ import annotations

from fractions import Fraction

import sympy

from ..answers import Distractor
from ..core import phrase, rational_text
from ..grading.inequality import set_text
from ._r3i_kit import accepted_by_grader
from ._w6a_kit import TooFewDistractors, enough, num_dist, num_fields, part, rat, retry_draw  # noqa: F401


def rt(value) -> str:
    """An exact number as a learner types it: `3`, `-3/4`."""
    return rational_text(sympy.Rational(rat(value).numerator, rat(value).denominator))


def sym(value):
    """A Fraction (or int) as a SymPy Rational; SymPy values pass through."""
    if isinstance(value, Fraction):
        return sympy.Rational(value.numerator, value.denominator)
    return sympy.sympify(value)


# -- inequalities --------------------------------------------------------------------------------------------------


# A small interval algebra over SymPy Interval pieces. SymPy's own Union/Complement/intersect simplify by
# testing containment, which costs milliseconds per call; here the pieces are merged by comparing floats of their
# bounds (two bounds that are equal as numbers are equal as floats, and distinct ones here differ by far more than 1e-9).


def _parts(s) -> list:
    if s == sympy.S.EmptySet:
        return []
    if isinstance(s, sympy.Union):
        return list(s.args)
    return [s]


def _span(piece):
    """(low, high, low_open, high_open) of an Interval or a one-point FiniteSet."""
    if isinstance(piece, sympy.FiniteSet):
        (v,) = tuple(piece)
        return v, v, False, False
    return piece.start, piece.end, bool(piece.left_open), bool(piece.right_open)


def _f(v) -> float:
    return float(v)


def _build(spans):
    """The set made of `spans` (low, high, low_open, high_open), merged where they overlap or touch."""
    spans = sorted(spans, key=lambda t: (_f(t[0]), t[2]))
    merged = []
    for lo, hi, lo_open, hi_open in spans:
        if merged:
            clo, chi, clo_open, chi_open = merged[-1]
            gap = _f(lo) - _f(chi)
            touching = abs(gap) < 1e-12 and not (chi_open and lo_open)
            if gap < -1e-12 or touching:
                if _f(hi) > _f(chi) + 1e-12 or (abs(_f(hi) - _f(chi)) < 1e-12 and not hi_open):
                    merged[-1] = (
                        clo,
                        hi,
                        clo_open,
                        hi_open if abs(_f(hi) - _f(chi)) > 1e-12 else (chi_open and hi_open),
                    )
                continue
        merged.append((lo, hi, lo_open, hi_open))
    pieces = []
    for lo, hi, lo_open, hi_open in merged:
        pieces.append(sympy.Interval(lo, hi, lo_open or lo == -sympy.oo, hi_open or hi == sympy.oo))
    if not pieces:
        return sympy.S.EmptySet
    if len(pieces) == 1:
        return pieces[0]
    return sympy.Union(*pieces, evaluate=False)


def sunion(*sets):
    return _build([_span(p) for s in sets for p in _parts(s)])


def scomp(s):
    """The complement of `s` in the real line."""
    spans = sorted((_span(p) for p in _parts(s)), key=lambda t: _f(t[0]))
    out, cursor, cursor_open = [], -sympy.oo, True
    for lo, hi, lo_open, hi_open in spans:
        if _f(lo) > _f(cursor) + 1e-12 or (abs(_f(lo) - _f(cursor)) < 1e-12 and not (cursor_open and lo_open)):
            out.append((cursor, lo, cursor_open if cursor != -sympy.oo else True, not lo_open))
        cursor, cursor_open = hi, not hi_open
    if cursor != sympy.oo:
        out.append((cursor, sympy.oo, cursor_open, True))
    return _build([t for t in out if _f(t[0]) < _f(t[1]) or (_f(t[0]) == _f(t[1]) and not t[2] and not t[3])])


def sinter(a, b):
    out = []
    for p in _parts(a):
        for q in _parts(b):
            lo1, hi1, lo1o, hi1o = _span(p)
            lo2, hi2, lo2o, hi2o = _span(q)
            if _f(lo1) > _f(lo2) + 1e-12 or (abs(_f(lo1) - _f(lo2)) < 1e-12 and lo1o):
                lo, lo_open = lo1, lo1o
            else:
                lo, lo_open = lo2, lo2o if abs(_f(lo1) - _f(lo2)) > 1e-12 else (lo1o or lo2o)
            if _f(hi1) < _f(hi2) - 1e-12:
                hi, hi_open = hi1, hi1o
            elif _f(hi1) > _f(hi2) + 1e-12:
                hi, hi_open = hi2, hi2o
            else:
                hi, hi_open = hi1, hi1o or hi2o
            if _f(lo) < _f(hi) - 1e-12 or (abs(_f(lo) - _f(hi)) < 1e-12 and not lo_open and not hi_open):
                out.append((lo, hi, lo_open, hi_open))
    return _build(out)


def has_isolated_point(solution) -> bool:
    """Whether the set has a single-point piece, which a reply cannot be typed as an inequality (x = 3 is not read)."""
    return any(isinstance(piece, sympy.FiniteSet) for piece in _parts(solution))


def ineq_fields(solution, var: str = "x") -> dict:
    if has_isolated_point(solution):
        raise TooFewDistractors("the solution set has an isolated point")
    text = set_text(solution, var)
    return dict(answer_kind="inequality", answer=text, answer_text=text, answer_latex="plain:" + text)


def ineq_cands(fields: dict, pairs, var: str = "x", n: int = 3) -> list[Distractor]:
    """Tagged distractors from `(solution set, tag)` pairs in priority order: none repeats the answer, none repeats
    another, and none is accepted by the real inequality grader."""
    out: list[Distractor] = []
    seen = {fields["answer_text"]}
    for solution, tag in pairs:
        if solution is None or has_isolated_point(solution):
            continue
        text = set_text(solution, var)
        if text in seen:  # set_text is canonical: a different text is a different set
            continue
        seen.add(text)
        out.append(Distractor(text, tag, text))
        if len(out) == n:
            break
    return out


def in_set(solution, point) -> bool:
    """Whether the exact rational `point` lies in `solution` (a SymPy real set), by comparing floats: a rational with a
    small denominator is never within 1e-9 of an irrational bound such as 2 - sqrt(5)."""
    x = float(point)
    if solution == sympy.S.Reals:
        return True
    if solution == sympy.S.EmptySet:
        return False
    pieces = solution.args if isinstance(solution, sympy.Union) else (solution,)
    for piece in pieces:
        if isinstance(piece, sympy.FiniteSet):
            if any(abs(float(v) - x) < 1e-12 for v in piece):
                return True
            continue
        lo, hi = float(piece.start), float(piece.end)
        above = x > lo or (x == lo and not piece.left_open)
        below = x < hi or (x == hi and not piece.right_open)
        if above and below:
            return True
    return False


# -- lists of numbers ----------------------------------------------------------------------------------------------


def _num(v):
    v = rat(v)
    return int(v) if v.denominator == 1 else rt(v)


def list_fields(values) -> dict:
    """An unordered list of exact numbers (graded by value, any order)."""
    values = sorted(rat(v) for v in values)
    text = ", ".join(rt(v) for v in values)
    return dict(answer_kind="list", answer=[_num(v) for v in values], answer_text=text, answer_latex="plain:" + text)


def list_cands(fields: dict, pairs, n: int = 3) -> list[Distractor]:
    out: list[Distractor] = []
    seen = {fields["answer_text"]}
    for values, tag in pairs:
        if not values:
            continue
        values = sorted(rat(v) for v in values)
        text = ", ".join(rt(v) for v in values)
        if text in seen or accepted_by_grader(fields, text):
            continue
        seen.add(text)
        out.append(Distractor(text, tag, [_num(v) for v in values]))
        if len(out) == n:
            break
    return out


def ordered_fields(values) -> dict:
    """A short ordered list of exact numbers, typed "3, -2, 5" (position matters)."""
    values = [rat(v) for v in values]
    text = ", ".join(rt(v) for v in values)
    return dict(
        answer_kind="ordered_list", answer=[_num(v) for v in values], answer_text=text, answer_latex="plain:" + text
    )


def ordered_cands(fields: dict, pairs, n: int = 3) -> list[Distractor]:
    out: list[Distractor] = []
    seen = {fields["answer_text"]}
    for values, tag in pairs:
        if values is None or any(v is None for v in values):
            continue
        values = [rat(v) for v in values]
        text = ", ".join(rt(v) for v in values)
        if text in seen or accepted_by_grader(fields, text):
            continue
        seen.add(text)
        out.append(Distractor(text, tag, [_num(v) for v in values]))
        if len(out) == n:
            break
    return out


# -- numbers (whole or exact fraction), with the grader kind chosen to suit the value -------------------------------


def number_fields(value, fraction: bool = False) -> dict:
    """The answer of an exact number. The grader kind is `number` for a whole value and `fraction` otherwise; pass
    `fraction=True` for a part that later follow-through could make fractional, so a learner's carried-forward value
    (3/2 where the canonical answer is the whole number 3) is still stated in the part's own kind."""
    value = rat(value)
    kind = "number" if value.denominator == 1 and not fraction else "fraction"
    return dict(answer_kind=kind, **num_fields(value))


def number_cands(correct, pairs, n: int = 3) -> list[Distractor]:
    """Distractors from `(value, tag)` pairs; a fraction is not offered against a whole-number answer."""
    whole = rat(correct).denominator == 1
    seen = {rat(correct)}
    out: list[Distractor] = []
    for value, tag in pairs:
        if value is None:
            continue
        value = rat(value)
        if value in seen or (whole and value.denominator != 1):
            continue
        seen.add(value)
        out.append(num_dist(correct, [(value, tag)], 1)[0])
        if len(out) == n:
            break
    return out


# -- option texts (multiple choice) ------------------------------------------------------------------------------


def mcq_fields(text: str) -> dict:
    return dict(answer_kind="mcq", answer=text, answer_text=text, answer_latex="plain:" + text)


def mcq_cands(correct: str, pairs, n: int = 4) -> list[Distractor]:
    """Option texts from `(text, tag)` pairs: none repeats the answer or another option."""
    out: list[Distractor] = []
    seen = {correct}
    for text, tag in pairs:
        if text is None or text in seen:
            continue
        seen.add(text)
        out.append(Distractor(text, tag, text))
        if len(out) == n:
            break
    return out


def spread(options, correct: str) -> None:
    """The correct option must not be the longest of the options by a clear margin (a giveaway)."""
    longest = max(len(o) for o in options)
    if (
        len(correct) == longest
        and sum(len(o) == longest for o in options) == 1
        and longest > 1.6 * (sum(len(o) for o in options) / len(options))
    ):
        raise TooFewDistractors("the answer is a long outlier")


NAMES = ("Aisha", "Ben", "Chen", "Dana", "Eli", "Farah", "Gus", "Hana")


def sub_fields(fields: dict) -> dict:
    """An answer dictionary without its `answer_kind`, as the `part()` helper wants it."""
    return {key: val for key, val in fields.items() if key != "answer_kind"}


def mc_question(rng, stems, key: str, wrongs, work, n: int = 4, **slots) -> dict:
    """A multiple-choice question: the stem is drawn from `stems` and filled with `slots`; `wrongs` are
    `(option text, misconception id)` pairs of which `n` distinct ones are kept."""
    fields = mcq_fields(key)
    cands = mcq_cands(key, wrongs, n)
    enough(cands, n)
    spread([key, *[c.text for c in cands]], key)
    return dict(text=phrase(rng, stems, **slots), working=work, distractors=cands, **fields)


def tier_pool(pools: dict, rng, tier: str):
    """Draw from the builders of `tier` (a ValueError for an unknown tier), redrawing a degenerate draw."""
    pool = pools.get(tier)
    if pool is None:
        raise ValueError(f"unknown tier {tier!r}")
    return retry_draw(rng, rng.choice(pool))


def length_balanced(rng, key: str, wrongs, n: int = 4):
    """`n` distinct `(text, tag)` wrongs, chosen at random, such that the key is not strictly the longest or the
    shortest option when the pool allows it (so the length of an option never gives the key away)."""
    seen = {key}
    pool = []
    for text, tag in wrongs:
        if text is not None and text not in seen:
            seen.add(text)
            pool.append((text, tag))
    n = min(n, len(pool))
    for _ in range(60):
        pick = rng.sample(pool, n)
        lens = [len(t) for t, _ in pick]
        if min(lens) <= len(key) <= max(lens):
            return pick
    return rng.sample(pool, n)
