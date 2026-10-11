"""Exam-tier source for deciding where a composition of ``f(x) = 1/(x - k)`` and ``g(x) = sqrt(x - a)`` is defined.

The practice tiers compose ``f`` after ``g`` once, so the domain is one ray with at most one point removed. Here
the composition is built from the same two functions in a different or longer pattern, and the learner has to
solve an equation or inequality that arises part-way through to find the domain.

Four question shapes, each built from chosen integers ``a`` and ``k``:

- ``self_composition``: ``f(f(x))`` is undefined at ``x = k`` and where ``1/(x - k) = k``, that is
  ``x = k + 1/k``.
- ``root_of_reciprocal``: ``g(f(x))`` needs ``x != k`` and ``1/(x - k) >= a``. For ``a > 0`` that gives
  ``k < x <= k + 1/a``.
- ``root_of_root``: ``g(g(x))`` needs ``x >= a`` and ``sqrt(x - a) >= a``. For ``a > 0`` that gives
  ``x >= a + a^2``.
- ``three_fold``: ``f(g(f(x)))`` needs ``k < x <= k + 1/a`` from the inner two functions, and the outer
  reciprocal fails where ``sqrt(1/(x - k) - a) = k``. For ``k > 0`` that removes ``x = k + 1/(a + k^2)``; for
  ``k < 0`` a square root can never equal ``k`` and nothing more is removed.

Every key is cross-checked by direct evaluation: the printed functions are evaluated with exact fractions (for
the reciprocals) and floating point (for the roots) on a grid of points around the critical values, and the
description is compared with whether the composite really evaluates. Distractors, each a named slip:

- ``self_composition``: removing only ``x = k``; solving ``1/(x - k) = k`` as ``x = 2k``; a sign error giving
  ``k - 1/k``; using ``1/k`` for the second point.
- ``root_of_reciprocal``: dropping the condition ``x > k``; flipping the inequality; making the upper bound
  strict; including ``x = k``.
- ``root_of_root``: only the domain of the inner root, ``x >= a``; adding instead of squaring (``x >= 2a``);
  a sign error (``x >= a - a^2``).
- ``three_fold``: forgetting the outer exclusion; applying the exclusion when ``k`` is negative; leaving ``k``
  unsquared when solving for the excluded point; dropping the lower bound ``x > k``; making the upper bound
  strict; including ``x = k``. Each is offered only where it changes the set of valid ``x``.
"""

from __future__ import annotations

import math
import random
from fractions import Fraction

from ..core import check, generator, latex_answer, pick_distractors, rational_text


def _shift(base: str, k: int) -> str:
    """``base - k`` with the sign chosen so the text never reads ``- -n``."""
    return f"{base} - {k}" if k >= 0 else f"{base} + {-k}"


def _f_text(k: int) -> str:
    return f"1/({_shift('x', k)})"


def _g_text(a: int) -> str:
    return f"sqrt({_shift('x', a)})"


def _grid(lo: Fraction, hi: Fraction, steps: int = 90) -> list[Fraction]:
    step = (hi - lo) / steps
    return [lo + step * i for i in range(steps + 1)]


def _defined(fn, pt: Fraction) -> bool:
    try:
        fn(pt)
    except (ValueError, ZeroDivisionError):
        return False
    return True


def _recip(v: Fraction, k: int) -> Fraction:
    return 1 / (v - k)  # ZeroDivisionError at v = k


def _root(v: float) -> float:
    return math.sqrt(v)  # ValueError for negatives


def _result(text: str, answer: str, candidates: list[str], working: list[str]) -> dict:
    # A slip can coincide with the key for one small parameter value (for example 2k = k + 1/k when k = 1); such
    # a candidate is dropped here, and the rest must still offer three distractors.
    candidates = [c for c in candidates if c != answer]
    check(len(candidates) >= 3, "fewer than three distinct distractors")
    return dict(
        text=text,
        answer=answer,
        answer_text=answer,
        answer_latex=latex_answer(answer, plain=True),
        working=working,
        distractors=pick_distractors(answer, candidates, n=3),
    )


def _self_composition(rng: random.Random) -> dict:
    k = rng.choice([v for v in range(-6, 7) if abs(v) >= 2])  # |k| = 1 makes several slips coincide
    other = Fraction(k) + Fraction(1, k)

    def composite(v: Fraction) -> float:
        return _recip(_recip(v, k), k)

    # Independent check: direct evaluation says the composite fails exactly at x = k and x = k + 1/k.
    points = {Fraction(k), other} | {Fraction(k) + Fraction(i, 3) for i in range(-9, 10)}
    for pt in points:
        check(_defined(composite, pt) == (pt not in (Fraction(k), other)), f"description disagrees at x = {pt}")
    answer = f"Defined for all real x, except x = {k} and x = {rational_text(other)}"
    candidates = [
        f"Defined for all real x, except x = {k}",
        f"Defined for all real x, except x = {k} and x = {2 * k}",
        f"Defined for all real x, except x = {k} and x = {rational_text(Fraction(k) - Fraction(1, k))}",
        f"Defined for all real x, except x = {k} and x = {rational_text(Fraction(1, k))}",
    ]
    text = f"Given f(x) = {_f_text(k)}, for what values of x is f(f(x)) defined? Select the correct description."
    working = [
        f"f(f(x)) needs x != {k} so that f(x) exists.",
        f"It also needs f(x) != {k}: 1/({_shift('x', k)}) = {k} gives {_shift('x', k)} = 1/{k}, so x = {rational_text(other)}.",
    ]
    return _result(text, answer, candidates, working)


def _root_of_reciprocal(rng: random.Random) -> dict:
    a = rng.randint(1, 6)
    k = rng.randint(-6, 6)
    top = Fraction(k) + Fraction(1, a)

    def composite(v: Fraction) -> float:
        return _root(float(_recip(v, k) - a))

    for pt in _grid(Fraction(k) - 2, top + 2) + [Fraction(k), top]:
        check(_defined(composite, pt) == (Fraction(k) < pt <= top), f"description disagrees at x = {pt}")
    t = rational_text(top)
    answer = f"Defined for {k} < x <= {t}"
    candidates = [
        f"Defined for x <= {t}",
        f"Defined for x >= {t}",
        f"Defined for {k} < x < {t}",
        f"Defined for {k} <= x <= {t}",
    ]
    text = (
        f"Given f(x) = {_f_text(k)} and g(x) = {_g_text(a)}, for what values of x is g(f(x)) defined? "
        "Select the correct description."
    )
    working = [
        f"g(f(x)) = sqrt(1/({_shift('x', k)}) - {a}) needs x != {k} and 1/({_shift('x', k)}) >= {a}.",
        f"Since {a} > 0 this forces {_shift('x', k)} > 0, and then {_shift('x', k)} <= 1/{a}, so {k} < x <= {t}.",
    ]
    return _result(text, answer, candidates, working)


def _root_of_root(rng: random.Random) -> dict:
    a = rng.randint(2, 9)  # a = 1 makes a, a^2 and a + a^2 - a coincide in several slips
    edge = a + a * a

    def composite(v: Fraction) -> float:
        return _root(_root(float(v - a)) - a)

    for pt in [Fraction(i, 2) for i in range(2 * (a - 3), 2 * (edge + 5))]:
        check(_defined(composite, pt) == (pt >= edge), f"description disagrees at x = {pt}")
    answer = f"Defined for x >= {edge}"
    candidates = [
        f"Defined for x >= {a}",
        f"Defined for x >= {2 * a}",
        f"Defined for x >= {a - a * a}",
        f"Defined for x >= {a * a}",
    ]
    text = f"Given g(x) = {_g_text(a)}, for what values of x is g(g(x)) defined? Select the correct description."
    working = [
        f"g(g(x)) = sqrt(sqrt({_shift('x', a)}) - {a}) needs x >= {a} for the inner root.",
        f"It also needs sqrt({_shift('x', a)}) >= {a}, so {_shift('x', a)} >= {a * a}, giving x >= {edge}.",
    ]
    return _result(text, answer, candidates, working)


def _three_fold(rng: random.Random) -> dict:
    a = rng.randint(1, 4)
    k = rng.choice([v for v in range(-5, 6) if v != 0])
    top = Fraction(k) + Fraction(1, a)
    removed = Fraction(k) + Fraction(1, a + k * k) if k > 0 else None

    def composite(v: Fraction) -> float:
        inner = _root(float(_recip(v, k) - a))
        if abs(inner - k) < 1e-12:
            raise ZeroDivisionError
        return 1 / (inner - k)

    grid = _grid(Fraction(k) - 2, top + 2) + [Fraction(k), top]
    if removed is not None:
        grid.append(removed)
    for pt in grid:
        expected = Fraction(k) < pt <= top and pt != removed
        check(_defined(composite, pt) == expected, f"description disagrees at x = {pt}")
    t = rational_text(top)
    base = f"{k} < x <= {t}"
    # Every candidate below changes the set of x values (a stated exclusion lies inside k < x <= top), so none
    # is secretly equivalent to the key.
    if removed is not None:
        answer = f"Defined for {base}, except x = {rational_text(removed)}"
        candidates = [
            f"Defined for {base}",
            f"Defined for {base}, except x = {rational_text(Fraction(k) + Fraction(1, a + k))}",
            f"Defined for {k} < x < {t}, except x = {rational_text(removed)}",
            f"Defined for x <= {t}, except x = {rational_text(removed)}",
        ]
        tail = (
            f"The outer reciprocal fails when sqrt(1/({_shift('x', k)}) - {a}) = {k}. Squaring, "
            f"1/({_shift('x', k)}) = {a + k * k}, so x = {rational_text(removed)}."
        )
    else:
        answer = f"Defined for {base}"
        candidates = [
            f"Defined for {base}, except x = {rational_text(Fraction(k) + Fraction(1, a + k * k))}",
            f"Defined for x <= {t}",
            f"Defined for {k} < x < {t}",
            f"Defined for {k} <= x <= {t}",
        ]
        tail = (
            f"The outer reciprocal would fail where sqrt(...) = {k}, but a square root is never negative, "
            "so nothing more is removed."
        )
    text = (
        f"Given f(x) = {_f_text(k)} and g(x) = {_g_text(a)}, for what values of x is f(g(f(x))) defined? "
        "Select the correct description."
    )
    working = [
        f"g(f(x)) needs {k} < x <= {t} (x != {k} and 1/({_shift('x', k)}) >= {a}).",
        tail,
    ]
    return _result(text, answer, candidates, working)


@generator(
    "exd_x_f2_when_defined",
    level=12,
    presets={"exam": dict(tier="exam")},
)
def exd_x_f2_when_defined(rng: random.Random, tier: str = "exam") -> dict:
    """Exam-tier domain-of-a-composition question: another pattern of composition, with an equation or inequality to solve."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}; this source only has 'exam'")
    shape = rng.choice(["self_composition", "root_of_reciprocal", "root_of_root", "three_fold"])
    if shape == "self_composition":
        return _self_composition(rng)
    if shape == "root_of_reciprocal":
        return _root_of_reciprocal(rng)
    if shape == "root_of_root":
        return _root_of_root(rng)
    return _three_fold(rng)
