"""Exam tier for vector and Cartesian equations of a plane: a plane through three points.

The practice tiers give a normal vector and a point. Here only three points ``A``, ``B``, ``C`` are
given. The learner must find a normal (``AB x AC``), write the Cartesian equation ``ax + by + cz = d``
using one of the points, and then use it to find the missing coordinate ``k`` of a fourth point that
lies on the plane.

Built backwards: ``P = A + s AB + t AC`` with small integer ``s``, ``t`` always lies on the plane and
has integer coordinates. One coordinate of ``P`` is hidden as ``k``; the hidden slot is chosen where the
normal has a nonzero component, so ``k`` is determined uniquely. Answer kind is ``number`` (an integer).
"""

from __future__ import annotations

import random

from sympy import Matrix

from ..core import check, generator, pick_distractors

_PATTERNS = (
    "The plane through A{a}, B{b} and C{c} contains the point P{p}. Find k.",
    "A, B and C are the points {a}, {b} and {c}. The point P{p} lies on the plane through A, B and C. Find the value of k.",
)
_AXES = ("x", "y", "z")


def _fmt(v) -> str:
    return "(" + ", ".join(str(x) for x in v) + ")"


def _sub(u, v):
    return [a - b for a, b in zip(u, v)]


def _dot(u, v) -> int:
    return sum(a * b for a, b in zip(u, v))


def _solve_slot(normal, d: int, point, slot: int):
    """Value of coordinate ``slot`` so that normal . point == d, or None if not a whole number."""
    rest = sum(normal[i] * point[i] for i in range(3) if i != slot)
    num = d - rest
    if normal[slot] == 0 or num % normal[slot]:
        return None
    return num // normal[slot]


def _build(rng: random.Random) -> dict:
    for _ in range(500):
        a, b, c = ([rng.randint(-4, 4) for _ in range(3)] for _ in range(3))
        ab, ac = _sub(b, a), _sub(c, a)
        n = [int(x) for x in Matrix(ab).cross(Matrix(ac))]
        if not any(n):
            continue
        s, t = rng.choice([-2, -1, 0, 1, 2]), rng.choice([-2, -1, 0, 1, 2])
        if (s, t) in {(0, 0), (1, 0), (0, 1)}:  # P would be A, B or C itself
            continue
        p = [a[i] + s * ab[i] + t * ac[i] for i in range(3)]
        slot = rng.choice([i for i in range(3) if n[i] != 0])
        k = p[slot]
        d = _dot(n, a)
        # Independent checks: P, B and C all satisfy the plane equation, and solving for the slot returns k.
        check(_dot(n, p) == d and _dot(n, b) == d and _dot(n, c) == d, "a point is off the plane")
        check(_solve_slot(n, d, p, slot) == k, "solving the plane equation does not return the hidden coordinate")
        # Principled wrong answers.
        forgot_divide = d - sum(
            n[i] * p[i] for i in range(3) if i != slot
        )  # numerator, never divided by the coefficient
        sum_normal = [ab[i] + ac[i] for i in range(3)]  # added the edge vectors instead of crossing them
        cands = [-k, forgot_divide]
        if any(sum_normal):
            cands.append(_solve_slot(sum_normal, _dot(sum_normal, a), p, slot))
        other = [i for i in range(3) if i != slot and n[i] != 0]
        if other:
            cands.append(_solve_slot_wrong_coefficient(n, d, p, slot, other[0]))
        # A sign slip moving the other terms across the equals sign.
        rest = sum(n[i] * p[i] for i in range(3) if i != slot)
        if (d + rest) % n[slot] == 0:
            cands.append((d + rest) // n[slot])
        cands = [x for x in cands if x is not None]
        distractors = pick_distractors(str(k), [str(x) for x in cands])
        if k == 0 or len(distractors) < 3:
            continue
        shown = [str(v) for v in p]
        shown[slot] = "k"
        text = rng.choice(_PATTERNS).format(a=_fmt(a), b=_fmt(b), c=_fmt(c), p="(" + ", ".join(shown) + ")")
        return dict(
            text=text,
            latex=text,
            answer=k,
            answer_text=str(k),
            answer_latex=str(k),
            distractors=distractors,
        )
    raise ValueError("plane: no usable draw in 500 attempts")


def _solve_slot_wrong_coefficient(normal, d, point, slot, wrong):
    """Divided by the wrong normal component (the coefficient of another axis)."""
    rest = sum(normal[i] * point[i] for i in range(3) if i != slot)
    num = d - rest
    return num // normal[wrong] if num % normal[wrong] == 0 else None


@generator(
    "exd_x_g1_vector_plane",
    level=12,
    presets={"exam": dict(tier="exam")},
)
def exd_x_g1_vector_plane(rng: random.Random, tier: str = "exam") -> dict:
    """The unknown coordinate of a point on the plane through three given points."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    return _build(rng)
