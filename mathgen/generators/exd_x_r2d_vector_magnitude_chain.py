"""Exam chain for vector magnitude: lengths and an angle, never components (four parts).

The stem gives ``|a|``, ``|b|`` and the angle between ``a`` and ``b`` (60 or 120 degrees), and names a combination
``p a + q b``.

(a) the dot product ``a . b = |a||b| cos(theta)``;
(b) ``|p a + q b|^2``, expanded with the cross term (follow-through ``p^2|a|^2 + q^2|b|^2 + 2pq`` times ``part_a``);
(c) the exact length, a simplified surd (follow-through ``sqrt(part_b)``);
(d) the angle between ``p a + q b`` and ``a`` to the nearest degree (follow-through from ``part_a`` and ``part_b``).

Every value is exact for the drawn lengths. The lengths are chosen so that ``a . b`` is a whole number, and a draw is
kept only when the squared length is not a perfect square and the final angle sits well clear of a rounding boundary.
"""

from __future__ import annotations

import math
import random

import sympy

from mathgen.answers import Distractor, exact_value
from mathgen.core import (
    attempts,
    check,
    generator,
    join_terms,
    latex_answer,
    paren_negative,
    pick_distractors,
)

_STEM = (
    "The vectors a and b satisfy |a| = {la} and |b| = {lb}, and the angle between a and b is {deg}°.",
    "Two vectors a and b have lengths |a| = {la} and |b| = {lb}, and the angle between them is {deg}°.",
)
_MIN_GAP = 0.02  # a surd distractor must sit well clear of the grader's decimal window around the key


def _int_dists(key: int, pairs, n: int = 3):
    cands = [Distractor(str(v), tag, exact_value(v)) for tag, v in pairs if v != key]
    out = pick_distractors(str(key), cands, n)
    check(len(out) >= 2, "too few distractors")
    return out


def _angle_deg(cos_value: float) -> float:
    return math.degrees(math.acos(max(-1.0, min(1.0, cos_value))))


def _surd(n: int):
    return sympy.sqrt(sympy.Integer(n))


@generator("exd_x_r2d_vector_magnitude_chain", level=12, presets={"exam": dict(tier="exam")})
def exd_x_r2d_vector_magnitude_chain(rng: random.Random, tier: str = "exam") -> dict:
    """a . b, |pa + qb|^2, the exact |pa + qb| and the angle between pa + qb and a."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    for _ in attempts(4000, "a combination with a clean angle"):
        la, lb = rng.randint(2, 8), rng.randint(2, 8)
        deg = rng.choice((60, 120))
        p, q = rng.randint(1, 4), rng.choice((-4, -3, -2, -1, 1, 2, 3, 4))
        if la * lb % 2 or max(p, abs(q)) < 2:
            continue
        sign = 1 if deg == 60 else -1
        ab = sign * la * lb // 2
        n = p * p * la * la + q * q * lb * lb + 2 * p * q * ab
        if n <= 0 or math.isqrt(n) ** 2 == n:
            continue
        cos_phi = (p * la * la + q * ab) / (math.sqrt(n) * la)
        phi = _angle_deg(cos_phi)
        if abs(phi % 1 - 0.5) < 0.06:
            continue
        break
    # Independent route: plane vectors a = (|a|, 0), b = |b| (cos, sin), with the combination worked out in components.
    cos_t, sin_t = sign / 2, math.sqrt(3) / 2  # noqa: F841 - sin_t is used by the component route
    vx, vy = p * la + q * lb * cos_t, q * lb * sin_t
    check(abs(vx * vx + vy * vy - n) < 1e-9, "component route disagrees with the expanded square")
    check(abs(vx * la / (math.sqrt(n) * la) - cos_phi) < 1e-9, "component route disagrees with the angle")
    phi_whole = int(math.floor(phi + 0.5))

    combo = join_terms([(p, "a"), (q, "b")])
    root = _surd(n)
    key_text = str(root)
    key_value = float(root)

    # (a) a . b
    a_pairs = [
        ("forgot-the-cosine", la * lb),
        ("wrong-sign-of-cosine", -ab),
    ]
    # (b) |pa + qb|^2
    b_pairs = [
        ("dropped-cross-term", p * p * la * la + q * q * lb * lb),
        ("wrong-sign-of-cosine", p * p * la * la + q * q * lb * lb - 2 * p * q * ab),
        ("coefficient-not-squared", p * la * la + q * lb * lb + 2 * p * q * ab),
        ("added-magnitudes-not-vectors", (p * la + abs(q) * lb) ** 2),
    ]
    b_pairs = [(t, v) for t, v in b_pairs if v > 0]
    # (c) the exact length: the square roots of the wrong squares, then the square itself.
    c_cands = []
    for tag, v in b_pairs[:2]:
        c_cands.append((tag, _surd(v)))
    c_cands.append(("forgets-square-root", sympy.Integer(n)))
    c_cands.append(("added-magnitudes-not-vectors", sympy.Integer(p * la + abs(q) * lb)))
    c_cands = [(t, e) for t, e in c_cands if abs(float(e) - key_value) > _MIN_GAP]
    c_dist = pick_distractors(key_text, [Distractor(str(e), t, str(e)) for t, e in c_cands], 3, tolerance=0)
    check(len(c_dist) >= 2, "too few distractors")
    # (d) the angle between pa + qb and a
    cos_b = (p * la * ab + q * lb * lb) / (math.sqrt(n) * lb)  # the angle with b instead of a
    d_pairs = [
        ("supplementary-angle", 180 - phi_whole),
        ("used-wrong-vector-pair", int(math.floor(_angle_deg(cos_b) + 0.5))),
        ("used-wrong-vector-pair", deg),
        ("swaps-sine-and-cosine", 90 - phi_whole),
    ]
    n_flip = p * p * la * la + q * q * lb * lb - 2 * p * q * ab  # the cosine's sign wrong, carried through
    if n_flip > 0:
        c_flip = (p * la * la - q * ab) / (math.sqrt(n_flip) * la)
        if abs(c_flip) < 1:
            d_pairs.insert(1, ("wrong-sign-of-cosine", int(math.floor(_angle_deg(c_flip) + 0.5))))
    d_pairs = [(t, v) for t, v in d_pairs if 0 < v < 180]

    stem = rng.choice(_STEM).format(la=la, lb=lb, deg=deg)
    pa2, qb2 = p * p * la * la, q * q * lb * lb
    return dict(
        text=stem,
        parts=[
            dict(
                prompt="Find a · b.",
                answer_kind="number",
                answer=ab,
                answer_text=str(ab),
                answer_latex=latex_answer(str(ab)),
                marks=2,
                working=[f"a · b = |a||b| cos {deg}° = {la} × {lb} × ({'-' if sign < 0 else ''}1/2) = {ab}."],
                distractors=_int_dists(ab, a_pairs),
            ),
            dict(
                prompt=f"Hence find |{combo}|².",
                answer_kind="number",
                answer=n,
                answer_text=str(n),
                answer_latex=latex_answer(str(n)),
                marks=2,
                sensible_range=(0, None),
                follow_through=f"{pa2} + {qb2} + ({2 * p * q})*part_a",
                working=[
                    f"|{combo}|² = {join_terms([(1, f'{paren_negative(p)}²|a|²'), (1, f'{paren_negative(q)}²|b|²'), (2 * p * q, '(a · b)')])}",
                    f"= {pa2} + {qb2} + ({2 * p * q})({ab}) = {n}.",
                ],
                distractors=_int_dists(n, b_pairs),
            ),
            dict(
                prompt=f"Hence find the exact value of |{combo}|, in simplest form.",
                answer_kind="expression",
                answer=key_text,
                answer_text=key_text,
                answer_latex=key_text.replace("sqrt", r"\sqrt"),
                answer_alt=round(key_value, 2),
                answer_form="simplified_surd",
                marks=1,
                follow_through="sqrt(part_b)",
                working=[f"|{combo}| = √{n} = {key_text}."],
                distractors=c_dist,
            ),
            dict(
                prompt=f"Find the angle between {combo} and a, to the nearest degree.",
                answer_kind="number",
                answer=phi_whole,
                answer_text=str(phi_whole),
                answer_latex=latex_answer(str(phi_whole)),
                marks=3,
                follow_through=f"180*acos(({p * la * la} + ({q})*part_a)/(sqrt(part_b)*{la}))/pi",
                working=[
                    f"({combo}) · a = {join_terms([(p, '|a|²'), (q, '(a · b)')])} = {p * la * la + q * ab}.",
                    f"cos φ = {p * la * la + q * ab} ÷ (√{n} × {la}), so φ = {phi_whole}°.",
                ],
                distractors=_int_dists(phi_whole, d_pairs),
            ),
        ],
    )
