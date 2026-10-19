"""More square-root and cube-root rearrangements for ``rearrange-simple-formulas`` (task w2).

``rearrange_powers_roots`` has seven fixed formula/target cases, so its tier showed only seven different
questions. These six add to them (surface area of a sphere, the other leg of a right triangle, the side of a
cube from its volume or surface area, the time of a fall, the diameter of a circle), in the same mould: the
target sits under a power, so the learner must undo it with a root.

Kept apart from ``p8b_rearrange_powers_build`` so nothing pinned there moves. Like that module, no case
puts a free symbol under a square root in a denominator: the expression grader cannot prove
``sqrt(P/R)`` equal to ``sqrt(P)/sqrt(R)`` for free ``P`` and ``R``, so only ``pi`` and numbers appear
there. The letter ``S`` is never a symbol (sympy reads it as its own singleton), so areas use ``A``.
"""

from __future__ import annotations

from sympy import Eq, Rational, latex, pi, solve, sqrt, symbols

from ..core import check, check_distinct, check_equivalent, generator, pick_distractors
from .p8b_rearrange_powers_build import _display

_A, _r, _s, _V, _d, _t, _a, _b, _c, _x, _y, _p, _q, _h = symbols("A r s V d t a b c x y p q h", positive=True)

# key, target name, target, lhs, rhs, formula text, tex, note, (two wrong answers by slip)
_CASES = [
    ("sphere_area_r", "r", _r, _A, 4 * pi * _r**2, "A = 4πr²", r"A = 4\pi r^2", "r is a radius, so r > 0"),
    (
        "pythagoras_b",
        "b",
        _b,
        _c,
        sqrt(_a**2 + _b**2),
        "c = √(a² + b²)",
        r"c = \sqrt{a^2 + b^2}",
        "b is a length, so b > 0",
    ),
    ("cube_volume_s", "s", _s, _V, _s**3, "V = s³", r"V = s^3", ""),
    ("cube_area_s", "s", _s, _A, 6 * _s**2, "A = 6s²", r"A = 6s^2", "s is a length, so s > 0"),
    ("fall_t", "t", _t, _d, 5 * _t**2, "d = 5t²", r"d = 5t^2", "t is a time, so t > 0"),
    ("circle_d", "d", _d, _A, pi * _d**2 / 4, "A = πd²/4", r"A = \frac{\pi d^2}{4}", "d is a diameter, so d > 0"),
    ("square_area_s", "s", _s, _A, _s**2, "A = s²", r"A = s^2", "s is a length, so s > 0"),
    ("hyp_a", "a", _a, _c**2, _a**2 + _b**2, "c² = a² + b²", r"c^2 = a^2 + b^2", "a is a length, so a > 0"),
    ("hyp_b", "b", _b, _c**2, _a**2 + _b**2, "c² = a² + b²", r"c^2 = a^2 + b^2", "b is a length, so b > 0"),
    ("semicircle_r", "r", _r, _A, pi * _r**2 / 2, "A = πr²/2", r"A = \frac{\pi r^2}{2}", "r is a radius, so r > 0"),
    ("hemisphere_r", "r", _r, _A, 2 * pi * _r**2, "A = 2πr²", r"A = 2\pi r^2", "r is a radius, so r > 0"),
    ("quarter_circle_r", "r", _r, _A, pi * _r**2 / 4, "A = πr²/4", r"A = \frac{\pi r^2}{4}", "r is a radius, so r > 0"),
    ("triple_area_s", "s", _s, _A, 3 * _s**2, "A = 3s²", r"A = 3s^2", "s is a length, so s > 0"),
    ("leg_p", "p", _p, _h**2, _p**2 + _q**2, "h² = p² + q²", r"h^2 = p^2 + q^2", "p is a length, so p > 0"),
    (
        "distance_x",
        "x",
        _x,
        _d,
        sqrt(_x**2 + _y**2),
        "d = √(x² + y²)",
        r"d = \sqrt{x^2 + y^2}",
        "x is a length, so x > 0",
    ),
    ("two_area_s", "s", _s, _A, 2 * _s**2, "A = 2s²", r"A = 2s^2", "s is a length, so s > 0"),
    ("five_area_s", "s", _s, _A, 5 * _s**2, "A = 5s²", r"A = 5s^2", "s is a length, so s > 0"),
    ("seven_area_s", "s", _s, _A, 7 * _s**2, "A = 7s²", r"A = 7s^2", "s is a length, so s > 0"),
    ("eight_area_s", "s", _s, _A, 8 * _s**2, "A = 8s²", r"A = 8s^2", "s is a length, so s > 0"),
    ("height_t", "t", _t, _h, 4 * _t**2, "h = 4t²", r"h = 4t^2", "t is a time, so t > 0"),
    ("three_pi_r", "r", _r, _A, 3 * pi * _r**2, "A = 3πr²", r"A = 3\pi r^2", "r is a radius, so r > 0"),
    ("leg_q", "q", _q, _h**2, _p**2 + _q**2, "h² = p² + q²", r"h^2 = p^2 + q^2", "q is a length, so q > 0"),
    (
        "distance_y",
        "y",
        _y,
        _d,
        sqrt(_x**2 + _y**2),
        "d = √(x² + y²)",
        r"d = \sqrt{x^2 + y^2}",
        "y is a length, so y > 0",
    ),
    (
        "pythagoras_a",
        "a",
        _a,
        _c,
        sqrt(_a**2 + _b**2),
        "c = √(a² + b²)",
        r"c = \sqrt{a^2 + b^2}",
        "a is a length, so a > 0",
    ),
]


_K_AREA = {"two_area_s": 2, "five_area_s": 5, "seven_area_s": 7, "eight_area_s": 8}


def _wrong(key, answer):
    """Two named slips per case, each different from the key for every value of the symbols."""
    if key == "sphere_area_r":
        return [-answer, _A / (4 * pi)]  # both signs offered; never took the root
    if key == "pythagoras_b":
        return [sqrt(_c**2 + _a**2), _c - _a]  # kept the addition; root spread over the difference
    if key == "cube_volume_s":
        return [_V / 3, sqrt(_V)]  # divided by the power; used a square root for a cube
    if key == "cube_area_s":
        return [_A / 6, sqrt(_A) / 6]  # never took the root; rooted A but still divided by 6
    if key == "fall_t":
        return [_d / 5, sqrt(_d) / 5]  # never took the root; rooted d but still divided by 5
    if key == "circle_d":
        return [4 * _A / pi, sqrt(_A / pi)]  # never took the root; stopped at the radius
    if key == "square_area_s":
        return [-answer, _A / 2]  # both signs offered; halved instead of taking the root
    if key == "hyp_a":
        return [sqrt(_c**2 + _b**2), _c - _b]  # kept the addition; root spread over the difference
    if key == "hyp_b":
        return [sqrt(_c**2 + _a**2), _c - _a]
    if key == "semicircle_r":
        return [2 * _A / pi, sqrt(_A / pi)]  # never took the root; forgot the factor of 2
    if key == "hemisphere_r":
        return [_A / (2 * pi), sqrt(_A / pi)]  # never took the root; dropped the 2 under the root
    if key == "quarter_circle_r":
        return [4 * _A / pi, sqrt(_A / pi)]  # never took the root; forgot the factor of 2
    if key == "triple_area_s":
        return [_A / 3, sqrt(_A) / 3]  # never took the root; rooted A but still divided by 3
    if key == "leg_p":
        return [sqrt(_h**2 + _q**2), _h - _q]  # kept the addition; root spread over the difference
    if key == "distance_x":
        return [sqrt(_d**2 + _y**2), _d - _y]  # kept the addition; root spread over the difference
    if key in _K_AREA:
        k = _K_AREA[key]
        return [_A / k, sqrt(_A) / k]  # never took the root; rooted A but still divided by k
    if key == "height_t":
        return [_h / 4, sqrt(_h) / 4]
    if key == "three_pi_r":
        return [_A / (3 * pi), sqrt(_A / pi) / 3]  # never took the root; took the 3 out of the root
    if key == "leg_q":
        return [sqrt(_h**2 + _p**2), _h - _p]
    if key == "distance_y":
        return [sqrt(_d**2 + _x**2), _d - _x]
    if key == "pythagoras_a":
        return [sqrt(_c**2 + _b**2), _c - _b]
    raise AssertionError(key)


@generator("w2_rearrange_roots_more", level=7)
def w2_rearrange_roots_more(rng):
    key, name, target, lhs, rhs, text_, tex, note = rng.choice(_CASES)
    solutions = solve(Eq(lhs, rhs), target)
    check(len(solutions) == 1, f"expected one root for {key}, got {solutions}")
    answer = solutions[0]
    check_equivalent(lhs.subs(target, answer), rhs.subs(target, answer))  # put it back into the formula
    wrong = _wrong(key, answer)
    check_distinct(answer, *wrong)
    shown = _display(answer)
    suffix = f" ({note})" if note else ""
    return dict(
        text=f"Make {name} the subject of the formula {text_}{suffix}",
        latex=tex,
        answer=str(answer),
        answer_text=shown,
        answer_latex=latex(answer),
        working=[f"Start from {text_}", f"{name} = {shown}"],
        distractors=pick_distractors(shown, [_display(w) for w in wrong]),
    )
