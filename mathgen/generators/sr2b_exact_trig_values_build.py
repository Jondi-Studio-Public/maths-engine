"""Generates `sr2b_exact_trig_values` questions (easy, medium, hard) and the exam chain
`sr2b_exam_exact_trig_values`.

Skill ``exact-trig-values-typed`` (``trigonometry``, answer ``expression``): the exact value of sine, cosine and
tangent at multiples of pi/6 and pi/4 in every quadrant, typed as a surd or fraction (``-sqrt(3)/2``), and the angle in
radians (typed ``5pi/6``) that has a stated value. The existing recognition skills for these values are multiple choice;
this one asks for the value itself.

* easy: sin, cos or tan of an angle in the first or second quadrant, from pi/6, pi/4 and pi/3 (and their supplements).
* medium: an angle in the third or fourth quadrant, a negative angle, or an angle larger than 2 pi, so the angle has to
  be reduced and the sign found from the quadrant.
* hard: either the angle in [0, 2 pi) that has a stated value of one function and a stated sign of another (the value
  alone gives two angles), or the exact value of a product of two values at angles in different quadrants.
* exam (``sr2b_exam_exact_trig_values``): for an angle in the second or third quadrant, (a) sin, (b) cos, (c) tan
  (follows through from (a) and (b)), (d) the other angle in [0, 2 pi) with the same sine.

The value of each function is found from the reference angle and the quadrant's sign, then checked against SymPy's
own evaluation of the function at the angle.
"""

from __future__ import annotations

import functools
import math
import re
from fractions import Fraction

import sympy as sp

from ..answers import Distractor
from ..core import attempts, check, generator, phrase
from ._sr2b_kit import pi_latex, pi_shown, pi_text, pick, retry_short

_S2, _S3 = sp.sqrt(2), sp.sqrt(3)
# value of each function at the reference angle k pi (first quadrant)
_REF = {
    Fraction(1, 6): {"sin": sp.Rational(1, 2), "cos": _S3 / 2, "tan": _S3 / 3},
    Fraction(1, 4): {"sin": _S2 / 2, "cos": _S2 / 2, "tan": sp.Integer(1)},
    Fraction(1, 3): {"sin": _S3 / 2, "cos": sp.Rational(1, 2), "tan": _S3},
}
_FN = {"sin": sp.sin, "cos": sp.cos, "tan": sp.tan}


def _typed(expr) -> str:
    return str(expr).replace("**", "^")


def _say(expr) -> str:
    """An exact value as the working prints it: 2√2, √3/2, -√3/3 (the answer itself is typed as 2*sqrt(2))."""
    text = _typed(expr).replace("*", "")
    return re.sub(r"sqrt\((\d+)\)", r"√\1", text)


def _latex(expr) -> str:
    return sp.latex(expr)


def reduce_angle(c: Fraction) -> Fraction:
    """An angle in units of pi, reduced into [0, 2)."""
    return c % 2


def reference(c: Fraction) -> tuple[Fraction, int, int]:
    """(reference angle, sign of sine, sign of cosine) for an angle in units of pi, not on an axis."""
    r = reduce_angle(c)
    if r < Fraction(1, 2):
        return r, 1, 1
    if r < 1:
        return 1 - r, 1, -1
    if r < Fraction(3, 2):
        return r - 1, -1, -1
    return 2 - r, -1, 1


def value(fn: str, c: Fraction):
    """Exact value of fn at c*pi by reference angle and quadrant sign (the construction route)."""
    ref, ss, cs = reference(c)
    sign = {"sin": ss, "cos": cs, "tan": ss * cs}[fn]
    return sign * _REF[ref][fn]


@functools.cache
def value_check(fn: str, c: Fraction):
    """The same value, confirmed against SymPy's evaluation of the function at the angle (once per angle)."""
    v = value(fn, c)
    exact = _FN[fn](sp.pi * sp.Rational(c.numerator, c.denominator))
    check(sp.simplify(exact - v) == 0, f"{fn}({c} pi) differs from SymPy")
    return v


def _float(fn: str, c: Fraction) -> float:
    """The function at c*pi as a float (a cheap independent check)."""
    return getattr(math, fn)(math.pi * float(c))


def _on_axis(c: Fraction) -> bool:
    return (c * 2).denominator == 1


def _angle_pool() -> list[Fraction]:
    out = []
    for den in (4, 6):
        for k in range(1, 2 * den):
            c = Fraction(k, den)
            if not _on_axis(c) and c not in out:
                out.append(c)
    return sorted(out)


_POOL = _angle_pool()


def _ex(expr, tag: str) -> Distractor:
    t = _typed(sp.sympify(expr))
    return Distractor(t, tag, t)


def _value_distractors(fn: str, c: Fraction, ans) -> list:
    """Wrong values for fn(c pi): each from an error a learner makes."""
    ref, ss, cs = reference(c)
    sign = {"sin": ss, "cos": cs, "tan": ss * cs}[fn]
    other = {"sin": "cos", "cos": "sin", "tan": "tan"}[fn]
    out = [_ex(-ans, "sr2b-exact-value-sign-error")]
    out.append(_ex(sign * _REF[ref][other], "sr2b-exact-value-sin-cos-swapped"))
    comp = Fraction(1, 2) - ref
    if comp in _REF:
        out.append(_ex(sign * _REF[comp][fn], "sr2b-exact-value-complement-used"))
    if fn == "tan":
        out.append(_ex(sign / _REF[ref]["tan"], "sr2b-exact-value-tan-inverted"))
        out.append(_ex(sign * _REF[ref]["sin"], "sr2b-exact-value-wrong-function"))
    else:
        out.append(_ex(sign * _REF[ref]["tan"], "sr2b-exact-value-wrong-function"))
        out.append(_ex(sign / _REF[ref][fn], "sr2b-exact-value-reciprocal-given"))
    return out


def _fn_label(fn: str, c: Fraction) -> str:
    return f"{fn}({pi_shown(c)})"


def _single(rng, pool, patterns):
    c = rng.choice(pool)
    fn = rng.choice(("sin", "cos", "tan"))
    mult = rng.choice((1, 1, 1, 2, 3, 4, 6, 8))
    base = value_check(fn, c)
    ans = mult * base
    ref, ss, cs = reference(c)
    sign = {"sin": ss, "cos": cs, "tan": ss * cs}[fn]
    text = _typed(ans)
    label = _fn_label(fn, c)
    stem = phrase(rng, patterns, e=label if mult == 1 else f"{mult} {label}")
    quad = {(1, 1): "first", (1, -1): "second", (-1, -1): "third", (-1, 1): "fourth"}[(ss, cs)]
    steps = []
    if c != reduce_angle(c):
        steps.append(f"The angle is equivalent to {pi_shown(reduce_angle(c))}, which is in the {quad} quadrant.")
    elif ref != c:
        steps.append(f"The angle is in the {quad} quadrant, with reference angle {pi_shown(ref)}.")
    if ref == reduce_angle(c):
        steps.append(f"{_fn_label(fn, ref)} = {_say(base)} is a standard value.")
    else:
        steps.append(
            f"{fn} is {'positive' if sign > 0 else 'negative'} in the {quad} quadrant, so {label} = {'' if sign > 0 else '-'}{fn}({pi_shown(ref)}) = {_say(base)}."
        )
    if mult != 1:
        steps.append(f"Multiplying by {mult} gives {_say(mult * base)}.")
    cands = [_ex(mult * sp.sympify(d.value), d.misconception) for d in _value_distractors(fn, c, base)]
    if mult != 1:
        cands.insert(1, _ex(base, "sr2b-exact-value-multiplier-dropped"))
    return dict(
        text=stem,
        answer_kind="expression",
        answer=text,
        answer_text=text,
        answer_latex=_latex(ans),
        working=steps,
        distractors=pick(text, cands, 3),
    )


_STEMS = (
    "Find the exact value of {e}.",
    "Evaluate {e} exactly. Type surds as sqrt(3), for example.",
    "What is the exact value of {e}?",
)


def _easy(rng):
    pool = [c for c in _POOL if c < 1]
    return _single(rng, pool, _STEMS)


def _medium(rng):
    pool = [c for c in _POOL if c > 1] + [c - 2 for c in _POOL] + [c + 2 for c in _POOL]
    return _single(rng, pool, _STEMS)


def _hard_inverse(rng):
    for _ in attempts(500, "an angle that a value and a sign fix"):
        theta = rng.choice(_POOL)
        fn = rng.choice(("sin", "cos", "tan"))
        cond = {
            "sin": rng.choice(("cos", "tan")),
            "cos": rng.choice(("sin", "tan")),
            "tan": rng.choice(("sin", "cos")),
        }[fn]
        v = value_check(fn, theta)
        cv = value_check(cond, theta)
        matches = [c for c in _POOL if value(fn, c) == v and (value(cond, c) > 0) == (cv > 0)]
        every = [c for c in _POOL if value(fn, c) == v]
        if matches == [theta] and len(every) == 2:
            break
    # second route: SymPy evaluation of both conditions at the answer, and at the rejected angle
    check(abs(_float(fn, theta) - float(v)) < 1e-9, "value differs")
    check((_float(cond, theta) > 0) == (cv > 0), "sign differs")
    other = next(c for c in every if c != theta)
    ref = reference(theta)[0]
    sign_word = ">" if cv > 0 else "<"
    ans = pi_text(theta)
    cands = [
        Distractor(pi_text(other), "sr2b-angle-wrong-quadrant", pi_text(other)),
        Distractor(pi_text(ref), "sr2b-angle-reference-angle-given", pi_text(ref)),
        Distractor(pi_text(reduce_angle(-theta)), "sr2b-angle-negated", pi_text(reduce_angle(-theta))),
        Distractor(pi_text(reduce_angle(theta + 1)), "sr2b-angle-added-pi", pi_text(reduce_angle(theta + 1))),
    ]
    cands = [x for x in cands if x.text != ans]
    stem = phrase(
        rng,
        (
            "Find the angle θ, where 0 ≤ θ < 2π, for which {fn} θ = {v} and {cond} θ {s} 0. Give θ exactly in radians.",
            "Given that {fn} θ = {v} and {cond} θ {s} 0, find θ in radians, where 0 ≤ θ < 2π. Give an exact value.",
        ),
        fn=fn,
        v=_typed(v),
        cond=cond,
        s=sign_word,
    )
    return dict(
        text=stem,
        answer_kind="expression",
        answer=ans,
        answer_text=ans,
        answer_latex=pi_latex(theta),
        working=[
            f"{fn} θ = {_say(v)} has two solutions in [0, 2π): {pi_shown(min(theta, other))} and {pi_shown(max(theta, other))}.",
            f"Only {pi_shown(theta)} has {cond} θ {sign_word} 0, so θ = {pi_shown(theta)}.",
        ],
        distractors=pick(ans, cands),
    )


def _hard_product(rng):
    for _ in attempts(500, "a product with a surd result that is not zero"):
        c1, c2 = rng.choice(_POOL), rng.choice(_POOL)
        f1, f2 = rng.choice(("sin", "cos")), rng.choice(("sin", "cos", "tan"))
        v1, v2 = value_check(f1, c1), value_check(f2, c2)
        prod = v1 * v2
        if reference(c1)[0] == reference(c2)[0] or abs(prod) in (1, sp.Rational(1, 4)):
            continue
        break
    check(abs(_float(f1, c1) * _float(f2, c2) - float(prod)) < 1e-9, "product differs from the float evaluation")
    ans = _typed(prod)
    cands = [
        _ex(abs(v1) * abs(v2) * (1 if prod > 0 else -1) * -1, "sr2b-exact-value-sign-error"),
        _ex(v1 + v2, "sr2b-exact-value-sum-for-product"),
        _ex(-prod, "sr2b-exact-value-sign-error"),
        _ex(value(f2 if f2 != "tan" else "sin", c1) * value(f1, c2), "sr2b-exact-value-sin-cos-swapped"),
    ]
    stem = f"Find the exact value of {_fn_label(f1, c1)} × {_fn_label(f2, c2)}. Type surds as sqrt(3), for example."
    return dict(
        text=stem,
        answer_kind="expression",
        answer=ans,
        answer_text=ans,
        answer_latex=_latex(prod),
        working=[
            f"{_fn_label(f1, c1)} = {_say(v1)} and {_fn_label(f2, c2)} = {_say(v2)}.",
            f"Their product is {_say(prod)}.",
        ],
        distractors=pick(ans, cands),
    )


@generator(
    "sr2b_exact_trig_values",
    level=12,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
@retry_short
def sr2b_exact_trig_values(rng, tier="easy"):
    """An exact trigonometric value or special angle; the tiers are described in the module docstring."""
    if tier == "easy":
        return _easy(rng)
    if tier == "medium":
        return _medium(rng)
    check(tier == "hard", f"unknown tier {tier!r}")
    return rng.choice([_hard_inverse, _hard_inverse, _hard_product])(rng)


# -- exam -------------------------------------------------------------------------------------------------------


@generator("sr2b_exam_exact_trig_values", level=12, presets={"exam": dict(tier="exam")})
@retry_short
def sr2b_exam_exact_trig_values(rng, tier="exam"):
    """Exam chain: sin, cos, tan (from the first two) and the other angle with the same sine."""
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(200, "an angle with enough wrong values at every part"):
        theta = rng.choice([c for c in _POOL if 0 < c < 2 and c not in (Fraction(1, 2), 1, Fraction(3, 2))])
        s, co, t = value_check("sin", theta), value_check("cos", theta), value_check("tan", theta)
        check(abs(float(s / co) - float(t)) < 1e-9, "tan is not sin over cos")
        other = reduce_angle(1 - theta)
        check(abs(_float("sin", other) - float(s)) < 1e-9, "other angle differs")
        check(other != theta, "the other angle is the same")
        ref = reference(theta)[0]
        sin_d = [
            _ex(-s, "sr2b-exact-value-sign-error"),
            _ex(co, "sr2b-exact-value-sin-cos-swapped"),
            _ex(reference(theta)[1] * _REF[ref]["tan"], "sr2b-exact-value-wrong-function"),
        ]
        cos_d = [
            _ex(-co, "sr2b-exact-value-sign-error"),
            _ex(s, "sr2b-exact-value-sin-cos-swapped"),
            _ex(reference(theta)[2] * _REF[ref]["tan"], "sr2b-exact-value-wrong-function"),
        ]
        tan_d = [
            _ex(-t, "sr2b-exact-value-sign-error"),
            _ex(1 / t, "sr2b-exact-value-tan-inverted"),
            _ex(co / s, "sr2b-exact-value-tan-inverted"),
            _ex(s, "sr2b-exact-value-wrong-function"),
            _ex(co, "sr2b-exact-value-wrong-function"),
        ]
        ang_d = [
            Distractor(pi_text(ref), "sr2b-angle-reference-angle-given", pi_text(ref)),
            Distractor(pi_text(reduce_angle(-theta)), "sr2b-angle-negated", pi_text(reduce_angle(-theta))),
            Distractor(pi_text(reduce_angle(theta + 1)), "sr2b-angle-added-pi", pi_text(reduce_angle(theta + 1))),
            Distractor(pi_text(reduce_angle(2 - ref)), "sr2b-angle-wrong-quadrant", pi_text(reduce_angle(2 - ref))),
        ]
        ang_d = [d for d in ang_d if d.text != pi_text(other)]
        lists = ((sin_d, s), (cos_d, co), (tan_d, t))
        if (
            all(len({d.text for d in lst if d.text != _typed(ans)}) >= 2 for lst, ans in lists)
            and len({d.text for d in ang_d}) >= 2
        ):
            break
    return dict(
        text=phrase(
            rng,
            ("Let θ = {t}.", "Consider the angle θ = {t}.", "The angle θ is {t} radians."),
            t=pi_shown(theta),
        ),
        parts=[
            dict(
                prompt="Find the exact value of sin θ. Type surds as sqrt(3), for example.",
                answer_kind="expression",
                answer=_typed(s),
                answer_text=_typed(s),
                answer_latex=_latex(s),
                marks=1,
                working=[
                    f"θ has reference angle {pi_shown(ref)}; sine is {'positive' if s > 0 else 'negative'} there, so sin θ = {_say(s)}."
                ],
                distractors=pick(_typed(s), [d for d in sin_d if d.text != _typed(s)]),
            ),
            dict(
                prompt="Find the exact value of cos θ.",
                answer_kind="expression",
                answer=_typed(co),
                answer_text=_typed(co),
                answer_latex=_latex(co),
                marks=1,
                working=[f"Cosine is {'positive' if co > 0 else 'negative'} at θ, so cos θ = {_say(co)}."],
                distractors=pick(_typed(co), [d for d in cos_d if d.text != _typed(co)]),
            ),
            dict(
                prompt="Hence find the exact value of tan θ.",
                answer_kind="expression",
                answer=_typed(t),
                answer_text=_typed(t),
                answer_latex=_latex(t),
                marks=1,
                follow_through="part_a/part_b",
                working=[f"tan θ = sin θ ÷ cos θ = ({_say(s)}) ÷ ({_say(co)}) = {_say(t)}."],
                distractors=pick(_typed(t), [d for d in tan_d if d.text != _typed(t)]),
            ),
            dict(
                prompt="There is one other angle α, with 0 ≤ α < 2π and α ≠ θ, for which sin α = sin θ. Find α exactly in radians.",
                answer_kind="expression",
                answer=pi_text(other),
                answer_text=pi_text(other),
                answer_latex=pi_latex(other),
                marks=2,
                working=[
                    f"The two angles with the same sine add to π, so α = π − {pi_shown(theta)} = {pi_shown(other)} (taken into [0, 2π))."
                ],
                distractors=pick(pi_text(other), ang_d),
            ),
        ],
    )
