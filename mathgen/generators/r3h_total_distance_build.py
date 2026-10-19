"""Generates `r3h_total_distance` questions (easy, medium, hard) and the exam chain `exd_x_r3h_total_distance`.

``total-distance-direction-change`` (``differentiation``, answer ``number``): a particle moves along a straight line and
turns round at least once in the time interval, so the distance it travels is more than the change in its position. The
learner finds when the velocity is zero, splits the interval there and adds the sizes of the changes in position.

* easy: ``s(t)`` is a quadratic and the particle turns once (the vertex), over a stated interval.
* medium: ``s(t)`` is a cubic; the particle turns twice, and the interval contains either both turns or only one.
* hard: the *velocity* is given, a quadratic with a rest at a fractional time; the total distance is found by integrating
  the speed over the interval (number, 2 decimal places).
* exam (``exd_x_r3h_total_distance``): (a) the times at rest, (b) the displacement over the interval, (c) the total
  distance, (d) the average speed (follows through from (c)).

Every total distance is checked against a numerical integration of the speed |v(t)| with Simpson's rule, which does not
use the times at rest at all.
"""

from __future__ import annotations

import sympy

from ..answers import Distractor
from ..core import attempts, check, generator, join_terms, pick_distractors, rounds_unambiguously
from ._r3h_kit import Redraw, build_with_redraws, need, number_options, number_part, prompt_part, say_marked

_R = sympy.Rational
t = sympy.Symbol("t")

_THINGS = (
    ("a cyclist on a straight path", "metres", "seconds", "post"),
    ("a tram on a straight track", "metres", "seconds", "depot"),
    ("a drone flying along a straight line", "metres", "seconds", "launch point"),
    ("a particle moving along a straight line", "metres", "seconds", "origin"),
    ("a lift in a tall building", "metres", "seconds", "ground floor"),
)


def _simpson_speed(v, lo, hi, n=20000):
    """The integral of |v| over [lo, hi] by Simpson's rule: the independent route."""
    h = (hi - lo) / n
    total = abs(v(lo)) + abs(v(hi))
    for i in range(1, n):
        total += (4 if i % 2 else 2) * abs(v(lo + i * h))
    return total * h / 3


def _fn(expr):
    return sympy.lambdify(t, expr, "math")


def _distance(s, rests, lo, hi):
    """Total distance of s(t) on [lo, hi], splitting at the rests inside it (exact)."""
    cuts = [lo] + [r for r in rests if lo < r < hi] + [hi]
    return sum(abs(s.subs(t, b) - s.subs(t, a)) for a, b in zip(cuts, cuts[1:]))


def _poly_text(expr) -> str:
    poly = sympy.Poly(sympy.expand(expr), t)
    terms = [(int(c), f"t^{m[0]}" if m[0] > 1 else ("t" if m[0] == 1 else "")) for m, c in poly.terms()]
    return join_terms(terms)


def _speeds_and_context(rng):
    thing, unit, tunit, origin = rng.choice(_THINGS)
    return thing, unit, tunit, origin


def _draw_quadratic(rng):
    t0 = rng.randint(1, 5)
    a = rng.choice([-3, -2, -1, 1, 2, 3])
    s0 = rng.randint(0, 20)
    T = t0 + rng.randint(2, 5)
    s = a * (t - t0) ** 2 + s0
    s = sympy.expand(s)
    rests = [t0]
    return s, rests, T


def _draw_cubic(rng, both: bool):
    for _ in attempts(500, "a cubic position function with integer coefficients"):
        a = rng.choice([1, 2])
        p = rng.randint(1, 4)
        q = p + rng.randint(2, 4)
        if (a * (p + q)) % 2:
            continue
        s0 = rng.randint(0, 15)
        sign = rng.choice([1, -1])
        s = sympy.expand(sign * (a * t**3 - _R(3 * a * (p + q), 2) * t**2 + 3 * a * p * q * t) + s0)
        T = q + rng.randint(1, 3) if both else p + rng.randint(1, q - p - 1)
        if not both and not (p < T < q):
            continue
        return s, [p, q], T
    raise AssertionError("unreachable")


def _pieces(s, rests, lo, hi):
    cuts = [lo] + [r for r in rests if lo < r < hi] + [hi]
    return [s.subs(t, b) - s.subs(t, a) for a, b in zip(cuts, cuts[1:])]


def _distance_wrongs(s, rests, T, v, a_fn):
    """Slips in finding the total distance, as (tag, value)."""
    pieces = _pieces(s, rests, 0, T)
    out = [
        ("r3h-motion-displacement-for-distance", abs(sum(pieces))),
        ("r3h-motion-stops-at-first-rest", abs(pieces[0])),
        ("r3h-motion-omits-last-segment", sum(abs(p) for p in pieces[:-1])),
        ("r3h-motion-position-for-distance", abs(s.subs(t, T))),
    ]
    if len(pieces) > 2:
        out.append(("r3h-motion-omits-middle-segment", abs(pieces[0]) + abs(pieces[-1])))
    # splitting where the acceleration is zero instead of where the velocity is zero
    acc_zero = [float(r) for r in sympy.solve(sympy.diff(s, t, 2), t) if 0 < float(r) < T]
    if acc_zero:
        cuts = [0] + sorted(acc_zero) + [T]
        out.append(
            (
                "r3h-motion-acceleration-zero-for-rest",
                sum(abs(float(s.subs(t, b) - s.subs(t, a))) for a, b in zip(cuts, cuts[1:])),
            )
        )
    # the signed sum of the pieces, with the last one counted the wrong way
    if len(pieces) > 1:
        out.append(("r3h-motion-signs-kept", abs(sum(pieces[:-1]) - pieces[-1])))
    return out


def _number_question(rng, tier):
    thing, unit, tunit, origin = _speeds_and_context(rng)
    if tier == "easy":
        s, rests, T = _draw_quadratic(rng)
        given = f"s = {_poly_text(s)}"
        form = "position"
    elif tier == "medium":
        s, rests, T = _draw_cubic(rng, both=rng.random() < 0.5)
        given = f"s = {_poly_text(s)}"
        form = "position"
    else:
        form = "velocity"
    if form == "position":
        v = sympy.diff(s, t)
        total = _distance(s, rests, 0, T)
        check(
            abs(float(total) - _simpson_speed(_fn(v), 0, T)) < 1e-6 * max(1, float(total)),
            "total distance fails the numerical check",
        )
        wrongs = _distance_wrongs(s, rests, T, v, None)
        stem, latex = say_marked(
            f"The position of {thing}, in {unit} from the {origin}, is «{given}» at time «t» {tunit}. "
            f"Find the total distance it travels in the first {T} {tunit}, in {unit}."
        )
        dp = 0
        answer = float(total)
        if not sympy.Rational(total).is_Integer:
            raise Redraw
        dist = number_options(answer, dp, [(tag, float(val)) for tag, val in wrongs if val is not None], gap=1)
        working = [
            f"The velocity is s' = {_poly_text(v)}, which is zero at t = {', '.join(str(r) for r in rests if 0 < r < T)}.",
            f"Distance = {' + '.join(str(abs(p)) for p in _pieces(s, rests, 0, T))} = {total}.",
        ]
        check(len(dist) >= 2, "too few total-distance distractors")
        return dict(
            text=stem,
            latex=latex,
            answer=int(total),
            answer_text=str(int(total)),
            answer_kind="number",
            distractors=dist,
            working=working,
        )
    return _velocity_question(rng, thing, unit, tunit, origin)


def _velocity_question(rng, thing, unit, tunit, origin):
    for _ in attempts(500, "a velocity with a fractional rest time and a clean distance"):
        m = rng.choice([2, 3])
        n = rng.choice([1, 3, 5]) if m == 2 else rng.choice([1, 2, 4, 5])
        r1 = _R(n, m)
        r2 = rng.randint(3, 7)
        if r1 >= r2 - 1:
            continue
        k = rng.choice([1, 2])
        v = sympy.expand(k * (m * t - n) * (t - r2))
        T = r2 + rng.randint(1, 3)
        s = sympy.integrate(v, t)
        total = _distance(s, [r1, r2], 0, T)
        answer = float(total)
        if not rounds_unambiguously(answer, 2, 0.1):
            continue
        check(
            abs(answer - _simpson_speed(_fn(v), 0, T)) < 1e-6 * max(1, answer),
            "total distance fails the numerical check",
        )
        pieces = _pieces(s, [r1, r2], 0, T)
        wrongs = [
            ("r3h-motion-displacement-for-distance", abs(float(sum(pieces)))),
            ("r3h-motion-stops-at-first-rest", abs(float(pieces[0]))),
            ("r3h-motion-omits-last-segment", float(sum(abs(p) for p in pieces[:-1]))),
            ("r3h-motion-omits-middle-segment", float(abs(pieces[0]) + abs(pieces[-1]))),
            ("r3h-motion-velocity-not-integrated", abs(float(v.subs(t, T)))),
            ("r3h-motion-average-for-distance", float(total) / T),
        ]
        dist = number_options(answer, 2, wrongs)
        if len(dist) < 2:
            continue
        break
    stem, latex = say_marked(
        f"The velocity of {thing}, in {unit} per {tunit[:-1]}, is «v = {_poly_text(v)}» at time «t» {tunit}. "
        f"Find the total distance it travels in the first {T} {tunit}, in {unit} correct to 2 decimal places."
    )
    return dict(
        text=stem,
        latex=latex,
        answer=round(answer, 2),
        answer_text=f"{answer:.2f}",
        answer_kind="number",
        distractors=dist,
        working=[
            f"v = 0 at t = {r1} and t = {r2}. Integrate |v| over [0, {r1}], [{r1}, {r2}] and [{r2}, {T}]: "
            + " + ".join(f"{abs(float(p)):.2f}" for p in pieces)
            + f" = {answer:.2f}."
        ],
    )


@generator(
    "r3h_total_distance",
    level=15,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def r3h_total_distance(rng, tier="easy"):
    return build_with_redraws(rng, lambda r: _number_question(r, tier))


# ---------------------------------------------------------------------------------------------------------------
# exam chain


def _exam_once(rng):
    thing, unit, tunit, origin = _speeds_and_context(rng)
    s, rests, T = _draw_cubic(rng, both=True)
    v = sympy.diff(s, t)
    pieces = _pieces(s, rests, 0, T)
    total = sum(abs(p) for p in pieces)
    disp = s.subs(t, T) - s.subs(t, 0)
    check(
        abs(float(total) - _simpson_speed(_fn(v), 0, T)) < 1e-6 * max(1, float(total)),
        "total distance fails the numerical check",
    )
    if abs(disp) == total or not rounds_unambiguously(float(total) / T, 2, 0.1) or total <= abs(disp):
        raise Redraw
    avg = float(total) / T
    stem, latex = say_marked(
        f"The position of {thing}, in {unit} from the {origin}, is «s = {_poly_text(s)}» at time «t» {tunit}, for «0 ≤ t ≤ {T}»."
    )
    # (a) the rest times, as a list
    p, q = rests
    a_answer = [p, q]
    seen = {(p, q)}
    a_opts = []
    for tag, w in (
        ("r3h-motion-one-rest-only", (p,)),
        (
            "r3h-motion-acceleration-zero-for-rest",
            tuple(int(r) for r in sympy.solve(sympy.diff(s, t, 2), t) if r == int(r)),
        ),
        ("r3h-motion-position-zero-for-rest", tuple(int(r) for r in sympy.solve(s, t) if r.is_Integer and r > 0)),
        ("r3h-motion-rest-sum-product", (p + q, p * q)),
        ("r3h-motion-sign-of-rest-times", (-p, -q)),
    ):
        if w and w not in seen and len(w) <= 2:
            seen.add(w)
            a_opts.append(Distractor(", ".join(str(x) for x in w), tag, list(w)))
    a_opts = pick_distractors(f"{p}, {q}", a_opts, 4)
    need(a_opts)
    pieces_f = [float(x) for x in pieces]
    wrongs_b = [
        ("r3h-motion-total-distance-for-displacement", float(total)),
        ("r3h-motion-position-for-displacement", float(s.subs(t, T))),
        ("r3h-motion-displacement-reversed", -float(disp)),
        ("r3h-motion-final-segment-only", pieces_f[-1]),
    ]
    wrongs_c = [(tag, float(val)) for tag, val in _distance_wrongs(s, rests, T, v, None)]
    parts = [
        dict(
            **prompt_part(
                f"Find the times, in {tunit}, at which the motion is momentarily at rest. Give them as a list, in increasing order."
            ),
            answer_kind="list",
            answer=a_answer,
            answer_text=f"{p}, {q}",
            marks=2,
            working=[f"s' = {_poly_text(v)} = 0 gives t = {p} and t = {q}."],
            distractors=a_opts,
        ),
        number_part(
            f"Find the displacement, the change in position, over the first {T} {tunit}, in {unit}.",
            float(disp),
            0,
            1,
            [f"s({T}) - s(0) = {s.subs(t, T)} - {s.subs(t, 0)} = {disp}."],
            [(tg, val) for tg, val in wrongs_b if val is not None],
            gap=1,
        ),
        number_part(
            f"Find the total distance travelled in the first {T} {tunit}, in {unit}.",
            float(total),
            0,
            3,
            [f"The pieces are {', '.join(str(abs(x)) for x in pieces)}, so the distance is {total}."],
            wrongs_c,
            gap=1,
            sensible_range=(0, None),
        ),
        number_part(
            f"Find the average speed over the first {T} {tunit}, in {unit} per {tunit[:-1]} correct to 2 decimal places.",
            avg,
            2,
            1,
            [f"Average speed = total distance / time = {total}/{T} = {avg:.2f}."],
            [
                ("r3h-motion-average-velocity-for-speed", float(disp) / T),
                ("r3h-motion-time-multiplied", float(total) * T),
                ("r3h-motion-total-per-segment", float(total) / (len(pieces))),
                ("r3h-motion-displacement-per-time-abs", abs(float(disp)) / T),
            ],
            follow_through=f"part_c/{T}",
            sensible_range=(0, None),
        ),
    ]
    return dict(text=stem, latex=latex, parts=parts)


@generator("exd_x_r3h_total_distance", level=15, presets={"exam": dict(tier="exam")})
def exd_x_r3h_total_distance(rng, tier="exam"):
    """Exam tier: the times at rest, the displacement, the total distance and the average speed."""
    check(tier == "exam", f"unknown tier {tier!r}")
    return build_with_redraws(rng, _exam_once)
