"""Exam chain for position vectors as functions of time: two particles that collide.

Particle A moves in a straight line, ``r_A(t) = (u t + p, v t + q)``, and particle B has a quadratic x-component,
``r_B(t) = (e t^2 + f t + g, w t + z)``. They collide at some time ``t* > 0``. Set the way a senior exam sets it:

(a) the time of the collision                                       number,   2 marks
(b) the x-coordinate of the collision point                         fraction, 1 mark,  follow-through (a)
(c) the y-coordinate of the collision point                         fraction, 1 mark,  follow-through (a)
(d) the speed of B at the moment of the collision, 1 d.p.           number,   2 marks, follow-through (a)

## Construction: choose the collision first

The x-coordinates differ by ``e (t - t*)(t - t0)`` with ``t0 <= 0`` (so ``t*`` is the only positive root), and the
y-coordinates differ by ``m (t - t*)``. The velocity of B at the collision, ``(2 e t* + f, w)``, is drawn from a
Pythagorean triple so the speed is a whole number. Everything is whole numbers, and each coordinate and the speed
are re-derived by evaluating both position functions at ``t*``.

## Distractors

* (a): the roots with their signs reversed, the rejected root used.
* (b), (c): the position at the rejected root, the squared term dropped, the coordinates swapped, the starting
  position dropped.
* (d): a single velocity component, the two components added instead of combined, the speed of the other particle.
"""

from __future__ import annotations

from fractions import Fraction
from math import hypot

from mathgen.answers import Distractor, exact_value
from mathgen.core import (
    check,
    generator,
    join_terms,
    latex_answer,
    paren_negative,
    phrase,
    pick_distractors,
    round_half_up,
    stated_precision_string,
    stated_precision_tolerance,
)

_TRIPLES = ((3, 4), (4, 3), (6, 8), (8, 6), (5, 12), (12, 5), (9, 12), (12, 9))

_STEMS = (
    "Two {things} A and B move in a flat region. Relative to a fixed origin, their position vectors at time t "
    "seconds are r_A(t) = ({ax}, {ay}) metres and r_B(t) = ({bx}, {by}) metres, for t ≥ 0. The {things} collide.",
    "The position vectors, in metres, of two {things} A and B at time t seconds (t ≥ 0) are "
    "r_A(t) = ({ax}, {ay}) and r_B(t) = ({bx}, {by}). The two {things} collide.",
    "Relative to a marker, {things} A and B have position vectors r_A(t) = ({ax}, {ay}) and "
    "r_B(t) = ({bx}, {by}), where distances are in metres and t ≥ 0 is the time in seconds. A and B collide.",
)
_THINGS = ("drones", "boats", "jet skis", "go-karts")


def _whole(prompt, value, kind, marks, working, wrong, follow=None, places=None, sensible=False):
    """A whole-number part (`fraction` kind, so a follow-through value stays exact) or a 1 d.p. `number` part."""
    if places is None:
        text = str(value)
        cands = [Distractor(str(v), tag, exact_value(Fraction(v))) for v, tag in wrong]
        answer = exact_value(Fraction(value)) if kind == "fraction" else int(value)
        tolerance = 0
    else:
        shown = round_half_up(Fraction(value), places)
        text = stated_precision_string(shown, places)
        cands = [
            Distractor(
                stated_precision_string(round_half_up(Fraction(v), places), places),
                tag,
                exact_value(round_half_up(Fraction(v), places)),
            )
            for v, tag in wrong
        ]
        answer = float(shown)
        tolerance = stated_precision_tolerance(places)
    part = dict(
        prompt=prompt,
        answer_kind=kind,
        answer=answer,
        answer_text=text,
        answer_latex=latex_answer(text),
        marks=marks,
        working=working,
        distractors=pick_distractors(text, cands, tolerance=tolerance),
    )
    if sensible:
        part["sensible_range"] = (0, None)
    if follow:
        part["follow_through"] = follow
    return part


@generator("exd_x_r2c_collision", level=14, presets={"exam": {"tier": "exam"}})
def exd_x_r2c_collision(rng, tier: str = "exam"):
    """Two particles with different position functions: collision time, point, and speed of B."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    for _ in range(1000):  # bounded rejection loop
        e = rng.choice([1, 2])
        t_star = rng.randint(1, 4)
        t_other = rng.choice([0, -1, -2, -3])
        m = rng.choice([-3, -2, -1, 1, 2, 3])
        w1, w2 = rng.choice(_TRIPLES)
        vb_x = w1 * rng.choice([-1, 1])  # B's x-velocity at the collision
        w = w2 * rng.choice([-1, 1])  # B's (constant) y-velocity
        f = vb_x - 2 * e * t_star
        u = f + e * (t_star + t_other)
        v = w - m
        p, q = rng.randint(-9, 9), rng.randint(-9, 9)
        g = p + e * t_star * t_other
        z = q - m * t_star
        if u != 0 and v != 0 and t_other != -t_star and all(abs(c) <= 20 for c in (f, u, v, g, z)):
            break
    else:  # pragma: no cover
        check(False, "no usable draw")

    def ax(t):
        return u * t + p

    def ay(t):
        return v * t + q

    def bx(t):
        return e * t * t + f * t + g

    def by(t):
        return w * t + z

    # Independent check: the particles coincide at t*, and at no other t >= 0.
    check(ax(t_star) == bx(t_star) and ay(t_star) == by(t_star), "the particles do not meet at t*")
    check(all(ax(t) != bx(t) or ay(t) != by(t) for t in range(0, 12) if t != t_star), "they meet at another time")
    cx, cy = ax(t_star), ay(t_star)
    speed_b = hypot(2 * e * t_star + f, w)
    check(speed_b == hypot(vb_x, w) and speed_b == round(speed_b), "the speed of B is not whole")
    speed_a = hypot(u, v)

    stem = phrase(
        rng,
        _STEMS,
        things=rng.choice(_THINGS),
        ax=join_terms([(u, "t"), (p, "")]),
        ay=join_terms([(v, "t"), (q, "")]),
        bx=join_terms([(e, "t^2"), (f, "t"), (g, "")]),
        by=join_terms([(w, "t"), (z, "")]),
    )

    part_a = _whole(
        "Find the time at which the collision occurs, in seconds.",
        t_star,
        "number",
        2,
        [
            f"Equate the x-components: {e}t² + ({f})t + ({g}) = {u}t + ({p}), so {e}(t - {t_star})(t - ({t_other})) = 0.",
            f"Only t = {t_star} gives t > 0, and the y-components also agree there.",
        ],
        [
            (-t_star, "sign-flip-both-roots"),
            (t_other, "wrong-root-chosen"),
            (-t_other, "sign-flip-both-roots"),
        ],
        sensible=True,
    )
    part_b = _whole(
        "Find the x-coordinate of the point of collision, in metres.",
        cx,
        "fraction",
        1,
        [f"x = {u}·{t_star} + ({p}) = {cx}."],
        [
            (ax(t_other), "wrong-root-chosen"),
            (f * t_star + g, "dropped-the-squared-term"),
            (cy, "swaps-x-and-y"),
            (ax(-t_star), "sign-flip-both-roots"),
        ],
        follow=f"{u}*part_a + ({p})",
    )
    part_c = _whole(
        "Find the y-coordinate of the point of collision, in metres.",
        cy,
        "fraction",
        1,
        [f"y = {v}·{t_star} + ({q}) = {cy}."],
        [
            (ay(t_other), "wrong-root-chosen"),
            (v * t_star, "dropped-constant-term"),
            (cx, "swaps-x-and-y"),
            (ay(-t_star), "sign-flip-both-roots"),
        ],
        follow=f"{v}*part_a + ({q})",
    )
    part_d = _whole(
        "Find the speed of B at the moment of the collision, in metres per second, correct to 1 decimal place.",
        Fraction(int(speed_b)),
        "number",
        2,
        [
            f"v_B(t) = ({2 * e}t + ({f}), {w}); at t = {t_star} this is ({vb_x}, {w}).",
            f"Speed = √({paren_negative(vb_x)}² + {paren_negative(w)}²) = {int(speed_b)} m/s.",
        ],
        [
            (abs(vb_x), "used-one-component-only"),
            (abs(w), "used-one-component-only"),
            (abs(vb_x) + abs(w), "speed-as-sum-of-components"),
            (Fraction(speed_a).limit_denominator(1000), "used-the-other-objects-speed"),
        ],
        follow=f"sqrt((2*{e}*part_a + ({f}))**2 + ({w})**2)",
        places=1,
    )
    return dict(text=stem, parts=[part_a, part_b, part_c, part_d])
