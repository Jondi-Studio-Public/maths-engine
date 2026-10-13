"""Generates `fy1ca_related_rates` and `fy1ca_related_rates_exam` questions.

One first-year skill, ``ucv1-related-rates``: relate the rates of change of two or more quantities with the chain
rule. Every single-answer key is an exact value (a rational, or a rational multiple of pi, or of 1/pi), graded as an
expression.

Tiers (the reasoning escalates; the numbers stay small):

- ``easy``: one formula is standard (area of a circle, volume or surface area of a sphere) and the rate of the radius
  is given. Differentiate with respect to time and substitute (an oil slick, a bushfire front, a weather balloon).
- ``medium``: the relation has to be set up from the situation: Pythagoras for a ladder against a wall or for two
  vehicles leaving an intersection (the signs of the rates matter), or similar triangles for the shadow of a person
  walking away from a lamp post.
- ``hard``: a variable has to be eliminated before differentiating: a cone of grain or an inverted conical tank whose
  radius is a fixed multiple of its depth (find the rate of the depth), or a melting hailstone, where the surface
  area's rate follows from the volume's through the radius.
- ``exam`` (topic ``fy1ca_related_rates_exam``): a four-part chain on a conical stockpile or an inflating balloon:
  (a) the volume at a stated size, (b) dV/dh there, (c) the rate of the size from the rate of the volume, (d) the time
  to double the size, each from the one before.

Every key is re-derived by a second route: the rate is estimated from a central difference of the quantity along a
time path on which the stated rates hold, and compared with the exact value.
"""

from __future__ import annotations

import math
import random
from fractions import Fraction

import sympy
from sympy import Rational, pi

from ..core import check, generator, latex_answer, phrase
from ._fy1ca_kit import (
    Redraw,
    build_with_redraws,
    clear_of_rounding,
    close,
    expression_distractors,
    frac,
    grid,
    need_distractors,
    number_distractors,
    number_part,
    shown,
    typed,
)

_TRIPLES = ((3, 4, 5), (5, 12, 13), (8, 15, 17), (7, 24, 25), (20, 21, 29))
_RATES = (Fraction(1, 2), Fraction(1), Fraction(3, 2), Fraction(2), Fraction(3), Fraction(5, 2), Fraction(4))


def _num(value) -> str:
    """A rate as it is written in a sentence: 2, 0.5, 1.5."""
    value = frac(value)
    if value.denominator == 1:
        return str(value.numerator)
    return f"{float(value):g}"


_RATIO_WORDS = {
    Fraction(1, 2): "half its height",
    Fraction(1): "equal to its height",
    Fraction(3, 2): "one and a half times its height",
    Fraction(2, 3): "two-thirds of its height",
    Fraction(3, 4): "three-quarters of its height",
}


def _ratio_text(value) -> str:
    """A rational as it is typed inside a formula: 1/2, 3/4, 1."""
    value = frac(value)
    return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"


def _with_h(value) -> str:
    """A ratio times h as written in a worked step: h, (1/2)h, 3h."""
    value = frac(value)
    if value == 1:
        return "h"
    return f"({_ratio_text(value)})h" if value.denominator != 1 else f"{value.numerator}h"


def _sym(value) -> sympy.Expr:
    value = frac(value)
    return Rational(value.numerator, value.denominator)


def _question(text: str, answer, wrong, working) -> dict:
    expr = sympy.nsimplify(answer) if not isinstance(answer, sympy.Basic) else answer
    distractors = expression_distractors(expr, [(w, tag) for w, tag in wrong])
    need_distractors(distractors)
    return dict(
        text=text,
        answer=str(expr),
        answer_text=typed(expr),
        answer_kind="expression",
        answer_latex=latex_answer(expr),
        working=working,
        distractors=distractors,
    )


# -- easy ------------------------------------------------------------------------------------------------------------


def _easy(rng: random.Random) -> dict:
    form = rng.choice(["slick", "fire", "balloon", "bubble"])
    r = rng.choice([2, 3, 4, 5, 6, 8, 10])
    dr = rng.choice([1, 2, 3, 4, 5])
    if form in ("slick", "fire"):
        answer = 2 * pi * r * dr
        wrong = [
            (2 * pi * dr, "fy1ca-rates-variable-dropped"),
            (2 * pi * r, "fy1ca-rates-chain-rule-forgotten"),
            (pi * r * dr, "fy1ca-rates-power-rule-slip"),
            (pi * r**2 * dr, "fy1ca-rates-formula-not-differentiated"),
        ]
        formula, derivative = "A = πr²", "dA/dt = 2πr · dr/dt"
        if form == "slick":
            text = phrase(
                rng,
                (
                    "An oil slick off the Western Australian coast spreads as a circle whose radius grows at {dr} m/min. How fast is the area of the slick increasing, in m²/min, when the radius is {r} m?",
                    "Oil leaking from a wrecked tanker forms a circular slick. Its radius is increasing at a steady {dr} m/min. Find the rate of increase of the slick's area, in m²/min, at the moment the radius is {r} m.",
                ),
                dr=dr,
                r=r,
            )
        else:
            text = phrase(
                rng,
                (
                    "A grass fire on a flat plain spreads outwards in a circle, its radius growing at {dr} km/h. At what rate is the burnt area growing, in km²/h, when the radius is {r} km?",
                    "The front of a bushfire on open grassland is a circle that is expanding at {dr} km/h. How quickly is the area inside the front changing, in km²/h, when it has a radius of {r} km?",
                ),
                dr=dr,
                r=r,
            )
        # second route: the area along the path r(t) = r + dr * t, differenced numerically
        area = lambda t: math.pi * (r + dr * t) ** 2  # noqa: E731
    elif form == "balloon":
        answer = 4 * pi * r**2 * dr
        wrong = [
            (4 * pi * r * dr, "fy1ca-rates-power-rule-slip"),
            (4 * pi * r**2, "fy1ca-rates-chain-rule-forgotten"),
            (Rational(4, 3) * pi * r**3 * dr, "fy1ca-rates-formula-not-differentiated"),
            (Rational(4, 3) * pi * r**2 * dr, "fy1ca-rates-wrong-coefficient"),
        ]
        formula, derivative = "V = (4/3)πr³", "dV/dt = 4πr² · dr/dt"
        text = phrase(
            rng,
            (
                "A weather balloon is being inflated so that its radius grows at {dr} cm/s. How fast is its volume increasing, in cm³/s, when the radius is {r} cm?",
                "The radius of a spherical weather balloon increases at {dr} cm/s as it is filled with helium. Find the rate at which the volume is increasing, in cm³/s, when the radius reaches {r} cm.",
            ),
            dr=dr,
            r=r,
        )
        area = lambda t: 4 / 3 * math.pi * (r + dr * t) ** 3  # noqa: E731
    else:
        answer = 8 * pi * r * dr
        wrong = [
            (8 * pi * dr, "fy1ca-rates-variable-dropped"),
            (4 * pi * r * dr, "fy1ca-rates-power-rule-slip"),
            (8 * pi * r, "fy1ca-rates-chain-rule-forgotten"),
            (4 * pi * r**2 * dr, "fy1ca-rates-formula-not-differentiated"),
        ]
        formula, derivative = "S = 4πr²", "dS/dt = 8πr · dr/dt"
        text = phrase(
            rng,
            (
                "A spherical soap bubble expands so that its radius grows at {dr} mm/s. How fast is the surface area of the bubble increasing, in mm²/s, when the radius is {r} mm?",
                "As a bubble is blown, its radius increases at {dr} mm/s. Find the rate of increase of the bubble's surface area, in mm²/s, at the instant the radius is {r} mm.",
            ),
            dr=dr,
            r=r,
        )
        area = lambda t: 4 * math.pi * (r + dr * t) ** 2  # noqa: E731
    h = 1e-4
    check(close((area(h) - area(-h)) / (2 * h), float(answer), 1e-6), "numerical rate disagrees")
    working = [
        f"The relation is {formula}.",
        f"Differentiate with respect to t: {derivative}.",
        f"Substitute r = {r} and dr/dt = {dr}.",
    ]
    return _question(text, answer, wrong, working)


# -- medium --------------------------------------------------------------------------------------------------------------


def _ladder(rng: random.Random) -> dict:
    a, b, c = rng.choice(_TRIPLES)
    scale = rng.choice([1, 1, 2])
    x, y, length = a * scale, b * scale, c * scale
    if rng.random() < 0.5:
        x, y = y, x
    v = rng.choice(_RATES)
    top_rate = -Fraction(x) * v / y  # dy/dt
    speed_asked = rng.random() < 0.6
    if speed_asked:
        answer = _sym(Fraction(x) * v / y)
        ask = "at what speed is the top of the ladder sliding down the wall, in m/s"
        wrong = [
            (_sym(Fraction(y) * v / x), "fy1ca-rates-ratio-inverted"),
            (_sym(v), "fy1ca-rates-rates-assumed-equal"),
            (_sym(Fraction(x) * v / length), "fy1ca-rates-wrong-length"),
            (-_sym(Fraction(x) * v / y), "fy1ca-rates-sign-for-speed"),
        ]
    else:
        answer = _sym(top_rate)
        ask = "what is the rate of change of the height of the top of the ladder above the ground, in m/s"
        wrong = [
            (-_sym(top_rate), "fy1ca-rates-sign-for-speed"),
            (_sym(-Fraction(y) * v / x), "fy1ca-rates-ratio-inverted"),
            (-_sym(v), "fy1ca-rates-rates-assumed-equal"),
            (_sym(-Fraction(x) * v / length), "fy1ca-rates-wrong-length"),
        ]
    # second route: the height along the path x(t) = x + v t keeps the ladder's length
    height = lambda t: math.sqrt(length**2 - (x + float(v) * t) ** 2)  # noqa: E731
    h = 1e-5
    check(close((height(h) - height(-h)) / (2 * h), float(top_rate), 1e-6), "numerical rate disagrees")
    text = phrase(
        rng,
        (
            "A {length} m ladder leans against a vertical wall. Its base is pulled away from the wall at {v} m/s. When the base is {x} m from the wall, {ask}?",
            "The foot of a {length} m ladder resting against a wall is dragged along the ground, away from the wall, at {v} m/s. At the moment its foot is {x} m from the wall, {ask}?",
        ),
        length=length,
        v=_num(v),
        x=x,
        ask=ask,
    )
    working = [
        f"Let x be the distance of the base from the wall and y the height of the top: x² + y² = {length}².",
        f"Differentiate: 2x dx/dt + 2y dy/dt = 0, so dy/dt = -(x/y)(dx/dt).",
        f"When x = {x}, y = {y}, so dy/dt = -({x}/{y})({_num(v)}) = {shown(_sym(top_rate))} m/s.",
    ]
    return _question(text, answer, wrong, working)


def _vehicles(rng: random.Random) -> dict:
    a, b, c = rng.choice(_TRIPLES)
    x, y, s = (a, b, c) if rng.random() < 0.5 else (b, a, c)
    # x km north of the junction (moving north at u), y km east of it (moving west, towards the junction, at w)
    u = rng.choice([40, 50, 60, 80])
    w = rng.choice([30, 40, 50, 60, 90])
    rate = (Fraction(x) * u - Fraction(y) * w) / s
    answer = _sym(rate)
    wrong = [
        (_sym(Fraction(x) * u + Fraction(y) * w) / s, "fy1ca-rates-direction-sign"),
        (_sym(Fraction(x) * u - Fraction(y) * w), "fy1ca-rates-division-dropped"),
        (_sym(u - w), "fy1ca-rates-rates-combined"),
        (_sym(Fraction(y) * u - Fraction(x) * w) / s, "fy1ca-rates-components-swapped"),
    ]
    # second route: the distance along the path x(t) = x + u t, y(t) = y - w t
    dist = lambda t: math.hypot(x + u * t, y - w * t)  # noqa: E731
    h = 1e-6
    check(close((dist(h) - dist(-h)) / (2 * h), float(rate), 1e-6), "numerical rate disagrees")
    text = phrase(
        rng,
        (
            "Two roads in the Riverina cross at right angles. A ute is {x} km north of the crossing and driving north at {u} km/h; a truck is {y} km east of it and driving west, towards the crossing, at {w} km/h. At what rate is the distance between them changing, in km/h? (A negative answer means the distance is shrinking.)",
            "At a crossroads outside Wagga Wagga, one car is {x} km north of the intersection heading north at {u} km/h, while another is {y} km east of it, travelling west towards it at {w} km/h. Find the rate of change of the distance between the cars, in km/h, at this moment. Use a negative value if they are getting closer.",
        ),
        x=x,
        y=y,
        u=u,
        w=w,
    )
    working = [
        f"Let s be the distance between the cars: s² = x² + y², with s = {s} when x = {x} and y = {y}.",
        "Differentiate: 2s ds/dt = 2x dx/dt + 2y dy/dt, with dx/dt = " + f"{u} and dy/dt = -{w}.",
        f"ds/dt = ({x}({u}) + {y}(-{w}))/{s} = {shown(answer)} km/h.",
    ]
    return _question(text, answer, wrong, working)


def _shadow(rng: random.Random) -> dict:
    lamp = rng.choice([4, 5, 6, 7, 9])
    person = rng.choice([Fraction(3, 2), Fraction(8, 5), Fraction(9, 5), 2])
    v = rng.choice(_RATES)
    person = frac(person)
    tip_rate = Fraction(lamp) * v / (lamp - person)
    length_rate = person * v / (lamp - person)
    ask_tip = rng.random() < 0.5
    answer = _sym(tip_rate if ask_tip else length_rate)
    wrong = [
        (_sym(length_rate if ask_tip else tip_rate), "fy1ca-rates-shadow-tip-and-length-confused"),
        (_sym(v), "fy1ca-rates-rates-assumed-equal"),
        (_sym(Fraction(lamp) * v / person if ask_tip else person * v / lamp), "fy1ca-rates-ratio-inverted"),
        (
            _sym(Fraction(lamp) * v / (lamp + person) if ask_tip else person * v / (lamp + person)),
            "fy1ca-rates-similar-triangles-sum",
        ),
    ]
    # second route: tip position T satisfies T / lamp = (T - d) / person for a walker at distance d from the post
    tip = lambda d: lamp * d / (lamp - float(person))  # noqa: E731
    length = lambda d: tip(d) - d  # noqa: E731
    h = 1e-5
    fn = tip if ask_tip else length
    check(close((fn(float(v) * h) - fn(-float(v) * h)) / (2 * h), float(answer), 1e-6), "numerical rate disagrees")
    which = "the tip of her shadow moving along the ground" if ask_tip else "the length of her shadow increasing"
    text = phrase(
        rng,
        (
            "A street lamp is {lamp} m high. A woman who is {person} m tall walks directly away from the base of the lamp at {v} m/s. How fast is {which}, in m/s?",
            "On a footpath in Hobart, a {lamp} m lamp post casts the shadow of a {person} m tall woman walking straight away from it at {v} m/s. At what rate is {which}, in m/s?",
        ),
        lamp=lamp,
        person=_num(person),
        v=_num(v),
        which=which,
    )
    working = [
        f"Let d be her distance from the post and T the distance of the shadow's tip from the post. Similar triangles give T/{lamp} = (T - d)/{_num(person)}.",
        f"So T = {lamp}d/({lamp} - {_num(person)}), and the shadow's length is T - d = {_num(person)}d/({lamp} - {_num(person)}).",
        f"With dd/dt = {_num(v)} this gives {shown(answer)} m/s.",
    ]
    return _question(text, answer, wrong, working)


def _medium(rng: random.Random) -> dict:
    return rng.choice([_ladder, _vehicles, _shadow])(rng)


# -- hard ----------------------------------------------------------------------------------------------------------------


def _cone_pile(rng: random.Random) -> dict:
    k = rng.choice([Fraction(1, 2), Fraction(1), Fraction(3, 2), Fraction(2, 3)])
    h0 = rng.choice([2, 3, 4, 5, 6])
    rate = rng.choice([2, 3, 4, 5, 6, 10])
    kk = _sym(k)
    answer = Rational(rate) / (pi * kk**2 * h0**2)
    wrong = [
        (Rational(rate) / (pi * kk * h0**2), "fy1ca-rates-ratio-not-squared"),
        (3 * Rational(rate) / (pi * kk**2 * h0**2), "fy1ca-rates-cone-third-forgotten"),
        (Rational(rate) / (3 * pi * kk**2 * h0**2), "fy1ca-rates-power-rule-slip"),
        (Rational(rate) / (pi * kk**2 * h0**3), "fy1ca-rates-formula-not-differentiated"),
    ]
    # second route: height along the path on which the volume grows at the stated rate
    v0 = math.pi * float(k) ** 2 * h0**3 / 3
    height = lambda t: (3 * (v0 + rate * t) / (math.pi * float(k) ** 2)) ** (1 / 3)  # noqa: E731
    h = 1e-5
    check(close((height(h) - height(-h)) / (2 * h), float(answer), 1e-6), "numerical rate disagrees")
    material = rng.choice(["wheat", "gravel", "sand", "mulch"])
    relation = f"the radius of its base is always {_RATIO_WORDS[k]}"
    text = phrase(
        rng,
        (
            "A conveyor pours {material} onto a heap at {rate} m³/min. The heap keeps the shape of a cone for which {relation}. How fast is the height of the heap increasing, in m/min, when it is {h0} m high?",
            "{Material} is tipped from a truck at {rate} m³/min and forms a conical pile; {relation}. Find the rate at which the height of the pile is rising, in m/min, at the instant the pile is {h0} m high.",
        ),
        material=material,
        Material=material.capitalize(),
        rate=rate,
        relation=relation,
        h0=h0,
    )
    working = [
        f"The radius is r = {_with_h(k)}, so V = (1/3)πr²h = (1/3)π({_with_h(k)})²h = ({shown(pi * kk**2 / 3)})h³.",
        f"Differentiate: dV/dt = ({shown(pi * kk**2)})h² · dh/dt.",
        f"With dV/dt = {rate} and h = {h0}: dh/dt = {rate}/({shown(pi * kk**2 * h0**2)}) = {shown(answer)}.",
    ]
    return _question(text, answer, wrong, working)


def _tank(rng: random.Random) -> dict:
    top, depth = rng.choice([(2, 4), (3, 6), (2, 6), (3, 4), (4, 6), (1, 2)])
    h0 = rng.choice([1, 2, 3]) * (1 if depth <= 4 else 1)
    h0 = min(h0, depth - 1)
    q = rng.choice([1, 2, 3, 4, 5])
    ratio = Rational(top, depth)
    # V = (1/3) pi (ratio h)^2 h; water leaves at q m^3/min, so the depth changes at -q / (pi ratio^2 h^2)
    rate = -Rational(q) / (pi * ratio**2 * h0**2)
    ask_fall = rng.random() < 0.5
    answer = -rate if ask_fall else rate
    wrong = [
        (-answer, "fy1ca-rates-sign-for-speed"),
        (answer * ratio**2 * 3, "fy1ca-rates-cone-third-forgotten"),
        (answer / 3, "fy1ca-rates-power-rule-slip"),
        (Rational(q) / (pi * ratio * h0**2) * (1 if ask_fall else -1), "fy1ca-rates-ratio-not-squared"),
    ]
    v0 = math.pi * float(ratio) ** 2 * h0**3 / 3
    height = lambda t: (3 * (v0 - q * t) / (math.pi * float(ratio) ** 2)) ** (1 / 3)  # noqa: E731
    h = 1e-5
    check(close((height(h) - height(-h)) / (2 * h), float(rate), 1e-6), "numerical rate disagrees")
    ask = (
        "how fast is the water level falling, in m/min"
        if ask_fall
        else "what is the rate of change of the depth of the water, in m/min (negative if the level is falling)"
    )
    text = phrase(
        rng,
        (
            "Water drains from an inverted conical tank, {top} m across at the top and {depth} m deep, at {q} m³/min. When the water is {h0} m deep, {ask}?",
            "A farm tank is an upside-down cone with a top diameter of {top2} m and a depth of {depth} m. It is being emptied at {q} m³/min. At the moment the water is {h0} m deep, {ask}?",
        ),
        top=top * 2,
        top2=top * 2,
        depth=depth,
        q=q,
        h0=h0,
        ask=ask,
    )
    working = [
        f"By similar triangles the radius of the water surface is r = ({shown(ratio)})h, so V = (1/3)πr²h = ({shown(pi * ratio**2 / 3)})h³.",
        f"Differentiate: dV/dt = ({shown(pi * ratio**2)})h² · dh/dt, and dV/dt = -{q}.",
        f"With h = {h0}: ({shown(pi * ratio**2 * h0**2)}) · dh/dt = -{q}, so dh/dt = {shown(rate)} m/min.",
    ]
    return _question(text, answer, wrong, working)


def _hailstone(rng: random.Random) -> dict:
    r0 = rng.choice([1, 2, 3, 4, 5])
    q = rng.choice([2, 3, 4, 6, 8, 12])
    # V' = -q (cm^3/min) so r' = -q / (4 pi r^2) and S' = 8 pi r r' = -2q / r
    answer = Rational(-2 * q, r0)
    ask_decrease = rng.random() < 0.5
    asked = -answer if ask_decrease else answer
    wrong = [
        (-asked, "fy1ca-rates-sign-for-speed"),
        (Rational(-q, r0) * (-1 if ask_decrease else 1), "fy1ca-rates-power-rule-slip"),
        (Rational(-2 * q) * (-1 if ask_decrease else 1), "fy1ca-rates-variable-dropped"),
        (Rational(-2 * q, 3 * r0) * (-1 if ask_decrease else 1), "fy1ca-rates-wrong-coefficient"),
    ]
    v0 = 4 / 3 * math.pi * r0**3
    surface = lambda t: 4 * math.pi * (3 * (v0 - q * t) / (4 * math.pi)) ** (2 / 3)  # noqa: E731
    h = 1e-5
    check(close((surface(h) - surface(-h)) / (2 * h), float(answer), 1e-6), "numerical rate disagrees")
    ask = (
        "at what rate is the surface area decreasing, in cm²/min"
        if ask_decrease
        else "what is the rate of change of its surface area, in cm²/min (negative if it is decreasing)"
    )
    text = phrase(
        rng,
        (
            "A spherical hailstone melts so that its volume decreases at {q} cm³/min. When its radius is {r0} cm, {ask}?",
            "As it falls, a spherical hailstone loses volume at {q} cm³/min and stays spherical. At the instant its radius is {r0} cm, {ask}?",
        ),
        q=q,
        r0=r0,
        ask=ask,
    )
    working = [
        "V = (4/3)πr³ and S = 4πr², so dV/dt = 4πr² dr/dt and dS/dt = 8πr dr/dt.",
        f"Eliminate dr/dt: dS/dt = (2/r) dV/dt. With dV/dt = -{q} and r = {r0}, dS/dt = {shown(answer)}"
        + (f", so the area is decreasing at {shown(asked)} cm²/min." if ask_decrease else " cm²/min."),
    ]
    return _question(text, asked, wrong, working)


def _hard(rng: random.Random) -> dict:
    return rng.choice([_cone_pile, _tank, _hailstone])(rng)


_TIERS = {"easy": _easy, "medium": _medium, "hard": _hard}


@generator("fy1ca_related_rates", level=19, presets={t: dict(tier=t) for t in _TIERS})
def fy1ca_related_rates(rng: random.Random, tier: str = "easy") -> dict:
    """Related rates: a standard formula (easy), a relation to set up (medium), a variable to eliminate first (hard)."""
    check(tier in _TIERS, f"unknown tier {tier!r}")
    return build_with_redraws(rng, _TIERS[tier], f"a {tier} related-rates question")


# -- exam ----------------------------------------------------------------------------------------------------------------


def _exam_build(rng: random.Random) -> dict:
    cone = rng.random() < 0.6
    h0 = rng.choice([2, 4, 5, 6]) if cone else rng.choice([1, 2, 3])
    rate = rng.choice([2, 3, 4, 5, 6])
    if cone:
        k = rng.choice([Fraction(1, 2), Fraction(3, 4), Fraction(1), Fraction(2, 3)])
        coef = math.pi * float(k) ** 2 / 3  # V = coef h^3
    else:
        k = None
        coef = 4 * math.pi / 3
    v_h = coef * h0**3
    dv_dh = 3 * coef * h0**2
    dh_dt = rate / dv_dh
    minutes = 7 * v_h / rate
    if not (
        clear_of_rounding(v_h, 2)
        and clear_of_rounding(dv_dh, 2)
        and clear_of_rounding(dh_dt, 3)
        and clear_of_rounding(minutes, 1)
    ):
        raise Redraw
    if grid(Fraction(v_h), 2) == grid(Fraction(dv_dh), 2) or min(dh_dt, minutes) < 0.01:
        raise Redraw
    # second route: integrate the volume numerically from h0 to 2*h0 (the volume added) and compare with 7 V(h0)
    n = 4000
    step = h0 / n
    added = sum(3 * coef * (h0 + (i + 0.5) * step) ** 2 for i in range(n)) * step
    check(close(added, 7 * v_h, 1e-6), "the volume added disagrees")
    check(close(v_h + 7 * v_h, coef * (2 * h0) ** 3), "doubling the size does not add 7 V")

    a_val, b_val = Fraction(v_h), Fraction(dv_dh)
    a_r, b_r = grid(a_val, 2), grid(b_val, 2)
    c_val = Fraction(dh_dt)
    d_val = Fraction(minutes)

    if cone:
        kf = Fraction(k)
        stem = phrase(
            rng,
            (
                "At a silo in the Wimmera, a conveyor builds a pile of {material} at a steady {rate} m³/min. The pile is always a cone whose base radius is {k_text}. Let h be the height of the pile in metres and V its volume in cubic metres.",
                "Gravel is tipped at a quarry at {rate} m³/min and forms a conical stockpile whose base radius is always {k_text}. Let h be the height of the stockpile in metres and V its volume in cubic metres.",
            ),
            material=rng.choice(["wheat", "barley", "canola"]),
            rate=rate,
            k_text=_RATIO_WORDS[kf],
        )
        # (a) the cone volume; wrong: forgot the third, radius = height, radius not squared, height not cubed
        a_wrong = [
            (a_val * 3, "fy1ca-rates-cone-third-forgotten"),
            (Fraction(math.pi * h0**3 / 3) if kf != 1 else a_val * 3, "fy1ca-rates-radius-equals-height"),
            (Fraction(math.pi * float(kf) * h0**3 / 3), "fy1ca-rates-ratio-not-squared"),
            (Fraction(coef * h0**2), "fy1ca-rates-power-rule-slip"),
        ]
        b_wrong = [
            (a_val, "fy1ca-rates-formula-not-differentiated"),
            (b_val / 3, "fy1ca-rates-power-rule-slip"),
            (Fraction(math.pi * h0**2), "fy1ca-rates-radius-equals-height"),
            (b_val * 2 / 3, "fy1ca-rates-power-rule-slip"),
        ]
        size_noun = "height"
        a_prompt = f"Find the volume of the pile when its height is {h0} m, correct to 2 decimal places."
        b_prompt = f"Find dV/dh when h = {h0}, correct to 2 decimal places."
    else:
        stem = phrase(
            rng,
            (
                "A spherical weather balloon is being inflated so that its volume increases at {rate} m³/min. Let r be its radius in metres and V its volume in cubic metres.",
                "A helium balloon at a weather station is filled at a steady {rate} m³/min and stays spherical. Let r be its radius in metres and V its volume in cubic metres.",
            ),
            rate=rate,
        )
        a_wrong = [
            (a_val * 3 / 4, "fy1ca-rates-wrong-coefficient"),
            (Fraction(4 * math.pi * h0**2), "fy1ca-rates-formula-confusion"),
            (Fraction(4 * math.pi * h0**3), "fy1ca-rates-sphere-third-forgotten"),
            (Fraction(4 / 3 * math.pi * h0**2), "fy1ca-rates-power-rule-slip"),
        ]
        b_wrong = [
            (a_val, "fy1ca-rates-formula-not-differentiated"),
            (b_val / 3, "fy1ca-rates-power-rule-slip"),
            (b_val * 2 / 3, "fy1ca-rates-wrong-coefficient"),
            (Fraction(8 * math.pi * h0), "fy1ca-rates-surface-differentiated"),
        ]
        size_noun = "radius"
        a_prompt = f"Find the volume of the balloon when its radius is {h0} m, correct to 2 decimal places."
        b_prompt = f"Find dV/dr when r = {h0}, correct to 2 decimal places."
    c_wrong = [
        (Fraction(rate) * b_r, "fy1ca-rates-ratio-inverted"),
        (b_r / rate, "fy1ca-rates-ratio-inverted"),
        (Fraction(rate) / a_r, "fy1ca-rates-wrong-derivative-used"),
        (Fraction(rate) / (b_r / 3), "fy1ca-rates-power-rule-slip"),
    ]
    d_wrong = [
        (a_val / rate, "fy1ca-rates-only-the-first-volume"),
        (8 * a_val / rate, "fy1ca-rates-volume-added-not-subtracted"),
        (2 * a_val / rate, "fy1ca-rates-linear-scaling"),
        (4 * a_val / rate, "fy1ca-rates-area-scaling"),
    ]
    a_dist = number_distractors(a_val, a_wrong, 2)
    b_dist = number_distractors(b_val, b_wrong, 2)
    c_dist = number_distractors(c_val, c_wrong, 3)
    d_dist = number_distractors(d_val, d_wrong, 1)
    for dist in (a_dist, b_dist, c_dist, d_dist):
        need_distractors(dist)
    var = "h" if cone else "r"
    c_prompt = f"Hence find the rate at which the {size_noun} is increasing at that moment, in m/min, correct to 3 decimal places."
    d_prompt = f"How many minutes does it take for the {size_noun} to grow from {h0} m to {2 * h0} m? Give your answer correct to 1 decimal place."
    parts = [
        number_part(
            a_prompt,
            a_val,
            2,
            2,
            [
                f"V = {'(1/3)π(' + _ratio_text(k) + 'h)²h' if cone else '(4/3)πr³'} gives V = {float(a_r):.2f} m³ when {var} = {h0}."
            ],
            a_dist,
            sensible_range=(0, None),
        ),
        number_part(
            b_prompt,
            b_val,
            2,
            2,
            [f"Differentiate V with respect to {var}: dV/d{var} = {float(b_r):.2f} when {var} = {h0}."],
            b_dist,
            sensible_range=(0, None),
        ),
        number_part(
            c_prompt,
            c_val,
            3,
            2,
            [f"d{var}/dt = (dV/dt)/(dV/d{var}) = {rate}/{float(b_r):.2f} = {float(c_val):.3f} m/min."],
            c_dist,
            follow_through=f"{rate}/part_b",
            sensible_range=(0, None),
        ),
        number_part(
            d_prompt,
            d_val,
            1,
            2,
            [
                f"The volume at {var} = {2 * h0} is 8 times the volume at {var} = {h0}, so the volume added is 7 × {float(a_r):.2f}.",
                f"Time = 7 × {float(a_r):.2f} / {rate} = {float(d_val):.1f} minutes.",
            ],
            d_dist,
            follow_through=f"7*part_a/{rate}",
            sensible_range=(0, None),
        ),
    ]
    return dict(text=stem, parts=parts)


@generator("fy1ca_related_rates_exam", level=19, presets={"exam": {}})
def fy1ca_related_rates_exam(rng: random.Random) -> dict:
    """A conical stockpile or an inflating balloon: V at a size, dV/dh there, dh/dt, and the time to double the size."""
    return build_with_redraws(rng, _exam_build, "a related-rates exam chain")
