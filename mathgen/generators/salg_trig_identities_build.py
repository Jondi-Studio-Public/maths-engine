"""Generates `salg_trig_identities` questions (easy, medium, hard) and the exam chain `salg_exam_trig_identities`.

Skill ``reciprocal-trig-identities`` (``trigonometry``, answer ``expression``; fraction and number answers are read by
the expression grader too): the identities 1 + tan²θ = sec²θ and 1 + cot²θ = cosec²θ, used to find a reciprocal
function from a tangent or cotangent, to solve an equation and to simplify.

* easy: ``tan θ = p/q`` (or ``cot θ = p/q``) with θ acute and p, q the legs of a Pythagorean triple; find sec θ (or
  cosec θ). The root is positive and rational.
* medium: ``sec θ = ±h/q`` (or ``cosec θ = ±h/p``) with θ in a stated quadrant outside the first; find tan θ (or
  cot θ). The identity gives the size and the quadrant gives the sign.
* hard: either an equation ``a sec²θ - b tan²θ = c`` (or ``a cosec²θ - b cot²θ = c``) whose acute solution is a standard
  angle, solved by replacing sec² with 1 + tan²; or an expression such as ``3(sec²x - 1)cos²x`` to simplify to a single
  term.
* exam (``salg_exam_trig_identities``): ``tan θ = ±p/q`` with θ in a stated quadrant; (a) sec²θ; (b) sec θ, using (a);
  (c) cos θ, using (b); (d) sin θ, using (c) and tan θ.

Every key is checked a second way: the legs and hypotenuse give the ratio directly, a float evaluation at the angle
the ratio defines confirms every value and sign, and each simplification is compared with its stem at several points.
"""

from __future__ import annotations

import math
from fractions import Fraction

import sympy as sp

from ..answers import Distractor, exact_value
from ..core import check, draw_until, generator, latex_answer, phrase, pick_distractors, rational_text
from ._salg_kit import frac_dist, shown, typed

_TRIPLES = [(3, 4, 5), (5, 12, 13), (8, 15, 17), (7, 24, 25), (20, 21, 29)]
_QUADRANTS = {
    1: ("0° < θ < 90°", 1, 1),
    2: ("90° < θ < 180°", 1, -1),
    3: ("180° < θ < 270°", -1, -1),
    4: ("270° < θ < 360°", -1, 1),
}  # text, sign of sin, sign of cos


def _frac_answer(value: Fraction, *, kind: str = "fraction", **extra):
    """A rational answer. A single question under this skill's expression grader keeps the fraction as text
    ("5/3"), where a part of an exam chain is a `fraction` part with the exact value."""
    return dict(
        answer_kind=kind,
        answer=exact_value(value) if kind == "fraction" else rational_text(value),
        answer_text=rational_text(value),
        answer_latex=latex_answer(sp.Rational(value.numerator, value.denominator)),
        **extra,
    )


def _pick(answer: Fraction, cands: list[tuple[Fraction, str]], count=3):
    seen = []
    out = []
    for value, tag in cands:
        value = Fraction(value)
        if value == answer or value in seen or value == 0:
            continue
        seen.append(value)
        out.append(frac_dist(value, tag))
    picked = pick_distractors(rational_text(answer), out, count)
    check(len(picked) >= 2, f"too few distractors for {answer}: {cands}")
    return picked


def _ratio_text(n: int, d: int, sign: int = 1) -> str:
    return f"{'-' if sign < 0 else ''}{n}/{d}"


# -- easy --------------------------------------------------------------------------------------------------------


def _easy(rng):
    a, b, h = rng.choice(_TRIPLES)
    if rng.random() < 0.5:
        a, b = b, a
    use_cot = rng.random() < 0.5
    given, wanted = ("cot", "cosec") if use_cot else ("tan", "sec")
    answer = Fraction(h, b)
    # Independent route: the angle the ratio defines, evaluated in floats.
    theta = math.atan2(a, b)
    float_value = 1 / (math.cos(theta) if not use_cot else math.sin(theta))
    if use_cot:
        theta = math.atan2(b, a)  # cot = a/b means tan = b/a
        float_value = 1 / math.sin(theta)
    check(abs(float_value - float(answer)) < 1e-9, "float evaluation differs")
    cands = [
        (Fraction(h, b) ** 2, "recip-identity-root-not-taken"),
        (Fraction(a + b, b), "recip-identity-tan-not-squared"),
        (Fraction(h, a), "recip-sec-cosec-swapped"),
        (Fraction(b, h), "recip-not-taken"),
    ]
    stem = phrase(
        rng,
        (
            "Given that {g} θ = {a}/{b} and θ is acute, find the exact value of {w} θ.",
            "The angle θ is acute and {g} θ = {a}/{b}. Find {w} θ.",
            "For an acute angle θ with {g} θ = {a}/{b}, find the value of {w} θ.",
        ),
        g=given,
        w=wanted,
        a=a,
        b=b,
    )
    sq = "sec²" if not use_cot else "cosec²"
    return dict(
        text=stem,
        **_frac_answer(
            answer,
            kind="expression",
            working=[
                f"{sq}θ = 1 + {given}²θ = 1 + ({a}/{b})² = {h * h}/{b * b}.",
                f"θ is acute, so {wanted} θ is positive: {wanted} θ = {h}/{b}.",
            ],
            distractors=_pick(answer, cands),
        ),
    )


# -- medium -------------------------------------------------------------------------------------------------------


def _medium(rng):
    a, b, h = rng.choice(_TRIPLES)
    if rng.random() < 0.5:
        a, b = b, a
    quadrant = rng.choice([2, 3, 4])
    text, sin_sign, cos_sign = _QUADRANTS[quadrant]
    use_cosec = rng.random() < 0.5
    if not use_cosec:
        # given sec = ±h/b (b = |adjacent|), find tan = sin/cos = ±a/b
        given_name, wanted_name = "sec", "tan"
        given_val = Fraction(cos_sign * h, b)
        answer = Fraction(sin_sign * cos_sign * a, b)
        theta = math.atan2(sin_sign * a, cos_sign * b)
        check(abs(1 / math.cos(theta) - float(given_val)) < 1e-9, "float sec differs")
        check(abs(math.tan(theta) - float(answer)) < 1e-9, "float tan differs")
        wrong_mag = Fraction(a, b)
        cands = [
            (-answer, "trig-sign-wrong-quadrant" if answer > 0 else "trig-quadrant-sign-ignored"),
            (answer * answer, "recip-identity-root-not-taken"),
            (1 / answer, "recip-cot-as-tan"),
            (given_val, "recip-identity-given-repeated"),
            (Fraction(sin_sign * a, h), "recip-sec-cosec-swapped"),
        ]
        sq = "tan²θ = sec²θ - 1"
    else:
        # given cosec = ±h/a (a = |opposite|), find cot = cos/sin = ±b/a
        given_name, wanted_name = "cosec", "cot"
        given_val = Fraction(sin_sign * h, a)
        answer = Fraction(sin_sign * cos_sign * b, a)
        theta = math.atan2(sin_sign * a, cos_sign * b)
        check(abs(1 / math.sin(theta) - float(given_val)) < 1e-9, "float cosec differs")
        check(abs(math.cos(theta) / math.sin(theta) - float(answer)) < 1e-9, "float cot differs")
        cands = [
            (-answer, "trig-sign-wrong-quadrant" if answer > 0 else "trig-quadrant-sign-ignored"),
            (answer * answer, "recip-identity-root-not-taken"),
            (1 / answer, "recip-cot-as-tan"),
            (given_val, "recip-identity-given-repeated"),
            (Fraction(cos_sign * b, h), "recip-sec-cosec-swapped"),
        ]
        sq = "cot²θ = cosec²θ - 1"
    stem = phrase(
        rng,
        (
            "Given that {g} θ = {v} and {q}, find the exact value of {w} θ.",
            "The angle θ satisfies {g} θ = {v} and {q}. Find {w} θ.",
        ),
        g=given_name,
        v=rational_text(sp.Rational(given_val.numerator, given_val.denominator)),
        q=text,
        w=wanted_name,
    )
    size = Fraction(abs(answer.numerator), answer.denominator)
    return dict(
        text=stem,
        **_frac_answer(
            answer,
            kind="expression",
            working=[
                f"{sq} = {rational_text(sp.Rational((given_val**2).numerator, (given_val**2).denominator))} - 1 = {rational_text(sp.Rational((answer**2).numerator, (answer**2).denominator))}, so |{wanted_name} θ| = {rational_text(sp.Rational(size.numerator, size.denominator))}.",
                f"In the quadrant {text}, {wanted_name} is {'positive' if answer > 0 else 'negative'}, so {wanted_name} θ = {rational_text(sp.Rational(answer.numerator, answer.denominator))}.",
            ],
            distractors=_pick(answer, cands),
        ),
    )


# -- hard: solve --------------------------------------------------------------------------------------------------

_T2 = {Fraction(1, 3): 30, Fraction(1): 45, Fraction(3): 60}  # tan² of the acute standard angles


def _wrong_angles(a, b, c, use_cot, angle):
    """(angle, error id) for the routes that go wrong: sec² read as tan² - 1, the identity not applied, and the
    complementary angle."""
    out = []
    for value, tag in (
        (Fraction(c + a, a - b), "recip-identity-sign-slip"),
        (Fraction(c, a - b), "recip-identity-not-applied"),
    ):
        if value in _T2:
            out.append((_T2[value] if not use_cot else 90 - _T2[value], tag))
    out.append((90 - angle, "trig-wrong-reference-angle"))
    out.extend((v, "trig-standard-angle-misread") for v in (30, 45, 60))
    seen = {angle}
    kept = []
    for v, t in out:
        if v not in seen:
            seen.add(v)
            kept.append((v, t))
    return kept


def _solve(rng):
    use_cot = rng.random() < 0.5

    def draw(r):
        target = r.choice(list(_T2))  # tan² (or cot²) of the acute angle's tan (or cot)
        a = r.randint(2, 6)
        d = r.choice([-6, -4, -3, -2, 2, 3, 4, 6, 8])
        return target, a, a - d, a + d * target

    def accept(t):
        target, a, b, c = t
        angle = _T2[target] if not use_cot else 90 - _T2[target]
        return b > 0 and b != a and c.denominator == 1 and c > 0

    target, a, b, c = draw_until(
        rng, draw, accept, tries=3000, what="coefficients with an integer right-hand side and two wrong angles"
    )
    c = int(c)
    angle = _T2[target] if not use_cot else 90 - _T2[target]
    cands = _wrong_angles(a, b, c, use_cot, angle)
    sec, tan = ("cosec", "cot") if use_cot else ("sec", "tan")
    # Independent route: substitute the angle back into the equation, in floats.
    r = math.radians(angle)
    s_val = 1 / math.sin(r) if use_cot else 1 / math.cos(r)
    t_val = 1 / math.tan(r) if use_cot else math.tan(r)
    check(abs(a * s_val**2 - b * t_val**2 - c) < 1e-9, "the angle does not satisfy the equation")
    distractors = pick_distractors(str(angle), [Distractor(str(v), t, v) for v, t in cands], 3)
    stem = phrase(
        rng,
        (
            "Find the acute angle θ, in degrees, that satisfies {a}{s}²θ - {b}{t}²θ = {c}.",
            "Solve {a}{s}²θ - {b}{t}²θ = {c} for the acute angle θ, in degrees.",
        ),
        a="" if a == 1 else a,
        b="" if b == 1 else b,
        c=c,
        s=sec,
        t=tan,
    )
    sign = "+" if a >= b else "-"
    gap = "" if abs(a - b) == 1 else abs(a - b)
    return dict(
        text=stem,
        answer_kind="number",
        answer=angle,
        answer_text=str(angle),
        answer_latex=str(angle),
        working=[
            f"Replace {sec}²θ using {sec}²θ = 1 + {tan}²θ: {a} {sign} {gap}{tan}²θ = {c}.",
            f"So {tan}²θ = {rational_text(sp.Rational(target.numerator, target.denominator))}, and for an acute angle θ = {angle}°.",
        ],
        distractors=distractors,
    )


# -- hard: simplify -----------------------------------------------------------------------------------------------

_X = sp.Symbol("x")


def _sec(e):
    return 1 / sp.cos(e)


def _cosec(e):
    return 1 / sp.sin(e)


def _same_function(left, right) -> bool:
    """Whether two expressions in x agree at five generic floats."""
    for v in (0.31, 0.77, 1.13, 1.42, 2.05):
        if abs(float(left.subs(_X, v)) - float(right.subs(_X, v))) > 1e-8:
            return False
    return True


def _families(k: int, m: int):
    """(stem text, stem expression, answer expression, answer text, [(wrong expression, wrong text, error id)])."""
    x = _X
    s2, c2 = sp.sin(x) ** 2, sp.cos(x) ** 2
    return [
        (
            f"{k}(sec²x - 1)cos²x",
            k * (_sec(x) ** 2 - 1) * c2,
            k * s2,
            f"{k}*sin(x)^2",
            [
                (-k * s2, f"-{k}*sin(x)^2", "recip-identity-sign-slip"),
                (k * sp.tan(x) ** 2, f"{k}*tan(x)^2", "recip-identity-factor-left"),
                (k * c2, f"{k}*cos(x)^2", "recip-sec-cosec-swapped"),
            ],
        ),
        (
            f"{k}(cosec²x - 1)sin²x",
            k * (_cosec(x) ** 2 - 1) * s2,
            k * c2,
            f"{k}*cos(x)^2",
            [
                (-k * c2, f"-{k}*cos(x)^2", "recip-identity-sign-slip"),
                (k * sp.cot(x) ** 2, f"{k}*cot(x)^2", "recip-identity-factor-left"),
                (k * s2, f"{k}*sin(x)^2", "recip-sec-cosec-swapped"),
            ],
        ),
        (
            f"{k}(1 + tan²x)cos x",
            k * (1 + sp.tan(x) ** 2) * sp.cos(x),
            k * _sec(x),
            f"{k}/cos(x)",
            [
                (k * sp.cos(x), f"{k}*cos(x)", "recip-not-taken"),
                (k * _sec(x) ** 2, f"{k}/cos(x)^2", "recip-identity-factor-left"),
                (k * _cosec(x), f"{k}/sin(x)", "recip-sec-cosec-swapped"),
            ],
        ),
        (
            f"{k}(1 + cot²x)sin x",
            k * (1 + sp.cot(x) ** 2) * sp.sin(x),
            k * _cosec(x),
            f"{k}/sin(x)",
            [
                (k * sp.sin(x), f"{k}*sin(x)", "recip-not-taken"),
                (k * _cosec(x) ** 2, f"{k}/sin(x)^2", "recip-identity-factor-left"),
                (k * _sec(x), f"{k}/cos(x)", "recip-sec-cosec-swapped"),
            ],
        ),
        (
            f"{k}tan²x + {m} - {k}sec²x",
            k * sp.tan(x) ** 2 + m - k * _sec(x) ** 2,
            sp.Integer(m - k),
            str(m - k),
            [
                (sp.Integer(m + k), str(m + k), "recip-identity-sign-slip"),
                (sp.Integer(m), str(m), "recip-identity-difference-as-zero"),
                (sp.Integer(k - m), str(k - m), "recip-identity-sign-slip"),
            ],
        ),
        (
            f"{k}(sec x - 1)(sec x + 1)/tan x",
            k * (_sec(x) - 1) * (_sec(x) + 1) / sp.tan(x),
            k * sp.tan(x),
            f"{k}*tan(x)",
            [
                (k * sp.tan(x) ** 2, f"{k}*tan(x)^2", "recip-identity-factor-left"),
                (k * sp.cot(x), f"{k}/tan(x)", "recip-cot-as-tan"),
                (-k * sp.tan(x), f"-{k}*tan(x)", "recip-identity-sign-slip"),
            ],
        ),
    ]


def _simplify(rng):
    k = rng.choice([2, 3, 4, 5])
    m = rng.choice([v for v in range(2, 9) if v != k])
    stem_text, stem_expr, answer, answer_text, wrongs = rng.choice(_families(k, m))
    check(_same_function(stem_expr, answer), f"{stem_text} does not simplify to {answer_text}")
    dist = []
    for expr, text, tag in wrongs:
        if _same_function(expr, answer):
            continue
        dist.append(Distractor(text, tag, text))
    picked = pick_distractors(answer_text, dist, 3)
    check(len(picked) >= 2, "too few simplification distractors")
    stem = phrase(rng, ("Simplify {e}.", "Write {e} as a single term."), e=stem_text)
    if "tan²x + " in stem_text:
        stem = f"Simplify {stem_text}."
    return dict(
        text=stem,
        answer_kind="expression",
        answer=answer_text,
        answer_text=answer_text,
        answer_latex=latex_answer(answer),
        working=[f"Using 1 + tan²x = sec²x and 1 + cot²x = cosec²x, {stem_text} = {shown(answer_text)}."],
        distractors=picked,
    )


def _hard(rng):
    return _solve(rng) if rng.random() < 0.5 else _simplify(rng)


@generator(
    "salg_trig_identities",
    level=12,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def salg_trig_identities(rng, tier="easy"):
    """1 + tan² = sec² and 1 + cot² = cosec² used to find, solve and simplify; see the module docstring."""
    if tier == "easy":
        return _easy(rng)
    if tier == "medium":
        return _medium(rng)
    check(tier == "hard", f"unknown tier {tier!r}")
    return _hard(rng)


# -- exam -------------------------------------------------------------------------------------------------------


@generator("salg_exam_trig_identities", level=12, presets={"exam": dict(tier="exam")})
def salg_exam_trig_identities(rng, tier="exam"):
    """Exam chain: sec²θ, sec θ, cos θ and sin θ from tan θ and the quadrant of θ."""
    check(tier == "exam", f"unknown tier {tier!r}")
    a, b, h = rng.choice(_TRIPLES)
    if rng.random() < 0.5:
        a, b = b, a
    quadrant = rng.choice([1, 2, 3, 4])
    q_text, sin_sign, cos_sign = _QUADRANTS[quadrant]
    tan_v = Fraction(sin_sign * cos_sign * a, b)
    sec2 = 1 + tan_v**2
    sec_v = Fraction(cos_sign * h, b)
    cos_v = Fraction(cos_sign * b, h)
    sin_v = Fraction(sin_sign * a, h)
    theta = math.atan2(sin_sign * a, cos_sign * b)
    check(abs(1 / math.cos(theta) ** 2 - float(sec2)) < 1e-9, "float sec² differs")
    check(abs(1 / math.cos(theta) - float(sec_v)) < 1e-9, "float sec differs")
    check(abs(math.cos(theta) - float(cos_v)) < 1e-9 and abs(math.sin(theta) - float(sin_v)) < 1e-9, "float sin/cos")
    check(sec_v * sec_v == sec2 and cos_v * sec_v == 1 and cos_v * tan_v == sin_v, "chain is inconsistent")
    tan_text = rational_text(sp.Rational(tan_v.numerator, tan_v.denominator))

    a_cands = [
        (tan_v**2, "recip-identity-one-omitted"),
        (1 - tan_v**2, "recip-identity-sign-slip"),
        (tan_v**2 - 1, "recip-identity-sign-slip"),
        ((1 + abs(tan_v)) ** 2, "recip-identity-tan-not-squared"),
    ]
    b_cands = [
        (-sec_v, "trig-sign-wrong-quadrant" if sec_v > 0 else "trig-quadrant-sign-ignored"),
        (sec2, "recip-identity-root-not-taken"),
        (Fraction(cos_sign * h, a), "recip-sec-cosec-swapped"),
        (1 + abs(tan_v) if cos_sign > 0 else -(1 + abs(tan_v)), "recip-identity-tan-not-squared"),
    ]
    c_cands = [
        (sec_v, "recip-not-taken"),
        (-cos_v, "trig-sign-wrong-quadrant" if cos_v > 0 else "trig-quadrant-sign-ignored"),
        (sin_v, "trig-sin-cos-swapped"),
        (Fraction(cos_sign * a, h), "recip-sec-cosec-swapped"),
    ]
    d_cands = [
        (cos_v, "recip-tan-not-applied"),
        (-sin_v, "trig-sign-wrong-quadrant" if sin_v > 0 else "trig-quadrant-sign-ignored"),
        (cos_v / tan_v, "recip-tan-divided"),
        (Fraction(sin_sign * h, a), "recip-sec-cosec-swapped"),
    ]

    def frac_part(prompt, key, marks, cands, working, ft=None, rng_=None):
        p = dict(prompt=prompt, marks=marks, working=working, distractors=_pick(key, cands))
        p.update(_frac_answer(key))
        if ft:
            p["follow_through"] = ft
        if rng_:
            p["sensible_range"] = rng_
        return p

    sign = "sqrt(part_a)" if cos_sign > 0 else "-sqrt(part_a)"
    return dict(
        text=f"Given that tan θ = {tan_text} and {q_text}.",
        parts=[
            frac_part(
                "Find the exact value of sec²θ.",
                sec2,
                1,
                a_cands,
                [
                    f"sec²θ = 1 + tan²θ = 1 + ({tan_text})² = {rational_text(sp.Rational(sec2.numerator, sec2.denominator))}."
                ],
                rng_=(1, None),
            ),
            frac_part(
                "Hence find the exact value of sec θ.",
                sec_v,
                2,
                b_cands,
                [
                    f"sec θ = ±sqrt({rational_text(sp.Rational(sec2.numerator, sec2.denominator))}).",
                    f"In the quadrant {q_text}, cos θ is {'positive' if cos_sign > 0 else 'negative'}, so sec θ = {rational_text(sp.Rational(sec_v.numerator, sec_v.denominator))}.",
                ],
                ft=sign,
                rng_=(1, None) if cos_sign > 0 else (None, -1),
            ),
            frac_part(
                "Hence find the exact value of cos θ.",
                cos_v,
                1,
                c_cands,
                [f"cos θ = 1/sec θ = {rational_text(sp.Rational(cos_v.numerator, cos_v.denominator))}."],
                ft="1/part_b",
            ),
            frac_part(
                "Hence find the exact value of sin θ.",
                sin_v,
                2,
                d_cands,
                [
                    f"sin θ = tan θ × cos θ = ({tan_text}) × ({rational_text(sp.Rational(cos_v.numerator, cos_v.denominator))}) = {rational_text(sp.Rational(sin_v.numerator, sin_v.denominator))}."
                ],
                ft=f"({tan_text})*part_c",
            ),
        ],
    )
