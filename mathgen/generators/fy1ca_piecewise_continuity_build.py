"""Generates `fy1ca_piecewise_continuity` and `fy1ca_piecewise_continuity_exam` questions.

One first-year skill, ``ucv1-piecewise-continuity``: choose the constant that makes a piecewise function continuous,
and classify the discontinuity that is left when it is not. Every answer is an exact rational, graded as a fraction.

Tiers (the reasoning escalates; the numbers stay small):

- ``easy``: two pieces and one junction. One piece holds the unknown constant, and the two values at the junction are
  equated (the constant is the slope of one piece, or the constant term of one).
- ``medium``: a piece whose value at the junction is a limit, not a substitution. The function is defined by a quotient
  (a difference of squares, a factorisable quadratic, sin(mx)/x or (e^(mx) - 1)/x) away from the junction and by the
  unknown constant at it, or a quotient piece meets a linear piece.
- ``hard``: three pieces and two junctions, with the unknown constant of the middle piece fixed by the first junction
  and then needed to fix the constant of the last piece: the second constant depends on the first.
- ``exam`` (topic ``fy1ca_piecewise_continuity_exam``): a four-part chain on one function with a stated value at the
  junction: (a) the right-hand limit, (b) the slope that makes the left-hand limit match, (c) classify the
  discontinuity that remains because the stated value is not the limit, (d) evaluate the function on the left piece.

Every answer is re-derived by a second route: the one-sided limits are evaluated numerically very close to the
junction and compared with the exact value.
"""

from __future__ import annotations

import math
import random
from fractions import Fraction

from ..core import attempts, check, draw_until, fmt_signed, generator, join_terms, paren_negative, phrase
from ._fy1ca_kit import (
    build_with_redraws,
    need_distractors,
    choice_part,
    close,
    fraction_distractors,
    fraction_part,
    frac,
    ratio_text,
)
from ..answers import Distractor

_NAMES = ("f", "g", "h", "p")
_EPS = 1e-6


def _lt(rng: random.Random, strict_left: bool, c: int) -> tuple[str, str]:
    """Two region phrases for the left and right pieces: ("x < 2", "x ≥ 2") or ("x ≤ 2", "x > 2")."""
    return (f"x < {c}", f"x ≥ {c}") if strict_left else (f"x ≤ {c}", f"x > {c}")


def _poly(*terms) -> str:
    return join_terms(list(terms))


def _two_pieces(name: str, left: str, right: str, regions: tuple[str, str]) -> str:
    return f"{name}(x) = {left} for {regions[0]}, and {name}(x) = {right} for {regions[1]}"


def _quad(p: int, r: int, s: int) -> str:
    return _poly((p, "x²"), (r, "x"), (s, ""))


def _eval_quad(p, r, s, x):
    return p * x * x + r * x + s


# -- easy ------------------------------------------------------------------------------------------------------------


def _easy(rng: random.Random) -> dict:
    name = rng.choice(_NAMES)
    c = rng.choice([-3, -2, -1, 1, 2, 3, 4])
    strict = rng.random() < 0.5
    regions = _lt(rng, strict, c)
    shape = rng.choice(["slope", "constant", "slope_right"])
    if shape in ("slope", "slope_right"):
        k = rng.choice([v for v in range(-6, 7) if v != 0])
        p, r, s = rng.choice([1, 2, -1, 3]), rng.choice([-4, -2, 0, 1, 3]), rng.randint(-5, 5)
        q_c = _eval_quad(p, r, s, c)
        u = q_c - k * c
        check(k * c + u == q_c, "the junction values disagree")
        lin = _poly((1, "kx"), (u, ""))
        quad = _quad(p, r, s)
        if shape == "slope":
            text_pieces = _two_pieces(name, lin, quad, regions)
        else:
            text_pieces = _two_pieces(name, quad, lin, regions)
        answer = Fraction(k)
        wrong = [
            ((q_c + u) / Fraction(c), "fy1ca-continuity-sign-slip"),
            (Fraction(q_c - u), "fy1ca-continuity-constant-not-divided"),
            (Fraction(2 * p * c + r), "fy1ca-continuity-gradients-matched"),
            (Fraction(q_c), "fy1ca-continuity-one-piece-only"),
        ]
        working = [
            f"For {name} to be continuous at x = {c}, the two pieces must have the same value there.",
            f"The quadratic piece gives {_eval_quad(p, r, s, c)} at x = {c}; the linear piece gives {join_terms([(c, 'k'), (u, '')])}.",
            f"So {join_terms([(c, 'k'), (u, '')])} = {q_c}, which gives k = {ratio_text(answer)}.",
        ]
        left_fn = (lambda x: k * x + u) if shape == "slope" else (lambda x: _eval_quad(p, r, s, x))
        right_fn = (lambda x: _eval_quad(p, r, s, x)) if shape == "slope" else (lambda x: k * x + u)
    else:
        k = rng.choice([v for v in range(-8, 9) if v != 0])
        m = rng.choice([-3, -2, 1, 2, 3, 4])
        n = rng.randint(-6, 6)
        # left piece x² + k, right piece mx + n; continuity: c² + k = mc + n
        answer = Fraction(k)
        n = k + c * c - m * c  # choose n so that the answer is the drawn k
        check(c * c + k == m * c + n, "the junction values disagree")
        left_text = _poly((1, "x²"), (1, "k"))
        right_text = _poly((m, "x"), (n, ""))
        text_pieces = _two_pieces(name, left_text, right_text, regions)
        wrong = [
            (Fraction(m * c + n + c * c), "fy1ca-continuity-sign-slip"),
            (Fraction(m * c + n), "fy1ca-continuity-one-piece-only"),
            (Fraction(m * c + n - 2 * c), "fy1ca-continuity-squared-as-doubled"),
            (Fraction(m - 2 * c), "fy1ca-continuity-gradients-matched"),
        ]
        working = [
            f"For {name} to be continuous at x = {c}, the two pieces must have the same value there.",
            f"The right piece gives {join_terms([(1, f'{paren_negative(m)} × {paren_negative(c)}'), (n, '')])} = {m * c + n}; the left piece gives {c * c} + k.",
            f"So {c * c} + k = {m * c + n}, which gives k = {ratio_text(answer)}.",
        ]
        left_fn = lambda x: x * x + k  # noqa: E731
        right_fn = lambda x: m * x + n  # noqa: E731
    # second route: the one-sided values just either side of the junction agree only at the answer
    check(close(left_fn(c - _EPS), right_fn(c + _EPS), 1e-4), "one-sided values do not meet at the junction")
    lead = phrase(
        rng,
        (
            "The function {name} is defined piece by piece: {pieces}. Find the value of k for which {name} is continuous at x = {c}.",
            "Let {pieces}. For what value of the constant k is {name} continuous at x = {c}?",
            "A function is given by {pieces}. Determine k so that there is no break in the graph of {name} at x = {c}.",
        ),
        name=name,
        pieces=text_pieces,
        c=c,
    )
    distractors = fraction_distractors(answer, wrong)
    need_distractors(distractors)
    return dict(
        text=lead,
        answer=_exact(answer),
        answer_text=ratio_text(answer),
        answer_kind="fraction",
        answer_latex=_latex(answer),
        working=working,
        distractors=distractors,
    )


def _exact(value: Fraction):
    from ..answers import exact_value
    from sympy import Rational

    return exact_value(Rational(value.numerator, value.denominator))


def _latex(value: Fraction) -> str:
    from ..core import latex_answer

    return latex_answer(ratio_text(value))


# -- medium ------------------------------------------------------------------------------------------------------------


def _medium(rng: random.Random) -> dict:
    name = rng.choice(_NAMES)
    kind = rng.choice(["difference_of_squares", "factorised", "sine", "exponential", "quotient_meets_line"])
    if kind == "difference_of_squares":
        p = rng.choice([-5, -4, -3, -2, 2, 3, 4, 5, 6])
        answer = Fraction(2 * p)
        quotient = f"(x² - {p * p})/({_lin_factor(p)})"
        fn = lambda x: (x * x - p * p) / (x - p)  # noqa: E731
        point = p
        wrong = [
            (Fraction(0), "fy1ca-limit-by-substitution"),
            (Fraction(p), "fy1ca-limit-cancel-slip"),
            (Fraction(-2 * p), "fy1ca-continuity-sign-slip"),
            (Fraction(p * p), "fy1ca-limit-cancel-slip"),
        ]
        working = [
            f"Away from x = {p}, factorise: x² - {p * p} = (x - {paren_negative(p)})(x + {paren_negative(p)}), so the quotient equals x + {paren_negative(p)}.",
            f"The limit as x → {p} is {p} + {paren_negative(p)} = {2 * p}, so k must be {2 * p}.",
        ]
    elif kind == "factorised":
        p = rng.choice([-4, -3, -2, -1, 1, 2, 3, 4])
        s = rng.choice([v for v in range(-6, 7) if v not in (0, -p)])
        # numerator (x - p)(x + s) = x² + (s - p)x - p s ; value at p: p + s
        answer = Fraction(p + s)
        numer = _poly((1, "x²"), (s - p, "x"), (-p * s, ""))
        quotient = f"({numer})/({_lin_factor(p)})"
        fn = lambda x: (x * x + (s - p) * x - p * s) / (x - p)  # noqa: E731
        point = p
        wrong = [
            (Fraction(0), "fy1ca-limit-by-substitution"),
            (Fraction(s), "fy1ca-limit-cancel-slip"),
            (Fraction(p - s), "fy1ca-continuity-sign-slip"),
            (Fraction(-p * s), "fy1ca-limit-cancel-slip"),
        ]
        working = [
            f"The numerator factorises as ({join_terms([(1, 'x'), (-p, '')])})({join_terms([(1, 'x'), (s, '')])}), "
            f"so away from x = {p} the quotient equals {join_terms([(1, 'x'), (s, '')])}.",
            f"The limit as x → {p} is {join_terms([(p, ''), (s, '')])} = {p + s}, so k must be {p + s}.",
        ]
    elif kind == "sine":
        m = rng.choice([2, 3, 4, 5, 6, 7])
        answer = Fraction(m)
        quotient = f"sin({m}x)/x"
        fn = lambda x: math.sin(m * x) / x  # noqa: E731
        point = 0
        wrong = [
            (Fraction(1), "fy1ca-standard-limit-factor-dropped"),
            (Fraction(0), "fy1ca-limit-by-substitution"),
            (Fraction(1, m), "fy1ca-standard-limit-inverted"),
            (Fraction(m * m), "fy1ca-standard-limit-factor-dropped"),
        ]
        working = [
            f"Write sin({m}x)/x as {m} × sin({m}x)/({m}x). As x → 0, sin({m}x)/({m}x) → 1.",
            f"So the limit is {m}, and k must be {m}.",
        ]
    elif kind == "exponential":
        m = rng.choice([2, 3, 4, 5, -2, -3])
        answer = Fraction(m)
        quotient = f"(e^({m}x) - 1)/x"
        fn = lambda x: (math.exp(m * x) - 1) / x  # noqa: E731
        point = 0
        wrong = [
            (Fraction(1), "fy1ca-standard-limit-factor-dropped"),
            (Fraction(0), "fy1ca-limit-by-substitution"),
            (Fraction(1, m), "fy1ca-standard-limit-inverted"),
            (Fraction(-m), "fy1ca-continuity-sign-slip"),
        ]
        working = [
            f"The derivative of e^({m}x) at x = 0 is {m}, and (e^({m}x) - 1)/x is the difference quotient of e^({m}x) at 0.",
            f"So the limit is {m}, and k must be {m}.",
        ]
    else:
        p = rng.choice([-3, -2, 2, 3, 4, 5])
        # (x² - p²)/(x - p) for x < p ; k x + 1 for x ≥ p.  2p = kp + 1
        answer = Fraction(2 * p - 1, p)
        left = f"(x² - {p * p})/({_lin_factor(p)})"
        right = "kx + 1"
        pieces = _two_pieces(name, left, right, (f"x < {p}", f"x ≥ {p}"))
        wrong = [
            (Fraction(2 * p - 1), "fy1ca-continuity-constant-not-divided"),
            (Fraction(2 * p + 1, p), "fy1ca-continuity-sign-slip"),
            (Fraction(-1, p), "fy1ca-limit-by-substitution"),
            (Fraction(p - 1, p), "fy1ca-limit-cancel-slip"),
        ]
        working = [
            f"As x → {p} from the left, (x² - {p * p})/({_lin_factor(p)}) = {join_terms([(1, 'x'), (p, '')])} → {2 * p}.",
            f"At x = {p} the right piece is {join_terms([(p, 'k'), (1, '')])}, so {join_terms([(p, 'k'), (1, '')])} = {2 * p} and k = {ratio_text(answer)}.",
        ]
        left_fn = lambda x: (x * x - p * p) / (x - p)  # noqa: E731
        right_fn = lambda x: float(answer) * x + 1  # noqa: E731
        check(close(left_fn(p - _EPS * 10), right_fn(p), 1e-4), "one-sided values do not meet at the junction")
        lead = phrase(
            rng,
            (
                "A function is defined by {pieces}. Find the value of k for which {name} is continuous at x = {p}.",
                "Let {pieces}. For what value of k does the graph of {name} have no break at x = {p}?",
            ),
            name=name,
            pieces=pieces,
            p=p,
        )
        return _single(lead, answer, wrong, working)
    # the "defined by a quotient except at the junction" templates
    check(
        close(fn(point + _EPS * 10), float(answer), 1e-4) and close(fn(point - _EPS * 10), float(answer), 1e-4),
        "limit disagrees",
    )
    pieces = f"{name}(x) = {quotient} for x ≠ {point}, and {name}({point}) = k"
    lead = phrase(
        rng,
        (
            "The function {name} is defined by {pieces}. Find the value of k that makes {name} continuous at x = {point}.",
            "Let {pieces}. What must k be for {name} to be continuous at x = {point}?",
            "Consider {pieces}. Determine k so that the discontinuity of {name} at x = {point} is removed.",
        ),
        name=name,
        pieces=pieces,
        point=point,
    )
    return _single(lead, answer, wrong, working)


def _lin_factor(p: int) -> str:
    return f"x - {p}" if p > 0 else f"x + {-p}"


def _single(text: str, answer: Fraction, wrong, working) -> dict:
    distractors = fraction_distractors(answer, wrong)
    need_distractors(distractors)
    return dict(
        text=text,
        answer=_exact(answer),
        answer_text=ratio_text(answer),
        answer_kind="fraction",
        answer_latex=_latex(answer),
        working=working,
        distractors=distractors,
    )


# -- hard ----------------------------------------------------------------------------------------------------------------


def _hard(rng: random.Random) -> dict:
    name = rng.choice(_NAMES)
    p = rng.choice([-2, -1, 1, 2])
    q = draw_until(rng, lambda r: p + r.choice([1, 2, 3]), lambda v: v != 0, what="a second junction")
    for _ in attempts(500, "a three-piece function with a non-trivial chain"):
        a = rng.choice([v for v in range(-4, 5) if v != 0])
        b = rng.choice([v for v in range(-5, 6) if v not in (0, a)])
        u = rng.choice([v for v in range(-4, 5) if v != 0])
        z = rng.randint(-6, 6)
        w_shift = rng.randint(-6, 6)
        v = a * p * p + z - u * p
        w = a * q * q + z - b * q
        if abs(v) <= 20 and abs(w) <= 25 and w != w_shift - 99:
            break
    first = _poly((u, "x"), (v, ""))
    middle = _poly((1, "ax²"), (z, ""))
    last = _poly((1, "bx"), (w, ""))
    check(u * p + v == a * p * p + z and a * q * q + z == b * q + w, "the junction values disagree")
    pieces = (
        f"{name}(x) = {first} for x < {p}, {name}(x) = {middle} for {p} ≤ x ≤ {q}, and {name}(x) = {last} for x > {q}"
    )
    answer = Fraction(b)
    a_true = Fraction(u * p + v - z, p * p)
    check(a_true == a, "a is not recovered by the first junction")
    wrong = [
        (Fraction(q * q + z - w, q), "fy1ca-continuity-first-constant-ignored"),
        (Fraction(a * q * q + z - w), "fy1ca-continuity-constant-not-divided"),
        (Fraction(a * q * q - z - w, q), "fy1ca-continuity-sign-slip"),
        ((Fraction(u * p + v + z, p * p) * q * q + z - w) / q, "fy1ca-continuity-first-constant-slip"),
    ]
    working = [
        f"At x = {p}: {join_terms([(1, f'{paren_negative(u)} × {paren_negative(p)}'), (v, '')])} = "
        f"{join_terms([(1, f'a × {paren_negative(p)}²'), (z, '')])}, "
        f"so {u * p + v} = {join_terms([(p * p, 'a'), (z, '')])} and a = {a}.",
        f"At x = {q}: {join_terms([(1, f'{paren_negative(a)} × {paren_negative(q)}²'), (z, '')])} = "
        f"{a * q * q + z} must equal {join_terms([(1, f'b × {paren_negative(q)}'), (w, '')])}.",
        f"So {fmt_signed(q, 'b')} = {a * q * q + z - w}, giving b = {b}.",
    ]
    # second route: one-sided numerical values at both junctions
    f1 = lambda x: u * x + v  # noqa: E731
    f2 = lambda x: a * x * x + z  # noqa: E731
    f3 = lambda x: b * x + w  # noqa: E731
    check(close(f1(p - _EPS), f2(p + _EPS), 1e-4) and close(f2(q - _EPS), f3(q + _EPS), 1e-4), "junctions do not meet")
    lead = phrase(
        rng,
        (
            "A function is defined by {pieces}, where a and b are constants. If {name} is continuous for every real x, find b.",
            "The constants a and b are chosen so that {pieces} is continuous everywhere. Find the value of b.",
            "Let {pieces}, with constants a and b. Given that {name} has no break anywhere, determine b.",
        ),
        name=name,
        pieces=pieces,
    )
    return _single(lead, answer, wrong, working)


_TIERS = {"easy": _easy, "medium": _medium, "hard": _hard}


@generator(
    "fy1ca_piecewise_continuity",
    level=19,
    presets={t: dict(tier=t) for t in _TIERS},
)
def fy1ca_piecewise_continuity(rng: random.Random, tier: str = "easy") -> dict:
    """A constant that makes a piecewise function continuous: easy matches values, medium a limit, hard two junctions."""
    check(tier in _TIERS, f"unknown tier {tier!r}")
    return build_with_redraws(rng, _TIERS[tier], f"a {tier} piecewise-continuity question")


# -- exam -------------------------------------------------------------------------------------------------------------


_KINDS = ("removable", "jump", "infinite", "continuous")
_KIND_TEXT = {
    "removable": "removable (the limit exists but differs from the value)",
    "jump": "jump (the one-sided limits differ)",
    "infinite": "infinite (the function is unbounded at the point)",
    "continuous": "none (the function is continuous at the point)",
}


@generator("fy1ca_piecewise_continuity_exam", level=19, presets={"exam": {}})
def fy1ca_piecewise_continuity_exam(rng: random.Random) -> dict:
    """f(x) = ax + u (x < p), f(p) = m, f(x) = x² + v (x > p): right limit, the slope a, the type of break, a value."""
    name = rng.choice(_NAMES)
    p = rng.choice([2, 3, 4])
    for _ in attempts(500, "an exam function whose four parts each have three distinct wrong answers"):
        v = rng.choice([w for w in range(-7, 8) if w != 0])
        a = rng.choice([w for w in range(-5, 6) if w not in (0, 1)])
        limit = p * p + v
        u = limit - a * p  # the left limit matches the right limit exactly when the slope is a
        m = rng.choice([w for w in range(-8, 14) if w != limit])
        r = p - rng.choice([1, 2, 3])
        if u == 0 or abs(u) > 20 or r in (0, p) or limit in (0, m):
            continue
        a_answer = Fraction(limit)
        a_dist = fraction_distractors(
            a_answer,
            [
                (Fraction(v), "fy1ca-limit-power-dropped"),
                (Fraction(p * p - v), "fy1ca-continuity-sign-slip"),
                (Fraction(2 * p + v), "squared-as-doubled"),
                (Fraction(m), "fy1ca-limit-equals-value"),
            ],
        )
        b_answer = Fraction(limit - u, p)
        b_dist = fraction_distractors(
            b_answer,
            [
                (Fraction(limit + u, p), "fy1ca-continuity-sign-slip"),
                (Fraction(limit - u), "fy1ca-continuity-constant-not-divided"),
                (Fraction(m - u, p), "fy1ca-limit-equals-value"),
                (Fraction(2 * p), "fy1ca-continuity-gradients-matched"),
            ],
        )
        d_answer = Fraction(a * r + u)
        d_dist = fraction_distractors(
            d_answer,
            [
                (Fraction(r * r + v), "fy1ca-continuity-wrong-piece"),
                (Fraction(a * r), "fy1ca-continuity-constant-dropped"),
                (Fraction(a * r - u), "fy1ca-continuity-sign-slip"),
                (Fraction(a * p + u), "fy1ca-continuity-value-at-junction"),
            ],
        )
        if min(len(a_dist), len(b_dist), len(d_dist)) >= 3:
            break
    # second route: one-sided numerical values just either side of the junction meet only for the true slope
    check(close(a * (p - _EPS) + u, (p + _EPS) ** 2 + v, 1e-4), "one-sided values do not meet for the true a")
    check(b_answer == a and limit != m, "the slope is not recovered, or the stated value equals the limit")

    left_text = _poly((1, "ax"), (u, ""))
    right_text = _poly((1, "x²"), (v, ""))
    stem = (
        f"Let {name}(x) = {left_text} for x < {p}, {name}({p}) = {m}, and {name}(x) = {right_text} for x > {p}, "
        f"where a is a constant."
    )
    options = list(_KINDS)
    rng.shuffle(options)
    c_dist = [
        Distractor(_KIND_TEXT["jump"], "fy1ca-removable-called-jump"),
        Distractor(_KIND_TEXT["continuous"], "fy1ca-removable-called-continuous"),
        Distractor(_KIND_TEXT["infinite"], "fy1ca-removable-called-infinite"),
    ]
    c_prompt = (
        f"Using the value of a from part (b), what type of discontinuity does {name} have at x = {p}? "
        f"Choose one of: {'; '.join(_KIND_TEXT[k] for k in options)}."
    )
    parts = [
        fraction_part(
            f"Find the limit of {name}(x) as x approaches {p} from the right.",
            a_answer,
            2,
            [
                f"For x > {p}, {name}(x) = x² + {paren_negative(v)}, so the limit is {p}² + {paren_negative(v)} = {limit}."
            ],
            a_dist,
            sensible_range=(None, None),
        ),
        fraction_part(
            f"Hence find the value of a for which the limit of {name}(x) from the left equals the limit from the right.",
            b_answer,
            2,
            [
                f"The left limit is {p}a + {paren_negative(u)}, and it must equal {limit}, so a = {ratio_text(b_answer)}."
            ],
            b_dist,
            follow_through=f"(part_a - ({u}))/{p}",
            sensible_range=(None, None),
        ),
        choice_part(
            c_prompt,
            _KIND_TEXT["removable"],
            c_dist,
            1,
            [
                f"Both one-sided limits are {limit}, so the limit exists, but {name}({p}) = {m} is a different number: removable."
            ],
        ),
        fraction_part(
            f"Using your value of a, evaluate {name}({r}).",
            d_answer,
            1,
            [
                f"Since {r} < {p}, use the left piece: {name}({r}) = {paren_negative(a)}({paren_negative(r)}) + {paren_negative(u)} = {a * r + u}."
            ],
            d_dist,
            follow_through=f"part_b*({r}) + ({u})",
            sensible_range=(None, None),
        ),
    ]
    return dict(text=stem, parts=parts)
