"""Generates `r3g_product_sum` questions (easy, medium, hard) and the exam chain `r3g_exam_product_sum`.

Skill ``product-to-sum-evaluate-solve`` (``trigonometry``, answer ``expression``): using the product-to-sum
and sum-to-product identities to evaluate exactly and to solve. The proof skills ask for the identities themselves;
this one uses them.

* easy: the exact value of ``2 sin A cos B``, ``2 cos A cos B`` or ``2 sin A sin B`` where A + B and A - B are standard
  angles, so the product is a sum or difference of two known values.
* medium: a product with no leading 2 (so the sum is halved), or a sum or difference such as ``sin 75° + sin 15°`` whose
  half-sum and half-difference are standard angles.
* hard: solve (and report the largest, second smallest or the sum of the solutions as an exact multiple of pi)
  ``2 sin px cos qx = sin (p - q)x`` or ``2 cos px cos qx = cos (p - q)x`` over 0 ≤ x < π (the sum form
  leaves one term), or ``sin px + sin qx = 0``, which factorises into a sine and a cosine and needs both families of
  solutions.
* exam (``r3g_exam_product_sum``): for ``2 sin px cos qx = sin (p - q)x``: (a) m in ``sin mx = 0``, (b) the smallest
  positive solution, (c) the number of solutions in 0 ≤ x < 2π, (d) the sum of the solutions in 0 ≤ x < π.

Values are computed with SymPy from the stated product and from the sum form, and every solution set is confirmed by a
float scan of the interval at multiples of pi/(4pq).
"""

from __future__ import annotations

import math
from fractions import Fraction

import sympy as sp

from ..core import attempts, check, generator
from ._r3g_kit import (
    vary_symbol,
    ex as _ex,
)
from ._r3g_kit import (
    TooFewDistractors,
    int_dist,
    latex,
    pi_dist,
    pi_latex,
    pi_shown,
    pi_text,
    pick,
    retry_short,
    typed,
)

F = Fraction
R = sp.Rational
_FN = {"sin": sp.sin, "cos": sp.cos}
_FMATH = {"sin": math.sin, "cos": math.cos}


def clean(expr):
    return sp.expand(sp.radsimp(sp.expand(sp.sympify(expr))))


def ex(expr, tag):
    return _ex(clean(expr), tag)


def at(fn: str, deg):
    """`fn` of an angle in degrees, exactly."""
    return _FN[fn](sp.pi * R(deg, 180))


def fl(fn: str, deg) -> float:
    return _FMATH[fn](math.radians(float(deg)))


_STANDARD = (0, 30, 45, 60, 90, 120, 135, 150, 180)


def _ask(rng, label: str) -> str:
    """The request to evaluate `label`, worded one of three ways."""
    return [
        f"Find the exact value of {label}. Type surds as sqrt(3), for example.",
        f"Evaluate {label} exactly, without a calculator. Type surds as sqrt(3), for example.",
        f"What is the exact value of {label}? Type surds as sqrt(3), for example.",
    ][rng.randrange(3)]


def _easy(rng):
    for _ in attempts(500, "a product whose sum and difference angles are standard"):
        s, d = rng.choice(_STANDARD), rng.choice(_STANDARD)
        if s <= d or (s + d) % 2 or (s - d) % 2:
            continue
        a, b = (s + d) // 2, (s - d) // 2
        if b == 0 or 90 in (a, b):
            continue
        kind = rng.choice(("sc", "cc", "ss"))
        break
    if kind == "sc":
        label, direct, via = f"2 sin {a}° cos {b}°", 2 * at("sin", a) * at("cos", b), at("sin", s) + at("sin", d)
        float_value = 2 * fl("sin", a) * fl("cos", b)
        wrong = [
            (at("sin", s) - at("sin", d), "r3g-product-sum-sign-slip"),
            ((at("sin", s) + at("sin", d)) / 2, "r3g-product-sum-factor-handled-wrongly"),
            (at("cos", s) + at("cos", d), "r3g-product-sum-wrong-function"),
            (at("sin", a + b) * at("cos", a - b), "r3g-product-sum-angles-not-added"),
        ]
        step = f"2 sin A cos B = sin(A + B) + sin(A - B) = sin {s}° + sin {d}°"
    elif kind == "cc":
        label, direct, via = f"2 cos {a}° cos {b}°", 2 * at("cos", a) * at("cos", b), at("cos", s) + at("cos", d)
        float_value = 2 * fl("cos", a) * fl("cos", b)
        wrong = [
            (at("cos", s) - at("cos", d), "r3g-product-sum-sign-slip"),
            ((at("cos", s) + at("cos", d)) / 2, "r3g-product-sum-factor-handled-wrongly"),
            (at("sin", s) + at("sin", d), "r3g-product-sum-wrong-function"),
            (at("cos", a + b) * at("cos", a - b), "r3g-product-sum-angles-not-added"),
        ]
        step = f"2 cos A cos B = cos(A + B) + cos(A - B) = cos {s}° + cos {d}°"
    else:
        label, direct, via = f"2 sin {a}° sin {b}°", 2 * at("sin", a) * at("sin", b), at("cos", d) - at("cos", s)
        float_value = 2 * fl("sin", a) * fl("sin", b)
        wrong = [
            (at("cos", s) - at("cos", d), "r3g-product-sum-sign-slip"),
            (at("cos", s) + at("cos", d), "r3g-product-sum-sign-slip"),
            ((at("cos", d) - at("cos", s)) / 2, "r3g-product-sum-factor-handled-wrongly"),
            (at("sin", s) - at("sin", d), "r3g-product-sum-wrong-function"),
        ]
        step = f"2 sin A sin B = cos(A - B) - cos(A + B) = cos {d}° - cos {s}°"
    check(sp.simplify(direct - via) == 0, "product differs from its sum form")
    check(abs(float(via) - float_value) < 1e-9, "sum form differs from the float product")
    text = typed(clean(via))
    return dict(
        text=_ask(rng, label),
        answer_kind="expression",
        answer=text,
        answer_text=text,
        answer_latex=latex(clean(via)),
        working=[f"Here A + B = {s}° and A - B = {d}°.", f"{step} = {text}."],
        distractors=pick(text, [ex(v, tag) for v, tag in wrong], 3),
    )


def _medium_product(rng):
    for _ in attempts(500, "a product with standard sum and difference angles"):
        s, d = rng.choice(_STANDARD), rng.choice(_STANDARD)
        if s <= d or (s + d) % 2 or (s - d) % 2:
            continue
        a, b = (s + d) // 2, (s - d) // 2
        if b == 0 or 90 in (a, b):
            continue
        break
    kind = rng.choice(("sc", "cs", "cc", "ss"))
    if kind == "sc":
        direct, via, label = at("sin", a) * at("cos", b), (at("sin", s) + at("sin", d)) / 2, f"sin {a}° cos {b}°"
        fv = fl("sin", a) * fl("cos", b)
        wrong = [at("sin", s) + at("sin", d), (at("sin", s) - at("sin", d)) / 2, (at("cos", s) + at("cos", d)) / 2]
    elif kind == "cs":
        direct, via, label = at("cos", a) * at("sin", b), (at("sin", s) - at("sin", d)) / 2, f"cos {a}° sin {b}°"
        fv = fl("cos", a) * fl("sin", b)
        wrong = [at("sin", s) - at("sin", d), (at("sin", s) + at("sin", d)) / 2, (at("sin", d) - at("sin", s)) / 2]
    elif kind == "cc":
        direct, via, label = at("cos", a) * at("cos", b), (at("cos", s) + at("cos", d)) / 2, f"cos {a}° cos {b}°"
        fv = fl("cos", a) * fl("cos", b)
        wrong = [at("cos", s) + at("cos", d), (at("cos", s) - at("cos", d)) / 2, (at("sin", s) + at("sin", d)) / 2]
    else:
        direct, via, label = at("sin", a) * at("sin", b), (at("cos", d) - at("cos", s)) / 2, f"sin {a}° sin {b}°"
        fv = fl("sin", a) * fl("sin", b)
        wrong = [at("cos", d) - at("cos", s), (at("cos", s) - at("cos", d)) / 2, (at("cos", d) + at("cos", s)) / 2]
    check(sp.simplify(direct - via) == 0, "product differs from its sum form")
    check(abs(float(via) - fv) < 1e-9, "sum form differs from the float product")
    tags = ["r3g-product-sum-factor-handled-wrongly", "r3g-product-sum-sign-slip", "r3g-product-sum-wrong-function"]
    text = typed(clean(via))
    return dict(
        text=_ask(rng, label),
        answer_kind="expression",
        answer=text,
        answer_text=text,
        answer_latex=latex(clean(via)),
        working=[
            f"With A = {a}° and B = {b}°, A + B = {s}° and A - B = {d}°.",
            f"The product is half of the sum form: {text}.",
        ],
        distractors=pick(text, [ex(v, t) for v, t in zip(wrong, tags)], 3),
    )


def _medium_sum(rng):
    for _ in attempts(500, "a sum with standard half angles"):
        p, q = rng.choice((30, 45, 60)), rng.choice((15, 30, 45))
        if p <= q or 90 in (p + q, p - q):
            continue
        a, b = p + q, p - q
        break
    kind = rng.choice(("ss+", "ss-", "cc+", "cc-"))
    fn = kind[0] == "s" and "sin" or "cos"
    plus = kind[2] == "+"
    if kind == "ss+":
        direct, via = at("sin", a) + at("sin", b), 2 * at("sin", p) * at("cos", q)
        label = f"sin {a}° + sin {b}°"
        fv = fl("sin", a) + fl("sin", b)
        wrong = [2 * at("cos", p) * at("sin", q), 2 * at("sin", p) * at("sin", q), at("sin", p) * at("cos", q)]
    elif kind == "ss-":
        direct, via = at("sin", a) - at("sin", b), 2 * at("cos", p) * at("sin", q)
        label = f"sin {a}° - sin {b}°"
        fv = fl("sin", a) - fl("sin", b)
        wrong = [2 * at("sin", p) * at("cos", q), -2 * at("cos", p) * at("sin", q), at("cos", p) * at("sin", q)]
    elif kind == "cc+":
        direct, via = at("cos", a) + at("cos", b), 2 * at("cos", p) * at("cos", q)
        label = f"cos {a}° + cos {b}°"
        fv = fl("cos", a) + fl("cos", b)
        wrong = [2 * at("sin", p) * at("sin", q), -2 * at("cos", p) * at("cos", q), at("cos", p) * at("cos", q)]
    else:
        direct, via = at("cos", a) - at("cos", b), -2 * at("sin", p) * at("sin", q)
        label = f"cos {a}° - cos {b}°"
        fv = fl("cos", a) - fl("cos", b)
        wrong = [2 * at("sin", p) * at("sin", q), -2 * at("cos", p) * at("cos", q), -at("sin", p) * at("sin", q)]
    del fn, plus
    check(sp.simplify(direct - via) == 0, "sum differs from its product form")
    check(abs(float(via) - fv) < 1e-9, "product form differs from the float sum")
    tags = ["r3g-product-sum-wrong-half-angles", "r3g-product-sum-sign-slip", "r3g-product-sum-factor-handled-wrongly"]
    text = typed(clean(via))
    return dict(
        text=_ask(rng, label),
        answer_kind="expression",
        answer=text,
        answer_text=text,
        answer_latex=latex(clean(via)),
        working=[
            f"The half-sum of the angles is {p}° and the half-difference is {q}°.",
            f"The sum is a product of two values: {text}.",
        ],
        distractors=pick(text, [ex(v, t) for v, t in zip(wrong, tags)], 3),
    )


def _medium(rng):
    return rng.choice([_medium_product, _medium_product, _medium_sum, _medium_sum])(rng)


def _scan_solutions(f, denom: int) -> list[Fraction]:
    """Roots in [0, 1) (units of pi) on a grid of 1/denom, by floats; refuses a sign change that is off the grid."""
    roots = []
    prev = None
    fine = denom * 10
    for k in range(fine):
        x = k / fine
        v = f(math.pi * x)
        if abs(v) < 1e-9:
            roots.append(F(k, fine))
            prev = None
            continue
        if prev is not None and prev * v < 0:
            raise AssertionError(f"sign change off the grid near {x}")
        prev = v
    return roots


_ASKS = {
    "largest": ("the largest solution", lambda v: max(v)),
    "second": ("the second smallest solution", lambda v: sorted(v)[1] if len(v) > 1 else sorted(v)[0]),
    "sum": ("the sum of all the solutions", lambda v: sum(v)),
}


def _pick_summary(sols, wrong_sets, tags, mode):
    """(stem tail, answer fields, distractors) asking for one summary value of a solution set, as an exact multiple of pi."""
    words, fn = _ASKS[mode]
    value = fn(sols)
    fields = dict(
        answer_kind="expression", answer=pi_text(value), answer_text=pi_text(value), answer_latex=pi_latex(value)
    )
    cands = []
    for w, t in zip(wrong_sets, tags):
        w = list(w)
        if w:
            cands.append(pi_dist(fn(w), t))
    cands = [c for c in cands if c.text != fields["answer_text"]]
    tail = f"Find {words}, exactly in radians, typed like 5pi/6."
    return tail, fields, cands


def _term(coef, q: int) -> str:
    return f"{q}x" if q != 1 else "x"


def _hard_single(rng):
    """2 sin px cos qx = sin (p-q)x   or   2 cos px cos qx = cos (p-q)x."""
    for _ in attempts(200, "p, q with a small m"):
        p, q = rng.randint(2, 5), rng.randint(1, 4)
        if p > q and p + q <= 7 and p - q >= 1:
            break
    fn = rng.choice(("sin", "cos"))
    m = p + q
    if fn == "sin":
        sols = [F(k, m) for k in range(m)]
        eq = f"2 sin {_term(0, p)} cos {_term(0, q)} = sin {_term(0, p - q)}"
        f = lambda x: 2 * math.sin(p * x) * math.cos(q * x) - math.sin((p - q) * x)  # noqa: E731
        slips = [
            (F(k, p) for k in range(p)),
            (F(k, m - 1) for k in range(m - 1)) if m > 2 else [],
            (F(k, m) for k in range(1, m)),
        ]
        tags = [
            "r3g-product-sum-wrong-multiple",
            "r3g-product-sum-wrong-multiple",
            "r3g-product-sum-zero-solution-dropped",
        ]
    else:
        sols = [F(2 * k + 1, 2 * m) for k in range(m)]
        eq = f"2 cos {_term(0, p)} cos {_term(0, q)} = cos {_term(0, p - q)}"
        f = lambda x: 2 * math.cos(p * x) * math.cos(q * x) - math.cos((p - q) * x)  # noqa: E731
        slips = [
            (F(k, 2 * m) for k in range(m)),
            (F(2 * k + 1, 2 * p) for k in range(p)),
            (F(2 * k + 1, 2 * m) for k in range(m - 1)),
        ]
        tags = [
            "r3g-product-sum-cosine-zero-as-sine-zero",
            "r3g-product-sum-wrong-multiple",
            "r3g-product-sum-solution-dropped",
        ]
    got = _scan_solutions(f, 4 * m * 2)
    check(got == sols, f"scan found {got}, construction {sols}")
    mode = rng.choice(("largest", "second", "sum"))
    tail, fields, cands = _pick_summary(sols, slips, tags, mode)
    other = "sin" if fn == "sin" else "cos"
    return dict(
        text=f"Solve {eq} for 0 ≤ x < π. {tail}",
        working=[
            f"Write the product as a sum: the left side is {other} {m}x + {other} {_term(0, p - q)}.",
            f"That leaves {other} {m}x = 0.",
            f"So {m}x = {'kπ' if fn == 'sin' else '(2k + 1)π/2'} and the solutions in the interval are {', '.join(pi_shown(s) for s in sols)}.",
        ],
        distractors=pick(fields["answer_text"], cands, 3),
        **fields,
    )


def _hard_factor(rng):
    """sin px + sin qx = 0 over [0, pi): 2 sin(mx) cos(nx) = 0."""
    for _ in attempts(200, "p, q with integer half-sum and half-difference"):
        m, n = rng.randint(3, 5), rng.randint(1, 3)
        if m > n:
            break
    p, q = m + n, m - n
    sine_roots = {F(k, m) for k in range(m)}
    cos_roots = {F(2 * k + 1, 2 * n) for k in range(n) if F(2 * k + 1, 2 * n) < 1}
    sols = sorted(sine_roots | cos_roots)
    f = lambda x: math.sin(p * x) + math.sin(q * x)  # noqa: E731
    got = _scan_solutions(f, 4 * m * n * 2)
    check(got == sols, f"scan found {got}, construction {sols}")
    mode = rng.choice(("largest", "sum", "second"))
    tail, fields, cands = _pick_summary(
        sols,
        [
            sine_roots,
            cos_roots,
            {F(k, p) for k in range(p)} | {F(k, q) for k in range(q)},
            sine_roots - {F(0)} | cos_roots,
        ],
        [
            "r3g-product-sum-one-factor-only",
            "r3g-product-sum-one-factor-only",
            "r3g-product-sum-wrong-multiple",
            "r3g-product-sum-zero-solution-dropped",
        ],
        mode,
    )
    return dict(
        text=(f"Solve sin {_term(0, p)} + sin {_term(0, q)} = 0 for 0 ≤ x < π. {tail}"),
        working=[
            f"sin {_term(0, p)} + sin {_term(0, q)} = 2 sin {_term(0, m)} cos {_term(0, n)}, so sin {_term(0, m)} = 0 or cos {_term(0, n)} = 0.",
            f"sin {m}x = 0 gives {', '.join(pi_shown(s) for s in sorted(sine_roots))}; cos {_term(0, n)} = 0 gives "
            f"{', '.join(pi_shown(s) for s in sorted(cos_roots))}.",
            "Together the solutions are " + ", ".join(pi_shown(s) for s in sols) + ".",
        ],
        distractors=pick(fields["answer_text"], cands, 3),
        **fields,
    )


@generator(
    "r3g_product_sum",
    level=12,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
@retry_short
@vary_symbol("x", ("x", "θ", "φ"))
def r3g_product_sum(rng, tier="easy"):
    """Products and sums of sines and cosines, evaluated and solved; the tiers are described in the module docstring."""
    if tier == "easy":
        return _easy(rng)
    if tier == "medium":
        return _medium(rng)
    check(tier == "hard", f"unknown tier {tier!r}")
    return rng.choice([_hard_single, _hard_single, _hard_factor])(rng)


# -- exam -------------------------------------------------------------------------------------------------------


@generator("r3g_exam_product_sum", level=12, presets={"exam": dict(tier="exam")})
@retry_short
@vary_symbol("x", ("x", "θ", "φ"))
def r3g_exam_product_sum(rng, tier="exam"):
    """Exam chain: m in sin mx = 0, the smallest positive solution, the number of solutions and their sum."""
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(200, "p, q with a spread of solutions"):
        p, q = rng.randint(2, 7), rng.randint(1, 4)
        if p > q and 4 <= p + q <= 9:
            break
    m = p + q
    f = lambda x: 2 * math.sin(p * x) * math.cos(q * x) - math.sin((p - q) * x)  # noqa: E731
    sols = [F(k, m) for k in range(m)]
    check(_scan_solutions(f, 4 * m * 2) == sols, "scan differs from the construction")
    smallest = F(1, m)
    total = sum(sols)
    check(total == F(m - 1, 2), "sum of the solutions differs from the formula")
    count2 = 2 * m
    prod = f"2 sin {p}x cos {q}x" if q != 1 else f"2 sin {p}x cos x"
    rhs = f"sin {p - q}x" if p - q != 1 else "sin x"

    def number_part(prompt, value, marks, working, wrongs, follow=None, rng_=None):
        d = dict(
            prompt=prompt,
            answer_kind="number",
            answer=value,
            answer_text=str(value),
            answer_latex=str(value),
            marks=marks,
            working=working,
            distractors=pick(str(value), [int_dist(v, t) for v, t in wrongs if v != value and v > 0], 3),
            sensible_range=(0, None),
        )
        if follow:
            d["follow_through"] = follow
        return d

    def expr_part(prompt, coeff, marks, working, wrongs, follow=None):
        d = dict(
            prompt=prompt,
            answer_kind="expression",
            answer=pi_text(coeff),
            answer_text=pi_text(coeff),
            answer_latex=pi_latex(coeff),
            marks=marks,
            working=working,
            distractors=pick(pi_text(coeff), [pi_dist(v, t) for v, t in wrongs if v != coeff and v > 0], 3),
        )
        if follow:
            d["follow_through"] = follow
        return d

    return dict(
        text=f"Consider the equation {prod} = {rhs}.",
        parts=[
            number_part(
                "Use a product-to-sum identity to write the equation as sin mx = 0. Find m.",
                m,
                1,
                [f"{prod} = sin {m}x + sin {_term(0, p - q)}, so the equation becomes sin {m}x = 0."],
                [
                    (p - q, "r3g-product-sum-angles-not-added"),
                    (p * q, "r3g-product-sum-angles-not-added"),
                    (p, "r3g-product-sum-wrong-multiple"),
                    (m + 1, "r3g-product-sum-wrong-multiple"),
                    (2 * m, "r3g-product-sum-wrong-multiple"),
                ],
            ),
            expr_part(
                "Find the smallest positive solution of the equation. Give it exactly in radians, typed like pi/6.",
                smallest,
                1,
                [f"sin {m}x = 0 first at {m}x = π, so x = π/{m}."],
                [
                    (F(1, p), "r3g-product-sum-wrong-multiple"),
                    (F(1, 2 * m), "r3g-product-sum-wrong-multiple"),
                    (F(2, m), "r3g-product-sum-zero-solution-dropped"),
                    (F(1, p - q) if p - q > 1 else F(1, m + 1), "r3g-product-sum-wrong-multiple"),
                ],
                "pi/part_a",
            ),
            number_part(
                "How many solutions does the equation have for 0 ≤ x < 2π?",
                count2,
                1,
                [f"There are {m} solutions in each interval of length π, so {count2} in 0 ≤ x < 2π."],
                [
                    (m, "r3g-product-sum-interval-not-doubled"),
                    (count2 - 1, "r3g-product-sum-zero-solution-dropped"),
                    (count2 + 1, "r3g-product-sum-endpoint-counted"),
                    (m + 1, "r3g-product-sum-wrong-multiple"),
                ],
                "2*part_a",
            ),
            expr_part(
                "Find the sum of the solutions in 0 ≤ x < π. Give it exactly in radians.",
                total,
                2,
                [
                    f"The solutions are 0, π/{m}, ..., {m - 1}π/{m}. Their sum is π(0 + 1 + ... + {m - 1})/{m} = {pi_shown(total)}."
                ],
                [
                    (F(m, 2), "r3g-product-sum-sum-limits-wrong"),
                    (F(m + 1, 2), "r3g-product-sum-sum-limits-wrong"),
                    (F(m - 1), "r3g-product-sum-sum-limits-wrong"),
                    (F(m - 2, 2) if m > 2 else F(1, 4), "r3g-product-sum-sum-limits-wrong"),
                ],
                "pi*(part_a - 1)/2",
            ),
        ],
    )
