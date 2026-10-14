"""Generates `r3b_integral_properties` questions (easy, medium, hard) and the exam chain
`r3b_exam_integral_properties`.

Skill ``definite-integral-properties`` (``integration``, answer ``number``): values of definite integrals found from
the properties of the integral, using stated values of other integrals of the same function, with no formula for f.

* easy: additivity and reversal: the integral over a whole interval from the integrals over two parts, a missing part
  from the whole and the other part, or the integral with its limits reversed.
* medium: symmetry. f is even or odd, its integral over [0, a] is given, and the integral of ``m f(x) + c`` over [-a, a],
  of f over [-a, 0], or of a mixture of an even and an odd function is found.
* hard: reflection, shifts and periodicity: ``m f(x) + n f(a + b - x)`` over [a, b], ``f(x + c)`` over a shifted
  interval, ``f`` of period p over several periods, and ``f(2a - x) = f(x)``.
* exam (``r3b_exam_integral_properties``): f even with its integral over [0, p] and [p, q] given: (a) over [-p, p], (b) over
  [0, q], (c) over [-q, q], (d) the integral of ``k f(x) + m x + n`` over [-q, q], where the odd term vanishes.

Each question is built from a model: f is a step function with an integer height on each unit interval, made even, odd,
periodic or reflected as the stem says. The stated integrals are read off the model, and the key is the model's own
integral, which is a second route to the value the property gives.
"""

from __future__ import annotations

import re

from ..core import attempts, check, generator, latex_from_inline
from ._r3b_kit import int_fields, part, pick_ints, plus, retry, say

_ASK = ("{P} Find {I}.", "{P} Evaluate {I}.", "{P} Calculate {I}.")


class Model:
    """A step function: `heights[i]` is the height of f on the unit interval [lo + i, lo + i + 1)."""

    def __init__(self, lo: int, heights: list[int]):
        self.lo, self.heights = lo, list(heights)

    def height(self, i: int) -> int:
        return self.heights[i - self.lo]

    def integral(self, a: int, b: int) -> int:
        """The integral over [a, b] (whole-number ends inside the model), negative when a > b."""
        if a > b:
            return -self.integral(b, a)
        return sum(self.height(i) for i in range(a, b))


def _cells(rng, n: int, lo: int = -4, hi: int = 7) -> list[int]:
    """n non-zero heights whose sum is not zero."""
    for _ in attempts(200, "step heights with a non-zero sum"):
        hs = [rng.choice([v for v in range(lo, hi + 1) if v]) for _ in range(n)]
        if sum(hs) != 0:
            return hs
    raise AssertionError  # pragma: no cover


def _even(rng, a: int) -> Model:
    right = _cells(rng, a)
    return Model(-a, list(reversed(right)) + right)


def _odd(rng, a: int) -> Model:
    right = _cells(rng, a)
    return Model(-a, [-h for h in reversed(right)] + right)


def _nonzero(rng, lo=-9, hi=9):
    return rng.choice([v for v in range(lo, hi + 1) if v])


def _I(lo, hi, body: str = "f(x)", body_tex: str | None = None):
    """An integral as (plain text, LaTeX)."""
    return f"∫ from {lo} to {hi} of {body} dx", rf"\int_{{{lo}}}^{{{hi}}} {body_tex or body}\,\mathrm{{d}}x"


def _eq(lo, hi, value, body: str = "f(x)", body_tex: str | None = None):
    t, m = _I(lo, hi, body, body_tex)
    return f"{t} = {value}", rf"{m} = {value}"


def _prem(words: str, *eqs, words_latex: str | None = None):
    """`words` then the equations, as a premise: (plain text, inline LaTeX). A lone f or g in `words` is set as maths
    in the LaTeX unless `words_latex` is given."""
    texts = [t for t, _ in eqs]
    maths = [rf"\({m}\)" for _, m in eqs]

    def andify(parts):
        return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]

    if words_latex is None:
        words_latex = re.sub(r"\b([fg])\b", r"\\(\1\\)", words)
    return f"{words}{andify(texts)}.", f"{words_latex}{andify(maths)}."


def _c(v) -> str:
    """A coefficient as written before a letter: 1 is blank and -1 is a minus sign."""
    return "" if v == 1 else "-" if v == -1 else str(v)


def _sgn(n) -> str:
    return "+" if n > 0 else "-"


def _build(rng, premise, ask, key, cands, working):
    (pt, pm), (at, am) = premise, ask
    text, latex = say(rng, _ASK, {"P": pt, "I": at}, {"P": pm, "I": rf"\({am}\)"})
    return dict(text=text, latex=latex, working=working, distractors=pick_ints(key, cands, rng=rng), **int_fields(key))


# -- easy ---------------------------------------------------------------------------------------------------------


def _easy(rng):
    kind = rng.choice(["sum", "missing", "reversed"])
    a = rng.choice([-2, -1, 0, 1])
    b = a + rng.choice([1, 2])
    c = b + rng.choice([1, 2, 3])
    for _ in attempts(200, "pieces with non-zero, distinct integrals"):
        model = Model(a, _cells(rng, c - a))
        P, Q, T = model.integral(a, b), model.integral(b, c), model.integral(a, c)
        if 0 not in (P, Q, T) and abs(P) != abs(Q):
            break
    if kind == "sum":
        premise = _prem("The function f satisfies ", _eq(a, b, P), _eq(b, c, Q))
        ask = _I(a, c)
        key, cands = (
            model.integral(a, c),
            [
                (P - Q, "r3b-prop-parts-subtracted"),
                (Q, "r3b-prop-one-part-only"),
                (P, "r3b-prop-one-part-only"),
                (-(P + Q), "r3b-prop-limits-reversed-sign"),
            ],
        )
        check(key == P + Q, "additivity differs from the model")
        working = [
            f"The interval from {a} to {c} is the interval from {a} to {b} followed by the interval from {b} to {c}: {P} + ({Q}) = {key}."
        ]
    elif kind == "missing":
        premise = _prem("The function f satisfies ", _eq(a, c, T), _eq(a, b, P))
        ask = _I(b, c)
        key, cands = (
            model.integral(b, c),
            [
                (T + P, "r3b-prop-parts-added"),
                (P - T, "r3b-prop-limits-reversed-sign"),
                (T, "r3b-prop-whole-for-part"),
                (-Q, "r3b-prop-limits-reversed-sign"),
            ],
        )
        check(key == T - P, "subtraction differs from the model")
        working = [f"The whole integral is the sum of the parts, so the part from {b} to {c} is {T} - ({P}) = {key}."]
    else:
        premise = _prem("The function f satisfies ", _eq(a, b, P), _eq(b, c, Q))
        ask = _I(c, a)
        key, cands = (
            model.integral(c, a),
            [
                (P + Q, "r3b-prop-limits-reversed-sign"),
                (P - Q, "r3b-prop-parts-subtracted"),
                (Q - P, "r3b-prop-parts-subtracted"),
                (-Q, "r3b-prop-one-part-only"),
            ],
        )
        check(key == -(P + Q), "reversal differs from the model")
        working = [
            f"The integral from {a} to {c} is {P} + ({Q}) = {P + Q}; reversing the limits changes the sign, giving {key}."
        ]
    return _build(rng, premise, ask, key, cands, working)


# -- medium -------------------------------------------------------------------------------------------------------


def _medium(rng):
    kind = rng.choice(["even", "odd", "mixed", "half"])
    a = rng.choice([2, 3, 4])
    m = _nonzero(rng, -4, 5)
    c = _nonzero(rng, -5, 6)
    if kind == "even":
        model = _even(rng, a)
        P = model.integral(0, a)
        premise = _prem("Let f be an even function with ", _eq(0, a, P))
        ask = _I(-a, a, f"({_c(m)}f(x) {_sgn(c)} {abs(c)})", f"({_c(m)}f(x) {_sgn(c)} {abs(c)})")
        key = m * model.integral(-a, a) + c * 2 * a
        check(key == 2 * m * P + 2 * a * c, "even symmetry differs from the model")
        cands = [
            (m * P + 2 * a * c, "r3b-prop-even-not-doubled"),
            (2 * a * c, "r3b-prop-even-treated-as-odd"),
            (2 * m * P + c, "r3b-prop-constant-not-over-width"),
            (2 * m * P, "r3b-prop-constant-dropped"),
        ]
        working = [
            f"f is even, so the integral of f over [{-a}, {a}] is 2 × {P} = {2 * P}; for the constant {c}, the integral is {c} × {2 * a} = {2 * a * c}."
        ]
    elif kind == "odd":
        model = _odd(rng, a)
        P = model.integral(0, a)
        premise = _prem("Let f be an odd function with ", _eq(0, a, P))
        ask = _I(-a, a, f"({_c(m)}f(x) {_sgn(c)} {abs(c)})", f"({_c(m)}f(x) {_sgn(c)} {abs(c)})")
        key = m * model.integral(-a, a) + c * 2 * a
        check(key == 2 * a * c, "odd symmetry differs from the model")
        cands = [
            (2 * m * P + 2 * a * c, "r3b-prop-odd-treated-as-even"),
            (m * P + 2 * a * c, "r3b-prop-even-not-doubled"),
            (c, "r3b-prop-constant-not-over-width"),
            (0, "r3b-prop-constant-dropped"),
        ]
        working = [
            f"f is odd, so its integral over [{-a}, {a}] is 0; only the constant {c} is left: {c} × {2 * a} = {2 * a * c}."
        ]
    elif kind == "mixed":
        mf, mg = _even(rng, a), _odd(rng, a)
        P, Q = mf.integral(0, a), mg.integral(0, a)
        n = _nonzero(rng, -4, 5)
        premise = (
            f"Let f be an even function and g an odd function, with ∫ from 0 to {a} of f(x) dx = {P} and ∫ from 0 to {a} of g(x) dx = {Q}.",
            rf"Let \(f\) be an even function and \(g\) an odd function, with \(\int_0^{{{a}}} f(x)\,\mathrm{{d}}x = {P}\) and \(\int_0^{{{a}}} g(x)\,\mathrm{{d}}x = {Q}\).",
        )
        ask = _I(-a, a, f"({_c(m)}f(x) {_sgn(n)} {_c(abs(n))}g(x))", f"({_c(m)}f(x) {_sgn(n)} {_c(abs(n))}g(x))")
        key = m * mf.integral(-a, a) + n * mg.integral(-a, a)
        check(key == 2 * m * P, "mixed symmetry differs from the model")
        cands = [
            (2 * m * P + 2 * n * Q, "r3b-prop-odd-treated-as-even"),
            (m * P + n * Q, "r3b-prop-even-not-doubled"),
            (2 * (m * P + n * Q), "r3b-prop-odd-treated-as-even"),
            (2 * m * P + n * Q, "r3b-prop-odd-part-kept"),
        ]
        working = [
            f"The even part doubles: 2 × {m} × {P} = {2 * m * P}. The odd part integrates to 0 over a symmetric interval."
        ]
    else:
        even = rng.random() < 0.5
        model = _even(rng, a) if even else _odd(rng, a)
        P = model.integral(0, a)
        word = "even" if even else "odd"
        premise = _prem(f"Let f be an {word} function with ", _eq(0, a, P))
        ask = _I(-a, 0)
        key = model.integral(-a, 0)
        check(key == (P if even else -P), "half-interval symmetry differs from the model")
        cands = [
            ((-P if even else P), "r3b-prop-even-treated-as-odd" if even else "r3b-prop-odd-treated-as-even"),
            (2 * P, "r3b-prop-even-not-doubled"),
            (0, "r3b-prop-odd-treated-as-even"),
            (-2 * P if not even else 2 * P, "r3b-prop-even-not-doubled"),
        ]
        working = [
            f"For an {word} function the integral over [{-a}, 0] is {'the same as' if even else 'the negative of'} the integral over [0, {a}], so it is {key}."
        ]
    return _build(rng, premise, ask, key, cands, working)


# -- hard ---------------------------------------------------------------------------------------------------------


def _hard(rng):
    kind = rng.choice(["reflect", "period", "shift", "mirror"])
    if kind == "reflect":
        a = rng.choice([0, 1, 2])
        b = a + rng.choice([2, 3, 4])
        model = Model(a, _cells(rng, b - a))
        P = model.integral(a, b)
        m, n = _nonzero(rng, -4, 5), _nonzero(rng, -4, 5)
        reflected = Model(a, list(reversed(model.heights)))
        key = m * model.integral(a, b) + n * reflected.integral(a, b)
        check(key == (m + n) * P, "reflection differs from the model")
        s = a + b
        premise = _prem("Let f be a function with ", _eq(a, b, P))
        ask = _I(
            a, b, f"({_c(m)}f(x) {_sgn(n)} {_c(abs(n))}f({s} - x))", f"({_c(m)}f(x) {_sgn(n)} {_c(abs(n))}f({s} - x))"
        )
        cands = [
            (m * P - n * P, "r3b-prop-reflection-reverses-sign"),
            (m * P, "r3b-prop-reflected-term-dropped"),
            (m * P + n * (b - a), "r3b-prop-reflection-as-constant"),
            (2 * (m + n) * P, "r3b-prop-reflection-doubled"),
        ]
        working = [
            f"Put u = {s} - x; the integral of f({s} - x) from {a} to {b} becomes the integral of f from {a} to {b}, which is {P}.",
            f"So the value is ({m} + ({n})) × {P} = {key}.",
        ]
    elif kind == "period":
        p = rng.choice([2, 3, 4])
        cnt = rng.choice([3, 4, 5])
        c = rng.choice([0, 1, 2, 3])
        base = _cells(rng, p)
        model = Model(0, base * (cnt + 2))
        P = model.integral(0, p)
        lo, hi = c, c + cnt * p
        key = model.integral(lo, hi)
        check(key == cnt * P, "periodicity differs from the model")
        premise = _prem(f"Let f be a function of period {p} with ", _eq(0, p, P))
        ask = _I(lo, hi)
        cands = [
            (P, "r3b-prop-one-period-only"),
            (cnt * p * P, "r3b-prop-period-multiplied-twice"),
            ((cnt + 1) * P, "r3b-prop-periods-miscounted"),
            ((cnt - 1) * P, "r3b-prop-periods-miscounted"),
        ]
        working = [
            f"The interval from {lo} to {hi} has width {cnt * p}, which is {cnt} periods of {p}. Any interval of one period has the same integral, {P}, so the value is {cnt} × {P} = {key}."
        ]
    elif kind == "shift":
        s = rng.choice([1, 2, 3])
        w = rng.choice([2, 3, 4])
        c = rng.choice([2, 3, 4, -2, -3])
        lo_f = s + max(c, 0)
        model = Model(min(s, s + c) - 1, _cells(rng, w + abs(c) + 2))
        P = model.integral(s + c, s + c + w)
        premise = _prem("Let f be a function with ", _eq(s + c, s + c + w, P))
        shift = f"x {_sgn(c)} {abs(c)}"
        ask = _I(s, s + w, f"f({shift})", f"f({shift})")
        shifted = Model(model.lo - c, model.heights)  # g(x) = f(x + c)
        key = shifted.integral(s, s + w)
        check(key == P, "shifting differs from the model")
        cands = [
            (P + c * w, "r3b-prop-shift-added-to-integral"),
            (-P, "r3b-prop-limits-reversed-sign"),
            (P * c, "r3b-prop-shift-multiplies-integral"),
            (P - c * w, "r3b-prop-shift-added-to-integral"),
        ]
        working = [
            f"Put u = {shift}; the integral of f({shift}) from {s} to {s + w} becomes the integral of f(u) from {s + c} to {s + c + w}, which is {P}."
        ]
    else:
        a = rng.choice([2, 3, 4])
        right = _cells(rng, a)
        model = Model(0, right + list(reversed(right)))
        P = model.integral(0, a)
        c = _nonzero(rng, -4, 5)
        premise = _prem(
            f"Let f be a function with f({2 * a} - x) = f(x) for all x, and ",
            _eq(0, a, P),
            words_latex=rf"Let \(f\) be a function with \(f({2 * a} - x) = f(x)\) for all \(x\), and ",
        )
        ask = _I(0, 2 * a, f"(f(x) {_sgn(c)} {abs(c)})", f"(f(x) {_sgn(c)} {abs(c)})")
        key = model.integral(0, 2 * a) + c * 2 * a
        check(key == 2 * P + 2 * a * c, "mirror symmetry differs from the model")
        cands = [
            (P + 2 * a * c, "r3b-prop-even-not-doubled"),
            (2 * P + c, "r3b-prop-constant-not-over-width"),
            (2 * P, "r3b-prop-constant-dropped"),
            (2 * P - 2 * a * c, "r3b-prop-constant-sign"),
        ]
        working = [
            f"The graph of f is symmetric about x = {a}, so the integral over [0, {2 * a}] is 2 × {P} = {2 * P}; the constant adds {c} × {2 * a} = {2 * a * c}."
        ]
    return _build(rng, premise, ask, key, cands, working)


@generator(
    "r3b_integral_properties",
    level=15,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
@retry
def r3b_integral_properties(rng, tier="easy"):
    """Definite integrals from the properties of the integral; the tiers are described in the module docstring."""
    if tier == "easy":
        return _easy(rng)
    if tier == "medium":
        return _medium(rng)
    check(tier == "hard", f"unknown tier {tier!r}")
    return _hard(rng)


# -- exam ---------------------------------------------------------------------------------------------------------


@generator("r3b_exam_integral_properties", level=15, presets={"exam": dict(tier="exam")})
@retry
def r3b_exam_integral_properties(rng, tier="exam"):
    """Exam chain: an even function's integrals over adjacent intervals, then a transformed integral."""
    check(tier == "exam", f"unknown tier {tier!r}")
    p = rng.choice([2, 3])
    q = p + rng.choice([1, 2, 3])
    model = _even(rng, q)
    P, Q = model.integral(0, p), model.integral(p, q)
    k = _nonzero(rng, -4, 5)
    m = _nonzero(rng, -5, 6)
    n = _nonzero(rng, -5, 6)
    for _ in attempts(200, "values whose sums are distinct and non-zero"):
        if P + Q != 0 and P != Q and P != 0 and Q != 0:
            break
        model = _even(rng, q)
        P, Q = model.integral(0, p), model.integral(p, q)
    a_val = model.integral(-p, p)
    b_val = model.integral(0, q)
    c_val = model.integral(-q, q)
    # the integral of m x over [-q, q] is 0 (x is odd); the constant n contributes n x 2q
    d_val = k * c_val + 0 * m + n * 2 * q
    check(a_val == 2 * P and b_val == P + Q and c_val == 2 * (P + Q), "the even-function rules differ from the model")
    check(d_val == k * 2 * (P + Q) + 2 * q * n, "the transformed integral differs from the rules")
    premise_text = (
        f"Let f be an even function with ∫ from 0 to {p} of f(x) dx = {P} and ∫ from {p} to {q} of f(x) dx = {Q}."
    )
    premise_latex = (
        rf"Let \(f\) be an even function with \(\int_0^{{{p}}} f(x)\,\mathrm{{d}}x = {P}\) "
        rf"and \(\int_{{{p}}}^{{{q}}} f(x)\,\mathrm{{d}}x = {Q}\)."
    )
    expr_text = f"({_c(k)}f(x) {_sgn(m)} {_c(abs(m))}x {_sgn(n)} {abs(n)})"
    pa = part(
        f"Find ∫ from {-p} to {p} of f(x) dx.",
        rf"Find \(\int_{{{-p}}}^{{{p}}} f(x)\,\mathrm{{d}}x\).",
        1,
        [f"f is even, so the integral over [{-p}, {p}] is twice the integral over [0, {p}]: 2 × {P} = {a_val}."],
        int_fields(a_val),
        pick_ints(
            a_val,
            [
                (P, "r3b-prop-even-not-doubled"),
                (0, "r3b-prop-even-treated-as-odd"),
                (P + Q, "r3b-prop-one-part-only"),
                (2 * Q, "r3b-prop-wrong-piece"),
            ],
            rng=rng,
        ),
        sensible_range=(-400, 400),
    )
    pb = part(
        f"Find ∫ from 0 to {q} of f(x) dx.",
        rf"Find \(\int_0^{{{q}}} f(x)\,\mathrm{{d}}x\).",
        1,
        [
            f"The interval from 0 to {q} is the interval from 0 to {p} followed by the interval from {p} to {q}: {P} + ({Q}) = {b_val}."
        ],
        int_fields(b_val),
        pick_ints(
            b_val,
            [
                (P - Q, "r3b-prop-parts-subtracted"),
                (2 * (P + Q), "r3b-prop-even-applied-when-not-asked"),
                (Q, "r3b-prop-one-part-only"),
                (P, "r3b-prop-one-part-only"),
            ],
            rng=rng,
        ),
        sensible_range=(-400, 400),
    )
    pc = part(
        f"Hence find ∫ from {-q} to {q} of f(x) dx.",
        rf"Hence find \(\int_{{{-q}}}^{{{q}}} f(x)\,\mathrm{{d}}x\).",
        2,
        [f"f is even, so the integral over [{-q}, {q}] is twice the integral over [0, {q}]: 2 × {b_val} = {c_val}."],
        int_fields(c_val),
        pick_ints(
            c_val,
            [
                (b_val, "r3b-prop-even-not-doubled"),
                (0, "r3b-prop-even-treated-as-odd"),
                (2 * P + Q, "r3b-prop-one-part-doubled"),
                (a_val + Q, "r3b-prop-one-part-doubled"),
            ],
            rng=rng,
        ),
        follow_through="2*part_b",
        sensible_range=(-800, 800),
    )
    pd = part(
        f"Hence find ∫ from {-q} to {q} of {expr_text} dx.",
        rf"Hence find \(\int_{{{-q}}}^{{{q}}} \left({_c(k)}f(x) {_sgn(m)} {_c(abs(m))}x {_sgn(n)} {abs(n)}\right)\mathrm{{d}}x\).",
        3,
        [
            f"The integral of f is {c_val}. The function x is odd, so its integral over [{-q}, {q}] is 0. The constant {n} over a width of {2 * q} gives {n * 2 * q}.",
            f"{k} × {c_val} + 0 {plus(n * 2 * q)} = {d_val}.",
        ],
        int_fields(d_val),
        pick_ints(
            d_val,
            [
                (k * c_val + n * q, "r3b-prop-constant-not-over-width"),
                (k * c_val + m * q * q + n * 2 * q, "r3b-prop-odd-part-kept"),
                (c_val + n * 2 * q, "r3b-prop-scale-factor-dropped"),
                (k * c_val, "r3b-prop-constant-dropped"),
                (k * c_val + n, "r3b-prop-constant-not-over-width"),
            ],
            rng=rng,
        ),
        follow_through=f"{k}*part_c+{n * 2 * q}",
        sensible_range=(-2000, 2000),
    )
    return dict(text=premise_text, latex=latex_from_inline(premise_latex), parts=[pa, pb, pc, pd])
