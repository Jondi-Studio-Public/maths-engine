"""Independent oracle for senior-examine-the-area-problem-and-use-sums-of
(ACMMM124, topic ``p13c_riemann_sum_area``).

Parses ONLY ``question.text``/``question.answer_text`` -- never imports
anything from ``mathgen.generators.p13c_riemann_sum_area_build`` (not a
helper, not a constant, not a table) -- and re-derives each Riemann sum from
scratch with ``sympy`` (symbolic substitution) and plain ``Fraction``
arithmetic over the STATED partition, a route independent of however the
generator's own construction built the running total.
"""

import re
from fractions import Fraction

import pytest
from sympy import Rational, Symbol, sympify

from mathgen import make

SEEDS = range(60)
TOPIC = "p13c_riemann_sum_area"

x = Symbol("x")

_UNIFORM_RE = re.compile(
    r"Using a (LEFT|RIGHT|MIDPOINT) Riemann sum with n = (\d+) equal subintervals, "
    r"estimate the area under the curve y = (?P<poly>.+) on \[(-?\d+), (-?\d+)\]\."
)
_EXAM_RE = re.compile(
    r"An interval is divided at breakpoints x = (?P<bps>[-\d, ]+) "
    r"\(each subinterval its own width\)\. Using a (LEFT|RIGHT) Riemann sum over this "
    r"partition, estimate the area under the curve y = (?P<poly>.+)\."
)


def _parse_poly(poly_str: str):
    # "3x^2 - 2x + 6" -> "3*x**2 - 2*x + 6" (sympify has no implicit
    # multiplication, so a digit immediately before "x" needs an explicit
    # "*"; a bare "x" with no digit in front is already a valid symbol).
    poly_str = poly_str.replace("^", "**")
    poly_str = re.sub(r"(\d)\s*(x)", r"\1*\2", poly_str)
    return sympify(poly_str, locals={"x": x})


def _fraction_str(value: Fraction) -> str:
    r = Rational(value.numerator, value.denominator)
    return str(r.p) if r.q == 1 else f"{r.p}/{r.q}"


@pytest.mark.parametrize("tier,rule", [("easy", "LEFT"), ("medium", "RIGHT"), ("hard", "MIDPOINT")])
def test_oracle_matches_uniform_riemann_sum(tier, rule):
    checked = 0
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=tier)
        m = _UNIFORM_RE.search(q.text)
        assert m is not None, f"{tier} seed={seed}: could not parse uniform-partition text {q.text!r}"
        assert m.group(1) == rule, f"{tier} seed={seed}: expected rule {rule}, text said {m.group(1)}"
        n = int(m.group(2))
        left, right = int(m.group(4)), int(m.group(5))
        poly = _parse_poly(m.group("poly"))
        width = Fraction(right - left, n)
        breakpoints = [Fraction(left) + i * width for i in range(n + 1)]
        if rule == "LEFT":
            xs = breakpoints[:-1]
        elif rule == "RIGHT":
            xs = breakpoints[1:]
        else:
            xs = [(breakpoints[i] + breakpoints[i + 1]) / 2 for i in range(n)]
        total = Fraction(0)
        for xi in xs:
            fx = poly.subs(x, Rational(xi.numerator, xi.denominator))
            fx_frac = Fraction(int(fx.p), int(fx.q)) if fx.is_Rational else Fraction(str(fx))
            total += fx_frac * width
        assert _fraction_str(total) == q.answer_text, (
            f"{tier} seed={seed}: oracle expected {_fraction_str(total)}, generator said {q.answer_text}"
        )
        checked += 1
    assert checked == len(SEEDS)


def test_oracle_matches_exam_non_uniform_riemann_sum():
    checked = 0
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty="exam")
        m = _EXAM_RE.search(q.text)
        assert m is not None, f"seed={seed}: could not parse exam-tier text {q.text!r}"
        rule = m.group(2)
        breakpoints = [Fraction(int(v.strip())) for v in m.group("bps").split(",")]
        poly = _parse_poly(m.group("poly"))
        dxs = [breakpoints[i + 1] - breakpoints[i] for i in range(len(breakpoints) - 1)]
        xs = breakpoints[:-1] if rule == "LEFT" else breakpoints[1:]
        total = Fraction(0)
        for xi, dx in zip(xs, dxs):
            fx = poly.subs(x, Rational(xi.numerator, xi.denominator))
            fx_frac = Fraction(int(fx.p), int(fx.q)) if fx.is_Rational else Fraction(str(fx))
            total += fx_frac * dx
        assert _fraction_str(total) == q.answer_text, (
            f"seed={seed}: oracle expected {_fraction_str(total)}, generator said {q.answer_text}"
        )
        # Confirm each strip's width really is individual (a genuinely
        # non-uniform partition), the exam tier's own defining feature.
        assert len(set(dxs)) > 1, f"seed={seed}: exam partition collapsed to uniform widths {dxs}"
        checked += 1
    assert checked == len(SEEDS)
