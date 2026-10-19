"""Independent oracle for the `p19c_continuous_distribution` questions: it re-derives each answer from the
rendered question text.

Parses ONLY ``question.text``/``question.answer_text`` -- never imports
anything from ``mathgen.generators.p19c_continuous_distribution_build`` (not
a helper, not a constant, not a table). Re-derives every value from scratch
with its own ``sympy`` integration/arithmetic.
"""

from __future__ import annotations

import re
from fractions import Fraction

import pytest
import sympy as sp

from mathgen import make
from mathgen.core import round_half_up, stated_precision_string

PD_TOPIC = "p19c_density_probability"
CUM_TOPIC = "p19c_continuous_uniform_moments"

SEEDS = range(300)  # 300 seeds x 4 tiers = 1200 questions per skill
TIERS = ("easy", "medium", "hard", "exam")

_x, _k = sp.symbols("x k")


# --------------------------------------------------------------------------
# p19c_density_probability oracle
# --------------------------------------------------------------------------

_FEXPR_RE = re.compile(r"f\(x\) = (?P<fexpr>[^,]+?) for ")
_FEXPR_RE_DEFINED = re.compile(r"f\(x\) = (?P<fexpr>.+?), defined on")
_FEXPR_RE_LET = re.compile(r"f\(x\) = (?P<fexpr>.+?) for x in")

_SUPPORT_INEQ_RE = re.compile(r"(-?\d+) <= x <= (-?\d+)")
_SUPPORT_BRACKET_RE = re.compile(r"x in \[(-?\d+), (-?\d+)\]")

_BOUNDS_RE = re.compile(r"between (?:x = )?(-?\d+) and (?:x = )?(-?\d+)")
_DP_RE = re.compile(r"correct to (\d+) decimal places")


def _parse_fexpr(text: str) -> str:
    for pattern in (_FEXPR_RE, _FEXPR_RE_DEFINED, _FEXPR_RE_LET):
        m = pattern.search(text)
        if m:
            return m.group("fexpr").strip()
    raise AssertionError(f"could not parse f(x) out of {text!r}")


def _to_sympy_density(fexpr: str, c: sp.Integer, d: sp.Integer) -> sp.Expr:
    """Parse a density expression string the way it is rendered to a
    learner, e.g. "1/7" (numeric k, easy tier), "kx", "k(x - 3)",
    "k(x - 2)^2" (LITERAL letter k, medium/hard/exam) -- inserting the
    explicit '*' sympify needs for the implicit-multiplication juxtapositions
    this generator's own display uses, then substituting '^' for '**'. When
    the letter k appears literally, this is the "is it a genuine density"
    proof in its strongest form: k is not read off the text at all, but
    SOLVED for independently from the requirement that the density
    integrates to exactly 1 over its own stated support [c, d]."""
    fixed = fexpr.replace("^", "**")
    fixed = re.sub(r"\)(?=[(xk\d])", ")*", fixed)  # "(1/72)(x - 3)" -> "(1/72)*(x - 3)"
    fixed = re.sub(r"k(?=[(x])", "k*", fixed)  # "k(x - 3)" -> "k*(x - 3)", "kx" -> "k*x"
    if re.search(r"(?<![a-zA-Z])k(?![a-zA-Z])", fixed):
        unnormalised = sp.sympify(fixed, locals={"x": _x, "k": _k})
        total_unnormalised = sp.integrate(unnormalised, (_x, c, d))
        k_value = sp.solve(sp.Eq(total_unnormalised, 1), _k)[0]
        return unnormalised.subs(_k, k_value)
    return sp.sympify(fixed, locals={"x": _x})


def _parse_support(text: str) -> tuple[sp.Integer, sp.Integer]:
    m = _SUPPORT_INEQ_RE.search(text)
    if m:
        return sp.Integer(m.group(1)), sp.Integer(m.group(2))
    m = _SUPPORT_BRACKET_RE.search(text)
    if m:
        return sp.Integer(m.group(1)), sp.Integer(m.group(2))
    raise AssertionError(f"could not parse support [c, d] out of {text!r}")


def _parse_bounds(text: str) -> tuple[sp.Integer, sp.Integer]:
    m = _BOUNDS_RE.search(text)
    assert m is not None, f"could not parse bounds p, q out of {text!r}"
    return sp.Integer(m.group(1)), sp.Integer(m.group(2))


def _parse_dp(text: str) -> int:
    m = _DP_RE.search(text)
    assert m is not None, f"could not parse stated precision out of {text!r}"
    return int(m.group(1))


@pytest.mark.parametrize("tier", TIERS)
def test_density_normalises_to_one_and_is_nonnegative(tier):
    """The "is it a genuine density" proof: independently re-integrate the
    stated density over its FULL stated support and confirm it is exactly
    1, and sample it across the support to confirm it is nonnegative
    everywhere there -- for every seed sampled."""
    checked = 0
    for seed in SEEDS:
        q = make(PD_TOPIC, seed=seed, difficulty=tier)
        fexpr = _parse_fexpr(q.text)
        c, d = _parse_support(q.text)
        f = _to_sympy_density(fexpr, c, d)
        total = sp.integrate(f, (_x, c, d))
        assert sp.nsimplify(total) == 1, (
            f"{tier} seed={seed}: density does not integrate to 1: {fexpr!r} on [{c},{d}] -> {total}"
        )
        for sample in range(int(c), int(d) + 1):
            value = f.subs(_x, sample)
            assert value >= 0, f"{tier} seed={seed}: density negative at x={sample}: {fexpr!r} -> {value}"
        checked += 1
    assert checked == len(SEEDS)


@pytest.mark.parametrize("tier", TIERS)
def test_density_probability_oracle_matches_answer(tier):
    checked = 0
    for seed in SEEDS:
        q = make(PD_TOPIC, seed=seed, difficulty=tier)
        fexpr = _parse_fexpr(q.text)
        c, d = _parse_support(q.text)
        f = _to_sympy_density(fexpr, c, d)
        p, qb = _parse_bounds(q.text)
        dp = _parse_dp(q.text)

        assert c <= p < qb <= d, f"{tier} seed={seed}: bounds not inside support: c={c} p={p} q={qb} d={d}"

        prob = sp.integrate(f, (_x, p, qb))
        raw = Fraction(int(sp.nsimplify(prob).p), int(sp.nsimplify(prob).q))
        rounded = round_half_up(raw, dp)
        expected_text = stated_precision_string(rounded, dp)

        assert expected_text == q.answer_text, (
            f"{tier} seed={seed}: oracle computed {expected_text!r} "
            f"(fexpr={fexpr!r}, c={c}, d={d}, p={p}, q={qb}), generator said {q.answer_text!r}"
        )
        checked += 1
    assert checked == len(SEEDS)


@pytest.mark.parametrize("tier", ("easy", "medium", "hard"))
def test_density_bounds_strictly_interior_below_exam(tier):
    for seed in SEEDS:
        q = make(PD_TOPIC, seed=seed, difficulty=tier)
        c, d = _parse_support(q.text)
        p, qb = _parse_bounds(q.text)
        assert c < p < qb < d, f"{tier} seed={seed}: bounds not strictly interior: c={c} p={p} q={qb} d={d}"


def test_density_exam_tier_touches_a_support_endpoint():
    checked = 0
    for seed in SEEDS:
        q = make(PD_TOPIC, seed=seed, difficulty="exam")
        c, d = _parse_support(q.text)
        p, qb = _parse_bounds(q.text)
        assert p == c or qb == d, f"exam seed={seed}: neither bound touches an endpoint: c={c} p={p} q={qb} d={d}"
        assert not (p == c and qb == d), f"exam seed={seed}: both bounds touch endpoints (degenerate): c={c} d={d}"
        checked += 1
    assert checked == len(SEEDS)


# --------------------------------------------------------------------------
# p19c_continuous_uniform_moments oracle
# --------------------------------------------------------------------------

_BRACKET_INTERVAL_RE = re.compile(r"\[(-?\d+), (-?\d+)\]")
_FROM_TO_INTERVAL_RE = re.compile(r"from (-?\d+) to (-?\d+)")
_SCENARIO_PATTERNS = (
    re.compile(r"at least (-?\d+) hours and at most (-?\d+) hours"),
    re.compile(r"minimum of (-?\d+) kilograms and a maximum of (-?\d+) kilograms"),
    re.compile(r"never less than (-?\d+) minutes and never more than (-?\d+) minutes"),
)


def _parse_interval(text: str) -> tuple[int, int]:
    m = _BRACKET_INTERVAL_RE.search(text)
    if m:
        return int(m.group(1)), int(m.group(2))
    m = _FROM_TO_INTERVAL_RE.search(text)
    if m:
        return int(m.group(1)), int(m.group(2))
    for pattern in _SCENARIO_PATTERNS:
        m = pattern.search(text)
        if m:
            return int(m.group(1)), int(m.group(2))
    raise AssertionError(f"could not parse interval [a, b] out of {text!r}")


def _oracle_moments(a: int, b: int) -> tuple[sp.Rational, sp.Rational, sp.Expr]:
    a_sp, b_sp = sp.Integer(a), sp.Integer(b)
    e = (a_sp + b_sp) / sp.Integer(2)
    var = (b_sp - a_sp) ** 2 / sp.Integer(12)
    sd = sp.sqrt(var)
    return e, var, sd


@pytest.mark.parametrize("tier", TIERS)
def test_continuous_uniform_moments_oracle_matches_answer(tier):
    checked = 0
    for seed in SEEDS:
        q = make(CUM_TOPIC, seed=seed, difficulty=tier)
        a, b = _parse_interval(q.text)
        assert a < b, f"{tier} seed={seed}: interval not ordered a < b: a={a} b={b}"
        dp = int(_DP_RE.search(q.text).group(1))

        e_sp, var_sp, sd_sp = _oracle_moments(a, b)
        e_frac = Fraction(int(sp.nsimplify(e_sp).p), int(sp.nsimplify(e_sp).q))
        var_frac = Fraction(int(sp.nsimplify(var_sp).p), int(sp.nsimplify(var_sp).q))
        sd_frac = Fraction(str(sd_sp.evalf(30)))

        e_text = stated_precision_string(round_half_up(e_frac, dp), dp)
        var_text = stated_precision_string(round_half_up(var_frac, dp), dp)
        sd_text = stated_precision_string(round_half_up(sd_frac, dp), dp)
        expected_text = f"({e_text}, {var_text}, {sd_text})"

        assert expected_text == q.answer_text, (
            f"{tier} seed={seed}: oracle computed {expected_text!r} (a={a}, b={b}), generator said {q.answer_text!r}"
        )
        checked += 1
    assert checked == len(SEEDS)


def test_continuous_uniform_moments_easy_tier_nonnegative_interval():
    for seed in SEEDS:
        q = make(CUM_TOPIC, seed=seed, difficulty="easy")
        a, _b = _parse_interval(q.text)
        assert a >= 0


def test_continuous_uniform_moments_hard_tier_negative_lower_bound():
    for seed in SEEDS:
        q = make(CUM_TOPIC, seed=seed, difficulty="hard")
        a, _b = _parse_interval(q.text)
        assert a < 0
