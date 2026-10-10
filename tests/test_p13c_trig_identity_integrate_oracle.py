"""Independent oracle for
senior-integrate-using-the-trigonometric-identities-mathrm-s-mathrm
(ACMSM116, topic ``p13c_trig_identity_integrate``).

Parses ONLY ``question.text``/``question.answer``/``question.answer_text`` --
never imports anything from
``mathgen.generators.p13c_trig_identity_integrate_build`` (not a helper, not
a constant, not a table). Extracts the integrand straight out of the
rendered prompt text, converts it to a sympy expression with a small local
parser (never `mathgen.grading._parse_expression` or any other project code),
and calls `sympy.integrate` on it directly -- a route completely independent
of however the generator applied the trigonometric identities by hand. This
is a genuinely different route from the generator's own `check_equivalent`,
which differentiates the CANDIDATE answer and compares to the integrand;
this oracle instead INTEGRATES the integrand from scratch via sympy's own
general integration routine and compares the result to the canonical.
"""

import re

import pytest
from sympy import Symbol, diff, integrate, sympify
from sympy.parsing.sympy_parser import (
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

from mathgen import make

SEEDS = range(60)
TIERS = ("easy", "medium", "hard", "exam")
TOPIC = "p13c_trig_identity_integrate"

x = Symbol("x")

_TRANSFORMATIONS = standard_transformations + (implicit_multiplication_application,)
_SQ_TRIG_RE = re.compile(r"(sin|cos|tan)\^2\(([^)]+)\)")
_INTEGRAND_RE = re.compile(r"∫ \((?P<integrand>.+?)\) dx")


def _to_sympy(expr_str: str):
    """Convert this generator's hand-built prompt notation (`sin^2(3x)`,
    never sympy's own `sin(3*x)**2` ordering) into a real sympy expression:
    rewrite `<fn>^2(<inner>)` to `(<fn>(<inner>))**2` first, then parse with
    implicit multiplication enabled so `4x` -> `4*x`."""
    rewritten = _SQ_TRIG_RE.sub(r"(\1(\2))**2", expr_str)
    return parse_expr(rewritten, transformations=_TRANSFORMATIONS)


def _extract_integrand(text: str) -> str:
    m = _INTEGRAND_RE.search(text)
    assert m is not None, f"could not find an ∫ (...) dx integrand in {text!r}"
    return m.group("integrand")


@pytest.mark.parametrize("tier", TIERS)
def test_oracle_reintegrates_and_matches_the_canonical(tier):
    checked = 0
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=tier)
        integrand_str = _extract_integrand(q.text)
        integrand = _to_sympy(integrand_str)

        # Independent route: integrate the integrand from scratch with
        # sympy's own general integration routine (never the generator's
        # own hand-applied identity formula).
        oracle_antiderivative = integrate(integrand, x)

        generator_value = sympify(q.answer.replace("^", "**"))
        assert (oracle_antiderivative - generator_value).simplify() == 0, (
            f"{tier} seed={seed}: oracle integrated {integrand_str!r} to "
            f"{oracle_antiderivative}, generator's answer is {q.answer!r}"
        )

        # A second, independent consistency check on the oracle's own route:
        # differentiating the oracle's antiderivative must recover the
        # integrand it started from (sympy.integrate should never need this
        # to be told twice, but this is the same discipline the generator's
        # own `check_equivalent` uses, run here by a reader who never saw
        # the generator's code).
        assert (diff(oracle_antiderivative, x) - integrand).simplify() == 0, (
            f"{tier} seed={seed}: oracle's own antiderivative does not "
            f"differentiate back to the integrand {integrand_str!r}"
        )

        assert q.answer_text.endswith(" + C"), (
            f"{tier} seed={seed}: answer_text {q.answer_text!r} does not end in ' + C'"
        )
        answer_text_value = sympify(q.answer_text[: -len(" + C")].replace("^", "**"))
        assert (answer_text_value - generator_value).simplify() == 0, (
            f"{tier} seed={seed}: answer_text value {answer_text_value} does not match generator's answer {q.answer!r}"
        )
        checked += 1
    assert checked == len(SEEDS)


# --- The boundary condition itself, read straight from the rendered text ---
# (never a bare linear sin(kx)/cos(kx) integrand -- integrate-trig's
# territory): every integrand extracted above must contain a "^2(" marker
# on at least one trig function.


@pytest.mark.parametrize("tier", TIERS)
def test_oracle_confirms_every_integrand_has_a_squared_trig_term(tier):
    checked = 0
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=tier)
        integrand_str = _extract_integrand(q.text)
        assert "^2(" in integrand_str, (
            f"{tier} seed={seed}: integrand has no squared trig term (bare "
            f"linear sin/cos, integrate-trig's territory): {integrand_str!r}"
        )
        checked += 1
    assert checked == len(SEEDS)
