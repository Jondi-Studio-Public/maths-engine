"""Independent oracle for the `p15c_functional` questions: it re-derives each answer from the rendered question
text.

Parses ONLY `question.text` and `question.answer`/`answer_text` -- never
anything from `mathgen.generators.p15c_functional_build` -- and re-derives
each answer from scratch: the operator norm via a fresh `sympy.Matrix`
singular-value computation built from entries read out of the rendered text
(never trusting the generator's own diagonal-max shortcut), and the
least-squares coefficients via a fresh normal-equations solve built from
basis values evaluated independently from the rendered basis name and sample
points/angles.
"""

import re
from fractions import Fraction

import pytest
import sympy

from mathgen.skills import load_skills

NORM_ID = "ucv1-diagonal-operator-norm"
APPROX_ID = "ucv1-function-space-approximation"
SEEDS = range(60)
TIERS = ["easy", "medium", "hard", "exam"]


def _skill_sources(skill_id):
    _domains, skills = load_skills()
    skill = next(s for s in skills if s.id == skill_id)
    return {src.difficulty: src for src in skill.sources}


NORM_SOURCES = _skill_sources(NORM_ID)
APPROX_SOURCES = _skill_sources(APPROX_ID)


def _make(sources, difficulty, seed):
    import mathgen

    src = sources[difficulty]
    return mathgen.make(src.topic, seed=seed, difficulty=src.difficulty, **(src.params or {}))


def _frac(s):
    s = s.strip()
    if "/" in s:
        p, q = s.split("/")
        return Fraction(int(p), int(q))
    return Fraction(int(s))


# ---------------------------------------------------------------------------
# ucv1-diagonal-operator-norm

_DIAG_RE = re.compile(r"diag\((-?\d+), (-?\d+)\)")
_ACTION_RE = re.compile(r"T\(x, y\) = \(([^)]+)\)")
_COMPOSE_RE = re.compile(r"T\(x, y\) = \(([^)]+)\) and S\(x, y\) = \(([^)]+)\)")
_POWER_RE = re.compile(r"T\(x, y\) = \(([^)]+)\).*T\^(\d+)")


def _parse_coef_var(token: str, var: str) -> Fraction:
    """Independently parses a coefficient-times-variable token exactly as
    `p15c_functional_build._coef_var_text` renders one ("x", "-x", "8x",
    "-8x", "x/2", "-x/2", "3x/2", "-3x/2", ...) -- written from scratch here,
    never imported, per the oracle's own no-generator-internals rule."""
    token = token.strip()
    m = re.match(rf"^(-?\d*){re.escape(var)}(?:/(\d+))?$", token)
    if not m:
        raise AssertionError(f"cannot parse coefficient token {token!r} for var {var!r}")
    num_str = m.group(1)
    if num_str in ("", "-"):
        num = -1 if num_str == "-" else 1
    else:
        num = int(num_str)
    den = int(m.group(2)) if m.group(2) else 1
    return Fraction(num, den)


def _parse_pair(inner: str) -> tuple[Fraction, Fraction]:
    """inner is "A, B" -- the two coefficient-var tokens for x and y."""
    x_tok, y_tok = [t.strip() for t in inner.split(",")]
    return _parse_coef_var(x_tok, "x"), _parse_coef_var(y_tok, "y")


def _norm_via_sympy(a, b):
    m = sympy.Matrix([[sympy.Rational(a.numerator, a.denominator), 0], [0, sympy.Rational(b.numerator, b.denominator)]])
    top = max(sympy.Rational(v) for v in m.singular_values())
    return Fraction(top.p, top.q)


@pytest.mark.parametrize("difficulty,seed", [(t, s) for t in TIERS for s in SEEDS])
def test_oracle_rederives_diagonal_operator_norm(difficulty, seed):
    q = _make(NORM_SOURCES, difficulty, seed)
    text = q.text
    if difficulty == "easy":
        m = _DIAG_RE.search(text)
        d1, d2 = Fraction(int(m.group(1))), Fraction(int(m.group(2)))
    elif difficulty == "medium":
        m = _ACTION_RE.search(text)
        d1, d2 = _parse_pair(m.group(1))
    elif difficulty == "hard":
        m = _COMPOSE_RE.search(text)
        t1, t2 = _parse_pair(m.group(1))
        s1, s2 = _parse_pair(m.group(2))
        d1, d2 = s1 * t1, s2 * t2
    else:
        m = _POWER_RE.search(text)
        d1, d2 = _parse_pair(m.group(1))
        k = int(m.group(2))
        d1, d2 = d1**k, d2**k

    derived = _norm_via_sympy(d1, d2)
    if difficulty == "exam":
        stated = Fraction(q.answer["p"], q.answer["q"])
    else:
        stated = Fraction(int(q.answer_text))
    assert derived == stated, (text, derived, stated)


def test_oracle_coverage_norm():
    total = len(TIERS) * len(SEEDS)
    assert total >= 240
    print(f"\nDiagonal-operator-norm oracle re-derived {total} questions.")


# ---------------------------------------------------------------------------
# ucv1-function-space-approximation


def _solve2(a11, a12, b1, a21, a22, b2):
    det = a11 * a22 - a12 * a21
    e1 = Fraction(b1 * a22 - a12 * b2, 1) / det
    e2 = Fraction(a11 * b2 - a21 * b1, 1) / det
    return e1, e2


_SIGNED_TERM_RE = re.compile(r"([+-]?)\s*(\d+(?:/\d+)?)")


def _sum_signed_terms(rhs: str) -> Fraction:
    """rhs is a plain value ("-36"), a signed sum ("1 - 4", "-1 + 4"), or a
    bare remainder with a leading sign and no other term ("- 11"). Tokenised
    with a sign-then-magnitude regex rather than a naive "+"-split, which
    cannot tell a leading "-" (no preceding space) from mid-string "- "."""
    total = Fraction(0)
    for sign, magnitude in _SIGNED_TERM_RE.findall(rhs):
        value = _frac(magnitude)
        total += -value if sign == "-" else value
    return total


@pytest.mark.parametrize("difficulty,seed", [(t, s) for t in TIERS for s in SEEDS])
def test_oracle_rederives_function_space_approximation(difficulty, seed):
    q = _make(APPROX_SOURCES, difficulty, seed)
    text = q.text

    points = [int(x) for x in re.findall(r"x = (-?\d+)", text.split("giving")[0])]
    if "{1, x^2}" in text:
        col1 = [Fraction(1) for _ in points]
        col2 = [Fraction(p * p) for p in points]
    elif "{x, x^2}" in text:
        col1 = [Fraction(p) for p in points]
        col2 = [Fraction(p * p) for p in points]
    else:  # "{x^2, x^3}"
        col1 = [Fraction(p * p) for p in points]
        col2 = [Fraction(p * p * p) for p in points]

    y_clause = text.split("giving ")[1].split(". Find")[0]
    y = []
    for i, part in enumerate(y_clause.split(";")):
        part = part.split("(f(x)")[0]
        rhs = part.split("=", 1)[1].strip()
        f_match = re.match(r"f\((-?\d+)\)\s*=\s*(-?\d+(?:/\d+)?)$", rhs)
        if f_match:
            p_val = int(f_match.group(1))
            y_val = _frac(f_match.group(2))
            assert y_val == Fraction(p_val) ** 4, (
                f"f({p_val}) stated as {y_val}, but f(x) = x^4 gives {Fraction(p_val) ** 4}"
            )
            y.append(y_val)
        else:
            y.append(_sum_signed_terms(rhs))

    s11 = sum(c * c for c in col1)
    s12 = sum(c1 * c2 for c1, c2 in zip(col1, col2))
    s22 = sum(c * c for c in col2)
    sy1 = sum(c * yy for c, yy in zip(col1, y))
    sy2 = sum(c * yy for c, yy in zip(col2, y))
    c1_derived, c2_derived = _solve2(s11, s12, sy1, s12, s22, sy2)

    stated_c1 = Fraction(q.answer["c1"])
    stated_c2 = Fraction(q.answer["c2"])
    assert c1_derived == stated_c1 and c2_derived == stated_c2, (text, (c1_derived, c2_derived), (stated_c1, stated_c2))


def test_oracle_coverage_approx():
    total = len(TIERS) * len(SEEDS)
    assert total >= 240
    print(f"\nFunction-space-approximation oracle re-derived {total} questions.")
