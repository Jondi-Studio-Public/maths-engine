"""Independent oracle for the `p14a_cubic_exp_technology` questions: it re-derives each answer from the rendered
question text.

Parses `question.text` / `question.answer_text` and re-derives the answer from scratch, using only
`mathgen.make` and the rendered text -- never anything from
`mathgen.generators.p14a_cubic_exp_technology_build`. This is a deliverable, not a convenience, and it must
not import the generator it is checking.
"""

from __future__ import annotations

import math
import re

import numpy as np
import pytest

from mathgen import make

TOPIC_A = "p14a_solve_cubic_technology"
TOPIC_B = "p14a_solve_exp_technology"

SEEDS = range(60)


def _parse_cubic_terms(cubic_text: str) -> dict:
    """Parse a displayed cubic like 'x^3 + 5x^2 - 25x - 50' into
    {"x^3": a, "x^2": b, "x": c, "": d}. Never reads generator code."""
    coeffs = {"x^3": 0, "x^2": 0, "x": 0, "": 0}
    s = cubic_text.strip()
    if not s.startswith(("+", "-")):
        s = "+" + s
    for raw in re.findall(r"[+-]\s*[^+-]+", s):
        term = raw.strip()
        sign = -1 if term.startswith("-") else 1
        body = term[1:].strip()
        sym = ""
        for candidate in ("x^3", "x^2", "x"):
            if body.endswith(candidate):
                sym = candidate
                body = body[: -len(candidate)].strip()
                break
        coef = int(body) if body else 1
        coeffs[sym] = sign * coef
    return coeffs


def _round3(value: float) -> str:
    return f"{value:.3f}"


# =============================================================================
# Skill A oracle: p14a_solve_cubic_technology
# =============================================================================


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_oracle_cubic_largest_root_tiers(difficulty):
    """easy/medium/hard all name 'the largest real root' -- re-derive it from
    the displayed cubic via numpy.roots (a full companion-matrix eigenvalue
    solve, independent of however the generator built the coefficients) and
    compare to answer_text at the stated 3 d.p. precision."""
    challenged = []
    for seed in SEEDS:
        q = make(TOPIC_A, seed=seed, difficulty=difficulty)
        m = re.match(r"^(.*?) = 0", q.text)
        assert m, q.text
        coeffs = _parse_cubic_terms(m.group(1))
        a, b, c, d = coeffs["x^3"], coeffs["x^2"], coeffs["x"], coeffs[""]
        roots = np.roots([a, b, c, d])
        real_roots = [r.real for r in roots if abs(r.imag) < 1e-6]
        if not real_roots:
            challenged.append((seed, q.text, "no real root found"))
            continue
        expected = _round3(max(real_roots))
        if expected != q.answer_text:
            challenged.append((seed, q.text, q.answer_text, expected))
    assert not challenged, challenged


def test_oracle_cubic_exam_physical_domain():
    """Exam tier: parse the box dimensions, target volume and stated domain
    bound directly from the prompt text, solve V(x) = x(p-2x)(q-2x) = v0 via
    numpy.roots, and confirm exactly one root lies in the stated (0, c) and
    matches answer_text."""
    pq_pattern = re.compile(r"(\d+) cm by (\d+) cm")
    v0_pattern = re.compile(r"volume\D*?(\d+) cm\^3")
    c_pattern = re.compile(r"0 < x < (\d+(?:\.\d+)?)")
    challenged = []
    for seed in SEEDS:
        q = make(TOPIC_A, seed=seed, difficulty="exam")
        m_pq = pq_pattern.search(q.text)
        m_v0 = v0_pattern.search(q.text)
        m_c = c_pattern.search(q.text)
        assert m_pq and m_v0 and m_c, q.text
        p, qd = int(m_pq.group(1)), int(m_pq.group(2))
        v0 = int(m_v0.group(1))
        c = float(m_c.group(1))
        coeffs = [4.0, -2.0 * (p + qd), float(p * qd), -float(v0)]
        roots = np.roots(coeffs)
        real_roots = [r.real for r in roots if abs(r.imag) < 1e-6]
        in_domain = [r for r in real_roots if 0 < r < c]
        if len(in_domain) != 1:
            challenged.append((seed, q.text, "domain root count != 1", in_domain))
            continue
        expected = _round3(in_domain[0])
        if expected != q.answer_text:
            challenged.append((seed, q.text, q.answer_text, expected))
    assert not challenged, challenged


# =============================================================================
# Skill B oracle: p14a_solve_exp_technology
# =============================================================================

_LINEAR_EXP_RE = re.compile(r"^(-?\d*)x(?:\s*([+-])\s*(\d+))?$")


def _parse_linear_exponent(exponent: str):
    m = _LINEAR_EXP_RE.match(exponent.strip())
    assert m, exponent
    coef_str, sign, m_str = m.groups()
    k = 1 if coef_str in ("", "+") else (-1 if coef_str == "-" else int(coef_str))
    m_val = 0
    if sign is not None:
        m_val = int(m_str) if sign == "+" else -int(m_str)
    return k, m_val


_ADDITIVE_RE = re.compile(r"^Solve (\d+) . e\^\(([^)]+)\) ([+-]) (\d+) = (-?\d+) for x")


@pytest.mark.parametrize("difficulty", ["easy", "medium"])
def test_oracle_exp_additive_tiers(difficulty):
    """easy/medium: parse A, the (possibly linear) exponent, the additive
    constant and the RHS directly out of the prompt text; isolate the
    exponential (subtract c, divide A), take ln, solve the linear exponent
    for x -- re-deriving via `math.log`/`math.exp`, never via the generator."""
    challenged = []
    for seed in SEEDS:
        q = make(TOPIC_B, seed=seed, difficulty=difficulty)
        m = _ADDITIVE_RE.match(q.text)
        assert m, q.text
        A = int(m.group(1))
        exponent = m.group(2)
        sign = 1 if m.group(3) == "+" else -1
        c = sign * int(m.group(4))
        d = int(m.group(5))
        k, m_val = _parse_linear_exponent(exponent)
        target = (d - c) / A
        if target <= 0:
            challenged.append((seed, q.text, "non-positive target for ln"))
            continue
        expected_x = (math.log(target) - m_val) / k
        expected = _round3(expected_x)
        if expected != q.answer_text:
            challenged.append((seed, q.text, q.answer_text, expected))
    assert not challenged, challenged


_HIDDEN_QUAD_RE = re.compile(r"(\d*e\^\(2x\) [+-] \d*e\^x [+-] \d+) = 0")


def _parse_quadratic_in_u_terms(eq_text: str) -> dict:
    """Parse 'Aqe^(2x) + Bqe^x + Cq' into {"e^(2x)": Aq, "e^x": Bq, "": Cq}."""
    coeffs = {"e^(2x)": 0, "e^x": 0, "": 0}
    s = eq_text.strip()
    if not s.startswith(("+", "-")):
        s = "+" + s
    for raw in re.findall(r"[+-]\s*[^+-]+", s):
        term = raw.strip()
        sign = -1 if term.startswith("-") else 1
        body = term[1:].strip()
        sym = ""
        for candidate in ("e^(2x)", "e^x"):
            if body.endswith(candidate):
                sym = candidate
                body = body[: -len(candidate)].strip()
                break
        coef = int(body) if body else 1
        coeffs[sym] = sign * coef
    return coeffs


def test_oracle_exp_hidden_quadratic():
    """hard: substitute u = e^x, solve the parsed quadratic in u via the
    quadratic formula (never the generator's own numpy call), discard
    non-positive u, and take the larger resulting x = ln(u)."""
    challenged = []
    for seed in SEEDS:
        q = make(TOPIC_B, seed=seed, difficulty="hard")
        m = _HIDDEN_QUAD_RE.search(q.text)
        assert m, q.text
        coeffs = _parse_quadratic_in_u_terms(m.group(1))
        Aq, Bq, Cq = coeffs["e^(2x)"], coeffs["e^x"], coeffs[""]
        disc = Bq * Bq - 4 * Aq * Cq
        if disc < 0:
            challenged.append((seed, q.text, "negative discriminant"))
            continue
        sqrt_disc = math.sqrt(disc)
        u1 = (-Bq + sqrt_disc) / (2 * Aq)
        u2 = (-Bq - sqrt_disc) / (2 * Aq)
        positive_u = [u for u in (u1, u2) if u > 0]
        if not positive_u:
            challenged.append((seed, q.text, "no positive u root"))
            continue
        expected_x = math.log(max(positive_u))
        expected = _round3(expected_x)
        if expected != q.answer_text:
            challenged.append((seed, q.text, q.answer_text, expected))
    assert not challenged, challenged


_TRANSCENDENTAL_FORM_A_RE = re.compile(r"e\^x = (\d+) - (-?\d*)x")
_TRANSCENDENTAL_FORM_B_RE = re.compile(r"e\^x \+ (-?\d*)x = (\d+)")
_TRANSCENDENTAL_INTERVAL_RE = re.compile(r"\[(-?\d+), (-?\d+)\]")


def test_oracle_exp_transcendental():
    """exam: parse a, b and the stated interval; re-derive the root via a
    bisection search written independently here (never the generator's own
    bisection/Newton routines)."""
    challenged = []
    for seed in SEEDS:
        q = make(TOPIC_B, seed=seed, difficulty="exam")
        m_interval = _TRANSCENDENTAL_INTERVAL_RE.search(q.text)
        assert m_interval, q.text
        lo, hi = float(m_interval.group(1)), float(m_interval.group(2))
        m = _TRANSCENDENTAL_FORM_A_RE.search(q.text)
        if m:
            a = int(m.group(1))
            b_str = m.group(2)
        else:
            m = _TRANSCENDENTAL_FORM_B_RE.search(q.text)
            assert m, q.text
            b_str = m.group(1)
            a = int(m.group(2))
        b = 1 if b_str in ("", "+") else (-1 if b_str == "-" else int(b_str))

        def f(x, a=a, b=b):
            return math.exp(x) - a + b * x

        flo, fhi = f(lo), f(hi)
        if flo * fhi >= 0:
            challenged.append((seed, q.text, "no sign change at stated interval"))
            continue
        for _ in range(100):
            mid = (lo + hi) / 2
            fmid = f(mid)
            if flo * fmid <= 0:
                hi, fhi = mid, fmid
            else:
                lo, flo = mid, fmid
        expected = _round3((lo + hi) / 2)
        if expected != q.answer_text:
            challenged.append((seed, q.text, q.answer_text, expected))
    assert not challenged, challenged
