"""Independent oracle for f10-real-number-applied-problems.

Imports NOTHING from `mathgen.generators.p18a_real_number_applied_build` --
only `mathgen.make` and `mathgen.skills.load_skills`. Parses each rendered
`question.text` with its own regexes and recomputes the answer with its own
sympy formulas, then compares against `question.answer_text` (never against
the generator's own internal `answer`/misconception variables).

Seed count is `N_SEEDS` (default 250 per tier -> 1000 questions over the four
tiers), overridable by `P18A_ORACLE_SEEDS`.

Per-template matrix (every prompt shape this oracle recognises, and which
formula it applies):

| tier  | template opener                              | formula                          |
|-------|-----------------------------------------------|-----------------------------------|
| easy  | "A running track is shaped like a rectangle"   | 2*L + W + (W/2)*pi                |
| easy  | "A garden bed is a rectangle"                  | 2*L + W + (W/2)*pi                |
| easy  | "A swimming lane is bounded by"                | 2*L + W + (W/2)*pi                |
| medium| "A path runs along the diagonal"               | p + q + sqrt(p**2 + q**2)         |
| medium| "Fencing is needed along the diagonal"         | p + q + sqrt(p**2 + q**2)         |
| medium| "A cable runs along the diagonal"              | p + q + sqrt(p**2 + q**2)         |
| hard  | (all three hard wordings, distinguished only   | round(2*L + (sqrt(a**2+b**2)/2)   |
|       | by carrier phrase, same L/a/b/d slots)         | *pi, 2)                            |
| exam  | (both exam wordings)                           | L = R / 2                         |
"""

import os
import re
from fractions import Fraction

import pytest
import sympy

from mathgen.skills import load_skills
from mathgen import make

N_SEEDS = int(os.environ.get("P18A_ORACLE_SEEDS", "250"))

_EASY_L_W_RE = re.compile(
    r"rectangle (\d+) m by (\d+) m|"
    r"two straight sides of length (\d+) m, one straight end of length (\d+) m"
)

_MEDIUM_PQ_RE = re.compile(r"rectangular (?:park|block|field) (\d+) m by (\d+) m")

_HARD_L_RE = re.compile(r"each ([\d.]+) m long")
_HARD_AB_RE = re.compile(r"rectangular plot (\d+) m by (\d+) m")
_HARD_D_RE = re.compile(r"rounded to (\d+) decimal places")

_EXAM_C_RE = re.compile(r"radius (\d+) m")
_EXAM_R_RE = re.compile(r"is (?:stated to be|exactly) (\S+) \+ \d+·π metres")


def _skill_sources(skill_id):
    _, skills = load_skills()
    skill = next(s for s in skills if s.id == skill_id)
    return skill, [(src.topic, src.difficulty, src.params or {}) for src in skill.sources]


def _oracle_easy(text: str) -> sympy.Expr:
    m = _EASY_L_W_RE.search(text)
    assert m, f"easy: could not parse L/W from {text!r}"
    if m.group(1) is not None:
        L, W = int(m.group(1)), int(m.group(2))
    else:
        L, W = int(m.group(3)), int(m.group(4))
    return sympy.Integer(2 * L + W) + sympy.Rational(W, 2) * sympy.pi


def _oracle_medium(text: str) -> sympy.Expr:
    m = _MEDIUM_PQ_RE.search(text)
    assert m, f"medium: could not parse p/q from {text!r}"
    p, q = int(m.group(1)), int(m.group(2))
    return sympy.Integer(p + q) + sympy.sqrt(p * p + q * q)


def _oracle_hard(text: str) -> str:
    lm = _HARD_L_RE.search(text)
    abm = _HARD_AB_RE.search(text)
    dm = _HARD_D_RE.search(text)
    assert lm and abm and dm, f"hard: could not parse L/a/b/d from {text!r}"
    L = Fraction(lm.group(1))
    a, b = int(abm.group(1)), int(abm.group(2))
    d = int(dm.group(1))
    r = sympy.sqrt(a * a + b * b) / 2
    total = sympy.Rational(L.numerator, L.denominator) * 2 + r * sympy.pi
    val = Fraction(str(total.evalf(40)))
    scaled = val * (10**d)
    from decimal import ROUND_HALF_UP, Decimal

    q_scaled = Decimal(scaled.numerator) / Decimal(scaled.denominator)
    rounded_int = int(q_scaled.quantize(Decimal("1"), rounding=ROUND_HALF_UP))
    sign = "-" if rounded_int < 0 else ""
    digits = str(abs(rounded_int)).zfill(d + 1)
    whole, frac = digits[:-d], digits[-d:]
    return f"{sign}{whole}.{frac}"


def _oracle_exam(text: str) -> sympy.Expr:
    cm = _EXAM_C_RE.search(text)
    rm = _EXAM_R_RE.search(text)
    assert cm and rm, f"exam: could not parse C/R from {text!r}"
    R = sympy.Rational(rm.group(1))
    return R / 2


_ORACLE = {"easy": _oracle_easy, "medium": _oracle_medium, "hard": _oracle_hard, "exam": _oracle_exam}


class TestOracle:
    @pytest.mark.parametrize("tier", ["easy", "medium", "hard", "exam"])
    def test_oracle_agrees_with_generator(self, tier):
        skill, sources = _skill_sources("f10-real-number-applied-problems")
        matches = [s for s in sources if s[1] == tier]
        assert matches, f"no source for tier {tier!r}"
        topic, difficulty, params = matches[0]
        checked = 0
        for seed in range(N_SEEDS):
            q = make(topic, seed=seed, difficulty=difficulty, **params)
            expected = _ORACLE[tier](q.text)
            if tier in ("easy", "medium"):
                assert sympy.simplify(sympy.sympify(q.answer_text) - expected) == 0, (
                    f"{tier} seed {seed}: oracle {expected} != generator answer {q.answer_text!r} for {q.text!r}"
                )
            elif tier == "hard":
                assert q.answer_text == expected, (
                    f"hard seed {seed}: oracle {expected!r} != generator answer {q.answer_text!r} for {q.text!r}"
                )
            else:  # exam
                assert sympy.sympify(q.answer_text) == expected, (
                    f"exam seed {seed}: oracle {expected} != generator answer {q.answer_text!r} for {q.text!r}"
                )
            checked += 1
        assert checked == N_SEEDS
        print(f"{tier}: oracle re-derived and matched {checked}/{N_SEEDS} questions")


class _FakeQuestion:
    """Minimal stand-in carrying only what the oracle reads (`text`,
    `answer_text`) -- never a real `mathgen.core.Question`, so a corrupted
    `answer_text` can be injected without touching the generator at all."""

    def __init__(self, text: str, answer_text: str):
        self.text = text
        self.answer_text = answer_text


class TestOracleCatchesCorruption:
    """Red-first proof: the oracle's own comparison genuinely fails when the
    question's `answer_text` is corrupted, run through the SAME assertion
    logic `TestOracle.test_oracle_agrees_with_generator` uses (not a looser
    inequality check) -- proof that the check has teeth, not just that two
    values happen to differ."""

    def test_easy_oracle_rejects_wrong_answer(self):
        real = make("p18a_real_number_applied", seed=0, difficulty="easy")
        corrupted_text = real.answer_text
        fake = _FakeQuestion(real.text, str(sympy.sympify(corrupted_text) + 1))
        expected = _oracle_easy(fake.text)
        with pytest.raises(AssertionError):
            assert sympy.simplify(sympy.sympify(fake.answer_text) - expected) == 0, "corrupted"

    def test_medium_oracle_rejects_wrong_answer(self):
        real = make("p18a_real_number_applied", seed=0, difficulty="medium")
        fake = _FakeQuestion(real.text, str(sympy.sympify(real.answer_text) + 1))
        expected = _oracle_medium(fake.text)
        with pytest.raises(AssertionError):
            assert sympy.simplify(sympy.sympify(fake.answer_text) - expected) == 0, "corrupted"

    def test_hard_oracle_rejects_wrong_answer(self):
        real = make("p18a_real_number_applied", seed=0, difficulty="hard")
        wrong = real.answer_text[:-1] + ("0" if real.answer_text[-1] != "0" else "1")
        fake = _FakeQuestion(real.text, wrong)
        expected = _oracle_hard(fake.text)
        with pytest.raises(AssertionError):
            assert fake.answer_text == expected, "corrupted"

    def test_exam_oracle_rejects_wrong_answer(self):
        real = make("p18a_real_number_applied", seed=0, difficulty="exam")
        fake = _FakeQuestion(real.text, str(sympy.sympify(real.answer_text) + 1))
        expected = _oracle_exam(fake.text)
        with pytest.raises(AssertionError):
            assert sympy.sympify(fake.answer_text) == expected, "corrupted"
