"""Tests `uc2-sturm-liouville-eigenvalues` (topic `s2c3_sturm_liouville`) and its exam chain.

The oracle reads the problem back out of the wording and re-solves it by finite differences: the differential operator
is discretised (ghost points for insulated and Robin ends, wrapped rows for periodic ends) and its eigenvalues found
with numpy, which shares nothing with the closed forms or the bisection used by the generator.
"""

from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np
import pytest
import sympy

import mathgen.generators.s2c3_sturm_liouville_build  # noqa: F401  (registers the topics)
from _s2c3_helpers import (
    assert_exam_grading,
    assert_follow_through,
    assert_key_accepted_and_distractors_rejected,
    assert_text_quality_and_determinism,
    skill_by_id,
    spread,
    tier_question,
)
from _s2c3_oracle import X, expr
from mathgen import make

TOPIC = "s2c3_sturm_liouville"
EXAM = "exd_x_s2c3_sturm_liouville"
SKILL = skill_by_id("uc2-sturm-liouville-eigenvalues")
TIERS = ("easy", "medium", "hard")
_SUB = {c: i for i, c in enumerate("₀₁₂₃₄₅₆₇₈₉")}


def _num(s):
    return float(sympy.N(sympy.sympify(s.replace("π", "*pi").replace("*pi", "*pi") if s != "π" else "pi")))


def _val(s):
    """A coordinate such as 3, -2, 1.25, π, 3π/4 or 2π/3."""
    s = s.strip()
    return float(sympy.N(expr(s.replace("π", "*pi").lstrip("*"))))


class Problem:
    def __init__(self, text):
        m = re.search(r"on \[([^,\]]+), ([^\]]+)\] with ", text)
        self.a, self.b = _val(m.group(1)), _val(m.group(2))
        self.ell = self.b - self.a
        cond = re.search(r"with (.*?)\.(?: |$)", text[m.start() :]).group(1)
        self.robin = None
        if "=" in cond and re.search(r"y\(.*?\) = y\(", cond):
            self.kind = "PER"
        elif "+" in cond:
            self.kind = "ROBIN"
            self.robin = float(re.search(r"\+ ([\d.]*)y\(", cond).group(1) or 1)
        else:
            left, right = cond.split(" and ")
            self.kind = ("D" if "′" not in left else "N") + ("D" if "′" not in right else "N")

    def eigenvalues(self, count, cells=700):
        n = cells
        h = self.ell / n
        if self.kind == "PER":
            A = np.zeros((n, n))
            for i in range(n):
                A[i, i] = 2
                A[i, (i - 1) % n] = -1
                A[i, (i + 1) % n] = -1
            return np.sort(np.linalg.eigvalsh(A / h**2))[:count]
        left = self.kind[0] if self.kind != "ROBIN" else "D"
        right = self.kind[1] if self.kind != "ROBIN" else "R"
        lo = 1 if left == "D" else 0
        hi = n - 1 if right == "D" else n
        idx = list(range(lo, hi + 1))
        A = np.zeros((len(idx), len(idx)))
        for r, i in enumerate(idx):
            A[r, r] = 2
            for j in (i - 1, i + 1):
                if lo <= j <= hi:
                    A[r, j - lo] -= 1
                elif j < 0:
                    A[r, r + 1 - 0] -= 1  # ghost: y_{-1} = y_1
                elif j > n and right in ("N",):
                    A[r, r - 1] -= 1
                elif j == n + 1:
                    if right == "N":
                        A[r, r - 1] -= 1
                    else:  # Robin: y_{n+1} = y_{n-1} - 2 h c y_n
                        A[r, r - 1] -= 1
                        A[r, r] += 2 * h * self.robin
        vals = np.linalg.eigvals(A / h**2)
        return np.sort(vals.real)[:count]

    def eigenvector(self, which, cells=700):
        raise NotImplementedError


def _index(text, marker):
    return _SUB[re.search(marker + r"([₀₁₂₃₄₅₆₇₈₉])", text).group(1)]


def _close(got, want, tol=0.004):
    return abs(got - want) <= tol * (1 + abs(want))


def test_skill_wiring():
    assert SKILL.answer == "number"
    assert [(s.topic, s.difficulty) for s in SKILL.sources] == [
        (TOPIC, "easy"),
        (TOPIC, "medium"),
        (TOPIC, "hard"),
        (EXAM, "exam"),
    ]
    assert SKILL.sources[-1].stages == ["exam"]
    assert "ucv1-wave-equation-separation" in SKILL.prerequisites
    assert SKILL.year_level == ("university",)


def test_easy_matches_finite_differences():
    for seed in range(30):
        q = tier_question(TOPIC, "easy", seed)
        p = Problem(q.text)
        assert p.kind in ("DN", "ND")
        n = _index(q.text, r"Find λ")
        assert _close(p.eigenvalues(n)[n - 1], float(q.answer)), (seed, q.text, q.answer)


def test_medium_covers_indexing_and_normalised_values():
    kinds = Counter()
    for seed in range(50):
        q = tier_question(TOPIC, "medium", seed)
        p = Problem(q.text)
        if "Find λ" in q.text:
            n = _index(q.text, r"Find λ")
            assert p.kind in ("NN", "PER")
            assert _close(p.eigenvalues(n)[n - 1], float(q.answer)), (seed, q.text, q.answer)
            kinds[p.kind] += 1
        else:
            n = _index(q.text, r"Let φ")
            lam = p.eigenvalues(n)[n - 1]
            k = math.sqrt(lam)
            x0 = _val(re.search(r"φ[₀₁₂₃₄₅₆₇₈₉]\(([^)]*)\), correct", q.text).group(1))
            use_sin = "sin" in q.text.split("positive multiple of")[1]
            trig = np.sin if use_sin else np.cos
            xs = np.linspace(p.a, p.b, 200001)
            norm = math.sqrt(np.trapezoid(np.square(trig(k * (xs - p.a))), xs))
            want = float(trig(k * (x0 - p.a))) / norm
            assert _close(want, float(q.answer), 0.006), (seed, q.text, q.answer, want)
            kinds["phi"] += 1
    assert kinds["NN"] >= 5 and kinds["PER"] >= 5 and kinds["phi"] >= 10


def test_hard_robin_and_expansion():
    kinds = Counter()
    for seed in range(40):
        q = tier_question(TOPIC, "hard", seed)
        if "Σ cₙφₙ" in q.text:
            ends = re.search(r"with (y.*?) and (y.*?), with leading", q.text)
            kind = ("D" if "′" not in ends.group(1) else "N") + ("D" if "′" not in ends.group(2) else "N")
            L = float(re.search(r"on \[0, (\d+)\] with", q.text).group(1))
            n = _index(q.text, r"Find c")
            f = expr(re.search(r"f\(x\) = (.*?) as f\(x\)", q.text).group(1).replace("²", "**2"))
            p = Problem(
                q.text.replace(
                    "Let φₙ be the eigenfunction for the n-th smallest eigenvalue of y″ + λy = 0",
                    "Consider the eigenvalue problem y″ + λy = 0",
                )
            )
            lam = p.eigenvalues(n)[n - 1]
            k = math.sqrt(lam)
            trig = np.sin if kind == "DN" else np.cos
            xs = np.linspace(0, L, 100001)
            fv = sympy.lambdify(X, f, "numpy")(xs) * np.ones_like(xs)
            want = np.trapezoid(fv * trig(k * xs), xs) / np.trapezoid(trig(k * xs) ** 2, xs)
            assert _close(want, float(q.answer), 0.004), (seed, q.text, q.answer, want)
            kinds["expansion"] += 1
        else:
            p = Problem(q.text)
            assert p.kind == "ROBIN"
            n = _index(q.text, r"Find λ")
            assert _close(p.eigenvalues(n, cells=1500)[n - 1], float(q.answer), 0.004), (seed, q.text, q.answer)
            kinds["robin"] += 1
    assert kinds["expansion"] >= 10 and kinds["robin"] >= 10


@pytest.mark.parametrize("tier", TIERS)
def test_key_accepted_and_every_distractor_rejected(tier):
    assert_key_accepted_and_distractors_rejected(SKILL, TOPIC, tier, minimum=3)


@pytest.mark.parametrize("tier", TIERS)
def test_text_quality_and_determinism(tier):
    assert_text_quality_and_determinism(TOPIC, tier)


@pytest.mark.parametrize("tier", TIERS)
def test_answer_space(tier):
    spread(TOPIC, tier, minimum_answers=20, minimum_texts=50)


def _exam_n(q):
    return _index(q.parts[1].prompt, r"Hence find λ")


def test_exam_shape_and_keys():
    assert_exam_grading(SKILL, EXAM, [2, 2, 3, 3], ["number"] * 4)
    q = make(EXAM, seed=2, difficulty="exam")
    assert [p.follow_through is None for p in q.parts] == [True, False, True, False]
    p = Problem(q.stem)
    eig = p.eigenvalues(5)
    for seed in range(40):
        q = make(EXAM, seed=seed, difficulty="exam")
        n = _exam_n(q)
        assert _close(eig[0], float(q.parts[0].answer))
        assert _close(eig[n - 1], float(q.parts[1].answer))
        k = math.sqrt(eig[n - 1])
        x0 = _val(re.search(r"φ[₀₁₂₃₄₅₆₇₈₉]\(([^)]*)\), correct", q.parts[2].prompt).group(1))
        xs = np.linspace(0, math.pi, 200001)
        phi = math.sin(k * x0) / math.sqrt(np.trapezoid(np.sin(k * xs) ** 2, xs))
        assert _close(phi, float(q.parts[2].answer), 0.006), seed
        f = 1.0 if "f(x) = 1 " in q.parts[3].prompt else xs
        want = np.trapezoid(f * np.sin(k * xs), xs) / np.trapezoid(np.sin(k * xs) ** 2, xs)
        assert _close(want, float(q.parts[3].answer), 0.004), seed


def _wrong_first(q):
    return "0.3000"


def test_exam_follow_through_from_a_wrong_lambda_1():
    assert_follow_through(
        SKILL,
        EXAM,
        "b",
        {"a": _wrong_first},
        lambda q, r: f"{(2 * _exam_n(q) - 1) ** 2 * float(r['a']):.4f}",
    )


def test_exam_follow_through_into_the_coefficient():
    def expected(q, replies):
        n = _exam_n(q)
        b = float(replies["b"])
        if "f(x) = x " in q.parts[3].prompt:
            return f"{2 * (-1) ** (n + 1) / (b * math.pi):.3f}"
        return f"{2 / (math.sqrt(b) * math.pi):.3f}"

    assert_follow_through(SKILL, EXAM, "d", {"b": "5.0000"}, expected)
