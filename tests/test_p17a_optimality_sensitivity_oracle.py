"""Re-derives, from `question.text` alone, which of the four rendered conclusions is justified -- using SymPy's
own differentiation/`solve` to find each optimum (a genuinely different route from the generator's closed-form
`Fraction(-b, 2*a)` arithmetic) -- and imports NOTHING from
`mathgen.generators.p17a_optimality_sensitivity_build`.

`N_SEEDS` (default 250 per tier, overridable by `P17A_ORACLE_SEEDS`) -- four
tiers, so >= 1000 questions re-derived over a full run.
"""

from __future__ import annotations

import os
import re
from fractions import Fraction

import pytest
from sympy import Symbol, diff, solve, sympify

from mathgen.skills import load_skills
from mathgen import make

SKILL_ID = "ucv1-optimality-sensitivity"
N_SEEDS = int(os.environ.get("P17A_ORACLE_SEEDS", "250"))

_x = Symbol("x")
_y = Symbol("y")

_OPT_RE = re.compile(r"([ABCD]): (.+?)(?=(?:\s+[ABCD]: )|$)")
_LETTER_INDEX = {"A": 0, "B": 1, "C": 2, "D": 3}


def _skill_sources():
    _domains, skills = load_skills()
    skill = next(s for s in skills if s.id == SKILL_ID)
    return {src.difficulty: src for src in skill.sources}


SOURCES = _skill_sources()


def _make(difficulty, seed):
    src = SOURCES[difficulty]
    return make(src.topic, seed=seed, difficulty=src.difficulty, **(src.params or {}))


def _to_expr(text: str, syms: dict):
    py = text.replace("^", "**")
    py = re.sub(r"(\d)([a-zA-Z])", r"\1*\2", py)
    return sympify(py, locals=syms)


def _options(text: str) -> dict:
    return dict(_OPT_RE.findall(text))


def _parse_direction_amount(option_text: str):
    m = re.search(r"(increases|decreases) by (-?\d+(?:/\d+)?)", option_text)
    assert m, option_text
    sign = 1 if m.group(1) == "increases" else -1
    return sign * Fraction(m.group(2))


def _parse_poly_1var(poly_str: str):
    e = _to_expr(poly_str, {"x": _x})
    a = int(e.coeff(_x, 2))
    b = int(e.coeff(_x, 1))
    c = int(e.subs(_x, 0))
    return a, b, c


def _parse_poly_2var(poly_str: str):
    e = _to_expr(poly_str, {"x": _x, "y": _y})
    a1 = int(e.coeff(_x, 2))
    b1 = int(e.coeff(_x, 1))
    a2 = int(e.coeff(_y, 2))
    b2 = int(e.coeff(_y, 1))
    c = int(e.subs({_x: 0, _y: 0}))
    return a1, b1, a2, b2, c


def _minimise_1var(a, b, c):
    e = a * _x**2 + b * _x + c
    xs = solve(diff(e, _x), _x)[0]
    val = e.subs(_x, xs)
    return Fraction(int(xs.p), int(xs.q)), Fraction(int(val.p), int(val.q))


def _perturbation_numbers(text: str, marker_pattern: str, known_old) -> int:
    """Find the perturbation sentence (matching `marker_pattern`, a regex
    fragment covering every stem wording) and return whichever of its two
    non-exponent integers is NOT `known_old`."""
    m = re.search(rf"[^.]*(?:{marker_pattern})[^.]*\.", text)
    assert m, (text, marker_pattern)
    sentence = m.group(0)
    nums = [int(n) for n in re.findall(r"(?<!\^)-?\d+", sentence)]
    candidates = [n for n in nums if n != known_old]
    assert candidates, (sentence, nums, known_old)
    return candidates[0]


# ==========================================================================
# easy: minimiser move under a linear-coefficient perturbation
# ==========================================================================


@pytest.mark.parametrize("tier", ["easy"])
def test_easy_oracle_matches(tier):
    checked = 0
    for seed in range(N_SEEDS):
        q = _make(tier, seed)
        poly_m = re.search(r"f\(x\) = (.+?)(?:,|\s+the minimiser)", q.text)
        a, b, c = _parse_poly_1var(poly_m.group(1))
        b_new = _perturbation_numbers(q.text, r"coefficient of x is|linear coefficient b is", b)
        x_star, _ = _minimise_1var(a, b, c)
        x_star_new, _ = _minimise_1var(a, b_new, c)
        true_delta = x_star_new - x_star

        opts = _options(q.text)
        matches = [letter for letter, text in opts.items() if _parse_direction_amount(text) == true_delta]
        assert matches == [q.answer_text], (seed, q.text, matches, true_delta)
        checked += 1
    assert checked == N_SEEDS


# ==========================================================================
# medium: minimum-value move under a curvature perturbation
# ==========================================================================


@pytest.mark.parametrize("tier", ["medium"])
def test_medium_oracle_matches(tier):
    checked = 0
    for seed in range(N_SEEDS):
        q = _make(tier, seed)
        poly_m = re.search(r"f\(x\) = (.+?)(?:,|\s+the minimiser)", q.text)
        a, b, c = _parse_poly_1var(poly_m.group(1))
        a_new = _perturbation_numbers(q.text, r"coefficient of x\^2 is|curvature coefficient a is", a)
        _, f_star = _minimise_1var(a, b, c)
        _, f_star_new = _minimise_1var(a_new, b, c)
        true_delta = f_star_new - f_star

        opts = _options(q.text)
        matches = [letter for letter, text in opts.items() if _parse_direction_amount(text) == true_delta]
        assert matches == [q.answer_text], (seed, q.text, matches, true_delta)
        checked += 1
    assert checked == N_SEEDS


# ==========================================================================
# hard: two-variable separable, minimum-value move, one part perturbed
# ==========================================================================


@pytest.mark.parametrize("tier", ["hard"])
def test_hard_oracle_matches(tier):
    checked = 0
    for seed in range(N_SEEDS):
        q = _make(tier, seed)
        poly_m = re.search(r"f\(x, y\) = (.+?)(?:,|\s+the minimiser)", q.text)
        a1, b1, a2, b2, c = _parse_poly_2var(poly_m.group(1))
        var_m = re.search(r"coefficient of (x|y)\^2|curvature coefficient of (x|y)", q.text)
        assert var_m, q.text
        var = var_m.group(1) or var_m.group(2)
        old_a = a1 if var == "x" else a2
        new_a = _perturbation_numbers(q.text, rf"coefficient of {var}\^2 is|curvature coefficient of {var} is", old_a)

        e = a1 * _x**2 + b1 * _x + a2 * _y**2 + b2 * _y + c
        crit = solve([diff(e, _x), diff(e, _y)], [_x, _y])
        f_star = e.subs(crit)

        if var == "x":
            e_new = new_a * _x**2 + b1 * _x + a2 * _y**2 + b2 * _y + c
        else:
            e_new = a1 * _x**2 + b1 * _x + new_a * _y**2 + b2 * _y + c
        crit_new = solve([diff(e_new, _x), diff(e_new, _y)], [_x, _y])
        f_star_new = e_new.subs(crit_new)

        true_delta = Fraction(int((f_star_new - f_star).p), int((f_star_new - f_star).q))

        opts = _options(q.text)
        matches = [letter for letter, text in opts.items() if _parse_direction_amount(text) == true_delta]
        assert matches == [q.answer_text], (seed, q.text, matches, true_delta)
        checked += 1
    assert checked == N_SEEDS


# ==========================================================================
# exam: compound crossing verdict + minimum-value move
# ==========================================================================


def _parse_exam_option(option_text: str):
    m = re.search(
        r"minimiser (does not move|moves) past t, and .*"
        r"(increases|decreases) by (-?\d+(?:/\d+)?)",
        option_text,
    )
    assert m, option_text
    crosses = m.group(1) == "moves"
    sign = 1 if m.group(2) == "increases" else -1
    amount = sign * Fraction(m.group(3))
    return crosses, amount


@pytest.mark.parametrize("tier", ["exam"])
def test_exam_oracle_matches(tier):
    checked = 0
    for seed in range(N_SEEDS):
        q = _make(tier, seed)
        poly_m = re.search(r"f\(x\) = (.+?)(?:,|\s+the minimiser)", q.text)
        a, b, c = _parse_poly_1var(poly_m.group(1))
        b_new = _perturbation_numbers(q.text, r"coefficient of x is|linear coefficient b is", b)
        t_m = re.search(r"Let t = (-?\d+(?:/\d+)?)\.", q.text)
        t = Fraction(t_m.group(1))

        x_star, f_star = _minimise_1var(a, b, c)
        x_star_new, f_star_new = _minimise_1var(a, b_new, c)
        true_crosses = (x_star - t) * (x_star_new - t) < 0
        true_delta = f_star_new - f_star

        opts = _options(q.text)
        matches = [letter for letter, text in opts.items() if _parse_exam_option(text) == (true_crosses, true_delta)]
        assert matches == [q.answer_text], (seed, q.text, matches, true_crosses, true_delta)
        checked += 1
    assert checked == N_SEEDS


def test_oracle_is_red_on_a_corrupted_answer():
    """Prove the oracle actually discriminates: force `answer_text` onto a
    wrong letter and confirm the oracle's own matching assertion goes red
    against that corrupted question, not just re-derives the true one."""
    from dataclasses import replace

    q = _make("easy", 0)
    poly_m = re.search(r"f\(x\) = (.+?)(?:,|\s+the minimiser)", q.text)
    a, b, c = _parse_poly_1var(poly_m.group(1))
    b_new = _perturbation_numbers(q.text, r"coefficient of x is|linear coefficient b is", b)
    x_star, _ = _minimise_1var(a, b, c)
    x_star_new, _ = _minimise_1var(a, b_new, c)
    true_delta = x_star_new - x_star
    opts = _options(q.text)
    matches = [letter for letter, text in opts.items() if _parse_direction_amount(text) == true_delta]
    assert matches == [q.answer_text]  # sane on the real question first

    bad_letter = next(l for l in "ABCD" if l != q.answer_text)
    corrupted = replace(q, answer_text=bad_letter)
    with pytest.raises(AssertionError):
        assert matches == [corrupted.answer_text]
