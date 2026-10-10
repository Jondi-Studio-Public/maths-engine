"""Independent oracle for senior-use-the-leibniz-notation-delta-x-and-delta.

Parses ONLY `question.text` and `question.answer_text` (never anything from
`mathgen.generators.p9c_leibniz_increments_build`) and re-derives δx and δy
from scratch: parse f(x)'s expression, x1 and x2 out of the question text,
evaluate f at both points with sympy, and subtract.
"""

import re

import pytest
from sympy import Symbol, sympify

import mathgen
from mathgen.skills import load_skills

SKILL_ID = "senior-use-the-leibniz-notation-delta-x-and-delta"
_QUESTION_RE = re.compile(
    r"y = f\(x\) = (.+?)\. x changes from x1 = (-?\d+) to x2 = (-?\d+)\. "
    r"What are δx and δy\?"
)
_ANSWER_RE = re.compile(r"δx = (-?\d+), δy = (-?\d+)")

_X = Symbol("x")


def _to_expr(f_text):
    """f_text is a plain-prose polynomial such as "3x^2 - 5x + 7" or "-x - 4"
    -- insert the multiplication sympify needs between a digit and the
    following "x", and turn "^" into "**"."""
    text = re.sub(r"(?<=\d)x", "*x", f_text)
    text = text.replace("^", "**")
    return sympify(text, locals={"x": _X})


def _solve(text):
    m = _QUESTION_RE.search(text)
    assert m, f"could not parse question text: {text}"
    f_text, x1_str, x2_str = m.group(1), m.group(2), m.group(3)
    x1, x2 = int(x1_str), int(x2_str)
    expr = _to_expr(f_text)
    y1 = int(expr.subs(_X, x1))
    y2 = int(expr.subs(_X, x2))
    return x2 - x1, y2 - y1


def _parse_answer(answer_text):
    m = _ANSWER_RE.fullmatch(answer_text)
    assert m, f"could not parse answer_text: {answer_text}"
    return int(m.group(1)), int(m.group(2))


def _skill_sources():
    _domains, skills = load_skills()
    skill = next(s for s in skills if s.id == SKILL_ID)
    return {src.difficulty: src for src in skill.sources}


SOURCES = _skill_sources()
SEEDS = range(60)


@pytest.mark.parametrize("difficulty,seed", [(tier, s) for tier in ["easy", "medium", "hard"] for s in SEEDS])
def test_oracle_rederives_delta_x_and_delta_y(difficulty, seed):
    src = SOURCES[difficulty]
    q = mathgen.make(src.topic, seed=seed, difficulty=src.difficulty, **(src.params or {}))
    derived = _solve(q.text)
    stated = _parse_answer(q.answer_text)
    assert derived == stated, (
        f"seed={seed} difficulty={difficulty}\ntext: {q.text}\nderived: {derived}\nstated: {stated}"
    )


def test_oracle_coverage():
    tiers = ["easy", "medium", "hard"]
    total = len(tiers) * len(SEEDS)
    assert len(SEEDS) >= 60
    assert total >= 180
    print(f"\nOracle re-derived {total} questions ({len(tiers)} tiers x {len(SEEDS)} seeds).")
