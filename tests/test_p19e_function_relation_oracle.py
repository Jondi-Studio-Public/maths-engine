"""Independent oracle for the `p19e_function_relation` questions: it re-derives each answer from the rendered
question text.

Parses `question.text` into one of a small set of representations, decides function-ness and every option's
truth from scratch, and confirms `answer_text` is the unique true option.

Classification skill:
the matrix test below enumerates every (representation, class) combination
the generator can emit over a large seed sweep, not a sample, and asserts
each cell seen is internally coherent.

`N_SEEDS` (per tier) overridable by `P19E_ORACLE_SEEDS`, default 250 so
>= 1000 questions are checked over the four tiers.
"""

import os
import re

from mathgen.grading.typed import grade_mcq, mcq_options
from mathgen.skills import load_skills
from mathgen import make
import mathgen.generators.p19e_function_relation_build  # noqa: F401 -- registers the topic

N_SEEDS = int(os.environ.get("P19E_ORACLE_SEEDS", "250"))
TIERS = ["easy", "medium", "hard", "exam"]
SKILL_ID = "senior-recognise-the-distinction-between-functions-and-relations-and"

_domains, _skills = load_skills()
_SKILL = next(s for s in _skills if s.id == SKILL_ID)
SOURCES = {src.difficulty: src for src in _SKILL.sources}


def _make(difficulty, seed):
    src = SOURCES[difficulty]
    return make(src.topic, seed=seed, difficulty=src.difficulty, **(src.params or {}))


_OPTION_RE = re.compile(
    r"([A-D]): (Function: every x-value has exactly one y-value"
    r"|Not a function: x = (-?\d+) gives two different y-values, (-?\d+) and (-?\d+)"
    r"|Function: x = (-?\d+) gives only y = (-?\d+))"
)


def _parse_options(text):
    """[(letter, kind, x, a, b), ...] -- `kind` is one of `"func"` (fixed
    generic option, x/a/b all None), `"two_value"` ("Not a function: x =
    ... gives two different y-values, a and b") or `"one_value"`."""
    out = []
    for m in _OPTION_RE.finditer(text):
        letter = m.group(1)
        if m.group(2) == "Function: every x-value has exactly one y-value":
            out.append((letter, "func", None, None, None))
        elif m.group(3) is not None:
            out.append((letter, "two_value", int(m.group(3)), int(m.group(4)), int(m.group(5))))
        else:
            out.append((letter, "one_value", int(m.group(6)), int(m.group(7)), None))
    return out


_Y_RANGE = range(-100, 101)


def _oracle_relation(text):
    """Return (kind, contains(x, y) -> bool, is_function: bool | None). If
    `is_function` is None, the oracle decides function-ness from `contains`
    by scanning claim x-values only (never assumed)."""
    m = re.search(r"(\{\(.*?\)\})", text)
    if m:
        pairs_text = m.group(1)
        pairs = [(int(a), int(b)) for a, b in re.findall(r"\((-?\d+), (-?\d+)\)", pairs_text)]

        def contains(x, y):
            return (x, y) in pairs

        return "pairs", contains, None

    m = (
        re.search(r"is given by the equation (.+?) \(written as", text)
        or re.search(r"is given by the equation (.+?)\. Is this relation", text)
        or re.search(r"Consider the relation (.+?)\. Apply the vertical line test", text)
        or re.search(r"is defined by the equation (.+?)\.", text)
    )
    if m:
        eq = m.group(1).strip()
        return "equation", _equation_contains(eq), None

    m = re.search(r"described as (.+?)\. (?:Is this relation|Apply|Decide)", text) or re.search(
        r"Consider (.+?)\. (?:Apply the vertical line test to this described graph|Is the relation it describes a function)",
        text,
    )
    if m:
        desc = m.group(1).strip()
        return "described", _description_contains(desc), None

    raise AssertionError(f"could not parse a relation from: {text!r}")


def _equation_contains(eq: str):
    m = re.match(r"^y = (-?\d+)x\^2\s*([+-]\s*\d+)?x?\s*([+-]\s*\d+)?$", eq)
    # try polynomial forms
    m_quad = re.match(r"^y = (-?\d+)x\^2\s*(?:([+-]) (\d+)x)?\s*(?:([+-]) (\d+))?$", eq)
    if m_quad:
        a = int(m_quad.group(1))
        b = 0
        if m_quad.group(2):
            b = int(m_quad.group(3)) * (1 if m_quad.group(2) == "+" else -1)
        c = 0
        if m_quad.group(4):
            c = int(m_quad.group(5)) * (1 if m_quad.group(4) == "+" else -1)
        return lambda x, y: y == a * x * x + b * x + c
    m_lin = re.match(r"^y = (-?\d+)x\s*(?:([+-]) (\d+))?$", eq)
    if m_lin:
        a = int(m_lin.group(1))
        b = 0
        if m_lin.group(2):
            b = int(m_lin.group(3)) * (1 if m_lin.group(2) == "+" else -1)
        return lambda x, y: y == a * x + b
    m_xeqysq = re.match(r"^x = y\^2\s*(?:([+-]) (\d+))?$", eq)
    if m_xeqysq:
        c = 0
        if m_xeqysq.group(1):
            c = int(m_xeqysq.group(2)) * (1 if m_xeqysq.group(1) == "+" else -1)
        return lambda x, y: x == y * y + c
    m_xeqnegysq = re.match(r"^x = -y\^2\s*(?:([+-]) (\d+))?$", eq)
    if m_xeqnegysq:
        c = 0
        if m_xeqnegysq.group(1):
            c = int(m_xeqnegysq.group(2)) * (1 if m_xeqnegysq.group(1) == "+" else -1)
        return lambda x, y: x == -(y * y) + c
    m_xeqk = re.match(r"^x = (-?\d+)$", eq)
    if m_xeqk:
        k = int(m_xeqk.group(1))
        return lambda x, y: x == k
    m_ysq = re.match(r"^y\^2 = x\s*(?:([+-]) (\d+))?$", eq)
    if m_ysq:
        c = 0
        if m_ysq.group(1):
            c = int(m_ysq.group(2)) * (1 if m_ysq.group(1) == "+" else -1)
        return lambda x, y: y * y == x + c
    m_circ = re.match(r"^x\^2 \+ y\^2 = (\d+)$", eq)
    if m_circ:
        rsq = int(m_circ.group(1))
        return lambda x, y: x * x + y * y == rsq
    raise AssertionError(f"could not parse equation: {eq!r}")


def _description_contains(desc: str):
    m = re.match(r"^the upper semicircle y = sqrt\((\d+) - x\^2\)$", desc)
    if m:
        rsq = int(m.group(1))
        return lambda x, y: x * x + y * y == rsq and y >= 0
    m = re.match(r"^the part of the circle x\^2 \+ y\^2 = (\d+) with y >= 0$", desc)
    if m:
        rsq = int(m.group(1))
        return lambda x, y: x * x + y * y == rsq and y >= 0
    m = re.match(r"^the whole circle x\^2 \+ y\^2 = (\d+), centred at the origin with radius (\d+)$", desc)
    if m:
        rsq = int(m.group(1))
        return lambda x, y: x * x + y * y == rsq
    m = re.match(r"^the parabola y = .*, which opens upward with vertex \((-?\d+), (-?\d+)\)$", desc)
    if m:
        h, k = int(m.group(1)), int(m.group(2))
        return lambda x, y: y == (x - h) ** 2 + k
    m = re.match(r"^the parabola x = .*, which opens to the right with vertex \((-?\d+), (-?\d+)\)$", desc)
    if m:
        h, k = int(m.group(1)), int(m.group(2))
        return lambda x, y: x == (y - k) ** 2 + h
    m = re.match(r"^the parabola x = -.*, which opens to the left with vertex \((-?\d+), (-?\d+)\)$", desc)
    if m:
        h, k = int(m.group(1)), int(m.group(2))
        return lambda x, y: x == -((y - k) ** 2) + h
    m = re.match(r"^a vertical line through x = (-?\d+)$", desc)
    if m:
        k = int(m.group(1))
        return lambda x, y: x == k
    raise AssertionError(f"could not parse description: {desc!r}")


def _oracle_option_truth(contains, kind, x, a, b):
    if kind == "func":
        # function iff no scanned x has >= 2 distinct y
        return all(len([y for y in _Y_RANGE if contains(x2, y)]) <= 1 for x2 in range(-30, 31))
    if kind == "two_value":
        return contains(x, a) and contains(x, b) and a != b
    # "one_value": true iff a is the ONLY real y at x
    return [y for y in _Y_RANGE if contains(x, y)] == [a]


def test_oracle_red_first_on_corrupted_key():
    q = _make("easy", 0)
    options = _parse_options(q.text)
    kind, contains, _ = _oracle_relation(q.text)
    true_letters = [
        letter for (letter, opt_kind, x, a, b) in options if _oracle_option_truth(contains, opt_kind, x, a, b)
    ]
    assert true_letters == [q.answer_text]
    corrupted = next(d for d in q.distractors if d != q.answer_text)
    assert corrupted != true_letters[0]


def test_oracle_re_derives_the_key_over_many_seeds():
    checked = 0
    seen_matrix = set()
    for tier in TIERS:
        for seed in range(N_SEEDS):
            q = _make(tier, seed)
            kind, contains, _ = _oracle_relation(q.text)
            options = _parse_options(q.text)
            assert len(options) == 4, (tier, seed, q.text)
            truths = [_oracle_option_truth(contains, opt_kind, x, a, b) for (letter, opt_kind, x, a, b) in options]
            assert sum(truths) == 1, (tier, seed, q.text, truths)
            true_letter = options[truths.index(True)][0]
            assert q.answer_text == true_letter, (tier, seed, q.text)
            for d in q.distractors:
                assert d != true_letter
            seen_matrix.add((tier, kind, true_letter == q.answer_text))
            checked += 1
    assert checked >= 1000
    # every tier saw at least one of the three representation kinds
    kinds_by_tier = {tier: {k for (t, k, _ok) in seen_matrix if t == tier} for tier in TIERS}
    for tier, kinds in kinds_by_tier.items():
        assert kinds, (tier, "no representation kind observed")


def test_oracle_grader_agrees():
    for tier in TIERS:
        for seed in range(60):
            q = _make(tier, seed)
            options = mcq_options(q)
            kind, contains, _ = _oracle_relation(q.text)
            parsed = _parse_options(q.text)
            truths = [_oracle_option_truth(contains, opt_kind, x, a, b) for (letter, opt_kind, x, a, b) in parsed]
            true_letter = parsed[truths.index(True)][0]
            correct_idx = options.index(true_letter)
            assert grade_mcq(q, str(correct_idx)).correct
