"""Shared checks for the first-year linear algebra skills (`tests/test_fy1la_*.py`).

Not a test module: it holds the helpers every one of those files uses, so each file can be about one skill's oracle,
the independent re-derivation that reads the numbers out of the rendered question and never imports the generator.
"""

from __future__ import annotations

import re
from collections import Counter

import sympy
from sympy.parsing.sympy_parser import (
    implicit_multiplication,
    parse_expr,
    split_symbols,
    standard_transformations,
)

from mathgen import make
from mathgen.engine import grade as skill_grade
from mathgen.grading import grade
from mathgen.misconceptions import CATALOGUE
from mathgen.skills import load_skills
from mathgen.text_quality import find_text_defects

SEEDS = range(40)
TIERS = ("easy", "medium", "hard")
_TRANSFORMS = standard_transformations + (split_symbols, implicit_multiplication)
_SKILLS = {s.id: s for s in load_skills()[1]}


def skill(skill_id):
    return _SKILLS[skill_id]


def parse(text: str, **local):
    """A SymPy expression from stem text such as "2x + ky": implicit products are read, so "ky" is k*y."""
    return parse_expr(text, transformations=_TRANSFORMS, local_dict=local)


def vector(text: str):
    """The entries of "(1, -2, k)" as SymPy expressions."""
    return [parse(e, k=sympy.Symbol("k")) for e in text.strip().strip("()").split(",")]


def vectors_in(text: str, names):
    """{name: entries} for "u = (..)" patterns in `text`."""
    out = {}
    for name in names:
        m = re.search(rf"\b{name} = (\([^)]*\))", text)
        out[name] = vector(m.group(1))
    return out


def matrix_in(text: str, name: str = "[A-Za-z]"):
    """The first "X = [[..], [..]]" matrix after a one-letter name, as a SymPy matrix."""
    m = re.search(rf"\b{name}\s*=\s*(\[\[.*?\]\])", text)
    rows = re.findall(r"\[([^\[\]]+)\]", m.group(1))
    return sympy.Matrix([[parse(e, k=sympy.Symbol("k")) for e in r.split(",")] for r in rows])


def system_in(text: str):
    """(coefficient matrix, right-hand side, unknowns) of the semicolon-separated equations in `text`."""
    flat = re.sub(r"x(\d)", r"x_\1", text)
    eqs = re.findall(r"(?<![A-Za-z])([-0-9kxyz_ +]*[xyz][-0-9kxyz_ +]*?) = (-?\d+)", flat)
    names = sorted({n for lhs, _ in eqs for n in re.findall(r"x_\d|(?<![a-z_])[xyz](?![a-z_\d])", lhs)})
    symbols = [sympy.Symbol(n) for n in names]
    local = dict(zip(names, symbols), k=sympy.Symbol("k"))
    exprs = [parse(lhs.strip(), **local) - sympy.Integer(rhs) for lhs, rhs in eqs]
    a, b = sympy.linear_eq_to_matrix(exprs, symbols)
    return a, b, symbols


def column(rows) -> sympy.Matrix:
    """A matrix answer value (list of rows of ints or {p, q}) as a SymPy matrix."""

    def entry(e):
        return sympy.Rational(e["p"], e["q"]) if isinstance(e, dict) else sympy.Integer(e)

    return sympy.Matrix([[entry(e) for e in row] for row in rows])


def text_matrix(text: str) -> sympy.Matrix:
    """A matrix or vector answer text such as "[[1, 2], [3, 4]]" or "(1, -2, 3)" as a SymPy matrix."""
    if text.startswith("("):
        return sympy.Matrix([sympy.Rational(e.strip()) for e in text.strip("()").split(",")])
    rows = re.findall(r"\[([^\[\]]+)\]", text)
    return sympy.Matrix([[sympy.Rational(e.strip()) for e in r.split(",")] for r in rows])


# -- generic checks ----------------------------------------------------------------------------------------------


def check_distractors(q, kind=None):
    """Every distractor is tagged with a catalogue id, is not the key, and is refused by the real grader."""
    kind = kind or q.answer_kind
    assert len(q.distractors) >= 2 and len(set(q.distractors)) == len(q.distractors)
    for d in q.distractors:
        assert d != q.answer_text
        assert q.option_notes[d] in CATALOGUE, d
        assert not grade(kind, q, d).correct, f"{d!r} is accepted as the answer to {q.text!r}"


def check_practice(topic, tier, oracle, seeds=SEEDS):
    """The key equals the oracle's re-derivation, grades correct, and every distractor grades wrong."""
    for seed in seeds:
        q = make(topic, seed=seed, difficulty=tier)
        oracle(q)
        assert grade(q.answer_kind, q, q.answer_text).correct, (topic, tier, seed)
        check_distractors(q)
        assert find_text_defects(q) == [], (topic, tier, seed, find_text_defects(q))
        assert q.to_json() == make(topic, seed=seed, difficulty=tier).to_json()


def exam(topic, seed):
    return make(topic, seed=seed, difficulty="exam")


def check_exam(topic, skill_id, oracle, marks, seeds=range(30)):
    """The exam chain: `oracle(q)` re-derives every part from the printed stem and prompts; the whole chain earns full
    marks through the skill's grader; every part's distractors are refused; a wrong part costs exactly its marks."""
    sk = skill(skill_id)
    for seed in seeds:
        q = exam(topic, seed)
        oracle(q)
        assert [p.marks for p in q.parts] == marks and 3 <= len(q.parts) <= 4
        assert find_text_defects(q) == [], (topic, seed, find_text_defects(q))
        assert q.to_json() == exam(topic, seed).to_json()
        full = {p.label: p.answer_text for p in q.parts}
        result = skill_grade(sk, q, full)
        assert result.correct and result.marks == result.max_marks == sum(marks)
        for p in q.parts:
            assert grade(p.answer_kind, p, p.answer_text).correct
            assert len(p.distractors) >= 2 and len(set(p.distractors)) == len(p.distractors)
            for d in p.distractors:
                assert d != p.answer_text and p.option_notes[d] in CATALOGUE
                assert not grade(p.answer_kind, p, d).correct, f"{topic} seed {seed}: part {p.label} accepts {d!r}"
            if p.sensible_range is not None:
                lo, hi = p.sensible_range
                assert (lo is None or p.answer > lo) and (hi is None or p.answer < hi)
        # a wrong answer to the last part (which no later part follows through from) costs exactly its marks
        last = q.parts[-1]
        wrong = dict(full)
        wrong[last.label] = last.distractors[0]
        partial = skill_grade(sk, q, wrong)
        assert partial.marks == sum(marks) - last.marks and not partial.correct


def check_follow_through(topic, skill_id, label, wrong_reply_for, seeds=range(20)):
    """A wrong earlier answer, used correctly by the part named `label` (which has `follow_through`), keeps that part's
    marks. `wrong_reply_for(q)` returns (earlier label, wrong reply, the reply to `label` that follows from it)."""
    sk = skill(skill_id)
    done = 0
    for seed in seeds:
        q = exam(topic, seed)
        earlier, wrong, follows = wrong_reply_for(q)
        replies = {p.label: p.answer_text for p in q.parts}
        replies[earlier] = wrong
        replies[label] = follows
        result = skill_grade(sk, q, replies)
        own = {p.label: p for p in q.parts}
        assert result.parts[label].correct, (seed, replies)
        assert result.parts[label].follow_through
        assert not result.parts[earlier].correct
        assert result.marks == sum(p.marks for p in q.parts) - own[earlier].marks
        done += 1
    assert done == len(list(seeds))


def check_variety(topic, difficulty, minimum=60, seeds=range(200)):
    texts = Counter(make(topic, seed=s, difficulty=difficulty).text for s in seeds)
    assert len(texts) >= minimum, len(texts)


def check_skill(skill_id, practice, exam_topic, answer, extra_exam_topics=()):
    sk = skill(skill_id)
    assert sk.status == "active" and sk.answer == answer
    tiers = [(s.topic, s.difficulty) for s in sk.sources]
    for t in TIERS:
        assert (practice, t) in tiers
    assert (exam_topic, "exam") in tiers
    assert sk.sources[-1].stages == ["exam"]
    ids = {m.id for m in sk.misconceptions}
    assert ids <= set(CATALOGUE)
    used = set()
    for topic, tier in [(practice, t) for t in TIERS] + [(exam_topic, "exam")]:
        for seed in range(30):
            q = make(topic, seed=seed, difficulty=tier)
            used |= set(q.option_notes.values())
            for p in q.parts:
                used |= set(p.option_notes.values())
    assert used <= ids, used - ids
