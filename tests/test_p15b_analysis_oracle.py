"""Independent oracle for the `p15b_analysis` questions: it re-derives each answer from the rendered question
text.

Parses ONLY `question.text`/`question.answer_text`/`question.distractors`
(via `mathgen.grading.mcq_options`) -- never anything from
`mathgen.generators.p15b_analysis_build` -- and re-derives the answer from
scratch: parses the sequence's stated terms as exact Fraction values, parses
each row's stated (epsilon, N), recomputes the tail spread a_N - a_last
directly from the PARSED terms (never trusting the generator's own spread
value), and independently decides which row (if any) is inconsistent.
"""

import re
from fractions import Fraction

import pytest

from mathgen.grading.typed import mcq_options
from mathgen.skills import load_skills
from mathgen import make

SKILL_ID = "ucv1-completeness-cauchy-check"
SEEDS = range(1000)
TIERS = ["easy", "medium", "hard", "exam"]


def _skill_sources(skill_id):
    _domains, skills = load_skills()
    skill = next(s for s in skills if s.id == skill_id)
    return {src.difficulty: src for src in skill.sources}


SOURCES = _skill_sources(SKILL_ID)


def _parse_fraction(text):
    text = text.strip()
    if "/" in text:
        num, den = text.split("/")
        return Fraction(int(num), int(den))
    return Fraction(int(text))


_TERM_RE = re.compile(r"a_(\d+) = (\d+(?:/\d+)?)")
_ROW_RE = re.compile(r"Row (\d+): epsilon = ([\d/]+(?: \+ [\d/]+)?), N = (\d+) \(claims")


def _parse_terms(text):
    terms = {}
    for m in _TERM_RE.finditer(text):
        terms[int(m.group(1))] = _parse_fraction(m.group(2))
    return terms


def _parse_rows(text):
    rows = []
    for m in _ROW_RE.finditer(text):
        row_num = int(m.group(1))
        eps_text = m.group(2)
        if " + " in eps_text:
            a, b = eps_text.split(" + ")
            epsilon = _parse_fraction(a) + _parse_fraction(b)
        else:
            epsilon = _parse_fraction(eps_text)
        n_index = int(m.group(3))
        rows.append((row_num, epsilon, n_index))
    return rows


@pytest.mark.parametrize("difficulty,seed", [(t, s) for t in TIERS for s in SEEDS])
def test_oracle_rederives_the_inconsistent_row(difficulty, seed):
    src = SOURCES[difficulty]
    q = make(src.topic, seed=seed, difficulty=src.difficulty, **(src.params or {}))

    terms = _parse_terms(q.text)
    last_index = max(terms)
    last_val = terms[last_index]
    rows = _parse_rows(q.text)
    assert len(rows) == 4, (q.text, rows)

    inconsistent = []
    for row_num, epsilon, n_index in rows:
        spread = terms[n_index] - last_val
        consistent = spread < epsilon
        if not consistent:
            inconsistent.append(row_num)

    if not inconsistent:
        derived = "All rows are consistent."
    else:
        assert len(inconsistent) == 1, (q.text, inconsistent)
        derived = f"Row {inconsistent[0]} is inconsistent."

    assert derived == q.answer_text, (
        f"seed={seed} difficulty={difficulty}\ntext: {q.text}\nderived: {derived}\nstated: {q.answer_text}"
    )
    options = mcq_options(q)
    assert len(options) == 5
    assert q.answer_text in options


def test_oracle_coverage():
    total = len(TIERS) * len(SEEDS)
    assert len(SEEDS) >= 60
    assert total >= 240
    print(f"\nOracle re-derived {total} questions ({len(TIERS)} tiers x {len(SEEDS)} seeds).")
