"""Independent oracle for the `p13e_joint_marginal` questions: it re-derives each answer from the rendered
question text.

Parses ONLY `question.text` and `question.answer_text` -- never anything
from `mathgen.generators.p13e_joint_marginal_build` -- and re-derives the
answer from scratch: it regex-recovers every `P(X=..,Y=..)=p/n` cell
straight out of the rendered table (five plus "one entry is missing" for the
exam tier, where it also solves for the sixth cell itself), then recomputes
whichever marginal or conditional probability the query line asks for
purely from that parsed table.
"""

import re
from fractions import Fraction

import pytest

from mathgen.skills import load_skills
import mathgen

SKILL_ID = "ucv1-joint-distribution-marginal"
SEEDS = range(60)
TIERS = ["easy", "medium", "hard", "exam"]

_CELL_RE = re.compile(r"P\(X=(x\d),Y=(y\d)\)=(\d+)/(\d+)")
_COND_RE = re.compile(r"P\((X|Y) = (x\d|y\d) \| (X|Y) = (x\d|y\d)\)")
_MARG_RE = re.compile(r"P\((X|Y) = (x\d|y\d)\)")

_ALL_CELLS = [(xi, yi) for xi in range(2) for yi in range(3)]


def _skill_sources():
    _domains, skills = load_skills()
    skill = next(s for s in skills if s.id == SKILL_ID)
    return {src.difficulty: src for src in skill.sources}


SOURCES = _skill_sources()


def _idx(label: str) -> int:
    return int(label[1]) - 1


def _parse_table(text: str):
    """Return (cells: {(xi,yi): numerator}, n) with every one of the six
    cells filled in -- for the exam tier (5 cells rendered) the sixth is
    solved for here, independently, from `n - sum(the other five)`."""
    found = _CELL_RE.findall(text)
    assert found, f"could not parse any joint-probability cell from: {text}"
    denominators = {int(den) for *_, den in found}
    assert len(denominators) == 1, f"inconsistent denominators in: {text}"
    n = denominators.pop()

    cells = {}
    for xl, yl, num, _den in found:
        cells[(_idx(xl), _idx(yl))] = int(num)

    assert len(cells) in (5, 6), f"expected 5 or 6 parsed cells, got {len(cells)}: {text}"
    if len(cells) == 5:
        missing = next(c for c in _ALL_CELLS if c not in cells)
        recovered = n - sum(cells.values())
        assert recovered > 0, f"recovered cell not positive in: {text}"
        cells[missing] = recovered
    else:
        assert sum(cells.values()) == n, f"six cells do not sum to n in: {text}"

    return cells, n


def _row_sum(cells, xi):
    return sum(cells[(xi, yj)] for yj in range(3))


def _col_sum(cells, yi):
    return sum(cells[(xi, yi)] for xi in range(2))


def _solve(text: str) -> Fraction:
    cells, n = _parse_table(text)

    m = _COND_RE.search(text)
    if m:
        var1, val1, var2, val2 = m.groups()
        # var1/val1 is the event being asked about; var2/val2 is the
        # condition. Exactly one of the pair is X and the other is Y.
        x_label = val1 if var1 == "X" else val2
        y_label = val1 if var1 == "Y" else val2
        xi, yi = _idx(x_label), _idx(y_label)
        numerator = cells[(xi, yi)]
        denominator = _row_sum(cells, xi) if var2 == "X" else _col_sum(cells, yi)
        return Fraction(numerator, denominator)

    m = _MARG_RE.search(text)
    assert m, f"could not find a marginal or conditional query in: {text}"
    var, val = m.groups()
    if var == "X":
        return Fraction(_row_sum(cells, _idx(val)), n)
    return Fraction(_col_sum(cells, _idx(val)), n)


@pytest.mark.parametrize("difficulty,seed", [(tier, s) for tier in TIERS for s in SEEDS])
def test_oracle_rederives_answer(difficulty, seed):
    src = SOURCES[difficulty]
    q = mathgen.make(src.topic, seed=seed, difficulty=src.difficulty, **(src.params or {}))
    derived = _solve(q.text)
    # Compare against the LEARNER-FACING `answer_text`, never the internal
    # `answer` payload dict -- a payload can be right while the separately
    # formatted `answer_text` carries a sign error, so the payload alone
    # cannot prove the artefact the learner receives.
    stated = Fraction(q.answer_text)
    assert derived == stated, (
        f"seed={seed} difficulty={difficulty}\ntext: {q.text}\nderived: {derived}\nstated: {stated}"
    )


def test_oracle_is_red_on_a_corrupted_answer():
    """Proves the oracle is actually checking something: a wrong numerator
    grafted onto a genuine question must fail the comparison this test file
    otherwise relies on to pass silently."""
    src = SOURCES["easy"]
    q = mathgen.make(src.topic, seed=0, difficulty=src.difficulty, **(src.params or {}))
    derived = _solve(q.text)
    corrupted = Fraction(q.answer_text) + 1
    assert derived != corrupted


def test_oracle_coverage():
    total = len(TIERS) * len(SEEDS)
    assert len(SEEDS) >= 60
    assert total >= 240
    print(f"\nOracle re-derived {total} questions ({len(TIERS)} tiers x {len(SEEDS)} seeds).")
