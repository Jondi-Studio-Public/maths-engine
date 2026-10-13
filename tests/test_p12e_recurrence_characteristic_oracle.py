"""Independent oracle for the `p12e_recurrence_characteristic` questions: it re-derives each answer from the
rendered question text.

Parses ONLY `question.text` and `question.answer_text` (never anything from
`mathgen.generators.p12e_recurrence_characteristic_build`) and re-derives the
answer from scratch, using cmath/Fraction rather than any helper the
generator itself calls.
"""

import dataclasses
import cmath
import re
from fractions import Fraction

import pytest

import mathgen
from mathgen.skills import load_skills

RECURRENCE_ID = "ucv1-recurrence-two-step"
ROOTS_ID = "ucv1-second-order-characteristic-roots"
SEEDS = range(60)

# "a_n = <expr>" itself appears, verbatim, in every easy/medium/hard wording
# (only the surrounding sentence varies) -- parsing just this substring is
# robust to which phrase() wording was drawn. p is always drawn from [2,5]
# (never 1, always an explicit digit); q can be +/-1 (no explicit digit).
_EXPR_RE = re.compile(r"a_n = (-?)(\d*)a_\(n-1\)\s*([+-])\s*(\d*)a_\(n-2\)")
_A1_A2_RE = re.compile(r"a_1 = (-?\d+) and a_2 = (-?\d+)")
_A2_A3_RE = re.compile(r"a_2 = (-?\d+) and a_3 = (-?\d+)")
_A1_A2_A3_RE = re.compile(r"a_1 = (-?\d+),\s*a_2 = (-?\d+)(?:,|\s+and)\s*a_3 = (-?\d+)")
_TARGET_RE = re.compile(r"a_(\d+)[.?]\s*$")
_PAIR_RE = re.compile(r"p = (-?\d+), q = (-?\d+)")


def _parse_pq(text):
    m = _EXPR_RE.search(text)
    assert m, f"could not parse a_n = ... expression from: {text}"
    p_sign, p_mag, q_sign, q_mag = m.groups()
    p = int(p_mag) if p_mag else 1
    if p_sign == "-":
        p = -p
    q = int(q_mag) if q_mag else 1
    if q_sign == "-":
        q = -q
    return p, q


def _solve_recurrence(text):
    m3 = _A1_A2_A3_RE.search(text)
    if m3 and "Pair A" in text:
        a1, a2, a3 = (int(g) for g in m3.groups())
        pairs = [(int(pg), int(qg)) for pg, qg in _PAIR_RE.findall(text)]
        assert len(pairs) == 2, f"expected exactly 2 candidate pairs, got {pairs}"
        consistent = [pq for pq in pairs if pq[0] * a2 + pq[1] * a1 == a3]
        assert len(consistent) == 1, f"expected exactly one consistent pair, got {consistent}"
        p, q = consistent[0]
        return p * a3 + q * a2

    m2 = _A2_A3_RE.search(text)
    if m2 and not _A1_A2_RE.search(text):
        p, q = _parse_pq(text)
        a2, a3 = (int(g) for g in m2.groups())
        a1 = Fraction(a3 - p * a2, q)
        assert a1.denominator == 1, f"non-integer a1: {a1}"
        return int(a1)

    m1 = _A1_A2_RE.search(text)
    if m1:
        p, q = _parse_pq(text)
        a1, a2 = (int(g) for g in m1.groups())
        tm = _TARGET_RE.search(text)
        assert tm, f"could not find target index in: {text}"
        target = int(tm.group(1))
        terms = [None, a1, a2]
        for i in range(3, target + 1):
            terms.append(p * terms[i - 1] + q * terms[i - 2])
        return terms[target]

    raise AssertionError(f"could not parse recurrence question text: {text}")


# The characteristic equation "r^2 <sign> <mag>r <sign> <mag> = 0" appears,
# verbatim, in every one of the easy/medium/hard phrase() wordings (only the
# surrounding sentence varies) -- so parsing THIS substring, rather than the
# whole sentence, is robust to which of the 3 wordings was drawn. The br term
# is OPTIONAL (join_terms drops a zero coefficient; b = -(r1+r2) can be 0 on
# the easy tier when r1 = -r2 -- never on medium/hard, whose roots/real-part
# are drawn nonzero by construction, so b there is never 0).
_QUAD_RE = re.compile(r"r\^2\s*(?:([+-])\s*(\d*)r)?\s*([+-])\s*(\d+)\s*=\s*0")
_REPEATED_ROOT_RE = re.compile(r"repeated")
_COMPLEX_ROOT_RE = re.compile(r"complex conjugate")
# The exam tier's three wordings all contain this substring verbatim.
_EXAM_ROOTS_RE = re.compile(r"roots? r1 = (-?\d+) and r2 = (-?\d+)")

_LABELLED_PAIR_RE = re.compile(r"(\w+) = (-?\d+), (\w+) = (-?\d+)")


def _parse_answer_pair(answer_text):
    m = _LABELLED_PAIR_RE.fullmatch(answer_text)
    assert m, f"could not parse answer_text: {answer_text}"
    label1, v1, label2, v2 = m.groups()
    return {label1: int(v1), label2: int(v2)}


def _parse_bc(text):
    m = _QUAD_RE.search(text)
    assert m, f"could not parse characteristic equation from: {text}"
    bsign, bmag, csign, cmag = m.groups()
    bmag_val = int(bmag) if bmag else 1
    b = 0 if bsign is None else (bmag_val if bsign == "+" else -bmag_val)
    c = int(cmag) if csign == "+" else -int(cmag)
    return b, c


def _solve_roots(text):
    # exam tier: roots given directly, never mentions "r^2 ... = 0" at all.
    m = _EXAM_ROOTS_RE.search(text)
    if m:
        r1, r2 = int(m.group(1)), int(m.group(2))
        b = -(r1 + r2)
        c = r1 * r2
        return {"b": b, "c": c}

    b, c = _parse_bc(text)
    disc = b * b - 4 * c

    if _COMPLEX_ROOT_RE.search(text):
        assert disc < 0
        root = cmath.sqrt(disc)  # purely imaginary for disc < 0
        real_part = round(-b / 2)
        imaginary_part = round(abs(root.imag) / 2)
        return {"real_part": real_part, "imaginary_part": imaginary_part}

    if _REPEATED_ROOT_RE.search(text):
        assert disc == 0
        r = round(-b / 2)
        return {"r1": r, "r2": r}

    # easy tier: two distinct real roots.
    assert disc >= 0
    root = cmath.sqrt(disc)
    hi = (-b + root.real) / 2
    lo = (-b - root.real) / 2
    r1, r2 = round(min(hi, lo)), round(max(hi, lo))
    return {"r1": r1, "r2": r2}


_SINGLE_ANSWER_EXAM = {"ucv1-second-order-characteristic-roots": "p12e_characteristic_roots"}


def _by_difficulty(skill):
    """Sources by difficulty. The exam tier of a skill listed above now serves a multi-part chain; these tests
    cover the single-answer generator it replaced, so that tier is pointed back at it."""
    sources = {src.difficulty: src for src in skill.sources}
    if skill.id in _SINGLE_ANSWER_EXAM:
        sources["exam"] = dataclasses.replace(sources["exam"], topic=_SINGLE_ANSWER_EXAM[skill.id], params={})
    return sources


def _skill_sources(skill_id):
    _domains, skills = load_skills()
    skill = next(s for s in skills if s.id == skill_id)
    return _by_difficulty(skill)


RECURRENCE_SOURCES = _skill_sources(RECURRENCE_ID)
ROOTS_SOURCES = _skill_sources(ROOTS_ID)


@pytest.mark.parametrize("difficulty,seed", [(tier, s) for tier in ["easy", "medium", "hard", "exam"] for s in SEEDS])
def test_oracle_rederives_recurrence_answer(difficulty, seed):
    src = RECURRENCE_SOURCES[difficulty]
    q = mathgen.make(src.topic, seed=seed, difficulty=src.difficulty, **(src.params or {}))
    derived = _solve_recurrence(q.text)
    stated = int(q.answer_text)
    assert derived == stated, (
        f"seed={seed} difficulty={difficulty}\ntext: {q.text}\nderived: {derived}\nstated: {stated}"
    )


@pytest.mark.parametrize("difficulty,seed", [(tier, s) for tier in ["easy", "medium", "hard", "exam"] for s in SEEDS])
def test_oracle_rederives_roots_answer(difficulty, seed):
    src = ROOTS_SOURCES[difficulty]
    q = mathgen.make(src.topic, seed=seed, difficulty=src.difficulty, **(src.params or {}))
    derived = _solve_roots(q.text)
    stated = _parse_answer_pair(q.answer_text)
    assert derived == stated, (
        f"seed={seed} difficulty={difficulty}\ntext: {q.text}\nderived: {derived}\nstated: {stated}"
    )


def test_oracle_coverage():
    tiers = ["easy", "medium", "hard", "exam"]
    total = len(tiers) * len(SEEDS) * 2  # both skills
    assert len(SEEDS) >= 60
    assert total >= 480
    print(f"\nOracle re-derived {total} questions ({len(tiers)} tiers x {len(SEEDS)} seeds x 2 skills).")
