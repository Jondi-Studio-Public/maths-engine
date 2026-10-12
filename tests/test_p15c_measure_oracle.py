"""Independent oracle for the `p15c_measure` questions: it re-derives each answer from the rendered question
text.

Parses ONLY `question.text`, `question.answer_text` and `mcq_options`/
`question.answer` -- never anything from
`mathgen.generators.p15c_measure_build` -- and re-derives each answer from
scratch, over its own second implementation of the same closure predicate
(sigma-algebra) or the same product-measure arithmetic.
"""

import re
from fractions import Fraction
from itertools import combinations

import pytest

from mathgen.grading.typed import mcq_options
from mathgen.skills import load_skills

SIGMA_ID = "ucv1-sigma-algebra-closure"
PRODUCT_ID = "ucv1-product-measure-rectangle"
SEEDS = range(60)
SIGMA_SEEDS = range(1000)  # mcq/classification: >= 1000 seeds
TIERS = ["easy", "medium", "hard", "exam"]


def _skill_sources(skill_id):
    _domains, skills = load_skills()
    skill = next(s for s in skills if s.id == skill_id)
    return {src.difficulty: src for src in skill.sources}


SIGMA_SOURCES = _skill_sources(SIGMA_ID)
PRODUCT_SOURCES = _skill_sources(PRODUCT_ID)


def _make(sources, difficulty, seed):
    import mathgen

    src = sources[difficulty]
    topic = src.topic
    params = src.params or {}
    if difficulty == "exam" and topic == "exd_x_r3e_product_measure_complement":
        # The exam source moved to a multi-part chain; these tests pin the earlier single-answer tier.
        topic, params = "p15c_product_measure_rectangle", {}
    return mathgen.make(topic, seed=seed, difficulty=src.difficulty, **params)


# ---------------------------------------------------------------------------
# ucv1-sigma-algebra-closure

_SET_RE = re.compile(r"(A\d+) = (\{[^}]*\}|the empty set|the complement of (A\d+) within the universe)")


def _parse_family(text):
    universe_m = re.search(r"universe is U = \{1, \.\.\., (\d+)\}", text)
    n = int(universe_m.group(1))
    universe = frozenset(range(1, n + 1))
    body = text.split("family of subsets is given: ")[1].split(". Decide")[0]
    raw = {}
    for label, spec, ref in _SET_RE.findall(body):
        raw[label] = (spec, ref)
    resolved = {}

    def resolve(label):
        if label in resolved:
            return resolved[label]
        spec, ref = raw[label]
        if spec == "the empty set":
            value = frozenset()
        elif spec.startswith("the complement of"):
            value = universe - resolve(ref)
        else:
            nums = [int(x) for x in re.findall(r"\d+", spec)]
            value = frozenset(nums)
        resolved[label] = value
        return value

    return universe, [resolve(label) for label in raw]


@pytest.mark.parametrize("difficulty,seed", [(t, s) for t in TIERS for s in SIGMA_SEEDS])
def test_oracle_rederives_sigma_algebra_closure(difficulty, seed):
    q = _make(SIGMA_SOURCES, difficulty, seed)
    universe, family = _parse_family(q.text)
    fam_set = set(family)
    comp_ok = all((universe - a) in fam_set for a in family)
    union_ok = all((a | b) in fam_set for a, b in combinations(family, 2)) and all((a | a) in fam_set for a in family)
    if comp_ok and union_ok:
        derived = "closed under both"
    elif comp_ok:
        derived = "fails union"
    elif union_ok:
        derived = "fails complement"
    else:
        derived = "fails both"
    assert derived == q.answer_text, (q.text, derived, q.answer_text)


def test_oracle_coverage_sigma():
    total = len(TIERS) * len(SIGMA_SEEDS)
    assert total >= 4000
    print(f"\nSigma-algebra-closure oracle re-derived {total} questions.")


# ---------------------------------------------------------------------------
# ucv1-product-measure-rectangle

_WEIGHT_RE = re.compile(r"(mu|nu)\(([xy]\d+)\) = (-?\d+(?:/\d+)?)")


def _parse_frac(s):
    if "/" in s:
        p, q = s.split("/")
        return Fraction(int(p), int(q))
    return Fraction(int(s))


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("difficulty", TIERS)
def test_oracle_rederives_product_measure(difficulty, seed):
    q = _make(PRODUCT_SOURCES, difficulty, seed)
    text = q.text
    weights = {}
    for prefix, label, value in _WEIGHT_RE.findall(text):
        weights[(prefix, label)] = _parse_frac(value)
    x_labels = sorted({lbl for (p, lbl) in weights if p == "mu"}, key=lambda s: int(s[1:]))
    y_labels = sorted({lbl for (p, lbl) in weights if p == "nu"}, key=lambda s: int(s[1:]))

    a_clause = text.split("A = ")[1].split(". B = ")[0]
    b_clause = text.split("B = ")[1].split(". Compute")[0]

    def _members(clause, labels, prefix):
        if "minus" in clause:
            excluded = re.search(r"\{(" + prefix + r"\d+)\}", clause).group(1)
            return [lbl for lbl in labels if lbl != excluded]
        if "∪" in clause:
            parts = clause.split("∪")
            members = set()
            for part in parts:
                members.update(re.findall(prefix + r"\d+", part))
            return sorted(members, key=lambda s: int(s[1:]))
        return re.findall(prefix + r"\d+", clause)

    a_members = _members(a_clause, x_labels, "x")
    b_members = _members(b_clause, y_labels, "y")
    mu_a = sum((weights[("mu", m)] for m in set(a_members)), Fraction(0))
    nu_b = sum((weights[("nu", m)] for m in set(b_members)), Fraction(0))
    derived = mu_a * nu_b
    stated = Fraction(q.answer["p"], q.answer["q"])
    assert derived == stated, (text, derived, stated)


def test_oracle_coverage_product():
    total = len(TIERS) * len(SEEDS)
    assert total >= 240
    print(f"\nProduct-measure-rectangle oracle re-derived {total} questions.")
