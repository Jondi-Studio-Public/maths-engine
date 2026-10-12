"""Checks for the multi-part exam sources of `propositional-truth-evaluation` and `propositional-formula-equivalence`.

(a) several prompt shapes, some absent from the hard tier; (b) every part's key is re-derived from the RENDERED
PROMPT by parsing the formulas with independent code, over 300 seeds; (c) the skill's grader accepts the key and
rejects every distractor; (d) a wrong count used correctly in the next part still earns that part's marks.
"""

import re
from fractions import Fraction
from itertools import product

import pytest

from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.text_quality import find_text_defects, prompt_shape

EVAL, EQUIV = "exd_x_sc4_prop_eval", "exd_x_sc4_prop_equiv"
SKILLS = {s.id: s for s in load_skills()[1]}
EVAL_SKILL = SKILLS["propositional-truth-evaluation"]
EQUIV_SKILL = SKILLS["propositional-formula-equivalence"]
SEEDS = range(300)
ROWS = list(product([True, False], repeat=4))


def _q(topic, seed):
    return make(topic, seed=seed, difficulty="exam")


@pytest.fixture(scope="module")
def evals():
    return [_q(EVAL, s) for s in SEEDS]


@pytest.fixture(scope="module")
def equivs():
    return [_q(EQUIV, s) for s in SEEDS]


# -- an independent reader for the rendered formulas: unit := ¬unit | atom | (expr); expr := unit [op unit] ------------


def parse(text):
    tokens = re.findall(r"[pqrs]|¬|∧|∨|→|\(|\)", text)
    pos = 0

    def unit():
        nonlocal pos
        tok = tokens[pos]
        pos += 1
        if tok == "¬":
            inner = unit()
            return lambda env: not inner(env)
        if tok == "(":
            inner = expr()
            assert tokens[pos] == ")"
            pos += 1
            return inner
        return lambda env: env[tok]

    def expr():
        nonlocal pos
        left = unit()
        if pos < len(tokens) and tokens[pos] in "∧∨→":
            op = tokens[pos]
            pos += 1
            right = unit()
            return {
                "∧": lambda env: left(env) and right(env),
                "∨": lambda env: left(env) or right(env),
                "→": lambda env: (not left(env)) or right(env),
            }[op]
        return left

    result = expr()
    assert pos == len(tokens), text
    return result


def envs():
    return [dict(zip("pqrs", row)) for row in ROWS]


def _word(value):
    return value == "true"


def eval_oracle(q):
    formula = parse(re.search(r"formula (.+)\.$", q.stem).group(1))
    rows = envs()
    k = sum(1 for env in rows if formula(env))
    atom, value = re.search(r"F true and ([pqrs]) (true|false)", q.parts[2].prompt).groups()
    other, ovalue = re.search(r"F true and ([pqrs]) (true|false)", q.parts[3].prompt).groups()
    a1, v1, a2, v2 = re.search(r"F true, ([pqrs]) (true|false) and ([pqrs]) (true|false)", q.parts[4].prompt).groups()
    return {
        "a": str(k),
        "b": _frac(Fraction(k, 16)),
        "c": str(sum(1 for env in rows if env[atom] == _word(value) and formula(env))),
        "d": str(sum(1 for env in rows if env[other] == _word(ovalue) and formula(env))),
        "e": str(sum(1 for env in rows if env[a1] == _word(v1) and env[a2] == _word(v2) and formula(env))),
    }


def equiv_oracle(q):
    f_text, g_text = re.search(r"formula (.+) and G the formula (.+)\.$", q.stem).groups()
    f, g = parse(f_text), parse(g_text)
    rows = envs()
    differ = sum(1 for env in rows if f(env) != g(env))
    atom, value = re.search(r"differ and ([pqrs]) is (true|false)", q.parts[5].prompt).groups()
    return {
        "a": str(differ),
        "b": _frac(Fraction(16 - differ, 16)),
        "c": str(sum(1 for env in rows if f(env) and g(env))),
        "d": str(sum(1 for env in rows if f(env) and not g(env))),
        "e": str(sum(1 for env in rows if not f(env) and not g(env))),
        "f": str(sum(1 for env in rows if env[atom] == _word(value) and f(env) != g(env))),
    }


def _frac(value):
    return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"


def _canonical(q):
    return {p.label: p.answer_text for p in q.parts}


# -- tests ------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("topic", [EVAL, EQUIV])
def test_no_text_defects(topic):
    for seed in range(60):
        assert find_text_defects(_q(topic, seed)) == [], seed


@pytest.mark.parametrize(
    "topic, hard",
    [(EVAL, "propositional_truth_value"), (EQUIV, "propositional_formula_equivalence")],
)
def test_prompt_variety_and_new_shapes(topic, hard):
    exam = {prompt_shape(_q(topic, s)) for s in range(60)}
    hard_shapes = {prompt_shape(make(hard, seed=s, difficulty="hard")) for s in range(60)}
    assert len(exam) >= 3, exam
    assert exam - hard_shapes, (exam, hard_shapes)


def test_wiring_and_shape():
    for skill, topic in ((EVAL_SKILL, EVAL), (EQUIV_SKILL, EQUIV)):
        assert [s.topic for s in skill.sources if s.difficulty == "exam"] == [topic]
    for seed in range(40):
        q = _q(EVAL, seed)
        assert [p.answer_kind for p in q.parts] == ["number", "fraction", "number", "number", "number"]
        assert [p.follow_through is not None for p in q.parts] == [False, True, False, False, False]
        q = _q(EQUIV, seed)
        assert [p.answer_kind for p in q.parts] == ["number", "fraction", "number", "number", "number", "number"]
        assert [p.follow_through is not None for p in q.parts] == [False, True, False, False, False, False]


def test_deterministic():
    for topic in (EVAL, EQUIV):
        for seed in (0, 1, 7, 123):
            assert _q(topic, seed).to_json() == _q(topic, seed).to_json()


def test_eval_keys_match_independent_derivation(evals):
    bad = [(s, q.stem, eval_oracle(q), _canonical(q)) for s, q in zip(SEEDS, evals) if eval_oracle(q) != _canonical(q)]
    assert bad == [], bad[:2]


def test_equiv_keys_match_independent_derivation(equivs):
    bad = [
        (s, q.stem, equiv_oracle(q), _canonical(q)) for s, q in zip(SEEDS, equivs) if equiv_oracle(q) != _canonical(q)
    ]
    assert bad == [], bad[:2]


def test_the_formulas_are_not_trivial(evals, equivs):
    assert len({eval_oracle(q)["a"] for q in evals}) >= 6
    assert len({equiv_oracle(q)["a"] for q in equivs}) >= 4
    for q in equivs:
        assert equiv_oracle(q)["a"] != "0", q.stem


@pytest.mark.parametrize("skill, topic", [(EVAL_SKILL, EVAL), (EQUIV_SKILL, EQUIV)])
def test_grader_accepts_key_and_rejects_every_distractor(skill, topic):
    bad = []
    for seed in range(150):
        q = _q(topic, seed)
        canon = _canonical(q)
        result = grade(skill, q, canon)
        if not (result.correct and result.marks == result.max_marks):
            bad.append(("key rejected", seed, canon))
        for p in q.parts:
            assert p.distractors, (seed, p.label)
            for d in p.distractors:
                if grade(skill, q, {**canon, p.label: d}).parts[p.label].marks != 0:
                    bad.append(("distractor accepted", seed, p.label, d))
    assert bad == [], bad[:3]


def test_distractors_carry_catalogue_tags(evals, equivs):
    """Most parts offer a tagged distractor; an option is left untagged only when no named error gives it."""
    untagged = 0
    parts = 0
    for q in [*evals, *equivs]:
        for p in q.parts:
            parts += 1
            assert len(p.distractors) >= 2
            if not [d for d in p.distractors if p.option_notes.get(d)]:
                untagged += 1
    assert untagged / parts < 0.1, (untagged, parts)


def test_eval_follow_through_from_a_wrong_count(evals):
    checked = 0
    for q in evals[:150]:
        for wrong in q.parts[0].distractors:
            probability = Fraction(int(wrong), 16)
            if not 0 < probability < 1:
                continue
            reply = {**_canonical(q), "a": wrong, "b": _frac(probability)}
            result = grade(EVAL_SKILL, q, reply)
            assert result.parts["a"].marks == 0
            assert result.parts["b"].marks == result.parts["b"].max_marks, (q.stem, reply)
            assert result.parts["b"].follow_through is True
            checked += 1
    assert checked >= 100


def test_equiv_follow_through_from_a_wrong_count(equivs):
    checked = 0
    for q in equivs[:150]:
        for wrong in q.parts[0].distractors:
            probability = 1 - Fraction(int(wrong), 16)
            if not 0 < probability < 1:
                continue
            reply = {**_canonical(q), "a": wrong, "b": _frac(probability)}
            result = grade(EQUIV_SKILL, q, reply)
            assert result.parts["a"].marks == 0
            assert result.parts["b"].marks == result.parts["b"].max_marks, (q.stem, reply)
            checked += 1
    assert checked >= 100


def test_all_wrong_earns_nothing(evals, equivs):
    for q in evals[:30]:
        assert grade(EVAL_SKILL, q, {"a": "99", "b": "9/7", "c": "98", "d": "97", "e": "96"}).marks == 0
    for q in equivs[:30]:
        assert grade(EQUIV_SKILL, q, {"a": "99", "b": "9/7", "c": "98", "d": "97", "e": "96", "f": "95"}).marks == 0


def test_answer_space_is_wide(evals, equivs):
    """Over 300 seeds the whole answer is hard to guess: at least 100 distinct answers, none above 5%."""
    from collections import Counter

    for questions in (evals, equivs):
        counts = Counter(q.answer_text for q in questions)
        assert len(counts) >= 100, len(counts)
        assert max(counts.values()) <= 15, counts.most_common(2)
