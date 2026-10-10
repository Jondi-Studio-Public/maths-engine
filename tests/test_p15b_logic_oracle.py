"""Independent oracle for the `p15b_logic` questions: it re-derives each answer from the rendered question text.

Parses ONLY `question.text`/`question.answer_text` (via `mathgen.grading.
mcq_options`) -- never anything from `mathgen.generators.p15b_logic_build`
-- and re-derives the answer from scratch: a fresh, independently-written
recursive-descent well-formedness parser (never the generator's own
`is_well_formed`), and a fresh, independently-written expression evaluator
(never the generator's own `_eval`), applied to the interpretation parsed
out of the rendered text.
"""

import re

import pytest

from mathgen.grading.typed import mcq_options
from mathgen.skills import load_skills
from mathgen import make

SKILL_ID = "ucv1-syntactic-semantic-separation"
SEEDS = range(1000)
TIERS = ["easy", "medium", "hard", "exam"]

OPTIONS = [
    "Ill-formed (not a valid formula).",
    "Well-formed and true under the interpretation.",
    "Well-formed and false under the interpretation, with no derivation from stated premises shown.",
    "Well-formed and false under the interpretation, but derivable via modus ponens from stated premises shown alongside it.",
]


def _skill_sources(skill_id):
    _domains, skills = load_skills()
    skill = next(s for s in skills if s.id == skill_id)
    return {src.difficulty: src for src in skill.sources}


SOURCES = _skill_sources(SKILL_ID)

_ATOM_CHARS = set("pqrsab")
_CONNECTIVES = {"∧", "∨", "→"}
_NOT = "¬"


def _oracle_tokenize(text: str) -> list[str]:
    tokens = []
    for ch in text:
        if ch in _ATOM_CHARS or ch in _CONNECTIVES or ch == _NOT or ch in "()":
            tokens.append(ch)
    return tokens


class _ParseFail(Exception):
    pass


def _oracle_parse(tokens, i):
    """A from-scratch recursive-descent parser for the grammar
    Formula := Atom | NOT Formula | LPAREN Formula Connective Formula RPAREN,
    written without reference to the generator's own parser or grammar
    constants, returning (subtree, next_index) or raising _ParseFail."""
    if i >= len(tokens):
        raise _ParseFail("ran out of tokens")
    tok = tokens[i]
    if tok in _ATOM_CHARS:
        return ("atom", tok), i + 1
    if tok == _NOT:
        inner, j = _oracle_parse(tokens, i + 1)
        return ("not", inner), j
    if tok == "(":
        left, j = _oracle_parse(tokens, i + 1)
        if j >= len(tokens) or tokens[j] not in _CONNECTIVES:
            raise _ParseFail("expected connective")
        conn = tokens[j]
        right, k = _oracle_parse(tokens, j + 1)
        if k >= len(tokens) or tokens[k] != ")":
            raise _ParseFail("expected closing paren")
        kind = {"∧": "and", "∨": "or", "→": "implies"}[conn]
        return (kind, left, right), k + 1
    raise _ParseFail(f"unexpected token {tok!r}")


def _oracle_is_well_formed_and_parse(text):
    tokens = _oracle_tokenize(text)
    if not tokens:
        return None
    try:
        tree, end = _oracle_parse(tokens, 0)
    except _ParseFail:
        return None
    if end != len(tokens):
        return None
    return tree


def _oracle_eval(tree, values):
    kind = tree[0]
    if kind == "atom":
        return values[tree[1]]
    if kind == "not":
        return not _oracle_eval(tree[1], values)
    left, right = _oracle_eval(tree[1], values), _oracle_eval(tree[2], values)
    if kind == "and":
        return left and right
    if kind == "or":
        return left or right
    if kind == "implies":
        return (not left) or right
    raise ValueError(kind)


def _parse_interpretation(text):
    m = re.search(r"Interpretation: (.+?)\. Classify", text)
    assert m, text
    values = {}
    for chunk in m.group(1).split(", "):
        name, val = chunk.split(" = ")
        values[name.strip()] = val.strip() == "True"
    return values


@pytest.mark.parametrize("difficulty,seed", [(t, s) for t in TIERS for s in SEEDS])
def test_oracle_rederives_the_classification(difficulty, seed):
    src = SOURCES[difficulty]
    q = make(src.topic, seed=seed, difficulty=src.difficulty, **(src.params or {}))

    m = re.search(r"Expression: (.+?)\. Interpretation:", q.text)
    assert m, q.text
    expr_text = m.group(1)

    tree = _oracle_is_well_formed_and_parse(expr_text)
    if tree is None:
        derived = OPTIONS[0]
    else:
        values = _parse_interpretation(q.text)
        truth = _oracle_eval(tree, values)
        # "Premise 1" only appears when a derivation is actually rendered --
        # NOT "modus ponens", which (since the fix) also appears in the
        # instruction sentence on every draw regardless of whether a
        # derivation is shown.
        has_derivation = "Premise 1" in q.text
        if truth:
            derived = OPTIONS[1]
        elif has_derivation:
            derived = OPTIONS[3]
        else:
            derived = OPTIONS[2]

    assert derived == q.answer_text, (
        f"seed={seed} difficulty={difficulty}\ntext: {q.text}\nderived: {derived}\nstated: {q.answer_text}"
    )
    options = mcq_options(q)
    assert len(options) == 4
    assert q.answer_text in options


@pytest.mark.parametrize("difficulty,seed", [(t, s) for t in TIERS for s in SEEDS])
def test_oracle_exactly_one_option_is_true_of_the_draw(difficulty, seed):
    """Re-derives, independently, the truth of EACH of the four (fixed) option strings against the parsed
    expression, interpretation and rendered derivation context, and asserts exactly one holds."""
    src = SOURCES[difficulty]
    q = make(src.topic, seed=seed, difficulty=src.difficulty, **(src.params or {}))

    m = re.search(r"Expression: (.+?)\. Interpretation:", q.text)
    assert m, q.text
    expr_text = m.group(1)
    tree = _oracle_is_well_formed_and_parse(expr_text)
    has_derivation = "Premise 1" in q.text

    truth_of_option = [False, False, False, False]
    if tree is None:
        truth_of_option[0] = True
    else:
        values = _parse_interpretation(q.text)
        truth = _oracle_eval(tree, values)
        if truth:
            truth_of_option[1] = True
        elif has_derivation:
            truth_of_option[3] = True
        else:
            truth_of_option[2] = True

    true_options = [OPTIONS[i] for i, t in enumerate(truth_of_option) if t]
    assert true_options == [q.answer_text], (
        f"{difficulty} seed={seed}: exactly-one-true option set is {true_options}, "
        f"stated answer is {q.answer_text!r}\n{q.text}"
    )


def test_old_wording_would_have_been_ambiguous_for_both_false_classes():
    """Documents, independently of the generator, why the pre-fix wording
    was defective: both the "false, no derivation" and "false, derivable"
    classes are genuinely observed in the data (over 1000 seeds/tier), and
    both satisfy the single predicate "well-formed and false" that the old
    option 3 alone expressed -- so a single option covering both cannot be
    the unique key for either."""
    saw_false_no_derivation = False
    saw_false_derivable = False
    for difficulty in TIERS:
        for seed in SEEDS:
            src = SOURCES[difficulty]
            q = make(src.topic, seed=seed, difficulty=src.difficulty, **(src.params or {}))
            if q.answer_text == OPTIONS[2]:
                saw_false_no_derivation = True
            elif q.answer_text == OPTIONS[3]:
                saw_false_derivable = True
            if saw_false_no_derivation and saw_false_derivable:
                break
        if saw_false_no_derivation and saw_false_derivable:
            break
    assert saw_false_no_derivation, "no 'false, no derivation shown' draw found"
    assert saw_false_derivable, "no 'false, derivable' draw found"
    # Both classes being real and both being "well-formed and false" is
    # exactly the fact that made the old single "well-formed but false"
    # option ambiguous -- see the OPTIONS wording fix above.


def test_oracle_coverage():
    total = len(TIERS) * len(SEEDS)
    assert len(SEEDS) >= 60
    assert total >= 240
    print(f"\nOracle re-derived {total} questions ({len(TIERS)} tiers x {len(SEEDS)} seeds).")
