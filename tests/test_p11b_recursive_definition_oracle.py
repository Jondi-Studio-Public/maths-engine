"""Independent oracle for the `p11b_recursive_definition` questions: it re-derives each answer from the rendered
question text.

Parses ONLY the rendered question (``question.text``, ``question.answer_text``, ``question.distractors``) and
re-derives the answer from scratch. Imports nothing from
``mathgen.generators.p11b_recursive_definition_build`` -- not a helper, not a constant, not the resample
logic.

Two independent checks:

1. The stated terms are genuinely arithmetic (constant first differences) or
   genuinely geometric (constant exact integer ratio) throughout, verified
   by an oracle-owned scan of EVERY consecutive pair, not just the first.
2. The canonical ``answer_text`` states exactly the first stated term and
   the constant this oracle independently derived, and every rendered
   distractor states something else.
"""

from __future__ import annotations

import re

from mathgen import make

TIERS = ["easy", "medium", "hard"]
SEEDS = range(60)

_TERMS_RE = re.compile(r"(-?\d+(?:,\s*-?\d+)+)\.\s+State ")
_ARITH_ANSWER_RE = re.compile(r"^first_term = (-?\d+), d = (-?\d+)$")
_GEOM_ANSWER_RE = re.compile(r"^first_term = (-?\d+), r = (-?\d+)$")


def _parse_terms(text: str) -> list[int]:
    match = _TERMS_RE.search(text)
    assert match, f"could not find a stated-terms run in: {text!r}"
    return [int(t.strip()) for t in match.group(1).split(",")]


def _is_arithmetic_question(text: str) -> bool:
    if "value of d in" in text:
        return True
    if "value of r in" in text:
        return False
    raise AssertionError(f"question names neither d nor r: {text!r}")


def _oracle_arithmetic(terms: list[int]) -> tuple[int, int]:
    """Independently derive (first_term, d) from the stated terms alone,
    confirming EVERY consecutive difference is the same constant."""
    diffs = {terms[i + 1] - terms[i] for i in range(len(terms) - 1)}
    assert len(diffs) == 1, f"terms are not arithmetic throughout: {terms}"
    return terms[0], next(iter(diffs))


def _oracle_geometric(terms: list[int]) -> tuple[int, int]:
    """Independently derive (first_term, r) from the stated terms alone,
    confirming EVERY consecutive pair divides exactly by the same constant
    integer ratio."""
    ratios = set()
    for i in range(len(terms) - 1):
        a, b = terms[i], terms[i + 1]
        assert a != 0 and b % a == 0, f"terms are not exactly geometric: {terms}"
        ratios.add(b // a)
    assert len(ratios) == 1, f"terms do not share a single ratio throughout: {terms}"
    return terms[0], next(iter(ratios))


def test_oracle_recomputes_the_canonical_answer_for_every_seed_and_tier():
    checked = {"arithmetic": 0, "geometric": 0}
    for topic in ["arithmetic_recursive_definition_state", "geometric_recursive_definition_state"]:
        for tier in TIERS:
            for seed in SEEDS:
                q = make(topic, seed=seed, difficulty=tier)
                terms = _parse_terms(q.text)
                is_arith = _is_arithmetic_question(q.text)
                if is_arith:
                    expected_first, expected_const = _oracle_arithmetic(terms)
                    match = _ARITH_ANSWER_RE.match(q.answer_text)
                    checked["arithmetic"] += 1
                else:
                    expected_first, expected_const = _oracle_geometric(terms)
                    match = _GEOM_ANSWER_RE.match(q.answer_text)
                    checked["geometric"] += 1
                assert match, f"unparseable canonical answer: {q.answer_text!r}"
                stated_first, stated_const = int(match.group(1)), int(match.group(2))
                assert stated_first == expected_first, (
                    f"{topic} {tier} seed {seed}: canonical first_term {stated_first} != "
                    f"oracle-derived {expected_first} from terms {terms}"
                )
                assert stated_const == expected_const, (
                    f"{topic} {tier} seed {seed}: canonical constant {stated_const} != "
                    f"oracle-derived {expected_const} from terms {terms}"
                )
    assert checked["arithmetic"] == len(TIERS) * len(SEEDS)
    assert checked["geometric"] == len(TIERS) * len(SEEDS)


def test_oracle_rejects_every_rendered_distractor():
    """Every distractor states a DIFFERENT (first_term, constant) pair from
    the one this oracle independently derived from the stated terms."""
    checked = 0
    for topic in ["arithmetic_recursive_definition_state", "geometric_recursive_definition_state"]:
        answer_re = _ARITH_ANSWER_RE if topic.startswith("arithmetic") else _GEOM_ANSWER_RE
        for tier in TIERS:
            for seed in SEEDS:
                q = make(topic, seed=seed, difficulty=tier)
                terms = _parse_terms(q.text)
                expected = _oracle_arithmetic(terms) if topic.startswith("arithmetic") else _oracle_geometric(terms)
                for distractor in q.distractors:
                    match = answer_re.match(distractor)
                    assert match, f"unparseable distractor: {distractor!r}"
                    given = (int(match.group(1)), int(match.group(2)))
                    assert given != expected, (
                        f"{topic} {tier} seed {seed}: distractor {distractor!r} states the "
                        f"same (first_term, constant) as the oracle-derived answer {expected}"
                    )
                    checked += 1
    assert checked > 0


def _lead_in_shape(text: str) -> str:
    """Classify which of the (>=3) abstract lead-in templates a
    context-free (easy/medium) rendering used, matching on a short literal
    fragment near the front of each -- never importing the generator's own
    template list (the oracle re-derives everything from rendered text)."""
    if text.startswith("Consider the"):
        return "consider"
    if "first few terms of" in text[:60]:
        return "first_few_terms"
    if "sequence begins " in text[:60]:
        return "begins"
    raise AssertionError(f"unrecognised lead-in shape: {text!r}")


def test_abstract_lead_in_has_at_least_three_shapes_over_60_seeds():
    """easy/medium never draw a context (context is None), so without
    lead-in variety every render used the identical single template --
    mathgen.text_quality.prompt_shape masks the digits inside {terms} away,
    collapsing all 60 seeds to one shape. Confirms, from rendered text alone, that all
    3 lead-ins genuinely appear."""
    for topic in ["arithmetic_recursive_definition_state", "geometric_recursive_definition_state"]:
        for tier in ["easy", "medium"]:
            shapes = {_lead_in_shape(make(topic, seed=seed, difficulty=tier).text) for seed in SEEDS}
            assert shapes == {"consider", "first_few_terms", "begins"}, (
                f"{topic} [{tier}]: only saw lead-in shapes {shapes} over {len(SEEDS)} seeds"
            )


def test_oracle_terms_run_never_states_t1_or_the_constant_labelled():
    """The stated-terms run itself never contains a label ('t1 =', 'd =',
    'r =') -- confirms the question genuinely withholds the rule, the whole
    premise this skill pair depends on (never re-reads the generator to
    confirm this; parses the rendered text only)."""
    for topic in ["arithmetic_recursive_definition_state", "geometric_recursive_definition_state"]:
        for tier in TIERS:
            for seed in SEEDS:
                q = make(topic, seed=seed, difficulty=tier)
                before_terms = q.text[: _TERMS_RE.search(q.text).start()]
                assert "t1 =" not in before_terms
                assert "d =" not in before_terms
                assert "r =" not in before_terms
