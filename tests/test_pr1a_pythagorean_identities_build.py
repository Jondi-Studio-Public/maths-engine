"""Build tests for
`senior-prove-and-apply-the-pythagorean-identities` (ACMSM046).

Fast (module-scope fixtures). The rubric-marking checks of each reference chain
and trap are not part of this package.
"""

from __future__ import annotations

import random
import re

import pytest
import sympy as sp

import mathgen
import mathgen.core
from mathgen.grading.typed import _parse_expression
from mathgen.generators import pr1a_pythagorean_identities_build as gen

TOPIC = "pr1a_pythagorean_identity_proof"
SKILL_ID = "senior-prove-and-apply-the-pythagorean-identities"
TIERS = ("easy", "medium", "hard", "exam")
N_SEEDS = 200


def _questions(difficulty: str, n: int = N_SEEDS):
    return [mathgen.make(TOPIC, seed=s, difficulty=difficulty) for s in range(n)]


@pytest.fixture(scope="module")
def all_questions():
    return {tier: _questions(tier) for tier in TIERS}


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_determinism_across_tiers():
    for tier in TIERS:
        for seed in (0, 1, 5, 17, 42):
            a = mathgen.make(TOPIC, seed=seed, difficulty=tier)
            b = mathgen.make(TOPIC, seed=seed, difficulty=tier)
            assert a.to_dict() == b.to_dict()


# ---------------------------------------------------------------------------
# Marking-dict shape (rules 1, 2, 4, 5) and text/answer_text cleanliness
# ---------------------------------------------------------------------------


def test_marking_dict_shape_and_text_cleanliness(all_questions):
    for tier, questions in all_questions.items():
        for q in questions:
            a = q.answer
            assert isinstance(a, dict)
            assert a["start"] and a["target"] and isinstance(a["symbols"], dict)
            assert len(a["checks"]) == 1
            check = a["checks"][0]
            assert check["type"] == "equivalence_chain"
            assert check["start"] == a["start"]
            assert check["target"] == a["target"]
            assert check["symbols"] == a["symbols"]
            assert check["authoritative"] is True
            assert a["reference_lines"][0] == a["start"]
            assert a["reference_lines"][-1] == a["target"]
            criterion_ids = {step["criterion_id"] for step in a["key_steps"]}
            assert criterion_ids <= {"correctness", "justified_steps", "communication"}
            assert "justified_steps" in criterion_ids
            assert "communication" in criterion_ids
            assert "**" not in q.text and "**" not in q.answer_text
            assert "*" not in q.text and "*" not in q.answer_text
            for trap in a["traps"]:
                assert trap["expected_check"] in (
                    "authoritative_failure",
                    "non_authoritative_failure",
                    "passes_chain",
                )
                assert trap["example_lines"]
            has_non_auth = any(t["expected_check"] == "non_authoritative_failure" for t in a["traps"])
            has_passes = any(t["expected_check"] == "passes_chain" for t in a["traps"])
            assert has_non_auth, (tier, a)
            assert has_passes, (tier, a)


def test_symbolic_items_carry_an_authoritative_trap(all_questions):
    for tier, questions in all_questions.items():
        for q in questions:
            a = q.answer
            if a["symbols"]:  # a prove item (symbolic); apply items are {} (numeric, rule 7's carve-out)
                assert any(t["expected_check"] == "authoritative_failure" for t in a["traps"]), (tier, a)


# ---------------------------------------------------------------------------
# Symbol safety (rule 6)
# ---------------------------------------------------------------------------


def test_symbol_names_round_trip_sympify_and_parse_expression():

    seen = set()
    for tier in TIERS:
        for s in range(60):
            q = mathgen.make(TOPIC, seed=s, difficulty=tier)
            seen |= set(q.answer["symbols"])
    assert seen <= {"x", "t", "u"}
    for name in seen:
        assert sp.sympify(name) == sp.Symbol(name)
        parsed = _parse_expression(name)
        assert parsed is not None and str(parsed) == name


# ---------------------------------------------------------------------------
# Notation and domain (rules 10-13, added by the task-1968 send-back).
# ---------------------------------------------------------------------------

_GREEK_SPELLED = ("theta", "alpha", "phi", "beta", "gamma")


def test_no_greek_letters_anywhere(all_questions):
    """Rule 10: Latin letters only -- no Greek glyph and no spelled-out Greek
    name, in either the prompt or the chain symbols/lines, anywhere."""
    for tier, questions in all_questions.items():
        for q in questions:
            a = q.answer
            for blob in (q.text, q.answer_text, a["start"], a["target"], *a["reference_lines"]):
                for greek_char in "θαφβγ":
                    assert greek_char not in blob, (tier, greek_char, blob)
                lowered = blob.lower()
                for name in _GREEK_SPELLED:
                    assert name not in lowered, (tier, name, blob)
            for sym in a["symbols"]:
                assert sym not in _GREEK_SPELLED


def test_notation_sentence_present_in_every_prompt(all_questions):
    """Rule 11: every prompt states the typeable notation (brackets round
    every angle, ^ for powers) in its own words."""
    for tier, questions in all_questions.items():
        for q in questions:
            text = q.text
            assert "sin(x)" in text and "cos(x)^2" in text, (tier, text)
            assert "**" not in text and "*" not in text


_PROVE_EXTRACT_PATTERNS = (
    re.compile(r"Start your proof from (.+?) and finish at (.+?),"),
    re.compile(r"Begin your proof from (.+?) and end at (.+?),"),
    re.compile(r"starting from (.+?), finishing at (.+?),"),
)

_APPLY_START_PATTERNS = (
    re.compile(r"[Ss]tart from ([a-z]+\^2\([^()]*\))"),
    re.compile(r"(?:find|evaluate) ([a-z]+\^2\([^()]*\)) exactly"),
    re.compile(r"the exact value of ([a-z]+\^2\([^()]*\))"),
)


def test_domain_stated_for_every_tan_sec_csc_cot_or_division_identity(all_questions):
    """Rule 13: an identity whose sides use tan/sec/csc/cot or a division by
    a trig term must never claim 'for all real x' / 'every real value', and
    must state its actual domain instead. `conj_prod`/`quartic_diff` are the
    only two families in this pool with neither tan/sec/csc/cot nor
    division -- both sides defined everywhere -- so they carry no domain
    clause; every other family must."""
    forbidden = ("all real", "every real")
    needs_domain_funcs = ("tan(", "sec(", "csc(", "cot(")
    for tier, questions in all_questions.items():
        for q in questions:
            text_lower = q.text.lower()
            for phrase_ in forbidden:
                assert phrase_ not in text_lower, (tier, phrase_, q.text)
            a = q.answer
            if not a["symbols"]:
                continue  # apply item: a specific numeric value, not a universal claim
            needs_domain = any(fn in a["start"] or fn in a["target"] for fn in needs_domain_funcs) or (
                "/" in a["start"] or "/" in a["target"]
            )
            if needs_domain:
                assert "is not 0" in q.text, (tier, "no domain condition stated", q.text)


# ---------------------------------------------------------------------------
# Every distinct marking block over 200 seeds per tier: reference chain
# passes, every trap produces exactly its expected_check.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# The in-generator check fires on a corrupted reference line.
# ---------------------------------------------------------------------------


def test_in_generator_check_fires_on_corrupted_reference_line(monkeypatch):
    family = dict(gen.FAMILIES_BY_TIER["easy"][0])
    bad_ref = list(family["ref_sym"])
    bad_ref[1] = "999"  # corrupt an intermediate reference line
    family["ref_sym"] = tuple(bad_ref)
    with pytest.raises(mathgen.core.VerificationError):
        gen._prove_item(random.Random("corrupt-test"), family)


# ---------------------------------------------------------------------------
# Skill loading
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Tier escalation substitute (deferred skills aren't seen by measure_tiers.py)
# ---------------------------------------------------------------------------


def test_tier_prompt_shapes_escalate(all_questions):
    from mathgen.text_quality import prompt_shape

    shapes_by_tier = {}
    for tier in TIERS:
        shapes_by_tier[tier] = {prompt_shape(q) for q in all_questions[tier][:60]}
    union_lower = set()
    for tier in ("easy", "medium", "hard", "exam"):
        added = shapes_by_tier[tier] - union_lower
        if tier in ("hard", "exam"):
            assert added, f"{tier} adds no new prompt_shape over lower tiers"
        union_lower |= shapes_by_tier[tier]
    for tier in TIERS:
        assert len(shapes_by_tier[tier]) >= 3, (tier, shapes_by_tier[tier])


def test_at_least_20_distinct_rendered_prompts_per_tier(all_questions):
    for tier in TIERS:
        texts = {q.text for q in all_questions[tier][:60]}
        assert len(texts) >= 20, (tier, len(texts))


def test_target_concentration_under_25_percent_at_200_seeds(all_questions):
    from collections import Counter

    for tier in TIERS:
        counts = Counter(q.answer_text for q in all_questions[tier])
        top = counts.most_common(1)[0][1]
        assert top / len(all_questions[tier]) <= 0.25, (tier, counts.most_common(3))


# --- rule 16: latex renders in KaTeX and is the same prompt as ---
# --- text. Static shape checks standing in for `katex_check.js` (node, run --
# --- separately by the task's own verification commands) and for the ------
# --- `is_mangled` detector in `tests/test_prompt_latex_mangled.py` (not ----
# --- imported here -- reimplemented as the minimum needed to prove this ----
# --- module's own latex can never regress into that failure mode). This is --
# --- also the module's first authored `latex`: before it every item --
# --- fell back to `mathgen.core.latex_prompt(text)`. --------

_ESCAPE_MACRO_RE_16 = re.compile(r"\\text(?:asciicircum|asciitilde|backslash)\{\}")
_TEXT_GROUP_RE_16 = re.compile(r"\\text\{[^{}]*\}")
_TRIG_NAME_RE_16 = re.compile(r"(?<!\\)\b(sin|cos|tan|sec|csc|cot)\b")


def _rule16_check(text: str, latex: str, tier: str, seed: int) -> None:
    assert latex, (tier, seed, "empty latex")
    assert r"\verb" not in latex, (tier, seed, latex)
    stripped = _ESCAPE_MACRO_RE_16.sub("", latex)
    spans = [m.span() for m in _TEXT_GROUP_RE_16.finditer(stripped)]

    def _inside(pos: int) -> bool:
        return any(a <= pos < b for a, b in spans)

    for m in re.finditer(r"\^", stripped):
        assert not _inside(m.start()), (tier, seed, "bare ^ inside \\text{} (should be \\textasciicircum{})", latex)
    for m in _TRIG_NAME_RE_16.finditer(stripped):
        assert _inside(m.start()), (tier, seed, f"unescaped {m.group()} outside \\text{{}}", latex)
    norm_latex = latex.replace(r"\text{", " ").replace("{", " ").replace("}", " ")
    norm_latex = norm_latex.replace("\\", " ")
    norm_latex = re.sub(r"\s+", " ", norm_latex).strip().lower()
    for phrase in ("exact value", "give your answer"):
        if phrase in text.lower():
            assert phrase in norm_latex, (tier, seed, phrase, latex)


@pytest.mark.parametrize("tier", TIERS)
def test_latex_rule16_shape(tier):
    checked = 0
    for s in range(30):
        q = mathgen.make(TOPIC, seed=s, difficulty=tier)
        _rule16_check(q.text, q.latex, tier, s)
        checked += 1
    assert checked == 30
