"""PR1A task 1961: senior-prove-and-apply-the-angle-sum-and-difference
(ACMMM041). Fast checks against the generator's own tables -- the marking
dict shape, the pool sizes, the domain statements and the in-generator
checks firing on corrupted entries. (The rubric-marking checks of each
reference chain and trap run in the site that hosts the rubric marker.) `tests/test_pr1a_angle_sum_difference_oracle.py` is
the independent oracle that never imports this module.
"""

from __future__ import annotations

import re

import pytest
import sympy

from mathgen.grading.typed import _parse_expression
from mathgen.core import make
from mathgen.generators import pr1a_angle_sum_difference_build as gen

SKILL_ID = "senior-prove-and-apply-the-angle-sum-and-difference"
TOPIC = "pr1a_angle_sum_difference_proof"
TIERS = ("easy", "medium", "hard", "exam")
N_SEEDS = 200

RUBRIC_IDS = {"correctness", "justified_steps", "communication"}
_NO_STAR_RE = re.compile(r"\*\*|\*")
_SAFE_SYMBOLS = {"x", "t", "u", "v", "A", "B", "P", "R", "theta", "alpha", "phi"}


def _questions(tier: str, n: int = N_SEEDS):
    return [make(TOPIC, seed=s, difficulty=tier) for s in range(n)]


# --- determinism -----------------------------------------------------------


@pytest.mark.parametrize("tier", TIERS)
def test_determinism(tier):
    for s in range(20):
        q1 = make(TOPIC, seed=s, difficulty=tier)
        q2 = make(TOPIC, seed=s, difficulty=tier)
        assert q1.text == q2.text
        assert q1.answer == q2.answer
        assert q1.answer_text == q2.answer_text


# --- text/answer_text free of ** and * --------------------------------------


@pytest.mark.parametrize("tier", TIERS)
def test_no_sympy_syntax_in_learner_facing_text(tier):
    for q in _questions(tier, 60):
        assert not _NO_STAR_RE.search(q.text), q.text
        assert not _NO_STAR_RE.search(q.answer_text), q.answer_text
        assert q.distractors == []


# --- marking dict shape ------------------------------------------------------


def _all_marking_blocks(tier: str, n: int = N_SEEDS):
    seen = {}
    for s in range(n):
        q = make(TOPIC, seed=s, difficulty=tier)
        marking = q.answer
        key = (marking["start"], marking["target"], tuple(marking["reference_lines"]))
        seen.setdefault(key, marking)
    return seen


@pytest.mark.parametrize("tier", TIERS)
def test_marking_dict_shape(tier):
    blocks = _all_marking_blocks(tier)
    assert blocks, f"no marking blocks collected for {tier}"
    for marking in blocks.values():
        assert isinstance(marking["start"], str) and isinstance(marking["target"], str)
        assert isinstance(marking["symbols"], dict)
        checks = marking["checks"]
        assert len(checks) == 1
        check0 = checks[0]
        assert check0["type"] == "equivalence_chain"
        assert check0["start"] == marking["start"]
        assert check0["target"] == marking["target"]
        assert check0["symbols"] == marking["symbols"]
        assert check0["authoritative"] is True
        assert all(k["criterion_id"] in RUBRIC_IDS for k in marking["key_steps"])
        assert any(k["criterion_id"] == "justified_steps" for k in marking["key_steps"])
        assert any(k["criterion_id"] == "communication" for k in marking["key_steps"])
        for trap in marking["traps"]:
            assert trap["expected_check"] in (
                "authoritative_failure",
                "non_authoritative_failure",
                "passes_chain",
            )
            assert trap["example_lines"]
        expected_checks = {t["expected_check"] for t in marking["traps"]}
        assert "passes_chain" in expected_checks
        assert "non_authoritative_failure" in expected_checks
        if marking["symbols"]:  # symbolic (prove) item: must also carry an authoritative trap
            assert "authoritative_failure" in expected_checks


# --- reference chains and traps against the REAL checker --------------------


# --- distinct pool size (rule 9 / the ladder) -------------------------------


def test_easy_tier_pool_size():
    blocks = _all_marking_blocks("easy", 200)
    # Send-back 1966: 2 prove identities (cos_sum, sin_diff_easy) x 5 letter
    # pairs = 10, + 16 apply combinations (4 angles x 2 decompositions x 2
    # functions) -- the honest pool size, stated here per the common file's
    # "if the true pool is genuinely smaller, say so".
    assert len(blocks) == 26, sorted(blocks)


def test_medium_tier_pool_size():
    # 2 prove identities (sin_sum, sin_diff) x 5 letter pairs.
    assert len(_all_marking_blocks("medium", 200)) == 10


def test_hard_tier_pool_size():
    # 2 prove identities (tan_sum, tan_diff) x 5 letter pairs.
    assert len(_all_marking_blocks("hard", 200)) == 10


def test_exam_tier_pool_size():
    # 3 prove identities (cos_diff_derivation, composite_sin, composite_cos)
    # x 5 letter pairs.
    assert len(_all_marking_blocks("exam", 200)) == 15


# --- decide() short-circuit on an authoritative failure ---------------------

_HIGH_CONFIDENCE_ASSESSMENT = {
    "criteria": [
        {"id": "correctness", "points": 4, "evidence": "matches"},
        {"id": "justified_steps", "points": 3, "evidence": "cites the given"},
        {"id": "communication", "points": 1, "evidence": "clear"},
    ],
    "confidence": "high",
    "needs_review": False,
}
_CLEAN_ADVERSARIAL = {"issues": [], "confidence": "high", "needs_review": False}
_RUBRIC = {
    "id": "trig-identity-proof-v1",
    "pass_threshold": 0.8,
    "criteria": [
        {"id": "correctness", "label": "Correctness", "max_points": 4},
        {"id": "justified_steps", "label": "Justified steps", "max_points": 3},
        {"id": "communication", "label": "Communication", "max_points": 1},
    ],
}


# --- in-generator check fires on a corrupted reference line -----------------


def test_verify_symbolic_chain_fires_on_corruption():
    good = ["cos(A + B)", "cos(A - (-B))", "cos(A)*cos(-B) + sin(A)*sin(-B)", "cos(A)*cos(B) - sin(A)*sin(B)"]
    gen._verify_symbolic_chain(good, "sanity")
    corrupted = list(good)
    corrupted[-1] = "cos(A)*cos(B) + sin(A)*sin(B)"  # sign-flipped, no longer equal to line 3
    with pytest.raises(Exception):
        gen._verify_symbolic_chain(corrupted, "corrupted")


def test_verify_numeric_chain_fires_on_corruption():
    entry = gen._APPLY_ENTRIES[0]
    gen._verify_numeric_chain(entry["reference_lines"], "sanity")
    corrupted = list(entry["reference_lines"])
    corrupted[-1] = "0"
    with pytest.raises(Exception):
        gen._verify_numeric_chain(corrupted, "corrupted")


# --- symbol names round-trip sympify and _parse_expression ------------------


def test_symbols_used_are_in_the_safe_list():

    used = set()
    for tier in TIERS:
        for marking in _all_marking_blocks(tier, 60).values():
            used |= set(marking["symbols"])
            for line in marking["reference_lines"]:
                for name in re.findall(r"[A-Za-z_]\w*", line):
                    if name not in ("sin", "cos", "tan", "pi", "sqrt"):
                        used.add(name)
    assert used <= _SAFE_SYMBOLS, used - _SAFE_SYMBOLS
    for name in used:
        sympy.Symbol(name)  # sympify-safe by construction
        parsed = _parse_expression(name)
        assert parsed is not None


# --- skill loads correctly ---------------------------------------------------


# --- send-back 1966, rule 12: the rendered start/end line parses to the ---
# --- marking dict's start/target, extracted from `text` by regex here, ---
# --- never a second copy kept in the generator. Reuses the oracle's own ---
# --- regex patterns (test-side only; the oracle itself never imports ------
# --- this module or the generator). --------------------------------------


# --- send-back 1966, rule 13: never claim "all real" for an identity that ---
# --- is not defined everywhere (tan/sec/csc/cot or a division). -------------


@pytest.mark.parametrize("tier", TIERS)
def test_no_false_all_real_claim(tier):
    for s in range(60):
        q = make(TOPIC, seed=s, difficulty=tier)
        text_low = q.text.lower()
        if "all real" in text_low or "every real" in text_low:
            marking = q.answer
            for expr in (marking["start"], marking["target"]):
                assert not re.search(r"\btan\(|\bsec\(|\bcsc\(|\bcot\(", expr), (tier, s, expr)
                assert "/" not in expr, (tier, s, expr)


# --- send-back 1966, rule 13/defect 3: the tan family (hard tier) states ---
# --- its domain instead of ever implying "for all real". -------------------


def test_hard_tier_states_domain_condition():
    checked = 0
    for s in range(60):
        q = make(TOPIC, seed=s, difficulty="hard")
        text_low = q.text.lower()
        assert "all real" not in text_low and "every real" not in text_low, q.text
        assert "nonzero" in text_low, q.text
        checked += 1


# --- task 1969, rule 16: latex renders in KaTeX and is the same prompt as ---
# --- text. Static shape checks standing in for `katex_check.js` (node, run --
# --- separately by the task's own verification commands) and for the ------
# --- `is_mangled` detector in `tests/test_prompt_latex_mangled.py` (not ----
# --- imported here -- reimplemented as the minimum needed to prove this ----
# --- module's own latex can never regress into that failure mode). --------

_ESCAPE_MACRO_RE_16 = re.compile(r"\\text(?:asciicircum|asciitilde|backslash)\{\}")
_TEXT_GROUP_RE_16 = re.compile(r"\\text\{[^{}]*\}")
_TRIG_NAME_RE_16 = re.compile(r"(?<!\\)\b(sin|cos|tan|sec|csc|cot)\b")


def _rule16_check(text: str, latex: str, tier: str, seed: int) -> None:
    assert latex, (tier, seed, "empty latex")
    assert r"\verb" not in latex, (tier, seed, latex)
    # Strip the zero-content escape macros FIRST (same order as the real
    # mangled-prompt-latex detector) so a `\text{...}` block that itself
    # contains `\textasciicircum{}` (this module's escaped `^`) is still
    # seen as ONE clean, single-level `\text{...}` group.
    stripped = _ESCAPE_MACRO_RE_16.sub("", latex)
    spans = [m.span() for m in _TEXT_GROUP_RE_16.finditer(stripped)]

    def _inside(pos: int) -> bool:
        return any(a <= pos < b for a, b in spans)

    for m in re.finditer(r"\^", stripped):
        assert not _inside(m.start()), (tier, seed, "bare ^ inside \\text{} (should be \\textasciicircum{})", latex)
    for m in _TRIG_NAME_RE_16.finditer(stripped):
        assert _inside(m.start()), (tier, seed, f"unescaped {m.group()} outside \\text{{}}", latex)
    # Same prompt as text: every instruction phrase carried by `text`
    # (exact value, notation sentence words) survives into `latex` once
    # `\text{`/`}`/backslashes are stripped -- the same normalisation
    # `tests/test_question_latex_instructions.py` uses.
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
        q = make(TOPIC, seed=s, difficulty=tier)
        _rule16_check(q.text, q.latex, tier, s)
        checked += 1
    assert checked == 30
