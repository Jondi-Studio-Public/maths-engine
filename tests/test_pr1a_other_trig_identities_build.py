"""PR1A task 1965 -- `senior-prove-and-apply-other-trigonometric-identities-such-as`
(ACMSM049) build tests. `tests/test_pr1a_other_trig_identities_oracle.py` is
the independent, generator-blind check on the learner-visible `text`/`start`/
`target`; this file proves the marking dict's own shape and the in-generator checks. The
rubric-marking checks of each reference chain and trap are not part of this package.
"""

from __future__ import annotations

import re

import pytest
import sympy

import mathgen
from mathgen.generators import pr1a_other_trig_identities_build as gen  # noqa: F401  (registers the topic)
from mathgen.grading.typed import _parse_expression

TOPIC = "pr1a_other_trig_identity_proof"
TIERS = ("easy", "medium", "hard", "exam")
N_SEEDS = 200

# ---------------------------------------------------------------------------
# Determinism and source wiring
# ---------------------------------------------------------------------------


def test_determinism():
    for tier in TIERS:
        q1 = mathgen.make(TOPIC, seed=7, difficulty=tier)
        q2 = mathgen.make(TOPIC, seed=7, difficulty=tier)
        assert q1.answer == q2.answer
        assert q1.text == q2.text


# ---------------------------------------------------------------------------
# Marking dict shape (common rules 1, 2, 4, 5) and readable text
# ---------------------------------------------------------------------------

_SAFE_NAMES = {"x", "t", "u", "v", "A", "B", "P", "R", "theta", "alpha", "phi"}
_GENERATOR_VARS = ("x", "t", "u", "v")


def test_symbol_names_round_trip_sympify_and_parse_expression():
    for name in _GENERATOR_VARS:
        assert name in _SAFE_NAMES
        expr = sympy.sympify(name)
        assert isinstance(expr, sympy.Symbol)
        parsed = _parse_expression(name)
        assert parsed is not None


_GREEK_RE = re.compile(r"[Ͱ-ορ-Ͽ]|\btheta\b|\balpha\b|\bphi\b")
_UNICODE_SUPERSCRIPT_RE = re.compile(r"[²⁴]")


@pytest.mark.parametrize("tier", TIERS)
def test_no_greek_and_no_unicode_superscript_anywhere(tier):
    """Common rule 10/12: Latin letters only, and every power is `^`, in
    both the rendered `text` and the marking dict's own strings. `√` is the
    one non-ASCII glyph this pool's apply items use (surd exact values),
    explicitly outside `_UNICODE_SUPERSCRIPT_RE`."""
    for q in _marking_blocks(tier, 60).values():
        assert not _GREEK_RE.search(q.text), q.text
        assert not _UNICODE_SUPERSCRIPT_RE.search(q.text), q.text
        assert not _GREEK_RE.search(q.answer_text), q.answer_text
        assert not _UNICODE_SUPERSCRIPT_RE.search(q.answer_text), q.answer_text
        for given in q.answer.get("key_steps", []):
            assert not _GREEK_RE.search(given["description"]), given


_TRIG_DIVISION_RE = re.compile(r"tan\(|/\s*\(?\s*(sin|cos|tan)\(|\)\s*/\s*(sin|cos|tan)\(")


@pytest.mark.parametrize("tier", TIERS)
def test_all_real_domain_claim_is_only_made_when_true(tier):
    """Common rule 13, checked in its literal form: every text containing
    'all real' or 'every real' has a target and start free of tan/sec/csc/cot
    and of division by a trig term. A restricted family (the exam-tier
    composite items) states 'given that ...' instead and never says
    'real' at all."""
    for q in _marking_blocks(tier, 60).values():
        restricted = bool(_TRIG_DIVISION_RE.search(q.answer["start"])) or bool(
            _TRIG_DIVISION_RE.search(q.answer["target"])
        )
        claims_domain = "all real" in q.text or "every real" in q.text
        if claims_domain:
            assert not restricted, q.text  # rule 13's literal test
        qualified = "given that" in q.text
        if not (claims_domain or qualified):
            continue  # an apply item never states a domain at all
        if restricted:
            assert qualified, q.text
            assert not claims_domain, q.text
        else:
            assert not qualified, q.text


@pytest.mark.parametrize("tier", TIERS)
def test_notation_sentence_present(tier):
    """Common rule 11: every prompt carries the short notation sentence, in
    both `text` and `latex`."""
    for q in _marking_blocks(tier, 20).values():
        assert "put brackets round every angle" in q.text
        assert "put brackets round every angle" in q.latex


_BARE_TRIG_ARGUMENT_RE = re.compile(r"\b(sin|cos|tan)\s+[A-Za-z0-9]")


def test_givens_bracket_every_function_argument():
    """Common rule 4: the stated givens use the same bracketed form as the
    identity they support."""
    for given in (
        gen.SIN_SUM_GIVEN,
        gen.COS_SUM_GIVEN,
        gen.PYTH_GIVEN,
        gen.SIN2_GIVEN,
        gen.COS2_2COS2M1_GIVEN,
        gen.COS2_1M2SIN2_GIVEN,
        gen.TRIPLE_COS_GIVEN,
        gen.TRIPLE_SIN_GIVEN,
    ):
        assert not _BARE_TRIG_ARGUMENT_RE.search(given), given
        assert "²" not in given and "⁴" not in given, given


def test_apply_items_state_radians_not_degrees_alone():
    """Common rule 14: the apply half states the angle in radians (`π/9`);
    a degree value may follow, but only inside the explicit "(that is, N°)"
    aside -- never as the angle the learner evaluates."""
    apply_qs = [q for q in _marking_blocks("easy", 200).values() if not q.answer["symbols"]]
    assert apply_qs
    for q in apply_qs:
        assert "π" in q.text, q.text
        for m in re.finditer(r"(\d+(?:\.\d+)?)°", q.text):
            before = q.text[: m.start()]
            assert before.rstrip().endswith("that is,"), q.text


def _marking_blocks(tier, n_seeds):
    seen = {}
    for seed in range(n_seeds):
        q = mathgen.make(TOPIC, seed=seed, difficulty=tier)
        key = (q.answer["start"], q.answer["target"], tuple(q.answer["reference_lines"]))
        seen[key] = q
    return seen


@pytest.mark.parametrize("tier", TIERS)
def test_marking_dict_shape(tier):
    for q in _marking_blocks(tier, 40).values():
        marking = q.answer
        assert isinstance(marking, dict)
        assert set(marking) >= {"start", "target", "symbols", "reference_lines", "key_steps", "traps", "checks"}
        assert marking["checks"] == [
            {
                "type": "equivalence_chain",
                "start": marking["start"],
                "target": marking["target"],
                "symbols": marking["symbols"],
                "authoritative": True,
            }
        ]
        assert len(marking["checks"]) == 1  # common rule 1: exactly one check, never a numeric check beside it
        for step in marking["key_steps"]:
            assert step["criterion_id"] in {"correctness", "justified_steps", "communication"}
        assert any(s["criterion_id"] == "justified_steps" for s in marking["key_steps"])
        assert any(s["criterion_id"] == "communication" for s in marking["key_steps"])
        for trap in marking["traps"]:
            assert trap["expected_check"] in {"authoritative_failure", "non_authoritative_failure", "passes_chain"}
            assert trap["example_lines"]
        assert "**" not in q.text and re.search(r"(?<![A-Za-z0-9])\*(?![A-Za-z0-9])", q.text) is None
        assert "**" not in q.answer_text


@pytest.mark.parametrize("tier", TIERS)
def test_at_least_two_named_identities_per_tier(tier):
    """The task file's pool-variety requirement: at least two named
    identities per tier, never 60 seeds of the one manifest example."""
    targets = {q.answer["target"] for q in _marking_blocks(tier, 60).values()}
    assert len(targets) >= 2, (tier, targets)


def test_every_required_trap_name_appears_somewhere_in_the_pool():
    """Task file's "traps (at least)" list -- every named trap appears at
    least once across the whole generator's pool (not necessarily on every
    item)."""
    required = {
        "triples_the_function",
        "sign_slip_in_triple_angle",
        "wrong_cos2x_form_midway",
        "starts_from_rhs",
        "cites_what_it_proves",
        "divides_by_zero_unstated",
        "wrong_special_value",
    }
    seen: set[str] = set()
    for tier in TIERS:
        for q in _marking_blocks(tier, N_SEEDS).values():
            for trap in q.answer["traps"]:
                seen.add(trap["name"])
    missing = required - seen
    assert not missing, missing


# ---------------------------------------------------------------------------
# decide() short-circuit (common rule: decision 3)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# In-generator check fires on a corrupted reference line
# ---------------------------------------------------------------------------


def test_in_generator_check_fires_on_corrupted_pool_entry(monkeypatch):
    real_right = gen._fam_cos3("x")["math_right"]
    broken = gen._fam_cos3

    def _corrupted(s):
        fam = broken(s)
        fam["math_right"] = lambda v: real_right(v) + 1.0  # break the identity
        return fam

    monkeypatch.setattr(gen, "_fam_cos3", _corrupted)
    monkeypatch.setattr(gen, "EASY_PROVE_BUILDERS", (_corrupted, gen._fam_sin3))
    saw_error = False
    from mathgen.core import VerificationError

    for seed in range(60):
        try:
            mathgen.make(TOPIC, seed=seed, difficulty="easy")
        except VerificationError:
            saw_error = True
            break
    assert saw_error, "corrupting a pool entry's math_right never raised VerificationError"


# --- common rule 16: latex renders in KaTeX and is the same prompt as text. --
# --- Static shape checks standing in for `katex_check.js` (node, run --------
# --- separately by the task's own verification commands). ------------------

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
