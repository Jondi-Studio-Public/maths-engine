"""Task 1321 (packet P13C, build 2/5): senior-use-the-notation-int-f-left-x-right
(ACMMM115, topic ``p13c_antideriv_notation``, answer: expression).
"""

import re

import pytest
from sympy import Symbol, diff, sympify

from mathgen.grading import GRADERS
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError, check_distinct, check_equivalent

SEEDS = range(60)
TIERS = ("easy", "medium", "hard", "exam")

TOPIC = "p13c_antideriv_notation"
SKILL = "senior-use-the-notation-int-f-left-x-right"

_grade_expression = GRADERS["expression"]

x = Symbol("x")


# --- Determinism -------------------------------------------------------


# --- Distractor hygiene --------------------------------------------------


# --- Real grader: accepts canonical, rejects distractors and a wrong value --


@pytest.mark.parametrize("tier", TIERS)
def test_real_grader_accepts_canonical_and_rejects_distractors(tier):
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=tier)
        result = _grade_expression(q, q.answer_text)
        assert result.correct, f"{tier} seed={seed}: canonical {q.answer_text!r} not accepted"
        for d in q.distractors:
            result = _grade_expression(q, d)
            assert not result.correct, f"{tier} seed={seed}: distractor {d!r} accepted"


@pytest.mark.parametrize("tier", TIERS)
def test_real_grader_rejects_a_deliberately_wrong_value(tier):
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=tier)
        wrong = f"{q.answer} + 999999 + C"
        result = _grade_expression(q, wrong)
        assert not result.correct, f"{tier} seed={seed}: deliberately wrong {wrong!r} accepted"


# --- The +C grading finding, proven directly through the real grader -------
# (mathgen/grading/typed.py::_expected_expression's docstring, quoted in the module
# docstring above): the DISPLAYED answer_text ("F(x) + C") is authoritative
# whenever it differs from the internal `answer` (bare F(x)) by exactly one
# free symbol -- so a learner who drops "+ C" is compared against the full
# "F(x) + C" and correctly rejected, not silently accepted against the bare
# internal value.


@pytest.mark.parametrize("tier", TIERS)
def test_plus_c_grading_finding_direct(tier):
    checked = 0
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=tier)
        # The canonical, WITH "+ C", is accepted.
        assert _grade_expression(q, q.answer_text).correct, f"{tier} seed={seed}: canonical with +C not accepted"
        # The bare internal answer, WITHOUT "+ C", is REJECTED -- this is the
        # "dropped +C" misconception firing through the real grader, not just
        # in-generator.
        assert not _grade_expression(q, q.answer).correct, (
            f"{tier} seed={seed}: bare answer (missing +C) was wrongly accepted"
        )
        # A different constant LETTER ("D") is a DIFFERENT free symbol to the
        # grader and must not be treated as equivalent to the canonical "+ C".
        wrong_letter = f"{q.answer} + D"
        assert not _grade_expression(q, wrong_letter).correct, f"{tier} seed={seed}: +D accepted as if it were +C"
        checked += 1
    assert checked == len(SEEDS)


# --- Independent-check tripwire ---------------------------------------------


def test_independent_check_fires_on_corrupted_answer():
    with pytest.raises(VerificationError):
        check_equivalent(1, 2, "should fail")
    with pytest.raises(VerificationError):
        check_distinct(1, 1, msg="should fail")


# --- Distinct rendered questions -------------------------------------------


# --- Answer kind: an expression, with the answer satisfying the notation ---


@pytest.mark.parametrize("tier", TIERS)
def test_answer_text_is_the_answer_plus_c(tier):
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=tier)
        # answer_text is "<answer, ** rewritten to ^> + C" -- q.answer itself
        # is the raw sympy str (kept with "**", never shown to a learner
        # directly), so compare by VALUE rather than exact string.
        assert q.answer_text.endswith(" + C"), (
            f"{tier} seed={seed}: answer_text {q.answer_text!r} does not end in ' + C'"
        )
        answer_part = q.answer_text[: -len(" + C")]
        assert sympify(answer_part.replace("^", "**")) == sympify(q.answer), (
            f"{tier} seed={seed}: answer_text {q.answer_text!r} does not match answer {q.answer!r}"
        )
        # The internal answer never itself carries "+ C" or "C" as a token.
        parsed = sympify(q.answer)
        assert Symbol("C") not in parsed.free_symbols, (
            f"{tier} seed={seed}: internal answer unexpectedly carries a free C"
        )


@pytest.mark.parametrize("tier", TIERS)
def test_distractors_formatted_like_the_answer(tier):
    # Every distractor is an expression string (optionally with "+ C"), same
    # shape family as the canonical -- never a bare number, never wrapped in
    # a different label. Distractors are VALUE strings (only "**" rewritten
    # to "^", a bare "*" left alone -- see `_value` in the generator), never
    # PROMPT strings, so no "·" is expected here.
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=tier)
        for d in q.distractors:
            assert re.match(r"^-?[0-9A-Za-z*/().\s+^-]+$", d), (
                f"{tier} seed={seed}: distractor not expression-shaped: {d!r}"
            )
            assert "**" not in d, f"{tier} seed={seed}: distractor still has Python ** operator: {d!r}"


# --- Boundary: never a bare "find the integral of f(x)" with no stated F ---
# (the integrate-polynomial/integrate-trig operation this row must not
# reproduce -- see module docstring "THIS IS THE HIGH-RISK ROW").


@pytest.mark.parametrize("tier", TIERS)
def test_premise_always_states_a_differentiation_fact(tier):
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=tier)
        assert "d/dx" in q.text or "derivative of" in q.text, (
            f"{tier} seed={seed}: no stated differentiation fact in {q.text!r}"
        )


@pytest.mark.parametrize("tier", TIERS)
def test_never_a_bare_integrate_polynomial_or_trig_prompt(tier):
    """Boundary against integrate-polynomial/integrate-trig: those skills'
    prompts are "Find ∫ (<bare integrand>) dx" with NO stated F/derivative
    fact anywhere in the text. Every prompt here must carry that fact.
    """
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=tier)
        # A bare-integrand prompt never mentions "d/dx" or "derivative of" at
        # all; ours always does (checked above). This test additionally
        # confirms the ONLY "Find" pattern (integrate-polynomial's own
        # verbatim template) does not appear on its own without that fact
        # immediately preceding it in the same sentence.
        assert not re.match(r"^Find ∫ \(.*\) dx$", q.text.strip()), (
            f"{tier} seed={seed}: bare integrate-polynomial/-trig-shaped prompt: {q.text!r}"
        )


@pytest.mark.parametrize("tier", TIERS)
def test_never_a_bare_solve_ode_prompt(tier):
    """Boundary against solve-ode-separable-poly: that skill's prompt is
    "Solve dy/dx = <polynomial>" with no F fact stated. Never produced here.
    """
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=tier)
        assert "dy/dx" not in q.text, f"{tier} seed={seed}: dy/dx ODE phrasing found: {q.text!r}"
        assert not q.text.strip().startswith("Solve "), f"{tier} seed={seed}: 'Solve ...' ODE-shaped prompt: {q.text!r}"


# --- hard/exam: F is not reachable via the polynomial power rule alone -----


def test_hard_tier_uses_a_non_polynomial_family():
    checked = 0
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty="hard")
        # "e^(" not "exp(": the prompt display fix (task 1325) renders
        # exp(...) as e^(...) for a senior audience -- see _exp_to_e_caret.
        assert any(tok in q.text for tok in ("e^(", "sin(", "cos(")), (
            f"seed={seed}: hard tier did not use exp/sin/cos family: {q.text!r}"
        )
        checked += 1
    assert checked == len(SEEDS)


# NOTE (task 1325, red-first evidence): before this fix, the hard tier drew
# an `ln` family (`F = a*ln(x)`, canonical `a*log(x) + C`) and a version of
# this test iterated every hard-tier seed in range(60) (16 of the 60 seeds
# were `ln`-family), building the canonical text, an `ln(x)` spelling, a
# `log(x)` spelling, a bare-juxtaposition spelling and an `ln|x|`
# bar-notation spelling, then asserted `_grade_expression` accepted all
# five. Run directly against the untouched generator on commit 985aeed, it
# failed on the FIRST ln-family seed reached (seed=0):
#
#   AssertionError: hard seed=0: ln-family spelling '-2*ln|x| + C' not
#   accepted (canonical was '-2*log(x) + C')
#   assert False
#    +  where False = GradeResult(correct=False,
#       canonical='-2*log(x) + C', given_normalised='-2*x', note=None,
#       note_kind=None).correct
#
# The `ln` family has now been
# removed from the hard tier by construction (see the module docstring's
# history note and `_draw_cos` below), so that scenario can no longer be
# exercised through the live generator. The known grader gap it exposed
# (`mathgen/grading/typed.py::_parse_expression` cannot parse `|...|` bar-absolute-
# value notation at all) is still real and still belongs to `mathgen/grading/typed.py`
# -- documented below as a standalone, generator-independent regression so a
# future reader does not have to rediscover it.


class _SyntheticLnQuestion:
    """Hand-built stand-in for a `Question`-like object carrying an
    `ln`-family answer, shaped exactly like the live generator's own
    `_finish()` output for the (now-removed) `ln` family: `answer` is the
    bare antiderivative string, `answer_text` is the same value with
    `" + C"` appended, matching every other question this module builds.
    """

    def __init__(self, answer: str):
        self.answer = answer
        self.answer_text = f"{answer} + C"
        self.answer_form = None
        self.answer_alt = None


def test_ln_bar_notation_grader_gap_is_undisturbed_documentation():
    """DOCUMENTATION test (task 1325): confirms the `mathgen/grading/typed.py::
    _parse_expression` bar-absolute-value gap the ln family used to trip is
    still exactly what it was -- unrelated to, and undisturbed by, this
    packet's family swap. `ln|x|` is NOT valid Python syntax, so the sympy
    parse throws and grading falls through to `spoken_to_expr`, which
    silently discards the `ln` and returns the bare symbol `x` -- so a
    correct `ln|x|` answer is (still) wrongly rejected. Ordinary spellings
    (`ln(x)`, `log(x)`, bare-juxtaposition) are unaffected and still accepted.
    This test is a permanent regression marker for the KNOWN grader gap, not
    a claim that removing the `ln` family fixed it -- it did not; nothing in
    `mathgen/grading/typed.py` changed. Fixing the parser itself belongs to that
    module, not to this generator.
    """
    q = _SyntheticLnQuestion("-2*log(x)")
    # Ordinary spellings: still accepted, exactly as before.
    assert _grade_expression(q, "-2*log(x) + C").correct
    assert _grade_expression(q, "-2*ln(x) + C").correct
    assert _grade_expression(q, "-2ln(x)+C").correct
    # The bar-absolute-value spelling: still REJECTED. This is the known,
    # pre-existing `mathgen/grading/typed.py` gap, not something this packet caused or
    # could safely fix (see module docstring's "Do not attempt to patch
    # mathgen/grading/typed.py").
    result = _grade_expression(q, "-2*ln|x| + C")
    assert not result.correct, (
        "mathgen/grading/typed.py now parses '|x|' bar-absolute-value notation -- if "
        "this starts passing, the grader gap this test documents has been "
        "fixed upstream; update the module docstring's history note "
        "accordingly (it no longer needs to explain why ln is excluded on "
        "notation-parsing grounds alone)."
    )


def test_hard_tier_cos_family_textbook_spellings_accepted():
    """Red-then-green (task 1325): on commit 985aeed the hard tier never
    draws a `cos` family at all (`family = rng.choice(["exp", "sin", "ln"])`),
    so this test failed with `checked_cos_seeds == 0` before the fix. After
    swapping `ln` for `cos` in `_hard`, this exercises the new family through
    the real generator + real grader: the canonical `cos(kx)` answer is
    accepted in several textbook-plausible spacing/juxtaposition spellings,
    and the `-a*k*sin(kx)` sign-error misconception distractor is correctly
    rejected.

    NOTE: the task file additionally asked for a "`.`/`·` vs `*`" spelling
    to be accepted. Checked directly and it is NOT: `mathgen/grading/typed.py` has no
    handling anywhere for the middle-dot multiplication character (grepped;
    none found), so `grade_expression(q, canonical.replace("*", "·"))`
    returns `correct=False` regardless of family -- the same class of
    parser gap as the `ln|x|` bar-notation finding, just for a different
    character, and equally out of scope here. The generator itself never emits "·" in a graded
    field -- `_value()`/`_display()` only ever produce "*", never "·" (that
    substitution is `_disp()`'s PROMPT-only concern) -- so this is a
    hypothetical learner spelling, not something the generator's own output
    needs to survive. Also, no cos-family answer here ever contains a "^"
    (there is no exponent in `a*cos(k*x)`), so a "^ vs **" spelling does not
    apply to this family either; that comparison is already exercised for
    the polynomial tiers by `test_answer_text_is_the_answer_plus_c`.
    """
    checked_cos_seeds = 0
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty="hard")
        # Filter on the ANSWER, not the prompt text: the sin family's stated
        # derivative fact is itself "a*k*cos(kx)", so "cos(" appears in
        # q.text for sin-family seeds too. The cos family is the only one
        # whose canonical F(x) -- q.answer_text -- contains "cos(".
        if "cos(" not in q.answer_text:
            continue
        checked_cos_seeds += 1
        canonical = q.answer_text
        assert "cos(" in canonical, f"hard seed={seed}: cos family answer missing cos(: {canonical!r}"
        assert "**" not in canonical
        spaced = canonical.replace("cos(", "cos( ").replace(")", " )")
        juxtaposed = canonical.replace("*cos(", "cos(").replace(" ", "")
        for spelling in (canonical, spaced, juxtaposed):
            result = _grade_expression(q, spelling)
            assert result.correct, (
                f"hard seed={seed}: cos-family spelling {spelling!r} not accepted (canonical was {canonical!r})"
            )
        for d in q.distractors:
            result = _grade_expression(q, d)
            assert not result.correct, f"hard seed={seed}: cos-family distractor {d!r} wrongly accepted"
    assert checked_cos_seeds > 0, "no cos-family hard-tier seed found in range(60)"


def test_exam_tier_combines_two_stated_facts():
    checked = 0
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty="exam")
        assert q.text.count("d/dx") == 2 or q.text.count("derivative of") == 2, (
            f"seed={seed}: exam tier does not state exactly two facts: {q.text!r}"
        )
        # "e^(" not "exp(": see the hard-tier test's comment above.
        assert any(tok in q.text for tok in ("e^(", "sin(")), (
            f"seed={seed}: exam tier's second fact is not exp/sin: {q.text!r}"
        )
        checked += 1
    assert checked == len(SEEDS)


# --- Skill loading -----------------------------------------------------


def test_skill_loads_with_correct_properties():
    _, skills = load_skills()
    skill = next((s for s in skills if s.id == SKILL), None)
    assert skill is not None, f"{SKILL} not found"
    assert skill.answer == "expression"
    assert skill.placement_eligible is False
    assert skill.prerequisites == ["differentiate-polynomial"]
    assert len(skill.sources) == 4
    for source in skill.sources:
        assert source.topic == TOPIC
    assert set(skill.sources[0].stages) == {0, 1, 2}
    assert set(skill.sources[1].stages) == {3, 4}
    assert set(skill.sources[2].stages) == {5, 6, "maintenance"}
    assert set(skill.sources[3].stages) == {"exam"}
    assert skill.sources[3].difficulty == "exam"
