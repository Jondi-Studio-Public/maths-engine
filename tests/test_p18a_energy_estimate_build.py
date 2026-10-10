"""Tests `ucv1-energy-estimate`: an energy estimate answered as an expression."""

import pytest

from mathgen.grading.typed import grade_expression
from mathgen.skills import load_skills
from mathgen import make

SKILL_ID = "ucv1-energy-estimate"
SEEDS = range(40)
SWEEP_SEEDS = range(60)
DRAW_SEEDS = range(200)
KEY_SWEEP_SEEDS = range(400)
TIERS = ["easy", "medium", "hard", "exam"]


def _skill_sources():
    _domains, skills = load_skills()
    skill = next(s for s in skills if s.id == SKILL_ID)
    return skill, {src.difficulty: src for src in skill.sources}


SKILL, SOURCES = _skill_sources()


def _make(difficulty, seed):
    src = SOURCES[difficulty]
    return make(src.topic, seed=seed, difficulty=src.difficulty, **(src.params or {}))


@pytest.mark.parametrize("difficulty", TIERS)
def test_real_grader_rejects_a_deliberately_wrong_answer(difficulty):
    for seed in SEEDS:
        q = _make(difficulty, seed)
        wrong = f"({q.answer_text}) + 1000"
        assert not grade_expression(q, wrong).correct, (
            f"seed {seed} {difficulty}: deliberate wrong {wrong} not rejected"
        )


def test_grader_probe_accepts_the_task_files_spellings():
    q = _make("easy", 0)
    canonical = q.answer_text  # e.g. "25*exp(-45*pi^2*t/2)"
    variants = [
        canonical,
        canonical.replace("pi", "π"),
        canonical.replace("exp(", "e^(").replace("*", ""),
        canonical.lower(),
    ]
    for v in variants:
        assert grade_expression(q, v).correct, f"probe spelling rejected: {v!r} (canonical {canonical!r})"


@pytest.mark.parametrize("difficulty", TIERS)
def test_at_least_two_distractors_on_most_draws(difficulty):
    counts = {0: 0, 1: 0, 2: 0, 3: 0}
    for seed in DRAW_SEEDS:
        q = _make(difficulty, seed)
        counts[min(len(q.distractors), 3)] += 1
    with_two_plus = counts[2] + counts[3]
    assert with_two_plus / len(DRAW_SEEDS) >= 0.85, (
        f"{difficulty}: only {with_two_plus}/{len(DRAW_SEEDS)} draws had >=2 distractors ({counts})"
    )


@pytest.mark.parametrize("difficulty", TIERS)
def test_top_key_share_under_15_percent(difficulty):
    keys = [_make(difficulty, seed).answer_text for seed in DRAW_SEEDS]
    top_share = max(keys.count(k) for k in set(keys)) / len(keys)
    assert top_share < 0.15, f"{difficulty}: top key share {top_share:.3f} >= 0.15"


def test_default_params_two_plus_distractors():
    # make(topic, seed) with no difficulty -- the generator's own default tier.
    for seed in range(60):
        q = make("p18a_energy_estimate", seed=seed)
        assert len(q.distractors) >= 1


def test_easy_is_dirichlet_full_mode():
    for seed in SEEDS:
        q = _make("easy", seed)
        assert "Dirichlet" in q.text
        assert "sin" in q.text and "cos" not in q.text


def test_medium_is_neumann_cosine_mode():
    for seed in SEEDS:
        q = _make("medium", seed)
        assert "insulated" in q.text or "Neumann" in q.text
        assert "cos" in q.text


def test_hard_is_mixed_or_offset():
    # The mixed variant always keeps [0, L] and states exactly one insulated
    # (derivative) end alongside a fixed end; the offset variant is pure
    # Dirichlet on [a, b] with a != 0 and never mentions a derivative end --
    # a clean discriminator between hard's two variants that does not depend
    # on how the mixed mode's fractional wavenumber happens to be displayed
    # (the display is the REDUCED fraction, e.g. "7*pi*x/12", never
    # the literal "n + 1/2").
    seen = {"mixed": False, "offset": False}
    for seed in SWEEP_SEEDS:
        q = _make("hard", seed)
        if "[0, " in q.text:
            seen["mixed"] = True
            assert "u_x(" in q.text
        else:
            seen["offset"] = True
            assert "u_x(" not in q.text
    assert seen["mixed"], "hard: mixed-condition variant never drawn over 60 seeds"
    assert seen["offset"], "hard: offset-interval variant never drawn over 60 seeds"


def test_exam_is_ic_given_or_neumann_zero():
    seen = {"ic": False, "zero": False}
    for seed in SWEEP_SEEDS:
        q = _make("exam", seed)
        if "n = 0" in q.text:
            seen["zero"] = True
        elif "u(x,0)" in q.text:
            seen["ic"] = True
    assert seen["ic"], "exam: IC-given variant never drawn over 60 seeds"
    assert seen["zero"], "exam: n=0 Neumann variant never drawn over 60 seeds"


def test_neumann_zero_energy_has_no_decay_term():
    for seed in SWEEP_SEEDS:
        q = _make("exam", seed)
        if "n = 0" in q.text:
            assert "exp(" not in q.answer_text, f"seed {seed}: n=0 answer should not decay: {q.answer_text}"


def test_skill_loads_through_api_with_correct_properties():
    assert SKILL.domain == "advanced-pdes-and-dynamics"
    assert SKILL.status == "active"
    assert SKILL.placement_eligible is False
    assert SKILL.answer == "expression"
    assert SKILL.prerequisites == ["ucv1-heat-equation-mode"]
    assert {src.difficulty for src in SKILL.sources} == {"easy", "medium", "hard", "exam"}
    exam_src = SOURCES["exam"]
    assert exam_src.stages == ["exam"]


# -- in-generator independent check fires on a corrupted value -------------


def test_independent_check_fires_on_corrupted_energy():
    from mathgen.core import VerificationError
    from mathgen.generators.p18a_energy_estimate_build import _verify_energy, _energy_key

    # A corrupted formula (missing the 1/2 factor) must fail the independent
    # double-angle re-derivation.
    import mathgen.generators.p18a_energy_estimate_build as mod

    original = mod._energy_key
    try:
        mod._energy_key = lambda A, family, m, L, k: A**2 * L  # drops the 1/2 AND the decay
        with pytest.raises(Exception):
            _verify_energy(5, "dirichlet", 2, 7, 3)
    finally:
        mod._energy_key = original


def test_independent_double_angle_check_fires_on_wrong_half_period():
    from mathgen.core import symbolically_equal, check

    # The independent double-angle antiderivative for a Dirichlet mode must
    # equal L/2 -- a corrupted claim of L/3 must fail the same `check()` call
    # `_verify_energy` makes.
    L = 7
    with pytest.raises(Exception):
        check(symbolically_equal(L / 2, L / 3), "deliberately corrupted L/2 claim")
