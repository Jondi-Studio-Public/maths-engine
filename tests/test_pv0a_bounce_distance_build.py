"""Pv0a_bounce_distance (senior-use-geometric-sequences-to-model-and-analyse-numerically, ACMGM074)."""

import functools
import re
from fractions import Fraction

import mathgen.generators.pv0a_bounce_distance_build  # noqa: F401 -- registers the topic
from mathgen.grading.typed import grade_number
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError, check, check_pairwise_distinct, rounds_unambiguously

DIFFICULTIES = ["easy", "medium", "hard", "exam"]
SEEDS = range(60)
TOPIC = "pv0a_bounce_distance"
SKILL_ID = "senior-use-geometric-sequences-to-model-and-analyse-numerically"


@functools.lru_cache(maxsize=None)
def _skills_dict():
    _, skills = load_skills()
    return {s.id: s for s in skills}


def _make_from_sources(difficulty, seed):
    skill = _skills_dict()[SKILL_ID]
    src = next(s for s in skill.sources if s.difficulty == difficulty)
    return make(src.topic, seed=seed, difficulty=src.difficulty, **(src.params or {}))


# =============================================================================
# Determinism and shape checks shared across tiers
# =============================================================================


def test_deterministic_across_seeds():
    for difficulty in DIFFICULTIES:
        for seed in SEEDS:
            a = make(TOPIC, seed=seed, difficulty=difficulty)
            b = make(TOPIC, seed=seed, difficulty=difficulty)
            assert a.text == b.text
            assert a.answer_text == b.answer_text
            assert a.distractors == b.distractors


def test_answer_is_money_style_2dp():
    for difficulty in DIFFICULTIES:
        for seed in SEEDS:
            q = make(TOPIC, seed=seed, difficulty=difficulty)
            assert "." in q.answer_text, (difficulty, seed, q.answer_text)
            whole, frac = q.answer_text.split(".")
            assert len(frac) == 2, (difficulty, seed, q.answer_text)
            for d in q.distractors:
                assert "." in d and len(d.split(".")[1]) == 2, (difficulty, seed, d)


def test_distractors_distinct_nonempty_never_equal_answer():
    for difficulty in DIFFICULTIES:
        for seed in SEEDS:
            q = make(TOPIC, seed=seed, difficulty=difficulty)
            assert len(q.distractors) >= 1, (difficulty, seed)
            assert len(set(q.distractors)) == len(q.distractors), (difficulty, seed)
            assert q.answer_text not in q.distractors, (difficulty, seed)


def test_at_least_two_distractors_on_most_draws():
    for difficulty in DIFFICULTIES:
        counts = [len(make(TOPIC, seed=s, difficulty=difficulty).distractors) for s in range(200)]
        at_least_two = sum(1 for c in counts if c >= 2)
        assert at_least_two / 200 >= 0.85, (difficulty, at_least_two)


def test_at_least_20_distinct_questions_per_tier_over_60_seeds():
    for difficulty in DIFFICULTIES:
        texts = {make(TOPIC, seed=s, difficulty=difficulty).text for s in range(60)}
        assert len(texts) >= 20, (difficulty, len(texts))


def test_skill_loads_with_expected_shape():
    skill = _skills_dict()[SKILL_ID]
    assert skill.answer == "number"
    assert skill.placement_eligible is False
    assert skill.prerequisites == ["simple-interest"]
    difficulties = {s.difficulty for s in skill.sources}
    assert difficulties == set(DIFFICULTIES)
    exam_source = next(s for s in skill.sources if s.difficulty == "exam")
    assert exam_source.stages == ["exam"]


# =============================================================================
# Real-grader accept/reject
# =============================================================================


def test_number_grader_rejects_a_deliberately_wrong_value():
    for difficulty in DIFFICULTIES:
        q = _make_from_sources(difficulty, 0)
        whole, frac = q.answer_text.split(".")
        wrong = f"{int(whole) + 97}.{frac}"
        assert not grade_number(q, wrong).correct, (difficulty, q.text, wrong)


# =============================================================================
# Independent in-generator checks fire on a corrupted answer
# =============================================================================


def test_check_pairwise_distinct_fires_on_a_corrupted_route():
    fired = False
    try:
        check_pairwise_distinct(Fraction(1), Fraction(1), msg="deliberately colliding")
    except VerificationError:
        fired = True
    assert fired


def test_check_fires_on_a_corrupted_condition():
    fired = False
    try:
        check(False, "deliberately false")
    except VerificationError:
        fired = True
    assert fired


def test_generators_own_impact_index_guard_fires():
    from mathgen.generators.pv0a_bounce_distance_build import _dist_to_impact

    fired = False
    try:
        _dist_to_impact(Fraction(1), Fraction(1, 2), 0)
    except VerificationError:
        fired = True
    assert fired


# =============================================================================
# Boundary/plausibility property: the exact total never sits on
# the 2dp rounding boundary -- proven RED first against an unconstrained draw.
# =============================================================================


def test_rounding_boundary_property_is_proven_red_first():
    """Reproduces the easy-tier construction with NO rounding-boundary guard
    and shows some draws land within the margin `rounds_unambiguously` rejects
    -- proof the real generator's retry loop is doing real work."""
    import random

    from mathgen.generators.pv0a_bounce_distance_build import _dist_to_impact

    failures = 0
    for seed in range(400):
        rng = random.Random(f"unconstrained:{seed}")
        h = Fraction(rng.randint(2, 40), 2)
        r = Fraction(rng.randint(40, 85), 100)
        k = rng.randint(3, 7)
        total = _dist_to_impact(h, r, k)
        if not rounds_unambiguously(float(total), 2):
            failures += 1
    assert failures > 0, "the unconstrained draw never landed on the rounding boundary -- test is not RED"


# The "holds on the real generator" half of this property is asserted by the
# oracle, which re-derives the
# EXACT total from the rendered text by stroke-by-stroke simulation and
# checks it directly with `rounds_unambiguously` -- a genuinely independent
# route from this module's own construction.


# =============================================================================
# Threshold margin property (exam tier): X is kept a relative 5%+ away from
# every candidate rebound height -- proven RED first, then shown to hold.
# =============================================================================

_H_RE = re.compile(r"from (?:a height of )?(\d+(?:\.\d+)?) m")
_R_RE = re.compile(r"(\d+(?:\.\d+)?)% of")
_X_RE = re.compile(r"(?:less than|lower than) (\d+(?:\.\d+)?) m", re.IGNORECASE)


def _parse_exam(text: str):
    h = Fraction(str(_H_RE.search(text).group(1)))
    r = Fraction(int(round(float(_R_RE.search(text).group(1)) * 100)), 10000)
    x = Fraction(_X_RE.search(text).group(1))
    return h, r, x


def test_exam_threshold_margin_is_proven_red_first():
    """Draws an UNCONSTRAINED threshold (uniform between two consecutive
    rebound heights, no 5% relative-margin floor) and shows it sometimes
    lands within 5% of a rebound height -- proof the real generator's margin
    check is doing real work."""
    import random

    close_calls = 0
    for seed in range(400):
        rng = random.Random(f"unconstrained-exam:{seed}")
        h = Fraction(rng.randint(2, 40), 2)
        r = Fraction(rng.randint(40, 85), 100)
        k_target = rng.randint(4, 9)
        heights = [h * r**i for i in range(1, k_target + 2)]
        above = heights[k_target - 2] if k_target >= 2 else h
        below = heights[k_target - 1]
        f = rng.uniform(0.0, 1.0)  # UNCONSTRAINED, unlike the real [0.25, 0.75]
        x = below + (above - below) * Fraction(str(round(f, 6)))
        if any(abs(x - hh) / hh <= Fraction(1, 20) for hh in heights):
            close_calls += 1
    assert close_calls > 0, "the unconstrained draw never landed within the 5% margin -- test is not RED"


# ============================================================================= No stated GIVEN
# number carries a trailing zero after the decimal point -- only the ANSWER is forced to 2 d.p.
# Proven RED first against the pre-fix formatting, then shown to hold on the real generator over 200
# seeds per tier. =============================================================================

_DECIMAL_RE = re.compile(r"\d+\.\d+")


def _decimal_tokens(text: str) -> list[str]:
    """Every decimal number literally written in `text` -- the precision
    instruction itself ("to 2 decimal places") has no decimal point, so it
    never appears here."""
    return _DECIMAL_RE.findall(text)


def test_no_trailing_zero_property_is_proven_red_first():
    """Reproduces the pre-fix formatting (forcing every GIVEN rebound height
    to 2dp via `stated_precision_string`, as the hard tier used to) and shows
    it produces a trailing zero on some draws -- proof the real generator's
    trimmed `_fmt_given` formatting is doing real work."""
    import random

    from mathgen.core import round_half_up, stated_precision_string
    from mathgen.generators.pv0a_bounce_distance_build import _draw_h, _draw_r

    failures = 0
    for seed in range(200):
        rng = random.Random(f"pre-fix-formatting:{seed}")
        h = _draw_h(rng)
        r = _draw_r(rng)
        start = rng.choice((1, 2))
        h1 = h * r**start
        h2 = h * r ** (start + 1)
        if round_half_up(h1, 2) != h1 or round_half_up(h2, 2) != h2:
            continue
        text = f"Its rebound reaches {stated_precision_string(h1, 2)} m and {stated_precision_string(h2, 2)} m."
        if any(tok.endswith("0") for tok in _decimal_tokens(text)):
            failures += 1
    assert failures > 0, "the pre-fix formatting never produced a trailing zero -- test is not RED"


def test_no_stated_number_has_a_trailing_zero_on_200_seeds_per_tier():
    for difficulty in DIFFICULTIES:
        for seed in range(200):
            q = make(TOPIC, seed=seed, difficulty=difficulty)
            for tok in _decimal_tokens(q.text):
                assert not tok.endswith("0"), (difficulty, seed, tok, q.text)


def test_exam_threshold_margin_holds_on_the_real_generator():
    for seed in range(150):
        q = make(TOPIC, seed=seed, difficulty="exam")
        h, r, x = _parse_exam(q.text)
        i = 1
        while True:
            height = h * r**i
            assert abs(x - height) / height > Fraction(1, 20), (seed, q.text, i, height, x)
            if height < x or i > 30:
                break
            i += 1
