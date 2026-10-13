"""Task 2143 (P19H build 4/4), reworked by task 2145 (send-back item 1a):
conventional rate-unit usage classification
(senior-identify-common-usage-of-rates-for-example-km, ACMEM014)."""

from __future__ import annotations

import pytest

from mathgen.grading import GRADERS, mcq_options
from mathgen.skills import load_skills
from mathgen import make
from mathgen.generators.p19h_rate_usage_build import _KEYS, _KEY_TYPE

TOPIC = "p19h_rate_usage_classification"
SKILL_ID = "senior-identify-common-usage-of-rates-for-example-km"
TIERS = ["easy", "medium", "hard", "exam"]
SEEDS = range(40)

# Forbidden-token list per key: the key string itself, its spelled-out long
# form, and the literal plural noun that would echo the unit name for the two
# rephrased non-speed keys (task 2145 required every clause for those two
# keys to say "pulse"/"heart rate" or "breathing rate"/"respiratory rate",
# never "beats" or "breaths" -- so "beats"/"breaths" as literal substrings
# are the forbidden tokens, not the broader stems "beat"/"breath", which
# would also flag the legitimate wording "breathing rate").
_FORBIDDEN_TOKENS = {
    "km/h": ["km/h", "kilometres per hour", "kilometers per hour"],
    "m/s": ["m/s", "metres per second", "meters per second"],
    "beats/minute": ["beats/minute", "beats per minute", "beats"],
    "breaths/minute": ["breaths/minute", "breaths per minute", "breaths"],
    "L/minute": ["l/minute", "litres per minute", "liters per minute"],
}


@pytest.fixture(scope="module")
def skills_dict():
    _, skills = load_skills()
    return {s.id: s for s in skills}


def test_skill_registered_with_all_four_tiers(skills_dict):
    skill = skills_dict[SKILL_ID]
    difficulties = {src.difficulty for src in skill.sources}
    assert difficulties == set(TIERS)
    for src in skill.sources:
        assert src.topic == TOPIC


# --- Determinism -------------------------------------------------------------


@pytest.mark.parametrize("difficulty", TIERS)
def test_deterministic_across_seeds(difficulty):
    for seed in SEEDS:
        q1 = make(TOPIC, seed=seed, difficulty=difficulty, tier=difficulty)
        q2 = make(TOPIC, seed=seed, difficulty=difficulty, tier=difficulty)
        assert q1.to_json() == q2.to_json()


# --- Distractor hygiene: never binary, distinct, never == answer -----------


@pytest.mark.parametrize("difficulty", TIERS)
def test_distractors_never_binary_and_distinct(difficulty):
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=difficulty, tier=difficulty)
        # Task 2153: >= 2 distractors (>= 3 options); the old ">= 3 distractors"
        # forced the fixed five options with dead padding (review 2149 item 3).
        assert len(q.distractors) >= 2, (difficulty, seed, q.distractors)
        assert q.answer_text not in q.distractors, (difficulty, seed, q.text)
        assert len(set(q.distractors)) == len(q.distractors), (difficulty, seed, q.distractors)
        assert q.answer_text in _KEYS


# --- Real grader accept/reject (mcq: pass an INDEX into mcq_options) -------


@pytest.mark.parametrize("difficulty", TIERS)
def test_real_grader_accepts_canonical_and_rejects_every_distractor(difficulty):
    grade_mcq = GRADERS["mcq"]
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=difficulty, tier=difficulty)
        options = mcq_options(q)
        correct_index = options.index(q.answer_text)
        assert grade_mcq(q, str(correct_index)).correct, (difficulty, seed, q.text)
        for i, opt in enumerate(options):
            if opt != q.answer_text:
                assert not grade_mcq(q, str(i)).correct, (difficulty, seed, q.text, opt)


@pytest.mark.parametrize("difficulty", TIERS)
def test_real_grader_rejects_out_of_range_index(difficulty):
    grade_mcq = GRADERS["mcq"]
    for seed in SEEDS[:10]:
        q = make(TOPIC, seed=seed, difficulty=difficulty, tier=difficulty)
        assert not grade_mcq(q, "99").correct, (difficulty, seed, q.text)


def test_raises_value_error_on_unknown_tier():
    with pytest.raises(ValueError):
        make(TOPIC, seed=1, tier="bogus")


# --- 200-seed answer-concentration + speed/non-speed balance ---------------


@pytest.mark.parametrize("difficulty", TIERS)
def test_key_concentration_under_30_percent_at_200_seeds(difficulty):
    """Key is drawn uniformly within its type (2 speed keys, 3 non-speed
    keys), so no single key's design target exceeds 50% (speed keys) or
    ~33% (non-speed keys). At a fixed 200-seed sample a fair-coin
    fluctuation can push the observed top share a few points above the
    nominal target (pv0b's "fair-coin slack" precedent, task 2142's result
    file) -- allow a generous band rather than the raw design target."""
    counts = {k: 0 for k in _KEYS}
    for seed in range(200):
        q = make(TOPIC, seed=seed, difficulty=difficulty, tier=difficulty)
        counts[q.answer_text] += 1
    top_share = max(counts.values()) / 200
    assert top_share < 0.30, (difficulty, counts)
    assert all(c > 0 for c in counts.values()), (difficulty, counts)


@pytest.mark.parametrize("difficulty", TIERS)
def test_speed_and_non_speed_both_represented_and_balanced_at_200_seeds(difficulty):
    """Construct-to-target: type is drawn uniformly {speed, non-speed}
    first, so both types should show up close to 50/50 over 200 seeds."""
    type_counts = {"speed": 0, "non-speed": 0}
    for seed in range(200):
        q = make(TOPIC, seed=seed, difficulty=difficulty, tier=difficulty)
        type_counts[_KEY_TYPE[q.answer_text]] += 1
    assert type_counts["speed"] > 0 and type_counts["non-speed"] > 0, (difficulty, type_counts)
    speed_share = type_counts["speed"] / 200
    assert 0.30 < speed_share < 0.70, (difficulty, type_counts)


# --- Key POSITION balance at 400 seeds --------------------------------------


@pytest.mark.parametrize("difficulty", TIERS)
def test_key_position_balance_at_400_seeds(difficulty):
    """Task 2153: option counts vary (3-5), so positions 0-2 exist in every
    question; each must hold a fair share and none may dominate."""
    positions = {}
    for seed in range(400):
        q = make(TOPIC, seed=seed, difficulty=difficulty, tier=difficulty)
        options = mcq_options(q)
        idx = options.index(q.answer_text)
        positions[idx] = positions.get(idx, 0) + 1
    assert len(positions) >= 3, (difficulty, positions)
    for pos in (0, 1, 2):
        assert positions.get(pos, 0) >= 400 * 0.15, (difficulty, positions)
    assert max(positions.values()) <= 400 * 0.45, (difficulty, positions)


# --- answer_latex never renders a bare unit word as math -------------------


@pytest.mark.parametrize("difficulty", TIERS)
def test_answer_latex_is_plain_for_bare_unit_words(difficulty):
    for seed in SEEDS[:10]:
        q = make(TOPIC, seed=seed, difficulty=difficulty, tier=difficulty)
        assert q.answer_latex.startswith("plain:"), (difficulty, seed, q.answer_latex)


# --- Every draw offers 3 to 5 options (never binary). Task 2153 replaced the
# old "always exactly 5" check: a fixed five carried 2+ dead options per
# question (review 2149 item 3). ---------------------------------------------


@pytest.mark.parametrize("difficulty", TIERS)
def test_every_draw_offers_three_to_five_options(difficulty):
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=difficulty, tier=difficulty)
        options = mcq_options(q)
        assert 3 <= len(options) <= 5, (difficulty, seed, options)


# ===========================================================================
# Task 2145 mechanical checks (send-back item 1a)
# ===========================================================================

# --- 1. Distinct prompts, 60 seeds per tier: pool widened well past the old
# 20/20/10/39. ---


@pytest.mark.parametrize("difficulty", TIERS)
def test_distinct_prompts_at_least_35_at_60_seeds(difficulty):
    texts = set()
    for seed in range(60):
        q = make(TOPIC, seed=seed, difficulty=difficulty, tier=difficulty)
        texts.add(q.text)
    assert len(texts) >= 35, (difficulty, len(texts))


@pytest.mark.parametrize("difficulty", TIERS)
def test_distinct_prompts_at_200_seeds_reported(difficulty, capsys):
    """Not a hard assertion beyond '>= the 60-seed floor' (more seeds can
    only ever find >= as many distinct prompts) -- this exists so the
    200-seed count is measured and printed for direct comparison with the
    the earlier pool's 20/20/10/39."""
    texts = set()
    for seed in range(200):
        q = make(TOPIC, seed=seed, difficulty=difficulty, tier=difficulty)
        texts.add(q.text)
    print(f"\n[task 2145] {difficulty}: distinct@200 = {len(texts)}")
    assert len(texts) >= 35, (difficulty, len(texts))


# --- 2. Key-not-in-prompt: the drawn key's own name/long-form/plural-noun
# must never leak into the prompt text itself. ------------------------------


@pytest.mark.parametrize("difficulty", TIERS)
def test_key_never_echoed_in_its_own_prompt_at_200_draws(difficulty):
    violations = []
    for seed in range(200):
        q = make(TOPIC, seed=seed, difficulty=difficulty, tier=difficulty)
        low = q.text.lower()
        for token in _FORBIDDEN_TOKENS[q.answer_text]:
            if token.lower() in low:
                violations.append((seed, q.answer_text, token, q.text))
    assert not violations, violations


# --- 3. Every offered option is on the key's plausible-unit list. Task 2153
# replaced the old "every option is one of the 5 keys" check, which forced the
# dead cross-type distractors (see test_2153_every_distractor_is_on_the_plausible_list). ---


@pytest.mark.parametrize("difficulty", TIERS)
def test_every_answer_is_a_real_key_at_200_draws(difficulty):
    for seed in range(200):
        q = make(TOPIC, seed=seed, difficulty=difficulty, tier=difficulty)
        assert q.answer_text in _KEYS, (difficulty, seed, q.answer_text)


# --- 4. All 5 keys reachable within 200 draws, every tier. ------------------


@pytest.mark.parametrize("difficulty", TIERS)
def test_all_five_keys_reachable_at_200_draws(difficulty):
    seen = set()
    for seed in range(200):
        q = make(TOPIC, seed=seed, difficulty=difficulty, tier=difficulty)
        seen.add(q.answer_text)
    assert seen == set(_KEYS), (difficulty, seen)


# --- 5. Escalation still holds: each tier asks a genuinely different
# reading question, not just a longer version of the previous one. This is
# asserted structurally rather than via a fragile text-shape proxy: easy and
# medium draw from disjoint single-clause phrase pools (never a shared
# phrase, so a "harder" difficulty is a different pool, not padding), hard
# names TWO quantities and picks one by ordinal position (a task easy/medium
# cannot pose because they have only one quantity), and exam names THREE
# quantities including a live km/h-vs-m/s pair and picks one by NAMED SOURCE
# (a task hard cannot pose because it never carries both speed conventions
# in the same item). See the module docstring for the
# full reasoning. -------------------------------------------------------


def test_easy_and_medium_phrase_pools_are_disjoint():
    from mathgen.generators.p19h_rate_usage_build import _EASY_POOL, _MEDIUM_POOL

    for key in _KEYS:
        easy_phrases = set(_EASY_POOL[key])
        medium_phrases = set(_MEDIUM_POOL[key])
        assert not (easy_phrases & medium_phrases), (key, easy_phrases & medium_phrases)


def test_hard_prompts_name_two_quantities_and_exam_prompts_name_three():
    for seed in range(20):
        hard_q = make(TOPIC, seed=seed, difficulty="hard", tier="hard")
        assert "; and separately" not in hard_q.text  # sanity: not exam-shaped
        assert " and separately " in hard_q.text, (seed, hard_q.text)
        exam_q = make(TOPIC, seed=seed, difficulty="exam", tier="exam")
        assert exam_q.text.count("; ") >= 2, (seed, exam_q.text)


def test_hard_and_exam_pools_are_scenario_based():
    """Task 2153: hard / exam draw from >= 7 scenarios, each carrying both
    speed keys and >= 3 clause variants per quantity."""
    from mathgen.generators.p19h_rate_usage_build import _SCENARIOS

    assert len(_SCENARIOS) >= 7, len(_SCENARIOS)
    for sc in _SCENARIOS:
        assert {"km/h", "m/s"} <= set(sc["variants"]), sc["intro"]
        for key, variants in sc["variants"].items():
            assert len(variants) >= 3, (sc["intro"], key)


# ===========================================================================
# Task 2153 (fixes for the blind review of 2149, items 2, 3, 11)
# ===========================================================================

# Test-owned, per-quantity list of the units a person could plausibly choose
# for that quantity (scale / convention alternatives). Independent of the
# generator's own lists.
_PLAUSIBLE = {
    "km/h": {"m/s", "mph"},
    "m/s": {"km/h", "mph", "cm/s"},
    "beats/minute": {"breaths/minute", "beats/second"},
    "breaths/minute": {"beats/minute", "breaths/second"},
    "L/minute": {"L/hour", "mL/minute", "L/second"},
}

# Words removed by task 2153 because their conventional unit is not km/h or
# m/s in real usage (cadence is rpm; ferries and aeroplanes report knots).
_DROPPED_CONTEXT_WORDS = ("leg-turnover", "ferry", "aeroplane", "flight computer", "flight-management")


@pytest.mark.parametrize("difficulty", TIERS)
def test_2153_every_distractor_is_on_the_plausible_list_at_300_seeds(difficulty):
    for seed in range(300):
        q = make(TOPIC, seed=seed, difficulty=difficulty, tier=difficulty)
        for d in q.distractors:
            assert d in _PLAUSIBLE[q.answer_text], (difficulty, seed, q.answer_text, d, q.text)


@pytest.mark.parametrize("difficulty", TIERS)
def test_2153_option_set_varies_and_stays_3_to_5(difficulty):
    sets = set()
    sizes = set()
    for seed in range(300):
        q = make(TOPIC, seed=seed, difficulty=difficulty, tier=difficulty)
        n = 1 + len(q.distractors)
        assert 3 <= n <= 5, (difficulty, seed, n)
        sizes.add(n)
        sets.add((q.answer_text, tuple(sorted(q.distractors))))
    assert len(sizes) >= 2, (difficulty, sizes)
    # 2155: padding units removed, so at most ~11 genuine (key, option-set) pairs exist.
    assert len(sets) >= 8, (difficulty, len(sets))


@pytest.mark.parametrize("difficulty", TIERS)
def test_2153_speed_questions_keep_the_other_speed_unit_as_a_distractor(difficulty):
    for seed in range(300):
        q = make(TOPIC, seed=seed, difficulty=difficulty, tier=difficulty)
        if q.answer_text == "km/h":
            assert "m/s" in q.distractors, (difficulty, seed)
        if q.answer_text == "m/s":
            assert "km/h" in q.distractors, (difficulty, seed)


@pytest.mark.parametrize("difficulty", TIERS)
def test_2153_no_dropped_context_words_at_300_seeds(difficulty):
    for seed in range(300):
        q = make(TOPIC, seed=seed, difficulty=difficulty, tier=difficulty)
        low = q.text.lower()
        for w in _DROPPED_CONTEXT_WORDS:
            assert w not in low, (difficulty, seed, w, q.text)


@pytest.mark.parametrize("difficulty", ["easy", "medium"])
def test_2153_distinct_prompts_at_least_150_at_200_seeds(difficulty):
    texts = {make(TOPIC, seed=s, difficulty=difficulty, tier=difficulty).text for s in range(200)}
    assert len(texts) >= 150, (difficulty, len(texts))


@pytest.mark.parametrize("difficulty", ["hard", "exam"])
def test_2153_distinct_prompts_at_least_150_at_200_seeds_upper_tiers(difficulty):
    texts = {make(TOPIC, seed=s, difficulty=difficulty, tier=difficulty).text for s in range(200)}
    assert len(texts) >= 150, (difficulty, len(texts))


_CUES_SINGLE = (
    "sprinter",
    "cyclist",
    "car on a highway",
    "train",
    "cricket ball",
    "tennis serve",
    "gust of wind",
    "skydiver",
    "skier",
    "motorbike",
    "racehorse",
    "cheetah",
    "runner on a treadmill",
    "radar gun",
)
_CUES_MULTI = ("sprinter", "cyclist", "wind", "racehorse")


@pytest.mark.parametrize("difficulty", TIERS)
def test_2153_no_cue_phrase_fixes_the_speed_key_at_800_seeds(difficulty):
    """Each subject / instrument cue must be seen with BOTH speed keys: a
    reporting context, not a cue word, settles km/h vs m/s."""
    cues = _CUES_SINGLE if difficulty in ("easy", "medium") else _CUES_MULTI
    seen = {c: set() for c in cues}
    for seed in range(800):
        q = make(TOPIC, seed=seed, difficulty=difficulty, tier=difficulty)
        low = q.text.lower()
        for c in cues:
            if c in low:
                seen[c].add(q.answer_text)
    for c in cues:
        assert {"km/h", "m/s"} <= seen[c], (difficulty, c, seen[c])


# Task 2155 (round-2 blind review): padding units nobody would choose.
_PADDING_UNITS = {"m/minute", "beats/hour", "breaths/hour", "kL/hour"}


@pytest.mark.parametrize("difficulty", TIERS)
def test_2155_no_padding_unit_is_ever_offered_at_300_seeds(difficulty):
    for seed in range(300):
        q = make(TOPIC, seed=seed, difficulty=difficulty, tier=difficulty)
        bad = set(q.distractors) & _PADDING_UNITS
        assert not bad, (difficulty, seed, bad, q.text)
