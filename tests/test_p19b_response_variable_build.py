"""Covers determinism, the real grader, the fixed four-option shape, the distractor class
draw, the three "consensus attacks" (overlap, explanatory column, response column), the
other option/stimulus tells, the scenario pool and the per-tier reasoning shapes. Every
measured property is computed by a function in this file that a reader can call.
"""

from __future__ import annotations

import re

import pytest

from mathgen.grading.typed import grade_mcq, mcq_options
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError
from mathgen.generators import p19b_dependent_variable_build as sibling
from mathgen.generators import p19b_response_variable_build as gen
from mathgen.generators.p19b_response_variable_build import (
    MANIPULABLE_SCENARIOS,
    QUESTIONS,
    SCENARIOS,
    TIER_POOLS,
    _verify,
    option_text,
)

TOPIC = "p19b_response_variable"
SKILL_ID = "senior-identify-the-response-variable-and-the-explanatory-variable"
TIERS = ("easy", "medium", "hard", "exam")
SEEDS = range(60)
N_ATTACK = 200
TOP_LIMIT = 0.40

_OPT_RE = re.compile(r"^Explanatory variable: (.+); response variable: (.+)$")


def _q(tier, seed):
    return make(TOPIC, seed=seed, difficulty=tier, tier=tier)


def _question_of(q):
    matches = [t for t, sentence in QUESTIONS.items() if q.text.endswith(" " + sentence)]
    assert matches
    return QUESTIONS[matches[0]]


def _stimulus(q):
    return q.text[: -len(" " + _question_of(q))]


def _scenario_of(q):
    """The scenario a question was drawn from, recovered from the key (its E and R)."""
    e, r = _OPT_RE.match(q.answer_text).groups()
    for sc in SCENARIOS:
        if sc["e"] == e and sc["r"] == r:
            return sc
    raise AssertionError(q.answer_text)


def _pairs(q):
    """[(E, R)] of the options, key first, then the distractors."""
    return [_OPT_RE.match(o).groups() for o in [q.answer_text, *q.distractors]]


def _class_of(pair, sc):
    e, r, z = sc["e"], sc["r"], sc["z"]
    table = {(e, r): "K", (r, e): "S", (e, z): "A", (z, r): "B", (z, e): "C", (r, z): "D"}
    return table[pair]


def _top_credit(scores):
    """Fractional credit the key (index 0) gets when the top scorer is named: ties share it."""
    best = max(scores)
    tops = [i for i, s in enumerate(scores) if s == best]
    return (1 / len(tops)) if 0 in tops else 0.0


def consensus_shares(tier, n=N_ATTACK):
    """(overlap, explanatory-column, response-column) share of draws on which the top-scoring
    option is the key, ties counted fractionally."""
    totals = [0.0, 0.0, 0.0]
    for seed in range(n):
        pairs = _pairs(_q(tier, seed))
        sets = [set(p) for p in pairs]
        overlap = [sum(len(sets[i] & sets[j]) for j in range(4) if j != i) for i in range(4)]
        e_col = [sum(1 for j in range(4) if j != i and pairs[j][0] == pairs[i][0]) for i in range(4)]
        r_col = [sum(1 for j in range(4) if j != i and pairs[j][1] == pairs[i][1]) for i in range(4)]
        for k, scores in enumerate((overlap, e_col, r_col)):
            totals[k] += _top_credit(scores)
    return tuple(t / n for t in totals)


def key_longest_share(tier, n=N_ATTACK):
    total = 0.0
    for seed in range(n):
        q = _q(tier, seed)
        lengths = [len(o) for o in [q.answer_text, *q.distractors]]
        total += _top_credit(lengths)
    return total / n


def first_named_echo_share(tier, n=N_ATTACK):
    """Share of draws on which the key is the ONLY option containing the stimulus's first-named
    variable."""
    hits = 0
    for seed in range(n):
        q = _q(tier, seed)
        sc = _scenario_of(q)
        stim = _stimulus(q)
        first = min((sc["e"], sc["r"], sc["z"]), key=stim.find)
        holders = [o for o in [q.answer_text, *q.distractors] if first in o]
        if holders == [q.answer_text]:
            hits += 1
    return hits / n


def key_in_stimulus_order_share(tier, n=N_ATTACK):
    """Share of draws on which the key's order (E then R) is the order the stimulus first
    names them in."""
    hits = 0
    for seed in range(n):
        q = _q(tier, seed)
        sc = _scenario_of(q)
        stim = _stimulus(q)
        if stim.find(sc["e"]) < stim.find(sc["r"]):
            hits += 1
    return hits / n


def position_shares(tier, n=400):
    counts = [0, 0, 0, 0]
    for seed in range(n):
        q = _q(tier, seed)
        counts[mcq_options(q).index(q.answer_text)] += 1
    return [c / n for c in counts]


def class_counts(tier, n=N_ATTACK):
    counts = {c: 0 for c in "SABCD"}
    s_draws = 0
    for seed in range(n):
        q = _q(tier, seed)
        sc = _scenario_of(q)
        classes = [_class_of(p, sc) for p in _pairs(q)[1:]]
        assert "K" not in classes
        for c in classes:
            counts[c] += 1
        s_draws += "S" in classes
    return counts, s_draws / n


# ---------------------------------------------------------------- basic contract


@pytest.mark.parametrize("tier", TIERS)
def test_deterministic_four_options_and_distinct(tier):
    for seed in SEEDS:
        q = _q(tier, seed)
        assert q.to_json() == _q(tier, seed).to_json()
        assert len(q.distractors) == 3
        options = [q.answer_text, *q.distractors]
        assert len(set(options)) == 4
        assert all(_OPT_RE.match(o) for o in options)
        assert q.answer_text.startswith("Explanatory variable: ")


@pytest.mark.parametrize("tier", TIERS)
def test_real_grader_accepts_key_and_rejects_every_distractor(tier):
    for seed in SEEDS:
        q = _q(tier, seed)
        options = mcq_options(q)
        assert grade_mcq(q, str(options.index(q.answer_text))).correct
        for d in q.distractors:
            assert not grade_mcq(q, str(options.index(d))).correct


def test_default_params_have_enough_distractors():
    for seed in range(100):
        q = make(TOPIC, seed=seed)
        assert len(q.distractors) >= 2


@pytest.mark.parametrize("tier", TIERS)
def test_giveaway_words_only_in_question_and_options(tier):
    for seed in range(N_ATTACK):
        q = _q(tier, seed)
        stim = _stimulus(q).lower()
        for root in ("explanat", "respon", "depend", "independent"):
            assert root not in stim, (tier, seed, root)
        assert q.text.endswith(QUESTIONS[tier])
        assert q.text.count("explanatory") == 1 and q.text.count("response") == 1
        assert "study" in stim


@pytest.mark.parametrize("tier", TIERS)
def test_sentence_case(tier):
    for seed in range(N_ATTACK):
        q = _q(tier, seed)
        for sentence in (s.strip() for s in re.split(r"(?<=[.?])\s+", q.text) if s.strip()):
            assert sentence[0].isupper(), (tier, seed, sentence)


def test_in_generator_check_fires_on_corrupted_value():
    q = _q("easy", 0)
    sc = _scenario_of(q)
    key, dis = q.answer_text, list(q.distractors)
    _verify(q.text, sc, key, dis, "easy")
    with pytest.raises(VerificationError):
        _verify(q.text.replace(sc["e"], "x"), sc, key, dis, "easy")
    with pytest.raises(VerificationError):
        _verify(q.text, sc, key, [dis[0], dis[1], key], "easy")
    with pytest.raises(VerificationError):
        _verify(q.text, sc, dis[0], [key, dis[1], dis[2]], "easy")
    with pytest.raises(VerificationError):
        _verify(q.text.replace("The ", "The response ", 1), sc, key, dis, "easy")
    with pytest.raises(VerificationError):
        _verify(q.text.replace("The researcher", "Z plays no part; the researcher", 1), sc, key, dis, "easy")
    with pytest.raises(VerificationError):
        _verify(q.text, sc, key, dis, "hard")


# ---------------------------------------------------------------- scenario pool


def test_scenario_pool():
    assert len(SCENARIOS) >= 40
    assert sum(1 for sc in SCENARIOS if sc["manipulable"]) >= 14
    assert set(map(id, MANIPULABLE_SCENARIOS)) <= set(map(id, SCENARIOS))
    assert all(sc["manipulable"] for sc in MANIPULABLE_SCENARIOS)
    sibling_names = set()
    for p in sibling.PAIRS:
        sibling_names.update((p["iv"], p["dv"], p["controlled"], p["other"], p["exam_trap"]))
    seen = set()
    for sc in SCENARIOS:
        names = (sc["e"], sc["r"], sc["z"])
        assert len(set(names)) == 3
        for name in names:
            assert 2 <= len(name.split()) <= 5, name
            assert name not in sibling_names, name
        for a in names:
            for b in names:
                assert a == b or a not in b, (a, b)
        assert len({len(n) for n in names}) == 3, names
        assert (sc["e"], sc["r"]) not in seen
        seen.add((sc["e"], sc["r"]))
    sib_pairs = {(p["iv"], p["dv"]) for p in sibling.PAIRS}
    assert not (seen & sib_pairs)
    # Z's length relative to E and R varies: longer than both, shorter than both, and between
    kinds = {"long": 0, "short": 0, "mid": 0}
    for sc in SCENARIOS:
        e_len, r_len, z_len = len(sc["e"]), len(sc["r"]), len(sc["z"])
        kinds["long" if z_len > max(e_len, r_len) else "short" if z_len < min(e_len, r_len) else "mid"] += 1
    assert all(v >= 10 for v in kinds.values()), kinds


def test_tier_pools_are_wide_and_medium_hard_differ():
    for tier, pool in TIER_POOLS.items():
        assert len(pool) >= 24, tier
    assert all(sc["manipulable"] for sc in TIER_POOLS["easy"])
    assert {id(x) for x in TIER_POOLS["medium"]} != {id(x) for x in TIER_POOLS["hard"]}


@pytest.mark.parametrize("tier", TIERS)
def test_distinct_triples_per_tier_at_200_seeds(tier):
    triples = set()
    for seed in range(N_ATTACK):
        sc = _scenario_of(_q(tier, seed))
        triples.add((sc["e"], sc["r"], sc["z"]))
    assert len(triples) >= 24, (tier, len(triples))


def key_length_rank_shares(tier, n=N_ATTACK):
    """Share of draws on which the key holds each length rank (1 = longest); ties split fractionally."""
    shares = [0.0] * 4
    for seed in range(n):
        q = _q(tier, seed)
        lengths = [len(o) for o in [q.answer_text, *q.distractors]]
        longer = sum(x > lengths[0] for x in lengths)
        tied = sum(x == lengths[0] for x in lengths)
        for i in range(tied):
            shares[longer + i] += 1 / tied
    return [x / n for x in shares]


@pytest.mark.parametrize("tier", TIERS)
def test_key_length_rank_shares_at_most_40_percent(tier):
    shares = key_length_rank_shares(tier)
    assert max(shares) <= TOP_LIMIT, (tier, shares)


# Phrases that would label the third variable irrelevant / excluded; none may appear in a stimulus.
Z_PHRASES = (
    "plays no part",
    "only a",
    "excludes",
    "covariate",
    "background",
    "grouping label",
    "not part",
    "does not involve",
    "leaves out",
    "only groups",
    "serves only",
    "excluded",
    "irrelevant",
)


@pytest.mark.parametrize("tier", TIERS)
def test_no_stimulus_labels_z_irrelevant(tier):
    for seed in range(N_ATTACK):
        stim = _stimulus(_q(tier, seed)).lower()
        for phrase_ in Z_PHRASES:
            assert phrase_ not in stim, (tier, seed, phrase_)


@pytest.mark.parametrize("tier", TIERS)
def test_z_is_recorded_neutrally_and_never_shares_a_sentence_with_a_direction_cue(tier):
    """Z appears in the recorded list; on easy/medium sometimes in one extra neutral note; on hard as the split
    of the second analysis; on exam as the colour key. Outside the recorded list it never shares a sentence with
    E (nor, except in the hard split analysis, with R)."""
    for seed in range(N_ATTACK):
        q = _q(tier, seed)
        sc = _scenario_of(q)
        sentences = re.split(r"(?<=\.)\s+", _stimulus(q))
        holders = [s for s in sentences if sc["z"] in s]
        lists = [s for s in holders if sc["e"] in s and sc["r"] in s]
        assert len(lists) == 1, (tier, seed, holders)
        extra = [s for s in holders if s is not lists[0]]
        if tier in ("hard", "exam"):
            assert len(extra) == 1, (tier, seed, holders)
        else:
            assert len(extra) <= 1, (tier, seed, holders)
        for s in extra:
            assert sc["e"] not in s, (tier, seed, s)
            if tier != "hard":
                assert sc["r"] not in s, (tier, seed, s)


def _occurrence_order(q, key):
    sc = _scenario_of(q)
    stim = _stimulus(q)
    return sorted((sc["e"], sc["r"], sc["z"]), key=lambda n: key(stim, n)), sc


def named_attack_share(tier, which, n=N_ATTACK):
    """A guesser's attack: parse where the three names occur in the stimulus, take the first-mentioned
    variable (earliest first occurrence) or the last-mentioned one (latest last occurrence, and, separately,
    latest first occurrence) and score each option by whether it uses that variable; return the share of draws on
    which the key is the top scorer (ties fractional)."""
    total = 0.0
    for seed in range(n):
        q = _q(tier, seed)
        if which == "first":
            order, _ = _occurrence_order(q, lambda s, n_: s.find(n_))
            target = order[0]
        elif which == "last":
            order, _ = _occurrence_order(q, lambda s, n_: s.rfind(n_))
            target = order[-1]
        else:  # "last-first-occurrence"
            order, _ = _occurrence_order(q, lambda s, n_: s.find(n_))
            target = order[-1]
        scores = [1 if target in o else 0 for o in [q.answer_text, *q.distractors]]
        total += _top_credit(scores)
    return total / n


def z_last_mentioned_share(tier, n=N_ATTACK):
    hits = 0
    for seed in range(n):
        q = _q(tier, seed)
        order, sc = _occurrence_order(q, lambda s, n_: s.rfind(n_))
        hits += order[-1] == sc["z"]
    return hits / n


@pytest.mark.parametrize("tier", TIERS)
@pytest.mark.parametrize("which", ("first", "last", "last-first-occurrence"))
def test_first_and_last_mentioned_attacks_at_most_35_percent(tier, which):
    share = named_attack_share(tier, which)
    assert share <= 0.35, (tier, which, share)


@pytest.mark.parametrize("tier", TIERS)
def test_z_is_last_mentioned_on_a_balanced_share_of_draws(tier):
    share = z_last_mentioned_share(tier)
    assert 0.25 <= share <= 0.45, (tier, share)


def test_exam_colour_sentence_before_axes_on_about_half_the_draws():
    before = 0
    for seed in range(N_ATTACK):
        q = _q("exam", seed)
        stim = _stimulus(q)
        sc = _scenario_of(q)
        colour_at = min(stim.find(t) for t in ("coloured", "labelled") if t in stim)
        axes_at = stim.find("axis")
        before += colour_at < axes_at
    assert 0.40 <= before / N_ATTACK <= 0.60, before / N_ATTACK


@pytest.mark.parametrize("tier", TIERS)
def test_recorded_list_order_is_a_uniform_permutation(tier):
    counts = {}
    for seed in range(N_ATTACK * 3):
        q = _q(tier, seed)
        sc = _scenario_of(q)
        stim = _stimulus(q)
        first = min((sc["e"], sc["r"], sc["z"]), key=stim.find)
        lst = next(s for s in re.split(r"(?<=\.)\s+", stim) if sc["e"] in s and sc["r"] in s and sc["z"] in s)
        order = tuple(
            "ERZ"[[sc["e"], sc["r"], sc["z"]].index(n)] for n in sorted((sc["e"], sc["r"], sc["z"]), key=lst.find)
        )
        counts[order] = counts.get(order, 0) + 1
    assert len(counts) == 6, counts
    assert all(0.10 <= c / (N_ATTACK * 3) <= 0.24 for c in counts.values()), counts


# ---------------------------------------------------------------- class draw and attacks


@pytest.mark.parametrize("tier", TIERS)
def test_every_distractor_class_drawn_and_s_share(tier):
    counts, s_share = class_counts(tier)
    assert all(v > 0 for v in counts.values()), counts
    assert 0.40 <= s_share <= 0.70, (tier, s_share)


@pytest.mark.parametrize("tier", TIERS)
def test_consensus_attacks_pick_the_key_on_at_most_40_percent(tier):
    overlap, e_col, r_col = consensus_shares(tier)
    assert overlap <= TOP_LIMIT, ("overlap", tier, overlap)
    assert e_col <= TOP_LIMIT, ("explanatory column", tier, e_col)
    assert r_col <= TOP_LIMIT, ("response column", tier, r_col)


@pytest.mark.parametrize("tier", TIERS)
def test_other_option_tells(tier):
    assert key_longest_share(tier) <= TOP_LIMIT
    assert first_named_echo_share(tier) <= TOP_LIMIT
    order = key_in_stimulus_order_share(tier)
    assert 0.40 <= order <= 0.60, (tier, order)


@pytest.mark.parametrize("tier", TIERS)
def test_key_position_balanced_at_400_seeds(tier):
    for share in position_shares(tier):
        assert 0.15 <= share <= 0.35, (tier, position_shares(tier))


# ---------------------------------------------------------------- variety and per-tier shape


def _shape(q):
    sc = _scenario_of(q)
    s = _stimulus(q)
    for tag, name in (("<E>", sc["e"]), ("<R>", sc["r"]), ("<Z>", sc["z"]), ("<U>", sc["unit"])):
        s = s.replace(name, tag)
    return s.replace(sc["unit"] + "s", "<U>s")


@pytest.mark.parametrize("tier", TIERS)
def test_at_least_three_shapes_and_twenty_prompts_over_60_seeds(tier):
    qs = [_q(tier, s) for s in SEEDS]
    assert len({q.text for q in qs}) >= 20
    assert len({_shape(q) for q in qs}) >= 3


_MANIPULATION = re.compile(r"\b(sets?|choos\w+|varies|varying|deliberately|setting|chosen)\b")
_INFLUENCE = re.compile(r"\b(influenc\w*|affect\w*|effect|causes?|leads?|lead)\b")


@pytest.mark.parametrize("tier", TIERS)
def test_tier_cues(tier):
    for seed in range(N_ATTACK):
        q = _q(tier, seed)
        stim = _stimulus(q)
        sc = _scenario_of(q)
        assert any(sc is x for x in TIER_POOLS[tier])
        body = stim
        if tier == "easy":
            assert sc["manipulable"] and _MANIPULATION.search(body)
            assert not _INFLUENCE.search(stim) and "observational" not in stim
        if tier == "medium":
            assert "observational" in body or "without changing anything" in body
            assert _INFLUENCE.search(body) or "changes according to" in body
            assert not re.search(r"\b(sets? the|chooses|predict\w*|estimate|line of best fit)\b", stim)
        if tier == "hard":
            assert "line of best fit" in stim and "mean" in stim and "subgroup" in stim
            assert "line of best fit" in _question_of(q)
            assert not _INFLUENCE.search(stim)
        if tier == "exam":
            assert "horizontal axis" in stim and "vertical axis" in stim
            assert re.search(r"coloured|labelled", stim) and "least-squares line" in stim
            assert "predictions" in stim and not re.search(r"predict(?!ions)|estimate", stim)
            assert not _INFLUENCE.search(stim)
            # the convention: the key's explanatory variable is the one on the horizontal axis
            m = re.search(
                re.escape(sc["e"])
                + r"[^.]*?horizontal|horizontal axis (?:and|shows|along)? ?[^.]*?"
                + re.escape(sc["e"]),
                stim,
            )
            assert m


def test_medium_influence_verb_in_most_but_not_all_templates():
    with_verb = 0
    for seed in range(N_ATTACK):
        with_verb += bool(_INFLUENCE.search(_stimulus(_q("medium", seed))))
    assert 0.70 <= with_verb / N_ATTACK < 1.0, with_verb / N_ATTACK


@pytest.mark.parametrize("tier", TIERS)
def test_response_named_before_explanatory_share(tier):
    n = 200
    r_first = 0
    for seed in range(n):
        q = _q(tier, seed)
        sc = _scenario_of(q)
        stim = _stimulus(q)
        r_first += stim.find(sc["r"]) < stim.find(sc["e"])
    assert 0.40 <= r_first / n <= 0.60, (tier, r_first / n)


@pytest.mark.parametrize("tier", TIERS)
def test_distractor_distribution_three_on_every_draw(tier):
    counts = {}
    for seed in range(N_ATTACK):
        counts[len(_q(tier, seed).distractors)] = counts.get(len(_q(tier, seed).distractors), 0) + 1
    assert counts == {3: N_ATTACK}


def test_skill_loads_with_expected_metadata():
    _domains, skills = load_skills()
    skill = {s.id: s for s in skills}.get(SKILL_ID)
    assert skill is not None
    assert skill.answer == "mcq"
    assert skill.prerequisites == []
    assert skill.placement_eligible is False
    assert {src.difficulty for src in skill.sources} == set(TIERS)
    assert {src.topic for src in skill.sources} == {TOPIC}
    exam = [src for src in skill.sources if src.difficulty == "exam"]
    assert len(exam) == 1 and exam[0].stages == ["exam"]
    assert option_text("a", "b") == "Explanatory variable: a; response variable: b"
    assert gen.S_SHARE > 0
