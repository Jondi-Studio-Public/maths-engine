"""P19g_particles_meet_mcq (ACMSM106).

Covers determinism, distractor hygiene, the real mcq grader, answer-space
construction (25-35% yes share, key-position balance, no-flavour
distribution), and skill loading. The independent oracle lives separately in
``tests/test_p19g_two_particles_meet_oracle.py`` and never imports this
module.
"""

from __future__ import annotations

import collections

import pytest

from fractions import Fraction

from mathgen.grading.typed import GRADERS, mcq_options
from mathgen import make
from mathgen.generators import p19g_particles_meet_build as _pm_mod
from mathgen.generators.p19g_particles_meet_build import _base_options

TOPIC = "p19g_particles_meet_mcq"
TIERS = ("easy", "medium", "hard", "exam")


# --- determinism -------------------------------------------------------------


@pytest.mark.parametrize("tier", TIERS)
def test_deterministic(tier):
    for seed in (0, 1, 7, 41):
        q1 = make(TOPIC, seed=seed, difficulty=tier, tier=tier)
        q2 = make(TOPIC, seed=seed, difficulty=tier, tier=tier)
        assert q1.text == q2.text
        assert q1.answer_text == q2.answer_text
        assert q1.distractors == q2.distractors


# --- distractor hygiene --------------------------------------------------


@pytest.mark.parametrize("tier", TIERS)
def test_distractors_distinct_nonempty_never_equal_answer(tier):
    for seed in range(60):
        q = make(TOPIC, seed=seed, difficulty=tier, tier=tier)
        assert len(q.distractors) == 3
        assert all(d for d in q.distractors)
        assert q.answer_text not in q.distractors
        assert len(set(q.distractors)) == 3


# --- real grader accept/reject -------------------------------------------


@pytest.mark.parametrize("tier", TIERS)
def test_real_grader_accepts_key_rejects_distractors(tier):
    grade_mcq = GRADERS["mcq"]
    for seed in range(60):
        q = make(TOPIC, seed=seed, difficulty=tier, tier=tier)
        options = mcq_options(q)
        correct_idx = options.index(q.answer_text)
        assert grade_mcq(q, str(correct_idx)).correct
        for i in range(len(options)):
            if i != correct_idx:
                assert not grade_mcq(q, str(i)).correct


@pytest.mark.parametrize("tier", TIERS)
def test_deliberately_wrong_index_rejected(tier):
    grade_mcq = GRADERS["mcq"]
    q = make(TOPIC, seed=3, difficulty=tier, tier=tier)
    options = mcq_options(q)
    correct_idx = options.index(q.answer_text)
    wrong_idx = (correct_idx + 1) % 4
    assert not grade_mcq(q, str(wrong_idx)).correct


# --- answer-space construction --------------------------------


@pytest.mark.parametrize("tier", TIERS)
def test_yes_share_between_25_and_35_percent(tier):
    yes = 0
    n = 200
    for seed in range(n):
        q = make(TOPIC, seed=seed, difficulty=tier, tier=tier)
        if q.answer_text.startswith("Yes"):
            yes += 1
    share = yes / n
    assert 0.25 <= share <= 0.35, (tier, share, yes, n)


@pytest.mark.parametrize("tier", TIERS)
def test_key_position_balanced_over_400_seeds(tier):
    counts = collections.Counter()
    n = 400
    for seed in range(n):
        q = make(TOPIC, seed=seed, difficulty=tier, tier=tier)
        options = mcq_options(q)
        counts[options.index(q.answer_text)] += 1
    assert set(counts) == {0, 1, 2, 3}
    for idx, c in counts.items():
        assert 0.15 * n <= c <= 0.35 * n, (tier, idx, c, n)


@pytest.mark.parametrize("tier", TIERS)
def test_no_flavour_distribution_at_least_two_flavours_overall(tier):
    flavours = collections.Counter()
    for seed in range(200):
        q = make(TOPIC, seed=seed, difficulty=tier, tier=tier)
        flavour = q.working[0].split("flavour=")[1].split(" ")[0]
        flavours[flavour] += 1
    # Non-meet ("no") flavours only, for the "at least two structurally
    # different no-flavours across draws" requirement.
    no_flavours = {k: v for k, v in flavours.items() if "meet" not in k or "outside" in k or "in_window" in k}
    assert len(flavours) >= 2, (tier, flavours)


@pytest.mark.parametrize("tier", TIERS)
def test_no_label_is_the_key_more_than_40_percent_of_draws(tier):
    """The answer's own "flavour" label (MEET / CROSS-different-times / NEVER-cross, or MEET-in/-outside-window
    on exam) must not be guessable from the case mix alone. Proved RED against the pre-fix generator (scratch-loaded into an
    isolated registry, never a checkout/stash): easy read cross_different_times=0.71
    (never_intersect never drawn, 0.0); medium read cross_different_times=0.46. Both exceed the 40% ceiling
    this test asserts; hard (0.37 max) and exam (0.285 max) were already clean."""
    flavours = collections.Counter()
    n = 200
    for seed in range(n):
        q = make(TOPIC, seed=seed, difficulty=tier, tier=tier)
        flavour = q.working[0].split("flavour=")[1].split(" ")[0]
        flavours[flavour] += 1
    shares = {k: v / n for k, v in flavours.items()}
    assert max(shares.values()) <= 0.40, (tier, shares)


@pytest.mark.parametrize("tier", ("easy", "medium"))
def test_never_cross_is_a_live_draw_at_easy_and_medium(tier):
    """NEVER-cross (parallel, non-coincident lines) must actually occur at easy and medium, not just hard/exam --
    it was entirely absent at easy (0/200) and under-weighted at medium (21%) before the fix."""
    flavours = collections.Counter()
    for seed in range(200):
        q = make(TOPIC, seed=seed, difficulty=tier, tier=tier)
        flavour = q.working[0].split("flavour=")[1].split(" ")[0]
        flavours[flavour] += 1
    assert flavours["never_intersect"] >= 200 * 0.15, (tier, flavours)


@pytest.mark.parametrize("tier", ("easy", "medium"))
def test_never_cross_carries_no_fraction_tell_at_easy_and_medium(tier):
    """Easy/medium's meet/cross draws use integer-only coefficients (``_nz``), so if a NEVER-cross draw's own
    parallel-scale factor ``k`` were ever a half-integer, "the prompt contains a '/' " would be a
    100%-accurate "pick NEVER" giveaway on every draw where it appears. Asserts no rendered question at these
    two tiers ever contains a '/', for ANY flavour, over 200 seeds -- not just that NEVER-cross itself avoids
    the tell."""
    for seed in range(200):
        q = make(TOPIC, seed=seed, difficulty=tier, tier=tier)
        assert "/" not in q.text, (tier, seed, q.text)


@pytest.mark.parametrize("tier,max_gap", (("easy", 6), ("medium", 8), ("hard", 9)))
def test_yes_at_t0_decoy_is_coherent_not_dead(tier, max_gap):
    """The fixed fourth-slot "Yes --... at t = 0" decoy states that particle A's own t=0 position (pA) equals
    particle B's stated start (pB) -- always false by construction (``pA != pB``), but before the fix pA and pB
    were typically far apart (easy median Manhattan gap 14, max 45, against a coordinate span of only 12; hard
    median 17, max 105.5, against a span of only 18), so the claim was dismissible on sight without doing any
    algebra ("dead" -- offered, and now bounded, at every tier that offers it: easy,
    medium, hard; exam has no fixed decoy). Spies on the module's own independent re-solve step
    (``_base_options``, called with the FINAL pA/vA/pB/vB on every draw) to check the actual gap used, without
    re-implementing any construction logic here. Bound: half each tier's own coordinate span (easy span 12 ->
    6; medium span 16 -> 8; hard span 18 -> 9) -- close enough that the decoy is a genuine near-miss, never
    zero (which would make it accidentally true, already independently guarded by the generator's own
    ``check`` calls)."""
    calls = []
    orig = _pm_mod._base_options

    def spy(pA, vA, pB, vB):
        calls.append((pA, pB))
        return orig(pA, vA, pB, vB)

    _pm_mod._base_options = spy
    try:
        for seed in range(200):
            make(TOPIC, seed=seed, difficulty=tier, tier=tier)
    finally:
        _pm_mod._base_options = orig

    assert len(calls) == 200
    for pA, pB in calls:
        gap = abs(pA[0] - pB[0]) + abs(pA[1] - pB[1])
        assert 0 < gap <= max_gap, (tier, pA, pB, gap)


def test_at_least_two_no_flavours_across_the_whole_topic():
    """The flavour requirement is a topic-wide guarantee, not a per-tier one."""
    flavours = set()
    for tier in TIERS:
        for seed in range(200):
            q = make(TOPIC, seed=seed, difficulty=tier, tier=tier)
            flavours.add(q.working[0].split("flavour=")[1].split(" ")[0])
    assert "cross_different_times" in flavours
    assert "never_intersect" in flavours
    assert len(flavours) >= 2


# --- at least 20 distinct rendered questions per tier over 60 seeds -------


@pytest.mark.parametrize("tier", TIERS)
def test_at_least_20_distinct_questions_over_60_seeds(tier):
    texts = {make(TOPIC, seed=s, difficulty=tier, tier=tier).text for s in range(60)}
    assert len(texts) >= 20, (tier, len(texts))


# --- options formatted alike (no shape tell) -------------------------------


@pytest.mark.parametrize("tier", TIERS)
def test_no_shape_tell_between_answer_and_distractors(tier):
    """Every option is a full sentence of the same rough shape (starts with
    "Yes" or "No", ends with a period) -- no option is identifiable purely by
    a different format."""
    for seed in range(40):
        q = make(TOPIC, seed=seed, difficulty=tier, tier=tier)
        for opt in [q.answer_text, *q.distractors]:
            assert opt.startswith("Yes") or opt.startswith("No")
            assert opt.endswith(".")


# --- the in-generator independent check fires on a corrupted value ---------


def test_independent_check_fires_on_a_corrupted_meet():
    """_base_options is the independent-of-construction re-derivation step
    (re-solves the linear system from the FINAL vectors, never trusting the
    caller's intended verdict). Feeding it a genuinely broken configuration
    (particles start at the same point) must raise, exercised directly."""
    from mathgen.core import VerificationError

    pA = (Fraction(0), Fraction(0))
    vA = (Fraction(1), Fraction(2))
    pB = (Fraction(0), Fraction(0))
    vB = (Fraction(2), Fraction(1))
    with pytest.raises(VerificationError):
        _base_options(pA, vA, pB, vB)


def test_independent_check_fires_on_coincident_parallel_lines():
    from mathgen.core import VerificationError

    pA = (Fraction(0), Fraction(0))
    vA = (Fraction(1), Fraction(2))
    pB = (Fraction(2), Fraction(4))  # on the same line as A (coincident)
    vB = (Fraction(2), Fraction(4))
    with pytest.raises(VerificationError):
        _base_options(pA, vA, pB, vB)


# --- exactly one option true, checked directly by re-deriving the verdict --


@pytest.mark.parametrize("tier", TIERS)
def test_exactly_one_option_true_by_independent_reconstruction(tier):
    """Re-parse each option's own claim back into a (point, times) triple by
    re-running the same option builder against the actual vectors used and
    confirm only the canonical answer's exact string is returned."""
    for seed in range(60):
        q = make(TOPIC, seed=seed, difficulty=tier, tier=tier)
        options = mcq_options(q)
        assert len(set(options)) == 4


# --- skill loads correctly ------------------------------------------------


def test_skill_loads_through_api():
    from mathgen.skills import load_skills

    _domains, skills = load_skills()
    by_id = {s.id: s for s in skills}
    skill = by_id["senior-examine-the-position-of-two-particles-each-described"]
    assert skill.answer == "mcq"
    assert skill.placement_eligible is False
    assert "vector-addition" in skill.prerequisites
    tiers_seen = {s.difficulty for s in skill.sources if s.topic == TOPIC}
    # The exam tier is a multi-part chain on its own topic (exd_x_r2i_particles_approach).
    assert tiers_seen == {"easy", "medium", "hard"}
    exam_sources = [s for s in skill.sources if s.difficulty == "exam"]
    assert exam_sources and exam_sources[0].stages == ["exam"]


# No structural tell on the options ---------------------------

import re as _re

_PT = _re.compile(r"\((-?[\d/]+), (-?[\d/]+)\)")
_YES_T = _re.compile(r"when t = (-?[\d/]+):")


def _frac(sv):
    return Fraction(sv)


def _opt_point(opt):
    m = _PT.search(opt)
    return m.group(0) if m else None


def _rule_wins(q):
    """Expected win (ties broken uniformly at random) of four rules a
    learner could apply without doing the maths."""
    opts = mcq_options(q)
    k = opts.index(q.answer_text)
    pts = [_opt_point(o) for o in opts]
    counts = collections.Counter(p for p in pts if p)

    def win(cands):
        if not cands:
            return 0.25
        return (1 / len(cands)) if k in cands else 0.0

    unique = [i for i, p in enumerate(pts) if p and counts[p] == 1]
    repeated = [i for i, p in enumerate(pts) if p and counts[p] > 1]
    yes_pos_t = [i for i, o in enumerate(opts) if o.startswith("Yes") and _frac(_YES_T.search(o).group(1)) != 0]
    sole_yes = yes_pos_t if len(yes_pos_t) == 1 else []
    pointless = [i for i, p in enumerate(pts) if p is None]
    # The rule: a "Yes ... t != 0" option whose point is not
    # repeated in another option.
    lone_yes = [i for i in unique if opts[i].startswith("Yes") and i in yes_pos_t]
    lone_yes = lone_yes if len(lone_yes) == 1 else []
    return win(unique), win(repeated), win(sole_yes), win(pointless), win(lone_yes)


@pytest.mark.parametrize("tier", TIERS)
def test_no_rule_of_thumb_finds_the_key_over_45_percent(tier):
    """The "Yes... at t != 0" option used to be the key in 100%
    of seeds where it was the only such option (easy 58, medium 66, hard 67
    of 200; exam 52/52), and the t = 0 "Yes" decoy was never the key. Rules
    tested: pick the option whose point is unique among the four; pick a
    repeated point; pick the sole Yes with t > 0; pick the option with no
    point; pick the lone Yes-t>0 whose point is not repeated (the rule). Each must win under 45% (ties random)."""
    n = 200
    tot = [0.0] * 5
    for seed in range(n):
        q = make(TOPIC, seed=seed, difficulty=tier, tier=tier)
        for i, w in enumerate(_rule_wins(q)):
            tot[i] += w
    rates = [t / n for t in tot]
    assert max(rates) < 0.45, (tier, rates)


@pytest.mark.parametrize("tier", TIERS)
def test_no_t0_yes_decoy_and_every_quoted_point_is_checkable(tier):
    """No option claims a meeting at t = 0 at B's start (dead, and false
    evidence), and every point/time quoted in any option is a real position
    of particle A or B at that time, read off the object shown."""
    calls = []
    name = "_window_options" if tier == "exam" else "_base_options"
    orig = getattr(_pm_mod, name)

    def spy(*args):
        out = orig(*args)
        calls.append((args, out))
        return out

    setattr(_pm_mod, name, spy)
    try:
        for seed in range(200):
            make(TOPIC, seed=seed, difficulty=tier, tier=tier)
    finally:
        setattr(_pm_mod, name, orig)
    assert len(calls) == 200
    for args, (answer, distractors, _v, _f) in calls:
        pA, vA, pB, vB = args[:4]
        for opt in [answer, *distractors]:
            assert " when t = 0:" not in opt, opt
            pt = _opt_point(opt)
            if pt is None:
                continue
            m = _YES_T.search(opt)
            if m:
                ts = [_frac(m.group(1))]
            else:
                ts = [_frac(x) for x in _re.findall(r"t = (-?[\d/]+)", opt)]
            ok = False
            for t in ts:
                for p, v in ((pA, vA), (pB, vB)):
                    x, y = p[0] + v[0] * t, p[1] + v[1] * t
                    if pt == f"({_pm_mod.rational_text(x)}, {_pm_mod.rational_text(y)})":
                        ok = True
            assert ok, (tier, opt, args)


# Hard-tier fraction tell; exam "Yes" claimed time -------------


def _is_whole_point(pt):
    return pt is not None and "/" not in pt


def test_hard_whole_number_point_does_not_identify_the_key():
    """At hard, 179/200 prompts had half-integer velocities but the key's meeting point was always whole, so "the
    option whose point is all-integer" found the key. Rule and converse (ties random) must each win under 45%,
    and the key's points must be a 35-65% mix of whole/fractional."""
    n = 200
    win_whole = win_frac = 0.0
    key_pts = []
    for seed in range(n):
        q = make(TOPIC, seed=seed, difficulty="hard", tier="hard")
        opts = mcq_options(q)
        k = opts.index(q.answer_text)
        pts = [_opt_point(o) for o in opts]
        if not any(pts):
            win_whole += 0.25
            win_frac += 0.25
            continue
        whole = [i for i, p in enumerate(pts) if _is_whole_point(p)]
        frac = [i for i, p in enumerate(pts) if p is not None and "/" in p]
        win_whole += (1 / len(whole)) if k in whole else 0.0 if whole else 0.25
        win_frac += (1 / len(frac)) if k in frac else 0.0 if frac else 0.25
        if pts[k] is not None:
            key_pts.append("/" in pts[k])
    assert win_whole / n < 0.45, win_whole / n
    assert win_frac / n < 0.45, win_frac / n
    share = sum(key_pts) / len(key_pts)
    assert 0.35 <= share <= 0.65, (share, len(key_pts))


def test_exam_fake_in_window_yes_claims_varied_times_and_all_false():
    """The fake in-window "Yes" always claimed t = T (148/148). Its claimed t must take >= 3 distinct relative
    values (t/T), equal T on under 50% of the seeds where it appears, and every claim must be false."""
    calls = []
    orig = _pm_mod._window_options

    def spy(*args):
        out = orig(*args)
        calls.append((args, out))
        return out

    _pm_mod._window_options = spy
    try:
        for seed in range(200):
            make(TOPIC, seed=seed, difficulty="exam", tier="exam")
    finally:
        _pm_mod._window_options = orig
    rel = []
    for args, (answer, distractors, _v, _f) in calls:
        pA, vA, pB, vB, T = args[:5]
        for opt in distractors:
            if not (opt.startswith("Yes") and "within the stated window" in opt):
                continue
            t = _frac(_YES_T.search(opt).group(1))
            assert 0 < t <= T
            assert _pm_mod._pos(pA, vA, t) != _pm_mod._pos(pB, vB, t), (opt, args)
            rel.append(t / T)
    assert len(rel) >= 50, len(rel)
    assert len(set(rel)) >= 3, set(rel)
    assert sum(1 for r in rel if r == 1) / len(rel) < 0.5
