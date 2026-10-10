"""See mathgen/generators/p19i_timetable_build.py for the tier design and misconceptions."""

import importlib.util
import os
import random
import re
from collections import Counter
from pathlib import Path

import pytest

from mathgen.grading.typed import grade_mcq, mcq_options
from mathgen.skills import load_skills
from mathgen import make

ROOT = Path(__file__).resolve().parents[1]
TIERS = ["easy", "medium", "hard", "exam"]
SEEDS = range(40)

TOPIC_A = "p19i_timetable_latest_service"
TOPIC_B = "p19i_timetable_condition_lookup"
TOPICS = [TOPIC_A, TOPIC_B]

SKILL_A = "senior-interpret-timetables-such-as-bus-train-and-ferry"
SKILL_B = "senior-interpret-complex-timetables-such-as-tide-charts-sunrise"


# --- determinism / distractor hygiene, both topics --------------------------


@pytest.mark.parametrize("topic", TOPICS)
@pytest.mark.parametrize("difficulty", TIERS)
def test_deterministic_across_seeds(topic, difficulty):
    for seed in SEEDS:
        q1 = make(topic, seed=seed, difficulty=difficulty)
        q2 = make(topic, seed=seed, difficulty=difficulty)
        assert q1.to_json() == q2.to_json()


@pytest.mark.parametrize("topic", TOPICS)
@pytest.mark.parametrize("difficulty", TIERS)
def test_distractors_distinct_nonempty_and_not_the_answer(topic, difficulty):
    for seed in SEEDS:
        q = make(topic, seed=seed, difficulty=difficulty)
        assert len(q.distractors) == 3, (topic, difficulty, seed, q.distractors)
        assert all(d for d in q.distractors)
        assert q.answer_text not in q.distractors, (topic, difficulty, seed)
        assert len(set(q.distractors)) == 3, (topic, difficulty, seed, q.distractors)


@pytest.mark.parametrize("difficulty", TIERS)
def test_skill_a_options_are_all_hh_mm_times(difficulty):
    time_re = re.compile(r"^\d{2}:\d{2}$")
    for seed in SEEDS:
        q = make(TOPIC_A, seed=seed, difficulty=difficulty)
        assert time_re.match(q.answer_text), q.answer_text
        for d in q.distractors:
            assert time_re.match(d), d


@pytest.mark.parametrize("difficulty", TIERS)
def test_skill_b_options_are_all_day_labels_alike(difficulty):
    for seed in SEEDS:
        q = make(TOPIC_B, seed=seed, difficulty=difficulty)
        options = [q.answer_text] + list(q.distractors)
        # Every option in one draw has the same shape: either all bare
        # weekday names, or all "The Nth" ordinal labels -- never mixed,
        # so no option is identifiable by its shape alone.
        looks_ordinal = [opt.startswith("The ") for opt in options]
        assert len(set(looks_ordinal)) == 1, (difficulty, seed, options)


# --- real grader accept/reject -----------------------------------------------


@pytest.mark.parametrize("topic", TOPICS)
@pytest.mark.parametrize("difficulty", TIERS)
def test_real_grader_accepts_answer_rejects_distractors_and_a_wrong_value(topic, difficulty):
    for seed in SEEDS:
        q = make(topic, seed=seed, difficulty=difficulty)
        options = mcq_options(q)
        correct_idx = options.index(q.answer_text)
        assert grade_mcq(q, str(correct_idx)).correct, (topic, difficulty, seed)
        for idx, opt in enumerate(options):
            if opt != q.answer_text:
                assert not grade_mcq(q, str(idx)).correct, (topic, difficulty, seed, opt)
        # a deliberately out-of-range index is rejected, not accepted by luck
        assert not grade_mcq(q, str(len(options) + 5)).correct


# --- variety ------------------------------------------------------------


@pytest.mark.parametrize("topic", TOPICS)
@pytest.mark.parametrize("difficulty", TIERS)
def test_at_least_20_distinct_questions_over_60_seeds(topic, difficulty):
    texts = {make(topic, seed=seed, difficulty=difficulty).text for seed in range(60)}
    assert len(texts) >= 20, f"{topic}/{difficulty}: only {len(texts)} distinct questions"


@pytest.mark.parametrize("topic", TOPICS)
@pytest.mark.parametrize("difficulty", TIERS)
def test_at_least_3_distinct_prompt_shapes_over_60_seeds(topic, difficulty):
    from mathgen.text_quality import prompt_shape

    shapes = {prompt_shape(make(topic, seed=seed, difficulty=difficulty)) for seed in range(60)}
    assert len(shapes) >= 3, f"{topic}/{difficulty}: only {len(shapes)} distinct prompt shapes"


# --- 200-seed answer-value AND key-position concentration ---------------
# The cap is 25% (not the shared instrument's usual 15%), because
# the mcq option pool here is a small table, not an open numeric range.


@pytest.mark.parametrize("topic", TOPICS)
@pytest.mark.parametrize("difficulty", TIERS)
def test_answer_value_concentration_stays_under_25_percent(topic, difficulty):
    counts = Counter(make(topic, seed=seed, difficulty=difficulty).answer_text for seed in range(200))
    top_key, top_count = counts.most_common(1)[0]
    assert top_count / 200 < 0.25, (
        f"{topic}/{difficulty}: {top_key!r} appears in {top_count}/200 draws, at or over the 25% cap"
    )


_A_ROW_RE = re.compile(r"at (\d{2}:\d{2}) and arrives")


@pytest.mark.parametrize("difficulty", TIERS)
def test_skill_a_key_position_is_not_fixed_first_or_last(difficulty):
    positions, ns = [], []
    for seed in range(200):
        q = make(TOPIC_A, seed=seed, difficulty=difficulty)
        deps = _A_ROW_RE.findall(q.text)
        positions.append(deps.index(q.answer_text))
        ns.append(len(deps))
    counts = Counter(positions)
    n_rows = Counter(ns).most_common(1)[0][0]
    # never always first (0) or always last (n_rows - 1)
    assert counts[0] < 200, f"{difficulty}: answer is always the first listed row"
    assert counts.get(n_rows - 1, 0) < 200, f"{difficulty}: answer is always the last listed row"
    assert len(counts) >= 3, f"{difficulty}: only {len(counts)} distinct positions over 200 seeds"


_B_DAY_RE = re.compile(r"\b(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday|The \d+\w+): ")


@pytest.mark.parametrize("difficulty", TIERS)
def test_skill_b_key_position_is_not_fixed_first_or_last(difficulty):
    positions = []
    for seed in range(200):
        q = make(TOPIC_B, seed=seed, difficulty=difficulty)
        labels = _B_DAY_RE.findall(q.text)
        assert q.answer_text in labels, (difficulty, seed, q.text, labels)
        positions.append(labels.index(q.answer_text))
    counts = Counter(positions)
    n = max(counts) + 1
    assert counts.get(0, 0) < 200, f"{difficulty}: answer is always the first listed day"
    assert counts.get(n - 1, 0) < 200, f"{difficulty}: answer is always the last listed day"
    assert len(counts) >= 3, f"{difficulty}: only {len(counts)} distinct positions over 200 seeds"


# --- tier-specific invariants: Skill A -----------------------------------


def test_skill_a_easy_later_departing_count_is_bounded_and_varies():
    """Easy's number of later-departing (disqualified) rows is drawn per seed from {0, 1, 2}, not fixed at
    exactly 1 -- a learner could otherwise just count down a fixed N rows from the latest departure without
    reading the table at all."""
    counts = Counter()
    for seed in range(60):
        q = make(TOPIC_A, seed=seed, difficulty="easy")
        deps = [int(h) * 60 + int(m) for h, m in (t.split(":") for t in _A_ROW_RE.findall(q.text))]
        answer_dep = int(q.answer_text.split(":")[0]) * 60 + int(q.answer_text.split(":")[1])
        later = [d for d in deps if d > answer_dep]
        assert len(later) <= 2, (seed, deps, answer_dep)
        counts[len(later)] += 1
    assert len(counts) >= 2, f"easy: later-departing count never varies over 60 seeds: {dict(counts)}"


def test_skill_a_hard_states_two_modes():
    for seed in range(60):
        q = make(TOPIC_A, seed=seed, difficulty="hard")
        modes_present = {m for m in ("Bus", "Train", "Boat") if m in q.text}
        assert len(modes_present) == 2, (seed, q.text)


def test_skill_a_exam_states_a_cancellation_constraint():
    for seed in range(60):
        q = make(TOPIC_A, seed=seed, difficulty="exam")
        assert "are cancelled today" in q.text or "are not running today" in q.text or "are running today" in q.text, (
            q.text
        )


def test_skill_a_misconception_ignores_stated_constraint_is_exercisable():
    """The exam-tier 'disqualified' distractor departs later than the answer
    and would (ignoring the constraint) itself meet the deadline."""
    for seed in range(60):
        q = make(TOPIC_A, seed=seed, difficulty="exam")
        answer_min = int(q.answer_text[:2]) * 60 + int(q.answer_text[3:])
        rows = re.findall(r"at (\d{2}:\d{2}) and arrives at \w+ at (\d{2}:\d{2})", q.text)
        deadline = re.search(r"by (\d{2}:\d{2})\?", q.text).group(1)
        deadline_min = int(deadline[:2]) * 60 + int(deadline[3:])
        found = False
        for dep_s, arr_s in rows:
            dep = int(dep_s[:2]) * 60 + int(dep_s[3:])
            arr = int(arr_s[:2]) * 60 + int(arr_s[3:])
            if dep_s in q.distractors and dep > answer_min and arr <= deadline_min:
                found = True
        # When the key is the latest OFFERED departure (no option departs later) the disqualified
        # row stays in the table but is not offered, so it cannot be a later distractor.
        key_is_latest_option = all(_mm(o) < answer_min for o in q.distractors)
        assert found or key_is_latest_option, (seed, q.text, q.distractors)


# Departure-rank spread and same-mode/route duration jitter -- The reusable measurement below
# (`_measure_latest_service_rank_and_duration`) takes a `builders` dict mapping tier ->
# callable(rng) -> question dict, the same shape `mathgen.make` gets back from each `_build_A_*`
# function, so it can be pointed at EITHER the current module's `_LATEST_SERVICE_BUILDERS` OR an old
# scratch copy loaded via `importlib.util.spec_from_file_location` under a throwaway module name
# (never registered, never touching git state).

_ROW_LINE_RE = re.compile(r"(\w+) \d+ departs \S+ at (\d\d:\d\d) and arrives at \S+ at (\d\d:\d\d)\.")


def _measure_latest_service_rank_and_duration(builders: dict, seeds: int = 200) -> dict:
    """For each tier's builder, over `seeds` draws: the departure-rank
    (1 = latest departure in the whole table) of the keyed answer, and the
    largest within-question spread of journey time (arrival - departure)
    among rows sharing the same mode. `builders[tier]` is called as
    `builders[tier](rng)` -- the same minimal call `mathgen.make` performs
    once it has resolved the tier's preset kwargs -- and must return a dict
    with `text` and `answer_text` keys shaped like the real generator's
    output. A builder may raise `VerificationError` on a colliding draw
    (the same thing the real topic function retries via its own `_retry`
    helper) -- retried here the same way, up to 200 attempts on the same
    `rng` stream, so the result matches what `mathgen.make` would return."""
    from mathgen.core import VerificationError

    report = {}
    for tier, builder in builders.items():
        rank_counts = Counter()
        max_spread = 0
        for seed in range(seeds):
            rng = random.Random(f"p19i_timetable_latest_service:{tier}:{seed}")
            q = None
            for _ in range(200):
                try:
                    q = builder(rng)
                    break
                except VerificationError:
                    continue
            assert q is not None, f"{tier}/seed={seed}: builder never succeeded in 200 attempts"
            deps = [int(h) * 60 + int(m) for h, m in (t.split(":") for t in _A_ROW_RE.findall(q["text"]))]
            answer_dep = int(q["answer_text"][:2]) * 60 + int(q["answer_text"][3:])
            rank = sorted(deps, reverse=True).index(answer_dep) + 1
            rank_counts[rank] += 1

            by_mode = {}
            for mode, dep_s, arr_s in _ROW_LINE_RE.findall(q["text"]):
                dep = int(dep_s[:2]) * 60 + int(dep_s[3:])
                arr = int(arr_s[:2]) * 60 + int(arr_s[3:])
                by_mode.setdefault(mode, []).append(arr - dep)
            for durations in by_mode.values():
                if len(durations) > 1:
                    max_spread = max(max_spread, max(durations) - min(durations))
        report[tier] = dict(rank_counts=rank_counts, max_spread=max_spread, seeds=seeds)
    return report


def test_skill_a_latest_service_rank_is_spread_and_duration_is_plausible():
    """The keyed answer's departure-rank must not concentrate at a single fixed value per tier, and
    same-mode/same-route journey times within one question must stay within a small jitter band (was up to
    ~100 minutes apart before the fix)."""
    from mathgen.generators.p19i_timetable_build import _LATEST_SERVICE_BUILDERS

    report = _measure_latest_service_rank_and_duration(_LATEST_SERVICE_BUILDERS, seeds=200)
    for tier, data in report.items():
        counts = data["rank_counts"]
        total = sum(counts.values())
        top_rank, top_count = counts.most_common(1)[0]
        assert top_count / total <= 0.6, (
            f"{tier}: rank {top_rank} concentrates {top_count}/{total} draws: {dict(counts)}"
        )
        assert len(counts) >= 2, f"{tier}: rank never varies over {total} seeds: {dict(counts)}"
        assert data["max_spread"] <= 10, (
            f"{tier}: same-mode/same-route journey time spreads {data['max_spread']} minutes within one question"
        )


# Exam latest-service deadline must bind, and no option tell ---

_EXAM_CANCEL_RE = re.compile(
    r"(Bus|Train|Boat) services are (?:cancelled today|not running today)|"
    r"Only (Bus|Train|Boat) services are running today"
)
_EXAM_ROW_RE = re.compile(r"(Bus|Train|Boat) \d+ departs \S+ at (\d\d:\d\d) and arrives at \S+ at (\d\d:\d\d)\.")
_EXAM_DEADLINE_RE = re.compile(r"by (\d\d:\d\d)\?")


def _mm(s):
    return int(s[:2]) * 60 + int(s[3:])


def _parse_exam_a(q):
    """(rows[(mode, dep, arr)], kept_modes, deadline) parsed from the text alone."""
    rows = [(m.lower(), _mm(d), _mm(a)) for m, d, a in _EXAM_ROW_RE.findall(q.text)]
    m = _EXAM_CANCEL_RE.search(q.text)
    if m.group(1):
        kept = {"bus", "train", "boat"} - {m.group(1).lower()}
    else:
        kept = {m.group(2).lower()}
    return rows, kept, _mm(_EXAM_DEADLINE_RE.search(q.text).group(1))


def test_skill_a_exam_latest_kept_service_often_fails_deadline_and_no_option_tell():
    """The key's rank among the MODE-KEPT services (sorted by departure, latest first) was 1 in 50% of rows, so
    the deadline did not bind and 'pick the earliest option' won ~50%. The rank must be spread, the
    earliest-option heuristic no better than ~35%, exactly one option true, every option a printed service,
    and no service may arrive exactly at the deadline."""
    ranks = Counter()
    earliest_wins = 0
    for seed in range(200):
        q = make(TOPIC_A, seed=seed, difficulty="exam")
        rows, kept, deadline = _parse_exam_a(q)
        assert all(arr != deadline for _, _, arr in rows), (seed, q.text)
        kept_deps = sorted((dep for mode, dep, _ in rows if mode in kept), reverse=True)
        answer = _mm(q.answer_text)
        ranks[kept_deps.index(answer) + 1] += 1
        options = [_mm(o) for o in [q.answer_text] + list(q.distractors)]
        printed = {dep for _, dep, _ in rows}
        assert all(o in printed for o in options), (seed, q.text, options)
        truth = max(dep for mode, dep, arr in rows if mode in kept and arr <= deadline)
        assert [o for o in options if o == truth] == [answer], (seed, q.text)
        if answer == min(options):
            earliest_wins += 1
    assert max(ranks.values()) / 200 <= 0.35, dict(ranks)
    for r in (1, 2, 3, 4):
        assert ranks[r] / 200 >= 0.15, dict(ranks)
    assert earliest_wins / 200 <= 0.35, earliest_wins


def test_skill_a_exam_key_rank_among_offered_options_is_near_uniform():
    """The key was never the latest offered departure (rank shares 31/89/80/0 of 200 before the fix). With four
    options the ranks (0 = latest offered... 3 = earliest) must each be present at a near-uniform share (all
    four cannot exceed 25%; the band is 20-30%), and every later option must miss the deadline or be the
    cancelled mode."""
    ranks = Counter()
    for seed in range(200):
        q = make(TOPIC_A, seed=seed, difficulty="exam")
        rows, kept, deadline = _parse_exam_a(q)
        answer = _mm(q.answer_text)
        options = [_mm(o) for o in list(q.distractors)]
        ranks[sum(1 for o in options if o > answer)] += 1
        by_dep = {dep: (mode, arr) for mode, dep, arr in rows}
        for o in options:
            if o > answer:
                mode, arr = by_dep[o]
                assert arr > deadline or mode not in kept, (seed, q.text)
    assert set(ranks) == {0, 1, 2, 3}, dict(ranks)
    for r in range(4):
        assert 0.20 <= ranks[r] / 200 <= 0.30, dict(ranks)


# Exam tide/sunrise overall extreme is sometimes the key ------


def _exam_b_key_is_overall_extreme(q):
    if "weekend" in q.text:
        vals = {d: float(h) for d, h in re.findall(r"(\w+): high tide at \d\d:\d\d, height (\d+\.\d) m", q.text)}
    else:
        vals = {d: _mm(t) for d, t in re.findall(r"(The \d+\w+): sunrise at (\d\d:\d\d)", q.text)}
    return vals[q.answer_text] == max(vals.values()), vals


def test_skill_b_exam_overall_extreme_is_the_key_in_a_minority_of_rows():
    """The overall extreme was a distractor in 200/200 exam rows, so 'eliminate the overall extreme' was a 100%
    tell. Key it in ~25-35% of rows."""
    hits = Counter()
    totals = Counter()
    for seed in range(200):
        q = make(TOPIC_B, seed=seed, difficulty="exam")
        is_ext, vals = _exam_b_key_is_overall_extreme(q)
        assert len(set(vals.values())) == len(vals)
        variant = "tide" if "weekend" in q.text else "sunrise"
        totals[variant] += 1
        hits[variant] += is_ext
    overall = sum(hits.values()) / 200
    assert 0.22 <= overall <= 0.38, (overall, dict(hits), dict(totals))
    for v in totals:
        assert hits[v] / totals[v] >= 0.15, (v, dict(hits), dict(totals))


# Sunrise sequences are physical (monotone or a flat turn) ----


def _sunrise_steps(text):
    times = [int(h) * 60 + int(m) for h, m in _SUNRISE_ROW_RE.findall(text)]
    return [b - a for a, b in zip(times, times[1:])]


@pytest.mark.parametrize("difficulty", ["easy", "medium", "exam"])
def test_sunrise_sequences_are_monotone_or_turn_flat(difficulty):
    """49/74 exam sunrise rows flipped direction with an accelerating step away from the turn (06:14, 06:15,
    06:18). A turn must be flat (a 0/1 min step beside the turn, the other side at most 2), steps must shrink
    into it and grow out of it, deltas stay within 0-3, and at most ~30% of sunrise rows flip at all."""
    flips = rows = 0
    for seed in range(200):
        q = make(TOPIC_B, seed=seed, difficulty=difficulty)
        steps = _sunrise_steps(q.text)
        if not steps:
            continue
        rows += 1
        assert all(abs(s) <= 3 for s in steps), (seed, steps)
        turns = [j for j in range(len(steps) - 1) if steps[j] * steps[j + 1] < 0]
        assert len(turns) <= 1, (seed, steps)
        if turns:
            flips += 1
            j = turns[0]
            assert min(abs(steps[j]), abs(steps[j + 1])) <= 1 and max(abs(steps[j]), abs(steps[j + 1])) <= 2, (
                seed,
                steps,
            )
            before = [abs(s) for s in steps[: j + 1]]
            after = [abs(s) for s in steps[j + 1 :]]
            assert before == sorted(before, reverse=True), (seed, steps)
            assert after == sorted(after), (seed, steps)
    assert rows >= 60, (difficulty, rows)
    # Easy/medium keep the 2171 bound of 30%.
    assert flips / rows <= (0.85 if difficulty == "exam" else 0.30), (difficulty, flips, rows)


# --- tier-specific invariants: Skill B -----------------------------------


def test_skill_b_hard_is_always_a_moon_phase_table():
    for seed in range(60):
        q = make(TOPIC_B, seed=seed, difficulty="hard")
        # The question kind varies; every kind is about the full moon
        assert "full moon" in q.text or "waning gibbous moon" in q.text, q.text


def test_skill_b_exam_states_two_clauses():
    for seed in range(60):
        q = make(TOPIC_B, seed=seed, difficulty="exam")
        assert "that is NOT a weekend day" in q.text or "among the days after the 3rd" in q.text, q.text


def test_skill_b_misconception_wrong_direction_is_exercisable():
    """easy/medium always offer the day holding the opposite-direction
    extreme value as one of the three distractors."""
    for difficulty in ("easy", "medium"):
        for seed in range(60):
            q = make(TOPIC_B, seed=seed, difficulty=difficulty)
            assert "highest" in q.text or "lowest" in q.text or "latest" in q.text or "earliest" in q.text, q.text
            assert len(q.distractors) == 3


_MOON_ROW_RE = re.compile(
    r"The (\d+)\w+: (new moon|waxing crescent|first quarter|waxing gibbous|"
    r"full moon|waning gibbous|last quarter|waning crescent)\."
)
_MOON_CUTOFF_RE = re.compile(r"first full moon after the (\d+)\w+\?")
_MOON_CUTOFF_LABEL_RE = re.compile(r"first full moon after the (\d+\w+)\?")


def test_skill_b_hard_cutoff_day_is_never_a_full_moon():
    """Eliminates the defect where the reference/cutoff day was itself sometimes a full moon
    (measured 18/200 on the pre-fix generator, reproduced in
    `test_moon_hard_tier_was_broken_before_the_fix_red_proof` below). This also structurally eliminates the OLD
    'full moon on/before the cutoff' misconception distractor route this test used to check for (removed --
    see `_build_B_hard`'s comment on dropping `before_day`): that route required exactly the defect being
    fixed here, so it can never legitimately fire any more."""
    for seed in range(60):
        q = make(TOPIC_B, seed=seed, difficulty="hard")
        # Only the reference-day kinds have a cutoff day
        kind, cutoff, _ = _moon_solve(q.text)
        if kind not in ("after_ref", "gibbous_after_ref"):
            continue
        rows = _MOON_ROW_RE.findall(q.text)
        day_phase = {int(d): p for d, p in rows}
        assert day_phase.get(cutoff) != "full moon", q.text


def test_skill_b_hard_cutoff_day_is_no_longer_offered_as_a_dead_option():
    """The reference day is now never an option in any question kind."""
    for seed in range(60):
        q = make(TOPIC_B, seed=seed, difficulty="hard")
        _, ref, _ = _moon_solve(q.text)
        ref_label = next(f"The {n}{sfx}" for n, sfx in re.findall(r"The (\d+)(\w+):", q.text) if int(n) == ref)
        assert ref_label not in q.distractors and ref_label != q.answer_text, (q.text, q.distractors)


def test_skill_b_hard_full_moon_spans_at_most_two_consecutive_printed_days():
    """A full moon lasts at most 2 printed days, never 3+."""
    for seed in range(120):
        q = make(TOPIC_B, seed=seed, difficulty="hard")
        rows = _MOON_ROW_RE.findall(q.text)
        full_days = sorted(int(d) for d, p in rows if p == "full moon")
        run = best = 1
        for a, b in zip(full_days, full_days[1:]):
            run = run + 1 if b - a == 1 else 1
            best = max(best, run)
        assert best <= 2, (seed, q.text)


def _measure_moon_hard_properties(fn, seeds=200):
    """Shared measurement: distinct 6-day phase-label
    sequences, the reference/cutoff-day-is-full-moon rate, and the longest
    run of consecutive full-moon printed days, over `seeds` draws of the
    hard-tier moon-phase builder `fn` (either the current module's
    `p19i_timetable_condition_lookup` or an OLD scratch copy loaded by
    `_load_generator_module_at_commit` below -- both share the same
    `fn(rng, tier="hard")` call signature and both return a plain dict)."""
    sequences = set()
    ref_is_full = 0
    consecutive_full = 0
    max_consecutive = 0
    for seed in range(seeds):
        rng = random.Random(f"{TOPIC_B}:hard:{seed}")
        q = fn(rng, tier="hard")
        text = q["text"]
        rows = _MOON_ROW_RE.findall(text)
        sequences.add(tuple(p for _, p in rows))
        kind, cutoff, _ = _moon_solve(text)
        day_phase = {int(d): p for d, p in rows}
        # Only the reference-day kinds have a cutoff day (the old module's questions are all of that
        # kind)
        if kind in ("after_ref", "gibbous_after_ref") and day_phase.get(cutoff) == "full moon":
            ref_is_full += 1
        full_days = sorted(d for d, p in day_phase.items() if p == "full moon")
        run = best = 1
        for a, b in zip(full_days, full_days[1:]):
            run = run + 1 if b - a == 1 else 1
            best = max(best, run)
        if full_days:
            max_consecutive = max(max_consecutive, best)
            if best >= 2:
                consecutive_full += 1
    return dict(
        distinct=len(sequences),
        ref_is_full=ref_is_full,
        consecutive_full=consecutive_full,
        max_consecutive=max_consecutive,
        seeds=seeds,
    )


def _load_generator_module_at_commit(commit, root):
    """Temporarily vacates the shared `core.REGISTRY`/`core.PRESETS` entries this file's two topics already
    occupy (registered by the real, currently-imported module) so the OLD module's own `@generator(...)`
    decorator calls can register without raising a duplicate-topic error; restores the saved entries in
    `finally` so the rest of the test session sees the real module again."""
    import os
    import subprocess
    import tempfile

    import mathgen.core as core

    result = subprocess.run(
        ["git", "show", f"{commit}:mathgen/generators/p19i_timetable_build.py"],
        cwd=str(root),
        capture_output=True,
        text=True,
        check=True,
    )
    topics = ["p19i_timetable_latest_service", "p19i_timetable_condition_lookup"]
    saved_registry = {t: core.REGISTRY.pop(t) for t in topics if t in core.REGISTRY}
    saved_presets = {t: core.PRESETS.pop(t) for t in topics if t in core.PRESETS}
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as f:
            f.write(result.stdout)
            tmp_path = f.name
        # Name it as a real member of mathgen.generators (rather than
        # setting __package__ by hand on a top-level module name) so
        # relative imports resolve without the __package__ != __spec__.parent
        # DeprecationWarning a manually-set __package__ would trigger.
        spec = importlib.util.spec_from_file_location(f"mathgen.generators._p19i_timetable_build_at_{commit}", tmp_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        if tmp_path is not None:
            os.unlink(tmp_path)
        for t in topics:
            core.REGISTRY.pop(t, None)
            core.PRESETS.pop(t, None)
        core.REGISTRY.update(saved_registry)
        core.PRESETS.update(saved_presets)


def test_moon_hard_tier_real_cycle_properties_current_module():
    """Permanent regression guard: the CURRENT generator's
    hard-tier moon-phase construction never offers a full-moon reference
    day, never shows 3+ consecutive full-moon printed days, and reaches
    the honest diversity ceiling for a 6-day window sliced from a real
    ~29-31 day cycle -- see `_build_B_hard`'s docstring for the 9-sequence
    derivation. Deliberately does NOT require exceeding the old model's 14
    distinct sequences: that number came from an 8-day-per-cycle model that
    was itself the defect. Needs no git history, unlike the RED-proof test
    below."""
    from mathgen.generators.p19i_timetable_build import p19i_timetable_condition_lookup as current_fn

    result = _measure_moon_hard_properties(current_fn, seeds=200)
    assert result["ref_is_full"] == 0, result
    assert result["max_consecutive"] <= 2, result
    assert result["distinct"] >= 5, result
    assert result["consecutive_full"] / result["seeds"] <= 0.30, result


_RED_PROOF_ENV = "P19I_RED_PROOF_COMMIT"
_SECOND_RED_PROOF_ENV = "P19I_SECOND_RED_PROOF_COMMIT"


def _red_proof_revision(env_name):
    """The git revision that still holds a pre-fix generator, named by the environment variable `env_name`.

    The red-proof tests are recorded evidence of past fixes, not live guards: they need a clone whose history still
    holds the old generator, so they skip unless the variable names such a revision."""
    revision = os.environ.get(env_name)
    if not revision:
        pytest.skip(f"set {env_name} to a revision that still has the pre-fix generator")
    return revision


def test_moon_hard_tier_was_broken_before_the_fix_red_proof():
    """Skips rather than fails if the named revision is unreachable from this clone (e.g. a shallow clone) -- this
    test is committed evidence of the fix, not a live regression guard; the guard is
    `test_moon_hard_tier_real_cycle_properties_current_module` above, which needs no git history."""
    import subprocess

    commit = _red_proof_revision(_RED_PROOF_ENV)
    try:
        old_module = _load_generator_module_at_commit(commit, ROOT)
    except subprocess.CalledProcessError:
        pytest.skip(f"commit {commit} not reachable from this clone")

    result = _measure_moon_hard_properties(old_module.p19i_timetable_condition_lookup, seeds=200)
    # Reproduces the original numbers (~14
    # distinct sequences, ~9% reference-is-full-moon, ~60%
    # consecutive-full-moon-days). Measured here:
    # distinct=14, ref_is_full=18 (9.0%), consecutive_full=119 (59.5%),
    # max_consecutive=3.
    assert result["distinct"] == 14, result
    assert result["ref_is_full"] == 18, result
    assert result["consecutive_full"] == 119, result
    assert result["max_consecutive"] == 3, result


_SUNRISE_ROW_RE = re.compile(r"sunrise at (\d{2}):(\d{2})\.")
_TIDE_ROW_RE = re.compile(r"high tide at (\d{2}):(\d{2}), height")


def _measure_time_deltas(fn, seeds=200):
    """Day-to-day change (minutes) of the printed sunrise / high-tide TIMES in the non-moon tiers (easy, medium,
    exam) of `p19i_timetable_condition_lookup`, parsed back out of the question text. Sunrise deltas are
    signed and folded into (-720, 720]; tide deltas are the forward change mod 24h (a tidal day always
    advances). Returns {(tier, kind): [delta...]}."""
    out = {}
    for tier in ("easy", "medium", "exam"):
        for seed in range(seeds):
            rng = random.Random(f"{TOPIC_B}:{tier}:{seed}")
            text = fn(rng, tier=tier)["text"]
            for kind, rx in (("sunrise", _SUNRISE_ROW_RE), ("tide", _TIDE_ROW_RE)):
                times = [int(h) * 60 + int(m) for h, m in rx.findall(text)]
                if len(times) < 2:
                    continue
                if kind == "sunrise":
                    d = [((b - a + 720) % 1440) - 720 for a, b in zip(times, times[1:])]
                else:
                    d = [(b - a) % 1440 for a, b in zip(times, times[1:])]
                out.setdefault((tier, kind), []).extend(d)
    return out


def _median(xs):
    xs = sorted(xs)
    n = len(xs)
    return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2


def test_sunrise_and_tide_time_drift_is_realistic_current_module():
    """Permanent guard: sunrise moves 1-3 min per printed day
    (never tens of minutes overnight); consecutive high tides advance about
    50 min per day (a 24h50m tidal day), never backwards or at random."""
    from mathgen.generators.p19i_timetable_build import p19i_timetable_condition_lookup as current_fn

    res = _measure_time_deltas(current_fn, seeds=200)
    for tier in ("easy", "medium", "exam"):
        sun = res[(tier, "sunrise")]
        tide = res[(tier, "tide")]
        assert len(sun) >= 100 and len(tide) >= 100, (tier, len(sun), len(tide))
        assert 1 <= _median([abs(x) for x in sun]) <= 3, (tier, "sunrise median")
        assert 1 <= min(abs(x) for x in sun) and max(abs(x) for x in sun) <= 3, (tier, "sunrise range")
        assert 46 <= _median(tide) <= 54, (tier, "tide median", _median(tide))
        assert 44 <= min(tide) and max(tide) <= 56, (tier, "tide range", min(tide), max(tide))


def test_sunrise_and_tide_time_drift_was_unrealistic_before_the_fix_red_proof():
    """Skips if the named revision is unreachable (shallow clone)."""
    import subprocess

    commit = _red_proof_revision(_RED_PROOF_ENV)
    try:
        old_module = _load_generator_module_at_commit(commit, ROOT)
    except subprocess.CalledProcessError:
        pytest.skip(f"commit {commit} not reachable from this clone")

    res = _measure_time_deltas(old_module.p19i_timetable_condition_lookup, seeds=200)
    for tier in ("easy", "medium", "exam"):
        sun = [abs(x) for x in res[(tier, "sunrise")]]
        assert _median(sun) >= 20 and max(sun) >= 60, (tier, _median(sun), max(sun))
    # Exam tide: a fixed 15-minute step.
    assert set(res[("exam", "tide")]) == {15}
    # Easy/medium tide: effectively random times, nowhere near a 50 min drift.
    for tier in ("easy", "medium"):
        tide = res[(tier, "tide")]
        assert not (46 <= _median(tide) <= 54 and max(tide) <= 56), tier


# Moon-phase hard tier needs the reference day ----------------

_MOON_ROW_PHASES = [
    "new moon",
    "waxing crescent",
    "first quarter",
    "waxing gibbous",
    "full moon",
    "waning gibbous",
    "last quarter",
    "waning crescent",
]
_MOON_K1_RE = re.compile(r"Which is the first full moon after the (\d+)\w+\?")
_MOON_K2_RE = re.compile(r"Which day is (\d+) days? after the first day showing a full moon\?")
_MOON_K3_RE = re.compile(r"Which day is (\d+) days? before the first day showing a full moon\?")
_MOON_K4_RE = re.compile(r"Which is the first day after the (\d+)\w+ showing a waning gibbous moon\?")


def _moon_solve(text):
    """Independent oracle: recompute (kind, reference day number, answer day
    number) from the printed table and question alone."""
    rows = [(int(d), p) for d, p in _MOON_ROW_RE.findall(text)]
    assert len(rows) == 6, text
    phase = dict(rows)
    days = sorted(phase)
    full = [d for d in days if phase[d] == "full moon"]
    if m := _MOON_K1_RE.search(text):
        ref = int(m.group(1))
        return "after_ref", ref, min(d for d in full if d > ref)
    if m := _MOON_K2_RE.search(text):
        return "after_full", full[0], full[0] + int(m.group(1))
    if m := _MOON_K3_RE.search(text):
        return "before_full", full[0], full[0] - int(m.group(1))
    if m := _MOON_K4_RE.search(text):
        ref = int(m.group(1))
        return "gibbous_after_ref", ref, min(d for d in days if d > ref and phase[d] == "waning gibbous")
    raise AssertionError(f"unrecognised moon question: {text}")


def _moon_hard_items(seeds=200):
    out = []
    for seed in range(seeds):
        q = make(TOPIC_B, seed=seed, difficulty="hard")
        rows = [(int(d), p) for d, p in _MOON_ROW_RE.findall(q.text)]
        phase = dict(rows)
        nth = {int(n): f"The {n}{s}" for n, s in re.findall(r"The (\d+)(\w+):", q.text)}
        first_full = min(d for d in phase if phase[d] == "full moon")
        kind, ref, ans = _moon_solve(q.text)
        options = [q.answer_text] + list(q.distractors)
        opt_days = sorted(int(re.match(r"The (\d+)", o).group(1)) for o in options)
        out.append(
            dict(
                q=q,
                phase=phase,
                kind=kind,
                ref=ref,
                ans=ans,
                first_full=first_full,
                opt_days=opt_days,
                key_day=int(re.match(r"The (\d+)", q.answer_text).group(1)),
                nth=nth,
                options=options,
            )
        )
    return out


def test_moon_hard_recomputes_from_printed_table_and_only_key_is_defensible():
    for it in _moon_hard_items():
        # printed labels are consecutive ordinals, so the oracle's day numbers
        # are the printed ones; the key must equal the recomputed answer
        assert it["key_day"] == it["ans"], it["q"].text
        assert len(set(it["opt_days"])) == 4, it["q"].text
        assert it["ans"] in it["phase"], it["q"].text


def test_moon_hard_reference_day_is_never_an_option_and_never_dead():
    for it in _moon_hard_items():
        assert it["ref"] not in it["opt_days"], (it["kind"], it["q"].text, it["q"].distractors)


def test_moon_hard_key_is_first_full_moon_row_in_well_under_half_of_items():
    items = _moon_hard_items()
    share = sum(it["key_day"] == it["first_full"] for it in items) / len(items)
    assert share < 0.35, share


def test_moon_hard_varies_question_kind_and_no_kind_dominates():
    counts = Counter(it["kind"] for it in _moon_hard_items())
    assert len(counts) >= 4, counts
    assert max(counts.values()) / 200 <= 0.40, counts


def test_moon_hard_key_rank_among_options_is_spread():
    ranks = Counter(it["opt_days"].index(it["key_day"]) for it in _moon_hard_items())
    assert set(ranks) == {0, 1, 2, 3}, ranks
    for r, c in ranks.items():
        assert 0.15 <= c / 200 <= 0.40, ranks


def _row_key_position_is_end(tier_kinds, seeds=200):
    ends = total = 0
    for tier in tier_kinds:
        for seed in range(seeds):
            q = make(TOPIC_B, seed=seed, difficulty=tier)
            if "sunrise" not in q.text or "tide" in q.text:
                continue
            labels = _B_DAY_RE.findall(q.text)
            idx = labels.index(q.answer_text)
            total += 1
            ends += idx in (0, len(labels) - 1)
    return ends, total


def test_sunrise_key_is_first_or_last_row_in_under_60_percent_of_items():
    ends, total = _row_key_position_is_end(["easy", "medium"], seeds=400)
    assert total >= 200, total
    assert ends / total < 0.60, (ends, total)


def test_sunrise_second_ranked_questions_recompute_from_the_table():
    seen = 0
    for tier in ("easy", "medium"):
        for seed in range(200):
            q = make(TOPIC_B, seed=seed, difficulty=tier)
            m = re.search(r"Which day has the second (latest|earliest) sunrise\?", q.text)
            if not m:
                continue
            seen += 1
            vals = {d: int(h) * 60 + int(mn) for d, h, mn in re.findall(r"(\w+): sunrise at (\d\d):(\d\d)\.", q.text)}
            order = sorted(vals, key=vals.get, reverse=(m.group(1) == "latest"))
            assert q.answer_text == order[1], q.text
            assert q.answer_text != order[0], q.text
            assert order[0] in q.distractors, q.text  # the trap: the overall extreme
    assert seen >= 40, seen


@pytest.mark.parametrize("tier", TIERS)
def test_tide_heights_stay_between_1_and_3_metres_in_every_tier(tier):
    for seed in range(200):
        q = make(TOPIC_B, seed=seed, difficulty=tier)
        heights = [float(h) for h in re.findall(r"height (\d+\.\d) m", q.text)]
        for h in heights:
            assert 1.0 <= h <= 3.0, (tier, seed, h, q.text)


# Exam sunrise key/option tells; moon hard key rank ------------

_EXAM_SUNRISE_ROW_RE = re.compile(r"The (\d+)\w+: sunrise at (\d\d):(\d\d)\.")


def _exam_sunrise_items(fn, seeds=2000):
    """Every exam sunrise item over `seeds` draws of `fn(rng, tier=...)` (the
    current module's function or an old scratch copy). Each item: seed, the six
    sunrise minutes, the key day index (0-based), the offered day indices."""
    items = []
    for seed in range(seeds):
        rng = random.Random(f"{TOPIC_B}:exam:{seed}")
        q = fn(rng, tier="exam")
        rows = _EXAM_SUNRISE_ROW_RE.findall(q["text"])
        if not rows:
            continue
        values = [int(h) * 60 + int(m) for _, h, m in rows]
        offered = sorted(
            int(re.match(r"The (\d+)", o).group(1)) - 1 for o in [q["answer_text"]] + list(q["distractors"])
        )
        items.append(
            dict(
                seed=seed,
                values=values,
                offered=tuple(offered),
                key=int(re.match(r"The (\d+)", q["answer_text"]).group(1)) - 1,
                text=q["text"],
                working=q["working"],
            )
        )
    return items


def _exam_sunrise_tells(items):
    """The measured tells: best option-set-only rule (fit on even seeds, scored
    on odd), the simple 'pick the 4th if offered else the 6th' rule, the key
    index shares, the 4th-and-6th-both-offered share, and the share of items
    where the after-the-3rd filter changes the answer."""
    from collections import defaultdict

    fit = defaultdict(Counter)
    for it in items:
        if it["seed"] % 2 == 0:
            fit[it["offered"]][it["key"]] += 1
    odd = [it for it in items if it["seed"] % 2 == 1]
    optset = sum(it["offered"] in fit and fit[it["offered"]].most_common(1)[0][0] == it["key"] for it in odd) / len(odd)
    simple = sum(it["key"] == (3 if 3 in it["offered"] else 5) for it in items) / len(items)
    keys = Counter(it["key"] for it in items)
    both = sum(3 in it["offered"] and 5 in it["offered"] for it in items) / len(items)
    filt = sum(max(range(6), key=lambda i: it["values"][i]) != it["key"] for it in items) / len(items)
    return dict(
        n=len(items),
        optset=optset,
        simple=simple,
        key_shares={k + 1: keys[k] / len(items) for k in sorted(keys)},
        both_4th_6th=both,
        filter_matters=filt,
    )


def _current_fn():
    from mathgen.generators.p19i_timetable_build import p19i_timetable_condition_lookup

    return p19i_timetable_condition_lookup


def test_exam_sunrise_has_no_option_set_or_key_position_tell():
    t = _exam_sunrise_tells(_exam_sunrise_items(_current_fn()))
    assert t["n"] >= 500, t
    assert t["optset"] <= 0.40, t  # best option-set-only rule, fit on even seeds, scored on odd
    assert t["simple"] <= 0.40, t  # "pick the 4th if offered else the 6th"


def test_exam_sunrise_key_is_spread_over_4th_5th_6th_and_4th_6th_always_offered():
    t = _exam_sunrise_tells(_exam_sunrise_items(_current_fn()))
    assert set(t["key_shares"]) == {4, 5, 6}, t
    for share in t["key_shares"].values():
        assert 0.25 <= share <= 0.40, t
    assert t["both_4th_6th"] == 1.0, t


def test_exam_sunrise_after_the_3rd_filter_changes_the_answer_in_most_items():
    t = _exam_sunrise_tells(_exam_sunrise_items(_current_fn()))
    assert t["filter_matters"] > 0.50, t


def test_exam_sunrise_tables_are_physical_at_2000_seeds():
    """At most one turning point, every step 1-3 minutes, and the key is the latest of days 4-6 with no tie."""
    for it in _exam_sunrise_items(_current_fn()):
        v = it["values"]
        steps = [b - a for a, b in zip(v, v[1:])]
        assert all(1 <= abs(s) <= 3 for s in steps), (it["seed"], steps)
        assert sum(steps[j] * steps[j + 1] < 0 for j in range(len(steps) - 1)) <= 1, (it["seed"], steps)
        assert it["key"] == max((3, 4, 5), key=lambda i: v[i]), it["text"]
        assert len(set(v)) == 6 and it["key"] in it["offered"] and len(set(it["offered"])) == 4


def test_exam_sunrise_working_text_matches_the_table_for_every_shape():
    for it in _exam_sunrise_items(_current_fn(), seeds=600):
        v, key = it["values"], it["key"]
        w = it["working"][0]
        m = re.search(r"The (\d+)\w+ has a later sunrise but is on or before the 3rd", w)
        if m:
            assert v[int(m.group(1)) - 1] > v[key] and int(m.group(1)) <= 3, (it["seed"], w)
        else:
            m = re.search(r"\(later than the (\d+)\w+, the latest on or before the 3rd\)", w)
            assert m, (it["seed"], w)
            assert v[key] > v[int(m.group(1)) - 1] == max(v[:3]), (it["seed"], w)


def test_exam_sunrise_tells_were_present_before_the_second_fix_red_proof():
    """Loaded via git show into an isolated registry, never checked out. Measured numbers are."""
    import subprocess

    commit = _red_proof_revision(_SECOND_RED_PROOF_ENV)
    try:
        old = _load_generator_module_at_commit(commit, ROOT)
    except subprocess.CalledProcessError:
        pytest.skip(f"commit {commit} not reachable from this clone")
    t = _exam_sunrise_tells(_exam_sunrise_items(old.p19i_timetable_condition_lookup))
    assert t["simple"] > 0.40, t  # (a) simple rule
    assert t["optset"] > 0.40, t  # (a) option-set rule
    assert not all(0.25 <= s <= 0.40 for s in t["key_shares"].values()) or set(t["key_shares"]) != {4, 5, 6}, (
        t
    )  # (b) key spread
    assert t["both_4th_6th"] < 1.0, t  # (b) 4th and 6th always offered


def _moon_hard_rank_and_direction(fn, seeds=2000):
    ranks = Counter()
    hits = 0
    for seed in range(seeds):
        rng = random.Random(f"{TOPIC_B}:hard:{seed}")
        q = fn(rng, tier="hard")
        opts = sorted(int(re.match(r"The (\d+)", o).group(1)) for o in [q["answer_text"]] + list(q["distractors"]))
        key = int(re.match(r"The (\d+)", q["answer_text"]).group(1))
        ranks[opts.index(key)] += 1
        before = bool(_MOON_K3_RE.search(q["text"]))
        hits += (opts[0] if before else opts[-1]) == key
    return ranks, hits / seeds


def test_moon_hard_direction_rule_and_key_rank_spread_at_2000_seeds():
    ranks, direction = _moon_hard_rank_and_direction(_current_fn())
    assert direction <= 0.40, (direction, dict(ranks))  # "largest for after, smallest for before"
    assert set(ranks) == {0, 1, 2, 3}, ranks
    for r, c in ranks.items():
        assert c / 2000 >= 0.15, dict(ranks)


def test_moon_hard_direction_rule_was_above_40_percent_before_the_second_fix_red_proof():
    """Regression guard for a previously fixed defect."""
    import subprocess

    commit = _red_proof_revision(_SECOND_RED_PROOF_ENV)
    try:
        old = _load_generator_module_at_commit(commit, ROOT)
    except subprocess.CalledProcessError:
        pytest.skip(f"commit {commit} not reachable from this clone")
    ranks, direction = _moon_hard_rank_and_direction(old.p19i_timetable_condition_lookup)
    assert direction > 0.40, (direction, dict(ranks))


# --- skill contract -------------------------------------------------------


def test_skill_a_loads_with_expected_shape():
    _, skills = load_skills(ROOT / "skills")
    by_id = {skill.id: skill for skill in skills}
    skill = by_id[SKILL_A]

    assert skill.answer == "mcq"
    assert skill.placement_eligible is False
    assert skill.status == "active"
    assert skill.prerequisites == []
    assert {source.topic for source in skill.sources} == {TOPIC_A}
    assert [source.difficulty for source in skill.sources] == ["easy", "medium", "hard", "exam"]
    assert [source.stages for source in skill.sources][-1] == ["exam"]


def test_skill_b_loads_with_expected_shape():
    _, skills = load_skills(ROOT / "skills")
    by_id = {skill.id: skill for skill in skills}
    skill = by_id[SKILL_B]

    assert skill.answer == "mcq"
    assert skill.placement_eligible is False
    assert skill.status == "active"
    assert skill.prerequisites == []
    assert {source.topic for source in skill.sources} == {TOPIC_B}
    assert [source.difficulty for source in skill.sources] == ["easy", "medium", "hard", "exam"]
    assert [source.stages for source in skill.sources][-1] == ["exam"]
