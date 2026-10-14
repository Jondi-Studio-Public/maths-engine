"""Re-derives the correct answer from `question.text` alone, via hand-written parsing/lookup logic -- does NOT
import `mathgen.generators.p19h_unit_choice_build`'s internal catalogues or lookup dicts (no shared constant
with construction).

The oracle below classifies the new prompt shapes."""

from __future__ import annotations

import re
from fractions import Fraction

from mathgen import make

ACTIVITY_TOPIC = "p19h_appropriate_activity_unit"
MILLIGRAMS_TOPIC = "p19h_need_for_milligrams"
TIERS = ["easy", "medium", "hard", "exam"]
SEEDS = range(60)

# ===========================================================================
# Oracle 1: appropriate_activity_unit
#
# Independent classification, judged on English meaning:
#   * a SPEED quantity is settled by the reporting convention the prompt
#     names: "SI base units" -> m/s; a general-audience medium -> km/h;
#   * a DISTANCE quantity is settled by the real scale of the object: the
#     tables below are hand-transcribed lists of the distance phrases, each
#     judged as coarse (km) or fine (m); a prompt's phrase is found by
#     longest match. No generator constant is imported.
# ===========================================================================

_KM_PHRASES = (
    "the distance a climbing team walks approaching a base camp",
    "the distance a crew rows in a long training session",
    "the distance a cyclist rides on a full day's tour",
    "the distance a marathon swimmer covers in an open-water crossing",
    "the distance a paddler covers on a multi-day kayaking trip",
    "the distance a runner covers in a long training run",
    "the distance a skier covers in a full day on the slopes",
    "the distance a swimmer covers in an open-water race",
    "the distance between two country towns",
    "the distance covered in a 24-hour relay running event",
    "the distance covered in a day of touring by car",
    "the distance covered on a horse-riding trek",
    "the distance covered on a long road trip",
    "the distance covered on a multi-day charity walk",
    "the distance covered on a multi-day hiking trip",
    "the distance covered on a multi-day mountain-bike tour",
    "the distance covered on a week-long cycling tour",
    "the distance driven in a day's touring",
    "the distance driven on a long outback road trip",
    "the distance driven on a road cycling classic",
    "the distance flown between two capital cities on a domestic flight",
    "the distance flown by a long-haul airliner between Sydney and Singapore",
    "the distance from a beach car park to a remote surf break by track",
    "the distance from a city to its regional airport",
    "the distance paddled on a multi-day canoe expedition",
    "the distance swum in a cross-strait charity swim",
    "the length of a city bike-path loop",
    "the length of a city fun-run course",
    "the length of a coastal cycling path",
    "the length of a coastal drive between two surf breaks",
    "the length of a coastal walking track",
    "the length of a cross-country hiking trail",
    "the length of a cross-country race course",
    "the length of a cross-country ski race",
    "the length of a cross-country skiing trail",
    "the length of a day-walk loop",
    "the length of a half-marathon course",
    "the length of a long-distance orienteering course",
    "the length of a mountain bike race course",
    "the length of a national park's hiking circuit",
    "the length of a ridge traverse",
    "the length of a river rafting run",
    "the length of a river rowing race course",
    "the length of a scenic drive along a coastline",
    "the length of a school cross-country run",
    "the length of a sea-kayak day paddle",
    "the length of a stage in a multi-day cycling race",
    "the length of a suburban train line",
    "the length of a triathlon's cycling leg",
    "the length of an orienteering course",
    "the length of an ultramarathon course",
    "the length of one stage of a multi-day cycling race",
    "the total distance of a 24-hour orienteering rogaine",
    "the total distance of a marathon",
)
_M_PHRASES = (
    "the depth a scuba diver descends to",
    "the depth of a diving pool",
    "the distance a discus thrower's throw travels",
    "the distance a hammer throw travels",
    "the distance a shot-putter throws the shot",
    "the distance between hurdles on a hurdles track",
    "the distance between two nearby checkpoints",
    "the distance of a long jump attempt",
    "the height a climber ascends on a single pitch",
    "the height a high jumper clears",
    "the height a pole vaulter clears",
    "the height of a basketball hoop",
    "the height of a boulder beside a trail",
    "the height of a climbing wall",
    "the height of a diving platform",
    "the length of a basketball court",
    "the length of a beach volleyball court",
    "the length of a bowling green rink",
    "the length of a car's parking space",
    "the length of a climbing rope",
    "the length of a cricket pitch",
    "the length of a family car",
    "the length of a football field",
    "the length of a javelin throw",
    "the length of a kayak",
    "the length of a mountain bike",
    "the length of a netball court",
    "the length of a road bike",
    "the length of a rowing eight boat",
    "the length of a single lap of an athletics track",
    "the length of a single oar",
    "the length of a single wave ride",
    "the length of a skateboard half-pipe",
    "the length of a sprint relay changeover zone",
    "the length of a surfboard",
    "the length of an Olympic swimming pool",
    "the length of one lap of a swimming pool",
    "the length of the home straight on an athletics track",
    "the vertical drop of a beginner slope",
    "the width of a bike's handlebars",
    "the width of a climbing wall's starting ledge",
    "the width of a groomed ski run",
    "the width of a narrow forest track",
    "the width of a narrow hiking trail",
    "the width of a narrow river channel",
    "the width of a narrow singletrack",
    "the width of a tennis court",
    "the width of an athletics track lane",
)


_PUBLIC_RE = re.compile(r"general|public|visitors|everyday|spectators")


def _speed_key_from_convention(clause: str) -> str:
    if "SI base units" in clause:
        assert not _PUBLIC_RE.search(clause), clause
        return "m/s"
    assert _PUBLIC_RE.search(clause), f"speed item names no recognised convention: {clause!r}"
    return "km/h"


def _distance_key(clause: str) -> str:
    hits = [(len(p), "km") for p in _KM_PHRASES if p in clause]
    hits += [(len(p), "m") for p in _M_PHRASES if p in clause]
    assert hits, f"unrecognised distance phrase: {clause!r}"
    hits.sort(reverse=True)
    top = hits[0]
    assert all(h[1] == top[1] for h in hits if h[0] == top[0]), clause
    return top[1]


def _classify_clause(clause: str) -> str:
    if re.search(r"\bspeed\b", clause):
        return _speed_key_from_convention(clause)
    return _distance_key(clause)


def _oracle_activity_single(text: str) -> str:
    """easy / medium / hard: exactly one quantity in the prompt."""
    assert "two quantities" not in text, text
    return _classify_clause(text)


def _oracle_activity_exam(text: str) -> str:
    m = re.search(
        r"First, (.+?)\. Second, (.+?)\. Considering only the (first|second) of these two",
        text,
    )
    assert m, text
    first, second, which = m.groups()
    return _classify_clause(first if which == "first" else second)


def test_oracle_activity_unit_easy_medium_hard():
    checked = 0
    for tier in ("easy", "medium", "hard"):
        for seed in SEEDS:
            q = make(ACTIVITY_TOPIC, seed=seed, difficulty=tier, tier=tier)
            expected = _oracle_activity_single(q.text)
            assert expected == q.answer_text, (tier, seed, q.text, q.answer_text, expected)
            checked += 1
    assert checked == 3 * len(SEEDS)


def test_oracle_activity_unit_exam():
    checked = 0
    for seed in SEEDS:
        q = make(ACTIVITY_TOPIC, seed=seed, difficulty="exam", tier="exam")
        expected = _oracle_activity_exam(q.text)
        assert expected == q.answer_text, (seed, q.text, q.answer_text, expected)
        checked += 1
    assert checked == len(SEEDS)


# ===========================================================================
# Oracle 2: need_for_milligrams
#
# Independent re-derivation: parse the stated masses (and, in medium/hard,
# the count of identical items) from the text, total them in mg with a
# hand-written ladder, and take the unit that expresses the total as a whole
# number of 1-999. Exactly one unit must qualify.
# ===========================================================================

_LADDER = {"mg": Fraction(1), "g": Fraction(1_000), "kg": Fraction(1_000_000), "t": Fraction(1_000_000_000)}
_MASS_UNITS = ("mg", "g", "kg", "t")
_VALUE_UNIT_RE = re.compile(r"([\d.]+)\s+(mg|g|kg|t)\b")
_COLLECTIVE_RE = re.compile(r"lists (\d+) [^,.]*?, each with a mass of ([\d.]+) (mg|g|kg|t)\b")


def _is_clean(value: Fraction) -> bool:
    return value.denominator == 1 and 0 < value <= 999


def _unique_clean_key(total_mg: Fraction, text: str) -> str:
    clean = [u for u in _MASS_UNITS if _is_clean(total_mg / _LADDER[u])]
    assert len(clean) == 1, (text, total_mg, clean)
    return clean[0]


def _oracle_milligrams_single(text: str) -> str:
    """easy: exactly one stated value+unit."""
    matches = _VALUE_UNIT_RE.findall(text)
    assert len(matches) == 1, (text, matches)
    value_str, unit = matches[0]
    return _unique_clean_key(Fraction(value_str) * _LADDER[unit], text)


def _oracle_milligrams_collective(text: str) -> str:
    """medium / hard: count x per-item mass, plus (hard) an extra stated mass."""
    m = _COLLECTIVE_RE.search(text)
    assert m, text
    count, per, unit = m.groups()
    total_mg = int(count) * Fraction(per) * _LADDER[unit]
    tail = text[m.end() :].split("Which unit")[0]
    for value_str, extra_unit in _VALUE_UNIT_RE.findall(tail):
        total_mg += Fraction(value_str) * _LADDER[extra_unit]
    return _unique_clean_key(total_mg, text)


def _oracle_milligrams_exam(text: str) -> str:
    """exam: two stated masses; the unit that expresses their COMBINED mass."""
    matches = _VALUE_UNIT_RE.findall(text)
    assert len(matches) == 2, (text, matches)
    total_mg = sum((Fraction(v) * _LADDER[u] for v, u in matches), Fraction(0))
    return _unique_clean_key(total_mg, text)


def _oracle_milligrams(text: str, tier: str) -> str:
    if tier == "easy":
        return _oracle_milligrams_single(text)
    if tier in ("medium", "hard"):
        return _oracle_milligrams_collective(text)
    return _oracle_milligrams_exam(text)


def test_oracle_milligrams_easy_medium_hard():
    checked = 0
    for tier in ("easy", "medium", "hard"):
        for seed in SEEDS:
            q = make(MILLIGRAMS_TOPIC, seed=seed, difficulty=tier, tier=tier)
            expected = _oracle_milligrams(q.text, tier)
            assert expected == q.answer_text, (tier, seed, q.text, q.answer_text, expected)
            checked += 1
    assert checked == 3 * len(SEEDS)


def test_oracle_milligrams_exam():
    checked = 0
    for seed in SEEDS:
        q = make(MILLIGRAMS_TOPIC, seed=seed, difficulty="exam", tier="exam")
        expected = _oracle_milligrams(q.text, "exam")
        assert expected == q.answer_text, (seed, q.text, q.answer_text, expected)
        checked += 1
    assert checked == len(SEEDS)


# ===========================================================================
# Negative controls: take a REAL generated question, corrupt the
# submitted answer to a wrong-but-plausible other key, and confirm (a) the
# real grader rejects it and (b) the oracle's OWN independent re-derivation
# from `question.text` alone (never reading `answer_text`) never agrees with
# the corrupted key -- proves the oracle actually discriminates rather than
# trivially agreeing with whatever key it's shown.
# ===========================================================================


def test_activity_oracle_rejects_a_corrupted_answer():
    from mathgen.grading.typed import GRADERS, mcq_options

    grade_mcq = GRADERS["mcq"]
    checked = 0
    for tier in TIERS:
        for seed in range(3):
            q = make(ACTIVITY_TOPIC, seed=seed, difficulty=tier, tier=tier)
            expected = _oracle_activity_exam(q.text) if tier == "exam" else _oracle_activity_single(q.text)
            assert expected == q.answer_text, (tier, q.text, q.answer_text, expected)
            options = mcq_options(q)
            wrong_option = next(o for o in options if o != q.answer_text)
            wrong_index = options.index(wrong_option)
            assert not grade_mcq(q, wrong_index).correct, (tier, q.text, wrong_option)
            assert expected != wrong_option, (tier, q.text, wrong_option)
            checked += 1
    assert checked == 3 * len(TIERS)


def test_milligrams_oracle_rejects_a_corrupted_answer():
    from mathgen.grading.typed import GRADERS, mcq_options

    grade_mcq = GRADERS["mcq"]
    checked = 0
    for tier in TIERS:
        for seed in range(3):
            q = make(MILLIGRAMS_TOPIC, seed=seed, difficulty=tier, tier=tier)
            expected = _oracle_milligrams(q.text, tier)
            assert expected == q.answer_text, (tier, q.text, q.answer_text, expected)
            options = mcq_options(q)
            wrong_option = next(o for o in options if o != q.answer_text)
            wrong_index = options.index(wrong_option)
            assert not grade_mcq(q, wrong_index).correct, (tier, q.text, wrong_option)
            assert expected != wrong_option, (tier, q.text, wrong_option)
            checked += 1
    assert checked == 3 * len(TIERS)


def test_oracle_is_not_a_rubber_stamp():
    """Feed the oracles hand-edited prompts whose key must change."""
    si = (
        "Which unit is the most appropriate for reporting a car's speed on a highway "
        "in a physics-class calculation that uses SI base units?"
    )
    pub = (
        "Which unit is the most appropriate for reporting a car's speed on a highway "
        "on a tourist brochure for visitors?"
    )
    assert _oracle_activity_single(si) == "m/s"
    assert _oracle_activity_single(pub) == "km/h"
    # 4 x 500 g = 2000 g = 2 kg; 4 x 100 g = 400 g.
    kg = "A dairy lists 4 blocks of butter, each with a mass of 500 g. Which unit"
    assert _oracle_milligrams_collective(kg) == "kg"
    g = "A dairy lists 4 blocks of butter, each with a mass of 100 g. Which unit"
    assert _oracle_milligrams_collective(g) == "g"


# ===========================================================================
# Enumerate every template x slot combination the generator can emit, over
# >= 1000 seeds total per skill, and mark each coherent.
# ===========================================================================


def test_activity_unit_enumerates_all_template_slot_combinations_coherent():
    total_checked = 0
    for tier in TIERS:
        for seed in range(300):
            q = make(ACTIVITY_TOPIC, seed=seed, difficulty=tier, tier=tier)
            expected = _oracle_activity_exam(q.text) if tier == "exam" else _oracle_activity_single(q.text)
            assert expected == q.answer_text, (tier, seed, q.text)
            total_checked += 1
    assert total_checked >= 1000


def test_milligrams_enumerates_all_template_slot_combinations_coherent():
    total_checked = 0
    for tier in TIERS:
        for seed in range(300):
            q = make(MILLIGRAMS_TOPIC, seed=seed, difficulty=tier, tier=tier)
            expected = _oracle_milligrams(q.text, tier)
            assert expected == q.answer_text, (tier, seed, q.text)
            total_checked += 1
    assert total_checked >= 1000
