"""Properties every `exd_x_r3*` exam chain must keep, checked across all of them at once.

Round-3 integration found three families of defect that each worker's own tests had missed, and this file pins them:

* a generator that raised `VerificationError` on a rare seed (the draw was accepted before the distractors of its last
  part were known to be distinct from the key);
* two parts of one question with the same answer, so a learner who knew one had the other for nothing and a reply
  that repeated part (a) could earn a later part by follow-through;
* a degenerate reply ("0" or "1" or the key of part (a) typed into every part) that earned marks it had not earned
  part by part.
"""

from __future__ import annotations

import pytest

from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.generators._exd_x_r3_distinct import answer_key, distinct_answers


def _chains():
    found = []
    for skill in load_skills()[1]:
        for source in skill.sources:
            if source.difficulty == "exam" and source.topic.startswith("exd_x_r3"):
                found.append((skill, source))
    return found


CHAINS = _chains()
IDS = [source.topic for _, source in CHAINS]
#: Topics whose rare failing seeds (found by sweeping 3000) were fixed at integration; they get a longer sweep.
FIXED = {
    "exd_x_r3i_product_group_orders",
    "exd_x_r3h_ideal_unknown_coefficients",
    "exd_x_r3j_euler_power_residue",
    "exd_x_r3g_rsa_keypair",
}


def _make(source, seed):
    return make(source.topic, seed=seed, **(source.params or {}))


def test_all_the_converted_chains_are_found():
    # 42 skills carry an r3 exam chain; senior-identify-common-usage-of-rates was reverted (recognition only).
    assert len(CHAINS) == 42
    assert not [s for _, s in CHAINS if s.topic == "exd_x_r3h_rates_speed_pulse"]


@pytest.mark.parametrize(("skill", "source"), CHAINS, ids=IDS)
def test_every_seed_builds(skill, source):
    for seed in range(1000 if source.topic in FIXED else 300):
        _make(source, seed)


@pytest.mark.parametrize(("skill", "source"), CHAINS, ids=IDS)
def test_no_two_parts_share_an_answer(skill, source):
    for seed in range(300):
        q = _make(source, seed)
        keys = [answer_key(p) for p in q.parts]
        assert len(set(keys)) == len(keys), (seed, [p.answer_text for p in q.parts])


@pytest.mark.parametrize(("skill", "source"), CHAINS, ids=IDS)
def test_a_degenerate_reply_earns_only_what_each_part_earns_alone(skill, source):
    for seed in range(40):
        q = _make(source, seed)
        labels = [p.label for p in q.parts]
        for reply in ("0", "1", "-1", q.parts[0].answer_text):
            together = grade(skill, q, {label: reply for label in labels})
            for p in q.parts:
                earned = together.parts[p.label].marks
                if earned:
                    alone = grade(skill, q, {**{label: "" for label in labels}, p.label: reply})
                    assert alone.parts[p.label].marks >= earned, (source.topic, seed, p.label, reply)


@pytest.mark.parametrize("topic", ["exd_x_r3j_euler_power_residue", "exd_x_r3g_rsa_keypair"])
def test_zeros_earn_nothing_where_the_chain_uses_a_floor(topic):
    """Both chains reduce with a floor in a follow-through; a reply of zero must not carry through it."""
    skill, source = next((s, src) for s, src in CHAINS if src.topic == topic)
    for seed in range(100):
        q = _make(source, seed)
        assert all(p.answer_text != "0" for p in q.parts)
        result = grade(skill, q, {p.label: "0" for p in q.parts})
        assert result.marks == 0, (seed, result.marks)


def test_residue_contour_has_exactly_two_poles_inside():
    source = next(src for _, src in CHAINS if src.topic == "exd_x_r3j_residue_contour")
    for seed in range(100):
        q = _make(source, seed)
        a, b, c, d = (p.answer_text for p in q.parts)
        assert d not in (a, b, c), seed


def test_distinct_answers_draws_again_until_the_parts_differ():
    class _Rng:
        def __init__(self):
            self.draws = 0

    def fake(rng):
        rng.draws += 1
        same = rng.draws < 3
        return {"parts": [{"answer_text": "2"}, {"answer_text": "2.0" if same else "3"}]}

    rng = _Rng()
    out = distinct_answers(fake)(rng)
    assert rng.draws == 3 and [p["answer_text"] for p in out["parts"]] == ["2", "3"]


def test_answer_key_reads_numbers_as_numbers():
    assert answer_key({"answer_text": "2.00"}) == answer_key({"answer_text": "2"})
    assert answer_key({"answer_text": "6/3"}) == answer_key({"answer_text": "2"})
    assert answer_key({"answer_text": "3:2"}) != answer_key({"answer_text": "2:3"})


def test_euler_edge_room_text_agrees_in_number():
    source = next(src for _, src in CHAINS if src.topic == "exd_x_r3i_euler_edge_room")
    seen = set()
    for seed in range(300):
        text = _make(source, seed).text
        assert " 1 are " not in text and "1 pentagons" not in text, text
        seen.update(w for w in ("1 is a triangle", "1 is a quadrilateral", "1 is a pentagon") if w in text)
    assert seen, "no draw had a single face of one kind"
