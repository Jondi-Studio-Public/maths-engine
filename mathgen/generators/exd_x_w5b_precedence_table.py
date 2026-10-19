"""Generates `exd_x_w5b_precedence_table`: a multi-part exam question on a project given as a precedence table.

The table has seven or eight activities and needs two to four dummy activities. The chain a senior exam sets:

(a) the fewest dummy activities the network diagram needs                      number, 2 marks
(b) the number of arrows in the diagram, activities and dummies together       number, 1 mark, part_a + activities
(c) the critical time of the project                                           number, 2 marks
(d) the new critical time when a critical activity is shortened by more
    than its margin over the nearest rival route                              number, 2 marks, part_c - saving

The dummy count is fixed first (the table is filed under it) and re-solved by `_w5b_dummy_kit`; the critical time comes
from a forward pass and a count of every route. For (d) the saving is smaller than the reduction, because another
route takes over, which is the step a learner who just subtracts the reduction misses.
"""

from __future__ import annotations

import random

from ..core import attempts, check, generator, phrase
from ._exd_distinct_parts import distinct_part_answers
from .exd_x_g6_project_network import _analyse, _forward, _length, _network_text, _routes
from .exd_x_r3b_common import number_part
from .w5b_precedence_table_build import dummy_candidates, dummy_working, pick_table, time_candidates, time_working

_STEMS = (
    "A project has the activities below, with durations in days and immediate predecessors. Work starts at time 0. {net} "
    "The project is shown as a network diagram in which each activity is an arrow, using as few dummy activities as "
    "possible.",
    "The precedence table for a project is shown. Dummy activities have zero duration. {net} "
    "A network diagram is drawn from the table with each activity as an arrow and the fewest possible dummies.",
    "A building project is planned as the activities below, starting at time 0. {net} "
    "The activities are drawn as arrows in a network diagram that uses as few dummy activities as it can.",
)


def _completion(table, dur):
    return max(_forward(table, dur)[1].values())


def _build(rng: random.Random) -> dict:
    activities = rng.choice([7, 8])
    dummies = rng.choice([2, 3, 4])
    table = pick_table(rng, activities, dummies)
    for _ in attempts(2000, "a table with a single critical route and a rival that a crash can overtake"):
        dur = {a: rng.randint(2, 12) for a in table}
        est, finish, _flt, routes, lengths = _analyse(table, dur)
        longest = [r for r in routes if lengths[r] == finish]
        if len(longest) != 1:
            continue
        options = []
        for y in sorted(longest[0]):
            rivals = [lengths[r] for r in routes if y not in r]
            if not rivals or dur[y] < 4:
                continue
            gap = finish - max(rivals)
            if 1 <= gap <= dur[y] - 3:
                options.append((y, gap))
        if options:
            break
    y, gap = rng.choice(options)
    reduction = rng.randint(gap + 1, dur[y] - 2)
    new_dur = dict(dur, **{y: dur[y] - reduction})
    new_finish = _completion(table, new_dur)
    saving = finish - new_finish
    check(saving == gap and 0 < saving < reduction, "the saving is not the margin over the nearest rival route")
    check(max(_length(r, new_dur) for r in _routes(table)) == new_finish, "new time disagrees with the routes")

    total = activities + dummies
    stem = phrase(rng, _STEMS, net=_network_text(table, dur))
    part_a = number_part(
        "How many dummy activities does the diagram contain?",
        dummies,
        0,
        2,
        dummy_working(table, dummies),
        dummy_candidates(table, dummies),
        sensible_range=(0, 10),
    )
    part_b = number_part(
        "How many arrows, activities and dummies together, does the diagram have?",
        total,
        0,
        1,
        [f"There are {activities} activities and {dummies} dummies: {activities} + {dummies} = {total}."],
        [
            ("counts-only-the-activities", activities),
            ("counts-only-the-dummies", dummies),
            ("arrow-count-one-too-many", total + 1),
            ("arrow-count-one-too-few", total - 1),
        ],
        follow_through=f"part_a + {activities}",
        sensible_range=(0, 20),
    )
    part_c = number_part(
        "Find the critical time of the project, in days.",
        finish,
        0,
        2,
        time_working(table, dur, finish, routes, lengths),
        time_candidates(table, dur, finish, est, routes, lengths),
        sensible_range=(0, None),
    )
    plural = "s" if gap != 1 else ""
    part_d = number_part(
        f"Activity {y} is shortened by {reduction} days, to {dur[y] - reduction} days, and every other duration is "
        "unchanged. Find the new critical time, in days.",
        new_finish,
        0,
        2,
        [
            f"Another route is only {gap} day{plural} shorter than the critical route.",
            f"The project is shortened by {saving} days, to {new_finish} days.",
        ],
        [
            ("crash-saving-taken-in-full", finish - reduction),
            ("ignored-the-reduction", finish),
            ("subtracted-overshoot-instead-of-saving", finish - (reduction - gap)),
        ],
        follow_through=f"part_c - {saving}",
        sensible_range=(0, None),
    )
    return dict(text=stem, parts=[part_a, part_b, part_c, part_d])


@generator("exd_x_w5b_precedence_table", level=12, presets={"exam": dict(tier="exam")})
@distinct_part_answers
def exd_x_w5b_precedence_table(rng: random.Random, tier: str = "exam") -> dict:
    """Dummy activities, arrows, critical time and the time after a crash, for a table of seven or eight activities."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    return _build(rng)
