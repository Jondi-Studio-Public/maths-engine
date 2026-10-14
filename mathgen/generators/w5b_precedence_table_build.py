"""Generates `w5b_precedence_table` questions (easy, medium, hard): activity networks drawn from a precedence table.

``critical-path-from-a-precedence-table`` (``financial-discrete``, answer ``number``). A project is given as a table of
activities, durations and immediate predecessors. The two things a senior question asks of the table, without drawing
the diagram, are how many dummy activities the network diagram needs and the critical time.

* easy: five or six activities; how many dummy activities does the diagram need (the fewest possible)?
* medium: six or seven activities with shared predecessors, so the diagram has a dummy; the critical time, found from
  the table's longest route. A dummy activity has no duration and adds nothing to a route.
* hard: seven or eight activities. Either the dummy count again, with two to four dummies; or the number of arrows in
  the finished diagram (activities plus dummies); or the critical time of a network whose rival route is within two
  days of the critical one.

Every table is chosen from `_w5b_dummy_tables` BY ITS DUMMY COUNT, so the answer is fixed before the question is built,
and is then re-solved: `fewest_dummies` searches every network and must agree, and the network it returns is read back
into a precedence table that must match the question's. The critical time is found by a forward pass and confirmed by
enumerating every route.
"""

from __future__ import annotations

import random

from ..core import check, draw_until, generator, phrase
from ._w5b_dummy_kit import closure_table, fewest_dummies, network_of, network_table
from ._w5b_dummy_tables import TABLES
from ._w5b_kit import number_question
from .exd_x_g6_project_network import _analyse, _network_text

_D_LO, _D_HI = 2, 12


def parse(row: str) -> dict:
    table = {}
    for item in row.split(";"):
        name, _, preds = item.partition(":")
        table[name] = preds.split(",") if preds else []
    return table


def pick_table(rng: random.Random, activities, dummies) -> dict:
    """A table with `activities` rows needing exactly `dummies` dummies, verified by solving it again."""
    rows = TABLES[(activities, dummies)]
    table = parse(rng.choice(rows))
    solved = fewest_dummies(table)
    check(solved == dummies, f"{table} needs {solved} dummies, not the {dummies} it was filed under")
    check(
        network_table(network_of(table)) == closure_table(table),
        "the solver's network does not reproduce the precedence table",
    )
    return table


def merges(table) -> int:
    return sum(1 for a in table if len(table[a]) >= 2)


def forks(table) -> int:
    return sum(1 for a in table if sum(1 for b in table if a in table[b]) >= 2)


def starts(table) -> int:
    return sum(1 for a in table if not table[a])


def dummy_candidates(table, dummies):
    return [
        ("counts-every-merge-as-a-dummy", merges(table)),
        ("counts-every-fork-as-a-dummy", forks(table)),
        ("counts-start-activities-beyond-the-first-as-dummies", max(starts(table) - 1, 0)),
        ("dummy-count-one-too-many", dummies + 1),
        ("dummy-count-one-too-few", dummies - 1 if dummies else None),
        ("dummy-count-two-too-many", dummies + 2),
    ]


def dummy_working(table, dummies):
    net_dummies = network_of(table)[1]
    return [
        "A dummy is needed wherever two activities share some, but not all, of their predecessors, or where two "
        "activities would otherwise start and finish at the same pair of events.",
        f"Checking each activity against the others, the fewest dummies that give the table's precedence is {dummies}.",
        f"The answer is {len(net_dummies)}." if len(net_dummies) == dummies else f"The answer is {dummies}.",
    ]


_COUNT_STEMS = (
    "A project has the activities below, with their durations in days. {net} The network diagram is drawn with each "
    "activity as an arrow and as few dummy activities as possible. How many dummy activities does it need?",
    "The activities of a small project, their durations and immediate predecessors are listed. {net} "
    "What is the smallest number of dummy activities with which this project can be drawn as an activity-on-arrow "
    "network?",
    "Below is the precedence table for a project. {net} Using as few dummy activities as possible, how many dummies "
    "must the network diagram contain?",
)
_TIME_STEMS = (
    "A project has the activities below. {net} Work starts at time 0. Dummy activities take no time. What is the "
    "critical time of the project, in days?",
    "The table lists each activity of a project with its duration and immediate predecessors. {net} "
    "Find the minimum time, in days, in which the whole project can be completed.",
    "A project is planned with the activities below, starting at time 0. {net} Find the length of the critical path, "
    "in days.",
)
_ARROW_STEMS = (
    "A project has the activities below. {net} The network diagram is drawn with as few dummy activities as possible. "
    "How many arrows, activities and dummies together, does the diagram have?",
    "The precedence table for a project is shown. {net} Counting every real activity and every dummy activity, "
    "how many arrows are in the diagram that uses the fewest dummies?",
)


def _count_question(rng, activities, dummies, stems):
    table = pick_table(rng, activities, dummies)
    dur = {a: rng.randint(_D_LO, _D_HI) for a in table}
    text = phrase(rng, stems, net=_network_text(table, dur))
    return number_question(text, dummies, 0, dummy_working(table, dummies), dummy_candidates(table, dummies))


def _easy(rng):
    activities = rng.choice([5, 5, 6])
    dummies = rng.choice([0, 1, 1, 2, 2] if activities == 5 else [1, 1, 2, 2, 3])
    return _count_question(rng, activities, dummies, _COUNT_STEMS)


def time_candidates(table, dur, finish, est, routes, lengths):
    longest = max(lengths.values())
    others = sorted({v for v in lengths.values() if v < longest}, reverse=True)
    eft = {a: est[a] + dur[a] for a in table}
    non_sink = [eft[a] for a in table if any(a in table[b] for b in table)]
    return [
        ("sums-all-durations", sum(dur.values())),
        ("takes-shorter-path", others[0] if others else None),
        ("stops-one-activity-early", max(non_sink) if non_sink else None),
    ]


def time_working(table, dur, finish, routes, lengths):
    best = [r for r in routes if lengths[r] == finish][0]
    return [
        f"Following every route through the table, the longest is {'-'.join(best)}.",
        f"Its length is {' + '.join(str(dur[a]) for a in best)} = {finish} days.",
    ]


def _time_question(rng, activities, dummies, close, stems):
    def draw(r):
        table = pick_table(r, activities, dummies)
        return table, {a: r.randint(_D_LO, _D_HI) for a in table}

    def accept(pair):
        table, dur = pair
        est, finish, _flt, routes, lengths = _analyse(table, dur)
        longest = [r for r in routes if lengths[r] == finish]
        if len(longest) != 1:
            return False
        rivals = [v for r, v in lengths.items() if r != longest[0]]
        gap = finish - max(rivals) if rivals else 99
        return (1 <= gap <= 2) if close else gap >= 1

    table, dur = draw_until(rng, draw, accept, tries=600, what="a network with a single critical route")
    est, finish, _flt, routes, lengths = _analyse(table, dur)
    check(finish == max(sum(dur[a] for a in r) for r in routes), "critical time disagrees with the route lengths")
    text = phrase(rng, stems, net=_network_text(table, dur))
    return number_question(
        text,
        finish,
        0,
        time_working(table, dur, finish, routes, lengths),
        time_candidates(table, dur, finish, est, routes, lengths),
    )


def _medium(rng):
    return _time_question(rng, rng.choice([6, 7]), rng.choice([1, 2]), False, _TIME_STEMS)


def _hard_count(rng):
    activities = rng.choice([7, 8])
    dummies = rng.choice([2, 3, 4] if activities == 7 else [2, 3, 4])
    return _count_question(rng, activities, dummies, _COUNT_STEMS)


def _hard_arrows(rng):
    activities = rng.choice([6, 7])
    dummies = rng.choice([2, 3] if activities == 6 else [2, 3, 4])
    table = pick_table(rng, activities, dummies)
    dur = {a: rng.randint(_D_LO, _D_HI) for a in table}
    text = phrase(rng, _ARROW_STEMS, net=_network_text(table, dur))
    total = activities + dummies
    working = [
        f"There are {activities} activities, and the fewest dummies that give this precedence is {dummies}.",
        f"Arrows: {activities} + {dummies} = {total}.",
    ]
    candidates = [
        ("counts-only-the-activities", activities),
        ("counts-only-the-dummies", dummies),
        ("arrow-count-one-too-many", total + 1),
        ("arrow-count-one-too-few", total - 1),
        ("counts-every-merge-as-a-dummy", activities + merges(table)),
    ]
    return number_question(text, total, 0, working, candidates)


def _hard_time(rng):
    return _time_question(rng, rng.choice([7, 8]), rng.choice([2, 3]), True, _TIME_STEMS)


@generator(
    "w5b_precedence_table",
    level=12,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def w5b_precedence_table(rng, tier="easy"):
    """Dummy activities and the critical time from a precedence table; the tiers are in the module docstring."""
    if tier == "easy":
        return _easy(rng)
    if tier == "medium":
        return _medium(rng)
    check(tier == "hard", f"unknown tier {tier!r}")
    return rng.choice([_hard_count, _hard_arrows, _hard_time])(rng)
