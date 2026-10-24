"""Generates `exd_x_r3b_critical_path`: a multi-part exam question on an activity network.

The network is one of the larger shapes the project-network exam tiers already use (eight or nine activities, two
or three merges). The chain is the one a senior exam sets:

(a) the minimum completion time of the project                    number, 2 marks
(b) the float of a non-critical activity                          number, 2 marks, part_a - (longest route through it)
(c) the new minimum completion time when a critical activity is
    shortened by more than the margin over its nearest rival      number, 2 marks, part_a - (saving)

The shortened activity is chosen so that the saving is smaller than the reduction (another route takes over),
which is the step a learner who just subtracts the reduction misses. Every key is computed by a forward and a
backward pass and confirmed by enumerating every start-to-finish route.
"""

from __future__ import annotations

import random

from ..core import attempts, check, generator, phrase
from .exd_x_g6_project_network import _TEMPLATES, _analyse, _forward, _length, _network_text, _routes
from .exd_x_r3b_common import number_part
from ._exd_distinct_parts import distinct_part_answers

_STEMS = (
    "A project has the activities below. The project starts at time 0. {net}",
    "The activities of a project, their durations and their predecessors are listed below. Work starts at time 0. {net}",
    "A building project is planned as the activities below, starting at time 0. {net}",
)


def _days(n) -> str:
    """ "day" for one, "days" otherwise."""
    return "day" if n == 1 else "days"


def _completion(preds, dur):
    return max(_forward(preds, dur)[1].values())


def _build(rng: random.Random) -> dict:
    preds = rng.choice(_TEMPLATES)
    for _ in attempts(4000, "a network with a unique critical route, a rival route and a non-critical activity"):
        dur = {a: rng.randint(2, 12) for a in preds}
        est, finish, flt, routes, lengths = _analyse(preds, dur)
        longest = [r for r in routes if lengths[r] == finish]
        if len(longest) != 1:
            continue
        critical = set(longest[0])
        free = [a for a in preds if flt[a] >= 2]
        if not free:
            continue
        options = []
        for y in sorted(critical):
            rivals = [lengths[r] for r in routes if y not in r]
            if not rivals or dur[y] < 4:
                continue
            gap = finish - max(rivals)
            if gap < 1 or gap > dur[y] - 3:
                continue
            options.append((y, gap))
        if options:
            break
    x = rng.choice(free)
    y, gap = rng.choice(options)
    reduction = rng.randint(gap + 1, dur[y] - 2)
    new_dur = dict(dur, **{y: dur[y] - reduction})
    new_finish = _completion(preds, new_dur)
    saving = finish - new_finish
    check(saving == gap and 0 < saving < reduction, "the saving is not the margin over the nearest rival route")
    through_x = max(lengths[r] for r in routes if x in r)
    check(flt[x] == finish - through_x, "float disagrees with the longest route through the activity")
    check(max(_length(r, new_dur) for r in _routes(preds)) == new_finish, "new completion time disagrees with routes")

    eft = _forward(preds, dur)[1]
    sink_time = max(eft[a] for a in preds if not any(a in ps for ps in preds.values()))
    check(sink_time == finish, "the latest sink does not finish at the completion time")
    ordered = sorted({lengths[r] for r in routes}, reverse=True)
    second = ordered[1] if len(ordered) > 1 else None
    non_sink = [eft[a] for a in preds if any(a in ps for ps in preds.values())]
    est_x = est[x]
    stem = phrase(rng, _STEMS, net=_network_text(preds, dur))
    part_a = number_part(
        "Find the minimum completion time of the project, in days.",
        finish,
        0,
        2,
        [f"The longest route is {'-'.join(longest[0])}, of length {finish} {_days(finish)}."],
        [
            ("sums-all-durations", sum(dur.values())),
            ("takes-shorter-path", second),
            ("stops-one-activity-early", max(non_sink) if non_sink else None),
        ],
        sensible_range=(0, None),
    )
    part_b = number_part(
        f"Find the float of activity {x}, in days (the number of days it can be delayed without delaying the project).",
        flt[x],
        0,
        2,
        [
            f"The longest route through {x} has length {through_x} {_days(through_x)}.",
            f"Float = {finish} - {through_x} = {flt[x]} {_days(flt[x])}.",
        ],
        [
            ("float-ignores-following-activities", finish - eft[x]),
            ("float-from-duration-only", finish - dur[x]),
            ("reported-earliest-start-as-float", est_x),
            ("float-sign-reversed", through_x - finish),
        ],
        follow_through=f"part_a - {through_x}",
    )
    part_c = number_part(
        f"Activity {y} is shortened by {reduction} {_days(reduction)}, to {dur[y] - reduction} {_days(dur[y] - reduction)}, and every other duration is "
        "unchanged. Find the new minimum completion time of the project, in days.",
        new_finish,
        0,
        2,
        [
            f"Another route is only {gap} day{'s' if gap != 1 else ''} shorter than the critical route.",
            f"The project is shortened by {saving} {_days(saving)}, to {new_finish} {_days(new_finish)}.",
        ],
        [
            ("crash-saving-taken-in-full", finish - reduction),
            ("ignored-the-reduction", finish),
            ("subtracted-overshoot-instead-of-saving", finish - (reduction - gap)),
        ],
        follow_through=f"part_a - {saving}",
    )
    return dict(text=stem, parts=[part_a, part_b, part_c])


@generator("exd_x_r3b_critical_path", level=12, presets={"exam": dict(tier="exam")})
@distinct_part_answers
def exd_x_r3b_critical_path(rng: random.Random, tier: str = "exam") -> dict:
    """An activity network: its minimum time, the float of a free activity, and the time after a crash."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    return _build(rng)
