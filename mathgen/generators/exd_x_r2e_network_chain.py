"""Exam sources for the two activity-network skills `senior-use-ests-and-lsts-to-locate-the-critical` and
`senior-use-forward-and-backward-scanning-to-determine-the`: multi-part chains on one project network.

* ``exd_x_r2e_network_crash`` -- locate the critical path (an ordered list of activities, 2 marks), give the
  minimum completion time (1 mark), find the longest route that avoids one critical activity (2 marks), and
  from those say what the minimum completion time becomes when that activity is shortened (3 marks,
  ``follow_through`` from the first two numbers). Sometimes the shortening is used up in full and sometimes a
  rival route takes over, so the learner has to choose.
* ``exd_x_r2e_network_missing`` -- one duration is not given, but the earliest start of a later activity is.
  Recover the duration (2 marks), give the minimum completion time (2 marks, ``follow_through``), the latest
  start time of an activity (2 marks, ``follow_through``) and the total float of another (2 marks,
  ``follow_through``).

## Construction

Both are drawn by choosing durations and then checking the property that makes every key unique: a single
longest route, a strictly binding predecessor. Keys come from a forward and backward pass and are confirmed by
enumerating every start-to-finish route. Follow-through expressions are written with that enumeration: the
maximum, over the relevant routes, of the sum of the durations on the route, with the recovered duration
replaced by ``part_a``.
"""

from __future__ import annotations

import random

from ..answers import Distractor
from ..core import attempts, check, generator, latex_answer, pick_distractors

# Activity letters are in topological order, so a pass in alphabetical order is a valid pass.
_NETWORKS = (
    {  # nine activities, one start, branches that re-merge twice
        "A": [],
        "B": ["A"],
        "C": ["A"],
        "D": ["B"],
        "E": ["B", "C"],
        "F": ["C"],
        "G": ["D", "E"],
        "H": ["E", "F"],
        "I": ["G", "H"],
    },
    {  # eight activities, two starts, a side branch off D
        "A": [],
        "B": [],
        "C": ["A"],
        "D": ["A"],
        "E": ["B", "C"],
        "F": ["D", "E"],
        "G": ["D"],
        "H": ["F", "G"],
    },
    {  # eight activities, one start, two merges
        "A": [],
        "B": ["A"],
        "C": ["A"],
        "D": ["B", "C"],
        "E": ["C"],
        "F": ["D"],
        "G": ["D", "E"],
        "H": ["F", "G"],
    },
    {  # eight activities, three starts
        "A": [],
        "B": [],
        "C": [],
        "D": ["A", "B"],
        "E": ["B", "C"],
        "F": ["D"],
        "G": ["D", "E"],
        "H": ["F", "G"],
    },
)
_SETTINGS = [
    "A school is building a new library.",
    "A council is upgrading a skate park.",
    "A community group is setting up a weekend market.",
    "A theatre company is preparing a new production.",
    "A club is renovating its clubrooms.",
]
_D_LO, _D_HI = 2, 12



def _days(n) -> str:
    """"day" for one, "days" otherwise."""
    return "day" if n == 1 else "days"

def _successors(preds):
    succ = {a: [] for a in preds}
    for a, ps in preds.items():
        for p in ps:
            succ[p].append(a)
    return succ


def _routes(preds):
    """Every start-to-finish route, as tuples of activity letters."""
    succ = _successors(preds)
    out = []

    def walk(path):
        if not succ[path[-1]]:
            out.append(tuple(path))
            return
        for s in succ[path[-1]]:
            walk(path + [s])

    for a in preds:
        if not preds[a]:
            walk([a])
    return out


def _forward(preds, dur):
    est, eft = {}, {}
    for a in preds:
        est[a] = max((eft[p] for p in preds[a]), default=0)
        eft[a] = est[a] + dur[a]
    return est, eft


def _backward(preds, dur, finish):
    succ = _successors(preds)
    lst, lft = {}, {}
    for a in reversed(list(preds)):
        lft[a] = min((lst[s] for s in succ[a]), default=finish)
        lst[a] = lft[a] - dur[a]
    return lst, lft


def _length(route, dur):
    return sum(dur[a] for a in route)


def _network_text(preds, dur, hidden=None):
    lines = []
    for a in preds:
        d = "not known" if a == hidden else f"{dur[a]} {_days(dur[a])}"
        after = f"predecessors: {', '.join(preds[a])}" if preds[a] else "no predecessors"
        lines.append(f"Activity {a}: duration {d}, {after}.")
    return " ".join(lines)


def _path_text(route) -> str:
    return ", ".join(route)


def _tag(value: int, tag: str) -> Distractor:
    return Distractor(str(value), tag, value)


def _route_distractor(route, tag) -> Distractor:
    return Distractor(_path_text(route), tag, list(route))


def _max_expr(sums):
    return f"({sums[0]})" if len(sums) == 1 else "Max(" + ", ".join(sums) + ")"


def _sum_expr(route, dur, hidden=None):
    return " + ".join("part_a" if a == hidden else str(dur[a]) for a in route)


def _choose(rng):
    preds = rng.choice(_NETWORKS)
    dur = {a: rng.randint(_D_LO, _D_HI) for a in preds}
    return preds, dur


@generator("exd_x_r2e_network_crash", level=12, presets={"exam": dict(tier="exam")})
def network_crash(rng: random.Random, tier: str = "exam") -> dict:
    """Critical path, completion time, the rival route, and the completion time after one activity is shortened."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    binding = rng.random() < 0.5
    for _ in attempts(6000, "a network with a unique critical path and a clean rival route"):
        preds, dur = _choose(rng)
        routes = _routes(preds)
        lengths = {r: _length(r, dur) for r in routes}
        total = max(lengths.values())
        critical = [r for r in routes if lengths[r] == total]
        if len(critical) != 1:
            continue
        path = critical[0]
        y = rng.choice([a for a in path if dur[a] >= 3])
        rivals = [lengths[r] for r in routes if y not in r]
        if not rivals:
            continue
        rival = max(rivals)
        gap = total - rival
        r = rng.randint(2, dur[y] - 1)
        if binding != (gap < r):
            continue
        est, eft = _forward(preds, dur)
        check(max(eft.values()) == total, "forward pass disagrees with the longest route")
        new_total = max(rival, total - r)
        shortened = dict(dur)
        shortened[y] -= r
        check(max(_length(rt, shortened) for rt in routes) == new_total, "re-run over every route disagrees")

        ordered = sorted(routes, key=lambda rt: -lengths[rt])
        second = next(rt for rt in ordered if rt != path)
        shortest = ordered[-1]
        greedy = [a for a in preds if not preds[a]]
        start = max(greedy, key=lambda a: dur[a])
        walk = [start]
        succ = _successors(preds)
        while succ[walk[-1]]:
            walk.append(max(succ[walk[-1]], key=lambda a: dur[a]))
        a_cands = [
            _route_distractor(second, "takes-second-longest-route"),
            _route_distractor(tuple(walk), "follows-the-longest-activity-at-each-step"),
            _route_distractor(shortest, "takes-shortest-route"),
        ]
        a_cands = [c for c in a_cands if list(c.value) != list(path)]
        b_cands = [
            _tag(sum(dur.values()), "adds-every-duration"),
            _tag(lengths[second], "takes-second-longest-route"),
            _tag(lengths[shortest], "takes-shortest-route"),
        ]
        c_cands = [
            _tag(total, "ignores-the-activity-being-avoided"),
            _tag(total - dur[y], "removes-the-activity-from-the-critical-path"),
            _tag(lengths[second], "takes-second-longest-route"),
            _tag(min(rivals), "takes-shortest-route"),
        ]
        if binding:
            d_cands = [_tag(total - r, "ignores-the-rival-route"), _tag(total, "ignores-the-change")]
        else:
            d_cands = [_tag(rival, "assumes-the-rival-route-binds"), _tag(total, "ignores-the-change")]
        d_cands.append(_tag(total - dur[y], "removes-the-activity-from-the-critical-path"))
        picks = [
            pick_distractors(", ".join(path), a_cands, 3),
            pick_distractors(str(total), b_cands, 3),
            pick_distractors(str(rival), c_cands, 3),
            pick_distractors(str(new_total), d_cands, 3),
        ]
        if not all(len(p) >= 2 for p in picks):  # slips coincide with the answers: draw another network
            continue
        break

    stem = (
        f"{rng.choice(_SETTINGS)} The project starts at time 0. {_network_text(preds, dur)} "
        f"Extra workers are hired so that activity {y} takes {r} {_days(r)} less than planned. "
        f"Every other activity takes exactly its stated duration."
    )
    return dict(
        text=stem,
        parts=[
            dict(
                prompt="Write down the critical path of the original project, listing the activities in order, "
                "separated by commas.",
                answer_kind="ordered_list",
                answer=list(path),
                answer_text=_path_text(path),
                answer_latex=latex_answer(_path_text(path), plain=True),
                marks=2,
                working=[f"The longest route is {' - '.join(path)}, {total} {_days(total)}."],
                distractors=picks[0],
            ),
            dict(
                prompt="Find the minimum completion time of the original project, in days.",
                answer_kind="number",
                answer=total,
                answer_text=str(total),
                answer_latex=latex_answer(str(total)),
                marks=1,
                sensible_range=(0, None),
                working=[f"Sum of the durations on the critical path = {total} {_days(total)}."],
                distractors=picks[1],
            ),
            dict(
                prompt=f"Find the length of the longest route from start to finish that does not use activity {y}, "
                f"in days.",
                answer_kind="number",
                answer=rival,
                answer_text=str(rival),
                answer_latex=latex_answer(str(rival)),
                marks=2,
                sensible_range=(0, None),
                working=[f"Of the routes that avoid {y}, the longest is {rival} {_days(rival)}."],
                distractors=picks[2],
            ),
            dict(
                prompt=f"Find the minimum completion time of the project after activity {y} is shortened, in days.",
                answer_kind="number",
                answer=new_total,
                answer_text=str(new_total),
                answer_latex=latex_answer(str(new_total)),
                marks=3,
                sensible_range=(0, None),
                follow_through=f"Max(part_c, part_b - {r})",
                working=[
                    f"Shortening {y} by {r} {_days(r)} would give {total} - {r} = {total - r} {_days(total - r)}, "
                    f"but no route can finish before the longest route avoiding {y}, which is {rival} {_days(rival)}.",
                    f"New minimum time = max({rival}, {total - r}) = {new_total} {_days(new_total)}.",
                ],
                distractors=picks[3],
            ),
        ],
    )


@generator("exd_x_r2e_network_missing", level=12, presets={"exam": dict(tier="exam")})
def network_missing(rng: random.Random, tier: str = "exam") -> dict:
    """Recover a missing duration from a later start time, then use it for the project time, an LST and a float."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    for _ in attempts(8000, "a network with a recoverable duration and two usable activities"):
        preds, dur = _choose(rng)
        est, eft = _forward(preds, dur)
        # The hidden activity u must be the strictly binding predecessor of some later activity v.
        pairs = [
            (u, v)
            for v in preds
            for u in preds[v]
            if len(preds[v]) >= 2 and all(eft[u] > eft[o] for o in preds[v] if o != u)
        ]
        if not pairs:
            continue
        u, v = rng.choice(pairs)
        routes = _routes(preds)
        total = max(eft.values())
        lst, _lft = _backward(preds, dur, total)
        floats = {a: lst[a] - est[a] for a in preds}
        slack = [a for a in preds if floats[a] > 0 and a != u]
        others = [a for a in preds if a != u and a != v]
        if not slack:
            continue
        w = rng.choice(slack)
        zs = [a for a in others if a != w and lst[a] > 0]
        if not zs:
            continue
        z = rng.choice(zs)
        break
    start_v = est[v]
    d_u = dur[u]
    check(start_v == est[u] + d_u, "the binding predecessor must determine the start of v")

    succ = _successors(preds)

    def from_z(node):
        if not succ[node]:
            return [(node,)]
        return [(node, *rest) for s in succ[node] for rest in from_z(s)]

    z_routes = from_z(z)
    w_routes = [rt for rt in routes if w in rt]
    lst_z = total - max(_length(rt, dur) for rt in z_routes)
    check(lst_z == lst[z], "latest start from routes disagrees with the backward pass")
    float_w = total - max(_length(rt, dur) for rt in w_routes)
    check(float_w == floats[w] and float_w > 0, "float from routes disagrees with the passes")

    shortest_all = min(_length(rt, dur) for rt in routes)
    other_pred = max((o for o in preds[v] if o != u), key=lambda o: eft[o])
    without_u = dict(dur)
    without_u[u] = 0
    a_cands = [
        _tag(start_v, "reads-start-time-as-duration"),
        _tag(start_v - est[other_pred], "uses-the-wrong-predecessor"),
        _tag(est[u], "reads-start-time-as-duration"),
        _tag(d_u + 1, "off-by-one-count"),
        _tag(start_v - eft[other_pred], "uses-the-wrong-predecessor"),
        _tag(start_v + est[u], "adds-instead-of-subtracts"),
        _tag(d_u - 1, "off-by-one-count"),
    ]
    with_start = dict(dur)
    with_start[u] = start_v
    b_cands = [
        _tag(sum(dur.values()) - dur[u] + d_u, "adds-every-duration"),
        _tag(max(_length(rt, without_u) for rt in routes), "leaves-out-the-unknown-activity"),
        _tag(max(_length(rt, with_start) for rt in routes), "reads-start-time-as-duration"),
        _tag(shortest_all, "takes-shortest-route"),
    ]
    c_cands = [
        _tag(est[z], "reports-earliest-not-latest"),
        _tag(total - dur[z], "ignores-later-activities-in-backward-pass"),
        _tag(lst_z + 1, "off-by-one-count"),
        _tag(lst_z - 1, "off-by-one-count"),
    ]
    through_w = [_length(rt, dur) for rt in w_routes]
    d_cands = [
        _tag(lst[w], "reports-latest-start-as-float"),
        _tag(total - eft[w], "ignores-later-activities-in-backward-pass"),
        _tag(total - min(through_w), "takes-shortest-route"),
        _tag(float_w + 1, "off-by-one-count"),
    ]
    picks = [
        pick_distractors(str(d_u), [c for c in a_cands if c.value > 0], 3),
        pick_distractors(str(total), b_cands, 3),
        pick_distractors(str(lst_z), c_cands, 3),
        pick_distractors(str(float_w), d_cands, 3),
    ]
    check(all(len(p) >= 2 for p in picks), f"a part has too few distractors: {[len(p) for p in picks]}")
    # Follow-through expressions, written with this question's durations: the recovered duration is `part_a`.
    ft_total = _max_expr([_sum_expr(rt, dur, u) for rt in routes])
    ft_lst = f"part_b - {_max_expr([_sum_expr(rt, dur, u) for rt in z_routes])}"
    ft_float = f"part_b - {_max_expr([_sum_expr(rt, dur, u) for rt in w_routes])}"
    stem = (
        f"{rng.choice(_SETTINGS)} The project starts at time 0. {_network_text(preds, dur, hidden=u)} "
        f"The earliest start time of activity {v} is {start_v} {_days(start_v)}."
    )
    return dict(
        text=stem,
        parts=[
            dict(
                prompt=f"Find the duration of activity {u}, in days.",
                answer_kind="number",
                answer=d_u,
                answer_text=str(d_u),
                answer_latex=latex_answer(str(d_u)),
                marks=2,
                sensible_range=(0, None),
                working=[
                    f"Activity {v} cannot start until all its predecessors finish, and {u} finishes last.",
                    f"Duration of {u} = {start_v} - {est[u]} = {d_u} {_days(d_u)}.",
                ],
                distractors=picks[0],
            ),
            dict(
                prompt="Hence find the minimum completion time of the project, in days.",
                answer_kind="number",
                answer=total,
                answer_text=str(total),
                answer_latex=latex_answer(str(total)),
                marks=2,
                sensible_range=(0, None),
                follow_through=ft_total,
                working=[f"The longest route through the network takes {total} {_days(total)}."],
                distractors=picks[1],
            ),
            dict(
                prompt=f"Find the latest start time of activity {z} that does not delay the project, in days.",
                answer_kind="number",
                answer=lst_z,
                answer_text=str(lst_z),
                answer_latex=latex_answer(str(lst_z)),
                marks=2,
                follow_through=ft_lst,
                working=[f"Latest start of {z} = {total} - (longest route from {z} to the end) = {lst_z} {_days(lst_z)}."],
                distractors=picks[2],
            ),
            dict(
                prompt=f"Find the total float of activity {w}, in days.",
                answer_kind="number",
                answer=float_w,
                answer_text=str(float_w),
                answer_latex=latex_answer(str(float_w)),
                marks=2,
                sensible_range=(0, None),
                follow_through=ft_float,
                working=[f"Float of {w} = {total} - (longest route through {w}) = {float_w} {_days(float_w)}."],
                distractors=picks[3],
            ),
        ],
    )
