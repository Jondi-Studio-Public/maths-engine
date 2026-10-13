"""Exam source for `senior-calculate-float-times-for-non-critical-activities`: float used, then a delay passed on.

The stem gives an activity network (durations in days and predecessors) and an activity X that is not critical,
delayed by t days (more than its float).

(a) the minimum completion time of the project (forward pass);
(b) the float of X, using (a): float = completion time - the longest route through X;
(c) by how many days the delay extends the project: t - float, using (b);
(d) the new minimum completion time: (a) plus (c).

Parts (b), (c) and (d) carry `follow_through`. The key is re-derived by listing every start-to-finish route, and the
delay is confirmed by running the forward pass again with X lengthened.
"""

from __future__ import annotations

from mathgen.core import attempts, check, generator

from .exd_x_r3c_parts import number_part
from ._exd_distinct_parts import distinct_part_answers

# Activities are lettered in topological order, so a pass in alphabetical order is a valid pass.
_TEMPLATES = (
    {"A": [], "B": [], "C": ["A"], "D": ["A", "B"], "E": ["B"], "F": ["C", "D"], "G": ["D", "E"], "H": ["F", "G"]},
    {
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
    {"A": [], "B": [], "C": ["A"], "D": ["A"], "E": ["B", "C"], "F": ["D"], "G": ["E", "F"], "H": ["E"]},
)
_INTROS = (
    "A building project is made up of the activities below and starts at time 0.",
    "The activities in a renovation, with their durations and predecessors, are listed below. The project starts at time 0.",
    "A school hall is to be refurbished in the stages below, starting at time 0.",
)


def _finish_times(preds, dur):
    eft = {}
    est = {}
    for a in preds:
        est[a] = max((eft[p] for p in preds[a]), default=0)
        eft[a] = est[a] + dur[a]
    return est, eft


def _routes(preds):
    succ = {a: [b for b in preds if a in preds[b]] for a in preds}
    out = []

    def walk(path):
        if not succ[path[-1]]:
            out.append(path)
            return
        for nxt in succ[path[-1]]:
            walk(path + [nxt])

    for a in preds:
        if not preds[a]:
            walk([a])
    return out


def _days(n):
    return f"{n} day" if n == 1 else f"{n} days"


def _network_text(preds, dur):
    lines = []
    for a in preds:
        tail = f"predecessors: {', '.join(preds[a])}." if preds[a] else "no predecessors."
        lines.append(f"Activity {a}: duration {dur[a]} days, {tail}")
    return " ".join(lines)


@generator("exd_x_r3c_network_float", level=14, presets={"exam": {"tier": "exam"}})
@distinct_part_answers
def exd_x_r3c_network_float(rng, tier="exam"):
    """Completion time, the float of a non-critical activity, the extension its delay causes, the new completion time."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    for _ in attempts(2000, "a network with a non-critical activity and distinct options"):
        preds = rng.choice(_TEMPLATES)
        dur = {a: rng.randint(2, 12) for a in preds}
        est, eft = _finish_times(preds, dur)
        finish = max(eft.values())
        routes = _routes(preds)
        lengths = {tuple(r): sum(dur[a] for a in r) for r in routes}
        check(max(lengths.values()) == finish, "the longest route is not the completion time")
        floats = {a: finish - max(v for r, v in lengths.items() if a in r) for a in preds}
        slack = [a for a in preds if floats[a] > 0]
        if not slack:
            continue
        x = rng.choice(slack)
        flt = floats[x]
        t = flt + rng.randint(1, 6)
        if t == 2 * flt:
            continue  # the float would equal the extension
        late = dict(dur, **{x: dur[x] + t})
        new_finish = max(_finish_times(preds, late)[1].values())
        extension = t - flt
        check(new_finish == finish + extension, "re-running the forward pass disagrees")
        through_x = finish - flt
        second = max((v for v in lengths.values() if v < finish), default=None)
        if second is None or second == through_x:
            continue
        # latest start of x: completion time less the longest route from x to the end (x included)
        after = max(sum(dur[a] for a in r[r.index(x) :]) for r in routes if x in r)
        lst = finish - after
        break

    stem = f"{rng.choice(_INTROS)} {_network_text(preds, dur)} Activity {x} takes {t} days longer than its stated duration, and every other activity takes exactly its stated duration."
    part_a = number_part(
        "Find the minimum completion time of the project with every activity taking its stated duration, in days.",
        finish,
        marks=2,
        sensible_range=(0, None),
        working=[
            f"The longest route through the network is {finish} days, so the minimum completion time is {finish} days."
        ],
        wrongs=[
            (sum(dur.values()), "sums-all-durations"),
            (second, "used-second-longest-route"),
            (through_x, "used-second-longest-route"),
        ],
    )
    part_b = number_part(
        f"Using your answer to (a), find the float time of activity {x}, in days.",
        flt,
        marks=2,
        sensible_range=(0, None),
        working=[
            f"The longest route through {x} is {through_x} days, so its float is {finish} - {through_x} = {_days(flt)}."
        ],
        wrongs=[
            (lst, "reports-lst-not-float"),
            (est[x], "reported-earliest-start-as-float"),
            (-flt, "reversed-subtraction"),
        ],
        follow_through=f"part_a - {through_x}",
    )
    part_c = number_part(
        f"Using your answer to (b), by how many days is the minimum completion time of the project extended by the longer time taken by activity {x}?",
        extension,
        sensible_range=(0, None),
        working=[
            f"The float of {_days(flt)} absorbs part of the {t}-day delay, so the project is {t} - {flt} = {_days(extension)} late."
        ],
        wrongs=[
            (t, "ignored-float-buffer"),
            (flt, "reported-float-not-extension"),
            (0, "no-delay-assumed"),
            (t + flt, "added-float-to-overrun"),
        ],
        follow_through=f"{t} - part_b",
    )
    part_d = number_part(
        "Using your answers to (a) and (c), find the new minimum completion time of the project, in days.",
        new_finish,
        sensible_range=(0, None),
        working=[f"{finish} + {extension} = {_days(new_finish)}."],
        wrongs=[
            (finish + t, "ignored-float-buffer"),
            (finish, "no-delay-assumed"),
            (finish + flt, "reported-float-not-extension"),
        ],
        follow_through="part_a + part_c",
    )
    return dict(text=stem, parts=[part_a, part_b, part_c, part_d])
