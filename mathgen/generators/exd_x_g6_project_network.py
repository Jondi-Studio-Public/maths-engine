"""Exam-tier questions on activity networks (critical path analysis).

Every earlier tier of the project-network skills uses a chain or a very small network and asks for ONE value
read straight off a forward or backward pass. The questions here are the kind a senior exam sets on the same
material: a larger network with several merges and many routes, and a question that needs the pass results to
be USED, not merely computed.

Four topics, one per skill, sharing a small network engine:

  exd_x_g6_delay_propagation        a non-critical activity overruns by t days: by how many days is the
                                    project late?  (total float used as a buffer, answer ``max(0, t - float)``)
  exd_x_g6_crash_shortest_time      a critical activity is shortened by r days: what is the new
                                    minimum completion time?  (the saving stops where a rival path takes over)
  exd_x_g6_missing_duration         one duration is missing but the earliest start of a later activity is
                                    given: recover the duration.  (a forward scan run in reverse)
  exd_x_g6_critical_path_after_change
                                    a critical activity is shortened: which route is the critical path now?
                                    (multiple choice; the stale answer is the old critical path)

Every question is drawn by choosing durations and then CHECKING the property that makes the key unique (a
strictly longest route, a strictly binding rival route, ...); the draw is repeated until it holds.  The key is
computed from a forward/backward pass and independently confirmed by enumerating every start-to-finish route.
"""

from __future__ import annotations

import random

from mathgen.core import check, check_distinct, check_equivalent, generator, latex_answer, pick_distractors

# Activity letters are in topological order, so a pass in alphabetical order is a valid pass.
_TEMPLATES = (
    {  # eight activities, two start points, two merge layers
        "A": [],
        "B": [],
        "C": ["A"],
        "D": ["A", "B"],
        "E": ["B"],
        "F": ["C", "D"],
        "G": ["D", "E"],
        "H": ["F", "G"],
    },
    {  # nine activities, one start point, branches that re-merge twice
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
    {  # eight activities, two start points, a late branch
        "A": [],
        "B": [],
        "C": ["A"],
        "D": ["A"],
        "E": ["B", "C"],
        "F": ["D"],
        "G": ["E", "F"],
        "H": ["E"],
    },
)
_D_LO, _D_HI = 2, 12
_TRIES = 4000


def _forward(preds, dur):
    est, eft = {}, {}
    for a in preds:
        est[a] = max((eft[p] for p in preds[a]), default=0)
        eft[a] = est[a] + dur[a]
    return est, eft


def _backward(preds, dur, finish):
    succ = {a: [] for a in preds}
    for a, ps in preds.items():
        for p in ps:
            succ[p].append(a)
    lst, lft = {}, {}
    for a in reversed(list(preds)):
        lft[a] = min((lst[s] for s in succ[a]), default=finish)
        lst[a] = lft[a] - dur[a]
    return lst, lft


def _routes(preds):
    """Every start-to-finish route, as tuples of activity letters (depth-first over successors)."""
    succ = {a: [] for a in preds}
    for a, ps in preds.items():
        for p in ps:
            succ[p].append(a)
    out = []

    def walk(path):
        nxt = succ[path[-1]]
        if not nxt:
            out.append(tuple(path))
            return
        for s in nxt:
            walk(path + [s])

    for a in preds:
        if not preds[a]:
            walk([a])
    return out


def _length(route, dur):
    return sum(dur[a] for a in route)


def _network_text(preds, dur, hidden=None):
    lines = []
    for a in preds:
        d = "unknown" if a == hidden else str(dur[a])
        unit = "" if a == hidden else " days"
        if preds[a]:
            lines.append(f"Activity {a}: duration {d}{unit}, predecessors: {', '.join(preds[a])}.")
        else:
            lines.append(f"Activity {a}: duration {d}{unit}, no predecessors.")
    return " ".join(lines)


def _draw(rng):
    preds = rng.choice(_TEMPLATES)
    dur = {a: rng.randint(_D_LO, _D_HI) for a in preds}
    return preds, dur


def _analyse(preds, dur):
    """Forward/backward pass, completion time, total float of every activity, and the route-enumeration
    cross-check that the completion time is the longest route and that float = completion - longest route
    through the activity."""
    est, eft = _forward(preds, dur)
    finish = max(eft.values())
    lst, _lft = _backward(preds, dur, finish)
    flt = {a: lst[a] - est[a] for a in preds}
    routes = _routes(preds)
    lengths = {r: _length(r, dur) for r in routes}
    check_equivalent(max(lengths.values()), finish, "longest route must equal the forward-pass completion time")
    for a in preds:
        through = max(lengths[r] for r in routes if a in r)
        check_equivalent(finish - through, flt[a], f"float of {a} must be completion time minus its longest route")
    return est, finish, flt, routes, lengths


def _result(text, answer, candidates, working):
    answer_text = str(answer)
    distractors = pick_distractors(answer_text, [str(c) for c in candidates])
    check(len(distractors) >= 2, "need at least two distractors")
    return dict(
        text=text,
        latex=rf"\text{{{text}}}",
        answer=answer,
        answer_text=answer_text,
        answer_latex=latex_answer(answer),
        working=working,
        distractors=distractors,
    )


# -- float skill: delay propagation ------------------------------------------------------------------------


@generator(
    "exd_x_g6_delay_propagation",
    level=12,
    presets={"exam": dict(tier="exam")},
)
def delay_propagation(rng: random.Random, tier: str = "exam") -> dict:
    """An activity with positive total float overruns by more than its float; report the project delay."""
    for _ in range(_TRIES):
        preds, dur = _draw(rng)
        _est, finish, flt, _routes_, _lengths = _analyse(preds, dur)
        slack = [a for a in preds if flt[a] > 0]
        if not slack:
            continue
        x = rng.choice(slack)
        t = flt[x] + rng.randint(1, 6)
        if t == 2 * flt[x]:
            continue  # would make the "reports the float" distractor equal the key
        delay = t - flt[x]
        # Second route: re-run the whole forward pass with the overrun applied.
        late = dict(dur)
        late[x] += t
        _e2, eft2 = _forward(preds, late)
        check_equivalent(max(eft2.values()) - finish, delay, "re-run forward pass must give the same delay")
        break
    else:
        raise RuntimeError("exd_x_g6_delay_propagation: no valid draw")
    text = (
        f"The project starts at time 0. {_network_text(preds, dur)} "
        f"Activity {x} overruns its stated duration by {t} days, and every other activity takes exactly its "
        f"stated duration. By how many days is the minimum completion time of the project extended?"
    )
    check_distinct(delay, t, flt[x], t + flt[x])
    return _result(
        text,
        delay,
        [t, flt[x], t + flt[x]],
        [
            f"Total float of {x} is {flt[x]} days; the overrun uses it up first, so the project is late by {t} - {flt[x]} = {delay} days."
        ],
    )


# -- minimum-time skill: crashing a critical activity -------------------------------------------------------


@generator(
    "exd_x_g6_crash_shortest_time",
    level=12,
    presets={"exam": dict(tier="exam")},
)
def crash_shortest_time(rng: random.Random, tier: str = "exam") -> dict:
    """A critical activity is shortened by r days; a rival route stops the saving short of r."""
    for _ in range(_TRIES):
        preds, dur = _draw(rng)
        _est, finish, flt, routes, lengths = _analyse(preds, dur)
        critical = [a for a in preds if flt[a] == 0 and dur[a] > 2]
        if not critical:
            continue
        x = rng.choice(critical)
        r = rng.randint(2, dur[x] - 1)
        crashed = dict(dur)
        crashed[x] -= r
        _e2, eft2 = _forward(preds, crashed)
        new_finish = max(eft2.values())
        saving = finish - new_finish
        if not 0 < saving < r:
            continue
        # Independent route: the longest route after the change, by enumeration.
        check_equivalent(max(_length(rt, crashed) for rt in routes), new_finish, "route enumeration disagrees")
        break
    else:
        raise RuntimeError("exd_x_g6_crash_shortest_time: no valid draw")
    text = (
        f"The project starts at time 0. {_network_text(preds, dur)} "
        f"Extra crew is hired so that activity {x} takes {r} days less than stated, and every other activity "
        f"is unchanged. What is the minimum completion time of the project now, in days?"
    )
    return _result(
        text,
        new_finish,
        [finish - r, finish, finish - dur[x]],
        [
            f"Completion time is {finish} days; shortening {x} by {r} days only saves {saving} days before another route governs, giving {new_finish}."
        ],
    )


# -- forward/backward scan skill: recover a missing duration -------------------------------------------------


@generator(
    "exd_x_g6_missing_duration",
    level=12,
    presets={"exam": dict(tier="exam")},
)
def missing_duration(rng: random.Random, tier: str = "exam") -> dict:
    """One duration is withheld and the earliest start of a later merge activity is given; recover it.

    The withheld activity must lie on the route that STRICTLY governs the merge's earliest start, otherwise
    the given start time would not determine it."""
    for _ in range(_TRIES):
        preds, dur = _draw(rng)
        est, _eft = _forward(preds, dur)
        merges = [a for a in preds if len(preds[a]) >= 2]
        m = rng.choice(merges)
        u = rng.choice([a for a in preds if a < m and a != m])
        # routes from a start point to the predecessors of m (activities strictly before m on a route to m)
        before = [r[: r.index(m)] for r in _routes(preds) if m in r and r.index(m) > 0]
        with_u = [r for r in before if u in r]
        without_u = [r for r in before if u not in r]
        if not with_u:
            continue
        target = est[m]
        best_with = max(_length(r, dur) for r in with_u)
        best_without = max((_length(r, dur) for r in without_u), default=0)
        if not best_with > best_without:
            continue  # u's routes must strictly govern the start of m
        if best_with != target:
            continue
        # Uniqueness: exactly one duration for u reproduces the given start time.
        hits = []
        for guess in range(1, 40):
            trial = dict(dur)
            trial[u] = guess
            if _forward(preds, trial)[0][m] == target:
                hits.append(guess)
        check(hits == [dur[u]], "the withheld duration must be uniquely determined")
        rest = best_with - dur[u]
        governing = max((r for r in with_u if _length(r, dur) == best_with), key=len)
        others = [dur[a] for a in governing if a != u]
        if not others:
            continue
        break
    else:
        raise RuntimeError("exd_x_g6_missing_duration: no valid draw")
    answer = dur[u]
    x = rng.choice(others)
    other_route = target - best_without
    check_equivalent(target - rest, answer, "duration must be start time minus the rest of the governing route")
    text = (
        f"The project starts at time 0. {_network_text(preds, dur, hidden=u)} "
        f"The earliest start time of activity {m} is day {target}. "
        f"Find the duration of activity {u}, in days."
    )
    return _result(
        text,
        answer,
        [answer + x, other_route, answer - x],
        [
            f"Activity {m} cannot start before the governing route to it is complete; that route is {target} days, of which {rest} are known, so {u} takes {answer} days."
        ],
    )


# -- locate skill: the critical path after a change (multiple choice) -----------------------------------------


def _route_text(route):
    return "-".join(route)


@generator(
    "exd_x_g6_critical_path_after_change",
    level=12,
    presets={"exam": dict(tier="exam")},
)
def critical_path_after_change(rng: random.Random, tier: str = "exam") -> dict:
    """A critical activity is shortened so far that a different route becomes the only critical path."""
    for _ in range(_TRIES):
        preds, dur = _draw(rng)
        _est, finish, flt, routes, lengths = _analyse(preds, dur)
        ranked = sorted(routes, key=lambda r: -lengths[r])
        if lengths[ranked[0]] == lengths[ranked[1]]:
            continue  # the original critical path must be unique
        old = ranked[0]
        x = rng.choice(old)
        if dur[x] <= 2:
            continue
        r = rng.randint(2, dur[x] - 1)
        crashed = dict(dur)
        crashed[x] -= r
        new_lengths = {rt: _length(rt, crashed) for rt in routes}
        order = sorted(routes, key=lambda rt: -new_lengths[rt])
        if new_lengths[order[0]] == new_lengths[order[1]]:
            continue  # the new critical path must be unique
        new = order[0]
        if new == old or x in new:
            continue
        rest = [rt for rt in ranked if rt not in (old, new)]
        if len(rest) < 2:
            continue
        break
    else:
        raise RuntimeError("exd_x_g6_critical_path_after_change: no valid draw")
    # Independent route: the new critical path is the route with the largest forward-pass finish.
    est2, eft2 = _forward(preds, crashed)
    check_equivalent(max(eft2.values()), new_lengths[new], "new completion time must equal the new longest route")
    answer = _route_text(new)
    candidates = [_route_text(old)] + [_route_text(rt) for rt in rest]
    distractors = pick_distractors(answer, candidates, n=3)
    check(len(distractors) >= 2, "need at least two distractors")
    text = (
        f"The project starts at time 0. {_network_text(preds, dur)} "
        f"Activity {x} is shortened by {r} days. Which one of the following routes is now the only critical path?"
    )
    return dict(
        text=text,
        latex=rf"\text{{{text}}}",
        answer=answer,
        answer_text=answer,
        answer_latex=latex_answer(answer, plain=True),
        working=[
            f"After shortening {x}, route {answer} is the longest at {new_lengths[new]} days; the old critical path {_route_text(old)} is now {new_lengths[old]} days."
        ],
        distractors=distractors,
    )
