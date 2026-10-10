"""Generates `p19i_timetable_latest_service` and `p19i_timetable_condition_lookup` questions: cross-table lookups
in a stated timetable.

Two ``answer: mcq`` generators, each a genuine cross-table lookup -- never a
single-cell read and never a duration between two named times (that is the
shipping neighbour ``f10-timetable-itinerary-duration``,
``mathgen/generators/p4b_timetable_build.py``, which computes a duration for
ONE journey and never compares across several services):

- ``p19i_timetable_latest_service`` (senior-interpret-timetables-such-as-bus-
  train-and-ferry): given a stated timetable of several bus/train/ferry
  services (each a departure from stop A and an arrival at stop B), choose
  from four offered departure times the LATEST service that still arrives by
  a stated deadline. The table's LISTED (printed) order is shuffled
  independently of departure-time order, so "read the last row" is never a
  shortcut.
- ``p19i_timetable_condition_lookup`` (senior-interpret-complex-timetables-
  such-as-tide-charts-sunrise): given a stated tide/sunrise/moon-phase table
  for several days, choose from four offered day labels the one meeting a
  stated condition (a superlative height/time, or the first full moon after
  a date).

**Boundary against each other**: Skill A's rows are bus/train/ferry services
with departure/arrival TIMES; the condition is always "latest service
meeting an arrival deadline". Skill B's rows are tide/sunrise/moon-phase
readings across several DAYS; the condition varies by table type (highest
tide, latest sunrise, first full moon after a date). Never interchangeable:
Skill A always compares departure times against one deadline; Skill B always
compares table entries against a table-type-specific superlative or date
filter.

**Key position is never fixed.** Both generators shuffle the table's printed
row order independently of the correct answer's structural role, so the
correct answer's row/column position varies across draws (measured at 200
seeds in the test file, never concentrated at first/last).

Construct-to-target throughout: each tier's structural roles are decided FIRST, then the stated times/values
are built to realise them, and an in-generator ``check`` re-derives the answer from the constructed rows a
second way before it is offered.
"""

from __future__ import annotations

from ..core import check, check_pairwise_distinct, generator, latex_answer, phrase, VerificationError

_RETRY_LIMIT = 200


def _retry(rng, builder, label: str):
    """Some draws collide by chance (two independently-drawn times landing
    on the same minute); retry the whole construction rather than fail the
    build. Each attempt consumes more of the same ``rng`` stream, so the
    result stays fully determined by (topic, tier, seed)."""
    last_error = None
    for _ in range(_RETRY_LIMIT):
        try:
            return builder(rng)
        except VerificationError as exc:
            last_error = exc
            continue
    raise RuntimeError(f"could not construct a collision-free {label} after {_RETRY_LIMIT} attempts") from last_error


# =========================================================================
# Skill A: p19i_timetable_latest_service
# =========================================================================

MODES = ["bus", "train", "boat"]
DURATION_RANGE = {"bus": (15, 40), "train": (20, 60), "boat": (30, 90)}
STOP_PAIRS = [
    ("Riverside", "Central"),
    ("Northgate", "Harbour"),
    ("Eastview", "Docklands"),
    ("Westfield", "University"),
    ("Southbank", "Airport"),
]


def _fmt_time(minutes: int) -> str:
    minutes %= 24 * 60
    h, m = divmod(minutes, 60)
    return f"{h:02d}:{m:02d}"


def _duration_base(rng, mode: str) -> int:
    lo, hi = DURATION_RANGE[mode]
    return rng.randint(lo, hi)


def _duration_plan(rng) -> dict:
    """One base journey-time draw per mode for THIS table. Previously every row drew its own duration
    independently from `DURATION_RANGE`, so two rows of the SAME mode on the SAME route within one question
    could land up to ~77-80 minutes apart -- implausible for a single route. Every row of a given mode in one
    question now uses this table's one base value for that mode, plus `_DURATION_JITTER` minutes of jitter."""
    return {mode: _duration_base(rng, mode) for mode in MODES}


_DURATION_JITTER = 5  # +/- minutes: same mode, same route stays within a 10-minute band


def _duration(rng, mode: str, plan: dict) -> int:
    lo, hi = DURATION_RANGE[mode]
    value = plan[mode] + rng.randint(-_DURATION_JITTER, _DURATION_JITTER)
    return max(lo, min(hi, value))


def _clean_deadline(rng, lo_h=7, hi_h=11) -> int:
    return rng.randint(lo_h, hi_h) * 60 + rng.choice([0, 15, 30, 45])


def _messy_deadline(rng, lo_h=7, hi_h=11) -> int:
    minute = rng.choice([m for m in range(60) if m % 5 != 0])
    return rng.randint(lo_h, hi_h) * 60 + minute


def _build_correct(rng, deadline: int, modes_pool: list[str], plan: dict) -> dict:
    """The correct (latest in-time) service. Never draw its arrival exactly AT the deadline -- always a genuine
    margin before it."""
    mode_c = rng.choice(modes_pool)
    correct_arr = deadline - rng.randint(3, 20)
    correct_dep = correct_arr - _duration(rng, mode_c, plan)
    return {"mode": mode_c, "dep": correct_dep, "arr": correct_arr}


def _build_later_rows(rng, count: int, base_dep: int, deadline: int, modes_pool: list[str], plan: dict) -> list[dict]:
    """`count` rows that depart progressively later than `base_dep` and all
    miss the deadline. Each row's journey time is drawn from the
    table's per-mode `plan`, not independently."""
    rows = []
    prev_dep = base_dep
    for _ in range(count):
        mode = rng.choice(modes_pool)
        dep = arr = None
        for _ in range(_RETRY_LIMIT):
            candidate_dep = prev_dep + rng.randint(15, 60)
            candidate_arr = candidate_dep + _duration(rng, mode, plan)
            if candidate_arr - deadline >= 5:
                dep, arr = candidate_dep, candidate_arr
                break
        check(dep is not None, "could not construct a too-late later service")
        rows.append({"mode": mode, "dep": dep, "arr": arr})
        prev_dep = dep
    return rows


def _build_earlier_rows(rng, count: int, base_arr: int, modes_pool: list[str], plan: dict) -> list[dict]:
    """`count` rows that depart (and arrive) progressively earlier than
    `base_arr`, each comfortably in time -- irrelevant to which row is
    "latest", just filling out the table. Journey times come from the same
    table's `plan` as every other row."""
    rows = []
    prev_arr = base_arr
    for _ in range(count):
        mode = rng.choice(modes_pool)
        arr = prev_arr - rng.randint(20, 60)
        dep = arr - _duration(rng, mode, plan)
        rows.append({"mode": mode, "dep": dep, "arr": arr})
        prev_arr = arr
    return rows


def _independent_latest_in_time(rows: list[dict], deadline: int, disqualified_mode: str | None) -> int:
    """In-generator independent route: recompute the latest in-time
    departure from the constructed rows directly (never trusting which row
    construction *labelled* "correct"), applying the mode filter if any."""
    candidates = [
        r["dep"] for r in rows if r["arr"] <= deadline and (disqualified_mode is None or r["mode"] != disqualified_mode)
    ]
    check(candidates, "no in-time service survives the deadline/constraint filter")
    return max(candidates)


def _assemble(
    rng,
    roles: dict,
    fillers: list[dict],
    deadline: int,
    offered_roles: list[str],
    constraint_mode: str | None = None,
    constraint_phrasing: str | None = None,
) -> dict:
    all_rows = [dict(role=role, **data) for role, data in roles.items()] + [dict(role="filler", **f) for f in fillers]

    deps = [r["dep"] for r in all_rows]
    check_pairwise_distinct(*deps, msg="two services share a departure time")
    arrs = [r["arr"] for r in all_rows]
    check_pairwise_distinct(*arrs, msg="two services share an arrival time")

    independent_answer = _independent_latest_in_time(all_rows, deadline, constraint_mode)
    check(
        independent_answer == roles["correct"]["dep"],
        "the labelled 'correct' role is not the independently-recomputed latest in-time service",
    )

    # "reaches Y by HH:MM" is ambiguous when a service's arrival lands EXACTLY on the deadline (a
    # learner reading the wording exclusively would defensibly reject it). Never draw a table with
    # such a tie -- reject-and-redraw so the correct answer always clears the deadline with a
    # genuine margin.
    check(
        all(r["arr"] != deadline for r in all_rows),
        "a service arrives exactly at the deadline -- inclusive/exclusive ambiguity",
    )

    # Shuffle print order. Dropped the old forced-adjacency swap (which always printed one specific
    # row directly next to "correct") -- it structurally telegraphed which row was "correct" once a
    # learner noticed the pattern, and it conflicts with randomising the number of later-departing
    # disqualified rows (there is no longer one single "adjacent" role to swap into place). Print
    # order is now a plain shuffle of every row.
    order = list(range(len(all_rows)))
    rng.shuffle(order)
    shuffled = [all_rows[i] for i in order]
    correct_idx = next(i for i, r in enumerate(shuffled) if r["role"] == "correct")

    stop_a, stop_b = rng.choice(STOP_PAIRS)
    counters = {"bus": 0, "train": 0, "boat": 0}
    row_sentences = []
    correct_label = None
    for r in shuffled:
        counters[r["mode"]] += 1
        label = f"{r['mode'].capitalize()} {counters[r['mode']]}"
        r["label"] = label
        if r["role"] == "correct":
            correct_label = label
        row_sentences.append(
            f"{label} departs {stop_a} at {_fmt_time(r['dep'])} and arrives at {stop_b} at {_fmt_time(r['arr'])}."
        )

    intro = "All times are shown in 24-hour format."
    constraint_sentence = f" {constraint_phrasing}" if constraint_phrasing else ""
    question = (
        f" Which of the following departures from {stop_a} is the latest service that "
        f"still reaches {stop_b} by {_fmt_time(deadline)}?"
    )
    text = intro + constraint_sentence + " " + " ".join(row_sentences) + question

    answer_dep = roles["correct"]["dep"]
    answer_text = _fmt_time(answer_dep)
    distractor_deps = [roles[role]["dep"] for role in offered_roles if role != "correct"]
    check_pairwise_distinct(answer_dep, *distractor_deps, msg="an offered option collides with the answer")
    distractors = [_fmt_time(d) for d in distractor_deps]

    working = [
        f"{correct_label} departs {stop_a} at {answer_text} and arrives at {stop_b} "
        f"at {_fmt_time(roles['correct']['arr'])}, which is before the {_fmt_time(deadline)} "
        "deadline -- and no later-departing service also meets it."
    ]

    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=answer_text,
        answer_text=answer_text,
        answer_latex=latex_answer(answer_text, plain=True),
        working=working,
        distractors=distractors,
    )


# The number of later-departing, disqualified rows (the "rank" of the correct answer among all rows
# sorted by departure time descending is exactly this count + 1) is drawn per seed from these small
# per-tier ranges, INCLUDING 0 -- a genuine share of draws where "correct" really is the latest
# departure in the whole table. Measured realised distributions are quoted
_EASY_TOTAL_EXTRA = 3
_EASY_LATER_CHOICES = [0, 1, 2]
_MEDIUM_TOTAL_EXTRA = 4
_MEDIUM_LATER_CHOICES = [0, 1, 2, 3]
_HARD_TOTAL_EXTRA = 5
_HARD_LATER_CHOICES = [0, 1, 2, 3]


def _build_A_easy(rng):
    mode = rng.choice(MODES)
    modes_pool = [mode]
    plan = _duration_plan(rng)
    deadline = _clean_deadline(rng)
    correct = _build_correct(rng, deadline, modes_pool, plan)
    k_later = rng.choice(_EASY_LATER_CHOICES)
    later_rows = _build_later_rows(rng, k_later, correct["dep"], deadline, modes_pool, plan)
    earlier_rows = _build_earlier_rows(rng, _EASY_TOTAL_EXTRA - k_later, correct["arr"], modes_pool, plan)
    extra_rows = later_rows + earlier_rows

    roles = {"correct": correct}
    for i, row in enumerate(extra_rows):
        roles[f"x{i}"] = row
    offered_roles = ["correct"] + [f"x{i}" for i in range(len(extra_rows))]
    return _assemble(rng, roles, [], deadline, offered_roles)


def _build_A_medium(rng):
    mode = rng.choice(MODES)
    modes_pool = [mode]
    plan = _duration_plan(rng)
    deadline = _clean_deadline(rng, 7, 12)
    correct = _build_correct(rng, deadline, modes_pool, plan)
    k_later = rng.choice(_MEDIUM_LATER_CHOICES)
    later_rows = _build_later_rows(rng, k_later, correct["dep"], deadline, modes_pool, plan)
    earlier_rows = _build_earlier_rows(rng, _MEDIUM_TOTAL_EXTRA - k_later, correct["arr"], modes_pool, plan)
    extra_rows = later_rows + earlier_rows

    roles = {"correct": correct}
    for i, row in enumerate(extra_rows[:3]):
        roles[f"x{i}"] = row
    fillers = extra_rows[3:]
    offered_roles = ["correct", "x0", "x1", "x2"]
    return _assemble(rng, roles, fillers, deadline, offered_roles)


def _build_A_hard(rng):
    modes_pool = rng.sample(MODES, 2)
    plan = _duration_plan(rng)
    deadline = _messy_deadline(rng, 7, 12)
    correct = _build_correct(rng, deadline, modes_pool, plan)
    k_later = rng.choice(_HARD_LATER_CHOICES)
    later_rows = _build_later_rows(rng, k_later, correct["dep"], deadline, modes_pool, plan)
    earlier_rows = _build_earlier_rows(rng, _HARD_TOTAL_EXTRA - k_later, correct["arr"], modes_pool, plan)
    extra_rows = later_rows + earlier_rows

    # "6 services across TWO modes mixed" requires BOTH modes to actually
    # appear -- independent rng.choice draws across correct + extra rows can
    # by chance all land on the same mode. Force coverage: if only one mode
    # is present, recompute the LAST extra row in the missing mode. With
    # `_HARD_TOTAL_EXTRA = 5` and `k_later` capped at 3, `earlier_rows`
    # always has >= 2 entries, so `extra_rows[-1]` is always an earlier row
    # here -- recompute its departure from its unchanged arrival and the
    # new mode's duration. (The `target not in later_rows` guard is kept
    # for safety in case that invariant ever changes; a later row's
    # lateness is defined purely by dep/arr and would need no recompute.)
    modes_used = {correct["mode"]} | {r["mode"] for r in extra_rows}
    if len(modes_used) < 2:
        missing_mode = next(m for m in modes_pool if m not in modes_used)
        target = extra_rows[-1]
        target["mode"] = missing_mode
        if target not in later_rows:
            target["dep"] = target["arr"] - _duration(rng, missing_mode, plan)

    roles = {"correct": correct}
    for i, row in enumerate(extra_rows[:3]):
        roles[f"x{i}"] = row
    fillers = extra_rows[3:]
    offered_roles = ["correct", "x0", "x1", "x2"]
    return _assemble(rng, roles, fillers, deadline, offered_roles)


_CANCEL_PHRASINGS = [
    "{Mode} services are cancelled today.",
    "{Mode} services are not running today.",
    "Only {other} services are running today.",
]


# Exam always needs its mode-disqualified "disq" row (the row that WOULD be the correct answer if
# the stated cancellation didn't apply -- the misconception the tier exists to exercise). The
# deadline must also BIND among the services the cancellation leaves running, so the rank of the key
# among the mode-KEPT services (latest departure first) is drawn per seed from 1..4 -- rank r means
# r-1 kept services depart later and miss the deadline. (Previously the later rows were a random mix
# of both modes and the key was the latest kept service in 50% of rows, so the deadline rarely
# mattered and "pick the earliest option" won ~50%.) The four extra rows are shuffled before two of
# them are offered, so offered options mix earlier and later departures.
_EXAM_TOTAL_EXTRAS = 4
_EXAM_OPTION_RANKS = [0, 1, 2, 3]


def _build_A_exam(rng):
    mode_keep, mode_disq = rng.sample(MODES, 2)
    modes_pool = [mode_keep, mode_disq]
    deadline = _messy_deadline(rng, 7, 12)
    # The key's rank AMONG THE FOUR OFFERED OPTIONS (by departure, 0 = the key is the latest offered
    # departure, 3 = the earliest) is drawn uniformly per seed. Rank r means r offered options
    # depart later than the key: the disqualified-mode row (always later than the key, in time but
    # for the cancellation) plus r-1 kept-mode rows that miss the deadline; r = 0 offers no later
    # option, so the disq row stays in the table unoffered. The rest are earlier in-time rows. The
    # count of kept-later rows is drawn from the range that can supply both counts.
    key_rank = rng.choice(_EXAM_OPTION_RANKS)
    k_kept_later = rng.choice(range(max(0, key_rank - 1), min(3, key_rank + 1) + 1))

    for _ in range(_RETRY_LIMIT):
        # Redraw the duration plan every attempt: when mode_disq's base
        # duration happens to run much longer than mode_keep's,
        # `disq_dep <= correct_dep` below is a structural near-certainty for
        # that one plan, and retrying with a fixed plan can never escape it.
        plan = _duration_plan(rng)
        correct_arr = deadline - rng.randint(15, 30)
        correct_dep = correct_arr - _duration(rng, mode_keep, plan)

        # Never let the disqualified-mode arrival land exactly on the deadline either. disq_dep >
        # correct_dep is required so "disq" is itself a later-departing, in-time-but-for-the-mode
        # row -- the row a learner who ignores the stated constraint would (wrongly) pick.
        disq_arr = deadline - rng.randint(2, 10)
        disq_dep = disq_arr - _duration(rng, mode_disq, plan)
        if disq_dep <= correct_dep:
            continue

        try:
            kept_later_rows = _build_later_rows(rng, k_kept_later, correct_dep, deadline, [mode_keep], plan)
        except VerificationError:
            continue
        earlier_rows = _build_earlier_rows(rng, _EXAM_TOTAL_EXTRAS - k_kept_later, correct_arr, modes_pool, plan)
        break
    else:
        raise RuntimeError("could not construct exam-tier timetable with a disqualified winner")

    correct = {"mode": mode_keep, "dep": correct_dep, "arr": correct_arr}
    disq = {"mode": mode_disq, "dep": disq_dep, "arr": disq_arr}
    kept_later_rows = list(kept_later_rows)
    earlier_rows = list(earlier_rows)
    rng.shuffle(kept_later_rows)
    rng.shuffle(earlier_rows)
    n_kept_offered = max(0, key_rank - 1)
    n_earlier_offered = 3 - key_rank
    offered_rows = kept_later_rows[:n_kept_offered] + earlier_rows[:n_earlier_offered]
    fillers = kept_later_rows[n_kept_offered:] + earlier_rows[n_earlier_offered:]
    check(
        len(offered_rows) == (3 if key_rank == 0 else 2),
        "not enough kept-later / earlier rows to realise the drawn key rank",
    )

    roles = {"correct": correct, "disq": disq}
    for i, row in enumerate(offered_rows):
        roles[f"x{i}"] = row
    offered_roles = ["correct"] + (["disq"] if key_rank >= 1 else []) + [f"x{i}" for i in range(len(offered_rows))]

    phrasing = rng.choice(_CANCEL_PHRASINGS)
    if "{other}" in phrasing:
        constraint_phrasing = phrasing.format(other=mode_keep.capitalize())
    else:
        constraint_phrasing = phrasing.format(Mode=mode_disq.capitalize())

    return _assemble(
        rng, roles, fillers, deadline, offered_roles, constraint_mode=mode_disq, constraint_phrasing=constraint_phrasing
    )


_LATEST_SERVICE_BUILDERS = {
    "easy": _build_A_easy,
    "medium": _build_A_medium,
    "hard": _build_A_hard,
    "exam": _build_A_exam,
}


@generator(
    "p19i_timetable_latest_service",
    level=9,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
        "exam": dict(tier="exam"),
    },
)
def p19i_timetable_latest_service(rng, tier="easy"):
    """Given a stated timetable of several bus/train/ferry services (each a
    departure from a named stop A and an arrival at a named stop B), choose
    from four offered departure times the LATEST service that still arrives
    at B by a stated deadline. Boundary: never a duration between two named
    times for one journey (that is ``f10-timetable-itinerary-duration``) --
    the table's printed row order is always decoupled from departure-time
    order."""
    builder = _LATEST_SERVICE_BUILDERS.get(tier)
    if builder is None:
        raise ValueError(f"unknown tier {tier!r}; expected one of {sorted(_LATEST_SERVICE_BUILDERS)}")
    return _retry(rng, builder, f"p19i_timetable_latest_service/{tier}")


# =========================================================================
# Skill B: p19i_timetable_condition_lookup
# =========================================================================

WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
WEEKEND = {"Saturday", "Sunday"}


def _ordinal(n: int) -> str:
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


_TIDE_MIN_TENTHS, _TIDE_MAX_TENTHS = 10, 30  # every printed tide height is 1.0-3.0 m


def _draw_extreme_values(
    rng, n: int, lo: int, hi: int, gap_lo: int, gap_hi: int, target_index: int, direction: str
) -> list[int]:
    """n distinct integer values in [lo, hi] with the value at ``target_index``
    a clean extreme (max if direction == "high", min if "low"), at least
    ``gap_lo`` and at most ``gap_hi`` clear of the runner-up -- never a tie.
    The gap is drawn first and the other values are sampled so that the
    extreme cannot leave [lo, hi]."""
    for _ in range(_RETRY_LIMIT):
        gap = rng.randint(gap_lo, gap_hi)
        if direction == "high":
            others = rng.sample(range(lo, hi - gap + 1), n - 1)
            extreme = max(others) + gap
        else:
            others = rng.sample(range(lo + gap, hi + 1), n - 1)
            extreme = min(others) - gap
        values = others[:]
        values.insert(target_index, extreme)
        if len(set(values)) == n and all(lo <= v <= hi for v in values):
            return values
    raise RuntimeError("could not draw a distinct extreme-value table")


def _drift_sequence(rng, n: int, start: int, step_lo: int, step_hi: int, signs: list[int]) -> list[int]:
    """n values starting at ``start`` where consecutive values differ by
    ``rng.randint(step_lo, step_hi)`` in the direction ``signs[j]`` (+1/-1)
    for step j. Every step is non-zero, so a run of one sign never ties."""
    values = [start]
    for j in range(n - 1):
        values.append(values[-1] + signs[j] * rng.randint(step_lo, step_hi))
    return values


def _sunrise_path(rng, n: int, step_lo: int, step_hi: int, turn: int, first_sign: int) -> list[int]:
    """Sunrise times (minutes after midnight) for ``n`` printed days, physically
    shaped: sunrise drifts steadily one way and only turns around
    a solstice, where the curve is FLAT.

    Steps before index ``turn`` move in ``first_sign``, steps from ``turn`` on
    move the other way. ``turn`` of 0 or ``n - 1`` is a monotone run (steps
    ``step_lo``..``step_hi`` each). An interior turn is flat: the step into
    the turn and the step out of it are 1 and 2 minutes (either way round, so
    the turning day's two neighbours never tie), magnitudes never grow on the
    approach and never shrink on the way out (e.g. 3, 2, 1 | 2, 3)."""
    if turn in (0, n - 1):
        sign = first_sign if turn == n - 1 else -first_sign
        mags = [rng.randint(step_lo, step_hi) for _ in range(n - 1)]
        steps = [sign * m for m in mags]
    else:
        # Exactly one side of the turn is the 1-minute flat step; the other is
        # 2 (or the two neighbours of the turning day would tie).
        if rng.random() < 0.5:
            approach = sorted((rng.randint(1, 3) for _ in range(turn - 1)), reverse=True) + [1]
            leave = [2] + sorted(rng.randint(2, 3) for _ in range(n - 2 - turn))
        else:
            approach = sorted((rng.randint(2, 3) for _ in range(turn - 1)), reverse=True) + [2]
            leave = [1] + sorted(rng.randint(1, 3) for _ in range(n - 2 - turn))
        steps = [first_sign * m for m in approach] + [-first_sign * m for m in leave]
    values = [rng.randint(340, 400)]
    for st in steps:
        values.append(values[-1] + st)
    return values


def _tide_times(rng, n: int) -> list[int]:
    """High-tide times one per printed day. A tidal day is about 24 h 50 min,
    so each high tide is later than the previous by about 50 minutes (46-54
    here), wrapping past midnight; the wrap never collides over <= 7 days."""
    times = _drift_sequence(rng, n, rng.randrange(0, 24 * 60, 5), 46, 54, [1] * (n - 1))
    check(len({t % (24 * 60) for t in times}) == n, "tide times collided after wrapping")
    return times


def _tide_or_sunrise_values_and_times(rng, n, close, table_type, target_index, direction):
    """(values, times): ``values`` is what the question ranks (tenths of a
    metre for tide, sunrise minutes for sunrise) with the target a clean
    extreme; ``times`` is what the table prints as a time."""
    if table_type == "tide":
        gap_lo, gap_hi = _tide_medium_gap() if close else _tide_easy_gap()
        heights = _draw_extreme_values(
            rng, n, _TIDE_MIN_TENTHS, _TIDE_MAX_TENTHS, gap_lo, gap_hi, target_index, direction
        )
        return heights, _tide_times(rng, n)
    lo, hi = (1, 3) if close else (2, 3)
    first_sign = 1 if direction == "high" else -1
    for _ in range(_RETRY_LIMIT):
        minutes = _sunrise_path(rng, n, lo, hi, target_index, first_sign)
        if len(set(minutes)) == n:
            return minutes, minutes
    raise RuntimeError("could not draw a distinct sunrise sequence")


def _tide_medium_gap():
    return (1, 2)


def _tide_easy_gap():
    return (4, 12)


_SUNRISE_SECOND_RANKED_PROB = 0.5  # share of sunrise rows asking for the SECOND latest/earliest
_SUNRISE_INTERIOR_TARGET_PROB = 0.2  # only ~1 in 5 sunrise tables turns


def _build_tide_or_sunrise(rng, n: int, close: bool, table_type: str):
    if table_type == "sunrise" and rng.random() >= _SUNRISE_INTERIOR_TARGET_PROB:
        target_index = rng.choice([0, n - 1])  # monotone drift: the extreme is an end row
    else:
        target_index = rng.randrange(1, n - 1) if table_type == "sunrise" else rng.randrange(n)
    direction = rng.choice(["high", "low"])
    start_dow = rng.randrange(7)
    days = [WEEKDAYS[(start_dow + i) % 7] for i in range(n)]

    if table_type == "tide":
        heights_tenths, times = _tide_or_sunrise_values_and_times(rng, n, close, table_type, target_index, direction)
        rows = [{"day": days[i], "time": times[i], "height_tenths": heights_tenths[i]} for i in range(n)]
        values = heights_tenths
        word = "highest" if direction == "high" else "lowest"
        question = f" Which day has the {word} high tide?"
        row_text = lambda r: f"{r['day']}: high tide at {_fmt_time(r['time'])}, height {r['height_tenths'] / 10:.1f} m."
        intro = "A tide chart shows the following high tides:"
    else:
        minutes, _ = _tide_or_sunrise_values_and_times(rng, n, close, table_type, target_index, direction)
        rows = [{"day": days[i], "time": minutes[i]} for i in range(n)]
        values = minutes
        word = "latest" if direction == "high" else "earliest"
        question = f" Which day has the {word} sunrise?"
        row_text = lambda r: f"{r['day']}: sunrise at {_fmt_time(r['time'])}."
        intro = "A table shows the following sunrise times:"

    check(len(set(values)) == n, "table entries collided; construction bug")
    if direction == "high":
        check(values[target_index] == max(values), "target index is not the constructed high extreme")
    else:
        check(values[target_index] == min(values), "target index is not the constructed low extreme")

    opp_index = values.index(max(values) if direction == "low" else min(values))
    trap_index = None
    if table_type == "sunrise" and rng.random() < _SUNRISE_SECOND_RANKED_PROB:
        # Ask for the SECOND latest/earliest sunrise, so the key is usually an interior row (a
        # monotone table's extreme is always an end row). The overall extreme is the trap option.
        order = sorted(range(n), key=lambda i: values[i], reverse=(direction == "high"))
        trap_index, target_index = order[0], order[1]
        text_word = "second " + word
        question = f" Which day has the {text_word} sunrise?"

    answer_day = days[target_index]
    # Offer: the correct day, the opposite-direction extreme (wrong-direction
    # misconception) -- or, for a second-ranked question, the overall extreme
    # as well -- and other genuinely different days.
    must = [target_index, opp_index] if trap_index is None else [target_index, trap_index, opp_index]
    remaining = [i for i in range(n) if i not in must]
    rng.shuffle(remaining)
    offered_indices = (must + remaining)[:4]
    check(len(set(offered_indices)) == 4, "offered options are not four distinct days")

    text = intro + " " + " ".join(row_text(r) for r in rows) + question
    distractor_days = [days[i] for i in offered_indices if i != target_index]

    if trap_index is None:
        working = [f"{answer_day} holds the {('highest' if direction == 'high' else 'lowest')} value in the table."]
    else:
        working = [f"{days[trap_index]} has the {word} sunrise, so {answer_day} has the second {word}."]

    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=answer_day,
        answer_text=answer_day,
        answer_latex=latex_answer(answer_day, plain=True),
        working=working,
        distractors=distractor_days,
    )


def _build_B_easy(rng):
    table_type = rng.choice(["tide", "sunrise"])
    return _build_tide_or_sunrise(rng, 4, close=False, table_type=table_type)


def _build_B_medium(rng):
    table_type = rng.choice(["tide", "sunrise"])
    return _build_tide_or_sunrise(rng, 5, close=True, table_type=table_type)


_PHASE_NAMES = [
    "new moon",
    "waxing crescent",
    "first quarter",
    "waxing gibbous",
    "full moon",
    "waning gibbous",
    "last quarter",
    "waning crescent",
]
_QUARTER_PHASES = {"new moon", "first quarter", "last quarter"}
_INTERMEDIATE_SPAN_CHOICES = [6, 7]  # waxing/waning crescent/gibbous each last ~6-7 real days
_FULL_MOON_TWO_DAY_PROB = 0.15  # full moon spans 2 printed days this often, otherwise 1; NEVER 3+

# Offset (target day - cutoff day) weighting: k=1 is weighted well below k
# in {2, 3...} so a single offset never dominates above ~40% of draws. Whenever target_day == 2, the
# feasible-k set is {1} only, so k==1 is forced regardless of these weights on that ~20% of draws
# --.2% realised share at 2000 seeds and why that structural floor cannot be tuned away by these
# weights alone.
_OFFSET_WEIGHT_K1 = 1
_OFFSET_WEIGHT_OTHER = 3


def _build_cycle_days(rng):
    """One draw's worth of a REAL ~29-31 day lunar cycle. The four NAMED
    INSTANT phases (new moon, first quarter, full moon, last quarter) each
    span one printed day (full moon occasionally two, per
    `_FULL_MOON_TWO_DAY_PROB`, but never three or more); the four
    crescent/gibbous phases each span 6 or 7 days, so the whole cycle lands
    close to 29-31 days without needing exact astronomy. Returns a list of
    (phase_name, span_in_days) in cycle order."""
    full_span = 2 if rng.random() < _FULL_MOON_TWO_DAY_PROB else 1
    spans = []
    for name in _PHASE_NAMES:
        if name == "full moon":
            spans.append(full_span)
        elif name in _QUARTER_PHASES:
            spans.append(1)
        else:
            spans.append(rng.choice(_INTERMEDIATE_SPAN_CHOICES))
    return list(zip(_PHASE_NAMES, spans))


def _phase_at_cycle_index(cycle_days, cycle_len, index):
    """Walk cycle_days (as built by _build_cycle_days) to find which phase
    owns absolute day-of-cycle `index` (any integer; wrapped mod the
    cycle's own total length). Never independent per printed day -- every
    day's phase is a deterministic function of one shared starting point,
    so consecutive printed days can only ever show the SAME phase (while
    inside one multi-day span) or the NEXT phase in the real cycle order,
    never an arbitrary jump."""
    idx = index % cycle_len
    acc = 0
    for name, span in cycle_days:
        if acc <= idx < acc + span:
            return name
        acc += span
    raise AssertionError("cycle_days did not cover its own declared length")  # pragma: no cover


def _build_B_hard(rng):
    """Draw the key rank once (uniform 0-3), then redraw the window,
    question and key until that rank can be supplied, so the marginal rank is
    exactly uniform rather than skewed toward what late keys can supply."""
    rank = rng.randrange(4)
    last = None
    for _ in range(_RETRY_LIMIT):
        try:
            return _build_B_hard_ranked(rng, rank)
        except VerificationError as exc:
            last = exc
    raise RuntimeError("could not construct a hard-tier moon-phase table") from last


def _build_B_hard_ranked(rng, rank):
    """Replaces the old fixed-block, 8-day compressed-cycle construction entirely. A 6-day printed window is now
    a genuinely random SLICE of a real ~29-31 day cycle (`_build_cycle_days` / `_phase_at_cycle_index`): pick
    a uniformly random starting day-of-cycle for day 1 of the window, derive each of the 6 printed days' phase
    from that one shared cycle, then require (a) the window contains a full moon at some day `target_day` >= 2
    (so a cutoff day can exist strictly before it within the window) and (b) the cutoff day itself is never a
    full moon and never has a full moon between it and `target_day` -- eliminating both earlier
    defects (reference day itself a full moon; impossible 8-day-per-cycle jumps) by construction rather than
    by a reject-and-redraw patch on the old model.

    Diversity ceiling, worked out and reported honestly rather than chased: with the crescent/gibbous phases
    spanning >=6 real days, ANY 6-day window that contains a full moon can only ever show "waxing gibbous"
    immediately before it and "waning gibbous" immediately after it (first quarter and last quarter are always
    too far away in real time to enter a 6-day window) -- so the space of distinct 6-day PHASE-LABEL sequences
    is small on physical grounds: 5 (a 1-day full moon at window position 2..6) + 4 (a 2-day full-moon block
    starting at position 2..5) = 9 maximum, well under the old model's 14 (which only reached 14 by being
    physically wrong).

    Offset (target day - cutoff day) weighting: see `_OFFSET_WEIGHT_K1`/
    `_OFFSET_WEIGHT_OTHER` above.

    The question kind varies across seeds (first full moon after a reference day; the day N days after /
    before the first full-moon day; the first waning-gibbous day after a reference day) so the key is usually
    NOT the first "full moon" row, and the reference day is never one of the four options.
    """
    n = 6
    for _ in range(_RETRY_LIMIT):
        cycle_days = _build_cycle_days(rng)
        cycle_len = sum(span for _, span in cycle_days)
        start = rng.randrange(cycle_len)  # day-of-cycle for printed day 1
        phase = {d: _phase_at_cycle_index(cycle_days, cycle_len, start + (d - 1)) for d in range(1, n + 1)}
        full_days = [d for d in range(1, n + 1) if phase[d] == "full moon"]
        if full_days:
            break
    else:
        raise RuntimeError("could not construct a hard-tier moon-phase table")

    p = full_days[0]  # first printed day showing a full moon
    block_end = p
    while block_end + 1 <= n and phase[block_end + 1] == "full moon":
        block_end += 1

    # The question kind varies so no single lookup heuristic wins, and in every kind the reference
    # day is NEVER an option. Only kinds this window can support are drawn.
    kinds = {}
    if p >= 2:
        kinds["after_ref"] = _MOON_KIND_WEIGHTS["after_ref"]
        kinds["before_full"] = _MOON_KIND_WEIGHTS["before_full"]
    if p <= n - 2:
        kinds["after_full"] = _MOON_KIND_WEIGHTS["after_full"]
    if p >= 2 and block_end + 1 <= n:
        kinds["gibbous_after_ref"] = _MOON_KIND_WEIGHTS["gibbous_after_ref"]
    kind = rng.choices(list(kinds), weights=list(kinds.values()), k=1)[0]

    # A random calendar offset: the table's six days are date_offset+1 ..
    # date_offset+6, never always 1st..6th. Internal reasoning stays on the
    # RELATIVE day index (1..6); only the printed label carries the offset.
    date_offset = rng.randint(0, 19)
    labels = {d: _ordinal(d + date_offset) for d in range(1, n + 1)}
    rows = [f"The {labels[d]}: {phase[d]}." for d in range(1, n + 1)]
    head = "A moon-phase table shows the following for six consecutive days: " + " ".join(rows)

    if kind == "after_ref":
        # first full moon after a reference day strictly before it
        feasible_ks = list(range(1, p))
        weights = [_OFFSET_WEIGHT_K1 if k == 1 else _OFFSET_WEIGHT_OTHER for k in feasible_ks]
        ref = p - rng.choices(feasible_ks, weights=weights, k=1)[0]
        check(phase[ref] != "full moon", "reference day must never itself be a full moon")
        key = p
        question = f" Which is the first full moon after the {labels[ref]}?"
        plausible = [block_end if block_end != p else None, p - 1, p + 1, p + 2]
        working = [f"The first full moon after the {labels[ref]} is on the {labels[key]}."]
    elif kind == "after_full":
        offset = rng.randint(2, n - p)
        ref, key = p, p + offset
        question = f" Which day is {offset} {'day' if offset == 1 else 'days'} after the first day showing a full moon?"
        plausible = [key - 1, key + 1, p - offset, key - 2, key + 2]
        working = [
            f"The first day showing a full moon is the {labels[p]}; {offset} {'day' if offset == 1 else 'days'} after it is the {labels[key]}."
        ]
    elif kind == "before_full":
        offset = rng.randint(1, p - 1)
        ref, key = p, p - offset
        question = (
            f" Which day is {offset} {'day' if offset == 1 else 'days'} before the first day showing a full moon?"
        )
        plausible = [key + 1, key - 1, p + offset, key + 2, key - 2]
        working = [
            f"The first day showing a full moon is the {labels[p]}; {offset} {'day' if offset == 1 else 'days'} before it is the {labels[key]}."
        ]
    else:  # gibbous_after_ref
        ref = rng.randint(1, p - 1)
        key = block_end + 1
        check(phase[key] == "waning gibbous", "day after the full moon must be waning gibbous")
        question = f" Which is the first day after the {labels[ref]} showing a waning gibbous moon?"
        plausible = [p, block_end if block_end != p else None, key + 1, p - 1, ref + 1]
        working = [f"The moon first shows waning gibbous on the {labels[key]}, the day after the full moon ends."]

    text = head + question

    # Three real, distinct distractors; never the key, never the reference
    # day. A target key rank (day order among the four options) is drawn
    # first, from the ranks this window can supply, and the options are
    # then filled from the plausible near-misses on each side of the key
    # (shuffled), then from any other day -- so the key's rank is spread
    # rather than pinned by where the window happens to put it.
    plausible = [d for d in dict.fromkeys(plausible) if d is not None and 1 <= d <= n and d not in (key, ref)]
    pool = [d for d in range(1, n + 1) if d not in (key, ref)]
    below = [d for d in pool if d < key]
    above = [d for d in pool if d > key]
    ranks = [r for r in range(4) if len(below) >= r and len(above) >= 3 - r]
    # ``rank`` (number of distractors BELOW the key) was drawn uniformly 0-3 by `_build_B_hard`, as
    # the fractions generator's `_share_options` does; a window that cannot supply it is redrawn
    # whole.
    check(rank in ranks, "this moon window cannot supply the drawn key rank")

    def take(side, count):
        near = [d for d in side if d in plausible]
        rest = [d for d in side if d not in plausible]
        rng.shuffle(near)
        rng.shuffle(rest)
        return (near + rest)[:count]

    chosen = take(below, rank) + take(above, 3 - rank)
    rng.shuffle(chosen)
    check(
        len(set(chosen + [key])) == 4 and ref not in chosen,
        "moon options must be four distinct days without the reference",
    )

    answer_day = f"The {labels[key]}"
    distractors = [f"The {labels[d]}" for d in chosen]

    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=answer_day,
        answer_text=answer_day,
        answer_latex=latex_answer(answer_day, plain=True),
        working=working,
        distractors=distractors,
    )


_MOON_KIND_WEIGHTS = {"after_ref": 2, "after_full": 3, "before_full": 3, "gibbous_after_ref": 3}


_EXAM_TIDE_KEY_EXTREME_PROB = 0.18  # share of exam tide rows whose key is the week's tallest tide
_EXAM_SUNRISE_KEY_WEIGHTS = [35, 30, 35]  # key on the 4th / 5th / 6th
_EXAM_SUNRISE_ASCENDING_SHARE = 0.25  # of 6th-keyed rows, a plain ascending run
_EXAM_SUNRISE_FIFTH_OFFERED_PROB = 0.8  # the 5th is offered this often when it is not the key
_EXAM_SUNRISE_TRAP_OFFERED_PROB = (
    0.1  # forced offers of the unfiltered latest (days 1-3); higher leaks the key through which day is offered (2178)
)


def _draw_exam_sunrise_values(rng, n):
    """(values, key_index) for the exam sunrise table. The key day
    among the 4th-6th is drawn first; the path is physical (1-3 min steps, at
    most one turning point, flat at an interior turn). 4th: descending or a
    peak on day 2/3. 5th: a peak on day 5. 6th: a valley then a rise with day 1
    later than day 6 (so the after-the-3rd filter matters), or, in a
    minority, a plain ascending run. The real key is recomputed as the argmax
    of days 4-6 and any draw that misses its intended day, or ties, is redrawn."""
    key = rng.choices([3, 4, 5], weights=_EXAM_SUNRISE_KEY_WEIGHTS)[0]
    ascending = key == 5 and rng.random() < _EXAM_SUNRISE_ASCENDING_SHARE  # decided once, not per retry
    for _ in range(_RETRY_LIMIT):
        if key == 3:
            values = _sunrise_path(rng, n, 1, 3, rng.choice([0, 1, 2]), 1)
        elif key == 4:
            values = _sunrise_path(rng, n, 1, 3, 4, 1)
        elif ascending:
            values = _sunrise_path(rng, n, 1, 3, n - 1, 1)
        else:
            values = _sunrise_path(rng, n, 1, 3, rng.randrange(1, 4), -1)
            if values[0] <= values[5]:
                continue
        if len(set(values)) != n:
            continue
        if max((3, 4, 5), key=lambda i: values[i]) != key:
            continue
        return values, key
    raise RuntimeError("could not construct exam-tier sunrise/date table")


def _build_B_exam(rng):
    n = 6
    # Sunrise drift realism (one direction change at most) leaves only the
    # 4th or 6th day able to be the correct answer to "latest after the 3rd"
    # (the 5th would need a second turning point), so this variant is drawn
    # less often to keep any single answer label under the 25% cap.
    variant = rng.choices(["tide_weekend", "sunrise_after_date"], weights=[3, 2])[0]

    if variant == "tide_weekend":
        start_dow = rng.randrange(7)
        days = [WEEKDAYS[(start_dow + i) % 7] for i in range(n)]
        weekend_idx = [i for i, d in enumerate(days) if d in WEEKEND]
        nonweekend_idx = [i for i in range(n) if i not in weekend_idx]
        check(weekend_idx and nonweekend_idx, "exam tide table needs both a weekend and a non-weekend day")

        key_is_extreme = rng.random() < _EXAM_TIDE_KEY_EXTREME_PROB
        for _ in range(_RETRY_LIMIT):
            target_i = rng.choice(nonweekend_idx)
            disq_i = rng.choice(weekend_idx)
            others = [i for i in range(n) if i not in (target_i, disq_i)]
            base = rng.sample(range(_TIDE_MIN_TENTHS, 19), len(others))
            if key_is_extreme:
                # The tallest tide of the whole week is the keyed (non-weekend) day; the weekend day
                # is a tall runner-up.
                disq_val = max(base) + rng.randint(2, 4)
                target_val = disq_val + rng.randint(2, 5)
            else:
                # Every height stays within 1.0-3.0 m.
                target_val = max(base) + rng.randint(3, 6)
                disq_val = target_val + rng.randint(3, 6)
            values = [None] * n
            for idx, v in zip(others, base):
                values[idx] = v
            values[target_i] = target_val
            values[disq_i] = disq_val
            if len(set(values)) == n and all(_TIDE_MIN_TENTHS <= v <= _TIDE_MAX_TENTHS for v in values):
                break
        else:
            raise RuntimeError("could not construct exam-tier tide/weekend table")

        opp_i = others[base.index(min(base))]
        remaining = [i for i in others if i != opp_i]
        rng.shuffle(remaining)
        filler_i = remaining[0]

        tide_times = _tide_times(rng, n)
        rows = [f"{days[i]}: high tide at {_fmt_time(tide_times[i])}, height {values[i] / 10:.1f} m." for i in range(n)]
        text = (
            "A tide chart shows the following high tides across a week: "
            + " ".join(rows)
            + " Which day has the highest high tide that is NOT a weekend day?"
        )
        answer_day = days[target_i]
        distractor_days = [days[disq_i], days[opp_i], days[filler_i]]
        if key_is_extreme:
            working = [
                f"{answer_day} has the highest tide of the week and is not a weekend day, "
                f"so it is the answer; {days[disq_i]} is a weekend day and is excluded anyway."
            ]
        else:
            working = [
                f"{days[disq_i]} has a higher tide but falls on a weekend, so it is excluded; "
                f"{answer_day} has the highest tide among the remaining days."
            ]
    else:
        # The key day (4th/5th/6th) is drawn FIRST, near-uniformly, then a physical path (monotone
        # or one flat turn) that produces it.
        values, target_i = _draw_exam_sunrise_values(rng, n)
        disq_i = max((0, 1, 2), key=lambda i: values[i])
        offered = {3, 5, target_i}
        if rng.random() < _EXAM_SUNRISE_FIFTH_OFFERED_PROB:
            offered.add(4)
        slots = 4 - len(offered)
        low = [0, 1, 2]
        rng.shuffle(low)
        if rng.random() < _EXAM_SUNRISE_TRAP_OFFERED_PROB:
            low.remove(disq_i)
            low.insert(0, disq_i)  # the unfiltered latest is offered
        offered_idx = sorted(offered) + low[:slots]
        offered_distractors = [i for i in offered_idx if i != target_i]
        rng.shuffle(offered_distractors)

        day_labels = [_ordinal(i + 1) for i in range(n)]
        rows = [f"The {day_labels[i]}: sunrise at {_fmt_time(values[i])}." for i in range(n)]
        text = (
            "A table shows the following sunrise times over six days: "
            + " ".join(rows)
            + " Which day has the latest sunrise among the days after the 3rd?"
        )
        answer_day = f"The {day_labels[target_i]}"
        distractor_days = [f"The {day_labels[i]}" for i in offered_distractors]
        if values[disq_i] > values[target_i]:
            working = [
                f"The {day_labels[disq_i]} has a later sunrise but is on or before the 3rd, so it is "
                f"excluded; {answer_day} has the latest sunrise among the days after the 3rd."
            ]
        else:
            working = [
                f"{answer_day} has the latest sunrise among the days after the 3rd "
                f"(later than the {day_labels[disq_i]}, the latest on or before the 3rd)."
            ]

    check(len(set(distractor_days)) == 3, "exam offered distractors are not three distinct days")

    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=answer_day,
        answer_text=answer_day,
        answer_latex=latex_answer(answer_day, plain=True),
        working=working,
        distractors=distractor_days,
    )


_CONDITION_LOOKUP_BUILDERS = {
    "easy": _build_B_easy,
    "medium": _build_B_medium,
    "hard": _build_B_hard,
    "exam": _build_B_exam,
}


@generator(
    "p19i_timetable_condition_lookup",
    level=9,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
        "exam": dict(tier="exam"),
    },
)
def p19i_timetable_condition_lookup(rng, tier="easy"):
    """Given a stated tide, sunrise/sunset or moon-phase table for several
    days, choose from four offered day labels the one meeting a stated
    condition (a superlative height/time, or the first full moon after a
    date). Boundary: the offered options are always day labels, never a raw
    duration or a single cell read in isolation -- see the module
    docstring."""
    builder = _CONDITION_LOOKUP_BUILDERS.get(tier)
    if builder is None:
        raise ValueError(f"unknown tier {tier!r}; expected one of {sorted(_CONDITION_LOOKUP_BUILDERS)}")
    return _retry(rng, builder, f"p19i_timetable_condition_lookup/{tier}")
