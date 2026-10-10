"""Critical-path location and float times, both senior ``acara-senior-v8-4``, domain ``financial-discrete``.

Two generators, sharing ONE fixed six-activity network fixture so the two
skills can never disagree about the same graph:

    A -> C -> D -> F        (Path 1, length sum1 = dA + dC + dD + dF)
    B -> E -> F              (Path 2, length sum2 = dB + dE + dF)

The library already has TWO project-network skills on the same curriculum
strand (``p6b_project_network_build.py``): raw EST/LST values (ACMGM105) and the
project's minimum completion time (ACMGM107). Neither of THIS module's two skills
asks for either of those; they grade different operations:

- ``p10c_critical_path_locate`` (``senior-use-ests-and-lsts-to-locate-the-
  critical``, ACMGM106) -- **classify/locate** which activities are on the
  critical path (never a raw EST/LST value): easy asks a Yes/No question
  about one named activity, medium asks a head-to-head comparison between
  one activity from each path, hard asks for a COUNT of critical activities.
  ``answer: mcq``.
- ``p10c_critical_path_float`` (``senior-calculate-float-times-for-non-
  critical-activities``, ACMGM108) -- **calculate the float (slack)** of a
  named non-critical activity (never the project's minimum completion time).
  ``answer: number``.

This module does not import from, or modify, ``p6b_project_network_build.py``;
every helper below (forward pass, backward pass, network-as-text formatter)
is written fresh.
"""

from __future__ import annotations

from mathgen.core import check, check_distinct, check_equivalent, generator, latex_answer, pick_distractors

ACTIVITIES = ("A", "B", "C", "D", "E", "F")
PREDECESSORS = {
    "A": [],
    "B": [],
    "C": ["A"],
    "D": ["C"],
    "E": ["B"],
    "F": ["D", "E"],
}
MIN_MARGIN = 2
DURATION_LOW, DURATION_HIGH = 2, 12


# --- Shared network helpers (forward pass, backward pass, formatter) -------


def _forward_pass(durations):
    """Genuine forward pass (EST/EFT) in topological order A, B, C, D, E, F.

    Generic over duration values -- not hardcoded per-seed arithmetic.
    """
    est = {}
    eft = {}
    for label in ACTIVITIES:
        preds = PREDECESSORS[label]
        est[label] = 0 if not preds else max(eft[p] for p in preds)
        eft[label] = est[label] + durations[label]
    return est, eft


def _successors():
    succ = {label: [] for label in ACTIVITIES}
    for label, preds in PREDECESSORS.items():
        for p in preds:
            succ[p].append(label)
    return succ


def _backward_pass(durations, completion_time):
    """Genuine backward pass (LST/LFT) in reverse topological order F..A."""
    succ = _successors()
    lst = {}
    lft = {}
    for label in reversed(ACTIVITIES):
        s = succ[label]
        lft[label] = completion_time if not s else min(lst[t] for t in s)
        lst[label] = lft[label] - durations[label]
    return lst, lft


def _format_network_text(durations):
    """One line per activity: "Activity X: duration N days, predecessors: A, B."
    / "...no predecessors."  Own fresh version of p6b's precedent style."""
    lines = []
    for label in ACTIVITIES:
        duration = durations[label]
        preds = PREDECESSORS[label]
        if not preds:
            lines.append(f"Activity {label}: duration {duration} days, no predecessors.")
        else:
            lines.append(f"Activity {label}: duration {duration} days, predecessors: {', '.join(preds)}.")
    return " ".join(lines)


def _path_sums(durations):
    sum1 = durations["A"] + durations["C"] + durations["D"] + durations["F"]
    sum2 = durations["B"] + durations["E"] + durations["F"]
    return sum1, sum2


def _draw_durations(rng, want_path1_wins: bool):
    """Draw the answer class FIRST (which path wins is already decided by the caller), then construct durations
    that realise it, with the required strict margin so "the two paths tie" never needs handling and no
    +/-1-day distractor can collide with the true classification."""
    for _ in range(1000):
        durations = {label: rng.randint(DURATION_LOW, DURATION_HIGH) for label in ACTIVITIES}
        sum1, sum2 = _path_sums(durations)
        margin = sum1 - sum2
        if want_path1_wins and margin >= MIN_MARGIN:
            return durations
        if not want_path1_wins and -margin >= MIN_MARGIN:
            return durations
    check(False, "could not draw durations meeting the margin after 1000 attempts")


# --- Independent check for the shared fixture itself -----------------------
# Explicit path enumeration, hardcoded against this fixed topology (only two
# paths) -- a genuinely different route from the forward/backward-pass
# recursion above (direct path summation vs. layer-by-layer recursion).


def _path_enumeration_est_lst(durations):
    dA, dB, dC, dD, dE, dF = (durations[k] for k in "ABCDEF")
    est = {
        "A": 0,
        "B": 0,
        "C": dA,
        "D": dA + dC,
        "E": dB,
        "F": max(dA + dC + dD, dB + dE),
    }
    completion_time = est["F"] + dF
    lst = {
        "F": completion_time - dF,
    }
    lst["D"] = lst["F"] - dD
    lst["C"] = lst["D"] - dC
    lst["A"] = lst["C"] - dA
    lst["E"] = lst["F"] - dE
    lst["B"] = lst["E"] - dB
    return est, lst, completion_time


def _verify_shared_fixture(durations):
    """Confirm every activity's EST/LST both ways (path enumeration vs.
    forward/backward-pass recursion) and assert equality."""
    est_fp, eft_fp = _forward_pass(durations)
    completion_time = max(eft_fp.values())
    lst_bp, lft_bp = _backward_pass(durations, completion_time)
    est_pe, lst_pe, completion_pe = _path_enumeration_est_lst(durations)
    check_equivalent(completion_pe, completion_time, "path-enumeration completion time must match forward pass")
    for label in ACTIVITIES:
        check_equivalent(est_pe[label], est_fp[label], f"EST({label}) must match between routes")
        check_equivalent(lst_pe[label], lst_bp[label], f"LST({label}) must match between routes")
    return est_fp, eft_fp, lst_bp, lft_bp, completion_time


# --- Skill 1: senior-use-ests-and-lsts-to-locate-the-critical (ACMGM106) ---


def _verify_path1_wins_classification(sum1, sum2, path1_wins):
    """Independent route for Skill 1: derive the classification straight from
    sum1/sum2, never touching EST/LST -- exercised directly with a corrupted
    ``path1_wins`` by the test file."""
    check(path1_wins == (sum1 > sum2), "path1_wins must equal sum1 > sum2")


def _mcq(text, answer, candidates, *, answer_latex, latex=None, working=None):
    """Own small mcq-shape helper."""
    return dict(
        text=text,
        latex=latex or rf"\text{{{text}}}",
        answer=answer,
        answer_text=answer,
        answer_latex=answer_latex,
        working=working or [],
        distractors=pick_distractors(answer, candidates, n=len(candidates)),
    )


def _generate_locate_easy(rng):
    """Is activity D on the critical path? (Yes/No, absolute check)."""
    path1_wins = rng.choice([True, False])
    durations = _draw_durations(rng, path1_wins)
    sum1, sum2 = _path_sums(durations)
    _verify_shared_fixture(durations)
    # Independent route: derive straight from sum1/sum2, never touching EST/LST.
    _verify_path1_wins_classification(sum1, sum2, path1_wins)
    answer = "Yes" if path1_wins else "No"
    other = "No" if path1_wins else "Yes"
    check_distinct(answer, other)
    text = f"The project starts at time 0. {_format_network_text(durations)} Is activity D on the critical path?"
    return _mcq(
        text,
        answer,
        [other],
        answer_latex=latex_answer(answer, plain=True),
        working=[f"sum(A,C,D,F)={sum1}, sum(B,E,F)={sum2}; the critical path is the longer of the two."],
    )


def _generate_locate_medium(rng):
    """Which of C or E is on the critical path? (head-to-head comparison)."""
    path1_wins = rng.choice([True, False])
    durations = _draw_durations(rng, path1_wins)
    sum1, sum2 = _path_sums(durations)
    _verify_shared_fixture(durations)
    _verify_path1_wins_classification(sum1, sum2, path1_wins)
    answer = "C" if path1_wins else "E"
    other = "E" if path1_wins else "C"
    check_distinct(answer, other)
    text = (
        f"The project starts at time 0. {_format_network_text(durations)} "
        f"Which of these two activities is on the critical path: C or E?"
    )
    return _mcq(
        text,
        answer,
        [other],
        answer_latex=latex_answer(answer, plain=True),
        working=[f"sum(A,C,D,F)={sum1}, sum(B,E,F)={sum2}; C is on Path 1, E is on Path 2."],
    )


def _generate_locate_hard(rng):
    """How many activities lie on the critical path? (count, with two
    collision-proof distractors)."""
    path1_wins = rng.choice([True, False])
    durations = _draw_durations(rng, path1_wins)
    sum1, sum2 = _path_sums(durations)
    _verify_shared_fixture(durations)
    _verify_path1_wins_classification(sum1, sum2, path1_wins)
    correct = 4 if path1_wins else 3
    other_path_count = 3 if path1_wins else 4
    # Collision-proof by construction: when correct == 4 the off-by-one is
    # correct + 1 == 5 (never equal to other_path_count == 3); when correct
    # == 3 the off-by-one is correct - 1 == 2 (never equal to
    # other_path_count == 4). {2, 3, 4, 5} are four consecutive integers and
    # each branch only ever touches three of them, so the two distractors
    # can never coincide with each other or with the answer.
    off_by_one = (correct + 1) if correct == 4 else (correct - 1)
    check_distinct(correct, other_path_count, off_by_one)
    check(len({correct, other_path_count, off_by_one}) == 3, "hard-tier options must be three distinct counts")
    text = (
        f"The project starts at time 0. {_format_network_text(durations)} "
        f"How many activities lie on the critical path, including its first and last activity?"
    )
    return _mcq(
        text,
        str(correct),
        [str(other_path_count), str(off_by_one)],
        answer_latex=latex_answer(correct),
        working=[
            f"sum(A,C,D,F)={sum1} (4 activities), sum(B,E,F)={sum2} (3 activities); "
            f"the critical path is the longer of the two, giving {correct} activities."
        ],
    )


@generator(
    "p10c_critical_path_locate",
    level=11,
    presets={
        "easy": dict(tier="locate_yes_no"),
        "medium": dict(tier="locate_head_to_head"),
        "hard": dict(tier="locate_count"),
    },
)
def p10c_critical_path_locate(rng, tier="locate_count"):
    """Use ESTs and LSTs to locate the critical path(s) for the project.

    Bare-call default is deliberately ``locate_count`` (the ``hard`` preset),
    not ``locate_yes_no`` (the ``easy`` preset): every skill source passes
    ``tier`` explicitly (see the three presets above), so this default only
    matters for a bare ``make("p10c_critical_path_locate")`` call with no
    difficulty -- and both ``locate_yes_no`` and ``locate_head_to_head`` are
    genuine 2-option mcqs with exactly 1 distractor by design, which fails
    the usual 85%-of-seeds-need->=2-distractors bar on every seed. Only ``locate_count``
    offers 3 options (2 distractors) unconditionally, so it is the only tier
    that can serve as the bare-call default.
    """
    if tier == "locate_yes_no":
        return _generate_locate_easy(rng)
    elif tier == "locate_head_to_head":
        return _generate_locate_medium(rng)
    elif tier == "locate_count":
        return _generate_locate_hard(rng)
    else:
        raise ValueError(f"unknown tier: {tier!r}")


# --- Skill 2: senior-calculate-float-times-for-non-critical-activities -----
# (ACMGM108)


def _direct_path_sum_float(durations, est, completion_time, query, between):
    """Independent check route: forward path-sum-to-completion, NOT the
    LST/EST backward-pass recursion construction uses.

    float(query) = (completion_time - dF) - (EST(query) + duration(query) +
    sum of durations of every activity STRICTLY BETWEEN query and sink F).

    NOTE ON A SIMPLER FORMULA: it is tempting to write this as ``completion_time - (EST(query)
    + duration(query) + sum(between))``, with NO ``- dF`` term. F's own duration is neither the queried
    activity nor "strictly between" it and the sink, but it still has to run, after the queried chain
    finishes, before the project is done; the true slack is against LST(F) = completion_time - dF, not against
    completion_time itself. The ``- dF`` term is required for this check to agree with the LST/EST route at
    all, so it is included here.
    """
    dF = durations["F"]
    downstream = sum(durations[label] for label in between)
    return (completion_time - dF) - (est[query] + durations[query] + downstream)


def _verify_float_independent(durations, est, completion_time, query, between, answer):
    """Independent route for Skill 2: direct path-sum-to-completion, never the
    LST/EST backward-pass recursion construction uses -- exercised directly with
    a corrupted ``answer`` by the test file."""
    checked = _direct_path_sum_float(durations, est, completion_time, query, between)
    check_equivalent(checked, answer, "direct path-sum float must match the LST/EST route")


def _float_generate(rng, *, path1_wins_target: bool, query: str, between: tuple[str, ...], reframed: bool):
    durations = _draw_durations(rng, path1_wins_target)
    sum1, sum2 = _path_sums(durations)
    _verify_shared_fixture(durations)
    est_fp, eft_fp = _forward_pass(durations)
    completion_time = max(eft_fp.values())
    lst_bp, _lft_bp = _backward_pass(durations, completion_time)

    # The true float, via the backward-pass (construction) route.
    answer = lst_bp[query] - est_fp[query]
    check(answer != 0, "answer must never be 0 -- every query targets the losing branch by construction")

    # Independent route: direct path-sum-to-completion, never touching LST.
    _verify_float_independent(durations, est_fp, completion_time, query, between, answer)

    # Shared-float property: every activity on the same losing chain shares
    # ONE float value (no branching within a chain). Assert it directly.
    if query in ("A", "C", "D"):
        losing_chain = ("A", "C", "D")
    else:
        losing_chain = ("B", "E")
    chain_floats = {label: lst_bp[label] - est_fp[label] for label in losing_chain}
    check(len(set(chain_floats.values())) == 1, "every activity on the losing chain must share one float value")
    check_equivalent(
        answer,
        sum2 - sum1 if query in ("A", "C", "D") else sum1 - sum2,
        "float must equal the margin between the two path sums",
    )

    # --- Misconceptions, each check_distinct'd unconditionally, every seed --
    reports_lst_not_float = lst_bp[query]
    if query == "A":
        # DEGENERATE CASE: A is always a source (EST(A) == 0
        # for every seed unconditionally), so LST(A) - EST(A) == LST(A) identically and "report the
        # raw LST" collides with the true float on every single seed for THIS activity only
        # (verified: A is the only query used by this module whose EST is structurally always 0).
        # Substituted with the raw LFT instead (an off-by-duration slip, LST + dA), which differs
        # from the float by dA >= 2 on every seed.
        reports_lst_not_float = lst_bp[query] + durations[query]
    wrong_branch_zero = 0
    candidates = [str(reports_lst_not_float), str(wrong_branch_zero)]
    misconception_ids = ["reports-lst-not-float", "wrong-branch-zero"]

    if between:
        # omitted-intermediate-duration: drop exactly one intermediate
        # activity's duration from the direct-path-sum route. Pick one
        # specific omission per draw.
        dropped = rng.choice(between)
        kept = tuple(b for b in between if b != dropped)
        omitted = _direct_path_sum_float(durations, est_fp, completion_time, query, kept)
        candidates.append(str(omitted))
        misconception_ids.append("omitted-intermediate-duration")

    reversed_sign = sum1 - sum2 if query in ("A", "C", "D") else sum2 - sum1
    # reversed_sign is the NEGATIVE of the true float (both candidate values,
    # sum2 - sum1 and sum1 - sum2, are negatives of each other and can only
    # coincide at 0 -- ruled out by the margin >= 2 guarantee) -- render it as
    # its own signed value, not a re-derived "positive" number.
    candidates.append(str(reversed_sign))
    misconception_ids.append("reversed-sign")

    check_distinct(answer, *[int(c) for c in candidates])

    activity_name = {"E": "E", "A": "A", "C": "C"}[query]
    if not reframed:
        text = (
            f"The project starts at time 0. {_format_network_text(durations)} "
            f"Calculate the float time of activity {activity_name}, in days."
        )
    else:
        text = (
            f"The project starts at time 0. {_format_network_text(durations)} "
            f"By how many days could activity {activity_name} "
            f"be delayed without delaying the project's overall completion time?"
        )
    answer_text = str(answer)
    distractors = pick_distractors(answer_text, candidates, n=len(candidates))
    return dict(
        text=text,
        latex=rf"\text{{{text}}}",
        answer=answer,
        answer_text=answer_text,
        answer_latex=latex_answer(answer),
        working=[
            f"Forward pass gives EST({query})={est_fp[query]}; backward pass gives LST({query})={lst_bp[query]}; "
            f"float({query}) = LST - EST = {answer} days."
        ],
        distractors=distractors,
    )


def _generate_float_easy(rng):
    """Path 2 loses; float of E (adjacent to the merge point F)."""
    return _float_generate(rng, path1_wins_target=True, query="E", between=(), reframed=False)


def _generate_float_medium(rng):
    """Path 1 loses; float of A (furthest from the merge, whole-chain slack)."""
    return _float_generate(rng, path1_wins_target=False, query="A", between=("C", "D"), reframed=False)


def _generate_float_hard(rng):
    """Path 1 loses; applied interpretation (never says "float") for C."""
    return _float_generate(rng, path1_wins_target=False, query="C", between=("D",), reframed=True)


@generator(
    "p10c_critical_path_float",
    level=11,
    presets={
        "easy": dict(tier="float_adjacent"),
        "medium": dict(tier="float_furthest"),
        "hard": dict(tier="float_delay_interpretation"),
    },
)
def p10c_critical_path_float(rng, tier="float_adjacent"):
    """Calculate float times for non-critical activities."""
    if tier == "float_adjacent":
        return _generate_float_easy(rng)
    elif tier == "float_furthest":
        return _generate_float_medium(rng)
    elif tier == "float_delay_interpretation":
        return _generate_float_hard(rng)
    else:
        raise ValueError(f"unknown tier: {tier!r}")
