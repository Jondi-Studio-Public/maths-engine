"""Project network -- EST/LST and critical path, both senior ``acara-senior-v8-4``, domain
``financial-discrete``, ``answer: number``.

Two generators:

- ``use_forward_and_backward_scanning_to_determine_the`` (``senior-use-forward-and-backward-
  scanning-to-determine-the``, ACMGM105) -- forward/backward scanning to find EST/LST.
  Three tiers: easy (4-activity chain, ask for EST of sink), medium (5-activity with one
  merge, ask for EST of merge), hard (6-activity with two merges, ask for LST of non-sink).

- ``use_the_critical_path_to_determine_the_minimum`` (``senior-use-the-critical-path-to-
  determine-the-minimum``, ACMGM107) -- critical path to find project completion time.
  Three tiers: easy (4-activity chain), medium (5-activity with one merge), hard (6-activity
  with two merges).

Both use integer durations drawn independently per seed. Topologies are fixed per tier so
reasoning escalation is guaranteed every seed.
"""

from __future__ import annotations

from mathgen.core import check, check_equivalent, generator, pick_distractors


# --- Shared network helpers (forward and backward pass) -------------------------


def _forward_pass(activities):
    """Forward pass: compute ES and EF for each activity.

    Args:
        activities: {label: {"duration": int, "predecessors": [label, ...]}}

    Returns:
        {label: (ES, EF)} in letter order (already topological)
    """
    es_ef = {}
    for label in sorted(activities):
        preds = activities[label]["predecessors"]
        es = 0 if not preds else max(es_ef[p][1] for p in preds)
        ef = es + activities[label]["duration"]
        es_ef[label] = (es, ef)
    return es_ef


def _successors(activities):
    """Build successor map: for each activity, list its immediate successors.

    Args:
        activities: {label: {"duration": int, "predecessors": [label, ...]}}

    Returns:
        {label: [successor, ...]}
    """
    succ = {label: [] for label in activities}
    for label, a in activities.items():
        for p in a["predecessors"]:
            succ[p].append(label)
    return succ


def _backward_pass(activities, es_ef, completion_time):
    """Backward pass: compute LS and LF for each activity.

    Args:
        activities: {label: {"duration": int, "predecessors": [label, ...]}}
        es_ef: {label: (ES, EF)} from forward pass
        completion_time: maximum EF (project completion time)

    Returns:
        {label: (LS, LF)} in reverse letter order
    """
    succ = _successors(activities)
    ls_lf = {}
    for label in sorted(activities, reverse=True):
        s = succ[label]
        lf = completion_time if not s else min(ls_lf[t][0] for t in s)
        ls = lf - activities[label]["duration"]
        ls_lf[label] = (ls, lf)
    return ls_lf


def _format_network_text(activities):
    """Format the network as a rigid text block: one line per activity.

    Args:
        activities: {label: {"duration": int, "predecessors": [label, ...]}}

    Returns:
        str with "Activity X: duration N days, ..." per activity
    """
    lines = []
    for label in sorted(activities):
        a = activities[label]
        duration = a["duration"]
        preds = a["predecessors"]
        if not preds:
            line = f"Activity {label}: duration {duration} {'day' if duration == 1 else 'days'}, no predecessors."
        else:
            pred_str = ", ".join(preds)
            line = (
                f"Activity {label}: duration {duration} {'day' if duration == 1 else 'days'}, predecessors: {pred_str}."
            )
        lines.append(line)
    return " ".join(lines)


# --- use_forward_and_backward_scanning_to_determine_the (ACMGM105) -------


def _verify_easy_est_chain(es_ef, answer, target):
    """Independent check for easy tier (EST of chain sink)."""
    check_equivalent(es_ef[target][0], answer, "answer must be ES of sink activity")
    # Second route: ES of D is the sum of all predecessors' durations
    # In a chain A:[], B:[A], C:[B], D:[C], the answer is always EF of C
    check_equivalent(es_ef["D"][0], es_ef["C"][1], "in a chain, ES of sink equals EF of previous")


def _verify_medium_est_merge(es_ef, answer, target):
    """Independent check for medium tier (EST of merge point E)."""
    check_equivalent(es_ef[target][0], answer, "answer must be ES of merge activity E")
    # Second route: ES of E is max(EF of C, EF of D)
    max_pred_finish = max(es_ef["C"][1], es_ef["D"][1])
    check_equivalent(es_ef["E"][0], max_pred_finish, "ES of merge equals max EF of predecessors")


def _verify_hard_lst_nonchain(ls_lf, es_ef, activities, answer, target):
    """Independent check for hard tier (LST of non-sink activity B).

    The answer is LS of B, computed as min(LS of successors) - duration.
    """
    check_equivalent(ls_lf[target][0], answer, "answer must be LS of activity B")
    # Second route: LS of B should equal min of successors' LS minus B's duration
    # Successors of B are D and E
    min_succ_ls = min(ls_lf["D"][0], ls_lf["E"][0])
    expected_ls = min_succ_ls - activities[target]["duration"]
    check_equivalent(answer, expected_ls, "LS(B) must be min(LS of successors) - duration(B)")


def _verify_completion_time_via_sinks(es_ef, activities, completion_time):
    """Independent check for completion time using only sink activities.

    Sink = activity with no successors. For fixed topologies:
    - easy: sink is D (only one)
    - medium: sink is E (only one)
    - hard: sink is F (only one)
    """
    # Find sinks dynamically
    succ = _successors(activities)
    sinks = [label for label in activities if not succ[label]]
    check(len(sinks) >= 1, "network must have at least one sink")
    sink_completion = max(es_ef[label][1] for label in sinks)
    check_equivalent(completion_time, sink_completion, "completion time must equal max EF of sink activities")


# ---- Easy tier topology (4 activities, chain) ----


def _generate_easy_est_chain(rng):
    """Easy tier: 4-activity chain. Durations drawn independently.

    Topology: A:[] -> B:[A] -> C:[B] -> D:[C]
    Ask for: ES of D (the sink)
    """
    durations = {label: rng.randint(2, 9) for label in "ABCD"}
    activities = {
        "A": {"duration": durations["A"], "predecessors": []},
        "B": {"duration": durations["B"], "predecessors": ["A"]},
        "C": {"duration": durations["C"], "predecessors": ["B"]},
        "D": {"duration": durations["D"], "predecessors": ["C"]},
    }

    es_ef = _forward_pass(activities)
    completion_time = max(ef for _, ef in es_ef.values())
    _verify_completion_time_via_sinks(es_ef, activities, completion_time)

    answer = es_ef["D"][0]  # ES of D
    _verify_easy_est_chain(es_ef, answer, "D")
    answer_text = str(answer)

    # Distractors
    distractor_1 = es_ef["C"][0]  # gives C's ES instead of C's EF
    distractor_2 = es_ef["D"][1]  # gives D's EF instead of ES
    distractor_3 = durations["D"]  # gives D's duration (ignores chain)

    candidates = [str(distractor_1), str(distractor_2), str(distractor_3)]
    distractors = pick_distractors(answer_text, candidates)
    text = (
        f"The project starts at time 0. "
        f"{_format_network_text(activities)} "
        f"What is the earliest starting time of activity D, in days?"
    )

    return dict(
        text=text,
        latex=r"\text{Chain of 4 activities; }ES(D) = ?",
        answer=answer,
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[
            f"Forward pass: ES(A)=0, EF(A)={es_ef['A'][1]}; ES(B)={es_ef['B'][0]}, EF(B)={es_ef['B'][1]}; "
            f"ES(C)={es_ef['C'][0]}, EF(C)={es_ef['C'][1]}; ES(D)={es_ef['D'][0]}."
        ],
        distractors=distractors,
    )


# ---- Medium tier topology (5 activities, one merge) ----


def _generate_medium_est_merge(rng):
    """Medium tier: 5-activity network with one merge.

    Topology: A:[] -> B:[A], C:[A] -> D:[B] -> E:[C,D] (merge)
    Ask for: ES of E (the merge, sink)
    """
    durations = {label: rng.randint(2, 9) for label in "ABCDE"}
    activities = {
        "A": {"duration": durations["A"], "predecessors": []},
        "B": {"duration": durations["B"], "predecessors": ["A"]},
        "C": {"duration": durations["C"], "predecessors": ["A"]},
        "D": {"duration": durations["D"], "predecessors": ["B"]},
        "E": {"duration": durations["E"], "predecessors": ["C", "D"]},
    }

    es_ef = _forward_pass(activities)
    completion_time = max(ef for _, ef in es_ef.values())
    _verify_completion_time_via_sinks(es_ef, activities, completion_time)

    answer = es_ef["E"][0]  # ES of E (merge point)
    _verify_medium_est_merge(es_ef, answer, "E")
    answer_text = str(answer)

    # Distractors
    distractor_1 = es_ef["C"][1] + es_ef["D"][1]  # sums predecessors' EF instead of taking max
    distractor_2 = min(es_ef["C"][1], es_ef["D"][1])  # takes min instead of max
    distractor_3 = es_ef["E"][1]  # gives E's EF instead of ES

    candidates = [str(distractor_1), str(distractor_2), str(distractor_3)]
    distractors = pick_distractors(answer_text, candidates)
    text = (
        f"The project starts at time 0. "
        f"{_format_network_text(activities)} "
        f"What is the earliest starting time of activity E, in days?"
    )

    return dict(
        text=text,
        latex=r"\text{5-activity network; }ES(E) = ?",
        answer=answer,
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[
            f"Forward pass: ES(A)=0, EF(A)={es_ef['A'][1]}; ES(B)={es_ef['B'][0]}, EF(B)={es_ef['B'][1]}; "
            f"ES(C)={es_ef['C'][0]}, EF(C)={es_ef['C'][1]}; ES(D)={es_ef['D'][0]}, EF(D)={es_ef['D'][1]}; "
            f"ES(E)=max({es_ef['C'][1]}, {es_ef['D'][1]})={answer}."
        ],
        distractors=distractors,
    )


# ---- Hard tier topology (6 activities, two merges) ----


def _generate_hard_lst_twopath(rng):
    """Hard tier: 6-activity network with one branch and two merges.

    Topology: A:[] -> B:[A], C:[A] -> D:[B] -> E:[B,C] -> F:[D,E]
    Ask for: LS of B (non-sink activity)
    """
    durations = {label: rng.randint(2, 9) for label in "ABCDEF"}
    activities = {
        "A": {"duration": durations["A"], "predecessors": []},
        "B": {"duration": durations["B"], "predecessors": ["A"]},
        "C": {"duration": durations["C"], "predecessors": ["A"]},
        "D": {"duration": durations["D"], "predecessors": ["B"]},
        "E": {"duration": durations["E"], "predecessors": ["B", "C"]},
        "F": {"duration": durations["F"], "predecessors": ["D", "E"]},
    }

    es_ef = _forward_pass(activities)
    completion_time = max(ef for _, ef in es_ef.values())
    _verify_completion_time_via_sinks(es_ef, activities, completion_time)

    ls_lf = _backward_pass(activities, es_ef, completion_time)
    answer = ls_lf["B"][0]  # LS of B
    _verify_hard_lst_nonchain(ls_lf, es_ef, activities, answer, "B")
    answer_text = str(answer)

    # Distractors
    distractor_1 = es_ef["B"][0]  # gives B's ES instead of LS (skips backward scan)
    distractor_2 = ls_lf["B"][1]  # gives B's LF instead of LS (forgets to subtract duration)
    distractor_3 = max(ls_lf["D"][0], ls_lf["E"][0]) - durations["B"]  # takes max instead of min

    candidates = [str(distractor_1), str(distractor_2), str(distractor_3)]
    distractors = pick_distractors(answer_text, candidates)
    # Verify no mention of "float" or "slack" (scope boundaries)
    text = (
        f"The project starts at time 0. "
        f"{_format_network_text(activities)} "
        f"What is the latest starting time of activity B, in days, without delaying "
        f"the earliest possible completion of the project?"
    )

    # Check that "float" and "slack" don't appear
    check("float" not in text.lower(), "text must not mention 'float'")
    check("slack" not in text.lower(), "text must not mention 'slack'")

    return dict(
        text=text,
        latex=r"\text{6-activity network; }LS(B) = ?",
        answer=answer,
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[
            f"Forward pass yields completion time = {completion_time}.",
            f"Backward pass: LS(B) = min(LS(D), LS(E)) - duration(B) = min({ls_lf['D'][0]}, {ls_lf['E'][0]}) - {durations['B']} = {answer}.",
        ],
        distractors=distractors,
    )


@generator(
    "use_forward_and_backward_scanning_to_determine_the",
    level=11,
    presets={
        "easy": dict(tier="chain_est"),
        "medium": dict(tier="merge_est"),
        "hard": dict(tier="nonchain_lst"),
    },
)
def use_forward_and_backward_scanning_to_determine_the(rng, tier="chain_est"):
    """Use forward and backward scanning to determine EST and LST for project activities."""
    if tier == "chain_est":
        return _generate_easy_est_chain(rng)
    elif tier == "merge_est":
        return _generate_medium_est_merge(rng)
    elif tier == "nonchain_lst":
        return _generate_hard_lst_twopath(rng)
    else:
        raise ValueError(f"unknown tier: {tier!r}")


# --- use_the_critical_path_to_determine_the_minimum (ACMGM107) -------


def _verify_easy_completion_chain(es_ef, completion_time, answer):
    """Independent check for easy tier (completion time of chain)."""
    check_equivalent(es_ef["D"][1], answer, "answer must be EF of sink D")
    check_equivalent(completion_time, answer, "completion time equals EF of sink")


def _verify_medium_completion_merge(es_ef, completion_time, answer):
    """Independent check for medium tier (completion time with merge)."""
    check_equivalent(es_ef["E"][1], answer, "answer must be EF of sink E")
    check_equivalent(completion_time, answer, "completion time equals EF of sink")


def _verify_hard_completion_twopath(es_ef, completion_time, answer):
    """Independent check for hard tier (completion time with two paths to sink)."""
    check_equivalent(es_ef["F"][1], answer, "answer must be EF of sink F")
    check_equivalent(completion_time, answer, "completion time equals EF of sink")


def _generate_easy_completion_chain(rng):
    """Easy tier: 4-activity chain. Completion time is sum of all durations.

    Topology: A:[] -> B:[A] -> C:[B] -> D:[C]
    Answer: ES(D) + duration(D) = completion time
    """
    durations = {label: rng.randint(2, 9) for label in "ABCD"}
    activities = {
        "A": {"duration": durations["A"], "predecessors": []},
        "B": {"duration": durations["B"], "predecessors": ["A"]},
        "C": {"duration": durations["C"], "predecessors": ["B"]},
        "D": {"duration": durations["D"], "predecessors": ["C"]},
    }

    es_ef = _forward_pass(activities)
    completion_time = max(ef for _, ef in es_ef.values())
    _verify_completion_time_via_sinks(es_ef, activities, completion_time)

    answer = completion_time
    _verify_easy_completion_chain(es_ef, completion_time, answer)
    answer_text = str(answer)

    # Distractors
    distractor_1 = es_ef["C"][1]  # stops one activity early (omits D's duration)
    distractor_2 = completion_time - durations["A"]  # omits first activity's duration
    distractor_3 = sum(durations.values()) + durations["B"]  # double-counts B

    candidates = [str(distractor_1), str(distractor_2), str(distractor_3)]
    distractors = pick_distractors(answer_text, candidates)
    text = (
        f"The project starts at time 0. "
        f"{_format_network_text(activities)} "
        f"What is the minimum time to complete the project, in days?"
    )

    return dict(
        text=text,
        latex=r"\text{Chain of 4 activities; completion time = ?}",
        answer=answer,
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[
            f"Forward pass: ES(A)=0, EF(A)={es_ef['A'][1]}; ES(B)={es_ef['B'][0]}, EF(B)={es_ef['B'][1]}; "
            f"ES(C)={es_ef['C'][0]}, EF(C)={es_ef['C'][1]}; ES(D)={es_ef['D'][0]}, EF(D)={es_ef['D'][1]}. "
            f"Completion time = {answer}."
        ],
        distractors=distractors,
    )


def _generate_medium_completion_merge(rng):
    """Medium tier: 5-activity network with one merge.

    Topology: A:[] -> B:[A], C:[A] -> D:[B] -> E:[C,D]
    Answer: EF(E) = max(paths) + duration(E)
    """
    durations = {label: rng.randint(2, 9) for label in "ABCDE"}
    activities = {
        "A": {"duration": durations["A"], "predecessors": []},
        "B": {"duration": durations["B"], "predecessors": ["A"]},
        "C": {"duration": durations["C"], "predecessors": ["A"]},
        "D": {"duration": durations["D"], "predecessors": ["B"]},
        "E": {"duration": durations["E"], "predecessors": ["C", "D"]},
    }

    es_ef = _forward_pass(activities)
    completion_time = max(ef for _, ef in es_ef.values())
    _verify_completion_time_via_sinks(es_ef, activities, completion_time)

    answer = completion_time
    _verify_medium_completion_merge(es_ef, completion_time, answer)
    answer_text = str(answer)

    # Distractors
    distractor_1 = sum(durations.values())  # sums all five durations (ignores branch)
    distractor_2 = min(es_ef["C"][1], es_ef["D"][1]) + durations["E"]  # takes shorter path
    distractor_3 = max(es_ef["C"][1], es_ef["D"][1])  # forgets to add E's duration

    candidates = [str(distractor_1), str(distractor_2), str(distractor_3)]
    distractors = pick_distractors(answer_text, candidates)
    text = (
        f"The project starts at time 0. "
        f"{_format_network_text(activities)} "
        f"What is the minimum time to complete the project, in days?"
    )

    return dict(
        text=text,
        latex=r"\text{5-activity network; completion time = ?}",
        answer=answer,
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[
            f"Forward pass: ES(A)=0, EF(A)={es_ef['A'][1]}; ES(B)={es_ef['B'][0]}, EF(B)={es_ef['B'][1]}; "
            f"ES(C)={es_ef['C'][0]}, EF(C)={es_ef['C'][1]}; ES(D)={es_ef['D'][0]}, EF(D)={es_ef['D'][1]}; "
            f"ES(E)={es_ef['E'][0]}, EF(E)={es_ef['E'][1]}. "
            f"Completion time = {answer}."
        ],
        distractors=distractors,
    )


def _generate_hard_completion_twopath(rng):
    """Hard tier: 6-activity network with two paths to sink F.

    Topology: A:[] -> B:[A], C:[A] -> D:[B] -> E:[B,C] -> F:[D,E]
    Answer: EF(F) = critical path length through network
    """
    durations = {label: rng.randint(2, 9) for label in "ABCDEF"}
    activities = {
        "A": {"duration": durations["A"], "predecessors": []},
        "B": {"duration": durations["B"], "predecessors": ["A"]},
        "C": {"duration": durations["C"], "predecessors": ["A"]},
        "D": {"duration": durations["D"], "predecessors": ["B"]},
        "E": {"duration": durations["E"], "predecessors": ["B", "C"]},
        "F": {"duration": durations["F"], "predecessors": ["D", "E"]},
    }

    es_ef = _forward_pass(activities)
    completion_time = max(ef for _, ef in es_ef.values())
    _verify_completion_time_via_sinks(es_ef, activities, completion_time)

    answer = completion_time
    _verify_hard_completion_twopath(es_ef, completion_time, answer)
    answer_text = str(answer)

    # Distractors
    distractor_1 = sum(durations.values())  # sums all six durations
    distractor_2 = durations["A"] + durations["B"] + durations["D"] + durations["F"]  # A-B-D-F path only
    distractor_3 = es_ef["E"][1]  # stops at E instead of continuing to F

    candidates = [str(distractor_1), str(distractor_2), str(distractor_3)]
    distractors = pick_distractors(answer_text, candidates)
    text = (
        f"The project starts at time 0. "
        f"{_format_network_text(activities)} "
        f"What is the minimum time to complete the project, in days?"
    )

    return dict(
        text=text,
        latex=r"\text{6-activity network; completion time = ?}",
        answer=answer,
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[
            f"Forward pass: ES(A)=0, EF(A)={es_ef['A'][1]}; ES(B)={es_ef['B'][0]}, EF(B)={es_ef['B'][1]}; "
            f"ES(C)={es_ef['C'][0]}, EF(C)={es_ef['C'][1]}; ES(D)={es_ef['D'][0]}, EF(D)={es_ef['D'][1]}; "
            f"ES(E)={es_ef['E'][0]}, EF(E)={es_ef['E'][1]}; "
            f"ES(F)={es_ef['F'][0]}, EF(F)={es_ef['F'][1]}. "
            f"Completion time = {answer}."
        ],
        distractors=distractors,
    )


@generator(
    "use_the_critical_path_to_determine_the_minimum",
    level=11,
    presets={
        "easy": dict(tier="chain_completion"),
        "medium": dict(tier="merge_completion"),
        "hard": dict(tier="twopath_completion"),
    },
)
def use_the_critical_path_to_determine_the_minimum(rng, tier="chain_completion"):
    """Use the critical path to determine the minimum time to complete the project."""
    if tier == "chain_completion":
        return _generate_easy_completion_chain(rng)
    elif tier == "merge_completion":
        return _generate_medium_completion_merge(rng)
    elif tier == "twopath_completion":
        return _generate_hard_completion_twopath(rng)
    else:
        raise ValueError(f"unknown tier: {tier!r}")
