"""Generates `pythagoras_truss_chain` questions: carry one computed length through two or three dependent right-
triangle calculations of a truss.

Pythagoras_truss_chain.

The learner already has `pythagorean-hypotenuse`/`pythagorean-leg` (single-step Pythagoras); this skill's
whole contract is tracking one computed length across two or three dependent right-triangle computations, the
shape of an exam-style item's labelled-truss question.

Picking one leg value h shared by all L triples
is impossible for L=3 without collapsing h1 and h2 to the same value; this
implements a walk instead: T1 free, then each subsequent triangle drawn from
the leg-value index at the *previous* stage's found leg, excluding every
triple already used in this chain.

  fresh hypotenuse is c2; find the other leg h2.
- Stage L (closing): the previous stage's found leg is one leg, a fresh own
  other leg a_L is given; find the hypotenuse c_L.

Every triangle comes from an exact scaled Pythagorean triple -- integer
arithmetic throughout, no float/sqrt of a non-perfect-square ever appears.
"""

from __future__ import annotations

import math

from ..core import check, check_distinct, generator, pick_distractors

# A mix of primitive triples and small multiples of each other, so that
# scaling by k in 1..3 makes leg values collide with each other on purpose.
_BANK = [
    (3, 4, 5),
    (5, 12, 13),
    (6, 8, 10),
    (8, 15, 17),
    (7, 24, 25),
    (20, 21, 29),
    (9, 40, 41),
    (12, 35, 37),
    (9, 12, 15),
    (12, 16, 20),
    (15, 20, 25),
    (16, 30, 34),
    (21, 28, 35),
    (10, 24, 26),
    (18, 24, 30),
]


def _build_pool() -> list[tuple[int, int, int]]:
    pool = set()
    for a, b, c in _BANK:
        for k in range(1, 4):
            pool.add((a * k, b * k, c * k))
    return sorted(pool)


_POOL = _build_pool()


def _build_index() -> dict[int, list[tuple[tuple[int, int, int], int, int]]]:
    """leg_value -> [(triple, other_leg, hyp), ...] for every scaled triple
    where leg_value is one of its two legs."""
    index: dict[int, list[tuple[tuple[int, int, int], int, int]]] = {}
    for triple in _POOL:
        a, b, c = triple
        index.setdefault(a, []).append((triple, b, c))
        index.setdefault(b, []).append((triple, a, c))
    return index


_INDEX = _build_index()

_MAX_TRIES = 5000


def _distinct_options(leg_value: int, exclude: set) -> list[tuple[tuple[int, int, int], int, int]]:
    """Options at a junction leg value, deduped by triple identity and with
    every triple already used in this chain removed."""
    by_triple: dict[tuple[int, int, int], tuple[tuple[int, int, int], int, int]] = {}
    for triple, other_leg, hyp in _INDEX.get(leg_value, []):
        if triple in exclude:
            continue
        by_triple[triple] = (triple, other_leg, hyp)
    return sorted(by_triple.values())


def _draw_chain(rng, length: int) -> dict:
    """Draw one length-`length` (2 or 3) truss chain.

    Returns a1, c1, h1 (stage 1), c2, h2 (stage 2, length==3 only),
    a_last, c_last (closing stage) -- all plain ints, reject-and-redraw
    until every uniqueness/constructibility condition holds.
    """
    tries = 0
    while True:
        tries += 1
        check(tries <= _MAX_TRIES, "pythagoras_truss_chain: exceeded max tries drawing a chain")

        t1 = rng.choice(_POOL)
        leg_x, leg_y, c1 = t1
        if rng.random() < 0.5:
            leg_x, leg_y = leg_y, leg_x
        a1, h1 = leg_x, leg_y
        used = {t1}
        shared = h1

        c2 = h2 = None
        if length == 3:
            opts = _distinct_options(shared, used)
            if not opts:
                continue
            t2, h2, c2 = rng.choice(opts)
            used.add(t2)
            shared = h2

        opts_last = _distinct_options(shared, used)
        if not opts_last:
            continue
        t_last, a_last, c_last = rng.choice(opts_last)
        used.add(t_last)

        # Reject if any two of the chosen triples are the same triple
        # (already impossible via `used`, but assert the count directly).
        if len(used) != length:
            continue

        # Reject if any two stages share an identical (given-pair) --
        # each stage's given values must be its own.
        given_pairs = [(a1, c1)]
        if length == 3:
            given_pairs.append((h1, c2))
        given_pairs.append((shared if length == 2 else h2, a_last))
        if len(set(given_pairs)) != len(given_pairs):
            continue

        # Reject unless every value this generator might report as an
        # answer or a "wrong-stage-value"/"an earlier stage's hypotenuse"
        # distractor is pairwise distinct: h1, (h2,) c_last and c1.
        target_pool = {h1, c_last} | ({h2} if length == 3 else set())
        watch = target_pool | {c1}
        if len(watch) != len(target_pool) + 1:
            continue

        return dict(a1=a1, c1=c1, h1=h1, c2=c2, h2=h2, a_last=a_last, c_last=c_last)


def _render_text(length: int, values: dict, final_sentence: str) -> str:
    parts = ["A support truss is modelled as a chain of right-angled triangles."]
    parts.append(
        f"Triangle 1: one leg is {values['a1']} m and the hypotenuse is "
        f"{values['c1']} m. Find the other leg — call it h1."
    )
    if length == 3:
        parts.append(
            f"Triangle 2: h1 (found above) is one leg, and its own "
            f"hypotenuse is {values['c2']} m. Find Triangle 2's other leg "
            f"— call it h2."
        )
    shared_label = f"h{length - 1}"
    parts.append(
        f"Triangle {length}: {shared_label} (found above) is one leg, and "
        f"its own other leg is {values['a_last']} m. {final_sentence}"
    )
    return " ".join(parts)


def _verify_chain_identities(length: int, values: dict) -> None:
    """The independent check: re-derive every stage's Pythagorean identity
    with a fresh `math.isqrt` computation -- a separate computation path
    from the triple-bank lookup that already asserts it true by
    construction (integer arithmetic throughout; no float/sqrt anywhere).
    Also asserts every side is strictly positive and the hypotenuse
    strictly exceeds each leg (the constructibility requirement). Factored
    out of the generator body so a test can call it directly on a
    corrupted `values` dict and assert it raises."""
    a1, c1, h1 = values["a1"], values["c1"], values["h1"]
    c2, h2 = values["c2"], values["h2"]
    a_last, c_last = values["a_last"], values["c_last"]
    shared_prev = h2 if length == 3 else h1

    diff1 = c1**2 - a1**2
    root1 = math.isqrt(diff1) if diff1 >= 0 else -1
    check(root1 * root1 == diff1 and root1 == h1, "pythagoras_truss_chain: stage 1 leg does not re-derive exactly")
    check(a1**2 + h1**2 == c1**2, "pythagoras_truss_chain: stage 1 Pythagorean identity fails")
    check(a1 > 0 and h1 > 0 and c1 > a1 and c1 > h1, "pythagoras_truss_chain: stage 1 triangle is not constructible")

    if length == 3:
        diff2 = c2**2 - h1**2
        root2 = math.isqrt(diff2) if diff2 >= 0 else -1
        check(root2 * root2 == diff2 and root2 == h2, "pythagoras_truss_chain: stage 2 leg does not re-derive exactly")
        check(h1**2 + h2**2 == c2**2, "pythagoras_truss_chain: stage 2 Pythagorean identity fails")
        check(
            h1 > 0 and h2 > 0 and c2 > h1 and c2 > h2, "pythagoras_truss_chain: stage 2 triangle is not constructible"
        )

    root_last = math.isqrt(a_last**2 + shared_prev**2)
    check(
        root_last * root_last == a_last**2 + shared_prev**2 and root_last == c_last,
        "pythagoras_truss_chain: closing stage hypotenuse does not re-derive exactly",
    )
    check(a_last**2 + shared_prev**2 == c_last**2, "pythagoras_truss_chain: closing stage Pythagorean identity fails")
    check(
        a_last > 0 and shared_prev > 0 and c_last > a_last and c_last > shared_prev,
        "pythagoras_truss_chain: closing stage triangle is not constructible",
    )


@generator(
    "pythagoras_truss_chain",
    level=11,
    presets={
        "easy": dict(length=2, target_mode="final"),
        "medium": dict(length=3, target_mode="final"),
        "hard": dict(length=3, target_mode="random"),
    },
)
def pythagoras_truss_chain(rng, length=2, target_mode="final"):
    """A chain of `length` (2 or 3) dependent right-angled triangles: each
    peel stage finds a leg that feeds the next triangle; the closing stage
    finds the final hypotenuse. `target_mode="random"` (hard tier) asks for
    a uniformly drawn quantity from {h1, h2, cL} instead of always the
    final hypotenuse -- the genuine escalation over medium, which always
    asks for cL: the learner must track *which* quantity is wanted mid-chain
    rather than pattern-matching "report the last number computed"."""
    values = _draw_chain(rng, length)
    a1, c1, h1 = values["a1"], values["c1"], values["h1"]
    c2, h2 = values["c2"], values["h2"]
    a_last, c_last = values["a_last"], values["c_last"]
    shared_prev = h2 if length == 3 else h1

    _verify_chain_identities(length, values)

    target_pool = {"h1": h1}
    if length == 3:
        target_pool["h2"] = h2
    target_pool["cL"] = c_last

    if target_mode == "final":
        target_key = "cL"
    else:
        target_key = rng.choice(list(target_pool.keys()))

    answer_val = target_pool[target_key]
    answer_text = str(answer_val)

    if target_key == "cL":
        final_sentence = "Find its hypotenuse, in m."
        stopped_at_intermediate = shared_prev
        added_instead = a_last + shared_prev
        wrong_stage_value = c1 if target_mode == "final" else h1
        check_distinct(answer_val, stopped_at_intermediate, msg="stopped-at-intermediate distractor equals the answer")
        check_distinct(answer_val, added_instead, msg="added-instead-of-pythagoras distractor equals the answer")
        check_distinct(answer_val, wrong_stage_value, msg="wrong-stage-value distractor equals the answer")
        candidates = [str(stopped_at_intermediate), str(added_instead), str(wrong_stage_value)]
    elif target_key == "h1":
        final_sentence = "Find the length of h1, in m."
        added_instead = math.isqrt(a1**2 + c1**2)
        wrong_stage_value = h2
        # stopped-at-intermediate is not exercisable here: the target IS the
        # intermediate value, so "reports h_{L-1}" collapses to the answer.
        check_distinct(answer_val, added_instead, msg="added-instead-of-pythagoras distractor equals the answer")
        check_distinct(answer_val, wrong_stage_value, msg="wrong-stage-value distractor equals the answer")
        candidates = [str(added_instead), str(wrong_stage_value)]
    else:  # target_key == "h2"
        final_sentence = "Find the length of h2, in m."
        added_instead = math.isqrt(h1**2 + c2**2)
        wrong_stage_value = c_last
        check_distinct(answer_val, added_instead, msg="added-instead-of-pythagoras distractor equals the answer")
        check_distinct(answer_val, wrong_stage_value, msg="wrong-stage-value distractor equals the answer")
        candidates = [str(added_instead), str(wrong_stage_value)]

    distractors = pick_distractors(answer_text, candidates)

    text = _render_text(length, values, final_sentence)

    working = [
        f"Triangle 1: {a1}^2 + h1^2 = {c1}^2, so h1 = {h1}.",
    ]
    if length == 3:
        working.append(f"Triangle 2: h1^2 + h2^2 = {c2}^2, so h2 = {h2}.")
    working.append(f"Triangle {length}: {a_last}^2 + {shared_prev}^2 = c{length}^2, so c{length} = {c_last}.")
    working.append(f"Answer: {answer_val} m.")

    return {
        "text": text,
        "latex": text,
        "answer": answer_val,
        "answer_text": answer_text,
        "answer_latex": answer_text,
        "distractors": distractors,
        "working": working,
    }
