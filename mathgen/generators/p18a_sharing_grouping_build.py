"""Generates `p18a_sharing_grouping` questions: decide whether a context is equal sharing or grouping and compute
the result.

Scope: "Given a stated practical context, determine whether it
is an equal-sharing or grouping situation and compute the result."

## Curriculum note

An earlier scoping note called grouping "multiplication", but in AC9M1N06 both
readings are DIVISION: sharing is partitive (a fixed number of
recipients/containers; the unknown is how many go in EACH one) and grouping
is quotitive (a fixed number per container; the unknown is HOW MANY
containers are needed). This module builds to the curriculum, not to that
loose paraphrase.

## Why this is `answer: mcq`, not a bare number

Sharing 12 things among 3 containers and grouping 12 things into containers
of 3 give the SAME number (4), so a bare number would never assess the
sharing-vs-grouping decision this skill adds over its prerequisites (which
already compute a fixed array operation they are told to perform). Every
option here is therefore "<number> <what it counts>": `4 apples in each bag`
(partitive) or `4 bags` (quotitive) -- never a bare number, so a learner who
gets the right number with the wrong reading picks a genuinely different,
wrong, option (the `swapped-reading` misconception below).

## Separation from `f10-multiply-divide-one-digit-arrays`, `equal-groups`,
`fair-sharing`

Those three all NAME the operation up front ("shared equally into", "arranged into equal groups of", "Share...
equally between") and ask a fixed stem ("How many... in each group?" / "How many equal groups?") for a bare
number. This skill's prompt never uses that stem -- it always asks the neutral "What is the result?" (or, at
exam, "... for one of those <containers>?") -- and the DECISION (which reading, and hence which unit the
number counts) is carried entirely by whether the stated fixed quantity is a container COUNT ("between {D}
bags") or a per-container SIZE ("each bag holds {D} apples"), never by an operation verb at medium/hard.

## Difficulty progression (genuine reasoning escalation, not size)

- easy: the context wording names the action plainly ("shares ... equally
  between" / "packs ... into ... that each hold"). DECISION: read which noun
  the stated number is attached to (a container count, or a per-container
  size) to choose partitive or quotitive, then divide.
- medium: the connector verb is neutral ("spread/split/divided evenly
  between" / "... and each ... will hold ..."); no verb names sharing or
  grouping at all. DECISION: the same reading-from-attachment decision as easy, now
  unavailable via a verb -- the learner must find the fixed quantity's own
  noun (container count vs per-container size) unaided by a naming verb.
- hard: the total does not divide the stated quantity evenly, so a leftover
  must be counted and reported IN the chosen unit ("... left over"), and the
  wrong-reading option, computed on the same numbers, would just be a
  legitimately different-but-plausible-looking wrong count. DECISION: the
  reading decision from medium PLUS deciding what the remainder means in the
  chosen reading (extra items unshared, or items that do not fill a further
  container) -- a step medium's exact division never raises.
- exam: two-step -- a total is first shared equally between a stated number
  of containers (always the partitive scene-setter), and then EACH share is
  itself either shared again or grouped again, with the sharing-vs-grouping
  decision now applying to the SECOND step only. DECISION: resolve the first
  division to get one share, then apply the same reading decision a second
  time to that share, rather than to the original total -- a genuinely new
  two-stage resolution no lower tier requires.

## Independent check for every draw

`_check_division` recomputes the answer's own division by multiplying the
divisor and quotient back together and adding any remainder, confirming it
reconstructs the original dividend -- a different route (multiplication) from
the one that produced the quotient (`divmod`). `check_pairwise_distinct` is
called on a SIGNATURE tuple `(reading, number, remainder)` per option (never
on the raw display text, which is free-form English that `sympify` cannot
parse) -- the true underlying value distinctness -- plus a plain string
dedup for the rendered options actually shown. The oracle
 parses `question.text` with
its own regexes over the same fixed vocabulary and re-derives (situation,
total, divisor, quotient, remainder) from scratch, then re-derives the
expected answer text's regex-matched (reading, number, remainder) triple
independently, never importing this module.
"""

from __future__ import annotations

from ..core import check, check_pairwise_distinct, generator, latex_answer, phrase

TOPIC = "p18a_sharing_grouping"
_TIERS = ("easy", "medium", "hard", "exam")

_ITEMS = [
    ("apple", "apples"),
    ("sticker", "stickers"),
    ("marble", "marbles"),
    ("counter", "counters"),
]
_CONTAINERS = [
    ("bag", "bags"),
    ("box", "boxes"),
    ("basket", "baskets"),
    ("cup", "cups"),
]
_NAMES = ("Mia", "Sam", "Ali", "Zoe", "Ben", "Leo")


def _unit_word(n: int, sg: str, pl: str) -> str:
    return sg if n == 1 else pl


def _each_text(q: int, item_pl: str, container_sg: str, item_sg: str, remainder: int = 0) -> str:
    text = f"{q} {item_pl} in each {container_sg}"
    if remainder:
        text += f", {remainder} {_unit_word(remainder, item_sg, item_pl)} left over"
    return text


def _count_text(q: int, container_sg: str, container_pl: str, item_sg: str, item_pl: str, remainder: int = 0) -> str:
    if remainder:
        return f"{q} full {container_pl}, {remainder} {_unit_word(remainder, item_sg, item_pl)} left over"
    return f"{q} {_unit_word(q, container_sg, container_pl)}"


def _check_division(total: int, divisor: int, quotient: int, remainder: int, msg: str) -> None:
    """Independent route: multiply divisor * quotient + remainder back and
    confirm it reconstructs the dividend -- never re-using `divmod`'s own
    result the way the construction obtained it."""
    check(divisor * quotient + remainder == total, msg)
    check(0 <= remainder < divisor, msg)


def _candidates_for_share(total, divisor, q, remainder, item_sg, item_pl, container_sg, container_pl):
    """Answer is partitive: `_each_text(q, ...)`. Returns (answer_sig, answer_text,
    [(sig, text), ...]) for the misconception candidates."""
    answer_sig = ("share", q, remainder)
    answer_text = _each_text(q, item_pl, container_sg, item_sg, remainder)

    raw = []
    # swapped-reading: the classic partitive/quotitive confusion -- the SAME
    # quotient and remainder as the true answer, but reported as if it were
    # a quotitive (group-count) answer instead of a partitive (per-container)
    # one -- exactly what a learner who divides correctly but misreads the
    # situation would pick.
    raw.append((("group", q, remainder), _count_text(q, container_sg, container_pl, item_sg, item_pl, remainder)))
    # restates-total: reports the given total as if it already were the answer.
    raw.append((("total", total, 0), f"{total} {item_pl}"))
    # off-by-one on the quotient, same reading, no remainder change.
    raw.append((("share", q + 1, remainder), _each_text(q + 1, item_pl, container_sg, item_sg, remainder)))
    return answer_sig, answer_text, raw


def _candidates_for_group(total, divisor, q, remainder, item_sg, item_pl, container_sg, container_pl):
    """Answer is quotitive: `_count_text(q, ...)`."""
    answer_sig = ("group", q, remainder)
    answer_text = _count_text(q, container_sg, container_pl, item_sg, item_pl, remainder)

    raw = []
    # swapped-reading: the SAME quotient and remainder as the true answer,
    # reported as if it were a partitive (per-container) answer instead of a
    # quotitive (group-count) one.
    raw.append((("share", q, remainder), _each_text(q, item_pl, container_sg, item_sg, remainder)))
    # restates-total.
    raw.append((("total", total, 0), f"{total} {item_pl}"))
    # off-by-one on the quotient, same reading, no remainder change.
    raw.append(
        (("group", q + 1, remainder), _count_text(q + 1, container_sg, container_pl, item_sg, item_pl, remainder))
    )
    return answer_sig, answer_text, raw


def _finalise(text, answer_sig, answer_text, raw_candidates):
    sigs = [answer_sig]
    texts_seen = {answer_text}
    distractors = []
    for sig, txt in raw_candidates:
        if sig in sigs or txt in texts_seen:
            continue
        sigs.append(sig)
        texts_seen.add(txt)
        distractors.append(txt)
    check(len(distractors) >= 1, "p18a_sharing_grouping: no distractor survived dedup")
    check_pairwise_distinct(*sigs, msg="p18a_sharing_grouping: two options share the same underlying value")

    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=answer_text,
        answer_text=answer_text,
        answer_latex=latex_answer(answer_text, plain=True),
        distractors=distractors,
        working=[f"answer: {answer_text}"],
    )


# ============================================================================
# Templates -- "between {D} {container_pl}" always marks the partitive
# (container-count-fixed) reading; "hold(s) {D} {item_pl}" always marks the
# quotitive (per-container-size-fixed) reading. These substrings are the
# oracle's own parse markers.
# ============================================================================

_EASY_SHARE = (
    "{name} shares {total} {item_pl} equally between {divisor} {container_pl}. What is the result?",
    "There are {total} {item_pl}. {name} shares them equally between {divisor} {container_pl}. What is the result?",
    "{name} has {total} {item_pl} and shares them equally between {divisor} {container_pl}. What is the result?",
)
_EASY_GROUP = (
    "{name} puts {total} {item_pl} into {container_pl} that each hold {divisor} {item_pl}. What is the result?",
    "There are {total} {item_pl}. {name} packs them into {container_pl} that "
    "each hold {divisor} {item_pl}. What is the result?",
    "{name} has {total} {item_pl} and groups them into {container_pl} that "
    "each hold {divisor} {item_pl}. What is the result?",
)
_MEDIUM_SHARE = (
    "{name} has {total} {item_pl}, spread evenly between {divisor} {container_pl}. What is the result?",
    "There are {total} {item_pl}, split evenly between {divisor} {container_pl}. What is the result?",
    "{name} has {total} {item_pl}, divided evenly between {divisor} {container_pl}. What is the result?",
)
_MEDIUM_GROUP = (
    "{name} has {total} {item_pl}, and each {container_sg} will hold {divisor} {item_pl}. What is the result?",
    "There are {total} {item_pl}. Each {container_sg} holds {divisor} {item_pl}. What is the result?",
    "{name} has {total} {item_pl} to arrange so that each {container_sg} "
    "holds {divisor} {item_pl}. What is the result?",
)
_HARD_SHARE = (
    "{name} has {total} {item_pl}, spread evenly between {divisor} "
    "{container_pl}, with some left over. What is the result?",
    "There are {total} {item_pl}, split evenly between {divisor} "
    "{container_pl}, with some left over. What is the result?",
    "{name} has {total} {item_pl}, divided evenly between {divisor} "
    "{container_pl}, but not all are used up. What is the result?",
)
_HARD_GROUP = (
    "{name} has {total} {item_pl}, and each {container_sg} will hold "
    "{divisor} {item_pl}, with some left over. What is the result?",
    "There are {total} {item_pl}. Each {container_sg} holds {divisor} "
    "{item_pl}, and the rest are left over. What is the result?",
    "{name} has {total} {item_pl} to arrange so that each {container_sg} "
    "holds {divisor} {item_pl}, but some will not fit. What is the result?",
)
_EXAM_SHARE = (
    "{name} has {total} {item_pl} and shares them equally between {g} "
    "{container_pl}. Each {container_sg}'s share is then shared equally "
    "among {s2} {container2_pl}. What is the final result, for one of those "
    "{container2_pl}?",
    "There are {total} {item_pl}. {name} shares them equally between {g} "
    "{container_pl}, then each {container_sg}'s share is shared equally "
    "among {s2} {container2_pl}. What is the final result, for one of those "
    "{container2_pl}?",
    "{name} shares {total} {item_pl} equally between {g} {container_pl}. "
    "Each {container_sg}'s share is shared again, equally among {s2} "
    "{container2_pl}. What is the final result, for one of those {container2_pl}?",
)
_EXAM_GROUP = (
    "{name} has {total} {item_pl} and shares them equally between {g} "
    "{container_pl}. Each {container_sg}'s share is then grouped into "
    "{container2_pl} of {s2} each. What is the final result, for one "
    "{container_sg}?",
    "There are {total} {item_pl}. {name} shares them equally between {g} "
    "{container_pl}, then each {container_sg}'s share is grouped into "
    "{container2_pl} of {s2} each. What is the final result, for one "
    "{container_sg}?",
    "{name} shares {total} {item_pl} equally between {g} {container_pl}. "
    "Each {container_sg}'s share is then packed into {container2_pl} of "
    "{s2} each. What is the final result, for one {container_sg}?",
)


def _draw_vocab(rng, exclude_container=None):
    item_sg, item_pl = rng.choice(_ITEMS)
    pool = [c for c in _CONTAINERS if c != exclude_container] if exclude_container else _CONTAINERS
    container_sg, container_pl = rng.choice(pool)
    name = rng.choice(_NAMES)
    return item_sg, item_pl, container_sg, container_pl, name


def _draw_simple(rng, divisor_hi, q_hi, total_cap):
    for _ in range(200):
        divisor = rng.randint(2, divisor_hi)
        q_max = max(2, min(q_hi, total_cap // divisor))
        q = rng.randint(2, q_max)
        total = divisor * q
        if total <= total_cap:
            return total, divisor, q
    check(False, "p18a_sharing_grouping: could not draw a simple total/divisor pair")


def _draw_with_remainder(rng, divisor_hi, q_hi, total_cap):
    for _ in range(200):
        divisor = rng.randint(2, divisor_hi)
        remainder = rng.randint(1, divisor - 1)
        q_max = max(2, min(q_hi, (total_cap - remainder) // divisor))
        q = rng.randint(2, q_max)
        total = divisor * q + remainder
        if total <= total_cap:
            return total, divisor, q, remainder
    check(False, "p18a_sharing_grouping: could not draw a remainder total/divisor pair")


def _divisors_with_quotient_at_least_2(n: int) -> list[int]:
    return [d for d in range(2, n // 2 + 1) if n % d == 0]


def _draw_exam(rng):
    for _ in range(200):
        g = rng.randint(2, 4)
        e = rng.randint(2, 8)
        total = g * e
        if total > 30:
            continue
        s2_choices = _divisors_with_quotient_at_least_2(e)
        if not s2_choices:
            continue
        s2 = rng.choice(s2_choices)
        c = e // s2
        return total, g, e, s2, c
    check(False, "p18a_sharing_grouping: could not draw a two-step exam draw")


def _build(rng, tier: str = "easy"):
    if tier not in _TIERS:
        raise ValueError(f"unknown tier {tier!r}")

    share = rng.random() < 0.5

    if tier == "easy":
        item_sg, item_pl, container_sg, container_pl, name = _draw_vocab(rng)
        total, divisor, q = _draw_simple(rng, divisor_hi=5, q_hi=6, total_cap=20)
        _check_division(total, divisor, q, 0, "p18a_sharing_grouping easy: division check failed")
        if share:
            text = phrase(
                rng, _EASY_SHARE, name=name, total=total, item_pl=item_pl, divisor=divisor, container_pl=container_pl
            )
            answer_sig, answer_text, raw = _candidates_for_share(
                total, divisor, q, 0, item_sg, item_pl, container_sg, container_pl
            )
        else:
            text = phrase(
                rng, _EASY_GROUP, name=name, total=total, item_pl=item_pl, divisor=divisor, container_pl=container_pl
            )
            answer_sig, answer_text, raw = _candidates_for_group(
                total, divisor, q, 0, item_sg, item_pl, container_sg, container_pl
            )
        return _finalise(text, answer_sig, answer_text, raw)

    if tier == "medium":
        item_sg, item_pl, container_sg, container_pl, name = _draw_vocab(rng)
        total, divisor, q = _draw_simple(rng, divisor_hi=6, q_hi=7, total_cap=24)
        _check_division(total, divisor, q, 0, "p18a_sharing_grouping medium: division check failed")
        if share:
            text = phrase(
                rng, _MEDIUM_SHARE, name=name, total=total, item_pl=item_pl, divisor=divisor, container_pl=container_pl
            )
            answer_sig, answer_text, raw = _candidates_for_share(
                total, divisor, q, 0, item_sg, item_pl, container_sg, container_pl
            )
        else:
            text = phrase(
                rng,
                _MEDIUM_GROUP,
                name=name,
                total=total,
                item_pl=item_pl,
                divisor=divisor,
                container_sg=container_sg,
                container_pl=container_pl,
            )
            answer_sig, answer_text, raw = _candidates_for_group(
                total, divisor, q, 0, item_sg, item_pl, container_sg, container_pl
            )
        return _finalise(text, answer_sig, answer_text, raw)

    if tier == "hard":
        item_sg, item_pl, container_sg, container_pl, name = _draw_vocab(rng)
        total, divisor, q, remainder = _draw_with_remainder(rng, divisor_hi=5, q_hi=5, total_cap=24)
        _check_division(total, divisor, q, remainder, "p18a_sharing_grouping hard: division check failed")
        if share:
            text = phrase(
                rng, _HARD_SHARE, name=name, total=total, item_pl=item_pl, divisor=divisor, container_pl=container_pl
            )
            answer_sig, answer_text, raw = _candidates_for_share(
                total, divisor, q, remainder, item_sg, item_pl, container_sg, container_pl
            )
        else:
            text = phrase(
                rng,
                _HARD_GROUP,
                name=name,
                total=total,
                item_pl=item_pl,
                divisor=divisor,
                container_sg=container_sg,
                container_pl=container_pl,
            )
            answer_sig, answer_text, raw = _candidates_for_group(
                total, divisor, q, remainder, item_sg, item_pl, container_sg, container_pl
            )
        return _finalise(text, answer_sig, answer_text, raw)

    # exam: two-step. Step 1 (partitive scene-setter) is always "shares ...
    # equally between g containers"; step 2 applies the share/group decision
    # to ONE share (e = total // g).
    item_sg, item_pl, container_sg, container_pl, name = _draw_vocab(rng)
    pool = [c for c in _CONTAINERS if c != (container_sg, container_pl)]
    container2_sg, container2_pl = rng.choice(pool)
    total, g, e, s2, c = _draw_exam(rng)
    _check_division(total, g, e, 0, "p18a_sharing_grouping exam: step-1 division check failed")
    _check_division(e, s2, c, 0, "p18a_sharing_grouping exam: step-2 division check failed")

    if share:
        text = phrase(
            rng,
            _EXAM_SHARE,
            name=name,
            total=total,
            item_pl=item_pl,
            g=g,
            container_sg=container_sg,
            container_pl=container_pl,
            s2=s2,
            container2_pl=container2_pl,
        )
        answer_sig = ("share", c, 0)
        answer_text = _each_text(c, item_pl, container2_sg, item_sg)
        raw = [
            # swapped-reading: the SAME final quotient c, reported under the
            # quotitive (group-count) unit instead of the partitive one.
            (("group", c, 0), _count_text(c, container2_sg, container2_pl, item_sg, item_pl)),
            # stops-after-step-1: reports the first-step share e under the
            # FINAL container, ignoring the second division. (Stated against
            # the first container it would be a true fact about the scene --
            # a second true option -- so it names the second container.)
            (("stage1", e, 0), _each_text(e, item_pl, container2_sg, item_sg)),
            (("total", total, 0), f"{total} {item_pl}"),
            (("share", c + 1, 0), _each_text(c + 1, item_pl, container2_sg, item_sg)),
        ]
    else:
        text = phrase(
            rng,
            _EXAM_GROUP,
            name=name,
            total=total,
            item_pl=item_pl,
            g=g,
            container_sg=container_sg,
            container_pl=container_pl,
            s2=s2,
            container2_pl=container2_pl,
        )
        answer_sig = ("group", c, 0)
        answer_text = _count_text(c, container2_sg, container2_pl, item_sg, item_pl)
        raw = [
            # swapped-reading: the SAME final quotient c, reported under the
            # partitive (per-container) unit instead of the quotitive one.
            (("share", c, 0), _each_text(c, item_pl, container2_sg, item_sg)),
            # stops-after-step-1: reports the first-step share e as a count
            # of the FINAL containers, ignoring the second division (never a
            # true fact about the scene; see the share branch above).
            (("stage1", e, 0), _count_text(e, container2_sg, container2_pl, item_sg, item_pl)),
            (("total", total, 0), f"{total} {item_pl}"),
            (("group", c + 1, 0), _count_text(c + 1, container2_sg, container2_pl, item_sg, item_pl)),
        ]
    return _finalise(text, answer_sig, answer_text, raw)


@generator(
    TOPIC,
    level=1,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
        "exam": dict(tier="exam"),
    },
)
def p18a_sharing_grouping(rng, tier: str = "easy"):
    return _build(rng, tier=tier)
