"""Generates `f10-partition-one-two-digit-numbers` questions (AC9M1N02). `answer: set`.

Scope: the learner lists every way a one- or two-digit number can be partitioned. A two-digit number is
partitioned into tens and ones, written (number of tens, number of ones), and a regrouped form such as
(2, 13) for 33 counts as a valid partition. A one-digit number is partitioned into two parts, written
(larger part, smaller part). The answer is the complete set of partitions, so a single valid partition is
not enough: nothing may be missing and nothing extra. Three-digit regrouping and decimal place value are
out of scope.

Ladder (reasoning escalation, not size):
  easy:   a number from 11 to 39 is split into tens and ones in every way, or a one-digit number is split
          into two parts in every way.
  medium: a two-digit number from 31 to 99 is split in every way that also meets a stated limit on the
          ones or on the tens, so the learner has to rule some partitions out.
  hard:   the number is not given. It is shown as a regrouped form, "3 tens and 25 ones", and the learner
          lists every other way, or it is described as some more or less than a stated number of tens and
          ones, so the number has to be found before it is partitioned.

The answer is built from the number, then checked against a second enumeration over every possible tens
count. Each distractor is a set a learner with one specific error would write; the generator checks that
the real set grader rejects every one.
"""

from __future__ import annotations

from mathgen.core import check, draw_until, generator, phrase
from mathgen.grading.typed import grade_set, set_answer_text

from types import SimpleNamespace

_TIERS = ("easy", "medium", "hard")

_RULE = "Use whole numbers; 0 is allowed."


def _tens_ones_splits(n: int) -> list[tuple[int, int]]:
    """Every (tens, ones) with 10 * tens + ones == n, most tens first."""
    return [(t, n - 10 * t) for t in range(n // 10, -1, -1)]


def _splits_by_search(n: int) -> list[tuple[int, int]]:
    """The same set found a second way, by trying every tens count up to n."""
    return sorted(((t, o) for t in range(0, n + 1) for o in range(0, n + 1) if 10 * t + o == n), reverse=True)


def _two_part_splits(n: int) -> list[tuple[int, int]]:
    """Every (larger, smaller) with larger + smaller == n and larger >= smaller."""
    return [(n - s, s) for s in range(0, n // 2 + 1)]


def _tens(count: int) -> str:
    return f"{count} ten" if count == 1 else f"{count} tens"


def _ones(count: int) -> str:
    return f"{count} one" if count == 1 else f"{count} ones"


def _text(items) -> str:
    return set_answer_text(items)


def _finish(answer, candidates):
    """(answer_text, distractors, option_notes). `candidates` is a list of (note_id, items); a candidate that is
    the same set as the answer, empty, or a repeat is dropped, and the set grader must reject every one kept."""
    answer_text = _text(answer)
    probe = SimpleNamespace(answer=list(answer), answer_text=answer_text)
    check(grade_set(probe, answer_text).correct, "the set grader rejects its own canonical answer")
    distractors: list[str] = []
    notes: dict[str, str] = {}
    answer_set = frozenset(answer)
    for note_id, items in candidates:
        if not items or frozenset(items) == answer_set:
            continue
        text = _text(items)
        if text in distractors or text == answer_text:
            continue
        check(not grade_set(probe, text).correct, "the set grader accepts a deliberately wrong set")
        distractors.append(text)
        notes[text] = note_id
        if len(distractors) == 3:
            break
    return answer_text, distractors, notes


def _tens_ones_candidates(n: int, answer, *, keep=lambda pair: True):
    """Wrong sets for 'split n into tens and ones' that a learner with one error would write."""
    t0, o0 = n // 10, n % 10
    trade_one_for_one = [(t0 - j, o0 + j) for j in range(t0 + 1)]
    return [
        ("trades-one-ten-for-one-one", [p for p in trade_one_for_one if keep(p)]),
        ("lists-only-the-standard-split", [(t0, o0)] if keep((t0, o0)) else []),
        ("leaves-out-all-ones-split", [p for p in answer if p != (0, n)]),
        ("swaps-tens-and-ones", [(o, t) for t, o in answer]),
        ("leaves-out-a-regrouped-split", answer[:-2] + answer[-1:] if len(answer) > 3 else []),
    ]


def _easy(rng):
    if rng.random() < 0.3:
        n = rng.randint(5, 9)
        answer = _two_part_splits(n)
        check(
            sorted(answer) == sorted((a, n - a) for a in range(n + 1) if a >= n - a),
            "two-part enumeration disagrees with the check",
        )
        text = phrase(
            rng,
            [
                "Split {n} into two parts in every way. Write each way as (larger part, smaller part). {rule}",
                "Split {n} into two parts in every way you can. Write each way as (larger part, smaller part). {rule}",
            ],
            n=n,
            rule=_RULE,
        )
        candidates = [
            ("lists-both-orders-as-different-splits", answer + [(b, a) for a, b in answer if a != b]),
            ("leaves-out-the-split-with-zero", [p for p in answer if 0 not in p]),
            ("lists-only-some-splits", answer[: max(2, len(answer) - 2)]),
            ("lists-parts-that-do-not-add-to-the-whole", [(a, b + 1) for a, b in answer]),
        ]
        working = [f"The parts add to {n}: " + ", ".join(f"{a} + {b}" for a, b in answer) + "."]
        answer_text, distractors, notes = _finish(answer, candidates)
        return text, answer, answer_text, distractors, notes, working
    n = rng.randint(11, 39)
    if n % 10 == 0:
        n += rng.choice((1, 2, 3))
    answer = _tens_ones_splits(n)
    check(answer == _splits_by_search(n), "tens and ones enumerations disagree")
    text = phrase(
        rng,
        [
            "Show {n} as tens and ones in every way. Write each way as (number of tens, number of ones). {rule}",
            "Show {n} as tens and ones in every way. Each way must make {n} in all. "
            "Write each way as (number of tens, number of ones). {rule}",
        ],
        n=n,
        rule=_RULE,
    )
    working = [f"{_tens(t)} and {_ones(o)} is {10 * t} + {o} = {n}." for t, o in answer]
    answer_text, distractors, notes = _finish(answer, _tens_ones_candidates(n, answer))
    return text, answer, answer_text, distractors, notes, working


def _medium_draw(rng):
    n = rng.randint(31, 99)
    if rng.random() < 0.5:
        limit = rng.choice((20, 30, 40))
        return n, ("ones", limit)
    return n, ("tens", rng.randint(1, max(1, n // 10 - 1)))


def _medium_keep(rule):
    kind, bound = rule
    if kind == "ones":
        return lambda pair: pair[1] < bound
    return lambda pair: pair[0] >= bound


def _medium_ok(drawn) -> bool:
    n, rule = drawn
    keep = _medium_keep(rule)
    kept = [p for p in _tens_ones_splits(n) if keep(p)]
    return n % 10 != 0 and 2 <= len(kept) < len(_tens_ones_splits(n))


def _medium(rng):
    n, rule = draw_until(rng, _medium_draw, _medium_ok, what="a number and a limit that keep some partitions")
    keep = _medium_keep(rule)
    all_splits = _tens_ones_splits(n)
    if rule[0] == "ones":
        condition = f"the number of ones is less than {rule[1]}"
    else:
        condition = f"there are at least {_tens(rule[1])}"
    answer = [p for p in all_splits if keep(p)]
    check(answer == [p for p in _splits_by_search(n) if keep(p)], "tens and ones enumerations disagree")
    check(2 <= len(answer) < len(all_splits), "the limit must keep more than one and rule at least one out")
    text = (
        f"Show {n} as tens and ones in every way where {condition}. "
        f"Write each way as (number of tens, number of ones). {_RULE}"
    )
    traded = [(n // 10 - j, n % 10 + j) for j in range(n // 10 + 1)]
    candidates = [
        ("ignores-the-stated-limit", all_splits),
        ("trades-one-ten-for-one-one", [p for p in traded if keep(p)]),
        ("lists-only-the-standard-split", [(n // 10, n % 10)]),
        ("applies-the-limit-the-wrong-way", [p for p in all_splits if not keep(p)]),
        ("swaps-tens-and-ones", [(o, t) for t, o in answer]),
        ("leaves-out-a-regrouped-split", answer[:1] + answer[2:] if len(answer) > 2 else answer[:1]),
    ]
    working = [f"Every way to make {n} is: " + ", ".join(f"({t}, {o})" for t, o in all_splits) + "."]
    working.append(f"Keep the ways where {condition}: " + ", ".join(f"({t}, {o})" for t, o in answer) + ".")
    answer_text, distractors, notes = _finish(answer, candidates)
    return text, answer, answer_text, distractors, notes, working


def _hard(rng):
    if rng.random() < 0.5:
        t0 = rng.randint(1, 6)
        o0 = rng.randint(11, 39)
        if o0 % 10 == 0:
            o0 += 1
        n = 10 * t0 + o0
        splits = _tens_ones_splits(n)
        answer = [p for p in splits if p != (t0, o0)]
        check(answer == [p for p in _splits_by_search(n) if p != (t0, o0)], "tens and ones enumerations disagree")
        check(len(answer) >= 2, "a hard question needs more than one other partition")
        text = (
            f"A number is shown as {_tens(t0)} and {o0} ones. List every other way to show the same number as "
            f"tens and ones. Write each way as (number of tens, number of ones). {_RULE}"
        )
        standard = (n // 10, n % 10)
        candidates = [
            ("includes-the-way-already-shown", splits),
            ("trades-one-ten-for-one-one", [(t - j, o + j) for t, o in [(t0, o0)] for j in range(1, t + 1)]),
            ("leaves-out-the-standard-split", [p for p in answer if p != standard]),
            ("swaps-tens-and-ones", [(o, t) for t, o in answer]),
        ]
        working = [f"{_tens(t0)} and {o0} ones is {10 * t0} + {o0} = {n}."]
        working.append(f"The other ways to make {n}: " + ", ".join(f"({t}, {o})" for t, o in answer) + ".")
    else:
        t0 = rng.randint(2, 8)
        o0 = rng.randint(0, 9)
        change, phrase = rng.choice(
            ((10, "10 more than"), (-10, "10 less than"), (5, "5 more than"), (-5, "5 less than"))
        )
        n = 10 * t0 + o0 + change
        if n < 20:
            n = 10 * t0 + o0 + abs(change)
            change, phrase = abs(change), phrase.replace("less", "more")
        answer = _tens_ones_splits(n)
        check(answer == _splits_by_search(n), "tens and ones enumerations disagree")
        check(len(answer) >= 2, "a hard question needs more than one partition")
        text = (
            f"A number is {phrase} {_tens(t0)} and {_ones(o0)}. Show the number as tens and ones in every way. "
            f"Write each way as (number of tens, number of ones). {_RULE}"
        )
        base = 10 * t0 + o0
        candidates = [
            ("partitions-the-starting-number", _tens_ones_splits(base)),
            ("trades-one-ten-for-one-one", [(n // 10 - j, n % 10 + j) for j in range(n // 10 + 1)]),
            ("lists-only-the-standard-split", [(n // 10, n % 10)]),
            ("leaves-out-all-ones-split", [p for p in answer if p != (0, n)]),
            ("swaps-tens-and-ones", [(o, t) for t, o in answer]),
        ]
        working = [f"{phrase} {_tens(t0)} and {_ones(o0)}: {base} {'+' if change > 0 else '-'} {abs(change)} = {n}."]
        working.append(f"Every way to make {n}: " + ", ".join(f"({t}, {o})" for t, o in answer) + ".")
    answer_text, distractors, notes = _finish(answer, candidates)
    return text, answer, answer_text, distractors, notes, working


_BUILDERS = {"easy": _easy, "medium": _medium, "hard": _hard}


@generator(
    "f10_partition_tens_ones",
    level=2,
    presets={tier: dict(tier=tier) for tier in _TIERS},
)
def f10_partition_tens_ones(rng, tier="easy"):
    check(tier in _BUILDERS, f"unknown tier {tier!r}")
    text, answer, answer_text, distractors, notes, working = _BUILDERS[tier](rng)
    return dict(
        text=text,
        latex="",
        answer=[list(p) for p in answer],
        answer_text=answer_text,
        working=working,
        distractors=distractors,
        option_notes=notes,
        answer_kind="set",
    )
