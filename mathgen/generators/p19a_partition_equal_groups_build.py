"""Generates `p19a_partition_equal_groups` questions: the size of each group when up to 120 objects are split
into a stated number of equal groups.

Scope: "Given a collection of up to 120 objects and a stated
number of equal groups to partition it into, determine the size of each
group using known number facts or skip counting."

## Answer direction

The number of groups is ALWAYS the given quantity; the group SIZE is always
the sole computed unknown (never the reverse direction). `answer: number`.

## Separation from `f10-multiply-divide-one-digit-arrays`

That skill's medium tier already renders "N counters are shared equally into
M equal groups. How many counters are in each group?" with one-digit
groups/each-group-size (2-9) and totals up to 81. This generator never uses
that fixed stem: every prompt names the METHOD the kernel requires -- skip
counting by the number of groups, or place-value partitioning of tens and
ones -- as the route to the answer, not a bare "shared equally into"
framing. `groups` is restricted to the skip-counting numbers {2, 5, 10}
(never any one-digit divisor as the arrays skill allows), totals run to 120
(vs the arrays skill's cap of 81), and medium/hard state the collection in
place-value units (bundles of ten and loose ones) that the arrays skill
never uses at all.

## Difficulty progression (genuine reasoning escalation)

- easy: a collection shared into 2, 5 or 10 equal groups, total up to 120,
  group sizes often two-digit; the prompt names skip counting by the number
  of groups as the check. DECISION: apply known multiplication/skip-counting
  facts by the stated group count to find the per-group size directly.
- medium: the collection is stated in place-value units (bundles of ten and
  loose ones), and both the bundle count and the loose count ALWAYS divide
  the number of groups evenly. DECISION: partition the tens and the ones
  SEPARATELY (whole bundles per group, then whole loose ones per group) and
  combine them -- a place-value decomposition step easy never requires.
- hard: the prompt is NEUTRAL -- it never states whether the bundle count
  divides evenly by the number of groups, only a standing hint ("open a
  bundle if you need to") that applies whether or not it turns out to be
  needed. About one third of hard draws divide evenly (like medium, no
  regrouping needed) and about two thirds leave exactly one bundle over,
  which must be unbundled into ten loose ones and combined with the stated
  loose ones before partitioning. DECISION: hard adds the one genuinely new
  step medium never requires -- CHECKING whether the bundle count divides
  evenly at all, and only then regrouping if it does not; a fixed hint that
  always announces the outcome ("does not share evenly") would remove that
  decision entirely, which is why hard never states the outcome.

## Independent check for every draw

`_check_partition` recomputes the total from the group count and the
group size by multiplication (`groups * group_size == total`), a different
route from the one used to construct the group size (division/decomposition
during the draw). The oracle parses `question.text` with its own regexes over the same fixed
vocabulary and re-derives the total, group count and (for medium/hard) the
bundle/loose split from scratch, never importing this module.
"""

from __future__ import annotations

from ..core import check, check_pairwise_distinct, generator, phrase

TOPIC = "p19a_partition_equal_groups"
_TIERS = ("easy", "medium", "hard")

_ITEMS = [
    ("pencil", "pencils"),
    ("counter", "counters"),
    ("sticker", "stickers"),
    ("bead", "beads"),
    ("shell", "shells"),
]
_NAMES = ("Mia", "Sam", "Ali", "Zoe", "Ben", "Leo")

# groups is always a skip-counting number, never an arbitrary one-digit
# divisor -- the separation from f10-multiply-divide-one-digit-arrays.
_GROUPS_EASY = (2, 5, 10)
# medium/hard: widened from {2, 5} so the number of groups genuinely varies -- every one of 2, 3, 4,
# 5 admits at least one loose-ones value that makes the unbundled combined-ones total divide evenly
# (see _draw_hard).
_GROUPS_MED_HARD = (2, 3, 4, 5)


def _check_partition(total: int, groups: int, group_size: int, msg: str) -> None:
    """Independent route: multiply groups * group_size back and confirm it
    reconstructs the total -- never re-using the division/decomposition that
    produced group_size during construction."""
    check(groups * group_size == total, msg)


def _draw_easy(rng):
    groups = rng.choice(_GROUPS_EASY)
    if groups == 2:
        lo, hi = 8, 60
    elif groups == 5:
        lo, hi = 4, 24
    else:
        lo, hi = 3, 12
    group_size = rng.randint(lo, hi)
    total = groups * group_size
    return groups, group_size, total


def _draw_medium(rng):
    for _ in range(200):
        groups = rng.choice(_GROUPS_MED_HARD)
        tens_per_group = rng.randint(1, 4)
        ones_per_group = rng.randint(0, 9)
        bundles = groups * tens_per_group
        loose = groups * ones_per_group
        total = bundles * 10 + loose
        if bundles >= 2 and total <= 120:
            return groups, bundles, loose, tens_per_group, ones_per_group
    check(False, "p19a_partition_equal_groups: could not draw a medium bundle/loose pair")


def _draw_hard_even(rng):
    """Bundles divide evenly by groups -- no unbundling actually needed this
    draw, exactly like medium's construction. The hard PROMPT never says so;
    the learner must check for themselves."""
    for _ in range(200):
        groups = rng.choice(_GROUPS_MED_HARD)
        tens_per_group = rng.randint(1, 4)
        ones_per_group = rng.randint(0, 9)
        bundles = groups * tens_per_group
        loose = groups * ones_per_group
        total = bundles * 10 + loose
        if bundles >= 2 and total <= 120:
            group_size = tens_per_group * 10 + ones_per_group
            return groups, bundles, loose, tens_per_group, ones_per_group, group_size, False
    check(False, "p19a_partition_equal_groups: could not draw a hard-even bundle/loose pair")


def _draw_hard_leftover(rng):
    """Bundles leave exactly one bundle over -- unbundling is genuinely
    needed this draw."""
    for _ in range(200):
        groups = rng.choice(_GROUPS_MED_HARD)
        tens_per_group = rng.randint(1, 4)
        bundles = groups * tens_per_group + 1  # exactly one leftover bundle
        loose_candidates = [l for l in range(0, 10) if (10 + l) % groups == 0]
        if not loose_candidates:
            continue
        loose = rng.choice(loose_candidates)
        total = bundles * 10 + loose
        if total > 120:
            continue
        ones_per_group = (10 + loose) // groups
        group_size = tens_per_group * 10 + ones_per_group
        return groups, bundles, loose, tens_per_group, ones_per_group, group_size, True
    check(False, "p19a_partition_equal_groups: could not draw a hard-leftover bundle/loose pair")


def _draw_hard(rng):
    """About one third of hard draws divide evenly (no unbundling needed);
    about two thirds leave exactly one bundle over (unbundling needed). The
    prompt itself never reveals which -- see the module docstring."""
    if rng.random() < 1 / 3:
        return _draw_hard_even(rng)
    return _draw_hard_leftover(rng)


_EASY_TEMPLATES = (
    "Count out {total} {item_pl} into {groups} equal groups by counting in "
    "{groups}s. How many {item_pl} are in each group?",
    "{name} has {total} {item_pl} to split into {groups} equal groups, "
    "using skip counting in {groups}s to check. How many {item_pl} go in "
    "each group?",
    "There are {total} {item_pl} to share into {groups} equal groups. Skip "
    "count in {groups}s to find how many {item_pl} are in each group.",
)

_MEDIUM_TEMPLATES = (
    "{bundles} bundles of ten and {loose} loose {item_pl} are shared "
    "equally into {groups} groups. How many {item_pl} are in each group?",
    "{name} has {bundles} bundles of ten {item_pl} and {loose} loose "
    "{item_pl}, shared equally into {groups} groups. How many {item_pl} "
    "are in each group?",
    "There are {bundles} bundles of ten {item_pl} and {loose} loose "
    "{item_pl} in total, split equally into {groups} equal groups. How "
    "many {item_pl} are in each group?",
)

# NEUTRAL wording -- deliberately never states whether the bundle count divides evenly by the number
# of groups.
_HARD_TEMPLATES = (
    "{bundles} bundles of ten and {loose} loose {item_pl} are to be shared "
    "equally into {groups} groups. Open a bundle if you need to. How many "
    "{item_pl} are in each group?",
    "{name} has {bundles} bundles of ten {item_pl} and {loose} loose "
    "{item_pl} to share equally into {groups} groups, opening a bundle if "
    "needed. How many {item_pl} are in each group?",
    "There are {bundles} bundles of ten {item_pl} and {loose} loose "
    "{item_pl} to split equally into {groups} groups. Open a bundle if you "
    "need to. How many {item_pl} are in each group?",
)


def _finalise(text, group_size, raw_distractor_values):
    answer_text = str(group_size)
    seen = {group_size}
    values = [group_size]
    distractors = []
    for v in raw_distractor_values:
        if v in seen or v < 0:
            continue
        seen.add(v)
        values.append(v)
        distractors.append(str(v))
    check(len(distractors) >= 1, "p19a_partition_equal_groups: no distractor survived dedup")
    check_pairwise_distinct(*values, msg="p19a_partition_equal_groups: two options share the same value")

    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=group_size,
        answer_text=answer_text,
        answer_latex=f"plain:{answer_text}",
        distractors=distractors,
        working=[f"answer: {answer_text}"],
    )


def _build(rng, tier: str = "easy"):
    if tier not in _TIERS:
        raise ValueError(f"unknown tier {tier!r}")

    item_sg, item_pl = rng.choice(_ITEMS)
    name = rng.choice(_NAMES)

    if tier == "easy":
        groups, group_size, total = _draw_easy(rng)
        _check_partition(total, groups, group_size, "p19a_partition_equal_groups easy: partition check failed")
        text = phrase(rng, _EASY_TEMPLATES, total=total, groups=groups, item_pl=item_pl, name=name)
        raw = [groups, total - groups, group_size + groups, group_size - groups]
        return _finalise(text, group_size, raw)

    if tier == "medium":
        groups, bundles, loose, tens_per_group, ones_per_group = _draw_medium(rng)
        total = bundles * 10 + loose
        group_size = tens_per_group * 10 + ones_per_group
        _check_partition(total, groups, group_size, "p19a_partition_equal_groups medium: partition check failed")
        text = phrase(rng, _MEDIUM_TEMPLATES, bundles=bundles, loose=loose, groups=groups, item_pl=item_pl, name=name)
        raw = [groups, total - groups, tens_per_group * 10, group_size + groups, group_size - groups]
        return _finalise(text, group_size, raw)

    # hard
    groups, bundles, loose, tens_per_group, ones_per_group, group_size, leftover = _draw_hard(rng)
    total = bundles * 10 + loose
    _check_partition(total, groups, group_size, "p19a_partition_equal_groups hard: partition check failed")
    text = phrase(rng, _HARD_TEMPLATES, bundles=bundles, loose=loose, groups=groups, item_pl=item_pl, name=name)
    raw = [groups, total - groups, tens_per_group * 10, group_size + groups, group_size - groups]
    if leftover:
        # unbundling-dropped: the leftover bundle is simply ignored instead
        # of unbundled -- the learner partitions only the STATED loose ones
        # (never combining in the leftover ten's own ones). `_finalise`'s own
        # dedup drops this candidate on the rare draw where it happens to
        # coincide with the true answer or another candidate. Only offered
        # when there IS a leftover bundle to drop.
        dropped = tens_per_group * 10 + (loose // groups)
        raw.append(dropped)
    return _finalise(text, group_size, raw)


@generator(
    TOPIC,
    level=1,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
    },
)
def p19a_partition_equal_groups(rng, tier: str = "easy"):
    return _build(rng, tier=tier)
