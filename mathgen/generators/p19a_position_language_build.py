"""Generates `p19a_position_language` questions: choose the positional term (above, between, behind and so on)
for one object relative to another.

Scope: "Given a stated linear arrangement of named objects or
people, choose the correct positional-language term (for example above,
below, next to, between, in front of, behind) describing one object's
position relative to another, from a small set of offered options."

## Arrangement axes

Three axes, each with its own fixed vocabulary and never mixed with another
axis's terms in the same question (a distractor is always a term plausible
for the STATED axis, per the skill's own contract):

- ``stack`` -- boxes stacked vertically: ``above`` / ``below``.
- ``queue`` -- people standing in a line: ``in front of`` / ``behind``.
- ``shelf`` -- items on a shelf: ``to the left of`` / ``to the right of``.

Every arrangement states its own listing direction explicitly (e.g. "listed
from top to bottom"); the question always names exactly two (or, for the
hard `between` variant, three) of the listed entities and never renders a
picture.

## Mutual exclusivity (never offer two true options)

``next to`` is also true of an adjacent pair that is genuinely
above/below (or in front of/behind, or left/right) of one another, so this
generator never offers ``next to`` alongside a directional term for an
ADJACENT pair: easy always asks an adjacent pair and offers only
{forward_term, backward_term, two false "between X and Y" pairs} (no ``next to``); medium and the
hard reversed-listing variant always ask a NON-adjacent pair, so ``next to``
is safely false and offered alongside the two directional terms and
``between``. The hard `between`-referent-pair variant offers ONLY
``between <X> and <Y>`` phrasings with different referent pairs (never a
plain directional term), so it can never collide with a directional truth
either.

The original build offered bare ``between`` unconditionally and always marked it False, which is wrong
whenever the target is interior to the arrangement (not at either end): a learner who notices the target sits
between its own neighbours can correctly defend "between" even though it was offered as a distractor. This is
now checked by `_independent_bare_between_truth` before every draw that would offer it: whenever the target IS
interior, the generator substitutes a genuinely false PARAMETERIZED ``between <A> and <B>`` option instead (a
referent pair that does NOT bracket the target -- the same mechanism the hard `between` variant already used),
and easy's pair selection is additionally restricted so the target is always an END entity (with only three
entities listed, no false parameterized pair exists to fall back on when the target is the middle one).

## Difficulty progression (genuine reasoning escalation)

- easy: four entities, an ADJACENT target/reference pair with the target at an end, listed in the
  axis's own natural direction. The options are the two directional terms and two false
  ``between <X> and <Y>`` pairs. DECISION: read which of the two
  axis-specific directional terms applies from the stated listing order.
- medium: four or five entities, a NON-adjacent target/reference pair (so
  ``next to`` is a genuine false distractor), and the asked pair's roles
  (which entity is the target) vary. DECISION: the same directional read as
  easy, now with ``next to`` and ``between`` offered as plausible-looking
  false options that must be ruled out.
- hard: EITHER (a) the arrangement is listed in the OPPOSITE direction to
  the offered terms' natural sense ("listed from bottom to top" while the
  options are still above/below), so a literal left-to-right reading of the
  list gives the wrong answer -- DECISION: invert the listing order before
  applying the directional read; OR (b) the question asks which of several
  ``between <X> and <Y>`` referent-pairs actually brackets the target --
  DECISION: check the target's position against *both* referents of more
  than one candidate pair, not just read a single relation.

## Independent check for every draw

`_independent_term_truth` recomputes a directional/next-to option's truth
directly from the raw listing `order` and the `listed_forward` flag via
`order.index(...)` -- a different route from the `ranks` dict comparison
used during construction. `_independent_between_truth` recomputes a
`between <X> and <Y>` option's truth via a sign-product test
(`(rank(a) - rank(target)) * (rank(b) - rank(target)) < 0`), a different
algebraic route from the index-range test (`i < r < j`) used to build the
candidate pairs. Both are asserted to agree with the option's own
construction-time flag on every draw, and exactly one offered option is
asserted true. The oracle
parses `question.text` from scratch (arrangement listing, direction phrase,
target and referents) and recomputes every option's truth without importing
this module.
"""

from __future__ import annotations

from ..core import check, generator, phrase

TOPIC = "p19a_position_language"
_TIERS = ("easy", "medium", "hard")

_AXES = {
    "stack": dict(
        label="Boxes",
        verb="are stacked",
        noun_pl="boxes",
        dir_forward="top to bottom",
        dir_backward="bottom to top",
        forward_term="above",
        backward_term="below",
        entities=("red", "blue", "green", "yellow", "purple"),
        display=lambda e: f"the {e} box",
    ),
    "queue": dict(
        label="Children",
        verb="are standing in a line",
        noun_pl="children",
        dir_forward="front to back",
        dir_backward="back to front",
        forward_term="in front of",
        backward_term="behind",
        entities=("Ava", "Ben", "Cora", "Dev", "Ella", "Finn"),
        display=lambda e: e,
    ),
    "shelf": dict(
        label="Items",
        verb="are arranged on a shelf",
        noun_pl="items",
        dir_forward="left to right",
        dir_backward="right to left",
        forward_term="to the left of",
        backward_term="to the right of",
        entities=("book", "ball", "hat", "cup", "lamp"),
        display=lambda e: f"the {e}",
    ),
}

_STATEMENT_TEMPLATES = (
    "{label} {verb}, listed from {direction}: {entities_list}.",
    "{label} {verb}. Listed from {direction}, they are: {entities_list}.",
    "{label} {verb} in this order, from {direction}: {entities_list}.",
)

_DIRECTIONAL_QUESTIONS = (
    "Which word correctly describes the position of {target} relative to {ref}?",
    "Which term correctly describes where {target} is, relative to {ref}?",
    "Choose the word that correctly describes {target}'s position relative to {ref}.",
)

_BETWEEN_QUESTIONS = (
    "Which of these correctly describes {target}'s position?",
    "Which statement correctly describes where {target} is?",
    "Choose the statement that correctly describes {target}'s position.",
)


def _build_ranks(order: list, listed_forward: bool) -> dict:
    n = len(order)
    return {e: (i if listed_forward else (n - 1 - i)) for i, e in enumerate(order)}


def _entities_list_text(axis, order):
    disp = axis["display"]
    return ", ".join(disp(e) for e in order)


def _statement(rng, axis, order, listed_forward):
    direction = axis["dir_forward"] if listed_forward else axis["dir_backward"]
    return phrase(
        rng,
        _STATEMENT_TEMPLATES,
        label=axis["label"],
        verb=axis["verb"],
        direction=direction,
        entities_list=_entities_list_text(axis, order),
    )


def _independent_term_truth(axis, order, listed_forward, term, target, ref) -> bool:
    """Recompute a directional/next-to term's truth directly from the raw
    `order` list and `listed_forward` flag -- never via the `ranks` dict
    used at construction time (a different route, per the module docstring)."""
    ti = order.index(target)
    ri = order.index(ref)
    target_is_forward_of_ref = (ti < ri) if listed_forward else (ti > ri)
    if term == axis["forward_term"]:
        return target_is_forward_of_ref
    if term == axis["backward_term"]:
        return (not target_is_forward_of_ref) and ti != ri
    if term == "next to":
        return abs(ti - ri) == 1
    if term == "between":
        return False  # only two referents are ever named alongside this term
    raise ValueError(f"unknown directional term {term!r}")


def _independent_between_truth(ranks, target, a, b) -> bool:
    """Recompute a `between <a> and <b>` option's truth via a sign-product
    test on rank differences -- a different algebraic route from the
    index-range test (`i < r < j`) used to build the candidate pairs."""
    return (ranks[a] - ranks[target]) * (ranks[b] - ranks[target]) < 0


def _independent_bare_between_truth(order, target) -> bool:
    """Recompute whether the bare word "between" is true of `target` -- it has an item on each side of it
    anywhere in the FULL stated arrangement -- directly from `order.index`, independent of which specific
    referent the question names alongside it."""
    i = order.index(target)
    return 0 < i < len(order) - 1


def _pick_false_between_pair(rng, order, target):
    """Pick two OTHER entities whose positions do NOT bracket `target` --
    used as a genuinely false parameterized `between <A> and <B>` option
    whenever the bare word `between` would itself be true of `target`."""
    ti = order.index(target)
    others = [e for e in order if e != target]
    false_pairs = []
    for x in range(len(others)):
        for y in range(x + 1, len(others)):
            a, b = others[x], others[y]
            ai, bi = order.index(a), order.index(b)
            lo, hi = min(ai, bi), max(ai, bi)
            if not (lo < ti < hi):
                false_pairs.append((a, b))
    check(len(false_pairs) >= 1, "p19a_position_language: no false between pair available")
    return rng.choice(false_pairs)


def _directional_options(rng, axis, order, listed_forward, target, ref, include_next_to, include_between):
    terms = [axis["forward_term"], axis["backward_term"]]
    if include_next_to:
        terms.append("next to")
    options = []
    for term in terms:
        truth = _independent_term_truth(axis, order, listed_forward, term, target, ref)
        options.append((term, truth))
    if include_between:
        if _independent_bare_between_truth(order, target):
            # bare "between" would be TRUE of this (interior) target --
            # substitute a parameterized, genuinely false referent pair.
            a, b = _pick_false_between_pair(rng, order, target)
            disp = axis["display"]
            pair_text = f"between {disp(a)} and {disp(b)}"
            ranks = _build_ranks(order, listed_forward)
            truth = _independent_between_truth(ranks, target, a, b)
            check(not truth, "p19a_position_language: fallback between pair is unexpectedly true")
            options.append((pair_text, truth))
        else:
            options.append(("between", False))
    return options


def _capitalise(sentence: str) -> str:
    return sentence[:1].upper() + sentence[1:]


def _finalise_directional(rng, text, disp_target, disp_ref, options, msg):
    true_count = sum(1 for _t, truth in options if truth)
    check(true_count == 1, f"{msg}: expected exactly one true option, got {true_count}")
    key_text = next(t for t, truth in options if truth)
    distractors = [t for t, truth in options if not truth]
    question = phrase(rng, _DIRECTIONAL_QUESTIONS, target=disp_target, ref=disp_ref)
    full_text = f"{text} {question}"
    return dict(
        text=full_text,
        latex=f"plain:{full_text}",
        answer=key_text,
        answer_text=key_text,
        answer_latex=f"plain:{key_text}",
        distractors=distractors,
        working=[_capitalise(f"{disp_target} is {key_text} {disp_ref}.")],
    )


def _draw_easy(rng):
    axis_key = rng.choice(list(_AXES))
    axis = _AXES[axis_key]
    order = list(rng.sample(axis["entities"], 4))
    listed_forward = True
    # The target is the first or last entity and the reference is its neighbour. At an end the target has nothing
    # on one side, so no "between" option can be true of it, and the other three entities give three referent
    # pairs that do not bracket it.
    target, ref = rng.choice(((order[0], order[1]), (order[3], order[2])))
    text = _statement(rng, axis, order, listed_forward)
    disp = axis["display"]
    options = _directional_options(
        rng, axis, order, listed_forward, target, ref, include_next_to=False, include_between=False
    )
    ranks = _build_ranks(order, listed_forward)
    ti = order.index(target)
    pairs = [(a, b) for x, a in enumerate(order) for b in order[x + 1 :] if target not in (a, b)]
    pairs = [(a, b) for a, b in pairs if not order.index(a) < ti < order.index(b)]
    check(len(pairs) == 3, "p19a_position_language easy: expected three non-bracketing referent pairs")
    for a, b in rng.sample(pairs, 2):
        truth = _independent_between_truth(ranks, target, a, b)
        check(not truth, "p19a_position_language easy: between pair is unexpectedly true")
        options.append((f"between {disp(a)} and {disp(b)}", truth))
    return _finalise_directional(rng, text, disp(target), disp(ref), options, "p19a_position_language easy")


def _draw_nonadjacent_pair(rng, order, k):
    for _ in range(50):
        i, j = rng.sample(range(k), 2)
        if abs(i - j) >= 2:
            return order[i], order[j]
    check(False, "p19a_position_language: could not draw a non-adjacent pair")


def _draw_medium(rng):
    axis_key = rng.choice(list(_AXES))
    axis = _AXES[axis_key]
    k = rng.choice((4, 5))
    order = list(rng.sample(axis["entities"], k))
    listed_forward = True
    a, b = _draw_nonadjacent_pair(rng, order, k)
    target, ref = (a, b) if rng.random() < 0.5 else (b, a)
    text = _statement(rng, axis, order, listed_forward)
    options = _directional_options(
        rng, axis, order, listed_forward, target, ref, include_next_to=True, include_between=True
    )
    disp = axis["display"]
    return _finalise_directional(rng, text, disp(target), disp(ref), options, "p19a_position_language medium")


def _draw_hard_reversed(rng):
    axis_key = rng.choice(list(_AXES))
    axis = _AXES[axis_key]
    k = rng.choice((4, 5))
    order = list(rng.sample(axis["entities"], k))
    listed_forward = False  # listed in the OPPOSITE direction to the terms' sense
    a, b = _draw_nonadjacent_pair(rng, order, k)
    target, ref = (a, b) if rng.random() < 0.5 else (b, a)
    text = _statement(rng, axis, order, listed_forward)
    options = _directional_options(
        rng, axis, order, listed_forward, target, ref, include_next_to=True, include_between=True
    )
    disp = axis["display"]
    return _finalise_directional(rng, text, disp(target), disp(ref), options, "p19a_position_language hard-reversed")


def _draw_hard_between(rng):
    axis_key = rng.choice(list(_AXES))
    axis = _AXES[axis_key]
    # k=5 only: with k=4, some target ranks cannot yield >= 2 false
    # (non-bracketing) referent pairs, so k=4 is never used for this variant.
    k = 5
    order = list(rng.sample(axis["entities"], k))
    listed_forward = True
    ranks = _build_ranks(order, listed_forward)
    # The target's rank: next to an end but never at one. Only there are at least three referent pairs that do not
    # bracket it (from the middle rank only two), which the four options need.
    r = rng.choice((1, k - 2))
    target = order[r]
    disp = axis["display"]

    true_pairs, false_pairs = [], []
    for i in range(k):
        for j in range(i + 1, k):
            if i == r or j == r:
                continue
            (true_pairs if i < r < j else false_pairs).append((i, j))
    check(len(true_pairs) >= 1, "p19a_position_language hard-between: no true pair")
    check(len(false_pairs) >= 3, "p19a_position_language hard-between: not enough false pairs")
    rng.shuffle(true_pairs)
    rng.shuffle(false_pairs)
    chosen = [true_pairs[0]] + false_pairs[:3]
    rng.shuffle(chosen)

    def pair_text(pair):
        i, j = pair
        return f"between {disp(order[i])} and {disp(order[j])}"

    options = []
    for i, j in chosen:
        truth = _independent_between_truth(ranks, target, order[i], order[j])
        options.append((pair_text((i, j)), truth))
    true_count = sum(1 for _t, truth in options if truth)
    check(true_count == 1, f"p19a_position_language hard-between: expected exactly one true option, got {true_count}")

    key_text = next(t for t, truth in options if truth)
    distractors = [t for t, truth in options if not truth]
    text = _statement(rng, axis, order, listed_forward)
    question = phrase(rng, _BETWEEN_QUESTIONS, target=disp(target))
    full_text = f"{text} {question}"
    return dict(
        text=full_text,
        latex=f"plain:{full_text}",
        answer=key_text,
        answer_text=key_text,
        answer_latex=f"plain:{key_text}",
        distractors=distractors,
        working=[_capitalise(f"{disp(target)} is {key_text}.")],
    )


def _build(rng, tier: str = "easy"):
    if tier not in _TIERS:
        raise ValueError(f"unknown tier {tier!r}")
    if tier == "easy":
        return _draw_easy(rng)
    if tier == "medium":
        return _draw_medium(rng)
    # hard: 25% between-referent-pair variant, 75% reversed-listing variant.
    if rng.random() < 0.25:
        return _draw_hard_between(rng)
    return _draw_hard_reversed(rng)


@generator(
    TOPIC,
    level=1,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
    },
)
def p19a_position_language(rng, tier: str = "easy"):
    return _build(rng, tier=tier)
