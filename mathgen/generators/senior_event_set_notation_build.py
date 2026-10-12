"""Generates `senior-use-set-language-and-notation-for-events-including` questions (ACMMM050). `answer: set`.

Scope: over a small, fully listed finite sample space, list the outcomes in an event that is
written with set notation: a complement (A'), an intersection (A ∩ B), a union (A ∪ B), or a combination of these.
The events are given either as explicit lists of outcomes or in words ("the number is even"). The answer is the
complete set of outcomes, so nothing may be missing and nothing extra, and it can be the empty set, which is how
mutually exclusive events show up (A ∩ B is empty exactly when A and B are mutually exclusive). Symbolic
set-builder answers, probabilities and unbounded sample spaces are out of scope.

Ladder (reasoning escalation):
  easy:   the sample space and the events are listed outright; one operation (A', A ∩ B or A ∪ B).
  medium: the events are described in words, so the learner first works out which outcomes each contains; one
          operation, now including the two-step A ∩ B' and A' ∩ B. The answer can be empty.
  hard:   a complement is applied to a compound event or two complements are combined ((A ∪ B)', A' ∩ B',
          A' ∪ B, ...), over dice, tiles and sequences of three coin tosses, where De Morgan slips are easy.
  exam:   outcomes are ordered pairs from two dice, or three events are combined in one expression.

Every answer is computed from the event definitions and then recomputed from the expression's own parse tree by a
second route; every distractor is a set that follows from a specific notation error, and the real set grader
must reject each one.
"""

from __future__ import annotations

import re
from itertools import product
from types import SimpleNamespace

from mathgen.core import check, draw_until, generator
from mathgen.grading.typed import grade_set, set_answer_text

_TIERS = ("easy", "medium", "hard", "exam")

# -- expressions: ("var", "A") | ("not", e) | ("and", e1, e2) | ("or", e1, e2) -------------------------------------


def _v(name):
    return ("var", name)


def _not(e):
    return ("not", e)


def _and(a, b):
    return ("and", a, b)


def _or(a, b):
    return ("or", a, b)


def _render(e, top=True) -> tuple[str, str]:
    """(plain text, LaTeX) for an expression, with the brackets the notation needs."""
    kind = e[0]
    if kind == "var":
        return e[1], e[1]
    if kind == "not":
        inner = e[1]
        if inner[0] == "var":
            return inner[1] + "'", inner[1] + "'"
        t, lx = _render(inner)
        return f"({t})'", f"({lx})'"
    t1, l1 = _render(e[1], top=False)
    t2, l2 = _render(e[2], top=False)
    symbol, latex_symbol = ("∩", r"\cap") if kind == "and" else ("∪", r"\cup")
    text = f"{t1} {symbol} {t2}"
    latex = f"{l1} {latex_symbol} {l2}"
    if top:
        return text, latex
    return f"({text})", f"({latex})"


def _eval(e, env, space):
    kind = e[0]
    if kind == "var":
        return env[e[1]]
    if kind == "not":
        return space - _eval(e[1], env, space)
    left, right = _eval(e[1], env, space), _eval(e[2], env, space)
    return left & right if kind == "and" else left | right


def _eval_by_outcome(e, env, outcomes):
    """The same value found a second way: test each outcome against the expression one at a time."""

    def holds(expr, outcome):
        kind = expr[0]
        if kind == "var":
            return outcome in env[expr[1]]
        if kind == "not":
            return not holds(expr[1], outcome)
        if kind == "and":
            return holds(expr[1], outcome) and holds(expr[2], outcome)
        return holds(expr[1], outcome) or holds(expr[2], outcome)

    return frozenset(o for o in outcomes if holds(e, o))


def _swap_and_or(e):
    kind = e[0]
    if kind == "var":
        return e
    if kind == "not":
        return _not(_swap_and_or(e[1]))
    return ("or" if kind == "and" else "and", _swap_and_or(e[1]), _swap_and_or(e[2]))


def _drop_not(e):
    kind = e[0]
    if kind == "var":
        return e
    if kind == "not":
        return _drop_not(e[1])
    return (kind, _drop_not(e[1]), _drop_not(e[2]))


def _wrong_de_morgan(e):
    """(P op Q)' read as P' op Q' with the operation unchanged. None when e is not a complemented compound."""
    if e[0] == "not" and e[1][0] in ("and", "or"):
        inner = e[1]
        return (inner[0], _not(inner[1]), _not(inner[2]))
    return None


def _flip_inner_not(e):
    """Toggle the complement on the first variable that appears."""
    kind = e[0]
    if kind == "var":
        return _not(e)
    if kind == "not":
        if e[1][0] == "var":
            return e[1]
        return _not(_flip_inner_not(e[1]))
    return (kind, _flip_inner_not(e[1]), e[2])


# -- contexts -------------------------------------------------------------------------------------------------------


def _die(n):
    space = list(range(1, n + 1))
    return space


_PRIMES = {2, 3, 5, 7, 11}
_SQUARES = {1, 4, 9}

# (words, predicate) for outcomes that are whole numbers
_NUMBER_EVENTS = [
    ("the number is even", lambda x: x % 2 == 0),
    ("the number is odd", lambda x: x % 2 == 1),
    ("the number is prime", lambda x: x in _PRIMES),
    ("the number is a multiple of 3", lambda x: x % 3 == 0),
    ("the number is a perfect square", lambda x: x in _SQUARES),
    ("the number is greater than 3", lambda x: x > 3),
    ("the number is less than 4", lambda x: x < 4),
    ("the number is at least 5", lambda x: x >= 5),
    ("the number is at most 2", lambda x: x <= 2),
    ("the number is a factor of 6", lambda x: 6 % x == 0),
]

_COIN2 = ["HH", "HT", "TH", "TT"]
_COIN2_EVENTS = [
    ("at least one head is shown", lambda o: "H" in o),
    ("both coins show the same face", lambda o: o[0] == o[1]),
    ("exactly one tail is shown", lambda o: o.count("T") == 1),
    ("the first coin shows a head", lambda o: o[0] == "H"),
    ("the second coin shows a tail", lambda o: o[1] == "T"),
]

_COIN3 = ["".join(p) for p in product("HT", repeat=3)]
_COIN3_EVENTS = [
    ("at least two heads are shown", lambda o: o.count("H") >= 2),
    ("the first coin shows a tail", lambda o: o[0] == "T"),
    ("all three coins show the same face", lambda o: len(set(o)) == 1),
    ("exactly one head is shown", lambda o: o.count("H") == 1),
    ("the last coin shows a head", lambda o: o[2] == "H"),
    ("no heads are shown", lambda o: "H" not in o),
]

_DICE2 = [(a, b) for a in range(1, 7) for b in range(1, 7)]
_DICE2_EVENTS = [
    ("the sum is 8", lambda o: o[0] + o[1] == 8),
    ("the sum is 7", lambda o: o[0] + o[1] == 7),
    ("the two dice show the same number", lambda o: o[0] == o[1]),
    ("the first die shows 6", lambda o: o[0] == 6),
    ("the second die shows 2", lambda o: o[1] == 2),
    ("the sum is at least 10", lambda o: o[0] + o[1] >= 10),
    ("the first die is odd", lambda o: o[0] % 2 == 1),
    ("the difference between the dice is 4", lambda o: abs(o[0] - o[1]) == 4),
    ("the product of the two numbers is 12", lambda o: o[0] * o[1] == 12),
]

_NUMBER_CONTEXTS = {
    6: "A fair six-sided die is rolled",
    8: "A spinner with 8 equal sectors numbered 1 to 8 is spun",
    10: "A tile is drawn from a bag of 10 tiles numbered 1 to 10",
    12: "A card is drawn from 12 cards numbered 1 to 12",
}


def _format_outcome(o):
    return str(o)


def _set_text(outcomes) -> str:
    return "{" + ", ".join(_format_outcome(o) for o in outcomes) + "}"


def _sorted(outcomes, order):
    return [o for o in order if o in outcomes]


def _answer_items(outcomes, order):
    return _sorted(outcomes, order)


# -- finishing --------------------------------------------------------------------------------------------------------


def _finish(answer_items, wrong_sets, order):
    """(answer_text, distractors, option_notes). `wrong_sets` is a list of (note_id, frozenset)."""
    answer_text = set_answer_text(answer_items)
    probe = SimpleNamespace(answer=list(answer_items), answer_text=answer_text)
    check(grade_set(probe, answer_text).correct, "the set grader rejects its own canonical answer")
    answer_set = frozenset(answer_items)
    distractors: list[str] = []
    notes: dict[str, str] = {}
    for note_id, wrong in wrong_sets:
        if wrong == answer_set:
            continue
        text = set_answer_text(_answer_items(wrong, order))
        if text in distractors or text == answer_text:
            continue
        check(not grade_set(probe, text).correct, "the set grader accepts a deliberately wrong set")
        distractors.append(text)
        notes[text] = note_id
        if len(distractors) == 3:
            break
    return answer_text, distractors, notes


_MAX_WRONG_SET = 12


def _wrong_sets_for(expr, env, space, order, value):
    """Sets a learner with one notation error would write for `expr`, in priority order."""
    candidates = []
    demorgan = _wrong_de_morgan(expr)
    if demorgan is not None:
        candidates.append(("moves-the-complement-inside-without-swapping", _eval(demorgan, env, space)))
    swapped = _swap_and_or(expr)
    if swapped != expr:
        candidates.append(("confuses-intersection-with-union", _eval(swapped, env, space)))
    dropped = _drop_not(expr)
    if dropped != expr:
        candidates.append(("ignores-the-complement", _eval(dropped, env, space)))
    flipped = _flip_inner_not(expr)
    candidates.append(("complements-the-wrong-event", _eval(flipped, env, space)))
    if expr[0] == "not" and expr[1][0] == "var":
        other = [name for name in env if name != expr[1][1]]
        if other:
            candidates.append(("reads-the-complement-as-the-other-event", env[other[0]]))
    candidates.append(("combines-the-events-the-other-way-round", _eval(_swap_and_or(flipped), env, space)))
    wrong = [(note, frozenset(found)) for note, found in candidates]
    wrong = [(note, found) for note, found in wrong if len(found) <= _MAX_WRONG_SET]
    # Slips of one outcome, always available and always wrong.
    ordered = _answer_items(value, order)
    if len(ordered) >= 2:
        wrong.append(("leaves-out-an-outcome", frozenset(ordered[:-1])))
    extra = [o for o in order if o not in value]
    if extra and len(value) < _MAX_WRONG_SET:
        wrong.append(("adds-an-outcome-that-does-not-belong", frozenset(value) | {extra[len(extra) // 2]}))
    return wrong


# -- builders ---------------------------------------------------------------------------------------------------------

_FORMAT_NOTE = "Write your answer as a set in braces, with the outcomes separated by commas; { } is the empty set."


def _numeric_space(rng, sizes):
    size = rng.choice(sizes)
    return size, _die(size)


def _easy(rng):
    size, space_list = _numeric_space(rng, (6, 8, 10, 12))
    space = frozenset(space_list)
    kind = rng.choice(("not", "and", "or"))

    def draw(_rng):
        a = frozenset(_rng.sample(space_list, _rng.randint(2, max(2, size // 2))))
        b = frozenset(_rng.sample(space_list, _rng.randint(2, max(2, size // 2))))
        return a, b

    def accept(pair):
        a, b = pair
        if a == b:
            return False
        if kind == "not":
            return len(a) < size
        if kind == "and":
            return bool(a & b)
        return len(a | b) < size

    a, b = draw_until(rng, draw, accept, what="two listed events")
    env = {"A": a, "B": b}
    expr = _not(_v("A")) if kind == "not" else (_and(_v("A"), _v("B")) if kind == "and" else _or(_v("A"), _v("B")))
    return _build(rng, expr, env, space, space_list, f"A sample space is S = {_set_text(space_list)}.", listed=True)


def _build(rng, expr, env, space, order, context_text, *, listed=False, descriptions=None):
    value = _eval(expr, env, space)
    check(value == _eval_by_outcome(expr, env, order), "set operations disagree with a one-outcome-at-a-time check")
    answer_items = _answer_items(value, order)
    if listed:
        event_text = " ".join(f"{name} = {_set_text(_answer_items(env[name], order))}." for name in sorted(env))
    else:
        event_text = " ".join(f"Event {name} is: {descriptions[name]}." for name in sorted(env))
    text, latex = _prompt_with_sets(f"{context_text} {event_text}", expr)
    wrong = _wrong_sets_for(expr, env, space, order, value)
    answer_text, distractors, notes = _finish(answer_items, wrong, order)
    return dict(
        text=text,
        latex=latex,
        answer=[list(i) if isinstance(i, tuple) else i for i in answer_items],
        answer_text=answer_text,
        working=_working(expr, env, space, order),
        distractors=distractors,
        option_notes=notes,
        answer_kind="set",
    )


def _prompt_with_sets(intro_text: str, expr):
    expr_text, expr_latex = _render(expr)
    lead = "List the outcomes in "
    text = f"{intro_text} {lead}{expr_text}. {_FORMAT_NOTE}"
    latex = _latex_prose(f"{intro_text} {lead}") + expr_latex + _latex_prose(f". {_FORMAT_NOTE}")
    return text, latex


_SET_PIECE_RE = re.compile(r"((?:\b[A-Z] = )?\{[^{}]*\})")


def _set_math(piece: str) -> str:
    """A listed set such as "A = {2, 4}" or "{ }" as LaTeX math, with word-like outcomes (HT, TTH) upright."""
    if piece.endswith("{ }"):
        return piece[: -len("{ }")] + r"\{\ \}"
    name, _, body = piece.rpartition("{")
    body = re.sub(r"[A-Z]{2,}", lambda m: r"\text{" + m.group(0) + "}", body[:-1])
    return name + r"\{" + body + r"\}"


def _latex_prose(prose: str) -> str:
    """Prose as LaTeX: words in \\text{}, listed sets as math so their braces typeset as braces."""
    out = []
    for piece in _SET_PIECE_RE.split(prose):
        if not piece:
            continue
        if _SET_PIECE_RE.fullmatch(piece):
            out.append(r"\," + _set_math(piece) + r"\,")
        else:
            out.append(r"\text{" + piece + "}")
    return "".join(out)


def _working(expr, env, space, order):
    steps = []
    for name in sorted(env):
        steps.append(f"{name} = {_set_text(_answer_items(env[name], order))}.")

    def go(e):
        t, _ = _render(e)
        value = _eval(e, env, space)
        if e[0] != "var":
            steps.append(f"{t} = {_set_text(_answer_items(value, order))}.")

    def walk(e):
        if e[0] == "var":
            return
        for child in e[1:]:
            walk(child)
        go(e)

    walk(expr)
    return steps


def _worded(rng, space_list, events, pairs_needed, expr_choices, *, context_text, allow_empty, max_size=None):
    """Pick distinct worded events and an expression over them; return the built question dict."""
    space = frozenset(space_list)
    names = ["A", "B", "C"][:pairs_needed]

    def draw(_rng):
        picks = _rng.sample(events, pairs_needed)
        expr = _rng.choice(expr_choices)
        return picks, expr

    def accept(draw_result):
        picks, expr = draw_result
        sets = [frozenset(o for o in space_list if pred(o)) for _text, pred in picks]
        if any(not s or s == space for s in sets):
            return False
        if len(set(sets)) < len(sets):
            return False
        env = dict(zip(names, sets))
        value = _eval(expr, env, space)
        if not value and not allow_empty:
            return False
        if value == space:
            return False
        if max_size is not None and len(value) > max_size:
            return False
        return True

    picks, expr = draw_until(rng, draw, accept, what="worded events with a usable answer")
    env = {n: frozenset(o for o in space_list if pred(o)) for n, (_text, pred) in zip(names, picks)}
    descriptions = {n: text for n, (text, _pred) in zip(names, picks)}
    return _build(rng, expr, env, space, space_list, context_text, descriptions=descriptions)


A, B, C = _v("A"), _v("B"), _v("C")

_MEDIUM_EXPRS = [_not(A), _not(B), _and(A, B), _or(A, B), _and(A, _not(B)), _and(_not(A), B)]
_HARD_EXPRS = [
    _not(_or(A, B)),
    _not(_and(A, B)),
    _and(_not(A), _not(B)),
    _or(_not(A), _not(B)),
    _or(_not(A), B),
    _or(A, _not(B)),
]
_EXAM_PAIR_EXPRS = [_and(A, B), _and(A, _not(B)), _and(_not(A), B), _and(_not(A), _not(B))]
_EXAM_TRIPLE_EXPRS = [
    _and(A, _or(B, C)),
    _and(_or(A, B), _not(C)),
    _and(_and(A, _not(B)), _not(C)),
    _or(_and(A, B), C),
    _and(_not(_or(A, B)), C),
    _not(_or(_and(A, B), C)),
]


def _medium(rng):
    where = rng.random()
    if where < 0.7:
        size, space_list = _numeric_space(rng, (6, 8, 10, 12))
        context = f"{_NUMBER_CONTEXTS[size]}. The sample space is S = {_set_text(space_list)}."
        events = _NUMBER_EVENTS
        return _worded(
            rng, space_list, events, 2, _MEDIUM_EXPRS, context_text=context, allow_empty=True, max_size=size - 1
        )
    context = (
        f"Two fair coins are tossed and the faces are recorded in order. The sample space is S = {_set_text(_COIN2)}."
    )
    return _worded(rng, _COIN2, _COIN2_EVENTS, 2, _MEDIUM_EXPRS, context_text=context, allow_empty=True)


def _hard(rng):
    where = rng.random()
    if where < 0.55:
        size, space_list = _numeric_space(rng, (8, 10, 12))
        context = f"{_NUMBER_CONTEXTS[size]}. The sample space is S = {_set_text(space_list)}."
        return _worded(rng, space_list, _NUMBER_EVENTS, 2, _HARD_EXPRS, context_text=context, allow_empty=True)
    if where < 0.75:
        context = f"Two fair coins are tossed and the faces are recorded in order. The sample space is S = {_set_text(_COIN2)}."
        return _worded(rng, _COIN2, _COIN2_EVENTS, 2, _HARD_EXPRS, context_text=context, allow_empty=True)
    context = (
        f"Three fair coins are tossed and the faces are recorded in order. The sample space is S = {_set_text(_COIN3)}."
    )
    return _worded(rng, _COIN3, _COIN3_EVENTS, 2, _HARD_EXPRS, context_text=context, allow_empty=True)


def _exam(rng):
    if rng.random() < 0.5:
        context = (
            "Two fair six-sided dice are rolled and the numbers are recorded as (first die, second die). "
            "The sample space S has all 36 ordered pairs."
        )
        return _worded(
            rng, _DICE2, _DICE2_EVENTS, 2, _EXAM_PAIR_EXPRS, context_text=context, allow_empty=True, max_size=9
        )
    size = 12
    space_list = _die(size)
    context = f"{_NUMBER_CONTEXTS[size]}. The sample space is S = {_set_text(space_list)}."
    return _worded(rng, space_list, _NUMBER_EVENTS, 3, _EXAM_TRIPLE_EXPRS, context_text=context, allow_empty=True)


_BUILDERS = {"easy": _easy, "medium": _medium, "hard": _hard, "exam": _exam}


@generator("senior_event_set_notation", level=17, presets={tier: dict(tier=tier) for tier in _TIERS})
def senior_event_set_notation(rng, tier="easy"):
    check(tier in _BUILDERS, f"unknown tier {tier!r}")
    return _BUILDERS[tier](rng)
