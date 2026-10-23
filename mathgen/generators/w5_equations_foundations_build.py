"""Foundations of equations for Years 7-8, built as auto-marked multiple choice.

Topics (each with easy, medium and hard presets and an exam chain `exd_x_<topic>`):

* `w5_meaning_of_equality` (`meaning-of-equality`): true and false number sentences, finding the missing number so the
  sides balance, and what keeps a balance true.
* `w5_inverse_operations` (`inverse-operations`): the operation that undoes another, the steps (in order) that take a
  function machine's output back to its input, and operations that undo themselves.
* `w5_solution_set` (`solution-and-solution-set`): which value is a solution, which equation a value solves, and the
  solution set of an equation over a stated set of numbers.
* `w5_identity_no_solution` (`identity-and-no-solution`): whether an equation has one solution, no solution or every
  number as a solution.
* `w5_check_substitution` (`check-solution-by-substitution`): what each side becomes when a value is substituted, and
  the verdict.

Every question is built from the values first. The key is then re-derived a second way (sides evaluated with exact
fractions, a machine run forward and back, a solution set found by testing each candidate) and every distractor is
checked to be false by the same independent evaluation.
"""

from __future__ import annotations

from fractions import Fraction as Fr

from ..core import attempts, check, fmt_signed, generator, join_terms, paren_negative, phrase
from ._w5_kit import (
    NAMES,
    mc_question,
    sub_fields,
    tier_pool,
    TooFewDistractors,
    enough,
    length_balanced,
    mcq_cands,
    mcq_fields,
    number_cands,
    number_fields,
    part,
    retry_draw,
    rt,
    spread,
)

_sub, _mc, _tiers = sub_fields, mc_question, tier_pool


# =====================================================================================================================
# meaning of equality
# =====================================================================================================================


def _eq_true_false(rng):
    ask_true = rng.random() < 0.55
    use_times = rng.random() < 0.35
    a, b = (rng.randint(2, 9), rng.randint(2, 9)) if use_times else (rng.randint(2, 12), rng.randint(2, 12))
    v = a * b if use_times else a + b
    op = "×" if use_times else "+"
    c = rng.randint(1, v - 1)
    d = v - c
    if (c, d) in ((a, b), (b, a)) and not use_times:
        raise TooFewDistractors("trivial sentence")
    true = f"{a} {op} {b} = {c} + {d}"
    check(_side(true, 0) == _side(true, 1), "built sentence is not true")
    nexts = [f"{a} {op} {b} = {v} + {e}" for e in rng.sample(range(1, 7), 2)]
    offs = []
    for _ in range(2):
        shift = rng.choice((-3, -2, -1, 1, 2, 3))
        dd = d + shift
        if dd >= 1 and dd != d:
            offs.append(f"{a} {op} {b} = {c} + {dd}")
    if ask_true:
        wrongs = [
            (nexts[0], "w5e-equals-means-answer-next"),
            (offs[0] if offs else None, "w5e-sides-not-compared"),
            (nexts[1], "w5e-equals-means-answer-next"),
        ]
        wrongs += [(o, "w5e-sides-not-compared") for o in offs[1:]]
        for s, _ in wrongs:
            if s is not None:
                check(_side(s, 0) != _side(s, 1), "a distractor sentence is true")
        key = true
        stems = (
            "Which of these number sentences is true?",
            "Which one of these statements is correct?",
            "Exactly one of these number sentences is true. Which one?",
        )
        work = [
            f"Work out both sides of each sentence. {a} {op} {b} = {v} and {c} + {d} = {v}, so {true} is true.",
            "In the other sentences the two sides have different values.",
        ]
    else:
        if v < 6:
            raise TooFewDistractors("too few true sentences")
        true_pool = [f"{a} {op} {b} = {cc} + {v - cc}" for cc in rng.sample(range(1, v), 4)]
        key = rng.choice([x for x in nexts + offs if x])
        wrongs = [(s, "w5e-sides-not-compared") for s in true_pool]
        check(_side(key, 0) != _side(key, 1), "the key sentence is true")
        for s, _ in wrongs:
            check(_side(s, 0) == _side(s, 1), "a distractor sentence is false")
        stems = (
            "Which of these number sentences is false?",
            "Which one of these statements is NOT correct?",
            "Exactly one of these number sentences is false. Which one?",
        )
        work = [
            f"Work out both sides: {key.split(' = ')[0]} = {_side(key, 0)} but {key.split(' = ')[1]} = {_side(key, 1)}.",
            "The other sentences have equal sides.",
        ]
    return _mc(rng, stems, key, wrongs, work)


def _side(sentence: str, which: int) -> int:
    text = sentence.split(" = ")[which]
    tokens = text.split(" ")
    total = int(tokens[0])
    i = 1
    while i < len(tokens):
        op, num = tokens[i], int(tokens[i + 1])
        total = total + num if op == "+" else total - num if op == "-" else total * num if op in ("×", "*") else total
        i += 2
    return total


def _eq_story(rng):
    name = rng.choice(NAMES)
    a, b = rng.randint(2, 12), rng.randint(2, 12)
    v = a + b
    is_true = rng.random() < 0.5
    if is_true:
        c = rng.randint(1, v - 1)
        d = v - c
        if (c, d) in ((a, b), (b, a)):
            raise TooFewDistractors("trivial")
        r = v
        key = f"Yes: both sides are equal to {v}"
        wrongs = [
            ("No: an equals sign must be followed by a single number", "w5e-equals-needs-single-number"),
            (f"No: the left side is {v} and the right side is {v + 2}", "w5e-wrong-side-value"),
            ("No: the two sides use different numbers", "w5e-compares-form"),
            (f"No: the right side is bigger than {v}", "w5e-wrong-side-value"),
        ]
        work = [
            f"Left side: {a} + {b} = {v}. Right side: {c} + {d} = {v}.",
            "The sides have the same value, so the sentence is true.",
        ]
    else:
        e = rng.randint(1, 6)
        c, d = v, e
        r = v + e
        key = f"No: the left side is {v} and the right side is {r}"
        wrongs = [
            (f"Yes: {a} + {b} makes {v}, and {v} is on the right side", "w5e-equals-means-answer-next"),
            ("Yes: both sides are sums, so they are equal", "w5e-compares-form"),
            (f"No: the left side is {v} and the right side is {r + 1}", "w5e-wrong-side-value"),
            (f"No: the left side is {r} and the right side is {v}", "w5e-wrong-side-value"),
        ]
        work = [
            f"Left side: {a} + {b} = {v}. Right side: {c} + {d} = {r}.",
            f"{v} is not equal to {r}, so the sentence is false.",
        ]
    stem = phrase(
        rng,
        (
            "{n} writes {s}. Is {n} right? Choose the best answer.",
            "{n} says that {s} is a true number sentence. Which reply is correct?",
            "Is {s} a true number sentence? Choose the answer that gives the right reason.",
        ),
        n=name,
        s=f"{a} + {b} = {c} + {d}",
    )
    fields = mcq_fields(key)
    cands = mcq_cands(key, wrongs, 4)
    enough(cands, 4)
    return dict(text=stem, working=work, distractors=cands, **fields)


def _eq_missing_add(rng):
    a, b = rng.randint(2, 12), rng.randint(2, 12)
    v = a + b
    c = rng.randint(1, v - 1)
    d = v - c
    if (c, d) in ((a, b), (b, a)):
        raise TooFewDistractors("trivial")
    hide = rng.choice(("c", "d", "b"))
    terms = {"a": a, "b": b, "c": c, "d": d}
    if hide == "b":
        text, key_v, others = f"{a} + n = {c} + {d}", b, a + c + d
    elif hide == "c":
        text, key_v, others = f"{a} + {b} = n + {d}", c, a + b + d
    else:
        text, key_v, others = f"{a} + {b} = {c} + n", d, a + b + c
    key = f"n = {key_v}"
    other_known = {"b": a, "c": d, "d": c}[hide]
    wrongs = [
        (f"n = {v}", "w5e-equals-means-answer-next"),
        (f"n = {others}", "w5e-adds-all-numbers"),
        (f"n = {other_known}", "w5e-copies-a-number"),
        (f"n = {key_v + 1}", "w5e-off-by-one"),
        (f"n = {key_v - 1}" if key_v > 1 else None, "w5e-off-by-one"),
    ]
    work = [
        f"The sides must have the same value. One side is a sum that makes {v}.",
        f"So n = {key_v} makes {text} balance.",
    ]
    return _mc(
        rng,
        (
            "Find the value of n that makes {t} true.",
            "What number should n be so that {t} is a true sentence?",
            "For which value of n does the sentence {t} balance?",
        ),
        key,
        wrongs,
        work,
        t=text,
    )


def _eq_missing_mult(rng):
    a, b = rng.randint(2, 9), rng.randint(2, 9)
    v = a * b
    divisors = [k for k in range(2, 10) if v % k == 0 and k not in (a, b)]
    if not divisors:
        raise TooFewDistractors("no second factorisation")
    c = rng.choice(divisors)
    n = v // c
    text = f"{a} × {b} = n × {c}"
    key = f"n = {n}"
    wrongs = [
        (f"n = {v}", "w5e-equals-means-answer-next"),
        (f"n = {v * c}", "w5e-adds-all-numbers"),
        (f"n = {c}", "w5e-copies-a-number"),
        (f"n = {n + 1}", "w5e-off-by-one"),
        (f"n = {a + b - c}" if a + b - c > 0 else None, "w5e-adds-instead-of-multiplies"),
    ]
    work = [f"{a} × {b} = {v}, so n × {c} must also be {v}.", f"{v} ÷ {c} = {n}, so n = {n}."]
    return _mc(
        rng,
        (
            "Find the value of n that makes {t} true.",
            "What number should n be so that {t} is a true sentence?",
            "For which value of n does {t} balance?",
        ),
        key,
        wrongs,
        work,
        t=text,
    )


def _eq_keep_balance(rng):
    a, b = rng.randint(3, 12), rng.randint(3, 12)
    v = a + b
    c = rng.randint(2, v - 2)
    d = v - c
    if (c, d) in ((a, b), (b, a)):
        raise TooFewDistractors("trivial")
    base = f"{a} + {b} = {c} + {d}"
    kind = rng.choice(("add", "subtract", "multiply"))
    k = rng.randint(2, 6)
    if kind == "subtract" and k >= v:
        raise TooFewDistractors("too large")
    if kind == "add":
        key = f"{a} + {b} + {k} = {c} + {d} + {k}"
        wrongs = [
            (f"{a} + {b} + {k} = {c} + {d}", "w5e-changes-one-side-only"),
            (f"{a} + {b} = {c} + {d} + {k}", "w5e-changes-one-side-only"),
            (f"{a} + {b} + {k} = {c} + {d} - {k}", "w5e-opposite-changes"),
            (f"{a} + {b} + {k} = {c} + {d} + {k + 1}", "w5e-unequal-changes"),
        ]
        word = f"adding {k} to both sides"
    elif kind == "subtract":
        key = f"{a} + {b} - {k} = {c} + {d} - {k}"
        wrongs = [
            (f"{a} + {b} - {k} = {c} + {d}", "w5e-changes-one-side-only"),
            (f"{a} + {b} = {c} + {d} - {k}", "w5e-changes-one-side-only"),
            (f"{a} + {b} - {k} = {c} + {d} + {k}", "w5e-opposite-changes"),
            (f"{a} + {b} - {k} = {c} + {d} - {k + 1}", "w5e-unequal-changes"),
        ]
        word = f"subtracting {k} from both sides"
    else:
        key = f"({a} + {b}) × {k} = ({c} + {d}) × {k}"
        wrongs = [
            (f"({a} + {b}) × {k} = {c} + {d}", "w5e-changes-one-side-only"),
            (f"{a} + {b} = ({c} + {d}) × {k}", "w5e-changes-one-side-only"),
            (f"({a} + {b}) × {k} = ({c} + {d}) + {k}", "w5e-opposite-changes"),
            (f"({a} + {b}) × {k} = ({c} + {d}) × {k + 1}", "w5e-unequal-changes"),
        ]
        word = f"multiplying both sides by {k}"
    work = [
        f"{base} is true, and both sides equal {v}.",
        f"Doing the same thing to both sides keeps them equal: {word} gives {key}.",
    ]
    stems = (
        "{b} is a true sentence. Which of these must also be true?",
        "Given that {b} is true, which sentence is certain to be true as well?",
        "The sentence {b} balances. Which change keeps it balanced?",
    )
    return _mc(rng, stems, key, wrongs, work, b=base)


def _eq_not_equal_expression(rng):
    a, b = rng.randint(3, 9), rng.randint(3, 9)
    v = a * b
    pool = []
    pool.append(f"{v - 4} + 4")
    pool.append(f"{v + 6} - 6")
    for f in range(2, 10):
        if v % f == 0 and f not in (a, b) and v // f != 1:
            pool.append(f"{f} × {v // f}")
    pool.append(f"{a + 1} × {b} - {b}")
    pool = [p for p in dict.fromkeys(pool)]
    rng.shuffle(pool)
    if len(pool) < 4:
        raise TooFewDistractors("few equal forms")
    bad_options = [
        f"{v - 3} + 4",
        f"{v + 5} - 6",
        f"{a + 1} × {b} + {b}",
        f"{v + 3} - 2",
        f"{a} × {b + 1}",
    ]
    key = rng.choice(bad_options)
    check(_side(key + " = 0", 0) != v if "(" not in key else True, "the key expression equals the target")
    wrongs = [(p, "w5e-sides-not-compared") for p in pool[:4]]
    for p, _ in wrongs:
        check(_side(p + " = 0", 0) == v, "a distractor does not equal the target")
    work = [
        f"{a} × {b} = {v}. Work out each expression and compare it with {v}.",
        f"{key} = {_side(key + ' = 0', 0)}, which is not {v}.",
    ]
    stems = (
        "Which of these expressions does NOT equal {a} × {b}?",
        "Four of these expressions have the same value as {a} × {b}. Which one does not?",
        "Which expression has a different value from {a} × {b}?",
    )
    return _mc(rng, stems, key, wrongs, work, a=a, b=b)


def _eq_exam(rng):
    a, b = rng.randint(3, 12), rng.randint(3, 12)
    v = a + b
    c = rng.randint(2, v - 2)
    n = v - c
    if n in (a, b) or c in (a, b):
        raise TooFewDistractors("trivial")
    k = rng.randint(2, 8)
    fa = number_fields(n)
    da = number_cands(
        n,
        [
            (v, "w5e-equals-means-answer-next"),
            (a + b + c, "w5e-adds-all-numbers"),
            (c, "w5e-copies-a-number"),
            (n + 1, "w5e-off-by-one"),
            (n - 1, "w5e-off-by-one"),
        ],
    )
    fb = number_fields(v)
    db = number_cands(
        v,
        [
            (n, "w5e-wrong-side-value"),
            (a, "w5e-wrong-side-value"),
            (v + c, "w5e-adds-all-numbers"),
            (v + 1, "w5e-off-by-one"),
            (v - 1, "w5e-off-by-one"),
        ],
    )
    key = f"Add {k} to the right side"
    wrongs = [
        ("Add nothing: the sides are already equal", "w5e-changes-one-side-only"),
        (f"Subtract {k} from the right side", "w5e-opposite-changes"),
        (f"Add {v} to the right side", "w5e-adds-all-numbers"),
        (f"Add {k + 1} to the right side", "w5e-unequal-changes"),
    ]
    fc = mcq_fields(key)
    dc = mcq_cands(key, wrongs, 4)
    for dd in (da, db, dc):
        enough(dd, 2)
    text = f"A balance has {a} + {b} on its left side and n + {c} on its right side, and it balances."
    return dict(
        text=text,
        parts=[
            part(
                "Find the value of n.",
                fa["answer_kind"],
                _sub(fa),
                1,
                [f"{a} + {b} = {v}, so n + {c} = {v} and n = {n}."],
                da,
                sensible_range=(0, None),
            ),
            part(
                "What is the value of each side of the balance?",
                fb["answer_kind"],
                _sub(fb),
                1,
                [f"{a} + {b} = {v}."],
                db,
                sensible_range=(0, None),
            ),
            part(
                f"{rng.choice(NAMES)} adds {k} to the left side only. What must be done to the right side to balance it again?",
                "mcq",
                _sub(fc),
                2,
                [f"The same amount must be added to both sides, so add {k} to the right side."],
                dc,
            ),
        ],
    )


@generator(
    "w5_meaning_of_equality",
    level=17,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def w5_meaning_of_equality(rng, tier="easy"):
    """Equals as a balance between two sides; the tiers are described in the module docstring."""
    pools = {
        "easy": (_eq_true_false, _eq_story),
        "medium": (_eq_missing_add, _eq_missing_add, _eq_missing_mult),
        "hard": (_eq_keep_balance, _eq_not_equal_expression),
    }
    return _tiers(pools, rng, tier)


@generator("exd_x_w5_meaning_of_equality", level=17, presets={"exam": dict(tier="exam")})
def exd_x_w5_meaning_of_equality(rng, tier="exam"):
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    return retry_draw(rng, _eq_exam)


# =====================================================================================================================
# inverse operations
# =====================================================================================================================

# A machine step is (op, n): "add", "sub", "mul", "div" a number, or "rsub" (n minus the input).
_OP_WORD = {
    "add": "add {n}",
    "sub": "subtract {n}",
    "mul": "multiply by {n}",
    "div": "divide by {n}",
    "rsub": "subtract the number from {n}",
}
_INVERSE = {"add": "sub", "sub": "add", "mul": "div", "div": "mul", "rsub": "rsub"}


def _word(step) -> str:
    op, n = step
    return _OP_WORD[op].format(n=n)


def _steps_text(steps) -> str:
    return ", then ".join(_word(s) for s in steps)


def _apply(step, x: Fr) -> Fr:
    op, n = step
    return {"add": x + n, "sub": x - n, "mul": x * n, "div": x / n, "rsub": n - x}[op]


def _run(steps, x: Fr) -> Fr:
    for s in steps:
        x = _apply(s, x)
    return x


def _undo(steps):
    """The steps that take the output back to the input: each inverse, in reverse order."""
    return [(_INVERSE[op], n) for op, n in reversed(steps)]


def _cap(text: str) -> str:
    return text[0].upper() + text[1:]


def _calc_value(text: str):
    """The value of a calculation such as 'Subtract 2 from 4' or 'Divide 12 by 3' (None for other texts)."""
    words = text.split(" ")
    if words[0] == "Add" and words[2] == "to":
        return Fr(int(words[3]) + int(words[1]))
    if words[0] == "Subtract" and words[2] == "from":
        return Fr(int(words[3]) - int(words[1]))
    if words[0] == "Multiply" and words[2] == "by":
        return Fr(int(words[1]) * int(words[3]))
    if words[0] == "Divide" and words[2] == "by":
        return Fr(int(words[1]), int(words[3]))
    return None


def _inv_pair(rng):
    op = rng.choice(("add", "sub", "mul", "div"))
    k = rng.randint(2, 12)
    if rng.random() < 0.5:
        stem = phrase(
            rng,
            ("Which operation undoes {w}?", "What do you do to reverse {w}?", "{c} is undone by which operation?"),
            w={
                "add": f"adding {k}",
                "sub": f"subtracting {k}",
                "mul": f"multiplying by {k}",
                "div": f"dividing by {k}",
            }[op],
            c={
                "add": f"Adding {k}",
                "sub": f"Subtracting {k}",
                "mul": f"Multiplying by {k}",
                "div": f"Dividing by {k}",
            }[op],
        )
        names = {"add": f"Add {k}", "sub": f"Subtract {k}", "mul": f"Multiply by {k}", "div": f"Divide by {k}"}
        key = names[_INVERSE[op]]
        check(_apply((_INVERSE[op], k), _apply((op, k), Fr(7))) == 7, "inverse does not undo")
        wrongs = [
            (names[op], "w5i-same-operation"),
            (names[{"add": "mul", "sub": "div", "mul": "add", "div": "sub"}[op]], "w5i-mixes-operation-pairs"),
            (names[{"add": "div", "sub": "mul", "mul": "sub", "div": "add"}[op]], "w5i-mixes-operation-pairs"),
            (f"{_cap(_OP_WORD[_INVERSE[op]].format(n=k + rng.randint(1, 3)))}", "w5i-wrong-amount"),
        ]
        work = [
            f"{_cap(_OP_WORD[op].format(n=k))} and {_OP_WORD[_INVERSE[op]].format(n=k)} are inverse operations.",
            "An inverse operation takes the result back to the number you started with.",
        ]
        return _mc(rng, (stem,), key, wrongs, work)
    x = rng.randint(2, 15)
    out = _apply((op, k), Fr(x))
    if out.denominator != 1:
        raise TooFewDistractors("fractional output")
    out = int(out)
    story = {
        "add": f"{k} is added to",
        "sub": f"{k} is subtracted from",
        "mul": f"is multiplied by {k}",
        "div": f"is divided by {k}",
    }
    started = {
        "add": f"{k} is added to a number n and the result is {out}.",
        "sub": f"{k} is subtracted from a number n and the result is {out}.",
        "mul": f"A number n is multiplied by {k} and the result is {out}.",
        "div": f"A number n is divided by {k} and the result is {out}.",
    }[op]
    inv = _INVERSE[op]
    word = lambda o, n: {
        "add": f"Add {n} to {out}",
        "sub": f"Subtract {n} from {out}",
        "mul": f"Multiply {out} by {n}",
        "div": f"Divide {out} by {n}",
    }[o]  # noqa: E731
    if op == "sub":
        started = f"A number n has {k} subtracted from it and the result is {out}."
    if op == "add":
        started = f"{k} is added to a number n and the result is {out}."
    key = word(inv, k)
    others = [o for o in ("add", "sub", "mul", "div") if o not in (inv,)]
    tags = {op: "w5i-same-operation"}
    wrongs = [(word(o, k), tags.get(o, "w5i-mixes-operation-pairs")) for o in others]
    wrongs.append(
        (
            f"Divide {k} by {out}" if inv == "div" else f"Subtract {out} from {k}" if inv == "sub" else None,
            "w5i-reverses-the-numbers",
        )
    )
    for text, _ in list(wrongs):
        if text is not None and _calc_value(text) == x:
            raise TooFewDistractors("a distractor calculation also finds n")
    work = [
        f"n {'+' if op == 'add' else '-' if op == 'sub' else '×' if op == 'mul' else '÷'} {k} = {out}"
        if op in ("sub", "mul", "div")
        else f"{k} + n = {out}",
        f"{key} to undo it: n = {x}.",
    ]
    return _mc(
        rng,
        ("{s} Which calculation finds n?", "{s} Which of these finds the value of n?"),
        key,
        wrongs,
        work,
        s=started,
    )


def _machine(rng, two: bool = True, allow_rsub: bool = False):
    ops = ["add", "sub", "mul", "div"]
    nsteps = 2 if two else 3
    steps = []
    for _ in range(nsteps):
        op = rng.choice(ops)
        steps.append((op, rng.randint(2, 9)))
    return steps


def _inv_machine_steps(rng):
    for _ in range(80):
        steps = _machine(rng, True)
        if steps[0][0] in ("mul", "div") and steps[1][0] in ("mul", "div"):
            continue
        if steps[0][0] in ("add", "sub") and steps[1][0] in ("add", "sub"):
            continue
        x = Fr(rng.randint(2, 20))
        if any(_run(steps[: i + 1], x).denominator != 1 for i in range(2)):
            continue
        out = _run(steps, x)
        if out <= 0 or out == x:
            continue
        return steps, x, int(out)
    raise TooFewDistractors("no machine")


def _inv_machine(rng):
    steps, x, out = _inv_machine_steps(rng)
    key_steps = _undo(steps)
    bad_orders = [
        (list(steps), "w5i-same-operation"),
        (list(reversed(steps)), "w5i-same-operation"),
        ([(_INVERSE[op], n) for op, n in steps], "w5i-wrong-order"),
    ]
    first_swap = [(_INVERSE[steps[0][0]], steps[0][1])]
    bad_orders.append((first_swap + [(_INVERSE[steps[1][0]], steps[1][1] + 1)], "w5i-wrong-amount"))
    bad_orders.append(([key_steps[0]], "w5i-stops-after-one-step"))
    key = _cap(_steps_text(key_steps))
    wrongs = []
    for seq, tag in bad_orders:
        if len(seq) == len(steps) and _run(seq, Fr(out)) == x:
            continue
        if len(seq) < len(steps) and _run(seq, Fr(out)) == x:
            continue
        wrongs.append((_cap(_steps_text(seq)), tag))
    check(_run(key_steps, Fr(out)) == x, "the key does not return the input")
    for text, _ in wrongs:
        pass
    work = [
        f"The machine does: {_steps_text(steps)}.",
        f"To go back, undo each step in the reverse order: {_steps_text(key_steps)}.",
    ]
    stem = phrase(
        rng,
        (
            "A machine takes an input x and does these steps in order: {s}. The output is {o}. Which steps, in order, take the output back to x?",
            "x goes into a machine that does: {s}. Which steps, done in order on the output, give x back?",
        ),
        s=_steps_text(steps),
        o=out,
    )
    return _mc(rng, (stem,), key, wrongs, work)


def _inv_machine_value(rng):
    steps, x, out = _inv_machine_steps(rng)
    key_steps = _undo(steps)
    key = f"x = {rt(x)}"
    cands = []
    for seq, tag in (
        (list(steps), "w5i-same-operation"),
        ([(_INVERSE[op], n) for op, n in steps], "w5i-wrong-order"),
        ([(_INVERSE[steps[0][0]], steps[0][1])], "w5i-stops-after-one-step"),
        ([key_steps[0]], "w5i-stops-after-one-step"),
        ([(_INVERSE[steps[0][0]], steps[0][1]), (steps[1][0], steps[1][1])], "w5i-same-operation"),
    ):
        v = _run(seq, Fr(out))
        if v != x:
            cands.append((f"x = {rt(v)}", tag))
    check(_run(key_steps, Fr(out)) == x and _run(steps, x) == out, "machine check failed")
    work = [
        f"The steps are: {_steps_text(steps)}.",
        f"Undo them in reverse order: {_steps_text(key_steps)}, which gives x = {rt(x)}.",
    ]
    stem = phrase(
        rng,
        (
            "A machine does these steps in order to x: {s}. The output is {o}. What was the input x?",
            "The output of a machine is {o}. The machine does these steps in order to x: {s}. Find x.",
        ),
        s=_steps_text(steps),
        o=out,
    )
    return _mc(rng, (stem,), key, cands, work)


def _inv_rsub_machine(rng):
    k = rng.randint(10, 30)
    m = rng.randint(2, 6)
    x = rng.randint(2, k - 2)
    steps = [("rsub", k), ("mul", m)]
    out = _run(steps, Fr(x))
    key_steps = _undo(steps)
    check(_run(key_steps, out) == x, "key does not undo")
    key = _cap(_steps_text(key_steps))
    cands_seq = [
        ([("div", m), ("sub", k)], "w5i-rsub-treated-as-sub"),
        ([("div", m), ("add", k)], "w5i-rsub-treated-as-sub"),
        ([("sub", k), ("div", m)], "w5i-wrong-order"),
        ([("rsub", k), ("div", m)], "w5i-same-operation"),
        ([("div", m), ("rsub", k + 1)], "w5i-wrong-amount"),
        ([("div", m), ("rsub", k - 2)], "w5i-wrong-amount"),
        ([("div", m)], "w5i-stops-after-one-step"),
    ]
    wrongs = []
    for seq, tag in cands_seq:
        if _run(seq, out) != x:
            wrongs.append((_cap(_steps_text(seq)), tag))
    work = [
        f"The machine does: {_steps_text(steps)}.",
        f"Subtracting from {k} is its own inverse: if y = {k} - x then x = {k} - y. So reverse the order: {_steps_text(key_steps)}.",
    ]
    stem = phrase(
        rng,
        (
            "A machine does these steps in order to x: {s}. Which steps, in order, take the output back to x?",
            "x is put through these steps in order: {s}. Which steps, done in order on the output, give x back?",
        ),
        s=_steps_text(steps),
    )
    return _mc(rng, (stem,), key, wrongs, work)


def _inv_self_inverse(rng):
    k = rng.randint(6, 40)
    n = k
    key = f"n → {k} - n"
    wrongs = [
        (f"n → n + {k}", "w5i-undoes-itself"),
        (f"n → n - {k}", "w5i-undoes-itself"),
        (f"n → {k} × n", "w5i-undoes-itself"),
        (f"n → n ÷ {k}", "w5i-undoes-itself"),
    ]
    for t in (7, 11, 13):
        check(Fr(k) - (Fr(k) - t) == t, "the key is not its own inverse")
        check(t + k + k != t and t - k - k != t and t * k * k != t and Fr(t) / k / k != t, "a distractor undoes itself")
    work = [
        f"Apply n → {k} - n twice to 7: {k} - 7 = {k - 7}, then {k} - {paren_negative(k - 7)} = 7. It returns to the start.",
        "Adding, subtracting, multiplying or dividing by a number needs a different operation to undo it.",
    ]
    stems = (
        "Which of these rules undoes itself, so that doing it twice returns the number you started with?",
        "Applying one of these rules twice always gives back the starting number. Which rule is it?",
        "One of these rules is its own inverse. Which one?",
    )
    return _mc(rng, stems, key, wrongs, work)


def _inv_three_value(rng):
    """A three-step machine run backwards from its output (the input is a whole number throughout)."""
    for _ in attempts(300, "a three step machine"):
        steps = _machine(rng, two=False)
        families = {op in ("mul", "div") for op, _ in steps}
        if len(families) < 2 or len({s for s in steps}) < 3:
            continue
        x = Fr(rng.randint(2, 20))
        if any(_run(steps[: i + 1], x).denominator != 1 or _run(steps[: i + 1], x) <= 0 for i in range(3)):
            continue
        out = _run(steps, x)
        if out == x:
            continue
        break
    key_steps = _undo(steps)
    check(_run(key_steps, out) == x and _run(steps, x) == out, "three step machine check failed")
    cands = []
    for seq, tag in (
        (list(steps), "w5i-same-operation"),
        ([(_INVERSE[op], n) for op, n in steps], "w5i-wrong-order"),
        (key_steps[:1], "w5i-stops-after-one-step"),
        (key_steps[:2], "w5i-stops-after-one-step"),
        ([key_steps[0], key_steps[2], key_steps[1]], "w5i-wrong-order"),
    ):
        v = _run(seq, out)
        if v != x:
            cands.append((f"x = {rt(v)}", tag))
    work = [
        f"The steps are: {_steps_text(steps)}.",
        f"Undo them in the reverse order: {_steps_text(key_steps)}, which gives x = {rt(x)}.",
    ]
    stem = phrase(
        rng,
        (
            "A machine does these three steps in order to x: {s}. The output is {o}. What was the input x?",
            "Starting from the output {o}, work backwards through a machine that does these steps to x in order: {s}. Find x.",
        ),
        s=_steps_text(steps),
        o=rt(out),
    )
    return _mc(rng, (stem,), f"x = {rt(x)}", cands, work)


def _inv_equation_steps(rng):
    """The inverse operations, in order, that solve a x + b = c (or a x - b = c)."""
    a = rng.randint(2, 9)
    b = rng.randint(1, 12)
    plus = rng.random() < 0.5
    x = rng.randint(1, 12)
    c = a * x + (b if plus else -b)
    eq = f"{a}x {'+' if plus else '-'} {b} = {c}"
    undo_b, wrong_b = ("Subtract", "Add") if plus else ("Add", "Subtract")
    key = f"{undo_b} {b}, then divide by {a}"
    wrongs = [
        (f"Divide by {a}, then {undo_b.lower()} {b}", "w5i-wrong-order"),
        (f"{wrong_b} {b}, then divide by {a}", "w5i-same-operation"),
        (f"{undo_b} {b}, then multiply by {a}", "w5i-mixes-operation-pairs"),
        (f"{undo_b} {a}, then divide by {b}", "w5i-wrong-amount"),
        (f"Divide by {a}", "w5i-stops-after-one-step"),
    ]
    check((c + (-b if plus else b)) / a == x, "the equation does not give x")
    work = [
        f"{a}x {'+' if plus else '-'} {b} = {c}: x is multiplied by {a}, then {b} is {'added' if plus else 'subtracted'}.",
        f"Undo in the reverse order: {undo_b.lower()} {b} {'to' if undo_b == 'Add' else 'from'} both sides, then divide by {a}, giving x = {x}.",
    ]
    stems = (
        "To solve {e}, which inverse operations, in order, should be applied to both sides?",
        "Which steps, in order, undo the operations in {e} to find x?",
    )
    return _mc(rng, stems, key, wrongs, work, e=eq)


def _inv_exam(rng):
    steps, x, out = _inv_machine_steps(rng)
    key_steps = _undo(steps)
    x2 = Fr(rng.randint(2, 25))
    if x2 == x or _run(steps, x2) <= 0 or any(_run(steps[: i + 1], x2).denominator != 1 for i in range(2)):
        raise TooFewDistractors("second input")
    out2 = int(_run(steps, x2))
    pa = mcq_fields(_cap(_steps_text(key_steps)))
    da = mcq_cands(
        _cap(_steps_text(key_steps)),
        [
            (_cap(_steps_text(list(steps))), "w5i-same-operation"),
            (_cap(_steps_text([(_INVERSE[o], n) for o, n in steps])), "w5i-wrong-order"),
            (_cap(_steps_text(list(reversed(steps)))), "w5i-same-operation"),
            (_cap(_steps_text([key_steps[0]])), "w5i-stops-after-one-step"),
        ],
        4,
    )
    pb = number_fields(x)
    db = number_cands(
        x,
        [
            (_run(steps, Fr(out)), "w5i-same-operation"),
            (_run([(_INVERSE[o], n) for o, n in steps], Fr(out)), "w5i-wrong-order"),
            (_run([key_steps[0]], Fr(out)), "w5i-stops-after-one-step"),
            (x + 1, "w5i-wrong-amount"),
        ],
    )
    pc = number_fields(x2)
    dc = number_cands(
        x2,
        [
            (_run(steps, Fr(out2)), "w5i-same-operation"),
            (_run([(_INVERSE[o], n) for o, n in steps], Fr(out2)), "w5i-wrong-order"),
            (_run([key_steps[0]], Fr(out2)), "w5i-stops-after-one-step"),
            (x2 + 1, "w5i-wrong-amount"),
        ],
    )
    for dd in (da, db, dc):
        enough(dd, 2)
    check(_run(key_steps, Fr(out)) == x and _run(key_steps, Fr(out2)) == x2, "machine check failed")
    return dict(
        text=f"A machine does these steps in order to an input x: {_steps_text(steps)}.",
        parts=[
            part(
                "Which steps, in order, take the output of the machine back to x?",
                "mcq",
                _sub(pa),
                1,
                [f"Undo each step in the reverse order: {_steps_text(key_steps)}."],
                da,
            ),
            part(
                f"The output is {out}. What is the input x?",
                pb["answer_kind"],
                _sub(pb),
                2,
                [f"Run the undo steps on {out}: x = {rt(x)}."],
                db,
            ),
            part(
                f"For a different input the output is {out2}. What is that input?",
                pc["answer_kind"],
                _sub(pc),
                1,
                [f"Run the undo steps on {out2}: the input is {rt(x2)}."],
                dc,
            ),
        ],
    )


@generator(
    "w5_inverse_operations",
    level=17,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def w5_inverse_operations(rng, tier="easy"):
    """Undoing operations; the tiers are described in the module docstring."""
    pools = {
        "easy": (_inv_pair,),
        "medium": (_inv_machine, _inv_machine_value),
        "hard": (_inv_rsub_machine, _inv_self_inverse, _inv_three_value, _inv_equation_steps),
    }
    return _tiers(pools, rng, tier)


@generator("exd_x_w5_inverse_operations", level=17, presets={"exam": dict(tier="exam")})
def exd_x_w5_inverse_operations(rng, tier="exam"):
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    return retry_draw(rng, _inv_exam)


# =====================================================================================================================
# solutions and solution sets
# =====================================================================================================================


def _lin_text(a: int, b: int, var: str = "n") -> str:
    return join_terms([(a, var), (b, "")])


def _holds(a: int, b: int, c: int, v: Fr) -> bool:
    """Whether a * v + b == c."""
    return a * v + b == c


def _fmt_set(values) -> str:
    return "{" + ", ".join(rt(v) for v in sorted(values)) + "}"


def _ss_find(rng):
    s = rng.choice([v for v in range(-9, 13) if v not in (0, 1)])
    a = rng.randint(2, 9)
    b = rng.choice([v for v in range(-12, 13) if v != 0])
    c = a * s + b
    eq = f"{_lin_text(a, b)} = {c}"
    key = f"n = {s}"
    cands = [
        (Fr(c), "w5s-takes-the-constant"),
        (Fr(c - b), "w5s-forgets-to-divide"),
        (Fr(c + b, a), "w5s-sign-slip"),
        (Fr(s + 1), "w5s-off-by-one"),
        (Fr(s - 1), "w5s-off-by-one"),
    ]
    wrongs = []
    for v, tag in cands:
        if v != s:
            check(not _holds(a, b, c, v), "a distractor solves the equation")
            wrongs.append((f"n = {rt(v)}", tag))
    check(_holds(a, b, c, Fr(s)), "the key does not solve the equation")
    work = [
        f"Substitute each value into {eq}. When n = {s}: {a} × {s} {'+' if b > 0 else '-'} {abs(b)} = {c}, which equals the right side.",
        "The other values make the left side different from the right side.",
    ]
    return _mc(
        rng,
        (
            "Which value is a solution of {e}?",
            "Which of these values of n makes {e} true?",
            "Which value of n solves {e}?",
        ),
        key,
        wrongs,
        work,
        e=eq,
    )


def _ss_which_equation(rng):
    s = rng.choice([v for v in range(-6, 12) if v not in (0, 1)])
    a = rng.randint(2, 7)
    b = rng.choice([v for v in range(-9, 10) if v != 0])
    c = a * s + b
    key = f"{_lin_text(a, b)} = {c}"
    offs = rng.sample([-6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6], 2)
    cands = [((a, b, c + offs[0]), "w5s-expression-value-not-checked")]
    cands.append(((a, b + rng.choice((-3, -2, -1, 1, 2, 3)), c), "w5s-expression-value-not-checked"))
    cands.append(((a, b, s), "w5s-solution-copied-from-equation"))
    cands.append(((a + rng.choice((-1, 1, 2)), b, c), "w5s-expression-value-not-checked"))
    cands.append(((a, b, c + offs[1]), "w5s-expression-value-not-checked"))
    cands.append(((a, -b, c), "w5s-sign-slip"))
    rng.shuffle(cands)
    wrongs = []
    for (aa, bb, cc), tag in cands:
        if not _holds(aa, bb, cc, Fr(s)):
            wrongs.append((f"{_lin_text(aa, bb)} = {cc}", tag))
    check(_holds(a, b, c, Fr(s)), "the key does not hold")
    work = [
        f"Substitute n = {s} into the left side of each equation and compare it with the right side.",
        f"{_lin_text(a, b)} = {c} is the only one where they are equal.",
    ]
    return _mc(
        rng,
        (
            "For which of these equations is n = {s} a solution?",
            "Which equation has n = {s} as a solution?",
            "Which equation is made true by n = {s}?",
        ),
        key,
        wrongs,
        work,
        s=s,
    )


def _ss_two_sides(rng):
    s = rng.choice([v for v in range(-8, 12) if v not in (0, 1)])
    c = rng.randint(1, 5)
    a = c + rng.randint(1, 5)
    b = rng.choice([v for v in range(-12, 13) if v != 0])
    d = (a - c) * s + b
    left, right = _lin_text(a, b), _lin_text(c, d)
    eq = f"{left} = {right}"
    key = f"n = {s}"

    def holds(v):
        return a * v + b == c * v + d

    cands = [
        (Fr(d + b, a - c), "w5s-sign-slip"),
        (Fr(d - b, a + c), "w5s-adds-the-coefficients"),
        (Fr(d - b), "w5s-forgets-to-divide"),
        (Fr(s + 1), "w5s-off-by-one"),
        (Fr(s - 1), "w5s-off-by-one"),
    ]
    wrongs = []
    for v, tag in cands:
        if v != s and not holds(v):
            wrongs.append((f"n = {rt(v)}", tag))
    check(holds(Fr(s)), "the key does not solve the equation")
    work = [
        f"Substitute n = {s}: left side {a * s + b}, right side {c * s + d}. They are equal.",
        "For the other values the two sides are different.",
    ]
    return _mc(
        rng,
        ("Which value is a solution of {e}?", "Which value of n makes both sides of {e} equal?"),
        key,
        wrongs,
        work,
        e=eq,
    )


def _roots_sets(rng):
    kind = rng.choice(("sq", "prod", "zero"))
    k = rng.randint(2, 9)
    if kind == "sq":
        eq, true, wrongs = (
            f"x^2 = {k * k}",
            {-k, k},
            [
                ({k}, "w5s-misses-the-negative"),
                ({-k}, "w5s-misses-the-positive"),
                ({k * k}, "w5s-gives-the-number-not-the-root"),
                ({-k * k, k * k}, "w5s-gives-the-number-not-the-root"),
            ],
        )
    elif kind == "zero":
        eq, true, wrongs = (
            f"x(x - {k}) = 0",
            {0, k},
            [
                ({k}, "w5s-misses-zero"),
                ({0}, "w5s-misses-the-other-root"),
                ({-k, 0}, "w5s-sign-slip"),
                ({-k}, "w5s-sign-slip"),
                ({k, k * k}, "w5s-gives-the-number-not-the-root"),
            ],
        )
    else:
        m = rng.choice([v for v in range(1, 10) if v != k])
        eq, true, wrongs = (
            f"(x - {k})(x + {m}) = 0",
            {k, -m},
            [
                ({k, m}, "w5s-sign-slip"),
                ({-k, m}, "w5s-sign-slip"),
                ({k}, "w5s-misses-the-other-root"),
                ({-m}, "w5s-misses-the-other-root"),
                ({-k, -m}, "w5s-sign-slip"),
            ],
        )
    return eq, true, wrongs


def _eval_eq(eq: str, x: int) -> bool:
    left, right = eq.split(" = ")
    if left.startswith("x^2"):
        return x * x == int(right)
    if left.startswith("x(x - "):
        k = int(left[6:-1])
        return x * (x - k) == int(right)
    k, m = (int(t) for t in left.replace("(x - ", " ").replace("(x + ", " ").replace(")", "").split())
    return (x - k) * (x + m) == int(right)


def _ss_set(rng):
    eq, true, wrongs = _roots_sets(rng)
    found = {x for x in range(-90, 91) if _eval_eq(eq, x)}
    check(found == true, "the solution set differs from the brute-force scan")
    key = _fmt_set(true)
    tagged = [(_fmt_set(w), t) for w, t in wrongs if set(w) != true]
    work = [
        f"Test the integers in {eq}: exactly {', '.join(str(v) for v in sorted(true))} make{'s' if len(true) == 1 else ''} it true.",
        "The solution set lists every value that works.",
    ]
    return _mc(
        rng,
        (
            "Which set is the solution set of {e}, when x can be any integer?",
            "Which set contains every integer solution of {e}?",
            "The equation {e} is solved over the integers. Which set lists all of its solutions?",
        ),
        key,
        tagged,
        work,
        e=eq,
    )


def _ss_restricted(rng):
    k = rng.randint(2, 9)
    kind = rng.choice(("sq", "zero"))
    if kind == "sq":
        eq, domain, true = f"x^2 = {k * k}", "a positive whole number", {k}
        wrongs = [
            ({-k, k}, "w5s-ignores-the-stated-set"),
            ({-k}, "w5s-misses-the-positive"),
            ({k * k}, "w5s-gives-the-number-not-the-root"),
            ({k, k * k}, "w5s-gives-the-number-not-the-root"),
        ]
        found = {x for x in range(1, 91) if x * x == k * k}
    else:
        eq, domain, true = f"x(x - {k}) = 0", "a positive whole number", {k}
        wrongs = [
            ({0, k}, "w5s-ignores-the-stated-set"),
            ({0}, "w5s-misses-the-other-root"),
            ({-k, 0}, "w5s-sign-slip"),
            ({k, k * k}, "w5s-gives-the-number-not-the-root"),
        ]
        found = {x for x in range(1, 91) if x * (x - k) == 0}
    check(found == true, "restricted solution set differs from the scan")
    if rng.random() < 0.4:
        # whole numbers (zero allowed) for the product equation
        if kind == "zero":
            domain, true = "a whole number (0, 1, 2, ...)", {0, k}
            found = {x for x in range(0, 91) if x * (x - k) == 0}
            check(found == true, "whole-number set differs from the scan")
            wrongs = [
                ({k}, "w5s-misses-zero"),
                ({-k, 0, k}, "w5s-ignores-the-stated-set"),
                ({0}, "w5s-misses-the-other-root"),
                ({k, k * k}, "w5s-gives-the-number-not-the-root"),
            ]
    key = _fmt_set(true)
    tagged = [(_fmt_set(w), t) for w, t in wrongs if set(w) != true]
    work = [f"Solve {eq}, then keep only the solutions that are {domain}.", f"The solution set is {key}."]
    return _mc(
        rng,
        (
            "Solve {e} where x is {d}. Which set is the solution set?",
            "x is {d}. Which set contains every solution of {e}?",
        ),
        key,
        tagged,
        work,
        e=eq,
        d=domain,
    )


def _poly_ok(roots, target) -> bool:
    return set(roots) == set(target)


def _ss_equation_for_set(rng):
    a, b = rng.sample([v for v in range(-8, 9) if v != 0], 2)
    key_roots = (a, b)
    style = rng.choice(("factored", "expanded"))

    def fmt(r):
        if style == "factored":
            return f"(x {'-' if r[0] > 0 else '+'} {abs(r[0])})(x {'-' if r[1] > 0 else '+'} {abs(r[1])}) = 0"
        return f"{join_terms([(1, 'x^2'), (-(r[0] + r[1]), 'x'), (r[0] * r[1], '')])} = 0"

    cand_roots = [
        ((-a, -b), "w5s-sign-slip"),
        ((a, -b), "w5s-sign-slip"),
        ((a + b, a * b), "w5s-sum-and-product-as-roots"),
        ((a, a), "w5s-repeats-one-root"),
        ((a * b, a + b), "w5s-sum-and-product-as-roots"),
    ]
    wrongs = []
    for roots, tag in cand_roots:
        if not _poly_ok(roots, key_roots) and fmt(roots) != fmt(key_roots):
            wrongs.append((fmt(roots), tag))
    key = fmt(key_roots)
    check(
        (a * a - (a + b) * a + a * b) == 0 and (b * b - (a + b) * b + a * b) == 0,
        "the key polynomial does not vanish at its roots",
    )
    work = [
        f"Substitute x = {a} and x = {b} into each equation. Only {key} is true for both.",
        "An equation with these two solutions has the brackets (x minus each solution).",
    ]
    return _mc(
        rng,
        (
            "Which equation has exactly the solutions x = {a} and x = {b}?",
            "The solution set of an equation is {{{a}, {b}}}. Which equation could it be?",
        ),
        key,
        wrongs,
        work,
        a=a,
        b=b,
    )


def _ss_exam(rng):
    s = rng.choice([v for v in range(-6, 10) if v not in (0, 1)])
    c = rng.randint(1, 4)
    a = c + rng.randint(1, 4)
    b = rng.choice([v for v in range(-10, 11) if v != 0])
    d = (a - c) * s + b
    eq = f"{_lin_text(a, b)} = {_lin_text(c, d)}"
    key_a = f"n = {s}"
    wr = [
        (Fr(d + b, a - c), "w5s-sign-slip"),
        (Fr(d - b, a + c), "w5s-adds-the-coefficients"),
        (Fr(d - b), "w5s-forgets-to-divide"),
        (Fr(s + 1), "w5s-off-by-one"),
    ]
    wa = [(f"n = {rt(v)}", t) for v, t in wr if v != s and a * v + b != c * v + d]
    fa = mcq_fields(key_a)
    da = mcq_cands(key_a, wa, 4)
    side = a * s + b
    check(side == c * s + d, "the sides differ at the solution")
    fb = number_fields(side)
    db = number_cands(
        side,
        [
            (s, "w5s-takes-the-constant"),
            (a * s, "w5s-sign-slip"),
            (side + a, "w5s-off-by-one"),
            (b + d, "w5s-sum-of-constants"),
        ],
    )
    k = rng.randint(2, 4)
    key_c = f"{_lin_text(a - c, b)} = {d}"
    # an equivalent equation: subtract c n from both sides
    check((a - c) * s + b == d, "the equivalent equation does not hold")
    cwr = [
        ((a - c, b, d + k), "w5s-expression-value-not-checked"),
        ((a + c, b, d), "w5s-adds-the-coefficients"),
        ((a - c, -b, d), "w5s-sign-slip"),
        ((a, b, d), "w5s-solution-copied-from-equation"),
    ]
    wc = [(f"{_lin_text(aa, bb)} = {dd}", t) for (aa, bb, dd), t in cwr if not _holds(aa, bb, dd, Fr(s))]
    fc = mcq_fields(key_c)
    dc = mcq_cands(key_c, wc, 4)
    enough(da, 4)
    enough(dc, 4)  # both choice parts always show five options
    enough(db, 2)
    return dict(
        text=f"Consider the equation {eq}.",
        parts=[
            part(
                "Which value of n is a solution of this equation?",
                "mcq",
                _sub(fa),
                1,
                [f"Substitute n = {s}: both sides equal {side}."],
                da,
            ),
            part(
                f"What does each side of the equation equal when n = {s}?",
                fb["answer_kind"],
                _sub(fb),
                1,
                [f"{_lin_text(a, b)} with n = {s} gives {side}."],
                db,
            ),
            part(
                "Which of these equations has the same solution as the given equation?",
                "mcq",
                _sub(fc),
                2,
                [f"Subtract {c if c != 1 else ''}n from both sides: {key_c}. It is true when n = {s}."],
                dc,
            ),
        ],
    )


@generator(
    "w5_solution_set",
    level=17,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def w5_solution_set(rng, tier="easy"):
    """Solutions and solution sets; the tiers are described in the module docstring."""
    pools = {
        "easy": (_ss_find, _ss_which_equation),
        "medium": (_ss_two_sides, _ss_set),
        "hard": (_ss_equation_for_set, _ss_restricted),
    }
    return _tiers(pools, rng, tier)


@generator("exd_x_w5_solution_set", level=17, presets={"exam": dict(tier="exam")})
def exd_x_w5_solution_set(rng, tier="exam"):
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    return retry_draw(rng, _ss_exam)


# =====================================================================================================================
# identities and equations with no solution
# =====================================================================================================================


def _render(rng, coef: int, const: int, style: str) -> str:
    """A side of an equation equal to coef * x + const, written plainly or with a bracket or split terms."""
    if style == "bracket" and coef not in (0, 1, -1):
        p = rng.choice([v for v in range(-5, 6) if v != 0])
        q = const - coef * p
        inner = f"x {'+' if p > 0 else '-'} {abs(p)}"
        return join_terms([(coef, f"({inner})"), (q, "")]) if coef != 1 else join_terms([(1, f"({inner})"), (q, "")])
    if style == "split" and abs(coef) >= 2:
        c1 = rng.randint(1, 4) + max(coef, 0)
        return join_terms([(c1, "x"), (coef - c1, "x"), (const, "")])
    return join_terms([(coef, "x"), (const, "")])


def _classify(a: int, b: int, c: int, d: int):
    """('one', x), ('none', None) or ('all', None) for a x + b = c x + d, by testing the difference at x = 0, 1."""
    d0 = (a * 0 + b) - (c * 0 + d)
    d1 = (a * 1 + b) - (c * 1 + d)
    if d0 == 0 and d1 == 0:
        return ("all", None)
    if d1 == d0:
        return ("none", None)
    return ("one", Fr(-d0, d1 - d0))


def _outcome_options(rng, kind: str, a: int, b: int, c: int, d: int, s):
    """Key and tagged wrong options for an equation of the given kind."""
    one = lambda v: f"Exactly one solution, x = {rt(v)}"  # noqa: E731
    if kind == "all":
        key = "Every number is a solution"
        k = b if b else 1
        wrongs = [
            ("No solution", "w5n-cancelling-means-no-solution"),
            (one(0), "w5n-zero-equals-zero-is-x-zero"),
            (one(Fr(b)) if b else one(1), "w5n-constant-as-solution"),
            (one(Fr(b - 1)) if b - 1 != 0 else one(2), "w5n-constant-as-solution"),
        ]
    elif kind == "none":
        key = "No solution"
        wrongs = [
            ("Every number is a solution", "w5n-cancelling-means-every-number"),
            (one(0), "w5n-zero-equals-zero-is-x-zero"),
            (one(Fr(b - d)), "w5n-difference-as-solution"),
            (one(Fr(d)), "w5n-constant-as-solution"),
        ]
    else:
        key = one(s)
        wrongs = [
            ("No solution", "w5n-cancelling-means-no-solution"),
            ("Every number is a solution", "w5n-cancelling-means-every-number"),
            (one(-s) if s != 0 else one(1), "w5n-sign-slip"),
            (one(Fr(d - b)) if Fr(d - b) != s else one(s + 1), "w5n-forgets-to-divide"),
        ]
    return key, wrongs


def _make_equation(rng, kind: str, style_l: str, style_r: str):
    for _ in range(80):
        a = rng.randint(2, 9)
        if kind == "one":
            c = rng.choice([v for v in range(-3, 10) if v != a])
            s = rng.choice([v for v in range(-8, 9) if v != 0])
            b = rng.choice([v for v in range(-12, 13) if v != 0])
            d = (a - c) * s + b
        else:
            c = a
            b = rng.choice([v for v in range(-12, 13) if v != 0])
            d = b if kind == "all" else b + rng.choice([v for v in range(-6, 7) if v != 0])
            s = None
        left, right = _render(rng, a, b, style_l), _render(rng, c, d, style_r)
        if left == right:
            continue
        got = _classify(a, b, c, d)
        check(got[0] == kind and (kind != "one" or got[1] == s), "classification differs from the construction")
        return a, b, c, d, s, left, right
    raise TooFewDistractors("no equation")


def _inos_classify(rng, styles, kinds=("one", "none", "all")):
    kind = rng.choice(kinds)
    a, b, c, d, s, left, right = _make_equation(rng, kind, rng.choice(styles), rng.choice(styles))
    key, wrongs = _outcome_options(rng, kind, a, b, c, d, s)
    eq = f"{left} = {right}"
    if kind == "all":
        work = [
            f"Simplify both sides: each side becomes {a}x {'+' if b > 0 else '-'} {abs(b)}.",
            "The two sides are always equal, so every number is a solution.",
        ]
    elif kind == "none":
        work = [
            f"Collect the x terms: both sides have {a}x, so they cancel and leave {b} = {d}.",
            f"{b} = {d} is never true, so there is no solution.",
        ]
    else:
        work = [f"Collect the x terms: {fmt_signed(a - c, 'x')} = {d - b}.", f"So x = {rt(s)}: exactly one solution."]
    stems = (
        "How many solutions does {e} have?",
        "Which statement about the equation {e} is correct?",
        "Solve {e}. Which description of its solutions is correct?",
    )
    return _mc(rng, stems, key, wrongs, work, e=eq)


def _inos_easy(rng):
    return _inos_classify(rng, ("plain",))


def _inos_final_form(rng):
    kind = rng.choice(("one", "none", "all"))
    if kind == "one":
        s = rng.choice([v for v in range(-9, 10) if v != 0])
        shown, a, b, c, d = f"x = {s}", 1, 0, 0, s
        key, wrongs = _outcome_options(rng, "one", 2, 0, 1, s, Fr(s))
        work = [f"The equation says x must be {s}, which is a single value, so there is exactly one solution."]
    elif kind == "all":
        v = rng.randint(1, 15)
        shown = f"{v} = {v}"
        key, wrongs = _outcome_options(rng, "all", 1, v, 1, v, None)
        work = [f"{v} = {v} is always true, whatever x is, so every number is a solution."]
    else:
        v = rng.randint(1, 15)
        w = v + rng.choice([-4, -3, -2, -1, 1, 2, 3, 4, 5])
        shown = f"{v} = {w}"
        key, wrongs = _outcome_options(rng, "none", 1, v, 1, w, None)
        work = [f"{v} = {w} is never true, whatever x is, so there is no solution."]
    stem = phrase(
        rng,
        (
            "After collecting terms, {n}'s working on an equation ends with {t}. What does this tell you about the solutions of the equation?",
            "Working on an equation ends with the line {t}. Which statement about the equation is correct?",
        ),
        n=rng.choice(NAMES),
        t=shown,
    )
    return _mc(rng, (stem,), key, wrongs, work)


def _inos_medium(rng):
    return _inos_classify(rng, ("plain", "bracket", "bracket"))


def _inos_hard(rng):
    return _inos_classify(rng, ("bracket", "split", "bracket"))


def _inos_find_k(rng):
    mode = rng.choice(("none", "all"))
    a = rng.randint(2, 9)
    b = rng.choice([v for v in range(-9, 10) if v != 0])
    d = b + rng.choice([v for v in range(-6, 7) if v != 0]) if mode == "none" else b
    eq = f"{join_terms([(a, 'x'), (b, '')])} = {join_terms([(1, 'kx'), (d, '')]).replace('1kx', 'kx')}"
    key = f"k = {a}"
    pool = [
        (b, "w5n-constant-as-solution"),
        (d, "w5n-constant-as-solution"),
        (a + 1, "w5n-off-by-one"),
        (a - 1, "w5n-off-by-one"),
        (b - d if mode == "none" else a * 2, "w5n-difference-as-solution"),
        (-a, "w5n-sign-slip"),
    ]
    wrongs = []
    for v, tag in pool:
        kind = _classify(a, b, v, d)[0]
        if v != a and kind == "one":
            wrongs.append((f"k = {v}", tag))
    check(_classify(a, b, a, d)[0] == mode, "the key does not give the stated behaviour")
    word = "no solution" if mode == "none" else "infinitely many solutions"
    work = [
        f"The equation has {word} only when the x terms match: {a}x and kx must be the same, so k = {a}.",
        f"Then the constants {b} and {d} {'differ' if mode == 'none' else 'are equal'}, giving {word}.",
    ]
    stems = (
        "For which value of k does the equation {e} have {w}?",
        "The equation {e} has {w}. What is the value of k?",
    )
    return _mc(rng, stems, key, wrongs, work, e=eq, w=word)


def _inos_exam(rng):
    a = rng.randint(2, 9)
    p = rng.randint(1, 9)
    q = rng.choice([v for v in range(-9, 10) if v != 0])
    r = a * p + q
    kk = r + rng.choice([-4, -3, -2, -1, 1, 2, 3, 4])
    eq = f"{join_terms([(a, f'(x + {p})'), (q, '')])} = {a}x + k"
    # (a) k = kk: no solution; (b) the k with infinitely many; (c) how many solutions when k = r (identity)
    ka = f"No solution, because {r} = {kk} is false"
    wa = [
        ("Every number is a solution, because the x terms cancel", "w5n-cancelling-means-every-number"),
        ("Exactly one solution, x = 0, because the x terms cancel to leave 0", "w5n-zero-equals-zero-is-x-zero"),
        (f"Exactly one solution, x = {kk}, because the constant on the right is left over", "w5n-constant-as-solution"),
        (f"Exactly one solution, x = {r - kk}, because the constants are subtracted", "w5n-difference-as-solution"),
    ]
    fa = mcq_fields(ka)
    da = mcq_cands(ka, wa, 4)
    fb = number_fields(r)
    db = number_cands(
        r,
        [
            (a * p, "w5n-constant-as-solution"),
            (a, "w5n-sign-slip"),
            (r + 1, "w5n-off-by-one"),
            (r - 1, "w5n-off-by-one"),
            (a + p, "w5n-adds-instead-of-multiplies"),
        ],
    )
    check(_classify(a, r, a, kk)[0] == "none" and _classify(a, r, a, r)[0] == "all", "the exam classification is wrong")
    key_c = "No solution"
    wc = [
        ("Every number is a solution", "w5n-cancelling-means-every-number"),
        (f"Exactly one solution, x = 0", "w5n-zero-equals-zero-is-x-zero"),
        (f"Exactly one solution, x = {r}", "w5n-constant-as-solution"),
        (f"Exactly one solution, x = {kk}", "w5n-constant-as-solution"),
    ]
    fc = mcq_fields(key_c)
    dc = mcq_cands(key_c, wc, 4)
    for dd in (da, db, dc):
        enough(dd, 2)
    return dict(
        text=f"The equation {eq} has an unknown number k in it.",
        parts=[
            part(
                f"{rng.choice(NAMES)} puts k = {kk} into the equation. Which statement about its solutions is correct?",
                "mcq",
                _sub(fa),
                1,
                [
                    f"Expand the left side: {join_terms([(a, 'x'), (r, '')])} = {join_terms([(a, 'x'), (kk, '')])}. The x terms cancel and {r} = {kk} is false: no solution."
                ],
                da,
            ),
            part(
                "For which value of k does the equation have infinitely many solutions?",
                fb["answer_kind"],
                _sub(fb),
                2,
                [f"Expanding the left side gives {join_terms([(a, 'x'), (r, '')])}, so k must be {r}."],
                db,
            ),
            part(
                f"Which statement is true for every value of k except {r}?",
                "mcq",
                _sub(fc),
                1,
                [f"The x terms cancel, leaving {r} = k. This is false whenever k is not {r}."],
                dc,
            ),
        ],
    )


@generator(
    "w5_identity_no_solution",
    level=17,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def w5_identity_no_solution(rng, tier="easy"):
    """One solution, none, or every number; the tiers are described in the module docstring."""
    pools = {
        "easy": (_inos_easy, _inos_final_form),
        "medium": (_inos_medium,),
        "hard": (_inos_hard, _inos_find_k),
    }
    return _tiers(pools, rng, tier)


@generator("exd_x_w5_identity_no_solution", level=17, presets={"exam": dict(tier="exam")})
def exd_x_w5_identity_no_solution(rng, tier="exam"):
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    return retry_draw(rng, _inos_exam)


# =====================================================================================================================
# checking a solution by substitution
# =====================================================================================================================


def _sides_option(left, right, v, flip: bool = False) -> str:
    """The values of the two sides and the conclusion drawn from them (reversed when `flip`)."""
    is_solution = (left == right) != flip
    return f"Left side {rt(left)}, right side {rt(right)}: x = {rt(v)} {'is' if is_solution else 'is not'} a solution"


def _check_eq_values(rng, bracket: bool = False, negative: bool = False, one_sided: bool = False):
    """(a, b, c, d, v, is_true) for a x + b = c x + d with a proposed value v (c = 0 means a constant right side)."""
    for _ in range(80):
        a = rng.randint(2, 9)
        c = 0 if one_sided else rng.choice([v for v in range(-4, 8) if v not in (0, a)])
        s = rng.choice([v for v in range(-9, 10) if v not in (0, 1)]) if negative else rng.randint(2, 12)
        if negative and s > 0:
            s = -s
        b = rng.choice([v for v in range(-10, 11) if v != 0])
        d = (a - c) * s + b
        if one_sided:
            d = a * s + b
        is_true = rng.random() < 0.5
        v = s if is_true else s + rng.choice((-3, -2, -1, 1, 2, 3))
        if v == 0 or (negative and v > 0):
            continue
        return a, b, c, d, v, is_true
    raise TooFewDistractors("no equation to check")


def _lhs(a, b, v):
    return a * v + b


def _rhs(c, d, v):
    return c * v + d


def _check_verdict(rng, one_sided: bool, negative: bool = False):
    a, b, c, d, v, is_true = _check_eq_values(rng, negative=negative, one_sided=one_sided)
    L, R = _lhs(a, b, v), _rhs(c, d, v)
    check((L == R) == is_true, "truth differs")
    left_txt = join_terms([(a, "x"), (b, "")])
    right_txt = str(d) if one_sided else join_terms([(c, "x"), (d, "")])
    eq = f"{left_txt} = {right_txt}"
    key = _sides_option(L, R, v)
    joined = int(f"{a}{v}") + b if v > 0 else None
    cands = [
        (_sides_option(L, R, v, flip=True), "w5k-wrong-verdict"),
    ]
    if joined is not None:
        cands.append((_sides_option(joined, R, v), "w5k-joins-digits"))
    cands.append((_sides_option(v + b, R, v), "w5k-ignores-coefficient"))
    cands.append((_sides_option(v + b, R, v, flip=True), "w5k-wrong-verdict"))
    if joined is not None:
        cands.append((_sides_option(joined, R, v, flip=True), "w5k-wrong-verdict"))
    if not one_sided:
        cands.append(
            (
                _sides_option(L, c + v + d if v > 0 else c * abs(v) + d, v),
                "w5k-negative-sign-dropped" if v < 0 else "w5k-ignores-coefficient",
            )
        )
        cands.append((_sides_option(abs(a * v) + b, R, v), "w5k-negative-sign-dropped"))
    else:
        cands.append((_sides_option(L, R + b, v), "w5k-moves-the-constant-wrongly"))
        cands.append((_sides_option(a * abs(v) + b if v < 0 else a + v + b, R, v), "w5k-adds-instead-of-multiplies"))
    wrongs = length_balanced(rng, key, [(text, tag) for text, tag in cands if text != key])
    work = [
        f"Substitute x = {v}: the left side is {a} × {rt(Fr(v))} {'+' if b > 0 else '-'} {abs(b)} = {L}.",
        f"The right side is {R}. {'The sides are equal' if L == R else 'The sides are not equal'}, so x = {v} {'is' if L == R else 'is not'} a solution.",
    ]
    stems = (
        "Check whether x = {v} is a solution of {e}. Which working and conclusion is correct?",
        "Substitute x = {v} into {e}. Which statement gives the correct value of each side and the correct conclusion?",
    )
    return _mc(rng, stems, key, wrongs, work, v=v, e=eq)


def _check_line(rng):
    a, b = rng.randint(2, 9), rng.choice([v for v in range(-9, 10) if v != 0])
    v = rng.randint(2, 12)
    L = a * v + b
    expr = join_terms([(a, "x"), (b, "")])
    sign, mag = ("+", b) if b > 0 else ("-", -b)
    key = f"{a} × {v} {sign} {mag} = {L}"
    cands = [
        (f"{a}{v} {sign} {mag} = {int(f'{a}{v}') + b}", "w5k-joins-digits"),
        (f"{a} + {v} {sign} {mag} = {a + v + b}", "w5k-adds-instead-of-multiplies"),
        (f"{a} × {v} × {mag} = {a * v * mag}", "w5k-multiplies-the-constant"),
        (f"{a} × {L} {sign} {mag} = {a * L + b}", "w5k-substitutes-the-answer"),
        (f"{a} × {v + b} = {a * (v + b)}", "w5k-brackets-the-constant"),
    ]
    work = [
        f"Replace x by {v} in {expr}: {a} × {v} {sign} {mag}.",
        f"{a} × {v} = {a * v}, then {a * v} {sign} {mag} = {L}.",
    ]
    stems = (
        "What is the correct working for the left side of {e} when x = {v}?",
        "Substitute x = {v} into {e}. Which line shows the left side worked out correctly?",
    )
    return _mc(rng, stems, key, [(t, tg) for t, tg in cands if t != key], work, e=expr, v=v)


def _check_find_mistake(rng):
    for _ in range(60):
        a = rng.randint(2, 6)
        p = rng.randint(1, 6)
        q = rng.choice([v for v in range(-8, 9) if v != 0])
        v = -rng.randint(2, 9)
        L = a * (v + p) + q
        mistakes = {
            "brackets": (
                a * v + p + q,
                "She multiplied only the x inside the bracket by the number outside it",
                "w5k-brackets-ignored",
            ),
            "negative": (
                a * (abs(v) + p) + q,
                "She treated the negative value of x as if it were positive",
                "w5k-negative-sign-dropped",
            ),
            "coefficient": (
                (v + p) + q,
                "She did not multiply the bracket by the number outside it",
                "w5k-ignores-coefficient",
            ),
            "constant": (
                a * (v + p) - q,
                "She changed the sign of the last number in the expression",
                "w5k-sign-slip-on-constant",
            ),
        }
        actual = rng.choice(list(mistakes))
        shown = mistakes[actual][0]
        if len({m[0] for m in mistakes.values()} | {L}) < 5 or shown == L:
            continue
        q_txt = f"{'+' if q > 0 else '-'} {abs(q)}"
        eq_left = f"{a}(x + {p}) {q_txt}"
        key = mistakes[actual][1]
        wrongs = [(m[1], m[2]) for k, m in mistakes.items() if k != actual]
        wrongs.append(("She made no mistake", "w5k-trusts-the-working"))
        check(shown != L, "the shown value is the correct value")
        work = [
            f"With x = {v}: {a}({v} + {p}) {q_txt} = {L}, not {shown}.",
            f"{shown} comes from this error: {key[0].lower() + key[1:]}.",
        ]
        stem = phrase(
            rng,
            (
                "Dana substitutes x = {v} into the expression {e} and gets {s}. Which mistake did she make?",
                "{n} works out {e} for x = {v} and gets {s}. What was {n}'s mistake?",
            ),
            v=v,
            e=eq_left,
            s=shown,
            n="Dana",
        )
        return _mc(rng, (stem,), key, wrongs, work)
    raise TooFewDistractors("no distinct mistake values")


def _check_exam(rng):
    a, c = rng.randint(3, 8), rng.randint(1, 2)
    s = rng.randint(2, 9)
    b = rng.choice([v for v in range(-9, 10) if v != 0])
    d = (a - c) * s + b
    works = rng.random() < 0.5  # whether the proposed value really is the solution
    v = s if works else s + rng.choice((1, 2, -1))
    if v <= 0:
        raise TooFewDistractors("bad proposed value")
    eq = f"{join_terms([(a, 'x'), (b, '')])} = {join_terms([(c, 'x'), (d, '')])}"
    L, R = _lhs(a, b, v), _rhs(c, d, v)
    check((L == R) == works, "the proposed value does not behave as intended")
    fa = number_fields(L)
    da = number_cands(
        L,
        [
            (int(f"{a}{v}") + b, "w5k-joins-digits"),
            (v + b, "w5k-ignores-coefficient"),
            (a * v - b, "w5k-sign-slip-on-constant"),
            (L + a, "w5k-off-by-one"),
        ],
    )
    fb = number_fields(R)
    db = number_cands(
        R,
        [
            (c + v + d, "w5k-ignores-coefficient"),
            (c * v - d, "w5k-sign-slip-on-constant"),
            (R + c, "w5k-off-by-one"),
            (L if not works else R + 1, "w5k-copies-the-left-side" if not works else "w5k-off-by-one"),
        ],
    )
    if works:
        key_c = f"Yes: both sides equal {L}"
        wc = [
            (f"No: the sides are not equal ({L} and {L + a})", "w5k-wrong-verdict"),
            (f"Yes: both sides equal {L + 1}", "w5k-off-by-one"),
            (f"No: x = {v} makes the left side larger than the right side", "w5k-wrong-verdict"),
            (f"No: the sides are not equal ({L + c} and {L})", "w5k-checks-one-side"),
        ]
        key_d = f"x = {v} is the only solution"
        wd = [
            ("Every number is a solution", "w5n-cancelling-means-every-number"),
            ("There is no solution", "w5n-cancelling-means-no-solution"),
            (f"x = {v} and x = {-v} are both solutions", "w5k-negative-sign-dropped"),
            (f"x = {v} and x = {v + 1} are both solutions", "w5k-checks-one-side"),
        ]
        prompt_d = f"{eq} is true for x = {v}. Which statement about its solutions is correct?"
        work_d = f"The x terms do not cancel ({a} and {c} are different), so there is exactly one solution, x = {v}."
    else:
        key_c = f"No: the sides are not equal ({L} and {R})"
        wc = [
            (f"Yes: both sides equal {L}", "w5k-copies-the-left-side"),
            (f"Yes: x = {v} gives a left side of {L}, so it works", "w5k-checks-one-side"),
            (f"No: x = {v} makes the right side larger than {R}", "w5k-wrong-verdict"),
            (f"Yes: both sides equal {R}", "w5k-checks-one-side"),
        ]
        key_d = f"x = {s}"
        wd = [
            (f"x = {v}", "w5k-wrong-verdict"),
            (f"x = {s + 2}" if s + 2 != v else f"x = {s - 1}", "w5k-off-by-one"),
            (f"x = {d}", "w5k-takes-the-constant"),
            (f"x = {L}", "w5k-takes-the-constant"),
        ]
        prompt_d = f"Which value of x does satisfy {eq}?"
        work_d = f"x = {s} gives {_lhs(a, b, s)} on both sides."
    fc = mcq_fields(key_c)
    dc = mcq_cands(key_c, wc, 4)
    fd = mcq_fields(key_d)
    dd = mcq_cands(key_d, [(t, g) for t, g in wd if t != key_d], 4)
    check(_lhs(a, b, s) == _rhs(c, d, s), "the true solution fails the check")
    enough(dc, 4)
    enough(dd, 4)  # both choice parts always show five options
    enough(da, 2)
    enough(db, 2)
    return dict(
        text=f"{rng.choice(NAMES)} wants to check whether x = {v} is a solution of {eq}.",
        parts=[
            part(
                f"What is the value of the left side when x = {v}?",
                fa["answer_kind"],
                _sub(fa),
                1,
                [f"{a} × {v} {'+' if b > 0 else '-'} {abs(b)} = {L}."],
                da,
                sensible_range=(None, None),
            ),
            part(
                f"What is the value of the right side when x = {v}?",
                fb["answer_kind"],
                _sub(fb),
                1,
                [f"{c} × {v} {'+' if d > 0 else '-'} {abs(d)} = {R}." if d else f"{c} × {v} = {R}."],
                db,
                sensible_range=(None, None),
            ),
            part(
                f"Is x = {v} a solution? Choose the correct conclusion.",
                "mcq",
                _sub(fc),
                1,
                [
                    f"{L} and {R} are {'the same' if works else 'different'}, so x = {v} {'is' if works else 'is not'} a solution."
                ],
                dc,
            ),
            part(
                prompt_d,
                "mcq",
                _sub(fd),
                2,
                [work_d],
                dd,
            ),
        ],
    )


@generator(
    "w5_check_substitution",
    level=17,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def w5_check_substitution(rng, tier="easy"):
    """Substitute to check; the tiers are described in the module docstring."""
    pools = {
        "easy": (_check_line, lambda r: _check_verdict(r, True)),
        "medium": (lambda r: _check_verdict(r, False),),
        "hard": (lambda r: _check_verdict(r, False, True), _check_find_mistake),
    }
    return _tiers(pools, rng, tier)


@generator("exd_x_w5_check_substitution", level=17, presets={"exam": dict(tier="exam")})
def exd_x_w5_check_substitution(rng, tier="exam"):
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    return retry_draw(rng, _check_exam)
