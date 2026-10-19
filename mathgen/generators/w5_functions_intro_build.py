"""The function concept and an introduction to domain and range: topics `w5_function_concept` and
`w5_domain_range_intro` (easy, medium, hard) with exam chains `exd_x_w5_function_concept` and
`exd_x_w5_domain_range_intro`.

Skill `function-concept` (domain `functions-sequences`, answer `mcq`): decide whether a relation is a function.
* easy: sets of ordered pairs. A function may repeat an output but never an input, so the options are built to catch
  the learner who rejects a repeated output and accepts a repeated input.
* medium: equations in x and y (which defines y as a function of x, or which does not), with the vertical line test
  applied to the shape of the curve rather than to whether the equation contains a square.
* hard: an unknown pair in a relation (which value of k breaks the rule), and mappings between two named sets, where
  every input needs exactly one output.
* exam: the number of different inputs, the input with two outputs, the pairs to remove, and the outputs.

Skill `domain-and-range-intro` (answer `list`): list the domain or range as a set of numbers.
* easy: the domain or the range of a set of ordered pairs (a repeated output is listed once).
* medium: the outputs of a rule on a stated set of inputs, including a practical context that limits the inputs.
* hard: the integers a rule can accept (a square root, a reciprocal), and the range of a quadratic or a product over whole
  inputs, where outputs repeat.
* exam: the range of a quadratic on whole inputs, how many values it has, its greatest value, and the inputs that give
  a stated output.

Every answer is built from the chosen relation or rule and re-derived a second way: the domain and range are recomputed
by evaluating the rule at every input (or by scanning the pairs), and the function test is repeated on a dictionary.
"""

from __future__ import annotations

from ..core import attempts, check, generator, join_terms, phrase
from ._w5_kit import (
    TooFewDistractors,
    enough,
    list_cands,
    list_fields,
    mcq_cands,
    mcq_fields,
    number_cands,
    number_fields,
    part,
    retry_draw,
    with_set_latex,
    spread,
)

CONCEPT = "w5_function_concept"
RANGE = "w5_domain_range_intro"


def _sub(fields: dict) -> dict:
    return {key: val for key, val in fields.items() if key != "answer_kind"}


def _set_text(pairs) -> str:
    return "{" + ", ".join(f"({x}, {y})" for x, y in pairs) + "}"


def is_function(pairs) -> bool:
    seen: dict[int, int] = {}
    for x, y in pairs:
        if seen.setdefault(x, y) != y:
            return False
    return True


def _has_repeated_output(pairs) -> bool:
    ys = [y for _, y in set(pairs)]
    return len(set(ys)) < len(ys)


def _function_pairs(rng, n: int, repeated_output: bool):
    xs = rng.sample(range(-4, 10), n)
    ys = [rng.randint(-5, 12) for _ in range(n)]
    if repeated_output:
        i, j = rng.sample(range(n), 2)
        ys[j] = ys[i]
    pairs = list(zip(xs, ys))
    check(is_function(pairs), "built pairs are not a function")
    if _has_repeated_output(pairs) != repeated_output:
        raise TooFewDistractors("repeated-output status differs")
    return pairs


def _non_function_pairs(rng, n: int, repeated_output: bool):
    """n pairs with exactly one input that has two different outputs."""
    base = _function_pairs(rng, n - 1, repeated_output)
    x, y = rng.choice(base)
    other = rng.choice([v for v in range(-5, 13) if v != y])
    pairs = base + [(x, other)]
    if repeated_output != _has_repeated_output(pairs):
        raise TooFewDistractors("repeated-output status differs")
    check(not is_function(pairs), "built pairs are a function")
    rng.shuffle(pairs)
    return pairs


# -- function concept: easy (sets of ordered pairs) ---------------------------------------------------------------------


def _fc_easy_input(rng):
    """One relation in the stem; name the input that is paired with two different outputs."""
    pairs = _non_function_pairs(rng, 6, rng.random() < 0.6)
    xs = sorted({x for x, _ in pairs})
    bad = [x for x in xs if len({y for xx, y in pairs if xx == x}) > 1]
    check(len(bad) == 1, "not exactly one conflicting input")
    key = f"x = {bad[0]}"
    wrong = []
    for x in xs:
        if x == bad[0]:
            continue
        shared = [xx for xx, yy in pairs if yy in [y for xx2, y in pairs if xx2 == x] and xx != x]
        wrong.append(
            (
                f"x = {x}",
                "w5f-confuses-repeated-output-with-repeated-input" if shared else "w5f-any-collection-is-function",
            )
        )
    wrong.sort(key=lambda t: t[1] != "w5f-confuses-repeated-output-with-repeated-input")
    fields = mcq_fields(key)
    cands = mcq_cands(key, wrong)
    enough(cands, 4)
    stem = phrase(
        rng,
        (
            "The relation {r} is not a function. Which input is paired with two different outputs?",
            "In the relation {r}, one input has two different outputs. Which input is it?",
        ),
        r=_set_text(pairs),
    )
    work = [
        f"The input {bad[0]} is listed twice, with different outputs.",
        "Repeated outputs are allowed; a repeated input with different outputs is not.",
    ]
    return dict(text=stem, working=work, distractors=cands, **fields)


def _fc_easy(rng):
    if rng.random() < 0.45:
        return _fc_easy_input(rng)
    n = rng.choice((4, 5))
    asks_function = rng.random() < 0.55
    if asks_function:
        ro = rng.random() < 0.6
        right = _function_pairs(rng, n, ro)
        rng.shuffle(right)
        wrong = []
        for i in range(4):
            # a relation with a repeated input; make the outputs all different when the key repeats one, so a
            # learner who forbids repeated outputs picks it
            want_repeat = (not ro) or i >= 2
            bad = _non_function_pairs(rng, n, want_repeat and rng.random() < 0.5)
            tag = (
                "w5f-requires-distinct-outputs"
                if (ro and not _has_repeated_output(bad))
                else "w5f-any-collection-is-function"
            )
            wrong.append((_set_text(bad), tag))
        key = _set_text(right)
        ask = rng.choice(
            (
                "Which of these relations is a function?",
                "Exactly one of the relations below is a function. Which one is it?",
                "Which set of ordered pairs defines a function?",
            )
        )
        work = [
            "A relation is a function when no input (first number) is paired with two different outputs.",
            "Check each set for a first number that appears twice with different second numbers.",
        ]
    else:
        right = _non_function_pairs(rng, n, rng.random() < 0.5)
        wrong = []
        for _ in range(4):
            good = _function_pairs(rng, n, rng.random() < 0.8)
            tag = "w5f-requires-distinct-outputs" if _has_repeated_output(good) else "w5f-any-collection-is-function"
            wrong.append((_set_text(sorted(good) if rng.random() < 0.5 else rng.sample(good, len(good))), tag))
        key = _set_text(right)
        ask = rng.choice(
            (
                "Which of these relations is NOT a function?",
                "Four of the relations below are functions. Which one is NOT a function?",
                "Which set of ordered pairs does NOT define a function?",
            )
        )
        work = [
            "A repeated output is allowed; a repeated input with different outputs is not.",
            "The relation that pairs one input with two outputs is not a function.",
        ]
    fields = mcq_fields(key)
    cands = mcq_cands(key, wrong)
    enough(cands, 4)
    spread([key, *[c.text for c in cands]], key)
    stem = phrase(rng, ("{a}", "Look at the five relations below. {a}"), a=ask)
    return dict(text=stem, working=work, distractors=cands, **fields)


# -- function concept: medium (equations) -----------------------------------------------------------------------------


def _function_equations(rng):
    a, b = rng.randint(2, 5), rng.randint(2, 9)
    return [
        (f"y = {a}x^2 + {b}", "squared"),
        (f"y = {a}x - {b}", "linear"),
        (f"y = |x - {b}| + {a}", "absolute"),
        (f"y = {a}/(x - {b})", "reciprocal"),
        (f"y = x^3 + {a}", "cubic"),
        (f"y = sqrt(x + {b})", "root"),
        (f"y = {b} - x^2", "squared"),
        (f"y = {a}x^2 - {b}x", "squared"),
    ]


def _relation_equations(rng):
    a, r = rng.randint(1, 6), rng.choice((3, 4, 5, 6, 7, 10))
    return [
        (f"x = y^2 + {a}", "sideways"),
        (f"x^2 + y^2 = {r * r}", "circle"),
        (f"y^2 = x + {a}", "sideways"),
        (f"|y| = x + {a}", "sideways"),
        (f"x = |y| - {a}", "sideways"),
        (f"(y - {a})^2 = x", "sideways"),
        (f"x^2 - y^2 = {a * a}", "circle"),
        (f"x^2 + {a}y^2 = {r * r * a}", "circle"),
    ]


def _fc_medium(rng):
    functions, relations = _function_equations(rng), _relation_equations(rng)
    if rng.random() < 0.55:
        key = rng.choice(functions)
        # the key starts "y = ", so at least one wrong equation must carry a leading "x = " label too
        lead = rng.choice([r for r in relations if r[0].startswith("x =")])
        wrongs = [lead, *rng.sample([r for r in relations if r is not lead], 3)]
        ask = "Which of these equations defines y as a function of x?"
        tagged = [(e, "w5f-curve-without-vertical-line-test") for e, _ in wrongs]
        work = [
            "y is a function of x when every x gives only one y, so a vertical line meets the graph at most once.",
            "Equations that need both +y and -y for the same x (squares or absolute values of y) are not functions.",
        ]
    else:
        key = rng.choice(relations)
        wrongs = rng.sample(functions, 4)
        ask = "Which of these equations does NOT define y as a function of x?"
        tagged = []
        for e, kind in wrongs:
            tag = {
                "squared": "w5f-squared-term-means-not-function",
                "absolute": "w5f-absolute-value-means-not-function",
            }.get(kind, "w5f-any-equation-with-restriction-fails")
            tagged.append((e, tag))
        work = [
            "Solve for y: when there is a +/- choice of y for the same x, a vertical line meets the graph twice.",
            "An x squared (or |x|) gives one y for each x, so those equations are functions.",
        ]
    fields = mcq_fields(key[0])
    cands = mcq_cands(key[0], tagged)
    enough(cands, 4)
    return dict(text=ask, working=work, distractors=cands, **fields)


def _fc_medium_table(rng):
    """Two-row tables: in which is y a function of x (or is not)."""
    n = rng.choice((4, 5))
    want_function = rng.random() < 0.55

    def table(pairs):
        return "x: " + ", ".join(str(x) for x, _ in pairs) + "  y: " + ", ".join(str(y) for _, y in pairs)

    if want_function:
        right = _function_pairs(rng, n, rng.random() < 0.6)
        wrong = [
            (table(_non_function_pairs(rng, n, rng.random() < 0.3)), "w5f-any-collection-is-function") for _ in range(4)
        ]
        ask = "In which table is y a function of x?"
    else:
        right = _non_function_pairs(rng, n, rng.random() < 0.5)
        wrong = []
        for _ in range(4):
            good = _function_pairs(rng, n, True)
            wrong.append((table(good), "w5f-requires-distinct-outputs"))
        ask = "In which table is y NOT a function of x?"
    key = table(right)
    fields = mcq_fields(key)
    cands = mcq_cands(key, wrong)
    enough(cands, 4)
    work = [
        "Read the x row and the y row together as pairs.",
        "A function never gives one x two different y values; two x values may share a y.",
    ]
    return dict(text=ask, working=work, distractors=cands, **fields)


def _fc_medium_row(rng):
    """A table that is a function and four extra rows; which row would stop y being a function of x."""
    n = rng.choice((4, 5))
    base = _function_pairs(rng, n, rng.random() < 0.5)
    table = "x: " + ", ".join(str(x) for x, _ in base) + "  y: " + ", ".join(str(y) for _, y in base)
    used_x = {x for x, _ in base}
    used_y = sorted({y for _, y in base})
    free_x = [v for v in range(-5, 12) if v not in used_x]
    x0, y0 = rng.choice(base)
    other = rng.choice([v for v in range(-5, 13) if v != y0])

    def row(x, y):
        return f"x = {x}, y = {y}"

    key = row(x0, other)
    x1, y1 = rng.choice([p for p in base if p != (x0, y0)])
    new_x = rng.sample(free_x, 3)
    wrong = [
        (row(x0, y0), "w5f-repeated-identical-pair-counts"),
        (row(new_x[0], rng.choice(used_y)), "w5f-confuses-input-and-output"),
        (row(new_x[1], rng.choice([v for v in range(-5, 13) if v not in used_y])), "w5f-any-collection-is-function"),
        (row(x1, y1), "w5f-repeated-identical-pair-counts"),
        (row(new_x[2], rng.choice(used_y)), "w5f-confuses-input-and-output"),
    ]
    fields = mcq_fields(key)
    cands = mcq_cands(key, wrong)
    enough(cands, 4)
    stem = phrase(
        rng,
        (
            "The table {t} shows y as a function of x. Which extra row would stop y from being a function of x?",
            "A function is given by the table {t}. Which new row, added to the table, would make it NOT a function?",
        ),
        t=table,
    )
    work = [
        f"A row breaks the rule only when its x is already in the table with a different y: x = {x0} already has y = {y0}.",
        "A new x never breaks the rule, and a row that repeats an existing pair exactly adds no second output.",
    ]
    return dict(text=stem, working=work, distractors=cands, **fields)


# -- function concept: hard ---------------------------------------------------------------------------------------------


def _fc_hard_k(rng):
    n = rng.choice((4, 5))
    base = _function_pairs(rng, n, rng.random() < 0.5)
    xs = [x for x, _ in base]
    ys = {y for _, y in base}
    x_hit, y_hit = rng.choice(base)
    new_y = rng.choice([v for v in range(-5, 13) if v not in ys])
    # the unknown pair is (k, new_y): it breaks the rule exactly when k is an existing input with another output
    others = [x for x, y in base if x != x_hit]
    same_pair_k = None
    # an option k that repeats an input with the SAME output as the new pair would not break the rule: build one by
    # aiming the new pair's output at an existing pair
    x_same, y_same = rng.choice(base)
    pair_y = y_same
    # re-derive with pair_y as the unknown pair's output
    breaking = [x for x, y in base if y != pair_y]
    check(len(breaking) >= 1, "no input breaks the rule")
    k_key = rng.choice(breaking)
    nonbreaking_same = [x for x, y in base if y == pair_y]
    ws = []
    ws.append((f"k = {nonbreaking_same[0]}", "w5f-repeated-identical-pair-counts"))
    out_only = [y for y in sorted(ys) if y not in xs and y != k_key]
    if out_only:
        ws.append((f"k = {rng.choice(out_only)}", "w5f-confuses-input-and-output"))
    fresh = [v for v in range(-4, 14) if v not in xs and v not in ys]
    ws.append((f"k = {rng.choice(fresh)}", "w5f-any-new-value-breaks-rule"))
    ws.append((f"k = {rng.choice([v for v in fresh if v != int(ws[-1][0][4:])])}", "w5f-any-new-value-breaks-rule"))
    rel = _set_text(base + [("k", pair_y)])
    key = f"k = {k_key}"
    check(not is_function(base + [(k_key, pair_y)]), "the key does not break the rule")
    for text, _ in ws:
        kk = int(text[4:])
        if kk not in nonbreaking_same:
            check(is_function(base + [(kk, pair_y)]), "a wrong option breaks the rule")
    fields = mcq_fields(key)
    cands = mcq_cands(key, ws)
    enough(cands, 3)
    spread([key, *[c.text for c in cands]], key)
    stem = f"The relation {rel} is not a function. Which value of k could it be?"
    work = [
        "A relation fails to be a function when one input has two different outputs.",
        f"The pair (k, {pair_y}) must repeat an input that is already paired with something other than {pair_y}.",
    ]
    return dict(text=stem, working=work, distractors=cands, **fields)


_LETTERS = "abcdef"


def _mapping_text(arrows) -> str:
    return ", ".join(f"{x}→{y}" for x, y in arrows)


def _fc_hard_mapping(rng):
    size = rng.choice((4, 5))
    inputs = list(range(1, size + 1))
    outputs = list(_LETTERS[:3])
    rng.shuffle(outputs)
    right = [(x, rng.choice(outputs)) for x in inputs]
    key = _mapping_text(right)
    # one input unmapped
    miss = rng.choice(inputs)
    unmapped = [(x, y) for x, y in right if x != miss]
    # one input with two outputs
    dup = rng.choice(inputs)
    two = list(right)
    other = rng.choice([y for y in outputs if y != dict(right)[dup]])
    two.insert(dup, (dup, other))
    # an output outside the stated set
    out_of_set = [(x, y) if x != rng.choice(inputs) else (x, "d") for x, y in right]
    if all(y != "d" for _, y in out_of_set):
        out_of_set[0] = (out_of_set[0][0], "d")
    # a second unmapped / duplicate variant to reach four wrong options
    miss2 = rng.choice([x for x in inputs if x != miss])
    unmapped2 = [(x, y) for x, y in right if x != miss2]
    extra_input = [*right, (size + 1, rng.choice(outputs))]
    wrong = [
        (_mapping_text(unmapped), "w5f-unmapped-input-allowed"),
        (_mapping_text(two), "w5f-any-collection-is-function"),
        (_mapping_text(out_of_set), "w5f-output-outside-codomain"),
        (_mapping_text(unmapped2), "w5f-unmapped-input-allowed"),
        (_mapping_text(extra_input), "w5f-input-outside-domain"),
    ]
    fields = mcq_fields(key)
    cands = mcq_cands(key, wrong, 4)
    enough(cands, 4)
    spread([key, *[c.text for c in cands]], key)
    stem = (
        f"A rule must send each member of A = {{{', '.join(map(str, inputs))}}} to a member of B = "
        f"{{{', '.join(outputs)}}}. Which of these describes a function from A to B?"
    )
    work = [
        "A function from A to B sends every member of A to exactly one member of B.",
        "Each option is checked for a missing input, an input with two outputs, or an output outside B.",
    ]
    return dict(text=stem, working=work, distractors=cands, **fields)


@generator(
    CONCEPT,
    level=17,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def w5_function_concept(rng, tier="easy"):
    """Is this relation a function? The tiers are described in the module docstring."""
    pool = {
        "easy": (_fc_easy,),
        "medium": (_fc_medium, _fc_medium_table, _fc_medium_row, _fc_medium_row),
        "hard": (_fc_hard_k, _fc_hard_mapping),
    }.get(tier)
    if pool is None:
        raise ValueError(f"unknown tier {tier!r}")
    return with_set_latex(retry_draw(rng, rng.choice(pool)))


def _fc_exam(rng):
    n = rng.choice((6, 7))
    ndup = rng.choice((1, 1, 2))
    for _ in attempts(500, "a relation with one or two conflicting inputs"):
        xs = rng.sample(range(1, 9), n - ndup)
        ys = [rng.randint(1, 9) for _ in range(n - ndup)]
        pairs = list(zip(xs, ys))
        dups = rng.sample(xs, ndup)
        for xd in dups:
            y0 = dict(zip(xs, ys))[xd]
            pairs.append((xd, rng.choice([v for v in range(1, 10) if v != y0])))
        if len(set(pairs)) == len(pairs):
            break
    x_dup = dups[0]
    rng.shuffle(pairs)
    xs_all = sorted({x for x, _ in pairs})
    distinct_x = len(xs_all)
    check(distinct_x == n - ndup, "distinct input count differs")
    to_remove = n - distinct_x
    ys_all = sorted({y for _, y in pairs})
    a = number_fields(distinct_x)
    a_d = number_cands(
        distinct_x,
        [
            (n, "w5f-counts-pairs-not-inputs"),
            (len(ys_all), "w5f-swaps-domain-and-range"),
            (n - 2, "w5f-miscounts-inputs"),
            (n + 1, "w5f-miscounts-inputs"),
        ],
    )
    key_b = f"x = {x_dup}"
    dup_outs = sorted(y for x, y in pairs if x == x_dup)
    holders = [x for x in xs_all if set(dup_outs) <= {yy for xx, yy in pairs if xx == x}]
    if len(dup_outs) != 2 or holders != [x_dup]:
        raise TooFewDistractors("the pair of outputs does not identify one input")
    repeated_outputs = [y for y in ys_all if sum(1 for _, yy in pairs if yy == y) > 1]
    wrong_b = []
    for y in repeated_outputs:
        xx = [x for x, yy in pairs if yy == y][0]
        if xx not in dups:
            wrong_b.append((f"x = {xx}", "w5f-confuses-repeated-output-with-repeated-input"))
    for x in xs_all:
        if x not in dups:
            wrong_b.append((f"x = {x}", "w5f-any-collection-is-function"))
    b = mcq_fields(key_b)
    b_d = mcq_cands(key_b, wrong_b, 4)
    c = number_fields(to_remove)
    c_d = number_cands(
        to_remove,
        [
            (n, "w5f-counts-pairs-not-inputs"),
            (distinct_x, "w5f-counts-inputs-kept-not-removed"),
            (to_remove + 1, "w5f-miscounts-inputs"),
            (len(repeated_outputs) or 2, "w5f-confuses-repeated-output-with-repeated-input"),
        ],
    )
    d = list_fields(ys_all)
    d_d = list_cands(
        d,
        [
            (xs_all, "w5f-swaps-domain-and-range"),
            ([y for _, y in pairs], "w5f-lists-repeated-outputs"),
            (ys_all[1:], "w5f-omits-a-value"),
        ],
    )
    for dd in (a_d, b_d, c_d, d_d):
        enough(dd)
    enough(b_d, 3)
    return dict(
        text=phrase(
            rng,
            (
                "A relation R is {r}.",
                "The relation R consists of these ordered pairs: {r}.",
                "Consider the set of ordered pairs R = {r}.",
            ),
            r=_set_text(pairs),
        ),
        parts=[
            part(
                "How many different inputs (first numbers) does R use?",
                a["answer_kind"],
                _sub(a),
                1,
                [f"The inputs are {', '.join(map(str, xs_all))}: {distinct_x} different values."],
                a_d,
            ),
            part(
                f"R is not a function. Which of these inputs is paired with both {dup_outs[0]} and {dup_outs[1]}?",
                "mcq",
                _sub(b),
                1,
                [f"Input {x_dup} is paired with both {dup_outs[0]} and {dup_outs[1]}."],
                b_d,
            ),
            part(
                "What is the smallest number of pairs that must be removed to turn R into a function?",
                c["answer_kind"],
                _sub(c),
                2,
                [f"Each input may keep one pair: {n} pairs - {distinct_x} inputs = {to_remove}."],
                c_d,
            ),
            part(
                "List the different outputs (second numbers) that R uses. Each value is listed once.",
                "list",
                _sub(d),
                1,
                ["Collect the second numbers and list each value once."],
                d_d,
            ),
        ],
    )


@generator("exd_x_w5_function_concept", level=17, presets={"exam": dict(tier="exam")})
def exd_x_w5_function_concept(rng, tier="exam"):
    """Exam chain: different inputs, the input with two outputs, pairs to remove and the outputs used."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    return with_set_latex(retry_draw(rng, _fc_exam))


# -- domain and range --------------------------------------------------------------------------------------------------


def _dr_easy(rng):
    n = rng.choice((4, 5, 6))
    for _ in attempts(100, "a function with a repeated output"):
        pairs = _function_pairs(rng, n, rng.random() < 0.7)
        break
    rng.shuffle(pairs)
    asks_range = rng.random() < 0.6
    xs, ys = sorted({x for x, _ in pairs}), sorted({y for _, y in pairs})
    if asks_range:
        fields = list_fields(ys)
        cands = list_cands(
            fields,
            [
                (xs, "w5f-swaps-domain-and-range"),
                ([y for _, y in pairs], "w5f-lists-repeated-outputs") if len(ys) < n else (None, ""),
                (ys[1:], "w5f-omits-a-value"),
                (ys + [max(ys) + 1], "w5f-adds-an-extra-value"),
            ],
        )
        ask = "List the range of this function. List each value once."
        work = [
            "The range is the set of outputs (second numbers).",
            f"The outputs are {', '.join(str(y) for _, y in pairs)}; a repeated output is listed once.",
        ]
    else:
        fields = list_fields(xs)
        cands = list_cands(
            fields,
            [
                (ys, "w5f-swaps-domain-and-range"),
                (xs[:-1], "w5f-omits-a-value"),
                (xs + [max(xs) + 1], "w5f-adds-an-extra-value"),
                ([x - 1 for x in xs], "w5f-shifts-the-values"),
            ],
        )
        ask = "List the domain of this function."
        work = ["The domain is the set of inputs (first numbers)."]
    enough(cands)
    return dict(text=f"A function is {_set_text(pairs)}. {ask}", working=work, distractors=cands, **fields)


_RULES = (
    ("f(x) = {a}x + {b}", lambda a, b: lambda x: a * x + b, "linear"),
    ("f(x) = {a}x - {b}", lambda a, b: lambda x: a * x - b, "linear"),
    ("f(x) = x^2 + {b}", lambda a, b: lambda x: x * x + b, "square"),
    ("f(x) = {b} - x^2", lambda a, b: lambda x: b - x * x, "square"),
    ("f(x) = {a}x^2 - {b}", lambda a, b: lambda x: a * x * x - b, "square"),
)

_CONTEXTS = (
    (
        "A taxi fare is C = {b} + {a}d dollars for a trip of d kilometres. Trips of {ds} km are offered.",
        lambda a, b: lambda d: b + a * d,
        "C",
    ),
    (
        "A shop sells n packs at ${a} each with a ${b} delivery charge, so the cost is C = {a}n + {b} dollars. Orders of {ds} packs are allowed.",
        lambda a, b: lambda n: a * n + b,
        "C",
    ),
    (
        "A tank holds {top} litres and loses {a} litres every minute, so V = {top} - {a}t litres after t minutes. Readings are taken at t = {ds} minutes.",
        None,
        "V",
    ),
)


def _dr_medium(rng):
    if rng.random() < 0.45:
        # a rule on a stated set of inputs
        template, build, kind = rng.choice(_RULES)
        a, b = rng.randint(2, 5), rng.randint(1, 9)
        f = build(a, b)
        lo = rng.randint(-3, 0)
        xs = (
            sorted(rng.sample(range(lo, lo + 6), 5)) if rng.random() < 0.5 else list(range(lo, lo + rng.choice((5, 6))))
        )
        outs = [f(x) for x in xs]
        ys = sorted(set(outs))
        if kind == "square" and len(ys) == len(outs):
            raise TooFewDistractors("a quadratic with no repeated output")
        fields = list_fields(ys)
        slip = build(a, -b) if kind == "linear" else (lambda x: -(x * x) + b if "- x^2" not in template else x * x - b)
        cands = list_cands(
            fields,
            [
                (outs, "w5f-lists-repeated-outputs") if len(ys) < len(outs) else (None, ""),
                (xs, "w5f-swaps-domain-and-range"),
                (sorted({slip(x) for x in xs}), "w5f-rule-applied-with-sign-slip"),
                (outs[:-1] if len(set(outs[:-1])) == len(set(outs)) else sorted(set(outs[:-1])), "w5f-omits-a-value"),
            ],
        )
        domain = "{" + ", ".join(map(str, xs)) + "}"
        stem = f"A function is f(x) = {template.split('=')[1].strip().format(a=a, b=b)}, with domain {domain}. List the range of f."
        work = [
            f"Substitute each domain value: the outputs are {', '.join(map(str, outs))}.",
            "List each output once, in order.",
        ]
    else:
        template, build, var = rng.choice(_CONTEXTS)
        a, b = rng.randint(2, 6), rng.randint(2, 9)
        top = a * rng.randint(8, 14)
        if build is None:
            first = rng.randint(0, 3)
            ds = list(range(first, first + 5))
            outs = [top - a * t for t in ds]
            stem = (
                template.format(a=a, b=b, ds=", ".join(map(str, ds)), top=top) + " List the possible volumes in litres."
            )
            slip_outs = [top + a * t for t in ds]
        else:
            ds = sorted(rng.sample(range(1, 9), 5))
            f = build(a, b)
            outs = [f(d) for d in ds]
            slip_outs = [a * d - b if var == "C" else a * d for d in ds]
            what = "fares" if "taxi" in template else "costs"
            stem = (
                template.format(a=a, b=b, ds=", ".join(map(str, ds)), top=top)
                + f" List the possible {what} in dollars."
            )
        ys = sorted(set(outs))
        fields = list_fields(ys)
        cands = list_cands(
            fields,
            [
                (ds, "w5f-swaps-domain-and-range"),
                (sorted(set(slip_outs)), "w5f-rule-applied-with-sign-slip"),
                (
                    list(range(min(ys), max(ys) + 1)) if max(ys) - min(ys) <= 30 else None,
                    "w5f-ignores-the-restriction-from-context",
                ),
                (ys[:-1], "w5f-omits-a-value"),
            ],
        )
        work = [
            "Only the stated inputs can occur, so find the output for each of them.",
            f"The outputs are {', '.join(map(str, outs))}.",
        ]
    enough(cands)
    return dict(text=stem, working=work, distractors=cands, **fields)


def _dr_hard_sqrt(rng):
    a = rng.randint(-3, 5)
    sgn = rng.choice((1, -1))  # sqrt(x - a) needs x >= a; sqrt(a - x) needs x <= a
    lo, hi = rng.randint(-5, 0), rng.randint(4, 8)
    ok = lambda x: sgn * (x - a) >= 0  # noqa: E731
    ints = [x for x in range(lo, hi + 1) if ok(x)]
    check(ints == [x for x in range(lo, hi + 1) if sgn * x >= sgn * a], "domain differs")
    if len(ints) < 2 or len(ints) == hi - lo + 1:
        raise TooFewDistractors("the restriction removes none or all of the whole numbers")
    fields = list_fields(ints)
    below = [x for x in range(lo, hi + 1) if not ok(x)]
    cands = list_cands(
        fields,
        [
            (list(range(lo, hi + 1)), "w5f-ignores-the-restriction"),
            ([x for x in range(lo, hi + 1) if sgn * (x - a) > 0], "w5f-excludes-the-boundary"),
            (below, "w5f-restriction-reversed"),
            ([x for x in range(lo, hi + 1) if sgn * (x + a) >= 0], "w5f-restriction-sign-slip"),
        ],
    )
    enough(cands)
    inner = join_terms([(sgn, "x"), (-sgn * a, "")])
    stem = f"The function f(x) = sqrt({inner}) is only defined when {inner} is zero or positive. List the whole numbers from {lo} to {hi} that are in its domain."
    work = [
        f"Solve {inner} ≥ 0: x {'≥' if sgn == 1 else '≤'} {a}.",
        f"The whole numbers from {lo} to {hi} with x {'≥' if sgn == 1 else '≤'} {a} are {', '.join(map(str, ints))}.",
    ]
    return dict(text=stem, working=work, distractors=cands, **fields)


def _dr_hard_excluded(rng):
    r1, r2 = sorted(rng.sample(range(-5, 6), 2))
    lo, hi = -rng.randint(5, 7), rng.randint(5, 7)
    out = [r1, r2]
    fields = list_fields(out)
    cands = list_cands(
        fields,
        [
            ([-r1, -r2], "w5f-restriction-sign-slip"),
            ([r1], "w5f-omits-a-value"),
            (list(range(lo, hi + 1)), "w5f-ignores-the-restriction"),
            ([x for x in range(lo, hi + 1) if x not in out], "w5f-restriction-reversed"),
        ],
    )
    enough(cands)
    den = f"({join_terms([(1, 'x'), (-r1, '')])})({join_terms([(1, 'x'), (-r2, '')])})"
    stem = f"The function f(x) = 1/({den}) cannot take an input that makes the denominator zero. List the whole numbers from {lo} to {hi} that are NOT in its domain."
    work = [
        f"The denominator is zero when x = {r1} or x = {r2}.",
        "Those two values are excluded; every other input is allowed.",
    ]
    return dict(text=stem, working=work, distractors=cands, **fields)


def _dr_hard_range(rng):
    kind = rng.choice(("quad", "area", "quad"))
    if kind == "quad":
        h = rng.randint(0, 3)
        c = rng.randint(-6, 6)
        sign = rng.choice((1, -1))
        lo, hi = h - rng.randint(2, 3), h + rng.randint(2, 4)
        xs = list(range(lo, hi + 1))
        f = lambda x: sign * (x - h) ** 2 + c  # noqa: E731
        outs = [f(x) for x in xs]
        ys = sorted(set(outs))
        body = join_terms([(sign, f"(x - {h})^2" if h else "x^2"), (c, "")])
        stem = f"A function is f(x) = {body}, with domain the whole numbers from {lo} to {hi} inclusive. List the range of f."
        slip = [(-((x - h) ** 2) + c if sign == 1 else (x - h) ** 2 + c) for x in xs]
        cands_src = [
            (outs, "w5f-lists-repeated-outputs"),
            (sorted(set(slip)), "w5f-negative-squared-as-negative"),
            (xs, "w5f-swaps-domain-and-range"),
            (sorted(set(f(x) for x in xs if x >= h)) if h > lo else None, "w5f-uses-half-the-domain"),
        ]
        work = [
            f"Substitute each whole number from {lo} to {hi}: {', '.join(map(str, outs))}.",
            "Inputs on either side of the turning point give the same output, so each value is listed once.",
        ]
    else:
        total = rng.randrange(8, 41, 2)
        ws = list(range(1, total))
        outs = [w * (total - w) for w in ws]
        ys = sorted(set(outs))
        stem = phrase(
            rng,
            (
                "A rectangle has perimeter {p} cm and whole-number side lengths. Its width is w cm, so its area is "
                "A = w({t} - w) square centimetres. List the possible areas.",
                "A farmer fences a rectangular pen with exactly {p} m of fencing, using whole-number side lengths. "
                "If the width is w m, the area is A = w({t} - w) square metres. List the possible areas.",
                "A rectangular poster has a border of length {p} cm all the way round and whole-number sides. "
                "With width w cm, its area is A = w({t} - w) square centimetres. List the possible areas.",
            ),
            p=2 * total,
            t=total,
        )
        cands_src = [
            (outs, "w5f-lists-repeated-outputs"),
            (ws, "w5f-swaps-domain-and-range"),
            (
                [w * (total + w) for w in ws if w * (total + w) not in ys][: total - 1]
                and sorted(set(w * (total + w) for w in ws)),
                "w5f-rule-applied-with-sign-slip",
            ),
            (ys[:-1], "w5f-omits-a-value"),
        ]
        work = [
            f"Width and length are whole numbers adding to {total}, so w can be 1, 2, ..., {total - 1}.",
            "Pairs of widths such as w and the matching length give the same area, so each area is listed once.",
        ]
    fields = list_fields(ys)
    cands = list_cands(fields, cands_src)
    enough(cands)
    return dict(text=stem, working=work, distractors=cands, **fields)


@generator(
    RANGE,
    level=17,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def w5_domain_range_intro(rng, tier="easy"):
    """List a domain or range; the tiers are described in the module docstring."""
    pool = {
        "easy": (_dr_easy,),
        "medium": (_dr_medium,),
        "hard": (_dr_hard_sqrt, _dr_hard_excluded, _dr_hard_range, _dr_hard_range),
    }.get(tier)
    if pool is None:
        raise ValueError(f"unknown tier {tier!r}")
    return with_set_latex(retry_draw(rng, rng.choice(pool)))


def _dr_exam(rng):
    h = rng.randint(-2, 4)
    q = rng.choice((1, 1, 2, 3))  # the coefficient of x^2
    lo, hi = h - rng.randint(2, 3), h + rng.randint(2, 4)
    c = rng.randint(-6, 6)
    b = -2 * q * h
    k = q * h * h + c
    xs = list(range(lo, hi + 1))
    f = lambda x: q * x * x + b * x + k  # noqa: E731
    outs = [f(x) for x in xs]
    ys = sorted(set(outs))
    body = join_terms([(q, "x^2"), (b, "x"), (k, "")])
    a = list_fields(ys)
    a_d = list_cands(
        a,
        [
            (outs, "w5f-lists-repeated-outputs"),
            (xs, "w5f-swaps-domain-and-range"),
            (sorted({-q * x * x + b * x + k for x in xs}), "w5f-negative-squared-as-negative"),
            (ys[:-1], "w5f-omits-a-value"),
        ],
    )
    b_f = number_fields(len(ys))
    b_d = number_cands(
        len(ys),
        [
            (len(xs), "w5f-counts-inputs-not-outputs"),
            (len(ys) + 1, "w5f-miscounts-inputs"),
            (len(ys) - 1, "w5f-miscounts-inputs"),
            (max(ys) - min(ys), "w5f-gives-the-spread"),
        ],
    )
    greatest = max(ys)
    c_f = number_fields(greatest)
    c_d = number_cands(
        greatest,
        [
            (f(hi) if f(hi) != greatest else f(lo), "w5f-takes-the-last-output"),
            (hi if hi != greatest else lo, "w5f-gives-the-input"),
            (min(ys), "w5f-gives-the-least"),
            (len(ys), "w5f-gives-the-count"),
        ],
    )
    target = ys[len(ys) // 2] if len(ys) > 1 else ys[0]
    inputs = [x for x in xs if f(x) == target]
    d = list_fields(inputs)
    d_d = list_cands(
        d,
        [
            ([target], "w5f-gives-the-output"),
            (inputs[:1], "w5f-one-input-only") if len(inputs) > 1 else (None, ""),
            ([x for x in xs if f(x) != target], "w5f-restriction-reversed"),
            ([-x for x in inputs], "w5f-sign-slip-on-inputs"),
        ],
    )
    for dd in (a_d, b_d, c_d, d_d):
        enough(dd)
    return dict(
        text=f"A function is f(x) = {body}, with domain the whole numbers from {lo} to {hi} inclusive.",
        parts=[
            part(
                "List the range of f. List each value once.",
                "list",
                _sub(a),
                2,
                [f"The outputs are {', '.join(map(str, outs))}."],
                a_d,
            ),
            part(
                "How many different values are in the range?",
                b_f["answer_kind"],
                _sub(b_f),
                1,
                [f"The range has {len(ys)} values."],
                b_d,
                sensible_range=(0, None),
            ),
            part(
                "What is the greatest value of f on this domain?",
                c_f["answer_kind"],
                _sub(c_f),
                1,
                [f"The greatest output is {greatest}."],
                c_d,
            ),
            part(
                f"List the inputs x in the domain for which f(x) = {target}.",
                "list",
                _sub(d),
                2,
                [f"Solve {body} = {target} and keep the whole numbers in the domain."],
                d_d,
            ),
        ],
    )


@generator("exd_x_w5_domain_range_intro", level=17, presets={"exam": dict(tier="exam")})
def exd_x_w5_domain_range_intro(rng, tier="exam"):
    """Exam chain: the range of a quadratic on whole inputs, its size, its greatest value and the inputs for one output."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    return with_set_latex(retry_draw(rng, _dr_exam))
