"""Generates `p19i_ratio_fraction_share` and `p19i_ratio_simplest_form` questions: a ratio part as a fraction of
the whole, and a ratio in simplest form.

Two generators, both ``answer: fraction`` (dict ``{"p": ..., "q": ...}``,
i.e. ``Fraction(p, q)``), and both deliberately kept visibly distinct from
each other and from their two shipping neighbours (``ratio-rate-proportion``,
``divide-quantity-in-ratio``):

- ``p19i_ratio_fraction_share`` (senior-understand-the-relationship-between-
  fractions-and-ratio): given a STATED two-term ratio a:b (or two raw counts
  standing in for one), express the fraction of the whole one named term
  represents -- ``a/(a+b)`` or ``b/(a+b)``. ``answer_text`` is a bare
  fraction ``"p/q"``, NEVER colon notation.
- ``p19i_ratio_simplest_form`` (senior-demonstrate-an-understanding-of-the-
  elementary-ideas-and): given two named whole-number QUANTITIES stated in a
  one-line real-world context, express their ratio in simplest whole-number
  a:b form. ``answer_text`` is colon notation ``"p:q"``, the simplified
  ratio itself, NEVER a bare fraction.

**Boundary against each other** (also stated in each generator's own
docstring below): ``p19i_ratio_fraction_share``'s deliverable is a bare
fraction of the whole, never a divided quantity and never a simplified a:b
ratio. ``p19i_ratio_simplest_form``'s deliverable is the simplified a:b
ratio itself, never a fraction of the whole.

**Boundary against the neighbours**: ``ratio-rate-proportion`` (topic
``ratio``, ``kind="simplify"``) is handed a ratio ALREADY written a:b and
simplifies it to its own simplest a:b form, or computes a unit rate -- it
never converts a ratio into a fraction-of-the-whole, and (unlike
``p19i_ratio_simplest_form``) it is never handed two named real-world
quantities to read as a ratio's terms first: its prompt states the ratio
outright. ``divide-quantity-in-ratio`` (topic ``p13a_divide_ratio``) divides
a SUPPLIED quantity into two computed shares (an ``answer: pair``) -- it
never returns a bare fraction and nothing here is ever divided into shares.

**Grader hazard (both generators)**: ``mathgen/grading/typed.py::grade_fraction``
compares by exact ``Fraction`` value, not by displayed form -- a "does not
simplify" distractor is invalid (it is accepted, not rejected). Every
distractor below is checked with ``check_distinct``/``check_pairwise_distinct``
to differ from the answer and from every other candidate in VALUE, never by
display alone. ``p19i_ratio_simplest_form`` additionally never relies on
``mathgen/grading/typed.py::_ratio_form_is_acceptable`` (the a:b-vs-a:b unsimplified-
form rejection) as a distractor strategy -- every one of its distractors
differs in value, exactly like ``p19i_ratio_fraction_share``'s.

Construct-to-target throughout, never solve-then-round: every stated quantity is built FROM drawn small
integers, and every tier's independent check re-derives the answer via a genuinely different algebraic route
(``check_equivalent``) before it is offered.
"""

from __future__ import annotations

import math
from fractions import Fraction

from ..core import (
    check,
    check_distinct,
    check_pairwise_distinct,
    check_equivalent,
    equals_chain,
    phrase,
    pick_distractors,
    generator,
)

_RETRY_LIMIT = 100

# --- shared real-world contexts -----------------------------------------
# (whole_label, first_label, second_label) -- kept short and countable, no
# absurd magnitudes; every count this module draws is a plausible real
# count for the label it is attached to.
_CONTEXTS_A = (
    ("the class", "boys", "girls", ("textbooks", "pencil cases", "laptops")),
    ("the choir", "sopranos", "altos", ("tenors", "basses")),
    ("the car park", "sedans", "hatchbacks", ("vans", "utes", "motorbikes")),
    ("the garden bed", "rose bushes", "lavender bushes", ("tulips", "daffodils", "marigolds")),
    ("the fruit bowl", "apples", "oranges", ("pears", "bananas", "mandarins")),
)


# =========================================================================
# Skill A: p19i_ratio_fraction_share
# =========================================================================


def _draw_ab(rng, lo, hi):
    """Two distinct, coprime small integers in [lo, hi] -- a stated ratio
    a:b that is not already reducible, matching ratio-rate-proportion's own
    coprime-draw convention."""
    for _ in range(_RETRY_LIMIT):
        a, b = rng.randint(lo, hi), rng.randint(lo, hi)
        if a != b and math.gcd(a, b) == 1:
            return a, b
    raise RuntimeError("could not draw a distinct coprime (a, b)")


def _share_options(rng, numerator, denom, named):
    """Pick three plausible WRONG shares for a key ``numerator/denom``. Every distractor is a proper fraction (0 < v < 1), since a
    share of a whole never exceeds 1, and is distinct from the key and from
    the others by value.

    ``named`` lists the tier's misconception candidates in priority order
    (Fractions); generic near-miss fillers (a part counted one too many or
    too few, the part added to the whole) follow so both sides of the key
    can be populated. A count of distractors BELOW the key (0-3) is drawn
    uniformly, so the key's rank among the four options is spread and it is
    not systematically the smallest or the largest. Returns the three
    Fractions, or None when this draw cannot fill the drawn rank (the
    caller redraws)."""
    key = Fraction(numerator, denom)
    fillers = (
        Fraction(numerator + 1, denom),  # counted one part too many
        Fraction(numerator, denom - 1),  # left one part out of the whole
        Fraction(numerator - 1, denom),  # counted one part too few
        Fraction(numerator, denom + numerator),  # part added to the whole
        Fraction(numerator, denom + 1),  # one extra part in the whole
        Fraction(numerator + 2, denom),
        Fraction(numerator, denom - 2),
    )
    seen = {key}
    above, below = [], []
    for c in list(named) + list(fillers):
        if not (0 < c < 1) or c in seen:
            continue
        seen.add(c)
        (above if c > key else below).append(c)
    n_below = rng.randint(0, 3)
    n_above = 3 - n_below
    if len(below) < n_below or len(above) < n_above:
        return None
    picked = below[:n_below] + above[:n_above]
    check_distinct(key, *picked, msg="a share distractor equals the answer")
    check_pairwise_distinct(*picked, msg="two share distractors coincide")
    check(all(0 < v < 1 for v in picked), "a share distractor is not a proper fraction")
    return picked


def _build_fraction_share_easy(rng):
    for _ in range(_RETRY_LIMIT):
        a, b = _draw_ab(rng, 2, 12)
        whole, x_label, y_label, _ = rng.choice(_CONTEXTS_A)
        denom = a + b
        numerator, other = a, b  # easy: always asks for the FIRST term
        answer = Fraction(numerator, denom)
        check_equivalent(answer, Fraction(a, a + b), "easy: fraction-of-whole disagrees with construction")
        other_fraction = Fraction(other, denom)
        # Independent route: the two named terms' shares must sum to the
        # whole -- a genuinely different identity from the direct division.
        check_equivalent(answer + other_fraction, 1, "easy: the two shares do not sum to the whole")

        # other term's share (wrong named term); a:b read directly as a
        # fraction when that is a proper fraction.
        cands = _share_options(rng, numerator, denom, [Fraction(other, denom), Fraction(a, b)])
        if cands is None:
            continue
        break
    else:
        raise RuntimeError("could not draw a distinct-distractor easy fraction-share question")

    patterns = (
        "In {whole}, the ratio of {x} to {y} is {a}:{b}. What fraction of {whole} are {x}?",
        "{Whole} has a ratio of {x} to {y} of {a}:{b}. What fraction of {whole} are {x}?",
        "The ratio of {x} to {y} in {whole} is {a}:{b}. Find the fraction of {whole} that are {x}.",
    )
    text = phrase(rng, patterns, whole=whole, Whole=whole[0].upper() + whole[1:], x=x_label, y=y_label, a=a, b=b)
    ans_text = f"{answer.numerator}/{answer.denominator}"
    working = [
        f"Total parts: {a} + {b} = {denom}.",
        f"Fraction that is {x_label}: {equals_chain(f'{a}/{denom}', ans_text)}.",
    ]
    candidates = [f"{c.numerator}/{c.denominator}" for c in cands]
    return dict(
        text=text,
        answer={"p": answer.numerator, "q": answer.denominator},
        answer_text=ans_text,
        working=working,
        distractors=pick_distractors(ans_text, candidates),
    )


def _build_fraction_share_medium(rng):
    for _ in range(_RETRY_LIMIT):
        a, b = _draw_ab(rng, 2, 20)
        whole, x_label, y_label, _ = rng.choice(_CONTEXTS_A)
        denom = a + b
        numerator, other = b, a  # medium: always asks for the SECOND term
        answer = Fraction(numerator, denom)
        check_equivalent(answer, Fraction(b, a + b), "medium: fraction-of-whole disagrees with construction")
        other_fraction = Fraction(other, denom)
        check_equivalent(answer + other_fraction, 1, "medium: the two shares do not sum to the whole")

        # other term's share (wrong named term); a:b read directly as a
        # fraction when that is a proper fraction.
        cands = _share_options(rng, numerator, denom, [Fraction(other, denom), Fraction(a, b)])
        if cands is None:
            continue
        break
    else:
        raise RuntimeError("could not draw a distinct-distractor medium fraction-share question")

    patterns = (
        "In {whole}, the ratio of {x} to {y} is {a}:{b}. What fraction of {whole} are {y}?",
        "{Whole} has a ratio of {x} to {y} of {a}:{b}. What fraction of {whole} are {y}?",
        "The ratio of {x} to {y} in {whole} is {a}:{b}. Find the fraction of {whole} that are {y}.",
    )
    text = phrase(rng, patterns, whole=whole, Whole=whole[0].upper() + whole[1:], x=x_label, y=y_label, a=a, b=b)
    ans_text = f"{answer.numerator}/{answer.denominator}"
    working = [
        f"Total parts: {a} + {b} = {denom}.",
        f"Fraction that is {y_label}: {equals_chain(f'{b}/{denom}', ans_text)}.",
    ]
    candidates = [f"{c.numerator}/{c.denominator}" for c in cands]
    return dict(
        text=text,
        answer={"p": answer.numerator, "q": answer.denominator},
        answer_text=ans_text,
        working=working,
        distractors=pick_distractors(ans_text, candidates),
    )


def _draw_counts(rng, lo, hi):
    """Two distinct plausible real-world counts -- not required coprime;
    these are literal counted things, not a ratio already stated in
    lowest terms."""
    for _ in range(_RETRY_LIMIT):
        c1, c2 = rng.randint(lo, hi), rng.randint(lo, hi)
        if c1 != c2:
            return c1, c2
    raise RuntimeError("could not draw two distinct counts")


_TOTAL_NOUN = {
    # whole_label -> what the stated total counts (both named groups are
    # members of it).
    "the class": "students",
    "the choir": "singers",
    "the car park": "cars",
    "the garden bed": "bushes",
    "the fruit bowl": "pieces of fruit",
}


def _build_fraction_share_hard(rng):
    """Hard: a stated TOTAL and ONE group's count; the asked group is the
    unstated remainder, so the learner must first subtract (other = total -
    given) before forming other/total. Easy/medium divide a stated ratio;
    exam adds irrelevant numbers -- this tier's escalation is the derived
    missing count."""
    for _ in range(_RETRY_LIMIT):
        given, other = _draw_counts(rng, 4, 40)
        whole, x_label, y_label, _ = rng.choice(_CONTEXTS_A)
        given_first = rng.choice([True, False])
        given_label, asked_label = (x_label, y_label) if given_first else (y_label, x_label)
        total = given + other
        answer = Fraction(other, total)
        # Independent route: the two groups' shares sum to the whole.
        check_equivalent(answer + Fraction(given, total), 1, "hard: the two groups' shares do not sum to the whole")
        check(total - given == other, "hard: derived missing count is inconsistent")

        # the stated group's share (not the asked group's); the stated count
        # over the asked count when that is a proper fraction.
        cands = _share_options(rng, other, total, [Fraction(given, total), Fraction(given, other)])
        if cands is None:
            continue
        break
    else:
        raise RuntimeError("could not draw a distinct-distractor hard fraction-share question")

    noun = _TOTAL_NOUN[whole]
    patterns = (
        "{Whole} has {total} {noun} in total. {given} of them are {gl} and the rest are {al}. "
        "What fraction of {whole} are {al}?",
        "There are {total} {noun} in {whole} altogether. {given} of them are {gl} and the rest "
        "are {al}. Find the fraction that are {al}.",
    )
    text = phrase(
        rng,
        patterns,
        whole=whole,
        Whole=whole[0].upper() + whole[1:],
        noun=noun,
        total=total,
        given=given,
        gl=given_label,
        al=asked_label,
    )
    ans_text = f"{answer.numerator}/{answer.denominator}"
    working = [
        f"Number that are {asked_label}: {total} - {given} = {other}.",
        f"Fraction that are {asked_label}: {equals_chain(f'{other}/{total}', ans_text)}.",
    ]
    candidates = [f"{c.numerator}/{c.denominator}" for c in cands]
    return dict(
        text=text,
        answer={"p": answer.numerator, "q": answer.denominator},
        answer_text=ans_text,
        working=working,
        distractors=pick_distractors(ans_text, candidates),
    )


# Exam contexts: (whole_label, what the whole counts, groups that
# are all genuine members of the whole). Every group belongs to the scene.
_EXAM_CONTEXTS = (
    ("the choir", ("sopranos", "altos", "tenors", "basses")),
    ("the car park", ("sedans", "hatchbacks", "vans", "utes")),
    ("the garden bed", ("rose bushes", "lavender bushes", "hydrangea bushes", "camellia bushes")),
    ("the fruit bowl", ("apples", "oranges", "pears", "bananas")),
)


def _build_fraction_share_exam(rng):
    """Exam: the total is NEVER stated. The whole is made of three groups
    whose sizes the learner must combine before forming the fraction --
    either three stated counts, or one count + a two-group ratio (so a second
    count is derived) + a third count. One step beyond hard, where the total
    is handed over."""
    for _ in range(_RETRY_LIMIT):
        whole, pool = rng.choice(_EXAM_CONTEXTS)
        x, y, z = rng.sample(pool, 3)
        variant = rng.choice(["counts", "ratio"])
        if variant == "counts":
            cx, cy, cz = (rng.randint(4, 40) for _ in range(3))
            numbers = (cx, cy, cz)
        else:
            a, b = _draw_ab(rng, 2, 6)
            k = rng.randint(2, 6)
            cy, cx = b * k, a * k  # x:y = a:b, y stated, x derived
            cz = rng.randint(4, 30)
            numbers = (cy, a, b, cz)
        counts = {x: cx, y: cy, z: cz}
        if len(set(counts.values())) != 3:
            continue
        total = cx + cy + cz
        asked = rng.choice([x, y, z])
        numerator = counts[asked]
        answer = Fraction(numerator, total)
        printed = set(numbers)
        if total in printed or answer.denominator in printed:
            continue  # the key denominator must not be printed
        # Independent route: the three groups' shares sum to the whole.
        check_equivalent(
            sum(Fraction(v, total) for v in counts.values()),
            1,
            "exam: the three groups' shares do not sum to the whole",
        )
        others = [g for g in (x, y, z) if g != asked]
        left_out = others[-1]
        cands = _share_options(
            rng,
            numerator,
            total,
            [
                Fraction(numerator, total - counts[left_out]),  # forgot one group in the whole
                Fraction(numerator, total - counts[others[0]]),  # forgot the other group
                Fraction(counts[others[0]], total),  # another group's share
                Fraction(counts[left_out], total),  # a third group's share
                Fraction(numerator, total + numerator),  # part over whole plus part
                Fraction(numerator + counts[others[0]], total),  # added another group to the part
            ],
        )
        if cands is None:
            continue
        break
    else:
        raise RuntimeError("could not draw a distinct-distractor exam fraction-share question")

    Whole = whole[0].upper() + whole[1:]
    if variant == "counts":
        patterns = (
            "{Whole} has {cx} {x}, {cy} {y} and {cz} {z}. What fraction of {whole} are {asked}?",
            "In {whole} there are {cx} {x}, {cy} {y} and {cz} {z}. What fraction are {asked}?",
        )
        text = phrase(rng, patterns, whole=whole, Whole=Whole, x=x, y=y, z=z, cx=cx, cy=cy, cz=cz, asked=asked)
        working = [f"Whole: {cx} + {cy} + {cz} = {total}."]
    else:
        patterns = (
            "{Whole} has {cy} {y}. The ratio of {x} to {y} is {a}:{b}, and there are {cz} {z}. "
            "What fraction of {whole} are {asked}?",
            "{Whole} holds {cy} {y}. The ratio of {x} to {y} is {a}:{b}, and there are "
            "{cz} {z}. Find the fraction of {whole} that are {asked}.",
        )
        text = phrase(rng, patterns, whole=whole, Whole=Whole, x=x, y=y, z=z, cy=cy, cz=cz, a=a, b=b, asked=asked)
        working = [
            f"Each ratio part is {cy} / {b} = {k}, so there are {a} × {k} = {cx} {x}.",
            f"Whole: {cx} + {cy} + {cz} = {total}.",
        ]
    ans_text = f"{answer.numerator}/{answer.denominator}"
    working.append(f"Fraction that are {asked}: {equals_chain(f'{numerator}/{total}', ans_text)}.")
    candidates = [f"{c.numerator}/{c.denominator}" for c in cands]
    return dict(
        text=text,
        answer={"p": answer.numerator, "q": answer.denominator},
        answer_text=ans_text,
        working=working,
        distractors=pick_distractors(ans_text, candidates),
    )


_FRACTION_SHARE_BUILDERS = {
    "easy": _build_fraction_share_easy,
    "medium": _build_fraction_share_medium,
    "hard": _build_fraction_share_hard,
    "exam": _build_fraction_share_exam,
}


@generator(
    "p19i_ratio_fraction_share",
    level=9,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
        "exam": dict(tier="exam"),
    },
)
def p19i_ratio_fraction_share(rng, tier="easy"):
    """Given a stated two-term ratio a:b (or, at hard, a total and one group's
    count, with the other group derived by subtraction), express the fraction of the whole that one named
    term represents. Boundary: this generator's deliverable is always a
    bare fraction of the whole (never colon notation, never a divided
    quantity) -- see the module docstring."""
    builder = _FRACTION_SHARE_BUILDERS.get(tier)
    if builder is None:
        raise ValueError(f"unknown tier {tier!r}; expected one of {sorted(_FRACTION_SHARE_BUILDERS)}")
    return builder(rng)


# =========================================================================
# Skill B: p19i_ratio_simplest_form
# =========================================================================

_CONTEXTS_B = (
    ("A fruit bowl", "apples", "oranges", ("pears", "bananas", "mandarins")),
    ("A choir", "sopranos", "altos", ("tenors", "basses")),
    ("A car park", "sedans", "hatchbacks", ("vans", "utes", "motorbikes")),
    ("A garden bed", "rose bushes", "lavender bushes", ("tulips", "daffodils", "marigolds")),
    ("A recipe box", "muffin trays", "cake tins", ("loaf tins", "pie dishes", "biscuit trays")),
)


def _draw_pq_coprime(rng, lo, hi):
    for _ in range(_RETRY_LIMIT):
        p, q = rng.randint(lo, hi), rng.randint(lo, hi)
        if p != q and math.gcd(p, q) == 1:
            return p, q
    raise RuntimeError("could not draw a distinct coprime (p, q)")


def _simplest_form_answer_text(p, q):
    return f"{p}:{q}"


def _build_simplest_form_easy(rng):
    for _ in range(_RETRY_LIMIT):
        p, q = _draw_pq_coprime(rng, 1, 9)
        g = rng.randint(2, 9)
        x, y = g * p, g * q
        if not (4 <= x <= 60 and 4 <= y <= 60):
            continue
        context, x_label, y_label, _ = rng.choice(_CONTEXTS_B)

        g2 = math.gcd(x, y)
        check_equivalent(
            Fraction(x // g2, y // g2), Fraction(p, q), "easy: gcd-derived ratio disagrees with construction"
        )

        swapped_terms = (q, p)  # swap the two terms
        reduced_second_only = (x, q)  # reduces only the second term
        reduced_first_only = (p, y)  # reduces only the first term
        try:
            check_distinct(
                Fraction(p, q),
                Fraction(*swapped_terms),
                Fraction(*reduced_second_only),
                Fraction(*reduced_first_only),
                msg="an easy-tier distractor equals the answer",
            )
            check_pairwise_distinct(
                Fraction(*swapped_terms),
                Fraction(*reduced_second_only),
                Fraction(*reduced_first_only),
                msg="two easy-tier distractors coincide",
            )
        except Exception:
            continue
        break
    else:
        raise RuntimeError("could not draw a distinct-distractor easy simplest-form question")

    patterns = (
        "{Context} has {x} {xl} and {y} {yl}. Express the ratio of {xl} to {yl} in simplest whole-number form.",
        "In {context_lower}, there are {x} {xl} and {y} {yl}. Write the ratio of {xl} to {yl} "
        "in simplest whole-number form.",
        "{Context} contains {x} {xl} and {y} {yl}. Simplify the ratio of {xl} to {yl} to whole numbers.",
    )
    text = phrase(
        rng, patterns, Context=context, context_lower=context[0].lower() + context[1:], x=x, y=y, xl=x_label, yl=y_label
    )
    ans_text = _simplest_form_answer_text(p, q)
    working = [
        f"Ratio of {x_label} to {y_label}: {x}:{y}.",
        f"gcd({x}, {y}) = {g2}. Divide both terms: {equals_chain(f'{x // g2}:{y // g2}', ans_text)}.",
    ]
    candidates = [f"{a}:{b}" for a, b in (swapped_terms, reduced_second_only, reduced_first_only)]
    return dict(
        text=text,
        answer={"p": p, "q": q},
        answer_text=ans_text,
        working=working,
        distractors=pick_distractors(ans_text, candidates),
    )


def _build_simplest_form_medium(rng):
    for _ in range(_RETRY_LIMIT):
        p, q = _draw_pq_coprime(rng, 1, 9)
        g = rng.randint(2, 9)
        x, y = g * p, g * q
        if not (4 <= x <= 60 and 4 <= y <= 60):
            continue
        context, x_label, y_label, third_pool = rng.choice(_CONTEXTS_B)
        z_label = rng.choice(third_pool)
        z = rng.randint(4, 60)

        g2 = math.gcd(x, y)
        check_equivalent(
            Fraction(x // g2, y // g2), Fraction(p, q), "medium: gcd-derived ratio disagrees with construction"
        )

        swapped_terms = (q, p)  # swap the two terms
        first_with_irrelevant_count = (x, z)  # wrong pair: first relevant count with the irrelevant one
        irrelevant_count_with_second = (z, y)  # wrong pair: irrelevant count with the second relevant one
        try:
            check_distinct(
                Fraction(p, q),
                Fraction(*swapped_terms),
                Fraction(*first_with_irrelevant_count),
                Fraction(*irrelevant_count_with_second),
                msg="a medium-tier distractor equals the answer",
            )
            check_pairwise_distinct(
                Fraction(*swapped_terms),
                Fraction(*first_with_irrelevant_count),
                Fraction(*irrelevant_count_with_second),
                msg="two medium-tier distractors coincide",
            )
        except Exception:
            continue
        break
    else:
        raise RuntimeError("could not draw a distinct-distractor medium simplest-form question")

    patterns = (
        "{Context} has {x} {xl}, {y} {yl} and {z} {zl}. Express the ratio of {xl} to {yl} in "
        "simplest whole-number form.",
        "In {context_lower}, there are {x} {xl}, {y} {yl} and {z} {zl}. Write the ratio of {xl} "
        "to {yl} in simplest whole-number form.",
        "{Context} contains {x} {xl}, {y} {yl} and {z} {zl}. Simplify the ratio of {xl} to {yl} to whole numbers.",
    )
    text = phrase(
        rng,
        patterns,
        Context=context,
        context_lower=context[0].lower() + context[1:],
        x=x,
        y=y,
        z=z,
        xl=x_label,
        yl=y_label,
        zl=z_label,
    )
    ans_text = _simplest_form_answer_text(p, q)
    working = [
        f"The {z_label} count ({z}) is not part of the {x_label}/{y_label} ratio.",
        f"Ratio of {x_label} to {y_label}: {x}:{y}.",
        f"gcd({x}, {y}) = {g2}. Divide both terms: {equals_chain(f'{x // g2}:{y // g2}', ans_text)}.",
    ]
    candidates = [f"{a}:{b}" for a, b in (swapped_terms, first_with_irrelevant_count, irrelevant_count_with_second)]
    return dict(
        text=text,
        answer={"p": p, "q": q},
        answer_text=ans_text,
        working=working,
        distractors=pick_distractors(ans_text, candidates),
    )


def _build_simplest_form_hard(rng):
    # Construct-to-target: draw the reduced ratio p:q FIRST (wide range, so
    # no single fraction denominator dominates the 200-draw concentration
    # sweep), then derive the stated fraction's own denominator as p + q --
    # gcd(p, p+q) == gcd(p, q) == 1, so the stated fraction is automatically
    # already in lowest terms and girls:boys is already p:q, no further
    # reduction ever changes the value.
    for _ in range(_RETRY_LIMIT):
        p, q = _draw_pq_coprime(rng, 1, 9)
        b_frac = p + q
        a_frac = p
        # Bound the class-size total to a realistic
        # classroom range (15-35) -- every multiplier m in this pool keeps
        # total = b_frac * m exact by construction (m is still an integer),
        # so the fraction-of-total division is never approximate.
        m_pool = [cand for cand in range(2, 11) if 15 <= b_frac * cand <= 35]
        if not m_pool:
            continue
        m = rng.choice(m_pool)
        total = b_frac * m
        girls = m * p
        boys = m * q

        g2 = math.gcd(girls, boys)
        check_equivalent(Fraction(girls, boys), Fraction(p, q), "hard: gcd-derived ratio disagrees with construction")
        check(girls * b_frac == total * a_frac, "hard: girls count disagrees with the stated fraction")

        swapped_terms = (q, p)  # swap the two terms
        fraction_denominator_as_term = (girls, b_frac)  # uses the fraction's denominator as one term directly
        total_as_second_term = (girls, total)  # forgot to subtract; used the total instead of boys
        try:
            check_distinct(
                Fraction(p, q),
                Fraction(*swapped_terms),
                Fraction(*fraction_denominator_as_term),
                Fraction(*total_as_second_term),
                msg="a hard-tier distractor equals the answer",
            )
            check_pairwise_distinct(
                Fraction(*swapped_terms),
                Fraction(*fraction_denominator_as_term),
                Fraction(*total_as_second_term),
                msg="two hard-tier distractors coincide",
            )
        except Exception:
            continue
        break
    else:
        raise RuntimeError("could not draw a distinct-distractor hard simplest-form question")

    patterns = (
        "A class of {total} students has {a_frac}/{b_frac} girls and the rest boys. Express "
        "the ratio of girls to boys in simplest whole-number form.",
        "Of {total} students in a class, {a_frac}/{b_frac} are girls and the rest are boys. "
        "Write the ratio of girls to boys in simplest whole-number form.",
        "A class has {total} students. {a_frac}/{b_frac} of them are girls, the rest boys. "
        "Simplify the ratio of girls to boys to whole numbers.",
    )
    text = phrase(rng, patterns, total=total, a_frac=a_frac, b_frac=b_frac)
    ans_text = _simplest_form_answer_text(p, q)
    working = [
        f"Girls: {a_frac}/{b_frac} x {total} = {girls}. Boys: {total} - {girls} = {boys}.",
        f"gcd({girls}, {boys}) = {g2}. Divide both terms: {equals_chain(f'{girls // g2}:{boys // g2}', ans_text)}.",
    ]
    candidates = [f"{a}:{b}" for a, b in (swapped_terms, fraction_denominator_as_term, total_as_second_term)]
    return dict(
        text=text,
        answer={"p": p, "q": q},
        answer_text=ans_text,
        working=working,
        distractors=pick_distractors(ans_text, candidates),
    )


def _build_simplest_form_exam(rng):
    for _ in range(_RETRY_LIMIT):
        p, q = _draw_pq_coprime(rng, 1, 9)
        t = rng.randint(5, 80)
        whole1, whole2 = t * p, t * q

        d1 = whole1 / 10
        d2 = whole2 / 10

        check_equivalent(
            Fraction(whole1, whole2), Fraction(p, q), "exam: cleared-decimal ratio disagrees with construction"
        )

        wrong_side = rng.choice(["first", "second"])
        if wrong_side == "first":
            decimal_cleared_by_100 = (whole1 * 10, whole2)  # cleared the first decimal by 100, not 10
        else:
            decimal_cleared_by_100 = (whole1, whole2 * 10)  # cleared the second decimal by 100, not 10
        swapped_terms = (q, p)  # swap the two terms
        reduced_second_only = (whole1, q)  # reduces only the second term
        try:
            check_distinct(
                Fraction(p, q),
                Fraction(*swapped_terms),
                Fraction(*decimal_cleared_by_100),
                Fraction(*reduced_second_only),
                msg="an exam-tier distractor equals the answer",
            )
            check_pairwise_distinct(
                Fraction(*swapped_terms),
                Fraction(*decimal_cleared_by_100),
                Fraction(*reduced_second_only),
                msg="two exam-tier distractors coincide",
            )
        except Exception:
            continue
        break
    else:
        raise RuntimeError("could not draw a distinct-distractor exam simplest-form question")

    d1_disp = f"{d1:.1f}"
    d2_disp = f"{d2:.1f}"
    patterns = (
        "A recipe uses {d1} kg of flour and {d2} kg of sugar. Express the ratio of flour to "
        "sugar in simplest whole-number form.",
        "A workshop uses {d1} litres of paint A and {d2} litres of paint B. Write the ratio of "
        "paint A to paint B in simplest whole-number form.",
        "A journey uses {d1} litres of fuel on the outbound leg and {d2} litres on the return "
        "leg. Simplify the ratio of outbound to return fuel to whole numbers.",
    )
    text = phrase(rng, patterns, d1=d1_disp, d2=d2_disp)
    ans_text = _simplest_form_answer_text(p, q)
    working = [
        f"Clear the decimals (multiply both by 10): {d1_disp}:{d2_disp} = {whole1}:{whole2}.",
        f"gcd({whole1}, {whole2}) = {t}. Divide both terms: {whole1 // t}:{whole2 // t} = {ans_text}.",
    ]
    candidates = [f"{a}:{b}" for a, b in (swapped_terms, decimal_cleared_by_100, reduced_second_only)]
    return dict(
        text=text,
        answer={"p": p, "q": q},
        answer_text=ans_text,
        working=working,
        distractors=pick_distractors(ans_text, candidates),
    )


_SIMPLEST_FORM_BUILDERS = {
    "easy": _build_simplest_form_easy,
    "medium": _build_simplest_form_medium,
    "hard": _build_simplest_form_hard,
    "exam": _build_simplest_form_exam,
}


@generator(
    "p19i_ratio_simplest_form",
    level=9,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
        "exam": dict(tier="exam"),
    },
)
def p19i_ratio_simplest_form(rng, tier="easy"):
    """Given two named whole-number quantities stated in a one-line
    real-world context (never a bare "Simplify the ratio a:b"), express
    their ratio in simplest whole-number a:b form. Boundary: this
    generator's deliverable is always the simplified a:b ratio itself
    (colon notation), never a bare fraction of the whole -- see the module
    docstring."""
    builder = _SIMPLEST_FORM_BUILDERS.get(tier)
    if builder is None:
        raise ValueError(f"unknown tier {tier!r}; expected one of {sorted(_SIMPLEST_FORM_BUILDERS)}")
    return builder(rng)
