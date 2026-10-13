"""Generates `ucv1_equivalence_relations` questions and the exam chain `exd_x_fy_equivalence_relations`.

``ucv1-equivalence-relations-classes`` (``discrete-mathematics``, answer ``number``): reflexive, symmetric and transitive
relations on a finite set, and the equivalence classes (a partition) they make.

* easy: congruence modulo n on {1, ..., N}: the size of the class of a given element, or the size of the largest class.
* medium: a relation R written out as a set of ordered pairs on a small set; how many ordered pairs must be added to make
  it an equivalence relation. The smallest equivalence relation containing R needs the reflexive, symmetric and
  transitive pairs together, so counting only one kind of missing pair gives a wrong answer.
* hard: how many equivalence classes a rule makes: a² ≡ b² (mod n) on {0, ..., n - 1}, gcd(a, n) = gcd(b, n) on
  {1, ..., n}, or "a ≡ b mod m and a ≡ b mod k" on {1, ..., N}. Each needs the classes counted, not guessed from n.
* exam (``exd_x_fy_equivalence_relations``): a ≡ b (mod m) on {1, ..., N}, with N not a multiple of m. (a) the number of
  classes, (b) the size of the largest (following through from (a)), (c) the number of ordered pairs in R (from (a) and
  (b)), (d) the number of unordered pairs {a, b} of distinct related elements (from (c)).

Every answer is re-derived by building the relation as a set of ordered pairs and finding its classes directly.
"""

from __future__ import annotations

from math import gcd, lcm

from ..core import attempts, check, generator, pick_distractors
from ._fy1cd_kit import F, divisor_count, int_dist, say


class _Retry(Exception):
    """A draw whose numbers leave too few distinct wrong answers."""


def _need(ds) -> None:
    if len(ds) < 2:
        raise _Retry


def _retrying(build, rng):
    for _ in attempts(100, "a draw with enough distinct wrong answers"):
        try:
            return build(rng)
        except _Retry:
            continue


def _dists(answer, candidates, n: int = 4):
    return pick_distractors(str(answer), [int_dist(v, t) for v, t in candidates if v > 0], n)


def _set(elems) -> F:
    body = ", ".join(str(e) for e in elems)
    return F("{" + body + "}", r"\{" + body.replace(", ", ",") + r"\}")


def _range_set(lo: int, hi: int) -> F:
    return F(f"{{{lo}, {lo + 1}, ..., {hi}}}", rf"\{{{lo},{lo + 1},\ldots,{hi}\}}")


def _some(count: int, lead: str = "") -> str:
    """'3 classes have' or 'the first class has', so a count of 1 reads correctly."""
    if count == 1:
        return f"{lead}class has" if lead else "one class has"
    return f"{lead}{count} classes have"


def _classes(elements, related) -> list[list[int]]:
    """The classes of an equivalence relation, found by sweeping the elements (no assumption about the rule)."""
    classes: list[list[int]] = []
    for e in elements:
        for cls in classes:
            if related(cls[0], e):
                cls.append(e)
                break
        else:
            classes.append([e])
    return classes


def _check_equivalence(elements, related) -> None:
    """The rule is an equivalence relation: related(a, b) holds exactly when a and b are in the same class of the
    sweep, which is reflexive, symmetric and transitive by construction."""
    classes = _classes(elements, related)
    where = {e: i for i, cls in enumerate(classes) for e in cls}
    for a in elements:
        for b in elements:
            check(related(a, b) == (where[a] == where[b]), "the rule is not an equivalence relation")


# -- easy -------------------------------------------------------------------------------------------------------


def _easy(rng):
    n = rng.randint(3, 7)
    big = rng.randint(2 * n + 1, 4 * n + 3)
    if big % n == 0:
        big += 1
    elements = list(range(1, big + 1))
    related = lambda a, b: (a - b) % n == 0  # noqa: E731
    _check_equivalence(elements[:14], related)
    classes = _classes(elements, related)
    check(len(classes) == n, "the classes are not the residues")
    lead = rng.choice(["On the set ", "Within the set ", "Given the set "])
    if rng.random() < 0.5:
        m = rng.choice(elements[: n * 2])
        cls = next(c for c in classes if m in c)
        answer = len(cls)
        ask = f"How many elements are in the equivalence class of {m}?"
        stem = say(
            lead,
            F(f"S = {_range_set(1, big)[0]}", f"S={_range_set(1, big)[1]}"),
            F(
                f", a relation R is defined by a R b when a - b is divisible by {n}",
                rf"\text{{, a relation }}R\text{{ is defined by }}a\,R\,b\text{{ when }}a-b\text{{ is divisible by }}{n}",
            ),
            f". {ask}",
        )
        cands = [
            (big // n, "equiv-class-rounds-down"),
            (n, "equiv-class-count-for-size"),
            ((big - m) // n, "equiv-class-counts-one-direction"),
            (m // n + 1 if m > n else big // n + 1, "equiv-class-counts-one-direction"),
            (big // n + 1, "equiv-class-rounds-up-always"),
        ]
        working = [
            f"The elements of the class of {m} are those that leave remainder {m % n} when divided by {n}.",
            f"There are {answer} of them in S.",
        ]
    else:
        answer = max(len(c) for c in classes)
        stem = say(
            lead,
            F(f"S = {_range_set(1, big)[0]}", f"S={_range_set(1, big)[1]}"),
            F(
                f", a relation R is defined by a R b when a - b is divisible by {n}",
                rf"\text{{, a relation }}R\text{{ is defined by }}a\,R\,b\text{{ when }}a-b\text{{ is divisible by }}{n}",
            ),
            ". What is the size of the largest equivalence class?",
        )
        cands = [
            (big // n, "equiv-class-rounds-down"),
            (n, "equiv-class-count-for-size"),
            (big // n + 2, "equiv-class-rounds-up-always"),
            (big - n, "equiv-class-count-for-size"),
        ]
        working = [
            f"{big} = {n} × {big // n} + {big % n}, so {_some(big % n, 'the first ')} {big // n + 1} elements and the rest have {big // n}.",
            f"The largest class has {answer} elements.",
        ]
    ds = _dists(answer, cands)
    _need(ds)
    return dict(
        text=stem[0],
        latex=stem[1],
        answer_kind="number",
        answer=answer,
        answer_text=str(answer),
        answer_latex=str(answer),
        working=working,
        distractors=ds,
    )


# -- medium -----------------------------------------------------------------------------------------------------

_PARTITIONS = [[3, 2], [2, 2, 1], [3, 1, 1], [4, 1], [3, 3], [2, 2, 2], [4, 2], [3, 2, 1]]


def _pairs_formula(pairs) -> F:
    body = ", ".join(f"({a}, {b})" for a, b in pairs)
    return F("{" + body + "}", r"\{" + body.replace(", ", ",").replace("),(", "),(") + r"\}")


def _smallest_equivalence(elements, pairs):
    """The smallest equivalence relation containing `pairs`: classes are the connected pieces of the graph."""
    parent = {e: e for e in elements}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in pairs:
        parent[find(a)] = find(b)
    groups: dict[int, list[int]] = {}
    for e in elements:
        groups.setdefault(find(e), []).append(e)
    return {(a, b) for g in groups.values() for a in g for b in g}


def _medium(rng):
    for _ in attempts(500, "a relation whose three kinds of missing pair all matter"):
        sizes = rng.choice(_PARTITIONS)
        elements = list(range(1, sum(sizes) + 1))
        rng.shuffle(elements)
        blocks, start = [], 0
        for s in sizes:
            blocks.append(sorted(elements[start : start + s]))
            start += s
        elements = sorted(elements)
        full = {(a, b) for blk in blocks for a in blk for b in blk}
        pairs = set(full)
        for e in rng.sample(elements, rng.randint(1, 2)):
            pairs.discard((e, e))
        for a, b in sorted(full):
            if a != b and rng.random() < 0.3:
                pairs.discard((a, b))
        closure = _smallest_equivalence(elements, pairs)
        refl_missing = sum(1 for e in elements if (e, e) not in pairs)
        sym = pairs | {(b, a) for a, b in pairs}
        sym_missing = len(sym) - len(pairs)
        sym_refl = sym | {(e, e) for e in elements}
        forgot_trans = len(sym_refl) - len(pairs)
        answer = len(closure) - len(pairs)
        distinct = {answer, refl_missing, sym_missing, forgot_trans}
        if len(distinct) == 4 and 0 < refl_missing and sym_missing > 0 and len(pairs) <= 14:
            break
    # the closure really is reflexive, symmetric and transitive, and nothing smaller contains `pairs`
    check(pairs <= closure, "R is not inside its closure")
    check(all((e, e) in closure for e in elements), "closure not reflexive")
    check(all((b, a) in closure for a, b in closure), "closure not symmetric")
    check(
        all((a, d) in closure for a, b in closure for c, d in closure if b == c),
        "closure not transitive",
    )
    ordered = sorted(pairs)
    stem = say(
        "Let ",
        F(f"S = {_set(elements)[0]}", f"S={_set(elements)[1]}"),
        " and ",
        F(f"R = {_pairs_formula(ordered)[0]}", f"R={_pairs_formula(ordered)[1]}"),
        ". What is the smallest number of ordered pairs that must be added to R to make it an equivalence relation on S?",
    )
    cands = [
        (refl_missing, "equiv-adds-only-reflexive"),
        (sym_missing, "equiv-adds-only-symmetric"),
        (forgot_trans, "equiv-forgets-transitivity"),
        (len(closure), "equiv-counts-whole-relation"),
    ]
    ds = _dists(answer, cands)
    _need(ds)
    return dict(
        text=stem[0],
        latex=stem[1],
        answer_kind="number",
        answer=answer,
        answer_text=str(answer),
        answer_latex=str(answer),
        working=[
            f"R is missing {refl_missing} pair(s) (a, a), so it is not reflexive.",
            f"Adding the reflexive and symmetric pairs gives {forgot_trans} more, but the result is still not transitive when pairs link through a middle element.",
            f"The smallest equivalence relation containing R has the classes "
            + ", ".join("{" + ", ".join(map(str, sorted(c))) + "}" for c in _classes_of(closure, elements))
            + f", so it has {len(closure)} pairs, which is {answer} more than R.",
        ],
        distractors=ds,
    )


def _classes_of(closure, elements):
    return _classes(elements, lambda a, b: (a, b) in closure)


# -- hard -------------------------------------------------------------------------------------------------------

_SQUARE_MODULI = [8, 12, 15, 16, 18, 20, 21, 24, 28, 30, 32, 36]
_GCD_MODULI = [12, 18, 20, 24, 30, 36, 40, 42, 48, 60]
_PAIRS = [(4, 6), (6, 9), (4, 10), (6, 8), (6, 10), (9, 12), (10, 15)]


def _totient(n: int) -> int:
    return sum(1 for k in range(1, n + 1) if gcd(k, n) == 1)


def _hard(rng):
    variant = rng.choice(["square", "gcd", "two_moduli"])
    if variant == "square":
        n = rng.choice(_SQUARE_MODULI)
        elements = list(range(n))
        related = lambda a, b: (a * a - b * b) % n == 0  # noqa: E731
        _check_equivalence(elements, related)
        answer = len(_classes(elements, related))
        stem = say(
            "A relation ~ on ",
            F(f"S = {_range_set(0, n - 1)[0]}", f"S={_range_set(0, n - 1)[1]}"),
            " is defined by ",
            F(f"a ~ b when a² ≡ b² (mod {n})", rf"a\sim b\text{{ when }}a^{{2}}\equiv b^{{2}}\pmod{{{n}}}"),
            ". How many equivalence classes does ~ have?",
        )
        cands = [
            (n, "equiv-classes-as-elements"),
            (n // 2 + 1, "equiv-squares-assumes-pm-only"),
            (n // 2, "equiv-squares-assumes-pm-only"),
            (_totient(n), "equiv-gcd-counts-units"),
        ]
        working = [
            f"Two elements are related when their squares agree mod {n}, so the classes are the distinct values of a² mod {n}.",
            f"Squaring 0, 1, ..., {n - 1} mod {n} gives {answer} different values.",
        ]
    elif variant == "gcd":
        n = rng.choice(_GCD_MODULI)
        elements = list(range(1, n + 1))
        related = lambda a, b: gcd(a, n) == gcd(b, n)  # noqa: E731
        _check_equivalence(elements, related)
        answer = len(_classes(elements, related))
        check(answer == divisor_count(n), "the classes are not counted by the divisors")
        stem = say(
            "A relation ~ on ",
            F(f"S = {_range_set(1, n)[0]}", f"S={_range_set(1, n)[1]}"),
            " is defined by ",
            F(f"a ~ b when gcd(a, {n}) = gcd(b, {n})", rf"a\sim b\text{{ when }}\gcd(a,{n})=\gcd(b,{n})"),
            ". How many equivalence classes does ~ have?",
        )
        cands = [
            (n, "equiv-classes-as-elements"),
            (_totient(n), "equiv-gcd-counts-units"),
            (answer - 1, "equiv-gcd-misses-n"),
            (answer + 1, "equiv-gcd-misses-n"),
        ]
        working = [
            f"gcd(a, {n}) is always a divisor of {n}, and every divisor of {n} occurs, so each divisor gives one class.",
            f"{n} has {answer} divisors.",
        ]
    else:
        m, k = rng.choice(_PAIRS)
        big = rng.randint(3, 5) * lcm(m, k)
        elements = list(range(1, big + 1))
        related = lambda a, b: (a - b) % m == 0 and (a - b) % k == 0  # noqa: E731
        _check_equivalence(elements[: 2 * lcm(m, k) + 1], related)
        answer = len(_classes(elements, related))
        check(answer == lcm(m, k), "the classes are not the residues mod lcm")
        stem = say(
            "On ",
            F(f"S = {_range_set(1, big)[0]}", f"S={_range_set(1, big)[1]}"),
            F(
                f", a ~ b means that a ≡ b (mod {m}) and a ≡ b (mod {k})",
                rf"\text{{, }}a\sim b\text{{ means that }}a\equiv b\pmod{{{m}}}\text{{ and }}a\equiv b\pmod{{{k}}}",
            ),
            ". How many equivalence classes does ~ have?",
        )
        cands = [
            (m * k, "equiv-lcm-as-product"),
            (m + k, "equiv-lcm-as-sum"),
            (max(m, k), "equiv-lcm-as-larger"),
            (gcd(m, k), "equiv-lcm-as-gcd"),
        ]
        working = [
            f"a ≡ b mod {m} and mod {k} together mean that {lcm(m, k)} divides a - b, since lcm({m}, {k}) = {lcm(m, k)}.",
            f"So the classes are the {answer} residues mod {lcm(m, k)}.",
        ]
    ds = _dists(answer, cands)
    _need(ds)
    return dict(
        text=stem[0],
        latex=stem[1],
        answer_kind="number",
        answer=answer,
        answer_text=str(answer),
        answer_latex=str(answer),
        working=working,
        distractors=ds,
    )


@generator(
    "ucv1_equivalence_relations",
    level=18,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def ucv1_equivalence_relations(rng, tier="easy"):
    """Equivalence relations and their classes; the tiers are described in the module docstring."""
    if tier == "easy":
        return _retrying(_easy, rng)
    if tier == "medium":
        return _retrying(_medium, rng)
    check(tier == "hard", f"unknown tier {tier!r}")
    return _retrying(_hard, rng)


# -- exam -------------------------------------------------------------------------------------------------------


def _exam(rng):
    m = rng.randint(3, 7)
    big = rng.randint(2 * m + 1, 4 * m + m - 1)
    if big % m == 0:
        big += 1
    rem = big % m
    elements = list(range(1, big + 1))
    related = lambda a, b: (a - b) % m == 0  # noqa: E731
    _check_equivalence(elements[: 2 * m + 2], related)
    classes = _classes(elements, related)
    sizes = [len(c) for c in classes]
    pairs = sum(1 for a in elements for b in elements if related(a, b))
    unordered = sum(1 for a in elements for b in elements if a < b and related(a, b))
    classes_n, largest = len(classes), max(sizes)
    check(classes_n == m and largest == -(-big // m), "classes or largest class differ")
    check(pairs == rem * (big // m + 1) ** 2 + (m - rem) * (big // m) ** 2, "ordered pairs differ from the class sizes")
    check(unordered * 2 == pairs - big, "unordered pairs differ")

    a_dist = _dists(
        classes_n,
        [
            (big, "equiv-classes-as-elements"),
            (largest, "equiv-class-count-for-size"),
            (big // m, "equiv-class-rounds-down"),
        ],
    )
    b_dist = _dists(
        largest,
        [
            (big // m, "equiv-class-rounds-down"),
            (m, "equiv-class-count-for-size"),
            (big // m + 2, "equiv-class-rounds-up-always"),
            (big - m, "equiv-class-count-for-size"),
        ],
    )
    c_dist = _dists(
        pairs,
        [
            (m * largest * largest, "equiv-pairs-all-classes-largest"),
            (big, "equiv-pairs-reflexive-only"),
            (big * largest, "equiv-pairs-each-element-with-largest"),
            (big * big // m, "equiv-pairs-equal-classes"),
        ],
    )
    d_dist = _dists(
        unordered,
        [
            (pairs - big, "equiv-pairs-ordered-not-halved"),
            (pairs // 2, "equiv-pairs-reflexive-not-removed"),
            (pairs, "equiv-pairs-reflexive-not-removed"),
            ((pairs - big) // 2 + big // 2, "equiv-pairs-reflexive-not-removed"),
        ],
    )

    def own_b(av):
        return -(-big // av) if av else None

    def own_c(av, bv):
        return rem * bv * bv + (av - rem) * (bv - 1) ** 2

    def own_d(cv):
        return (cv - big) // 2 if (cv - big) % 2 == 0 else None

    b_dist = [d for d in b_dist if int(d.text) not in {own_b(int(x.text)) for x in a_dist}]
    a_values, b_values = [classes_n] + [int(x.text) for x in a_dist], [largest] + [int(x.text) for x in b_dist]
    c_dist = [d for d in c_dist if int(d.text) not in {own_c(av, bv) for av in a_values for bv in b_values} - {pairs}]
    c_values = [pairs] + [int(x.text) for x in c_dist]
    d_dist = [d for d in d_dist if int(d.text) not in {own_d(cv) for cv in c_values} - {unordered}]
    for ds in (a_dist, b_dist, c_dist, d_dist):
        _need(ds)

    stem = say(
        rng.choice(["A relation R on ", "The relation R on ", "A binary relation R on "]),
        F(f"S = {_range_set(1, big)[0]}", f"S={_range_set(1, big)[1]}"),
        " is defined by ",
        F(f"a R b when a ≡ b (mod {m})", rf"a\,R\,b\text{{ when }}a\equiv b\pmod{{{m}}}"),
        ". R is an equivalence relation.",
    )
    return dict(
        text=stem[0],
        latex=stem[1],
        parts=[
            dict(
                prompt="How many equivalence classes does R have?",
                answer_kind="number",
                answer=classes_n,
                answer_text=str(classes_n),
                answer_latex=str(classes_n),
                marks=2,
                sensible_range=(0, None),
                working=[
                    f"Each class is the set of elements with one remainder mod {m}, and there are {m} remainders."
                ],
                distractors=a_dist,
            ),
            dict(
                prompt="What is the size of the largest class?",
                answer_kind="number",
                answer=largest,
                answer_text=str(largest),
                answer_latex=str(largest),
                marks=2,
                sensible_range=(0, None),
                follow_through=f"ceiling({big}/part_a)",
                working=[f"{big} = {m} × {big // m} + {rem}, so the largest classes have {largest} elements."],
                distractors=b_dist,
            ),
            dict(
                prompt="How many ordered pairs (a, b) are in R?",
                answer_kind="number",
                answer=pairs,
                answer_text=str(pairs),
                answer_latex=str(pairs),
                marks=2,
                sensible_range=(0, None),
                follow_through=f"{rem}*part_b**2 + (part_a - {rem})*(part_b - 1)**2",
                working=[
                    f"{_some(rem)} {largest} elements and {_some(m - rem)} {largest - 1}. A class of size s gives s² ordered pairs.",
                    f"{rem} × {largest}² + {m - rem} × {largest - 1}² = {pairs}.",
                ],
                distractors=c_dist,
            ),
            dict(
                prompt="How many unordered pairs {a, b} of different elements are related by R?",
                prompt_latex=r"\text{How many unordered pairs }\{a,b\}\text{ of different elements are related by }R\text{?}",
                answer_kind="number",
                answer=unordered,
                answer_text=str(unordered),
                answer_latex=str(unordered),
                marks=2,
                sensible_range=(0, None),
                follow_through=f"(part_c - {big})/2",
                working=[
                    f"Remove the {big} pairs (a, a) from {pairs}, leaving {pairs - big} ordered pairs of different elements.",
                    f"Each unordered pair gives two ordered pairs, so there are {unordered}.",
                ],
                distractors=d_dist,
            ),
        ],
    )


@generator("exd_x_fy_equivalence_relations", level=18, presets={"exam": dict(tier="exam")})
def exd_x_fy_equivalence_relations(rng, tier="exam"):
    """Exam chain: classes, the largest class, ordered pairs and unordered pairs of a congruence relation."""
    check(tier == "exam", f"unknown tier {tier!r}")
    return _retrying(_exam, rng)
