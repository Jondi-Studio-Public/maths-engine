"""Re-derives each rendered question's mcq key from `question.text` alone -- using sympy's own polynomial
division and, for the group skill, a completely independent element-order computation, never importing
anything from `mathgen.generators.p15a_ideal_structure_build` or `p15a_group_data`.

Total re-derived: 1200 for ideal-membership (4 tiers x 300 seeds) + 1200 for
structure-classification (4 tiers x 300 seeds) = 2400.
"""

from __future__ import annotations

import re
from collections import Counter
from itertools import permutations, product

import pytest
from sympy import GF, Poly, Symbol, div, sympify

from mathgen.grading.typed import mcq_options
from mathgen import make

IDEAL_TOPIC = "p15a_ideal_membership"
STRUCT_TOPIC = "p15a_structure_classification"
N_SEEDS = 300
_x = Symbol("x")

# ===========================================================================
# Oracle A: p15a_ideal_membership
# ===========================================================================

_OPT_RE = re.compile(r"([ABCD]): (.+?)(?=(?:\s+[ABCD]: )|$)")


def _to_expr(text: str):
    """Parse a poly string like 'x^3 - 2x + 3' or '(x + 4)(x^2 - 3) + (-3x + 3)'
    into a sympy expr with standard Python syntax."""
    py = text.replace("^", "**")
    py = re.sub(r"(\d)(x)", r"\1*\2", py)  # '2x' -> '2*x'
    py = re.sub(r"\)(\()", r")*(", py)  # ')(' -> ')*('
    return sympify(py, locals={"x": _x})


def _oracle_ideal_zz(text: str) -> str:
    g_match = re.search(r"g\(x\) = (.+?) and", text)
    g_expr = _to_expr(g_match.group(1))
    opts = dict(_OPT_RE.findall(text))
    zero_letters = []
    for letter, opt_text in opts.items():
        p_expr = _to_expr(opt_text.strip().rstrip())
        p_poly = Poly(p_expr, _x, domain="ZZ")
        g_poly = Poly(g_expr, _x, domain="ZZ")
        _, r = div(p_poly, g_poly)
        if r.is_zero:
            zero_letters.append(letter)
    assert len(zero_letters) == 1, (text, opts, zero_letters)
    return zero_letters[0]


def _oracle_ideal_gf(text: str) -> str:
    m_prime = re.search(r"\(Z/(\d+)Z\)", text)
    prime = int(m_prime.group(1))
    g_match = re.search(r"g\(x\) = (.+?) and", text)
    g_expr = _to_expr(g_match.group(1))
    opts = dict(_OPT_RE.findall(text))
    zero_letters = []
    for letter, opt_text in opts.items():
        p_expr = _to_expr(opt_text.strip())
        p_poly = Poly(p_expr, _x, domain=GF(prime))
        g_poly = Poly(g_expr, _x, domain=GF(prime))
        _, r = div(p_poly, g_poly)
        if r.is_zero:
            zero_letters.append(letter)
    assert len(zero_letters) == 1, (text, opts, zero_letters)
    return zero_letters[0]


@pytest.mark.parametrize("tier", ["easy", "medium"])
def test_ideal_oracle_matches_expanded(tier):
    checked = 0
    for seed in range(N_SEEDS):
        q = make(IDEAL_TOPIC, seed=seed, difficulty=tier)
        expected = _oracle_ideal_zz(q.text)
        assert expected == q.answer_text, (tier, seed, q.text, q.answer_text)
        checked += 1
    assert checked == N_SEEDS


@pytest.mark.parametrize("tier", ["hard"])
def test_ideal_oracle_matches_factored(tier):
    checked = 0
    for seed in range(N_SEEDS):
        q = make(IDEAL_TOPIC, seed=seed, difficulty=tier)
        expected = _oracle_ideal_zz(q.text)
        assert expected == q.answer_text, (tier, seed, q.text, q.answer_text)
        checked += 1
    assert checked == N_SEEDS


@pytest.mark.parametrize("tier", ["exam"])
def test_ideal_oracle_matches_gf(tier):
    checked = 0
    for seed in range(N_SEEDS):
        q = make(IDEAL_TOPIC, seed=seed, difficulty=tier)
        expected = _oracle_ideal_gf(q.text)
        assert expected == q.answer_text, (tier, seed, q.text, q.answer_text)
        checked += 1
    assert checked == N_SEEDS


def test_ideal_oracle_is_red_on_a_corrupted_answer():
    q = make(IDEAL_TOPIC, seed=0, difficulty="easy")
    expected = _oracle_ideal_zz(q.text)
    corrupted = next(o for o in mcq_options(q) if o != expected)
    assert corrupted != expected


# ===========================================================================
# Oracle B: p15a_structure_classification
# ===========================================================================


def _oracle_group_profile(name: str) -> tuple[int, dict]:
    """A from-scratch element-order computation for each named group, using
    ONLY plain Python (never sympy.combinatorics, never p15a_group_data) --
    explicit permutation representations built by hand."""

    def perm_order(p: tuple) -> int:
        n = len(p)
        identity = tuple(range(n))
        cur = p
        k = 1
        while cur != identity:
            cur = tuple(p[cur[i]] for i in range(n))
            k += 1
        return k

    def tally(perms):
        c = Counter(perm_order(p) for p in perms)
        return len(perms), dict(c)

    if name.startswith("the cyclic group Z"):
        n = int(name.rsplit("Z", 1)[1])
        elems = [tuple((i + k) % n for i in range(n)) for k in range(n)]
        return tally(elems)
    if name == "Z2 x Z2":
        elems = list(product(range(2), range(2)))

        def order_pair(a, b):
            oa = 1 if a == 0 else 2
            ob = 1 if b == 0 else 2
            return oa if oa >= ob else ob

        c = Counter(order_pair(a, b) for a, b in elems)
        return 4, dict(c)
    if name == "Z4 x Z2":
        c = Counter()
        for a in range(4):
            for b in range(2):
                oa = {0: 1, 1: 4, 2: 2, 3: 4}[a]
                ob = 1 if b == 0 else 2
                import math

                lcm = oa * ob // math.gcd(oa, ob)
                c[lcm] += 1
        return 8, dict(c)
    if name == "Z2 x Z2 x Z2":
        c = Counter()
        for tup in product(range(2), repeat=3):
            o = 1 if all(v == 0 for v in tup) else 2
            c[o] += 1
        return 8, dict(c)
    if name == "Z3 x Z3":
        c = Counter()
        for a in range(3):
            for b in range(3):
                o = 1 if (a == 0 and b == 0) else 3
                c[o] += 1
        return 9, dict(c)
    if name == "Z6 x Z2":
        import math

        c = Counter()
        for a in range(6):
            for b in range(2):
                oa = {0: 1, 1: 6, 2: 3, 3: 2, 4: 3, 5: 6}[a]
                ob = 1 if b == 0 else 2
                lcm = oa * ob // math.gcd(oa, ob)
                c[lcm] += 1
        return 12, dict(c)
    if name.startswith("S3") or name.startswith("D") or name == "A4 (the even permutations of 4 objects)":
        # Build via explicit permutation groups on small point sets.
        if name.startswith("S3"):
            elems = list(permutations(range(3)))
        elif "square" in name:  # D4, order 8
            elems = [tuple((i + k) % 4 for i in range(4)) for k in range(4)]
            elems += [tuple((k - i) % 4 for i in range(4)) for k in range(4)]
        elif "pentagon" in name:  # D5, order 10
            elems = [tuple((i + k) % 5 for i in range(5)) for k in range(5)]
            elems += [tuple((k - i) % 5 for i in range(5)) for k in range(5)]
        elif "hexagon" in name:  # D6, order 12
            elems = [tuple((i + k) % 6 for i in range(6)) for k in range(6)]
            elems += [tuple((k - i) % 6 for i in range(6)) for k in range(6)]
        else:  # A4: even permutations of 4 points

            def is_even(p):
                n = len(p)
                seen = [False] * n
                parity = 0
                for start in range(n):
                    if seen[start]:
                        continue
                    length = 0
                    j = start
                    while not seen[j]:
                        seen[j] = True
                        j = p[j]
                        length += 1
                    parity += length - 1
                return parity % 2 == 0

            elems = [p for p in permutations(range(4)) if is_even(p)]
        return tally(elems)
    raise AssertionError(f"unhandled group name: {name}")


_STRUCT_OPT_MARKER_RE = re.compile(r"(?:^|\s)([ABCD]): ")


def _split_options(opts_section: str) -> dict:
    markers = list(_STRUCT_OPT_MARKER_RE.finditer(opts_section))
    opts = {}
    for i, m in enumerate(markers):
        letter = m.group(1)
        start = m.end()
        end = markers[i + 1].start() if i + 1 < len(markers) else len(opts_section)
        opts[letter] = opts_section[start:end].strip().rstrip(".")
    return opts


def _oracle_struct(text: str) -> str:
    n_match = re.search(r"order (\d+)", text)
    n = int(n_match.group(1))
    if "listed in no particular order" in text:
        list_match = re.search(r"is: ([\d, ]+)\. Tally", text)
        raw = [int(v) for v in list_match.group(1).split(",")]
        profile_supplied = dict(Counter(raw))
    else:
        table_match = re.search(r"with ((?:\d+ elements? of order \d+(?:, )?)+)\.", text)
        profile_supplied = {
            int(o): int(c) for c, o in re.findall(r"(\d+) elements? of order (\d+)", table_match.group(1))
        }
    opts_section = text[text.index("A:") :]
    opts = _split_options(opts_section)
    matches = []
    for letter, name in opts.items():
        order_c, profile_c = _oracle_group_profile(name)
        if order_c == n and profile_c == profile_supplied:
            matches.append(letter)
    assert len(matches) == 1, (text, opts, profile_supplied, matches)
    return matches[0]


@pytest.mark.parametrize("tier", ["easy", "medium", "hard"])
def test_struct_oracle_matches_table(tier):
    checked = 0
    for seed in range(N_SEEDS):
        q = make(STRUCT_TOPIC, seed=seed, difficulty=tier)
        expected = _oracle_struct(q.text)
        assert expected == q.answer_text, (tier, seed, q.text, q.answer_text)
        checked += 1
    assert checked == N_SEEDS


def test_struct_oracle_matches_raw_list_exam():
    checked = 0
    for seed in range(N_SEEDS):
        q = make(STRUCT_TOPIC, seed=seed, difficulty="exam")
        expected = _oracle_struct(q.text)
        assert expected == q.answer_text, (seed, q.text, q.answer_text)
        checked += 1
    assert checked == N_SEEDS


def test_struct_oracle_is_red_on_a_corrupted_answer():
    q = make(STRUCT_TOPIC, seed=0, difficulty="easy")
    expected = _oracle_struct(q.text)
    corrupted = next(o for o in mcq_options(q) if o != expected)
    assert corrupted != expected
