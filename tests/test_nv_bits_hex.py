"""Tests `bits-hex-and-digit-counts`: every key is re-derived from the printed stem with plain integer arithmetic."""

from __future__ import annotations

import math
import re

import pytest

from mathgen.engine import grade
from tests import r3b_support as sup

SKILL = "bits-hex-and-digit-counts"
TOPIC = "nv_bits_hex_counts"
EXAM = "exd_x_nv_bits_hex_counts"
FORMAT = " Type the digits only: no 0x, no #, no spaces."


def ints(text):
    return [int(x) for x in re.findall(r"\d+", text)]


def smallest_bits(n):
    """Fewest bits whose patterns reach 0..n, found by counting up (not by bit_length)."""
    k = 1
    while 2**k <= n:
        k += 1
    return k


def count_oracle(q):
    """The numeric key recomputed from the stem alone, by plain loops and comparisons."""
    t = q.text
    n = ints(t)[0]
    if re.search(r"different \d+-digit hexadecimal|locker code|device identifier", t):
        return 16**n
    if re.search(
        r"smallest number of bits that can hold|least number of binary digits|How many bits does it need|last ticket|fewest bits that can store|age field",
        t,
    ):
        return smallest_bits(max(ints(t)))
    if re.search(
        r"largest whole number it can hold|largest whole number that can be stored|largest reading it can send", t
    ):
        return int("1" * n, 2)
    if re.search(r"different values|different codes are possible|different settings", t):
        return len({format(i, f"0{n}b") for i in range(2**n)})
    if re.search(r"hexadecimal digits are needed|greatest number of hexadecimal digits", t):
        return len(format(2 ** ints(t)[-1] - 1, "x"))
    if re.search(r"smallest number of bits|bits per barcode|at least \d+ different bit patterns", t):
        k = 0
        while 2**k < n:
            k += 1
        return k
    if re.search(r"largest whole number \(in decimal\)|largest value, in decimal", t):
        return int("F" * n, 16)
    raise AssertionError(t)


def test_wired():
    sup.assert_wired(SKILL, TOPIC, EXAM, "mcq", ["number-bases-primes-and-divisors"], {"tier": "exam"})


@pytest.mark.parametrize("tier", ["easy", "medium", "hard"])
def test_numeric_keys_match_the_stem(tier):
    checked = 0
    for seed in range(200):
        q = sup.tier_question(TOPIC, tier, seed)
        if q.answer_kind != "number":
            continue
        assert int(q.answer_text) == count_oracle(q), (q.text, q.answer_text)
        checked += 1
    assert checked >= 40


def hex_oracle(q):
    t = q.text.replace(FORMAT, "")
    nums = ints(t)
    if re.search(r"bit pattern|binary number|binary numeral", t):
        bits = re.search(r"\b[01]{6,}\b", t).group()
        padded = bits.rjust(-(-len(bits) // 4) * 4, "0")
        return "".join("0123456789ABCDEF"[int(padded[i : i + 4], 2)] for i in range(0, len(padded), 4))
    if re.search(r"\bred\b", t):
        return "".join(f"{v:02X}" for v in nums[-3:])
    if re.search(r"register of|bits of a field|largest whole number that can be stored", t):
        return f"{int('1' * nums[0], 2):X}"
    if re.search(r"exactly|-digit hexadecimal numeral", t):
        width = nums[0] if "exactly" not in t or t.startswith("A memory") else None
        if t.startswith("A memory"):
            return f"{nums[1]:X}".rjust(nums[0], "0")
        if t.startswith("Write"):
            return f"{nums[0]:X}".rjust(nums[1], "0")
        return f"{nums[1]:X}".rjust(nums[0], "0")
    return f"{nums[0]:X}"


@pytest.mark.parametrize("tier", ["medium", "hard"])
def test_hex_keys_match_the_stem(tier):
    seen = 0
    for seed in range(300):
        q = sup.tier_question(TOPIC, tier, seed)
        if q.answer_kind != "ordered_list":
            continue
        assert q.answer == [q.answer_text]
        assert q.answer_text == hex_oracle(q), (q.text, q.answer_text)
        seen += 1
    assert seen >= 30


def test_hex_grading_is_case_insensitive_and_keeps_leading_zeros():
    sk = sup.skill_object(SKILL)
    padded = unpadded = lower = 0
    for seed in range(300):
        q = sup.tier_question(TOPIC, "hard", seed)
        if q.answer_kind != "ordered_list":
            continue
        key = q.answer_text
        assert grade(sk, q, key).correct
        assert grade(sk, q, key.lower()).correct and grade(sk, q, " " + key.lower() + " ").correct
        lower += 1
        if key.startswith("0") and "exactly" in q.text:
            stripped = key.lstrip("0")
            assert not grade(sk, q, stripped).correct, (q.text, key)
            assert not grade(sk, q, "0" + key).correct
            padded += 1
        assert not grade(sk, q, key[1:]).correct if len(key) > 1 else True
        unpadded += 1
    assert padded >= 5 and lower >= 20 and unpadded >= 20


@pytest.mark.parametrize("tier", ["easy", "medium", "hard"])
def test_clean_graded_and_varied(tier):
    sup.assert_clean_and_deterministic(TOPIC, tier)
    sup.assert_single_graded(SKILL, TOPIC, tier)
    sup.assert_answer_space(TOPIC, tier, distinct=8)


def test_tiers_differ():
    sup.assert_tier_shapes(TOPIC, EXAM)


def test_hard_boundary_questions_are_really_at_the_boundary():
    """256 needs 9 bits (not 8), 255 needs 8: the key is never the exponent of a power-of-two limit."""
    boundary = 0
    for seed in range(300):
        q = sup.tier_question(TOPIC, "hard", seed)
        if "last ticket" in q.text or "whole number from 0 up to" in q.text or "age field" in q.text:
            n = max(ints(q.text))
            if n & (n - 1) == 0:
                assert int(q.answer_text) == int(math.log2(n)) + 1
                boundary += 1
    assert boundary >= 3


def test_largest_n_bit_hex_has_the_short_top_digit():
    checked = 0
    for seed in range(300):
        q = sup.tier_question(TOPIC, "hard", seed)
        if q.answer_kind == "ordered_list" and re.search(
            r"filled with 1s|set to 1|largest whole number that can be stored", q.text
        ):
            n = ints(q.text)[0]
            assert q.answer_text[0] != "F" or n % 4 == 0
            assert int(q.answer_text, 16) == 2**n - 1
            checked += 1
    assert checked >= 5


def test_exam_chain_follows_through_and_is_graded():
    sk = sup.skill_object(SKILL)
    kinds = set()
    for seed in range(120):
        q = sup.exam_question(EXAM, seed)
        top = int(re.search(r"largest reading it ever has to record is (\d+)", q.text).group(1))
        a, b, c, d = q.parts
        bits = smallest_bits(top)
        assert int(a.answer_text) == bits
        assert int(b.answer_text) == 2**bits
        assert int(c.answer_text) == 2**bits - top - 1 and int(c.answer_text) > 0
        if d.answer_kind == "ordered_list":
            assert d.answer_text == f"{top:X}"
        else:
            assert int(d.answer_text) == -(-bits // 4)
        kinds.add(d.answer_kind)
        assert [bool(p.follow_through) for p in q.parts] == [False, True, True, d.answer_kind == "number"]
    assert kinds == {"number", "ordered_list"}
    sup.assert_exam_graded(SKILL, EXAM)
    sup.assert_exam_clean_and_deterministic(EXAM)
    sup.assert_exam_answer_space(EXAM)


def test_exam_follow_through():
    sk = sup.skill_object(SKILL)
    import sympy

    for seed in range(60):
        q = sup.exam_question(EXAM, seed)
        values = {}
        for p in q.parts:
            if p.answer_kind != "number":
                continue
            if p.follow_through:
                assert sympy.sympify(p.follow_through, locals=values) == int(p.answer_text), (seed, p.label)
            values[f"part_{p.label}"] = sympy.Integer(int(p.answer_text))
    from tests.r3d_support import assert_follow_through_credit

    assert_follow_through_credit(sk, EXAM, range(40), min_checked=10)


def test_unknown_tier_raises():
    import random

    from mathgen.generators.nv_bits_hex_build import exd_x_nv_bits_hex_counts, nv_bits_hex_counts

    with pytest.raises(ValueError):
        nv_bits_hex_counts(random.Random(1), tier="exam")
    with pytest.raises(ValueError):
        exd_x_nv_bits_hex_counts(random.Random(1), tier="hard")


def test_every_hex_question_states_the_format_to_type():
    """The shared ordered-list grader compares the typed text, so each hex question says what to type."""
    seen = 0
    for tier in ("medium", "hard"):
        for seed in range(200):
            q = sup.tier_question(TOPIC, tier, seed)
            if q.answer_kind == "ordered_list":
                assert q.text.endswith(FORMAT), q.text
                seen += 1
    for seed in range(100):
        for p in sup.exam_question(EXAM, seed).parts:
            if p.answer_kind == "ordered_list":
                assert FORMAT in p.prompt
                seen += 1
    assert seen >= 100


def _unit_rank_and_position(unit):
    from mathgen.core import mcq_options

    key, wrong = unit.answer_text, list(unit.distractors)
    try:
        value = lambda t: int(t, 16) if unit.answer_kind == "ordered_list" else float(t)
        rank = sum(value(w) < value(key) for w in wrong)
    except ValueError:
        rank = sum(w < key for w in wrong)
    return rank, mcq_options(unit).index(key)


@pytest.mark.parametrize("tier", ["easy", "medium", "hard", "exam"])
def test_key_rank_and_position_vary(tier):
    """Over 240 seeds each part or question has its key at three or more different ranks and in several list places."""
    from collections import defaultdict

    ranks, places = defaultdict(set), defaultdict(set)
    for seed in range(240):
        q = sup.exam_question(EXAM, seed) if tier == "exam" else sup.tier_question(TOPIC, tier, seed)
        for label, unit in [(p.label, p) for p in q.parts] or [("q", q)]:
            r, pos = _unit_rank_and_position(unit)
            ranks[label].add(r)
            places[label].add(pos)
    for label in ranks:
        assert len(ranks[label]) >= 3, (tier, label, ranks[label])
        assert len(places[label]) >= 2, (tier, label, places[label])


def test_key_rank_does_not_follow_the_stem_kind():
    """Each question kind on its own: a fixed rank for one kind is what the slips-first-listed pattern produces."""
    from collections import Counter, defaultdict

    kinds = defaultdict(Counter)
    for tier in ("easy", "medium", "hard"):
        for seed in range(400):
            q = sup.tier_question(TOPIC, tier, seed)
            kinds[(tier, " ".join(re.sub(r"\d+", "#", q.text).split()[:3]))][_unit_rank_and_position(q)[0]] += 1
    for kind, counts in kinds.items():
        n = sum(counts.values())
        if n >= 40:
            assert max(counts.values()) / n <= 0.6, (kind, counts)


# -- what each stem rules out ---------------------------------------------------------------------------------------
# One row per part: (where it is asked, what its stem implies about every option, the key's value included).
#   bits needed (easy, hard boundary, hard codes, exam a):  a whole number of bits, at least 1, no more than 3 x the key
#   digits needed (medium, exam d):                         a whole number of digits, at least 1
#   largest value / number of values / codes (easy..hard):  a whole number, at least 1
#   exam b (values those bits hold):                        a whole number, at least 2
#   exam c (values left unused):                            a whole number from 0 to (values - 1): fewer than the values
#   hex numeral (medium, hard, exam d):                     digits 0-9 A-F only, no leading zero unless the key has one
#   hex numeral with a stated width (padded, colour):       exactly that many digits
BITS_COUNT = re.compile(
    r"smallest number of bits|least number of binary digits|How many bits does it need|last ticket|fewest bits|age field"
    r"|different binary code|binary barcode|smallest number of bits for which"
)
WIDTH = re.compile(
    r"exactly (\d) hexadecimal digits|using exactly (\d) digits|a (\d)-digit hexadecimal|six-digit|six hexadecimal digits"
)


def hex_violations(text, key, options):
    out = []
    width = None
    if m := WIDTH.search(text):
        width = 6 if "six" in m.group(0) else int(next(g for g in m.groups() if g))
        if len(key) != width:
            out.append(f"key {key} is not {width} digits")
    for o in options:
        if not re.fullmatch(r"[0-9A-F]+", o.upper()):
            out.append(f"{o} is not hexadecimal")
        if width is not None and len(o) != width:
            out.append(f"{o} is not exactly {width} digits")
        if o.startswith("0") and not key.startswith("0"):
            out.append(f"{o} starts with 0 beside {key}")
    return out


def number_violations(text, key, options, *, part=None, context=None):
    out = []
    k = int(key)
    nums = [int(o) for o in options]
    if any(n < 1 for n in nums):
        out.append(f"option below 1: {options}")
    if BITS_COUNT.search(text) and any(n > 3 * k for n in nums):
        out.append(f"magnitude leak beside bits {k}: {options}")
    return out


@pytest.mark.parametrize("tier", ["easy", "medium", "hard"])
def test_no_option_is_ruled_out_by_the_stem(tier):
    bad, seen = [], {"hex": 0, "num": 0, "width": 0}
    for seed in range(400):
        q = sup.tier_question(TOPIC, tier, seed)
        options = list(q.distractors)
        if q.answer_kind == "ordered_list" or not q.answer_text.isdigit() or FORMAT in q.text:
            seen["hex"] += 1
            seen["width"] += bool(WIDTH.search(q.text))
            found = hex_violations(q.text, q.answer_text, options)
        else:
            seen["num"] += 1
            found = number_violations(q.text, q.answer_text, options)
        bad += [f"seed {seed}: {f}" for f in found]
    assert not bad, bad[:5]
    assert seen["hex" if tier != "easy" else "num"] > 0


def test_exam_options_respect_each_part():
    bad = []
    for seed in range(400):
        q = sup.exam_question(EXAM, seed)
        top = int(re.search(r"record is (\d+)", q.text).group(1))
        a, b, c, d = q.parts
        bits, values = int(a.answer_text), int(b.answer_text)
        if any(not 1 <= int(o) <= 3 * bits for o in a.distractors):
            bad.append(f"seed {seed} (a): {a.distractors} beside {bits}")
        if any(int(o) < 2 for o in b.distractors):
            bad.append(f"seed {seed} (b): {b.distractors}")
        if any(not 0 <= int(o) < values for o in c.distractors):
            bad.append(f"seed {seed} (c): {c.distractors} for {values} values")
        if FORMAT in d.prompt:
            bad += [f"seed {seed} (d): {v}" for v in hex_violations(d.prompt, d.answer_text, d.distractors)]
        elif any(int(o) < 1 for o in d.distractors):
            bad.append(f"seed {seed} (d): {d.distractors}")
        assert top >= 1
    assert not bad, bad[:5]


def test_hex_options_are_mixed_in_shape():
    """The key is not the only option with a letter in it, and keys without a letter turn up."""
    letterless_keys = key_only_letters = 0
    for seed in range(600):
        q = sup.tier_question(TOPIC, "medium", seed)
        if FORMAT not in q.text:
            continue
        has = [any(c.isalpha() for c in o) for o in q.distractors]
        if not any(c.isalpha() for c in q.answer_text):
            letterless_keys += 1
        elif not any(has):
            key_only_letters += 1
    assert letterless_keys >= 30, letterless_keys
    assert key_only_letters <= 60, key_only_letters


def _is_pow2(o):
    return o.isdigit() and int(o) >= 1 and int(o) & (int(o) - 1) == 0


PROPERTIES = {
    "power of two": _is_pow2,
    "multiple of 10": lambda o: o.isdigit() and int(o) % 10 == 0 and int(o) > 0,
    "a hexadecimal letter": lambda o: any(c in "ABCDEFabcdef" for c in o) and not o.isdigit(),
}


def _lone(label, key, options, seed, typed_hex=False):
    """The properties the key has that no wrong option has (a typed hexadecimal numeral has only the letter property)."""
    out = []
    for name, has in PROPERTIES.items():
        if typed_hex != (name == "a hexadecimal letter"):
            continue
        if has(key) and not any(has(o) for o in options):
            out.append(f"{label} seed {seed}: the key {key} is the only option that is {name}: {options}")
    return out


@pytest.mark.parametrize("tier", ["easy", "medium", "hard"])
def test_key_is_never_the_only_option_with_a_property(tier):
    bad = []
    for seed in range(1000):
        q = sup.tier_question(TOPIC, tier, seed)
        bad += _lone(tier, q.answer_text, list(q.distractors), seed, q.answer_kind == "ordered_list")
    assert not bad, bad[:5]


def test_exam_key_is_never_the_only_option_with_a_property():
    bad = []
    for seed in range(1000):
        q = sup.exam_question(EXAM, seed)
        for label, p in zip("abcd", q.parts):
            bad += _lone(label, p.answer_text, list(p.distractors), seed, p.answer_kind == "ordered_list")
    assert not bad, bad[:5]


def test_exam_values_part_has_at_least_three_powers_of_two():
    """Part (b) asks for 2^n, so the key must keep company: at least three of the four options are powers of two."""
    few = []
    for seed in range(1200):
        p = sup.exam_question(EXAM, seed).parts[1]
        options = [int(p.answer_text), *(int(o) for o in p.distractors)]
        if sum(1 for v in options if v & (v - 1) == 0) < 3:
            few.append((seed, options))
    assert not few, few[:5]
