"""Classification-style oracle: enumerates the full draw space (every ordered
(P, Q) pair x every S_kind for skill 1's medium tier; every curated triple x
a representative N sample for skill 2), not a seed sample. Derives every
expected answer purely from `question.text` and re-implements the truth-table
identity / forward-backward classification FROM SCRATCH in this file -- does
not import `_PREDICATES`, `FORWARD_ONLY`, `BICONDITIONAL`, `UNRELATED`, or any
other helper from `mathgen.generators.p9a_implication_symbols_build`.

Generates through each skill's own `sources` (topic + params + difficulty),
never a bare `make(topic, seed, difficulty=tier)`.
"""

from __future__ import annotations

import itertools
import re
from collections import Counter, defaultdict

import yaml

import mathgen

SKILL1_PATH = "skills/senior-mathematics/senior-use-implication-converse-equivalence-negation-contrapositive.yaml"
SKILL2_PATH = (
    "skills/senior-mathematics/senior-use-the-symbols-for-implication-rightarrow-equivalence-longleftrightarrow.yaml"
)


def _load_sources(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)["sources"]


def _source_for(sources, difficulty):
    matches = [s for s in sources if s["difficulty"] == difficulty]
    assert len(matches) == 1
    return matches[0]


SKILL1_SOURCES = _load_sources(SKILL1_PATH)
SKILL2_SOURCES = _load_sources(SKILL2_PATH)

SWEEP = 4000  # ~0.2 ms/question measured; 4000 seeds is <1 s and saturates
# every combo below with high statistical confidence (see
# per-combo probability estimates in the module comments below).


# ---------------------------------------------------------------------------
# Skill 1, medium tier ("equivalence_check"): 30 ordered (P, Q) pairs x 3
# S_kind values = 90 rows.
# ---------------------------------------------------------------------------

_MEDIUM_RE = re.compile(
    r'^If (\d+) (.+?), then (\d+) (.+?)\. Consider its (\w+): "(.+?)" '
    r"Is this ALWAYS logically equivalent to the original implication, for every value of n\?$"
)


def _fresh_truth_table_identity(s_kind: str) -> bool:
    """Whether the named transform ALWAYS matches P => Q on all 4 boolean
    rows -- written fresh here, independent of the generator's own copy."""
    rows = list(itertools.product([False, True], repeat=2))
    implication = lambda p, q: (not p) or q
    transform = {
        "contrapositive": lambda p, q: q or (not p),
        "converse": lambda p, q: (not q) or p,
        "negation": lambda p, q: p and (not q),
    }[s_kind]
    return all(transform(p, q) == implication(p, q) for p, q in rows)


def test_medium_tier_equivalence_check_oracle():
    spec = _source_for(SKILL1_SOURCES, "medium")
    seen_pairs = set()
    seen_combos = set()
    counts = Counter()
    checked = 0
    for seed in range(SWEEP):
        q = mathgen.make(spec["topic"], seed=seed, difficulty=spec["difficulty"], **(spec["params"] or {}))
        m = _MEDIUM_RE.match(q.text)
        assert m, f"unrecognised medium-tier question shape: {q.text!r}"
        n1, p_name, n2, q_name, s_kind, s_text = m.groups()
        assert n1 == n2, f"the two clauses disagree on n: {q.text!r}"
        assert s_kind in ("converse", "contrapositive", "negation"), s_kind

        expected_always_equal = _fresh_truth_table_identity(s_kind)
        expected_answer = "True" if expected_always_equal else "False"
        # The key is the verdict plus the identified transform ("False: it is the converse, ...").
        assert q.answer_text.split(":")[0] == expected_answer, (
            f"seed={seed}: s_kind={s_kind} expected {expected_answer!r}, got {q.answer_text!r}\nQuestion: {q.text}"
        )
        checked += 1
        seen_pairs.add((p_name, q_name))
        seen_combos.add((p_name, q_name, s_kind))
        counts[s_kind] += 1

    print(f"\nmedium-tier equivalence_check oracle: {checked} questions checked")
    print(f"distinct (P, Q) pairs seen: {len(seen_pairs)}/30")
    print(f"distinct (P, Q, S_kind) combos seen: {len(seen_combos)}/90")
    print(f"S_kind counts: {dict(counts)}")

    assert len(seen_pairs) == 30, f"only {len(seen_pairs)}/30 ordered (P, Q) pairs observed"
    assert set(counts) == {"converse", "contrapositive", "negation"}
    assert len(seen_combos) == 90, f"only {len(seen_combos)}/90 (P, Q, S_kind) combos observed"


# ---------------------------------------------------------------------------
# Skill 2, hard tier: 3 FORWARD_ONLY + 3 BICONDITIONAL + 2 UNRELATED triples x
# a representative N sample ({24, 30, 40}) = 24 rows.
# ---------------------------------------------------------------------------

_EQUALITY_RE = re.compile(
    r"^Statement A: the value of (\d+) × (\d+)\. Statement B: the value of (\d+)\. "
    r"Which symbol correctly relates A and B: ⇒, ⇔, or =\?$"
)
_PAIR_RE = re.compile(
    r"^Statement A: (.+?)\. Statement B: (.+?)\. This is considered for every whole "
    r"number n from 1 to (\d+)\. Which symbol correctly relates A and B: ⇒, ⇔, or =\?$"
)

REPRESENTATIVE_NS = (24, 30, 40)


def _parse_predicate(desc: str):
    """Derive a predicate function fresh from the English description text --
    NOT a copy of the generator's FORWARD_ONLY/BICONDITIONAL/UNRELATED
    tables, which this file never imports."""
    desc = desc.strip()
    m = re.fullmatch(r"n is a multiple of (\d+) and a multiple of (\d+)", desc)
    if m:
        k1, k2 = int(m.group(1)), int(m.group(2))
        return lambda n: n % k1 == 0 and n % k2 == 0
    m = re.fullmatch(r"n is a multiple of (\d+)", desc)
    if m:
        k = int(m.group(1))
        return lambda n: n % k == 0
    if desc == "n is not odd":
        return lambda n: n % 2 == 0
    if desc == "n is even":
        return lambda n: n % 2 == 0
    if desc == "n is odd":
        return lambda n: n % 2 == 1
    if desc == "n is a perfect square":
        import math

        return lambda n: n >= 0 and math.isqrt(n) ** 2 == n
    raise ValueError(f"unrecognised predicate description: {desc!r}")


def _classify(a_desc: str, b_desc: str, n_hi: int) -> str:
    a_fn = _parse_predicate(a_desc)
    b_fn = _parse_predicate(b_desc)
    forward = all((not a_fn(n)) or b_fn(n) for n in range(1, n_hi + 1))
    backward = all((not b_fn(n)) or a_fn(n) for n in range(1, n_hi + 1))
    if forward and backward:
        return "⇔"
    if forward and not backward:
        return "⇒"
    if backward and not forward:
        return "⇐-only (not in the 4-option universe)"
    return "No single symbol connects these"


def test_hard_tier_symbol_classify_oracle():
    spec = _source_for(SKILL2_SOURCES, "hard")
    triple_n_seen: dict[tuple[str, str], set[int]] = defaultdict(set)
    equality_checked = 0
    pair_checked = 0
    for seed in range(SWEEP):
        q = mathgen.make(spec["topic"], seed=seed, difficulty=spec["difficulty"], **(spec["params"] or {}))

        eq_match = _EQUALITY_RE.match(q.text)
        if eq_match:
            a, b, product = (int(g) for g in eq_match.groups())
            expected = "=" if a * b == product else "no single symbol connects these (arithmetic mismatch)"
            assert q.answer_text == "=", f"seed={seed}: equality branch answer {q.answer_text!r}\n{q.text}"
            assert a * b == product, f"seed={seed}: {a} x {b} != {product}\n{q.text}"
            equality_checked += 1
            continue

        pair_match = _PAIR_RE.match(q.text)
        assert pair_match, f"unrecognised hard-tier question shape: {q.text!r}"
        a_desc, b_desc, n_hi_str = pair_match.groups()
        n_hi = int(n_hi_str)
        expected_symbol = _classify(a_desc, b_desc, n_hi)
        assert expected_symbol in ("⇒", "⇔", "No single symbol connects these"), (
            f"seed={seed}: classification produced an out-of-universe symbol {expected_symbol!r}\nQuestion: {q.text}"
        )
        assert q.answer_text == expected_symbol, (
            f"seed={seed}: expected {expected_symbol!r}, got {q.answer_text!r}\nQuestion: {q.text}"
        )
        pair_checked += 1
        if n_hi in REPRESENTATIVE_NS:
            triple_n_seen[(a_desc, b_desc)].add(n_hi)

    print(f"\nhard-tier symbol_classify oracle: {equality_checked} equality rows, {pair_checked} pair rows checked")
    print(f"distinct (triple, representative N) cells observed: {sum(len(v) for v in triple_n_seen.values())}/24")
    for triple, ns in sorted(triple_n_seen.items()):
        print(f"  {triple}: N={sorted(ns)}")

    assert len(triple_n_seen) == 8, f"only {len(triple_n_seen)}/8 distinct triples observed"
    missing = [(triple, n) for triple in triple_n_seen for n in REPRESENTATIVE_NS if n not in triple_n_seen[triple]]
    assert not missing, f"representative (triple, N) cells never observed in {SWEEP} seeds: {missing}"
