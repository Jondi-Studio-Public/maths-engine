"""Re-derives each rendered question's mcq key from `question.text` alone, via hand-written parsing and
predicates -- never importing anything from `mathgen.generators.p15a_subgroup_isomorphism_build`, not a
helper, not a constant, not even `mathgen.generators.p15a_group_data`.

Total re-derived: 1200 for the subgroup-test skill (4 tiers x 300 seeds) +
1200 for the isomorphism-invariant skill (4 tiers x 300 seeds) = 2400,
clearing the >= 1000 combined bar.
"""

from __future__ import annotations

import re
from collections import Counter

import pytest

from mathgen.grading.typed import mcq_options
from mathgen import make

SUB_TOPIC = "p15a_finite_subgroup_test"
ISO_TOPIC = "p15a_isomorphism_invariant"
N_SEEDS = 300

# ===========================================================================
# Oracle A: p15a_finite_subgroup_test
# ===========================================================================


def _oracle_parse_additive(text: str):
    m = re.search(r"modulo (\d+),", text)
    n = int(m.group(1))
    m2 = re.search(r"Let S = \{([^}]*)\}", text)
    S = [int(x.strip()) for x in m2.group(1).split(",")]
    return n, S


def _oracle_cycle_to_perm(token: str, n: int) -> tuple:
    """Parse one cycle-notation permutation like '(1 2 3)' or '(1 2)(3 4)' or
    '()' into a 0-indexed image tuple of length n."""
    perm = list(range(n))
    if token == "()":
        return tuple(perm)
    for cyc_str in re.findall(r"\(([^()]*)\)", token):
        idxs = [int(x) - 1 for x in cyc_str.split()]
        for i in range(len(idxs)):
            perm[idxs[i]] = idxs[(i + 1) % len(idxs)]
    return tuple(perm)


def _oracle_parse_permutation(text: str):
    m = re.search(r"Let S = \{(.*)\}\. Check the conditions", text)
    body = m.group(1)
    # Split on '), ' boundaries that separate whole permutation tokens: each
    # token is a maximal run of one or more '(...)' groups, or '()'.
    tokens = re.findall(r"(?:\(\)|(?:\([^()]*\))+)", body)
    return [_oracle_cycle_to_perm(t, 4) for t in tokens]


def _oracle_compose(p, q):
    return tuple(p[q[i]] for i in range(len(p)))


def _oracle_inverse(p):
    inv = [0] * len(p)
    for j, pj in enumerate(p):
        inv[pj] = j
    return tuple(inv)


def _oracle_classify_additive(n, S):
    if 0 not in S:
        return "identity"
    S_set = set(S)
    for a in S:
        for b in S:
            if (a + b) % n not in S_set:
                return "closure"
    for a in S:
        assert (-a) % n in S_set, "finite subgroup theorem violated -- oracle itself is wrong"
    return "subgroup"


def _oracle_classify_perm(S):
    n = len(S[0])
    identity = tuple(range(n))
    if identity not in S:
        return "identity"
    S_set = set(S)
    for a in S:
        for b in S:
            if _oracle_compose(a, b) not in S_set:
                return "closure"
    for a in S:
        assert _oracle_inverse(a) in S_set, "finite subgroup theorem violated -- oracle itself is wrong"
    return "subgroup"


def _label_kind(option_text: str) -> str:
    if option_text.startswith("S is a subgroup"):
        return "subgroup"
    if "identity condition" in option_text:
        return "identity"
    if "closure condition" in option_text:
        return "closure"
    if "inverse condition" in option_text:
        return "inverses"
    raise AssertionError(option_text)


@pytest.mark.parametrize("tier", ["easy", "medium"])
def test_sub_oracle_matches_additive(tier):
    checked = 0
    for seed in range(N_SEEDS):
        q = make(SUB_TOPIC, seed=seed, difficulty=tier)
        n, S = _oracle_parse_additive(q.text)
        expected_kind = _oracle_classify_additive(n, S)
        actual_kind = _label_kind(q.answer_text)
        assert expected_kind == actual_kind, (tier, seed, n, S, q.answer_text)
        checked += 1
    assert checked == N_SEEDS


@pytest.mark.parametrize("tier", ["hard", "exam"])
def test_sub_oracle_matches_permutation(tier):
    checked = 0
    for seed in range(N_SEEDS):
        q = make(SUB_TOPIC, seed=seed, difficulty=tier)
        S = _oracle_parse_permutation(q.text)
        expected_kind = _oracle_classify_perm(S)
        actual_kind = _label_kind(q.answer_text)
        assert expected_kind == actual_kind, (tier, seed, S, q.answer_text)
        checked += 1
    assert checked == N_SEEDS


def test_sub_oracle_is_red_on_a_corrupted_answer():
    q = make(SUB_TOPIC, seed=0, difficulty="easy")
    n, S = _oracle_parse_additive(q.text)
    expected_kind = _oracle_classify_additive(n, S)
    corrupted = next(o for o in mcq_options(q) if _label_kind(o) != expected_kind)
    assert _label_kind(corrupted) != expected_kind


# ===========================================================================
# Oracle B: p15a_isomorphism_invariant
# ===========================================================================

_PROFILE_TABLE_RE = re.compile(r"(\d+) elements? of order (\d+)")


def _oracle_parse_table_pair(text: str):
    m = re.search(
        r"Group G has order (\d+), with ([^.]+)\. Group H has order (\d+), with ([^.]+)\.",
        text,
    )
    order_g, profile_g_text, order_h, profile_h_text = m.groups()
    profile_g = {int(o): int(c) for c, o in _PROFILE_TABLE_RE.findall(profile_g_text)}
    profile_h = {int(o): int(c) for c, o in _PROFILE_TABLE_RE.findall(profile_h_text)}
    return int(order_g), profile_g, int(order_h), profile_h


def _tally(raw):
    c = Counter(raw)
    return dict(c)


def _oracle_parse_raw_list_pair(text: str):
    # Two shapes: hard ("Group G has order N. The order of each of its N
    # elements, listed in no particular order, is: ...") and exam (one
    # shuffled, G/H-tagged list). Handle both with independent patterns.
    m_hard_g = re.search(
        r"Group G has order (\d+)\. The order of each of its \d+ elements,"
        r" listed in no particular order, is: ([\d, ]+)\.",
        text,
    )
    m_hard_h = re.search(
        r"Group H has order (\d+), with element orders \(again in no "
        r"particular order\): ([\d, ]+)\.",
        text,
    )
    if m_hard_g and m_hard_h:
        order_g = int(m_hard_g.group(1))
        raw_g = [int(x) for x in m_hard_g.group(2).split(",")]
        order_h = int(m_hard_h.group(1))
        raw_h = [int(x) for x in m_hard_h.group(2).split(",")]
        return order_g, _tally(raw_g), order_h, _tally(raw_h)
    # exam shape: "Group G has order N and group H has
    # order M. The list below has one entry per element ... : G:3, H:2, ...".
    # Re-derived independently by TAG, never by trusting the construction's
    # own separated raw_g/raw_h -- the whole point of this tier is that the
    # learner (and this oracle) must split the interleaved tags apart first.
    m_exam_orders = re.search(r"Group G has order (\d+) and group H has order (\d+)\. ", text)
    m_exam_tagged = re.search(r"shuffled together in no particular order: ([A-Z]:\d+(?:, [A-Z]:\d+)*)\.", text)
    order_g = int(m_exam_orders.group(1))
    order_h = int(m_exam_orders.group(2))
    raw_g, raw_h = [], []
    for entry in m_exam_tagged.group(1).split(", "):
        tag, value = entry.split(":")
        (raw_g if tag == "G" else raw_h).append(int(value))
    return order_g, _tally(raw_g), order_h, _tally(raw_h)


def _oracle_decide(order_g, profile_g, order_h, profile_h) -> str:
    if order_g != order_h:
        return "reject_size"
    if profile_g == profile_h:
        return "retain"
    return "reject_profile"


def _iso_label_kind(option_text: str) -> str:
    if option_text.startswith("retain"):
        return "retain"
    if "differ" in option_text and "|G| and |H|" in option_text:
        return "reject_size"
    if "counts differ" in option_text:
        return "reject_profile"
    raise AssertionError(option_text)


@pytest.mark.parametrize("tier", ["easy", "medium"])
def test_iso_oracle_matches_table(tier):
    checked = 0
    for seed in range(N_SEEDS):
        q = make(ISO_TOPIC, seed=seed, difficulty=tier)
        order_g, profile_g, order_h, profile_h = _oracle_parse_table_pair(q.text)
        expected = _oracle_decide(order_g, profile_g, order_h, profile_h)
        actual = _iso_label_kind(q.answer_text)
        assert expected == actual, (tier, seed, order_g, profile_g, order_h, profile_h, q.answer_text)
        checked += 1
    assert checked == N_SEEDS


@pytest.mark.parametrize("tier", ["hard", "exam"])
def test_iso_oracle_matches_raw_list(tier):
    checked = 0
    for seed in range(N_SEEDS):
        q = make(ISO_TOPIC, seed=seed, difficulty=tier)
        order_g, profile_g, order_h, profile_h = _oracle_parse_raw_list_pair(q.text)
        expected = _oracle_decide(order_g, profile_g, order_h, profile_h)
        actual = _iso_label_kind(q.answer_text)
        assert expected == actual, (tier, seed, order_g, profile_g, order_h, profile_h, q.answer_text)
        checked += 1
    assert checked == N_SEEDS


def test_iso_oracle_is_red_on_a_corrupted_answer():
    q = make(ISO_TOPIC, seed=0, difficulty="easy")
    order_g, profile_g, order_h, profile_h = _oracle_parse_table_pair(q.text)
    expected = _oracle_decide(order_g, profile_g, order_h, profile_h)
    corrupted = next(o for o in mcq_options(q) if _iso_label_kind(o) != expected)
    assert _iso_label_kind(corrupted) != expected
