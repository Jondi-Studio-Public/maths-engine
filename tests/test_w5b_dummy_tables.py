"""Certifies the dummy-activity solver (`_w5b_dummy_kit`) and every table filed in `_w5b_dummy_tables`.

The hand cases are worked out on paper (each is the classic situation named in its id), so the solver is checked
against counts that do not come from itself. Every filed table is re-solved, and the network the solver returns is
read back into a precedence table that must equal the table's closure.
"""

from __future__ import annotations

import pytest

from mathgen.generators._w5b_dummy_kit import closure_table, fewest_dummies, network_of, network_table
from mathgen.generators._w5b_dummy_tables import TABLES
from mathgen.generators.w5b_precedence_table_build import parse

HAND_CASES = {
    "chain": ({"A": [], "B": ["A"], "C": ["B"]}, 0),
    "two_separate_paths_rejoin": ({"A": [], "B": ["A"], "C": ["A"], "D": ["B"], "E": ["C"], "F": ["D", "E"]}, 0),
    "parallel_pair_then_join": ({"A": [], "B": ["A"], "C": ["A"], "D": ["B", "C"]}, 1),
    "two_starts_share_one_follower": ({"A": [], "B": [], "C": ["A", "B"]}, 1),
    "partial_overlap": ({"A": [], "B": [], "C": ["A", "B"], "D": ["B"]}, 1),
    "nested_predecessors": ({"A": [], "B": [], "C": ["A"], "D": ["A", "B"]}, 1),
    "overlap_both_sides": ({"A": [], "B": [], "C": ["A"], "D": ["A", "B"], "E": ["B"]}, 2),
    "twin_starts_and_twin_followers": ({"A": [], "B": [], "C": ["A", "B"], "D": ["A", "B"]}, 2),
    "two_parallel_pairs": ({"A": [], "B": ["A"], "C": ["A"], "D": ["B", "C"], "E": ["B", "C"]}, 2),
}


@pytest.mark.parametrize("name", sorted(HAND_CASES))
def test_hand_worked_cases(name):
    table, expected = HAND_CASES[name]
    assert fewest_dummies(table) == expected
    assert network_table(network_of(table)) == closure_table(table)


def test_every_filed_table_needs_the_dummies_it_is_filed_under():
    count = 0
    for (activities, dummies), rows in TABLES.items():
        assert len(set(rows)) == len(rows), (activities, dummies)
        for row in rows:
            table = parse(row)
            assert len(table) == activities, row
            assert list(table) == sorted(table), "activities must be lettered in order"
            for name, preds in table.items():
                assert all(p < name for p in preds), (row, name)
            assert fewest_dummies(table) == dummies, row
            assert network_table(network_of(table)) == closure_table(table), row
            count += 1
    assert count >= 100


def test_every_dummy_count_is_offered_for_each_size():
    sizes = {}
    for activities, dummies in TABLES:
        sizes.setdefault(activities, set()).add(dummies)
    assert sizes[5] == {0, 1, 2}
    assert {2, 3, 4} <= sizes[7] and {2, 3, 4} <= sizes[8]
