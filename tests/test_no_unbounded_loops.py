"""No new uncapped `while` loops in mathgen/generators/.

Rejection sampling that can spin forever on an unlucky seed is replaced by `core.draw_until`
(one value) or `core.attempts` (several values per try), which raise VerificationError when the
budget runs out. Loops that predate the helper are listed in tests/unbounded_loops_allowlist.yaml;
this test fails on a loop that is not listed and on a listed loop that no longer exists, so the
list only shrinks. See scripts/list_unbounded_loops.py for what counts as a finding.
"""

from collections import Counter

import pytest

from scripts.list_unbounded_loops import ALLOWLIST, KINDS, load_allowlist, scan_source, scan_tree

REGEN = "python scripts/list_unbounded_loops.py --write"


def _per_function(rows):
    return Counter((r["module"], r["function"]) for r in rows)


def test_no_new_unbounded_loops():
    found, allowed = _per_function(scan_tree()), _per_function(load_allowlist())
    new = sorted(key for key, n in found.items() if n > allowed.get(key, 0))
    assert not new, (
        "unbounded `while` loop in mathgen/generators/ (in "
        + ", ".join(f"{m}.{f}" for m, f in new)
        + "). Use mathgen.core.draw_until(rng, draw, accept, tries=..., what=...) for rejection sampling, or "
        "`for _ in core.attempts(limit, what):` for a loop that draws several values, so it raises "
        "VerificationError instead of spinning forever. Do not add it to the allowlist."
    )


def test_allowlist_has_no_stale_entries():
    found, allowed = _per_function(scan_tree()), _per_function(load_allowlist())
    stale = sorted(key for key, n in allowed.items() if n > found.get(key, 0))
    assert not stale, (
        "allowlist entries whose loop is gone or now bounded (good): "
        + ", ".join(f"{m}.{f}" for m, f in stale)
        + f". Shrink the list with `{REGEN}`."
    )


def test_allowlist_is_regenerated_not_hand_edited():
    assert load_allowlist() == scan_tree(), f"{ALLOWLIST.name} differs from the tree: run `{REGEN}`"


def test_allowlist_entries_are_well_formed():
    rows = load_allowlist()
    assert rows, "the allowlist is empty: delete it and this check only when every loop is converted"
    keys = [(r["module"], r["function"], r["index"]) for r in rows]
    assert len(keys) == len(set(keys))
    for r in rows:
        assert r["kind"] in KINDS, r
        assert bool(r.get("reason")) == (r["kind"] == "algorithmic"), f"only algorithmic loops carry a reason: {r}"


@pytest.mark.parametrize(
    "source, expected",
    [
        ("def f(rng):\n    while True:\n        a = rng.randint(1, 5)\n        if a:\n            break\n", 1),
        ("def f(rng):\n    while a == b:\n        b = rng.randint(1, 5)\n", 1),
        ("def f(rng):\n    while ok(rng):\n        pass\n", 1),
        (
            "def f(rng):\n    t = 0\n    while True:\n        t += 1\n        a = rng.random()\n        if a > .5 or t >= 9:\n            break\n",
            0,
        ),
        ("def f(rng):\n    t = 0\n    while t < 5:\n        t += 1\n        a = rng.random()\n", 0),
        ("def f(n):\n    while n % 2 == 0:\n        n //= 2\n", 0),
        ("def f(a, b):\n    while b:\n        a, b = b, a % b\n", 0),
        (
            "class C:\n    def g(self, rng):\n        while True:\n            rng.random()\n            def h():\n                while True:\n                    pass\n",
            2,
        ),
    ],
)
def test_scanner_detects_the_documented_shapes(source, expected):
    assert len(scan_source(source)) == expected


def test_scanner_indexes_by_function_not_line():
    src = "def f(rng):\n    while True:\n        rng.random()\n    while True:\n        rng.random()\n"
    shifted = "\n\n# unrelated edit\n" + src
    assert scan_source(src) == scan_source(shifted) == [("f", 0, "while_true"), ("f", 1, "while_true")]
