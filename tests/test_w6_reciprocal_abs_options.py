"""`senior-graph-reciprocal-abs-compare`: the identify-g tiers offer five options and key all of them in turn."""

from __future__ import annotations

from mathgen import make

TOPIC = "ey_graph_reciprocal_abs"
MENU = {"g(x) = |f(x)|", "g(x) = -f(x)", "g(x) = f(|x|)", "g(x) = 1/f(x)", "g(x) = -|f(x)|", "g(x) = f(-x)"}


def test_identify_tiers_have_five_options_from_the_six_item_menu():
    for tier in ("easy", "medium"):
        keys = set()
        for seed in range(60):
            q = make(TOPIC, seed=seed, difficulty=tier)
            options = {q.answer_text, *q.distractors}
            assert len(options) == 5 and options <= MENU, (tier, seed, options)
            keys.add(q.answer_text)
        assert len(keys) >= (4 if tier == "easy" else 5), (tier, keys)
