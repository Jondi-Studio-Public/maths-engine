"""Tests the seven `dg_*` skills: graph terms, constructing column/picture graphs, dot plots and stem-and-leaf plots,
comparing two displays, reading real-data graphs, and placing features on a grid map.

Each skill gets the shared wiring and grading checks (every tier's key accepted, every distractor rejected through the
real grader and tagged with a catalogue id, text clean and deterministic, exam chains graded part by part with
follow-through) and an oracle that re-derives the key from what the learner sees (the question text, or the figure's
spec) by a route that does not share code with the generator: a recount from the printed list, `statistics`, a
SymPy rank or line, or a fresh parse of the map.
"""

from __future__ import annotations

import json
import re
import statistics
from collections import Counter
from fractions import Fraction

import pytest
import sympy

from mathgen import make
from mathgen.engine import grade
from mathgen.misconceptions import CATALOGUE
from tests import w7b_support as sup

SEEDS = range(70)
TIERS = ("easy", "medium", "hard")

# skill id, topic, chain topic, answer kind
SKILLS = [
    ("graph-terms-from-a-network", "dg_graph_terms", "dg_graph_terms_chain", "number"),
    ("construct-column-and-picture-graphs", "dg_construct_columns", "dg_construct_columns_chain", "bars"),
    ("construct-dot-plots-from-data", "dg_construct_dotplot", "dg_construct_dotplot_chain", "bars"),
    ("construct-stem-and-leaf-plots", "dg_construct_stem", "dg_construct_stem_chain", "table"),
    ("compare-two-data-displays", "dg_compare_displays", "dg_compare_displays_chain", "number"),
    ("read-values-and-rates-from-real-data-graphs", "dg_real_data_graphs", "dg_real_data_graphs_chain", "number"),
    ("place-features-on-a-grid-map", "dg_grid_map", "dg_grid_map_chain", "grid_cells"),
]
IDS = [s[0] for s in SKILLS]
BY_ID = {s[0]: s for s in SKILLS}


@pytest.fixture(params=SKILLS, ids=IDS)
def case(request):
    sid, topic, chain, kind = request.param
    return sup.skill_of(sid), topic, chain, kind


def _skill_and_topics(sid):
    _, topic, chain, _ = BY_ID[sid]
    return sup.skill_of(sid), topic, chain


# ---------------------------------------------------------------------------------------------- shared checks


def test_wiring_is_four_sources_easy_medium_hard_exam(case):
    skill, topic, chain, kind = case
    assert skill.status == "active" and skill.answer == kind
    sup.check_wiring(skill, topic, chain, kind)
    assert skill.sources[-1].params == {"tier": "exam"}
    assert skill.prerequisites and all(p in {s.id for s in sup.load_skills()[1]} for p in skill.prerequisites)
    assert skill.difficulty_progression and set(skill.difficulty_progression) == {"easy", "medium", "hard", "exam"}
    steps = [skill.difficulty_progression[t].steps for t in ("easy", "medium", "hard", "exam")]
    assert steps == sorted(steps)


@pytest.mark.parametrize("tier", TIERS)
def test_every_tier_key_is_accepted_and_every_distractor_rejected(case, tier):
    skill, topic, _chain, _kind = case
    sup.check_single_tier(skill, topic, tier, seeds=60)


def test_exam_chain_grades_part_by_part(case):
    skill, _topic, chain, _kind = case
    for seed in range(40):
        q = sup.exam(chain, seed)
        marks = [p.marks for p in q.parts]
        assert [p.label for p in q.parts] == [chr(97 + i) for i in range(len(marks))]
        assert sup.find_text_defects(q) == []
        assert q.to_json() == sup.exam(chain, seed).to_json()
        full = grade(skill, q, q.answer_text)
        assert full.correct and full.marks == full.max_marks == sum(marks), (seed, full)
        canon = {p.label: p.answer_text for p in q.parts}
        for p in q.parts:
            assert len(p.distractors) >= 2
            for text in p.distractors:
                result = grade(skill, q, {**canon, p.label: text})
                assert result.parts[p.label].marks == 0, (seed, p.label, text, p.answer_text)
                assert p.option_notes[text] in CATALOGUE


def test_every_tier_is_its_own_pool_with_enough_variety(case):
    skill, topic, chain, _kind = case
    pools = {}
    for tier in TIERS:
        qs = [sup.single(topic, tier, s) for s in range(60)]
        assert len({(q.text, json.dumps(q.figure, sort_keys=True)) for q in qs}) >= 20, tier
        assert len({q.answer_text for q in qs}) >= 6, tier
        pools[tier] = {q.text for q in qs}
    assert pools["easy"] != pools["medium"] != pools["hard"]
    exam = [sup.exam(chain, s) for s in range(60)]
    assert len({q.answer_text for q in exam}) >= 12


def test_distractor_ids_are_new_dg_ids_in_the_catalogue():
    mine = {k for k in CATALOGUE if k.startswith("dg-")}
    assert len(mine) >= 80
    for sid, topic, chain, _ in SKILLS:
        for tier in TIERS:
            for seed in range(30):
                q = sup.single(topic, tier, seed)
                assert all(v.startswith("dg-") for v in q.option_notes.values()), (sid, tier, seed)


# ---------------------------------------------------------------------------------------------- graph terms


def _fig_graph(fig):
    names = [v["id"] for v in fig["vertices"]]
    idx = {v: i for i, v in enumerate(names)}
    edges = sorted({tuple(sorted((idx[e["from"]], idx[e["to"]]))) for e in fig["edges"]})
    thick = {tuple(sorted((idx[e["from"]], idx[e["to"]]))) for e in fig["edges"] if e.get("highlight")}
    return names, edges, thick


def _components_by_laplacian(n, edges):
    lap = sympy.zeros(n, n)
    for u, v in edges:
        lap[u, u] += 1
        lap[v, v] += 1
        lap[u, v] -= 1
        lap[v, u] -= 1
    return n - lap.rank()  # the nullity of the Laplacian is the number of components


def test_drawn_graph_answers_match_a_sympy_recount():
    seen = Counter()
    for tier in ("easy", "medium"):
        for seed in range(150):
            q = sup.single("dg_graph_terms", tier, seed)
            names, edges, thick = _fig_graph(q.figure)
            n = len(names)
            adj = sympy.zeros(n, n)
            for u, v in edges:
                adj[u, v] = adj[v, u] = 1
            deg = [int(sum(adj.row(i))) for i in range(n)]
            comps = _components_by_laplacian(n, edges)
            t = q.text
            x = re.search(r"[Vv]ertex ([A-J])\b", t)
            if "not adjacent" in t:
                assert q.answer == n - 1 - deg[names.index(x.group(1))]
                seen["nonadj"] += 1
            elif "are adjacent to vertex" in t:
                assert q.answer == deg[names.index(x.group(1))]
                seen["adj"] += 1
            elif "greatest" in t:
                assert q.answer == max(deg)
                seen["max"] += 1
            elif "edges does the network" in t or "Count the edges" in t:
                assert q.answer == sum(deg) // 2 == len(edges)
                seen["edges"] += 1
            elif "fewest number" in t or "At least how many" in t:
                assert q.answer == comps - 1
                seen["connect"] += 1
            elif "component" in t:
                assert q.answer == comps
                seen["comps"] += 1
            elif "How many vertices" in t:
                assert q.answer == len({v for e in thick for v in e})
                seen["subv"] += 1
            else:
                assert "not in H" in t or "left out of H" in t, t
                assert q.answer == len(edges) - len(thick)
                seen["sube"] += 1
    assert min(seen.values()) >= 10 and len(seen) == 8, seen


def _multigraph_from(text):
    listed = re.search(
        r"edges[^:]*: ([A-J]{2}(?:, [A-J]{2})*)",
        text.split(" An edge written")[0] if "An edge written" in text else text,
    )
    if listed is None:
        listed = re.search(r"are ([A-J]{2}(?:, [A-J]{2})*)\.", text)
    return [(p[0], p[1]) for p in listed.group(1).split(", ")]


def test_multigraph_answers_match_a_recount_from_the_printed_edge_list():
    seen = Counter()
    for seed in range(200):
        q = sup.single("dg_graph_terms", "hard", seed)
        assert q.figure is None
        es = _multigraph_from(q.text)
        loops = [e for e in es if e[0] == e[1]]
        pair_counts = Counter(tuple(sorted(e)) for e in es if e[0] != e[1])
        t = q.text
        if "How many loops" in t:
            assert q.answer == len(loops)
            seen["loops"] += 1
        elif "pairs of vertices are joined by multiple edges" in t:
            assert q.answer == sum(1 for c in pair_counts.values() if c > 1)
            seen["pairs"] += 1
        elif "extra edges" in t:
            assert q.answer == sum(c - 1 for c in pair_counts.values())
            seen["extra"] += 1
        elif "simple graph" in t:
            assert q.answer == len(pair_counts)
            seen["simple"] += 1
        else:
            x = re.search(r"degree of vertex ([A-J])", t).group(1)
            assert q.answer == sum((a == x) + (b == x) for a, b in es)  # a loop has both ends at x, so it adds 2
            seen["degree"] += 1
    assert min(seen.values()) >= 15 and len(seen) == 5, seen


def test_the_graph_terms_chain_has_two_shapes_and_the_set_part_is_graded_as_a_set():
    shapes = Counter()
    skill = sup.skill_of("graph-terms-from-a-network")
    for seed in range(80):
        q = sup.exam("dg_graph_terms_chain", seed)
        kinds = tuple(p.answer_kind for p in q.parts)
        shapes[kinds] += 1
        if kinds == ("number", "set", "number", "number", "number"):
            names, edges, _ = _fig_graph(q.figure)
            x = re.search(r"vertex ([A-J])", q.parts[0].prompt).group(1)
            xi = names.index(x)
            nbrs = {names[b if a == xi else a] for a, b in edges if xi in (a, b)}
            assert set(q.parts[1].answer) == nbrs and int(q.parts[0].answer) == len(nbrs)
            m = len(edges)
            assert int(q.parts[2].answer) == m and int(q.parts[3].answer) == m - len(nbrs)
            rest = [(a, b) for a, b in edges if xi not in (a, b)]
            others = [i for i in range(len(names)) if i != xi]
            idx = {v: i for i, v in enumerate(others)}
            assert int(q.parts[4].answer) == _components_by_laplacian(len(others), [(idx[a], idx[b]) for a, b in rest])
            # the typed set is order-free and exact
            shuffled = "{" + ", ".join(reversed(sorted(nbrs))) + "}"
            res = grade(skill, q, {p.label: p.answer_text for p in q.parts} | {"b": shuffled})
            assert res.parts["b"].marks == q.parts[1].marks
            res = grade(
                skill, q, {p.label: p.answer_text for p in q.parts} | {"b": "{" + ", ".join(sorted(nbrs | {x})) + "}"}
            )
            assert res.parts["b"].marks == 0
        else:
            assert kinds == ("number",) * 4
    assert len(shapes) == 2


def test_the_multigraph_chain_follows_through_from_loops_and_extra_edges():
    checked = 0
    for seed in range(120):
        q = sup.exam("dg_graph_terms_chain", seed)
        if q.figure is not None:
            continue
        skill = sup.skill_of("graph-terms-from-a-network")
        canon = {p.label: p.answer_text for p in q.parts}
        total = int(re.search(r"and (\d+) edges", q.text).group(1))
        # a wrong (a) and (b) still earn (c) when (c) is the learner's own total - a - b
        a, b = int(canon["a"]) + 1, int(canon["b"]) + 1
        if total - a - b <= 0 or total - a - b in (
            a,
            b,
        ):  # a carried zero or negative count earns nothing, by the grader's rule
            continue
        res = grade(skill, q, {**canon, "a": str(a), "b": str(b), "c": str(total - a - b)})
        assert res.parts["c"].marks == q.parts[2].marks, (seed, res.parts["c"])
        checked += 1
    assert checked >= 15


# ---------------------------------------------------------------------------------------------- column / picture graphs


def _list_after_colon(text):
    return [w.strip() for w in text.split(": ", 1)[1].split(". ", 1)[0].rstrip(".").split(",")]


def _bars_reply(q):
    answer = q.answer
    return dict(zip(answer["categories"], answer["heights"]))


def test_column_heights_match_a_recount_of_the_printed_data():
    seen = Counter()
    for tier in TIERS:
        for seed in range(120):
            q = sup.single("dg_construct_columns", tier, seed)
            got = {k: Fraction(v) if isinstance(v, str) else Fraction(v) for k, v in _bars_reply(q).items()}
            t = q.text
            if "picture graph" in t:
                key = int(re.search(r"standing for (\d+) ", t).group(1))
                table = re.search(r"\. ((?:[A-Za-z ]+ \d+(?:, )?)+)\. A picture graph", t).group(1)
                counts = {m.group(1): int(m.group(2)) for m in re.finditer(r"([A-Za-z ]+?) (\d+)(?:, |$)", table)}
                assert {k: Fraction(v, key) for k, v in counts.items()} == got
                assert any(v.denominator == 2 for v in got.values())  # at least one half symbol
                seen["picture"] += 1
            elif "In all," in t:
                total = int(re.search(r"In all, (\d+)", t).group(1))
                known = {
                    m.group(1): int(m.group(2))
                    for m in re.finditer(r"([A-Za-z ]+?) (\d+)(?:, |;)", t.split("counted. ", 1)[1])
                }
                (missing,) = [k for k in got if k not in known]
                assert got[missing] == total - sum(known.values()) and all(got[k] == v for k, v in known.items())
                seen["missing"] += 1
            else:
                items = _list_after_colon(t)
                counts = Counter(items)
                if tier == "hard":
                    assert {k: Fraction(counts.get(k, 0)) for k in got} == got
                    assert any(v == 0 for v in got.values())
                    seen["zero"] += 1
                else:
                    assert {k: Fraction(counts[k]) for k in got} == got and set(counts) == set(got)
                    seen["list"] += 1
    assert min(seen.values()) >= 20 and len(seen) == 4, seen


def test_column_chain_difference_and_symbols_follow_through():
    skill = sup.skill_of("construct-column-and-picture-graphs")
    for seed in range(40):
        q = sup.exam("dg_construct_columns_chain", seed)
        counts = [int(v) for v in q.parts[0].answer["heights"]]
        assert int(q.parts[1].answer) == max(counts) - min(counts)
        assert [Fraction(v) for v in q.parts[2].answer["heights"]] == [Fraction(c, 2) for c in counts]
        assert Fraction(str(q.parts[3].answer)) == Fraction(max(counts) - min(counts), 2)
        canon = {p.label: p.answer_text for p in q.parts}
        wrong_b = int(canon["b"]) + 2
        res = grade(
            skill,
            q,
            {
                **canon,
                "b": str(wrong_b),
                "d": str(Fraction(wrong_b, 2)).replace("/", ".") if wrong_b % 2 else str(wrong_b // 2),
            },
        )
        assert res.parts["d"].marks == q.parts[3].marks and res.parts["d"].follow_through is True


# ---------------------------------------------------------------------------------------------- dot plots


def test_dot_plot_counts_match_a_recount_and_the_chain_stats_match_statistics():
    for tier in TIERS:
        for seed in range(100):
            q = sup.single("dg_construct_dotplot", tier, seed)
            counts = Counter(_list_after_colon(q.text))
            reply = _bars_reply(q)
            assert all(counts.get(k, 0) == v for k, v in reply.items()) and set(counts) <= set(reply)
            if tier == "hard":
                assert any(v == 0 for v in reply.values())
            if tier == "medium":
                assert all(re.fullmatch(r"\d+\.\d", k) for k in reply)
    for seed in range(60):
        q = sup.exam("dg_construct_dotplot_chain", seed)
        data = [int(w) for w in _list_after_colon(q.text)]
        assert int(q.parts[1].answer) == statistics.multimode(data)[0]
        assert Fraction(str(q.parts[2].answer)) == Fraction(str(statistics.median(data)))
        assert (
            Fraction(str(q.parts[3].answer)) == Fraction(str(statistics.median(data))) - statistics.multimode(data)[0]
        )
        assert sum(int(v) for v in q.parts[0].answer["heights"]) == len(data)


def test_a_dot_plot_reply_with_one_dot_per_distinct_value_is_wrong_and_diagnosed():
    skill = sup.skill_of("construct-dot-plots-from-data")
    hits = 0
    for seed in range(60):
        q = sup.single("dg_construct_dotplot", "easy", seed)
        for text, tag in q.option_notes.items():
            if tag == "dg-dp-one-dot-per-value":
                res = grade(skill, q, text)
                assert not res.correct and res.misconception == tag
                hits += 1
    assert hits >= 20


# ---------------------------------------------------------------------------------------------- stem-and-leaf


def _rows_from_values(values, decimal):
    rows = {}
    for v in sorted(values):
        if decimal:
            tenths = round(Fraction(v) * 10)
            stem, leaf = divmod(tenths, 10)
        else:
            stem, leaf = divmod(int(v), 10)
        rows.setdefault(stem, []).append(leaf)
    return {str(s): " ".join(map(str, ls)) for s, ls in rows.items()}


def test_stem_and_leaf_rows_match_an_independent_sort_of_the_printed_values():
    seen = Counter()
    for tier in TIERS:
        for seed in range(100):
            q = sup.single("dg_construct_stem", tier, seed)
            raw = _list_after_colon(q.text)
            decimal = any("." in w for w in raw)
            values = [Fraction(w) for w in raw]
            hundred = tier == "hard" and not decimal and min(values) >= 100
            if hundred:
                rows = {}
                for v in sorted(values):
                    rows.setdefault(str(int(v) // 10), []).append(str(int(v) % 10))
                rows = {k: " ".join(ls) for k, ls in rows.items()}
            else:
                rows = _rows_from_values(values, decimal)
            blanks = {q.answer["rows"][int(k.split(",")[0])][0]: v for k, v in q.answer["blanks"].items()}
            assert blanks == rows, (tier, seed)
            assert len(set(blanks)) == len(blanks)
            seen["decimal" if decimal else "hundred" if hundred else "tens"] += 1
            # leaves in a row are increasing (non-decreasing), and a jumbled row is graded wrong
            for leaves in blanks.values():
                ls = leaves.split()
                assert ls == sorted(ls)
    assert min(seen.values()) >= 10 and len(seen) == 3, seen


def test_stem_and_leaf_chain_counts_and_median_match_statistics():
    for seed in range(60):
        q = sup.exam("dg_construct_stem_chain", seed)
        values = [int(w) for w in _list_after_colon(q.text)]
        n = len(values)
        assert int(q.parts[1].answer) == n
        assert Fraction(str(q.parts[2].answer)) == Fraction(n + 1, 2)
        assert Fraction(str(q.parts[3].answer)) == Fraction(str(statistics.median(values)))
        assert sum(len(v.split()) for v in q.parts[0].answer["blanks"].values()) == n
        assert "x y z" in q.parts[0].prompt  # the format hint never gives a real row away


def test_the_stem_and_leaf_grader_wants_ordered_leaves():
    skill = sup.skill_of("construct-stem-and-leaf-plots")
    q = sup.single("dg_construct_stem", "medium", 3)
    assert grade(skill, q, q.answer_text).correct
    key = q.answer["blanks"]
    first = sorted(key, key=lambda k: tuple(map(int, k.split(","))))[0]
    leaves = key[first].split()
    assert len(leaves) >= 2 and leaves != leaves[::-1]
    swapped = q.answer_text.replace(f"{key[first]}", " ".join(leaves[::-1]), 1)
    assert not grade(skill, q, swapped).correct


# ---------------------------------------------------------------------------------------------- compare displays


def _compare_data(q):
    a = [d["at"] for d in q.figure["dots"] for _ in range(d["count"])]
    body = q.text.split("means", 1)[1].split(". ", 1)[1].split(". ", 1)[0].rstrip(".")
    b = []
    for row in body.split("; "):
        stem, _, leaves = row.partition("|")
        b += [int(stem) * 10 + int(x) for x in leaves.split()]
    return a, b


def test_compare_displays_keys_match_statistics_on_both_displays():
    seen = Counter()
    for tier in TIERS:
        for seed in range(100):
            q = sup.single("dg_compare_displays", tier, seed)
            a, b = _compare_data(q)
            t = q.text
            med = lambda v: Fraction(str(statistics.median(v)))  # noqa: E731
            if "mode of" in t:
                assert q.answer == abs(statistics.multimode(a)[0] - statistics.multimode(b)[0])
                seen["mode"] += 1
            elif "range of all" in t:
                assert q.answer == max(a + b) - min(a + b)
                seen["combined"] += 1
            elif "range of" in t:
                assert q.answer == abs((max(a) - min(a)) - (max(b) - min(b)))
                seen["range"] += 1
            elif "How many more values" in t:
                assert q.answer == abs(len(a) - len(b))
                seen["count"] += 1
            elif "By how much" in t:
                assert Fraction(str(q.answer)) == abs(med(a) - med(b))
                seen["median"] += 1
            elif "greater than the median" in t:
                assert q.answer == sum(v > med(a) for v in b)
                seen["above"] += 1
            else:
                assert "less than the median" in t
                assert q.answer == sum(v < med(b) for v in a)
                seen["below"] += 1
    assert min(seen.values()) >= 8 and len(seen) == 7, seen


def test_compare_chain_follows_through_from_both_medians():
    skill = sup.skill_of("compare-two-data-displays")
    checked = 0
    for seed in range(40):
        q = sup.exam("dg_compare_displays_chain", seed)
        a, b = _compare_data(q)
        ma, mb = Fraction(str(statistics.median(a))), Fraction(str(statistics.median(b)))
        assert Fraction(str(q.parts[0].answer)) == ma and Fraction(str(q.parts[1].answer)) == mb
        assert Fraction(str(q.parts[2].answer)) == mb - ma > 0
        canon = {p.label: p.answer_text for p in q.parts}
        wrong_a = ma - 1 if mb - ma < 2 else ma + 1
        if mb - wrong_a == mb - ma:
            continue
        res = grade(skill, q, {**canon, "a": str(wrong_a), "c": str(mb - wrong_a)})
        assert res.parts["c"].marks == q.parts[2].marks and res.parts["c"].follow_through is True, (
            seed,
            res.parts["c"],
        )
        checked += 1
    assert checked >= 30


def test_compare_figure_never_shows_the_stem_plot_or_the_answer():
    for seed in range(40):
        q = sup.single("dg_compare_displays", "medium", seed)
        assert q.figure["kind"] == "dot_plot" and "|" not in str(q.figure)
        assert "|" in q.text  # the second display is written in the question


# ---------------------------------------------------------------------------------------------- real-data graphs


def _segments(fig):
    return [(tuple(s["from"]), tuple(s["to"])) for s in fig["segments"]]


def _sym_line(fig):
    """The line through the first and last drawn points, as a SymPy expression y(x), by SymPy."""
    (x0, y0), (x1, y1) = _segments(fig)[0][0], _segments(fig)[-1][1]
    x = sympy.Symbol("x")
    return x, sympy.Rational(y0) + sympy.Rational(y1 - y0, x1 - x0) * (x - x0)


def _nums(text):
    return [sympy.Rational(m) for m in re.findall(r"\d+(?:\.\d+)?", text)]


def _is_journey(fig):
    return len(fig["segments"]) == 3


def test_line_graph_readings_match_a_sympy_line_through_the_drawn_ends():
    seen = Counter()
    for tier in ("easy", "medium", "hard"):
        for seed in range(150):
            q = sup.single("dg_real_data_graphs", tier, seed)
            fig, t = q.figure, q.text
            assert fig["equal_scale"] is False
            ans = sympy.Rational(str(q.answer))
            if _is_journey(fig):
                (t0, d0), (t1, d1), (t2, d2), (t3, d3) = [s[0] for s in _segments(fig)] + [_segments(fig)[-1][1]]
                if "average speed" in t:
                    assert ans == sympy.Rational(d3, t3)
                    seen["average"] += 1
                elif "stop" in t:
                    assert ans == t2 - t1 and d2 == d1
                    seen["stopped"] += 1
                else:
                    ta, tb = _nums(t)[-2:]
                    pts = {t0: d0, t1: d1, t2: d2, t3: d3}
                    assert ans == sympy.Rational(pts[int(tb)] - pts[int(ta)]) / (tb - ta)
                    seen["speed"] += 1
                continue
            x, f = _sym_line(fig)
            xs = [int(v) for v in _nums(t) if v == int(v)]
            if "fixed charge" in t or "deposit" in t or "call-out fee" in t or "at the start of the trip" in t:
                assert ans == f.subs(x, 0)
                seen["fixed"] += 1
            elif "graph stops at" in t:
                X = xs[-1]
                assert ans == f.subs(x, X)
                seen["extra"] += 1
            elif re.search(
                r"for every 100 km|go up for each|per kilogram|1 euro worth|speed, in kilometres per hour", t
            ):
                per = 100 if "100 km" in t else 1
                assert ans == abs(sympy.diff(f, x)) * per
                seen["rate"] += 1
            elif (
                re.search(
                    r"can be bought for|are worth|can the car drive|can you travel|can the bike|has the car driven|hours has the cyclist|hours of work can",
                    t,
                )
                or "For how many hours" in t
            ):
                y = _nums(t)[-1]
                (sol,) = sympy.solve(sympy.Eq(f, y), x)
                assert ans == sol
                seen["inverse"] += 1
            else:
                X = _nums(t)[0] if tier != "x" else None
                assert ans == f.subs(x, xs[0]), (t, ans)
                seen["read"] += 1
    assert len(seen) >= 7 and min(seen.values()) >= 5, seen


def test_real_data_chain_follows_through_from_charge_and_rate():
    skill = sup.skill_of("read-values-and-rates-from-real-data-graphs")
    for seed in range(40):
        q = sup.exam("dg_real_data_graphs_chain", seed)
        a, b = int(q.parts[0].answer), int(q.parts[1].answer)
        X = int(
            next(
                g
                for g in re.search(
                    r"for a (\d+) km trip|hire the bike for (\d+) hours|charge for (\d+) hours", q.parts[2].prompt
                ).groups()
                if g
            )
        )
        assert int(q.parts[2].answer) == a + b * X
        budget = int(re.search(r"\$(\d+)", q.parts[3].prompt).group(1))
        assert int(q.parts[3].answer) * b + a == budget
        canon = {p.label: p.answer_text for p in q.parts}
        res = grade(skill, q, {**canon, "a": str(a + 2), "c": str(a + 2 + b * X)})
        assert res.parts["c"].marks == q.parts[2].marks and res.parts["c"].follow_through is True
        # a wrong starting charge carried into (d) is followed too, when the learner divides what is left by the right rate
        if (budget - a - b) % b == 0 and budget - a - b > 0 and (budget - a - b) // b not in (a + b, b):
            res = grade(skill, q, {**canon, "a": str(a + b), "d": str((budget - a - b) // b)})
            assert res.parts["d"].marks == q.parts[3].marks and res.parts["d"].follow_through is True, (
                seed,
                res.parts["d"],
            )


def test_every_drawn_point_of_a_real_data_graph_is_on_a_gridline_crossing():
    for tier in TIERS:
        for seed in range(60):
            fig = sup.single("dg_real_data_graphs", tier, seed).figure
            for (x0, y0), (x1, y1) in _segments(fig):
                for x, y in ((x0, y0), (x1, y1)):
                    assert x % fig["x_grid"] == 0 and y % fig["y_grid"] == 0, (tier, seed, x, y)


# ---------------------------------------------------------------------------------------------- grid map


def _landmarks(fig):
    names = {
        "house": "house", "school": "school", "church": "church", "hospital": "hospital", "bridge": "bridge",
        "campground": "campground", "lookout": "lookout", "post_office": "post office",
        "railway_station": "railway station", "lighthouse": "lighthouse", "picnic_area": "picnic area",
    }  # fmt: skip
    return {names[s["symbol"]]: (int(s["at"][0]), int(s["at"][1])) for s in fig["symbols"]}


def _ref(c, r):
    return "ABCDEFGHIJ"[c] + str(r + 1)


def _cell(ref):
    return "ABCDEFGHIJ".index(ref[0]), int(ref[1:]) - 1


def _text_cells(text):
    return {tuple(map(int, m)) for m in re.findall(r"\((\d+), (\d+)\)", text)}


def test_grid_map_keys_match_a_fresh_parse_of_the_map_and_the_question():
    seen = Counter()
    for tier in TIERS:
        for seed in range(120):
            q = sup.single("dg_grid_map", tier, seed)
            fig, t = q.figure, q.text
            lm = _landmarks(fig)
            got = {tuple(c) for c in q.answer["cells"]}
            assert q.answer["match"] == "exact" and q.figure["canvas"] == "grid"
            assert fig["column_labels"][0] == "A" and fig["row_labels"][0] == "1"  # rows count from the bottom
            if m := re.search(r"corners ([A-J]\d+) and ([A-J]\d+)", t):
                (c0, r0), (c1, r1) = _cell(m.group(1)), _cell(m.group(2))
                assert got == {(c, r) for c in range(c0, c1 + 1) for r in range(r0, r1 + 1)}
                seen["region"] += 1
            elif "between the two landmarks" in t:
                m = re.match(r"The (.+?) and the (.+?) are in the same (row|column)", t)
                (ac, ar), (bc, br) = lm[m.group(1)], lm[m.group(2)]
                if m.group(3) == "row":
                    assert ar == br and got == {(c, ar) for c in range(min(ac, bc) + 1, max(ac, bc))}
                else:
                    assert ac == bc and got == {(ac, r) for r in range(min(ar, br) + 1, max(ar, br))}
                assert not got & set(lm.values())
                seen["between"] += 1
            elif m := re.search(r"([A-J]\d+)", t):
                assert got == {_cell(m.group(1))}
                seen["reference"] += 1
            else:
                move = re.search(
                    r"((?:\d+ squares? (?:east|west|north|south))(?: and \d+ squares? (?:east|west|north|south))?)", t
                ).group(1)
                name = next(n for n in lm if f"the {n}" in t)
                c, r = lm[name]
                for k, way in re.findall(r"(\d+) squares? (east|west|north|south)", move):
                    c += int(k) * {"east": 1, "west": -1}.get(way, 0)
                    r += int(k) * {"north": 1, "south": -1}.get(way, 0)
                assert got == {(c, r)}
                seen["move"] += 1
    assert min(seen.values()) >= 15 and len(seen) == 4, seen


def test_grid_map_chain_counts_and_follow_through():
    skill = sup.skill_of("place-features-on-a-grid-map")
    for seed in range(40):
        q = sup.exam("dg_grid_map_chain", seed)
        lm = _landmarks(q.figure)
        names = re.findall(r"The (.+?) is in square ([A-J]\d+), and the (.+?) is in square ([A-J]\d+)", q.text)[0]
        (xc, xr), (yc, yr) = _cell(names[1]), _cell(names[3])
        assert lm[names[0]] == (xc, xr) and lm[names[2]] == (yc, yr)
        assert int(q.parts[1].answer) == abs(yc - xc) and int(q.parts[2].answer) == abs(yr - xr)
        assert int(q.parts[3].answer) == abs(yc - xc) + abs(yr - xr)
        ref = re.search(r"square ([A-J]\d+)", q.parts[0].prompt).group(1)
        assert {tuple(c) for c in q.parts[0].answer["cells"]} == {_cell(ref)}
        canon = {p.label: p.answer_text for p in q.parts}
        res = grade(skill, q, {**canon, "b": str(int(canon["b"]) + 1), "d": str(int(canon["b"]) + 1 + int(canon["c"]))})
        assert res.parts["d"].marks == q.parts[3].marks and res.parts["d"].follow_through is True


def test_a_reversed_grid_reference_is_diagnosed():
    skill = sup.skill_of("place-features-on-a-grid-map")
    hits = 0
    for seed in range(80):
        q = sup.single("dg_grid_map", "easy", seed)
        for text, tag in q.option_notes.items():
            if tag == "dg-gm-reference-reversed":
                res = grade(skill, q, text)
                assert not res.correct and res.misconception == tag
                hits += 1
    assert hits >= 10


def test_the_figure_never_holds_the_answer_cells():
    for tier in TIERS:
        for seed in range(40):
            q = sup.single("dg_grid_map", tier, seed)
            assert "cells" not in q.figure  # nothing is pre-shaded: the answer is the learner's to draw
            assert set(q.figure) <= {
                "kind",
                "alt",
                "grid",
                "symbols",
                "legend",
                "column_labels",
                "row_labels",
                "north",
                "canvas",
            }


# ---------------------------------------------------------------------------------------------- a note on the visual gate


def test_make_is_the_only_entry_the_generators_need():
    # the topics are registered by importing the package, with no extra wiring
    for _sid, topic, chain, _kind in SKILLS:
        assert make(topic, seed=0, difficulty="easy").text
        assert make(chain, seed=0, difficulty="exam").parts


# --- review fixes: articles, a guessable stop, ----------------------------------------------------------------


def test_prompts_use_a_and_an_correctly_in_the_grid_map_and_real_data_graphs():
    a_before_vowel = re.compile(r"\ba (?:[aeiou]\w|8\d*\b|11\b|18\b)")
    an_before_consonant = re.compile(r"\ban (?:[b-df-hj-np-tv-z]\w|[0-79]\d*\b)(?<!\b11)(?<!\b18)")
    for topic in ("dg_grid_map", "dg_real_data_graphs"):
        for tier in ("easy", "medium", "hard"):
            for seed in range(250):
                q = make(topic, seed=seed, difficulty=tier)
                for t in [q.text, *q.working, *(p.prompt for p in q.parts)]:
                    assert not a_before_vowel.search(t), (topic, tier, seed, t)
                    assert not an_before_consonant.search(t), (topic, tier, seed, t)


def test_the_journey_stop_key_has_no_fixed_rank_among_the_options():
    keys, ranks = Counter(), Counter()
    for seed in range(1500):
        q = make("dg_real_data_graphs", seed=seed, difficulty="hard")
        if "For how many hours did the cyclist stop" not in q.text:
            continue
        keys[float(q.answer)] += 1
        ranks[sum(float(d) < float(q.answer) for d in q.distractors)] += 1
    total = sum(ranks.values())
    assert set(keys) >= {2.0, 3.0, 4.0, 5.0}, keys
    assert set(ranks) >= {0, 1, 2} and max(ranks.values()) / total <= 0.5, ranks
