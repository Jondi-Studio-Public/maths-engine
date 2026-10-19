"""The `network` figure kind (vertices, straight edges, labels, highlights), its renderer and its layouts, and the
network questions that now carry the picture of the network their text prints."""

from __future__ import annotations

import copy
import re

import pytest
from snet_support import network_from_text

from mathgen import figures, make, svg
from mathgen.figure_kinds import network as net
from mathgen.generators import _network_fig as fig


def _hexagon(**over):
    pos = net.circle_layout(6)
    verts = [(c, *pos[i]) for i, c in enumerate("ABCDEF")]
    edges = [
        {"from": "A", "to": "B", "label": "4"},
        {"from": "B", "to": "C"},
        {"from": "C", "to": "D", "directed": True},
    ]
    return dict(alt="A network of six vertices.", vertices=verts, edges=edges, **over)


def test_kind_is_registered_and_renderable():
    assert "network" in figures.kinds() and "network" in svg.rendered_kinds()


def test_a_valid_network_renders_deterministically_and_shows_only_its_texts():
    spec = net.network(**_hexagon(highlight_vertices=["A"]))
    assert figures.validate(spec) == []
    assert set(figures.labels(spec)) >= {"A", "B", "C", "D", "E", "F", "4"}
    one, two = svg.figure_svg(spec), svg.figure_svg(copy.deepcopy(spec))
    assert one == two and one.startswith("<svg") and "<polygon" in one  # the arrowhead


def test_highlight_makes_an_edge_thicker():
    plain = net.network(**_hexagon())
    hot = copy.deepcopy(plain)
    hot["edges"][0]["highlight"] = True
    assert svg.figure_svg(hot) != svg.figure_svg(plain)
    assert 'stroke-width="4.5"' in svg.figure_svg(hot)


@pytest.mark.parametrize(
    "mutate,message",
    [
        (lambda s: s["edges"].append({"from": "A", "to": "Z"}), "unknown vertices"),
        (lambda s: s["edges"].append({"from": "A", "to": "A"}), "loop"),
        (lambda s: s["edges"].append({"from": "B", "to": "A"}), "second edge"),
        (lambda s: s["vertices"][1].update(id="A"), "repeated"),
        (
            lambda s: s["vertices"][1].update(at=[s["vertices"][0]["at"][0] + 2, s["vertices"][0]["at"][1]]),
            "closer than",
        ),
        (lambda s: s["vertices"][1].update(at=[500, 5]), "inside the frame"),
        (lambda s: s["edges"][0].update(label="a very long label"), "label"),
        (lambda s: s["edges"][0].update(label_t=0.95), "label_t"),
        (lambda s: s.update(highlight_vertices=["Q"]), "unknown vertex"),
    ],
)
def test_invalid_networks_are_rejected(mutate, message):
    spec = net.network(**_hexagon())
    mutate(spec)
    problems = figures.validate(spec)
    assert any(message in p for p in problems), problems


def test_an_edge_may_not_pass_through_a_vertex_that_is_not_its_end():
    spec = net.network(**_hexagon())
    spec["edges"].append({"from": "A", "to": "D"})  # a hexagon's diameter is clear of the other vertices
    assert figures.validate(spec) == []
    spec["vertices"][2]["at"] = [50.0, 51.0]  # move C onto that diameter
    assert any("passes through vertex C" in p for p in figures.validate(spec))


def test_a_pair_of_opposite_directed_edges_is_allowed_but_not_an_undirected_repeat():
    spec = net.network(**_hexagon())
    spec["edges"] = [{"from": "A", "to": "B", "directed": True}, {"from": "B", "to": "A", "directed": True}]
    assert figures.validate(spec) == []


def test_plane_true_rejects_crossings_and_accepts_a_planar_drawing():
    pos = net.circle_layout(4)
    verts = [(c, *pos[i]) for i, c in enumerate("ABCD")]
    ring = [("A", "B"), ("B", "C"), ("C", "D"), ("D", "A")]
    net.network(alt="A square.", vertices=verts, edges=ring + [("A", "C")], plane=True)
    with pytest.raises(ValueError, match="cross"):
        net.network(alt="A square.", vertices=verts, edges=ring + [("A", "C"), ("B", "D")], plane=True)
    net.network(alt="A square.", vertices=verts, edges=ring + [("A", "C"), ("B", "D")], plane=False)


def test_named_side_vertices_allow_long_names_and_a_bipartite_layout():
    left, right = net.two_row_layout(3, 3)
    verts = [(w, *p, "left") for w, p in zip(("Ana", "Ben", "Cara"), left)]
    verts += [(j, *p, "right") for j, p in zip(("plumbing", "carpentry", "microscopes"), right)]
    spec = net.network(alt="Workers and jobs.", vertices=verts, edges=[("Ana", "plumbing"), ("Cara", "microscopes")])
    assert "microscopes" in svg.figure_svg(spec)
    with pytest.raises(ValueError, match="1 to 8"):
        net.network(alt="x", vertices=[("microscopes", 20, 20), ("B", 70, 70)], edges=[])


def test_place_labels_keeps_labels_of_crossing_edges_apart():
    pos = net.circle_layout(6)
    names = "ABCDEF"
    verts = {c: tuple(pos[i]) for i, c in enumerate(names)}
    edges = [{"from": names[i], "to": names[(i + 3) % 6], "label": str(i)} for i in range(3)]  # three diameters
    net.place_labels(verts, edges)
    spots = [net.label_point(verts[e["from"]], verts[e["to"]], e["label_t"]) for e in edges]
    assert all(
        ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5 >= net.MIN_LABEL_GAP
        for i, a in enumerate(spots)
        for b in spots[i + 1 :]
    )


def test_best_circle_layout_finds_a_planar_order_when_there_is_one():
    ring_with_chord = [(0, 2), (2, 4), (4, 1), (1, 3), (3, 0), (0, 4)]  # a 5-cycle scrambled, plus a chord
    pos = net.best_circle_layout(5, ring_with_chord)
    assert net.crossing_count({i: pos[i] for i in range(5)}, ring_with_chord) == 0
    assert net.best_circle_layout(5, ring_with_chord) == pos  # deterministic


def test_layered_layout_puts_source_left_sink_right_and_clears_vertices():
    names = ["S", "A", "B", "C", "T"]
    arcs = [("S", "A"), ("S", "B"), ("A", "C"), ("B", "A"), ("B", "T"), ("C", "T")]
    pos = dict(zip(names, fig.layered_layout(names, arcs)))
    assert pos["S"][0] < pos["A"][0] < pos["T"][0] and pos["S"][0] == min(p[0] for p in pos.values())
    assert pos["T"][0] == max(p[0] for p in pos.values())
    spec = fig.graph_figure(
        5, [(0, 1), (0, 2), (1, 3), (2, 1), (2, 4), (3, 4)], names=names, positions=list(pos.values()), directed=True
    )
    assert figures.validate(spec) == []


# --------------------------------------------------------------- the network questions carry the picture

_DRAWN = [
    ("snet_eulerian", "easy"),
    ("snet_eulerian", "medium"),
    ("snet_eulerian_chain", "exam"),
    ("snet_hamiltonian", "easy"),
    ("snet_hamiltonian", "medium"),
    ("snet_hamiltonian", "hard"),
    ("snet_hamiltonian_chain", "exam"),
    ("snet_route_types", "easy"),
    ("snet_route_types", "hard"),
    ("snet_route_types_chain", "exam"),
    ("snet_adjacency_chain", "exam"),
    ("shortest_path", None),
    ("exd_x_r3a_connector_reroute", None),
    ("exd_x_sc2_network_closure", "exam"),
    ("use_minimal_spanning_trees_to_solve_minimal_connector", "medium"),
    ("solve_small_scale_network_flow_problems_including_the", "medium"),
    ("w7a_network_cut", "easy"),
    ("exd_x_w7a_network_cut", None),
]


@pytest.mark.parametrize("topic,difficulty", _DRAWN)
def test_the_figure_is_the_network_the_text_prints(topic, difficulty):
    n_drawn = 0
    for seed in range(1, 13):
        q = make(topic, seed=seed, difficulty=difficulty)
        if q.figure is None:
            # A figure is dropped only when a printed weight equals an answer (the leak gate), on every redraw.
            with pytest.raises(AssertionError):
                figures.assert_no_leak(fig.figure_from_text(q.text), [q.answer_text, *(p.answer_text for p in q.parts)])
            continue
        n_drawn += 1
        assert figures.validate(q.figure) == []
        names, arcs, directed = fig.network_from_text(q.text)
        drawn = {(e["from"], e["to"]) if directed else tuple(sorted((e["from"], e["to"]))) for e in q.figure["edges"]}
        printed = {(a, b) if directed else tuple(sorted((a, b))) for a, b, _ in arcs}
        assert drawn == printed, (topic, seed)
        assert {v["id"] for v in q.figure["vertices"]} == set(names)
        weights = {(e["from"], e["to"]): e.get("label") for e in q.figure["edges"]}
        for a, b, w in arcs:
            if w is not None and any(weights.values()):
                assert (weights.get((a, b)) or weights.get((b, a))) == w, (topic, seed, a, b)
        svg.figure_svg(q.figure)
    assert n_drawn >= 4, (topic, n_drawn)


def test_unweighted_text_networks_read_back_to_the_same_edges_as_the_oracle_helper():
    for seed in range(1, 25):
        q = make("snet_eulerian", seed=seed, difficulty="easy")
        n, edges = network_from_text(q.text)
        drawn = sorted(tuple(sorted("ABCDEFGH".index(x) for x in (e["from"], e["to"]))) for e in q.figure["edges"])
        assert drawn == edges and len(q.figure["vertices"]) == n


def test_questions_that_print_no_single_network_get_no_figure():
    assert make("snet_eulerian", seed=3, difficulty="hard").figure is None  # four networks
    assert make("snet_adjacency", seed=3, difficulty="easy").figure is None  # a matrix
    assert make("sr2a_network_model", seed=3, difficulty="easy").figure is None  # a situation in words


def test_planar_faces_easy_draws_a_plane_graph_with_no_crossings():
    for seed in range(1, 41):
        q = make("sr2a_planar_faces", seed=seed, difficulty="easy")
        spec = q.figure
        assert spec["plane"] is True and figures.validate(spec) == []
        pos = {v["id"]: v["at"] for v in spec["vertices"]}
        assert net.crossing_count(pos, [(e["from"], e["to"]) for e in spec["edges"]]) == 0
        listed = re.search(r"are: ((?:[A-I]{2}(?:, )?)+)\.", q.text).group(1).split(", ")
        assert sorted(tuple(sorted(p)) for p in listed) == sorted(
            tuple(sorted((e["from"], e["to"]))) for e in spec["edges"]
        )


@pytest.mark.parametrize("tier", ["easy", "hard"])
def test_bipartite_matching_draws_each_can_do_line(tier):
    for seed in range(1, 21):
        q = make("sr2a_bipartite", seed=seed, difficulty=tier)
        spec = q.figure
        assert figures.validate(spec) == []
        sides = {v["id"]: v["side"] for v in spec["vertices"]}
        assert all(sides[e["from"]] == "left" and sides[e["to"]] == "right" for e in spec["edges"])
        for e in spec["edges"]:
            assert re.search(rf"{e['from']} can [^.]*\b{e['to']}\b", q.text), (seed, e)


@pytest.mark.parametrize(
    "topic,difficulty",
    [("shortest_path", None), ("exd_x_sc2_network_closure", "exam"), ("exd_x_r3a_connector_reroute", None)],
)
def test_these_weighted_questions_are_never_without_their_figure(topic, difficulty):
    for seed in range(1, 41):
        q = make(topic, seed=seed, difficulty=difficulty)
        assert q.figure is not None, (topic, seed)
        printed = {w for _, _, w in fig.network_from_text(q.text)[1]}
        labels = {e.get("label") for e in q.figure["edges"]}
        # Drawn with its weights, or without them when an answer equals one of the printed weights.
        assert labels == printed or labels == {None}, (topic, seed)
        if topic == "exd_x_r3a_connector_reroute":
            assert labels == {None}  # its answers include a printed cable cost, so the drawing carries no costs
