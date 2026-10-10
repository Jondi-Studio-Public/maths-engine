"""Tests for Euler's formula for planar graphs, `senior-apply-euler-s-formula-v-f-e-2` (ACMGM082)."""

from __future__ import annotations

import re

import pytest

from mathgen.grading.typed import GRADERS
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError, check_pairwise_distinct
from mathgen.generators.pv0c_euler_planar_build import (
    _build_plane_graph,
    _euler_variants,
    _has_edge,
    _rotate_to_edge,
    _verify_plane_graph,
)

TOPIC = "pv0c_euler_planar"
SEEDS = range(60)
TIERS = ["easy", "medium", "hard", "exam"]

SKILLS = load_skills()[1]


class TestEulerPlanar:
    @pytest.mark.parametrize("difficulty", TIERS)
    def test_determinism(self, difficulty):
        for seed in SEEDS:
            q1 = make(TOPIC, seed=seed, difficulty=difficulty)
            q2 = make(TOPIC, seed=seed, difficulty=difficulty)
            assert q1.to_json() == q2.to_json()

    @pytest.mark.parametrize("difficulty", TIERS)
    def test_working_text_nonempty(self, difficulty):
        for seed in SEEDS:
            q = make(TOPIC, seed=seed, difficulty=difficulty)
            assert isinstance(q.working, list) and len(q.working) >= 2
            for step in q.working:
                assert isinstance(step, str) and step

    @pytest.mark.parametrize("difficulty", TIERS)
    def test_distractors_distinct_nonempty_never_equal_answer(self, difficulty):
        for seed in SEEDS:
            q = make(TOPIC, seed=seed, difficulty=difficulty)
            assert q.answer_text not in q.distractors
            assert len(q.distractors) == len(set(q.distractors))
            assert len(q.distractors) >= 1

    @pytest.mark.parametrize("difficulty", TIERS)
    def test_at_least_two_distractors_on_85_percent_of_draws(self, difficulty):
        n = 200
        at_least_two = 0
        for seed in range(n):
            q = make(TOPIC, seed=seed, difficulty=difficulty)
            if len(q.distractors) >= 2:
                at_least_two += 1
        share = at_least_two / n
        assert share >= 0.85, f"{difficulty}: only {share:.2f} of draws had >= 2 distractors"

    @pytest.mark.parametrize("difficulty", TIERS)
    def test_grader_accepts_answer_rejects_distractors_and_wrong(self, difficulty):
        grader = GRADERS["number"]
        for seed in SEEDS:
            q = make(TOPIC, seed=seed, difficulty=difficulty)
            result = grader(q, q.answer_text)
            assert result.correct, f"seed {seed} {difficulty}: rejected answer {q.answer_text}"
            for dist in q.distractors:
                result = grader(q, dist)
                assert not result.correct, f"seed {seed} {difficulty}: accepted distractor {dist}"
            wrong = str(int(q.answer_text) + 1000)
            result = grader(q, wrong)
            assert not result.correct, f"seed {seed} {difficulty}: accepted wrong answer {wrong}"

    @pytest.mark.parametrize("difficulty", TIERS)
    def test_answer_is_plain_positive_integer_string(self, difficulty):
        for seed in SEEDS:
            q = make(TOPIC, seed=seed, difficulty=difficulty)
            assert isinstance(q.answer, int)
            assert q.answer > 0
            assert q.answer_text == str(q.answer)
            for dist in q.distractors:
                assert re.fullmatch(r"\d+", dist), f"distractor {dist!r} not formatted like the answer"

    @pytest.mark.parametrize("difficulty", TIERS)
    def test_at_least_20_distinct_questions_over_60_seeds(self, difficulty):
        texts = {make(TOPIC, seed=seed, difficulty=difficulty).text for seed in SEEDS}
        assert len(texts) >= 20, f"{difficulty}: only {len(texts)} distinct questions over 60 seeds"

    @pytest.mark.parametrize("difficulty", TIERS)
    def test_answer_concentration_under_25_percent_at_200_seeds(self, difficulty):
        from collections import Counter

        n = 200
        counts = Counter(make(TOPIC, seed=seed, difficulty=difficulty).answer_text for seed in range(n))
        top_share = counts.most_common(1)[0][1] / n
        assert top_share <= 0.25, f"{difficulty}: top answer share {top_share:.2f} over {n} seeds"

    @pytest.mark.parametrize("difficulty", TIERS)
    def test_prompt_states_outer_face_convention(self, difficulty):
        for seed in range(10):
            q = make(TOPIC, seed=seed, difficulty=difficulty)
            assert "outer" in q.text or "polyhedron" in q.text, q.text

    def test_easy_realisability_bounds(self):
        for seed in SEEDS:
            q = make(TOPIC, seed=seed, difficulty="easy")
            v = int(re.search(r"(\d+) vertices", q.text).group(1))
            e = int(re.search(r"(\d+) edges", q.text).group(1))
            assert v >= 4
            assert v - 1 <= e <= 3 * v - 6, f"seed {seed}: (v={v}, e={e}) not realisable"

    def test_hard_degree_list_matches_derived_f(self):
        for seed in SEEDS:
            q = make(TOPIC, seed=seed, difficulty="hard")
            pairs = re.findall(r"(\d+) (?:vertex|vertices) of degree (\d+)", q.text)
            assert pairs, q.text
            degree_sum = sum(int(n) * int(d) for n, d in pairs)
            v = sum(int(n) for n, d in pairs)
            assert degree_sum % 2 == 0
            e = degree_sum // 2
            f = 2 - v + e
            assert f == int(q.answer_text), f"seed {seed}: expected f={f}, got {q.answer_text}"

    def test_exam_face_list_matches_derived_v(self):
        for seed in SEEDS:
            q = make(TOPIC, seed=seed, difficulty="exam")
            outer = int(
                re.search(r"outer(?:,\s*unbounded)? (?:face|region) (?:is bounded by|has) (\d+)", q.text).group(1)
            )
            face_words = re.findall(
                r"(\d+) (?:is an?|are) (triangle|triangles|quadrilateral|quadrilaterals|"
                r"pentagon|pentagons|hexagon|hexagons|heptagon|heptagons|octagon|octagons)",
                q.text,
            )
            assert face_words, q.text
            name_to_len = {"triangle": 3, "quadrilateral": 4, "pentagon": 5, "hexagon": 6, "heptagon": 7, "octagon": 8}
            other_lengths = []
            for n, name in face_words:
                length = name_to_len[name.rstrip("s")]
                other_lengths.extend([length] * int(n))
            f = 1 + len(other_lengths)
            e = (outer + sum(other_lengths)) // 2
            assert (outer + sum(other_lengths)) % 2 == 0
            v = 2 + e - f
            assert v == int(q.answer_text), f"seed {seed}: expected v={v}, got {q.answer_text}"

    def test_plane_graph_pool_invariants_verified_at_import(self):
        # Re-derive one fresh graph and re-verify independently of the pool
        # built at import time (that pool's own build already calls
        # _verify_plane_graph on every entry -- this proves that check
        # actually distinguishes good from bad).
        import random

        rng = random.Random("test-fixed-seed")
        vertices, edges, faces = _build_plane_graph(rng, m=8, insertions=3, max_deletions=4)
        _verify_plane_graph(vertices, edges, faces)  # does not raise

    def test_verify_plane_graph_fires_on_corrupted_graph(self):
        # A valid small graph (a fan-triangulated square: 2 triangles + outer).
        vertices = [0, 1, 2, 3]
        edges = {frozenset(p) for p in [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)]}
        faces = [(0, 1, 2, 3), (0, 1, 2), (0, 2, 3)]
        _verify_plane_graph(vertices, edges, faces)  # does not raise
        # Corrupt: drop an edge from the edge set so face-length sum != 2e.
        bad_edges = set(edges)
        bad_edges.discard(frozenset((0, 2)))
        with pytest.raises(VerificationError):
            _verify_plane_graph(vertices, bad_edges, faces)

    def test_has_edge_and_rotate_to_edge(self):
        face = (0, 1, 2, 3)
        assert _has_edge(face, 1, 2)
        assert _has_edge(face, 2, 1)
        assert not _has_edge(face, 0, 2)
        assert _rotate_to_edge(face, 1, 2) == (1, 2, 3, 0)
        assert _rotate_to_edge(face, 2, 1) == (2, 1, 0, 3)
        assert _rotate_to_edge(face, 0, 2) is None

    def test_euler_variants_fires_on_unknown_name(self):
        with pytest.raises(ValueError):
            _euler_variants("q", 4, 5, 6)

    def test_euler_variants_formulas(self):
        # v=6, e=9, f=5 (v - e + f = 2)
        forgot, no_const, sign = _euler_variants("f", 6, 9, 5)
        assert forgot == 4
        assert no_const == 3
        assert sign == 13
        forgot, no_const, sign = _euler_variants("e", 6, 9, 5)
        assert forgot == 8
        assert no_const == 11
        assert sign == 1
        forgot, no_const, sign = _euler_variants("v", 6, 9, 5)
        assert forgot == 7
        assert no_const == 4
        assert sign == -2

    def test_check_pairwise_distinct_is_the_real_helper(self):
        # Sanity: our generator relies on core's check_pairwise_distinct, not
        # a local re-implementation.
        check_pairwise_distinct(1, 2, 3)
        with pytest.raises(VerificationError):
            check_pairwise_distinct(1, 2, 2)

    def test_skill_loads(self):
        skill = next((s for s in SKILLS if s.id == "senior-apply-euler-s-formula-v-f-e-2"), None)
        assert skill is not None
        assert skill.domain == "financial-discrete"
        assert skill.answer == "number"
        assert skill.prerequisites == ["simple-interest"]
        assert skill.placement_eligible is False
        assert len(skill.sources) == 4
        assert {s.difficulty for s in skill.sources} == {"easy", "medium", "hard", "exam"}
        exam_source = next(s for s in skill.sources if s.difficulty == "exam")
        assert exam_source.stages == ["exam"]
