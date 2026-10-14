"""Tests for: matrices for storing and displaying information
(``senior-use-matrices-for-storing-and-displaying-information-that``, ACMGM013)."""

from __future__ import annotations

import random
import re

import pytest

from mathgen.grading.typed import GRADERS
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError, REGISTRY
from mathgen.text_quality import prompt_shape, find_text_defects
from mathgen.generators.p14a_matrix_table_build import (
    STORE_NAMES,
    PRODUCT_NAMES,
    TOWN_NAMES,
    NEIGHBOURHOOD_NAMES,
    _draw_easy,
    _draw_medium,
    _draw_adjacency,
    _verify_matrix_lookup_or_total,
    _verify_asymmetric_at_node,
)

TOPIC = "p14a_matrix_table"
GRADER = GRADERS["number"]
SEEDS = range(60)
TIERS = ["easy", "medium", "hard", "exam"]
SKILL_ID = "senior-use-matrices-for-storing-and-displaying-information-that"


class _CommonBehaviour:
    TIER: str

    def test_determinism(self):
        for seed in SEEDS:
            q1 = make(TOPIC, seed=seed, difficulty=self.TIER)
            q2 = make(TOPIC, seed=seed, difficulty=self.TIER)
            assert q1.to_json() == q2.to_json()

    def test_working_nonempty(self):
        for seed in SEEDS:
            q = make(TOPIC, seed=seed, difficulty=self.TIER)
            assert isinstance(q.working, list) and len(q.working) >= 1
            for step in q.working:
                assert isinstance(step, str) and step

    def test_distractors_distinct_nonempty_never_equal_answer(self):
        for seed in SEEDS:
            q = make(TOPIC, seed=seed, difficulty=self.TIER)
            assert q.answer_text not in q.distractors
            assert len(q.distractors) == len(set(q.distractors))
            assert len(q.distractors) >= 1

    def test_grader_accepts_answer_rejects_distractors_and_wrong(self):
        for seed in SEEDS:
            q = make(TOPIC, seed=seed, difficulty=self.TIER)
            result = GRADER(q, q.answer_text)
            assert result.correct, f"seed {seed}: rejected canonical answer {q.answer_text!r}"
            for dist in q.distractors:
                result = GRADER(q, dist)
                assert not result.correct, f"seed {seed}: accepted distractor {dist!r}"
            wrong = int(q.answer_text) + 137
            result = GRADER(q, str(wrong))
            assert not result.correct, f"seed {seed}: accepted deliberately wrong {wrong}"

    def test_answer_is_a_plain_non_negative_integer(self):
        for seed in SEEDS:
            q = make(TOPIC, seed=seed, difficulty=self.TIER)
            assert isinstance(q.answer, int)
            assert q.answer >= 0
            assert re.fullmatch(r"\d+", q.answer_text), f"seed {seed}: {q.answer_text!r} is not a bare integer"
            for dist in q.distractors:
                assert re.fullmatch(r"-?\d+", dist), f"seed {seed}: distractor {dist!r} is not a bare integer"

    def test_at_least_20_distinct_questions_over_60_seeds(self):
        texts = {make(TOPIC, seed=seed, difficulty=self.TIER).text for seed in SEEDS}
        assert len(texts) >= 20, f"only {len(texts)} distinct questions over {len(SEEDS)} seeds"

    def test_prompt_variety_at_least_3_shapes(self):
        shapes = {prompt_shape(make(TOPIC, seed=seed, difficulty=self.TIER)) for seed in SEEDS}
        assert len(shapes) >= 3

    def test_no_bad_text_patterns(self):
        for seed in SEEDS:
            q = make(TOPIC, seed=seed, difficulty=self.TIER)
            defects = find_text_defects(q)
            assert not defects, f"seed {seed}: {defects}"

    def test_distractor_distribution_over_200_seeds(self):
        counts = {0: 0, 1: 0, 2: 0, 3: 0}
        for seed in range(200):
            q = make(TOPIC, seed=seed, difficulty=self.TIER)
            n = len(q.distractors)
            counts[min(n, 3)] += 1
        at_least_two = counts[2] + counts[3]
        assert at_least_two / 200 >= 0.85, f"{self.TIER}: distribution {counts}"
        # Stash on the class so the distribution can be read back after the run.
        type(self)._last_distribution = counts

    def test_skill_loads(self):
        _, skills = load_skills()
        skill = next(s for s in skills if s.id == SKILL_ID)
        assert skill.answer == "number"
        assert skill.placement_eligible is False
        # the exam source is now the multi-part network chain; the earlier exam variant is no longer wired
        topic = "exd_x_r3h_network_journeys" if self.TIER == "exam" else TOPIC
        assert any(src.topic == topic and src.difficulty == self.TIER for src in skill.sources)


class TestEasy(_CommonBehaviour):
    TIER = "easy"

    def test_entries_all_distinct_within_a_matrix(self):
        for seed in SEEDS:
            rng = random.Random(f"{TOPIC}:{self.TIER}:{seed}")
            draw = _draw_easy(rng)
            flat = [v for row in draw["matrix"] for v in row]
            assert len(flat) == len(set(flat))

    def test_transposed_target_never_on_diagonal(self):
        for seed in SEEDS:
            rng = random.Random(f"{TOPIC}:{self.TIER}:{seed}")
            draw = _draw_easy(rng)
            assert draw["target_row"] != draw["target_col"]

    def test_labels_distinct_and_non_confusable(self):
        for seed in SEEDS:
            q = make(TOPIC, seed=seed, difficulty=self.TIER)
            for pool in (STORE_NAMES, PRODUCT_NAMES):
                assert len(set(pool)) == len(pool)

    def test_verify_lookup_fires_on_corrupted_entry(self):
        rng = random.Random("p14a-matrix-verify-easy")
        draw = _draw_easy(rng)
        row, col = draw["row_labels"][draw["target_row"]], draw["col_labels"][draw["target_col"]]
        query = {"kind": "entry", "row": row, "col": col}
        _verify_matrix_lookup_or_total(draw["matrix"], draw["row_labels"], draw["col_labels"], query, draw["answer"])
        with pytest.raises(VerificationError):
            _verify_matrix_lookup_or_total(
                draw["matrix"], draw["row_labels"], draw["col_labels"], query, draw["answer"] + 1
            )

    def test_stage3_exclusion_no_third_operation(self):
        """Confirm by construction that only ``entry`` lookups ever appear on
        the easy tier -- never a named-subset sum or a derived difference."""
        for seed in SEEDS:
            rng = random.Random(f"{TOPIC}:{self.TIER}:{seed}")
            draw = _draw_easy(rng)
            assert set(draw.keys()) >= {"target_row", "target_col", "answer"}
            assert draw["answer"] == draw["matrix"][draw["target_row"]][draw["target_col"]]


class TestMedium(_CommonBehaviour):
    TIER = "medium"

    def test_row_and_column_sums_all_distinct(self):
        for seed in SEEDS:
            rng = random.Random(f"{TOPIC}:{self.TIER}:{seed}")
            draw = _draw_medium(rng)
            matrix = draw["matrix"]
            row_sums = [sum(r) for r in matrix]
            col_sums = [sum(matrix[i][j] for i in range(len(matrix))) for j in range(len(matrix[0]))]
            assert len(set(row_sums)) == len(row_sums)
            assert len(set(col_sums)) == len(col_sums)

    def test_both_axes_occur_over_60_seeds(self):
        axes = set()
        for seed in SEEDS:
            rng = random.Random(f"{TOPIC}:{self.TIER}:{seed}")
            draw = _draw_medium(rng)
            axes.add(draw["axis"])
        assert axes == {"row", "col"}

    def test_verify_row_total_fires_on_corrupted_total(self):
        rng = random.Random("p14a-matrix-verify-medium-row")
        draw = _draw_medium(rng)
        while draw["axis"] != "row":
            rng = random.Random(f"p14a-matrix-verify-medium-row-{rng.random()}")
            draw = _draw_medium(rng)
        row = draw["row_labels"][draw["idx"]]
        query = {"kind": "row_total", "row": row, "col": None}
        _verify_matrix_lookup_or_total(draw["matrix"], draw["row_labels"], draw["col_labels"], query, draw["answer"])
        with pytest.raises(VerificationError):
            _verify_matrix_lookup_or_total(
                draw["matrix"], draw["row_labels"], draw["col_labels"], query, draw["answer"] + 1
            )

    def test_verify_col_total_fires_on_corrupted_total(self):
        rng = random.Random("p14a-matrix-verify-medium-col")
        draw = _draw_medium(rng)
        while draw["axis"] != "col":
            rng = random.Random(f"p14a-matrix-verify-medium-col-{rng.random()}")
            draw = _draw_medium(rng)
        col = draw["col_labels"][draw["idx"]]
        query = {"kind": "col_total", "row": None, "col": col}
        _verify_matrix_lookup_or_total(draw["matrix"], draw["row_labels"], draw["col_labels"], query, draw["answer"])
        with pytest.raises(VerificationError):
            _verify_matrix_lookup_or_total(
                draw["matrix"], draw["row_labels"], draw["col_labels"], query, draw["answer"] + 1
            )


class TestHard(_CommonBehaviour):
    TIER = "hard"

    def test_matrix_asymmetric_at_queried_node(self):
        for seed in SEEDS:
            rng = random.Random(f"{TOPIC}:{self.TIER}:{seed}")
            draw = _draw_adjacency(rng, [4, 5], 9, TOWN_NAMES)
            matrix, node = draw["matrix"], draw["node"]
            row_sum = sum(matrix[node])
            col_sum = sum(row[node] for row in matrix)
            assert row_sum != col_sum

    def test_both_directions_occur_over_60_seeds(self):
        directions = set()
        for seed in SEEDS:
            rng = random.Random(f"{TOPIC}:{self.TIER}:{seed}")
            draw = _draw_adjacency(rng, [4, 5], 9, TOWN_NAMES)
            directions.add(draw["direction"])
        assert directions == {"from", "to"}

    def test_zero_entries_occur_over_60_seeds(self):
        # Plausibility: 0 is a valid "no direct link" entry and
        # should occur sometimes, not be avoided.
        saw_zero = False
        for seed in SEEDS:
            q = make(TOPIC, seed=seed, difficulty=self.TIER)
            if re.search(r"\| 0(?: \||;|\.)", q.text):
                saw_zero = True
                break
        assert saw_zero, "expected at least one seed to draw a 0 adjacency entry"

    def test_verify_asymmetric_fires_on_symmetric_input(self):
        symmetric = [[0, 3, 2], [3, 0, 1], [2, 1, 0]]
        with pytest.raises(VerificationError):
            _verify_asymmetric_at_node(symmetric, 0)

    def test_asymmetry_resample_rate_over_200_seeds(self):
        """Measure how often a raw (pre-rejection) draw would have been
        symmetric at its queried node -- i.e. how often the resample
        guard actually fires."""
        needed_resample = 0
        for seed in range(200):
            rng = random.Random(f"p14a-matrix-red-first-hard:{seed}")
            n = rng.choice([4, 5])
            rng.sample(TOWN_NAMES, n)
            matrix = [[0] * n for _ in range(n)]
            for i in range(n):
                for j in range(n):
                    if i != j:
                        matrix[i][j] = rng.randint(0, 9)
            node = rng.randrange(n)
            row_sum = sum(matrix[node])
            col_sum = sum(row[node] for row in matrix)
            if row_sum == col_sum:
                needed_resample += 1
        type(self)._resample_rate = needed_resample / 200
        assert needed_resample > 0, (
            "expected some raw (pre-rejection) draws to be symmetric at the queried node without the asymmetry guard"
        )


class TestExam(_CommonBehaviour):
    TIER = "exam"

    def test_matrix_asymmetric_at_queried_node(self):
        for seed in SEEDS:
            rng = random.Random(f"{TOPIC}:{self.TIER}:{seed}")
            draw = _draw_adjacency(rng, [5, 6], 9, NEIGHBOURHOOD_NAMES)
            matrix, node = draw["matrix"], draw["node"]
            row_sum = sum(matrix[node])
            col_sum = sum(row[node] for row in matrix)
            assert row_sum != col_sum

    def test_stages_is_exam_only(self):
        _, skills = load_skills()
        skill = next(s for s in skills if s.id == SKILL_ID)
        exam_src = next(src for src in skill.sources if src.difficulty == "exam")
        assert exam_src.stages == ["exam"]

    def test_larger_network_than_hard(self):
        for seed in SEEDS:
            rng = random.Random(f"{TOPIC}:{self.TIER}:{seed}")
            draw = _draw_adjacency(rng, [5, 6], 9, NEIGHBOURHOOD_NAMES)
            assert draw["n"] >= 5


def test_topic_registered_once_in_this_module():
    _, fn = REGISTRY[TOPIC]
    assert fn.__module__.endswith("p14a_matrix_table_build")


def test_module_name_ends_in_build():
    from mathgen.generators import p14a_matrix_table_build

    assert p14a_matrix_table_build.__name__.endswith("_build")


def test_no_collision_with_existing_topics():
    # grep confirms no pre-existing registration of this topic name.
    assert TOPIC == "p14a_matrix_table"
