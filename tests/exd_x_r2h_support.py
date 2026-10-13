"""Shared checks for the `test_exd_x_r2h_*` files: every multi-part exam source is held to the same contract.

A test file passes its oracle, which re-derives every part's answer text from the rendered stem and prompts alone;
nothing here imports a generator module.
"""

from __future__ import annotations

import dataclasses
from collections import Counter
from fractions import Fraction

from mathgen import make
from mathgen.engine import grade
from mathgen.misconceptions import CATALOGUE
from mathgen.skills import load_skills
from mathgen.text_quality import find_text_defects, prompt_shape


def skill_of(skill_id):
    return next(s for s in load_skills()[1] if s.id == skill_id)


def exam(topic, seed):
    return make(topic, seed=seed, difficulty="exam")


def answer_tuple(q):
    return tuple(p.answer_text for p in q.parts)


def standard_checks(topic, skill_id, oracle, labels, marks, ft_labels, hard_topic, *, max_share=15, n_distract=3):
    """Return a dict of named check functions; the test file wraps each in a test."""
    skill = skill_of(skill_id)

    def wired():
        exam_sources = [s for s in skill.sources if s.difficulty == "exam"]
        assert len(exam_sources) == 1
        src = exam_sources[0]
        assert (src.topic, src.stages, src.params) == (topic, ["exam"], {"tier": "exam"})
        assert skill.sources[-1] is src

    def shape():
        for s in range(40):
            q = exam(topic, s)
            assert [p.label for p in q.parts] == labels
            assert [p.marks for p in q.parts] == marks
            assert [p.follow_through is not None for p in q.parts] == [lab in ft_labels for lab in labels]
            assert find_text_defects(q) == [], (s, q.text)

    def shapes():
        shapes_exam = {prompt_shape(exam(topic, s)) for s in range(60)}
        shapes_hard = {prompt_shape(make(hard_topic, seed=s, difficulty="hard")) for s in range(60)}
        assert len(shapes_exam) >= 3, shapes_exam
        assert shapes_exam - shapes_hard

    def oracle_agrees():
        bad = []
        for s in range(300):
            q = exam(topic, s)
            got = {p.label: p.answer_text for p in q.parts}
            expected = oracle(q)
            if expected != got:
                bad.append((s, expected, got))
        assert bad == [], bad[:3]

    def composed_full_marks():
        for s in range(100):
            q = exam(topic, s)
            result = grade(skill, q, q.answer_text)
            assert result.correct and result.marks == result.max_marks == sum(marks), (s, q.answer_text)

    def distractors_rejected():
        for s in range(100):
            q = exam(topic, s)
            canon = {p.label: p.answer_text for p in q.parts}
            for p in q.parts:
                assert len(p.distractors) == n_distract, (s, p.label, p.distractors)
                for d in p.distractors:
                    assert p.option_notes.get(d) in CATALOGUE, (s, p.label, d)
                    result = grade(skill, q, {**canon, p.label: d})
                    assert result.parts[p.label].marks == 0, (s, p.label, d, canon)

    def answer_space():
        tuples = Counter(answer_tuple(exam(topic, s)) for s in range(300))
        assert len(tuples) >= 100, len(tuples)
        assert max(tuples.values()) <= max_share, tuples.most_common(3)

    def deterministic():
        for s in (0, 1, 7, 99):
            assert exam(topic, s).to_json() == exam(topic, s).to_json()

    return dict(
        wired=wired,
        shape=shape,
        shapes=shapes,
        oracle_agrees=oracle_agrees,
        composed_full_marks=composed_full_marks,
        distractors_rejected=distractors_rejected,
        answer_space=answer_space,
        deterministic=deterministic,
    )


def _degenerate(replies):
    """Whether the replies hold a zero or two equal values: the engine withholds follow-through marks then (a carried
    zero, or an answer that only repeats an earlier reply, is not new work), so such a seed says nothing here."""
    values = []
    for text in replies.values():
        try:
            values.append(Fraction(text))
        except (ValueError, ZeroDivisionError):
            return False
    return 0 in values or len(set(values)) < len(values)


def follow_through_check(topic, skill_id, wrong_chain, seeds=60, at_least=20):
    """`wrong_chain(q)` returns (wrong_label, replies, expected_marks): the replies a learner would give after a
    slip at `wrong_label`, carried through correctly, and the marks that earns. A seed whose replies hold a zero or a
    repeated value is skipped; at least `at_least` seeds must be checked."""
    skill = skill_of(skill_id)
    checked = 0
    for s in range(seeds):
        q = exam(topic, s)
        wrong_label, replies, expected = wrong_chain(q)
        if _degenerate(replies):
            continue
        checked += 1
        result = grade(skill, q, replies)
        assert result.parts[wrong_label].marks == 0, (s, replies)
        assert result.marks == expected, (s, replies, result.marks, expected)
        for label, part in result.parts.items():
            if label != wrong_label and part.marks:
                assert part.max_marks == part.marks
    assert checked >= at_least, checked


#: The exam source each of these skills used before it became a multi-part chain. The older single-answer test files
#: still describe that generator, which stays in the library, so they read the exam source through
#: `legacy_exam_sources`.
LEGACY_EXAM_TOPICS = {
    "ucv1-fourier-transform-gaussian": "p15e_fourier_transform_gaussian",
    "ucv1-fundamental-group-map": "p15b_fundamental_group_map",
    "ucv1-inverse-regularised-solve": "p14c_inverse_regularised_solve",
    "ucv1-poisson-count-probability": "p12e_poisson_count",
    "ucv1-sigma-algebra-closure": "p15c_sigma_algebra_closure",
    "ucv1-simplex-homology": "p15b_simplex_homology",
    "ucv1-structure-classification": "p15a_structure_classification",
    "ucv1-symmetry-conservation": "p15c_symmetry_conservation",
    "ucv1-syntactic-semantic-separation": "p15b_syntactic_semantic_separation",
    "ucv1-topology-basis-membership": "p15c_topology_basis_membership",
}


def legacy_exam_sources(skill_id, sources):
    """`sources` (difficulty -> Source) with the exam entry pointed back at the pre-chain generator, when the skill is
    one of those converted to a chain; any other skill's sources come back unchanged."""
    old = LEGACY_EXAM_TOPICS.get(skill_id)
    if old is None:
        return sources
    return {**sources, "exam": dataclasses.replace(sources["exam"], topic=old, params={})}
