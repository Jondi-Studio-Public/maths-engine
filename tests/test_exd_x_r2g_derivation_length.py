"""Tests for the multi-part exam source of ucv1-formal-derivation-check: Modus Ponens steps, unused premises, other
steps and the total lines of a shortest derivation. Keys are re-derived by building the derivation from the rendered
premises and target: every formula that must appear on a line, each counted once, classified by how it is obtained."""

from __future__ import annotations

import re

import mathgen.generators.exd_x_r2g_derivation_length  # noqa: F401  (registers the topic)
from tests.exd_x_r2g_support import exam, follow_through_check, standard_checks

TOPIC = "exd_x_r2g_derivation_length"
SKILL = "ucv1-formal-derivation-check"
LABELS = ["a", "b", "c", "d"]
MARKS = [1, 1, 2, 2]


def _parse(q):
    first = re.search(r"derive (.+?) from the premises (.+?)\. Count", q.stem)
    if first:
        return first.group(2).split("; "), first.group(1)
    second = re.search(r"From the premises (.+?), a shortest derivation of (.+?) is wanted", q.stem)
    return second.group(1).split("; "), second.group(2)


def _shortest(q):
    premises, target = _parse(q)
    implications = {}  # consequent -> antecedent
    conjunction_premises = []
    for p in premises:
        if " -> " in p:
            a, b = p.split(" -> ")
            implications[b] = (a, p)
        else:
            a, b = p.split(" & ")
            conjunction_premises.append((a, b, p))
    lines: dict[str, str] = {}  # formula -> how it is obtained

    def need(formula: str) -> None:
        if formula in lines:
            return
        if formula in premises:
            lines[formula] = "premise"
            return
        if formula in implications:
            antecedent, premise = implications[formula]
            lines[formula] = "mp"
            need(premise)
            need(antecedent)
            return
        for a, b, p in conjunction_premises:
            if formula in (a, b):
                lines[formula] = "and-elim"
                need(p)
                return
        raise AssertionError(f"cannot derive {formula}")

    disjunction = re.fullmatch(r"\((.+)\) \| (\w)", target)
    body = disjunction.group(1) if disjunction else target
    atoms = body.split(" & ")
    node = atoms[0]
    need(node)
    for atom in atoms[1:]:
        need(atom)
        node = f"({node} & {atom})"
        lines[node] = "and-intro"
    if disjunction:
        lines[target] = "or-intro"
    kinds = list(lines.values())
    return dict(
        mp=kinds.count("mp"),
        other=sum(1 for k in kinds if k in ("and-elim", "and-intro", "or-intro")),
        used=kinds.count("premise"),
        total=len(lines),
        n=len(premises),
    )


def oracle(q):
    d = _shortest(q)
    return {"a": str(d["mp"]), "b": str(d["n"] - d["used"]), "c": str(d["other"]), "d": str(d["total"])}


CHECKS = standard_checks(TOPIC, SKILL, oracle, LABELS, MARKS, {"b", "d"}, "p15d_formal_derivation_check")


def test_source_wired():
    CHECKS["wired"]()


def test_shape_and_text():
    CHECKS["shape"]()


def test_prompt_shapes_beyond_hard_tier():
    CHECKS["shapes"]()


def test_keys_match_independent_derivation():
    CHECKS["oracle_agrees"]()


def test_composed_answer_earns_full_marks():
    CHECKS["composed_full_marks"]()


def test_distractors_rejected_and_tagged():
    CHECKS["distractors_rejected"]()


def test_answer_space():
    CHECKS["answer_space"]()


def test_deterministic():
    CHECKS["deterministic"]()


def test_premise_lists_have_no_duplicates():
    for s in range(120):
        premises, _target = _parse(exam(TOPIC, s))
        assert len(premises) == len(set(premises))


def test_follow_through_from_wrong_step_count():
    """A Modus Ponens count one too high is carried into the unused premises and the line total."""

    def wrong_chain(q):
        d = _shortest(q)
        wrong = d["mp"] + 1
        if d["n"] - 1 - wrong == wrong:
            return None
        replies = {
            "a": str(wrong),
            "b": str(d["n"] - 1 - wrong),
            "c": str(d["other"]),
            "d": str((1 + wrong) + wrong + d["other"]),
        }
        return "a", replies, 1 + 2 + 2

    follow_through_check(TOPIC, SKILL, wrong_chain)
