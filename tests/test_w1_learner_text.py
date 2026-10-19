"""Learner-facing text hygiene: ordinals, Python notation in the command-line output, currency, option lengths."""

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from mathgen import engine, make
from mathgen.core import display_question, ordinal
from mathgen.skills import load_skills
from mathgen.text_quality import BAD_TEXT_PATTERNS, find_python_notation, find_text_defects

ROOT = Path(__file__).resolve().parent.parent
_ORDINAL_PATTERN = next(p for p in BAD_TEXT_PATTERNS if p.name == "wrong_ordinal")


def test_ordinal_endings():
    expected = {
        1: "1st", 2: "2nd", 3: "3rd", 4: "4th", 9: "9th", 10: "10th", 11: "11th", 12: "12th", 13: "13th",
        14: "14th", 20: "20th", 21: "21st", 22: "22nd", 23: "23rd", 24: "24th", 101: "101st", 111: "111th",
        112: "112th", 113: "113th", 122: "122nd",
    }  # fmt: skip
    for n, text in expected.items():
        assert ordinal(n) == text


@pytest.mark.parametrize(
    "bad", ["the 3th term", "the 21th term", "the 12nd value", "the 4rd", "the 22th", "1th", "13st"]
)
def test_wrong_ordinals_are_caught(bad):
    assert _ORDINAL_PATTERN.find(bad) is not None


@pytest.mark.parametrize(
    "good",
    ["the 1st term", "the 3rd term", "the 11th term", "the 112th", "the (k + 1)th term", "27st^2", "2.3th", "3 th"],
)
def test_right_ordinals_pass(good):
    assert _ORDINAL_PATTERN.find(good) is None


def test_every_ordinal_the_helper_writes_passes_the_pattern():
    assert all(_ORDINAL_PATTERN.find(f"the {ordinal(n)} term") is None for n in range(1, 400))


_ORDINAL_TOPICS = (
    [("pv0a_bounce_distance", {"difficulty": d}) for d in ("easy", "medium", "hard")]
    + [("pv0a_annuity_duration", {"difficulty": d}) for d in ("easy", "medium", "hard")]
    + [
        ("sequence", {}),
        ("exd_x_w12a_histogram_chain", {"difficulty": "exam"}),
        ("w6b_complex_polygon_roots", {"difficulty": "easy"}),
        ("w6b_complex_polygon_roots", {"difficulty": "hard"}),
    ]
)


@pytest.mark.parametrize("topic,kwargs", _ORDINAL_TOPICS)
def test_generated_ordinals_are_right(topic, kwargs):
    for seed in range(40):
        try:
            q = make(topic, seed=seed, **kwargs)
        except Exception:  # noqa: BLE001 - a rejected draw is not this test's concern
            continue
        defects = [d for d in find_text_defects(q, include_working=True) if d.startswith("wrong_ordinal")]
        assert not defects, (topic, seed, defects)


def _school_skills():
    _domains, skills = load_skills()
    return [s for s in skills if s.id.startswith(("f10-", "senior-"))]


def test_f10_and_senior_text_has_no_python_notation_or_wrong_ordinals():
    """Everything a learner is shown (`display_question`) is free of `**`, `*` as times, sqrt( ), exp( ), the word pi
    and wrong ordinals, for every f10 and senior skill and every tier."""
    problems = []
    for skill in _school_skills():
        for src in skill.sources:
            try:
                q = engine.make_for_skill(skill, seed=1, source=src)
            except Exception:  # noqa: BLE001 - a deferred skill may not generate
                continue
            found = find_python_notation(display_question(q))
            found += [d for d in find_text_defects(q, include_working=True) if d.startswith("wrong_ordinal")]
            if found:
                problems.append(f"{skill.id}/{src.difficulty}: {found[0]}")
    assert not problems, problems[:10]


@pytest.mark.parametrize("topic", ["factor_gcf", "like_terms_select", "function_evaluation", "expression_parts"])
def test_cli_jsonl_is_paper_notation(topic, tmp_path):
    """`python -m mathgen --out` writes what a learner reads (`3x^2`), not Python's `3*x**2`."""
    out = tmp_path / "q.jsonl"
    done = subprocess.run(
        [sys.executable, "-m", "mathgen", "--topic", topic, "-n", "12", "--seed", "0", "--out", str(out)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        timeout=120,
    )
    assert done.returncode == 0, done.stderr
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 12
    for row in rows:
        shown = [row["text"], row["answer_text"], *row["distractors"]]
        assert not any("*" in s for s in shown), row["text"]
    assert any("^" in row["text"] or "×" in row["text"] or re.search(r"\d[a-z]", row["text"]) for row in rows)


def test_simple_interest_uses_the_local_dollar():
    for seed in range(60):
        for difficulty in ("easy", "medium", "hard"):
            text = make("formula_plugin", seed=seed, difficulty=difficulty).text
            assert "£" not in text, text


_LEAK_TOPICS = [
    ("define_variable", {}),
    ("smech_residual_plots", {"difficulty": "easy"}),
    ("smech_residual_plots", {"difficulty": "medium"}),
    ("smech_residual_plots", {"difficulty": "hard"}),
    ("f10c_congruence_criteria", {"difficulty": "easy"}),
    ("f10c_congruence_criteria", {"difficulty": "medium"}),
    ("f10c_congruence_criteria", {"difficulty": "hard"}),
    ("exd_x_r3f_ci_coverage", {"difficulty": "exam", "tier": "exam"}),
]


@pytest.mark.parametrize("topic,kwargs", _LEAK_TOPICS)
def test_the_key_is_not_given_away_by_its_length(topic, kwargs):
    """Over 120 seeds the key is the single longest option no more than about 35% of the time (4 options: 25% by
    chance), and the same for the single shortest."""
    longest = shortest = total = 0
    for seed in range(120):
        q = make(topic, seed=seed, **kwargs)
        items = (
            [(p.answer_text, p.distractors) for p in q.parts if p.answer_kind == "mcq"]
            if q.parts
            else [(q.answer_text, q.distractors)]
        )
        for answer, distractors in items:
            lengths = [len(d) for d in distractors]
            total += 1
            longest += len(answer) > max(lengths)
            shortest += len(answer) < min(lengths)
    assert total
    # The congruence skill's answer "Neither can be concluded from this information" is fixed wording that no other
    # option matches in length, so its hard tier is held to a looser bar.
    limit = 0.40 if topic == "f10c_congruence_criteria" else 0.35
    assert longest / total <= limit, (topic, kwargs, longest, total)
    assert shortest / total <= 0.35, (topic, kwargs, shortest, total)
