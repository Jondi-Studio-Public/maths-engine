"""`present` gives a page everything it shows for a question: clean text, typesetting-ready maths, the figure."""

from __future__ import annotations

import json
import re

import pytest

from mathgen import engine
from mathgen.core import mcq_options
from mathgen.present import present, prompt_tex, value_tex

SPAN = re.compile(r"\\\((.*?)\\\)", re.S)


@pytest.fixture(scope="module")
def skills():
    return {s.id: s for s in engine.load_skills()[1]}


def _pairs(page):
    """Every (text, tex) pair a page carries."""
    yield page["prompt"]
    yield page["answer"]
    yield from page.get("options", [])
    yield from page["working"]
    for part in page.get("parts", []):
        yield part["prompt"]
        yield part["answer"]
        yield from part.get("options", [])
        yield from part["working"]


# ------------------------------------------------------------- prompt_tex


@pytest.mark.parametrize(
    ("text", "latex", "tex"),
    [
        # prose with maths: prose stays text, each maths run gets its own delimiters
        ("Solve x^2 = 4 for x.", r"\text{Solve }x^{2} = 4\text{ for x.}", r"Solve \(x^{2} = 4\) for x."),
        # pure maths with no word: latex_prompt returns it unchanged, and it is all maths
        ("56 + 71", "56 + 71", r"\(56 + 71\)"),
        # all prose: nothing to typeset
        ("Which shape has more sides?", "Which shape has more sides?", None),
        # escaped prose characters come back as themselves
        (
            "It costs $4 & more.",
            r"\text{It costs \$4 \& more for }x + 1",
            r"It costs $4 & more for \(x + 1\)",
        ),
        # a \text{...} inside a maths group belongs to the maths
        (
            "sigma",
            r"\text{Find }\sigma = \frac{2.5\,\text{kN}}{A}\text{.}",
            r"Find \(\sigma = \frac{2.5\,\text{kN}}{A}\).",
        ),
        # spacing commands at a run's edge become an ordinary space
        ("f", r"f(x) = 3x,\ \text{find } f(-3)", r"\(f(x) = 3x,\) find \(f(-3)\)"),
        # not the shape it can split: leave it to `text`
        ("odd", r"\text{a {b}", None),
        ("plain", "plain:Obtuse", None),
        ("", "", None),
    ],
)
def test_prompt_tex(text, latex, tex):
    assert prompt_tex(text, latex) == tex


def test_value_tex():
    assert value_tex("6*sqrt(21)") == r"\(6 \sqrt{21}\)"
    assert value_tex("Obtuse") is None
    assert value_tex("x", "plain:Obtuse") is None
    assert value_tex("ignored", r"\frac{1}{2}") == r"\(\frac{1}{2}\)"


# ------------------------------------------------------------- present


def test_typed_answer_question(skills):
    skill = skills["simplify-surds"]
    q = engine.make_for_skill(skill, seed=7)
    page = engine.present(skill, q)
    assert page["prompt"]["text"] == q.text
    assert page["answer"] == {"text": "6√21", "tex": r"\(6 \sqrt{21}\)"}
    assert "options" not in page and page["figure"] is None
    json.dumps(page)


def test_multiple_choice_with_a_figure(skills):
    skill = skills["f10-clock-half-quarter-hour"]
    q = engine.make_for_skill(skill, seed=3)
    page = engine.present(skill, q)
    assert [o["text"] for o in page["options"]] == mcq_options(q)
    right = [o["text"] for o in page["options"]].index(q.answer_text)
    assert engine.grade(skill, q, str(right)).correct
    fig = page["figure"]
    assert fig["spec"] == q.figure and fig["alt"] == q.figure["alt"]
    assert fig["svg"].startswith("<svg") and 'role="img"' in fig["svg"]
    assert engine.present(skill, q, draw_figure=False)["figure"]["svg"] is None
    assert "#161c2b" in engine.present(skill, q, theme="dark")["figure"]["svg"]


def test_learner_view_hides_the_answer(skills):
    skill = skills["tell-time-to-hour"]
    q = engine.make_for_skill(skill, seed=1)
    page = engine.present(skill, q, reveal=False)
    assert "answer" not in page and "working" not in page
    assert set(page["figure"]) == {"svg", "alt"}
    assert [o["text"] for o in page["options"]] == mcq_options(q)
    assert "spec" in engine.present(skill, q)["figure"]


def test_multi_part_question(skills):
    for skill in skills.values():
        if any((s.difficulty or "").startswith("exam") for s in skill.sources):
            for i, _ in enumerate(skill.sources):
                try:
                    q = engine.make_for_skill(skill, 1, source=i)
                except Exception:
                    continue
                if q.parts:
                    break
            else:
                continue
            break
    else:
        pytest.skip("no multi-part question in the library")
    page = engine.present(skill, q)
    assert [p["label"] for p in page["parts"]] == [p.label for p in q.parts]
    for part, src in zip(page["parts"], q.parts):
        assert part["prompt"]["text"] == src.text or "**" in src.text
        assert part["answer"]["text"]
    hidden = engine.present(skill, q, reveal=False)
    assert all("answer" not in p and "working" not in p for p in hidden["parts"])


CHUNKS = 8  # the library sweep is split so pytest-xdist can share it


@pytest.mark.parametrize("chunk", range(CHUNKS))
def test_no_python_notation_or_broken_delimiters_anywhere(skills, chunk):
    """Across the library: no `x**2` reaches a learner, and every tex string has balanced delimiters."""
    for skill in sorted(skills.values(), key=lambda s: s.id)[chunk::CHUNKS]:
        for i, _ in enumerate(skill.sources):
            try:
                q = engine.make_for_skill(skill, 1, source=i)
            except Exception:  # a source that cannot generate is another test's concern
                continue
            page = engine.present(skill, q, draw_figure=False)
            for pair in _pairs(page):
                assert "**" not in pair["text"], (skill.id, pair)
                tex = pair["tex"]
                if tex is None:
                    continue
                assert tex.count(r"\(") == tex.count(r"\)") >= 1, (skill.id, tex)
                outside = SPAN.sub("", tex)
                assert r"\(" not in outside and r"\)" not in outside, (skill.id, tex)
                for span in SPAN.findall(tex):
                    assert span.strip() and not span.rstrip().endswith("\\"), (skill.id, tex)
