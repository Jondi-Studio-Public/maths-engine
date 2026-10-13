"""Every figure kind has a full description: complete, deterministic, plain text built only from the spec."""

from __future__ import annotations

import re

import pytest

from mathgen import engine, figures
from mathgen.describe import describable_kinds, describe

SEEDS = range(1, 4)
CHUNKS = 8  # the generated-figure sweep is split so pytest-xdist can share it
_NUMBER = re.compile(r"(?<![\w.])[−-]?\d+(?:\.\d+)?(?![\w.])")


def _generated_figures(chunk):
    _, skills = engine.load_skills()
    for skill in sorted(skills, key=lambda s: s.id)[chunk::CHUNKS]:
        for i, _src in enumerate(skill.sources):
            try:
                first = engine.make_for_skill(skill, 1, source=i)
            except Exception:  # a source that cannot generate is another test's concern
                continue
            if not first.figure:
                continue
            for seed in SEEDS:
                q = first if seed == 1 else engine.make_for_skill(skill, seed, source=i)
                if q.figure:
                    yield f"{skill.id}[{i}]@{seed}", q.figure


def test_every_figure_kind_has_a_describer():
    assert set(figures.kinds()) <= set(describable_kinds())


@pytest.mark.parametrize("chunk", range(CHUNKS))
def test_describes_every_generated_figure(chunk):
    for where, spec in _generated_figures(chunk):
        text = describe(spec)
        assert describe(spec) == text, f"{where}: not deterministic"
        assert text.strip() == text and "  " not in text, where
        assert text[-1] in ".?!" and text[0].isupper(), f"{where}: {text!r}"
        assert not re.search(r"[<>{}\\]|\*\*|\bNone\b|\bnan\b|\binf\b|\d\.0\b|\[|\]", text), f"{where}: {text!r}"
        assert len(text) <= 3000, f"{where}: {len(text)} characters"
        # It says more than the alt text: a description that adds nothing gives a blind learner nothing.
        assert len(text) > len(spec["alt"]), f"{where}: {text!r}"
        # Every label the figure draws (beyond the alt text) is in the description.
        for label in figures.labels(spec)[1:]:
            assert label in text, f"{where}: {label!r} is drawn but not described: {text!r}"


def test_title_leads_the_description():
    spec = figures.number_line(range_=(0, 10), alt="A number line.", marks=[{"at": 3}])
    spec["title"] = "Jumps of three"
    assert describe(spec).startswith("Jumps of three.")


def test_invalid_spec_is_refused():
    with pytest.raises(ValueError):
        describe({"kind": "number_line", "alt": "", "range": [0, 1], "tick": 1})
