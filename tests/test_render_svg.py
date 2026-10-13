"""The SVG renderer draws every figure kind the engine emits, as well-formed, self-contained, leak-free SVG."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET

import pytest

from mathgen import engine, figures
from mathgen.render import figure_svg, renderable_kinds

SVG = "{http://www.w3.org/2000/svg}"
SEEDS = range(1, 7)

# Text a renderer may draw that is not one of the spec's own strings: tick and dial numbers, multiples of pi.
_NUMERIC = re.compile(r"^[−-]?(\d+(\.\d+)?)?(π)?(/\d+)?$")
# Fixed captions a renderer writes in front of one of the spec's strings ("P = 9/25", "Unit: block").
_CAPTIONS = ("P = ", "Unit: ")
_NON_FINITE = re.compile(r"\b(nan|inf|infinity)\b", re.I)


CHUNKS = 8  # the generated-figure sweep is split so pytest-xdist can share it


def _generated_figures(chunk):
    """Every figure one slice of the active skills draws, a few seeds each."""
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


def test_every_figure_kind_has_a_renderer():
    assert set(figures.kinds()) <= set(renderable_kinds())


@pytest.mark.parametrize("chunk", range(CHUNKS))
def test_renders_every_generated_figure(chunk):
    for where, spec in _generated_figures(chunk):
        svg = figure_svg(spec)
        try:
            root = ET.fromstring(svg)
        except ET.ParseError as e:  # pragma: no cover - the message is the point
            pytest.fail(f"{where}: not well-formed XML: {e}")
        assert root.tag == f"{SVG}svg", where
        assert root.get("role") == "img", where
        title = root.find(f"{SVG}title")
        assert title is not None and title.text == spec["alt"], where
        _, _, vw, vh = (float(v) for v in root.get("viewBox").split())
        assert 40 <= vw <= 1600 and 40 <= vh <= 1600, f"{where}: viewBox {vw} x {vh}"
        # A non-finite coordinate shows up as "nan" or "inf" in an attribute; text (a "banana" label) may say anything.
        for node in root.iter():
            for name, value in node.attrib.items():
                assert not _NON_FINITE.search(value), f"{where}: {node.tag} {name}={value!r}"
        assert 'href="http' not in svg and "<script" not in svg, where
        shown = set(figures.labels(spec))
        for node in root.iter(f"{SVG}text"):
            s = "".join(node.itertext()).strip()
            bare = next((s[len(c) :] for c in _CAPTIONS if s.startswith(c)), s)
            if s and s + " " not in _CAPTIONS and bare not in shown and not _NUMERIC.match(bare):
                pytest.fail(f"{where}: draws {s!r}, which is not one of the spec's strings")
        assert figure_svg(spec) == svg, f"{where}: not deterministic"


def test_ids_are_unique_between_two_figures():
    a = figures.number_line(range_=(0, 10), alt="a number line", marks=[{"at": 3}])
    b = figures.number_line(range_=(0, 10), alt="a number line", marks=[{"at": 4}])
    ids_a = set(re.findall(r'id="([^"]+)"', figure_svg(a)))
    ids_b = set(re.findall(r'id="([^"]+)"', figure_svg(b)))
    assert ids_a and not ids_a & ids_b


def test_dark_theme_and_custom_prefix():
    spec = figures.number_line(range_=(0, 10), alt="a number line", marks=[{"at": 3}])
    dark = figure_svg(spec, theme="dark", id_prefix="q1")
    assert "#161c2b" in dark and 'id="q1-' in dark
    with pytest.raises(ValueError):
        figure_svg(spec, theme="sepia")


def test_style_is_scoped_to_the_figure():
    spec = figures.number_line(range_=(0, 10), alt="a number line")
    style = re.search(r"<style>(.*)</style>", figure_svg(spec)).group(1)
    for rule in re.findall(r"([^{}]+)\{", style):
        for selector in rule.split(","):
            assert selector.strip().startswith(".mgf"), selector


def test_invalid_spec_is_refused():
    with pytest.raises(ValueError):
        figure_svg({"kind": "number_line", "alt": "", "range": [0, 1], "tick": 1})
