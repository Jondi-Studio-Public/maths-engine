"""The HTML, Moodle XML and QTI 2.1 exports of a paper: they parse, hold every question, and only the key shows answers."""

import html
import io
import json
import re
import xml.etree.ElementTree as ET
import zipfile
from decimal import Decimal
from html.parser import HTMLParser

import pytest

from mathgen import export, service

MULTI_PART = "cyclic-group-element-order"  # exam source: three numeric parts with marks
TABLE = "f10-two-way-table-relationship"
FIGURE_MCQ = "count-to-10"  # an mcq with a counting figure

NS_QTI = "{http://www.imsglobal.org/xsd/imsqti_v2p1}"


@pytest.fixture(scope="module")
def lib():
    return service.Library.load()


@pytest.fixture(scope="module")
def paper(lib):
    mcq = service.find_skills(lib, answer_kind="mcq", year="8")[0].id
    number = service.find_skills(lib, answer_kind="number", year="7")[0].id
    expression = service.find_skills(lib, answer_kind="expression", year="9")[0].id
    spec = {
        "title": "Export <check> & more",
        "seed": 11,
        "versions": 2,
        "sections": [
            {
                "title": "Choice",
                "items": [{"skill": FIGURE_MCQ, "count": 2}, {"skill": mcq, "count": 2}],
            },
            {
                "title": "Answers",
                "items": [
                    {"skill": number, "count": 2},
                    {"skill": expression, "count": 2},
                    {"skill": MULTI_PART, "source": "exam", "count": 1},
                    {"skill": TABLE, "count": 1},
                ],
            },
        ],
    }
    return service.build_paper(lib, spec)


def _questions(paper, label="A"):
    version = next(v for v in paper["versions"] if v["label"] == label)
    return [q for s in version["sections"] for q in s["questions"]]


def _answer_blocks(q):
    return [p["answer"] for p in q["parts"]] if q.get("parts") else [q["answer"]]


class _Collect(HTMLParser):
    def __init__(self):
        super().__init__()
        self.stack, self.numbers, self.keys, self.errors, self.scripts = [], [], 0, [], 0
        self._in_num = False

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag in ("script", "link") or (tag == "img" and "src" in attrs):
            self.scripts += 1
        if tag not in ("meta", "br", "img", "input", "hr", "link"):
            self.stack.append(tag)
        cls = attrs.get("class") or ""
        self._in_num = tag == "div" and cls == "num"
        if "key" in cls.split():
            self.keys += 1

    def handle_endtag(self, tag):
        if not self.stack or self.stack[-1] != tag:
            self.errors.append(f"unbalanced </{tag}> with {self.stack[-3:]}")
            if tag in self.stack:
                while self.stack.pop() != tag:
                    pass
        else:
            self.stack.pop()
        self._in_num = False

    def handle_data(self, data):
        if self._in_num and data.strip():
            self.numbers.append(int(data))


def _parse_html(doc):
    p = _Collect()
    p.feed(doc)
    p.close()
    return p


def test_html_parses_and_numbers_every_question(paper):
    doc = export.paper_html(paper)
    p = _parse_html(doc)
    assert p.errors == [] and p.stack == []
    assert p.scripts == 0, "no scripts, links or external images"
    expected = [q["number"] for v in paper["versions"] for q in _questions(paper, v["label"])]
    assert p.numbers == expected
    assert doc.count('class="version"') == 2
    assert "Version A" in doc and "Version B" in doc
    assert "Export &lt;check&gt; &amp; more" in doc and "Export <check>" not in doc
    assert f"Paper seed {paper['seed']}" in doc
    for line in paper["attribution"]:
        assert html.escape(line) in doc


def test_html_single_version_and_total_marks(paper):
    doc = export.paper_html(paper, version="B")
    assert "Version B" in doc and "Version A" not in doc
    total = sum(q["marks"] for q in _questions(paper, "B"))
    assert f"{total} marks" in doc
    assert "[3 marks]" in doc or "[2 marks]" in doc  # a multi-part question shows its parts' marks
    with pytest.raises(ValueError):
        export.paper_html(paper, version="Z")


def test_html_hides_answers_unless_key(paper):
    plain = export.paper_html(paper, version="A")
    keyed = export.paper_html(paper, version="A", key=True)
    assert _parse_html(plain).keys == 0
    entries = sum(len(q.get("parts") or [1]) for q in _questions(paper))
    assert _parse_html(keyed).keys == entries
    assert _parse_html(keyed).errors == []
    checked = 0
    for q in _questions(paper):
        shown = q.get("text") or q["stem"]["text"]
        for a in _answer_blocks(q):
            for step in a["working"]:
                assert html.escape(step) not in plain or html.escape(step) in html.escape(shown)
                assert html.escape(step) in keyed
            if q["answer_kind"] in ("number", "mcq") and len(a["text"]) > 3 and a["text"] not in shown:
                if q["answer_kind"] == "number":
                    assert html.escape(a["text"]) not in plain
                checked += 1
                assert html.escape(a["text"]) in keyed
    assert checked >= 1
    mcqs = [q for q in _questions(paper) if q["answer_kind"] == "mcq" and not q.get("parts")]
    for q in mcqs:
        assert f"<b>Correct option:</b> {export._letter(q['answer']['option'])}." in keyed
        assert "Correct option" not in plain
        assert 'class="ol">A</span>' in plain and 'class="ol">D</span>' in plain


def test_html_requires_answers_for_key(lib):
    bare = service.build_paper(lib, {"seed": 1, "sections": [{"items": [{"skill": FIGURE_MCQ}]}]}, answers=False)
    assert "Correct option" not in export.paper_html(bare)
    with pytest.raises(ValueError):
        export.paper_html(bare, key=True)
    with pytest.raises(ValueError):
        export.paper_moodle_xml(bare)
    with pytest.raises(ValueError):
        export.paper_qti21_zip(bare)


def test_html_tables_and_figures(paper, monkeypatch):
    doc = export.paper_html(paper, version="A")
    assert '<table class="grid">' in doc and 'class="blank"' in doc
    assert "Table (" not in doc, "the table is drawn as a grid, not repeated as text"
    keyed = export.paper_html(paper, version="A", key=True)
    assert 'class="filled"' in keyed
    figures = [q["figure"] for q in _questions(paper) if q["figure"]]
    assert figures
    if export._figure_svg(figures[0]):
        assert "<svg" in doc
    # without mathgen.svg the description stands in, in a box
    monkeypatch.setattr(export, "_figure_svg", lambda spec, width=400: None)
    assert "figbox" in export.paper_html(paper, version="A") and "a group of shapes to count" in export.paper_html(
        paper
    )


def test_deterministic(paper):
    assert export.paper_html(paper) == export.paper_html(paper)
    assert export.paper_moodle_xml(paper) == export.paper_moodle_xml(paper)
    assert export.paper_qti21_zip(paper) == export.paper_qti21_zip(paper)
    assert export.paper_qti21_zip(paper, version="B") != export.paper_qti21_zip(paper, version="A")


# --- Moodle ----------------------------------------------------------------------------------------------------


def _expected_moodle_type(q):
    if q.get("parts"):
        return "cloze"
    kind = q["answer_kind"]
    if kind == "mcq":
        return "multichoice"
    if kind == "number" and export._numeric(q["answer"]["text"]):
        return "numerical"
    if kind == "table" and q.get("table") and export._table_answers(q["table"], q["answer"]["text"]):
        return "cloze"
    return "shortanswer"


def test_moodle_xml_types_and_counts(paper):
    root = ET.fromstring(export.paper_moodle_xml(paper, version="A").encode())
    questions = root.findall("question")
    assert questions[0].get("type") == "category"
    assert "Export <check> & more" in questions[0].findtext("category/text")
    body = questions[1:]
    expected = [_expected_moodle_type(q) for q in _questions(paper, "A")]
    assert [q.get("type") for q in body] == expected
    assert {"multichoice", "numerical", "shortanswer", "cloze"} <= set(expected)
    for q in body:
        assert q.findtext("generalfeedback/text").strip(), "the answer and working are in the feedback"
        if q.get("type") == "multichoice":
            assert q.findtext("shuffleanswers") == "0" and q.findtext("single") == "true"
            fractions = [a.get("fraction") for a in q.findall("answer")]
            assert fractions.count("100") == 1 and len(fractions) == 4
        if q.get("type") == "numerical":
            assert q.find("answer/tolerance") is not None


def test_moodle_multichoice_marks_the_right_option(paper):
    root = ET.fromstring(export.paper_moodle_xml(paper).encode())
    for src, node in zip(_questions(paper), root.findall("question")[1:]):
        if node.get("type") != "multichoice":
            continue
        right = [a.findtext("text") for a in node.findall("answer") if a.get("fraction") == "100"]
        assert right == [src["answer"]["text"]]


def test_moodle_cloze_for_parts_and_tables(paper):
    root = ET.fromstring(export.paper_moodle_xml(paper).encode())
    body = root.findall("question")[1:]
    qs = _questions(paper)
    multi = next(n for q, n in zip(qs, body) if q.get("parts"))
    text = multi.findtext("questiontext/text")
    assert text.count(":NUMERICAL:=") == 3
    assert re.search(r"\{2:NUMERICAL:=14:0\}", text) or "{2:NUMERICAL:=" in text
    assert multi.findtext("defaultgrade").startswith("5.")
    q_table, table = next((q, n) for q, n in zip(qs, body) if q["answer_kind"] == "table")
    assert table.get("type") == "cloze" and "<table" in table.findtext("questiontext/text")
    assert table.findtext("questiontext/text").count("NUMERICAL") == len(q_table["table"]["blanks"])


def test_moodle_figures_are_embedded_files(paper):
    root = ET.fromstring(export.paper_moodle_xml(paper).encode())
    figured = [n for q, n in zip(_questions(paper), root.findall("question")[1:]) if q["figure"]]
    assert figured
    if export._figure_svg(_questions(paper)[0]["figure"]):
        for n in figured:
            f = n.find("questiontext/file")
            assert f is not None and f.get("encoding") == "base64"
            assert f"@@PLUGINFILE@@/{f.get('name')}" in n.findtext("questiontext/text")
    else:
        assert all("[Figure:" in n.findtext("questiontext/text") for n in figured)


def test_cdata_guard():
    root = ET.fromstring(f"<a>{export._cdata('x ]]> y')}</a>")
    assert root.text == "x ]]> y"


@pytest.mark.parametrize(
    "text, context, expected",
    [
        ("14", "", ("14", "0")),
        ("-11", "", ("-11", "0")),
        ("−3.5", "", ("-3.5", "0")),
        ("$1,250.50", "", ("1250.50", "0")),
        ("3.14", "Give your answer to 2 decimal places", ("3.14", "0.005")),
        ("0.5", "", ("0.5", "0")),
        ("3/4", "", ("0.75", "0")),
        ("1/3", "", ("0.333333333333", "0.00033")),
        ("1/3000000", "", ("0.000000333333333333", "0.00000000033")),
        ("x = 8", "", ("8", "0")),
        ("f(-3) = 30", "", ("30", "0")),
        ("\u03b8 = 77.3\u00b0", "", ("77.3", "0")),
        ("\u03c3 = 50.00 MPa", "", ("50.00", "0")),
        ("F = 60.00 N", "", ("60.00", "0")),
        ("E = 70.0 GPa", "Give E to 1 decimal place", ("70.0", "0.05")),
        ("A$352.50", "", ("352.50", "0")),
        ("v = -4.5 m/s", "", ("-4.5", "0")),
        ("12 cm\u00b2", "", ("12", "0")),
        ("45\u00b0", "", ("45", "0")),
        ("2.5%", "", ("2.5", "0")),
        ("2.75", "the ground floor", ("2.75", "0")),
        ("2.75", "rounded to 2 decimal places", ("2.75", "0.005")),
        ("6√21", "", None),
        ("n ≤ 0", "", None),
    ],
)
def test_numeric_tolerance(text, context, expected):
    assert export._numeric(text, context) == expected


# --- QTI -------------------------------------------------------------------------------------------------------


def _open_zip(paper, version="A"):
    return zipfile.ZipFile(io.BytesIO(export.paper_qti21_zip(paper, version=version)))


def _item_names(zf):
    """The item files in paper order, as the assessmentTest lists them."""
    test = ET.fromstring(zf.read("test.xml"))
    return [r.get("href") for r in test.iter(f"{NS_QTI}assessmentItemRef")]


def test_qti_package_lists_and_parses_every_item(paper):
    zf = _open_zip(paper)
    assert zf.testzip() is None
    names = zf.namelist()
    assert names[0] == "imsmanifest.xml"
    manifest = ET.fromstring(zf.read("imsmanifest.xml"))
    ns = "{http://www.imsglobal.org/xsd/imscp_v1p1}"
    hrefs = {r.get("href") for r in manifest.iter(f"{ns}resource")}
    assert hrefs == {n for n in names if n != "imsmanifest.xml" and not n.startswith("figures/")}
    listed = {f.get("href") for f in manifest.iter(f"{ns}file")}
    assert set(names) - {"imsmanifest.xml"} == listed
    qs = _questions(paper)
    items = _item_names(zf)
    assert sorted(items) == sorted(n for n in names if n.startswith("q"))
    assert len(items) == len(qs)
    for name in names:
        ET.fromstring(zf.read(name))
    ids = [ET.fromstring(zf.read(n)).get("identifier") for n in items]
    assert len(set(ids)) == len(ids) and all(re.fullmatch(r"[A-Za-z][A-Za-z0-9_.-]*", i) for i in ids)


def test_qti_mcq_correct_identifier_matches_option(paper):
    zf = _open_zip(paper)
    qs = _questions(paper)
    items = _item_names(zf)
    checked = 0
    for q, name in zip(qs, items):
        root = ET.fromstring(zf.read(name))
        if q["answer_kind"] != "mcq" or q.get("parts"):
            continue
        correct = root.findtext(f"{NS_QTI}responseDeclaration/{NS_QTI}correctResponse/{NS_QTI}value")
        choices = {c.get("identifier"): c.text for c in root.iter(f"{NS_QTI}simpleChoice")}
        assert choices[correct] == q["answer"]["text"]
        assert list(choices.values()) == q["options"], "order is kept"
        assert root.find(f"{NS_QTI}responseProcessing").get("template").endswith("match_correct")
        checked += 1
    assert checked >= 2


def test_qti_number_text_parts_and_tables(paper):
    zf = _open_zip(paper)
    qs = _questions(paper)
    items = _item_names(zf)
    seen = set()
    for q, name in zip(qs, items):
        root = ET.fromstring(zf.read(name))
        decl = {d.get("identifier"): d for d in root.iter(f"{NS_QTI}responseDeclaration")}
        interactions = list(root.iter(f"{NS_QTI}textEntryInteraction")) + list(root.iter(f"{NS_QTI}choiceInteraction"))
        assert len(interactions) == len(decl)
        if q.get("parts"):
            assert len(interactions) == len(q["parts"])
            assert {d.get("baseType") for d in decl.values()} == {"float"}
            seen.add("parts")
        elif q["answer_kind"] == "table":
            assert len(interactions) == len(q["table"]["blanks"])
            seen.add("table")
        elif q["answer_kind"] == "number" and export._numeric(q["answer"]["text"]):
            (d,) = decl.values()
            assert d.get("baseType") == "float"
            assert d.findtext(f"{NS_QTI}correctResponse/{NS_QTI}value") == export._numeric(q["answer"]["text"])[0]
            assert root.find(f".//{NS_QTI}equal") is not None
            seen.add("number")
        elif q["answer_kind"] not in ("mcq",):
            (d,) = decl.values()
            assert d.get("baseType") == "string"
            assert root.find(f".//{NS_QTI}stringMatch") is not None
            assert b"<!--" in zf.read(name)
            seen.add("text")
    assert seen == {"parts", "table", "number", "text"}


def test_qti_figures_are_package_files(paper):
    zf = _open_zip(paper)
    figs = [n for n in zf.namelist() if n.startswith("figures/")]
    if export._figure_svg(_questions(paper)[0]["figure"]):
        assert figs
        for n in figs:
            ET.fromstring(zf.read(n))
        manifest = zf.read("imsmanifest.xml").decode()
        assert all(f'href="{n}"' in manifest for n in figs)
        assert any(b"<img " in zf.read(n) for n in zf.namelist() if n.startswith("q"))
    else:
        assert not figs
        assert any(b"[Figure:" in zf.read(n) for n in zf.namelist() if n.startswith("q"))


def test_paper_is_json_ready(paper):
    json.dumps(paper)  # exports read plain data, never objects


def test_numeric_value_never_collapses_to_zero():
    value, tol = export._numeric("1/3000000")
    assert Decimal(value) > 0 and Decimal(tol) < Decimal(value)


def test_unknown_version_is_a_value_error_everywhere(paper):
    for call in (export.paper_html, export.paper_moodle_xml, export.paper_qti21_zip):
        with pytest.raises(ValueError, match="no version 'Z' in this paper"):
            call(paper, version="Z")


def test_lone_surrogates_do_not_break_any_format(lib):
    bad = service.build_paper(
        lib, {"title": "T\ud800x", "seed": 3, "sections": [{"title": "S\udfff", "items": [{"skill": FIGURE_MCQ}]}]}
    )
    bad["attribution"] = ["a\ud800b"]
    export.paper_html(bad).encode("utf-8")
    export.paper_html(bad, key=True).encode("utf-8")
    export.paper_moodle_xml(bad).encode("utf-8")
    zf = zipfile.ZipFile(io.BytesIO(export.paper_qti21_zip(bad)))
    for name in zf.namelist():
        ET.fromstring(zf.read(name))


def test_cloze_text_answers_are_plain_not_html():
    slot = {"kind": "text", "answer": "x < 3 / y} ~ #*"}
    cloze = export._slot_cloze(slot, 1)

    assert "&lt;" not in cloze and "x < 3" in cloze
    assert "\\}" in cloze and "\\~" in cloze and "\\#" in cloze and "\\/" in cloze


def test_rendered_figure_shape_is_used_as_is(monkeypatch):
    spec = {"alt": "a drawn thing", "svg": '<?xml version="1.0"?><svg xmlns="http://www.w3.org/2000/svg"></svg>'}
    assert export._figure_svg(spec) == '<svg xmlns="http://www.w3.org/2000/svg"></svg>'


def test_answer_with_name_and_unit_is_numerical(lib):
    q = {
        "id": "x/y/1/abc",
        "skill": "x",
        "number": 1,
        "answer_kind": "number",
        "marks": 1,
        "figure": None,
        "text": "Find F.",
        "answer": {"text": "F = 60.00 N", "latex": None, "working": []},
    }
    paper = {
        "title": "t",
        "seed": 1,
        "answers": True,
        "attribution": [],
        "engine": {},
        "versions": [{"label": "A", "sections": [{"title": None, "questions": [q]}]}],
    }
    root = ET.fromstring(export.paper_moodle_xml(paper).encode())
    num = root.findall("question")[1]
    assert num.get("type") == "numerical" and num.findtext("answer/text") == "60.00"
