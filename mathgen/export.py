"""Papers as files other tools open: a printable HTML worksheet and key, a Moodle XML question bank and an IMS QTI 2.1
package (the format Canvas imports).

    from mathgen import service, export

    lib = service.Library.load()
    paper = service.build_paper(lib, {"title": "Surds quiz", "seed": 42, "versions": 2,
                                      "sections": [{"items": [{"skill": "simplify-surds", "count": 5}]}]})
    open("worksheet.html", "w").write(export.paper_html(paper))
    open("key.html", "w").write(export.paper_html(paper, key=True))
    open("bank.xml", "w").write(export.paper_moodle_xml(paper, version="A"))
    open("quiz.zip", "wb").write(export.paper_qti21_zip(paper, version="A"))

`paper` is what `service.build_paper` returns. The HTML worksheet needs nothing but a browser; the Moodle and QTI
exports mark answers, so they need a paper built with `answers=True` (the default) and raise `ValueError` without.

What an LMS can mark itself, it does. A multiple-choice question becomes a single-answer choice, a number a numerical
answer within a tolerance (none when the answer is an exact integer or terminating decimal), a table a grid with one
box per blank, and a multi-part question one item (Moodle: a cloze) with a box or menu per part. Every other kind of
answer (an expression, a pair, an inequality, a fraction ...) becomes a text answer compared with the one answer we
give, ignoring case: a student who writes an equivalent form is marked wrong, so those items suit a teacher who
reads the responses. The full answer and working go in the feedback.

Figures come from `mathgen.svg.figure_svg` and are embedded as SVG; without it each figure's description is printed
in a box instead.
"""

from __future__ import annotations

import base64
import html
import io
import re
import zipfile
from decimal import Context, Decimal
from fractions import Fraction
from xml.sax.saxutils import escape as _xml_escape
from xml.sax.saxutils import quoteattr as _quoteattr

# --- shared model ----------------------------------------------------------------------------------------------

_BAD_XML = re.compile("[\x00-\x08\x0b\x0c\x0e-\x1f￾￿]")
_TABLE_SUFFIX = re.compile(r"\s*Table \(.*\Z", re.DOTALL)
_ROUND_CUE = re.compile(r"≈|approx|\bround|decimal place|significant|\bd\.?p\b|\bs\.?f\b", re.IGNORECASE)
_DECIMAL = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)")
_FRACTION = re.compile(r"[+-]?\d+/\d+")
#: "x = ", "F_net = ", "A" before a dollar sign, and the like, ahead of the number
_LEAD = re.compile(r"^\s*(?:[^\W\d][\w']*(?:\([\w ,+\-]*\))?\s*=\s*)?(?:[A-Z]{1,3}\$|\$)?\s*")
#: a unit after the number: letters, %, degrees, powers and slashes ("GPa", "m/s", "cm²", "m^2", "kg m/s")
_UNIT = re.compile(r"^[A-Za-zµμΩ°%][A-Za-zµμΩ°%²³·/^ \-\d]{0,14}$")


def _clean(text) -> str:
    """`text` as a string every output format can encode: lone surrogates become "?", control characters go."""
    out = "" if text is None else str(text)
    out = out.encode("utf-8", "replace").decode("utf-8")
    return _BAD_XML.sub("", out)


def _letter(i: int) -> str:
    return chr(ord("A") + i) if i < 26 else str(i + 1)


def _marks_label(n) -> str:
    return f"[{n} mark{'' if n == 1 else 's'}]"


def _select_version(paper: dict, version: str | None) -> list[dict]:
    versions = paper.get("versions") or []
    if version is None:
        return versions
    chosen = [v for v in versions if v.get("label") == version]
    if not chosen:
        raise ValueError(f"no version {version!r} in this paper")
    return chosen


def _one_version(paper: dict, version: str | None) -> dict:
    if version is None:
        version = "A"
    return _select_version(paper, version)[0]


def _figure_svg(spec: dict | None, width: int = 400) -> str | None:
    """The figure as a standalone SVG string, or None when there is none or `mathgen.svg` cannot draw it."""
    if not spec:
        return None
    if spec.get("svg"):  # already rendered (a paper built without answers carries the picture, not the spec)
        svg = spec["svg"]
    else:
        try:
            from .svg import figure_svg
        except ImportError:
            return None
        try:
            svg = figure_svg(spec, width=width)
        except Exception:  # a figure that cannot be drawn must not stop the export; the alt text stands in
            return None
    if not svg:
        return None
    svg = re.sub(r"^\s*<\?xml[^>]*\?>\s*", "", _clean(svg))
    return re.sub(r"^\s*<!DOCTYPE[^>]*>\s*", "", svg).strip()


def _numeric(text: str, context: str = "") -> tuple[str, str] | None:
    """(value, tolerance) for an answer that is a plain number, else None. A leading "x = " or currency sign and a
    trailing unit are ignored ("F = 60.00 N", "A$352.50", "70.0 GPa"). The tolerance is "0" for an integer, a
    terminating decimal or a fraction that is one, and half a unit in the last place for a decimal the question or
    working says was rounded. A fraction like 1/3 gets a tolerance of 0.1% of its value, at most 0.0005."""
    s = _clean(text).strip().replace("\u2212", "-").replace(",", "")
    s = s[_LEAD.match(s).end() :]
    m = re.match(r"[+-]?(?:\d+/\d+|\d+(?:\.\d*)?|\.\d+)", s)
    if not m:
        return None
    rest = s[m.end() :].strip()
    if rest and not _UNIT.match(rest):
        return None
    s = m.group(0)
    if _DECIMAL.fullmatch(s):
        value = s.lstrip("+")
        if "." in s and _ROUND_CUE.search(context):
            places = len(s.split(".")[1])
            return value, format(Decimal(5).scaleb(-(places + 1)), "f")
        return value, "0"
    frac = Fraction(s)
    if frac.denominator == 0:
        return None
    den = frac.denominator
    for p in (2, 5):
        while den % p == 0:
            den //= p
    ctx = Context(prec=12)
    value = ctx.divide(Decimal(frac.numerator), Decimal(frac.denominator))
    if den == 1:
        return format(value, "f"), "0"
    tol = min(Decimal("0.0005"), abs(value) * Decimal("0.001"))
    return format(value, "f"), format(Decimal(f"{tol:.2g}"), "f")


def _table_answers(table: dict, answer_text: str) -> dict | None:
    """{(row, col): value} for a table's blanks, read from an answer of the form "Child Returned = 4, ...", or None
    when any blank cannot be matched."""
    found: dict[str, str] = {}
    last = None
    for piece in answer_text.split(", "):
        key, sep, value = piece.partition(" = ")
        if sep:
            last = key
            found[key] = value
        elif last is not None:
            found[last] += ", " + piece
    out = {}
    for blank in table.get("blanks") or []:
        try:
            r, c = (int(x) for x in str(blank).split(","))
            key = f"{table['rows'][r][0]} {table['columns'][c]}"
        except (ValueError, IndexError, KeyError):
            return None
        if key not in found:
            return None
        out[(r, c)] = found[key].strip()
    return out or None


def _scalar_slot(answer: str, kind: str, context: str) -> dict:
    num = _numeric(answer, context) if kind == "number" else None
    if num:
        return {"kind": "number", "answer": answer, "value": num[0], "tol": num[1]}
    return {"kind": "text", "answer": answer}


def _entry(block: dict, kind: str, label, marks: int, *, need_answer: bool) -> dict:
    """One answerable piece of a question (the whole question, or a part), with the boxes a student fills."""
    ans = block.get("answer")
    if ans is None and need_answer:
        raise ValueError("this export marks answers: build the paper with answers=True")
    table = block.get("table")
    text = block.get("text") or ""
    if table:
        text = _TABLE_SUFFIX.sub("", text)
    entry = {
        "label": label,
        "marks": marks,
        "kind": kind,
        "text": text,
        "table": table,
        "options": block.get("options") if kind == "mcq" else None,
        "answer": ans,
        "cells": {},
        "slots": [],
    }
    if ans is None:
        return entry
    answer_text = _clean(ans.get("text"))
    working = " ".join(ans.get("working") or [])
    context = f"{text} {working} {answer_text}"
    if kind == "mcq":
        options = [_clean(o) for o in block.get("options") or []]
        entry["slots"] = [{"kind": "mcq", "answer": answer_text, "options": options, "correct": ans.get("option", 0)}]
    elif kind == "table" and table:
        values = _table_answers(table, answer_text)
        if values:
            for pos in sorted(values):
                slot = _scalar_slot(values[pos], "number", context)
                entry["cells"][pos] = slot
                entry["slots"].append(slot)
        else:
            entry["slots"] = [{"kind": "text", "answer": answer_text}]
    else:
        entry["slots"] = [_scalar_slot(answer_text, kind, context)]
    return entry


def _items(version: dict, *, need_answer: bool) -> list[dict]:
    """The version's questions in order, each normalised for the exporters."""
    items = []
    for section in version.get("sections") or []:
        for q in section.get("questions") or []:
            if q.get("parts"):
                entries = [
                    _entry(
                        p,
                        p.get("answer_kind", q.get("answer_kind")),
                        p["label"],
                        p.get("marks", 1),
                        need_answer=need_answer,
                    )
                    for p in q["parts"]
                ]
                stem = (q.get("stem") or {}).get("text") or ""
            else:
                entries = [_entry(q, q.get("answer_kind"), None, q.get("marks", 1), need_answer=need_answer)]
                stem = ""
            items.append(
                {
                    "id": q.get("id", ""),
                    "skill": q.get("skill", ""),
                    "number": q.get("number", len(items) + 1),
                    "marks": q.get("marks", 1),
                    "stem": stem,
                    "figure": q.get("figure"),
                    "entries": entries,
                    "section": section.get("title"),
                }
            )
    return items


def _feedback_lines(item: dict) -> list[tuple[str | None, str, list[str]]]:
    """(part label, answer text, working steps) for each entry, for general feedback."""
    out = []
    for e in item["entries"]:
        a = e["answer"] or {}
        out.append((e["label"], _clean(a.get("text")), [_clean(w) for w in a.get("working") or []]))
    return out


# --- HTML ------------------------------------------------------------------------------------------------------

_CSS = """
@page { size: A4; margin: 16mm 15mm 18mm; }
* { box-sizing: border-box; }
html { background: #fff; }
body { font: 11pt/1.45 "Segoe UI", "Helvetica Neue", Helvetica, Arial, sans-serif; color: #111; margin: 0 auto;
       max-width: 190mm; padding: 8mm 0; }
.version { break-after: page; page-break-after: always; }
.version:last-child { break-after: auto; page-break-after: auto; }
h1 { font-size: 20pt; margin: 0 0 2mm; }
.sub { color: #444; margin: 0 0 4mm; }
.namebar { display: flex; gap: 8mm; margin: 0 0 5mm; }
.namebar span { flex: 1; border-bottom: 1px solid #111; padding-bottom: 1mm; color: #444; font-size: 10pt; }
.total { font-weight: 600; }
h2 { font-size: 13pt; margin: 6mm 0 3mm; padding-bottom: 1mm; border-bottom: 2px solid #111; }
.q { display: flex; gap: 4mm; margin: 0 0 6mm; break-inside: avoid; page-break-inside: avoid; }
.num { flex: 0 0 9mm; font-weight: 700; text-align: right; }
.body { flex: 1; min-width: 0; }
.marks { color: #444; font-size: 9.5pt; white-space: nowrap; }
.text { margin: 0 0 2mm; white-space: pre-wrap; overflow-wrap: anywhere; }
.fig { margin: 2mm 0 3mm; }
.fig svg { width: 60mm; max-width: 100%; height: auto; }
.figbox { border: 1px solid #111; padding: 3mm; font-style: italic; color: #333; }
.opts { list-style: none; margin: 0; padding: 0; }
.opts li { display: flex; gap: 3mm; margin: 1.5mm 0; align-items: baseline; }
.opts .ol { flex: 0 0 6mm; height: 6mm; border: 1px solid #111; border-radius: 50%; text-align: center;
            font-size: 9pt; line-height: 5mm; }
.opts li.correct { font-weight: 700; }
.opts li.correct .ol { background: #111; color: #fff; }
.note { color: #444; font-size: 9.5pt; font-weight: 400; }
.part { display: flex; gap: 3mm; margin: 3mm 0 0; }
.pl { flex: 0 0 7mm; }
.pb { flex: 1; min-width: 0; }
.ansline { display: flex; gap: 2mm; align-items: flex-end; margin-top: 3mm; }
.ansline .line { flex: 1; border-bottom: 1px solid #111; height: 6mm; }
.work { border: 1px solid #888; margin-top: 2mm; }
table.grid { border-collapse: collapse; margin: 2mm 0; }
table.grid th, table.grid td { border: 1px solid #111; padding: 1.5mm 4mm; min-width: 20mm; height: 8mm; text-align: center; }
table.grid th { background: #eee; }
table.grid td.blank { background: #fff; }
table.grid td.filled { font-weight: 700; }
.key { margin-top: 2mm; padding: 2mm 3mm; background: #f1f1f1; border-left: 3px solid #111; }
.key ol { margin: 1mm 0 0 5mm; padding: 0; }
.foot { margin-top: 8mm; padding-top: 2mm; border-top: 1px solid #888; font-size: 8pt; color: #444; }
.foot p { margin: 0 0 1mm; }
@media print { body { padding: 0; max-width: none; } }
"""


def _h(value) -> str:
    return html.escape(_clean(value), quote=True)


def _figure_html(spec: dict | None) -> str:
    if not spec:
        return ""
    svg = _figure_svg(spec)
    if svg:
        return f'<div class="fig" role="img" aria-label="{_h(spec.get("alt") or "figure")}">{svg}</div>'
    return f'<div class="fig"><div class="figbox">Figure: {_h(spec.get("alt") or "see the printed figure")}</div></div>'


def _table_html(entry: dict, key: bool) -> str:
    table = entry["table"]
    if not table:
        return ""
    out = ['<table class="grid"><thead><tr>']
    out += [f"<th>{_h(c)}</th>" for c in table.get("columns") or []]
    out.append("</tr></thead><tbody>")
    for r, row in enumerate(table.get("rows") or []):
        out.append("<tr>")
        for c, cell in enumerate(row):
            if cell is None:
                slot = entry["cells"].get((r, c)) if key else None
                if slot:
                    out.append(f'<td class="filled">{_h(slot["answer"])}</td>')
                else:
                    out.append('<td class="blank"></td>')
            else:
                out.append(f"<td>{_h(cell)}</td>")
        out.append("</tr>")
    out.append("</tbody></table>")
    return "".join(out)


def _answer_area(entry: dict, key: bool) -> str:
    kind = entry["kind"]
    a = entry["answer"] or {}
    parts = []
    if kind == "mcq":
        correct = a.get("option")
        notes = a.get("option_notes") or {}
        lis = []
        for i, opt in enumerate(entry["options"] or []):
            is_right = key and i == correct
            note = f' <span class="note">{_h(notes[str(i)])}</span>' if key and str(i) in notes else ""
            lis.append(
                f'<li class="{"correct" if is_right else ""}"><span class="ol">{_letter(i)}</span>'
                f"<span>{_h(opt)}{note}</span></li>"
            )
        parts.append(f'<ul class="opts">{"".join(lis)}</ul>')
    else:
        if entry["table"]:
            parts.append(_table_html(entry, key))
        elif not key:
            if entry["marks"] >= 2:
                parts.append(f'<div class="work" style="height:{min(entry["marks"], 6) * 9}mm"></div>')
            parts.append('<div class="ansline"><span>Answer:</span><span class="line"></span></div>')
    if key and a:
        steps = a.get("working") or []
        body = f"<b>Answer:</b> {_h(a.get('text'))}"
        if kind == "mcq" and a.get("option") is not None:
            body = f"<b>Correct option:</b> {_letter(a['option'])}. {_h(a.get('text'))}"
        if steps:
            body += "<ol>" + "".join(f"<li>{_h(s)}</li>" for s in steps) + "</ol>"
        parts.append(f'<div class="key">{body}</div>')
    return "".join(parts)


def _question_html(item: dict, key: bool) -> str:
    out = [
        f'<section class="q" id="q{int(item["number"])}"><div class="num">{_h(item["number"])}</div><div class="body">'
    ]
    multi = item["entries"][0]["label"] is not None
    if multi:
        out.append(f'<p class="text">{_h(item["stem"])} <span class="marks">{_marks_label(item["marks"])}</span></p>')
    if multi:
        out.append(_figure_html(item["figure"]))
    for e in item["entries"]:
        text = f'<p class="text">{_h(e["text"])}'
        if multi:
            text += f' <span class="marks">{_marks_label(e["marks"])}</span>'
        text += "</p>"
        area = _answer_area(e, key)
        if multi:
            out.append(
                f'<div class="part"><div class="pl">({_h(e["label"])})</div><div class="pb">{text}{area}</div></div>'
            )
        else:
            out.append(f'{text}{_figure_html(item["figure"])}<div class="marks">{_marks_label(e["marks"])}</div>{area}')
    out.append("</div></section>")
    return "".join(out)


def paper_html(paper: dict, *, version: str | None = None, key: bool = False) -> str:
    """A self-contained printable HTML document (A4, no scripts, fonts or links out), one page break between
    versions. `key=True` gives the answer key: each answer, its working and, for multiple choice, the right option.
    `version` picks one version by label; None prints them all."""
    versions = _select_version(paper, version)
    if key and not paper.get("answers", True):
        raise ValueError("an answer key needs a paper built with answers=True")
    title = _clean(paper.get("title") or "Worksheet")
    engine_info = paper.get("engine") or {}
    pages = []
    for v in versions:
        items = _items(v, need_answer=key)
        total = sum(i["marks"] for i in items)
        label = _clean(v.get("label"))
        head = [f"<h1>{_h(title)}</h1>"]
        sub = (
            f"{'Answer key, ' if key else ''}Version {_h(label)}" if len(paper.get("versions") or []) > 1 or key else ""
        )
        head.append(
            f'<p class="sub">{sub + " &middot; " if sub else ""}<span class="total">{total} marks</span> in total</p>'
        )
        if not key:
            head.append(
                '<div class="namebar"><span>Name</span><span>Date</span><span>Score / ' + str(total) + "</span></div>"
            )
        body = []
        n = 0
        for section in v.get("sections") or []:
            count = len(section.get("questions") or [])
            sec_items = items[n : n + count]
            n += count
            sec_marks = sum(i["marks"] for i in sec_items)
            if section.get("title"):
                body.append(f'<h2>{_h(section["title"])} <span class="marks">{_marks_label(sec_marks)}</span></h2>')
            body.extend(_question_html(i, key) for i in sec_items)
        foot = [f"<p>{_h(a)}</p>" for a in paper.get("attribution") or []]
        regen = (
            f"Paper seed {_h(paper.get('seed'))}, version {_h(label)}. Generated by maths-engine "
            f"{_h(engine_info.get('engine', ''))} with skill library {_h(engine_info.get('library', ''))}: "
            "the same seed, spec and library regenerate this paper."
        )
        foot.append(f"<p>{regen}</p>")
        pages.append(
            f'<main class="version">{"".join(head)}{"".join(body)}<footer class="foot">{"".join(foot)}</footer></main>'
        )
    return (
        '<!DOCTYPE html>\n<html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{_h(title)}{' (answer key)' if key else ''}</title><style>{_CSS}</style></head>\n"
        f"<body>\n{chr(10).join(pages)}\n</body></html>\n"
    )


# --- Moodle XML ------------------------------------------------------------------------------------------------


def _cdata(text: str) -> str:
    return "<![CDATA[" + _clean(text).replace("]]>", "]]]]><![CDATA[>") + "]]>"


def _xml(text) -> str:
    return _xml_escape(_clean(text))


def _wild(text: str) -> str:
    """Moodle's short-answer matching treats * as a wildcard: escape it, and the backslash that does the escaping."""
    return text.replace("\\", "\\\\").replace("*", "\\*")


def _cloze(text: str) -> str:
    return re.sub(r'([\\}#~/"])', r"\\\1", text)


def _html_p(text: str) -> str:
    return f"<p>{_h(text)}</p>"


def _grid_html(entry: dict, cell_markup) -> str:
    """The table as an HTML grid; `cell_markup(r, c)` gives the markup for a blank cell."""
    table = entry["table"]
    out = ['<table border="1" cellpadding="4"><tr>']
    out += [f"<th>{_h(c)}</th>" for c in table.get("columns") or []]
    out.append("</tr>")
    for r, row in enumerate(table.get("rows") or []):
        out.append("<tr>")
        for c, cell in enumerate(row):
            out.append(f"<td>{cell_markup(r, c) if cell is None else _h(cell)}</td>")
        out.append("</tr>")
    out.append("</table>")
    return "".join(out)


def _feedback_html(item: dict) -> str:
    out = []
    for label, answer, working in _feedback_lines(item):
        prefix = f"({label}) " if label else ""
        out.append(f"<p><b>{_h(prefix)}Answer:</b> {_h(answer)}</p>")
        if working:
            out.append("<ol>" + "".join(f"<li>{_h(w)}</li>" for w in working) + "</ol>")
    return "".join(out)


def _slot_cloze(slot: dict, weight) -> str:
    w = f"{weight:g}"
    if slot["kind"] == "mcq":
        opts = [_cloze(_clean(o)) for o in slot["options"]]
        body = "~".join(("=" if i == slot["correct"] else "") + o for i, o in enumerate(opts))
        return "{" + f"{w}:MULTICHOICE:{body}" + "}"
    if slot["kind"] == "number":
        return "{" + f"{w}:NUMERICAL:={slot['value']}:{slot['tol']}" + "}"
    return "{" + f"{w}:SHORTANSWER:={_cloze(_wild(_clean(slot['answer'])))}" + "}"


def _cloze_entry_html(entry: dict) -> str:
    marks = entry["marks"]
    slots = entry["slots"]
    if entry["table"] and entry["cells"]:
        by_pos = entry["cells"]
        grid = _grid_html(entry, lambda r, c: _slot_cloze(by_pos[(r, c)], 1) if (r, c) in by_pos else "")
        text = _h(entry["text"])
        return f"<p>{text}</p>{grid}"
    box = _slot_cloze(slots[0], marks)
    if entry["kind"] == "mcq":
        options = "".join(f"<br>{_letter(i)}. {_h(o)}" for i, o in enumerate(entry["options"] or []))
        return f"<p>{_h(entry['text'])}{options}</p><p>Answer: {box}</p>"
    table = _grid_html(entry, lambda r, c: "") if entry["table"] else ""
    return f"<p>{_h(entry['text'])}</p>{table}<p>Answer: {box}</p>"


def _moodle_question(item: dict, figure_no: int | None, svg: str | None) -> str:
    entries = item["entries"]
    single = len(entries) == 1 and entries[0]["label"] is None
    only = entries[0]
    native = single and len(only["slots"]) == 1 and not only["table"]
    if native:
        slot = only["slots"][0]
        qtype = {"mcq": "multichoice", "number": "numerical"}.get(slot["kind"], "shortanswer")
    elif single and only["table"] and not only["cells"]:
        qtype = "shortanswer"
    else:
        qtype = "cloze"
    figure_md = ""
    if item["figure"]:
        alt = _h(item["figure"].get("alt") or "figure")
        figure_md = (
            f'<p><img src="@@PLUGINFILE@@/figure-{figure_no}.svg" alt="{alt}" width="400"></p>'
            if svg
            else f"<p><i>[Figure: {alt}]</i></p>"
        )
    # question text
    if qtype == "cloze":
        stem = _html_p(item["stem"]) if item["stem"] else ""
        parts = []
        for e in entries:
            body = _cloze_entry_html(e)
            if e["label"] is not None:
                body = f"<p><b>({_h(e['label'])})</b> <small>{_marks_label(e['marks'])}</small></p>{body}"
            parts.append(body)
        qtext = stem + figure_md + "".join(parts)
        grade = sum(len(e["cells"]) if e["table"] and e["cells"] else e["marks"] for e in entries)
    else:
        text = _h(only["text"])
        table = ""
        if only["table"]:
            table = _grid_html(only, lambda r, c: "&nbsp;&nbsp;&nbsp;&nbsp;")
        qtext = f"<p>{text}</p>{figure_md}{table}"
        grade = only["marks"]
    file_xml = ""
    if svg:
        data = base64.b64encode(svg.encode("utf-8", "replace")).decode("ascii")
        file_xml = f'<file name="figure-{figure_no}.svg" path="/" encoding="base64">{data}</file>'
    name = f"{item['number']}. {item['skill']}"
    out = [
        f'<question type="{qtype}">',
        f"<name><text>{_xml(name)}</text></name>",
        f'<questiontext format="html"><text>{_cdata(qtext)}</text>{file_xml}</questiontext>',
        f'<generalfeedback format="html"><text>{_cdata(_feedback_html(item))}</text></generalfeedback>',
        f"<defaultgrade>{grade:.7f}</defaultgrade><penalty>0.0000000</penalty><hidden>0</hidden>",
        f"<idnumber>{_xml(item['id'])}</idnumber>",
    ]
    if qtype == "multichoice":
        slot = only["slots"][0]
        notes = (only["answer"] or {}).get("option_notes") or {}
        out.append("<single>true</single><shuffleanswers>0</shuffleanswers><answernumbering>abc</answernumbering>")
        out.append('<correctfeedback format="html"><text></text></correctfeedback>')
        out.append('<partiallycorrectfeedback format="html"><text></text></partiallycorrectfeedback>')
        out.append('<incorrectfeedback format="html"><text></text></incorrectfeedback>')
        for i, opt in enumerate(slot["options"]):
            fb = notes.get(str(i), "")
            out.append(
                f'<answer fraction="{100 if i == slot["correct"] else 0}" format="html"><text>{_cdata(_h(opt))}</text>'
                f'<feedback format="html"><text>{_cdata(_h(fb) if fb else "")}</text></feedback></answer>'
            )
    elif qtype == "numerical":
        slot = only["slots"][0]
        out.append(
            f'<answer fraction="100" format="plain_text"><text>{_xml(slot["value"])}</text>'
            f'<feedback format="html"><text></text></feedback><tolerance>{slot["tol"]}</tolerance></answer>'
        )
        out.append(
            "<unitgradingtype>0</unitgradingtype><unitpenalty>0.1000000</unitpenalty><showunits>3</showunits><unitsleft>0</unitsleft>"
        )
    elif qtype == "shortanswer":
        out.append("<usecase>0</usecase>")
        out.append(
            f'<answer fraction="100" format="plain_text"><text>{_xml(_wild(only["slots"][0]["answer"]))}</text>'
            '<feedback format="html"><text></text></feedback></answer>'
        )
    out.append("</question>")
    return "".join(out)


def paper_moodle_xml(paper: dict, *, version: str = "A") -> str:
    """A Moodle XML question bank for one version: a category named after the paper, then one question per
    question. Multiple choice becomes multichoice, a number numerical, a multi-part question or a table a cloze
    (multianswer); any other answer kind becomes shortanswer with the answer as the one accepted answer. Every
    question's general feedback carries the full answer and working."""
    chosen = _one_version(paper, version)
    items = _items(chosen, need_answer=True)
    title = _clean(paper.get("title") or "Worksheet")
    out = ['<?xml version="1.0" encoding="UTF-8"?>\n<quiz>']
    out.append(
        f'<question type="category"><category><text>{_xml("$course$/top/" + title + " (version " + str(version) + ")")}</text></category></question>'
    )
    n = 0
    for item in items:
        svg = _figure_svg(item["figure"])
        if svg:
            n += 1
        out.append(_moodle_question(item, n if svg else None, svg))
    out.append("</quiz>\n")
    return _clean("\n".join(out))


# --- QTI 2.1 ---------------------------------------------------------------------------------------------------

_QTI_NS = "http://www.imsglobal.org/xsd/imsqti_v2p1"
_XSI = "http://www.w3.org/2001/XMLSchema-instance"
_NOT_NAME = re.compile(r"[^A-Za-z0-9_.-]")


def _ident(item: dict) -> str:
    return f"q{item['number']}_{_NOT_NAME.sub('_', str(item['id']))}"


def _qa(value) -> str:
    return _quoteattr(_clean(value))


def _qti_slot_xml(slot: dict, rid: str, idx: int) -> str:
    if slot["kind"] == "mcq":
        choices = "".join(
            f'<simpleChoice identifier="{rid}_C{i}">{_xml(o)}</simpleChoice>' for i, o in enumerate(slot["options"])
        )
        return f'<choiceInteraction responseIdentifier="{rid}" shuffle="false" maxChoices="1">{choices}</choiceInteraction>'
    length = max(8, min(40, len(slot["answer"]) + 4))
    return f'<textEntryInteraction responseIdentifier="{rid}" expectedLength="{length}"/>'


def _qti_item(item: dict, figure_file: str | None, svg: str | None) -> str:
    slots = [(e, s) for e in item["entries"] for s in e["slots"]]
    single = len(slots) == 1
    rids = {id(s): ("RESPONSE" if single else f"RESPONSE_{i}") for i, (_e, s) in enumerate(slots, 1)}
    ident = _ident(item)
    decls, body = [], []
    notes = []
    for _e, s in slots:
        rid = rids[id(s)]
        if s["kind"] == "mcq":
            decls.append(
                f'<responseDeclaration identifier="{rid}" cardinality="single" baseType="identifier">'
                f"<correctResponse><value>{rid}_C{s['correct']}</value></correctResponse></responseDeclaration>"
            )
        elif s["kind"] == "number":
            decls.append(
                f'<responseDeclaration identifier="{rid}" cardinality="single" baseType="float">'
                f"<correctResponse><value>{_xml(s['value'])}</value></correctResponse></responseDeclaration>"
            )
        else:
            decls.append(
                f'<responseDeclaration identifier="{rid}" cardinality="single" baseType="string">'
                f"<correctResponse><value>{_xml(s['answer'])}</value></correctResponse></responseDeclaration>"
            )
            notes.append(rid)
    decls.append(
        '<outcomeDeclaration identifier="SCORE" cardinality="single" baseType="float"><defaultValue><value>0</value></defaultValue></outcomeDeclaration>'
    )
    decls.append(
        f'<outcomeDeclaration identifier="MAXSCORE" cardinality="single" baseType="float"><defaultValue><value>{item["marks"]}</value></defaultValue></outcomeDeclaration>'
    )
    if notes:
        body.append(
            "<!-- text answers ("
            + ", ".join(notes)
            + "): the one accepted answer is compared as typed, ignoring case; "
            "an equivalent form is marked wrong, so a teacher should read these responses -->"
        )
    if item["stem"]:
        body.append(f"<p>{_xml(item['stem'])}</p>")
    if item["figure"]:
        alt = _xml(item["figure"].get("alt") or "figure")
        if figure_file and svg:
            body.append(f'<p><img src="{figure_file}" alt="{alt}" width="400"/></p>')
        else:
            body.append(f"<p><em>[Figure: {alt}]</em></p>")
    for e in item["entries"]:
        prefix = f"({_xml(e['label'])}) " if e["label"] is not None else ""
        suffix = f" {_marks_label(e['marks'])}" if e["label"] is not None else ""
        if e["kind"] == "mcq" and e["slots"]:
            s = e["slots"][0]
            body.append(f"<p>{prefix}{_xml(e['text'])}{suffix}</p>")
            body.append(_qti_slot_xml(s, rids[id(s)], 0))
        elif e["table"] and e["cells"]:
            body.append(f"<p>{prefix}{_xml(e['text'])}{suffix}</p>")
            rows = [
                "<table><thead><tr>"
                + "".join(f"<th>{_xml(c)}</th>" for c in e["table"].get("columns") or [])
                + "</tr></thead><tbody>"
            ]
            for r, row in enumerate(e["table"].get("rows") or []):
                cells = []
                for c, cell in enumerate(row):
                    if cell is None and (r, c) in e["cells"]:
                        s = e["cells"][(r, c)]
                        cells.append(f"<td>{_qti_slot_xml(s, rids[id(s)], 0)}</td>")
                    else:
                        cells.append(f"<td>{'' if cell is None else _xml(cell)}</td>")
                rows.append("<tr>" + "".join(cells) + "</tr>")
            rows.append("</tbody></table>")
            body.append("".join(rows))
        else:
            s = e["slots"][0]
            grid = ""
            if e["table"]:
                grid = (
                    "<table><thead><tr>"
                    + "".join(f"<th>{_xml(c)}</th>" for c in e["table"].get("columns") or [])
                    + "</tr></thead><tbody>"
                    + "".join(
                        "<tr>" + "".join(f"<td>{'' if cell is None else _xml(cell)}</td>" for cell in row) + "</tr>"
                        for row in e["table"].get("rows") or []
                    )
                    + "</tbody></table>"
                )
            body.append(f"<p>{prefix}{_xml(e['text'])}{suffix}</p>{grid}<p>{_qti_slot_xml(s, rids[id(s)], 0)}</p>")
    # response processing
    if single and slots[0][1]["kind"] == "mcq":
        rp = '<responseProcessing template="http://www.imsglobal.org/question/qti_v2p1/rptemplates/match_correct"/>'
    else:
        conds = []
        for e, s in slots:
            rid = rids[id(s)]
            weight = e["marks"] / len(e["slots"])
            var, cor = f'<variable identifier="{rid}"/>', f'<correct identifier="{rid}"/>'
            if s["kind"] == "number":
                tol = s["tol"]
                attrs = 'toleranceMode="exact"' if tol == "0" else f'toleranceMode="absolute" tolerance="{tol} {tol}"'
                test = f"<equal {attrs}>{var}{cor}</equal>"
            elif s["kind"] == "text":
                test = f'<stringMatch caseSensitive="false">{var}{cor}</stringMatch>'
            else:
                test = f"<match>{var}{cor}</match>"
            conds.append(
                f"<responseCondition><responseIf>{test}"
                '<setOutcomeValue identifier="SCORE"><sum><variable identifier="SCORE"/>'
                f'<baseValue baseType="float">{weight:g}</baseValue></sum></setOutcomeValue></responseIf></responseCondition>'
            )
        rp = "<responseProcessing>" + "".join(conds) + "</responseProcessing>"
    feedback = _feedback_html_plain(item)
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<assessmentItem xmlns="{_QTI_NS}" xmlns:xsi="{_XSI}" '
        f'xsi:schemaLocation="{_QTI_NS} http://www.imsglobal.org/xsd/qti/qtiv2p1/imsqti_v2p1.xsd" '
        f'identifier="{ident}" title={_qa(str(item["number"]) + ". " + str(item["skill"]))} adaptive="false" timeDependent="false">'
        + "".join(decls)
        + f"<itemBody><div>{''.join(body)}</div></itemBody>"
        + rp
        + f"<!-- answer: {_comment_safe(feedback)} -->"
        + "</assessmentItem>\n"
    )


def _comment_safe(text: str) -> str:
    return _xml(text).replace("--", "- -")


def _feedback_html_plain(item: dict) -> str:
    out = []
    for label, answer, working in _feedback_lines(item):
        prefix = f"({label}) " if label else ""
        out.append(f"{prefix}{answer}" + (" -- " + " ".join(working) if working else ""))
    return " | ".join(out)


def paper_qti21_zip(paper: dict, *, version: str = "A") -> bytes:
    """An IMS QTI 2.1 content package for one version: `imsmanifest.xml`, `test.xml` (an assessmentTest with one
    section per paper section), one assessmentItem per question and any figures as SVG under `figures/`. Multiple
    choice is a choiceInteraction marked by the match_correct template; a number is a float text entry marked with
    an `equal` tolerance; a multi-part question or table is one item with several interactions; every other answer
    kind is a string text entry marked against the one accepted answer, ignoring case."""
    chosen = _one_version(paper, version)
    items = _items(chosen, need_answer=True)
    title = _clean(paper.get("title") or "Worksheet")
    files: dict[str, bytes] = {}
    item_files: list[tuple[dict, str, list[str]]] = []
    fig_no = 0
    for item in items:
        svg = _figure_svg(item["figure"])
        fig_name = None
        if svg:
            fig_no += 1
            fig_name = f"figures/figure-{fig_no}.svg"
            files[fig_name] = svg.encode("utf-8", "replace")
        fname = f"{_ident(item)}.xml"
        files[fname] = _qti_item(item, fig_name, svg).encode("utf-8", "replace")
        item_files.append((item, fname, [fig_name] if fig_name else []))

    sections = []
    i = 0
    for s_index, section in enumerate(chosen.get("sections") or [], 1):
        refs = []
        for _q in section.get("questions") or []:
            item, fname, _figs = item_files[i]
            i += 1
            refs.append(f'<assessmentItemRef identifier="{_ident(item)}" href="{fname}"/>')
        sections.append(
            f'<assessmentSection identifier="section{s_index}" title={_qa(section.get("title") or title)} visible="true">{"".join(refs)}</assessmentSection>'
        )
    test_id = "test_" + _NOT_NAME.sub("_", f"{paper.get('seed', 0)}_{version}")
    files["test.xml"] = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<assessmentTest xmlns="{_QTI_NS}" xmlns:xsi="{_XSI}" '
        f'xsi:schemaLocation="{_QTI_NS} http://www.imsglobal.org/xsd/qti/qtiv2p1/imsqti_v2p1.xsd" '
        f'identifier="{test_id}" title={_qa(title)}>'
        '<outcomeDeclaration identifier="SCORE" cardinality="single" baseType="float"/>'
        '<testPart identifier="part1" navigationMode="nonlinear" submissionMode="individual">'
        + "".join(sections)
        + "</testPart></assessmentTest>\n"
    ).encode("utf-8", "replace")

    resources = [
        f'<resource identifier="RES_{test_id}" type="imsqti_test_xmlv2p1" href="test.xml"><file href="test.xml"/></resource>'
    ]
    for item, fname, figs in item_files:
        deps = "".join(f'<file href="{f}"/>' for f in figs)
        resources.append(
            f'<resource identifier="RES_{_ident(item)}" type="imsqti_item_xmlv2p1" href="{fname}"><file href="{fname}"/>{deps}</resource>'
        )
    files["imsmanifest.xml"] = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<manifest xmlns="http://www.imsglobal.org/xsd/imscp_v1p1" '
        f'xmlns:xsi="{_XSI}" identifier="MANIFEST_{test_id}" '
        'xsi:schemaLocation="http://www.imsglobal.org/xsd/imscp_v1p1 http://www.imsglobal.org/xsd/qti/qtiv2p1/qtiv2p1_imscpv1p2_v1p0.xsd">'
        "<metadata><schema>IMS Content</schema><schemaversion>1.1.4</schemaversion></metadata>"
        "<organizations/><resources>" + "".join(resources) + "</resources></manifest>\n"
    ).encode("utf-8", "replace")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name in ["imsmanifest.xml", "test.xml"] + sorted(
            n for n in files if n not in ("imsmanifest.xml", "test.xml")
        ):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            zf.writestr(info, files[name])
    return buf.getvalue()
