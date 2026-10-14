"""The calls a worker process runs (see `pool.py`). Each takes the worker's `Library` and returns JSON-ready data,
or a `(media type, text or bytes)` pair for a file."""

from __future__ import annotations

from .. import service
from .pool import job

#: Formats that carry the answers (an answer key, or an LMS bank that marks with them).
NEEDS_ANSWERS = frozenset({"html-key", "moodle", "qti"})


@job
def make_questions(lib, **kwargs):
    return service.make_questions(lib, **kwargs)


@job
def question(lib, *, qid, answers):
    return service.question_by_id(lib, qid, answers=answers)


@job
def grade(lib, *, qid, reply, reveal):
    return service.grade_reply(lib, qid, reply, reveal=reveal)


@job
def paper(lib, *, spec, answers, format, version):
    built = service.build_paper(lib, spec, answers=answers or format in NEEDS_ANSWERS)
    if format == "json":
        return built
    from .. import export

    try:
        return _export(export, built, format, version)
    except ValueError as exc:  # a version the paper does not have, say
        raise service.BadRequest(str(exc)) from None


def _export(export, built, format, version):
    if format == "html":
        return "text/html; charset=utf-8", export.paper_html(built, version=version)
    if format == "html-key":
        return "text/html; charset=utf-8", export.paper_html(built, version=version, key=True)
    if format == "moodle":
        return "application/xml", export.paper_moodle_xml(built, version=version or "A")
    if format == "qti":
        return "application/zip", export.paper_qti21_zip(built, version=version or "A")
    raise service.BadRequest(f"unknown format {format!r}")


@job
def figure_svg(lib, *, qid):
    # The learner view's figure is already the picture: `{"alt": ..., "svg": ...}`, never the spec.
    view = service.question_by_id(lib, qid)
    if not view.get("figure"):
        raise service.NotFound(f"question {qid} has no figure")
    return view["figure"]["svg"]
