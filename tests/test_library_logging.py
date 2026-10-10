"""The library does not print its own log records.

Python writes a warning or an error to stderr when no handler at all is set up for the logger it came from. A
library should leave that to the application, so `mathgen` attaches a `NullHandler` to its own logger: records
from `mathgen.*` are dropped unless the application sets up logging, and reach it normally when it does.
"""

from __future__ import annotations

import contextlib
import io
import logging

import mathgen  # noqa: F401  (importing the package is what attaches the handler)
from mathgen.core import Question
from mathgen.grading import grade


@contextlib.contextmanager
def _no_logging_set_up():
    """The state of a script that has not configured logging: no handler on the root logger (pytest's own is
    taken off for the length of the block), and the last-resort handler, which Python falls back to, writing to
    a stream the test can read."""
    stream = io.StringIO()
    last_resort = logging.StreamHandler(stream)
    last_resort.setLevel(logging.WARNING)
    root = logging.getLogger()
    saved_handlers, saved_last_resort = root.handlers[:], logging.lastResort
    root.handlers = []
    logging.lastResort = last_resort
    try:
        yield stream
    finally:
        root.handlers = saved_handlers
        logging.lastResort = saved_last_resort


def test_the_package_logger_has_a_null_handler():
    assert any(isinstance(h, logging.NullHandler) for h in logging.getLogger("mathgen").handlers)


def test_a_library_error_is_not_printed_when_the_application_set_up_no_logging():
    with _no_logging_set_up() as printed:
        logging.getLogger("mathgen.grading").error("boom")
        logging.getLogger("mathgen.grading").warning("careful")
        logging.getLogger("mathgen").error("boom")
    assert printed.getvalue() == ""


def test_other_loggers_are_unaffected():
    with _no_logging_set_up() as printed:
        logging.getLogger("some_other_library").error("this is for the application's stderr")
    assert "this is for the application's stderr" in printed.getvalue()


def test_an_application_handler_still_receives_the_records():
    seen = io.StringIO()
    handler = logging.StreamHandler(seen)
    logger = logging.getLogger("mathgen")
    logger.addHandler(handler)
    try:
        with _no_logging_set_up() as printed:
            logging.getLogger("mathgen.grading").error("boom")
    finally:
        logger.removeHandler(handler)
    assert "boom" in seen.getvalue()
    assert printed.getvalue() == ""


def test_a_grader_failure_is_logged_for_the_application_and_printed_by_nobody(monkeypatch):
    from mathgen.grading import typed

    def broken(question, given):
        raise RuntimeError("a grader bug")

    monkeypatch.setitem(typed.GRADERS, "number", broken)
    question = Question(
        topic="test", level=0, seed=1, text="q", latex="q", answer=1, answer_text="1", answer_latex="",
        working=[], difficulty="easy", distractors=[],
    )  # fmt: skip
    seen = io.StringIO()
    handler = logging.StreamHandler(seen)
    logger = logging.getLogger("mathgen")
    logger.addHandler(handler)
    try:
        with _no_logging_set_up() as printed:
            assert grade("number", question, "1").correct is False
    finally:
        logger.removeHandler(handler)
    assert "a grader bug" in seen.getvalue()  # the traceback is there for whoever collects logs
    assert printed.getvalue() == ""
