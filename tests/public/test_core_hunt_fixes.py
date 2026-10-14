"""Regressions from the 2026-10-14 core bug hunt: each of these once hung, crashed or misread input."""

import pytest
from fastapi.testclient import TestClient

from mathgen.grading import values
from mathgen.scheduling import prerequisites as graph
from mathgen.server import Config, create_app
from mathgen.server.config import parse_keys
from mathgen.server.pool import WorkerPool
from mathgen.skills import Skill, Source

AUTHOR = "a" * 24
LONG = "k" * 20


def _skill(sid, prerequisites=()):
    return Skill(
        id=sid,
        name=sid,
        domain="arithmetic",
        status="active",
        prerequisites=list(prerequisites),
        answer="number",
        sources=[Source(topic="addition", params={}, difficulty="easy")],
    )


def test_a_prerequisite_cycle_is_an_error_not_a_hang():
    cycle = [_skill("a", ["b"]), _skill("b", ["c"]), _skill("c", ["b"])]
    with pytest.raises(ValueError, match="cycle"):
        graph.gaps_before("a", cycle, [])


@pytest.mark.parametrize("text", ["2 000", "1 000", "3 4", "12 5"])
def test_digits_side_by_side_are_not_multiplied(text):
    assert values.read_value(text) is None


@pytest.mark.parametrize("text, expected", [("2 1/2", "5/2"), ("2sqrt(3)", "2*sqrt(3)"), ("2 sqrt(3)", "2*sqrt(3)")])
def test_mixed_numbers_and_spaced_products_still_read(text, expected):
    assert str(values.read_value(text)) == expected


def test_a_space_after_the_role_colon_is_not_part_of_the_key():
    assert parse_keys("author: " + LONG) == [(LONG, "author")]


def test_a_lone_surrogate_in_a_request_is_a_422_not_a_500():
    pool = WorkerPool(1)
    try:
        with TestClient(create_app(Config(keys=[(AUTHOR, "author")]), pool=pool)) as client:
            r = client.post(
                "/v1/questions",
                content=b'{"skill":"\\ud800"}',
                headers={"Authorization": f"Bearer {AUTHOR}", "content-type": "application/json"},
            )
            assert r.status_code in (404, 422), r.text
    finally:
        pool.close()
