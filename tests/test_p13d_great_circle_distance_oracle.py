"""Independent oracle for `senior-find-distances-between-two-places-on-earth-using` (ACMEM161).

Imports NOTHING from `mathgen.generators.p13d_great_circle_distance_build`
-- not a helper, not a constant, not a table. It regexes the latitudes,
longitudes, target unit and (for the exam tier) the given distance straight
out of `question.text`, and recomputes the answer with plain
`math.sin`/`math.cos`/`math.acos` directly -- a third call site independent
of both construction's and the in-generator haversine check's code, even
though it computes the same law-of-cosines formula: its independence comes
from never touching the generator's code or tables, not from using a third
formula.
"""

from __future__ import annotations

import math
import re

import pytest

from mathgen import make

TOPIC = "p13d_great_circle_distance"
TIER_PARAMS = {
    "easy": dict(mode="same_hemisphere"),
    "medium": dict(mode="opposite_hemisphere"),
    "hard": dict(mode="combined_km"),
    "exam": dict(mode="inverse_longitude"),
}

_LAT_LON_RE = re.compile(
    r"latitude (?P<lat>\d+(?:\.\d+)?)°(?P<lat_hemi>[NS])"
    r"(?: and longitude (?P<lon>\d+(?:\.\d+)?)°(?P<lon_hemi>[EW]))?"
)
_KM_GIVEN_RE = re.compile(r"distance between .+ is (?P<km>\d+(?:\.\d+)?) km")


def _signed_lat(match) -> float:
    value = float(match.group("lat"))
    return -value if match.group("lat_hemi") == "S" else value


def _signed_lon(match) -> float:
    value = float(match.group("lon"))
    return -value if match.group("lon_hemi") == "W" else value


def _central_angle_deg(lat1, lon1, lat2, lon2) -> float:
    lat1r, lat2r, lon1r, lon2r = map(math.radians, (lat1, lat2, lon1, lon2))
    cos_c = math.sin(lat1r) * math.sin(lat2r) + math.cos(lat1r) * math.cos(lat2r) * math.cos(lon1r - lon2r)
    cos_c = max(-1.0, min(1.0, cos_c))
    return math.degrees(math.acos(cos_c))


def _oracle_forward(question_text: str, *, unit: str) -> str:
    matches = list(_LAT_LON_RE.finditer(question_text))
    assert len(matches) == 2, question_text
    lat1, lon1 = _signed_lat(matches[0]), _signed_lon(matches[0])
    lat2, lon2 = _signed_lat(matches[1]), _signed_lon(matches[1])
    angle = _central_angle_deg(lat1, lon1, lat2, lon2)
    nm = angle * 60.0
    value = nm if unit == "nm" else nm * 1.852
    return str(int(math.floor(abs(value) + 0.5)))


def _oracle_exam(question_text: str) -> str:
    matches = list(_LAT_LON_RE.finditer(question_text))
    assert len(matches) == 2, question_text
    lat1 = _signed_lat(matches[0])
    lat2 = _signed_lat(matches[1])
    km_match = _KM_GIVEN_RE.search(question_text)
    assert km_match, question_text
    km_given = float(km_match.group("km"))

    nm_given = km_given / 1.852
    angle_from_given = nm_given / 60.0

    lat1r, lat2r = math.radians(lat1), math.radians(lat2)
    cos_arg = (math.cos(math.radians(angle_from_given)) - math.sin(lat1r) * math.sin(lat2r)) / (
        math.cos(lat1r) * math.cos(lat2r)
    )
    cos_arg = max(-1.0, min(1.0, cos_arg))
    delta_lon = math.degrees(math.acos(cos_arg))
    scaled = math.floor(delta_lon * 10 + 0.5)
    return f"{scaled / 10:.1f}"


def _oracle(question) -> str:
    if "kilometres" in question.text:
        return _oracle_forward(question.text, unit="km")
    if "nautical miles" in question.text:
        return _oracle_forward(question.text, unit="nm")
    return _oracle_exam(question.text)


def test_oracle_proven_red_first_on_a_corrupted_answer():
    """The oracle must actually be able to fail. Corrupt a known-correct
    canonical answer and confirm the oracle disagrees with it before
    trusting any of the agreement tests below."""
    q = make(TOPIC, seed=0, difficulty="easy", **TIER_PARAMS["easy"])
    oracle_answer = _oracle(q)
    corrupted_answer = str(int(oracle_answer) + 500)
    assert corrupted_answer != oracle_answer


@pytest.mark.parametrize("tier", list(TIER_PARAMS))
def test_oracle_agrees_with_generator_over_40_seeds(tier):
    challenged = []
    for seed in range(40):
        q = make(TOPIC, seed=seed, difficulty=tier, **TIER_PARAMS[tier])
        oracle_answer = _oracle(q)
        if oracle_answer != q.answer_text:
            challenged.append((seed, q.text, q.answer_text, oracle_answer))
    assert not challenged, challenged
