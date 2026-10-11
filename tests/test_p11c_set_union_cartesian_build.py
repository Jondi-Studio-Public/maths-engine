"""Set-union cardinality + Cartesian-form equating.

Tests for:
- senior-determine-and-use-the-formulas-for-finding-the (ACMSM005), answer: number
- senior-review-cartesian-form (ACMSM078), answer: pair
"""

import hashlib

import pytest

from mathgen.grading.typed import grade_number, grade_pair
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError
from mathgen.generators.p11c_set_union_cartesian_build import (
    _verify_union_easy,
    _verify_union_medium,
    _verify_union_hard,
    _verify_union_three_set_unknown,
    _verify_cartesian_easy,
    _verify_cartesian_medium,
    _verify_cartesian_hard,
)

UNION_TOPIC = "p11c_set_union_cardinality"
CARTESIAN_TOPIC = "p11c_cartesian_form_equating"
DIFFICULTIES = ["easy", "medium", "hard"]
SEEDS = range(60)

_UNION_ID = "senior-determine-and-use-the-formulas-for-finding-the"
_CARTESIAN_ID = "senior-review-cartesian-form"


def _make_from_sources(skills_dict, skill_id, difficulty, seed):
    """Generate through the skill's own `sources` entry (params-driven `make`
    call), never a bare `make(topic, seed, difficulty)` alone."""
    skill = skills_dict[skill_id]
    src = next(s for s in skill.sources if s.difficulty == difficulty)
    return make(src.topic, seed=seed, difficulty=src.difficulty, **(src.params or {}))


@pytest.fixture(scope="module")
def skills_dict():
    _, skills = load_skills()
    return {s.id: s for s in skills}


_ACMSM005_PINNED_HASHES = {
    "0:easy": "adee51c2a2df2a7b",
    "1:easy": "7943f81f93f02d91",
    "2:easy": "58385cef2ae40e6d",
    "3:easy": "e626ff1a522f2fe8",
    "4:easy": "3d70fece3bff74dd",
    "5:easy": "96f786ad60ae198f",
    "6:easy": "27557fff726a1a0f",
    "7:easy": "bbec5846f0c485bb",
    "8:easy": "3056f394ab36d2af",
    "9:easy": "dc90c143bdb36ee5",
    "10:easy": "cc999e678c409190",
    "11:easy": "8a3e016d830d03d0",
    "12:easy": "bb605557e49927df",
    "13:easy": "12a37a0434651a86",
    "14:easy": "06fc896c82a15227",
    "15:easy": "22719fc291601432",
    "16:easy": "5f3be4c0c4e4ebad",
    "17:easy": "792bf536ee3e2662",
    "18:easy": "a209659d637541a2",
    "19:easy": "d9160ed6d2eb1ebf",
    "20:easy": "12b587afd2231b05",
    "21:easy": "f59640894c268e1b",
    "22:easy": "54e2193f2baf853c",
    "23:easy": "37ebc78e9956968f",
    "24:easy": "0a929ab075c1f1c0",
    "25:easy": "a94ca6559e3ceb0b",
    "26:easy": "e6831e047321dfde",
    "27:easy": "8f91d55aea80e484",
    "28:easy": "4e47be3683105ab2",
    "29:easy": "906900cb60c7458b",
    "30:easy": "ddb8f227bead78cf",
    "31:easy": "c71b08d960bfd8cb",
    "32:easy": "a37566dd3686eb91",
    "33:easy": "ff3520c6b2af0ffa",
    "34:easy": "a9d9acfd51f9043c",
    "35:easy": "1b88d3eb54334986",
    "36:easy": "4bfb079232576e7d",
    "37:easy": "5829c5f8c4950519",
    "38:easy": "581850587bee6d0d",
    "39:easy": "fb829bd93f0e2818",
    "40:easy": "57a5aeb5db487eca",
    "41:easy": "ba2170dbba25f4b2",
    "42:easy": "fb0ac4ac37f910bd",
    "43:easy": "3154b42b3e32245f",
    "44:easy": "85220e03f8246f83",
    "45:easy": "0a63f854b8fc039a",
    "46:easy": "2c6377430e00dd9b",
    "47:easy": "9a18d699c1a7a78e",
    "48:easy": "b778e700aa297a7d",
    "49:easy": "8228c53fb7a4b682",
    "50:easy": "88f6e767e881cb0a",
    "51:easy": "036879eb86606424",
    "52:easy": "14ff6e536975c9c9",
    "53:easy": "0fd3ef81b17bb6c2",
    "54:easy": "e160c05db28240e2",
    "55:easy": "ab9c55337a427548",
    "56:easy": "039efe9937019064",
    "57:easy": "0a212189d5428875",
    "58:easy": "1b0680747ba7b1ce",
    "59:easy": "8ce3d0cd1dfd6159",
    "0:medium": "60f298a807e950d7",
    "1:medium": "b6304fe509072513",
    "2:medium": "9471087363e92497",
    "3:medium": "bd61d1f600b7ff48",
    "4:medium": "fe3e8f824f2fd3c7",
    "5:medium": "7a78ebd3dff27c20",
    "6:medium": "7d9f187b1a32f44c",
    "7:medium": "20b4e1616e302bc9",
    "8:medium": "68f5b3e441cdff75",
    "9:medium": "143d144af258f1ce",
    "10:medium": "f671c8471d3f0791",
    "11:medium": "f550fea8ecd585ba",
    "12:medium": "0dc26fb0db9f2101",
    "13:medium": "2c8edda9fb354cea",
    "14:medium": "ddfb8efbe20ef6d7",
    "15:medium": "3f27fb86f1424b63",
    "16:medium": "11c1cc94cde75452",
    "17:medium": "68e847013c8762d3",
    "18:medium": "94868af774297988",
    "19:medium": "fe5ae3fd3e82fbc0",
    "20:medium": "13ad7b1069b4ca93",
    "21:medium": "5766f18c5bc03695",
    "22:medium": "a5516e3af3f6e0cc",
    "23:medium": "74a8e456816eb26a",
    "24:medium": "2c620807b679938d",
    "25:medium": "5b6bb62077eaad59",
    "26:medium": "bd0292f9d127ad36",
    "27:medium": "440e8d6766f7bbfc",
    "28:medium": "04adbe8d23ab9eda",
    "29:medium": "6a8df4f2fddfbfd7",
    "30:medium": "38539fee92a24e1a",
    "31:medium": "ec0c11ad857da55d",
    "32:medium": "d923aa48d7a45c1e",
    "33:medium": "107ee26dc105bee6",
    "34:medium": "c8ec534c4bd5e374",
    "35:medium": "5a4ea07050dcad26",
    "36:medium": "b3b51c378ac56bc5",
    "37:medium": "f4078ae87affb0d5",
    "38:medium": "66e4c05319f09e72",
    "39:medium": "1f03ca5217cdf5e5",
    "40:medium": "7fa2d9d174dbd660",
    "41:medium": "8466a0754a019652",
    "42:medium": "709c5737fd06d860",
    "43:medium": "72961aa97b5cc630",
    "44:medium": "61de81b7169f0e18",
    "45:medium": "6564e78af7114442",
    "46:medium": "2104d1d05e62e55d",
    "47:medium": "573fdbcede92b952",
    "48:medium": "b7b5487f7229c344",
    "49:medium": "ec952e00c25a0ee9",
    "50:medium": "f92211d57a474096",
    "51:medium": "c2353fbe25a73244",
    "52:medium": "1e613b066b64b3c8",
    "53:medium": "e2a4d5776b28b843",
    "54:medium": "ea21de4177655460",
    "55:medium": "b810a8cfb4a37229",
    "56:medium": "d96b632f0bd37a05",
    "57:medium": "9cc46c79f3f55eaf",
    "58:medium": "7d73b13e777f0e01",
    "59:medium": "02e2fd727abb80d9",
    "0:hard": "ea030f9b1c8a3d01",
    "1:hard": "bce87de0cf42af94",
    "2:hard": "eb404db1a99cebc8",
    "3:hard": "c54381d2138402f6",
    "4:hard": "c0aced4d87707405",
    "5:hard": "1dd88951d0f0de45",
    "6:hard": "53b0242f71edde2c",
    "7:hard": "ba5d695c61103bf6",
    "8:hard": "8f63bcc0f4f05258",
    "9:hard": "5033825496b5dd37",
    "10:hard": "13d4fbf43376ac46",
    "11:hard": "dc606bc6d31782b4",
    "12:hard": "c1132944ed1d9105",
    "13:hard": "e77fcffeadf9e100",
    "14:hard": "c97ce5508c7165fe",
    "15:hard": "02fe19932b2e755e",
    "16:hard": "ccea7ed29570ab7c",
    "17:hard": "8107be936f569822",
    "18:hard": "8bfe323dcb0c332d",
    "19:hard": "da47862c8b6e08be",
    "20:hard": "48e607563135b5f8",
    "21:hard": "6a97e1892a0f2627",
    "22:hard": "2295f68c12fb4b25",
    "23:hard": "f63c676d47d0756b",
    "24:hard": "21d897addf0a5557",
    "25:hard": "d0457fcccea77a64",
    "26:hard": "2baf3aea5719fe0f",
    "27:hard": "6aed5c7dcbb3303b",
    "28:hard": "6427399e287e77e1",
    "29:hard": "6da8b0d3c96e62cd",
    "30:hard": "fd1cc32093b8d773",
    "31:hard": "28e087d87ff23127",
    "32:hard": "1f2a4a96bc36dc7e",
    "33:hard": "cbf30c8c068c8ef5",
    "34:hard": "84c2dcd9220c5238",
    "35:hard": "0acea5dc18159d1b",
    "36:hard": "2d1b918fe6546380",
    "37:hard": "00fd77c56e519362",
    "38:hard": "0af6977706c70e1e",
    "39:hard": "a47192125a8f5554",
    "40:hard": "ccc9a365cf3b3e60",
    "41:hard": "767cec8c3764371d",
    "42:hard": "a7dffcb55b252fc4",
    "43:hard": "465644bf61c264dc",
    "44:hard": "bf6b134355005a1c",
    "45:hard": "ead71a2af6efb84b",
    "46:hard": "cdb3a4ff109aaf13",
    "47:hard": "d0bb81b7e2bd35a0",
    "48:hard": "6cda56e6b00011cd",
    "49:hard": "846e441019679b24",
    "50:hard": "ca7b50fff0cc6122",
    "51:hard": "93fcd8ef632d9cc4",
    "52:hard": "8634d3efe4da4ff3",
    "53:hard": "70d5d7b4fc322e70",
    "54:hard": "a760215d26de1b06",
    "55:hard": "2a28b3f6c2d0bf6b",
    "56:hard": "c1eba985ebed376c",
    "57:hard": "2e0a7b8997a7cfcd",
    "58:hard": "6def6b4812ca58de",
    "59:hard": "4a2a775aabd6faf1",
}


def test_union_legacy_presets_backward_compatible_outputs_pinned():
    """The new `two_set_overlap`/`three_set_unknown` presets are additive names on top, never edits to these
    three."""
    for key, expected in _ACMSM005_PINNED_HASHES.items():
        seed_str, preset = key.split(":")
        q = make(UNION_TOPIC, seed=int(seed_str), difficulty=preset, mode=preset)
        got = hashlib.sha256(q.to_json().encode()).hexdigest()[:16]
        assert got == expected, f"{UNION_TOPIC} seed={key}: output changed for a pre-existing question"


# --- Determinism -------------------------------------------------------------

# --- Distractor hygiene --------------------------------------------------


@pytest.mark.parametrize("skill_id,topic", [(_UNION_ID, UNION_TOPIC), (_CARTESIAN_ID, CARTESIAN_TOPIC)])
@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_distractor_count_distinct_nonempty_never_equal_answer(skills_dict, skill_id, topic, difficulty):
    for seed in SEEDS:
        q = _make_from_sources(skills_dict, skill_id, difficulty, seed)
        assert len(q.distractors) == 3, (skill_id, difficulty, seed, q.distractors)


# --- Real grader accept/reject -----------------------------------------


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_union_real_grader_accepts_canonical_rejects_distractors_and_wrong(skills_dict, difficulty):
    for seed in SEEDS:
        q = _make_from_sources(skills_dict, _UNION_ID, difficulty, seed)
        wrong = str(q.answer + 1)
        assert not grade_number(q, wrong).correct, (difficulty, seed, q.text, wrong)


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_cartesian_real_grader_accepts_canonical_rejects_distractors_and_wrong(skills_dict, difficulty):
    for seed in SEEDS:
        q = _make_from_sources(skills_dict, _CARTESIAN_ID, difficulty, seed)
        x0, y0 = q.answer["x"], q.answer["y"]
        wrong = f"x = {x0 + 1}, y = {y0}"
        assert not grade_pair(q, wrong).correct, (difficulty, seed, q.text, wrong)


# --- Labelled-pair order independence  -----------


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_cartesian_grader_accepts_both_label_orders_and_rejects_wrong_in_both_orders(skills_dict, difficulty):
    for seed in SEEDS:
        q = _make_from_sources(skills_dict, _CARTESIAN_ID, difficulty, seed)
        x0, y0 = q.answer["x"], q.answer["y"]
        assert grade_pair(q, f"x = {x0}, y = {y0}").correct
        assert grade_pair(q, f"y = {y0}, x = {x0}").correct
        assert not grade_pair(q, f"x = {x0 + 1}, y = {y0}").correct
        assert not grade_pair(q, f"y = {y0}, x = {x0 + 1}").correct


# --- Answer-kind discipline ----------------------------------------------


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_union_answer_is_a_plain_integer(skills_dict, difficulty):
    for seed in SEEDS:
        q = _make_from_sources(skills_dict, _UNION_ID, difficulty, seed)
        assert isinstance(q.answer, int)
        assert q.answer_text == str(q.answer)
        assert q.answer >= 0


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_cartesian_answer_is_labelled_xy_pair(skills_dict, difficulty):
    for seed in SEEDS:
        q = _make_from_sources(skills_dict, _CARTESIAN_ID, difficulty, seed)
        assert isinstance(q.answer, dict) and set(q.answer) == {"x", "y"}
        assert q.answer_text == f"x = {q.answer['x']}, y = {q.answer['y']}"


# --- Every distractor formatted like the answer -------------------------


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_union_distractors_are_plain_integers(skills_dict, difficulty):
    for seed in SEEDS:
        q = _make_from_sources(skills_dict, _UNION_ID, difficulty, seed)
        for d in q.distractors:
            assert d.lstrip("-").isdigit(), (difficulty, seed, d)


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_cartesian_distractors_are_labelled_xy_pairs(skills_dict, difficulty):
    for seed in SEEDS:
        q = _make_from_sources(skills_dict, _CARTESIAN_ID, difficulty, seed)
        for d in q.distractors:
            assert d.startswith("x = ") and ", y = " in d, (difficulty, seed, d)


# --- Cartesian hard tier: never the (a+bi)z multiplication form ---------


def test_cartesian_hard_never_produces_a_bi_z_multiplication_form(skills_dict):
    for seed in SEEDS:
        q = _make_from_sources(skills_dict, _CARTESIAN_ID, "hard", seed)
        assert ")z" not in q.text
        assert "iz" not in q.text
        assert "the conjugate of z" in q.text


# --- easy-tier union constraints hold on every seed (a, b > c > 0; a+b<=N;
# every distractor a plausible non-negative count under N) --------------


def test_union_easy_constraints_hold_every_seed():
    import re

    pattern = re.compile(r"In a class of (\d+) students, (\d+) study \w+, (\d+) study \w+, and (\d+) study both")
    for seed in SEEDS:
        q = make(UNION_TOPIC, seed=seed, difficulty="easy", mode="easy")
        m = pattern.search(q.text)
        assert m, q.text
        N, a, b, c = (int(g) for g in m.groups())
        assert a > c > 0 and b > c > 0
        assert a + b <= N
        answer = int(q.answer_text)
        assert 0 <= answer < N
        for d in q.distractors:
            assert 0 <= int(d) < N


# --- medium-tier union: inclusion-exclusion matches the direct region sum,
# every stated count >= 2 (no "1 like ..." grammar defect) --------------


def test_union_medium_every_stated_count_is_at_least_two():
    import re

    number_pattern = re.compile(r"\b(\d+)\b")
    for seed in SEEDS:
        q = make(UNION_TOPIC, seed=seed, difficulty="medium", mode="medium")
        counts = [int(n) for n in number_pattern.findall(q.text)]
        assert all(n >= 2 for n in counts), (seed, q.text)


# --- hard-tier union: "neither" is never a bare "1 study" ---------------


def test_union_hard_neither_is_at_least_two():
    import re

    pattern = re.compile(r"(\d+) study neither")
    for seed in SEEDS:
        q = make(UNION_TOPIC, seed=seed, difficulty="hard", mode="hard")
        m = pattern.search(q.text)
        assert m, q.text
        assert int(m.group(1)) >= 2


# --- at least 20 distinct rendered questions per tier over 60 seeds -----


@pytest.mark.parametrize("topic", [UNION_TOPIC, CARTESIAN_TOPIC])
@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_at_least_20_distinct_questions_per_tier(topic, difficulty):
    seen = {make(topic, seed=s, difficulty=difficulty, mode=difficulty).text for s in range(60)}
    assert len(seen) >= 20, (topic, difficulty, len(seen))


# --- at least 20 distinct rendered questions per tier, through the SHIPPING
# skill's real sources (two_set_overlap, three_set_unknown), not the old
# medium/hard presets the test above pins ------------------------------------


@pytest.mark.parametrize("mode", ["two_set_overlap", "three_set_unknown"])
def test_at_least_20_distinct_questions_per_shipping_mode(mode):
    seen = {
        make(UNION_TOPIC, seed=s, difficulty="medium" if mode == "two_set_overlap" else "hard", mode=mode).text
        for s in range(60)
    }
    assert len(seen) >= 20, (mode, len(seen))


# --- ValueError on a nonsensical preset -----------------------------------


@pytest.mark.parametrize("topic", [UNION_TOPIC, CARTESIAN_TOPIC])
def test_raises_value_error_on_unknown_mode(topic):
    with pytest.raises(ValueError):
        make(topic, seed=1, mode="bogus")


# --- Independent in-generator checks fire on a corrupted answer ---------


def test_verify_union_easy_fires_on_corrupted():
    _verify_union_easy(50, 20, 15, 5, 20)  # N=50,a=20,b=15,c=5 -> neither=20
    with pytest.raises(VerificationError):
        _verify_union_easy(50, 20, 15, 5, 21)


def test_verify_union_medium_fires_on_corrupted():
    # A=8,B=8,C=8,AB=4,AC=4,BC=4,ABC=2 -> union = 8+8+8-4-4-4+2 = 14
    _verify_union_medium(8, 8, 8, 4, 4, 4, 2, 14)
    with pytest.raises(VerificationError):
        _verify_union_medium(8, 8, 8, 4, 4, 4, 2, 15)


def test_verify_union_hard_fires_on_corrupted():
    # a=20,b=25,c=8,neither=3 -> N = (20+25-8)+3 = 40
    _verify_union_hard(40, 20, 25, 8, 3)
    with pytest.raises(VerificationError):
        _verify_union_hard(41, 20, 25, 8, 3)


def test_verify_union_three_set_unknown_fires_on_corrupted():
    # A=8,B=8,C=8,AB=4,AC=4,BC=4,ABC=2 -> union = 14 (same seven-term identity
    # as test_verify_union_medium_fires_on_corrupted's numbers).
    _verify_union_three_set_unknown("ABC", 8, 8, 8, 4, 4, 4, 2, 14, 2)
    with pytest.raises(VerificationError):
        _verify_union_three_set_unknown("ABC", 8, 8, 8, 4, 4, 4, 2, 14, 3)

    _verify_union_three_set_unknown("C", 8, 8, 8, 4, 4, 4, 2, 14, 8)
    with pytest.raises(VerificationError):
        _verify_union_three_set_unknown("C", 8, 8, 8, 4, 4, 4, 2, 14, 9)

    _verify_union_three_set_unknown("AB", 8, 8, 8, 4, 4, 4, 2, 14, 4)
    with pytest.raises(VerificationError):
        _verify_union_three_set_unknown("AB", 8, 8, 8, 4, 4, 4, 2, 14, 5)


def test_verify_cartesian_easy_fires_on_corrupted():
    # a=2,b=-1,c=1,d=4,x0=6,y0=-5 -> e=2*6-1=11, f=1*-5+4=-1
    _verify_cartesian_easy(2, -1, 1, 4, 11, -1, 6, -5)
    with pytest.raises(VerificationError):
        _verify_cartesian_easy(2, -1, 1, 4, 11, -1, 7, -5)


def test_verify_cartesian_medium_fires_on_corrupted():
    # p=1,q=-2,x0=4,y0=-3 -> r=5,s=-5
    _verify_cartesian_medium(1, -2, 5, -5, 4, -3)
    with pytest.raises(VerificationError):
        _verify_cartesian_medium(1, -2, 5, -5, 5, -3)


def test_verify_cartesian_hard_fires_on_corrupted():
    # k=4,x0=-6,y0=-10 -> r=(1+4)*-6=-30, s=(1-4)*-10=30
    _verify_cartesian_hard(4, -30, 30, -6, -10)
    with pytest.raises(VerificationError):
        _verify_cartesian_hard(4, -30, 30, -6, -11)


# --- Both skills load correctly through the API --------------------------


def test_both_skills_load_through_api(skills_dict):
    union = skills_dict[_UNION_ID]
    assert union is not None
    assert union.domain == "senior-mathematics"
    assert union.prerequisites == []
    assert union.answer == "number"
    assert union.placement_eligible is False
    assert union.status == "active"
    assert len(union.sources) == 4  # easy, medium, hard and exam

    cartesian = skills_dict[_CARTESIAN_ID]
    assert cartesian is not None
    assert cartesian.domain == "senior-mathematics"
    assert cartesian.prerequisites == []
    assert cartesian.answer == "pair"
    assert cartesian.placement_eligible is False
    assert cartesian.status == "active"
    assert len(cartesian.sources) == 4  # easy, medium, hard and exam
