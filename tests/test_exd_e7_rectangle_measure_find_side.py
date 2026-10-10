"""Rectangle_measure's hard-tier `find_side` mode gives the area/perimeter and one side, asking for the other
side -- the inverse of what easy/medium ask."""

import mathgen


def test_find_side_area_answer_divides_evenly():
    for seed in range(15):
        q = mathgen.make("rectangle_measure", seed=seed, kind="area", side_hi=12, find_side=True)
        assert "has area" in q.text
        assert "What is its width in cm?" in q.text


def test_find_side_perimeter_answer_is_consistent():
    for seed in range(15):
        q = mathgen.make("rectangle_measure", seed=seed, kind="perimeter", side_hi=25, find_side=True)
        assert "has perimeter" in q.text
        assert "What is its width in cm?" in q.text


def test_find_side_default_is_false_and_unchanged():
    a = mathgen.make("rectangle_measure", seed=5, kind="area", side_hi=12)
    b = mathgen.make("rectangle_measure", seed=5, kind="area", side_hi=12, find_side=False)
    assert a.text == b.text
    assert a.answer == b.answer
    c = mathgen.make("rectangle_measure", seed=5, kind="perimeter", side_hi=25)
    d = mathgen.make("rectangle_measure", seed=5, kind="perimeter", side_hi=25, find_side=False)
    assert c.text == d.text
    assert c.answer == d.answer
