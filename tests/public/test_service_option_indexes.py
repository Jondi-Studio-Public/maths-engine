"""An integer reply to a multiple-choice question is always the option's index, never option text."""

import pytest

from mathgen import service


@pytest.fixture(scope="module")
def lib():
    return service.Library.load()


def test_an_integer_reply_is_the_index_even_when_it_equals_another_options_text(lib):
    # count-to-10 has numeric options, so the digits of an index can be another option's text.
    for seed in range(40):
        (view,) = service.make_questions(lib, "count-to-10", seed=seed, answers=True)
        options, right = view["options"], view["answer"]["option"]
        for index in range(len(options)):
            got = service.grade_reply(lib, view["id"], index)["correct"]
            assert got == (index == right), (seed, options, index)
