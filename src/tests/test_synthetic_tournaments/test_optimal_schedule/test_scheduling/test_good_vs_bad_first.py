import pytest

import synthetic_tournaments.optimal_schedule.scheduling.good_vs_bad_first as gbf


@pytest.fixture
def parameters():
    return [
        (["a", "b", "c", "d"], 1),
        (["a", "b", "c", "d"], 2),
        (["a", "b", "c", "d", "e", "f"], 1),
    ]


@pytest.fixture
def all_expected():
    return [
        [
            (("a", "d"), ("b", "c")),
            (("b", "d"), ("a", "c")),
            (("a", "b"), ("c", "d")),
            (("d", "a"), ("c", "b")),
            (("d", "b"), ("c", "a")),
            (("b", "a"), ("d", "c")),
        ],
        [
            (("a", "d"), ("b", "c")),
            (("b", "d"), ("a", "c")),
            (("a", "b"), ("c", "d")),
            (("d", "a"), ("c", "b")),
            (("d", "b"), ("c", "a")),
            (("b", "a"), ("d", "c")),
            (("a", "d"), ("b", "c")),
            (("b", "d"), ("a", "c")),
            (("a", "b"), ("c", "d")),
            (("d", "a"), ("c", "b")),
            (("d", "b"), ("c", "a")),
            (("b", "a"), ("d", "c")),
        ],
        [
            (("a", "f"), ("b", "e"), ("c", "d")),
            (("c", "f"), ("a", "e"), ("b", "d")),
            (("b", "f"), ("c", "e"), ("a", "d")),
            (("a", "c"), ("d", "f")),
            (("a", "b"), ("d", "e")),
            (("b", "c"), ("e", "f")),
            (("f", "a"), ("e", "b"), ("d", "c")),
            (("f", "c"), ("e", "a"), ("d", "b")),
            (("f", "b"), ("e", "c"), ("d", "a")),
            (("c", "a"), ("f", "d")),
            (("b", "a"), ("e", "d")),
            (("c", "b"), ("f", "e")),
        ],
    ]


def test_create_double_rr(parameters, all_expected):
    for parameter, expected in zip(parameters, all_expected):
        assert gbf.create_double_rr(*parameter) == expected


def test_create_double_rr_random(parameters, all_expected):
    for (team_names, num_schedules), expected in zip(parameters, all_expected):
        result = gbf.create_double_rr(team_names, num_schedules)

        # make sure that all matches happened twice per schedule
        for round in result:
            for home, away in round:
                num_matches = sum(
                    ((home, away) in expected_round) ^ ((away, home) in expected_round)
                    for expected_round in expected
                )
                assert num_matches == 2 * num_schedules
