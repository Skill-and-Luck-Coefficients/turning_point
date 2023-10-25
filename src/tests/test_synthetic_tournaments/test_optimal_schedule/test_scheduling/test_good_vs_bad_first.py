import pytest

import synthetic_tournaments.optimal_schedule.scheduling.good_vs_bad_first as gbf


@pytest.fixture
def parameters():
    return [
        (["a", "b", "c", "d"], 1),
        (["d", "c", "b", "a"], 2),
        (["a", "b", "c", "d", "e", "f"], 1),
    ]


@pytest.fixture
def all_expected():
    return [
        [
            (("a", "d"), ("b", "c")),
            (("a", "c"), ("b", "d")),
            (("a", "b"), ("c", "d")),
            (("d", "a"), ("c", "b")),
            (("c", "a"), ("d", "b")),
            (("b", "a"), ("d", "c")),
        ],
        [
            (("d", "a"), ("c", "b")),
            (("d", "b"), ("c", "a")),
            (("d", "c"), ("b", "a")),
            (("a", "d"), ("b", "c")),
            (("b", "d"), ("a", "c")),
            (("c", "d"), ("a", "b")),
            (("d", "a"), ("c", "b")),
            (("d", "b"), ("c", "a")),
            (("d", "c"), ("b", "a")),
            (("a", "d"), ("b", "c")),
            (("b", "d"), ("a", "c")),
            (("c", "d"), ("a", "b")),
        ],
        [
            (("a", "f"), ("b", "e"), ("c", "d")),
            (("a", "e"), ("b", "d"), ("c", "f")),
            (("a", "d"), ("b", "f"), ("c", "e")),
            (("a", "c"), ("d", "f")),
            (("a", "b"), ("d", "e")),
            (("b", "c"), ("e", "f")),
            (("f", "a"), ("e", "b"), ("d", "c")),
            (("e", "a"), ("d", "b"), ("f", "c")),
            (("d", "a"), ("f", "b"), ("e", "c")),
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
