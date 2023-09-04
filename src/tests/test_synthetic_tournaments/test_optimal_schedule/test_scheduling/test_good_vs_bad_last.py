import pytest

import synthetic_tournaments.optimal_schedule.scheduling.good_vs_bad_last as gbl


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
            (("d", "c"), ("b", "a")),
            (("c", "a"), ("d", "b")),
            (("c", "b"), ("d", "a")),
            (("c", "d"), ("a", "b")),
            (("a", "c"), ("b", "d")),
            (("b", "c"), ("a", "d")),
        ],
        [
            (("d", "c"), ("b", "a")),
            (("c", "a"), ("d", "b")),
            (("c", "b"), ("d", "a")),
            (("c", "d"), ("a", "b")),
            (("a", "c"), ("b", "d")),
            (("b", "c"), ("a", "d")),
            (("d", "c"), ("b", "a")),
            (("c", "a"), ("d", "b")),
            (("c", "b"), ("d", "a")),
            (("c", "d"), ("a", "b")),
            (("a", "c"), ("b", "d")),
            (("b", "c"), ("a", "d")),
        ],
        [
            (("f", "e"), ("c", "b")),
            (("e", "d"), ("b", "a")),
            (("f", "d"), ("c", "a")),
            (("d", "a"), ("e", "c"), ("f", "b")),
            (("d", "b"), ("e", "a"), ("f", "c")),
            (("d", "c"), ("e", "b"), ("f", "a")),
            (("e", "f"), ("b", "c")),
            (("d", "e"), ("a", "b")),
            (("d", "f"), ("a", "c")),
            (("a", "d"), ("c", "e"), ("b", "f")),
            (("b", "d"), ("a", "e"), ("c", "f")),
            (("c", "d"), ("b", "e"), ("a", "f")),
        ],
    ]


def test_create_double_rr(parameters, all_expected):
    for parameter, expected in zip(parameters, all_expected):
        assert gbl.create_double_rr(*parameter) == expected


def test_create_double_rr_random(parameters, all_expected):
    for (team_names, num_schedules), expected in zip(parameters, all_expected):
        result = gbl.create_double_rr(team_names, num_schedules)

        # make sure that all matches happened twice per schedule
        for round in result:
            for home, away in round:
                num_matches = sum(
                    ((home, away) in expected_round) ^ ((away, home) in expected_round)
                    for expected_round in expected
                )
                assert num_matches == 2 * num_schedules
