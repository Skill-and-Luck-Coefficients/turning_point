from random import sample

import synthetic_tournaments.optimal_schedule.algorithm.recursive_optimal_schedule as opt


def test_generate_optimal_schedule_between_groups():
    parameters = [
        ([], []),
        ([], [0]),
        ([0], []),
        ([0], [1]),
        ([0, 1], [2, 3]),
        ([0, 2], [3, 1]),
        (["0", "2"], ["3", "1"]),
        ([0, 1, 2], [3, 4, 5]),
    ]

    all_expected = [
        [],
        [],
        [],
        [((0, 1),)],
        [((0, 3), (1, 2)), ((0, 2), (1, 3))],
        [((0, 3), (1, 2)), ((0, 1), (2, 3))],
        [(("0", "3"), ("1", "2")), (("0", "1"), ("2", "3"))],
        [((0, 5), (1, 4), (2, 3)), ((0, 4), (1, 3), (2, 5)), ((0, 3), (1, 5), (2, 4))],
    ]

    for parameter, expected in zip(parameters, all_expected):
        assert opt.generate_optimal_schedule_between_groups(*parameter) == expected


def test_generate_optimal_schedule_between_groups_determinism():
    param1 = [0, 1, 2]
    param2 = [3, 4, 5]
    expected = opt.generate_optimal_schedule_between_groups(param1, param2)

    for _ in range(5):
        sampled1 = sample(param1, k=len(param1))
        sampled2 = sample(param2, k=len(param2))

        result = opt.generate_optimal_schedule_between_groups(sampled1, sampled2)
        assert result == expected


def test_generate_optimal_schedule():
    parameters = [-1, 0, 1, 2, 3, 4, 5, 6, ["a", "b", "c", "d"]]

    all_expected = [
        [tuple()],
        [tuple()],
        [tuple()],
        [((0, 1),)],
        [((0, 2),), ((0, 1),), ((1, 2),)],
        [((0, 3), (1, 2)), ((0, 2), (1, 3)), ((0, 1), (2, 3))],
        [
            ((0, 4), (1, 3)),
            ((0, 3), (1, 2)),
            ((0, 2), (1, 4)),
            ((0, 1), (2, 4)),
            ((2, 3),),
            ((3, 4),),
        ],
        [
            ((0, 5), (1, 4), (2, 3)),
            ((0, 4), (1, 3), (2, 5)),
            ((0, 3), (1, 5), (2, 4)),
            ((0, 2), (3, 5)),
            ((0, 1), (3, 4)),
            ((1, 2), (4, 5)),
        ],
        [(("a", "d"), ("b", "c")), (("a", "c"), ("b", "d")), (("a", "b"), ("c", "d"))],
    ]

    for parameter, expected in zip(parameters, all_expected):
        assert opt.generate_recursive_optimal_schedule(parameter) == expected


def test_generate_optimal_schedule_determinism():
    for integer in range(20):
        expected = opt.generate_recursive_optimal_schedule(integer)

        list_teams = list(range(integer))
        assert expected == opt.generate_recursive_optimal_schedule(list_teams)

        # Is this desired behaviour?
        random_list = sample(list_teams, k=len(list_teams))
        assert expected == opt.generate_recursive_optimal_schedule(random_list)
