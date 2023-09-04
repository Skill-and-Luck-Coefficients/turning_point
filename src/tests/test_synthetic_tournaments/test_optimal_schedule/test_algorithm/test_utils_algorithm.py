import synthetic_tournaments.optimal_schedule.algorithm.utils as utils


def test_split_in_middle():
    all_expected = [
        ([], []),
        ([], [0]),
        ([0], [1]),
        ([0], [1, 2]),
        ([0, 1], [2, 3]),
        ([0, 1], [2, 3, 4]),
        ([0, 1, 2], [3, 4, 5]),
        ([0, 1, 2], [3, 4, 5, 6]),
        ([0, 1, 2, 3], [4, 5, 6, 7]),
        ([0, 1, 2, 3], [4, 5, 6, 7, 8]),
    ]

    for integer, expected in zip(range(10), all_expected):
        assert utils.split_in_middle(list(range(integer))) == expected


def test_sum_sequence_values():
    parameters = [([0, 1], [3, 4]), ([0, 1, 2], [3, 4, 5])]
    all_expected = [[0, 1, 3, 4], [0, 1, 2, 3, 4, 5]]

    for parameter, expected in zip(parameters, all_expected):
        assert utils.sum_sequence_values(parameter) == expected
