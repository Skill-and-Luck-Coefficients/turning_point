from itertools import zip_longest
from typing import Any, Sequence

from .utils import Round, split_in_middle, sum_sequence_values


def generate_optimal_schedule_between_groups(
    group_one_teams: Sequence[Any], group_two_teams: Sequence[Any]
) -> list[Round]:
    """
    Given two groups, generate the schedule between them.

    Schedule is created by rotating the largest group (sorted in ascending order)
    while maintaining the smallest group intact (sorted in descending order).

    ----
    Example:
        group_one_teams: [0, 1, 2]
        group_two_teams: [3, 4, 5]

        Result:
            [
                ((0, 5), (1, 4), (2, 3)),
                ((2, 5), (0, 4), (1, 3)),
                ((1, 5), (2, 4), (0, 3)),
            ]
    """
    if not group_one_teams or not group_two_teams:
        return []

    largest = sorted(group_one_teams, reverse=False)
    smallest = sorted(group_two_teams, reverse=True)
    if len(group_one_teams) < len(group_two_teams):
        largest, smallest = smallest, largest

    initial_largest = largest.copy()

    schedule: list[Round] = []

    while True:
        # if one half has more teams than the other, zip will ignore it
        round_schedule = zip(smallest, largest)
        lower_number_first = map(sorted, round_schedule)
        matches_as_tuples = map(tuple, lower_number_first)
        schedule.append(tuple(matches_as_tuples))

        # need to rotate the largest, otherwise some matches may be skipped
        largest.insert(0, largest.pop())

        if initial_largest == largest:
            break

    return schedule


def generate_optimal_schedule(
    teams: int | Sequence[Any],
) -> list[Round]:
    """
    Given a list of teams, schedule an entire tournament by splitting
    the list in half and applying `generate_schedule_half_split`.

    ----
    Parameters:
        teams: int | Sequence[int]
            Teams identifiers (integers)

            int: teams being an integer is equivalent to list(range(teams)).

    ----
    Example:
        teams: [0, 1, 2, 3, 4, 5]

        Result:
            [
                ((0, 5), (1, 4), (2, 3)),
                ((2, 5), (0, 4), (1, 3)),
                ((1, 5), (2, 4), (0, 3)),
                ((0, 2), (3, 5)),
                ((1, 2), (4, 5)),
            ]
    """
    if isinstance(teams, int):
        teams = list(range(teams))

    teams = sorted(teams)
    num_teams = len(teams)

    if num_teams <= 1:
        return [tuple()]

    first_half, second_half = split_in_middle(teams)

    schedule: list[Round] = []
    schedule.extend(generate_optimal_schedule_between_groups(first_half, second_half))

    first_half_schedule = generate_optimal_schedule(first_half)
    second_half_schedule = generate_optimal_schedule(second_half)
    both_halves_schedule = [first_half_schedule, second_half_schedule]
    first_and_second: list[Round] = [
        sum_sequence_values(first_second)  # type: ignore
        for first_second in zip_longest(*both_halves_schedule, fillvalue=tuple())
    ]
    schedule.extend(first_and_second)

    return [matches for matches in schedule if matches]