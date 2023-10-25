from itertools import zip_longest
from typing import Any, Sequence

from .utils import Round, split_in_middle


def generate_optimal_schedule_between_groups(
    group_one_teams: Sequence[Any], group_two_teams: Sequence[Any]
) -> list[Round]:
    """
    Given two groups, generate the schedule between them.

    Schedule is created by rotating the largest group (sorted in ascending order)
    while maintaining the smallest group intact (sorted in descending order).

    ----
    Example:
        group_one_teams: [0, 1, 2] -> Good teams
        group_two_teams: [3, 4, 5] -> Bad teams

        Result:
            [
                ((0, 5), (1, 4), (2, 3)),
                ((0, 4), (1, 3), (2, 5)),
                ((0, 3), (1, 5), (2, 4)),
            ]
    """
    if not group_one_teams or not group_two_teams:
        return []

    smallest = sorted(group_one_teams)
    largest = sorted(group_two_teams)
    if len(largest) < len(smallest):
        largest, smallest = smallest, largest

    largest = list(reversed(largest))
    initial_largest = largest.copy()

    schedule: list[Round] = []

    while True:
        # if one half has more teams than the other, zip will ignore it
        round_schedule = zip(smallest, largest)
        lower_number_first = map(sorted, round_schedule)
        matches_as_tuples = map(tuple, lower_number_first)
        schedule.append(tuple(matches_as_tuples))

        # need to left rotate the largest, otherwise some matches may be skipped
        largest.append(largest.pop(0))

        if initial_largest == largest:
            break

    return schedule


def generate_optimal_schedule(
    teams: int | Sequence[Any],
) -> list[Round]:
    """
    Given a list of teams, schedule an entire tournament by splitting
    the list in half and applying `generate_optimal_schedule_between_groups`.

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
                ((0, 4), (1, 3), (2, 5)),
                ((0, 3), (1, 5), (2, 4)),
                ((0, 2), (3, 5)),
                ((0, 1), (3, 4)),
                ((1, 2), (4, 5)),
            ]
    """
    if isinstance(teams, int):
        teams = list(range(teams))

    if len(teams) <= 1:
        return [tuple()]

    teams = sorted(teams)
    first_half, second_half = split_in_middle(teams)

    schedule: list[Round] = []
    schedule.extend(generate_optimal_schedule_between_groups(first_half, second_half))

    first_half_rounds = generate_optimal_schedule(first_half)
    second_half_rounds = generate_optimal_schedule(second_half)
    both_halves = zip_longest(first_half_rounds, second_half_rounds, fillvalue=tuple())
    joined_on_rounds: list[Round] = [first + second for first, second in both_halves]
    schedule.extend(joined_on_rounds)

    return [matches for matches in schedule if matches]  # Remove empty entries
