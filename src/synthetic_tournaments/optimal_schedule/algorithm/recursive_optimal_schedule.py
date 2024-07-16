from itertools import zip_longest
from typing import Sequence

from .utils import Round, split_in_middle


def _parse_groups(
    group_one: Sequence[int], group_two: Sequence[int]
) -> tuple[Sequence[int], Sequence[int]]:
    """
    ----
    Returns:
        tuple[
            list[int],  # Sorted smallest group
            list[int],  # Sorted largest group
        ]
    """
    smallest = sorted(group_one)
    largest = sorted(group_two)

    if len(largest) < len(smallest):
        largest, smallest = smallest, largest

    return smallest, largest


def generate_optimal_schedule_between_groups(
    group_one_teams: Sequence[int], group_two_teams: Sequence[int]
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

    smallest, largest = _parse_groups(group_one_teams, group_two_teams)

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


def generate_recursive_optimal_schedule(rankings: int | Sequence[float]) -> list[Round]:
    """
    Given a list of rankings, schedule an entire tournament by splitting
    the list in half and applying `generate_optimal_schedule_between_groups`.

    ----
    Parameters:
        rankings: int | Sequence[int]
            int: Number of teams.
                Equivalent to list(range(rankings)).

            Sequence[float]: Team rankings.
                Only order matters, the smallest the number the better the team.
    ----
    Example:
        rankings: 6

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
    if isinstance(rankings, int):
        rankings = list(range(rankings))

    if len(rankings) <= 1:
        return [tuple()]

    rankings = sorted(rankings)
    first_half, second_half = split_in_middle(rankings)

    schedule: list[Round] = []
    schedule.extend(generate_optimal_schedule_between_groups(first_half, second_half))

    first_half_rounds = generate_recursive_optimal_schedule(first_half)
    second_half_rounds = generate_recursive_optimal_schedule(second_half)

    both_halves = zip_longest(first_half_rounds, second_half_rounds, fillvalue=tuple())
    joined_on_rounds: list[Round] = [first + second for first, second in both_halves]
    schedule.extend(joined_on_rounds)

    return [matches for matches in schedule if matches]  # Remove empty entries
