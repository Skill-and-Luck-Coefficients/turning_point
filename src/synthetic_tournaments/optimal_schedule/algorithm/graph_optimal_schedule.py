from itertools import combinations
from typing import Callable, Iterable, Sequence

import networkx as nx

from .utils import Round


def _create_weighted_graph(
    strengths: Iterable[float], skill_diff_fn: Callable[[float, float], float]
) -> nx.Graph:
    """
    Weights: skill discrepancy between the teams.
    """
    team_ids = list(range(len(strengths)))

    matches = combinations(team_ids, r=2)
    matches_and_skill_diff = (
        (home, away, skill_diff_fn(strengths[home], strengths[away]))
        for home, away in matches
    )

    graph = nx.Graph()
    graph.add_weighted_edges_from(matches_and_skill_diff)
    return graph


def _sort_round(graph: nx.Graph, matches: Iterable[tuple[int, int]]):
    """
    Sort priority:
        - Largest skill discrepancy
        - Strongest team
    """

    def _extract_weight(edge: tuple[int, int]) -> float:
        return graph.get_edge_data(*edge)["weight"]

    best_as_home = (tuple(sorted(match)) for match in matches)
    sorted_by_best = sorted(best_as_home, key=lambda match: match[0])
    return tuple(sorted(sorted_by_best, key=_extract_weight, reverse=True))


def generate_optimal_graph_schedule(
    strengths: int | Sequence[float],
    skill_diff_fn: Callable[[float, float], float] = lambda x, y: (x - y) ** 2,
) -> list[Round]:
    """
    Greedy algorithm that finds the maximum skill discrepancy between pairs of teams and se

    ----
    Parameters:
        strengths: int | Iterable[float]
            int: Number of teams.
                Considers the rankings (integers) as the teams strengths.
            Iterable[float]: Team strengths.
                Remark: Results consider strengths sorted in descending order!

        skill_diff_fn: Callable[
            [float, float],  # Strenghts: [strength of the best team, strength of the other team]
            float            # Skill Discrepancy (non-negative)
        ]
            Returns the skill discrepancy between the teams: larger values means larges discrepancy.

    ----
    Returns:
        list[
            tuple[int, int]  # Matches (tuple of identifiers)
        ]
            Team identifier: team strength ranking
                Index for their respective (descending-sorted) `strengths`.
                    Best team: 0
                    Second best: 1
                    ...
                    Worst team: number_of_teams - 1
    """
    if isinstance(strengths, int):
        strengths = list(range(strengths))

    if len(strengths) <= 1:
        return [tuple()]

    strengths = sorted(strengths, reverse=True)
    graph = _create_weighted_graph(strengths, skill_diff_fn)

    schedule = []

    while graph.edges:
        next_round = nx.matching.max_weight_matching(graph)
        schedule.append(_sort_round(graph, next_round))

        graph.remove_edges_from(next_round)

    return schedule
