from itertools import permutations
from typing import Callable

import networkx as nx
import numpy as np

import synthetic_tournaments.optimal_schedule.algorithm.graph_optimal_schedule as gos


def test_create_weighted_graph__empty():
    empty_result = gos._create_weighted_graph([], lambda x, y: x - y)
    assert len(empty_result.edges) == 0


def test_create_weighted_graph():
    strengths = [0, 1, 2]
    strengths_to_skill_diff = lambda x, y: abs(x - y) ** 2
    expected = {
        (0, 1): {"weight": 1},
        (0, 2): {"weight": 4},
        (1, 2): {"weight": 1},
    }

    result = gos._create_weighted_graph(strengths, strengths_to_skill_diff)
    result_edges = {
        tuple(sorted(edge)): result.get_edge_data(*edge) for edge in result.edges
    }
    assert result_edges == expected

    strengths = [0, 1, 2, 3]
    strengths_to_skill_diff = lambda x, y: abs(x - y)
    expected = {
        (0, 1): {"weight": 1},
        (0, 2): {"weight": 2},
        (0, 3): {"weight": 3},
        (1, 2): {"weight": 1},
        (1, 3): {"weight": 2},
        (2, 3): {"weight": 1},
    }

    result = gos._create_weighted_graph(strengths, strengths_to_skill_diff)
    result_edges = {
        tuple(sorted(edge)): result.get_edge_data(*edge) for edge in result.edges
    }
    assert result_edges == expected


def test_sort_round__empty():
    result = gos._sort_round(nx.Graph(), [])
    assert result == tuple()


def test_sort_round():
    edge_weights = [(0, 1, 4), (0, 2, 3), (1, 2, 3)]
    graph = nx.Graph()
    graph.add_weighted_edges_from(edge_weights)

    all_expected = [
        ((0, 1),),
        ((0, 1), (0, 2)),
        ((0, 2), (1, 2)),
        ((0, 1), (0, 2), (1, 2)),
    ]
    for expected in all_expected:
        for round_ in permutations(expected, r=len(expected)):
            assert expected == gos._sort_round(graph, round_)


def generate_optimal_graph_schedule__empty():
    result = gos.generate_optimal_graph_schedule([])
    assert result == [tuple()]


def generate_optimal_graph_schedule__invariance():

    def is_skill_diff_non_ascending(
        schedule: list[tuple[tuple[int, int]]],
        strengths: list[float],
        skill_diff_fn: Callable[[float, float], float],
    ) -> bool:
        """
        Checks that the total skill discrepancy in round `i` is always bigger than or equal to the one from round `i + 1`.
        """
        skill_diff_per_round = [
            sum(
                skill_diff_fn(strengths[home], strengths[away]) for home, away in round_
            )
            for round_ in schedule
        ]
        return skill_diff_per_round == sorted(skill_diff_per_round, reverse=True)

    skill_diff_fns = [
        lambda x, y: abs(x - y),
        lambda x, y: (x - y) ** 2,
        lambda x, y: np.exp(abs(x - y)),
        lambda x, y: x / (x + y),
    ]

    for skill_diff_fn in skill_diff_fns:
        strenghts = 4
        strenghts_list = list(reversed(range(strenghts)))
        result = gos.generate_optimal_graph_schedule(strenghts, skill_diff_fn)
        assert is_skill_diff_non_ascending(result, strenghts_list, skill_diff_fn)

        strenghts = list(reversed(range(8)))
        result = gos.generate_optimal_graph_schedule(strenghts, skill_diff_fn)
        assert is_skill_diff_non_ascending(result, strenghts_list, skill_diff_fn)


def generate_optimal_graph_schedule():
    expected = [
        ((0, 3), (1, 2)),
        ((0, 2), (1, 3)),
        ((0, 1), (2, 3)),
    ]

    strenghts = 4
    skill_diff_fn = lambda x, y: (x - y) ** 2
    result = gos.generate_optimal_graph_schedule(strenghts, skill_diff_fn)
    assert result == expected

    expected = [
        ((0, 7), (1, 6), (2, 5), (3, 4)),
        ((0, 6), (1, 7), (2, 4), (3, 5)),
        ((0, 5), (2, 7), (1, 4), (3, 6)),
        ((0, 4), (1, 5), (2, 6), (3, 7)),
        ((0, 3), (4, 7), (1, 2), (5, 6)),
        ((0, 2), (1, 3), (4, 6), (5, 7)),
        ((0, 1), (2, 3), (4, 5), (6, 7)),
    ]
    strenghts = 8
    skill_diff_fn = lambda x, y: np.exp(abs(x - y))
    result = gos.generate_optimal_graph_schedule(strenghts, skill_diff_fn)
    assert result == expected
