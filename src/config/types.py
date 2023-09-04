from typing import Literal, TypedDict

Sports = list[str]
OptimalScheduleTypes = Literal[
    "tp_minimizer", "tp_maximizer", "tp_minimizer_random", "tp_maximizer_random"
]


class PermutedMatchesParameters(TypedDict):
    """
    num_permutations: int
    """

    num_permutations: int


class PermutedMatches(TypedDict):
    """
    should_create_it: bool
    seed: int
    parameters: PermutedMatchesParameters
    """

    should_create_it: bool
    seed: int
    parameters: PermutedMatchesParameters


class TurningPointParameters(TypedDict):
    """
    num_iteration_simulation: list[int]
    """

    num_iteration_simulation: list[int]


class TurningPointConfig(TypedDict):
    """
    should_calculate_it: bool
    seed: int
    parameters: TurningPointParameters
    """

    should_calculate_it: bool
    seed: int
    parameters: TurningPointParameters


class RealConfig(TypedDict):
    """
    sports: Sports
    turning_point: TurningPointConfig
    """

    sports: Sports
    turning_point: TurningPointConfig


class PermutedConfig(TypedDict):
    """
    sports: Sports
    matches: PermutedMatches
    turning_point: TurningPointConfig
    """

    sports: Sports
    matches: PermutedMatches
    turning_point: TurningPointConfig


class OptimalMatchesParameters(TypedDict):
    """
    types: OptimalScheduleTypes | list[OptimalScheduleTypes]
    """

    types: OptimalScheduleTypes | list[OptimalScheduleTypes]


class OptimalMatchesConfig(TypedDict):
    """
    should_create_it: bool
    seed: int
    parameters: OptimalMatchesParameters
    """

    should_create_it: bool
    seed: int
    parameters: OptimalMatchesParameters


class OptimalConfig(TypedDict):
    """
    sports: Sports
    matches: OptimalMatchesConfig
    turning_point: TurningPointConfig
    """

    sports: Sports
    matches: OptimalMatchesConfig
    turning_point: TurningPointConfig


class ConfigurationType(TypedDict):
    """
    REAL_MATCHES: RealMatchesConfig
    PERMUTED_MATCHES: PermutedConfig
    """

    REAL_MATCHES: RealConfig
    PERMUTED_MATCHES: PermutedConfig
    OPTIMAL_SCHEDULE: OptimalConfig
