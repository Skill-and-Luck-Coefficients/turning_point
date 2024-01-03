from typing import Literal, Mapping, TypedDict

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
    winner_type = Literal["winner", "result"]
    winner_to_points = Mapping[str, tuple[float, float]]
    """

    num_iteration_simulation: list[int]
    winner_type: Literal["winner", "result"]
    winner_to_points: Mapping[str, tuple[float, float]]


class TurningPointConfig(TypedDict):
    """
    should_calculate_it: bool
    seed: int | list[int]
    quantile: float | list[float]
    parameters: TurningPointParameters
    """

    should_calculate_it: bool
    seed: int | list[int]
    quantile: float | list[float]
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
    REAL_MATCHES: RealConfig
    PERMUTED_MATCHES: PermutedConfig
    OPTIMAL_SCHEDULE: OptimalConfig
    DIFFERENT_POINT_SYSTEM: RealConfig
    DIFFERENT_QUANTILE: RealConfig
    """

    REAL_MATCHES: RealConfig
    PERMUTED_MATCHES: PermutedConfig
    OPTIMAL_SCHEDULE: OptimalConfig
    DIFFERENT_POINT_SYSTEM: RealConfig
    DIFFERENT_QUANTILE: RealConfig
