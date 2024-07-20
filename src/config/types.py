from typing import Literal, Mapping, TypedDict

Sports = list[str]

AlgType = Literal["graph", "recursive"]
TPType = Literal["tp_minimizer", "tp_maximizer"]
ScheduleType = Literal[
    "mirrored",
    "reversed",
    "random_mirrored",
    "random_reversed",
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
    metric: str | list[str]
    parameters: TurningPointParameters
    """

    should_calculate_it: bool
    seed: int | list[int]
    quantile: float | list[float]
    metric: str | list[str]
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


OptimalMatchesTypeParameter = dict[
    AlgType,
    dict[TPType, ScheduleType | list[ScheduleType]],
]


class OptimalMatchesParameters(TypedDict):
    """
    types: dict[, OptimalMatchesTPParameter]
    """

    types: OptimalMatchesTypeParameter


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


class BradleyTerryMatchesParameters(TypedDict):
    """
    strengths: list[float]
    n_different_results: int
    n_permutations_per_result: int
    number_of_drr: int
    """

    strengths: list[float]
    n_different_results: int
    n_permutations_per_result: int
    number_of_drr: int


class BradleyTerryMatchesConfig(TypedDict):
    """
    should_create_it: bool
    seed: int
    parameters: dict[str, BradleyTerryMatchesParameters]
    """

    should_create_it: bool
    seed: int
    parameters: dict[str, BradleyTerryMatchesParameters]


class BradleyTerryConfig(TypedDict):
    """
    sports: Sports
    matches: BradleyTerryMatchesConfig
    turning_point: TurningPointConfig
    """

    sports: Sports
    matches: BradleyTerryMatchesConfig
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
    BRADLEY_TERRY: BradleyTerryConfig
