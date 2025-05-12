from math import prod

import numpy as np
from scipy.optimize import milp

from tournament_simulations.schedules import Round

from .conversion import convert_schedule_list_to_tensor, convert_schedule_tensor_to_list
from .integer_programming import get_bounds, get_constraints, get_ilp_coefficients


def min_break_schedule_from_tensor_schedule(schedule: np.ndarray) -> np.ndarray:
    """
    Finds the break-minimizer first-turn version of `schedule`.

    Parameters:
        schedule (np.ndarray): Tournament schedule in tensor format.
            Shape: [num_matchdays, num_teams, num_teams]
    """

    def _extract_optimal_schedule_from_result() -> np.ndarray:
        optimal_tensor: np.ndarray = np.abs(result.x).astype(int)
        return optimal_tensor[: prod(schedule.shape)].reshape(schedule.shape)

    num_rounds_one_turn = schedule.shape[0]
    num_teams = schedule.shape[1]
    ilp_coefs = get_ilp_coefficients(num_teams, num_rounds_one_turn)

    c = ilp_coefs.get("flat_vector")
    integrality = np.ones_like(c)
    bounds = get_bounds(ilp_coefs, schedule)
    constraints = get_constraints(ilp_coefs, schedule, num_teams, num_rounds_one_turn)
    result = milp(c, integrality=integrality, bounds=bounds, constraints=constraints)

    return _extract_optimal_schedule_from_result()


def min_break_schedule_from_list_schedule(schedule: list[Round]) -> list[Round]:
    """
    Finds the break-minimizer first-turn version of `schedule`.

    Parameters:
        schedule (list[Round]): Tournament schedule in list format.
            ```
            list[                   # Schedule
                tuple[              # Round
                    tuple[int, int] # Match
                ]
            ]
            ```
    """
    schedule_tensor = convert_schedule_list_to_tensor(schedule)
    min_break_tensor = min_break_schedule_from_tensor_schedule(schedule_tensor)
    return convert_schedule_tensor_to_list(min_break_tensor)
