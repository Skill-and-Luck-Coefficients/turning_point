from math import prod

import numpy as np
from scipy.optimize import OptimizeResult, milp

from tournament_simulations.schedules import Round

from .conversion import convert_schedule_list_to_tensor, convert_schedule_tensor_to_list
from .integer_programming import get_bounds, get_constraints, get_ilp_coefficients


def min_break_schedule_from_tensor_schedule(
    schedule: np.ndarray,
    presolve_options: list[bool] = [True, False],
) -> np.ndarray:
    """
    Finds the break-minimizer first-turn version of `schedule`.

    Parameters:
        schedule (np.ndarray): Tournament schedule in tensor format.
            Shape: [num_matchdays, num_teams, num_teams]

        presolve_options: list[bool]
            Which `milp` presolve options to attempt.
    """

    def _extract_optimal_schedule_from_result(_result: OptimizeResult) -> np.ndarray:
        optimal_tensor: np.ndarray = np.abs(_result.x).astype(int)
        return optimal_tensor[: prod(schedule.shape)].reshape(schedule.shape)

    def _validate_optimal_schedule(_optimal_schedule: np.ndarray):
        _symmetric_initial_condition = schedule + schedule.transpose(0, 2, 1)
        _symmetric_result = _optimal_schedule + _optimal_schedule.transpose(0, 2, 1)
        return np.all(_symmetric_initial_condition == _symmetric_result)

    def _get_optimal_schedule():
        """
        Sometimes the solution doesn't work corretly.

        This functions attempts with both 'presolve' options.
        """
        for _presolve in presolve_options:
            _result = milp(c, **params, options={"presolve": _presolve})
            _optimal_schedule = _extract_optimal_schedule_from_result(_result)

            if _validate_optimal_schedule(_optimal_schedule):
                return _optimal_schedule

        return _optimal_schedule

    num_rounds_one_turn = schedule.shape[0]
    num_teams = schedule.shape[1]
    ilp_coefs = get_ilp_coefficients(num_teams, num_rounds_one_turn)

    c = ilp_coefs.get("flat_vector")
    integrality = np.ones_like(c)
    bounds = get_bounds(ilp_coefs, schedule)
    constraints = get_constraints(ilp_coefs, schedule, num_teams, num_rounds_one_turn)

    params = {
        "integrality": integrality,
        "bounds": bounds,
        "constraints": constraints,
    }
    return _get_optimal_schedule()


def min_break_schedule_from_list_schedule(
    schedule: list[Round],
    presolve_options: list[bool] = [True, False],
) -> list[Round]:
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

        presolve_options: list[bool]
            Which `milp` presolve options to attempt.
    """
    tensor = convert_schedule_list_to_tensor(schedule)
    min_break_tensor = min_break_schedule_from_tensor_schedule(tensor, presolve_options)
    return convert_schedule_tensor_to_list(min_break_tensor)
