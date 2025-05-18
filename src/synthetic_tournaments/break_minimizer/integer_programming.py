from itertools import combinations
from typing import Literal

import numpy as np
from scipy.optimize import Bounds, LinearConstraint

from .utils import stack_all_flattened

ILPCoefficientKeys = Literal["schedule", "home_breaks", "away_breaks", "flat_vector"]
ILPCoefficients = dict[ILPCoefficientKeys, np.ndarray]


def get_ilp_coefficients(num_teams: int, num_rounds_one_turn: int) -> ILPCoefficients:
    """
    Get Integer Linear Programming coefficients for minimization.

    Minimization: minimize number of breaks (home and away).

    Coefficients:
        schedule: not in minimization. S_{t, i, j} - team i faced j as the home team in matchday t.
        home_breaks: in minimization. h_{t, i} - home break for team i in between matchdays (t-1) and t.
        away_breaks: in minimization. a_{t, i} - away break for team i in between matchdays (t-1) and t.
        flat_vector: flat version of stack(schedule, home_breaks, away_breaks)
    """

    def _break_coefs():
        """
        Since h_{t, i} and a_{t, i} compare the matchdays (t-1) and t, there is t=0 is not necessary.
        """
        empty_break_matchday = np.zeros((1, num_teams))
        valid_break_matchdays = np.ones((num_rounds_one_turn - 1, num_teams))

        return np.vstack((empty_break_matchday, valid_break_matchdays))

    schedule_coefs = np.zeros((num_rounds_one_turn, num_teams, num_teams))
    home_break_coefs = _break_coefs()
    away_break_coefs = _break_coefs()

    flat_coefs = stack_all_flattened(schedule_coefs, home_break_coefs, away_break_coefs)
    return {
        "schedule": schedule_coefs,
        "home_breaks": home_break_coefs,
        "away_breaks": away_break_coefs,
        "flat_vector": flat_coefs,
    }


def get_bounds(
    coefficients: ILPCoefficients,
    initial_condition: np.ndarray,
) -> Bounds:
    """
    Gets lower and upper bounds for ILPCoefficients.

        Lower_bounds: always 0.

        Upper_bounds: 0 if the coefficients shouldn't be changed, 1 otherwise.

            (1) Schedule: 1 when a match happend in that matchday in `initial_condition`.
            (2) Breaks: 1 when a break can happen.
    """

    def _schedule_upper_bound():
        """
        Given the initial schedule (when matches happens), allows home/away to be available for both teams.

        Available: 1 as upper bound.
        Unavailable: 0 as upper bound.
        """
        symmetric_tensor = initial_condition + initial_condition.transpose(0, 2, 1)
        return symmetric_tensor.flatten()

    def _upper_bounds():
        schedule_upper_bound = _schedule_upper_bound()
        home_break_upper_bound = np.copy(coefficients.get("home_breaks"))
        away_break_upper_bound = np.copy(coefficients.get("away_breaks"))
        return stack_all_flattened(
            schedule_upper_bound, home_break_upper_bound, away_break_upper_bound
        )

    upper_bound = _upper_bounds()
    lower_bound = np.zeros_like(upper_bound)
    return Bounds(lower_bound, upper_bound)


def get_constraints(
    coefficients: ILPCoefficients,
    initial_condition: np.ndarray,
    num_teams: int,
    num_round_one_turn: int,
) -> LinearConstraint:
    """
    Get ILP constraints.
    """

    def _create_empty_constraints():
        return {
            "schedule": np.zeros_like(coefficients.get("schedule")),
            "home_breaks": np.zeros_like(coefficients.get("home_breaks")),
            "away_breaks": np.zeros_like(coefficients.get("away_breaks")),
        }

    def _add_home_break_constraints():
        """
        Adds one to 'home_break' count if team `i` played home two rounds in a row.

        sum_{j} (S_{t - 1, i, j} + S_{t, i, j}) = 1 + h_{t, i}
        """
        for _team_i in range(num_teams):
            for _round in range(1, num_round_one_turn):

                _constraints = _create_empty_constraints()
                _constraints["schedule"][_round - 1, _team_i, :] = 1
                _constraints["schedule"][_round, _team_i, :] = 1
                _constraints["home_breaks"][_round, _team_i] = -1
                _coefficients = stack_all_flattened(*_constraints.values())

                constraint_lower_bound.append(0)
                constraint_upper_bound.append(1)
                constraint_coefficients.append(_coefficients)

    def _add_away_break_constraints():
        """
        Adds one to 'away_break' count if team `i` played away two rounds in a row.

        sum_{j} (S_{t - 1, j, i} + S_{t, j, i}) = 1 + a_{t, i}
        """
        for _team_i in range(num_teams):
            for _round in range(1, num_round_one_turn):

                _constraints = _create_empty_constraints()
                _constraints["schedule"][_round - 1, :, _team_i] = 1
                _constraints["schedule"][_round, :, _team_i] = 1
                _constraints["away_breaks"][_round, _team_i] = -1
                _coefficients = stack_all_flattened(*_constraints.values())

                constraint_lower_bound.append(0)
                constraint_upper_bound.append(1)
                constraint_coefficients.append(_coefficients)

    def _add_one_match_per_round_constraints():
        """
        Limits each team to play at most once in a matchday.

        sum_{j} S_{t, j, i} = 1
        """
        for _team_i in range(num_teams):
            for _round in range(1, num_round_one_turn):

                _constraints = _create_empty_constraints()
                _constraints["schedule"][_round, _team_i, :] = 1
                _coefficients = stack_all_flattened(*_constraints.values())

                constraint_lower_bound.append(0)
                constraint_upper_bound.append(1)
                constraint_coefficients.append(_coefficients)

    def _add_home_or_away_constraints():
        """
        If a match should happen in round t according to `initial_condition`, guarantees that it either happens home or away.

        S_{t, i, j} + S_{t, j, i} = `initial_condition`_{t, i, j}
        """

        for _team_i, _team_j in combinations(range(num_teams), r=2):
            for _round in range(num_round_one_turn):

                if symmetric_schedule_tensor[_round, _team_i, _team_j] == 0:
                    continue

                _constraints = _create_empty_constraints()
                _constraints["schedule"][_round, _team_i, _team_j] = 1
                _constraints["schedule"][_round, _team_j, _team_i] = 1
                _coefficients = stack_all_flattened(*_constraints.values())

                constraint_lower_bound.append(1)
                constraint_upper_bound.append(1)
                constraint_coefficients.append(_coefficients)

    constraint_lower_bound = []
    constraint_upper_bound = []
    constraint_coefficients = []
    symmetric_schedule_tensor = initial_condition + initial_condition.transpose(0, 2, 1)

    _add_home_break_constraints()
    _add_away_break_constraints()
    _add_one_match_per_round_constraints()
    _add_home_or_away_constraints()

    lower_bound = np.array(constraint_lower_bound)
    upper_bound = np.array(constraint_upper_bound)
    coefficients = np.vstack(constraint_coefficients)
    return LinearConstraint(coefficients, lower_bound, upper_bound)
