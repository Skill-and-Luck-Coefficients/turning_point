import numpy as np
import pandas as pd

from tournament_simulations.schedules import Round


def convert_schedule_list_to_tensor(schedule: list[Round]) -> np.ndarray:
    """
    Converts a double round-robin schedule in tensor format to list format.

    Parameters:
        schedule (list[Round]): Schedule in list format.
            ```
            list[                   # Schedule
                tuple[              # Round
                    tuple[int, int] # Match
                ]
            ]
            ```

    Returns:
        Schedule (np.ndarray):
            Tensor S_{t, i, j}. <br>
            Team i faced team j as the home-team in matchday t.
    """

    def _num_teams():
        _unique_teams = set()

        for round in schedule:
            for home, away in round:
                _unique_teams.add(home)
                _unique_teams.add(away)

        return len(_unique_teams)

    num_rounds = len(schedule)
    num_teams = _num_teams()

    schedule_tensor = np.zeros((num_rounds, num_teams, num_teams), dtype=int)

    for round_number, round in enumerate(schedule):
        for home, away in round:
            schedule_tensor[round_number, home, away] = 1

    return schedule_tensor


def convert_schedule_tensor_to_list(schedule: np.ndarray) -> list[Round]:
    """
    Converts a double round-robin schedule in tensor format to list format.

    Parameters:
        schedule (np.ndarray):
            Tensor S_{t, i, j}. <br>
            Team i faced team j as the home-team in matchday t.

    Returns:
        Schedule (list[Round]): Schedule in list format.
            ```
            list[                   # Schedule
                tuple[              # Round
                    tuple[int, int] # Match
                ]
            ]
            ```
    """

    def _non_zero_schedule():
        """
        Returns:
            Schedule (list[tuple]): Entries where tensor is non-zero
            ```
            list[
                tuple[
                    int,    # Matchday
                    int,    # Home team
                    int,    # Away team
                ]
            ]
            ```
        """
        return list(zip(*schedule.nonzero()))

    non_zero_schedule = _non_zero_schedule()
    num_matchdays = max(value[0] for value in non_zero_schedule)

    schedule_list = [[] for _ in range(num_matchdays + 1)]

    for matchday, home, away in non_zero_schedule:
        schedule_list[matchday].append((home, away))

    return [tuple(sorted(round, key=min)) for round in schedule_list]


def convert_schedule_df_to_tensor(schedule: pd.DataFrame) -> np.ndarray:
    """
    Converts a double round-robin schedule in DataFrame format to tensor format.

    Parameters:
        schedule (pd.DataFrame): Schedule in DataFrame format.
            ```
            pd.DataFrame[
                index: [
                    "date number": int      # Matchday
                ],
                columns: [
                    "home": int     # Home team
                    "away": int     # Away team
                ]
            ]
            ```

    Returns:
        Schedule (np.ndarray):
            Tensor S_{t, i, j}. <br>
            Team i faced team j as the home-team in matchday t.
    """

    def _num_teams():
        home_teams = schedule["home"].unique()
        away_teams = schedule["away"].unique()
        return len(set(home_teams) | set(away_teams))

    def _when_teams_played() -> np.ndarray:
        """
        Returns:
            WhenPlayed (np.ndarray):
                Shape: (:, 3)
                Each row reprensets when the match was played: (matchday, team_home, team_away)
        """
        return schedule.reset_index("date number").to_numpy()

    def _fill_when_played_tensor():
        for index in when_played:
            when_played_tensor[tuple(index)] = 1

    num_teams = _num_teams()
    num_rounds_one_turn = num_teams - 1 if num_teams % 2 == 0 else num_teams

    when_played = _when_teams_played()
    when_played_tensor = np.zeros(shape=(2 * num_rounds_one_turn, num_teams, num_teams))
    _fill_when_played_tensor()

    return when_played_tensor
