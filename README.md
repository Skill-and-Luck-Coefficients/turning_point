# **Turning Point**
Repository for calculating turning points.

## **Reproducibility**

**Seeds**: `src/parameters.json` contains the default seeds used.

Sample for tounament match information:

| | | "home" | "away" | "result" | "winner" | "date" | "odds home" | "odds tie" | "odds away" |
| --- | ---- | ---- | ---- | ---- | ---- | ----- | ----- |     --     |    --- |
| "id" | "date number" | | |  |  |  |  |  |  |
| 'acb@/basketball/spain/acb-2010-2011/' | 0 | 'Estudiantes'  | 'Real Madrid' | '79:84' | 'a' | '30.09.2010' | 3.33 |     <not avalaible>     |    1.31 |
| 'acb@/basketball/spain/acb-2010-2011/' | 1 |  'Real Betis' | 'Joventut Badalona' | '85:80' | 'h	' | '02.10.2010' | 1.31 |     <not avalaible>     |    3.35 |

**Remarks**:
- 'odds tie' is not required for basketball and volleyball.
- 'date' is not necessary, only the matchday is ('date number').
- If 'winner' is provided, 'result' is not necessary for the default turning point calculation (3 points for win, 1 for tie and 0 for loss).

**Download**: csvs containing the exact matches we used are available at [Tournament Matches](https://github.com/Skill-and-Luck-Coefficients/scrape_tournament_matches/tree/main/data)


## **Dependencies**

### **Submodules**

Make sure to also clone the submodules. It can be done by cloning a repository with:
```
$ git clone --recurse-submodules <repository>
```

If you cloned it without recursing submodules, you can run:
```
$ git submodule init
$ git submodule update
```

### **Virtual Environments**

If you prefer **Pip**:

```
$ python3 -m venv env                           # create env
$ source env/bin/activate                       # activate env
$ python3 -m pip install -r requirements.txt    # install packages
```

If you prefer **Conda**:

```
$ conda env create -f environment.yml  # create env with all packages
$ conda activate skill_coef            # activate env
```

### **Installing submodule**

Submodule `tournament_simulations` is not on [PyPy](https://www.pypy.org/), so it should be installed manually.

With your virtual environment activated, go to submodule's directory:
```
$ cd tournament_simulations
```

Then you can install it with:

```
$ python3 -m pip install --upgrade build
$ python3 -m build
$ python3 -m pip install dist/*.whl
```

### **Run**

**Turning point calculation:**
- Place the matches in its expected directory: [Paths](#paths).
- Update the configuration if necessary: [Configuration](#configuration).
- Run: 
```
$ cd src
$ python create_dataset.py
```

**Graphs/Results:**
After creating the dataset, run the desired notebook.

## **Tournaments and Coefficients**

### **Matches**

For information about what data is expected, see data frame documentation in [Matches](https://github.com/EstefanoB/tournament_simulations/blob/main/src/tournament_simulations/data_structures/matches/matches.py).

- **Real tournaments**: all data come from real matches. 
- **Permutations**: Real matches but the order has been permuted. 
    - It is not a simple permutation, since it follows a double robin schedule in an attempt to prevent schedules that would never happen in real life.
    - For more information, see [Tournament Simulations](https://github.com/EstefanoB/tournament_simulations/blob/main/README.md).


**Real tournament matches should be provided.** 
- It should not have invalid values (like NaN) for required data.
- It should not have a match (A, B) which happens more than once in a day, otherwise the tournament can't be permuted correctly.
- See path details in [Paths Section](#paths).

### **Turning Point**

**Coefficient**

For a tournament, coefficient is calculated by finding when simulated ranking variances becomes considerably smaller than its real ranking variance.

For each expanding window: 
- Calculate ranking variance for real tournament.
- Calculate 95% percentile of ranking variances over all simulated tournaments.

[Turning Point](https://github.com/EstefanoB/turning_point/blob/main/src/turning_point/normal_coefficient/calculate_turning_point.py) is defined to be the date from which the real ranking variance is always above the simulated 95% percentile. 

**DataFrames**

Date (or match) after which the tournament can't be considered random anymore.

- **Date Turning Point**: 
    - `coef`
        - Date until tournament can't be considered random.
        - It doesn't always happen: represented by infinity.
    - `%coef`
        - Same as before, but each tournament is normalized by its last date.
        - As long as `coef` exists, this will be between 0 and 1.
    - See documentation about data frame in [Turning Point](https://github.com/EstefanoB/turning_point/blob/main/src/turning_point/normal_coefficient/turning_point.py).
- **Match Turning Point**: 
    - `coef`
        - Number of matches until tournament can't be considered random.
        - It doesn't always happen: represented by infinity.
    - `%coef`
        - Same as before, but each tournament is normalized by its total number of matches.
        - As long as `coef` exists, this will be between 0 and 1.
    - See documentation about data frame in [Match Turning](https://github.com/EstefanoB/turning_point/blob/main/src/turning_point/match_coefficient/match_turning_point.py).


## **Dataset**

### **Paths**
All default paths are defined in [`src/config/dataset_paths.py`](https://github.com/EstefanoB/turning_point/blob/main/src/config/dataset_paths.py).

Directory for different quantiles and metrics are of the form: "{default}/{quantile}/{metric}/"

Filename for a sport should be `f"{sport}.csv"`.

## **Configuration**
Default values are in [`src/parameters.json`](https://github.com/EstefanoB/turning_point/blob/main/src/parameters.json).

- **sports**
    - List of strings
    - Sports/filenames which should be considered.
    - As mentioned in [Paths Section](#paths), filename for a sport is `f"{sport}.csv"`
- **should_<...>_it**
    - Boolen value
    - Whether or not the code associated with that portion should be run.
- **quantile**
    - Float (or list of floats)
    - Quantile values to use when calculating the envelope.
- **seed**
    - Integer (or list of integers)
    - Seed which should be used for random events. 
    - If more than one quantile is passed as argument, you can pass a different seed for each one.
- **num_iteration_simulation** 
    - List with two integers
    - Respectively, number of iterations and number of simulations per iteration.
        - It is split into "batches" to avoid using too much RAM.
- **num_permutations**
    - Integer
    - How many permutations should be created for each sport.
- **winner_type**
    - Literal["winner", "result"]
    - Whether points should be based off of the winner or the result
- **winner_to_points**
    - Mapping[str, tuple[float, float]]
    - Maps the winner (or result if winner_type is "result") to the points home and away teams gained after the match (respectively).
- **metric**
    - str | Iterable[str]
    - Which metric should be used.
    - Options: See https://github.com/EstefanoB/turning_point/blob/main/src/turning_point/metrics/
- **types** : OPTIMAL_SCHEDULE["matches"]["parameters"]
    - dict[str, dict[str, str | list[str]]]
    - Keys: 
        - Options: "graph", "recurvise"
        - Which optimal algorithm should be used
    - Values: dict[str, str | list[str]]
        - Keys: 
            - Options: "tp_maximizer", "tp_minimizer"
            - Which optimization should be run: increase (maximizer) or descrease (minimizer) perceived balance.
        - Values: str | list[str]
            - Options:
                - "mirrored": Second turn will be a mirrored version of the first
                - "reversed": Second portion/turn will have the opposite order. The last match to happen in the first turn will be the first to happen in the second.
                - "random_mirrored": Similar to mirrored, but which team plays as home/away in all first turn matches is randomized.
                - "random_reversed": Similar to reversed, but which team plays as home/away in all first turn matches is randomized.
- **parameters**: BRADLEY_TERRY["matches"]
    - dict[str, dict[str, values]]
        - Key: str (corresponding filename/sport)
        - dict[str, values]
            - "strengths": list[float]
                - Teams strenghts
            - "n_different_results": int
                - Given a set of team strengths, how many simulations should be run.
            - "n_permutations_per_result": int
                - How many permutations should compose a simulations for each strength.
            - "number_of_drr": int
                - Number of double round-robin schedules to compose a single permutation.

## **Licensing**
---

This repository is licensed under the Apache License, Version 2.0. 

See [LICENSE](https://github.com/EstefanoB/turning_point/blob/main/LICENSE) for the full license text.