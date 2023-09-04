# **Turning Point**
Repository for calculating turning points.


## **Dependencies**
---

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

## **Tournaments and Coefficients**
---

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
---

### **Paths**
All paths are defined in [`src/config/dataset_paths.py`](https://github.com/EstefanoB/turning_point/blob/main/src/config/dataset_paths.py).

Filename for a sport should be `f"{sport}.csv"`.

### **Creation**

Analysis can be done using files in `src/`.

You should move to `src/` before running scripts.

- [`create_dataset.py`](https://github.com/EstefanoB/turning_point/blob/main/src/create_dataset.py)
    - Create permutation if necessary.
    - Calculate turning point if desired.
- [`plots.ipynb`](https://github.com/EstefanoB/turning_point/blob/main/src/plots.ipynb)
    - Generate all images
- [`parameters.json`](https://github.com/EstefanoB/turning_point/blob/main/src/parameters.json)
    - Configuration for creating dataset.
    - See [Configuration Section](#configuration) for details.

## **Configuration**
---
Default values are in [`parameters.json`](https://github.com/EstefanoB/turning_point/blob/main/src/parameters.json).

- **sports**
    - List of strings
    - Sports which should be considered.
    - As mentioned in [Paths Section](#paths), filename for a sport is `f"{sport}.csv"`
- **should_<...>_it**
    - Boolen value
    - Whether or not the code associated with that portion should be run.
- **seed**
    - Integer
    - Seed which should be used for random events. 
- **num_iteration_simulation** 
    - List with two integers
    - Respectively, number of iterations and number of simulations per iteration.
        - It is split into "batches" to avoid using too much RAM.
- **num_permutations**
    - Integer
    - How many permutations should be created for each sport.
- **types**
    - In OPTIMAL_SCHEDULE["matches"]["parameters"]
    - One of the values (strings) below, or a list of them.
        - "tp_minimizer": schedule that minimizes the turning point
            - Deterministic: In the first time two teams face each other, the best one will play as home-team.
        - "tp_maximizer": schedule that maximizes the turning point
            - Deterministic: In the first time two teams face each other, the worst one will play as home-team.
        - "tp_minimizer_random": schedule that minimizes the turning point
            - Non-Deterministic: In the first time two teams face each other, the home-team will be chosen randomly.
        - "tp_maximizer_random": schedule that maximizes the turning point
            - Non-Deterministic: In the first time two teams face each other, the home-team will be chosen randomly.

## **Licensing**
---

This repository is licensed under the Apache License, Version 2.0. 

See [LICENSE](https://github.com/EstefanoB/turning_point/blob/main/LICENSE) for the full license text.