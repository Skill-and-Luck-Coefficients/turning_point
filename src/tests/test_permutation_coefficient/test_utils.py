import pandas as pd

import turning_point.permutation_coefficient.utils as utils


def test_add_second_level_to_column_names():

    test = pd.DataFrame(
        {
            "id": pd.Categorical([0, 1, 3, 5]),
            "col1": ["a", "b", "a", "c"],
            "col2": [0.54, 9.85, 928.5, 945.8],
        }
    ).set_index("id")

    for name in ["ok", "test", "level2", 59]:

        expected = pd.DataFrame(
            {
                "id": pd.Categorical([0, 1, 3, 5]),
                ("col1", name): ["a", "b", "a", "c"],
                ("col2", name): [0.54, 9.85, 928.5, 945.8],
            }
        ).set_index("id")

        assert utils.add_second_level_to_column_names(test, name).equals(expected)
