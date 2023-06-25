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


def test_get_permutation_numbers():
    test = pd.DataFrame(
        {
            "id": pd.Categorical(
                [
                    "current@/one/two/three@0",
                    "current@/one/two/three@1",
                    "current@/one/two/three@2",
                ]
            ),
        }
    ).set_index("id")

    expected = sorted(["0", "1", "2"])
    assert sorted(utils.get_permutation_numbers(test)) == expected

    test = pd.DataFrame(
        {
            "id": pd.Categorical(
                [
                    "current@/one/two/three",
                    "current@/",
                ]
            ),
        }
    ).set_index("id")

    expected = sorted([""])
    assert sorted(utils.get_permutation_numbers(test)) == expected

    test = pd.DataFrame(
        {
            "id": pd.Categorical(
                [
                    "current@/one/two/three@0",
                    "current@/one/two/three@2",
                    "current@/one/two/three@two",
                ]
            ),
        }
    ).set_index("id")

    expected = sorted(["", "0", "2"])
    assert sorted(utils.get_permutation_numbers(test)) == expected

    test = pd.DataFrame(
        {
            "id": pd.Categorical(
                [
                    "one@two",
                    "one@two@0",
                    "one@two@10",
                    "one@two@02",
                    "one@two@100",
                    "one@two@2",
                ]
            ),
        }
    ).set_index("id")

    expected = sorted(["", "0", "10", "02", "100", "2"])
    assert sorted(utils.get_permutation_numbers(test)) == expected


def test_filter_ith_permutation():
    test = pd.DataFrame(
        {
            "id": pd.Categorical(
                [
                    "current@/one/two/three",
                    "current@/one/two/three",
                    "current@/one/two/three",
                ]
            ),
            "col": [0, 1, 2],
        }
    ).set_index("id")

    assert utils.get_ith_permutation(test, "").equals(test)

    test = pd.DataFrame(
        {
            "id": pd.Categorical(
                [
                    "current@/one/two/three",
                    "current@/one/two/three@1",
                    "current@/one/two/three@11",
                    "current@/one/two/three@21",
                ]
            ),
            "col": [0, 1, 2, 3],
        }
    ).set_index("id")

    assert utils.get_ith_permutation(test, "").equals(test)

    expected_index = ["current@/one/two/three@1"]
    assert utils.get_ith_permutation(test, "1").equals(test.loc[expected_index])

    expected_index = ["current@/one/two/three@11"]
    assert utils.get_ith_permutation(test, "11").equals(test.loc[expected_index])

    expected_index = ["current@/one/two/three@21"]
    assert utils.get_ith_permutation(test, "21").equals(test.loc[expected_index])

    test = pd.DataFrame(
        {
            "id": pd.Categorical(
                [
                    "current@/one@0",
                    "current@/one@0",
                    "current@/two@0",
                    "current@/two@0",
                    "current@/two@0",
                    "current@/two@0",
                    "current@/one@1",
                    "current@/one@1",
                    "current@/two@1",
                    "current@/two@1",
                    "current@/two@2",
                    "current@/two@2",
                ]
            ),
            "col": list(range(12)),
        }
    ).set_index("id")

    assert utils.get_ith_permutation(test, "").equals(test)

    expected_index = ["current@/one@0", "current@/two@0"]
    assert utils.get_ith_permutation(test, "0").equals(test.loc[expected_index])

    expected_index = ["current@/one@1", "current@/two@1"]
    assert utils.get_ith_permutation(test, "1").equals(test.loc[expected_index])

    expected_index = ["current@/two@2"]
    assert utils.get_ith_permutation(test, "2").equals(test.loc[expected_index])
