import pandas as pd

from turning_point.normal_coefficient import TurningPoint


def _create_result_regex(result_id: str, permutation: str) -> str:
    return rf".+?@result_{result_id}@{permutation}"


def _extract_tp_results(df: pd.DataFrame, result_id: str):
    """
    ----
    Returns:
        tuple[
            pd.DataFrame,  # results for permutations
            pd.DataFrame,  # result for optimal graph schedule
            pd.DataFrame,  # result for optimal recursive schedule
        ]
    """

    def _filter_specific_result(df: pd.DataFrame, regex: str) -> pd.DataFrame:
        return df[df.index.get_level_values("id").str.fullmatch(regex)]

    permutations_regex = _create_result_regex(result_id, permutation="\d+")
    permutations_tp = _filter_specific_result(df, permutations_regex)

    graph_regex = _create_result_regex(result_id, permutation="graph_optimal")
    graph_optimal_tp = _filter_specific_result(df, graph_regex)

    recursive_regex = _create_result_regex(result_id, permutation="recusive_optimal")
    recursive_optimal_tp = _filter_specific_result(df, recursive_regex)

    return permutations_tp, graph_optimal_tp, recursive_optimal_tp


def _optimal_values(
    graph_series: pd.Series, recursive_series: pd.Series
) -> dict[str, float]:
    return {
        "graph optimal": graph_series.iloc[0],
        "recursive optimal": recursive_series.iloc[0],
    }


def _permutations_stats(bt_df: pd.DataFrame, tp_column: str) -> dict[str, float]:
    return {
        "average permutation": bt_df[("permutations", tp_column)].mean(),
        "median permutation": bt_df[("permutations", tp_column)].median(),
    }


def _optimal_tp_diff_stats(series: pd.Series, key: str) -> dict[str, float]:
    return {
        f"{key} diff < 0": (series < 0).mean(),
        f"avg {key} diff [{key} < 0]": series[series < 0].abs().mean(),
        f"avg {key} diff [{key} > 0]": series[series > 0].abs().mean(),
    }


def _get_validation_summary_one_file(df: pd.DataFrame, tp_column: str) -> pd.DataFrame:

    extract_results_regex = _create_result_regex(result_id="(.+?)", permutation=".+")
    unique_values = df.index.str.extract(extract_results_regex, expand=False).unique()

    all_results = []

    for result_id in unique_values:

        permutations, graph_opt, recursive_opt = _extract_tp_results(df, result_id)

        bt_df = pd.concat(
            {
                "permutations": permutations,
                "graph optimal diff": graph_opt.to_numpy() - permutations,
                "recursive optimal diff": recursive_opt.to_numpy() - permutations,
            },
            axis="columns",
        )

        optimal_values = _optimal_values(graph_opt[tp_column], recursive_opt[tp_column])
        permutation_stats = _permutations_stats(bt_df, tp_column)

        graph_col = bt_df[("graph optimal diff", "turning point")]
        graph_diff = _optimal_tp_diff_stats(graph_col, key="graph")

        recursive_col = bt_df[("recursive optimal diff", "turning point")]
        recursive_diff = _optimal_tp_diff_stats(recursive_col, key="recursive")

        bt_result = optimal_values | permutation_stats | graph_diff | recursive_diff
        all_results.append(pd.Series(bt_result, name=result_id))

    return pd.concat(all_results, axis="columns")


def get_validation_summary_all_files(
    filename_to_tp: dict[str, TurningPoint],
    filenames: list[str],
    tp_column: str = "%turning point",
) -> dict[str, pd.DataFrame]:
    return {
        fname: _get_validation_summary_one_file(filename_to_tp[fname].df, tp_column)
        for fname in filenames
    }
