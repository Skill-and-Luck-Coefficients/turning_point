import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from data_analysis.plots.utils import plot_functions as pf
from turning_point.permutation_coefficient import PermutationTurningPoint


def plot_bradley_terry_validation(
    fig: plt.Figure,
    axs: list[list[plt.Axes]],
    key_to_bradley_terry_tp: dict[str, PermutationTurningPoint],
    key_to_strengths: dict[str, list[float]],
):
    def _build_type_column():
        _type_values = bt_tp.index.str.extract("BT_(.+?)@.+", expand=False)
        return _type_values.map(
            {
                "purely_random": "Random",
                "graph_optimal": "Min\niMWM",
                "graph_optimal_max": "Max\niMWM",
                "recursive_optimal": "Min\nREC",
                "recursive_optimal_max": "Max\nREC",
            }
        )

    def _parse_bt():
        return (
            bt_tp.sort_values(by="type", ascending=False)
            .replace({np.inf: 1.005})
            .dropna()
        )

    def _build_title_from_strength():
        return ", ".join(map(str, strength[: len(strength) // 2]))

    flat_axs = flat_axs = pf.flatten_axes(axs)

    for ax, (key, strength) in zip(flat_axs, key_to_strengths.items()):
        bt_tp = key_to_bradley_terry_tp[key].df
        bt_tp["type"] = _build_type_column()
        bt_tp = _parse_bt()

        sns.boxplot(data=bt_tp, y="%turning point", x="type", ax=ax)
        ax.set_xlabel("")
        ax.set_ylabel("")

        ax.set_title(_build_title_from_strength(), fontsize=30)

    pf.add_ylabels_to_nth_col(fig, axs, r"PCB $\tau_\%$", n=0)
