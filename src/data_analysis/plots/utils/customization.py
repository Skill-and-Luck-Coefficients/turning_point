from seaborn import set_theme

# NOTE:
# newer versions o matplotlib should have an option called
# constrained_layout but when I wrote this, it was still experimental.
# Thus, subplot parameters are hardcoded with fig.subplots_adjust
# for each graph separately

# matplotlib.rcParams.keys() gives all options
MATPLOTLIB_RC = {
    "axes.titlesize": 35,
    "axes.titlepad": 10,
    "axes.labelsize": 30,
    "axes.labelpad": 15,
    "legend.fontsize": 30,
    "legend.framealpha": 1,
    "lines.markersize": 15,
    "legend.handletextpad": 0.3,
    "legend.handlelength": 1,
    "figure.dpi": 250,
    "savefig.dpi": 300,
    "savefig.format": "pdf",
    "xtick.major.pad": 15,
    "xtick.labelsize": 30,
    "xtick.major.size": 0,
    "ytick.labelsize": 30,
    "ytick.major.size": 0,
}

set_theme(
    context="talk",
    style="darkgrid",
    palette="deep",
    font="Serif",
    font_scale=1,
    color_codes=True,
    rc=MATPLOTLIB_RC,
)
