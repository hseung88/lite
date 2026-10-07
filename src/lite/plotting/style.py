import matplotlib.pyplot as plt

GP_PANEL_SIZE = (8.0, 8.0 * 0.6789)
GP_STYLE = {
    "font.family": "DejaVu Sans",
    "font.size": 18,
    "axes.labelsize": 20,
    "axes.titlesize": 18,
    "legend.fontsize": 22,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
}


def setup_gp_style() -> None:
    plt.rcParams.update(GP_STYLE)


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 12,
        }
    )
