"""Plot helpers for the energy-budget workspace."""
import sys
import numpy as np
import matplotlib.pyplot as plt
from analysis import PROJECT_ROOT

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from utils_plot import plot_colors

COLORS = {"total_ke": plot_colors.ke_color, "eddy_ke": plot_colors.eddy_color,
          "seiche_ke": plot_colors.iw_color, "total_pe": plot_colors.pe_color,
          "wind_24h": plot_colors.wind_color, "total_energy": "tab:purple"}
LABELS = {"total_ke": "Total KE", "eddy_ke": "Eddy KE", "seiche_ke": "Internal seiche KE",
          "total_pe": "Available potential energy", "wind_24h": "Wind input — last 24h",
          "total_energy": "KE + APE"}


def timeseries(data, keys=("total_ke", "eddy_ke", "seiche_ke"), *, title="Energy budget", ylim=None):
    fig, ax = plt.subplots(figsize=(12, 5))
    for key in keys:
        data[key].plot(ax=ax, label=LABELS.get(key, key), color=COLORS.get(key))
    ax.set(title=title, ylabel="Energy [MJ]", xlabel="", ylim=ylim)
    ax.legend()
    fig.tight_layout()
    return fig, ax


def fractions(ratios, means, *, title="KE fractions", depth_label=""):
    fig, ax = plt.subplots(figsize=(9, 5))
    for key in ratios:
        ratios[key].plot(ax=ax, color=COLORS[key], label=f"{LABELS[key]} ({means[key]:.1f}%)")
        ax.axhline(means[key], color=COLORS[key], linestyle="--")
    ax.text(.02, .98, depth_label, transform=ax.transAxes, va="top")
    ax.set(title=title, ylabel="Fraction of total KE [%]", xlabel="", ylim=(0, 100))
    ax.legend()
    fig.tight_layout()
    return fig, ax


def monthly_radar(monthly):
    if monthly.empty:
        raise ValueError("No monthly data to plot.")
    angles = np.linspace(0, 2 * np.pi, len(monthly), endpoint=False)
    closed_angles = np.r_[angles, angles[0]]
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"projection": "polar"})
    for key, values in {"total_ke": monthly.total_ke,
                        "seiche_ke": monthly.eddy_ke + monthly.seiche_ke,
                        "eddy_ke": monthly.eddy_ke}.items():
        closed = np.r_[values.to_numpy(), values.iloc[0]]
        label = "Eddy + seiche KE" if key == "seiche_ke" else LABELS[key]
        ax.plot(closed_angles, closed, label=label, color=COLORS[key])
        ax.fill(closed_angles, closed, alpha=.1, color=COLORS[key])
    ax.set_xticks(angles, monthly.index.strftime("%b %Y"))
    ax.set_title("Monthly kinetic energy [MJ]")
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    return fig, ax


def calm_scatter(calm):
    values = calm[["hours_without_wind", "eddy_fraction_change"]].dropna()
    fig, ax = plt.subplots(figsize=(7, 5))
    x, y = values.hours_without_wind, values.eddy_fraction_change
    ax.scatter(x, y, s=2, alpha=.3, color=COLORS["eddy_ke"])
    if len(values) > 1 and x.nunique() > 1:
        slope, intercept = np.polyfit(x, y, 1)
        limits = np.array([x.min(), x.max()])
        ax.plot(limits, slope * limits + intercept, color="black", label="Linear fit")
        ax.legend()
    ax.set(xlabel="Hours without wind", ylabel="Eddy fraction change [percentage points]")
    return fig, ax


def interactive(data, keys, *, ylabel="Energy [MJ]"):
    from bokeh.plotting import figure
    plot = figure(title="Energy distribution", x_axis_type="datetime", width=1200,
                  height=450, x_axis_label="Date", y_axis_label=ylabel,
                  tools="xpan,xwheel_zoom,box_zoom,reset,save", active_scroll="xwheel_zoom")
    for key in keys:
        plot.line(data[key].index, data[key].to_numpy(), legend_label=LABELS.get(key, key),
                  line_color=COLORS[key])
    plot.legend.click_policy = "hide"
    return plot
