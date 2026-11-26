import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
# Style for the figures of the report / presentation, so that every plot looks the same.
#
# Usage in a plot_maker file:
#     import source_style as ss
#     ss.set_style("report")                         # or "presentation", once at the top
#     fig, ax = ss.figure()                          # 1 panel, full text width
#     fig, axes = ss.figure(ncols=2)                 # 2 panels side by side
#     ax.plot(x, y, **ss.METHOD["A"])                # same color / marker for a method in every plot
#     ss.panel_labels(axes)                          # (a), (b), ... in the corners
#     ss.save(fig, "my_plot")                        # -> results/my_plot.pdf and results/my_plot.png


# ---------------------------------------------------------------- colors and markers
# Colorblind-safe categorical palette (fixed order: always use them in this order).
COLORS = ["#2a78d6",   # blue
          "#eb6834",   # orange
          "#1baf7a",   # aqua
          "#eda100",   # yellow
          "#6250d6",   # violet
          "#e34948"]   # red
MARKERS = ["o", "s", "^", "D", "v", "P"]
INK = "#2b2b2b"        # text / axes
REF = "#000000"        # reference lines (theory, mu_eq, ...)

# Same look for a method in every figure (color + marker, so it is readable in black and white too).
METHOD = {
    "naive": dict(color=COLORS[2], marker=MARKERS[2], label="naive"),
    "A":     dict(color=COLORS[0], marker=MARKERS[0], label="method A"),
    "B":     dict(color=COLORS[1], marker=MARKERS[1], label="method B"),
}


# ---------------------------------------------------------------- presets
# width = width of a full-width figure in inches.
# report: LaTeX article text width (~16 cm), fonts the size of the text (10-11 pt).
# presentation: 16:9 slide, big fonts readable from the back of the room.
PRESETS = {
    "report":       dict(width=6.3, base=10, lw=1.4, ms=4.5, serif=True),
    "presentation": dict(width=11.0, base=18, lw=2.5, ms=8.0, serif=False),
}
_current = "report"


def set_style(mode="report"):
    """Set matplotlib rcParams for 'report' or 'presentation'. Call it once before plotting."""
    global _current
    if mode not in PRESETS:
        raise ValueError(f"mode must be one of {list(PRESETS)}")
    _current = mode
    p = PRESETS[mode]
    b = p["base"]
    mpl.rcParams.update({
        # fonts
        "font.family": "serif" if p["serif"] else "sans-serif",
        "font.serif": ["CMU Serif", "Computer Modern Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm" if p["serif"] else "dejavusans",
        "font.size": b,
        "axes.titlesize": b,
        "axes.labelsize": b + 1,
        "xtick.labelsize": b - 1,
        "ytick.labelsize": b - 1,
        "legend.fontsize": b - 1.5,
        "figure.titlesize": b + 1,
        # lines and markers
        "lines.linewidth": p["lw"],
        "lines.markersize": p["ms"],
        "errorbar.capsize": 0.6 * p["ms"],
        "axes.prop_cycle": mpl.cycler(color=COLORS),
        # axes: light, no top/right spines
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#d9d9d9",
        "grid.linewidth": 0.6,
        "grid.alpha": 0.8,
        "axes.axisbelow": True,
        # ticks
        "xtick.color": INK,
        "ytick.color": INK,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "xtick.minor.size": 2,
        "ytick.minor.size": 2,
        # legend
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "#cccccc",
        "legend.fancybox": False,
        "legend.borderpad": 0.4,
        "legend.handlelength": 1.8,
        # output
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
        "pdf.fonttype": 42,          # editable text in the pdf
    })


def figure(ncols=1, nrows=1, width=1.0, aspect=0.75, **kwargs):
    """New figure sized for the current style.
    width  = fraction of the full width (1.0 = whole text width, 0.5 = half, for 2 figures side by side)
    aspect = height / width of ONE panel.
    Returns (fig, ax) or (fig, axes) like plt.subplots."""
    W = PRESETS[_current]["width"] * width
    panel_w = W / ncols
    H = panel_w * aspect * nrows
    return plt.subplots(nrows, ncols, figsize=(W, H), constrained_layout=True, **kwargs)


def panel_labels(axes, start="a"):
    """Write (a), (b), ... in the top-left corner of each panel."""
    for i, ax in enumerate(np.atleast_1d(np.asarray(axes, dtype=object)).flat):
        ax.text(-0.02, 1.02, f"({chr(ord(start) + i)})", transform=ax.transAxes,
                ha="right", va="bottom", fontweight="bold")


def vline(ax, x, label=None, color=REF, ls="--", lw=None):
    """Vertical reference line (mu_opt, mu_eq, thresholds, ...)."""
    ax.axvline(x, color=color, ls=ls, lw=lw or 0.8 * mpl.rcParams["lines.linewidth"], label=label, zorder=1)


def save(fig, name, folder="results", formats=("png",)):
    """Save in results/ as pdf (for LaTeX, vector) and png (for slides / quick look)."""
    os.makedirs(folder, exist_ok=True)
    for f in formats:
        fig.savefig(os.path.join(folder, f"{name}.{f}"))
    plt.close(fig)


set_style("report")      # default when the module is imported
