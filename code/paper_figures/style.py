"""Shared figure style.

Matches the look of the IEEE TSTE reference (Omar & El-Shatshat, 2025): sans-serif type,
boxed axes, dashed light grid, framed legends, Dark2-type colours, centred panel titles and
the panel letter (a), (b), ... printed under each panel.
"""
import os
import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COL1, COL2 = 3.5, 7.16          # single / double column width, inches

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "mathtext.fontset": "dejavusans",
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.8,
    "axes.linewidth": 0.8, "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "xtick.major.size": 3, "ytick.major.size": 3,
    "xtick.direction": "out", "ytick.direction": "out",
    "axes.spines.top": True, "axes.spines.right": True,
    "axes.grid": True, "grid.color": "#b0b0b0", "grid.linestyle": "--", "grid.linewidth": 0.45,
    "grid.alpha": 0.7, "axes.axisbelow": True, "lines.linewidth": 1.3,
    "legend.frameon": True, "legend.framealpha": 0.95, "legend.edgecolor": "#c8c8c8",
    "legend.fancybox": True, "legend.handlelength": 1.8, "legend.borderpad": 0.35,
    "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.titlelocation": "center", "axes.titleweight": "normal", "axes.titlepad": 4,
})

INK, INK2, MUTED = "#000000", "#404040", "#8c8c8c"

# Dark2 / tab colours, as in the reference figures
TEAL, ORANGE, PURPLE, MAGENTA, GREEN, MUSTARD, BROWN, GREY = (
    "#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#e6ab02", "#a6761d", "#666666")
BLUE, RED, CYAN = "#1f77b4", "#d62728", "#17becf"
AQUA = TEAL

METHOD = {
    "mpc":     ("MPC", BLUE),
    "miqp":    ("MIQP", ORANGE),
    "il":      ("MPC-imitation", GREY),
    "fpppo":   ("FP-PPO", PURPLE),
    "pdppo":   ("PD-PPO", TEAL),
    "focops":  ("FOCOPS", MUSTARD),
    "cpo":     ("CPO", GREEN),
    "miqpdrl": ("MIQP+CPO", BROWN),
    "mpcdrl":  ("MPC-CPO", MAGENTA),
}
NAME = {k: v[0] for k, v in METHOD.items()}
COLOR = {k: v[1] for k, v in METHOD.items()}

# energy sources (stacked bars), as in the reference's dispatch figure
SRC = {"pv": "#e6ab02", "wind": "#1b9e77", "bess": "#1f77b4", "fc": "#7570b3", "el": "#e7298a",
       "bc": "#9ecae1", "unserved": "#d62728", "curt": "#bdbdbd", "load": "#000000"}


def panel(ax, letter, text=""):
    """Centred title above the panel; the letter is printed under the panel by save()."""
    ax.set_title(text)
    ax._letter = letter


def _letters(fig):
    for ax in fig.axes:
        letter = getattr(ax, "_letter", "")
        if not letter:
            continue
        xl = ax.get_xlabel()
        ticks_visible = any(t.get_visible() and t.get_text() for t in ax.get_xticklabels())
        if xl and ticks_visible:
            ax.set_xlabel(xl + "\n(%s)" % letter)
        else:
            ax.set_title("(%s) %s" % (letter, ax.get_title()))


def legend_below(ax, **kw):
    """Ask save() to put this panel's legend under the panel, below its x label and letter."""
    ax._below_legend = kw


def _place_below_legends(fig, pad=3):
    from matplotlib.transforms import offset_copy
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    for ax in fig.axes:
        kw = getattr(ax, "_below_legend", None)
        if kw is None:
            continue
        lab = ax.xaxis.label.get_window_extent(r)
        tick = ax.xaxis.get_tightbbox(r)
        y0 = min(lab.y0, tick.y0) if tick is not None else lab.y0
        dy = (ax.bbox.y0 - y0) * 72.0 / fig.dpi + pad          # points below the axes frame
        tr = offset_copy(ax.transAxes, fig=fig, y=-dy, units="points")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, 0.0), bbox_transform=tr, **kw)


def _place_figure_legend(fig, spec, pad=4):
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    y0 = min(ax.get_tightbbox(r).y0 for ax in fig.axes if ax.get_visible())
    y = (y0 - pad * fig.dpi / 72.0) / fig.bbox.height
    handles = spec.pop("handles")
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, y), **spec)


def save(fig, name, out=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures"),
         letters=True, relayout=True, fig_legend=None):
    os.makedirs(out, exist_ok=True)
    if letters:
        fig.canvas.draw()                      # make tick labels exist before testing them
        _letters(fig)
    _place_below_legends(fig)
    if relayout:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fig.tight_layout()
    if fig_legend:
        _place_figure_legend(fig, dict(fig_legend))
    fig.savefig(os.path.join(out, name + ".pdf"))
    fig.savefig(os.path.join(out, name + ".png"), dpi=600)
    plt.close(fig)
