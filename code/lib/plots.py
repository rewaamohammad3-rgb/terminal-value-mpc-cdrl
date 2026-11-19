# -*- coding: utf-8 -*-
"""All figures and the summary table for one run.

    python plots.py <run_dir>

Reads only what the stages wrote, so it can be re-run any time (python run.py --stages plots).
Figures go to <run_dir>/figures/{network,mpc,drl,summary}/ as PNG and PDF.
"""
import csv
import glob
import io
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                   # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core                                                       # noqa: E402

# Categorical slots in fixed order (validated reference palette), plus ink and limits.
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8c8a85", "#e4e3df"
LIMIT = "#b3261e"
DEV = {"PV": ("PV", C[3]), "WIND": ("Wind", C[2]), "BESS": ("Battery", C[0]),
       "HFC": ("Hydrogen (EL/FC)", C[6])}
ALGO_NAME = {"il": "MPC-imitation", "fpppo": "FP-PPO", "pdppo": "PD-PPO", "focops": "FOCOPS",
             "cpo": "CPO", "miqp": "MIQP",
             "miqpdrl": os.environ.get("V3_MIQPDRL_LABEL", "MIQP+DRL"),
             "mpcdrl": os.environ.get("V3_MPCDRL_LABEL", "MPC+DRL values")}
ALGO_COL = {"MPC": C[0], "il": C[5], "fpppo": C[1], "pdppo": C[2], "focops": C[3], "cpo": C[4],
            "miqp": C[6], "miqpdrl": C[7], "mpcdrl": C[7]}   # mpcdrl and miqpdrl never share a run
ALGO_ORDER = ("il", "fpppo", "pdppo", "focops", "cpo", "miqp", "miqpdrl", "mpcdrl")

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9, "legend.fontsize": 8,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.6,
    "legend.frameon": False, "savefig.dpi": 300, "savefig.bbox": "tight"})

# IEEE-33 topology: (from, to) for branches 1..32, and feeder sections for grouping
BRANCHES = [(i, i + 1) for i in range(1, 18)] + [(2, 19), (19, 20), (20, 21), (21, 22),
                                                  (3, 23), (23, 24), (24, 25), (6, 26)] + \
           [(i, i + 1) for i in range(26, 33)]
SECTIONS = [("main feeder", list(range(1, 19))), ("lateral", list(range(19, 23))),
            ("lateral", list(range(23, 26))), ("lateral", list(range(26, 34)))]


def bus_xy():
    xy = {b: (b - 1, 0.0) for b in range(1, 19)}
    xy.update({b: (1 + 0.8 * (b - 18), 1.6) for b in range(19, 23)})
    xy.update({b: (2 + 0.8 * (b - 22), -1.6) for b in range(23, 26)})
    xy.update({b: (5 + 0.9 * (b - 25), -3.2) for b in range(26, 34)})
    return xy


def section_axis():
    """Buses 1..33 in order on a continuous axis (the usual IEEE 33-bus profile plot)."""
    order = list(range(1, 34))
    return order, np.arange(33, dtype=float)


def mark_sections(ax, xs, top):
    """Thin separators where a lateral starts, labelled with the bus it branches from."""
    labels = [("main feeder", 0, 17), ("off bus 2", 18, 21),
              ("off bus 3", 22, 24), ("off bus 6", 25, 32)]
    for name, a, b in labels:
        if a > 0:
            ax.axvline(xs[a] - 0.5, color=MUTED, lw=0.7, ls=":")
        ax.text((xs[a] + xs[b]) / 2, top, name, ha="center", va="bottom", fontsize=7,
                color=INK2)


class Run(object):
    def __init__(self, run_dir):
        self.dir = run_dir
        self.name = os.path.basename(os.path.normpath(run_dir))
        self.cfg = core.load_config(os.path.join(run_dir, "config_used.py"))
        self.fig = os.path.join(run_dir, "figures")
        self.mpc = {}
        for f in sorted(glob.glob(os.path.join(run_dir, "mpc", "kpi_*.json"))):
            k = core.read_json(f)
            self.mpc[int(k["year"])] = k
        self.drl = []
        for f in sorted(glob.glob(os.path.join(run_dir, "drl", "result_*.json"))):
            self.drl.append(core.read_json(f))

    def net(self, year):
        p = os.path.join(self.dir, "mpc", "network_%d.npz" % year)
        return dict(np.load(p)) if os.path.exists(p) else None

    def trace(self, year):
        p = os.path.join(self.dir, "mpc", "trace_%d.npz" % year)
        return dict(np.load(p)) if os.path.exists(p) else None

    def save(self, fig, sub, name):
        d = os.path.join(self.fig, sub)
        os.makedirs(d, exist_ok=True)
        for ext in getattr(self.cfg, "FIG_FORMATS", ["png", "pdf"]):
            fig.savefig(os.path.join(d, "%s.%s" % (name, ext)))
        plt.close(fig)
        print("  figures/%s/%s" % (sub, name))

    def drl_file(self, kind, algo, year, seed=0):
        if kind == "trace":
            p = os.path.join(self.dir, "drl", "traces", "%s_%d_s%d.npz" % (algo, year, seed))
        else:
            p = os.path.join(self.dir, "drl", "%s_%s_%d_s%d.npz" % (kind, algo, year, seed))
        return dict(np.load(p)) if os.path.exists(p) else None

    def algos(self):
        return [a for a in ALGO_ORDER if any(k["algo"] == a for k in self.drl)]

    def plot_year(self):
        y = int(getattr(self.cfg, "PLOT_YEAR", 2020))
        return y if y in self.mpc else (sorted(self.mpc)[0] if self.mpc else None)


# ───────────────────────────────────────────────────────────── network figures
def fig_layout(r):
    layout = dict(r.cfg.LAYOUT)
    caps = {int(b): q for b, q in r.cfg.CAPACITORS_MVAR.items()}
    plant = {}
    if r.mpc:
        plant = next(iter(r.mpc.values())).get("plant", {})
        layout = {k: int(v) for k, v in plant.get("layout", layout).items()}
    xy = bus_xy()
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    for a, b in BRANCHES + [(2, 19), (3, 23), (6, 26)]:
        (x1, y1), (x2, y2) = xy[a], xy[b]
        if y1 != y2:      # lateral take-off: drop vertically then run along
            ax.plot([x1, x1, x2], [y1, y2, y2], color=MUTED, lw=1.2, zorder=1)
        else:
            ax.plot([x1, x2], [y1, y2], color=MUTED, lw=1.2, zorder=1)
    for b, (x, y) in xy.items():
        ax.scatter([x], [y], s=16, color="white", edgecolor=INK2, lw=0.9, zorder=2)
        ax.text(x, y - 0.3 if y >= 0 else y + 0.3, str(b), ha="center",
                va="top" if y >= 0 else "bottom", fontsize=6.5, color=INK2)
    ax.scatter([xy[1][0]], [xy[1][1]], s=90, marker="s", color=INK, zorder=3)
    ax.text(xy[1][0], xy[1][1] + 0.35, "Bus 1\n(slack, 1.0 pu)", ha="center", va="bottom",
            fontsize=7, color=INK)
    rating = {"PV": "%.2f MWp" % r.cfg.PV_MWP, "WIND": "%.2f MW" % r.cfg.WIND_MW,
              "BESS": "%.1f MW / %.1f MWh" % (r.cfg.BESS_P_MW, r.cfg.BESS_USABLE_MWH),
              "HFC": "FC %.1f / EL %.1f MW" % (r.cfg.FC_MW, r.cfg.EL_MW)}
    stack = {}
    for k in ("PV", "WIND", "BESS", "HFC"):
        if k not in layout:
            continue
        bus = layout[k]
        x, y = xy[bus]
        n = stack.get(bus, 0)
        stack[bus] = n + 1
        dy = (0.62 + 0.55 * n) * (1 if y >= 0 else -1)      # away from the feeder line
        ax.scatter([x], [y + dy], s=70, marker="o", color=DEV[k][1], edgecolor="white",
                   lw=1.2, zorder=4)
        ax.plot([x, x], [y, y + dy], color=DEV[k][1], lw=1.0, zorder=3)
        ax.text(x + 0.28, y + dy, "%s, %s (bus %d)" % (DEV[k][0], rating[k], bus), va="center",
                fontsize=6.8, color=INK, zorder=5,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
    for bus, q in caps.items():
        x, y = xy[bus]
        dc = -0.75 if y >= 0 else 0.75                        # opposite side to the devices
        ax.scatter([x], [y + dc], s=36, marker="v", color=INK2, zorder=4)
        ax.text(x + 0.25, y + dc, "C %.2f Mvar (bus %d)" % (q, bus), va="center", fontsize=6.5,
                color=INK2)
    ys = [p[1] for p in xy.values()]
    ax.set_xlim(-0.8, 20.5)
    ax.set_ylim(min(ys) - 1.6, max(ys) + 1.8)
    ax.axis("off")
    ax.set_title("IEEE 33-bus feeder and plant layout (%s)" % r.name, loc="left", color=INK)
    r.save(fig, "network", "feeder_layout")


def fig_voltage_envelope(r):
    years = sorted(r.mpc)
    vs, vl = [], []
    for y in years:
        n = r.net(y)
        if n is not None:
            vs.append(n["v_exact"])
            vl.append(n["v_linear"])
    if not vs:
        return
    v, l = np.vstack(vs), np.vstack(vl)
    order, xs = section_axis()
    idx = np.array(order) - 1
    lo, mean, hi = v.min(0)[idx], v.mean(0)[idx], v.max(0)[idx]
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    ax.fill_between(xs, lo, hi, color=C[0], alpha=0.18, lw=0, label="exact min-max")
    ax.plot(xs, mean, color=C[0], lw=1.8, marker="o", ms=2.5, label="exact mean")
    ax.plot(xs, l.max(0)[idx], color=C[1], lw=1.0, ls="--", label="linearised max")
    ax.plot(xs, l.min(0)[idx], color=C[1], lw=1.0, ls=":", label="linearised min")
    top = max(1.062, hi.max() + 0.004)
    mark_sections(ax, xs, top)
    for lim in (0.95, 1.05):
        ax.axhline(lim, color=LIMIT, lw=1.0, ls="--")
    ax.set_xticks(xs)
    ax.set_xticklabels(order, fontsize=6.5)
    ax.set_xlabel("bus")
    ax.set_ylabel("voltage, pu")
    ax.set_ylim(min(0.90, lo.min() - 0.005), top + 0.012)
    ax.legend(loc="lower left", ncol=4)
    ax.set_title("Per-bus voltage range over %s, %d hours (red: 0.95 / 1.05 pu limits)"
                 % ("-".join(str(y) for y in (years[0], years[-1])) if len(years) > 1
                    else years[0], len(v)), loc="left")
    r.save(fig, "network", "voltage_envelope")


def fig_voltage_snapshots(r):
    """True single-hour profiles: each line is the whole feeder in one hour."""
    y = r.plot_year()
    n = r.net(y) if y else None
    tr = r.trace(y) if y else None
    if n is None or tr is None:
        return
    v = n["v_exact"]
    net = (tr["P_load"] - tr["P_pv"] - tr["P_wind"]) / 1e6
    picks = [("highest demand hour", int(np.argmax(tr["P_load"]))),
             ("largest renewable surplus hour", int(np.argmin(net))),
             ("lowest-voltage hour", int(np.argmin(v[:, 1:].min(1)))),
             ("highest-voltage hour", int(np.argmax(v[:, 1:].max(1))))]
    order, xs = section_axis()
    idx = np.array(order) - 1
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    for (lab, h), c in zip(picks, (C[0], C[2], C[6], C[1])):
        d, hh = divmod(h, 24)
        ax.plot(xs, v[h, idx], color=c, lw=1.6, marker="o", ms=2.5,
                label="%s (day %d, %02d:00)" % (lab, d + 1, hh))
    for lim in (0.95, 1.05):
        ax.axhline(lim, color=LIMIT, lw=1.0, ls="--")
    top = max(1.062, v[:, 1:].max() + 0.004)
    mark_sections(ax, xs, top)
    ax.set_ylim(min(0.92, v.min() - 0.005), top + 0.012)
    ax.set_xticks(xs)
    ax.set_xticklabels(order, fontsize=6.5)
    ax.set_xlabel("bus")
    ax.set_ylabel("voltage, pu")
    ax.legend(loc="lower left", fontsize=7)
    ax.set_title("Feeder voltage profile in four representative hours of %d (exact)" % y,
                 loc="left")
    r.save(fig, "network", "voltage_profile_hours_%d" % y)


def fig_violations_by_bus(r):
    over, under = np.zeros(33), np.zeros(33)
    for k in r.mpc.values():
        for b, c in k.get("over_by_bus", {}).items():
            over[int(b) - 1] += c
        for b, c in k.get("under_by_bus", {}).items():
            under[int(b) - 1] += c
    order, xs = section_axis()
    idx = np.array(order) - 1
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    w = 0.38
    ax.bar(xs - w / 2, over[idx], w, color=C[1], label="overvoltage (> 1.05 pu)")
    ax.bar(xs + w / 2, under[idx], w, color=C[0], label="undervoltage (< 0.95 pu)")
    ax.set_xticks(xs)
    ax.set_xticklabels(order, fontsize=6.5)
    ax.set_xlabel("bus where the hour's worst violation occurs")
    ax.set_ylabel("hours")
    tot = sum(k.get("hours", 0) for k in r.mpc.values())
    ax.set_title("Exact-network violations by bus, MPC, %d of %d hours (%.2f %%)"
                 % (over.sum() + under.sum(), tot, 100 * (over.sum() + under.sum()) / max(tot, 1)),
                 loc="left")
    if over.sum() + under.sum() == 0:
        ax.text(0.5, 0.5, "no violating hours", transform=ax.transAxes, ha="center",
                color=INK2)
        ax.set_ylim(0, 1)
    ax.legend(loc="upper left")
    r.save(fig, "network", "violations_by_bus")


def fig_branch_power(r):
    ps = [r.net(y)["P_br"] for y in sorted(r.mpc) if r.net(y) is not None]
    if not ps:
        return
    P = np.vstack(ps)
    names = [str(b) for _, b in BRANCHES]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    ax.fill_between(x, P.min(0), P.max(0), color=C[2], alpha=0.25, lw=0, label="min-max")
    ax.plot(x, P.mean(0), color=C[2], lw=1.8, label="mean")
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=6.5)
    ax.set_xlabel("branch, named by its downstream bus")
    ax.set_ylabel("active power, MW")
    rev = int(np.sum((P < -1e-6).any(0)))
    ax.set_title("Active power along the feeder (exact), positive = away from bus 1; "
                 "%d of 32 branches reverse in some hour" % rev, loc="left")
    ax.legend(loc="upper right")
    r.save(fig, "network", "branch_active_power")


def fig_voltage_year(r):
    y = r.plot_year()
    n = r.net(y) if y else None
    if n is None:
        return
    v = n["v_exact"][:, 1:]
    T = len(v)
    d = T // 24
    vmin = v.min(1)[:d * 24].reshape(d, 24)
    vmax = v.max(1)[:d * 24].reshape(d, 24)
    days = np.arange(d)
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    ax.fill_between(days, vmin.min(1), vmax.max(1), color=C[0], alpha=0.18, lw=0,
                    label="daily range of feeder min / max")
    ax.plot(days, vmax.max(1), color=C[1], lw=1.0, label="daily highest bus voltage")
    ax.plot(days, vmin.min(1), color=C[0], lw=1.0, label="daily lowest bus voltage")
    for lim in (0.95, 1.05):
        ax.axhline(lim, color=LIMIT, lw=1.0, ls="--")
    ax.set_xlim(0, d)
    ax.set_xlabel("day of %d" % y)
    ax.set_ylabel("voltage, pu")
    ax.legend(loc="lower left", ncol=3)
    ax.set_title("Feeder voltage extremes through %d (exact)" % y, loc="left")
    r.save(fig, "network", "voltage_extremes_%d" % y)


def fig_linear_vs_exact(r):
    y = r.plot_year()
    n = r.net(y) if y else None
    if n is None:
        return
    ve, vl = n["v_exact"][:, 1:], n["v_linear"][:, 1:]
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.0))
    for ax, fn, lab, col in ((axs[0], np.min, "lowest", C[0]), (axs[1], np.max, "highest", C[1])):
        a, b = fn(vl, 1), fn(ve, 1)
        ax.scatter(a, b, s=4, color=col, alpha=0.35, lw=0)
        lo, hi = min(a.min(), b.min()) - 0.003, max(a.max(), b.max()) + 0.003
        ax.plot([lo, hi], [lo, hi], color=INK2, lw=0.8)
        for lim in (0.95, 1.05):
            if lo < lim < hi:
                ax.axhline(lim, color=LIMIT, lw=0.8, ls="--")
                ax.axvline(lim, color=LIMIT, lw=0.8, ls="--")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel("linearised (planning) model, pu")
        ax.set_ylabel("exact DistFlow, pu")
        ax.set_title("%s bus voltage each hour, %d" % (lab.capitalize(), y), loc="left")
    fig.tight_layout()
    r.save(fig, "network", "linear_vs_exact_%d" % y)


# ───────────────────────────────────────────────────────────── MPC figures
def fig_dispatch_week(r, tr=None, sub="mpc", label="MPC"):
    y = r.plot_year()
    if tr is None:
        tr = r.trace(y) if y else None
    if tr is None:
        return
    d0 = int(getattr(r.cfg, "PLOT_WEEK_START_DAY", 90))
    s = slice(24 * d0, min(24 * (d0 + 7), len(tr["P_load"])))
    if s.start >= len(tr["P_load"]):
        s = slice(0, min(168, len(tr["P_load"])))
    h = np.arange(s.stop - s.start)
    L, pv, wd = tr["P_load"][s] / 1e6, tr["P_pv"][s] / 1e6, tr["P_wind"][s] / 1e6
    pb, ph = tr["act_b"][s] / 1e6, tr["act_h"][s] / 1e6
    fig, axs = plt.subplots(3, 1, figsize=(7.2, 6.0), sharex=True)
    ax = axs[0]
    ax.plot(h, L, color=INK, lw=1.6, label="demand")
    ax.plot(h, pv, color=C[3], label="PV available")
    ax.plot(h, wd, color=C[2], label="wind available")
    if "P_def" in tr:
        ax.fill_between(h, 0, tr["P_def"][s] / 1e6, color=C[7], alpha=0.25, lw=0,
                        label="unserved")
    ax.set_ylabel("MW")
    ax.set_ylim(0, max(L.max(), pv.max(), wd.max()) * 1.35)
    ax.legend(loc="upper left", ncol=4)
    ax.set_title("Demand, renewables and unserved demand", loc="left")
    ax = axs[1]
    ax.axhline(0, color=INK2, lw=0.8)
    ax.plot(h, pb, color=C[0], label="battery, delivered (+ discharge)")
    ax.plot(h, ph, color=C[6], label="hydrogen, delivered (+ FC, - EL)")
    if "schedule" in tr:
        sc = tr["schedule"][s]
        ax.plot(h, sc[:, 0], color=C[0], lw=1.0, ls="--", label="battery, day-ahead schedule")
        ax.plot(h, sc[:, 1], color=C[6], lw=1.0, ls="--", label="hydrogen, day-ahead schedule")
        for a_ in axs:
            for d_ in range(0, len(h), 24):
                a_.axvline(d_, color=MUTED, lw=0.6, ls=":")
    ax.set_ylabel("MW")
    lim = max(np.abs(pb).max(), np.abs(ph).max(), 0.1)
    ax.set_ylim(-1.1 * lim, 1.9 * lim)
    ax.legend(loc="upper left", ncol=2, fontsize=7)
    ax.set_title("Storage dispatch (dotted lines: 00:00, when each day-ahead schedule is "
                 "committed)" if "schedule" in tr else "Storage dispatch", loc="left")
    ax = axs[2]
    if "soc" in tr:
        ax.plot(h, tr["soc"][s], color=C[0], label="battery SoC")
    if "soh2" in tr:
        ax.plot(h, tr["soh2"][s], color=C[6], label="hydrogen tank level")
    ax.set_ylim(0, 1)
    ax.set_ylabel("fraction")
    ax.set_xlabel("hour from day %d of %d" % (d0, y))
    ax.legend(loc="upper left", ncol=2)
    ax.set_title("Storage state", loc="left")
    fig.suptitle("%s, one week of day-ahead operation" % label, x=0.01, ha="left", y=1.0,
                 fontsize=10)
    fig.tight_layout()
    r.save(fig, sub, "dispatch_week_%d" % y)


def fig_mpc_years(r):
    if not r.mpc:
        return
    ys = sorted(r.mpc)
    x = np.arange(len(ys))
    dfc = [r.mpc[y].get("D_L_corrected_pct", np.nan) for y in ys]
    kap = [r.mpc[y]["kappa_exact_pct"] for y in ys]
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.8))
    axs[0].bar(x, dfc, 0.55, color=C[0])
    axs[0].set_ylabel("demand deficit, %")
    axs[0].set_title("MPC deficit (tank-corrected)", loc="left")
    axs[1].bar(x, kap, 0.55, color=C[2])
    axs[1].set_ylabel("hours inside 0.95-1.05 pu, %")
    axs[1].set_ylim(min(90, min(kap) - 1), 100.5)
    axs[1].set_title("MPC exact voltage compliance", loc="left")
    for ax, vals in ((axs[0], dfc), (axs[1], kap)):
        ax.set_xticks(x)
        ax.set_xticklabels(ys)
        for xi, v in zip(x, vals):
            ax.text(xi, v, "%.2f" % v, ha="center", va="bottom", fontsize=7, color=INK)
    fig.tight_layout()
    r.save(fig, "mpc", "kpis_by_year")


# ───────────────────────────────────────────────────────────── DRL figures
def drl_table(r):
    """{algo: {year: [results over seeds]}}"""
    t = {}
    for k in r.drl:
        t.setdefault(k["algo"], {}).setdefault(int(k["test_year"]), []).append(k)
    return t


def fig_drl_vs_mpc(r):
    t = drl_table(r)
    if not t:
        return
    ys = sorted(set(r.mpc) | {y for a in t.values() for y in a})
    methods = ["MPC"] + [a for a in ALGO_ORDER if a in t]
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.0))
    w = 0.8 / len(methods)
    for j, mth in enumerate(methods):
        for ax, key in ((axs[0], "D_L_corrected_pct"), (axs[1], "kappa_exact_pct")):
            vals, errs = [], []
            for y in ys:
                if mth == "MPC":
                    v = [r.mpc[y][key]] if y in r.mpc else []
                else:
                    v = [k.get(key, np.nan) for k in t[mth].get(y, [])]
                vals.append(np.mean(v) if v else np.nan)
                errs.append(np.std(v) if len(v) > 1 else 0.0)
            ax.bar(np.arange(len(ys)) + (j - (len(methods) - 1) / 2) * w, vals, w * 0.92,
                   yerr=errs if any(errs) else None, color=ALGO_COL[mth],
                   label=ALGO_NAME.get(mth, mth), error_kw={"lw": 0.8, "ecolor": INK2})
    axs[0].set_ylabel("demand deficit, %")
    axs[0].set_title("Deficit on the held-out year", loc="left")
    axs[1].set_ylabel("exact compliance, %")
    axs[1].set_title("Exact voltage compliance", loc="left")
    lo = np.nanmin([k.get("kappa_exact_pct", 100) for k in r.drl] +
                   [k["kappa_exact_pct"] for k in r.mpc.values()] or [95])
    axs[1].set_ylim(min(95, lo - 1), 100.3)
    for ax in axs:
        ax.set_xticks(np.arange(len(ys)))
        ax.set_xticklabels(ys)
    axs[0].legend(loc="upper left", ncol=2)
    fig.tight_layout()
    r.save(fig, "drl", "mpc_vs_drl_by_year")


def fig_learning(r):
    hs = sorted(glob.glob(os.path.join(r.dir, "drl", "history_*.json")))
    if not hs:
        return
    by = {}
    for f in hs:
        algo = os.path.basename(f)[len("history_"):].split("_")[0]
        h = core.read_json(f)
        pts = [(e["ep"], e["val_deficit"]) for e in h if "val_deficit" in e]
        if pts:
            by.setdefault(algo, []).append(np.array(pts))
    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    for algo in ALGO_ORDER:
        for i, p in enumerate(by.get(algo, [])):
            ax.plot(p[:, 0], p[:, 1], color=ALGO_COL[algo], lw=1.0, alpha=0.8,
                    label=ALGO_NAME[algo] if i == 0 else None)
    if r.mpc:
        ax.axhline(np.mean([k["D_L_corrected_pct"] for k in r.mpc.values()]), color=C[0],
                   ls="--", lw=1.0, label="MPC mean (test years)")
    ax.set_xlabel("training episode (one year of hours each)")
    ax.set_ylabel("validation deficit, %")
    ax.set_title("DRL learning curves: deficit on 2017-2018 at each checkpoint", loc="left")
    ax.legend(loc="upper right", ncol=3)
    r.save(fig, "drl", "learning_curves")



# ───────────────────────────────────────────────────────────── DRL convergence and comparison
def _smooth(x, w=5):
    x = np.asarray(x, float)
    if len(x) < w:
        return x
    k = np.ones(w) / w
    y = np.convolve(x, k, mode="valid")
    return np.concatenate([np.full(w - 1, np.nan), y])


def fig_drl_convergence(r):
    hs = sorted(glob.glob(os.path.join(r.dir, "drl", "history_*.json")))
    if not hs:
        return
    by = {}
    for f in hs:
        algo = os.path.basename(f)[len("history_"):].split("_")[0]
        by.setdefault(algo, []).append(core.read_json(f))
    fig, axs = plt.subplots(2, 2, figsize=(7.2, 5.4))
    panels = [(axs[0, 0], "train_return", "return per training year",
               "Training return (higher = lower cost)"),
              (axs[0, 1], "val_deficit", "validation deficit, %",
               "Validation deficit, 2017-2018 (selection metric)"),
              (axs[1, 0], "train_cost", "voltage-violation cost (sum over the year)",
               "Constraint cost during training"),
              (axs[1, 1], "mult", "Lagrange multiplier", "Dual variable (constrained methods)")]
    for ax, key, ylab, title in panels:
        for algo in ALGO_ORDER:
            if algo in ("il", "miqp"):
                continue
            runs = by.get(algo, [])
            curves = []
            for h in runs:
                pts = [(e["ep"], e[key]) for e in h if key in e and e[key] is not None]
                if pts:
                    curves.append(np.array(pts, float))
            if not curves:
                continue
            n = min(len(c) for c in curves)
            ep = curves[0][:n, 0]
            Y = np.vstack([c[:n, 1] for c in curves])
            m = Y.mean(0)
            if key == "train_return" and len(m) > 15:
                m = _smooth(m)
            ax.plot(ep, m, color=ALGO_COL[algo], lw=1.6, label=ALGO_NAME[algo])
            if len(curves) > 1:
                ax.fill_between(ep, Y.min(0), Y.max(0), color=ALGO_COL[algo], alpha=0.12, lw=0)
        if key == "val_deficit" and r.mpc:
            pass
        ax.set_xlabel("episode")
        ax.set_ylabel(ylab)
        ax.set_title(title, loc="left", fontsize=8.5)
    h, l = axs[0, 1].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.legend(h, l, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle("DRL convergence (mean over test-year cells, band = min-max); "
                 "episode 0 = policy initialised from the MPC", y=0.955, fontsize=8, color=INK2)
    r.save(fig, "drl", "convergence")


def fig_methods_summary(r):
    """Five-year means per method: deficit, exact compliance, losses, curtailment."""
    if not r.mpc or not r.drl:
        return
    t = drl_table(r)
    ys = sorted(r.mpc)
    methods = ["MPC"] + [a for a in ALGO_ORDER if a in t]
    keys = [("D_L_corrected_pct", "deficit, %"), ("kappa_exact_pct", "exact compliance, %"),
            ("loss_MWh", "losses, MWh/yr"), ("C_curt_pct", "curtailment, %")]
    fig, axs = plt.subplots(1, 4, figsize=(7.2, 2.6))
    for ax, (k, lab) in zip(axs, keys):
        vals, errs = [], []
        for mth in methods:
            if mth == "MPC":
                v = [r.mpc[y].get(k, np.nan) for y in ys]
            else:
                v = [c.get(k, np.nan) for y in ys for c in t[mth].get(y, [])]
            vals.append(np.nanmean(v) if v else np.nan)
            errs.append(np.nanstd(v) if len(v) > 1 else 0.0)
        x = np.arange(len(methods))
        ax.bar(x, vals, 0.65, yerr=errs, color=[ALGO_COL[m] for m in methods],
               error_kw={"lw": 0.8, "ecolor": INK2})
        ax.set_xticks(x)
        ax.set_xticklabels([ALGO_NAME.get(m, m) for m in methods], rotation=45, ha="right",
                           fontsize=7)
        ax.set_title(lab, loc="left", fontsize=8.5)
        if k == "kappa_exact_pct":
            lo = np.nanmin(vals)
            ax.set_ylim(min(95, lo - 1), 100.3)
    fig.suptitle("Methods over the held-out years (bar = mean, whisker = std over years and seeds)",
                 x=0.01, ha="left", fontsize=9)
    fig.tight_layout()
    r.save(fig, "drl", "methods_summary")


def fig_drl_system(r):
    """Per DRL method: dispatch week and exact voltage envelope; and all methods' envelopes."""
    y = r.plot_year()
    if not r.drl or y is None:
        return
    order, xs = section_axis()
    idx = np.array(order) - 1
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    top = 1.062
    meths = [("MPC", None)] + [(a, a) for a in r.algos()]
    for name, algo in meths:
        vs = []
        years = sorted(r.mpc) if algo is None else sorted({k["test_year"] for k in r.drl
                                                            if k["algo"] == algo})
        for yy in years:
            n = r.net(yy) if algo is None else r.drl_file("network", algo, yy)
            if n is not None:
                vs.append(n["v_exact"])
        if not vs:
            continue
        v = np.vstack(vs)
        c = ALGO_COL[name if algo is None else algo]
        ax.plot(xs, v.max(0)[idx], color=c, lw=1.0, ls="--")
        ax.plot(xs, v.min(0)[idx], color=c, lw=1.0, ls="--")
        ax.plot(xs, v.mean(0)[idx], color=c, lw=1.6, marker="o", ms=2,
                label=ALGO_NAME.get(name, name))
        top = max(top, v.max() + 0.004)
    for lim in (0.95, 1.05):
        ax.axhline(lim, color=LIMIT, lw=1.0, ls="--")
    mark_sections(ax, xs, top)
    ax.set_ylim(None, top + 0.012)
    ax.set_xticks(xs)
    ax.set_xticklabels(order, fontsize=6.5)
    ax.set_xlabel("bus")
    ax.set_ylabel("voltage, pu")
    ax.legend(loc="lower left", ncol=5, fontsize=7)
    ax.set_title("Exact bus voltages over the held-out years: mean (solid), min/max (dashed)",
                 loc="left")
    r.save(fig, "drl", "voltage_envelope_methods")


# ───────────────────────────────────────────────────────────── the 24 h operation day
def _cmap_seq():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("seq", ["#dce9f8", "#2a78d6", "#0d2f5c"])


def _cmap_div():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("div", ["#1d5fae", "#e9e8e4", "#c4501f"])


def operation_day(r):
    """Which day to show: OPERATION_DATE in the config, else the day with the widest voltage swing."""
    import datetime
    d = getattr(r.cfg, "OPERATION_DATE", None)
    if d:
        dt = datetime.date.fromisoformat(str(d))
        return dt.year, dt.timetuple().tm_yday - 1, dt.isoformat()
    y = r.plot_year()
    n = r.net(y) if y else None
    if n is None:
        return None
    v = n["v_exact"][:, 1:]
    D = len(v) // 24
    sw = v[:D * 24].max(1).reshape(D, 24).max(1) - v[:D * 24].min(1).reshape(D, 24).min(1)
    day = int(np.argmax(sw))
    dt = datetime.date(y, 1, 1) + datetime.timedelta(days=day)
    return y, day, dt.isoformat()


def fig_operation_day(r):
    """The day-ahead problem for one day: the 24 h schedule committed at 00:00, what was
    delivered, and the feeder's exact voltages and active/reactive flows over those 24 h."""
    od = operation_day(r)
    if od is None:
        return
    y, day, date = od
    s = slice(24 * day, 24 * day + 24)
    h = np.arange(24)
    meths = [("MPC", r.trace(y), r.net(y))]
    for a in r.algos():
        meths.append((ALGO_NAME[a], r.drl_file("trace", a, y), r.drl_file("network", a, y)))
    meths = [m for m in meths if m[1] is not None and m[2] is not None]
    if not meths:
        return
    sub = os.path.join("operation_%s" % date)
    cmp_rows = []
    for name, tr, net in meths:
        col = ALGO_COL.get(name, C[0]) if name == "MPC" else \
            ALGO_COL[[k for k, v in ALGO_NAME.items() if v == name][0]]
        L, pv, wd = tr["P_load"][s] / 1e6, tr["P_pv"][s] / 1e6, tr["P_wind"][s] / 1e6
        pb, ph, qh = tr["act_b"][s] / 1e6, tr["act_h"][s] / 1e6, tr["qh"][s]
        sc = tr["schedule"][s] if "schedule" in tr else None
        v = net["v_exact"][s]
        P, Q = net["P_br"][s], net["Q_br"][s]
        Ps = net["P_slack"][s]
        Qs = net["Q_slack"][s] if "Q_slack" in net else np.full(24, np.nan)

        # (1) schedule and operation
        fig, axs = plt.subplots(3, 1, figsize=(7.2, 6.4), sharex=True)
        ax = axs[0]
        ax.plot(h, L, color=INK, label="demand")
        ax.plot(h, pv, color=C[3], label="PV")
        ax.plot(h, wd, color=C[2], label="wind")
        if "P_def" in tr:
            ax.fill_between(h, 0, tr["P_def"][s] / 1e6, color=C[7], alpha=0.25, lw=0,
                            label="unserved")
        ax.set_ylabel("MW")
        ax.set_ylim(0, max(L.max(), pv.max(), wd.max()) * 1.4)
        ax.legend(loc="upper left", ncol=4)
        ax.set_title("Demand and renewables (realised)", loc="left")
        ax = axs[1]
        ax.axhline(0, color=INK2, lw=0.8)
        ax.plot(h, pb, color=C[0], marker="o", ms=2.5, label="battery, delivered")
        ax.plot(h, ph, color=C[6], marker="o", ms=2.5, label="hydrogen, delivered")
        ax.plot(h, qh, color=C[1], marker="o", ms=2.5, label="converter Q, Mvar")
        if sc is not None:
            ax.plot(h, sc[:, 0], color=C[0], ls="--", lw=1.0, label="battery, 00:00 schedule")
            ax.plot(h, sc[:, 1], color=C[6], ls="--", lw=1.0, label="hydrogen, 00:00 schedule")
        lim = max(0.2, np.abs(np.concatenate([pb, ph, qh])).max())
        ax.set_ylim(-1.15 * lim, 2.0 * lim)
        ax.set_ylabel("MW / Mvar")
        ax.legend(loc="upper left", ncol=3, fontsize=6.8)
        ax.set_title("24 h schedule committed at 00:00 (dashed) and delivered (solid)", loc="left")
        ax = axs[2]
        if "soc" in tr:
            ax.plot(h, tr["soc"][s], color=C[0], label="battery SoC")
        if "soh2" in tr:
            ax.plot(h, tr["soh2"][s], color=C[6], label="hydrogen tank")
        ax.set_ylim(0, 1)
        ax.set_ylabel("fraction")
        ax.set_xlabel("hour of %s" % date)
        ax.set_xticks(range(0, 24, 2))
        ax.legend(loc="upper left", ncol=2)
        ax.set_title("Storage state", loc="left")
        fig.suptitle("%s - day-ahead operation of %s" % (name, date), x=0.01, ha="left",
                     fontsize=10)
        fig.tight_layout()
        r.save(fig, sub, "%s_schedule" % name)

        # (2) voltage over the 24 h
        order, xs = section_axis()
        idx = np.array(order) - 1
        fig, axs = plt.subplots(2, 1, figsize=(7.2, 5.6),
                                gridspec_kw={"height_ratios": [1.15, 1]})
        ax = axs[0]
        vmax_dev = max(0.051, np.abs(v - 1.0).max())
        im = ax.imshow(v[:, idx].T, aspect="auto", origin="lower", cmap=_cmap_div(),
                       vmin=1 - vmax_dev, vmax=1 + vmax_dev, extent=(-0.5, 23.5, 0.5, 33.5))
        cb = fig.colorbar(im, ax=ax, pad=0.01)
        cb.set_label("voltage, pu")
        for lim_ in (0.95, 1.05):
            cb.ax.axhline(lim_, color=INK, lw=0.8)
        ax.set_ylabel("bus")
        ax.set_xlabel("hour")
        ax.set_xticks(range(0, 24, 2))
        ax.set_yticks([1, 6, 12, 18, 22, 25, 33])
        ax.grid(False)
        for b_ in (18.5, 22.5, 25.5):
            ax.axhline(b_, color="white", lw=0.8)
        ax.set_title("Exact bus voltage, every bus and hour", loc="left")
        ax = axs[1]
        cm = _cmap_seq()
        for k in range(24):
            ax.plot(xs, v[k, idx], color=cm(k / 23.0), lw=1.0)
        for lim_ in (0.95, 1.05):
            ax.axhline(lim_, color=LIMIT, lw=1.0, ls="--")
        top = max(1.06, v.max() + 0.004)
        mark_sections(ax, xs, top)
        ax.set_ylim(min(0.93, v.min() - 0.005), top + 0.012)
        ax.set_xticks(xs)
        ax.set_xticklabels(order, fontsize=6.5)
        ax.set_xlabel("bus")
        ax.set_ylabel("voltage, pu")
        sm = plt.cm.ScalarMappable(cmap=cm, norm=plt.Normalize(0, 23))
        cb = fig.colorbar(sm, ax=ax, pad=0.01)
        cb.set_label("hour")
        ax.set_title("Voltage profile along the feeder, one line per hour", loc="left")
        fig.suptitle("%s - feeder voltage on %s" % (name, date), x=0.01, ha="left", fontsize=10)
        fig.tight_layout()
        r.save(fig, sub, "%s_voltage" % name)

        # (3) active and reactive power over the 24 h
        fig, axs = plt.subplots(3, 1, figsize=(7.2, 7.0),
                                gridspec_kw={"height_ratios": [0.8, 1, 1]})
        ax = axs[0]
        ax.axhline(0, color=INK2, lw=0.8)
        ax.plot(h, Ps, color=C[0], marker="o", ms=2.5, label="P at bus 1, MW (import +)")
        ax.plot(h, Qs, color=C[1], marker="o", ms=2.5, label="Q at bus 1, Mvar (import +)")
        ax.plot(h, net["loss_p"][s].sum(1), color=INK2, ls=":", label="feeder losses, MW")
        ax.set_xticks(range(0, 24, 2))
        ax.set_xlabel("hour")
        ax.legend(loc="upper left", ncol=3, fontsize=7)
        ax.set_title("Substation exchange and losses", loc="left")
        names = [str(b) for _, b in BRANCHES]
        for ax, M, lab in ((axs[1], P, "active power, MW"), (axs[2], Q, "reactive power, Mvar")):
            m = max(1e-3, np.abs(M).max())
            im = ax.imshow(M.T, aspect="auto", origin="lower", cmap=_cmap_div(), vmin=-m, vmax=m,
                           extent=(-0.5, 23.5, -0.5, len(names) - 0.5))
            cb = fig.colorbar(im, ax=ax, pad=0.01)
            cb.set_label(lab)
            ax.set_yticks(range(0, len(names), 2))
            ax.set_yticklabels(names[::2], fontsize=6.5)
            ax.set_ylabel("branch (to bus)")
            ax.grid(False)
            ax.set_xticks(range(0, 24, 2))
            ax.set_xlabel("hour")
            ax.set_title("Branch %s (orange = away from bus 1, blue = reverse)" % lab.split(",")[0],
                         loc="left")
        fig.suptitle("%s - power flows on %s (exact DistFlow)" % (name, date), x=0.01, ha="left",
                     fontsize=10)
        fig.tight_layout()
        r.save(fig, sub, "%s_power_flow" % name)

        # 24 h table
        rows = []
        for k in range(24):
            rows.append({"hour": k, "load_MW": L[k], "pv_MW": pv[k], "wind_MW": wd[k],
                         "sched_bess_MW": sc[k, 0] if sc is not None else "",
                         "sched_h2_MW": sc[k, 1] if sc is not None else "",
                         "sched_q_Mvar": sc[k, 2] if sc is not None else "",
                         "bess_MW": pb[k], "h2_MW": ph[k], "q_Mvar": qh[k],
                         "soc": tr["soc"][s][k] if "soc" in tr else "",
                         "tank": tr["soh2"][s][k] if "soh2" in tr else "",
                         "unserved_MW": tr["P_def"][s][k] / 1e6 if "P_def" in tr else "",
                         "v_min_pu": v[k, 1:].min(), "v_max_pu": v[k, 1:].max(),
                         "P_bus1_MW": Ps[k], "Q_bus1_Mvar": Qs[k]})
        d_ = os.path.join(r.fig, sub)
        os.makedirs(d_, exist_ok=True)
        with io.open(os.path.join(d_, "%s_24h.csv" % name), "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            for row in rows:
                w.writerow({k: (round(v_, 5) if isinstance(v_, (float, np.floating)) else v_)
                            for k, v_ in row.items()})
        cmp_rows.append((name, col, h, v, Ps, Qs, pb, ph, tr))

    # (4) all methods on the same day
    fig, axs = plt.subplots(2, 2, figsize=(7.2, 5.2), sharex=True)
    for name, col, h_, v, Ps, Qs, pb, ph, tr in cmp_rows:
        axs[0, 0].plot(h_, v[:, 1:].max(1), color=col, lw=1.4, label=name)
        axs[0, 0].plot(h_, v[:, 1:].min(1), color=col, lw=1.0, ls="--")
        axs[0, 1].plot(h_, (tr["P_def"][s] / 1e6) if "P_def" in tr else np.zeros(24), color=col,
                       lw=1.4)
        axs[1, 0].plot(h_, pb, color=col, lw=1.4)
        axs[1, 1].plot(h_, ph, color=col, lw=1.4)
    for lim_ in (0.95, 1.05):
        axs[0, 0].axhline(lim_, color=LIMIT, lw=0.9, ls="--")
    axs[0, 0].set_title("Highest (solid) and lowest (dashed) bus voltage, pu", loc="left",
                        fontsize=8.5)
    axs[0, 1].set_title("Unserved demand, MW", loc="left", fontsize=8.5)
    axs[1, 0].set_title("Battery, MW (+ discharge)", loc="left", fontsize=8.5)
    axs[1, 1].set_title("Hydrogen, MW (+ FC, - EL)", loc="left", fontsize=8.5)
    for ax in axs[1]:
        ax.set_xlabel("hour of %s" % date)
        ax.set_xticks(range(0, 24, 3))
    h0, l0 = axs[0, 0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.legend(h0, l0, loc="upper center", ncol=len(cmp_rows), bbox_to_anchor=(0.5, 1.0))
    r.save(fig, sub, "methods_compared")

# ───────────────────────────────────────────────────────────── utilisation of the plant
SUPPLY = [("PV", "pv_MWh", C[3]), ("wind", "wind_MWh", C[2]), ("battery", "bess_dis_MWh", C[0]),
          ("fuel cell", "fc_MWh", C[6]), ("unserved", "unserved_MWh", C[7])]


def utilisation(tr, cfg, s=slice(None)):
    """Energy accounting of one trace (whole year or one day). MWh and %."""
    L = tr["P_load"][s] / 1e6
    pv, wd = tr["P_pv"][s] / 1e6, tr["P_wind"][s] / 1e6
    ab, ah = tr["act_b"][s] / 1e6, tr["act_h"][s] / 1e6
    res = tr["P_res"][s] / 1e6 if "P_res" in tr else pv + wd
    avail = pv + wd
    curt = np.minimum(np.clip(res + ab + ah - L, 0.0, None), avail)
    share = np.divide(pv, avail, out=np.zeros_like(pv), where=avail > 0)
    pv_d, wd_d = pv - curt * share, wd - curt * (1 - share)
    dis, chg = np.clip(ab, 0, None), np.clip(-ab, 0, None)
    fc, el = np.clip(ah, 0, None), np.clip(-ah, 0, None)
    uns = tr["P_def"][s] / 1e6 if "P_def" in tr else np.clip(L - pv_d - wd_d - ab - ah, 0, None)
    # renewable energy that went straight to demand (the rest charged the battery or the EL)
    re_d = pv_d + wd_d
    to_load = np.clip(re_d - chg - el, 0.0, None)
    f = np.divide(to_load, re_d, out=np.zeros_like(re_d), where=re_d > 0)
    pv_load, wd_load = pv_d * f, wd_d * f
    n = len(L)
    E = lambda x: float(np.sum(x))
    load = E(L)
    out = {"hours": n, "demand_MWh": load,
           "pv_avail_MWh": E(pv), "wind_avail_MWh": E(wd),
           "pv_MWh": E(pv_d), "wind_MWh": E(wd_d), "curtailed_MWh": E(curt),
           "bess_dis_MWh": E(dis), "bess_chg_MWh": E(chg), "fc_MWh": E(fc), "el_MWh": E(el),
           "unserved_MWh": E(uns), "pv_to_load_MWh": E(pv_load), "wind_to_load_MWh": E(wd_load)}
    pct = lambda a, b: 100.0 * a / b if b > 0 else float("nan")
    out.update({
        "PV used, % of available": pct(out["pv_MWh"], out["pv_avail_MWh"]),
        "wind used, % of available": pct(out["wind_MWh"], out["wind_avail_MWh"]),
        "renewables used, % of available": pct(out["pv_MWh"] + out["wind_MWh"],
                                               out["pv_avail_MWh"] + out["wind_avail_MWh"]),
        "demand from PV, %": pct(out["pv_to_load_MWh"], load),
        "demand from wind, %": pct(out["wind_to_load_MWh"], load),
        "demand from battery, %": pct(out["bess_dis_MWh"], load),
        "demand from fuel cell, %": pct(out["fc_MWh"], load),
        "demand unserved, %": pct(out["unserved_MWh"], load),
        "renewable energy to electrolyser, %": pct(out["el_MWh"], out["pv_MWh"] + out["wind_MWh"]),
        "renewable energy to battery, %": pct(out["bess_chg_MWh"], out["pv_MWh"] + out["wind_MWh"]),
        "electrolyser utilisation, % of rating x hours": pct(out["el_MWh"], float(cfg.EL_MW) * n),
        "fuel cell utilisation, % of rating x hours": pct(out["fc_MWh"], float(cfg.FC_MW) * n),
        "electrolyser running hours, %": pct(float(np.sum(el > 1e-6)), n),
        "fuel cell running hours, %": pct(float(np.sum(fc > 1e-6)), n),
        "battery full cycles": out["bess_dis_MWh"] / max(float(cfg.BESS_USABLE_MWH), 1e-9),
    })
    if "schedule" in tr:                     # what the day-ahead plan intended
        sc = tr["schedule"][s]
        out["planned_el_MWh"] = E(np.clip(-sc[:, 1], 0, None))
        out["planned_fc_MWh"] = E(np.clip(sc[:, 1], 0, None))
        out["planned_bess_dis_MWh"] = E(np.clip(sc[:, 0], 0, None))
        out["planned_bess_chg_MWh"] = E(np.clip(-sc[:, 0], 0, None))
    return out


def _method_traces(r):
    """[(display name, colour key, {year: trace})] for the MPC and each DRL method (seed 0)."""
    out = [("MPC", "MPC", {y: r.trace(y) for y in sorted(r.mpc)})]
    for a in r.algos():
        yrs = sorted({k["test_year"] for k in r.drl if k["algo"] == a})
        out.append((ALGO_NAME[a], a, {y: r.drl_file("trace", a, y) for y in yrs}))
    return [(n, c, {y: t for y, t in d.items() if t is not None}) for n, c, d in out]


UTIL_COLS = ["PV used, % of available", "wind used, % of available",
             "renewables used, % of available", "demand from PV, %", "demand from wind, %",
             "demand from battery, %", "demand from fuel cell, %", "demand unserved, %",
             "renewable energy to electrolyser, %", "renewable energy to battery, %",
             "electrolyser utilisation, % of rating x hours",
             "fuel cell utilisation, % of rating x hours", "electrolyser running hours, %",
             "fuel cell running hours, %", "battery full cycles"]


def _write_table(path_csv, path_md, rows, title):
    if not rows:
        return
    with io.open(path_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        for row in rows:
            w.writerow({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()})
    keys = list(rows[0])
    lines = ["# " + title, "", "| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)]
    for row in rows:
        lines.append("| " + " | ".join(("%.2f" % v) if isinstance(v, float) else str(v)
                                       for v in row.values()) + " |")
    with io.open(path_md, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def fig_utilisation(r):
    """Annual utilisation of PV, wind, battery and hydrogen under day-ahead operation."""
    meths = _method_traces(r)
    if not meths or not meths[0][2]:
        return
    d = os.path.join(r.fig, "utilisation")
    os.makedirs(d, exist_ok=True)
    rows, mean = [], {}
    for name, ck, trs in meths:
        us = []
        for y, tr in trs.items():
            u = utilisation(tr, r.cfg)
            us.append(u)
            rows.append(dict([("method", name), ("year", y)] + [(c, u[c]) for c in UTIL_COLS]))
        mean[name] = (ck, {c: float(np.nanmean([u[c] for u in us])) for c in UTIL_COLS})
    _write_table(os.path.join(d, "utilisation_by_year.csv"),
                 os.path.join(d, "utilisation_by_year.md"), rows,
                 "Plant utilisation by method and year (day-ahead operation)")
    summ = [dict([("method", n)] + list(v[1].items())) for n, v in mean.items()]
    _write_table(os.path.join(d, "utilisation_summary.csv"),
                 os.path.join(d, "utilisation_summary.md"), summ,
                 "Plant utilisation, mean over the held-out years")

    # (a) where the demand was supplied from (stacked, one bar per method)
    names = list(mean)
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    x = np.arange(len(names))
    bottom = np.zeros(len(names))
    for lab, key, col in SUPPLY:
        k = {"PV": "demand from PV, %", "wind": "demand from wind, %",
             "battery": "demand from battery, %", "fuel cell": "demand from fuel cell, %",
             "unserved": "demand unserved, %"}[lab]
        v = np.array([mean[n][1][k] for n in names])
        ax.bar(x, v, 0.6, bottom=bottom, color=col, label=lab, edgecolor="white", lw=1.0,
               alpha=0.35 if lab == "unserved" else 1.0)
        for xi, (b_, vi) in enumerate(zip(bottom, v)):
            if vi > 3:
                ax.text(xi, b_ + vi / 2, "%.1f" % vi, ha="center", va="center", fontsize=7,
                        color=INK)
        bottom += v
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel("share of demand, %")
    ax.set_ylim(0, 105)
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    ax.set_title("Where the demand was supplied from, mean over the held-out years", loc="left")
    r.save(fig, "utilisation", "demand_supply_mix")

    # (b) utilisation of each component
    groups = [("Renewables used, % of available",
               ["PV used, % of available", "wind used, % of available"], ["PV", "wind"]),
              ("Hydrogen and battery",
               ["electrolyser utilisation, % of rating x hours",
                "fuel cell utilisation, % of rating x hours",
                "electrolyser running hours, %", "fuel cell running hours, %"],
               ["EL energy\n% of rating", "FC energy\n% of rating", "EL hours\nrunning %",
                "FC hours\nrunning %"])]
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.0), gridspec_kw={"width_ratios": [1, 2]})
    for ax, (title, keys, labels) in zip(axs, groups):
        w = 0.8 / len(names)
        for j, n in enumerate(names):
            ck = mean[n][0]
            ax.bar(np.arange(len(keys)) + (j - (len(names) - 1) / 2) * w,
                   [mean[n][1][k] for k in keys], w * 0.92, color=ALGO_COL[ck], label=n)
        ax.set_xticks(np.arange(len(keys)))
        ax.set_xticklabels(labels, fontsize=7.5)
        ax.set_ylabel("%")
        ax.set_title(title, loc="left", fontsize=9)
    lo = min(mean[n][1][k] for n in names for k in groups[0][1])
    axs[0].set_ylim(max(0, lo - 5), 100.5)
    h, l = axs[1].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.legend(h, l, loc="upper center", ncol=len(names), bbox_to_anchor=(0.5, 1.0), fontsize=7.5)
    r.save(fig, "utilisation", "component_utilisation")


def fig_operation_utilisation(r):
    """For the operation day: hourly supply/sink stack and utilisation table, every method."""
    od = operation_day(r)
    if od is None:
        return
    y, day, date = od
    s = slice(24 * day, 24 * day + 24)
    sub = "operation_%s" % date
    rows = []
    h = np.arange(24)
    for name, ck, trs in _method_traces(r):
        tr = trs.get(y)
        if tr is None:
            continue
        u = utilisation(tr, r.cfg, s)
        row = dict([("method", name)] + [(c, u[c]) for c in UTIL_COLS])
        for k in ("planned_el_MWh", "el_MWh", "planned_fc_MWh", "fc_MWh",
                  "planned_bess_dis_MWh", "bess_dis_MWh", "curtailed_MWh", "unserved_MWh"):
            if k in u:
                row[k] = u[k]
        rows.append(row)

        L = tr["P_load"][s] / 1e6
        pv, wd = tr["P_pv"][s] / 1e6, tr["P_wind"][s] / 1e6
        ab, ah = tr["act_b"][s] / 1e6, tr["act_h"][s] / 1e6
        res = tr["P_res"][s] / 1e6 if "P_res" in tr else pv + wd
        avail = pv + wd
        curt = np.minimum(np.clip(res + ab + ah - L, 0.0, None), avail)
        sh = np.divide(pv, avail, out=np.zeros_like(pv), where=avail > 0)
        pv_d, wd_d = pv - curt * sh, wd - curt * (1 - sh)
        uns = tr["P_def"][s] / 1e6 if "P_def" in tr else np.zeros(24)
        sup = [("PV", pv_d, C[3]), ("wind", wd_d, C[2]), ("battery discharge", np.clip(ab, 0, None), C[0]),
               ("fuel cell", np.clip(ah, 0, None), C[6]), ("unserved", uns, C[7])]
        snk = [("battery charge", np.clip(-ab, 0, None), C[0]),
               ("electrolyser", np.clip(-ah, 0, None), C[6]), ("curtailed", curt, MUTED)]
        fig, ax = plt.subplots(figsize=(7.2, 3.6))
        base = np.zeros(24)
        for lab, v, col in sup:
            ax.bar(h, v, 0.85, bottom=base, color=col, label=lab, edgecolor="white", lw=0.6,
                   alpha=0.35 if lab == "unserved" else 1.0)
            base += v
        base = np.zeros(24)
        for lab, v, col in snk:
            ax.bar(h, -v, 0.85, bottom=-base, color=col, label=lab, edgecolor="white", lw=0.6,
                   hatch="///" if lab != "curtailed" else None, alpha=0.6)
            base += v
        ax.step(h, L, where="mid", color=INK, lw=1.4, label="demand")
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_xticks(range(0, 24, 2))
        ax.set_xlabel("hour of %s" % date)
        ax.set_ylabel("MW  (+ supply, - absorbed)")
        ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=7)
        ax.set_title("%s - hourly energy balance of the day-ahead operation" % name, loc="left")
        r.save(fig, sub, "%s_energy_balance" % name)
    d = os.path.join(r.fig, sub)
    os.makedirs(d, exist_ok=True)
    _write_table(os.path.join(d, "utilisation_day.csv"), os.path.join(d, "utilisation_day.md"),
                 rows, "Utilisation on %s (24 h day-ahead operation)" % date)

def fig_daily_paired(r):
    """Day-by-day comparison with the MPC: on how many days does each DRL method do better?"""
    meths = _method_traces(r)
    if len(meths) < 2:
        return
    mpc = meths[0][2]
    rows = []
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    allv = []
    for name, ck, trs in meths[1:]:
        diffs = []
        for y, tr in trs.items():
            if y not in mpc:
                continue
            m = mpc[y]
            D = min(len(m["P_def"]), len(tr["P_def"])) // 24
            dm = m["P_def"][:D * 24].reshape(D, 24).sum(1) / 1e6
            dd = tr["P_def"][:D * 24].reshape(D, 24).sum(1) / 1e6
            diffs.append(dd - dm)
        if not diffs:
            continue
        d = np.concatenate(diffs)
        allv.append((name, ck, d))
        rows.append({"method": name, "days": len(d),
                     "days DRL better (less unserved)": int((d < -1e-3).sum()),
                     "days DRL worse": int((d > 1e-3).sum()),
                     "days equal (within 1 kWh)": int((np.abs(d) <= 1e-3).sum()),
                     "mean daily difference, MWh (DRL - MPC)": float(d.mean()),
                     "largest DRL gain, MWh": float(-d.min()), "largest DRL loss, MWh": float(d.max())})
    if not allv:
        return
    lim = max(np.percentile(np.abs(np.concatenate([v for _, _, v in allv])), 99), 0.5)
    bins = np.linspace(-lim, lim, 41)
    for name, ck, d in allv:
        hist, edges = np.histogram(np.clip(d, -lim, lim), bins=bins)
        ax.step(0.5 * (edges[1:] + edges[:-1]), hist, where="mid", color=ALGO_COL[ck], lw=1.4,
                label="%s (better %d / worse %d days)" % (name, (d < -1e-3).sum(), (d > 1e-3).sum()))
    ax.axvline(0, color=INK, lw=0.9)
    ax.set_xlabel("daily unserved energy, DRL minus MPC (MWh)   <- DRL better | MPC better ->")
    ax.set_ylabel("days")
    ax.legend(loc="upper left", fontsize=7)
    ax.set_title("Day-by-day comparison with the MPC over the held-out years", loc="left")
    r.save(fig, "drl", "daily_paired_vs_mpc")
    d_ = os.path.join(r.fig, "drl")
    _write_table(os.path.join(d_, "daily_paired_vs_mpc.csv"), os.path.join(d_, "daily_paired_vs_mpc.md"),
                 rows, "Day-by-day comparison with the MPC (unserved energy per day)")

# ───────────────────────────────────────────────────────────── summary
def write_summary(r):
    rows = []
    for y in sorted(r.mpc):
        k = r.mpc[y]
        rows.append({"method": "MPC", "year": y, "seed": "",
                     "deficit_pct": k.get("D_L_corrected_pct"), "raw_deficit_pct": k.get("D_L_pct"),
                     "curtail_pct": k.get("C_curt_pct"),
                     "kappa_exact_pct": k["kappa_exact_pct"],
                     "kappa_linear_pct": k.get("kappa_linear_pct"),
                     "over_h": k["over_hours"], "under_h": k["under_hours"],
                     "v_min": k["v_min_exact"], "v_min_bus": k["v_min_bus"],
                     "v_max": k["v_max_exact"], "v_max_bus": k["v_max_bus"],
                     "loss_MWh": k["loss_MWh"]})
    for k in sorted(r.drl, key=lambda z: (z["algo"], z["test_year"], z["seed"])):
        rows.append({"method": ALGO_NAME.get(k["algo"], k["algo"]), "year": k["test_year"],
                     "seed": k["seed"], "deficit_pct": k.get("D_L_corrected_pct"),
                     "raw_deficit_pct": k.get("D_L_pct"), "curtail_pct": k.get("C_curt_pct"),
                     "kappa_exact_pct": k.get("kappa_exact_pct"), "kappa_linear_pct": None,
                     "over_h": None, "under_h": None,
                     "v_min": k.get("v_min_exact"), "v_min_bus": None,
                     "v_max": k.get("v_max_exact"), "v_max_bus": None,
                     "loss_MWh": k.get("loss_MWh")})
    if not rows:
        return
    d = os.path.join(r.fig, "summary")
    os.makedirs(d, exist_ok=True)
    for path in (os.path.join(r.dir, "summary.csv"), os.path.join(d, "summary.csv")):
        with io.open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            for row in rows:
                w.writerow({k: (round(v, 4) if isinstance(v, float) else v)
                            for k, v in row.items()})
    # means per method
    lines = ["# %s" % r.name, ""]
    if r.mpc:
        p = next(iter(r.mpc.values())).get("plant", {})
        lines += ["Plant in force: " + json.dumps(p), ""]
    lines += ["| method | years | deficit % | exact compliance % | worst-year compliance % |",
              "|---|---|---|---|---|"]
    meth = {}
    for row in rows:
        meth.setdefault(row["method"], []).append(row)
    for mname, rs in meth.items():
        dd = [x["deficit_pct"] for x in rs if x["deficit_pct"] is not None]
        kk = [x["kappa_exact_pct"] for x in rs if x["kappa_exact_pct"] is not None]
        lines.append("| %s | %d | %.3f | %.2f | %.2f |"
                     % (mname, len(set(x["year"] for x in rs)), np.mean(dd), np.mean(kk),
                        np.min(kk)))
    with io.open(os.path.join(r.dir, "summary.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def make_all(run_dir):
    r = Run(run_dir)
    print("figures for %s (%d MPC years, %d DRL cells)" % (r.name, len(r.mpc), len(r.drl)))
    for fn in (fig_layout, fig_voltage_envelope, fig_voltage_snapshots, fig_violations_by_bus, fig_branch_power,
               fig_voltage_year, fig_linear_vs_exact, fig_mpc_years,
               fig_drl_vs_mpc, fig_learning, fig_drl_convergence, fig_methods_summary,
               fig_drl_system, fig_operation_day, fig_utilisation, fig_operation_utilisation,
               fig_daily_paired,
               write_summary):
        try:
            fn(r)
        except Exception as exc:                                  # noqa: BLE001
            print("  skipped %s: %s: %s" % (fn.__name__, type(exc).__name__, exc))


if __name__ == "__main__":
    make_all(sys.argv[1])
