"""Every data figure of the paper, from the raw result files (data.py).

Style follows the IEEE TSTE reference (style.py): boxed axes, dashed grid, framed legends,
centred titles, panel letters under the panels, stacked dispatch bars, shaded training curves
and cumulative curves with markers and end-value labels.
"""
import datetime
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.colors import TwoSlopeNorm
from style import (plt, COL1, COL2, NAME, COLOR, SRC, INK, INK2, MUTED, panel, save, legend_below,
                   BLUE, RED, ORANGE, TEAL, PURPLE, MAGENTA, GREEN, GREY)
from data import U, YEARS, SEEDS, kpi, trace, network, history, values, daily

CASE1 = ["fpppo", "pdppo", "focops", "cpo"]
OPDAY = datetime.date(2020, 5, 23)
D0 = (OPDAY - datetime.date(2020, 1, 1)).days * 24
HRS = np.arange(24)
DIV = "RdBu_r"
SEEDCOL = ["#e7298a", "#7570b3", "#1b9e77", "#666666"]


def D(m, y, s=0):
    return kpi(m, y, s)["D_L_corrected_pct"]


def smooth(x, k):
    x = np.asarray(x, float)
    return np.convolve(x, np.ones(k) / k, mode="valid")


# ───────────────────────────────────────────────────── concept: how the agent scales the values
def fig_value_concept():
    E_HD, a_lo, a_hi, s_res = 280.73, 1.5, 0.5, 0.3
    lam_B, E_B = 0.5, 8.571
    S = np.linspace(0, 1, 301)
    VH = E_HD * np.minimum(a_lo * S, a_hi * S + (a_lo - a_hi) * s_res)
    fig, (a, b) = plt.subplots(1, 2, figsize=(COL2, 2.5), gridspec_kw=dict(width_ratios=[1, 1.35]))
    a.fill_between(S * 100, VH / 3, VH * 3, color=MAGENTA, alpha=0.12, lw=0, label=r"range $w_H\in[1/3,\,3]$")
    for w, ls, lab in ((3, "--", r"$w_H=3$"), (1, "-", r"$w_H=1$ (MPC)"), (1 / 3, ":", r"$w_H=1/3$")):
        a.plot(S * 100, w * VH, ls, color=MAGENTA if w != 1 else BLUE, lw=1.4, label=lab)
    a.axvline(s_res * 100, color=INK2, lw=0.8, ls="-.")
    a.text(s_res * 100 + 1.2, 160, "reserve $s_{res}$", fontsize=6.6, color=INK2, rotation=90)
    a.annotate("slope $a_{lo}E_{HD}$", xy=(15, VH[45]), xytext=(1.5, 300), fontsize=6.4,
               arrowprops=dict(arrowstyle="->", lw=0.7, color=INK2))
    a.annotate("slope $a_{hi}E_{HD}$", xy=(70, VH[210]), xytext=(55, 300), fontsize=6.4,
               arrowprops=dict(arrowstyle="->", lw=0.7, color=INK2))
    a.set_xlabel(r"end-of-day tank level $S_{H,K}$ (%)")
    a.set_ylabel(r"terminal value $w_H V_H$ (MWh-equivalent)")
    a.set_xlim(0, 100)
    a.set_ylim(0, 800)
    legend_below(a, ncol=2, fontsize=6.0, columnspacing=0.8, handlelength=1.8)
    panel(a, "a", "Terminal value of stored hydrogen")
    # (b) one CDRL action per day, the planner's 24 h horizon inside each day
    w = values(2020, 0)["w"]
    d0, nd = 141, 7
    days = np.arange(nd)
    for j, col, lab in ((0, MAGENTA, r"$w_H$ (hydrogen)"), (1, BLUE, r"$w_B$ (battery)")):
        y = w[d0:d0 + nd, j]
        b.step(np.r_[days, nd] * 24, np.r_[y, y[-1]], where="post", color=col, lw=1.5, label=lab)
        b.plot(days * 24, y, "o", color=col, ms=3.5)
    b.axhline(1.0, color=INK, lw=0.9, ls="--", label="MPC constants ($w=1$)")
    for k in range(nd + 1):
        b.axvline(k * 24, color=RED, lw=0.6, ls=":")
    b.annotate("", xy=(24, 0.915), xytext=(48, 0.915), arrowprops=dict(arrowstyle="<->", lw=0.8))
    b.text(36, 0.919, r"$t_d$: one action per day (00:00)", ha="center", va="bottom", fontsize=6.4)
    b.annotate("", xy=(96, 0.893), xytext=(120, 0.893), arrowprops=dict(arrowstyle="<->", lw=0.8, color=TEAL))
    b.text(108, 0.897, r"$k=1,\dots,24$: MPC horizon", ha="center", va="bottom", fontsize=6.4, color=TEAL)
    b.set_xticks(np.arange(0, nd * 24 + 1, 24))
    b.set_xticklabels([(datetime.date(2020, 1, 1) + datetime.timedelta(days=int(d0 + k))).strftime("%d %b")
                       for k in range(nd + 1)])
    b.set_xlim(0, nd * 24)
    b.set_ylim(0.88, 1.10)
    b.set_xlabel("time (h), 2020")
    b.set_ylabel("value multiplier")
    legend_below(b, ncol=3, fontsize=6.2, columnspacing=1.0)
    panel(b, "b", "Daily actions of the CDRL agent (seed 0)")
    save(fig, "value_concept")


# ───────────────────────────────────────────────────── Case 1: annual comparison
def fig_case1_annual():
    fig, (a, b) = plt.subplots(1, 2, figsize=(COL2, 2.55), gridspec_kw=dict(width_ratios=[1.25, 1]))
    rows = CASE1[::-1]
    for i, m in enumerate(rows):
        d = np.array([D(m, y) - D("mpc", y) for y in YEARS])
        a.barh(i, d.mean(), color=COLOR[m], alpha=0.75, height=0.6, edgecolor="black", lw=0.5)
        a.scatter(d, np.full(5, i) + np.linspace(-0.18, 0.18, 5), s=10, color="black", zorder=3, lw=0)
        a.text(d.max() + 0.025, i, "%+.3f" % d.mean(), fontsize=6.6, va="center")
    a.axvline(0, color=INK, lw=0.9)
    a.set_yticks(range(len(rows)))
    a.set_yticklabels([NAME[m] for m in rows])
    a.set_xlabel(r"$\Delta D_L^{corr}$ against MPC (pp)")
    a.set_xlim(-0.02, 0.52)
    panel(a, "a", "Difference from MPC (bar: mean, dots: years)")
    for m, mk, ls in (("mpc", "o", "-"), ("fpppo", "s", "-"), ("pdppo", "^", "--"), ("focops", "D", ":"),
                      ("cpo", "v", "-.")):
        b.plot(YEARS, [D(m, y) for y in YEARS], ls, color=COLOR[m], marker=mk, ms=3.2, label=NAME[m])
    b.set_xticks(YEARS)
    b.set_xlabel("test year")
    b.set_ylabel(r"$D_L^{corr}$ (%)")
    legend_below(b, ncol=5, fontsize=6.0, columnspacing=0.8, handlelength=1.6)
    b.set_ylim(22.5, 33.0)
    panel(b, "b", "Corrected deficit per test year")
    save(fig, "case1_annual")


def fig_case1_training():
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(COL2, 2.45))
    k = 12
    for m in ("fpppo", "pdppo", "focops", "cpo"):
        h = history(m, 2020)
        ep = np.array([x["ep"] for x in h if "val_deficit" in x])
        vd = np.array([x["val_deficit"] for x in h if "val_deficit" in x])
        a.plot(ep, vd, color=COLOR[m], label=NAME[m], lw=1.2)
        tr = np.array([x["train_return"] for x in h[1:]]) / 1e3
        b.plot(np.arange(k, len(tr) + 1), smooth(tr, k), color=COLOR[m], lw=1.3, label=NAME[m])
        if m != "fpppo":
            mu = np.array([x["mult"] for x in h[1:]])
            c.plot(np.arange(1, len(mu) + 1), mu, color=COLOR[m], lw=0.5, alpha=0.25)
            c.plot(np.arange(k, len(mu) + 1), smooth(mu, k), color=COLOR[m], lw=1.3, label=NAME[m])
    il0 = history("fpppo", 2020)[0]["val_deficit"]
    a.axhline(il0, color=INK, ls="--", lw=0.9)
    a.set_xlabel("Episodes")
    a.set_ylabel(r"Validation $D_L^{corr}$ (%)")
    panel(a, "a", "Validation deficit (2017-2018)")
    b.set_xlabel("Episodes")
    b.set_ylabel(r"Return, 12-episode mean ($\times10^3$)")
    panel(b, "b", "Training return")
    c.set_xlabel("Episodes")
    c.set_ylabel(r"Lagrange multiplier $\lambda$")
    c.set_yscale("symlog", linthresh=0.1)
    panel(c, "c", "Dual variable")
    hs = [Line2D([], [], color=COLOR[m], lw=1.3, label=NAME[m]) for m in ("fpppo", "pdppo", "focops", "cpo")]
    hs.append(Line2D([], [], color=INK, ls="--", lw=0.9, label="episode 0 = cloned MPC policy (a)"))
    hs.append(Line2D([], [], color=GREY, lw=0.5, alpha=0.5, label="raw multiplier (c)"))
    save(fig, "case1_training", fig_legend=dict(handles=hs, ncol=6, fontsize=6.2, columnspacing=1.0,
                                                 handlelength=1.8))


# ───────────────────────────────────────────────────── day-by-day paired comparison (both cases)
def fig_daily():
    meths = ["fpppo", "pdppo", "focops", "cpo", "mpcdrl"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(COL2, 2.6), gridspec_kw=dict(width_ratios=[1, 1.2]))
    marks = {"fpppo": "s", "pdppo": "^", "focops": "D", "cpo": "v", "mpcdrl": "o"}
    for i, m in enumerate(meths[::-1]):
        dd = np.concatenate([daily(trace(m, y)["P_def"]) / 1e6 - daily(trace("mpc", y)["P_def"]) / 1e6 for y in YEARS])
        better, worse = int((dd < -1e-3).sum()), int((dd > 1e-3).sum())
        equal = len(dd) - better - worse
        n = float(len(dd))
        kw = dict(height=0.62, edgecolor="black", lw=0.4)
        a.barh(i, better / n * 100, color=TEAL, **kw)
        a.barh(i, equal / n * 100, left=better / n * 100, color="#d9d9d9", **kw)
        a.barh(i, worse / n * 100, left=(better + equal) / n * 100, color=ORANGE, **kw)
        a.text(better / n * 50, i, "%d" % better, ha="center", va="center", fontsize=6.3, color="white")
        a.text((better + equal + worse / 2) / n * 100, i, "%d" % worse, ha="center", va="center", fontsize=6.3,
               color="white")
        if m in marks:
            cum = np.concatenate([[0], np.cumsum(dd)])
            x = np.arange(len(cum)) / 365.25 + 2019
            b.plot(x, cum, color=COLOR[m], lw=1.2, marker=marks[m], ms=3, markevery=90, label=NAME[m])
            if m in ("cpo", "mpcdrl"):
                b.text(x[-1] + 0.08, cum[-1], "%+.0f" % cum[-1], color=COLOR[m], fontsize=6.4, va="center")
    a.set_yticks(range(len(meths)))
    a.set_yticklabels([NAME[m] for m in meths[::-1]])
    a.set_xlim(0, 100)
    a.set_xlabel("Share of the 1826 test days (%)")
    legend_below(a, handles=[Patch(facecolor=TEAL, edgecolor="black", lw=0.4, label="less unserved than MPC"),
                      Patch(facecolor="#d9d9d9", edgecolor="black", lw=0.4, label="equal (1 kWh)"),
                      Patch(facecolor=ORANGE, edgecolor="black", lw=0.4, label="more unserved")],
             ncol=3, fontsize=6.2)
    panel(a, "a", "Day-by-day comparison with MPC")
    b.axhline(0, color=INK, lw=0.8)
    b.set_xlabel("Test period")
    b.set_ylabel("Cumulative extra unserved energy (MWh)")
    b.set_xticks(YEARS)
    b.set_xlim(2019, 2024.85)
    legend_below(b, ncol=5, fontsize=6.0, columnspacing=0.8, handlelength=1.6)
    panel(b, "b", "Cumulative difference from MPC")
    save(fig, "daily_paired")


# ───────────────────────────────────────────────────── 24 h operation of one day
def fig_operation():
    F = np.load(U + "/runs/my_new_scenario9_MIQP_DRL/forecasts/forecast_residual_2020.npy")
    f = F[D0, :24] / 1e6
    t = trace("mpc", 2020)
    sl = slice(D0, D0 + 24)
    L, pv, wd = t["P_load"][sl] / 1e6, t["P_pv"][sl] / 1e6, t["P_wind"][sl] / 1e6
    fig = plt.figure(figsize=(COL2, 6.6), layout="constrained")
    gs = fig.add_gridspec(3, 2, height_ratios=[1.3, 1, 1])
    a = fig.add_subplot(gs[0, 0])
    a.plot(HRS, L, color=INK, label="real-time demand")
    a.plot(HRS, f[:, 0], color=INK, ls="--", lw=1.0, label="forecast demand")
    a.plot(HRS, pv, color=SRC["pv"], label="real-time PV")
    a.plot(HRS, f[:, 1], color=SRC["pv"], ls="--", lw=1.0, label="forecast PV")
    a.plot(HRS, wd, color=SRC["wind"], label="real-time wind")
    a.plot(HRS, f[:, 2], color=SRC["wind"], ls="--", lw=1.0, label="forecast wind")
    a.set_ylabel("Power (MW)")
    a.set_xlabel("Time (Hours)")
    a.set_ylim(0, 4.6)
    legend_below(a, ncol=3, fontsize=5.8, columnspacing=0.8)
    panel(a, "a", "Forecast (00:00) and realized values")
    # (b) MPC hourly dispatch, stacked bars as in the reference's day-ahead schedule figure
    b = fig.add_subplot(gs[0, 1])
    ab, ah = t["act_b"][sl] / 1e6, t["act_h"][sl] / 1e6
    bd, fc = np.maximum(ab, 0), np.maximum(ah, 0)
    un = t["P_def"][sl] / 1e6
    bc, el = -np.minimum(ab, 0), -np.minimum(ah, 0)
    cu = t["P_curt"][sl] / 1e6
    bot = np.zeros(24)
    kw = dict(width=0.7, edgecolor="none")
    for v, c, lab in ((pv, SRC["pv"], "P_PV"), (wd, SRC["wind"], "P_W"), (bd, SRC["bess"], "Dis_BS"),
                      (fc, SRC["fc"], "P_FC"), (un, SRC["unserved"], "Unserved")):
        b.bar(HRS, v, bottom=bot, color=c, label=lab, **kw)
        bot += v
    neg = np.zeros(24)
    for v, c, lab in ((bc, SRC["bc"], "Ch_BS"), (el, SRC["el"], "P_EL"), (cu, SRC["curt"], "Spill")):
        b.bar(HRS, -v, bottom=-neg, color=c, label=lab, **kw)
        neg += v
    assert np.abs(bot - neg - L).max() < 0.05
    b.plot(HRS, L, color="#6a51a3", marker="o", ms=2.6, lw=1.1, label="L_E")
    b.axhline(0, color=INK, lw=0.6)
    b.set_ylabel("Electrical Power (MW)")
    b.set_xlabel("Time (Hours)")
    b.set_xticks(HRS)
    b.tick_params(axis="x", labelrotation=90, labelsize=6)
    b.set_ylim(-3.0, 5.4)
    legend_below(b, ncol=5, fontsize=5.6, columnspacing=0.6, handlelength=1.2)
    panel(b, "b", "MPC day-ahead schedule")
    meths = ["mpc", "fpppo", "pdppo", "focops", "cpo", "mpcdrl"]
    tr = {m: trace(m, 2020) for m in meths}
    c = fig.add_subplot(gs[1, 0]); d = fig.add_subplot(gs[1, 1])
    e = fig.add_subplot(gs[2, 0]); g = fig.add_subplot(gs[2, 1])
    for m in meths:
        T = tr[m]
        kw = dict(color=COLOR[m], lw=1.2, label=NAME[m], where="mid")
        c.step(HRS, T["act_b"][sl] / 1e6, **kw)
        d.step(HRS, T["act_h"][sl] / 1e6, **kw)
        e.plot(HRS, T["soc"][sl], color=COLOR[m], lw=1.2)
        g.plot(HRS, T["soh2"][sl] * 100, color=COLOR[m], lw=1.2)
    for ax, lim in ((c, 1.2), (d, 1.0)):
        ax.axhline(lim, color=RED, lw=1.0, ls="--")
        ax.axhline(-lim, color=RED, lw=1.0, ls="--")
        ax.set_ylabel("Power (MW)")
    e.axhline(0.9, color=RED, lw=1.0, ls="--")
    e.axhline(0.2, color=RED, lw=1.0, ls="--")
    g.axhline(30, color=RED, lw=1.0, ls="--")
    panel(c, "c", "Battery power (+ discharge)")
    panel(d, "d", "Hydrogen unit power (+ fuel cell, - electrolyzer)")
    e.set_ylabel("SoC")
    e.set_ylim(0.1, 1.0)
    panel(e, "e", "Battery state of charge")
    g.set_ylabel("Tank level (%)")
    panel(g, "f", "Hydrogen tank level")
    for ax in (c, d, e, g):
        ax.set_xlim(-0.6, 23.6)
        ax.set_xticks(range(0, 24, 3))
        ax.set_xlabel("Time (Hours)")
    a.set_xlim(-0.6, 23.6)
    a.set_xticks(range(0, 24, 3))
    b.set_xlim(-0.6, 23.6)
    hs = [Line2D([], [], color=COLOR[m], lw=1.4, label=NAME[m]) for m in meths]
    hs.append(Line2D([], [], color=RED, lw=1.0, ls="--", label="limits / reserve"))
    save(fig, "operation_day", relayout=False, fig_legend=dict(handles=hs, ncol=7, fontsize=6.2, columnspacing=1.0,
                                                title="Panels (c)-(f)", title_fontsize=6.2))


# ───────────────────────────────────────────────────── voltage
ORDER = list(range(1, 19)) + list(range(19, 23)) + list(range(23, 26)) + list(range(26, 34))


def fig_voltage():
    n = network("mpc")
    V = n["v_exact"]
    x = np.arange(1, 34)
    LIM = dict(color="#c44e52", lw=0.8, ls=(0, (4, 3)), alpha=0.85, zorder=1)
    LEG_Y = -0.36          # legends go below each panel, under its letter
    fig, axs = plt.subplots(1, 3, figsize=(COL2, 2.75), gridspec_kw=dict(width_ratios=[1.12, 1, 1]))
    a, b, c = axs

    # (a) heat map: a softer diverging map, centred on 1 pu
    day = V[D0:D0 + 24][:, [i - 1 for i in ORDER]]
    im = a.imshow(day.T, aspect="auto", origin="lower", cmap="coolwarm",
                  norm=TwoSlopeNorm(1.0, 0.95, 1.05), extent=[-0.5, 23.5, 0.5, 33.5],
                  interpolation="nearest")
    a.set_yticks([1, 18, 22, 25, 33])
    a.set_xticks([0, 6, 12, 18, 23])
    for yb in (18.5, 22.5, 25.5):
        a.axhline(yb, color="white", lw=0.8, alpha=0.9)
    a.set_xlabel("Hour")
    a.set_ylabel("Bus (feeder order)")
    a.grid(False)
    cb = fig.colorbar(im, ax=a, fraction=0.06, pad=0.03, ticks=[0.95, 0.975, 1.0, 1.025, 1.05])
    cb.set_label("Voltage (pu)", fontsize=7)
    cb.ax.tick_params(labelsize=6)
    cb.outline.set_linewidth(0.5)
    panel(a, "a", "MPC, 23 May 2020")

    def feeder_axes(ax):
        for lim in (0.95, 1.05):
            ax.axhline(lim, **LIM)
        for xb in (18.5, 22.5, 25.5):
            ax.axvline(xb, color=MUTED, lw=0.5, ls=":", zorder=1)
        # section names sit above the frame, outside the data area
        for xc, lab in ((9.5, "trunk"), (20.5, "L1"), (24, "L2"), (29.5, "L3")):
            ax.text(xc, 1.01, lab, ha="center", va="bottom", fontsize=6.0, color=INK2,
                    transform=ax.get_xaxis_transform(), clip_on=False)
        ax.set_xticks([1, 6, 12, 18, 25, 33])
        ax.set_xlim(0.5, 33.5)
        ax.set_ylim(0.94, 1.06)
        ax.set_yticks([0.94, 0.96, 0.98, 1.00, 1.02, 1.04, 1.06])
        ax.set_xlabel("Bus")
        ax.set_ylabel("Voltage (pu)")

    # (b) four hours of the same day; the legend sits in the empty upper-left corner
    hours = [3, 9, 13, 20]
    cols = [BLUE, TEAL, ORANGE, MAGENTA]
    mks = ["o", "s", "^", "D"]
    for h, col, mk in zip(hours, cols, mks):
        b.plot(x, V[D0 + h][[i - 1 for i in ORDER]], color=col, lw=1.1, marker=mk, ms=2.6,
               markevery=4, label="%02d:00" % h, zorder=3)
    feeder_axes(b)
    b.legend(loc="upper center", bbox_to_anchor=(0.5, LEG_Y), ncol=4, fontsize=6.0,
             handlelength=1.5, columnspacing=0.8, handletextpad=0.4)
    panel(b, "b", "Profiles along the feeder")
    b.set_title(b.get_title(), pad=11)

    # (c) the six envelopes differ by at most 0.002 pu: shade the MPC range and draw the
    #     MPC, the CDRL family (one grey style) and MPC-CPO on top
    env = {m: network(m)["v_exact"][:, [i - 1 for i in ORDER]]
           for m in ("mpc", "fpppo", "pdppo", "focops", "cpo", "mpcdrl")}
    lo, hi = env["mpc"].min(0), env["mpc"].max(0)
    c.fill_between(x, lo, hi, color=COLOR["mpc"], alpha=0.10, lw=0, zorder=1)
    for m in ("fpppo", "pdppo", "focops", "cpo"):
        c.plot(x, env[m].min(0), color="#9a9a9a", lw=0.8, zorder=2)
        c.plot(x, env[m].max(0), color="#9a9a9a", lw=0.8, zorder=2)
    c.plot(x, lo, color=COLOR["mpc"], lw=1.4, zorder=3)
    c.plot(x, hi, color=COLOR["mpc"], lw=1.4, zorder=3)
    c.plot(x, env["mpcdrl"].min(0), color=COLOR["mpcdrl"], lw=1.1, ls=(0, (3, 2)), zorder=4)
    c.plot(x, env["mpcdrl"].max(0), color=COLOR["mpcdrl"], lw=1.1, ls=(0, (3, 2)), zorder=4)
    feeder_axes(c)
    handles = [Line2D([], [], color=COLOR["mpc"], lw=1.4, label="MPC"),
               Line2D([], [], color="#9a9a9a", lw=0.8, label="CDRL (4 policies)"),
               Line2D([], [], color=COLOR["mpcdrl"], lw=1.1, ls=(0, (3, 2)), label="MPC-CPO")]
    c.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, LEG_Y), ncol=3,
             fontsize=6.0, handlelength=1.8, columnspacing=0.8, handletextpad=0.4)
    panel(c, "c", "2020 min-max envelope (8784 h)")
    c.set_title(c.get_title(), pad=11)
    a.set_title(a.get_title(), pad=11)
    save(fig, "voltage")


# ───────────────────────────────────────────────────── active and reactive power
def fig_power():
    n = network("mpc")
    t = trace("mpc", 2020)
    P, Q = n["P_br"][D0:D0 + 24], n["Q_br"][D0:D0 + 24]
    br_order = [i - 2 for i in ORDER[1:]]
    fig, axs = plt.subplots(1, 3, figsize=(COL2, 2.55), gridspec_kw=dict(width_ratios=[1, 1, 0.95]))
    a, b, c = axs
    for ax, X, lab in ((a, P, "P (MW)"), (b, Q, "Q (Mvar)")):
        lim = np.abs(X).max()
        im = ax.imshow(X[:, br_order].T, aspect="auto", origin="lower", cmap=DIV, norm=TwoSlopeNorm(0, -lim, lim),
                       extent=[-0.5, 23.5, 1.5, 33.5], interpolation="nearest")
        ax.set_yticks([2, 18, 22, 25, 33])
        ax.set_xlabel("Time (Hours)")
        ax.set_ylabel("Receiving bus of branch")
        ax.grid(False)
        cb = fig.colorbar(im, ax=ax, fraction=0.06, pad=0.02)
        cb.set_label(lab + ", + toward load", fontsize=6.5)
        cb.ax.tick_params(labelsize=6)
    panel(a, "a", "Branch active power")
    panel(b, "b", "Branch reactive power")
    ph, qh = t["act_h"] / 1e6, t["qh"]
    th = np.linspace(0, 2 * np.pi, 200)
    c.plot(np.cos(th), np.sin(th), color=RED, lw=1.0, ls="--", label=r"$S_{conv}$ limit")
    c.scatter(ph, qh, s=1.2, color=BLUE, alpha=0.25, lw=0, rasterized=True, label="2020")
    c.scatter(ph[D0:D0 + 24], qh[D0:D0 + 24], s=9, color=MAGENTA, lw=0, zorder=3, label="23 May")
    c.set_aspect("equal")
    c.set_xlim(-1.15, 1.15)
    c.set_ylim(-1.15, 1.15)
    c.set_xlabel(r"$p_H$ (MW, + fuel cell)")
    c.set_ylabel(r"$q_H$ (Mvar)")
    # legend under the panel letter, outside the data area
    handles = [Line2D([], [], color=RED, lw=1.0, ls="--", label=r"$S_{conv}$ limit"),
               Line2D([], [], ls="", marker="o", ms=3.0, color=BLUE, alpha=0.6, label="2020"),
               Line2D([], [], ls="", marker="o", ms=3.6, color=MAGENTA, label="23 May")]
    c.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.36), ncol=3,
             fontsize=6.0, handletextpad=0.3, columnspacing=0.8, handlelength=1.6)
    panel(c, "c", "Converter operating points")
    save(fig, "power_flow")


# ───────────────────────────────────────────────────── Case 2
def fig_case2_training():
    fig, (a, b) = plt.subplots(1, 2, figsize=(COL2, 2.45))
    mks = ["o", "s", "^", "D"]
    for s in SEEDS:
        h = history("mpcdrl", seed=s)
        ep = [x["ep"] for x in h if "val_deficit" in x]
        vd = [x["val_deficit"] for x in h if "val_deficit" in x]
        a.plot(ep, vd, color=SEEDCOL[s], marker=mks[s], ms=3, lw=1.1, label="seed %d" % s)
        best = int(np.argmin(vd))
        a.scatter([ep[best]], [vd[best]], s=60, facecolor="none", edgecolor=SEEDCOL[s], lw=1.2, zorder=4)
        if s == 0:
            wh = np.array([x.get("w_H_mean", 1.0) for x in h[1:]])
            wb = np.array([x.get("w_B_mean", 1.0) for x in h[1:]])
            for v, col, lab in ((wh, MAGENTA, r"mean $w_H$"), (wb, BLUE, r"mean $w_B$")):
                b.plot(range(1, len(v) + 1), v, color=col, lw=0.6, alpha=0.35)
                b.plot(range(5, len(v) + 1), smooth(v, 5), color=col, lw=1.4, label=lab + " (5-episode mean)")
    base = history("mpcdrl", seed=0)[0]["val_deficit"]
    a.axhline(base, color=BLUE, ls="--", lw=1.1, label="MPC (episode 0)")
    a.set_xlabel("Episodes")
    a.set_ylabel(r"Validation $D_L^{corr}$ (%)")
    legend_below(a, ncol=5, fontsize=6.0, columnspacing=0.7, handlelength=1.5)
    a.set_ylim(25.40, 25.65)
    panel(a, "a", "Validation deficit (circles: kept checkpoint)")
    b.axhline(1.0, color=INK, lw=0.9, ls="--")
    b.set_xlabel("Episodes")
    b.set_ylabel("Mean multiplier per training year")
    legend_below(b, ncol=2, fontsize=6.0, columnspacing=0.8)
    panel(b, "b", "Multipliers explored in training (seed 0)")
    save(fig, "case2_training")


def fig_case2_values():
    from scipy.ndimage import uniform_filter1d
    y = 2020
    W = np.stack([values(y, s)["w"] for s in (0, 1, 2)])
    days = np.arange(1, W.shape[1] + 1)
    fig, axs = plt.subplots(3, 1, figsize=(COL2 * 0.78, 4.6), sharex=True)
    for ax, j, sym, col, title in ((axs[0], 0, "w_H", MAGENTA, "Hydrogen value multiplier"),
                                   (axs[1], 1, "w_B", BLUE, "Battery value multiplier")):
        ax.axhline(1.0, color=INK, lw=0.9, ls="--", label="MPC constant ($w=1$)")
        ax.fill_between(days, W[:, :, j].min(0), W[:, :, j].max(0), color=col, alpha=0.18, lw=0,
                        label="range, seeds 0-2")
        med = np.median(W[:, :, j], 0)
        ax.plot(days, med, color=col, lw=0.5, alpha=0.6, label="median")
        ax.plot(days, uniform_filter1d(med, 7, mode="nearest"), color=col, lw=1.5, label="7-day mean")
        ax.set_ylabel("$%s$" % sym)
        ax.set_ylim(0.78, 1.25)
        ax.set_title(title)
    tm = trace("mpc", y)["soh2"]
    td = trace("mpcdrl", y, 0)["soh2"]
    axs[2].plot(days, daily(tm)[:len(days)] / 24 * 100, color=COLOR["mpc"], lw=1.2, label=NAME["mpc"])
    axs[2].plot(days, daily(td)[:len(days)] / 24 * 100, color=COLOR["mpcdrl"], lw=1.2, ls="--",
                label=NAME["mpcdrl"] + " (seed 0)")
    axs[2].axhline(30, color=RED, lw=1.0, ls=":", label="reserve")
    axs[2].set_ylabel("Tank level (%)")
    axs[2].set_title("Daily mean hydrogen tank level")
    mt = [datetime.date(y, mo, 1).timetuple().tm_yday for mo in range(1, 13)]
    axs[2].set_xticks(mt)
    axs[2].set_xticklabels([datetime.date(y, mo, 1).strftime("%b") for mo in range(1, 13)])
    axs[2].set_xlim(1, days[-1])
    axs[2].set_xlabel("Month of 2020")
    for ax, l in zip(axs, "abc"):
        ax.set_title("(%s) %s" % (l, ax.get_title()))
    hs = [Line2D([], [], color=INK, lw=0.9, ls="--", label="MPC constant ($w=1$), (a)-(b)"),
          Patch(facecolor=GREY, alpha=0.25, lw=0, label="range, seeds 0-2, (a)-(b)"),
          Line2D([], [], color=GREY, lw=0.5, label="median, (a)-(b)"),
          Line2D([], [], color=GREY, lw=1.5, label="7-day mean, (a)-(b)"),
          Line2D([], [], color=COLOR["mpc"], lw=1.2, label=NAME["mpc"] + ", (c)"),
          Line2D([], [], color=COLOR["mpcdrl"], lw=1.2, ls="--", label=NAME["mpcdrl"] + " (seed 0), (c)"),
          Line2D([], [], color=RED, lw=1.0, ls=":", label="reserve, (c)")]
    save(fig, "case2_values", letters=False,
         fig_legend=dict(handles=hs, ncol=4, fontsize=6.0, columnspacing=1.0, handlelength=1.8))


def fig_case2_years():
    fig, ax = plt.subplots(figsize=(COL2 * 0.62, 2.4))
    mks = ["o", "s", "^", "D"]
    for i, y in enumerate(YEARS):
        d = [D("mpcdrl", y, s) - D("mpc", y) for s in SEEDS]
        ax.bar(i, np.mean(d), color=MAGENTA, alpha=0.55, width=0.6, edgecolor="black", lw=0.5,
               label="mean over seeds" if i == 0 else None)
        for s in SEEDS:
            ax.scatter(i + (s - 1.5) * 0.1, d[s], s=16, color=SEEDCOL[s], marker=mks[s], zorder=3, lw=0,
                       label=("seed %d" % s) if i == 0 else None)
        ax.text(i, 0.006, "%.0f MWh" % (-np.mean(d) / 100 * kpi("mpc", y)["load_MWh"]),
                ha="center", va="bottom", fontsize=6.4)
    ax.axhline(0, color=INK, lw=0.9)
    ax.set_xticks(range(5))
    ax.set_xticklabels(YEARS)
    ax.set_xlabel("Test year")
    ax.set_ylabel(r"$\Delta D_L^{corr}$ against MPC (pp)")
    ax.set_ylim(-0.2, 0.04)
    legend_below(ax, ncol=5, fontsize=6.0, handletextpad=0.3, columnspacing=0.8)
    save(fig, "case2_years", letters=False)


if __name__ == "__main__":
    import sys
    names = sys.argv[1:] or [k for k in dict(globals()) if k.startswith("fig_")]
    for k in names:
        globals()[k]()
        print("ok", k)
