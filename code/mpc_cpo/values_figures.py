# -*- coding: utf-8 -*-
"""Figures and tables specific to MPC + DRL values.   python values_figures.py <res_dir>

    figures/mpc_drl_values/learned_values_<year>   what the agent chose each day (hydrogen and
                                                   battery value multipliers), and the tank level
                                                   it led to, against the plain MPC
    figures/mpc_drl_values/mpc_vs_mpcdrl.csv/.md   per test year: MPC (main run), MPC re-run
                                                   through this code (check), MPC + DRL values
                                                   (mean, std, best over seeds), difference
"""
import glob
import io
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "lib"))
import core                      # noqa: E402
import plots as P                # noqa: E402  (style, palette and save helper)

plt = P.plt


def table(r, out):
    rows = []
    res = {}
    for f in glob.glob(os.path.join(r.dir, "drl", "result_mpcdrl_*.json")):
        k = core.read_json(f)
        res.setdefault(int(k["test_year"]), []).append(k)
    if not res:
        return
    for y in sorted(res):
        ks = res[y]
        d = np.array([k["D_L_corrected_pct"] for k in ks])
        kap = np.array([k["kappa_exact_pct"] for k in ks])
        mpc = r.mpc.get(y, {}).get("D_L_corrected_pct", np.nan)
        cp = os.path.join(r.dir, "values", "mpccheck_%d.json" % y)
        chk = core.read_json(cp)["D_L_corrected_pct"] if os.path.exists(cp) else np.nan
        rows.append(dict(year=y, mpc=mpc, mpc_check=chk, mpcdrl_mean=d.mean(),
                         mpcdrl_std=d.std() if len(d) > 1 else 0.0, mpcdrl_best=d.min(),
                         seeds=len(d), diff_mean=d.mean() - mpc,
                         seeds_better=int(np.sum(d < mpc - 1e-9)),
                         compliance_min=kap.min()))
    keys = list(rows[0])
    with io.open(os.path.join(out, "mpc_vs_mpcdrl.csv"), "w", encoding="utf-8") as fh:
        fh.write(",".join(keys) + "\n")
        for row in rows:
            fh.write(",".join(("%.4f" % v) if isinstance(v, float) else str(v)
                              for v in row.values()) + "\n")
    m = lambda key: float(np.nanmean([x[key] for x in rows]))          # noqa: E731
    lines = ["# MPC vs MPC + DRL storage values (%s)" % r.name, "",
             "Deficit is the corrected deficit D_L_corrected, %. Lower is better. "
             "MPC check = the plain MPC re-run through this code (must equal the main run).", "",
             "| year | MPC | MPC check | MPC+DRL mean | std | best seed | diff (mean - MPC) "
             "| seeds better | min compliance % |",
             "|---|---|---|---|---|---|---|---|---|"]
    for x in rows:
        lines.append("| %d | %.3f | %.3f | %.3f | %.3f | %.3f | %+.3f | %d/%d | %.2f |"
                     % (x["year"], x["mpc"], x["mpc_check"], x["mpcdrl_mean"], x["mpcdrl_std"],
                        x["mpcdrl_best"], x["diff_mean"], x["seeds_better"], x["seeds"],
                        x["compliance_min"]))
    lines.append("| **mean** | **%.3f** | %.3f | **%.3f** | | %.3f | **%+.3f** | | |"
                 % (m("mpc"), m("mpc_check"), m("mpcdrl_mean"), m("mpcdrl_best"), m("diff_mean")))
    with io.open(os.path.join(out, "mpc_vs_mpcdrl.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def daily(x, n_days):
    x = np.asarray(x, float)
    n = n_days * 24
    x = np.concatenate([x, np.full(max(0, n - len(x)), x[-1])])[:n]
    return x.reshape(n_days, 24).mean(1)


def fig_values(r, out):
    files = sorted(glob.glob(os.path.join(r.dir, "values", "values_*_s*.npz")))
    if not files:
        return
    years = sorted({int(os.path.basename(f).split("_")[1]) for f in files})
    y = int(getattr(r.cfg, "PLOT_YEAR", years[0]))
    y = y if y in years else years[0]
    vs = [dict(np.load(f)) for f in files if int(os.path.basename(f).split("_")[1]) == y]
    nd = min(len(v["w"]) for v in vs)
    W = np.stack([v["w"][:nd] for v in vs])                       # (seeds, days, 2)
    days = np.arange(1, nd + 1)
    col = P.ALGO_COL["mpcdrl"]
    fig, axs = plt.subplots(3, 1, figsize=(7.2, 6.0), sharex=True)
    for ax, j, name in ((axs[0], 0, "hydrogen value multiplier  w_H"),
                        (axs[1], 1, "battery value multiplier  w_B")):
        ax.axhline(1.0, color=P.C[0], lw=1.2, ls="--", label="MPC (fixed, w = 1)")
        if len(W) > 1:
            ax.fill_between(days, W[:, :, j].min(0), W[:, :, j].max(0), color=col, alpha=0.15,
                            lw=0, label="MPC + DRL, range over %d seeds" % len(W))
        ax.plot(days, np.median(W[:, :, j], 0), color=col, lw=1.4,
                label="MPC + DRL" + (" (median over seeds)" if len(W) > 1 else ""))
        ax.set_yscale("log", base=3)
        ax.set_ylim(1 / 3.2, 3.2)
        ax.set_yticks([1 / 3, 1 / 2, 1, 2, 3])
        ax.set_yticklabels(["1/3", "1/2", "1", "2", "3"])
        ax.set_ylabel(name.split("  ")[1])
        ax.set_title(name.split("  ")[0].capitalize() + ": what the agent asked the MPC to "
                     "assume for each stored MWh at day's end", loc="left", fontsize=8.5)
    axs[0].legend(loc="upper right", ncol=3, fontsize=7)
    ax = axs[2]
    tr = r.trace(y)
    if tr is not None and "soh2" in tr:
        ax.plot(days, daily(tr["soh2"], nd), color=P.C[0], lw=1.4, label="MPC")
    ax.plot(days, daily(vs[0]["soh2"], nd), color=col, lw=1.4, label="MPC + DRL (seed 0)")
    ax.axhline(0.3, color=P.MUTED, lw=0.9, ls=":", label="reserve 0.3")
    ax.set_ylabel("tank level (daily mean)")
    ax.set_xlabel("day of %d" % y)
    ax.set_title("Hydrogen tank level that resulted", loc="left", fontsize=8.5)
    ax.legend(loc="upper right", ncol=3, fontsize=7)
    ax.set_xlim(1, nd)
    fig.tight_layout()
    r.save(fig, "mpc_drl_values", "learned_values_%d" % y)


def main(res_dir):
    r = P.Run(res_dir)
    out = os.path.join(r.fig, "mpc_drl_values")
    os.makedirs(out, exist_ok=True)
    for fn in (table, fig_values):
        try:
            fn(r, out) if fn is table else fn(r, out)
        except Exception as exc:                                  # noqa: BLE001
            print("  skipped %s: %s: %s" % (fn.__name__, type(exc).__name__, exc))


if __name__ == "__main__":
    main(sys.argv[1])
