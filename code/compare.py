# -*- coding: utf-8 -*-
"""Side-by-side figures for two or more finished runs.

    python compare.py base H2bus6_cap33-0.3
    python compare.py base H2bus6_cap33-0.3 another_run --name my_comparison

Output: v3/comparisons/<name>/  (PNG + PDF, comparison.csv, comparison.md)
The first run named is treated as the reference.
"""
import argparse
import csv
import io
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "lib"))
import plots as PL                                                # noqa: E402
import matplotlib.pyplot as plt                                   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--name", default=None)
    a = ap.parse_args()
    runs = [PL.Run(os.path.join(HERE, "runs", n)) for n in a.runs]
    for r in runs:
        if not r.mpc:
            raise SystemExit("run %s has no MPC results yet" % r.name)
    name = a.name or "_vs_".join(r.name for r in runs)
    out = os.path.join(HERE, "comparisons", name)
    os.makedirs(out, exist_ok=True)
    cols = [PL.C[i % len(PL.C)] for i in range(len(runs))]
    fmts = getattr(runs[0].cfg, "FIG_FORMATS", ["png", "pdf"])

    def save(fig, fname):
        for ext in fmts:
            fig.savefig(os.path.join(out, "%s.%s" % (fname, ext)))
        plt.close(fig)
        print("  comparisons/%s/%s" % (name, fname))

    years = sorted(set.intersection(*[set(r.mpc) for r in runs]))

    # 1. voltage envelopes
    order, xs = PL.section_axis()
    idx = np.array(order) - 1
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    top = 1.062
    for r, c in zip(runs, cols):
        v = np.vstack([r.net(y)["v_exact"] for y in years])
        ax.fill_between(xs, v.min(0)[idx], v.max(0)[idx], color=c, alpha=0.14, lw=0)
        ax.plot(xs, v.max(0)[idx], color=c, lw=1.0, ls="--")
        ax.plot(xs, v.min(0)[idx], color=c, lw=1.0, ls="--")
        ax.plot(xs, v.mean(0)[idx], color=c, lw=1.8, marker="o", ms=2.5, label=r.name)
        top = max(top, v.max() + 0.004)
    PL.mark_sections(ax, xs, top)
    ax.set_ylim(None, top + 0.012)
    for lim in (0.95, 1.05):
        ax.axhline(lim, color=PL.LIMIT, lw=1.0, ls="--")
    ax.set_xticks(xs)
    ax.set_xticklabels(order, fontsize=6.5)
    ax.set_xlabel("bus")
    ax.set_ylabel("voltage, pu")
    ax.legend(loc="lower left")
    ax.set_title("Exact bus voltages, %s: mean (solid) and min/max (dashed)"
                 % ", ".join(map(str, years)), loc="left")
    save(fig, "voltage_envelope")

    # 2. violations by bus
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    w = 0.8 / len(runs)
    for j, (r, c) in enumerate(zip(runs, cols)):
        tot = np.zeros(33)
        for y in years:
            for key in ("over_by_bus", "under_by_bus"):
                for b, n in r.mpc[y].get(key, {}).items():
                    tot[int(b) - 1] += n
        ax.bar(xs + (j - (len(runs) - 1) / 2) * w, tot[idx], w * 0.92, color=c,
               label="%s (%d h)" % (r.name, tot.sum()))
    ax.set_xticks(xs)
    ax.set_xticklabels(order, fontsize=6.5)
    ax.set_xlabel("bus where the hour's worst violation occurs")
    ax.set_ylabel("hours")
    ax.legend(loc="upper left")
    ax.set_title("Exact-network violating hours by bus (over + under)", loc="left")
    save(fig, "violations_by_bus")

    # 3. KPIs by year
    keys = [("D_L_corrected_pct", "demand deficit, %"),
            ("kappa_exact_pct", "exact compliance, %"),
            ("loss_MWh", "feeder losses, MWh")]
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.6))
    x = np.arange(len(years))
    for ax, (k, lab) in zip(axs, keys):
        for j, (r, c) in enumerate(zip(runs, cols)):
            ax.bar(x + (j - (len(runs) - 1) / 2) * w, [r.mpc[y][k] for y in years], w * 0.92,
                   color=c, label=r.name)
        ax.set_xticks(x)
        ax.set_xticklabels([str(y)[2:] for y in years])
        ax.set_title(lab, loc="left")
        if k == "kappa_exact_pct":
            lo = min(r.mpc[y][k] for r in runs for y in years)
            ax.set_ylim(min(95, lo - 1), 100.3)
    h, l = axs[0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.legend(h, l, loc="upper center", ncol=len(runs), bbox_to_anchor=(0.5, 1.0))
    save(fig, "kpis_by_year")

    # 4. layouts side by side
    for r in runs:
        PL.fig_layout(r)

    # table
    rows = []
    for r in runs:
        for y in years:
            k = r.mpc[y]
            rows.append({"run": r.name, "year": y, "deficit_pct": k["D_L_corrected_pct"],
                         "kappa_exact_pct": k["kappa_exact_pct"], "over_h": k["over_hours"],
                         "under_h": k["under_hours"], "v_min": k["v_min_exact"],
                         "v_max": k["v_max_exact"], "loss_MWh": k["loss_MWh"]})
    with io.open(os.path.join(out, "comparison.csv"), "w", newline="", encoding="utf-8") as fh:
        wtr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wtr.writeheader()
        for row in rows:
            wtr.writerow({k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()})
    lines = ["| run | layout | capacitors | mean deficit % | mean exact compliance % | "
             "worst year % | violating h | losses MWh/yr |", "|---|---|---|---|---|---|---|---|"]
    for r in runs:
        ks = [r.mpc[y] for y in years]
        p = ks[0].get("plant", {})
        lines.append("| %s | %s | %s | %.3f | %.2f | %.2f | %d | %.0f |" % (
            r.name, p.get("layout"), p.get("capacitors_Mvar"),
            np.mean([k["D_L_corrected_pct"] for k in ks]),
            np.mean([k["kappa_exact_pct"] for k in ks]),
            np.min([k["kappa_exact_pct"] for k in ks]),
            sum(k["violating_hours"] for k in ks), np.mean([k["loss_MWh"] for k in ks])))
    with io.open(os.path.join(out, "comparison.md"), "w", encoding="utf-8") as fh:
        fh.write("Years: %s\n\n" % years + "\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
