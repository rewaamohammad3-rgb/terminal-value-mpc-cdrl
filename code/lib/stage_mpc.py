# -*- coding: utf-8 -*-
"""One closed-loop MPC year, then exact DistFlow scoring of it.

    python stage_mpc.py <run_dir> <year> [--hours N]

Uses the core pipeline's controller unchanged: `run_stage3.run` (MPC2 + dispatch layer + voltage
clip) with all four of its corrections applied through `settings.apply`, and its exact scorer
`network_profiles.solve_distflow` (served-load and curtailment corrections included).

Writes into <run_dir>/mpc/:
    trace_<year>.npz     hourly plant record (also the residual DRL arm's reference)
    network_<year>.npz   exact voltages per bus, branch flows, losses, linearised voltages
    kpi_<year>.json      the study's indicators plus exact compliance and violation counts
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core  # noqa: E402

VIOL_LO, VIOL_HI = 0.95, 1.05


def main():
    run_dir, year = sys.argv[1], int(sys.argv[2])
    hours = int(sys.argv[sys.argv.index("--hours") + 1]) if "--hours" in sys.argv else None
    cfg = core.load_config(os.path.join(run_dir, "config_used.py"))
    b = core.bootstrap(cfg, run_dir)
    M2, S3, IND, NP = b["M2"], b["S3"], b["IND"], b["NP"]
    out = os.path.join(run_dir, "mpc")
    os.makedirs(out, exist_ok=True)

    cap = {}
    orig_vs = M2.voltage_study_q

    def capture(env, qh_mvar=None, chunk=50000):
        L, n = env.log, env.N
        cap.update(P_load=env.P_load.copy(), P_res=env.P_res.copy(), P_pv=env.P_pv.copy(),
                   P_wind=env.P_wind.copy(), load_scale=env.load_scale.copy(),
                   cmd_b=L["cmd_b"][:n].copy(), cmd_h=L["cmd_h"][:n].copy(),
                   act_b=L["act_b"][:n].copy(), act_h=L["act_h"][:n].copy(),
                   qh=(np.zeros(n) if qh_mvar is None else np.asarray(qh_mvar, float)))
        for key in ("soc", "soh2", "P_def", "P_curt"):
            if key in L:
                cap[key] = np.asarray(L[key][:n], float).copy()
        return orig_vs(env, qh_mvar, chunk)

    M2.voltage_study_q = capture
    S3.M2.voltage_study_q = capture

    t0 = time.time()
    mode = getattr(cfg, "MPC_MODE", "day_ahead")
    if mode == "day_ahead":
        import dayahead
        extra = {}
        k = IND.clean(dayahead.run_mpc_year(b, cfg, year, hours=hours, capture=extra))
        cap["schedule"] = extra["schedule"]             # the committed 24 h plans, (N, 3)
    else:                                               # its original: re-plan every hour
        k = IND.clean(S3.run(year=year, forecast="v3", horizon=int(cfg.HORIZON), rho=0.01,
                             lam_v=10.0, lam_term=800.0, s_conv=float(cfg.S_CONV_MVA),
                             c_rated=b["c_rated"], p_max=float(cfg.BESS_P_MW) * 1e6,
                             hours=hours))
    k["mpc_mode"] = mode
    np.savez_compressed(os.path.join(out, "trace_%d.npz" % year), **cap)

    # exact network (served load, delivered renewables), and the linearised one beside it
    ex = NP.solve_distflow(cap)
    lin = NP.solve_network(cap)
    v = ex["v"]
    vm = v[:, 1:]
    over = (vm > VIOL_HI).any(1)
    under = (vm < VIOL_LO).any(1)
    bad = over | under
    bus_over = np.bincount(vm.argmax(1)[over] + 2, minlength=34)[2:]
    bus_under = np.bincount(vm.argmin(1)[under] + 2, minlength=34)[2:]
    np.savez_compressed(os.path.join(out, "network_%d.npz" % year),
                        v_exact=v.astype(np.float32), v_linear=lin["v"].astype(np.float32),
                        P_br=ex["P_br"].astype(np.float32), Q_br=ex["Q_br"].astype(np.float32),
                        loss_p=ex["loss_p"].astype(np.float32),
                        P_slack=ex["P_slack"].astype(np.float32),
                        Q_slack=ex["Q_slack"].astype(np.float32))
    k.update(year=year, hours=int(len(v)),
             kappa_exact_pct=100.0 * (1.0 - bad.mean()),
             kappa_linear_pct=100.0 * float(np.mean(np.all(
                 (lin["v"][:, 1:] >= VIOL_LO) & (lin["v"][:, 1:] <= VIOL_HI), axis=1))),
             violating_hours=int(bad.sum()), over_hours=int(over.sum()),
             under_hours=int(under.sum()),
             v_min_exact=float(vm.min()), v_max_exact=float(vm.max()),
             v_min_bus=int(vm.min(0).argmin() + 2), v_max_bus=int(vm.max(0).argmax() + 2),
             over_by_bus={int(i + 2): int(c) for i, c in enumerate(bus_over) if c},
             under_by_bus={int(i + 2): int(c) for i, c in enumerate(bus_under) if c},
             loss_MWh=float(ex["loss_p"].sum()),
             slack_import_MWh=float(np.clip(ex["P_slack"], 0, None).sum()),
             slack_export_MWh=float(np.clip(-ex["P_slack"], 0, None).sum()),
             reverse_branches=int(np.sum((ex["P_br"] < -1e-6).any(0))),
             wall_minutes=(time.time() - t0) / 60.0,
             plant=core.plant_summary(b))
    core.dump_json(k, os.path.join(out, "kpi_%d.json" % year))
    print("MPC %d: deficit %.3f %%, exact compliance %.2f %% (%d over, %d under), "
          "V %.4f..%.4f pu, %.1f min"
          % (year, k.get("D_L_corrected_pct", float("nan")), k["kappa_exact_pct"],
             k["over_hours"], k["under_hours"], k["v_min_exact"], k["v_max_exact"],
             k["wall_minutes"]), flush=True)


if __name__ == "__main__":
    main()
