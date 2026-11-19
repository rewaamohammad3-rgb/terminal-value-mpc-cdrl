# -*- coding: utf-8 -*-
"""MPC day-ahead demonstrations for one historical year, on the DRL plant.

    python stage_demos.py <run_dir> <year>

The MPC plans each day at 00:00 and its 24 h schedule is executed by the same day-ahead DRL
plant the agent will use (same real-time layer, same voltage shield). At every 00:00 the
agent's observation and the MPC's 24 h schedule are stored, so the agent can be taught to
reproduce the MPC from the same information before it is fine-tuned with RL.

Writes <run_dir>/demos/demo_<year>.npz with obs (days, obs_dim), act (days, 72) in [-1, 1],
reward (days,), cost (days,), plus the year's deficit as a check.
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core       # noqa: E402
import dayahead   # noqa: E402


def main():
    run_dir, year = sys.argv[1], int(sys.argv[2])
    cfg = core.load_config(os.path.join(run_dir, "config_used.py"))
    b = core.bootstrap(cfg, run_dir)
    import drl_env as E
    E.PROD_DATA = b["forecast_dir"]
    core.set_defaults(E.Plant.__init__, s_conv=float(cfg.S_CONV_MVA), horizon=24)
    DA = dayahead.make_plant_class(E, b)

    t0 = time.time()
    pl = DA(year=year)
    obs, act, rew, cost = [], [], [], []
    o = pl.reset_day()
    pol = dayahead.mpc_policy(b, cfg, pl)
    done = False
    while not done:
        a = pol(o)
        obs.append(o)
        act.append(a)
        o, r, done, info = pl.step_day(a)
        rew.append(r)
        cost.append(info["cost"])
    k = pl.metrics(exact=False)
    out = os.path.join(run_dir, "demos")
    os.makedirs(out, exist_ok=True)
    np.savez_compressed(os.path.join(out, "demo_%d.npz" % year),
                        obs=np.asarray(obs, np.float32), act=np.asarray(act, np.float32),
                        reward=np.asarray(rew, np.float64), cost=np.asarray(cost, np.float64),
                        deficit=k["D_L_corrected_pct"])
    print("demos %d: %d days, MPC deficit on the DRL plant %.3f %%, %.1f min"
          % (year, len(obs), k["D_L_corrected_pct"], (time.time() - t0) / 60), flush=True)


if __name__ == "__main__":
    main()
