# -*- coding: utf-8 -*-
"""MPC + DRL, option A: DRL sets the MPC's end-of-day storage values (called by run_targets_drl.py).

    python targets_drl.py train <res_dir> <algo> <seed> [--episodes N] [--hours H] [--check]

**Why this split.** The day-ahead MPC sees 24 hours. Whatever is left in the hydrogen tank and the
battery at the end of that window is worth something to the following days and seasons, and the MPC
can only guess that worth: its MPC2 prices it with fixed constants, the same on 1 January as on
1 July, the same before a calm week as before a windy one,

    hydrogen   z <= a_lo * S_end,  z <= a_hi * S_end + (a_lo - a_hi) * s_res,   cost  -E_HD * z
               (a_lo = 1.5 below the 30 % reserve, a_hi = 0.5 above it, per MWh the tank returns)
    battery    lam_b = 0.5 per MWh left in the battery

Those numbers decide how eagerly the MPC fills the tank from surplus, how readily it burns hydrogen
to serve load, and how much charge it carries into the next day. They are the only part of the
MPC that is a guess rather than a calculation, and a 24 h optimiser cannot learn them. A DRL agent
can: it sees the season, the recent weather regime and the state of both stores, and it is scored
on the whole year.

**The agent's action**, once per day at 00:00, two numbers in [-1, 1]:

    w_H = 3 ** a[0]     multiplies the hydrogen end-of-day value          (1/3 .. 3)
    w_B = 3 ** a[1]     multiplies the battery end-of-day value lam_b     (1/3 .. 3)

The MPC then plans the 24 hours with those values, exactly as before, so every voltage, power-flow
and converter constraint stays with the optimiser. a = (0, 0) gives w = 1, which is its MPC
unchanged; the agent's last layer starts at zero, so the untrained agent IS the MPC, and the best
checkpoint on the validation years is kept (episode 0, the plain MPC, is one of the candidates).

**Reward.** Each day: minus the unserved energy of that day, plus the change in the hydrogen the
tank can return (E_HD * delta S_H2), in MWh. Over a year this adds up exactly to minus the
corrected deficit the paper reports (D_L_corrected = unserved energy + hydrogen drawn from the
tank over the year), so the agent is trained on the paper's own indicator, with a dense signal.
The constrained methods also get a cost: the share of that day's hours with any bus outside
0.95-1.05 pu (the same check the MPC rows are scored with).

**The plant** is the one the main run's MPC rows were scored on (run_mpc_year: its THESSEnv, its
real-time Dispatch with the minimum-load rule, VBand, and the no-shed-charging rule), stepped one
day at a time. So "MPC" and "MPC + DRL" differ only in the two numbers, and each test cell also
re-runs the plain MPC (a = 0) through this code as a check against the main run.
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
V3 = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(V3, "lib"))
import core       # noqa: E402
import dayahead   # noqa: E402

TRAIN_YEARS = list(range(2005, 2017))
VAL_YEARS = [2017, 2018]
H = 24
W_RANGE = 3.0                       # w = W_RANGE ** a, a in [-1, 1]
REWARD_SCALE = 0.1                  # MWh -> reward units (advantages are normalised anyway)
VIOL_LO, VIOL_HI = 0.95, 1.05


def _boot(res_dir):
    cfg = core.load_config(os.path.join(res_dir, "config_used.py"))
    b = core.bootstrap(cfg, res_dir)
    return cfg, b


# ───────────────────────────────────────────────────────────── the day-stepped MPC plant
class TargetsYear(object):
    """dayahead.run_mpc_year, one day per step, with the MPC's storage values set per day."""

    OBS_DIM = 17

    def __init__(self, b, cfg, year, hours=None):
        self.b, self.cfg, self.year, self.hours = b, cfg, year, hours
        M2 = b["M2"]
        self.s_conv, self.s_res, self.soh2_0 = float(cfg.S_CONV_MVA), 0.3, 0.5
        self.fc_min = self.el_min = 0.15
        self.no_shed = bool(getattr(cfg, "RT_NO_SHED_CHARGING", True))
        self.df = b["W"].profile_frame(year)
        if hours:
            self.df = self.df.iloc[:int(hours)].reset_index(drop=True)
        self.ctl = dayahead.new_controller(b, cfg)
        self.c0 = self.ctl.c.copy()
        nv = M2.NV
        k = np.arange(self.ctl.H) * nv
        self.i_bd, self.i_bc = k + M2.BD, k + M2.BC
        self.eps = float(self.ctl.par["eps"]) * M2.DT_H
        self.lam_b = float(self.ctl.par["lam_b"])
        self.b_unit = M2.E_B * M2.A_B                              # battery value per MW-hour
        self.E_HD = M2.E_HD
        self.P_scale = max(float(cfg.PV_MWP) + float(cfg.WIND_MW), 1.0)
        self._F = None

    # ---- parameters
    def set_values(self, a):
        """a in [-1, 1]^2 -> the MPC's cost vector for the coming day."""
        a = np.clip(np.asarray(a, float).ravel(), -1.0, 1.0)
        w_h, w_b = W_RANGE ** a[0], W_RANGE ** a[1]
        c = self.c0.copy()
        c[self.ctl.iz] = self.c0[self.ctl.iz] * w_h
        lb = self.lam_b * w_b
        c[self.i_bd] = lb * self.b_unit + self.eps
        c[self.i_bc] = -lb * self.b_unit + self.eps
        if a[0] == 0.0 and a[1] == 0.0:
            c = self.c0.copy()                                   # bit-for-bit the plain MPC
        self.ctl.c = c
        return w_h, w_b

    # ---- episode
    def reset(self):
        b, cfg = self.b, self.cfg
        m, M2, S3, IND = b["m"], b["M2"], b["S3"], b["IND"]
        from dispatch import Dispatch
        bess = S3.set_bess(b["c_rated"], float(cfg.BESS_P_MW) * 1e6)
        env = m.THESSEnv(self.df, 3600.0, m.Feeder(m.CAPACITORS_MVAR), bess_p=bess,
                         hfc_p=m.HFCParams(), soh2_0=self.soh2_0)
        env.reset()
        env.sc = IND.NoSC()
        self.env = env
        self.N = env.N
        if self._F is None:
            self._F = b["forecasts"](self.year)[:self.N] / 1e6     # (N, 24, 3) MW
        self.F = self._F
        self.L_true, self.R_true = env.P_load / 1e6, env.P_res / 1e6
        self.ctl.c = self.c0.copy()
        self.dsp = Dispatch(s_res=self.s_res, fc_min=self.fc_min, el_min=self.el_min, min_run_h=2)
        self.pb_prev = self.ph_prev = 0.0
        self.q_log = np.zeros(self.N)
        self.sched = np.zeros((self.N, 3))
        self.values = np.ones((int(np.ceil(self.N / float(H))), 2))
        self.t, self.fails, self.reopened, self.solves = 0, 0, 0, 0
        return self.obs()

    def obs(self):
        """What the agent sees at 00:00: the two stores, the season, the day's forecast and the
        weather regime of the past week and month (realised, so no look-ahead beyond the forecast)."""
        t = min(self.t, self.N - 1)
        env = self.env
        doy = t // H + 1                                           # the year starts on 1 January
        ang = 2.0 * np.pi * (float(doy) - 1.0) / 365.25
        Fd = self.F[t, :H]
        L, R = Fd[:, 0], Fd[:, 1] + Fd[:, 2]
        sc = self.P_scale

        def past(n):
            a = max(0, t - n)
            if t <= a:
                return 0.0, 0.0
            net = self.R_true[a:t] - self.L_true[a:t]
            return float(np.clip(net, 0, None).mean()) / sc, float(np.clip(-net, 0, None).mean()) / sc

        s7, d7 = past(24 * 7)
        s30, d30 = past(24 * 30)
        return np.array([
            (env.bess.soc - 0.55) / 0.35,
            (env.hfc.soh2 - 0.5) / 0.25,
            (env.hfc.soh2 - self.s_res) / 0.25,
            np.sin(ang), np.cos(ang),
            L.mean() / sc, Fd[:, 1].mean() / sc, Fd[:, 2].mean() / sc,
            np.clip(L - R, 0, None).mean() / sc, np.clip(R - L, 0, None).mean() / sc,
            s7, d7, s30, d30,
            1.0 - t / float(self.N),
            self.pb_prev / 1.0, self.ph_prev / 1.0,
        ], np.float32)

    def step(self, a):
        """One day: the MPC plans with this day's values, then the real-time layer runs 24 h."""
        b, cfg = self.b, self.cfg
        M2, S3 = b["M2"], b["S3"]
        from run_stage2 import terminal_target
        env, N, t0 = self.env, self.N, self.t
        w_h, w_b = self.set_values(a)
        self.values[t0 // H] = (w_h, w_b)
        s_term = terminal_target(t0, N, H, self.soh2_0, self.s_res)
        plan = dayahead.plan_day(b, self.ctl, env.bess.soc, env.hfc.soh2, self.F[t0, :H],
                                 self.pb_prev, self.ph_prev, s_term)
        self.solves += 1
        if plan is None:
            self.fails += 1
        Fday = self.F[t0, :H]
        s_h0 = env.hfc.soh2
        t_end = min(t0 + H, N)
        for t in range(t0, t_end):
            k = t - t0
            if plan is None:
                pb = ph = qh = 0.0
            else:
                pb, ph = plan[0][k], plan[1][k]
                qh = float(np.clip(plan[2][k], -self.s_conv, self.s_conv))
            self.sched[t] = (pb, ph, qh)
            e = (self.L_true[t] - self.R_true[t]) - (Fday[k, 0] - Fday[k, 1] - Fday[k, 2])
            band = S3.VBand(self.L_true[t], env.P_pv[t] / 1e6, env.P_wind[t] / 1e6, qh,
                            self.s_conv, p_min_fc=self.fc_min * M2.P_FC,
                            p_min_el=self.el_min * M2.P_EL)
            pb, ph = self.dsp(env, pb, ph, e, band)
            if self.no_shed:
                pb, ph = dayahead.no_shed_charging(pb, ph, self.L_true[t], self.R_true[t],
                                                   self.el_min * M2.P_EL)
            self.reopened += band.reopened
            self.pb_prev, self.ph_prev = pb, ph
            room = max(0.0, self.s_conv ** 2 - ph ** 2) ** 0.5 if self.s_conv else 0.0
            self.q_log[t] = float(np.clip(band.q, -room, room)) if self.s_conv else 0.0
            env.step(t, pb * 1e6, 0.0, ph * 1e6)
        self.t = t_end
        sl = slice(t0, t_end)
        def_mwh = float(env.log["P_def"][sl].sum()) / 1e6
        dh2 = (env.hfc.soh2 - s_h0) * self.E_HD
        r = REWARD_SCALE * (-def_mwh + dh2)
        cost = self._day_violation(sl)
        done = self.t >= N
        return self.obs(), r, done, {"cost": cost, "deficit_MWh": def_mwh, "dH2_MWh": dh2}

    def _day_violation(self, sl):
        """Share of the day's hours with any monitored bus outside 0.95-1.05 pu (voltage_study_q)."""
        M2, env = self.b["M2"], self.env
        f = env.feeder
        n = sl.stop - sl.start
        q = np.zeros((n, 33))
        q[:, M2.I_H] = self.q_log[sl] * M2.PU
        v2 = f.v_squared(env.injections(sl), env.load_scale[sl]) + 2.0 * (q @ f.X)
        v = np.sqrt(np.clip(v2, 0, None))[:, 1:]
        ok = np.all((v >= VIOL_LO) & (v <= VIOL_HI), axis=1)
        return float(1.0 - ok.mean())

    # ---- end of year
    def capture(self):
        env, L, n = self.env, self.env.log, self.N
        cap = dict(P_load=env.P_load.copy(), P_res=env.P_res.copy(), P_pv=env.P_pv.copy(),
                   P_wind=env.P_wind.copy(), load_scale=env.load_scale.copy(),
                   cmd_b=L["cmd_b"][:n].copy(), cmd_h=L["cmd_h"][:n].copy(),
                   act_b=L["act_b"][:n].copy(), act_h=L["act_h"][:n].copy(),
                   qh=self.q_log.copy(), schedule=self.sched.copy())
        for key in ("soc", "soh2", "P_def", "P_curt"):
            if key in L:
                cap[key] = np.asarray(L[key][:n], float).copy()
        return cap

    def kpis(self):
        b = self.b
        M2, IND = b["M2"], b["IND"]
        volt = M2.voltage_study_q(self.env, self.q_log if self.s_conv else None)
        k = IND.five(self.env, volt, self.soh2_0)
        k.update(year=self.year, mode="day_ahead", horizon=H, hours=self.N,
                 infeasible=self.fails, day_solves=self.solves, q_reopened=self.reopened)
        return IND.clean(k)


def score(b, cap, k):
    """Exact DistFlow scoring, as stage_mpc does for the MPC rows."""
    NP = b["NP"]
    ex = NP.solve_distflow(cap)
    lin = NP.solve_network(cap)
    v = ex["v"]
    vm = v[:, 1:]
    over = (vm > VIOL_HI).any(1)
    under = (vm < VIOL_LO).any(1)
    bad = over | under
    net = dict(v_exact=v.astype(np.float32), v_linear=lin["v"].astype(np.float32),
               P_br=ex["P_br"].astype(np.float32), Q_br=ex["Q_br"].astype(np.float32),
               loss_p=ex["loss_p"].astype(np.float32), P_slack=ex["P_slack"].astype(np.float32),
               Q_slack=ex["Q_slack"].astype(np.float32))
    k.update(kappa_exact_pct=100.0 * (1.0 - bad.mean()),
             kappa_linear_pct=100.0 * float(np.mean(np.all(
                 (lin["v"][:, 1:] >= VIOL_LO) & (lin["v"][:, 1:] <= VIOL_HI), axis=1))),
             violating_hours=int(bad.sum()), over_hours=int(over.sum()),
             under_hours=int(under.sum()),
             v_min_exact=float(vm.min()), v_max_exact=float(vm.max()),
             loss_MWh=float(ex["loss_p"].sum()))
    return k, net


def run_year(pl, policy):
    """One full year with policy(obs) -> action; returns (sum of rewards, kpis)."""
    o, done, R = pl.reset(), False, 0.0
    while not done:
        o, r, done, _ = pl.step(policy(o))
        R += r
    return R, pl.kpis()


# ───────────────────────────────────────────────────────────── training + test, one seed
def stage_train(res_dir, algo, seed, episodes=None, hours=None, check=False):
    import torch
    cfg, b = _boot(res_dir)
    import agents as AG
    AG.COST_LIMIT = float(getattr(cfg, "DRL_COST_LIMIT", AG.COST_LIMIT))
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    np.random.seed(seed)
    out = os.path.join(res_dir, "drl")
    os.makedirs(os.path.join(out, "traces"), exist_ok=True)
    os.makedirs(os.path.join(res_dir, "values"), exist_ok=True)

    def log(msg):
        print(msg, flush=True)

    plants = {}

    def plant(y):
        if y not in plants:
            plants[y] = TargetsYear(b, cfg, y, hours=hours)
        return plants[y]

    obs_dim, act_dim = TargetsYear.OBS_DIM, 2
    kw = {"init_log_std": float(getattr(cfg, "TARGETS_DRL_LOG_STD", -1.0))}
    if algo == "cpo" and hasattr(cfg, "TARGETS_DRL_DELTA"):
        kw["delta"] = float(cfg.TARGETS_DRL_DELTA)
    if algo != "cpo" and hasattr(cfg, "TARGETS_DRL_LR"):
        kw["lr"] = float(cfg.TARGETS_DRL_LR)
    kw["gamma"] = float(getattr(cfg, "TARGETS_DRL_GAMMA", 0.99))
    agent = AG.make(algo, obs_dim, act_dim, seed=seed, zero_init=True, **kw)

    def greedy(o):
        return agent.act(o, greedy=True)[0]

    def validate():
        ds = []
        for y in VAL_YEARS:
            _, k = run_year(plant(y), greedy)
            ds.append(k["D_L_corrected_pct"])
        return float(np.mean(ds))

    t0 = time.time()
    vd = validate()
    best = {"score": vd, "state": agent.state(), "ep": 0}
    hist = [{"ep": 0, "val_deficit": vd, "phase": "MPC (default values)"}]
    log("  ep   0  (plain MPC, default storage values)  val deficit %.3f %%" % vd)

    n_ep = int(episodes or getattr(cfg, "TARGETS_DRL_EPISODES", 60))
    every = int(getattr(cfg, "DRL_EVAL_EVERY", 5))
    buf = AG.Rollout(400, obs_dim, act_dim)
    order = list(TRAIN_YEARS)
    rng = np.random.default_rng(seed)
    for ep in range(1, n_ep + 1):
        if (ep - 1) % len(order) == 0:
            rng.shuffle(order)
        pl = plant(order[(ep - 1) % len(order)])
        o, done, R, C, W = pl.reset(), False, 0.0, 0.0, []
        while not done:
            a, lp, v, vc = agent.act(o)
            o2, r, done, info = pl.step(a)
            buf.push(o, a, lp, r, info["cost"], v, vc, done)
            R += r
            C += info["cost"]
            W.append(a)
            o = o2
        agent.update(buf)
        W = np.asarray(W)
        h = {"ep": ep, "year": pl.year, "train_return": R, "train_cost": C,
             "mult": float(getattr(agent, "mult", 0.0)),
             "train_deficit": pl.kpis()["D_L_corrected_pct"],
             "w_H_mean": float(np.mean(W_RANGE ** W[:, 0])),
             "w_B_mean": float(np.mean(W_RANGE ** W[:, 1]))}
        if ep % every == 0 or ep == n_ep:
            vd = validate()
            h["val_deficit"] = vd
            if vd < best["score"]:
                best = {"score": vd, "state": agent.state(), "ep": ep}
            log("  ep %3d  train %d  deficit %.3f %%  w_H %.2f  w_B %.2f  |  val deficit %.3f %%"
                "  (best %.3f @ ep %d)  %.0f min"
                % (ep, pl.year, h["train_deficit"], h["w_H_mean"], h["w_B_mean"], vd,
                   best["score"], best["ep"], (time.time() - t0) / 60))
        hist.append(h)
    agent.load(best["state"])
    torch.save(best["state"], os.path.join(out, "agent_mpcdrl_s%d.pt" % seed))
    core.dump_json(hist, os.path.join(out, "history_mpcdrl_train_s%d.json" % seed))
    log("training done: best validation %.3f %% at episode %d (episode 0 = plain MPC %.3f %%)"
        % (best["score"], best["ep"], hist[0]["val_deficit"]))

    # held-out years: the trained agent, and the plain MPC through the same code as a check
    main_mpc = os.path.join(res_dir, "mpc")
    for y in cfg.DRL_TEST_YEARS:
        pl = TargetsYear(b, cfg, y, hours=hours)
        res = {}
        for tag, pol in (("mpccheck", lambda o: np.zeros(2)), ("mpcdrl", greedy)):
            if tag == "mpccheck" and not check:
                continue
            _, k = run_year(pl, pol)
            cap = pl.capture()
            k, net = score(b, cap, k)
            k.update(algo=tag, test_year=y, seed=seed, best_episode=best["ep"],
                     val_deficit_pct=best["score"], episodes=n_ep, base_algo=algo,
                     minutes=(time.time() - t0) / 60, plant=core.plant_summary(b))
            if tag == "mpcdrl":
                np.savez_compressed(os.path.join(out, "traces", "mpcdrl_%d_s%d.npz" % (y, seed)), **cap)
                np.savez_compressed(os.path.join(out, "network_mpcdrl_%d_s%d.npz" % (y, seed)), **net)
                core.dump_json(k, os.path.join(out, "result_mpcdrl_%d_s%d.json" % (y, seed)))
                np.savez_compressed(os.path.join(res_dir, "values", "values_%d_s%d.npz" % (y, seed)),
                                    w=pl.values, soh2=cap["soh2"], soc=cap["soc"])
            else:
                core.dump_json(k, os.path.join(res_dir, "values", "mpccheck_%d.json" % y))
            res[tag] = k
        ref = os.path.join(main_mpc, "kpi_%d.json" % y)
        ref_d = core.read_json(ref)["D_L_corrected_pct"] if os.path.exists(ref) else float("nan")
        chk = res.get("mpccheck", {}).get("D_L_corrected_pct", float("nan"))
        log("TEST %d s%d: MPC %.3f %% (main run %.3f %%) -> MPC+DRL values %.3f %%, "
            "compliance %.2f %%" % (y, seed, chk, ref_d, res["mpcdrl"]["D_L_corrected_pct"],
                                    res["mpcdrl"]["kappa_exact_pct"]))
    log("done in %.0f min" % ((time.time() - t0) / 60))


if __name__ == "__main__":
    a = sys.argv
    ep = int(a[a.index("--episodes") + 1]) if "--episodes" in a else None
    hrs = int(a[a.index("--hours") + 1]) if "--hours" in a else None
    if a[1] == "train":
        stage_train(a[2], a[3], int(a[4]), episodes=ep, hours=hrs, check="--check" in a)
    else:
        raise SystemExit(__doc__)
