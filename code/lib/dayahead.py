# -*- coding: utf-8 -*-
"""Day-ahead operation: one 24-hour schedule per day, committed at 00:00, plus a real-time layer.

    MPC      At hour 0 of every day the core pipeline's MPC2 (same SOCP, same corrections) is solved
             once on the 24 h forecast issued at that hour. Its whole 24-hour plan (battery,
             hydrogen, converter reactive power) is committed. During the day its real-time
             dispatch layer only absorbs the forecast error of each hour (battery first, then
             hydrogen) and its voltage clip (VBand) keeps the realised band, exactly as in
             run_stage3.run. Nothing is re-optimised intraday.

    DRL      The same day-ahead structure: at 00:00 the agent observes the plant state, the
             calendar and the 24 h forecast, and outputs a 24 x 3 schedule (72 actions in
             [-1, 1]). The plant executes it hour by hour through the same real-time layer.
             The agent is first trained to imitate the MPC's day-ahead schedules on the
             historical years (behaviour cloning), then fine-tuned with constrained RL.
"""
import numpy as np

import core

H_DAY = 24


# ───────────────────────────────────────────────────────────── the full 24 h MPC plan
class _Capture(object):
    """Wraps clarabel.DefaultSolver inside mpc2 so the whole solution vector is kept.

    MPC2.solve returns only the first hour; the day-ahead schedule needs all 24.
    """
    last = None

    def __init__(self, real):
        self.real = real

    def __call__(self, *a, **kw):
        solver = self.real(*a, **kw)
        cap = self

        class _S(object):
            def solve(self_inner):
                sol = solver.solve()
                try:
                    cap.last = np.asarray(sol.x, float)
                except Exception:                                     # noqa: BLE001
                    cap.last = None
                return sol
        return _S()


def install_capture(M2):
    if not isinstance(M2.clarabel.DefaultSolver, _Capture):
        import types
        shim = types.ModuleType("clarabel_shim")
        for k in dir(M2.clarabel):
            setattr(shim, k, getattr(M2.clarabel, k))
        shim.DefaultSolver = _Capture(M2.clarabel.DefaultSolver)
        M2.clarabel = shim
    return M2.clarabel.DefaultSolver


def new_controller(b, cfg, horizon=H_DAY):
    M2 = b["M2"]
    install_capture(M2)
    return M2.MPC2(horizon=horizon, rho=0.01, lam_v=10.0, lam_term=800.0, s_res=0.3,
                   s_conv=float(cfg.S_CONV_MVA))


def plan_day(b, ctl, soc, soh2, Fday, pb_prev, ph_prev, s_term):
    """Solve once; return (pb, ph, q) arrays of length 24 in MW / Mvar, or None if infeasible.

    Fday: (24, 3) forecast in MW (load, PV, wind) issued at 00:00 for the coming day.
    """
    M2 = b["M2"]
    cap = M2.clarabel.DefaultSolver
    L, pv, wd = Fday[:, 0], Fday[:, 1], Fday[:, 2]
    x0 = ctl.solve(soc, soh2, L, pv + wd, pv, wd, pb_prev, ph_prev, s_term)
    x = getattr(ctl, "last_x", None)             # the mixed-integer planner keeps its own
    if x is None:
        x = cap.last
    if x0 is None or x is None:
        return None
    H, NV = ctl.H, M2.NV
    k = np.arange(H) * NV
    pb = np.maximum(x[k + M2.BD], 0) - np.maximum(x[k + M2.BC], 0)
    ph = np.maximum(x[k + M2.HD], 0) - np.maximum(x[k + M2.HC], 0)
    q = x[ctl.iq:ctl.iq + H].copy() if ctl.s_conv else np.zeros(H)
    # the first hour must be the one MPC2.solve itself returned
    assert abs(pb[0] - (x0[M2.BD] - x0[M2.BC])) < 1e-6, "solution layout mismatch"
    return pb, ph, q


def no_shed_charging(pb, ph, L, R, el_min):
    """Never charge storage while demand is being shed.

    The real-time layer pushes each hour's forecast error into the battery and hydrogen unit.
    When the realised renewable output is below what the 00:00 forecast assumed, that can leave
    the battery or electrolyser charging in an hour where load is unserved: the plant then sheds
    demand to fill storage, which is never worth it (the energy comes back at 96 % or 38 %).
    Charging is therefore limited to the realised renewable surplus; the electrolyser is cut
    first (lower round trip), then the battery. Discharging is untouched.
    """
    if L - R - pb - ph <= 1e-9:                       # demand is fully served: nothing to do
        return pb, ph
    excess = (max(-pb, 0.0) + max(-ph, 0.0)) - max(R - L, 0.0)
    if excess <= 1e-9:
        return pb, ph
    if ph < 0:
        cut = min(excess, -ph)
        ph += cut
        excess -= cut
        if -el_min < ph < 0:                          # below the electrolyser's minimum load
            excess += -ph
            ph = 0.0
    if excess > 1e-9 and pb < 0:
        pb += min(excess, -pb)
    return pb, ph


# ───────────────────────────────────────────────────────────── MPC closed loop, day-ahead
def run_mpc_year(b, cfg, year, hours=None, capture=None):
    """run_stage3.run, with the plan made once per day instead of once per hour."""
    import time
    m, W, M2, S3, IND = b["m"], b["W"], b["M2"], b["S3"], b["IND"]
    from dispatch import Dispatch
    from run_stage2 import terminal_target
    s_conv, s_res, soh2_0 = float(cfg.S_CONV_MVA), 0.3, 0.5
    fc_min = el_min = 0.15
    bess = S3.set_bess(b["c_rated"], float(cfg.BESS_P_MW) * 1e6)
    df = W.profile_frame(year)
    if hours:
        df = df.iloc[:int(hours)].reset_index(drop=True)
    env = m.THESSEnv(df, 3600.0, m.Feeder(m.CAPACITORS_MVAR), bess_p=bess,
                     hfc_p=m.HFCParams(), soh2_0=soh2_0)
    env.reset()
    env.sc = IND.NoSC()
    N = env.N
    F = b["forecasts"](year)[:N] / 1e6                          # (N, 24, 3) MW
    L_true, R_true = env.P_load / 1e6, env.P_res / 1e6

    ctl = new_controller(b, cfg)
    dsp = Dispatch(s_res=s_res, fc_min=fc_min, el_min=el_min, min_run_h=2)
    pb_prev = ph_prev = 0.0
    q_log = np.zeros(N)
    plan = None
    t0, fails, reopened, solves = time.time(), 0, 0, 0
    sched = np.zeros((N, 3))                                    # the committed day-ahead plan
    for t in range(N):
        k = t % H_DAY
        if k == 0:
            s_term = terminal_target(t, N, H_DAY, soh2_0, s_res)
            plan = plan_day(b, ctl, env.bess.soc, env.hfc.soh2, F[t, :H_DAY], pb_prev, ph_prev,
                            s_term)
            solves += 1
            if plan is None:
                fails += 1
            Fday = F[t, :H_DAY]
        if plan is None:
            pb = ph = qh = 0.0
        else:
            pb, ph, qh = plan[0][k], plan[1][k], float(np.clip(plan[2][k], -s_conv, s_conv))
        sched[t] = (pb, ph, qh)
        # forecast error of this hour against the forecast the day's plan was made on
        e = (L_true[t] - R_true[t]) - (Fday[k, 0] - Fday[k, 1] - Fday[k, 2])
        band = S3.VBand(L_true[t], env.P_pv[t] / 1e6, env.P_wind[t] / 1e6, qh, s_conv,
                        p_min_fc=fc_min * M2.P_FC, p_min_el=el_min * M2.P_EL)
        pb, ph = dsp(env, pb, ph, e, band)
        if getattr(cfg, "RT_NO_SHED_CHARGING", True):
            pb, ph = no_shed_charging(pb, ph, L_true[t], R_true[t], el_min * M2.P_EL)
        reopened += band.reopened
        pb_prev, ph_prev = pb, ph
        room = max(0.0, s_conv ** 2 - ph ** 2) ** 0.5 if s_conv else 0.0
        q_log[t] = float(np.clip(band.q, -room, room)) if s_conv else 0.0
        env.step(t, pb * 1e6, 0.0, ph * 1e6)

    volt = M2.voltage_study_q(env, q_log if s_conv else None)
    k = IND.five(env, volt, soh2_0)
    k.update(year=year, mode="day_ahead", horizon=H_DAY, hours=N, infeasible=fails,
             day_solves=solves, q_reopened=reopened, minutes=(time.time() - t0) / 60)
    if capture is not None:
        capture["schedule"] = sched
    return k


# ───────────────────────────────────────────────────────────── DRL environment, day-ahead
def make_plant_class(E, b):
    """A day-ahead wrapper around the core pipeline's hourly DRL plant (drl_env.Plant)."""
    M2 = b["M2"]

    class DayAheadPlant(E.Plant):
        ACT = 3 * H_DAY

        def __init__(self, *a, **kw):
            super(DayAheadPlant, self).__init__(*a, **kw)
            # the day-ahead forecast of every hour: the one issued at 00:00 of its day
            t = np.arange(self.N)
            self.da_net = self.Fnet[(t // H_DAY) * H_DAY, t % H_DAY]
            self.days = int(np.ceil(self.N / float(H_DAY)))
            self.sched = np.zeros((self.N, 3))          # committed 24 h schedules, MW / Mvar
            self._no_shed = bool(getattr(DayAheadPlant, "NO_SHED", True))

        def reset(self):
            self.sched = np.zeros((self.N, 3))
            return super(DayAheadPlant, self).reset()

        def trace(self):
            tr = super(DayAheadPlant, self).trace()
            L = self.env.log
            tr["schedule"] = self.sched.copy()
            for key in ("soc", "soh2", "P_def", "P_curt"):
                if key in L:
                    tr[key] = np.asarray(L[key][:self.N], float).copy()
            return tr

        # real-time layer: the hour's forecast error, battery first, then hydrogen
        def project(self, a):
            pb, ph, q = super(DayAheadPlant, self).project(a)
            t = self.t
            e = self.net[t] - self.da_net[t]
            want = pb + e
            env = self.env
            soc = env.bess.soc
            lim_d = min(self.P_B, max(0.0, (soc - self.SOC_MIN) * self.E_B / (self.ETA_C * self.DT_H)))
            lim_c = min(self.P_B, max(0.0, (self.SOC_MAX - soc) * self.E_B / (self.ETA_C * self.DT_H)))
            pb_new = min(max(want, -lim_c), lim_d)
            left = want - pb_new
            if left > 0 and env.hfc.soh2 > self.s_res:
                ph = min(ph + left, self.P_FC)
            elif left < 0:
                ph = max(ph + left, -self.P_EL)
            if self.s_conv:
                room = max(0.0, self.s_conv ** 2 - q ** 2) ** 0.5
                ph = float(np.clip(ph, -room, room))
            if self._no_shed:
                pb_new, ph = no_shed_charging(pb_new, ph, self.L[t], self.PV[t] + self.WD[t],
                                              self.el_min * self.P_EL)
            return pb_new, ph, q

        def day_obs(self):
            return self._obs(min(self.t, self.N - 1), self.pb_prev, self.ph_prev)

        def reset_day(self):
            self.reset()
            return self.day_obs()

        def step_day(self, A):
            """Execute one 24 h schedule. A: (72,) or (24, 3) in [-1, 1]."""
            A = np.asarray(A, np.float64).reshape(H_DAY, 3)
            t0 = self.t
            k = min(H_DAY, self.N - t0)
            a = np.clip(A[:k], -1.0, 1.0)
            self.sched[t0:t0 + k, 0] = a[:, 0] * self.P_B
            self.sched[t0:t0 + k, 1] = np.where(a[:, 1] >= 0, a[:, 1] * self.P_FC, a[:, 1] * self.P_EL)
            self.sched[t0:t0 + k, 2] = a[:, 2] * self.s_conv if self.s_conv else 0.0
            R, C, n = 0.0, 0.0, 0
            done = False
            for k in range(H_DAY):
                _, r, done, info = self.step(A[k])
                R += r
                C += info["cost"]
                n += 1
                if done:
                    break
            return self.day_obs(), R, done, {"cost": C / max(n, 1)}

    return DayAheadPlant


def to_action(plant, pb, ph, q):
    """MW schedule -> the agent's normalised action space, (n, 3) in [-1, 1]."""
    a = np.zeros((len(pb), 3))
    a[:, 0] = pb / plant.P_B
    a[:, 1] = np.where(ph >= 0, ph / plant.P_FC, ph / plant.P_EL)
    a[:, 2] = q / plant.s_conv if plant.s_conv else 0.0
    return np.clip(a, -1.0, 1.0)


def mpc_policy(b, cfg, plant):
    """The MPC as a day-ahead policy on the DRL plant: used to generate demonstrations."""
    from run_stage2 import terminal_target
    ctl = new_controller(b, cfg)
    F = plant.F                                               # (N, 24, 3) MW

    def policy(_obs):
        t = plant.t
        s_term = terminal_target(t, plant.N, H_DAY, plant.soh2_0, plant.s_res)
        out = plan_day(b, ctl, plant.env.bess.soc, plant.env.hfc.soh2, F[t, :H_DAY],
                       plant.pb_prev, plant.ph_prev, s_term)
        if out is None:
            return np.zeros(3 * H_DAY)
        return to_action(plant, *out).ravel()
    return policy
