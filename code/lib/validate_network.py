# -*- coding: utf-8 -*-
"""Check the network formulation used for scoring, against two independent references.

    python validate_network.py [run_dir] [year]

1. IEEE 33-bus benchmark (Baran & Wu 1989): nominal load, no DER, no capacitors.
   Published: V_min 0.9038 pu at bus 18, active losses 202.7 kW, reactive 135.1 kvar.
2. Every hour of a finished MPC year, scored by the core pipeline's exact DistFlow
   (`network_profiles.solve_distflow`) and by an independent complex-voltage
   backward/forward sweep written here (constant-power loads, no DistFlow algebra).
   The two must agree on |V| at every bus and on the substation P and Q.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core  # noqa: E402


def bfs_complex(F, p_net, q_net, tol=1e-10, it=100):
    """Complex backward/forward sweep. p_net, q_net: (T, 33) net injections in pu (gen +)."""
    T, n = p_net.shape
    S = p_net + 1j * q_net                         # injection at each bus
    z = F.r + 1j * F.x
    order = sorted(range(len(F.fr)), key=lambda k: len(F.path[F.to[k]]))
    V = np.ones((T, n), complex)
    for _ in range(it):
        I_inj = np.conj(S / V)                      # injected current (gen +)
        I_inj[:, 0] = 0.0
        I_br = np.zeros((T, len(F.fr)), complex)
        for k in reversed(order):                   # backward: branch current = load beyond
            b = F.to[k]
            I_br[:, k] = -I_inj[:, b]
            for kk in range(len(F.fr)):
                if F.fr[kk] == b:
                    I_br[:, k] += I_br[:, kk]
        Vn = V.copy()
        for k in order:                             # forward
            Vn[:, F.to[k]] = Vn[:, F.fr[k]] - z[k] * I_br[:, k]
        d = np.max(np.abs(Vn - V))
        V = Vn
        if d < tol:
            break
    root = [k for k in range(len(F.fr)) if F.fr[k] == 0]
    S_slack = (V[:, [0]] * np.conj(I_br[:, root])).sum(1)
    loss = (np.abs(I_br) ** 2 * F.r).sum(1) + 1j * (np.abs(I_br) ** 2 * F.x).sum(1)
    return np.abs(V), S_slack, loss


def main():
    v3 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    run_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(v3, "runs", "base_delivered")
    year = int(sys.argv[2]) if len(sys.argv) > 2 else 2020
    cfg = core.load_config(os.path.join(run_dir, "config_used.py"))
    b = core.bootstrap(cfg, run_dir)
    NP, m = b["NP"], b["m"]
    F = NP.FEEDER
    base = m.S_BASE_W / 1e6

    # 1. benchmark, through the core pipeline's own exact scorer with no shedding
    n = NP.FEEDER.N_BUS
    tr0 = {"P_load": np.array([3.715e6]), "load_scale": np.array([1.0]),
           "P_pv": np.zeros(1), "P_wind": np.zeros(1), "P_res": np.zeros(1),
           "act_b": np.zeros(1), "act_h": np.zeros(1), "qh": np.zeros(1),
           "_shed_applied": True}
    qcap, F.q_cap = F.q_cap.copy(), np.zeros(n)
    ex = NP.solve_distflow(tr0, iters=30)
    vb, Sb, lb = bfs_complex(F, -F.P0[None, :], -F.Q0[None, :])
    F.q_cap = qcap
    v = ex["v"][0]
    print("1. IEEE 33-bus benchmark (nominal load, no DER, no capacitors)")
    print("   published      : V_min 0.9038 pu @ bus 18, losses 202.7 kW / 135.1 kvar")
    print("   exact DistFlow : V_min %.4f pu @ bus %d, losses %.1f kW / %.1f kvar"
          % (v.min(), v.argmin() + 1, ex["loss_p"].sum() * 1e3, ex["loss_q"].sum() * 1e3))
    print("   complex BFS    : V_min %.4f pu @ bus %d, losses %.1f kW / %.1f kvar"
          % (vb[0].min(), vb[0].argmin() + 1, lb[0].real * base * 1e3, lb[0].imag * base * 1e3))

    # 2. a real year, exactly as the scorer sees it (served load, delivered renewables)
    p = os.path.join(run_dir, "mpc", "trace_%d.npz" % year)
    if not os.path.exists(p):
        print("2. skipped: no %s" % p)
        return
    tr = dict(np.load(p))
    import voltagefix as VF
    t = dict(tr)
    t["load_scale"] = tr["load_scale"] * VF.shed_factor(tr, F, NP.injections)
    t["_shed_applied"] = True
    ex = NP.solve_distflow(t, iters=30)
    p_gen, q_gen = NP.injections(t)
    s = t["load_scale"][:, None]
    vb, Sb, lb = bfs_complex(F, p_gen - s * F.P0, q_gen + F.q_cap - s * F.Q0)
    dv = np.abs(ex["v"] - vb)
    print("2. %s %d, %d hours: exact DistFlow vs independent complex BFS"
          % (os.path.basename(run_dir), year, len(vb)))
    print("   max |dV| over all buses and hours : %.2e pu" % dv.max())
    print("   max |dP| at substation            : %.2e MW" % np.abs(ex["P_slack"] - Sb.real * base).max())
    print("   max |dQ| at substation            : %.2e Mvar" % np.abs(ex["Q_slack"] - Sb.imag * base).max())
    print("   annual losses                     : %.1f vs %.1f MWh"
          % (ex["loss_p"].sum(), lb.real.sum() * base))
    print("   substation P range                : %.3f .. %.3f MW (import +)"
          % (Sb.real.min() * base, Sb.real.max() * base))


if __name__ == "__main__":
    main()
