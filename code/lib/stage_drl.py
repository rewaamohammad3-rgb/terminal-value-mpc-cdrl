# -*- coding: utf-8 -*-
"""One DRL cell: train on 2005-2016, select the checkpoint on 2017-2018 by deficit, score once on
the held-out test year with exact DistFlow.

    python stage_drl.py <run_dir> <algo> <test_year> <seed> [--episodes N]

DRL_MODE = "day_ahead" (default)
    The agent decides once per day at 00:00: observation = plant state + calendar + the 24 h
    forecast; action = the 24 h schedule (battery, hydrogen, converter reactive power, 72 values).
    0. MPC advisor: a small neural network is trained on the MPC's day-ahead decisions of
       DRL_DEMO_YEARS (default four years) to predict the MPC's 24 h schedule from the 00:00
       observation. Its prediction is appended to the agent's observation every day
       (DRL_MPC_HINT), so the agent always "sees" what the MPC would do, without solving an MPC.
    1. Behaviour cloning: the policy is initialised to reproduce the MPC's schedules on the same
       years, and the value critics are fitted to the MPC's returns (DRL_IMITATION).
    2. Fine-tuning with the core pipeline's constrained algorithm (FP-PPO, PD-PPO, FOCOPS or CPO),
       one training year per episode. Episode 0 (the cloned policy) is a candidate checkpoint, so
       fine-tuning can never make the selected agent worse on validation than the clone.
DRL_MODE = "hourly"
    The core pipeline's original protocol (train.run): one action per hour.

Writes into <run_dir>/drl/: result_*.json, history_*.json, traces/*.npz
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core  # noqa: E402

TRAIN_YEARS = list(range(2005, 2017))
VAL_YEARS = [2017, 2018]


def rollout(pl, policy):
    o = pl.reset_day()
    R, done = 0.0, False
    while not done:
        o, r, done, _ = pl.step_day(policy(o))
        R += r
    return R


def score(pl, agent, exact):
    R = rollout(pl, lambda o: agent.act(o, greedy=True)[0])
    k = pl.metrics(exact=exact)
    k["return"] = R
    return k


def load_demos(run_dir, years):
    out = []
    for y in years:
        p = os.path.join(run_dir, "demos", "demo_%d.npz" % y)
        if not os.path.exists(p):
            raise SystemExit("missing %s - run the demos stage first (python run.py)" % p)
        out.append(dict(np.load(p)))
    return out


def train_advisor(demos, seed, log, epochs=400):
    """MPC advisor: base observation at 00:00 -> the MPC's 24 h schedule (72 values in [-1, 1])."""
    import torch
    import torch.nn as nn
    X = torch.as_tensor(np.concatenate([d["obs"] for d in demos]), dtype=torch.float32)
    Y = torch.as_tensor(np.clip(np.concatenate([d["act"] for d in demos]), -0.995, 0.995),
                        dtype=torch.float32)
    torch.manual_seed(seed)
    net = nn.Sequential(nn.Linear(X.shape[1], 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(),
                        nn.Linear(256, Y.shape[1]), nn.Tanh())
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
    g = torch.Generator().manual_seed(seed)
    n = len(X)
    for ep in range(epochs):
        perm = torch.randperm(n, generator=g)
        tot = 0.0
        for i in range(0, n, 128):
            j = perm[i:i + 128]
            loss = ((net(X[j]) - Y[j]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * len(j)
        if ep % 100 == 0 or ep == epochs - 1:
            log("    advisor epoch %3d  MSE vs MPC schedule %.4f" % (ep, tot / n))
    net.eval()
    return net, tot / n


def behaviour_clone(agent, demos, cfg, log, hint=None):
    """Fit tanh(mu(obs)) to the MPC's actions, and the critics to the MPC's discounted returns."""
    import torch
    X, Y, G, GC = [], [], [], []
    for d in demos:
        X.append(d["obs"] if hint is None else hint(d["obs"]))
        Y.append(np.clip(d["act"], -0.995, 0.995))
        r, c = d["reward"], d["cost"]
        g = np.zeros_like(r)
        gc = np.zeros_like(c)
        run = runc = 0.0
        for i in range(len(r) - 1, -1, -1):
            run = r[i] + agent.gamma * run
            runc = c[i] + agent.gamma * runc
            g[i], gc[i] = run, runc
        G.append(g)
        GC.append(gc)
    X = torch.as_tensor(np.concatenate(X), dtype=torch.float32)
    Y = torch.as_tensor(np.concatenate(Y), dtype=torch.float32)
    G = torch.as_tensor(np.concatenate(G), dtype=torch.float32).unsqueeze(1)
    GC = torch.as_tensor(np.concatenate(GC), dtype=torch.float32).unsqueeze(1)
    n = len(X)
    epochs = int(getattr(cfg, "DRL_BC_EPOCHS", 300))
    opt = torch.optim.Adam(agent.actor.mu.parameters(), lr=1e-3)
    nets = [agent.v] + ([agent.vc] if getattr(agent, "vc", None) is not None else [])
    targets = [G] + ([GC] if len(nets) > 1 else [])
    optv = [torch.optim.Adam(nn.parameters(), lr=1e-3) for nn in nets]
    g = torch.Generator().manual_seed(0)
    for ep in range(epochs):
        perm = torch.randperm(n, generator=g)
        tot = 0.0
        for i in range(0, n, 256):
            j = perm[i:i + 256]
            loss = ((torch.tanh(agent.actor.mu(X[j])) - Y[j]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * len(j)
            for net, tgt, o in zip(nets, targets, optv):
                lv = ((net(X[j]) - tgt[j]) ** 2).mean()
                o.zero_grad()
                lv.backward()
                o.step()
        if ep % 50 == 0 or ep == epochs - 1:
            log("    BC epoch %3d  action MSE %.4f" % (ep, tot / n))
    with torch.no_grad():
        agent.actor.log_std.fill_(float(getattr(cfg, "DRL_BC_LOG_STD", -2.3)))
    agent._refresh()
    return tot / n


def run_dayahead(run_dir, cfg, algo, test, seed, episodes, log):
    import torch
    b = core.bootstrap(cfg, run_dir)
    import drl_env as E
    import agents as AG
    import dayahead
    E.PROD_DATA = b["forecast_dir"]
    AG.COST_LIMIT = float(getattr(cfg, "DRL_COST_LIMIT", AG.COST_LIMIT))
    core.set_defaults(E.Plant.__init__, s_conv=float(cfg.S_CONV_MVA), horizon=24)
    DA = dayahead.make_plant_class(E, b)
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    np.random.seed(seed)
    assert test not in TRAIN_YEARS + VAL_YEARS

    t0 = time.time()
    demo_years = list(getattr(cfg, "DRL_DEMO_YEARS", [2013, 2014, 2015, 2016]))
    bad = [y for y in demo_years if y not in TRAIN_YEARS]
    if bad:
        raise SystemExit("DRL_DEMO_YEARS must lie in the training years 2005-2016, got %s" % bad)
    use_hint = bool(getattr(cfg, "DRL_MPC_HINT", True))
    use_bc = bool(getattr(cfg, "DRL_IMITATION", True))
    demos = load_demos(run_dir, demo_years) if (use_hint or use_bc) else []
    hint, adv_mse = None, None
    if use_hint:
        log("  MPC advisor network, trained on the MPC's day-ahead schedules of %s" % demo_years)
        adv, adv_mse = train_advisor(demos, seed, log)
        torch.save(adv.state_dict(), os.path.join(run_dir, "drl", "advisor_%s_%d_s%d.pt"
                                                   % (algo, test, seed)))

        def hint(o):
            o = np.asarray(o, np.float32)
            with torch.no_grad():
                h = adv(torch.as_tensor(o.reshape(-1, o.shape[-1]))).numpy()
            return np.concatenate([o.reshape(-1, o.shape[-1]), h], 1).reshape(
                o.shape[:-1] + (o.shape[-1] + h.shape[-1],))

        class Hinted(DA):                       # the advisor's 24 h suggestion joins the obs
            def day_obs(self):
                return hint(DA.day_obs(self))
        Plant = Hinted
    else:
        Plant = DA

    tr = {}
    def plant(y):
        if y not in tr:
            tr[y] = Plant(year=y)
        return tr[y]
    pv = [plant(y) for y in VAL_YEARS]
    obs_dim = len(pv[0].reset_day())
    imitation_only = (algo == "il")
    if imitation_only:                       # the MPC-imitation policy alone, no RL
        episodes = 0
        agent = AG.make("fpppo", obs_dim, DA.ACT, seed=seed)
    else:
        # Fine-tuning starts from a policy that already matches the MPC, so it must move
        # gently: a small step size (or trust region for CPO) keeps the policy from being
        # thrown away by the first noisy updates.
        hyper = {"delta": float(getattr(cfg, "DRL_CPO_DELTA", 0.002))} if algo == "cpo" else \
            {"lr": float(getattr(cfg, "DRL_FT_LR", 2e-5))}
        agent = AG.make(algo, obs_dim, DA.ACT, seed=seed, **hyper)

    hist = []
    bc_mse = None
    if use_bc:
        log("  behaviour cloning on the MPC's day-ahead schedules of %s" % demo_years)
        bc_mse = behaviour_clone(agent, demos, cfg, log, hint=hint)

    def validate():
        ks = [score(p, agent, exact=False) for p in pv]
        return float(np.mean([k["D_L_corrected_pct"] for k in ks])), \
            float(np.mean([k["return"] for k in ks]))

    vd, vr = validate()
    # Episode 0 (the imitation policy) is a selection candidate only for the "il" method, or
    # if DRL_SELECT_EP0 is set; otherwise each algorithm is reported on its own fine-tuning.
    sel0 = imitation_only or bool(getattr(cfg, "DRL_SELECT_EP0", False))
    best = {"score": vd if sel0 else np.inf, "state": agent.state(), "ep": 0}
    hist.append({"ep": 0, "val_deficit": vd, "val_return": vr, "phase": "imitation"})
    log("    ep   0  (imitation policy)  val deficit %.3f %%" % vd)

    every = max(1, int(getattr(cfg, "DRL_EVAL_EVERY", 5)))
    buf = AG.Rollout(400, obs_dim, DA.ACT)
    steps = 0
    # Critic warm-up: a few years of rollouts that train only the value functions, with the
    # policy put back after each update, so the first policy steps use sensible advantages.
    warm = 0 if imitation_only else int(getattr(cfg, "DRL_CRITIC_WARMUP", 5))
    for w in range(warm):
        keep = {k: v.detach().clone() for k, v in agent.actor.state_dict().items()}
        pl = plant(TRAIN_YEARS[(w + 7) % len(TRAIN_YEARS)])
        o = pl.reset_day()
        done = False
        while not done:
            a, lp, v, vc = agent.act(o)
            o2, r, done, info = pl.step_day(a)
            buf.push(o, a, lp, r, info["cost"], v, vc, done)
            o = o2
        agent.update(buf)
        agent.actor.load_state_dict(keep)
        agent._refresh()
    if warm:
        log("    critic warm-up: %d training years, policy held at the imitation policy" % warm)
    for ep in range(1, episodes + 1):
        pl = plant(TRAIN_YEARS[ep % len(TRAIN_YEARS)])
        o = pl.reset_day()
        R, C, done = 0.0, 0.0, False
        while not done:
            a, lp, v, vc = agent.act(o)
            o2, r, done, info = pl.step_day(a)
            buf.push(o, a, lp, r, info["cost"], v, vc, done)
            R += r
            C += info["cost"]
            steps += 1
            o = o2
        agent.update(buf)
        h = {"ep": ep, "year": pl.year, "train_return": R, "train_cost": C,
             "mult": getattr(agent, "mult", 0.0)}
        if ep % every == 0 or ep == episodes:
            vd, vr = validate()
            h.update(val_deficit=vd, val_return=vr)
            if vd < best["score"]:
                best = {"score": vd, "state": agent.state(), "ep": ep}
            log("    ep %3d  train %+9.1f  val deficit %.3f %%  (best %.3f @ ep %d)"
                % (ep, R, vd, best["score"], best["ep"]))
        hist.append(h)

    agent.load(best["state"])
    torch.save(best["state"], os.path.join(run_dir, "drl", "agent_%s_%d_s%d.pt" % (algo, test, seed)))
    pt = Plant(year=test)
    out = score(pt, agent, exact=True)
    # the test year's network, kept like the MPC's: exact voltages, branch P and Q, losses
    tr = pt.trace()
    NP = b["NP"]
    ex = NP.solve_distflow(tr)
    lin = NP.solve_network(tr)
    np.savez_compressed(os.path.join(run_dir, "drl", "network_%s_%d_s%d.npz" % (algo, test, seed)),
                        v_exact=ex["v"].astype(np.float32), v_linear=lin["v"].astype(np.float32),
                        P_br=ex["P_br"].astype(np.float32), Q_br=ex["Q_br"].astype(np.float32),
                        loss_p=ex["loss_p"].astype(np.float32),
                        P_slack=ex["P_slack"].astype(np.float32),
                        Q_slack=ex["Q_slack"].astype(np.float32))
    np.savez_compressed(os.path.join(run_dir, "drl", "traces", "%s_%d_s%d.npz" % (algo, test, seed)),
                        **tr)
    vm = ex["v"][:, 1:]
    out.update(over_hours=int((vm > 1.05).any(1).sum()), under_hours=int((vm < 0.95).any(1).sum()),
               v_min_bus=int(vm.min(0).argmin() + 2), v_max_bus=int(vm.max(0).argmax() + 2))
    out.update(algo=algo, test_year=test, seed=seed, mode="day_ahead", critic_warmup=warm,
               imitation=use_bc, mpc_hint=use_hint, demo_years=demo_years,
               advisor_mse=adv_mse, bc_action_mse=bc_mse,
               train_years=TRAIN_YEARS, val_year=VAL_YEARS, episodes=episodes,
               env_steps_days=steps, best_episode=best["ep"], val_deficit_pct=best["score"],
               multiplier=getattr(agent, "mult", 0.0), minutes=(time.time() - t0) / 60.0)
    return out, hist


def main():
    run_dir, algo, test, seed = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    cfg = core.load_config(os.path.join(run_dir, "config_used.py"))
    episodes = (int(sys.argv[sys.argv.index("--episodes") + 1]) if "--episodes" in sys.argv
                else int(cfg.DRL_EPISODES))
    out = os.path.join(run_dir, "drl")
    tag = "%s_%d_s%d" % (algo, test, seed)
    os.makedirs(os.path.join(out, "traces"), exist_ok=True)
    os.environ["DRL_RUNS_DIR"] = os.path.join(out, "_train_cache")
    os.environ["DRL_TRACE_DIR"] = os.path.join(out, "traces")
    os.environ["DRL_TRACE_TAG"] = tag

    def log(msg):
        print(msg, flush=True)

    t0 = time.time()
    if getattr(cfg, "DRL_MODE", "day_ahead") == "day_ahead":
        res, hist = run_dayahead(run_dir, cfg, algo, test, seed, episodes, log)
    else:
        if cfg.DRL_RESIDUAL:
            os.environ["DRL_MPC_TRACE"] = os.path.join(run_dir, "mpc", "trace_%d.npz")
        b = core.bootstrap(cfg, run_dir)
        import drl_env as E
        import train as T
        import agents as AG
        E.PROD_DATA = b["forecast_dir"]
        AG.COST_LIMIT = float(getattr(cfg, "DRL_COST_LIMIT", AG.COST_LIMIT))
        core.set_defaults(E.Plant.__init__, s_conv=float(cfg.S_CONV_MVA),
                          horizon=int(cfg.HORIZON))
        res, hist = T.run(algo, test, seed, episodes=episodes, quiet=False,
                          residual=bool(cfg.DRL_RESIDUAL))
        res["mode"] = "hourly"
    b = core.bootstrap(cfg, run_dir)
    res["wall_minutes"] = (time.time() - t0) / 60.0
    res["plant"] = core.plant_summary(b)
    core.dump_json(hist, os.path.join(out, "history_%s.json" % tag))
    core.dump_json(res, os.path.join(out, "result_%s.json" % tag))
    print("DRL %s: deficit %.3f %%, exact compliance %.2f %%, %.0f min"
          % (tag, res.get("D_L_corrected_pct", float("nan")),
             res.get("kappa_exact_pct", float("nan")), res["wall_minutes"]), flush=True)


if __name__ == "__main__":
    main()
