# -*- coding: utf-8 -*-
"""
MPC + DRL, option A: the DRL agent sets the MPC's end-of-day storage values.   Run from v3:

    python mpc_drl_targets\\run_targets_drl.py                     uses mpc_drl_targets\\config.py
    python mpc_drl_targets\\run_targets_drl.py --config other.py    any other config file
    python mpc_drl_targets\\run_targets_drl.py --stages plots       redraw only
    python mpc_drl_targets\\run_targets_drl.py --workers 4 --force

What it does
  1. drl    one job per seed: train the agent on 2005-2016 (one year per episode, the MPC solving
            every day of it), keep the checkpoint with the lowest deficit on 2017-2018, then run
            it on every test year. The first seed also re-runs the plain MPC through the same code
            on every test year, as a check against the main run's MPC rows.
  2. plots  one comparison over MPC, MPC-imitation, the best DRL method of the main run and
            MPC + DRL values: every figure and table of the main pipeline, plus the learned
            values over the year, into mpc_drl_targets/results/<RUN_NAME>/figures/

Its config is mpc_drl_targets\\config.py (RUN_NAME there must be a finished main run, with the
same plant). Settings for this run (defaults in brackets)
    TARGETS_DRL_ALGO      ["auto"]        "auto" (best of the main run) or fpppo, pdppo, focops, cpo
    TARGETS_DRL_EPISODES  [60]            training episodes, one training year each (about 25 s each)
    TARGETS_DRL_SEEDS     [0, 1, 2, 3]    one job per seed, run in parallel on WORKERS cores
    TARGETS_DRL_LOG_STD   [-1.0]          exploration noise of the two values (in [-1, 1] units)
    TARGETS_DRL_GAMMA     [0.99]          discount per day (0.99 looks about 100 days ahead)

The main run (python run.py, same RUN_NAME) should be finished first.
"""
import argparse
import csv
import io
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
V3 = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(V3, "lib"))
import build_tree  # noqa: E402
import core        # noqa: E402

PY = sys.executable
NAME_TO_KEY = {"FP-PPO": "fpppo", "PD-PPO": "pdppo", "FOCOPS": "focops", "CPO": "cpo"}
KEY_TO_NAME = {v: k for k, v in NAME_TO_KEY.items()}


def say(log, msg):
    line = "%s  %s" % (time.strftime("%H:%M:%S"), msg)
    print(line, flush=True)
    with io.open(log, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def pool(jobs, workers, log):
    pending, running, failed = list(jobs), [], []
    while pending or running:
        while pending and len(running) < workers:
            name, cmd, lf = pending.pop(0)
            fh = io.open(lf, "w", encoding="utf-8")
            env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8")
            running.append((name, subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT,
                                                   cwd=HERE, env=env), fh, time.time()))
            say(log, "  started  %s" % name)
        time.sleep(2)
        for item in list(running):
            name, p, fh, t0 = item
            if p.poll() is not None:
                fh.close()
                running.remove(item)
                with io.open(fh.name, encoding="utf-8", errors="replace") as f:
                    lines = [l for l in f.read().splitlines() if l.strip()]
                if p.returncode == 0:
                    say(log, "  finished %s (%.1f min)" % (name, (time.time() - t0) / 60))
                    for l in lines:
                        if l.startswith(("TEST", "training done")):
                            say(log, "      " + l)
                else:
                    failed.append(name)
                    say(log, "  FAILED   %s - see %s" % (name, fh.name))
                    for l in lines[-6:]:
                        say(log, "      " + l)
    return failed


def best_algo(cfg, base_run, log):
    a = str(getattr(cfg, "TARGETS_DRL_ALGO", "auto")).lower()
    if a != "auto":
        return a
    p = os.path.join(base_run, "summary.csv")
    if os.path.exists(p):
        acc = {}
        with io.open(p, encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                if row["method"] in NAME_TO_KEY and row["deficit_pct"]:
                    acc.setdefault(row["method"], []).append(float(row["deficit_pct"]))
        if acc:
            name = min(acc, key=lambda k: sum(acc[k]) / len(acc[k]))
            say(log, "best DRL method of the main run: %s (mean deficit %.3f %%)"
                % (name, sum(acc[name]) / len(acc[name])))
            return NAME_TO_KEY[name]
    say(log, "no finished main run found; using CPO")
    return "cpo"


def copy_main_results(base_run, res_dir, algo):
    """Bring the MPC, MPC-imitation and best-DRL results of the main run into the comparison."""
    if not os.path.isdir(base_run):
        return
    src_mpc = os.path.join(base_run, "mpc")
    if os.path.isdir(src_mpc):
        os.makedirs(os.path.join(res_dir, "mpc"), exist_ok=True)
        for f in os.listdir(src_mpc):
            shutil.copy2(os.path.join(src_mpc, f), os.path.join(res_dir, "mpc", f))
    src = os.path.join(base_run, "drl")
    if not os.path.isdir(src):
        return
    for a in ("il", algo):
        for f in os.listdir(src):
            if f.startswith(("result_%s_" % a, "history_%s_" % a, "network_%s_" % a)):
                shutil.copy2(os.path.join(src, f), os.path.join(res_dir, "drl", f))
        tdir = os.path.join(src, "traces")
        for f in (os.listdir(tdir) if os.path.isdir(tdir) else []):
            if f.startswith("%s_" % a):
                shutil.copy2(os.path.join(tdir, f), os.path.join(res_dir, "drl", "traces", f))


def main():
    ap = argparse.ArgumentParser()
    own = os.path.join(HERE, "config.py")
    ap.add_argument("--config", default=own if os.path.exists(own) else os.path.join(V3, "config.py"))
    ap.add_argument("--stages", default="drl,plots")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--episodes", type=int, default=None, help="override TARGETS_DRL_EPISODES")
    ap.add_argument("--hours", type=int, default=None, help="test only: hours per year")
    a = ap.parse_args()
    cfg = core.load_config(os.path.abspath(a.config))
    res_dir = os.path.join(HERE, "results", cfg.RUN_NAME)
    base_run = os.path.join(V3, "runs", cfg.RUN_NAME)
    for d in ("drl", os.path.join("drl", "traces"), "logs", "figures", "values", "mpc"):
        os.makedirs(os.path.join(res_dir, d), exist_ok=True)
    shutil.copy2(os.path.abspath(a.config), os.path.join(res_dir, "config_used.py"))
    log = os.path.join(res_dir, "logs", "run.log")
    workers = a.workers or int(cfg.WORKERS)
    stages = a.stages.split(",")
    say(log, "MPC + DRL storage values for %s  (results in %s)" % (cfg.RUN_NAME, res_dir))
    say(log, "config: %s" % os.path.abspath(a.config))
    if not os.path.isdir(base_run):
        say(log, "WARNING: no main run at %s - the comparison will have no MPC rows" % base_run)
    if build_tree.stale():
        build_tree.build()
    core.prepare_forecasts(res_dir, cfg, list(range(2005, 2024)))
    algo = best_algo(cfg, base_run, log)
    copy_main_results(base_run, res_dir, algo)               # the MPC rows, for the check too

    if "drl" in stages:
        seeds = list(getattr(cfg, "TARGETS_DRL_SEEDS", [0, 1, 2, 3]))
        jobs = []
        for s in seeds:
            done = all(os.path.exists(os.path.join(res_dir, "drl", "result_mpcdrl_%d_s%d.json" % (y, s)))
                       for y in cfg.DRL_TEST_YEARS)
            if done and not a.force:
                continue
            cmd = [PY, "targets_drl.py", "train", res_dir, algo, str(s)]
            if a.episodes:
                cmd += ["--episodes", str(a.episodes)]
            if a.hours:
                cmd += ["--hours", str(a.hours)]
            if s == seeds[0]:
                cmd += ["--check"]
            jobs.append(("MPC+%s values, seed %d" % (KEY_TO_NAME.get(algo, algo), s), cmd,
                         os.path.join(res_dir, "logs", "train_s%d.log" % s)))
        ep = a.episodes or int(getattr(cfg, "TARGETS_DRL_EPISODES", 60))
        say(log, "training %d seed(s) with %s, %d episodes each (about %d min per seed)"
            % (len(jobs), KEY_TO_NAME.get(algo, algo), ep, int(ep * 0.45 + 15)))
        pool(jobs, workers, log)

    if "plots" in stages:
        say(log, "figures and tables")
        label = "MPC+%s values" % KEY_TO_NAME.get(algo, algo)
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", V3_MPCDRL_LABEL=label)
        subprocess.run([PY, os.path.join(V3, "lib", "plots.py"), res_dir],
                       cwd=os.path.join(V3, "lib"), env=env)
        subprocess.run([PY, os.path.join(HERE, "values_figures.py"), res_dir], cwd=HERE, env=env)
    say(log, "done: %s" % res_dir)


if __name__ == "__main__":
    main()
