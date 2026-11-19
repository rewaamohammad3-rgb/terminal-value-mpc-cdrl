# -*- coding: utf-8 -*-
"""
v3 pipeline runner.  Edit config.py, then:

    python run.py                    MPC on all MPC_YEARS, MPC demos, DRL (if enabled), figures
    python run.py --quick            2-week smoke test of every stage (~5 min), into runs/<RUN_NAME>_quick
    python run.py --stages mpc       only the MPC (+ exact network scoring)
    python run.py --stages demos,drl only the DRL (demos = MPC schedules 2005-2016 to imitate)
    python run.py --stages plots     only redraw figures from results already on disk
    python run.py --workers 6        override WORKERS
    python run.py --force            redo finished years/cells of this RUN_NAME
    python run.py --config my_other_config.py

    python compare.py base H2bus6_cap33-0.3      side-by-side figures of finished runs

Output (all inside v3/runs/<RUN_NAME>/):
    config_used.py   exact copy of the config this run used
    plant_used.json  the plant read back from the live model (sizes, buses, capacitors)
    mpc/             per-year traces, exact network results, KPI json
    drl/             per-cell results, training histories, test-year traces
    figures/         network/, mpc/, drl/, summary/   (PNG + PDF)
    summary.csv      one row per year: MPC and DRL deficit, compliance, losses, violations
    logs/            one log per process, plus run.log
"""
import argparse
import filecmp
import io
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
LIB = os.path.join(HERE, "lib")
sys.path.insert(0, LIB)
import build_tree  # noqa: E402
import core        # noqa: E402

PY = sys.executable
ALL_YEARS = list(range(2005, 2024))


def say(log, msg):
    line = "%s  %s" % (time.strftime("%H:%M:%S"), msg)
    print(line, flush=True)
    with io.open(log, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def pool(jobs, workers, log):
    """Run (name, cmd, logfile) jobs, `workers` at a time. Returns the names that failed."""
    pending, running, failed = list(jobs), [], []
    while pending or running:
        while pending and len(running) < workers:
            name, cmd, lf = pending.pop(0)
            fh = io.open(lf, "w", encoding="utf-8")
            env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8")
            p = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=LIB, env=env)
            running.append((name, p, fh, time.time()))
            say(log, "  started  %s" % name)
        time.sleep(2)
        for item in list(running):
            name, p, fh, t0 = item
            if p.poll() is not None:
                fh.close()
                running.remove(item)
                mins = (time.time() - t0) / 60
                if p.returncode == 0:
                    with io.open(fh.name, encoding="utf-8", errors="replace") as f:
                        last = [l for l in f.read().splitlines() if l.strip()]
                    say(log, "  finished %s (%.1f min)  %s" % (name, mins, last[-1] if last else ""))
                else:
                    failed.append(name)
                    say(log, "  FAILED   %s (exit %s) - see %s" % (name, p.returncode, fh.name))
    return failed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "config.py"))
    ap.add_argument("--stages", default="mpc,demos,drl,plots")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()

    cfg = core.load_config(os.path.abspath(a.config))
    name = cfg.RUN_NAME + ("_quick" if a.quick else "")
    run_dir = os.path.join(HERE, "runs", name)
    for d in ("mpc", "demos", "drl", "figures", "logs"):
        os.makedirs(os.path.join(run_dir, d), exist_ok=True)
    log = os.path.join(run_dir, "logs", "run.log")
    stages = [s.strip() for s in a.stages.split(",")]
    workers = a.workers or int(cfg.WORKERS)

    # the same RUN_NAME must never mix results from two different configs
    used = os.path.join(run_dir, "config_used.py")
    if os.path.exists(used) and not filecmp.cmp(used, os.path.abspath(a.config), shallow=False):
        old, new = core.config_dict(core.load_config(used)), core.config_dict(cfg)
        diff = [k for k in new if old.get(k) != new.get(k) and k not in
                ("WORKERS", "PLOT_YEAR", "PLOT_WEEK_START_DAY", "FIG_FORMATS", "DRL_ENABLE",
                 "MPC_YEARS", "DRL_TEST_YEARS", "DRL_SEEDS", "DRL_ALGOS")]
        if diff and not a.force:
            raise SystemExit("runs/%s was made with a different config (%s).\n"
                             "Change RUN_NAME in config.py, or pass --force to wipe that run."
                             % (name, ", ".join(diff)))
        if diff:
            for d in ("mpc", "demos", "drl", "forecasts", "figures"):
                shutil.rmtree(os.path.join(run_dir, d), ignore_errors=True)
                os.makedirs(os.path.join(run_dir, d), exist_ok=True)
    shutil.copy2(os.path.abspath(a.config), used)

    say(log, "run %s  stages %s  workers %d" % (name, ",".join(stages), workers))
    if build_tree.stale():
        say(log, "building the code tree from ../pipeline_codes_20261113")
        build_tree.build()
    core.prepare_forecasts(run_dir, cfg, ALL_YEARS)

    mpc_years = [2020] if a.quick else list(cfg.MPC_YEARS)
    if "mpc" in stages:
        jobs = []
        for y in mpc_years:
            if os.path.exists(os.path.join(run_dir, "mpc", "kpi_%d.json" % y)) and not a.force:
                continue
            cmd = [PY, "stage_mpc.py", run_dir, str(y)] + (["--hours", "336"] if a.quick else [])
            jobs.append(("MPC %d" % y, cmd, os.path.join(run_dir, "logs", "mpc_%d.log" % y)))
        say(log, "MPC: %d year(s) to solve (about 3-6 min each)" % len(jobs))
        bad = pool(jobs, workers, log)
        if bad:
            say(log, "MPC failures: %s" % bad)

    dayahead_drl = getattr(cfg, "DRL_MODE", "day_ahead") == "day_ahead"
    if "demos" in stages and (cfg.DRL_ENABLE or a.quick) and dayahead_drl:
        jobs = []
        need = (getattr(cfg, "DRL_IMITATION", True) or getattr(cfg, "DRL_MPC_HINT", True))
        for y in (list(getattr(cfg, "DRL_DEMO_YEARS", [2013, 2014, 2015, 2016])) if need else []):
            if os.path.exists(os.path.join(run_dir, "demos", "demo_%d.npz" % y)) and not a.force:
                continue
            jobs.append(("demos %d" % y, [PY, "stage_demos.py", run_dir, str(y)],
                         os.path.join(run_dir, "logs", "demos_%d.log" % y)))
        say(log, "MPC demonstrations for imitation: %d year(s) (about 0.5 min each)" % len(jobs))
        bad = pool(jobs, workers, log)
        if bad:
            say(log, "demo failures: %s - DRL cannot start without them" % bad)
            stages = [s_ for s_ in stages if s_ != "drl"]

    if "drl" in stages and (cfg.DRL_ENABLE or a.quick):
        algos = list(cfg.DRL_ALGOS)[:1] if a.quick else list(cfg.DRL_ALGOS)
        if (dayahead_drl and getattr(cfg, "DRL_IMITATION", True) and "il" not in algos
                and getattr(cfg, "DRL_REPORT_IMITATION", True)):
            algos = ["il"] + algos            # the MPC-imitation policy alone, as a reference
        years = [2020] if a.quick else list(cfg.DRL_TEST_YEARS)
        seeds = [0] if a.quick else list(cfg.DRL_SEEDS)
        jobs = []
        for al in algos:
            for y in years:
                for s in seeds:
                    tag = "%s_%d_s%d" % (al, y, s)
                    if (os.path.exists(os.path.join(run_dir, "drl", "result_%s.json" % tag))
                            and not a.force):
                        continue
                    cmd = [PY, "stage_drl.py", run_dir, al, str(y), str(s)]
                    if a.quick:
                        cmd += ["--episodes", "2"]
                    jobs.append(("DRL %s" % tag, cmd,
                                 os.path.join(run_dir, "logs", "drl_%s.log" % tag)))
        say(log, "DRL: %d cell(s) to train, %d at a time" % (len(jobs), workers))
        bad = pool(jobs, workers, log)
        if bad:
            say(log, "DRL failures: %s" % bad)

    if "plots" in stages:
        say(log, "figures and summary")
        cmd = [PY, os.path.join(LIB, "plots.py"), run_dir]
        r = subprocess.run(cmd, cwd=LIB, env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
        if r.returncode != 0:
            say(log, "plotting failed (exit %s)" % r.returncode)
    say(log, "done: %s" % run_dir)


if __name__ == "__main__":
    main()
