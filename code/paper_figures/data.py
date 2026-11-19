"""Loads every result the paper uses. Single source for numbers and figures."""
import json, glob, os
import numpy as np
import pathlib
U = str(pathlib.Path(__file__).resolve().parents[2])
RUN = U + '/runs/my_new_scenario9_MIQP_DRL'
MQ = U + '/miqp_drl/results/my_new_scenario9_MIQP_DRL'
TG = U + '/mpc_drl_targets/results/my_new_scenario9_MIQP_DRL'
YEARS = [2019, 2020, 2021, 2022, 2023]
SEEDS = [0, 1, 2, 3]

MPCDRL_TRAIN = {0: dict(best_episode=10, val_deficit_pct=25.418), 1: dict(best_episode=10, val_deficit_pct=25.428),
                2: dict(best_episode=10, val_deficit_pct=25.464), 3: dict(best_episode=0, val_deficit_pct=25.544)}

def rj(p): return json.load(open(p))

def kpi(method, y, seed=0):
    if method == 'mpc': return rj(RUN + '/mpc/kpi_%d.json' % y)
    if method in ('il', 'fpppo', 'pdppo', 'focops', 'cpo'):
        return rj(RUN + '/drl/result_%s_%d_s0.json' % (method, y))
    if method in ('miqp', 'miqpdrl'): return rj(MQ + '/drl/result_%s_%d_s0.json' % (method, y))
    if method == 'mpcdrl':
        import csv
        for r in csv.DictReader(open(TG + '/summary.csv')):
            if r['method'].startswith('MPC+') and int(r['year']) == y and int(r['seed']) == seed:
                d = dict(D_L_corrected_pct=float(r['deficit_pct']), D_L_pct=float(r['raw_deficit_pct']),
                         C_curt_pct=float(r['curtail_pct']), kappa_exact_pct=float(r['kappa_exact_pct']),
                         v_min_exact=float(r['v_min']), v_max_exact=float(r['v_max']), loss_MWh=float(r['loss_MWh']))
                d.update(MPCDRL_TRAIN[seed]); return d
        raise KeyError((y, seed))
    raise KeyError(method)

def trace(method, y, seed=0):
    if method == 'mpc': p = RUN + '/mpc/trace_%d.npz' % y
    elif method in ('il', 'fpppo', 'pdppo', 'focops', 'cpo'): p = RUN + '/drl/traces/%s_%d_s0.npz' % (method, y)
    elif method == 'miqp': p = MQ + '/drl/traces/miqp_%d_s0.npz' % y
    elif method == 'mpcdrl': p = TG + '/drl/traces/mpcdrl_%d_s%d.npz' % (y, seed)
    else: raise KeyError(method)
    return dict(np.load(p))

def network(method, y=2020):
    p = {'mpc': RUN + '/mpc/network_%d.npz' % y, 'cpo': RUN + '/drl/network_cpo_2020_s0.npz',
         'fpppo': RUN + '/drl/network_fpppo_2020_s0.npz', 'pdppo': RUN + '/drl/network_pdppo_2020_s0.npz',
         'focops': RUN + '/drl/network_focops_2020_s0.npz',
         'miqp': MQ + '/drl/network_miqp_2020_s0.npz', 'mpcdrl': TG + '/drl/network_mpcdrl_2020_s0.npz'}[method]
    return dict(np.load(p))

def history(method, y=None, seed=0):
    if method in ('fpppo', 'pdppo', 'focops', 'cpo', 'il'): return rj(RUN + '/drl/history_%s_%d_s0.json' % (method, y))
    if method == 'miqpdrl': return rj(MQ + '/drl/history_miqpdrl_%d_s0.json' % y)
    if method == 'mpcdrl': return rj(TG + '/drl/history_mpcdrl_train_s%d.json' % seed)

def values(y, seed): return dict(np.load(TG + '/values/values_%d_s%d.npz' % (y, seed)))

def daily(x):
    x = np.asarray(x, float); n = len(x) // 24
    return x[:n * 24].reshape(n, 24).sum(1)
