# -*- coding: utf-8 -*-
"""Shared machinery for the v3 pipeline: paths, config loading, and applying the config to
the core pipeline's plant model *before* its controller is imported.

Why order matters. `mpc2` (its controller) computes every rating, the battery energy, the
tank energy, the voltage sensitivities and the device buses once, at import time, from the
plant module. So sizes, locations and capacitors are patched onto the plant module first, and
only then is anything that imports `mpc2` loaded.

Why dataclass defaults are patched and not class attributes. `BESSParams` and `HFCParams` are
dataclasses. Assigning `HFCParams.P_FC_max = x` (what 00_config/plant_sizes.py does) changes
the class attribute but NOT the default the generated __init__ uses, so `HFCParams()` still
returns the delivered 1 MW. This module rewrites the __init__ defaults instead, and checks it.
"""
import importlib.util
import inspect
import io
import json
import os
import sys

import numpy as np

LIB = os.path.dirname(os.path.abspath(__file__))
V3 = os.path.dirname(LIB)
TREE = os.path.join(V3, "_tree")
PAPER = os.path.join(TREE, "paper", "main", "journal")
J = os.path.join(TREE, "Journal_")
DELIVERED_FORECASTS = os.path.join(J, "Fix", "forecasts")

# What the forecast files on disk were produced for (plant B, as delivered).
FORECAST_PLANT = {"pv_scale": 1.0, "wind_scale": 1.2379, "load_scale": 1.0}
DELIVERED = {"A_PV_m2": 10160.0, "PV_MWp": 0.18 * 10160.0 / 1000.0, "WIND_MW": 3.0,
             "C_rated_Ah": 8000.0, "BESS_usable_MWh": None}


# ───────────────────────────────────────────────────────────── config
def load_config(path):
    spec = importlib.util.spec_from_file_location("v3_config", path)
    cfg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg)
    return cfg


def config_dict(cfg):
    out = {}
    for k in dir(cfg):
        if k.isupper():
            v = getattr(cfg, k)
            if isinstance(v, dict):
                v = {str(a): b for a, b in v.items()}
            if isinstance(v, (tuple, range)):
                v = list(v)
            out[k] = v
    return out


def scales(cfg):
    """Plant multipliers relative to the delivered plant model (PV 1.83 MWp, wind 3.0 MW)."""
    return {"pv_scale": float(cfg.PV_MWP) / DELIVERED["PV_MWp"],
            "wind_scale": float(cfg.WIND_MW) / DELIVERED["WIND_MW"],
            "load_scale": float(cfg.LOAD_SCALE)}


def forecast_factors(cfg):
    s = scales(cfg)
    return {"load": s["load_scale"] / FORECAST_PLANT["load_scale"],
            "pv": s["pv_scale"] / FORECAST_PLANT["pv_scale"],
            "wind": s["wind_scale"] / FORECAST_PLANT["wind_scale"]}


def forecast_dir(run_dir, cfg):
    f = forecast_factors(cfg)
    if all(abs(v - 1.0) < 1e-12 for v in f.values()):
        return DELIVERED_FORECASTS
    return os.path.join(run_dir, "forecasts")


def prepare_forecasts(run_dir, cfg, years):
    """Write rescaled copies of the forecasts when ratings or load differ from plant B.

    The forecaster normalises every channel by its rating, so a resized plant's forecast is the
    delivered forecast times the rating ratio (the core pipeline measured this to float32 rounding,
    Fix/code/forecast_equivariance.py). No retraining is needed.
    """
    d = forecast_dir(run_dir, cfg)
    if d == DELIVERED_FORECASTS:
        return d
    os.makedirs(d, exist_ok=True)
    f = forecast_factors(cfg)
    k = np.array([f["load"], f["pv"], f["wind"]], np.float32)
    for y in years:
        dst = os.path.join(d, "forecast_residual_%d.npy" % y)
        if os.path.exists(dst):
            continue
        a = np.load(os.path.join(DELIVERED_FORECASTS, "forecast_residual_%d.npy" % y))
        np.save(dst, (a * k).astype(np.float32))
    return d


# ───────────────────────────────────────────────────────────── patch helpers
def set_defaults(func, **kw):
    """Rewrite the default values of named keyword parameters of a function."""
    params = [p for p in inspect.signature(func).parameters.values()
              if p.kind == p.POSITIONAL_OR_KEYWORD and p.default is not p.empty]
    names = [p.name for p in params]
    d = list(func.__defaults__)
    off = len(d) - len(names)
    for k, v in kw.items():
        d[off + names.index(k)] = v
    func.__defaults__ = tuple(d)


def c_rated_for(m, usable_mwh, soc_min=0.20, soc_max=0.90):
    """Battery capacity (Ah) that gives the requested usable energy, as mpc2 computes it."""
    per_ah = m.BESS(m.BESSParams()).v_oc(0.6) * 3600 / 3.6e9          # MWh per Ah
    return float(usable_mwh) / ((soc_max - soc_min) * per_ah)


_BOOT = {}


def bootstrap(cfg, run_dir, hours=None):
    """Apply the config to the plant, then import the controller stack. Idempotent."""
    if _BOOT:
        return _BOOT
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    sys.dont_write_bytecode = True
    for p in (os.path.join(J, "DRL", "code"), os.path.join(J, "Network", "code"),
              os.path.join(J, "Grid_connection", "code"),
              os.path.join(PAPER, "Stage 3", "code"), os.path.join(PAPER, "Stage 2", "code"),
              os.path.join(PAPER, "Stage 1", "code"), os.path.join(J, "Fix", "code"), J):
        if p not in sys.path:
            sys.path.insert(0, p)

    import weather_real as W
    m = W.m

    # 1. ratings, as dataclass defaults so every later instantiation sees them
    c_rated = c_rated_for(m, cfg.BESS_USABLE_MWH)
    set_defaults(m.BESSParams.__init__, C_rated=c_rated, P_max=float(cfg.BESS_P_MW) * 1e6)
    set_defaults(m.HFCParams.__init__, P_FC_max=float(cfg.FC_MW) * 1e6,
                 P_elec_max=float(cfg.EL_MW) * 1e6, V_tank=float(cfg.TANK_M3))
    h = m.HFCParams()
    assert abs(h.P_FC_max - cfg.FC_MW * 1e6) < 1e-6 and abs(h.V_tank - cfg.TANK_M3) < 1e-9

    # 2. where everything sits on the feeder, and the fixed capacitors
    bad = [k for k in cfg.LAYOUT if k not in m.LAYOUT]
    if bad:
        raise SystemExit("config LAYOUT: unknown key(s) %s, use %s" % (bad, sorted(m.LAYOUT)))
    for k, v in cfg.LAYOUT.items():
        if not 2 <= int(v) <= 33:
            raise SystemExit("config LAYOUT: %s at bus %s, must be 2..33" % (k, v))
        m.LAYOUT[k] = int(v)
    m.CAPACITORS_MVAR = {int(b): float(q) for b, q in cfg.CAPACITORS_MVAR.items()}

    # 3. demand level, on the weather builder every environment reads
    ls = float(cfg.LOAD_SCALE)
    if abs(ls - 1.0) > 1e-12:
        orig_build = W.build

        def build_load_scaled(year):
            d = dict(orig_build(year))
            d["load"] = d["load"] * ls
            return d
        W.build = build_load_scaled

    # 4. the core pipeline's settings: plant split, guard band, forecast location
    import settings
    sc = scales(cfg)
    settings.PLANT_B = True
    settings.PLANT_SPLIT = (sc["pv_scale"], sc["wind_scale"])
    settings.GUARD_BAND_PU = float(cfg.GUARD_BAND_PU)
    set_defaults(settings.apply, margin=float(cfg.GUARD_BAND_PU))
    fdir = forecast_dir(run_dir, cfg)
    settings.forecast_dir = lambda: fdir

    # 5. the controller stack, now that the plant is final
    import mpc2 as M2
    import run_stage3 as S3
    import indicators as IND
    set_defaults(S3.set_bess, c_rated=c_rated, p_max=float(cfg.BESS_P_MW) * 1e6)
    settings.apply()

    def _fc(year, mode="v3"):
        return np.load(os.path.join(fdir, "forecast_residual_%d.npy" % year))
    S3.FC.forecasts = _fc

    import network_profiles as NP                       # applies the scoring corrections
    _BOOT.update(W=W, m=m, M2=M2, S3=S3, IND=IND, NP=NP, settings=settings,
                 c_rated=c_rated, forecast_dir=fdir, forecasts=_fc)
    return _BOOT


def plant_summary(b):
    """The plant actually in force, read back from the live modules (not from the config)."""
    m, M2, W = b["m"], b["M2"], b["W"]
    return {"PV_MWp": round(0.18 * float(W.A_PV) / 1000.0, 4),
            "WIND_MW": round(float(W.P_RATED) / 1e6, 4),
            "BESS_P_MW": M2.P_B, "BESS_E_full_MWh": round(M2.E_B, 3),
            "BESS_usable_MWh": round(M2.E_B * (M2.SOC_MAX - M2.SOC_MIN), 3),
            "FC_MW": M2.P_FC, "EL_MW": M2.P_EL,
            "tank_return_MWh": round(M2.E_HD, 2), "tank_fill_MWh": round(M2.E_HC, 2),
            "layout": dict(m.LAYOUT), "capacitors_Mvar": dict(m.CAPACITORS_MVAR),
            "guard_band_pu": M2.V_MARGIN}


def dump_json(obj, path):
    def conv(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)
    with io.open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=1, default=conv)


def read_json(path):
    with io.open(path, encoding="utf-8") as fh:
        return json.load(fh)
