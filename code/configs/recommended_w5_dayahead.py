# -*- coding: utf-8 -*-
"""
==========================================================================================
  v3 PIPELINE CONFIGURATION  -  edit this file, then run:   python run.py
==========================================================================================
Every run is saved under  runs/<RUN_NAME>/  (results, logs, figures, a copy of this file).
Give each scenario its own RUN_NAME so runs never overwrite each other. Re-running the same
RUN_NAME resumes: finished years / DRL cells are skipped (use --force to redo them).

Delivered values (the core pipeline's plant B, as in the v2 manuscript) are shown on the right.
"""

# ---------------------------------------------------------------- run identity
RUN_NAME = "recommended_w5_dayahead"

# ---------------------------------------------------------------- generation
PV_MWP = 1.8288                      # PV peak, MWp                      delivered 1.8288
WIND_MW = 5.0                     # wind turbine rating, MW           delivered 3.7137 (plant B)
                                     #                                   (plant A was 3.0)

# ---------------------------------------------------------------- demand
LOAD_SCALE = 1.0                     # multiplies the whole feeder demand delivered 1.0
                                     # (IEEE-33 nominal 3.715 MW; modelled peak ~3.08 MW)

# ---------------------------------------------------------------- battery
BESS_P_MW = 1.2                      # charge/discharge power, MW        delivered 1.2
BESS_USABLE_MWH = 6.3891             # usable energy (SoC 0.2-0.9), MWh  delivered 6.3891

# ---------------------------------------------------------------- hydrogen
FC_MW = 1.0                          # fuel cell power, MW               delivered 1.0
EL_MW = 1.0                          # electrolyser power, MW            delivered 1.0
TANK_M3 = 500.0                      # tank volume, m3 (5-350 bar)       delivered 500 (~281 MWh back)
S_CONV_MVA = 1.0                     # hydrogen converter rating, MVA    delivered 1.0

# ---------------------------------------------------------------- locations (bus 2..33)
# Keys are the plant model's own names. HFC = the hydrogen unit (electrolyser + fuel cell).
LAYOUT = {"PV": 33, "WIND": 25, "BESS": 22, "HFC": 6}
                                     # delivered {"PV": 33, "WIND": 25, "BESS": 22, "HFC": 18}

# ---------------------------------------------------------------- fixed capacitors {bus: Mvar}
CAPACITORS_MVAR = {18: 0.4, 33: 0.3}  # delivered {18: 0.4, 33: 0.6}

# ---------------------------------------------------------------- controller
MPC_MODE = "day_ahead"               # "day_ahead": one 24 h schedule committed at 00:00 each day,
                                     #   real-time layer absorbs forecast errors (no intraday re-plan)
                                     # "rolling": its original, re-plans every hour on a 24 h window
GUARD_BAND_PU = 0.012                # planning band 0.95+g .. 1.05-g    delivered 0.012
HORIZON = 24                         # MPC look-ahead, h (forecasts supply 24) delivered 24

# ---------------------------------------------------------------- what to run
MPC_YEARS = [2019, 2020, 2021, 2022, 2023]   # the five held-out test years
WORKERS = 4                          # parallel processes (years / DRL cells). <= CPU cores - 1

# Deep RL (the core pipeline's four constrained agents; train 2005-2016, select the checkpoint on
# 2017-2018, test once on the held-out year)
DRL_ENABLE = True
DRL_MODE = "day_ahead"               # "day_ahead": at 00:00 the agent outputs the whole 24 h
                                     #   schedule (72 actions) from state + 24 h forecast;
                                     #   same real-time layer as the MPC executes it
                                     # "hourly": its original, one action per hour
DRL_DEMO_YEARS = [2013, 2014, 2015, 2016]  # years of day-ahead MPC results the agent learns
                                     #   from (must be inside the training years 2005-2016)
DRL_MPC_HINT = True                  # an "MPC advisor" network, trained on those years, predicts
                                     #   the MPC's 24 h schedule; it is added to the agent's
                                     #   observation every day (no MPC solve needed at run time)
DRL_IMITATION = True                 # start the agent from a copy of the MPC (behaviour cloning
                                     #   on DRL_DEMO_YEARS), then fine-tune with RL on 2005-2016
DRL_BC_EPOCHS = 300                  # imitation training passes over the MPC schedules
DRL_BC_LOG_STD = -2.3                # exploration noise after imitation (lower = stay near MPC)
DRL_FT_LR = 2e-5                     # RL fine-tuning step size (FP-PPO, PD-PPO, FOCOPS)
DRL_CPO_DELTA = 0.002                # RL fine-tuning trust region (CPO)
DRL_CRITIC_WARMUP = 5                # years of critic-only training before the policy moves
DRL_SELECT_EP0 = False               # False: each algorithm is reported on its own fine-tuned
                                     #   policy; the imitation policy is reported separately as
                                     #   "MPC-imitation". True: imitation may be selected too.
DRL_EVAL_EVERY = 5                   # validate on 2017-2018 every N episodes
DRL_ALGOS = ["fpppo", "pdppo", "focops", "cpo"]   # any subset of these four
DRL_TEST_YEARS = [2019, 2020, 2021, 2022, 2023]
DRL_SEEDS = [0]                      # paper used 5 seeds; 1 seed = 20 runs instead of 100
DRL_EPISODES = 60                    # RL fine-tuning episodes after imitation (each = one
                                     # training year, 365 day-ahead decisions). ~0.3 min each.
                                     # (hourly mode: paper used 250)
DRL_RESIDUAL = False                 # True: agent corrects the MPC instead of replacing it
DRL_COST_LIMIT = 0.00376             # allowed fraction of hours outside 0.95-1.05 pu for the
                                     # constrained agents (paper: the MPC's own 2020 rate, 0.00376)

# ---------------------------------------------------------------- figures
PLOT_YEAR = 2020                     # year used for the single-year figures
OPERATION_DATE = None                # day for the 24 h operation figures, e.g. "2023-07-15"
                                     #   (must be in a test year); None = the day with the
                                     #   widest voltage swing in PLOT_YEAR
PLOT_WEEK_START_DAY = 90             # first day of the one-week dispatch figure
FIG_FORMATS = ["png", "pdf"]
