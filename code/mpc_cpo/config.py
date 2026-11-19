# -*- coding: utf-8 -*-
r"""
==========================================================================================
  MPC + DRL STORAGE VALUES (option A)  -  edit this file, then run from the v3 folder:
        python mpc_drl_targets\run_targets_drl.py
==========================================================================================
This run uses this file (not v3\config.py). Results go to
    mpc_drl_targets\results\<RUN_NAME>\

IMPORTANT
  * RUN_NAME must be the name of a FINISHED main run (runs\<RUN_NAME>\): the MPC rows, the
    MPC-imitation and the best DRL method are taken from it for the comparison.
  * The plant (sizes, layout, capacitors, guard band, load) must be the SAME as that main run,
    otherwise MPC and MPC+DRL are compared on different plants. The values below are copied
    from my_new_scenario9_MIQP_DRL. Change them only together with RUN_NAME.
  * The settings meant to be tuned for this run are in the last section (TARGETS_DRL_*).
"""

# ---------------------------------------------------------------- run identity
RUN_NAME = "my_new_scenario9_MIQP_DRL"   # the finished main run this one builds on

# ---------------------------------------------------------------- plant (same as the main run)
PV_MWP = 1.8
WIND_MW = 4
LOAD_SCALE = 1.0
BESS_P_MW = 1.2
BESS_USABLE_MWH = 6
FC_MW = 1.0
EL_MW = 1.0
TANK_M3 = 500.0
S_CONV_MVA = 1.0
LAYOUT = {"PV": 33, "WIND": 25, "BESS": 22, "HFC": 6}
CAPACITORS_MVAR = {18: 0.4, 33: 0.3}
GUARD_BAND_PU = 0.012
HORIZON = 24
MPC_MODE = "day_ahead"
RT_NO_SHED_CHARGING = True

# ---------------------------------------------------------------- years and machine
MPC_YEARS = [2019, 2020, 2021, 2022, 2023]
DRL_TEST_YEARS = [2019, 2020, 2021, 2022, 2023]   # held-out years the trained agent is scored on
WORKERS = 4                          # parallel jobs = seeds trained side by side (<= CPU cores - 1)
DRL_SEEDS = [0]                      # (main-run setting, not used here; see TARGETS_DRL_SEEDS)
DRL_EVAL_EVERY = 5                   # validate on 2017-2018 every N episodes
DRL_COST_LIMIT = 0.00376             # constrained methods: allowed share of hours outside 0.95-1.05

# ================================================================ THIS RUN: the DRL agent
TARGETS_DRL_ALGO = "auto"            # "auto" = best DRL method of the main run (CPO for scenario 9)
                                     # or one of "fpppo", "pdppo", "focops", "cpo"
TARGETS_DRL_EPISODES = 60            # training episodes, one training year (2005-2016) each,
                                     # about 25-30 s per episode. More = longer, maybe better.
TARGETS_DRL_SEEDS = [0, 1, 2, 3]     # one training job per seed; report mean and spread
TARGETS_DRL_LOG_STD = -1.0           # exploration noise on the two values (log std, action units)
                                     # -1.0 -> std 0.37 -> multipliers explored ~0.67x..1.5x
                                     # lower (e.g. -1.6) = stays closer to the plain MPC
TARGETS_DRL_GAMMA = 0.99             # discount per day: 0.99 looks ~100 days ahead (seasonal)
# TARGETS_DRL_LR = 3e-4              # step size for fpppo / pdppo / focops (uncomment to change)
# TARGETS_DRL_DELTA = 0.01           # trust region for cpo (uncomment to change)

# ---------------------------------------------------------------- figures
PLOT_YEAR = 2020                     # year of the learned-values and 24 h operation figures
OPERATION_DATE = None                # day of the 24 h operation figures (None = automatic)
FIG_FORMATS = ["png", "pdf"]
