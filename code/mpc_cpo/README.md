# MPC + DRL storage values (option A)

The DRL agent does not replace the MPC and does not correct its schedule. Once a day, at 00:00, it
sets the **two numbers the 24 h MPC cannot work out for itself**: how much a MWh left in the
hydrogen tank, and in the battery, is worth at the end of the day. The MPC then plans the day
with those values, with every voltage, power-flow and converter constraint unchanged.

## Why
Her MPC2 values whatever is left in storage at the end of its 24 h window with fixed constants
(hydrogen: 1.5 below the 30 % reserve, 0.5 above it, per MWh the tank returns; battery: 0.5 per
MWh). Those are the same in January and July, before a calm week and before a windy one. They
decide how eagerly the MPC fills the tank from surplus, how readily it burns hydrogen, and how much
charge it carries into the next day. That is the one part of the MPC that is a guess, and the one
part a learner with a year-long view can improve.

## Action, reward, plant
- action, per day: `w_H = 3**a0`, `w_B = 3**a1`, with `a` in [-1, 1]; multipliers of the two
  values, from 1/3 to 3. `a = (0, 0)` is its MPC exactly; the agent starts there (zero-initialised
  last layer) and the best validation checkpoint is kept, so episode 0 (plain MPC) is a candidate.
- reward, per day: `-(unserved MWh) + E_HD * change in tank level`. Summed over a year it equals
  minus the corrected deficit the paper reports, so the agent is trained on the paper's indicator.
- cost (constrained methods): share of the day's hours with a bus outside 0.95-1.05 pu.
- plant: the same day-ahead closed loop as the main run's MPC rows (its THESSEnv, Dispatch with
  minimum load, VBand, no-shed-charging). Checked: the plain MPC through this code gives the main
  run's deficit to the 4th decimal (2020: 30.4049 % both).
- observation (17): battery SoC, tank level, distance to reserve, season (sin/cos day of year),
  the day's 24 h forecast (load, PV, wind, forecast deficit and surplus), realised surplus and
  deficit over the past 7 and 30 days, fraction of the year left, last battery/hydrogen powers.

## Run (from the v3 folder, after the main run of the same RUN_NAME has finished)
```
python mpc_drl_targets\run_targets_drl.py
```
It reads its own `mpc_drl_targets\config.py` (falls back to `v3\config.py` if that file is
missing; `--config other.py` picks any other). In that file `RUN_NAME` must be a finished main
run and the plant must match it; the settings for this run are in the TARGETS_DRL_ section:
`TARGETS_DRL_ALGO = "auto"` (best DRL method of the main run), `TARGETS_DRL_EPISODES = 60`,
`TARGETS_DRL_SEEDS = [0, 1, 2, 3]`, `TARGETS_DRL_LOG_STD = -1.0`, `TARGETS_DRL_GAMMA = 0.99`. One job per seed; with 4 workers the 4 seeds run side by side,
about 35-45 min in total. Re-running skips finished seeds (`--force` redoes them,
`--stages plots` only redraws).

## Output: `mpc_drl_targets\results\<RUN_NAME>\`
- `summary.md` / `summary.csv`: MPC, MPC-imitation, best DRL, MPC+DRL values
- `figures\mpc_drl_values\mpc_vs_mpcdrl.md`: per test year, MPC vs MPC+DRL (mean, std and best
  over seeds, difference, how many seeds beat MPC) and the MPC check
- `figures\mpc_drl_values\learned_values_<PLOT_YEAR>.png`: the values the agent chose each day
  and the tank level that resulted, against the plain MPC
- `figures\...`: every figure of the main pipeline with the new method included
- `logs\train_s<seed>.log`: training progress per seed (validation deficit every 5 episodes)

## What to expect
Fixed values already move the deficit: on 2017 the plain MPC gives 21.627 %, battery value 1/3
gives 21.543 %, hydrogen value x1.7 gives 21.598 %, and hoarding (both x3) gives 22.720 %. A
policy that adapts the values to season and weather can do better than any fixed choice, but the
room is tenths of a point: in 2020 the tank sits at its reserve 57 % of the year and most spilled
energy came when the electrolyser was already at full power.
