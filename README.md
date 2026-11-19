# terminal-value-mpc-cdrl

Supporting material for the manuscript

> **Learning Terminal Storage Values for Model Predictive Control of a Hydrogen–Battery Microgrid**

The repository lets readers check the reported results: the scripts written for this study, the
run configurations, the weather input, and the result files behind every table and figure of the
paper (annual indicators, training histories, trained policy weights and run logs).

## Contents

```
code/
  run.py, config.py, compare.py      pipeline driver and run configuration (MPC and Case 1 CDRL policies)
  lib/                               day-ahead wrappers, training stages, network validation, plots
  configs/                           the plant/run configurations used during the study
  mpc_cpo/                           the proposed MPC-CPO (CPO agent that sets the MPC terminal values)
  paper_figures/                     scripts that draw the data figures of the manuscript
  requirements.txt
data/weather/
  pvgis_hurghada_2005_2023.csv       hourly PVGIS-SARAH3 data, Hurghada, Egypt (27.18 N, 33.80 E), 2005-2023
results/
  mpc_and_cdrl/                      MPC and the four Case 1 CDRL policies (FP-PPO, PD-PPO, FOCOPS, CPO)
  mpc_cpo/                           Case 2: MPC-CPO, four seeds
```

Each results folder holds `summary.md` / `summary.csv` (five-year indicators), `config_used.py`
(the exact configuration of the run), `mpc/` and `drl/` (per-year result and training-history JSON
files and the trained network weights, `.pt`), `values/` (checks that MPC-CPO with a zero action
reproduces the MPC), `figures/` (PDF figures and the CSV/Markdown tables behind them) and `logs/`. The internal run name `my_new_scenario9_MIQP_DRL` appears in some files; it is only a
folder label.

## Where the paper's numbers are

| Paper | File |
|---|---|
| Table 6 (online performance, 2019–2023) | `results/mpc_and_cdrl/summary.md`, `results/mpc_cpo/summary.md` (MPC-CPO is listed as "MPC+CPO values") |
| Table 5, Figure 8 (Case 1 training) | `results/mpc_and_cdrl/drl/history_*_s0.json` |
| Table 7, Figures 10–11 (Case 2) | `results/mpc_cpo/drl/history_mpcdrl_train_s*.json`, `results/mpc_cpo/drl/result_mpcdrl_*.json` |
| Figure 13 (day-by-day comparison) | `results/*/figures/drl/daily_paired_vs_mpc.csv` |
| Table 8 (plant utilization) | `results/*/figures/utilisation/utilisation_summary.csv` |
| Plant and run settings (Tables 2–4) | `results/*/config_used.py` |

## What is not included

* **Core simulation code.** The plant model, the SOCP-MPC controller, the real-time layer, the CDRL
  agents and the exact DistFlow scorer that the scripts in `code/` call are not part of this
  repository; they are available from the corresponding author on reasonable request. The scripts
  here therefore document the procedure but do not run on their own.
* **Hourly traces and forecasts** (`.npz`, `.npy`, about 400 MB) are omitted for size; they are needed
  to redraw the hourly figures and are available on request.

## Data source and terms

The weather data are from the Photovoltaic Geographical Information System (PVGIS) of the European
Commission Joint Research Centre, https://re.jrc.ec.europa.eu/pvg_tools/en/ (PVGIS © European Union,
2001–2026). Reuse is authorised provided the source is acknowledged.

No licence has been chosen yet for the code and results in this repository; until one is added, all
rights are reserved by the authors.
