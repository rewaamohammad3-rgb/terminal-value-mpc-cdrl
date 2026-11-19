# my_new_scenario9_MIQP_DRL

Plant in force: {"PV_MWp": 1.8, "WIND_MW": 4.0, "BESS_P_MW": 1.2, "BESS_E_full_MWh": 8.571, "BESS_usable_MWh": 6.0, "FC_MW": 1.0, "EL_MW": 1.0, "tank_return_MWh": 280.73, "tank_fill_MWh": 737.17, "layout": {"BESS": 22, "PV": 33, "SC": 33, "WIND": 25, "HFC": 6}, "capacitors_Mvar": {"18": 0.4, "33": 0.3}, "guard_band_pu": 0.012}

| method | years | deficit % | exact compliance % | worst-year compliance % |
|---|---|---|---|---|
| MPC | 5 | 28.169 | 100.00 | 100.00 |
| CPO | 5 | 28.501 | 100.00 | 99.99 |
| FOCOPS | 5 | 28.509 | 100.00 | 99.99 |
| FP-PPO | 5 | 28.516 | 100.00 | 99.99 |
| MPC-imitation | 5 | 28.500 | 100.00 | 99.99 |
| PD-PPO | 5 | 28.517 | 100.00 | 99.99 |
