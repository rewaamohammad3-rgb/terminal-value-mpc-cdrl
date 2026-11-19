# Day-by-day comparison with the MPC (unserved energy per day)

| method | days | days DRL better (less unserved) | days DRL worse | days equal (within 1 kWh) | mean daily difference, MWh (DRL - MPC) | largest DRL gain, MWh | largest DRL loss, MWh |
|---|---|---|---|---|---|---|---|
| MPC-imitation | 1826 | 478 | 707 | 641 | 0.17 | 11.58 | 17.36 |
| FP-PPO | 1826 | 469 | 709 | 648 | 0.18 | 8.13 | 17.36 |
| PD-PPO | 1826 | 470 | 711 | 645 | 0.18 | 8.14 | 17.36 |
| FOCOPS | 1826 | 487 | 687 | 652 | 0.18 | 11.43 | 17.51 |
| CPO | 1826 | 497 | 682 | 647 | 0.17 | 11.59 | 17.36 |
