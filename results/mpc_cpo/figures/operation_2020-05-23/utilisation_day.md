# Utilisation on 2020-05-23 (24 h day-ahead operation)

| method | PV used, % of available | wind used, % of available | renewables used, % of available | demand from PV, % | demand from wind, % | demand from battery, % | demand from fuel cell, % | demand unserved, % | renewable energy to electrolyser, % | renewable energy to battery, % | electrolyser utilisation, % of rating x hours | fuel cell utilisation, % of rating x hours | electrolyser running hours, % | fuel cell running hours, % | battery full cycles | planned_el_MWh | el_MWh | planned_fc_MWh | fc_MWh | planned_bess_dis_MWh | bess_dis_MWh | curtailed_MWh | unserved_MWh |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MPC | 98.78 | 88.79 | 90.18 | 15.11 | 80.93 | 3.89 | 0.00 | 0.08 | 22.07 | 9.14 | 66.69 | 0.00 | 91.67 | 0.00 | 0.34 | 9.36 | 16.00 | 0.00 | 0.00 | 0.00 | 2.02 | 7.89 | 0.04 |
| MPC-imitation | 90.79 | 86.78 | 87.34 | 14.92 | 80.97 | 4.11 | 0.00 | 0.00 | 19.47 | 9.61 | 56.97 | 0.00 | 79.17 | 0.00 | 0.36 | 9.56 | 13.67 | 0.00 | 0.00 | 0.25 | 2.13 | 10.18 | 0.00 |
| CPO | 90.83 | 86.77 | 87.34 | 14.94 | 80.97 | 4.09 | 0.00 | 0.00 | 19.48 | 9.60 | 56.98 | 0.00 | 79.17 | 0.00 | 0.35 | 9.62 | 13.68 | 0.00 | 0.00 | 0.26 | 2.12 | 10.18 | 0.00 |
| MPC+CPO values | 98.77 | 88.79 | 90.18 | 15.10 | 80.93 | 3.89 | 0.00 | 0.08 | 22.07 | 9.15 | 66.68 | 0.00 | 91.67 | 0.00 | 0.34 | 9.36 | 16.00 | 0.00 | 0.00 | 0.00 | 2.02 | 7.89 | 0.04 |
