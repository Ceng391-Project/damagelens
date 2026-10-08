# Extreme heat — summer 2023 heatwaves (MODIS surface temperature vs stations)

## Summary
- **10 cities** (Phoenix, Las Vegas, El Paso, Seville, Rome, Palermo, Athens, Antalya, Adana, Beijing), June–August 2023; calibration uses summer 2022 only.
- Daytime satellite land-surface temperature (LST) vs station daily maximum air temperature: mean **r = 0.65**; linear calibration error **3.0 °C** RMSE on average.
- **Extreme-heat day detection** (station Tmax ≥ 2013–2022 summer P90): ranking is good (**pooled AUC 0.82**), but the thresholding rule decides the result: linear calibration F1 0.20 (estimates shrink towards the mean), quantile matching F1 **0.50** (precision 0.70, recall 0.40).
- The Mediterranean LST anomaly map clearly shows the July 2023 "Cerberus" heatwave over Sicily, Sardinia, Greece, North Africa and around Adana (land mean +0.4 °C).
- This is a single-date measurement, not pre/post damage detection; it is outside the #30 change-detection pipeline.

## Data and ground truth
| data | platform | resolution |
|---|---|---|
| MODIS Terra MOD11A1 daily daytime LST — Planetary Computer | satellite | 1 km |
| MODIS Terra MOD11A2 8-day LST (anomaly map) | satellite | 1 km → 0.05° |
| Meteostat (NOAA ISD/DWD stations) daily Tmax, 2013–2023 | ground station | point |

## Results — per city (test year 2023)
| city | station | days (clear) | extreme days | r | RMSE °C | AUC | F1 linear | F1 quantile |
|---|---|---|---|---|---|---|---|---|
| Phoenix | Phoenix Sky Harbor Airport | 77 | 34 | 0.573 | 3.694 | 0.814 | 0.000 | 0.542 |
| Las Vegas | McCarran International Airport | 78 | 16 | 0.790 | 2.834 | 0.893 | 0.000 | 0.476 |
| El Paso | El Paso International Airport | 75 | 32 | 0.733 | 2.884 | 0.884 | 0.000 | 0.512 |
| Seville | Sevilla / San Pablo | 78 | 20 | 0.637 | 2.618 | 0.771 | 0.429 | 0.476 |
| Rome | Roma Fiumicino | 72 | 21 | 0.468 | 2.995 | 0.653 | 0.000 | 0.333 |
| Palermo | Palermo / Punta Raisi | 75 | 12 | 0.544 | 3.377 | 0.753 | 0.000 | 0.125 |
| Athens | El Venizelos / Liádha | 81 | 20 | 0.689 | 2.587 | 0.892 | 0.462 | 0.622 |
| Antalya | Antalya | 82 | 14 | 0.868 | 3.105 | 0.963 | 0.000 | 0.250 |
| Adana | Adana / Sakirpasa | 66 | 27 | 0.585 | 2.890 | 0.800 | 0.000 | 0.432 |
| Beijing | Beijing | 46 | 24 | 0.651 | 2.777 | 0.854 | 0.684 | 0.766 |

![LST–Tmax relation, extreme-day ROC and per-city performance](img/heat_metrics.png)
*LST–Tmax relation, extreme-day ROC and per-city performance*

![Summer 2023 time series: station Tmax and satellite estimate](img/heat_timeseries.png)
*Summer 2023 time series: station Tmax and satellite estimate*

![12–19 July 2023 LST anomaly (vs the 2018–2022 mean)](img/heat_anomaly_map.png)
*12–19 July 2023 LST anomaly (vs the 2018–2022 mean)*


## Limitations
- LST is surface, not air temperature; urban surfaces and bare soil can be 10–20 °C hotter during the day.
- No measurement on cloudy days (see the days column); in Beijing the monsoon removes about half of the summer days.
- 2023 was clearly hotter than 2022: thresholds calibrated on the 2022 rate under-predict the number of extreme days in 2023 (low recall).

## Sources
- MODIS LST (Wan et al.), Microsoft Planetary Computer `modis-11A1-061`, `modis-11A2-061`
- Meteostat: https://meteostat.net
