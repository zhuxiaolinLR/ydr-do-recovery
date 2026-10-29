# ydr-do-recovery

Analysis code for the manuscript:

> **The false recovery trap in a regulated river: Decoupling between dissolved
> oxygen recovery and oxygen-demand pressure**

Xiaolin Zhu, Yiping Li — College of Environment, Hohai University.

This repository contains the Python pipeline used to (1) classify dissolved-oxygen
(DO) hypoxia events with Random Forest and a logistic-regression baseline under
leave-one-year-out (LOYO) temporal validation, (2) attribute the drivers with SHAP,
and (3) construct the System Fatigue Index (SFI), a parameter-free composite
indicator of residual oxygen-demand pressure during recovery.

All reported numbers in the manuscript are produced with the **B-calibre dataset**
(2021–2025, with 2020 excluded entirely), 12 variables, for the Tanghuan Highway
Bridge (THGLQ) station of the Yongding River, China.

---

## Repository layout

```
code/
  01_rf_sfi_loyo_pipeline.py   Main pipeline: RF regression + classification,
                               LOYO validation, SHAP, SFI, ROC/P90 summary.
                               Writes all intermediate CSVs to outputs/.
  02_lr_shap_tables.py         Logistic-regression baseline + SHAP figures,
                               and renders Table 1 / S2 / S7 / S8 / Fig S5 / S6.
  03_false_recovery_figures.py False-recovery classification and SFI distribution
                               (Fig 4) and per-year LOYO AUC (Fig S4).
data/
  sample/                      De-identified 4-month demo subset (2023-07-01 to
                               2023-10-31, hypoxia season) for smoke-testing the
                               code without the full dataset.
  input/                       Put the full 12-variable table here (see below).
outputs/                       Created at runtime; holds all derived results.
```

## Data availability

The full high-frequency (4-h) monitoring record (DO, water temperature, pH,
electrical conductivity, turbidity, CODMn, NH₃-N, TN, TP, discharge, rainfall)
for the two Yongding River stations (NDHQ, THGLQ; 2020–2025) was obtained from the
National Surface Water Environmental Monitoring Network of China. **The raw records
are third-party national-network data and are therefore not redistributed in this
repository.**

To reproduce the exact reported numbers, obtain the full record from the National
Surface Water Environmental Monitoring Network's real-time data-release platform,
format it as the 12-variable table described below, and place it at
`data/input/THGLQ_12var_2021-2025.xlsx`.

A **de-identified sample subset** (`data/sample/THGLQ_sample_2023-07-01_2023-10-31.csv`,
650 rows, hypoxia season) is included so that the code can be run end-to-end as a
smoke test. Because it covers only four months, it does **not** reproduce the
5-year LOYO statistics in the manuscript; it demonstrates the pipeline and the
input format.

### Input format

Column names may be Chinese or English; both are auto-detected. The expected
columns (and their aliases) are:

| English | Chinese      | Unit / note            |
|---------|--------------|------------------------|
| datetime| 监测时间     | 4-h time stamps        |
| site    | (optional)   | station, e.g. THGLQ    |
| DO      | 溶解氧       | mg/L                   |
| WT      | 水温         | °C                     |
| pH      | pH           | —                      |
| EC      | 电导率       | µS/cm                  |
| turbidity| 浊度        | NTU                    |
| CODMn   | 高锰酸盐指数 | mg/L                   |
| NH3-N   | 氨氮         | mg/L                   |
| TN      | 总氮         | mg/L                   |
| TP      | 总磷         | mg/L                   |
| Q       | 流量         | m³/s                   |
| Rain6h/24h/72h | 累积降水6h/24h/72h | mm          |

## Installation

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate     Linux/macOS:  source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

Run the scripts in order (each is self-contained; `01` must run first because
`02` and `03` read its outputs):

```bash
python code/01_rf_sfi_loyo_pipeline.py   # RF + SFI + LOYO + ROC/P90 -> outputs/
python code/02_lr_shap_tables.py         # LR baseline + SHAP + tables/figures
python code/03_false_recovery_figures.py # Fig 4 + Fig S4
```

`01` takes a few minutes on a laptop. Its output directory is `outputs/`, and it
writes the CSVs `01_processed_timeseries.csv`, `02–07_*`, `08_SFI_overall_ROC.csv`,
`09_SFI_LOYO_ROC.csv`, and `10_SFI_P90_risk_summary.csv`, plus PDP/ROC figures.

## Reproducing the reported numbers

With the full B-calibre dataset in place, `01` reproduces:

- RF regression LOYO R² = 0.663 ± 0.076
- RF hypoxia-classification LOYO AUC = 0.969 ± 0.010
- SFI overall ROC-AUC (24 h / 48 h) = 0.611 / 0.627
- SFI LOYO AUC (24 h / 48 h) = 0.580 ± 0.080 / 0.602 ± 0.083
- SFI P90 = 1.431 (mean 0.616 ± 0.869, median 0.339, range 0–13.96)

`02` reproduces the logistic-regression baseline (LOYO AUC = 0.967 ± 0.018) and the
SHAP feature importances (pH and water temperature dominate, mean |SHAP| = 0.206
and 0.122), and renders Tables 1 / S2 / S7 / S8 and Figs S5 / S6.

`03` reproduces the false-recovery classification (Fig 4) and the per-year SFI
LOYO AUC (Fig S4).

## License

MIT — see `LICENSE`. The monitoring data are © their original data provider and are
not covered by this license.
