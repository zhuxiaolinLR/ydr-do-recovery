# data/input

Place the full 12-variable table here as `THGLQ_12var_2021-2025.xlsx` before
running `01_rf_sfi_loyo_pipeline.py`.

The full high-frequency (4-h) monitoring record is third-party data from the
National Surface Water Environmental Monitoring Network of China and is **not**
included in this repository. Obtain it from the network's real-time data-release
platform and format the columns as described in the top-level `README.md`
(`datetime`, `site`, `DO`, `WT`, `pH`, `EC`, `turbidity`, `CODMn`, `NH3-N`,
`TN`, `TP`, `Q`, `Rain6h`, `Rain24h`, `Rain72h`).

A de-identified smoke-test subset is available at `../sample/`.
