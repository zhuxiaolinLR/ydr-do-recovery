# -*- coding: utf-8 -*-
"""
B口径 (2020 excluded entirely, 2021-2025) figure/table regeneration for manuscript resubmission.

Regenerates the 6 model-result figures/tables with B口径 12-variable numbers:
  - Figures/Table1_LOYO_Performance (.csv/.xlsx/.png/.tiff)
  - Supplementary Tables/TableS2_Model_Performance_Variable_Importance
  - Supplementary Tables/TableS7_RF_LR_LOYO_AUC_Comparison
  - Supplementary Tables/TableS8_SHAP_Feature_Importance
  - Supplementary Charts/FigS5_SHAP_beeswarm
  - Supplementary Charts/FigS6_SHAP_importance_bar

Sources of truth (LOCKED):
  - RF regression LOYO       -> outputs/02_RF_regression_LOYO.csv
  - RF classification LOYO    -> .../05_RF_classification_LOYO.csv
  - regression importance     -> .../04_RF_regression_importance_summary.csv
  - classification importance -> .../07_RF_classification_importance_summary.csv
  - LR LOYO + SHAP + 10-fold  -> recomputed here (reproduces b_run_full.py exactly)
"""
import os, sys, io, shutil, csv, datetime, warnings
warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 仓库根
DATA_FILE = os.path.join(REPO, "data", "input", "THGLQ_12var_2021-2025.xlsx")  # 完整数据（未分发）
AUTH_DIR  = os.path.join(REPO, "outputs")   # 01_rf_sfi_loyo_pipeline.py 的输出
OUT       = os.path.join(REPO, "outputs")

FIG_DIR   = os.path.join(OUT, "Figures")
STAB_DIR  = os.path.join(OUT, "Supplementary Tables")
SCHART_DIR= os.path.join(OUT, "Supplementary Charts")

MIR_FIG   = FIG_DIR
MIR_STAB  = STAB_DIR
MIR_SCHART= SCHART_DIR

COL_MAP = {"监测时间":"datetime","水温":"WT","pH":"pH","溶解氧":"DO","高锰酸盐指数":"CODMn",
           "氨氮":"NH3N","总磷":"TP","总氮":"TN","电导率":"EC","浊度":"Turbidity",
           "流量":"Q","累积降水6h":"Rain6h","累积降水24h":"Rain24h","累积降水72h":"Rain72h"}
FEATURES = ["WT","Q","pH","EC","Turbidity","CODMn","NH3N","TN","TP","Rain6h","Rain24h","Rain72h"]
DISPLAY = {"WT":"WT","Q":"Flow (Q)","pH":"pH","EC":"EC","Turbidity":"Turbidity",
           "CODMn":"CODMn","NH3N":"NH3-N","TN":"TN","TP":"TP",
           "Rain6h":"Rain_6h","Rain24h":"Rain_24h","Rain72h":"Rain_72h"}
# canonical order = classification-importance descending (= SHAP descending)
ORDER = ["pH","WT","TP","Turbidity","TN","EC","Rain72h","CODMn","NH3N","Q","Rain24h","Rain6h"]

RFC = dict(n_estimators=500, max_depth=8, class_weight="balanced", random_state=42, n_jobs=-1)

# ----------------------------------------------------------------------------
# Backup existing figure/table files
# ----------------------------------------------------------------------------
ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
BK = os.path.join(OUT, "backup_figs_Bkoujing_" + ts)
os.makedirs(BK, exist_ok=True)
TARGETS = [
    (FIG_DIR,   "Table1_LOYO_Performance"),
    (STAB_DIR,  "TableS2_Model_Performance_Variable_Importance"),
    (STAB_DIR,  "TableS7_RF_LR_LOYO_AUC_Comparison"),
    (STAB_DIR,  "TableS8_SHAP_Feature_Importance"),
    (SCHART_DIR,"FigS5_SHAP_beeswarm"),
    (SCHART_DIR,"FigS6_SHAP_importance_bar"),
]
for d, stem in TARGETS:
    for ext in (".csv", ".xlsx", ".png", ".tiff"):
        p = os.path.join(d, stem + ext)
        if os.path.exists(p):
            shutil.copy2(p, os.path.join(BK, stem + ext))
print("backed up figures/tables to", BK)

# ----------------------------------------------------------------------------
# Load authoritative B口径 CSVs
# ----------------------------------------------------------------------------
def read_auth(name):
    return pd.read_csv(os.path.join(AUTH_DIR, name), encoding="utf-8-sig")

reg_loyo  = read_auth("02_RF_regression_LOYO.csv")     # test_year,n_train,n_test,R2,RMSE,MAE (+Mean,SD rows)
cls_loyo  = read_auth("05_RF_classification_LOYO.csv")  # test_year,n_train,n_test,positive_test,AUC,Acc,Sen,Spe,F1
reg_imp   = read_auth("04_RF_regression_importance_summary.csv")     # feature,importance_mean,importance_sd
cls_imp   = read_auth("07_RF_classification_importance_summary.csv") # feature,importance_mean,importance_sd

# ----------------------------------------------------------------------------
# Recompute LR LOYO + SHAP (reproduces b_run_full.py exactly)
# ----------------------------------------------------------------------------
df = pd.read_excel(DATA_FILE).rename(columns=COL_MAP)
df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
for c in FEATURES + ["DO"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.dropna(subset=["datetime", "DO"]).sort_values("datetime").reset_index(drop=True)

df["low"] = (df["DO"] <= 2.0).astype(int)
dt_h = df["datetime"].diff().dt.total_seconds() / 3600
new_event = df["low"].astype(bool) & (~df["low"].shift(1).fillna(False).astype(bool) | (dt_h > 4 * 1.5))
df["_grp"] = new_event.cumsum()
df["hypoxia"] = 0
for _, sub in df[df["low"] == 1].groupby("_grp"):
    dur = max((sub["datetime"].iloc[-1] - sub["datetime"].iloc[0]).total_seconds() / 3600, 4)
    if dur >= 4.0:
        df.loc[sub.index, "hypoxia"] = 1
df = df.drop(columns=["low", "_grp"])
df["year"] = df["datetime"].dt.year

imputer = SimpleImputer(strategy="median")
X_full = imputer.fit_transform(df[FEATURES])
y_full = df["hypoxia"].astype(int).values
years = sorted(df["year"].unique())

# --- LR LOYO ---
lr_rows = []
for y in years:
    tr = df[df["year"] != y]; te = df[df["year"] == y]
    sc = StandardScaler()
    Xtr = sc.fit_transform(imputer.fit_transform(tr[FEATURES]))
    Xte = sc.transform(imputer.transform(te[FEATURES]))
    m = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    m.fit(Xtr, tr["hypoxia"].astype(int))
    lr_rows.append((y, roc_auc_score(te["hypoxia"].astype(int), m.predict_proba(Xte)[:, 1])))
lr_dict = dict(lr_rows)
lr_mean = float(np.mean([v for _, v in lr_rows]))
lr_sd   = float(np.std([v for _, v in lr_rows], ddof=1))
print("LR LOYO per-year:", {y: round(v, 6) for y, v in lr_rows})
print(f"LR mean={lr_mean:.6f} sd={lr_sd:.6f}")

# --- SHAP ---
mc = RandomForestClassifier(**RFC); mc.fit(X_full, y_full)
expl = shap.TreeExplainer(mc)
idx = np.random.default_rng(42).choice(len(X_full), size=min(2000, len(X_full)), replace=False)
X_sample = X_full[idx]
sv = expl.shap_values(X_sample)
if isinstance(sv, list):
    sv = sv[1]
elif sv.ndim == 3:
    sv = sv[:, :, 1]
mean_shap = np.abs(sv).mean(axis=0)
shap_dict = {f: float(v) for f, v in zip(FEATURES, mean_shap)}
print("SHAP mean|SHAP|:", {f: round(v, 6) for f, v in shap_dict.items()})

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def f3(x): return f"{x:.3f}"
def f2(x): return f"{x:.2f}"
def fint(x): return str(int(round(x)))

def fmt_pct(mean, sd):
    """importance as % with 1 decimal (2 decimals when mean% < 0.1)."""
    pm, ps = mean * 100.0, sd * 100.0
    if pm < 0.1:
        return f"{pm:.2f} ± {ps:.2f}"
    mstr = f"{pm:.1f}"
    sstr = f"{ps:.1f}"
    if sstr == "0.0" and ps > 0:
        sstr = f"{ps:.2f}"
    return f"{mstr} ± {sstr}"

def plusminus(mean, sd, nd=3):
    return f"{mean:.{nd}f} ± {sd:.{nd}f}"

# lookup importance (feature -> (mean, sd)) from authoritative summaries
reg_imp_map = {r.feature: (r.importance_mean, r.importance_sd) for r in reg_imp.itertuples()}
cls_imp_map = {r.feature: (r.importance_mean, r.importance_sd) for r in cls_imp.itertuples()}

# ----------------------------------------------------------------------------
# Table 1 CSV
# ----------------------------------------------------------------------------
reg_rows = reg_loyo[reg_loyo["test_year"].astype(str).str.isdigit()].copy()
cls_rows = cls_loyo[cls_loyo["test_year"].astype(str).str.isdigit()].copy()

t1_lines = []
t1_lines.append("Table 1. Random forest leave-one-year-out (LOYO) performance (cross-year Mean ± SD).")
t1_lines.append("")
t1_lines.append("(a) Regression (continuous DO)")
t1_lines.append('"Test year","n_train","n_test","R²","RMSE (mg/L)","MAE (mg/L)"')
for r in reg_rows.itertuples():
    t1_lines.append(f'"{int(r.test_year)}","{fint(r.n_train)}","{fint(r.n_test)}","{f3(r.R2)}","{f2(r.RMSE)}","{f3(r.MAE)}"')
r2m, r2s = reg_loyo[reg_loyo["test_year"]=="Mean"]["R2"].iloc[0], reg_loyo[reg_loyo["test_year"]=="SD"]["R2"].iloc[0]
rmm, rms = reg_loyo[reg_loyo["test_year"]=="Mean"]["RMSE"].iloc[0], reg_loyo[reg_loyo["test_year"]=="SD"]["RMSE"].iloc[0]
mam, mas = reg_loyo[reg_loyo["test_year"]=="Mean"]["MAE"].iloc[0], reg_loyo[reg_loyo["test_year"]=="SD"]["MAE"].iloc[0]
t1_lines.append(f'"Mean ± SD","—","—","{plusminus(r2m,r2s)}","{plusminus(rmm,rms,2)}","{plusminus(mam,mas)}"')
t1_lines.append("")
t1_lines.append("(b) Classification (hypoxia events)")
t1_lines.append('"Test year","Positive","AUC","Accuracy","Sensitivity","Specificity","F1"')
for r in cls_rows.itertuples():
    t1_lines.append(f'"{int(r.test_year)}","{fint(r.positive_test)}","{f3(r.AUC)}","{f3(r.Accuracy)}","{f3(r.Sensitivity)}","{f3(r.Specificity)}","{f3(r.F1)}"')
aucm, aucs = cls_loyo[cls_loyo["test_year"]=="Mean"]["AUC"].iloc[0], cls_loyo[cls_loyo["test_year"]=="SD"]["AUC"].iloc[0]
accm, accs = cls_loyo[cls_loyo["test_year"]=="Mean"]["Accuracy"].iloc[0], cls_loyo[cls_loyo["test_year"]=="SD"]["Accuracy"].iloc[0]
senm, sens = cls_loyo[cls_loyo["test_year"]=="Mean"]["Sensitivity"].iloc[0], cls_loyo[cls_loyo["test_year"]=="SD"]["Sensitivity"].iloc[0]
spem, spes = cls_loyo[cls_loyo["test_year"]=="Mean"]["Specificity"].iloc[0], cls_loyo[cls_loyo["test_year"]=="SD"]["Specificity"].iloc[0]
f1m, f1s = cls_loyo[cls_loyo["test_year"]=="Mean"]["F1"].iloc[0], cls_loyo[cls_loyo["test_year"]=="SD"]["F1"].iloc[0]
t1_lines.append(f'"Mean ± SD","—","{plusminus(aucm,aucs)}","{plusminus(accm,accs)}","{plusminus(senm,sens)}","{plusminus(spem,spes)}","{plusminus(f1m,f1s)}"')
t1_csv = "\n".join(t1_lines) + "\n"
with open(os.path.join(FIG_DIR, "Table1_LOYO_Performance.csv"), "w", encoding="utf-8-sig", newline="") as f:
    f.write(t1_csv)
print("Table1 CSV written")

# ----------------------------------------------------------------------------
# Table S2 CSV
# ----------------------------------------------------------------------------
s2_header = '"Variable","DO Dynamics (regression, %)","Hypoxia Events (classification, %)"'
s2_data = []
for feat in ORDER:
    rm, rs = reg_imp_map[feat]
    cm, cs = cls_imp_map[feat]
    s2_data.append(f'"{DISPLAY[feat]}","{fmt_pct(rm, rs)}","{fmt_pct(cm, cs)}"')
s2_note = ("\"Model performance (leave-one-year-out, mean ± SD across 2021–2025 test years): "
           f"regression R² = {plusminus(r2m,r2s)}, RMSE = {plusminus(rmm,rms,2)} mg/L; "
           f"classification AUC = {plusminus(aucm,aucs)}, accuracy = {plusminus(accm,accs)}, "
           f"sensitivity = {plusminus(senm,sens)}, specificity = {plusminus(spem,spes)}.\"")
s2_csv = s2_header + "\n" + "\n".join(s2_data) + "\n" + s2_note + "\n"
with open(os.path.join(STAB_DIR, "TableS2_Model_Performance_Variable_Importance.csv"), "w", encoding="utf-8-sig", newline="") as f:
    f.write(s2_csv)
print("TableS2 CSV written")

# ----------------------------------------------------------------------------
# Table S7 CSV (RF vs LR LOYO AUC)
# ----------------------------------------------------------------------------
s7_lines = ['"Test year","LR baseline AUC","RF AUC"']
for r in cls_rows.itertuples():
    y = int(r.test_year)
    s7_lines.append(f'"{y}","{f3(lr_dict[y])}","{f3(r.AUC)}"')
s7_lines.append(f'"Mean ± SD","{plusminus(lr_mean, lr_sd)}","{plusminus(aucm, aucs)}"')
s7_csv = "\n".join(s7_lines) + "\n"
with open(os.path.join(STAB_DIR, "TableS7_RF_LR_LOYO_AUC_Comparison.csv"), "w", encoding="utf-8-sig", newline="") as f:
    f.write(s7_csv)
print("TableS7 CSV written")

# ----------------------------------------------------------------------------
# Table S8 CSV (SHAP feature importance)
# ----------------------------------------------------------------------------
shap_sorted = sorted(shap_dict.items(), key=lambda t: -t[1])
s8_lines = ['"Feature","Mean |SHAP value|"']
for feat, v in shap_sorted:
    s8_lines.append(f'"{DISPLAY[feat]}","{f3(v)}"')
s8_csv = "\n".join(s8_lines) + "\n"
with open(os.path.join(STAB_DIR, "TableS8_SHAP_Feature_Importance.csv"), "w", encoding="utf-8-sig", newline="") as f:
    f.write(s8_csv)
print("TableS8 CSV written")

# ----------------------------------------------------------------------------
# Render table PNG/TIFF
# ----------------------------------------------------------------------------
HEADER = "#2C7BB6"; ALT = "#F5F8FB"; GRID = "#CCCCCC"

def style_table(table, ncols, nrows, header_rows=1):
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    for j in range(ncols):
        c = table[(0, j)]
        c.set_facecolor(HEADER)
        c.set_text_props(weight="bold", color="white")
    for i in range(header_rows, nrows):
        for j in range(ncols):
            c = table[(i, j)]
            if (i - header_rows) % 2 == 1:
                c.set_facecolor(ALT)
            c.set_edgecolor(GRID)

def read_csv_lines(path):
    with open(path, encoding="utf-8-sig") as f:
        return [r for r in csv.reader(f)]

def save_png_tiff(fig, dir_new, dir_mir, stem):
    for d in (dir_new, dir_mir):
        fig.savefig(os.path.join(d, stem + ".png"), dpi=200, facecolor="white", bbox_inches="tight")
        fig.savefig(os.path.join(d, stem + ".tiff"), dpi=200, facecolor="white", bbox_inches="tight", format="tiff")
    plt.close(fig)

# ---- Table 1 ----
rows = read_csv_lines(os.path.join(FIG_DIR, "Table1_LOYO_Performance.csv"))
title = rows[0][0]
sub_a, sub_b = [], []
cur = None
for r in rows[1:]:
    if not r or not r[0].strip(): continue
    if r[0].startswith("(a)"): cur = "a"; continue
    if r[0].startswith("(b)"): cur = "b"; continue
    (sub_a if cur == "a" else sub_b).append(r)
head_a, data_a = sub_a[0], sub_a[1:]
head_b, data_b = sub_b[0], sub_b[1:]

fig = plt.figure(figsize=(10, 8), dpi=200)
fig.text(0.5, 0.965, title, ha="center", va="top", fontsize=12, weight="bold")
fig.text(0.06, 0.60, "(a) Regression (continuous DO)", ha="left", va="bottom", fontsize=10, weight="bold")
fig.text(0.06, 0.28, "(b) Classification (hypoxia events)", ha="left", va="bottom", fontsize=10, weight="bold")
ax_a = fig.add_axes([0.06, 0.62, 0.88, 0.30]); ax_a.axis("off")
tab_a = ax_a.table(cellText=data_a, colLabels=head_a, cellLoc="center", loc="upper center")
style_table(tab_a, len(head_a), len(data_a) + 1); tab_a.scale(1, 1.4)
ax_b = fig.add_axes([0.06, 0.30, 0.88, 0.30]); ax_b.axis("off")
tab_b = ax_b.table(cellText=data_b, colLabels=head_b, cellLoc="center", loc="upper center")
style_table(tab_b, len(head_b), len(data_b) + 1); tab_b.scale(1, 1.4)
save_png_tiff(fig, FIG_DIR, MIR_FIG, "Table1_LOYO_Performance")
print("Table1 PNG/TIFF rendered")

# ---- Table S2 ----
rows = read_csv_lines(os.path.join(STAB_DIR, "TableS2_Model_Performance_Variable_Importance.csv"))
head = rows[0]; data = rows[1:-1]; note = rows[-1][0]
fig = plt.figure(figsize=(9, 6.5), dpi=200)
ax = fig.add_axes([0.04, 0.22, 0.92, 0.72]); ax.axis("off")
tab = ax.table(cellText=data, colLabels=head, cellLoc="center", loc="upper center")
style_table(tab, len(head), len(data) + 1); tab.scale(1, 1.5)
fig.text(0.04, 0.14, note, ha="left", va="top", fontsize=8.5, wrap=True)
save_png_tiff(fig, STAB_DIR, MIR_STAB, "TableS2_Model_Performance_Variable_Importance")
print("TableS2 PNG/TIFF rendered")

# ---- Table S7 / S8 (generic) ----
def render_simple_table(stem, title):
    csv_path = os.path.join(STAB_DIR, stem + ".csv")
    dfc = pd.read_csv(csv_path, encoding="utf-8-sig")
    n_rows, n_cols = dfc.shape
    fig_w = max(7, n_cols * 2.2)
    fig_h = max(4, (n_rows + 1) * 0.45)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=200)
    ax.axis("off")
    tbl = ax.table(cellText=dfc.values, colLabels=dfc.columns, cellLoc="center", loc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1, 1.7)
    for j in range(n_cols):
        c = tbl[(0, j)]; c.set_facecolor("#4472C4"); c.set_text_props(weight="bold", color="white")
    for i in range(1, n_rows + 1):
        for j in range(n_cols):
            if i % 2 == 0:
                tbl[(i, j)].set_facecolor("#D9E1F2")
    ax.set_title(title, fontsize=12, weight="bold", pad=16)
    save_png_tiff(fig, STAB_DIR, MIR_STAB, stem)

render_simple_table("TableS7_RF_LR_LOYO_AUC_Comparison", "Random Forest vs Logistic Regression LOYO AUC")
render_simple_table("TableS8_SHAP_Feature_Importance", "SHAP Feature Importance (hypoxia classification)")
print("TableS7/S8 PNG/TIFF rendered")

# ----------------------------------------------------------------------------
# SHAP figures (beeswarm + bar)
# ----------------------------------------------------------------------------
X_disp = pd.DataFrame(X_sample, columns=[DISPLAY[f] for f in FEATURES])

# beeswarm
shap.summary_plot(sv, X_disp, show=False, max_display=12)
plt.tight_layout()
for d in (SCHART_DIR, MIR_SCHART):
    plt.savefig(os.path.join(d, "FigS5_SHAP_beeswarm.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(d, "FigS5_SHAP_beeswarm.tiff"), dpi=300, bbox_inches="tight", format="tiff")
plt.close("all")
print("FigS5 beeswarm rendered")

# bar
disp_order = [DISPLAY[f] for f in ORDER]
vals = [shap_dict[f] for f in ORDER]
fig, ax = plt.subplots(figsize=(7, 5))
ypos = np.arange(len(ORDER))
ax.barh(ypos, vals, color="#4472C4")
ax.set_yticks(ypos)
ax.set_yticklabels(disp_order)
ax.invert_yaxis()
ax.set_xlabel("Mean |SHAP value|")
ax.set_title("SHAP feature importance (hypoxia classification)")
for i, v in enumerate(vals):
    ax.text(v + 0.002, i, f"{v:.3f}", va="center", fontsize=8)
plt.tight_layout()
for d in (SCHART_DIR, MIR_SCHART):
    fig.savefig(os.path.join(d, "FigS6_SHAP_importance_bar.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(d, "FigS6_SHAP_importance_bar.tiff"), dpi=300, bbox_inches="tight", format="tiff")
plt.close(fig)
print("FigS6 bar rendered")

# ----------------------------------------------------------------------------
# Write XLSX (Table1, TableS2, TableS7, TableS8) + sync csv/xlsx to mirror
# ----------------------------------------------------------------------------
import openpyxl

def csv_to_xlsx(src_csv, xlsx_path):
    rows = read_csv_lines(src_csv)
    wb = openpyxl.Workbook()
    ws = wb.active
    for i, r in enumerate(rows, start=1):
        for j, cell in enumerate(r, start=1):
            ws.cell(row=i, column=j, value=cell)
    wb.save(xlsx_path)

for d, stem in TARGETS:
    if stem.startswith("FigS"):
        continue  # SHAP figures have no csv/xlsx
    src_csv = os.path.join(d, stem + ".csv")
    mir_dir = MIR_FIG if d == FIG_DIR else MIR_STAB
    # xlsx in both new and mirror
    csv_to_xlsx(src_csv, os.path.join(d, stem + ".xlsx"))
    csv_to_xlsx(src_csv, os.path.join(mir_dir, stem + ".xlsx"))
    # sync csv to mirror
    shutil.copy2(src_csv, os.path.join(mir_dir, stem + ".csv"))

print("XLSX written + csv synced to mirror")

print("\nDONE — all B口径 figures/tables regenerated")
