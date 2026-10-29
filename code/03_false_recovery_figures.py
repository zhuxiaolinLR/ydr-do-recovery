# -*- coding: utf-8 -*-
"""重渲染 Fig4(e) 与 FigS4（逐年 LOYO AUC）。

数据源（01_rf_sfi_loyo_pipeline.py 的 B口径 2021–2025 输出，位于 outputs/）：
  01_processed_timeseries.csv  -> Fig4 恢复分类/Delta/再塌陷/轨迹/SFI 分布
  09_SFI_LOYO_ROC.csv          -> FigS4 逐年 onset AUC（24h/48h，2021–2025）

Fig4(a-d) 恢复分类与 Fig4(e) SFI 分布、FigS4 逐年 AUC 均使用 B口径（2021–2025，排除 2020）。
"""
import os, numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 仓库根
BASE = os.path.join(REPO, "outputs")   # 01_rf_sfi_loyo_pipeline.py 的 B口径(2021–2025) 输出
FIG  = os.path.join(REPO, "outputs", "Figures")
SUP  = os.path.join(REPO, "outputs", "Supplementary Charts")

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9,
    'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 7.5,
    'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.6,
})

C_BLUE = '#2C7BB6'; C_RED = '#D7191C'; C_ORANGE = '#FDAE61'
C_LIGHT = '#ABD9E9'; C_GREY = '#888888'; C_NAVY = '#4575b4'
H = np.timedelta64(1, 'h')

def save(fig, name, d):
    os.makedirs(d, exist_ok=True)
    for ext in ['png', 'tiff']:
        kw = dict(dpi=300, bbox_inches='tight', pad_inches=0.05, facecolor='white')
        if ext == 'tiff':
            kw['pil_kwargs'] = {'compression': 'tiff_lzw'}
        fig.savefig(os.path.join(d, f'{name}.{ext}'), **kw)
    print(f'[OK] {name} -> {d}')

def panel(ax, label):
    ax.text(-0.08, 1.05, label, transform=ax.transAxes, fontsize=11, fontweight='bold',
            va='bottom', ha='left')

def valid(a): return (~np.isnan(a)) & (a >= 0)

# =====================================================================
#  载入新时序
# =====================================================================
df = pd.read_csv(os.path.join(BASE, '01_processed_timeseries.csv'), encoding='utf-8-sig',
                 low_memory=False)
df['t'] = pd.to_datetime(df['datetime'], errors='coerce')
df = df.sort_values('t').reset_index(drop=True)
do  = pd.to_numeric(df['DO'], errors='coerce').values.astype(float)
cod = pd.to_numeric(df['CODMn'], errors='coerce').values.astype(float)
nh3 = pd.to_numeric(df['NH3N'], errors='coerce').values.astype(float)
tns = df['t'].values.astype('datetime64[ns]')
N = len(df)

# ---- event clusters ----
hyp_idx = [i for i in range(N) if do[i] <= 2.0]
clusters = []; cur = []
for i in hyp_idx:
    if not cur: cur = [i]
    elif (tns[i] - tns[cur[-1]]) / H < 12: cur.append(i)
    else: clusters.append(cur); cur = [i]
if cur: clusters.append(cur)
clusters = [c for c in clusters if (tns[c[-1]] - tns[c[0]]) / H >= 4]

# ---- recovered + genuine/false ----
rec_events = []
for c in clusters:
    end = c[-1]; found = None
    for k in range(end + 1, N):
        if (tns[k] - tns[end]) / H > 72: break
        if do[k] >= 5.0: found = k; break
    if found is not None: rec_events.append((c, found))

d_cod = []; d_nh3 = []
for c, rec_k in rec_events:
    end = c[-1]; t_end = tns[end]
    ev_cod = cod[c][valid(cod[c])]; ev_nh3 = nh3[c][valid(nh3[c])]
    post_idx = [k for k in range(end + 1, N) if (tns[k] - t_end) / H <= 72]
    po_cod = cod[post_idx][valid(cod[post_idx])]
    po_nh3 = nh3[post_idx][valid(nh3[post_idx])]
    if len(ev_cod) and len(po_cod) and len(ev_nh3) and len(po_nh3):
        d_cod.append(po_cod.mean() - ev_cod.mean())
        d_nh3.append(po_nh3.mean() - ev_nh3.mean())
d_cod = np.array(d_cod); d_nh3 = np.array(d_nh3)
genuine = (d_cod < 0) & (d_nh3 < 0)
n_gen = int(genuine.sum()); n_fal = int((~genuine).sum())

# ---- re-collapse ----
gen_rec = 0; fal_rec = 0; false_traj = None
for (c, rec_k), g in zip(rec_events, genuine):
    t0 = tns[rec_k]; hit = False
    for k in range(rec_k + 1, N):
        dt = (tns[k] - t0) / H
        if dt > 24: break
        if do[k] <= 2.0:
            hit = True
            if false_traj is None and not g:
                false_traj = (c, rec_k, k)
            break
    if hit:
        if g: gen_rec += 1
        else: fal_rec += 1

print(f"clusters={len(clusters)} recovered={len(rec_events)} genuine={n_gen} false={n_fal}")
print(f"re-collapse<=24h: genuine {100.0*gen_rec/n_gen:.1f}%  false {100.0*fal_rec/n_fal:.1f}%")

# ---- 新 SFI ----
sfi = pd.to_numeric(df['SFI'], errors='coerce').values.astype(float)
sfi_ok = sfi[~np.isnan(sfi)]
p90 = np.percentile(sfi_ok, 90); med = np.percentile(sfi_ok, 50)
print(f"SFI: n={len(sfi_ok)} mean={sfi_ok.mean():.3f} sd={sfi_ok.std(ddof=1):.3f} "
      f"median={med:.3f} P90={p90:.3f} max={sfi_ok.max():.3f}")

# =====================================================================
#  FIG 4  (a-d 恢复分类不变) (e) 新 SFI 分布
# =====================================================================
fig4 = plt.figure(figsize=(8.6, 9.4))
gs4 = fig4.add_gridspec(3, 2, height_ratios=[1, 1, 1.25], hspace=0.55, wspace=0.35)

ax = fig4.add_subplot(gs4[0, 0]); panel(ax, '(a)')
lbl = ['Genuine', 'False (pseudo)']
cnt = [n_gen, n_fal]
bars = ax.bar(lbl, cnt, color=[C_BLUE, C_RED], edgecolor='white', lw=0.4, width=0.5)
for b, v in zip(bars, cnt):
    ax.text(b.get_x()+b.get_width()/2, v+1, f'{v}\n({v/len(genuine)*100:.1f}%)',
            ha='center', fontsize=8, fontweight='bold')
ax.set_ylabel('Number of recovery events'); ax.set_ylim(0, 62)
ax.set_title('Recovery classification', fontweight='bold', fontsize=10, pad=6)

ax = fig4.add_subplot(gs4[0, 1]); panel(ax, '(b)')
cat = ['\u0394CODMn', '\u0394NH$_3$-N']
gen_means = [d_cod[genuine].mean(), d_nh3[genuine].mean()]
fal_means = [d_cod[~genuine].mean(), d_nh3[~genuine].mean()]
x = np.arange(2); w = 0.35
ax.bar(x-w/2, gen_means, w, color=C_BLUE, edgecolor='white', lw=0.4, label='Genuine')
ax.bar(x+w/2, fal_means, w, color=C_RED, edgecolor='white', lw=0.4, label='False')
ax.axhline(0, color=C_GREY, lw=0.5)
for i, (gm, fm) in enumerate(zip(gen_means, fal_means)):
    ax.text(i-w/2, gm + (0.03 if gm >= 0 else -0.08), f'{gm:.2f}', ha='center', fontsize=7.5)
    ax.text(i+w/2, fm + (0.03 if fm >= 0 else -0.08), f'{fm:.2f}', ha='center', fontsize=7.5)
ax.set_xticks(x); ax.set_xticklabels(cat)
ax.set_ylabel('\u0394 during recovery (mg/L)')
ax.legend(fontsize=7.5, frameon=False)
ax.set_title('Recovery-phase \u0394 (genuine vs false)', fontweight='bold', fontsize=10, pad=6)

ax = fig4.add_subplot(gs4[1, 0]); panel(ax, '(c)')
rlbl = ['Genuine', 'False']
rv = [100.0*gen_rec/n_gen, 100.0*fal_rec/n_fal]
bars = ax.bar(rlbl, rv, color=[C_BLUE, C_RED], edgecolor='white', lw=0.4, width=0.5)
for b, v in zip(bars, rv):
    ax.text(b.get_x()+b.get_width()/2, v+1, f'{v:.1f}%', ha='center', fontsize=8, fontweight='bold')
ax.set_ylabel('24-h re-collapse risk (%)'); ax.set_ylim(0, 60)
ax.set_title('Recovery stability comparison', fontweight='bold', fontsize=10, pad=6)

ax = fig4.add_subplot(gs4[1, 1]); panel(ax, '(d)')
if false_traj is not None:
    c, rec_k, rc_k = false_traj
    lo = max(0, c[0] - 6); hi = min(N - 1, rc_k + 6)
    seg = slice(lo, hi + 1)
    tt = tns[seg]; dd = do[seg]
    ax.plot(tt, dd, color=C_RED, lw=1.1)
    ax.scatter([tns[rec_k]], [do[rec_k]], color=C_BLUE, s=28, zorder=5,
               label='Recovery (DO > 5)')
    ax.scatter([tns[rc_k]], [do[rc_k]], color=C_RED, s=28, zorder=5, marker='^',
               label='Re-collapse (DO \u2264 2)')
    ax.axhline(2.0, color=C_GREY, ls='--', lw=0.8, alpha=0.6)
    ax.axhline(5.0, color=C_GREY, ls=':', lw=0.8, alpha=0.6)
    ax.axvspan(tns[c[0]], tns[c[-1]], color=C_RED, alpha=0.10)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(mdates.AutoDateLocator()))
    ax.set_xlabel('Time'); ax.set_ylabel('DO (mg/L)')
    ax.set_title('Representative false recovery trajectory', fontweight='bold', fontsize=10, pad=6)
    ax.legend(fontsize=7, frameon=False, loc='lower right')
    ax.annotate('', xy=(tns[rc_k], do[rc_k]), xytext=(tns[rec_k], do[rec_k]),
                arrowprops=dict(arrowstyle='<->', color=C_BLUE, lw=0.8))
    gap_h = (tns[rc_k] - tns[rec_k]) / H
    ax.text((tns[rec_k] + (tns[rc_k]-tns[rec_k])/2), 3.5, f'{gap_h:.0f} h',
            ha='center', fontsize=7, color=C_BLUE)
else:
    ax.text(0.5, 0.5, 'No false-recovery\nre-collapse event found', ha='center',
            va='center', fontsize=9, color=C_GREY)
    ax.set_axis_off()
    ax.set_title('Representative false recovery trajectory', fontweight='bold', fontsize=10, pad=6)

ax = fig4.add_subplot(gs4[2, :]); panel(ax, '(e)')
bins = np.linspace(0, 2.5, 51)
ax.hist(sfi_ok, bins=bins, color=C_LIGHT, edgecolor='white', lw=0.3, alpha=0.9)
ax.axvline(p90, color=C_RED, ls='--', lw=1.0, label=f'P90 = {p90:.3f}')
ax.axvline(sfi_ok.mean(), color=C_BLUE, ls='-', lw=0.8, label=f'Mean = {sfi_ok.mean():.3f}')
ax.axvline(med, color=C_GREY, ls=':', lw=0.8, label=f'Median = {med:.3f}')
ax.set_xlabel('System Fatigue Index (SFI)'); ax.set_ylabel('Frequency')
ax.set_title(f'SFI distribution (n = {len(sfi_ok):,}; range 0\u2013{sfi_ok.max():.2f})',
             fontweight='bold', fontsize=10, pad=6)
ax.legend(fontsize=7.5, frameon=False)

save(fig4, 'Fig4_False_Recovery_SFI', FIG)
plt.close(fig4)

# =====================================================================
#  FIG S4 — 逐年 onset LOYO AUC（2021–2025，排除 2020）
# =====================================================================
roc = pd.read_csv(os.path.join(BASE, '09_SFI_LOYO_ROC.csv'))
roc = roc[roc['test_year'].notna()].copy()
roc = roc[~roc['test_year'].astype(str).str.contains('Mean', na=False)]
roc['test_year'] = roc['test_year'].astype(int)
roc = roc[roc['test_year'] != 2020].copy()   # 排除 2020（base 不稳定）
roc['AUC'] = roc['AUC'].astype(float)
years = sorted(roc['test_year'].unique())
h24 = roc[roc['horizon_h'] == 24].set_index('test_year')['AUC'].reindex(years)
h48 = roc[roc['horizon_h'] == 48].set_index('test_year')['AUC'].reindex(years)

# 打印验证均值
m24, s24 = h24.mean(), h24.std(ddof=1)
m48, s48 = h48.mean(), h48.std(ddof=1)
print(f"FigS4 2021-2025: mean 24h={m24:.3f}±{s24:.3f}  mean 48h={m48:.3f}±{s48:.3f}")

figS4 = plt.figure(figsize=(6.4, 4.0))
ax = figS4.add_subplot(111)
x = np.arange(len(years)); w = 0.38
ax.bar(x-w/2, h24.values, w, color=C_BLUE, edgecolor='white', lw=0.4, label='24 h')
ax.bar(x+w/2, h48.values, w, color=C_ORANGE, edgecolor='white', lw=0.4, label='48 h')
ax.axhline(0.5, color=C_GREY, ls='--', lw=0.8, alpha=0.7)
ax.text(len(years)-0.5, 0.52, 'random (AUC = 0.5)', ha='right', fontsize=7, color=C_GREY)
for xi, (a24, a48) in enumerate(zip(h24.values, h48.values)):
    ax.text(xi-w/2, a24+0.02, f'{a24:.2f}', ha='center', fontsize=7)
    ax.text(xi+w/2, a48+0.02, f'{a48:.2f}', ha='center', fontsize=7)
ax.set_xticks(x); ax.set_xticklabels([str(y) for y in years])
ax.set_xlabel('Test year'); ax.set_ylabel('AUC')
ax.set_ylim(0, 0.95)
ax.set_title('SFI prospective onset prediction (LOYO, per-year AUC)',
             fontweight='bold', fontsize=10, pad=6)
ax.legend(fontsize=8, frameon=False, loc='lower left')
figS4.text(0.5, -0.04,
           f'Mean 24 h = {m24:.3f} \u00b1 {s24:.3f}    |    Mean 48 h = {m48:.3f} \u00b1 {s48:.3f}',
           ha='center', fontsize=8, color='black')
save(figS4, 'FigS4_SFI_per_year_LOYO_AUC', SUP)
plt.close(figS4)

print('DONE')
