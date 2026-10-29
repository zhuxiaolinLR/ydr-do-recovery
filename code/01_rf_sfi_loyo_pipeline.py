# -*- coding: utf-8 -*-
"""
JEM - Yongding River DO / hypoxia validation pipeline
======================================================

用途
----
1) RF回归：连续DO预测，Leave-One-Year-Out (LOYO)时间阻塞验证
2) RF分类：低氧事件识别，LOYO验证
3) 变量重要性：跨年份 Mean ± SD
4) PDP：关键变量的非线性响应窗口
5) SFI：基于当前时刻以前数据计算，并预测未来24/48 h低氧
6) SFI ROC/AUC：总体ROC + LOYO年度外推验证
7) 自动输出CSV、PNG、Excel结果

与你当前JEM稿件的对应关系
--------------------------
- 当前稿件：RF 500棵树、max_depth=8、原10折CV
- 本脚本：保留RF参数，但改为LOYO时间阻塞验证
- 低氧定义：DO <= 2.0 mg/L，持续 >= 4 h
- 快速下降事件：DO下降幅度 >= 3.0 mg/L 且下降速率 >= 1.5 mg/L/h
- SFI：OI × RR
  OI = 恢复/当前时刻前24 h DO一阶差分绝对值的均值
  RR = ln(CODMn/base + NH3-N/base)
  base默认采用“非低氧时段中位数”，可在CONFIG中修改
- SFI未来24/48 h标签：只使用当前时刻之后的已识别低氧事件，不允许未来信息进入SFI

重要
----
运行前只需要修改 CONFIG 中的 INPUT_FILE。
如果你的Excel列名与脚本识别的中文/英文别名不同，
请在 COLUMN_ALIASES 中加入实际列名。

建议环境：
Python >= 3.10
pandas, numpy, scikit-learn, matplotlib, openpyxl
"""

from pathlib import Path
import sys
import warnings
warnings.filterwarnings("ignore")
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 仓库根目录（本脚本位于 code/ 下）
REPO = Path(__file__).resolve().parent.parent

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    roc_auc_score, roc_curve, accuracy_score,
    recall_score, f1_score, confusion_matrix
)
from sklearn.inspection import PartialDependenceDisplay
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


# ============================================================
# 0. CONFIG —— 只需要首先修改这里
# ============================================================

CONFIG = {
    # ---------- 输入文件 ----------
    # 完整 12 变量监测表（第三方国家网数据，未随仓库分发）。
    # 请将完整数据放入 data/input/ 下并命名为 THGLQ_12var_2021-2025.xlsx，
    # 或将本行改为你的实际路径。列名见 COLUMN_ALIASES（中英文自动识别）。
    "INPUT_FILE": str(REPO / "data" / "input" / "THGLQ_12var_2021-2025.xlsx"),
    "SHEET_NAME": None,          # None = 自动读取第一个sheet

    # ---------- 分析断面 ----------
    # 推荐JEM当前SFI/低氧重点分析THGLQ。
    # 如果Excel没有站点列，设为None。
    "SITE": "THGLQ",

    # 如果想把所有站点分别跑一遍，改成：
    # "SITE": "ALL"

    # ---------- 时间 ----------
    "EXPECTED_INTERVAL_HOURS": 4,

    # ---------- 低氧事件 ----------
    "HYPOXIA_DO_THRESHOLD": 2.0,
    "HYPOXIA_MIN_DURATION_HOURS": 4.0,

    # ---------- 快速下降 ----------
    "RAPID_DROP_MAGNITUDE": 3.0,
    "RAPID_DROP_RATE": 1.5,

    # ---------- RF ----------
    "N_ESTIMATORS": 500,
    "MAX_DEPTH": 8,
    "MIN_SAMPLES_LEAF": 1,
    "RANDOM_STATE": 42,
    "N_JOBS": -1,

    # ---------- SFI ----------
    "SFI_LOOKBACK_HOURS": 24,
    "SFI_FUTURE_HOURS": [24, 48],

    # SFI中base的定义：
    # "non_hypoxia_median" = 非低氧时段中位数
    # "all_median" = 全部数据中位数
    "SFI_BASE_METHOD": "non_hypoxia_median",

    # ---------- ROC ----------
    "ROC_MIN_SAMPLES": 20,

    # ---------- 输出 ----------
    "OUTPUT_DIR": str(REPO / "outputs"),

    # 是否保存PDP图
    "MAKE_PDP": True,

    # 最多画几个变量的PDP
    "N_PDP_FEATURES": 6,
}


# ============================================================
# 1. Excel列名自动识别
# ============================================================

COLUMN_ALIASES = {
    "datetime": [
        "datetime", "date_time", "timestamp", "time", "date",
        "日期时间", "监测时间", "时间", "日期"
    ],
    "site": [
        "site", "station", "section", "monitoring_site",
        "断面", "监测断面", "站点", "测站", "监测点"
    ],
    "DO": [
        "DO", "do", "Dissolved Oxygen", "dissolved_oxygen",
        "溶解氧", "溶解氧(mg/L)", "DO(mg/L)"
    ],
    "WT": [
        "WT", "wt", "Water Temperature", "water_temperature",
        "水温", "水温(℃)", "温度"
    ],
    "Q": [
        "Q", "Flow", "flow", "Discharge", "流量",
        "流量(m3/s)", "流量（m³/s）", "径流量"
    ],
    "pH": [
        "pH", "PH", "ph", "pH值", "pH（无量纲）"
    ],
    "EC": [
        "EC", "ec", "Conductivity", "Electrical Conductivity",
        "电导率", "电导率(μS/cm)", "电导率（μS/cm）"
    ],
    "Turbidity": [
        "Turbidity", "turbidity", "浊度", "浊度(NTU)", "浊度（NTU）"
    ],
    "CODMn": [
        "CODMn", "COD", "codmn", "COD_Mn",
        "高锰酸盐指数", "CODMn(mg/L)", "CODMn（mg/L）"
    ],
    "NH3N": [
        "NH3-N", "NH3N", "NH3_N", "ammonia",
        "氨氮", "氨氮(mg/L)", "NH₃-N", "NH₃-N(mg/L)"
    ],
    "TN": [
        "TN", "tn", "总氮", "总氮(mg/L)", "TN(mg/L)"
    ],
    "TP": [
        "TP", "tp", "总磷", "总磷(mg/L)", "TP(mg/L)"
    ],
    "Rain6h": [
        "Rain6h", "precip6h", "Pcp6h", "累积降水6h",
        "cumulative precipitation 6h"
    ],
    "Rain24h": [
        "Rain24h", "precip24h", "Pcp24h", "累积降水24h",
        "cumulative precipitation 24h"
    ],
    "Rain72h": [
        "Rain72h", "precip72h", "Pcp72h", "累积降水72h",
        "cumulative precipitation 72h"
    ],
}


def normalize_name(x):
    """列名标准化，用于模糊匹配。"""
    return (
        str(x).strip()
        .lower()
        .replace(" ", "")
        .replace("_", "")
        .replace("-", "")
        .replace("（", "(")
        .replace("）", ")")
    )


def find_column(df, key, required=False):
    normalized = {normalize_name(c): c for c in df.columns}

    for alias in COLUMN_ALIASES[key]:
        a = normalize_name(alias)
        if a in normalized:
            return normalized[a]

    # 再做包含式匹配
    for c in df.columns:
        nc = normalize_name(c)
        for alias in COLUMN_ALIASES[key]:
            a = normalize_name(alias)
            if len(a) >= 2 and a in nc:
                return c

    if required:
        raise ValueError(
            f"找不到必须字段 [{key}]。\n"
            f"当前Excel列名：{list(df.columns)}\n"
            f"请把实际列名加入 COLUMN_ALIASES['{key}']。"
        )
    return None


# ============================================================
# 2. 数据读取
# ============================================================

def load_data():
    input_file = Path(CONFIG["INPUT_FILE"])
    if not input_file.exists():
        raise FileNotFoundError(
            f"\n找不到输入文件：{input_file}\n"
            f"请修改 CONFIG['INPUT_FILE'] 为你的Excel实际路径。"
        )

    print("=" * 70)
    print("JEM RF + SFI 时间外推验证")
    print("=" * 70)
    print(f"输入文件：{input_file}")

    sheet = CONFIG["SHEET_NAME"]
    if sheet is None:
        xls = pd.ExcelFile(input_file)
        sheet = xls.sheet_names[0]
        print(f"自动使用第一个sheet：{sheet}")

    df = pd.read_excel(input_file, sheet_name=sheet)
    print(f"原始数据：{df.shape[0]} 行 × {df.shape[1]} 列")

    colmap = {}
    colmap["datetime"] = find_column(df, "datetime", required=True)
    colmap["DO"] = find_column(df, "DO", required=True)

    for key in COLUMN_ALIASES:
        if key not in colmap:
            colmap[key] = find_column(df, key, required=False)

    rename_dict = {
        v: k for k, v in colmap.items()
        if v is not None
    }
    df = df.rename(columns=rename_dict)

    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime", "DO"]).copy()

    # 数值列
    numeric_cols = [
        "DO", "WT", "Q", "pH", "EC", "Turbidity",
        "CODMn", "NH3N", "TN", "TP",
        "Rain6h", "Rain24h", "Rain72h"
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # 站点
    if "site" in df.columns:
        df["site"] = df["site"].astype(str).str.strip()

    # 排序
    sort_cols = ["datetime"]
    if "site" in df.columns:
        sort_cols = ["site", "datetime"]
    df = df.sort_values(sort_cols).reset_index(drop=True)

    print("识别后的字段：")
    for k, v in colmap.items():
        print(f"  {k:12s} -> {v}")

    return df


# ============================================================
# 3. 站点筛选
# ============================================================

def filter_site(df):
    target = CONFIG["SITE"]

    if target is None:
        return df.copy()

    if "site" not in df.columns:
        print("警告：Excel没有站点列，因此不进行站点筛选。")
        return df.copy()

    if str(target).upper() == "ALL":
        return df.copy()

    sites = df["site"].dropna().unique().tolist()
    target_upper = str(target).upper()

    matched = [s for s in sites if str(s).upper() == target_upper]

    if not matched:
        raise ValueError(
            f"找不到站点 {target}。\n"
            f"Excel中的站点包括：{sites[:50]}"
        )

    out = df[df["site"].isin(matched)].copy()
    print(f"分析断面：{matched[0]}；数据量：{len(out)}")
    return out


# ============================================================
# 4. 低氧事件识别
# ============================================================

def identify_hypoxia_events(group):
    """
    按站点、时间排序。
    DO <= 2 mg/L 连续 >= 4 h 定义为低氧事件。
    注意：这里严格按实际时间间隔计算持续时间，
    而不是简单假设每个点间隔都等于4 h。
    """
    g = group.sort_values("datetime").copy()

    g["low_oxygen_point"] = (
        g["DO"] <= CONFIG["HYPOXIA_DO_THRESHOLD"]
    )

    # 相邻记录时间
    dt_h = g["datetime"].diff().dt.total_seconds() / 3600.0

    # 新事件：
    # 1) 第一个点
    # 2) 当前/前一个低氧状态不连续
    # 3) 时间间隔明显大于预期采样间隔
    new_event = (
        g["low_oxygen_point"] &
        (
            (~g["low_oxygen_point"].shift(1).fillna(False)) |
            (dt_h > CONFIG["EXPECTED_INTERVAL_HOURS"] * 1.5)
        )
    )

    g["event_group"] = new_event.cumsum()

    # 只有低氧点才属于事件
    g["hypoxia_event_id"] = np.nan

    event_counter = 0

    for grp_id, sub in g[g["low_oxygen_point"]].groupby("event_group"):
        if len(sub) == 0:
            continue

        duration_h = (
            sub["datetime"].iloc[-1] -
            sub["datetime"].iloc[0]
        ).total_seconds() / 3600.0

        # 对离散4 h数据：
        # 一个点本身代表一个4 h观测窗口，因此至少1个点也满足4 h。
        represented_duration = max(
            duration_h,
            CONFIG["EXPECTED_INTERVAL_HOURS"]
        )

        if represented_duration >= CONFIG["HYPOXIA_MIN_DURATION_HOURS"]:
            event_counter += 1
            g.loc[sub.index, "hypoxia_event_id"] = event_counter

    g["hypoxia_event"] = g["hypoxia_event_id"].notna().astype(int)

    return g


def add_hypoxia_events(df):
    if "site" in df.columns:
        df = (
            df.groupby("site", group_keys=False)
            .apply(identify_hypoxia_events)
            .reset_index(drop=True)
        )
    else:
        df = identify_hypoxia_events(df)

    print(
        f"低氧点数：{df['hypoxia_event'].sum():,.0f}；"
        f"低氧事件标签已生成。"
    )
    return df


# ============================================================
# 5. 快速下降事件
# ============================================================

def add_rapid_drop(df):
    def calc(g):
        g = g.sort_values("datetime").copy()
        g["DO_prev"] = g["DO"].shift(1)
        g["dt_h"] = (
            g["datetime"].diff().dt.total_seconds() / 3600
        )

        g["DO_drop"] = g["DO_prev"] - g["DO"]
        g["DO_drop_rate"] = g["DO_drop"] / g["dt_h"]

        g["rapid_drop"] = (
            (g["DO_drop"] >= CONFIG["RAPID_DROP_MAGNITUDE"]) &
            (g["DO_drop_rate"] >= CONFIG["RAPID_DROP_RATE"])
        ).astype(int)

        return g

    if "site" in df.columns:
        return (
            df.groupby("site", group_keys=False)
            .apply(calc)
            .reset_index(drop=True)
        )
    return calc(df)


# ============================================================
# 6. RF特征准备
# ============================================================

def get_rf_features(df):
    candidates = [
        "WT", "Q", "pH", "EC", "Turbidity",
        "CODMn", "NH3N", "TN", "TP",
        "Rain6h", "Rain24h", "Rain72h"
    ]

    features = [c for c in candidates if c in df.columns]

    if len(features) < 3:
        raise ValueError(
            f"RF可用环境变量少于3个：{features}\n"
            "请检查Excel列名或COLUMN_ALIASES。"
        )

    return features


def make_regression_model():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("rf", RandomForestRegressor(
            n_estimators=CONFIG["N_ESTIMATORS"],
            max_depth=CONFIG["MAX_DEPTH"],
            min_samples_leaf=CONFIG["MIN_SAMPLES_LEAF"],
            random_state=CONFIG["RANDOM_STATE"],
            n_jobs=CONFIG["N_JOBS"]
        ))
    ])


def make_classification_model():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("rf", RandomForestClassifier(
            n_estimators=CONFIG["N_ESTIMATORS"],
            max_depth=CONFIG["MAX_DEPTH"],
            min_samples_leaf=CONFIG["MIN_SAMPLES_LEAF"],
            class_weight="balanced",
            random_state=CONFIG["RANDOM_STATE"],
            n_jobs=CONFIG["N_JOBS"]
        ))
    ])


# ============================================================
# 7. LOYO RF回归
# ============================================================

def run_rf_regression(df, features):
    print("\n" + "=" * 70)
    print("A. RF回归：Leave-One-Year-Out")
    print("=" * 70)

    data = df.dropna(subset=["DO"]).copy()
    data["year"] = data["datetime"].dt.year

    years = sorted(data["year"].dropna().unique())

    yearly_results = []
    importance_rows = []

    for test_year in years:
        train = data[data["year"] != test_year].copy()
        test = data[data["year"] == test_year].copy()

        if len(train) < 50 or len(test) < 10:
            print(f"{test_year}: 数据不足，跳过")
            continue

        X_train = train[features]
        y_train = train["DO"]
        X_test = test[features]
        y_test = test["DO"]

        model = make_regression_model()
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        r2 = r2_score(y_test, pred)
        rmse = np.sqrt(mean_squared_error(y_test, pred))
        mae = mean_absolute_error(y_test, pred)

        yearly_results.append({
            "test_year": test_year,
            "n_train": len(train),
            "n_test": len(test),
            "R2": r2,
            "RMSE": rmse,
            "MAE": mae
        })

        rf = model.named_steps["rf"]
        imp = rf.feature_importances_

        for feature, value in zip(features, imp):
            importance_rows.append({
                "test_year": test_year,
                "feature": feature,
                "importance": value
            })

        print(
            f"{test_year}: "
            f"R²={r2:.3f}, RMSE={rmse:.3f}, MAE={mae:.3f}"
        )

    results = pd.DataFrame(yearly_results)
    importance = pd.DataFrame(importance_rows)

    if not results.empty:
        summary = pd.DataFrame([{
            "test_year": "Mean",
            "n_train": results["n_train"].mean(),
            "n_test": results["n_test"].mean(),
            "R2": results["R2"].mean(),
            "RMSE": results["RMSE"].mean(),
            "MAE": results["MAE"].mean()
        }, {
            "test_year": "SD",
            "n_train": results["n_train"].std(),
            "n_test": results["n_test"].std(),
            "R2": results["R2"].std(),
            "RMSE": results["RMSE"].std(),
            "MAE": results["MAE"].std()
        }])

        results_with_summary = pd.concat(
            [results, summary],
            ignore_index=True
        )
    else:
        results_with_summary = results

    return results_with_summary, importance


# ============================================================
# 8. LOYO RF分类
# ============================================================

def safe_specificity(y_true, y_pred):
    cm = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    )

    tn, fp, fn, tp = cm.ravel()

    if (tn + fp) == 0:
        return np.nan

    return tn / (tn + fp)


def run_rf_classification(df, features):
    print("\n" + "=" * 70)
    print("B. RF分类：Leave-One-Year-Out")
    print("=" * 70)

    data = df.dropna(subset=["hypoxia_event"]).copy()
    data["year"] = data["datetime"].dt.year

    years = sorted(data["year"].dropna().unique())

    yearly_results = []
    importance_rows = []

    for test_year in years:
        train = data[data["year"] != test_year].copy()
        test = data[data["year"] == test_year].copy()

        if len(train) < 50 or len(test) < 10:
            print(f"{test_year}: 数据不足，跳过")
            continue

        # 训练集和测试集都必须同时包含0/1
        if train["hypoxia_event"].nunique() < 2:
            print(f"{test_year}: 训练集没有同时包含0/1，跳过")
            continue

        if test["hypoxia_event"].nunique() < 2:
            print(
                f"{test_year}: 测试集只有一个类别，"
                "AUC不可计算；仍保留分类结果。"
            )

        X_train = train[features]
        y_train = train["hypoxia_event"].astype(int)

        X_test = test[features]
        y_test = test["hypoxia_event"].astype(int)

        model = make_classification_model()
        model.fit(X_train, y_train)

        prob = model.predict_proba(X_test)[:, 1]
        pred_class = (prob >= 0.5).astype(int)

        auc = (
            roc_auc_score(y_test, prob)
            if y_test.nunique() == 2 else np.nan
        )

        accuracy = accuracy_score(y_test, pred_class)
        sensitivity = recall_score(
            y_test, pred_class, zero_division=0
        )
        specificity = safe_specificity(
            y_test, pred_class
        )
        f1 = f1_score(
            y_test, pred_class, zero_division=0
        )

        yearly_results.append({
            "test_year": test_year,
            "n_train": len(train),
            "n_test": len(test),
            "positive_test": int(y_test.sum()),
            "AUC": auc,
            "Accuracy": accuracy,
            "Sensitivity": sensitivity,
            "Specificity": specificity,
            "F1": f1
        })

        rf = model.named_steps["rf"]
        imp = rf.feature_importances_

        for feature, value in zip(features, imp):
            importance_rows.append({
                "test_year": test_year,
                "feature": feature,
                "importance": value
            })

        print(
            f"{test_year}: "
            f"AUC={auc:.3f} | "
            f"Sens={sensitivity:.3f} | "
            f"Spec={specificity:.3f}"
        )

    results = pd.DataFrame(yearly_results)
    importance = pd.DataFrame(importance_rows)

    if not results.empty:
        numeric_cols = [
            "n_train", "n_test", "positive_test",
            "AUC", "Accuracy", "Sensitivity",
            "Specificity", "F1"
        ]

        mean_row = {"test_year": "Mean"}
        sd_row = {"test_year": "SD"}

        for c in numeric_cols:
            mean_row[c] = results[c].mean()
            sd_row[c] = results[c].std()

        results_with_summary = pd.concat(
            [results, pd.DataFrame([mean_row, sd_row])],
            ignore_index=True
        )
    else:
        results_with_summary = results

    return results_with_summary, importance


# ============================================================
# 9. 变量重要性汇总
# ============================================================

def summarize_importance(importance):
    if importance.empty:
        return importance

    summary = (
        importance
        .groupby("feature")["importance"]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={
            "mean": "importance_mean",
            "std": "importance_sd"
        })
        .sort_values("importance_mean", ascending=False)
    )

    return summary


# ============================================================
# 10. PDP
# ============================================================

def make_pdp_plots(df, features, importance_summary, output_dir):
    if not CONFIG["MAKE_PDP"]:
        return

    print("\n" + "=" * 70)
    print("C. PDP：关键变量非线性响应")
    print("=" * 70)

    if importance_summary.empty:
        print("没有importance结果，跳过PDP。")
        return

    selected = (
        importance_summary["feature"]
        .head(CONFIG["N_PDP_FEATURES"])
        .tolist()
    )

    # 用全数据拟合一个描述性模型。
    # 注意：这张PDP不是LOYO性能验证本身。
    model = make_regression_model()
    model.fit(df[features], df["DO"])

    for feature in selected:
        try:
            fig, ax = plt.subplots(figsize=(6.5, 4.5))
            PartialDependenceDisplay.from_estimator(
                model,
                df[features],
                [feature],
                ax=ax
            )
            ax.set_title(
                f"Partial dependence of DO on {feature}"
            )
            fig.tight_layout()
            fig.savefig(
                output_dir / f"PDP_{feature}.png",
                dpi=300,
                bbox_inches="tight"
            )
            plt.close(fig)
        except Exception as e:
            print(f"PDP {feature}失败：{e}")


# ============================================================
# 11. SFI计算
# ============================================================

def calculate_sfi_for_group(g):
    g = g.sort_values("datetime").copy()

    # 只允许当前及过去数据
    # 24 h对应6个4h间隔。
    lookback = int(
        CONFIG["SFI_LOOKBACK_HOURS"] /
        CONFIG["EXPECTED_INTERVAL_HOURS"]
    )

    # OI：过去24h DO一阶差分绝对值均值
    diff = g["DO"].diff().abs()
    g["OI"] = diff.rolling(
        window=lookback,
        min_periods=max(2, lookback // 2)
    ).mean()

    # base：expanding window，仅用当前时刻之前的非低氧观测逐步更新
    # 与英文稿件 §2.4 一致："calculated at each time point using only
    # non-hypoxic observations available before that time point and
    # were updated independently at each time step"
    if CONFIG["SFI_BASE_METHOD"] == "non_hypoxia_median":
        if "hypoxia_event" in g.columns:
            hypoxia_mask = g["hypoxia_event"] == 0
        else:
            hypoxia_mask = pd.Series(True, index=g.index)
    else:
        hypoxia_mask = pd.Series(True, index=g.index)

    # 非低氧时刻保留原值，低氧时刻置NaN
    # shift(1)确保严格"当前时刻之前"；expanding().median()自动跳过NaN
    cod_nonhyp = g["CODMn"].where(hypoxia_mask)
    nh3_nonhyp = g["NH3N"].where(hypoxia_mask)
    cod_base_s = cod_nonhyp.shift(1).expanding(min_periods=2).median()
    nh3_base_s = nh3_nonhyp.shift(1).expanding(min_periods=2).median()

    # 兜底：初始点数据不足时使用全局非低氧中位数
    cod_global = g.loc[hypoxia_mask, "CODMn"].median()
    nh3_global = g.loc[hypoxia_mask, "NH3N"].median()
    if not np.isfinite(cod_global) or cod_global <= 0:
        cod_global = g["CODMn"].median()
    if not np.isfinite(nh3_global) or nh3_global <= 0:
        nh3_global = g["NH3N"].median()
    cod_base_s = cod_base_s.fillna(cod_global)
    nh3_base_s = nh3_base_s.fillna(nh3_global)
    cod_base_s = cod_base_s.where(cod_base_s > 0, cod_global)
    nh3_base_s = nh3_base_s.where(nh3_base_s > 0, nh3_global)

    # RR按照当前JEM稿件公式：
    # ln(CODMn/base + NH3-N/base)
    ratio = (
        g["CODMn"] / cod_base_s +
        g["NH3N"] / nh3_base_s
    )

    ratio = ratio.replace([np.inf, -np.inf], np.nan)

    # RR = max{ln(...), 0}，与稿件公式一致（低于背景值截断为0）
    g["RR"] = np.log(
        ratio.where(ratio > 0)
    ).clip(lower=0)

    g["SFI"] = g["OI"] * g["RR"]

    return g


def calculate_sfi(df):
    print("\n" + "=" * 70)
    print("D. SFI计算")
    print("=" * 70)

    if "CODMn" not in df.columns or "NH3N" not in df.columns:
        raise ValueError(
            "SFI需要CODMn和NH3N两列，但当前数据没有同时识别到。"
        )

    if "site" in df.columns:
        out = (
            df.groupby("site", group_keys=False)
            .apply(calculate_sfi_for_group)
            .reset_index(drop=True)
        )
    else:
        out = calculate_sfi_for_group(df)

    valid = out["SFI"].notna().sum()
    print(f"有效SFI点数：{valid:,}")
    print(
        f"SFI P90 = {out['SFI'].quantile(0.90):.6f}"
    )

    return out


# ============================================================
# 12. 未来24/48 h低氧标签
# ============================================================

def add_future_hypoxia_label_group(g, future_hours):
    g = g.sort_values("datetime").copy()

    event_times = (
        g.loc[
            g["hypoxia_event"] == 1,
            "datetime"
        ]
        .dropna()
        .sort_values()
        .values
    )

    if len(event_times) == 0:
        g[f"future_{future_hours}h_low"] = 0
        return g

    current_times = g["datetime"].values

    labels = []

    for t in current_times:
        t0 = pd.Timestamp(t)
        t1 = t0 + pd.Timedelta(hours=future_hours)

        # 严格预测未来，不把当前时刻作为未来
        exists = np.any(
            (event_times > np.datetime64(t0)) &
            (event_times <= np.datetime64(t1))
        )

        labels.append(int(exists))

    g[f"future_{future_hours}h_low"] = labels
    return g


def add_future_labels(df):
    for hours in CONFIG["SFI_FUTURE_HOURS"]:
        if "site" in df.columns:
            df = (
                df.groupby("site", group_keys=False)
                .apply(
                    lambda g: add_future_hypoxia_label_group(
                        g, hours
                    )
                )
                .reset_index(drop=True)
            )
        else:
            df = add_future_hypoxia_label_group(
                df, hours
            )

    return df


# ============================================================
# 13. SFI ROC
# ============================================================

def roc_statistics(y_true, score):
    mask = (
        pd.notna(y_true) &
        pd.notna(score)
    )

    y = np.asarray(y_true[mask]).astype(int)
    s = np.asarray(score[mask]).astype(float)

    if len(y) < CONFIG["ROC_MIN_SAMPLES"]:
        return None

    if len(np.unique(y)) < 2:
        return None

    auc = roc_auc_score(y, s)
    fpr, tpr, thresholds = roc_curve(y, s)

    j = tpr - fpr
    idx = np.nanargmax(j)

    best_threshold = thresholds[idx]
    sensitivity = tpr[idx]
    specificity = 1 - fpr[idx]

    return {
        "AUC": auc,
        "Youden_threshold": best_threshold,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "n": len(y),
        "positive": int(y.sum())
    }


def run_sfi_overall_roc(df):
    print("\n" + "=" * 70)
    print("E. SFI总体ROC：未来24/48 h")
    print("=" * 70)

    rows = []

    for hours in CONFIG["SFI_FUTURE_HOURS"]:
        label = f"future_{hours}h_low"

        res = roc_statistics(
            df[label],
            df["SFI"]
        )

        if res is None:
            print(f"{hours} h：无法计算ROC")
            continue

        res["horizon_h"] = hours
        rows.append(res)

        print(
            f"{hours} h: "
            f"AUC={res['AUC']:.3f}, "
            f"threshold={res['Youden_threshold']:.6f}, "
            f"Sens={res['Sensitivity']:.3f}, "
            f"Spec={res['Specificity']:.3f}"
        )

        # ROC图
        mask = (
            df[label].notna() &
            df["SFI"].notna()
        )

        y = df.loc[mask, label].astype(int)
        s = df.loc[mask, "SFI"].astype(float)

        fpr, tpr, _ = roc_curve(y, s)

        fig, ax = plt.subplots(figsize=(5.5, 5.0))
        ax.plot(
            fpr, tpr,
            linewidth=2,
            label=f"AUC = {res['AUC']:.3f}"
        )
        ax.plot(
            [0, 1], [0, 1],
            linestyle="--",
            linewidth=1
        )
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.set_title(
            f"SFI prediction of hypoxia within {hours} h"
        )
        ax.legend()
        fig.tight_layout()
        fig.savefig(
            Path(CONFIG["OUTPUT_DIR"]) /
            f"SFI_ROC_{hours}h.png",
            dpi=300,
            bbox_inches="tight"
        )
        plt.close(fig)

    return pd.DataFrame(rows)


# ============================================================
# 14. SFI LOYO ROC
# ============================================================

def run_sfi_loyo_roc(df):
    print("\n" + "=" * 70)
    print("F. SFI LOYO时间外推ROC")
    print("=" * 70)

    data = df.copy()
    data["year"] = data["datetime"].dt.year

    rows = []

    for hours in CONFIG["SFI_FUTURE_HOURS"]:
        label = f"future_{hours}h_low"

        for year in sorted(data["year"].dropna().unique()):
            test = data[data["year"] == year].copy()

            res = roc_statistics(
                test[label],
                test["SFI"]
            )

            if res is None:
                print(
                    f"{year} / {hours}h："
                    "样本或类别不足，AUC=NA"
                )
                continue

            rows.append({
                "test_year": year,
                "horizon_h": hours,
                **res
            })

            print(
                f"{year} / {hours}h: "
                f"AUC={res['AUC']:.3f}"
            )

    result = pd.DataFrame(rows)

    if not result.empty:
        summary_rows = []

        for hours in CONFIG["SFI_FUTURE_HOURS"]:
            sub = result[
                result["horizon_h"] == hours
            ]

            if len(sub) == 0:
                continue

            summary_rows.append({
                "test_year": "Mean",
                "horizon_h": hours,
                "AUC": sub["AUC"].mean(),
                "AUC_SD": sub["AUC"].std(),
                "Sensitivity": sub["Sensitivity"].mean(),
                "Sensitivity_SD": sub["Sensitivity"].std(),
                "Specificity": sub["Specificity"].mean(),
                "Specificity_SD": sub["Specificity"].std(),
                "n_years": len(sub)
            })

        summary = pd.DataFrame(summary_rows)

        result = pd.concat(
            [result, summary],
            ignore_index=True,
            sort=False
        )

    return result


# ============================================================
# 15. SFI P90风险比较
# ============================================================

def sfi_p90_risk_summary(df):
    p90 = df["SFI"].quantile(0.90)

    out = []

    for hours in CONFIG["SFI_FUTURE_HOURS"]:
        label = f"future_{hours}h_low"

        high = df.loc[
            df["SFI"] >= p90, label
        ].dropna()

        background = df.loc[
            df["SFI"] < p90, label
        ].dropna()

        high_risk = high.mean() if len(high) else np.nan
        background_risk = (
            background.mean()
            if len(background)
            else np.nan
        )

        risk_ratio = (
            high_risk / background_risk
            if background_risk and background_risk > 0
            else np.nan
        )

        out.append({
            "horizon_h": hours,
            "P90_threshold": p90,
            "high_SFI_risk": high_risk,
            "background_risk": background_risk,
            "risk_ratio": risk_ratio,
            "n_high": len(high),
            "n_background": len(background)
        })

    return pd.DataFrame(out)


# ============================================================
# 16. 保存Excel
# ============================================================

def save_results(
    df,
    rf_reg_results,
    rf_reg_importance,
    rf_reg_importance_summary,
    rf_cls_results,
    rf_cls_importance,
    rf_cls_importance_summary,
    sfi_roc,
    sfi_loyo,
    sfi_risk
):
    output_dir = Path(CONFIG["OUTPUT_DIR"])
    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # 完整时间序列
    df.to_csv(
        output_dir / "01_processed_timeseries.csv",
        index=False,
        encoding="utf-8-sig"
    )

    rf_reg_results.to_csv(
        output_dir / "02_RF_regression_LOYO.csv",
        index=False,
        encoding="utf-8-sig"
    )

    rf_reg_importance.to_csv(
        output_dir / "03_RF_regression_importance_by_year.csv",
        index=False,
        encoding="utf-8-sig"
    )

    rf_reg_importance_summary.to_csv(
        output_dir / "04_RF_regression_importance_summary.csv",
        index=False,
        encoding="utf-8-sig"
    )

    rf_cls_results.to_csv(
        output_dir / "05_RF_classification_LOYO.csv",
        index=False,
        encoding="utf-8-sig"
    )

    rf_cls_importance.to_csv(
        output_dir / "06_RF_classification_importance_by_year.csv",
        index=False,
        encoding="utf-8-sig"
    )

    rf_cls_importance_summary.to_csv(
        output_dir / "07_RF_classification_importance_summary.csv",
        index=False,
        encoding="utf-8-sig"
    )

    sfi_roc.to_csv(
        output_dir / "08_SFI_overall_ROC.csv",
        index=False,
        encoding="utf-8-sig"
    )

    sfi_loyo.to_csv(
        output_dir / "09_SFI_LOYO_ROC.csv",
        index=False,
        encoding="utf-8-sig"
    )

    sfi_risk.to_csv(
        output_dir / "10_SFI_P90_risk_summary.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # Excel总表
    excel_path = output_dir / "validation_results.xlsx"

    with pd.ExcelWriter(
        excel_path,
        engine="openpyxl"
    ) as writer:

        rf_reg_results.to_excel(
            writer,
            sheet_name="RF_reg_LOYO",
            index=False
        )

        rf_reg_importance.to_excel(
            writer,
            sheet_name="RF_reg_importance_year",
            index=False
        )

        rf_reg_importance_summary.to_excel(
            writer,
            sheet_name="RF_reg_importance",
            index=False
        )

        rf_cls_results.to_excel(
            writer,
            sheet_name="RF_cls_LOYO",
            index=False
        )

        rf_cls_importance.to_excel(
            writer,
            sheet_name="RF_cls_importance_year",
            index=False
        )

        rf_cls_importance_summary.to_excel(
            writer,
            sheet_name="RF_cls_importance",
            index=False
        )

        sfi_roc.to_excel(
            writer,
            sheet_name="SFI_ROC",
            index=False
        )

        sfi_loyo.to_excel(
            writer,
            sheet_name="SFI_LOYO",
            index=False
        )

        sfi_risk.to_excel(
            writer,
            sheet_name="SFI_P90",
            index=False
        )

    print("\n结果已保存：")
    print(output_dir)
    print(excel_path)


# ============================================================
# 17. 主程序
# ============================================================

def main():

    output_dir = Path(CONFIG["OUTPUT_DIR"])
    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # 1. 读取
    df = load_data()

    # 2. 站点
    df = filter_site(df)

    # 如果SITE=ALL，下面会把多个站点混在一起。
    # 对严格JEM分析，建议一次运行一个站点。
    if (
        CONFIG["SITE"] == "ALL" and
        "site" in df.columns
    ):
        print(
            "\n警告：当前SITE=ALL，多个站点将进入同一模型。"
            "如果论文需要断面独立模型，请逐断面运行。"
        )

    # 3. 低氧
    df = add_hypoxia_events(df)

    # 4. 快速下降
    df = add_rapid_drop(df)

    # 5. RF features
    features = get_rf_features(df)

    print("\nRF输入变量：")
    for f in features:
        print("  -", f)

    # 6. RF regression
    reg_results, reg_imp = run_rf_regression(
        df, features
    )
    reg_imp_summary = summarize_importance(reg_imp)

    # 7. RF classification
    cls_results, cls_imp = run_rf_classification(
        df, features
    )
    cls_imp_summary = summarize_importance(cls_imp)

    # 8. PDP
    make_pdp_plots(
        df,
        features,
        reg_imp_summary,
        output_dir
    )

    # 9. SFI
    required_sfi = {"CODMn", "NH3N"}
    if required_sfi.issubset(df.columns):
        df = calculate_sfi(df)

        # 10. future labels
        df = add_future_labels(df)

        # 11. ROC
        sfi_roc = run_sfi_overall_roc(df)

        # 12. LOYO
        sfi_loyo = run_sfi_loyo_roc(df)

        # 13. P90
        sfi_risk = sfi_p90_risk_summary(df)

    else:
        print(
            "\n警告：CODMn或NH3N缺失，无法计算SFI。"
        )
        sfi_roc = pd.DataFrame()
        sfi_loyo = pd.DataFrame()
        sfi_risk = pd.DataFrame()

    # 14. 保存
    save_results(
        df,
        reg_results,
        reg_imp,
        reg_imp_summary,
        cls_results,
        cls_imp,
        cls_imp_summary,
        sfi_roc,
        sfi_loyo,
        sfi_risk
    )

    # 15. 控制台核心摘要
    print("\n" + "=" * 70)
    print("分析完成")
    print("=" * 70)

    print("\n【RF回归 LOYO】")
    print(reg_results)

    print("\n【RF分类 LOYO】")
    print(cls_results)

    print("\n【RF回归变量重要性】")
    print(reg_imp_summary)

    print("\n【RF分类变量重要性】")
    print(cls_imp_summary)

    print("\n【SFI总体ROC】")
    print(sfi_roc)

    print("\n【SFI LOYO ROC】")
    print(sfi_loyo)

    print("\n【SFI P90风险】")
    print(sfi_risk)

    print("\n请重点检查：")
    print("1. 每个测试年份的AUC/R²是否稳定")
    print("2. pH、WT、TP等关键变量重要性是否跨年份稳定")
    print("3. PDP风险窗口是否跨年份重复出现")
    print("4. SFI 24 h / 48 h AUC是否明显高于0.5")
    print("5. Youden阈值是否具有跨年份稳定性")
    print("6. 不要直接把脚本中的示例阈值写入论文，必须使用实际输出结果")


if __name__ == "__main__":
    main()
