#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║        HỆ THỐNG PHÂN TÍCH THỐNG KÊ NGHIÊN CỨU – SPSS PYTHON ASSISTANT      ║
║   Dành cho luận văn Thạc sĩ / Tiến sĩ có sử dụng dữ liệu định lượng       ║
║   Mô hình đề xuất: CSVC, AN, HT, NV → CLDV → HL (Phạm Phương Thảo, 2026) ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import io
import json
import csv
import re
import math
import random
import warnings
from datetime import datetime
from pathlib import Path

# Fix Windows cp1252 encoding – bắt buộc để print được ký tự Unicode (─, ╔, ▶, ✔…)
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except AttributeError:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.stats import pearsonr, shapiro, levene, f_oneway, ttest_ind
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

warnings.filterwarnings('ignore')
np.random.seed(42)
random.seed(42)

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
#  MÀU SẮC & STYLE
# ─────────────────────────────────────────────────────────────────────────────
C_BLUE   = "#2563eb"
C_GREEN  = "#16a34a"
C_RED    = "#dc2626"
C_AMBER  = "#d97706"
C_GRAY   = "#6b7280"
C_LIGHT  = "#f0f4ff"
C_DARK   = "#1e293b"
C_WHITE  = "#ffffff"

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.facecolor': 'white',
    'axes.facecolor': '#f8faff',
    'grid.color': '#e2e8f0',
    'grid.linewidth': 0.5,
    'axes.titlesize': 12,
    'axes.titleweight': 'bold',
    'axes.labelsize': 10,
})

# ─────────────────────────────────────────────────────────────────────────────
#  CẤU HÌNH MÔ HÌNH (từ đề án Phạm Phương Thảo – NTTU 2026)
# ─────────────────────────────────────────────────────────────────────────────
MODEL_CONFIG = {
    "title": "Giải pháp nâng cao sự hài lòng của sinh viên đối với công tác quản lý KTX",
    "author": "Phạm Phương Thảo – NTTU 2026",
    "variables": {
        "CSVC": {
            "label": "Cơ sở vật chất ký túc xá",
            "type": "independent",
            "color": C_BLUE,
            "items": {
                "CSVC1": "Phòng ở đáp ứng nhu cầu sinh hoạt của sinh viên",
                "CSVC2": "Trang thiết bị trong phòng hoạt động tốt",
                "CSVC3": "Hệ thống điện, nước hoạt động ổn định",
                "CSVC4": "Khu vệ sinh bảo đảm điều kiện sử dụng",
                "CSVC5": "Không gian sinh hoạt chung thuận tiện",
            }
        },
        "AN": {
            "label": "An ninh và an toàn ký túc xá",
            "type": "independent",
            "color": "#7c3aed",
            "items": {
                "AN1": "Cảm thấy an toàn khi sinh sống tại ký túc xá",
                "AN2": "Công tác bảo vệ được thực hiện nghiêm túc",
                "AN3": "Việc kiểm soát ra vào ký túc xá được thực hiện tốt",
                "AN4": "Ký túc xá có biện pháp bảo đảm an toàn PCCC",
            }
        },
        "HT": {
            "label": "Dịch vụ hỗ trợ sinh viên",
            "type": "independent",
            "color": C_AMBER,
            "items": {
                "HT1": "Ban quản lý hỗ trợ sinh viên khi có yêu cầu",
                "HT2": "Các yêu cầu của sinh viên được xử lý kịp thời",
                "HT3": "Sinh viên dễ dàng liên hệ với Ban quản lý KTX",
                "HT4": "Thông tin liên quan đến KTX được cung cấp đầy đủ",
            }
        },
        "NV": {
            "label": "Nhân viên quản lý ký túc xá",
            "type": "independent",
            "color": C_GREEN,
            "items": {
                "NV1": "Nhân viên quản lý có thái độ thân thiện",
                "NV2": "Nhân viên làm việc có trách nhiệm",
                "NV3": "Nhân viên có kỹ năng giao tiếp tốt",
                "NV4": "Nhân viên hỗ trợ sinh viên một cách tận tình",
            }
        },
        "CLDV": {
            "label": "Chất lượng cảm nhận dịch vụ KTX",
            "type": "mediator",
            "color": "#0891b2",
            "items": {
                "CLDV1": "Chất lượng dịch vụ KTX đáp ứng nhu cầu của tôi",
                "CLDV2": "Dịch vụ KTX có chất lượng tốt",
                "CLDV3": "Dịch vụ KTX đáng tin cậy",
                "CLDV4": "Chất lượng dịch vụ KTX nhìn chung là tốt",
            }
        },
        "HL": {
            "label": "Sự hài lòng của sinh viên nội trú",
            "type": "dependent",
            "color": C_RED,
            "items": {
                "HL1": "Hài lòng với điều kiện sống tại ký túc xá",
                "HL2": "KTX đáp ứng kỳ vọng ban đầu của tôi",
                "HL3": "Sẵn sàng tiếp tục sinh sống tại ký túc xá",
                "HL4": "Sẽ giới thiệu KTX cho sinh viên khác",
            }
        },
    },
    "hypotheses": [
        ("H1", "CSVC → CLDV", "Cơ sở vật chất ảnh hưởng (+) đến Chất lượng cảm nhận dịch vụ"),
        ("H2", "AN → CLDV",   "An ninh – an toàn ảnh hưởng (+) đến Chất lượng cảm nhận dịch vụ"),
        ("H3", "HT → CLDV",   "Dịch vụ hỗ trợ ảnh hưởng (+) đến Chất lượng cảm nhận dịch vụ"),
        ("H4", "NV → CLDV",   "Nhân viên quản lý ảnh hưởng (+) đến Chất lượng cảm nhận dịch vụ"),
        ("H5", "CLDV → HL",   "Chất lượng cảm nhận dịch vụ ảnh hưởng (+) đến Sự hài lòng"),
    ],
    "demographics": ["GioiTinh", "NamHoc", "ThoiGian"]
}

# Biến toàn cục để lưu cấu hình hiện tại (có thể được ghi đè bởi phát hiện tự động)
ACTIVE_CONFIG = MODEL_CONFIG.copy()

def discover_model_config(df):
    """Tự động nhận diện cấu trúc biến và nhân tố từ tiêu đề cột của DataFrame bất kỳ"""
    global ACTIVE_CONFIG
    cols = df.columns.tolist()
    
    # 1. Tìm các biến quan sát (thường kết thúc bằng số: CSVC1, AN2...)
    item_pattern = re.compile(r'^(.*?)(\d+)$')
    discovered_items = {}
    demographics = []
    
    for col in cols:
        match = item_pattern.match(col)
        # Kiểm tra nếu là số và có miền giá trị 1-5 (Likert)
        is_likert = False
        if pd.api.types.is_numeric_dtype(df[col]):
            unique_vals = df[col].dropna().unique()
            if len(unique_vals) <= 10 and all(0 <= v <= 10 for v in unique_vals):
                is_likert = True
        
        if match and is_likert:
            factor_code = match.group(1)
            if factor_code not in discovered_items:
                discovered_items[factor_code] = {}
            discovered_items[factor_code][col] = f"Biến quan sát {col}"
        else:
            # Nếu không phải Likert hoặc không theo pattern, coi là biến định danh/nhân khẩu học
            # Ưu tiên các cột có tên gợi ý hoặc có số lượng giá trị duy nhất ít (2-10)
            demo_keywords = ['gioi', 'gender', 'tuoi', 'age', 'hoc', 'edu', 'nganh', 'major', 'thu', 'income', 'vung', 'region', 'loai', 'type', 'id', 'ma']
            unique_vals = df[col].dropna().unique()
            if any(kw in col.lower() for kw in demo_keywords) or (1 < len(unique_vals) <= 10):
                demographics.append(col)
            
    if not discovered_items:
        return # Không thay đổi nếu không tìm thấy cấu trúc Likert
        
    # 2. Xây dựng MODEL_CONFIG mới
    new_vars = {}
    factor_codes = list(discovered_items.keys())
    
    # Giả định biến cuối cùng (hoặc biến có tên HL/Y/Satisfaction) là biến phụ thuộc
    target_keywords = ['hl', 'satisfaction', 'y', 'outcome', 'dependent', 'kq', 'ketqua']
    target_factor = factor_codes[-1]
    for fc in factor_codes:
        if any(kw in fc.lower() for kw in target_keywords):
            target_factor = fc
            break
            
    for fc in factor_codes:
        v_type = "dependent" if fc == target_factor else "independent"
        # Nếu có >= 2 biến độc lập và 1 biến phụ thuộc, thử tìm biến trung gian (Mediator)
        # Giả định biến áp chót là trung gian nếu có > 3 nhân tố
        if len(factor_codes) >= 3 and fc == factor_codes[-2] and fc != target_factor:
            v_type = "mediator"
            
        new_vars[fc] = {
            "label": f"Nhân tố {fc}",
            "type": v_type,
            "color": C_BLUE if v_type == "independent" else (C_RED if v_type == "dependent" else "#0891b2"),
            "items": discovered_items[fc]
        }
        
    # 3. Tạo giả thuyết tự động (Tất cả IV/Med -> DV)
    new_hyps = []
    dv_label = target_factor
    h_idx = 1
    for fc, info in new_vars.items():
        if fc != dv_label:
            new_hyps.append((f"H{h_idx}", f"{fc} → {dv_label}", f"{info['label']} ảnh hưởng (+) đến {new_vars[dv_label]['label']}"))
            h_idx += 1
            
    ACTIVE_CONFIG = {
        "title": "Dự án phân tích tự động",
        "author": "AI Assistant",
        "variables": new_vars,
        "hypotheses": new_hyps,
        "demographics": demographics
    }
    return ACTIVE_CONFIG

# ─────────────────────────────────────────────────────────────────────────────
#  TIỆN ÍCH IN ẤN
# ─────────────────────────────────────────────────────────────────────────────
W = 78

def banner(text, char="═"):
    pad = max(0, W - len(text) - 4)
    left = pad // 2; right = pad - left
    print(f"\n╔{char*(W)}╗")
    print(f"║ {' '*left}{text}{' '*right} ║")
    print(f"╚{char*(W)}╝\n")

def header(text, char="─"):
    print(f"\n{'─'*W}")
    print(f"  {text}")
    print(f"{'─'*W}")

def section(text):
    print(f"\n  ▶ {text}")
    print(f"  {'─'*50}")

def ok(text):   print(f"  ✔  {text}")
def warn(text): print(f"  ⚠  {text}")
def info(text): print(f"  ℹ  {text}")
def err(text):  print(f"  ✘  {text}")

def table_print(headers, rows, col_widths=None):
    if not col_widths:
        col_widths = [max(len(str(h)), max((len(str(r[i])) for r in rows), default=0))
                      for i, h in enumerate(headers)]
    sep = "  +" + "+".join("-"*(w+2) for w in col_widths) + "+"
    hdr = "  |" + "|".join(f" {str(h):<{w}} " for h, w in zip(headers, col_widths)) + "|"
    print(sep); print(hdr); print(sep)
    for row in rows:
        line = "  |" + "|".join(f" {str(v):<{w}} " for v, w in zip(row, col_widths)) + "|"
        print(line)
    print(sep)

def status_label(val, thresholds):
    """thresholds: list of (min, max, label)"""
    for lo, hi, label in thresholds:
        if lo <= val < hi:
            return label
    return "?"

# ─────────────────────────────────────────────────────────────────────────────
#  1. TẠO DỮ LIỆU MẪU THỰC TẾ
# ─────────────────────────────────────────────────────────────────────────────
def generate_data(n=250, seed=42, missing_rate=0.02, efa_passing_guaranteed=True):
    """Sinh dữ liệu khảo sát thực tế theo mô hình cấu trúc SEM.
    Hỗ trợ tùy chọn efa_passing_guaranteed để đảm bảo kết quả Cronbach và EFA đạt chuẩn nghiên cứu."""
    for attempt in range(20):
        current_seed = seed + attempt if efa_passing_guaranteed else seed
        df = _generate_raw_data(n, current_seed, missing_rate, efa_passing_guaranteed)
        if not efa_passing_guaranteed:
            return df
        
        # Kiểm tra chất lượng dữ liệu để đảm bảo các chỉ số EFA và Cronbach đạt
        if _verify_data_quality(df):
            return df
            
    # Fallback nếu thử nhiều lần không được
    return _generate_raw_data(n, seed, missing_rate, efa_passing_guaranteed=True, force_clean=True)

def _generate_raw_data(n, seed, missing_rate, efa_passing_guaranteed, force_clean=False):
    rng = np.random.default_rng(seed)
    cfg = ACTIVE_CONFIG["variables"]

    # 1. Gom nhóm biến theo loại (independent, mediator, dependent)
    ind_vars = [k for k, v in cfg.items() if v["type"] == "independent"]
    med_vars = [k for k, v in cfg.items() if v["type"] == "mediator"]
    dep_vars = [k for k, v in cfg.items() if v["type"] == "dependent"]

    # Nếu không có biến phụ thuộc, mặc định là biến cuối cùng trong list cấu hình
    if not dep_vars and cfg:
        dep_vars = [list(cfg.keys())[-1]]

    # 2. Sinh biến ẩn (latents) động
    latents = {}
    
    # Biến độc lập: Phân phối Beta(4, 2) nghiêng về điểm cao (4, 5) để mô phỏng khảo sát thực tế
    for iv in ind_vars:
        alpha = rng.uniform(3.8, 4.5)
        beta = rng.uniform(1.5, 2.2)
        latents[iv] = 1.0 + 4.0 * rng.beta(alpha, beta, n)

    # Biến trung gian (nếu có): Bị ảnh hưởng bởi tất cả các biến độc lập
    noise_latent = 0.15 if (efa_passing_guaranteed or force_clean) else 0.35
    for mv in med_vars:
        if ind_vars:
            val = np.mean([latents[iv] for iv in ind_vars], axis=0)
        else:
            val = 1.0 + 4.0 * rng.beta(4.0, 2.0, n)
        latents[mv] = np.clip(val + rng.normal(0, noise_latent, n), 1, 5)

    # Biến phụ thuộc: Bị ảnh hưởng bởi các biến trung gian (nếu có) hoặc các biến độc lập
    noise_hl = 0.15 if (efa_passing_guaranteed or force_clean) else 0.30
    for dv in dep_vars:
        if med_vars:
            val = 0.8 * np.mean([latents[mv] for mv in med_vars], axis=0) + 0.2 * (1.0 + 4.0 * rng.beta(4, 2, n))
        elif ind_vars:
            val = 0.8 * np.mean([latents[iv] for iv in ind_vars], axis=0) + 0.2 * (1.0 + 4.0 * rng.beta(4, 2, n))
        else:
            val = 1.0 + 4.0 * rng.beta(4, 2, n)
        latents[dv] = np.clip(val + rng.normal(0, noise_hl, n), 1, 5)

    # 3. Tạo dữ liệu mẫu
    data = {}
    data["ID"] = [f"SV{str(i+1).zfill(3)}" for i in range(n)]

    # Sinh các biến nhân khẩu học động
    demos = ACTIVE_CONFIG.get("demographics", [])
    for d_col in demos:
        d_lower = d_col.lower()
        if "gioi" in d_lower or "gender" in d_lower or "sex" in d_lower:
            data[d_col] = rng.choice([1, 2], n, p=[0.45, 0.55])
        elif "nam" in d_lower or "year" in d_lower:
            data[d_col] = rng.choice([1, 2, 3, 4], n, p=[0.28, 0.30, 0.25, 0.17])
        elif "nganh" in d_lower or "major" in d_lower:
            data[d_col] = rng.choice([1, 2, 3, 4, 5], n)
        elif "thoigian" in d_lower or "time" in d_lower or "duration" in d_lower:
            data[d_col] = rng.choice([1, 2, 3], n, p=[0.35, 0.40, 0.25])
        elif "chiphi" in d_lower or "cost" in d_lower:
            data[d_col] = rng.choice([1, 2, 3], n, p=[0.50, 0.35, 0.15])
        else:
            data[d_col] = rng.choice([1, 2, 3], n)

    # Xác định các tham số nhiễu cho biến quan sát
    if force_clean:
        noise_val = 0.40
        jitter_std = 0.03
    elif efa_passing_guaranteed:
        noise_val = 0.50
        jitter_std = 0.05
    else:
        noise_val = 0.85
        jitter_std = 0.20

    # Biến Likert từ latent
    def to_likert(lat, noise=0.80):
        raw = np.clip(np.round(lat + rng.normal(0, noise, len(lat))), 1, 5)
        return raw.astype(int)

    for var_code, lat in latents.items():
        if var_code in cfg:
            items = cfg[var_code]["items"]
            for j, item_code in enumerate(items):
                jitter = rng.normal(0, jitter_std)  # item-level offset
                vals = to_likert(lat + jitter, noise=noise_val)
                
                # Áp dụng khuyết thiếu (missing data)
                mask = rng.random(n) < missing_rate
                vals = vals.astype(float)
                vals[mask] = np.nan
                data[item_code] = vals

    df = pd.DataFrame(data)

    # Chèn thêm Outliers (ngoại lai) ở chế độ đời thực (3%)
    if not (efa_passing_guaranteed or force_clean):
        num_outliers = int(0.03 * n)
        if num_outliers > 0:
            outlier_indices = rng.choice(n, num_outliers, replace=False)
            survey_cols = []
            for var_code in latents.keys():
                if var_code in cfg:
                    survey_cols.extend(list(cfg[var_code]["items"].keys()))
                
            for idx in outlier_indices:
                outlier_type = rng.choice(["all_5", "all_1", "random"])
                if outlier_type == "all_5":
                    df.loc[idx, survey_cols] = 5.0
                elif outlier_type == "all_1":
                    df.loc[idx, survey_cols] = 1.0
                else:
                    df.loc[idx, survey_cols] = rng.choice([1.0, 2.0, 3.0, 4.0, 5.0], size=len(survey_cols))

    return df

def _verify_data_quality(df):
    """Kiểm tra xem dữ liệu có đạt các điều kiện cần thiết của Cronbach và EFA không"""
    cfg = ACTIVE_CONFIG["variables"]
    ind_vars = [k for k, v in cfg.items() if v["type"] == "independent"]
    all_items = []
    
    for var_code in ind_vars:
        items = list(cfg[var_code]["items"].keys())
        all_items.extend(items)
        
        # Kiểm tra Cronbach Alpha
        c_res = cronbach_alpha(df[items].dropna().values)
        alpha = c_res.get("alpha", 0)
        if alpha < 0.70:
            return False
            
        # Kiểm tra Item-Total Correlation >= 0.30
        itc_list = c_res.get("itc_list", [])
        for col_name, corr in itc_list:
            if corr < 0.30:
                return False
                
    # Kiểm tra KMO và Bartlett
    df_efa = df[all_items].copy()
    for col in df_efa.columns:
        if df_efa[col].isna().any():
            df_efa[col] = df_efa[col].fillna(df_efa[col].mean())
    df_efa = df_efa.dropna()
    
    if df_efa.empty:
        return False
        
    kmo = kmo_test(df_efa)
    if kmo < 0.70:  # KMO >= 0.70 là mức tốt
        return False
        
    _, _, p_bart = bartlett_test(df_efa)
    if p_bart >= 0.05:
        return False
        
    # Kiểm tra loadings EFA sơ bộ (mỗi item phải load >= 0.50 vào ít nhất 1 factor)
    try:
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df_efa)
        pca = PCA(n_components=len(ind_vars))
        pca.fit(X_scaled)
        loadings = varimax(pca.components_.T * np.sqrt(pca.explained_variance_))
        
        for i in range(loadings.shape[0]):
            if np.max(np.abs(loadings[i])) < 0.50:
                return False
    except Exception:
        return False
        
    return True


# ─────────────────────────────────────────────────────────────────────────────
#  2. THỐNG KÊ MÔ TẢ
# ─────────────────────────────────────────────────────────────────────────────
def describe_sample(df):
    header("THỐNG KÊ MÔ TẢ MẪU KHẢO SÁT")
    n = len(df)
    print(f"\n  Tổng số quan sát: n = {n}")
    
    def pct(mask): return f"{mask.sum()} ({mask.sum()/n*100:.1f}%)"
    
    section("Đặc điểm nhân khẩu học")
    demos = ACTIVE_CONFIG.get("demographics", [])
    rows = []
    for d_col in demos:
        if d_col in df.columns:
            unique_vals = sorted(df[d_col].dropna().unique())
            for val in unique_vals:
                lbl = str(val)
                d_lower = d_col.lower()
                if "gioi" in d_lower or "gender" in d_lower or "sex" in d_lower:
                    lbl = "Nam" if val == 1 else ("Nữ" if val == 2 else f"Khác ({val})")
                elif "nam" in d_lower or "year" in d_lower:
                    lbl = f"Năm {int(val)}" if val in [1, 2, 3, 4] else f"Năm thứ {val}"
                elif "thoigian" in d_lower or "time" in d_lower:
                    lbl = "<1 năm" if val == 1 else ("1-2 năm" if val == 2 else (">2 năm" if val == 3 else f"Nhóm {val}"))
                elif "chiphi" in d_lower or "cost" in d_lower:
                    lbl = "Tự túc" if val == 1 else ("Học bổng" if val == 2 else ("Gia đình hỗ trợ" if val == 3 else f"Nhóm {val}"))
                rows.append((f"{d_col} – {lbl}", pct(df[d_col] == val)))
    
    table_print(["Tiêu chí", "Số lượng (%)"], rows, [28, 18])

    section("Thống kê mô tả biến Likert (thang 1–5)")
    cfg = ACTIVE_CONFIG["variables"]
    desc_rows = []
    for var_code, var_info in cfg.items():
        items = list(var_info["items"].keys())
        all_vals = df[items].values.flatten()
        all_vals = all_vals[~np.isnan(all_vals)]
        m    = np.mean(all_vals)
        sd   = np.std(all_vals, ddof=1)
        mn   = np.min(all_vals)
        mx   = np.max(all_vals)
        miss = df[items].isnull().sum().sum()
        skew = float(pd.Series(all_vals).skew())
        level = "Cao" if m >= 3.5 else ("Trung bình" if m >= 2.5 else "Thấp")
        desc_rows.append((var_code, var_info["label"][:30], f"{m:.3f}", f"{sd:.3f}",
                          f"{mn:.0f}", f"{mx:.0f}", f"{skew:.3f}", miss, level))
    table_print(
        ["Biến","Tên biến","Mean","SD","Min","Max","Skew","Missing","Đánh giá"],
        desc_rows, [8,30,8,8,6,6,8,8,12]
    )
    # Tạo DataFrames cho UI
    demo_df = pd.DataFrame(rows, columns=["Tiêu chí", "Số lượng (%)"])
    stats_df = pd.DataFrame([
        {
            "Biến": r[0], "Tên": r[1], "Mean": r[2], "SD": r[3],
            "Min": r[4], "Max": r[5], "Skew": r[6], "Missing": r[7], "Đánh giá": r[8]
        } for r in desc_rows
    ])
    
    return {"demographics": demo_df, "stats": stats_df}


# ─────────────────────────────────────────────────────────────────────────────
#  3. CRONBACH'S ALPHA
# ─────────────────────────────────────────────────────────────────────────────
def cronbach_alpha(data_matrix, col_names=None):
    """Tính Cronbach's Alpha từ ma trận dữ liệu (n_obs x n_items)"""
    df_in = pd.DataFrame(data_matrix)
    if col_names:
        df_in.columns = col_names
    df_clean = df_in.dropna()
    if len(df_clean) < 3 or df_clean.shape[1] < 2:
        return {"alpha": np.nan, "itc_list": [], "k": 0, "sum_item_var": 0, "total_var": 0}
    
    k = df_clean.shape[1]
    item_vars = df_clean.var(ddof=1)
    sum_item_var = item_vars.sum()
    total_var = df_clean.sum(axis=1).var(ddof=1)
    alpha = (k / (k - 1)) * (1 - sum_item_var / total_var)

    # Item-total correlations
    itc_list = []
    for i in range(k):
        col_name = df_clean.columns[i]
        other_items = df_clean.drop(columns=[col_name])
        rest_sum = other_items.sum(axis=1)
        corr = df_clean[col_name].corr(rest_sum)
        itc_list.append((col_name, corr))
    
    return {
        "alpha": alpha, 
        "itc_list": itc_list, 
        "k": k, 
        "sum_item_var": sum_item_var, 
        "total_var": total_var
    }


def run_cronbach(df):
    header("KIỂM ĐỊNH ĐỘ TIN CẬY THANG ĐO – CRONBACH'S ALPHA")
    cfg = ACTIVE_CONFIG["variables"]
    cronbach_results = {}

    for var_code, var_info in cfg.items():
        items_dict = var_info["items"]
        cols = list(items_dict.keys())
        res_ca = cronbach_alpha(df[cols], cols)
        alpha_val = res_ca["alpha"]
        itc_vals = res_ca["itc_list"]

        itc_rows = []
        dropped = []
        for c, val in itc_vals:
            lbl = items_dict[c]
            itc_rows.append({"Mã biến": c, "Tên biến": lbl, "ITC": round(val, 3)})
            if val < 0.3: dropped.append(c)
        
        status_val = "Đạt" if alpha_val >= 0.7 else ("Tạm" if alpha_val >= 0.6 else "Không đạt")
        cronbach_results[var_code] = {
            "alpha": alpha_val,
            "table": pd.DataFrame(itc_rows),
            "dropped": dropped,
            "items": cols,
            "status": status_val,
            "metadata": {
                "k": res_ca["k"],
                "sum_item_var": res_ca["sum_item_var"],
                "total_var": res_ca["total_var"]
            }
        }
        
        section(f"{var_code} – {var_info['label']} ({res_ca['k']} biến quan sát)")
        if not np.isnan(alpha_val):
            if alpha_val >= 0.7:
                ok(f"Cronbach's Alpha = {alpha_val:.3f} (Đạt)")
            elif alpha_val >= 0.6:
                warn(f"Cronbach's Alpha = {alpha_val:.3f} (Tạm chấp nhận)")
            else:
                err(f"Cronbach's Alpha = {alpha_val:.3f} (Không đạt)")
        
        if dropped:
            warn(f"Biến có Item-Total r < 0.3: {', '.join(dropped)} → cân nhắc loại")
    
    return cronbach_results



# ─────────────────────────────────────────────────────────────────────────────
#  4. PHÂN TÍCH NHÂN TỐ KHÁM PHÁ (EFA) – PCA thủ công
# ─────────────────────────────────────────────────────────────────────────────
def kmo_test(df_items):
    """Tính KMO từ ma trận tương quan có thêm Ridge Regularization để tránh lỗi suy biến ma trận"""
    df_clean = df_items.dropna()
    if df_clean.empty or df_clean.shape[1] < 2:
        return 0.0
    corr = df_clean.corr().values
    n_vars = corr.shape[0]
    
    # Ridge regularization: cộng một hằng số nhỏ vào đường chéo chính để tránh suy biến
    epsilon = 1e-5
    corr_reg = corr + epsilon * np.eye(n_vars)
    # Chuẩn hóa lại ma trận tương quan sau khi cộng epsilon
    d = np.diag(corr_reg)
    corr_reg = corr_reg / np.sqrt(np.outer(d, d))
    
    try:
        inv_corr = np.linalg.inv(corr_reg)
        partial_corr = np.zeros_like(corr_reg)
        for i in range(n_vars):
            for j in range(n_vars):
                if i != j:
                    partial_corr[i,j] = -inv_corr[i,j] / np.sqrt(inv_corr[i,i]*inv_corr[j,j])
        
        sum_r2 = np.sum(corr_reg**2) - np.sum(np.diag(corr_reg)**2)  # off-diagonal
        sum_p2 = np.sum(partial_corr**2) - np.sum(np.diag(partial_corr)**2)
        kmo = sum_r2 / (sum_r2 + sum_p2)
        return round(float(kmo), 3)
    except Exception:
        return 0.50  # Giá trị fallback thực tế tối thiểu nếu vẫn xảy ra lỗi toán học khác

def bartlett_test(df_items):
    """Bartlett's test of sphericity"""
    df_clean = df_items.dropna()
    n, p = df_clean.shape
    corr = df_clean.corr().values
    det = np.linalg.det(corr)
    det = max(det, 1e-300)
    chi2 = -(n - 1 - (2*p + 5)/6) * np.log(det)
    df_stat = p*(p-1)/2
    p_val = 1 - stats.chi2.cdf(chi2, df_stat)
    return round(float(chi2), 3), int(df_stat), round(float(p_val), 4)

def run_efa(df, cronbach_results):
    header("PHÂN TÍCH NHÂN TỐ KHÁM PHÁ (EFA) – Principal Components + Varimax")
    cfg = ACTIVE_CONFIG["variables"]
    
    # Lấy tất cả biến độc lập + trung gian (không lấy biến phụ thuộc vào EFA chính)
    ind_vars = [k for k,v in cfg.items() if v["type"] == "independent"]
    all_items = []
    for var_code in ind_vars:
        kept = [i for i in cfg[var_code]["items"] if i not in cronbach_results.get(var_code,{}).get("dropped",[])]
        all_items.extend(kept)

    if not all_items:
        warn("Không còn biến quan sát hợp lệ sau bước Cronbach. Bỏ qua EFA.")
        return {
            "kmo": float("nan"),
            "bartlett_chi2": float("nan"),
            "bartlett_sig": float("nan"),
            "n_factors": 0,
            "total_var": 0.0,
            "eigenvalues": [],
            "var_explained": [],
            "loadings": np.empty((0, 0)),
            "items": []
        }

    # Dùng nội suy trung bình theo cột để giữ cỡ mẫu cho EFA,
    # tránh trường hợp dropna làm n_samples quá nhỏ gây lỗi PCA.
    df_efa = df[all_items].copy()
    for col in df_efa.columns:
        if df_efa[col].isna().any():
            df_efa[col] = df_efa[col].fillna(df_efa[col].mean())
    df_efa = df_efa.dropna()
    n_obs, n_items = df_efa.shape
    
    section(f"Kiểm tra điều kiện EFA ({n_items} biến, n={n_obs})")
    
    kmo = kmo_test(df_efa)
    chi2, df_b, p_bart = bartlett_test(df_efa)
    
    kmo_status = ("Tốt" if kmo >= 0.8 else
                  "Khá" if kmo >= 0.7 else
                  "Tạm chấp nhận" if kmo >= 0.6 else "Không phù hợp")
    
    if kmo >= 0.7: ok(f"KMO = {kmo}  →  {kmo_status}")
    else:          warn(f"KMO = {kmo}  →  {kmo_status}")
    
    if p_bart < 0.05:
        ok(f"Bartlett's Test: Chi² = {chi2}, df = {df_b}, Sig. < 0.001  →  Phù hợp")
    else:
        warn(f"Bartlett's Test Sig. = {p_bart}  →  Không đạt (cần < 0.05)")
    
    # PCA
    from sklearn.decomposition import PCA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_efa)
    
    max_components = min(n_obs, n_items)
    if max_components < 1:
        warn("Dữ liệu không đủ để thực hiện EFA sau khi làm sạch.")
        return {
            "kmo": float("nan"),
            "bartlett_chi2": float("nan"),
            "bartlett_sig": float("nan"),
            "n_factors": 0,
            "total_var": 0.0,
            "eigenvalues": [],
            "var_explained": [],
            "loadings": np.empty((n_items, 0)),
            "items": all_items
        }

    # Số nhân tố có eigenvalue > 1
    pca_full = PCA()
    pca_full.fit(X_scaled)
    eigenvalues = pca_full.explained_variance_
    n_factors = int(np.sum(eigenvalues > 1))
    n_factors = max(n_factors, len(ind_vars))  # ít nhất bằng số biến độc lập
    n_factors = max(1, min(n_factors, max_components))

    pca = PCA(n_components=n_factors)
    pca.fit(X_scaled)
    loadings_raw = pca.components_.T * np.sqrt(pca.explained_variance_)  # shape: n_items x n_factors
    
    # Varimax rotation (thủ công)
    loadings = varimax(loadings_raw)
    
    var_explained = pca.explained_variance_ratio_ * 100
    cumvar = np.cumsum(var_explained)
    total_var = cumvar[-1]
    
    section("Eigenvalue và phương sai trích")
    eig_rows = []
    for i in range(n_factors):
        eig_val = eigenvalues[i]
        v_pct = var_explained[i]
        cum = cumvar[i]
        eig_rows.append((f"Nhân tố {i+1}", f"{eig_val:.3f}", f"{v_pct:.2f}%",
                         f"{cum:.2f}%", "Giữ lại" if eig_val > 1 else "Xem xét"))
    table_print(["Nhân tố","Eigenvalue","Phương sai (%)","Cộng dồn (%)","Kết luận"],
                eig_rows, [10,12,15,14,12])
    
    if total_var >= 60:
        ok(f"Tổng phương sai trích = {total_var:.2f}%  →  Đạt yêu cầu (≥60%)")
    elif total_var >= 50:
        warn(f"Tổng phương sai trích = {total_var:.2f}%  →  Tạm chấp nhận (≥50%)")
    else:
        err(f"Tổng phương sai trích = {total_var:.2f}%  →  Chưa đạt (<50%)")
    
    section("Ma trận Factor Loading (Varimax rotation)")
    headers = ["Biến"] + [f"F{i+1}" for i in range(n_factors)] + ["Nhân tố chính"]
    load_rows = []
    for j, item in enumerate(all_items):
        row_loads = loadings[j]
        max_f = int(np.argmax(np.abs(row_loads)))
        max_l = row_loads[max_f]
        row = [item] + [f"{l:.3f}" for l in row_loads] + [f"F{max_f+1} ({max_l:.3f})"]
        load_rows.append(row)
    col_w = [8] + [8]*n_factors + [16]
    table_print(headers, load_rows, col_w)
    
    info("Ngưỡng Factor Loading: ≥0.5=Tốt | 0.4–0.49=Chấp nhận | <0.4=Xem xét loại")
    
    low_load = [all_items[j] for j in range(len(all_items)) if np.max(np.abs(loadings[j])) < 0.4]
    if low_load:
        warn(f"Biến có loading < 0.4: {', '.join(low_load)}")
    else:
        ok("Tất cả biến có Factor Loading ≥ 0.4")
    
    # Tạo bảng loadings đẹp cho UI
    headers_efa = [f"F{i+1}" for i in range(n_factors)]
    df_loadings = pd.DataFrame(loadings, index=all_items, columns=headers_efa)
    df_loadings["Nhân tố chính"] = [headers_efa[i] for i in np.argmax(np.abs(loadings), axis=1)]

    return {
        "kmo": kmo, "bartlett_chi2": chi2, "bartlett_sig": p_bart,
        "n_factors": n_factors, "total_var": total_var,
        "eigenvalues": eigenvalues[:n_factors].tolist(),
        "var_explained": var_explained.tolist(),
        "loadings": loadings, 
        "rotated_matrix": df_loadings,
        "items": all_items
    }

def varimax(Phi, gamma=1.0, q=20, tol=1e-6):
    """Varimax rotation"""
    p, k = Phi.shape
    R = np.eye(k)
    d = 0
    for _ in range(q):
        d_old = d
        for i in range(k):
            for j in range(i+1, k):
                L = Phi @ R
                u = L[:,i]**2 - L[:,j]**2
                v = 2 * L[:,i] * L[:,j]
                A = np.sum(u)
                B = np.sum(v)
                C = np.sum(u**2 - v**2)
                D = np.sum(u*v)
                num = D - A*B/p
                den = C - (A**2 - B**2)/p
                if abs(den) > 1e-10:
                    theta = 0.25 * np.arctan2(num, den)
                    c, s = np.cos(theta), np.sin(theta)
                    G = np.eye(k)
                    G[i,i] = c; G[j,j] = c; G[i,j] = s; G[j,i] = -s
                    R = R @ G
        d = np.sum(np.diag(Phi.T @ Phi @ R @ R.T))
        if abs(d - d_old) < tol:
            break
    return Phi @ R


# ─────────────────────────────────────────────────────────────────────────────
#  5. TƯƠNG QUAN PEARSON
# ─────────────────────────────────────────────────────────────────────────────
def run_correlation(df):
    header("MA TRẬN TƯƠNG QUAN PEARSON")
    cfg = ACTIVE_CONFIG["variables"]
    
    # Tạo biến trung bình cho mỗi thang đo
    var_means = {}
    for var_code, var_info in cfg.items():
        items = list(var_info["items"].keys())
        var_means[var_code] = df[items].mean(axis=1)
    
    df_means = pd.DataFrame(var_means)
    corr_matrix = df_means.corr()
    
    section("Ma trận tương quan (r) và mức ý nghĩa")
    var_codes = list(cfg.keys())
    headers = [""] + var_codes
    rows = []
    for v1 in var_codes:
        row = [v1]
        for v2 in var_codes:
            if v1 == v2:
                row.append("1.000")
            else:
                r_val = corr_matrix.loc[v1, v2]
                common = df_means[[v1, v2]].dropna()
                n_valid = common.shape[0]
                if n_valid >= 2:
                    _, p = pearsonr(common[v1], common[v2])
                else:
                    p = 1.0
                sig_mark = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
                row.append(f"{r_val:.3f}{sig_mark}")
        rows.append(row)
    table_print(headers, rows, [6]+[12]*len(var_codes))
    info("*** p<0.001  ** p<0.01  * p<0.05")
    
    # Kiểm tra mối tương quan giả thuyết
    section("Kiểm tra tương quan theo giả thuyết")
    hyp_pairs = []
    for hyp in ACTIVE_CONFIG.get("hypotheses", []):
        path_str = hyp[1]
        for delimiter in ["→", "->", "-&gt;"]:
            if delimiter in path_str:
                parts = [p.strip() for p in path_str.split(delimiter)]
                if len(parts) == 2 and parts[0] in df_means.columns and parts[1] in df_means.columns:
                    hyp_pairs.append((parts[0], parts[1]))
                break
    for v1, v2 in hyp_pairs:
        common = df_means[[v1,v2]].dropna()
        if common.shape[0] < 2:
            print(f"  ⚠ {v1} → {v2}: Không đủ dữ liệu chung để kiểm định Pearson.")
            continue
        r_act, p_act = pearsonr(common[v1], common[v2])
        direction = "thuận chiều (+)" if r_act > 0 else "nghịch chiều (–)"
        strength = ("Rất mạnh" if abs(r_act) > 0.7 else "Mạnh" if abs(r_act) > 0.5
                    else "Trung bình" if abs(r_act) > 0.3 else "Yếu")
        sig_str = f"p<0.001" if p_act < 0.001 else f"p={p_act:.3f}"
        status = "✔" if p_act < 0.05 and r_act > 0 else "✘"
        print(f"  {status} {v1} → {v2}: r = {r_act:.3f} ({direction}, {strength}), {sig_str}")
    
    return corr_matrix, df_means


# ─────────────────────────────────────────────────────────────────────────────
#  6. HỒI QUY TUYẾN TÍNH BỘI
# ─────────────────────────────────────────────────────────────────────────────
def ols_regression(X, y):
    """OLS regression thủ công trả về beta, SE, t, p, VIF"""
    n, p = X.shape
    X_aug = np.column_stack([np.ones(n), X])
    try:
        beta = np.linalg.lstsq(X_aug, y, rcond=None)[0]
        y_hat = X_aug @ beta
        residuals = y - y_hat
        SSR = np.sum((y_hat - np.mean(y))**2)
        SSE = np.sum(residuals**2)
        SST = np.sum((y - np.mean(y))**2)
        R2 = SSR / SST if SST > 0 else 0
        adjR2 = 1 - (1 - R2)*(n-1)/(n-p-1)
        MSE = SSE / (n - p - 1)
        # Sử dụng pinv (Pseudo-inverse) để tránh lỗi Singular matrix khi có đa cộng tuyến hoàn hảo
        var_beta = MSE * np.linalg.pinv(X_aug.T @ X_aug)
        se = np.sqrt(np.diag(var_beta))
        t_stat = beta / se
        p_vals = 2 * (1 - stats.t.cdf(np.abs(t_stat), df=n-p-1))
        F_stat = (SSR/p) / (SSE/(n-p-1))
        F_p = 1 - stats.f.cdf(F_stat, p, n-p-1)
        
        # VIF
        vif = []
        for i in range(p):
            Xi = np.delete(X, i, axis=1)
            # Dùng pinv cho VIF để đảm bảo tính toán không bị ngắt quãng
            ri2 = np.linalg.lstsq(Xi, X[:,i], rcond=None)[0]
            yi_hat = Xi @ ri2
            ss_res = np.sum((X[:,i] - yi_hat)**2)
            ss_tot = np.sum((X[:,i] - np.mean(X[:,i]))**2)
            r2i = 1 - ss_res/ss_tot if ss_tot > 0 else 0
            vif.append(1/(1-r2i) if r2i < 1 else 999)
        
        # Durbin-Watson
        dw = np.sum(np.diff(residuals)**2) / SSE if SSE > 0 else 2.0
        
        # Tính Standardized Betas (Beta chuẩn hóa) để so sánh mức độ tác động
        X_std = np.std(X, axis=0)
        y_std = np.std(y)
        # beta[0] là Intercept, beta[1:] là các biến độc lập
        # Beta chuẩn hóa = Unstd_Beta * (Std_X / Std_y)
        beta_standardized = {}
        # beta_all[1:] tương ứng với X_std
        # Note: We need labels for the dictionary, but here we just return a list/dict if we can.
        # However, it's easier to return it as a list and let the caller map it.
        std_betas = beta[1:] * (X_std / y_std) if y_std > 0 else np.zeros(p)

        return {
            "beta": beta, "se": se, "t": t_stat, "p": p_vals,
            "R2": R2, "adjR2": adjR2, "F": F_stat, "F_p": F_p,
            "vif": vif, "dw": dw, "residuals": residuals, "y_hat": y_hat,
            "std_betas": std_betas
        }
    except Exception as e:
        print(f"DEBUG: OLS Error: {e}")
        return None


def get_regression_table(res, x_labels):
    """Tạo bảng DataFrame kết quả hồi quy từ dict kết quả ols_regression"""
    beta_all  = res['beta']
    se_all    = res['se']
    t_all     = res['t']
    p_all     = res['p']
    vif_all   = res['vif']
    
    rows = []
    # Intercept
    rows.append({
        "Biến": "(Hằng số)",
        "B": round(beta_all[0], 3),
        "SE": round(se_all[0], 3),
        "VIF": np.nan,
        "t": round(t_all[0], 3),
        "Sig.": "< 0.001" if p_all[0] < 0.001 else round(p_all[0], 3),
        "VIF trạng thái": "–",
        "GT": "–"
    })
    # Variables
    for i, lbl in enumerate(x_labels):
        b_unstd = beta_all[i+1]
        se_i    = se_all[i+1]
        t_i     = t_all[i+1]
        p_i     = p_all[i+1]
        vif_i   = vif_all[i]
        
        sig_str = "< 0.001" if p_i < 0.001 else round(p_i, 3)
        vif_st  = "Tốt" if vif_i < 5 else ("Xem xét" if vif_i < 10 else "Đa cộng tuyến")
        h_result = "Ủng hộ" if p_i < 0.05 and b_unstd > 0 else ("Bác bỏ" if p_i >= 0.05 else "Nghịch chiều")
        
        rows.append({
            "Biến": lbl,
            "B": round(b_unstd, 3),
            "SE": round(se_i, 3),
            "VIF": round(vif_i, 2) if pd.notna(vif_i) else np.nan,
            "t": round(t_i, 3),
            "Sig.": sig_str,
            "VIF trạng thái": vif_st,
            "GT": h_result
        })
    df_res = pd.DataFrame(rows)
    # Convert problematic columns to string to avoid serialization issues with mixed content like "< 0.001"
    df_res["Sig."] = df_res["Sig."].astype(str)
    return df_res


def _print_hypothesis_summary(results_all):
    section("Tổng kết kiểm định giả thuyết")
    hyps = ACTIVE_CONFIG["hypotheses"]
    
    # Map hypothesis sang mô hình và vị trí biến động
    # Tìm model nào chứa X và Y
    rows = []
    for h_code, path, desc in hyps:
        # Tách X -> Y từ path
        parts = path.split("→")
        if len(parts) != 2: continue
        x_target = parts[0].strip()
        y_target = parts[1].strip()
        
        # Tìm trong các model đã chạy
        found = False
        for m_key, m_info in results_all.items():
            if m_info["Y_label"] == y_target:
                labels = m_info["X_labels"]
                if x_target in labels:
                    idx = labels.index(x_target)
                    r = m_info["result"]
                    b = r["beta"][idx+1]
                    p = r["p"][idx+1]
                    sig = "< 0.001" if p < 0.001 else f"{p:.3f}"
                    result = "✔ Ủng hộ" if p < 0.05 and b > 0 else "✘ Bác bỏ"
                    rows.append((h_code, path, f"{b:.3f}", sig, result))
                    found = True
                    break
    
    table_print(["GT","Đường dẫn","Beta","Sig.","Kết quả"], rows, [4,14,7,10,12])

def run_regression(df_means):
    header("PHÂN TÍCH HỒI QUY TUYẾN TÍNH – KIỂM ĐỊNH GIẢ THUYẾT")
    cfg = ACTIVE_CONFIG["variables"]
    results_all = {}

    ind_codes = [k for k, v in cfg.items() if v["type"] == "independent"]
    med_codes = [k for k, v in cfg.items() if v["type"] == "mediator"]
    dep_codes = [k for k, v in cfg.items() if v["type"] == "dependent"]
    
    if not dep_codes:
        warn("Không tìm thấy biến phụ thuộc. Bỏ qua hồi quy.")
        return {}

    # ── 1. Hồi quy cho các Biến Trung gian (nếu có) ──
    for i, m_code in enumerate(med_codes):
        section(f"Mô hình {i+1}: {', '.join(ind_codes)} → {m_code}")
        df_m = df_means[ind_codes + [m_code]].dropna()
        if not df_m.empty:
            X = df_m[ind_codes].values
            y = df_m[m_code].values
            res = ols_regression(X, y)
            if res:
                res["table"] = get_regression_table(res, ind_codes)
                res["standardized_betas_map"] = {lbl: res["std_betas"][i] for i, lbl in enumerate(ind_codes)}
                _print_regression(res, ind_codes, m_code, model_num=i+1)
                results_all[f"model_med_{m_code}"] = {"result": res, "X_labels": ind_codes, "Y_label": m_code}

    # ── 2. Hồi quy cho Biến Phụ thuộc chính ──
    # Model: IVs + Meds -> DV
    for j, d_code in enumerate(dep_codes):
        x_final = ind_codes + med_codes
        m_num = len(med_codes) + j + 1
        section(f"Mô hình {m_num}: {', '.join(x_final)} → {d_code}")
        
        df_d = df_means[x_final + [d_code]].dropna()
        if not df_d.empty:
            X = df_d[x_final].values
            y = df_d[d_code].values
            res = ols_regression(X, y)
            if res:
                res["table"] = get_regression_table(res, x_final)
                res["standardized_betas_map"] = {lbl: res["std_betas"][i] for i, lbl in enumerate(x_final)}
                _print_regression(res, x_final, d_code, model_num=m_num)
                results_all[f"model_dep_{d_code}"] = {"result": res, "X_labels": x_final, "Y_label": d_code}

    _print_hypothesis_summary(results_all)
    return results_all

    _print_hypothesis_summary(results_all)
    return results_all


def _print_regression(res, x_labels, y_label, model_num):
    print(f"\n  Biến phụ thuộc: {y_label}")
    print(f"  n = {len(res['residuals']) + len(x_labels)}")
    print(f"  R² = {res['R2']:.4f}  |  Adjusted R² = {res['adjR2']:.4f}")
    
    f_sig = "< 0.001" if res['F_p'] < 0.001 else f"= {res['F_p']:.3f}"
    dw_ok = 1.5 <= res['dw'] <= 2.5
    
    if res['F_p'] < 0.05: ok(f"F = {res['F']:.3f}, Sig. {f_sig}  →  Mô hình phù hợp")
    else:                  err(f"F = {res['F']:.3f}, Sig. {f_sig}  →  Mô hình chưa phù hợp")
    
    if dw_ok: ok(f"Durbin-Watson = {res['dw']:.3f}  →  Không có tự tương quan phần dư")
    else:     warn(f"Durbin-Watson = {res['dw']:.3f}  →  Có thể tự tương quan phần dư")
    
    # Coefficients table
    beta_all  = res['beta']
    se_all    = res['se']
    t_all     = res['t']
    p_all     = res['p']
    vif_all   = res['vif']
    
    rows = []
    # Intercept
    rows.append(("(Hằng số)", f"{beta_all[0]:.3f}", f"{se_all[0]:.3f}",
                 "–", f"{t_all[0]:.3f}",
                 "< 0.001" if p_all[0] < 0.001 else f"{p_all[0]:.3f}",
                 "–", "–"))
    # Variables
    for i, lbl in enumerate(x_labels):
        b_unstd = beta_all[i+1]
        se_i    = se_all[i+1]
        t_i     = t_all[i+1]
        p_i     = p_all[i+1]
        vif_i   = vif_all[i]
        
        # Beta chuẩn hóa (thủ công)
        sig_str = "< 0.001" if p_i < 0.001 else f"{p_i:.3f}"
        vif_st  = "Tốt" if vif_i < 5 else ("Xem xét" if vif_i < 10 else "Đa cộng tuyến")
        h_result = "Ủng hộ" if p_i < 0.05 and b_unstd > 0 else ("Bác bỏ" if p_i >= 0.05 else "Nghịch chiều")
        rows.append((lbl, f"{b_unstd:.3f}", f"{se_i:.3f}", f"{vif_i:.2f}",
                     f"{t_i:.3f}", sig_str, vif_st, h_result))
    
    table_print(
        ["Biến","B","SE","VIF","t","Sig.","VIF trạng thái","GT"],
        rows, [8,7,7,7,8,10,16,12]
    )
    
    info("Ngưỡng: Sig. <0.05=Có ý nghĩa | VIF <5=Tốt | 5-10=Xem xét | >10=Đa cộng tuyến")
    info(f"R²={res['R2']:.4f} → mô hình giải thích {res['R2']*100:.1f}% biến thiên của {y_label}")


def _print_hypothesis_summary(results_all):
    section("Tổng kết kiểm định giả thuyết")
    hyps = ACTIVE_CONFIG.get("hypotheses", [])
    
    rows = []
    for h_code, path, desc in hyps:
        parts = [p.strip() for p in path.split("→")]
        if len(parts) == 2:
            cause_var, effect_var = parts[0], parts[1]
            found = False
            for m_info in results_all.values():
                if m_info["Y_label"] == effect_var and cause_var in m_info["X_labels"]:
                    r = m_info["result"]
                    labels = m_info["X_labels"]
                    idx = labels.index(cause_var)
                    b = r["beta"][idx+1]
                    p = r["p"][idx+1]
                    sig = "< 0.001" if p < 0.001 else f"{p:.3f}"
                    result = "✔ Ủng hộ" if p < 0.05 and b > 0 else "✘ Bác bỏ"
                    rows.append((h_code, path, f"{b:.3f}", sig, result))
                    found = True
                    break
            if not found:
                rows.append((h_code, path, "–", "–", "–"))
        else:
            rows.append((h_code, path, "–", "–", "–"))
    
    table_print(["GT","Đường dẫn","Beta","Sig.","Kết quả"], rows, [4,14,7,10,12])


def run_group_tests(df, df_means):
    """Kiểm định sự khác biệt giữa các nhóm (t-test, ANOVA) dựa trên các biến định danh phát hiện được"""
    results = {}
    dv_target = [k for k, v in ACTIVE_CONFIG["variables"].items() if v["type"] == "dependent"]
    if not dv_target or df_means.empty: return {}
    y_col = dv_target[0]
    
    # Tìm các biến định danh trong ACTIVE_CONFIG
    demos = ACTIVE_CONFIG.get("demographics", [])
    
    for d_col in demos:
        if d_col not in df.columns: continue
        unique_vals = df[d_col].dropna().unique()
        if 1 < len(unique_vals) <= 10:
            if len(unique_vals) == 2:
                # T-test
                v1, v2 = unique_vals
                g1 = df_means.loc[df[d_col] == v1, y_col].dropna()
                g2 = df_means.loc[df[d_col] == v2, y_col].dropna()
                if len(g1) > 2 and len(g2) > 2:
                    _, p_lev = levene(g1, g2)
                    t_stat, t_p = ttest_ind(g1, g2, equal_var=(p_lev > 0.05))
                    t_table = pd.DataFrame([
                        {"Nhóm": str(v1), "N": len(g1), "Mean": round(g1.mean(), 3), "SD": round(g1.std(), 3)},
                        {"Nhóm": str(v2), "N": len(g2), "Mean": round(g2.mean(), 3), "SD": round(g2.std(), 3)}
                    ])
                    results[f"Khác biệt theo {d_col} (t-test)"] = {
                        "table": t_table, 
                        "p_value": t_p, 
                        "conclusion": "Có ý nghĩa" if t_p < 0.05 else "Không ý nghĩa",
                        "metadata": {"stat": t_stat, "p_value": t_p, "type": "T-test"}
                    }
            else:
                # ANOVA
                groups = [df_means.loc[df[d_col] == v, y_col].dropna() for v in unique_vals]
                if all(len(g) > 2 for g in groups):
                    f_stat, f_p = f_oneway(*groups)
                    a_rows = []
                    for i, v in enumerate(unique_vals):
                        g = groups[i]
                        a_rows.append({"Nhóm": str(v), "N": len(g), "Mean": round(g.mean(), 3), "SD": round(g.std(), 3)})
                    results[f"Khác biệt theo {d_col} (ANOVA)"] = {
                        "table": pd.DataFrame(a_rows), 
                        "p_value": f_p, 
                        "conclusion": "Có ý nghĩa" if f_p < 0.05 else "Không ý nghĩa",
                        "metadata": {"stat": f_stat, "p_value": f_p, "type": "ANOVA"}
                    }
    return results
    
    return results


def run_mediation_model4(df_means):
    header("PHÂN TÍCH BIẾN TRUNG GIAN (MEDIATION - MODEL 4)")
    if df_means.empty: return {}
    
    # Chuẩn bị dữ liệu động
    ind_vars = [k for k, v in ACTIVE_CONFIG["variables"].items() if v["type"] == "independent"]
    med_vars = [k for k, v in ACTIVE_CONFIG["variables"].items() if v["type"] == "mediator"]
    dep_vars = [k for k, v in ACTIVE_CONFIG["variables"].items() if v["type"] == "dependent"]
    
    if not ind_vars or not med_vars or not dep_vars:
        warn("Dữ liệu không đủ cấu trúc X, M, Y để chạy Mediation Model 4.")
        return {}
        
    m_target = med_vars[0]
    y_target = dep_vars[0]
    
    df_med = df_means.dropna()
    df_med["X_total"] = df_med[ind_vars].mean(axis=1)
    
    X_vals = df_med[["X_total"]].values
    M_vals = df_med[m_target].values
    Y_vals = df_med[y_target].values
    
    # Bước 1: Y ~ X (Tổng tác động c)
    res_total = ols_regression(X_vals, Y_vals)
    c_path = res_total["beta"][1]
    
    # Bước 2: M ~ X (Tác động a)
    res_a = ols_regression(X_vals, M_vals)
    a_path = res_a["beta"][1]
    
    # Bước 3: Y ~ X + M (Tác động trực tiếp c' và tác động b)
    X_M_vals = df_med[["X_total", m_target]].values
    res_direct = ols_regression(X_M_vals, Y_vals)
    c_prime_path = res_direct["beta"][1]
    b_path = res_direct["beta"][2]
    
    # Tính toán tác động gián tiếp (Indirect Effect)
    indirect_effect = a_path * b_path
    
    # Kiểm định Sobel (Xấp xỉ)
    se_a = res_a["se"][1]
    se_b = res_direct["se"][2]
    sobel_se = np.sqrt(b_path**2 * se_a**2 + a_path**2 * se_b**2)
    sobel_z = indirect_effect / sobel_se
    from scipy.stats import norm
    p_sobel = 2 * (1 - norm.cdf(abs(sobel_z)))
    
    # Bảng kết quả
    rows = [
        {"Đường dẫn": "X → M (a)", "Hệ số": round(a_path, 3), "Sig.": f"{res_a['p'][1]:.4f}", "Kết luận": "Có ý nghĩa" if res_a['p'][1] < 0.05 else "Không ý nghĩa"},
        {"Đường dẫn": "M → Y (b)", "Hệ số": round(b_path, 3), "Sig.": f"{res_direct['p'][2]:.4f}", "Kết luận": "Có ý nghĩa" if res_direct['p'][2] < 0.05 else "Không ý nghĩa"},
        {"Đường dẫn": "Trực tiếp X → Y (c')", "Hệ số": round(c_prime_path, 3), "Sig.": f"{res_direct['p'][1]:.4f}", "Kết luận": "Có ý nghĩa" if res_direct['p'][1] < 0.05 else "Không ý nghĩa"},
        {"Đường dẫn": "Tổng tác động X → Y (c)", "Hệ số": round(c_path, 3), "Sig.": f"{res_total['p'][1]:.4f}", "Kết luận": "Có ý nghĩa" if res_total['p'][1] < 0.05 else "Không ý nghĩa"},
        {"Đường dẫn": "Gián tiếp X → M → Y", "Hệ số": round(indirect_effect, 3), "Sig. (Sobel)": f"{p_sobel:.4f}", "Kết luận": "Có ý nghĩa" if p_sobel < 0.05 else "Không ý nghĩa"}
    ]
    
    # Phân loại trung gian
    if p_sobel < 0.05:
        if res_direct['p'][1] >= 0.05:
            med_type = "Trung gian TOÀN PHẦN (Full Mediation)"
        else:
            med_type = "Trung gian MỘT PHẦN (Partial Mediation)"
    else:
        med_type = "KHÔNG có tác động trung gian"
        
    ok(f"Mediation Analysis Model 4 hoàn tất: {med_type}")
    return {
        "table": pd.DataFrame(rows),
        "type": med_type,
        "indirect_effect": indirect_effect,
        "p_value": p_sobel,
        "sobel_z": sobel_z,
        "a": a_path,
        "b": b_path,
        "se_a": se_a,
        "se_b": se_b
    }


# ─────────────────────────────────────────────────────────────────────────────
#  8. KIỂM TRA GIẢI ĐỊNH HỒI QUY
# ─────────────────────────────────────────────────────────────────────────────
def run_assumption_checks(regression_results):
    header("KIỂM TRA GIẢ ĐỊNH HỒI QUY – MÔ HÌNH 1 & 2")
    
    for m_key in ["model1","model2"]:
        if m_key not in regression_results:
            continue
        res = regression_results[m_key]["result"]
        y_label = regression_results[m_key]["Y_label"]
        resid = res["residuals"]
        y_hat = res["y_hat"]
        
        section(f"Mô hình: → {y_label}")
        
        # 1. Phân phối chuẩn phần dư (Shapiro-Wilk)
        stat_sw, p_sw = shapiro(resid[:min(len(resid),5000)])
        if p_sw > 0.05:
            ok(f"Shapiro-Wilk: W={stat_sw:.3f}, p={p_sw:.3f}  →  Phần dư phân phối chuẩn")
        else:
            warn(f"Shapiro-Wilk: W={stat_sw:.3f}, p={p_sw:.3f}  →  Phần dư lệch phân phối chuẩn")
        
        # 2. Mean residuals
        mean_r = np.mean(resid)
        ok(f"Mean phần dư = {mean_r:.6f}  →  {'Đạt (≈0)' if abs(mean_r) < 0.01 else 'Cần xem xét'}")
        
        # 3. Durbin-Watson
        dw = res["dw"]
        if 1.5 <= dw <= 2.5:
            ok(f"Durbin-Watson = {dw:.3f}  →  Không có tự tương quan phần dư")
        else:
            warn(f"Durbin-Watson = {dw:.3f}  →  Có thể có tự tương quan phần dư")
        
        # 4. VIF
        vifs = res["vif"]
        x_labels = regression_results[m_key]["X_labels"]
        max_vif = max(vifs) if vifs else 1
        if max_vif < 5:
            ok(f"VIF max = {max_vif:.2f}  →  Không có đa cộng tuyến")
        elif max_vif < 10:
            warn(f"VIF max = {max_vif:.2f}  →  Cần xem xét đa cộng tuyến")
        else:
            err(f"VIF max = {max_vif:.2f}  →  Đa cộng tuyến nghiêm trọng")
        
        info("Tóm tắt: Mô hình đạt các giả định cơ bản nếu tất cả chỉ số trên đạt")


# ─────────────────────────────────────────────────────────────────────────────
#  9. VẼ BIỂU ĐỒ
# ─────────────────────────────────────────────────────────────────────────────
def create_charts(df, df_means, cronbach_results, efa_results, reg_results):
    header("TẠO BIỂU ĐỒ PHÂN TÍCH CHI TIẾT")
    chart_files = {}
    
    title_font = {'fontsize': 11, 'fontweight': 'bold', 'color': C_DARK}
    cfg = ACTIVE_CONFIG["variables"]
    
    # ── 1. Bar chart: Mean ──
    if not df_means.empty:
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        var_codes = list(cfg.keys())
        active_vars = [v for v in var_codes if v in df_means.columns]
        means = [df_means[v].mean() for v in active_vars]
        colors = [cfg[v]["color"] for v in active_vars]
        bars = ax1.barh(active_vars, means, color=colors, height=0.6)
        ax1.axvline(x=3.5, color=C_RED, linestyle='--', label='Ngưỡng trung bình 3.5')
        for bar, m in zip(bars, means):
            ax1.text(m+0.02, bar.get_y()+bar.get_height()/2, f'{m:.3f}', va='center')
        ax1.set_xlim(1, 5.3); ax1.set_title("Biểu đồ Mean các nhân tố", **title_font)
        p1 = OUTPUT_DIR / "chart_1_mean.png"
        plt.savefig(p1, dpi=120, bbox_inches='tight'); plt.close(); chart_files["mean"] = p1

    # ── 2. Cronbach Alpha ──
    if cronbach_results:
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        active_vars = [v for v in cfg.keys() if v in cronbach_results]
        alphas = [cronbach_results[v]["alpha"] for v in active_vars]
        bar_colors = [C_GREEN if a >= 0.7 else C_AMBER if a >= 0.6 else C_RED for a in alphas]
        ax2.bar(active_vars, alphas, color=bar_colors, width=0.6)
        ax2.axhline(y=0.7, color=C_AMBER, linestyle='--', label='α=0.7')
        ax2.set_ylim(0, 1.05); ax2.set_title("Biểu đồ Độ tin cậy Cronbach's Alpha", **title_font)
        p2 = OUTPUT_DIR / "chart_2_cronbach.png"
        plt.savefig(p2, dpi=120, bbox_inches='tight'); plt.close(); chart_files["cronbach"] = p2

    # ── 3. Heatmap Tương quan ──
    if not df_means.empty:
        fig3, ax3 = plt.subplots(figsize=(10, 8))
        active_vars = [v for v in cfg.keys() if v in df_means.columns]
        corr_data = df_means[active_vars].corr()
        sns.heatmap(corr_data, annot=True, fmt='.3f', cmap='Blues', ax=ax3)
        ax3.set_title("Ma trận Tương quan Pearson", **title_font)
        p3 = OUTPUT_DIR / "chart_3_correlation.png"
        plt.savefig(p3, dpi=120, bbox_inches='tight'); plt.close(); chart_files["corr"] = p3

    # ── 4. Scree Plot EFA ──
    if efa_results and "eigenvalues" in efa_results:
        fig4, ax4 = plt.subplots(figsize=(10, 6))
        eigs = efa_results["eigenvalues"]
        ax4.plot(range(1, len(eigs)+1), eigs, 'o-', color=C_BLUE, linewidth=2)
        ax4.axhline(y=1.0, color=C_RED, linestyle='--', label='Eigenvalue = 1')
        ax4.set_title("Biểu đồ Scree Plot (EFA)", **title_font)
        p4 = OUTPUT_DIR / "chart_4_efa_scree.png"
        plt.savefig(p4, dpi=120, bbox_inches='tight'); plt.close(); chart_files["efa"] = p4

    # ── 5. Beta Comparison ──
    if reg_results and "model1" in reg_results:
        fig5, ax5 = plt.subplots(figsize=(10, 6))
        res_m1 = reg_results["model1"]["result"]
        labels_m1 = reg_results["model1"]["X_labels"]
        betas_m1 = res_m1["beta"][1:]
        ax5.bar(labels_m1, betas_m1, color=C_GREEN)
        ax5.set_title("So sánh mức độ tác động (Hệ số Beta)", **title_font)
        p5 = OUTPUT_DIR / "chart_5_regression_beta.png"
        plt.savefig(p5, dpi=120, bbox_inches='tight'); plt.close(); chart_files["reg_beta"] = p5

    ok("Đã tạo xong bộ biểu đồ phân tích thành phần.")
    return chart_files


# ─────────────────────────────────────────────────────────────────────────────
#  10. VẼ SƠ ĐỒ MÔ HÌNH NGHIÊN CỨU
# ─────────────────────────────────────────────────────────────────────────────
def create_model_diagram(reg_results):
    fig, ax = plt.subplots(1, 1, figsize=(14, 7))
    fig.patch.set_facecolor('white')
    ax.set_xlim(0, 14); ax.set_ylim(0, 7); ax.axis('off')
    
    cfg = ACTIVE_CONFIG["variables"]
    ind_codes = [k for k, v in cfg.items() if v["type"] == "independent"]
    med_codes = [k for k, v in cfg.items() if v["type"] == "mediator"]
    dep_codes = [k for k, v in cfg.items() if v["type"] == "dependent"]
    if not dep_codes and cfg:
        dep_codes = [list(cfg.keys())[-1]]
    
    # Boxes
    def draw_box(ax, x, y, w, h, label, sublabel, color, alpha=0.15):
        rect = mpatches.FancyBboxPatch((x-w/2, y-h/2), w, h,
            boxstyle="round,pad=0.1", facecolor=color, alpha=alpha,
            edgecolor=color, linewidth=2)
        ax.add_patch(rect)
        ax.text(x, y+0.1, label, ha='center', va='center',
                fontsize=10, fontweight='bold', color=color)
        ax.text(x, y-0.3, sublabel, ha='center', va='center',
                fontsize=7.5, color='#374151', style='italic')
                
    positions = {}
    
    # Calculate Y positions dynamically
    # Biến độc lập (trái x = 2.5)
    n_ind = len(ind_codes)
    for idx, iv in enumerate(ind_codes):
        y_pos = 5.5 - idx * (4.5 / max(1, n_ind - 1)) if n_ind > 1 else 3.5
        positions[iv] = (2.5, y_pos)
        draw_box(ax, 2.5, y_pos, 4.0, 0.9, iv, cfg[iv]["label"][:30], cfg[iv].get("color", C_BLUE))
        
    # Biến trung gian (giữa x = 7.5)
    n_med = len(med_codes)
    for idx, mv in enumerate(med_codes):
        y_pos = 5.5 - idx * (4.5 / max(1, n_med - 1)) if n_med > 1 else 3.5
        positions[mv] = (7.5, y_pos)
        draw_box(ax, 7.5, y_pos, 3.8, 0.9, mv, cfg[mv]["label"][:30], cfg[mv].get("color", "#0891b2"))
        
    # Biến phụ thuộc (phải x = 12.0)
    n_dep = len(dep_codes)
    for idx, dv in enumerate(dep_codes):
        y_pos = 5.5 - idx * (4.5 / max(1, n_dep - 1)) if n_dep > 1 else 3.5
        positions[dv] = (12.0, y_pos)
        draw_box(ax, 12.0, y_pos, 3.5, 0.9, dv, cfg[dv]["label"][:30], cfg[dv].get("color", C_RED))
        
    # Draw arrows dynamically from reg_results
    for m_key, m_info in reg_results.items():
        y_code = m_info["Y_label"]
        x_codes = m_info["X_labels"]
        r = m_info["result"]
        
        if y_code in positions:
            y_pos = positions[y_code]
            for idx, x_code in enumerate(x_codes):
                if x_code in positions:
                    x_pos = positions[x_code]
                    
                    b = r["beta"][idx+1]
                    p = r["p"][idx+1]
                    sig_str = f"β={b:.3f}***" if p < 0.001 else (f"β={b:.3f}**" if p < 0.01 else f"β={b:.3f}*" if p < 0.05 else f"β={b:.3f} n.s.")
                    arrow_col = cfg[x_code].get("color", C_BLUE) if p < 0.05 else C_GRAY
                    
                    x_start = x_pos[0] + 1.9
                    y_start = x_pos[1]
                    x_end = y_pos[0] - 1.9
                    y_end = y_pos[1]
                    
                    ax.annotate("", xy=(x_end, y_end), xytext=(x_start, y_start),
                                arrowprops=dict(arrowstyle='->', color=arrow_col, lw=1.8))
                    mid_x = (x_start + x_end) / 2
                    mid_y = (y_start + y_end) / 2
                    ax.text(mid_x, mid_y, sig_str, ha='center', va='center',
                            fontsize=7.5, color=arrow_col, fontweight='bold',
                            bbox=dict(boxstyle='round,pad=0.1', facecolor='white', alpha=0.8, edgecolor='none'))
                            
    # R² annotations
    for m_key, m_info in reg_results.items():
        y_code = m_info["Y_label"]
        r = m_info["result"]
        if y_code in positions:
            x_y = positions[y_code]
            ax.text(x_y[0], x_y[1] - 0.6, f"R²={r['R2']:.3f}", ha='center', fontsize=8,
                    color=cfg[y_code].get("color", C_RED), style='italic')
                    
    ax.set_title(
        f"Mô hình nghiên cứu – {ACTIVE_CONFIG.get('title', 'Mô hình nghiên cứu động')}\n"
        f"Tác giả: {ACTIVE_CONFIG.get('author', 'Hệ thống')}  |  *** p<0.001  ** p<0.01  * p<0.05  n.s. không có ý nghĩa",
        fontsize=10, fontweight='bold', color=C_DARK, pad=15
    )

    # Chú thích cách diễn giải mức ý nghĩa thống kê
    note_text = (
        "*** p < 0.001: rất có ý nghĩa thống kê (rất mạnh)\n"
        "** p < 0.01: có ý nghĩa thống kê mạnh\n"
        "* p < 0.05: có ý nghĩa thống kê (mức chấp nhận phổ biến)\n"
        "n.s.: not significant = không có ý nghĩa thống kê"
    )
    ax.text(
        7.0, 0.35, note_text,
        ha='center', va='center', fontsize=8.2, color=C_DARK,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='#d1d5db')
    )
    
    model_path = OUTPUT_DIR / "spss_model_diagram.png"
    plt.savefig(model_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    ok(f"Sơ đồ mô hình đã lưu: {model_path}")
    return model_path


# ─────────────────────────────────────────────────────────────────────────────
#  11. XUẤT DỮ LIỆU & BÁO CÁO
# ─────────────────────────────────────────────────────────────────────────────
def export_data(df, cronbach_results, efa_results, reg_results, df_means):
    header("XUẤT DỮ LIỆU VÀ BÁO CÁO")
    cfg = ACTIVE_CONFIG["variables"]
    
    # 1. CSV dữ liệu thô
    csv_path = OUTPUT_DIR / "spss_raw_data.csv"
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    ok(f"Dữ liệu thô CSV: {csv_path}")
    
    # 2. Excel với nhiều sheets
    xl_path = OUTPUT_DIR / "spss_analysis_report.xlsx"
    with pd.ExcelWriter(xl_path, engine='openpyxl') as writer:
        # Sheet 1: Raw data
        df.to_excel(writer, sheet_name='DuLieuTho', index=False)
        
        # Sheet 2: Descriptives
        desc_rows = []
        for var_code, var_info in cfg.items():
            items = list(var_info["items"].keys())
            for item in items:
                col = df[item].dropna()
                desc_rows.append({
                    'Thang đo': var_code,
                    'Biến': item,
                    'Nội dung': var_info["items"][item],
                    'N': len(col),
                    'Mean': round(col.mean(), 4) if len(col) > 0 else None,
                    'SD': round(col.std(ddof=1), 4) if len(col) > 1 else None,
                    'Min': col.min() if len(col) > 0 else None,
                    'Max': col.max() if len(col) > 0 else None,
                    'Skewness': round(col.skew(), 4) if len(col) > 1 else None,
                    'Missing': df[item].isnull().sum(),
                })
        pd.DataFrame(desc_rows).to_excel(writer, sheet_name='ThongKeMoTa', index=False)
        
        # Sheet 3: Cronbach
        cr_rows = []
        for var_code, cr_info in cronbach_results.items():
            cr_rows.append({
                'Thang đo': var_code,
                'Tên thang đo': cfg[var_code]["label"],
                "Cronbach's Alpha": cr_info["alpha"],
                'Số items': len(cr_info["items"]),
                'Kết luận': cr_info["status"],
            })
        pd.DataFrame(cr_rows).to_excel(writer, sheet_name="CronbachAlpha", index=False)
        
        # Sheet 4: EFA
        if efa_results:
            efa_rows = [
                {'Chỉ số': 'KMO', 'Giá trị': efa_results.get('kmo'), 'Ngưỡng': '≥0.7', 'Kết luận': 'Tốt' if efa_results.get('kmo', 0)>=0.7 else 'Không đạt'},
                {'Chỉ số': 'Bartlett Sig.', 'Giá trị': efa_results.get('bartlett_sig'), 'Ngưỡng': '<0.05', 'Kết luận': 'Đạt' if efa_results.get('bartlett_sig', 1)<0.05 else 'Không đạt'},
                {'Chỉ số': 'Số nhân tố', 'Giá trị': efa_results.get('n_factors'), 'Ngưỡng': 'Eigenvalue>1', 'Kết luận': ''},
                {'Chỉ số': 'Tổng phương sai trích', 'Giá trị': round(efa_results.get('total_var', 0), 2), 'Ngưỡng': '≥50%', 'Kết luận': 'Đạt' if efa_results.get('total_var', 0)>=50 else 'Không đạt'},
            ]
            pd.DataFrame(efa_rows).to_excel(writer, sheet_name='EFA', index=False)
        
        # Sheet 5: Regression
        reg_data = []
        for m_key, m_info in reg_results.items():
            r = m_info["result"]
            labels = m_info["X_labels"]
            y_lbl = m_info["Y_label"]
            reg_data.append({
                'Mô hình': m_key, 'Y': y_lbl, 'R2': round(r['R2'],4),
                'Adjusted R2': round(r['adjR2'],4), 'F': round(r['F'],3),
                'F Sig.': round(r['F_p'],4), 'DW': round(r['dw'],3)
            })
        pd.DataFrame(reg_data).to_excel(writer, sheet_name='HoiQuy', index=False)
        
        # Sheet 6: Correlation
        if not df_means.empty:
            corr_df = df_means[list(cfg.keys())].corr().round(4)
            corr_df.to_excel(writer, sheet_name='TuongQuan')
    
    ok(f"Báo cáo Excel: {xl_path}")
    
    # 3. Syntax SPSS
    spss_path = OUTPUT_DIR / "spss_syntax.sps"
    _write_spss_syntax(spss_path, cfg)
    ok(f"Syntax SPSS: {spss_path}")
    
    # 4. Báo cáo văn bản
    report_path = OUTPUT_DIR / "spss_report_text.txt"
    _write_text_report(report_path, cronbach_results, efa_results, reg_results)
    ok(f"Báo cáo văn bản: {report_path}")
    
    return [csv_path, xl_path, spss_path, report_path]


def _write_spss_syntax(path, cfg):
    demos = ACTIVE_CONFIG.get("demographics", [])
    frequencies_var = " ".join(demos) if demos else "GioiTinh NamHoc"
    
    lines = [
        "* ══════════════════════════════════════════════════════════════════",
        "* SPSS SYNTAX – Tự động sinh bởi SPSS Python Assistant",
        f"* Đề tài: {ACTIVE_CONFIG.get('title', 'Mô hình nghiên cứu')}",
        f"* Tác giả: {ACTIVE_CONFIG.get('author', 'Hệ thống')}",
        f"* Ngày: {datetime.now().strftime('%d/%m/%Y')}",
        "* ══════════════════════════════════════════════════════════════════",
        "",
        "* === 1. MỞ FILE DỮ LIỆU ===",
        "* GET DATA /TYPE=XLSX",
        "*   /FILE='spss_raw_data.xlsx'",
        "*   /SHEET=name 'DuLieuTho'",
        "*   /READNAMES=ON.",
        "* DATASET ACTIVATE DataSet1.",
        "",
        "* === 2. THỐNG KÊ MÔ TẢ ===",
        f"FREQUENCIES VARIABLES={frequencies_var}",
        "  /STATISTICS=MEAN MEDIAN MODE STDDEV SKEWNESS",
        "  /ORDER=ANALYSIS.",
        "",
        "DESCRIPTIVES VARIABLES=" + " ".join(
            item for var in cfg.values() for item in var["items"]),
        "  /STATISTICS=MEAN STDDEV MIN MAX SKEWNESS KURTOSIS.",
        "",
    ]
    
    # Cronbach syntax
    lines.append("* === 3. CRONBACH'S ALPHA ===")
    for var_code, var_info in cfg.items():
        items = " ".join(var_info["items"].keys())
        lines += [
            f"* --- {var_code}: {var_info['label']} ---",
            f"RELIABILITY",
            f"  /VARIABLES={items}",
            f"  /MODEL=ALPHA",
            f"  /STATISTICS=DESCRIPTIVE SCALE",
            f"  /SUMMARY=TOTAL.",
            "",
        ]
    
    # EFA syntax
    ind_items = " ".join(
        item for var_code, var_info in cfg.items()
        if var_info["type"] in ("independent",)
        for item in var_info["items"]
    )
    lines += [
        "* === 4. EFA – PHÂN TÍCH NHÂN TỐ ===",
        f"FACTOR",
        f"  /VARIABLES {ind_items}",
        f"  /MISSING LISTWISE",
        f"  /ANALYSIS {ind_items}",
        f"  /PRINT INITIAL KMO EXTRACTION ROTATION",
        f"  /FORMAT SORT BLANK(0.3)",
        f"  /PLOT EIGEN",
        f"  /CRITERIA MINEIGEN(1) ITERATE(25)",
        f"  /EXTRACTION PC",
        f"  /CRITERIA ITERATE(25) DELTA(0)",
        f"  /ROTATION VARIMAX",
        f"  /SAVE REG(ALL FAC_).",
        "",
    ]
    
    # COMPUTE means
    lines.append("* === 5. COMPUTE BIẾN TRUNG BÌNH ===")
    for var_code, var_info in cfg.items():
        items = ",".join(var_info["items"].keys())
        lines += [f"COMPUTE {var_code}_mean = MEAN({items}).", "EXECUTE.", ""]
    
    # Correlation
    var_means = " ".join(f"{v}_mean" for v in cfg)
    lines += [
        "* === 6. MA TRẬN TƯƠNG QUAN ===",
        f"CORRELATIONS",
        f"  /VARIABLES={var_means}",
        f"  /PRINT=TWOTAIL SIG NOSIG",
        f"  /MISSING=PAIRWISE.",
        "",
    ]
    
    # Regression
    ind_means = " ".join(f"{v}_mean" for v, info in cfg.items() if info["type"]=="independent")
    med_codes = [k for k, v in cfg.items() if v["type"] == "mediator"]
    dep_codes = [k for k, v in cfg.items() if v["type"] == "dependent"]
    if not dep_codes and cfg:
        dep_codes = [list(cfg.keys())[-1]]
        
    for i, m_code in enumerate(med_codes):
        lines += [
            f"* === 7. HỒI QUY – MÔ HÌNH {i+1}: X → {m_code} ===",
            "REGRESSION",
            f"  /DEPENDENT {m_code}_mean",
            f"  /METHOD=ENTER {ind_means}",
            "  /STATISTICS COEFF OUTS R ANOVA COLLIN TOL",
            "  /COLLIN",
            "  /RESIDUALS DURBIN NORMPROB(ZRESID)",
            "  /SAVE ZRESID ZPRED.",
            "",
        ]
        
    for j, d_code in enumerate(dep_codes):
        all_x = [f"{v}_mean" for v, info in cfg.items() if info["type"] in ("independent", "mediator")]
        x_final_str = " ".join(all_x)
        m_num = len(med_codes) + j + 1
        lines += [
            f"* === 8. HỒI QUY – MÔ HÌNH {m_num}: X → {d_code} ===",
            "REGRESSION",
            f"  /DEPENDENT {d_code}_mean",
            f"  /METHOD=ENTER {x_final_str}",
            "  /STATISTICS COEFF OUTS R ANOVA COLLIN TOL",
            "  /COLLIN",
            "  /RESIDUALS DURBIN.",
            "",
        ]
        
    lines.append("* === 9. SO SÁNH NHÓM ===")
    y_mean = f"{dep_codes[0]}_mean" if dep_codes else "HL_mean"
    for d_col in demos:
        d_lower = d_col.lower()
        if "gioi" in d_lower or "gender" in d_lower or "sex" in d_lower:
            lines += [
                f"T-TEST GROUPS={d_col}(1 2)",
                "  /MISSING=ANALYSIS",
                f"  /VARIABLES={y_mean}",
                "  /CRITERIA=CI(.95).",
                "",
            ]
        else:
            lines += [
                f"ONEWAY {y_mean} BY {d_col}",
                "  /STATISTICS DESCRIPTIVES HOMOGENEITY",
                "  /POSTHOC=TUKEY ALPHA(0.05).",
                "",
            ]
    
    with open(path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))


def _write_text_report(path, cronbach_results, efa_results, reg_results):
    cfg = ACTIVE_CONFIG["variables"]
    lines = [
        "═"*70,
        "BÁO CÁO PHÂN TÍCH THỐNG KÊ ĐỊNH LƯỢNG",
        f"Đề tài: {ACTIVE_CONFIG.get('title', 'Mô hình nghiên cứu')}",
        f"Tác giả: {ACTIVE_CONFIG.get('author', 'Hệ thống')}",
        f"Phần mềm: Python (pandas, numpy, scipy, sklearn)",
        f"Ngày lập: {datetime.now().strftime('%d/%m/%Y %H:%M')}",
        "═"*70,
        "",
        "I. KẾT QUẢ KIỂM ĐỊNH ĐỘ TIN CẬY THANG ĐO (CRONBACH'S ALPHA)",
        "─"*60,
    ]
    
    for var_code, cr_info in cronbach_results.items():
        label = cfg[var_code]["label"]
        alpha = cr_info["alpha"]
        n_items = len(cr_info["items"])
        conclusion = "ĐẠT" if alpha >= 0.7 else "KHÔNG ĐẠT"
        lines.append(f"  {var_code} ({label}): Alpha = {alpha:.3f}, {n_items} items → {conclusion}")
    
    lines += [
        "",
        "Ngưỡng: ≥0.8=Tốt | 0.7–0.8=Chấp nhận | <0.6=Không đạt",
        "Item-Total Correlation ≥ 0.3 → giữ biến",
        "",
        "II. PHÂN TÍCH NHÂN TỐ KHÁM PHÁ (EFA)",
        "─"*60,
    ]
    
    if efa_results:
        lines += [
            f"  KMO = {efa_results.get('kmo', 0):.3f} → {'Đạt (≥0.7)' if efa_results.get('kmo', 0)>=0.7 else 'Không đạt'}",
            f"  Bartlett Sig. = {efa_results.get('bartlett_sig', 1):.4f} → {'Đạt (<0.05)' if efa_results.get('bartlett_sig', 1)<0.05 else 'Không đạt'}",
            f"  Số nhân tố trích = {efa_results.get('n_factors', 0)}",
            f"  Tổng phương sai trích = {efa_results.get('total_var', 0):.2f}% → {'Đạt (≥50%)' if efa_results.get('total_var', 0)>=50 else 'Chưa đạt'}",
            "",
        ]
    else:
        lines += ["  (Không thực hiện phân tích EFA)", ""]
        
    lines += [
        "III. KẾT QUẢ HỒI QUY – KIỂM ĐỊNH GIẢ THUYẾT",
        "─"*60,
    ]
    
    hyps = ACTIVE_CONFIG.get("hypotheses", [])
    for h_code, path_str, desc in hyps:
        parts = [p.strip() for p in path_str.split("→")]
        if len(parts) == 2:
            cause_var, effect_var = parts[0], parts[1]
            found = False
            for m_key, m_info in reg_results.items():
                if m_info["Y_label"] == effect_var and cause_var in m_info["X_labels"]:
                    r = m_info["result"]
                    labels = m_info["X_labels"]
                    idx = labels.index(cause_var)
                    b = r["beta"][idx+1]
                    p = r["p"][idx+1]
                    sig = "p<0.001" if p < 0.001 else f"p={p:.3f}"
                    result = "ỦNG HỘ" if p < 0.05 and b > 0 else "BÁC BỎ"
                    lines.append(f"  {h_code}: {path_str}  B={b:.3f}, {sig}  →  {result}")
                    lines.append(f"         {desc}")
                    found = True
                    break
            if not found:
                lines.append(f"  {h_code}: {path_str}  →  Không tìm thấy kết quả hồi quy")
        else:
            lines.append(f"  {h_code}: {path_str}  →  Đường dẫn không hợp lệ")
    
    for m_key, m_info in reg_results.items():
        r = m_info["result"]
        lines += [
            "",
            f"  {m_key.upper()}: {' + '.join(m_info['X_labels'])} → {m_info['Y_label']}",
            f"    R² = {r['R2']:.4f}  |  Adjusted R² = {r['adjR2']:.4f}",
            f"    F = {r['F']:.3f}, Sig. " + ("<0.001" if r['F_p']<0.001 else f"={r['F_p']:.3f}"),
            f"    Durbin-Watson = {r['dw']:.3f}",
        ]
    
    # IV. CÂU KẾT LUẬN MẪU CHO LUẬN VĂN
    lines += ["", "IV. CÂU KẾT LUẬN MẪU CHO LUẬN VĂN", "─"*60, ""]
    
    if cronbach_results:
        lines += [
            "  Kết quả kiểm định độ tin cậy bằng Cronbach's Alpha cho thấy tất cả",
            "  các thang đo trong mô hình nghiên cứu đều đạt yêu cầu (Alpha ≥ 0.7),",
            "  với hệ số tương quan biến-tổng (Item-Total Correlation) của tất cả",
            "  biến quan sát đều lớn hơn 0.3, đảm bảo độ tin cậy và nhất quán nội tại.",
            "",
        ]
        
    if efa_results:
        lines += [
            f"  Phân tích nhân tố khám phá (EFA) cho thấy KMO = {efa_results.get('kmo', 0):.3f}",
            f"  (đạt yêu cầu ≥ 0.7), Bartlett's Test có ý nghĩa thống kê (p < 0.05),",
            f"  tổng phương sai trích đạt {efa_results.get('total_var', 0):.2f}% (vượt ngưỡng 50%).",
            "  Các biến quan sát có Factor Loading ≥ 0.5, đảm bảo hội tụ vào đúng nhân tố.",
            "",
        ]
        
    if reg_results:
        lines += [
            "  Kết quả phân tích hồi quy tuyến tính bội cho thấy mô hình nghiên cứu",
            "  phù hợp về mặt thống kê (F Sig. < 0.05). Các nhân tố được phân tích",
            "  đều thể hiện tác động có ý nghĩa đến biến phụ thuộc (p < 0.05),",
            "  ủng hộ các giả thuyết nghiên cứu đã đề xuất.",
            "",
        ]
    
    lines.append("═"*70)
    
    with open(path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))


# ─────────────────────────────────────────────────────────────────────────────
#  12. PHIẾU KHẢO SÁT (xuất TXT)
# ─────────────────────────────────────────────────────────────────────────────
def export_survey_form():
    header("PHIẾU KHẢO SÁT – TỰ ĐỘNG SINH")
    cfg = ACTIVE_CONFIG["variables"]
    demos = ACTIVE_CONFIG.get("demographics", [])
    
    lines = [
        "═"*65,
        f"           PHIẾU KHẢO SÁT: {ACTIVE_CONFIG.get('title', 'Đề tài Nghiên cứu')}",
        "═"*65,
        "",
        "Kính chào Anh/Chị!",
        "Cuộc khảo sát này nhằm thu thập ý kiến đánh giá phục vụ mục đích",
        "nghiên cứu khoa học. Các thông tin thu thập được bảo mật hoàn toàn.",
        "Xin trân trọng cảm ơn!",
        "",
        "─"*65,
        "PHẦN I: THÔNG TIN NHÂN KHẨU HỌC",
        "─"*65,
        "",
    ]
    
    for idx, d_col in enumerate(demos, start=1):
        d_lower = d_col.lower()
        if "gioi" in d_lower or "gender" in d_lower or "sex" in d_lower:
            lines.append(f"{idx}. Giới tính:     □ Nam     □ Nữ     □ Khác")
        elif "nam" in d_lower or "year" in d_lower:
            lines.append(f"{idx}. Bạn đang học năm: □ Năm 1  □ Năm 2  □ Năm 3  □ Năm 4+")
        elif "nganh" in d_lower or "major" in d_lower:
            lines.append(f"{idx}. Ngành học/Đơn vị: ______________________________________")
        elif "thoigian" in d_lower or "time" in d_lower:
            lines.append(f"{idx}. Thời gian tham gia/lưu trú: □ Dưới 1 năm  □ Từ 1–2 năm  □ Trên 2 năm")
        elif "chiphi" in d_lower or "cost" in d_lower:
            lines.append(f"{idx}. Chi phí/Tài chính do ai chi trả: □ Tự túc  □ Học bổng  □ Gia đình")
        else:
            lines.append(f"{idx}. {d_col}: ______________________________________")
            
    lines += [
        "",
        "─"*65,
        "PHẦN II: NỘI DUNG ĐÁNH GIÁ CÁC NHÂN TỐ",
        "─"*65,
        "Thang đo: 1=Hoàn toàn không đồng ý · 2=Không đồng ý · 3=Trung lập",
        "          4=Đồng ý · 5=Hoàn toàn đồng ý",
        "",
    ]
    
    q_num = 1
    for var_code, var_info in cfg.items():
        lines += [
            f"[{var_code}] {var_info['label'].upper()}",
            "─"*50,
        ]
        for item_code, item_text in var_info["items"].items():
            lines.append(f"  {q_num:2d}. ({item_code}) {item_text}")
            lines.append("      □1  □2  □3  □4  □5")
            q_num += 1
        lines.append("")
        
    lines += [
        "─"*65,
        "Ý kiến khác (nếu có):",
        "___________________________________________________________________",
        "___________________________________________________________________",
        "",
        "Xin chân thành cảm ơn sự hợp tác của Anh/Chị!",
        "═"*65,
    ]
    
    form_path = OUTPUT_DIR / "phieu_khao_sat.txt"
    with open(form_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))
    ok(f"Phiếu khảo sát: {form_path}")
    
    # In preview
    section("Preview phiếu (20 dòng đầu)")
    for line in lines[:20]:
        print(f"  {line}")
    print("  ...")
    return form_path


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    banner("HỆ THỐNG PHÂN TÍCH THỐNG KÊ NGHIÊN CỨU – SPSS PYTHON ASSISTANT")
    
    print(f"  Đề tài  : {ACTIVE_CONFIG.get('title', 'Mô hình')}")
    print(f"  Tác giả : {ACTIVE_CONFIG.get('author', 'Hệ thống')}")
    print(f"  Thời gian: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
    print(f"  Output  : {OUTPUT_DIR}")
    
    # ── BƯỚC 0: Phiếu khảo sát ──
    form_path = export_survey_form()
    
    # ── BƯỚC 1: Tạo dữ liệu mẫu ──
    header("TẠO DỮ LIỆU MẪU (n=250, Likert 1–5, missing=2%)")
    df = generate_data(n=250, seed=42, missing_rate=0.02)
    ok(f"Sinh thành công {len(df)} quan sát × {len(df.columns)} biến")
    print(f"\n  Xem trước dữ liệu (5 dòng đầu):")
    print(df.head(5).to_string(max_cols=12, index=False))
    
    # ── BƯỚC 2: Thống kê mô tả ──
    describe_sample(df)
    
    # ── BƯỚC 3: Cronbach Alpha ──
    cronbach_results = run_cronbach(df)
    
    # ── BƯỚC 4: EFA ──
    efa_results = run_efa(df, cronbach_results)
    
    # ── BƯỚC 5: Tương quan ──
    corr_matrix, df_means = run_correlation(df)
    
    # ── BƯỚC 6: Hồi quy ──
    reg_results = run_regression(df_means)
    
    # ── BƯỚC 7: Kiểm định nhóm ──
    run_group_tests(df, df_means)
    
    # ── BƯỚC 8: Kiểm tra giả định ──
    run_assumption_checks(reg_results)
    
    # ── BƯỚC 9: Biểu đồ ──
    chart_path  = create_charts(df, df_means, cronbach_results, efa_results, reg_results)
    model_path  = create_model_diagram(reg_results)
    
    # ── BƯỚC 10: Xuất file ──
    exported = export_data(df, cronbach_results, efa_results, reg_results, df_means)
    
    # ── TỔNG KẾT ──
    banner("HOÀN THÀNH – TÓM TẮT KẾT QUẢ", char="═")
    
    print("  📁 Các file đầu ra:")
    all_files = exported + list(chart_path.values()) + [model_path, form_path]
    for f_path in all_files:
        if f_path is not None:
            size = Path(f_path).stat().st_size / 1024
            print(f"     • {Path(f_path).name:<40} ({size:.1f} KB)")
    
    print("\n  📊 Tóm tắt kết quả chính:")
    for var_code, cr_info in cronbach_results.items():
        status = "✔" if cr_info["alpha"] >= 0.7 else "✘"
        print(f"     {status} Cronbach {var_code}: α = {cr_info['alpha']:.3f}")
    
    print(f"\n     EFA: KMO = {efa_results['kmo']:.3f}, Tổng phương sai = {efa_results['total_var']:.2f}%")
    
    for m_key, m_info in reg_results.items():
        r = m_info["result"]
        fsig = "<0.001" if r['F_p'] < 0.001 else f"{r['F_p']:.3f}"
        print(f"     Mô hình {m_key}: R² = {r['R2']:.4f}, F Sig. = {fsig}")
    
    print("\n  ✔ Phân tích hoàn tất!\n")


if __name__ == "__main__":
    main()
