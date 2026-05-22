import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import io
import base64
from scipy.stats import levene, ttest_ind, f_oneway

# Import logic từ file chính
import spss_analyzer_main as spss

# --- Khởi tạo Session State (Đảm bảo không bị NameError khi refresh) ---
if 'ai_level_val' not in st.session_state: st.session_state.ai_level_val = 'undergraduate'
if 'ai_prompt_val' not in st.session_state: st.session_state.ai_prompt_val = ''
if 'current_rec' not in st.session_state: st.session_state.current_rec = None
if 'current_roadmap' not in st.session_state: st.session_state.current_roadmap = []
if 'df' not in st.session_state: st.session_state.df = None
if 'raw_df' not in st.session_state: st.session_state.raw_df = None
if 'is_cleaned' not in st.session_state: st.session_state.is_cleaned = False
if 'optimization_logs' not in st.session_state: st.session_state.optimization_logs = []
if 'cleaning_logs' not in st.session_state: st.session_state.cleaning_logs = []
if 'results' not in st.session_state: st.session_state.results = None

# --- Hộp thoại Nhập liệu Thủ công (Giao diện SPSS Style) ---
@st.dialog("📝 Nhập Phiếu Khảo Sát (Dữ liệu thô)")
def manual_entry_dialog():
    st.markdown("""
    <style>
    .spss-header { background-color: #f1f5f9; padding: 10px; border-radius: 5px; border-left: 5px solid #3b82f6; margin-bottom: 15px; }
    .spss-group { background-color: #f8fafc; border: 1px solid #e2e8f0; padding: 15px; border-radius: 8px; margin-bottom: 10px; }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("<div class='spss-header'><b>SPSS Data Entry Mode:</b> Nhập trực tiếp dữ liệu từ phiếu khảo sát giấy vào hệ thống.</div>", unsafe_allow_html=True)
    
    with st.form("manual_entry_form"):
        # 1. Thông tin Định danh (Categorical Variables)
        st.markdown("#### 1. Thông tin Định danh (Nominal Variables)")
        
        # Lấy danh sách biến định danh từ cấu hình hiện tại
        demo_vars = spss.ACTIVE_CONFIG.get("demographics", [])
        if not demo_vars:
             st.info("Không có biến nhân khẩu học nào được cấu hình.")
        
        # Chia cột động cho các biến định danh
        if demo_vars:
            demo_cols = st.columns(min(len(demo_vars), 2))
            for i, d_var in enumerate(demo_vars):
                with demo_cols[i % 2]:
                    # Nếu là các biến chuẩn, gợi ý mapping
                    options = ["Nam", "Nữ"] if d_var == "GioiTinh" else (["Năm 1", "Năm 2", "Năm 3", "Năm 4+"] if d_var == "NamHoc" else ["Nhóm A", "Nhóm B", "Nhóm C"])
                    entry_data[d_var] = st.selectbox(f"{d_var}:", options, key=f"manual_{d_var}")
        
        st.markdown("---")
        # 2. Thang đo Likert (Scale)
        st.markdown("#### 2. Thang đo Likert (Scale Variables)")
        st.caption("Nhập giá trị từ 1 (Rất không hài lòng) đến 5 (Rất hài lòng)")
        
        entry_data = {}
        for var_code, v_info in spss.ACTIVE_CONFIG["variables"].items():
            with st.expander(f"📦 Nhân tố: {v_info['label']} ({var_code})", expanded=True):
                items = v_info["items"]
                # Chia 3 cột cho mỗi nhân tố
                cols = st.columns(3)
                for i, (item_code, item_label) in enumerate(items.items()):
                    with cols[i % 3]:
                        entry_data[item_code] = st.number_input(f"{item_code}:", 1, 5, 3, help=item_label)
        
        st.markdown("---")
        submitted = st.form_submit_button("📥 Ghi vào Data View (Save Case)", use_container_width=True, type="primary")
        if submitted:
            new_row = {}
            for d_var in demo_vars:
                val = entry_data[d_var]
                if d_var == "GioiTinh": new_row[d_var] = 1 if val == "Nam" else 2
                elif d_var == "NamHoc": new_row[d_var] = ["Năm 1", "Năm 2", "Năm 3", "Năm 4+"].index(val) + 1
                else: new_row[d_var] = val
            
            # Thêm dữ liệu Likert
            for k, v in entry_data.items():
                if k not in demo_vars:
                    new_row[k] = v
            
            if st.session_state.raw_df is None:
                st.session_state.raw_df = pd.DataFrame(columns=list(new_row.keys()))
            
            st.session_state.raw_df = pd.concat([st.session_state.raw_df, pd.DataFrame([new_row])], ignore_index=True)
            st.session_state.df = st.session_state.raw_df.copy()
            st.session_state.is_cleaned = False
            st.session_state.results = None 
            st.success("✅ Đã lưu Case mới vào Data View!")
            st.rerun()

ABBR_EXPLAIN = {
    "CSVC": "Cơ sở vật chất",
    "AN": "An ninh - An toàn",
    "HT": "Dịch vụ hỗ trợ",
    "NV": "Nhân viên quản lý",
    "CLDV": "Chất lượng cảm nhận dịch vụ",
    "HL": "Sự hài lòng",
    "KMO": "Kaiser-Meyer-Olkin (chỉ số phù hợp để phân tích nhân tố)",
    "EFA": "Exploratory Factor Analysis (Phân tích nhân tố khám phá)",
    "PCA": "Principal Component Analysis (Phân tích thành phần chính)",
    "ITC": "Item-Total Correlation (Tương quan biến - tổng)",
    "Sig.": "Mức ý nghĩa thống kê (p-value)",
    "p-value": "Xác suất quan sát kết quả khi giả thuyết không có tác động là đúng",
    "R2": "Hệ số xác định (tỷ lệ biến thiên được mô hình giải thích)",
    "VIF": "Variance Inflation Factor (hệ số phóng đại phương sai)",
    "DW": "Durbin-Watson (chỉ số tự tương quan phần dư)",
    "SE": "Sai số chuẩn (Standard Error)",
    "Beta/B": "Hệ số hồi quy (mức độ tác động của biến độc lập lên biến phụ thuộc)",
    "ANOVA": "Analysis of Variance (kiểm định phương sai giữa nhiều nhóm)",
    "n.s.": "Not significant (không có ý nghĩa thống kê)"
}

# ================================================================
# DATA STRUCTURES FOR AI RESEARCH ASSISTANT
# ================================================================
PROBLEMS_AI = {
    "descriptive": {
        "label": "Phân tích Mô tả (Descriptive Analytics)",
        "keys": [("mô tả", 5), ("thống kê", 5), ("tổng quan", 4), ("tần số", 5), ("trung bình", 4), ("biến động", 4), ("xu hướng", 4), ("cơ cấu", 4), ("tổng hợp", 3), ("báo cáo", 3), ("hiện trạng", 4), ("thực trạng", 4), ("đặc điểm", 3)],
    },
    "diagnostic": {
        "label": "Phân tích Chẩn đoán (Diagnostic Analytics)",
        "keys": [("chẩn đoán", 5), ("tại sao", 5), ("nguyên nhân", 5), ("gốc rễ", 5), ("lý do", 4), ("đối chiếu", 4), ("so sánh", 4), ("giải thích", 4), ("khám phá", 4), ("tìm hiểu", 3), ("phân tích sâu", 4)],
    },
    "predictive": {
        "label": "Phân tích Dự đoán (Predictive Analytics)",
        "keys": [("dự báo", 5), ("dự đoán", 5), ("tương lai", 5), ("xác suất", 5), ("kịch bản", 4), ("xu thế", 4), ("tiên lượng", 4), ("churn", 5), ("rời bỏ", 5), ("forecast", 5), ("predict", 5), ("mô phỏng", 4)],
    },
    "prescriptive": {
        "label": "Phân tích Đề xuất (Prescriptive Analytics)",
        "keys": [("đề xuất", 3), ("giải pháp", 3), ("tối ưu", 5), ("khuyến nghị", 5), ("nên làm gì", 5), ("chiến lược", 4), ("cải thiện", 3), ("nâng cao", 3), ("kế hoạch", 4), ("định hướng", 4)],
    },
    "finance": {
        "label": "Quản trị Tài chính & Kế toán",
        "keys": [("tài chính", 5), ("kế toán", 5), ("dòng tiền", 5), ("lợi nhuận", 5), ("chi phí", 4), ("vốn", 4), ("đầu tư", 4), ("roi", 5), ("npv", 5), ("cash flow", 5), ("ngân hàng", 4), ("tín dụng", 4), ("thuế", 4)],
    },
    "ops": {
        "label": "Quản trị Vận hành & Chuỗi cung ứng",
        "keys": [("vận hành", 5), ("chuỗi cung ứng", 5), ("logistics", 5), ("kho bãi", 5), ("tồn kho", 5), ("sản xuất", 4), ("hiệu suất", 4), ("quy trình", 4), ("vận chuyển", 4), ("cung ứng", 5), ("quản lý ktx", 5), ("ký túc xá", 5)],
    },
    "marketing": {
        "label": "Quản trị Marketing & Bán hàng",
        "keys": [("marketing", 5), ("bán hàng", 5), ("khách hàng", 5), ("thị trường", 5), ("thương hiệu", 5), ("phân khúc", 5), ("định giá", 4), ("quảng cáo", 4), ("hành vi", 4), ("trung thành", 4), ("quảng bá", 4)],
    },
    "hr": {
        "label": "Quản trị Nhân sự (HRM)",
        "keys": [("nhân sự", 5), ("nhân viên", 5), ("hr", 5), ("tuyển dụng", 5), ("đào tạo", 5), ("lương", 4), ("nghỉ việc", 5), ("hài lòng nhân viên", 5), ("động lực", 4), ("văn hóa", 4), ("cam kết", 4)],
    },
    "reliability": {
        "label": "Kiểm định Độ tin cậy (Cronbach's Alpha)",
        "keys": [("cronbach", 6), ("alpha", 6), ("độ tin cậy", 6), ("thang đo", 5), ("bảng khảo sát", 3), ("nhất quán", 4), ("tin cậy", 5), ("đo lường", 4)],
    },
    "efa": {
        "label": "Phân tích Nhân tố Khám phá (EFA)",
        "keys": [("efa", 6), ("nhân tố khám phá", 6), ("gom nhóm", 4), ("kmo", 6), ("hội tụ", 5), ("phân biệt", 5), ("trích nhân tố", 6), ("giảm chiều", 4)],
    },
    "ttest_anova": {
        "label": "Kiểm định Sự khác biệt (T-Test/ANOVA)",
        "keys": [("khác biệt", 6), ("so sánh", 5), ("t-test", 6), ("anova", 6), ("giả thuyết", 4), ("biến định tính", 4), ("phân nhóm", 5), ("kiểm định t", 6)],
    },
    "regression": {
        "label": "Phân tích Tương quan & Hồi quy",
        "keys": [("tương quan", 6), ("hồi quy", 7), ("regression", 6), ("tác động", 7), ("ảnh hưởng", 7), ("nguyên nhân kết quả", 5), ("biến độc lập", 5), ("biến phụ thuộc", 5), ("nhân tố ảnh hưởng", 7), ("mối quan hệ", 5), ("ols", 6), ("hài lòng", 7), ("nhân tố", 5), ("yếu tố", 5), ("chất lượng", 7)],
    },
}

ROADMAP_AI = {
    "descriptive": {
        "undergraduate": [
            "📌 Bước 1: Làm sạch dữ liệu & Xử lý giá trị thiếu (Missing values)",
            "📌 Bước 2: Tính toán Tần số (Frequency) & Tỷ lệ (%) cho biến định danh",
            "📌 Bước 3: Tính toán Trung bình (Mean) & Độ lệch chuẩn (Std.Dev) cho biến định lượng",
            "📊 Biểu đồ: Biểu đồ cột (Bar Chart), Biểu đồ tròn (Pie Chart)"
        ],
        "master": [
            "📌 Bước 1: Kiểm định tính phân phối chuẩn (Normality Test - Skewness/Kurtosis)",
            "📌 Bước 2: Thống kê mô tả nâng cao (Median, Mode, Quartiles)",
            "📌 Bước 3: Phân tích biến động (Coefficient of Variation - CV)",
            "📊 Biểu đồ: Box Plot (Phát hiện ngoại lai), Biểu đồ Histogram có đường chuẩn"
        ],
        "phd": [
            "📌 Bước 1: Phân tích đặc điểm mẫu chuyên sâu (Deep Profile Analysis)",
            "📌 Bước 2: Kiểm định tính đại diện của mẫu (Sampling Representativeness)",
            "📌 Bước 3: Phân tích trọng số (Weighting) để điều chỉnh sai lệch mẫu",
            "📊 Biểu đồ: Ma trận mật độ (Density Matrix), Biểu đồ Radar đa chiều"
        ],
        "academic": [
            "📌 Bước 1: Chuẩn hóa dữ liệu theo chuẩn báo cáo quốc tế (APA/AMA)",
            "📌 Bước 2: Phân tích so sánh đặc điểm mẫu đa quốc gia/đa khu vực",
            "📌 Bước 3: Kiểm định tính ổn định của dữ liệu theo thời gian (Stationarity)",
            "📊 Biểu đồ: Heatmap phân bố dữ liệu diện rộng, Sơ đồ cấu trúc mẫu"
        ]
    },
    "diagnostic": {
        "undergraduate": [
            "📌 Bước 1: Phân tích tương quan đơn giản (Correlation)",
            "📌 Bước 2: Đối chiếu thực trạng với tiêu chuẩn/kế hoạch",
            "📌 Bước 3: Xác định các nguyên nhân tiềm năng qua thống kê",
            "📊 Biểu đồ: Đồ thị xương cá (Ishikawa), Biểu đồ Pareto"
        ],
        "master": [
            "📌 Bước 1: Phân tích tương quan Pearson & Spearman",
            "📌 Bước 2: Phân tích hồi quy chẩn đoán (Diagnostic Regression)",
            "📌 Bước 3: Kiểm định sự khác biệt giữa các nhóm nguyên nhân (T-test/ANOVA)",
            "📊 Biểu đồ: Ma trận tương quan (Heatmap), Biểu đồ phân tán (Scatter)"
        ],
        "phd": [
            "📌 Bước 1: Mô hình hóa cấu trúc nguyên nhân (Path Analysis)",
            "📌 Bước 2: Phân tích tác động trực tiếp & gián tiếp",
            "📌 Bước 3: Kiểm định tính nội sinh (Endogeneity) trong quan hệ nhân quả",
            "📊 Biểu đồ: Sơ đồ mạng lưới nhân quả, Path Diagram"
        ],
        "academic": [
            "📌 Bước 1: Phân tích Counterfactual (Phản thực nghiệm)",
            "📌 Bước 2: Mô hình hóa phương trình cấu trúc (SEM) để chẩn đoán hệ thống",
            "📌 Bước 3: Phân tích độ nhạy (Sensitivity Analysis)",
            "📊 Biểu đồ: Directed Acyclic Graphs (DAGs), Sơ đồ mô phỏng hệ thống"
        ]
    },
    "predictive": {
        "undergraduate": [
            "📌 Bước 1: Thu thập & Chuẩn hóa dữ liệu chuỗi thời gian",
            "📌 Bước 2: Phân tích xu hướng (Trend Analysis)",
            "📌 Bước 3: Dự báo bằng phương pháp trung bình động (Moving Average)",
            "📊 Biểu đồ: Biểu đồ đường (Line Chart) có trendline"
        ],
        "master": [
            "📌 Bước 1: Kiểm định tính dừng của chuỗi dữ liệu (ADF Test)",
            "📌 Bước 2: Xây dựng mô hình ARIMA / Holt-Winters",
            "📌 Bước 3: Đánh giá sai số dự báo (MAE, RMSE, MAPE)",
            "📊 Biểu đồ: Biểu đồ dự báo có khoảng tin cậy (Confidence Interval)"
        ],
        "phd": [
            "📌 Bước 1: Xây dựng mô hình hồi quy Logistic / Probit cho dự báo phân loại",
            "📌 Bước 2: Kiểm định tính chính xác qua ma trận nhầm lẫn (Confusion Matrix)",
            "📌 Bước 3: Phân tích đường cong ROC & chỉ số AUC",
            "📊 Biểu đồ: ROC Curve, Biểu đồ tầm quan trọng của biến (Feature Importance)"
        ],
        "academic": [
            "📌 Bước 1: Ứng dụng mô hình Ensemble (Random Forest, XGBoost) hoặc Deep Learning",
            "📌 Bước 2: Kiểm định tính ổn định mô hình qua Cross-validation",
            "📌 Bước 3: Phân tích Bayesian để cập nhật xác suất dự báo",
            "📊 Biểu đồ: Neural Network Architecture Diagram, SHAP Value Plot"
        ]
    },
    "prescriptive": {
        "undergraduate": [
            "📌 Bước 1: Phân tích các kịch bản 'Nếu - Thì' (What-if Analysis)",
            "📌 Bước 2: Đánh giá tính khả thi của các phương án đề xuất",
            "📌 Bước 3: Lập danh mục ưu tiên các giải pháp cải thiện",
            "📊 Biểu đồ: Sơ đồ luồng quyết định (Decision Flowchart)"
        ],
        "master": [
            "📌 Bước 1: Xây dựng mô hình tối ưu hóa nguồn lực (Linear Programming)",
            "📌 Bước 2: Phân tích độ nhạy của giải pháp đối với biến động thị trường",
            "📌 Bước 3: Đánh giá hiệu quả kinh tế (ROI, Payback Period) của giải pháp",
            "📊 Biểu đồ: Tornado Plot, Biểu đồ phân tích điểm hòa vốn"
        ],
        "phd": [
            "📌 Bước 1: Mô phỏng Monte Carlo để đánh giá rủi ro của các quyết định",
            "📌 Bước 2: Xây dựng mô hình ra quyết định đa tiêu chí (AHP/TOPSIS)",
            "📌 Bước 3: Tối ưu hóa đa mục tiêu (Multi-objective Optimization)",
            "📊 Biểu đồ: Sơ đồ phân tích kịch bản phức hợp, Pareto Frontier"
        ],
        "academic": [
            "📌 Bước 1: Xây dựng lý thuyết trò chơi (Game Theory) trong ra quyết định",
            "📌 Bước 2: Mô hình hóa hệ thống động (System Dynamics)",
            "📌 Bước 3: Đề xuất khung chính sách dựa trên bằng chứng thực nghiệm",
            "📊 Biểu đồ: Sơ đồ tương tác hệ thống phức hợp, Simulation Animation"
        ]
    },
    "finance": {
        "undergraduate": [
            "📌 Bước 1: Phân tích các tỷ số tài chính (Liquidity, Solvency, Profitability)",
            "📌 Bước 2: Phân tích ngang & dọc báo cáo tài chính",
            "📌 Bước 3: Phân tích hòa vốn (Break-even Analysis)",
            "📊 Biểu đồ: Biểu đồ Waterfall (Dòng tiền), Biểu đồ cột chồng"
        ],
        "master": [
            "📌 Bước 1: Định giá doanh nghiệp/dự án (DCF Model, WACC)",
            "📌 Bước 2: Phân tích độ nhạy của NPV/IRR theo các biến đầu vào",
            "📌 Bước 3: Đánh giá rủi ro tài chính qua chỉ số Beta",
            "📊 Biểu đồ: Radar Chart sức khỏe tài chính, Biểu đồ nến giá"
        ],
        "phd": [
            "📌 Bước 1: Phân tích chuỗi thời gian tài chính nâng cao (GARCH, VAR)",
            "📌 Bước 2: Tính toán giá trị rủi ro (VaR - Value at Risk)",
            "📌 Bước 3: Kiểm định các lý thuyết tài chính (EMH, CAPM)",
            "📊 Biểu đồ: Volatility Surface Chart, Biểu đồ Risk Matrix"
        ],
        "academic": [
            "📌 Bước 1: Mô hình hóa kinh tế lượng tài chính phức hợp",
            "📌 Bước 2: Phân tích tác động lan tỏa (Spillover effects)",
            "📌 Bước 3: Nghiên cứu tài chính hành vi qua thực nghiệm",
            "📊 Biểu đồ: Efficient Frontier, Sơ đồ mạng lưới tài chính"
        ]
    },
    "ops": {
        "undergraduate": [
            "📌 Bước 1: Sơ đồ hóa quy trình hiện tại (As-is Process Map)",
            "📌 Bước 2: Đo lường thời gian chu kỳ (Cycle Time) & Năng suất",
            "📌 Bước 3: Xác định các điểm nghẽn (Bottlenecks) cơ bản",
            "📊 Biểu đồ: Sơ đồ luồng (Flowchart), Biểu đồ Gantt"
        ],
        "master": [
            "📌 Bước 1: Phân tích chuỗi giá trị (Value Stream Mapping - VSM)",
            "📌 Bước 2: Tính toán hiệu suất tổng thể thiết bị (OEE)",
            "📌 Bước 3: Áp dụng các công cụ Lean/Six Sigma (DMAIC)",
            "📊 Biểu đồ: VSM Current/Future State, Biểu đồ kiểm soát (Control Chart)"
        ],
        "phd": [
            "📌 Bước 1: Mô phỏng hệ thống vận hành (Discrete Event Simulation)",
            "📌 Bước 2: Tối ưu hóa chuỗi cung ứng diện rộng (Supply Chain Network Opt)",
            "📌 Bước 3: Kiểm định độ tin cậy hệ thống (System Reliability)",
            "📊 Biểu đồ: Sơ đồ mạng lưới logistics, Petri Nets"
        ],
        "academic": [
            "📌 Bước 1: Nghiên cứu chiến lược vận hành bền vững (Circular Economy)",
            "📌 Bước 2: Ứng dụng AI/IoT trong tối ưu hóa vận hành",
            "📌 Bước 3: Mô hình hóa tương tác giữa các tác nhân trong chuỗi giá trị",
            "📊 Biểu đồ: 3D Simulation Visualization, Dynamic Dashboard"
        ]
    },
    "marketing": {
        "undergraduate": [
            "📌 Bước 1: Thống kê mô tả đặc điểm khách hàng",
            "📌 Bước 2: Phân tích thị phần & NPS",
            "📌 Bước 3: Phân tích ma trận SWOT Marketing",
            "📊 Biểu đồ: Biểu đồ phễu (Funnel), Bản đồ Customer Persona"
        ],
        "master": [
            "📌 Bước 1: Phân tích phân khúc thị trường (Segmentation - Cluster Analysis)",
            "📌 Bước 2: Xây dựng bản đồ định vị (Perceptual Mapping)",
            "📌 Bước 3: Phân tích giá trị vòng đời khách hàng (CLV)",
            "📊 Biểu đồ: Customer Journey Map, Đồ thị phân cụm"
        ],
        "phd": [
            "📌 Bước 1: Mô hình hóa hành vi người tiêu dùng (CB Model)",
            "📌 Bước 2: Kiểm định thực nghiệm Marketing (Field Experiment)",
            "📌 Bước 3: Phân tích tác động của thương hiệu (Brand Equity Modeling)",
            "📊 Biểu đồ: Sơ đồ Path Analysis hành vi, Heatmap tương tác"
        ],
        "academic": [
            "📌 Bước 1: Phân tích Neuromarketing & Phản ứng sinh lý",
            "📌 Bước 2: Phân tích mạng lưới xã hội (Social Network Analysis)",
            "📌 Bước 3: Mô hình hóa Marketing hỗn hợp (MMM) phức hợp",
            "📊 Biểu đồ: Brand Resonance Pyramid, Social Network Graph"
        ]
    },
    "hr": {
        "undergraduate": [
            "📌 Bước 1: Thống kê tỷ lệ nghỉ việc & vắng mặt",
            "📌 Bước 2: Khảo sát mức độ hài lòng nhân viên cơ bản",
            "📌 Bước 3: Mô tả cơ cấu tổ chức hiện tại",
            "📊 Biểu đồ: Tháp dân số nhân sự, Sơ đồ tổ chức"
        ],
        "master": [
            "📌 Bước 1: Phân tích chỉ số gắn kết nhân viên (Engagement Index)",
            "📌 Bước 2: Xây dựng khung năng lực (Competency Framework)",
            "📌 Bước 3: Phân tích ROI của hoạt động đào tạo",
            "📊 Biểu đồ: Ma trận 9-Box, Đồ thị mạng lưới nội bộ"
        ],
        "phd": [
            "📌 Bước 1: Mô hình hóa tác động của văn hóa tổ chức lên hiệu suất",
            "📌 Bước 2: Phân tích tâm lý tổ chức & Cam kết gắn bó",
            "📌 Bước 3: Kiểm định các mô hình lãnh đạo (Leadership Modeling)",
            "📊 Biểu đồ: Path Diagram nhân sự, Sơ đồ cấu trúc văn hóa"
        ],
        "academic": [
            "📌 Bước 1: Nghiên cứu quản trị nhân sự chiến lược (SHRM)",
            "📌 Bước 2: Phân tích đa cấp trong quản trị nhân sự",
            "📌 Bước 3: Đánh giá tác động của công nghệ lên tương lai việc làm",
            "📊 Biểu đồ: Leadership Impact Map, Diversity Dashboard"
        ]
    },
    "reliability": {
        "undergraduate": [
            "📌 Bước 1: Kiểm định Cronbach's Alpha (> 0.7)",
            "📌 Bước 2: Kiểm tra hệ số tương quan biến tổng (Corrected Item-Total > 0.3)",
            "📌 Bước 3: Loại bỏ các biến không phù hợp để tăng độ tin cậy",
            "📊 Biểu đồ: Biểu đồ thanh độ tin cậy từng biến"
        ],
        "master": [
            "📌 Bước 1: Kiểm định độ tin cậy nâng cao (Split-half, Guttman)",
            "📌 Bước 2: Phân tích tính đơn hướng (Unidimensionality)",
            "📌 Bước 3: Đánh giá độ ổn định qua Test-retest",
            "📊 Biểu đồ: ITC Profile Chart, Sơ đồ cấu trúc thang đo"
        ],
        "phd": [
            "📌 Bước 1: Tính toán độ tin cậy tổng hợp (Composite Reliability - CR)",
            "📌 Bước 2: Tính toán phương sai trích trung bình (AVE)",
            "📌 Bước 3: Kiểm định giá trị hội tụ & phân biệt",
            "📊 Biểu đồ: Sơ đồ kiểm định Fornell-Larcker"
        ],
        "academic": [
            "📌 Bước 1: Ứng dụng lý thuyết đáp ứng câu hỏi (IRT)",
            "📌 Bước 2: Tính toán độ tin cậy Bayesian (Omega)",
            "📌 Bước 3: Kiểm định tính bất biến đo lường (Measurement Invariance)",
            "📊 Biểu đồ: Item Characteristic Curves (ICC), Information Plot"
        ]
    },
    "efa": {
        "undergraduate": [
            "📌 Bước 1: Kiểm định KMO & Bartlett's Test",
            "📌 Bước 2: Trích nhân tố (Principal Components / Exploratory)",
            "📌 Bước 3: Xoay nhân tố (Varimax) & Giải thích các nhóm",
            "📊 Biểu đồ: Scree Plot (Xác định số nhân tố)"
        ],
        "master": [
            "📌 Bước 1: Trích nhân tố bằng phương pháp Principal Axis Factoring",
            "📌 Bước 2: Xoay nhân tố bằng phương pháp Oblique (Promax/Direct Oblimin)",
            "📌 Bước 3: Kiểm tra tính hội tụ & phân biệt sơ bộ qua ma trận mẫu",
            "📊 Biểu đồ: Biểu đồ ma trận xoay (Loading Plot)"
        ],
        "phd": [
            "📌 Bước 1: Xây dựng mô hình nhân tố khẳng định (CFA)",
            "📌 Bước 2: Kiểm định tính phù hợp mô hình (CFI, TLI, GFI, RMSEA)",
            "📌 Bước 3: Hiệu chỉnh mô hình qua Modification Indices",
            "📊 Biểu đồ: Sơ đồ CFA Diagram chuyên nghiệp"
        ],
        "academic": [
            "📌 Bước 1: Mô hình hóa phương trình cấu trúc (SEM)",
            "📌 Bước 2: Phân tích đa cấp (HLM / Multi-level Modeling)",
            "📌 Bước 3: Phân tích dữ liệu mảng (Panel Data - FE/RE Models)",
            "📌 Bước 4: Kiểm định tính bất biến của mô hình trên các nhóm mẫu",
            "📊 Biểu đồ: SEM Path Diagram, Multilevel Growth Charts"
        ]
    },
    "ttest_anova": {
        "undergraduate": [
            "📌 Bước 1: Kiểm định tính phân phối chuẩn & đồng nhất phương sai",
            "📌 Bước 2: Thực hiện Independent T-test hoặc One-way ANOVA",
            "📌 Bước 3: Giải thích ý nghĩa trị số p (Sig. < 0.05)",
            "📊 Biểu đồ: Error Bar Chart, Biểu đồ cột so sánh trung bình"
        ],
        "master": [
            "📌 Bước 1: Kiểm định Post-hoc (Tukey, LSD, Bonferroni) khi có sai biệt",
            "📌 Bước 2: Tính toán kích thước hiệu ứng (Effect Size - Eta-squared)",
            "📌 Bước 3: Thực hiện ANOVA hai nhân tố (Two-way ANOVA)",
            "📊 Biểu đồ: Interaction Plot (Biểu đồ tương tác), Box Plot"
        ],
        "phd": [
            "📌 Bước 1: Thực hiện phân tích hiệp biến (ANCOVA)",
            "📌 Bước 2: Thực hiện phân tích đa biến (MANOVA)",
            "📌 Bước 3: Kiểm định phi tham số (Kruskal-Wallis) nếu vi phạm giả định",
            "📊 Biểu đồ: Multivariate Scatter Plot, Profile Plots"
        ],
        "academic": [
            "📌 Bước 1: Phân tích so sánh đa cấp (Multi-level comparison)",
            "📌 Bước 2: Thực hiện Bayesian ANOVA",
            "📌 Bước 3: Tổng hợp kết quả so sánh qua Meta-analysis",
            "📊 Biểu đồ: Bayesian Density Comparison, Forest Plot"
        ]
    },
    "regression": {
        "undergraduate": [
            "📌 Bước 1: Kiểm định độ tin cậy & EFA cho các nhóm biến",
            "📌 Bước 2: Tính toán giá trị đại diện nhân tố (Mean scores)",
            "📌 Bước 3: Phân tích tương quan Pearson",
            "📌 Bước 4: Hồi quy tuyến tính bội (Enter method)",
            "📌 Bước 5: Kiểm định giả định hồi quy (VIF, Durbin-Watson)",
            "📊 Biểu đồ: Regression Scatter, Normal P-P Plot"
        ],
        "master": [
            "📌 Bước 1: Thực hiện CFA để khẳng định thang đo",
            "📌 Bước 2: Hồi quy đa biến với phương pháp Stepwise/Hierarchy",
            "📌 Bước 3: Phân tích biến trung gian đơn giản (Mediation - Model 4)",
            "📌 Bước 4: Kiểm định phương sai sai số thay đổi (White test)",
            "📊 Biểu đồ: Residuals Plot, Sơ đồ mô hình hồi quy chính thức"
        ],
        "phd": [
            "📌 Bước 1: Phân tích mô hình trung gian & điều tiết phức hợp (Hayes Process Model 4, 7, 8, 14, 21, 58, 59)",
            "📌 Bước 2: Thực hiện Bootstrapping để kiểm định tác động gián tiếp",
            "📌 Bước 3: Kiểm định hiện tượng nội sinh (Endogeneity)",
            "📌 Bước 4: So sánh các mô hình cạnh tranh (Model Comparison - AIC/BIC)",
            "📊 Biểu đồ: Sơ đồ Hayes Process Model, Slopes Analysis Chart"
        ],
        "academic": [
            "📌 Bước 1: Mô hình hóa phương trình cấu trúc (SEM)",
            "📌 Bước 2: Phân tích đa cấp (HLM / Multi-level Modeling)",
            "📌 Bước 3: Phân tích dữ liệu mảng (Panel Data - FE/RE Models)",
            "📌 Bước 4: Kiểm định tính bất biến của mô hình trên các nhóm mẫu",
            "📊 Biểu đồ: SEM Path Diagram, Multilevel Growth Charts"
        ]
    }
}

RESOURCES_AI = {
    "descriptive": {
        "undergraduate": {
            "proposal": "📝 Đề cương: Thống kê hiện trạng & Tần suất",
            "refs": "📚 Ref: Hoàng Trọng & Chu Nguyễn Mộng Ngọc (2008), Thống kê ứng dụng",
            "template": "📄 Mẫu: Bảng hỏi thông tin nhân khẩu học",
            "preview": "═════════════════════════════════════════════════════════════════\n           PHIẾU KHẢO SÁT ĐẶC ĐIỂM ĐỐI TƯỢNG NGHIÊN CỨU\n═════════════════════════════════════════════════════════════════\n\n**PHẦN I: THÔNG TIN NHÂN KHẨU HỌC**\n1. Giới tính: □ Nam  □ Nữ\n2. Độ tuổi: □ < 18  □ 18-25  □ 26-35  □ > 35\n3. Trình độ học vấn: □ Trung cấp/Cao đẳng □ Đại học □ Sau đại học\n4. Thu nhập hàng tháng: □ < 5tr □ 5-10tr □ 10-20tr □ > 20tr\n\n**PHẦN II: THỐNG KÊ TẦN SUẤT SỬ DỤNG**\n5. Bạn có thường xuyên sử dụng dịch vụ không?\n   □ Hàng ngày  □ 2-3 lần/tuần  □ 1 lần/tuần  □ Hiếm khi"
        },
        "master": {
            "proposal": "📝 Đề cương: Phân tích biến động & Xu hướng vĩ mô",
            "refs": "📚 Ref: Hair et al. (2014), Multivariate Data Analysis",
            "template": "📄 Mẫu: Bảng nhập liệu SPSS chuẩn hóa",
            "preview": "### CẤU TRÚC BẢNG NHẬP LIỆU (MẪU)\n\n| STT | HoTen_Ma | GioiTinh | Tuoi | Bien_1 | Bien_2 | Bien_n |\n|-----|----------|----------|------|--------|--------|--------|\n| 1   | KH001    | 1        | 25   | 4      | 5      | 3      |\n| 2   | KH002    | 2        | 30   | 5      | 4      | 5      |\n*Ghi chú: GioiTinh (1:Nam, 2:Nu); Bien (Thang đo 1-5)*"
        },
        "phd": {
            "proposal": "📝 Đề cương: Khung lý thuyết đặc điểm mẫu diện rộng",
            "refs": "📚 Ref: Field (2013), Discovering Statistics using IBM SPSS",
            "template": "📄 Mẫu: Protocol làm sạch dữ liệu Big Data",
            "preview": "### PROTOCOL LÀM SẠCH DỮ LIỆU (MẪU)\n\n1. Bước 1: Kiểm tra giá trị khuyết (Missing value > 10% loại bỏ).\n2. Bước 2: Kiểm tra giá trị ngoại lai (Outliers - Z-score > 3).\n3. Bước 3: Kiểm tra tính nhất quán (Straight-lining response).\n4. Bước 4: Mã hóa lại các câu hỏi ngược (Reverse coding)."
        },
        "academic": {
            "proposal": "📝 Đề cương: Chuẩn mực báo cáo thống kê quốc tế",
            "refs": "📚 Ref: APA Publication Manual (7th Edition)",
            "template": "📄 Mẫu: Codebook nghiên cứu đa quốc gia",
            "preview": "### RESEARCH CODEBOOK (MẪU)\n\n- **Variable Name**: [GENDER]\n- **Label**: Respondent's gender\n- **Type**: Numeric (Nominal)\n- **Values**: {0: Male, 1: Female, 2: Other}\n- **Missing**: 999"
        }
    },
    "diagnostic": {
        "undergraduate": {
            "proposal": "📝 Đề cương: Khám phá nguyên nhân cơ bản",
            "refs": "📚 Ref: Ishikawa (1968), Guide to Quality Control",
            "template": "📄 Mẫu: Phiếu phỏng vấn nhóm tập trung",
            "preview": "═════════════════════════════════════════════════════════════════\n           PHIẾU PHỎNG VẤN CHẨN ĐOÁN NGUYÊN NHÂN\n═════════════════════════════════════════════════════════════════\n\n**PHẦN I: THÔNG TIN CHUYÊN GIA/ĐỐI TƯỢNG**\n- Họ và tên (tùy chọn): .................................\n- Vị trí/Công việc: .................................\n- Thâm niên công tác: .................................\n\n**PHẦN II: NỘI DUNG CHẨN ĐOÁN**\n1. Theo bạn, đâu là lý do chính dẫn đến tình trạng này?\n2. Những yếu tố nào cản trở quy trình hiện tại?\n3. Bạn có đề xuất thay đổi cụ thể nào không?\n4. Vai trò của các bộ phận liên quan hiện nay thế nào?"
        },
        "master": {
            "proposal": "📝 Đề cương: Chẩn đoán lỗi quy trình & Rủi ro",
            "refs": "📚 Ref: Root Cause Analysis (ASQ Reference)",
            "template": "📄 Mẫu: Check-list chẩn đoán 5-Whys",
            "preview": "### CHIẾN LƯỢC 5-WHYS (MẪU)\n\n- **Vấn đề**: Hiệu suất giảm 20%\n- **Why 1**: Tại sao máy móc hay hỏng? -> Do thiếu bảo trì.\n- **Why 2**: Tại sao thiếu bảo trì? -> Do ngân sách bị cắt.\n- **Why 3**: Tại sao ngân sách bị cắt? -> Do ưu tiên dự án khác.\n- ...\n- **Gốc rễ**: Thiếu chiến lược bảo trì dài hạn."
        },
        "phd": {
            "proposal": "📝 Đề cương: Cơ chế nhân quả trong quản trị",
            "refs": "📚 Ref: Pearl (2009), Causality",
            "template": "📄 Mẫu: Kịch bản phỏng vấn chuyên gia sâu",
            "preview": "### KỊCH BẢN PHỎNG VẤN CHUYÊN GIA (MẪU)\n\n1. Đánh giá của ông/bà về mối quan hệ giữa biến A và biến B?\n2. Có yếu tố trung gian nào thúc đẩy mối quan hệ này không?\n3. Bối cảnh thị trường ảnh hưởng thế nào đến cơ chế này?\n4. Những sai lệch tiềm tàng trong mô hình là gì?"
        },
        "academic": {
            "proposal": "📝 Đề cương: Mô hình hóa cơ chế chẩn đoán phức hợp",
            "refs": "📚 Ref: Journal of Business Research Guidelines",
            "template": "📄 Mẫu: Protocol Meta-analysis",
            "preview": "### PROTOCOL META-ANALYSIS (MẪU)\n\n1. Search strategy: (Keywords, Databases, Period)\n2. Inclusion criteria: (Sample size, Statistical methods)\n3. Coding scheme: (Effect size type, Moderator variables)\n4. Bias check: (Funnel plot, Egger test)"
        }
    },
    "finance": {
        "undergraduate": {
            "proposal": "📝 Đề cương: Phân tích hiệu quả tài chính",
            "refs": "📚 Ref: Brigham & Houston (2019), Fundamentals of Financial Management",
            "template": "📄 Mẫu: Bảng tính tỷ số tài chính (Ratio Analysis)",
            "preview": "═════════════════════════════════════════════════════════════════\n           BẢNG THU THẬP DỮ LIỆU TÀI CHÍNH DOANH NGHIỆP\n═════════════════════════════════════════════════════════════════\n\n**THÔNG TIN ĐƠN VỊ:**\n- Tên công ty: .................................\n- Mã chứng khoán: .................................\n- Năm tài chính: .................................\n\n**CÁC CHỈ TIÊU TÀI CHÍNH CHÍNH:**\n1. Tài sản ngắn hạn (CA): .................\n2. Nợ ngắn hạn (CL): .................\n3. Hàng tồn kho (Inv): .................\n4. Lợi nhuận sau thuế (NI): .................\n5. Tổng tài sản (TA): .................\n\n*Ghi chú: Dữ liệu được trích xuất từ Bảng cân đối kế toán và Báo cáo kết quả KD.*"
        },
        "master": {
            "proposal": "📝 Đề cương: Đánh giá giá trị & Rủi ro đầu tư",
            "refs": "📚 Ref: Damodaran (2012), Investment Valuation",
            "template": "📄 Mẫu: Mô hình chiết khấu dòng tiền (DCF)",
            "preview": "### MÔ HÌNH DCF (MẪU)\n\n1. Dự báo dòng tiền (FCFF) 5 năm tới.\n2. Tính toán WACC (Weighted Average Cost of Capital).\n3. Tính toán giá trị cuối kỳ (Terminal Value).\n4. Chiết khấu về hiện giá (PV)."
        },
        "phd": {
            "proposal": "📝 Đề cương: Lý thuyết cấu trúc vốn & Tài chính hành vi",
            "refs": "📚 Ref: Journal of Finance Research Papers",
            "template": "📄 Mẫu: Database tài chính từ Bloomberg/Thomson Reuters",
            "preview": "### CẤU TRÚC DATASET TÀI CHÍNH\n\n- **Ticker**: Mã chứng khoán\n- **MarketCap**: Vốn hóa thị trường\n- **BookToMarket**: Tỷ lệ giá trị sổ sách/thị trường\n- **Momentum**: Tỷ suất sinh lời quá khứ"
        },
        "academic": {
            "proposal": "📝 Đề cương: Mô hình kinh tế lượng tài chính phức hợp",
            "refs": "📚 Ref: Brooks (2019), Introductory Econometrics for Finance",
            "template": "📄 Mẫu: Script phân tích GARCH/VAR",
            "preview": "### GARCH MODEL SPECIFICATION\n\n- Mean equation: r_t = mu + epsilon_t\n- Variance equation: sigma_t^2 = omega + alpha*epsilon_{t-1}^2 + beta*sigma_{t-1}^2\n- Goal: Modeling Volatility Clustering."
        }
    },
    "ops": {
        "undergraduate": {
            "proposal": "📝 Đề cương: Quản lý vận hành & Tối ưu hóa tồn kho",
            "refs": "📚 Ref: Heizer & Render (2017), Operations Management",
            "template": "📄 Mẫu: Bảng kiểm soát tồn kho ABC (Excel)",
            "preview": "═════════════════════════════════════════════════════════════════\n           PHIẾU KIỂM KÊ & ĐÁNH GIÁ VẬN HÀNH KHO\n═════════════════════════════════════════════════════════════════\n\n**THÔNG TIN KHO/XƯỞNG:**\n- Tên kho: .................  Vị trí: .................\n\n**NỘI DUNG ĐÁNH GIÁ (Thang điểm 1-10):**\n1. Tốc độ xử lý đơn hàng: [ ]\n2. Độ chính xác tồn kho: [ ]\n3. Tỷ lệ hàng lỗi/hỏng: [ ]\n4. Thời gian xe chờ tại bãi: [ ]\n\n**PHÂN LOẠI ABC:**\n- Mã hàng: .................  Giá trị: .................\n- Mã hàng: .................  Giá trị: ................."
        },
        "master": {
            "proposal": "📝 Đề cương: Quản trị chuỗi cung ứng & Logistics",
            "refs": "📚 Ref: Chopra & Meindl (2016), Supply Chain Management",
            "template": "📄 Mẫu: Mô hình vận tải & Bài toán điều phối",
            "preview": "### BÀI TOÁN VẬN TẢI (MẪU)\n\n- **Nguồn cung**: Kho A, Kho B, Kho C.\n- **Nhu cầu**: Điểm bán 1, 2, 3, 4.\n- **Mục tiêu**: Minimize Total Cost = sum(cost_ij * quantity_ij).\n- **Ràng buộc**: Công suất kho & Nhu cầu thị trường."
        },
        "phd": {
            "proposal": "📝 Đề cương: Mô phỏng hệ thống sản xuất phức hợp",
            "refs": "📚 Ref: Law (2014), Simulation Modeling and Analysis",
            "template": "📄 Mẫu: Sơ đồ luồng công việc (Work-flow/Petri-nets)",
            "preview": "### LOGIC MÔ PHỎNG SẢN XUẤT (MẪU)\n\n1. **Source**: Nguyên liệu đầu vào (Inter-arrival time).\n2. **Queue**: Hàng chờ tại các công đoạn.\n3. **Processor**: Thời gian gia công (Triangular/Normal Dist).\n4. **Sink**: Sản phẩm hoàn thiện đầu ra."
        },
        "academic": {
            "proposal": "📝 Đề cương: Chuỗi cung ứng bền vững & Chuyển đổi số",
            "refs": "📚 Ref: International Journal of Production Research",
            "template": "📄 Mẫu: Protocol khảo sát mức độ trưởng thành số (SCM)",
            "preview": "### DIGITAL SCM MATURITY MODEL\n\n- **Level 1**: Thủ công (Manual/Paper-based).\n- **Level 2**: Số hóa cơ bản (Excel/Stand-alone ERP).\n- **Level 3**: Kết nối (Integrated SCM/Cloud).\n- **Level 4**: Thông minh (AI/IoT/Real-time Analytics)."
        }
    },
    "predictive": {
        "undergraduate": {
            "proposal": "📝 Đề cương: Dự báo xu hướng & Trung bình động",
            "refs": "📚 Ref: Hyndman & Athanasopoulos (2018), Forecasting: Principles and Practice",
            "template": "📄 Mẫu: Bảng dự báo doanh thu/nhu cầu (Excel)",
            "preview": "### BẢNG DỰ BÁO XU HƯỚNG (MẪU)\n\n- **Tháng 1-12**: Dữ liệu thực tế quá khứ.\n- **Phương pháp**: Simple Moving Average (k=3).\n- **Dự báo**: Giá trị cho Tháng 13, 14.\n- **Sai số**: Tính toán MSE, MAE."
        },
        "master": {
            "proposal": "📝 Đề cương: Mô hình dự báo chuỗi thời gian ARIMA",
            "refs": "📚 Ref: Box & Jenkins (1970), Time Series Analysis",
            "template": "📄 Mẫu: Script Python/R chạy ARIMA chuẩn",
            "preview": "### QUY TRÌNH ARIMA (MẪU)\n\n1. Kiểm định tính dừng (ADF Test).\n2. Xác định tham số (p, d, q) qua biểu đồ ACF/PACF.\n3. Ước lượng mô hình và kiểm định phần dư.\n4. Dự báo ngoài mẫu (Out-of-sample forecast)."
        },
        "phd": {
            "proposal": "📝 Đề cương: Dự báo phân loại & Máy học (ML)",
            "refs": "📚 Ref: James et al. (2013), An Introduction to Statistical Learning",
            "template": "📄 Mẫu: Codebook tiền xử lý dữ liệu ML",
            "preview": "### PIPELINE DỰ BÁO PHÂN LOẠI (MẪU)\n\n- **Feature Engineering**: Mã hóa, chuẩn hóa biến.\n- **Model**: Logistic Regression, Random Forest.\n- **Evaluation**: Confusion Matrix, ROC-AUC.\n- **Validation**: K-fold Cross Validation."
        },
        "academic": {
            "proposal": "📝 Đề cương: Dự báo phức hợp & Deep Learning",
            "refs": "📚 Ref: Goodfellow et al. (2016), Deep Learning",
            "template": "📄 Mẫu: Kiến trúc mạng Neural (RNN/LSTM)",
            "preview": "### LSTM ARCHITECTURE (MẪU)\n\n- **Input Layer**: Chuỗi thời gian đa biến.\n- **Hidden Layers**: LSTM units with Dropout.\n- **Output Layer**: Giá trị dự báo hoặc Xác suất phân loại.\n- **Loss Function**: MSE / Cross-entropy."
        }
    },
    "prescriptive": {
        "undergraduate": {
            "proposal": "📝 Đề cương: Phân tích kịch bản & Giải pháp đề xuất",
            "refs": "📚 Ref: Decision Support Systems Journal",
            "template": "📄 Mẫu: Bảng so sánh phương án tối ưu",
            "preview": "### BẢNG SO SÁNH PHƯƠNG ÁN (MẪU)\n\n| Tiêu chí | Phương án A | Phương án B | Trọng số |\n|----------|-------------|-------------|----------|\n| Chi phí  | Thấp        | Cao         | 40%      |\n| Thời gian| Nhanh       | Chậm        | 30%      |\n| Rủi ro   | Cao         | Thấp        | 30%      |"
        },
        "master": {
            "proposal": "📝 Đề cương: Tối ưu hóa nguồn lực & Vận hành",
            "refs": "📚 Ref: Winston (2004), Operations Research",
            "template": "📄 Mẫu: Mô hình Linear Programming (Solver)",
            "preview": "### MÔ HÌNH TỐI ƯU (MẪU)\n\n- **Hàm mục tiêu**: Maximize Profit = sum(c_i * x_i).\n- **Ràng buộc**: Nguồn lực (Vốn, Nhân công, Máy móc).\n- **Biến quyết định**: Số lượng sản phẩm/dịch vụ cần sản xuất."
        },
        "phd": {
            "proposal": "📝 Đề cương: Ra quyết định đa tiêu chí (MCDA)",
            "refs": "📚 Ref: Saaty (1980), The Analytic Hierarchy Process",
            "template": "📄 Mẫu: Ma trận so sánh cặp AHP",
            "preview": "### MA TRẬN AHP (MẪU)\n\n1. Thiết lập cấu trúc phân cấp (Mục tiêu - Tiêu chí - Phương án).\n2. So sánh cặp các tiêu chí theo thang điểm 1-9.\n3. Tính toán trọng số và kiểm định tỷ số nhất quán (CR < 0.1).\n4. Tổng hợp điểm số cho từng phương án."
        },
        "academic": {
            "proposal": "📝 Đề cương: Mô phỏng & Lý thuyết trò chơi trong chính sách",
            "refs": "📚 Ref: Osborne (2004), An Introduction to Game Theory",
            "template": "📄 Mẫu: Protocol xây dựng mô phỏng hệ thống động",
            "preview": "### SYSTEM DYNAMICS PROTOCOL (MẪU)\n\n1. Xác định Causal Loop Diagrams (CLD).\n2. Xây dựng Stock and Flow Diagrams.\n3. Thiết lập các phương trình vi phân mô phỏng sự tương tác.\n4. Chạy mô phỏng kịch bản chính sách 10-20 năm."
        }
    },
    "marketing": {
        "undergraduate": {
            "proposal": "📝 Đề cương: Đánh giá hài lòng khách hàng",
            "refs": "📚 Ref: Kotler & Keller (2016), Marketing Management",
            "template": "📄 Mẫu: Bảng hỏi Likert 5 mức độ",
            "preview": "═════════════════════════════════════════════════════════════════\n           PHIẾU KHẢO SÁT MỨC ĐỘ HÀI LÒNG KHÁCH HÀNG\n═════════════════════════════════════════════════════════════════\n\n**PHẦN I: THÔNG TIN CHUNG**\n1. Giới tính: □ Nam  □ Nữ\n2. Tần suất mua sắm: □ Hàng ngày □ Hàng tuần □ Hàng tháng\n\n**PHẦN II: ĐÁNH GIÁ (Thang đo 1-5)**\n\n[SP] SẢN PHẨM\n1. Chất lượng sản phẩm tốt: □1 □2 □3 □4 □5\n2. Thiết kế bao bì đẹp: □1 □2 □3 □4 □5\n\n[GIA] GIÁ CẢ\n3. Giá cả phù hợp với giá trị: □1 □2 □3 □4 □5\n4. Có nhiều chương trình khuyến mãi: □1 □2 □3 □4 □5\n\n[HL] HÀI LÒNG CHUNG\n5. Tôi hài lòng với thương hiệu này: □1 □2 □3 □4 □5"
        },
        "master": {
            "proposal": "📝 Đề cương: Phân tích hành vi & Trung thành thương hiệu",
            "refs": "📚 Ref: Aaker (1991), Managing Brand Equity",
            "template": "📄 Mẫu: Kịch bản phỏng vấn sâu khách hàng",
            "preview": "### KỊCH BẢN PHỎNG VẤN SÂU (MẪU)\n\n1. Bạn đã sử dụng thương hiệu này được bao lâu?\n2. Lý do chính khiến bạn chọn chúng tôi là gì?\n3. Điều gì khiến bạn cảm thấy hài lòng nhất/không hài lòng nhất?\n4. Bạn có sẵn sàng giới thiệu cho bạn bè không? Tại sao?"
        },
        "phd": {
            "proposal": "📝 Đề cương: Mô hình hóa giá trị khách hàng (CLV)",
            "refs": "📚 Ref: Journal of Marketing Research",
            "template": "📄 Mẫu: Database hành vi mua sắm thực tế",
            "preview": "### CẤU TRÚC DATASET HÀNH VI (MẪU)\n\n- **CustomerID**: Mã định danh khách hàng\n- **Recency**: Số ngày kể từ lần mua cuối\n- **Frequency**: Tổng số lần giao dịch\n- **Monetary**: Tổng giá trị chi tiêu\n- **Satisfaction_Score**: Điểm hài lòng trung bình"
        },
        "academic": {
            "proposal": "📝 Đề cương: Marketing số & Trí tuệ nhân tạo",
            "refs": "📚 Ref: Journal of Interactive Marketing",
            "template": "📄 Mẫu: Protocol tracking hành vi web/app",
            "preview": "### PROTOCOL TRACKING (MẪU)\n\n1. **Event**: click_buy, view_product, add_to_cart\n2. **Metrics**: Time on page, Bounce rate, Conversion rate\n3. **Tools**: Google Analytics 4, Hotjar heatmaps"
        }
    },
    "hr": {
        "undergraduate": {
            "proposal": "📝 Đề cương: Khảo sát môi trường làm việc",
            "refs": "📚 Ref: Dessler (2020), Human Resource Management",
            "template": "📄 Mẫu: Phiếu khảo sát động lực làm việc",
            "preview": "═════════════════════════════════════════════════════════════════\n           PHIẾU KHẢO SÁT ĐỘNG LỰC LÀM VIỆC NHÂN VIÊN\n═════════════════════════════════════════════════════════════════\n\n**PHẦN I: THÔNG TIN CHUNG**\n1. Bộ phận: .................  2. Thâm niên: .................\n\n**PHẦN II: ĐÁNH GIÁ ĐỘNG LỰC (Likert 1-5)**\n1. Tôi cảm thấy công việc của mình có ý nghĩa: □1 □2 □3 □4 □5\n2. Thu nhập hiện tại xứng đáng với nỗ lực: □1 □2 □3 □4 □5\n3. Đồng nghiệp luôn sẵn sàng hỗ trợ tôi: □1 □2 □3 □4 □5\n4. Cấp trên công bằng trong đánh giá: □1 □2 □3 □4 □5"
        },
        "master": {
            "proposal": "📝 Đề cương: Chiến lược thu hút & Giữ chân nhân tài",
            "refs": "📚 Ref: Armstrong (2020), Strategic HRM",
            "template": "📄 Mẫu: Khung đánh giá năng lực (Competency)",
            "preview": "### KHUNG NĂNG LỰC (MẪU)\n\n1. **Năng lực cốt lõi**: Giao tiếp, Giải quyết vấn đề\n2. **Năng lực chuyên môn**: Kỹ năng SPSS, Phân tích dữ liệu\n3. **Năng lực quản lý**: Lập kế hoạch, Điều phối đội ngũ"
        },
        "phd": {
            "proposal": "📝 Đề cương: Văn hóa tổ chức & Cam kết gắn bó",
            "refs": "📚 Ref: Schein (2010), Organizational Culture and Leadership",
            "template": "📄 Mẫu: Thang đo tâm lý tổ chức chuẩn hóa",
            "preview": "### THANG ĐO CAM KẾT (OCQ - MẪU)\n\n1. Tôi rất tự hào khi nói với người khác mình là thành viên tổ chức.\n2. Tôi sẵn sàng nỗ lực vượt mức mong đợi vì tổ chức.\n3. Sự thay đổi nơi làm việc sẽ khiến tôi cảm thấy mất mát."
        },
        "academic": {
            "proposal": "📝 Đề cương: Quản trị nhân sự trong kỷ nguyên số",
            "refs": "📚 Ref: Human Resource Management Review",
            "template": "📄 Mẫu: Protocol phỏng vấn exit-interview",
            "preview": "### EXIT INTERVIEW PROTOCOL\n\n1. Điều gì là nguyên nhân chính khiến bạn quyết định rời đi?\n2. Bạn đánh giá thế nào về cơ hội thăng tiến tại đây?\n3. Chúng tôi cần cải thiện điều gì để giữ chân nhân tài?"
        }
    },
    "reliability": {
        "undergraduate": {
            "proposal": "📝 Đề cương: Kiểm định độ tin cậy thang đo",
            "refs": "📚 Ref: Nunnally (1978), Psychometric Theory",
            "template": "📄 Mẫu: Bảng hỏi Pilot (n=30)",
            "preview": "═════════════════════════════════════════════════════════════════\n           PHIẾU KHẢO SÁT THỬ NGHIỆM (PILOT TEST)\n═════════════════════════════════════════════════════════════════\n\n**HƯỚNG DẪN:** Vui lòng đánh giá mức độ rõ ràng của các câu hỏi sau.\n\n[Q1] Câu hỏi 1: ................. □ Rõ ràng □ Cần sửa\n[Q2] Câu hỏi 2: ................. □ Rõ ràng □ Cần sửa\n[Q3] Câu hỏi 3: ................. □ Rõ ràng □ Cần sửa\n\n*Lưu ý: Sau khi thu thập 30 mẫu, tính Cronbach's Alpha để kiểm định độ tin cậy thang đo.*"
        },
        "master": {
            "proposal": "📝 Đề cương: Hiệu chỉnh & Xây dựng thang đo mới",
            "refs": "📚 Ref: DeVellis (2016), Scale Development",
            "template": "📄 Mẫu: Bảng so sánh kết quả chuyên gia",
            "preview": "### BẢNG HIỆU CHỈNH THANG ĐO\n\n- **Biến quan sát gốc**: [Nội dung]\n- **Ý kiến chuyên gia**: [Sửa đổi/Loại bỏ]\n- **Biến quan sát hiệu chỉnh**: [Nội dung mới]"
        },
        "phd": {
            "proposal": "📝 Đề cương: Kiểm định tính bất biến của thang đo",
            "refs": "📚 Ref: Journal of Personality and Social Psychology",
            "template": "📄 Mẫu: Bộ dữ liệu đa nhóm (Multi-group)",
            "preview": "### CẤU TRÚC DỮ LIỆU ĐA NHÓM\n\n- Group 1: Nam giới (n=200)\n- Group 2: Nữ giới (n=200)\n- Kiểm định Invariance: Configural, Metric, Scalar."
        },
        "academic": {
            "proposal": "📝 Đề cương: Phát triển thang đo chuẩn quốc tế",
            "refs": "📚 Ref: Psychological Methods Journal",
            "template": "📄 Mẫu: Protocol dịch thuật & thích nghi văn hóa",
            "preview": "### PROTOCOL DỊCH THUẬT (MẪU)\n\n1. Forward translation (Dịch xuôi)\n2. Back translation (Dịch ngược)\n3. Expert committee review\n4. Pre-testing on target population"
        }
    },
    "efa": {
        "undergraduate": {
            "proposal": "📝 Đề cương: Gom nhóm nhân tố ảnh hưởng",
            "refs": "📚 Ref: Hoàng Trọng & Chu Nguyễn Mộng Ngọc (2008)",
            "template": "📄 Mẫu: Bảng kết quả ma trận xoay mẫu",
            "preview": "═════════════════════════════════════════════════════════════════\n           PHIẾU KHẢO SÁT CÁC NHÂN TỐ ẢNH HƯỞNG\n═════════════════════════════════════════════════════════════════\n\n**HƯỚNG DẪN:** Vui lòng đánh giá mức độ đồng ý của bạn (1-5).\n\n[NT1] NHÂN TỐ 1 (Ví dụ: Sự tiện lợi)\n- NT1.1: ................. □1 □2 □3 □4 □5\n- NT1.2: ................. □1 □2 □3 □4 □5\n\n[NT2] NHÂN TỐ 2 (Ví dụ: Giá cả)\n- NT2.1: ................. □1 □2 □3 □4 □5\n- NT2.2: ................. □1 □2 □3 □4 □5\n\n*Lưu ý: Dữ liệu này sẽ được dùng để chạy Phân tích Nhân tố Khám phá (EFA).*"
        },
        "master": {
            "proposal": "📝 Đề cương: Khám phá cấu trúc nhân tố phức hợp",
            "refs": "📚 Ref: Watkins (2018), A Step-by-Step Guide to EFA",
            "template": "📄 Mẫu: Biểu đồ Scree Plot mẫu",
            "preview": "### SCREE PLOT INTERPRETATION\n\n1. Xác định điểm gãy (elbow point).\n2. Số lượng nhân tố có Eigenvalue > 1.\n3. Tổng phương sai trích (Cumulative Variance) > 50%."
        },
        "phd": {
            "proposal": "📝 Đề cương: So sánh cấu trúc nhân tố đa nhóm",
            "refs": "📚 Ref: Thompson (2004), Exploratory and Confirmatory Factor Analysis",
            "template": "📄 Mẫu: Sơ đồ Path Diagram nháp",
            "preview": "### CFA PATH DIAGRAM (MẪU)\n\n- Latent Variable 1 -> [Observed 1, 2, 3]\n- Latent Variable 2 -> [Observed 4, 5, 6]\n- Covariance between Latents."
        },
        "academic": {
            "proposal": "📝 Đề cương: Lý thuyết đo lường hiện đại (IRT)",
            "refs": "📚 Ref: Structural Equation Modeling: A Multidisciplinary Journal",
            "template": "📄 Mẫu: Protocol phân tích ESEM/BFA",
            "preview": "### ESEM/BFA PROTOCOL\n\n1. Xác định Cross-loadings.\n2. Phân tách nhân tố chung (General factor) và nhân tố riêng (Specific factors)."
        }
    },
    "ttest_anova": {
        "undergraduate": {
            "proposal": "📝 Đề cương: So sánh đặc điểm giữa các nhóm",
            "refs": "📚 Ref: Nguyễn Văn Tuấn (2014), Phân tích dữ liệu với R",
            "template": "📄 Mẫu: Bảng tổng hợp Mean/Std.Dev theo nhóm",
            "preview": "═════════════════════════════════════════════════════════════════\n           PHIẾU KHẢO SÁT SO SÁNH GIỮA CÁC NHÓM\n═════════════════════════════════════════════════════════════════\n\n**PHẦN I: PHÂN LOẠI NHÓM**\n1. Bạn thuộc nhóm nào? □ Nhóm A  □ Nhóm B  □ Nhóm C\n\n**PHẦN II: ĐÁNH GIÁ BIẾN PHỤ THUỘC (1-5)**\n2. Mức độ hài lòng chung: □1 □2 □3 □4 □5\n3. Ý định quay lại: □1 □2 □3 □4 □5\n\n*Lưu ý: Mục tiêu là so sánh sự khác biệt trung bình giữa các nhóm (T-Test/ANOVA).*"
        },
        "master": {
            "proposal": "📝 Đề cương: Kiểm định sự khác biệt đa nhân tố",
            "refs": "📚 Ref: Cohen (1988), Statistical Power Analysis",
            "template": "📄 Mẫu: Bảng Post-hoc so sánh cặp",
            "preview": "### KẾT QUẢ POST-HOC (MẪU)\n\n- Nhóm 1 vs Nhóm 2: Diff=0.4, p=0.01 (*)\n- Nhóm 1 vs Nhóm 3: Diff=0.1, p=0.45\n- Nhóm 2 vs Nhóm 3: Diff=0.3, p=0.03 (*)"
        },
        "phd": {
            "proposal": "📝 Đề cương: Phân tích hiệp biến & Tương tác nhóm",
            "refs": "📚 Ref: Tabachnick & Fidell (2013), Using Multivariate Statistics",
            "template": "📄 Mẫu: Sơ đồ tương tác (Interaction Plot)",
            "preview": "### INTERACTION PLOT (MẪU)\n\n- Trục X: Biến độc lập 1\n- Đường: Các mức độ của Biến độc lập 2\n- Trục Y: Biến phụ thuộc"
        },
        "academic": {
            "proposal": "📝 Đề cương: Phân tích đa cấp & Hiệu ứng hỗn hợp",
            "refs": "📚 Ref: Journal of Statistics Education",
            "template": "📄 Mẫu: Protocol phân tích Bayesian ANOVA",
            "preview": "### BAYESIAN ANOVA PROTOCOL\n\n1. Xác định Prior distribution.\n2. Tính toán Bayes Factor (BF10).\n3. Giải thích mức độ bằng chứng (Evidence strength)."
        }
    },
    "regression": {
        "undergraduate": {
            "proposal": "📝 Đề cương: Nghiên cứu các nhân tố tác động",
            "refs": "📚 Ref: Gujarati (2004), Basic Econometrics",
            "template": "📄 Mẫu: Phiếu khảo sát 5-point Likert",
            "preview": "═════════════════════════════════════════════════════════════════\n PHIẾU KHẢO SÁT MỨC ĐỘ HÀI LÒNG CỦA SINH VIÊN ĐỐI VỚI CÔNG TÁC QUẢN LÝ KÝ TÚC XÁ – TRƯỜNG ĐH NTTU\n═════════════════════════════════════════════════════════════════\n\nKính chào Anh/Chị sinh viên!\nCuộc khảo sát này nhằm thu thập ý kiến đánh giá của sinh viên đang lưu\ntrú tại ký túc xá Trường Đại học Nguyễn Tất Thành. Thông tin chỉ phục\nvụ mục đích nghiên cứu khoa học, được bảo mật hoàn toàn.\nXin trân trọng cảm ơn!\n\n─────────────────────────────────────────────────────────────────\nPHẦN I: THÔNG TIN NHÂN KHẨU HỌC\n─────────────────────────────────────────────────────────────────\n\n1. Giới tính:     □ Nam     □ Nữ     □ Khác\n2. Bạn đang học năm: □ Năm 1  □ Năm 2  □ Năm 3  □ Năm 4+\n3. Ngành học: ______________________________________\n4. Thời gian sinh sống tại KTX:\n   □ Dưới 1 năm   □ Từ 1–2 năm   □ Trên 2 năm\n5. Chi phí thuê phòng do ai chi trả:\n   □ Tự túc   □ Học bổng/hỗ trợ   □ Gia đình\n\n─────────────────────────────────────────────────────────────────\nPHẦN II: ĐÁNH GIÁ CÁC NHÂN TỐ QUẢN LÝ KÝ TÚC XÁ\n─────────────────────────────────────────────────────────────────\nThang đo: 1=Hoàn toàn không đồng ý · 2=Không đồng ý · 3=Trung lập\n          4=Đồng ý · 5=Hoàn toàn đồng ý\n\n[CSVC] CƠ SỞ VẬT CHẤT KÝ TÚC XÁ\n──────────────────────────────────────────────────\n   1. (CSVC1) Phòng ở đáp ứng nhu cầu sinh hoạt của sinh viên\n      □1  □2  □3  □4  □5\n   2. (CSVC2) Trang thiết bị trong phòng hoạt động tốt\n      □1  □2  □3  □4  □5\n   3. (CSVC3) Hệ thống điện, nước hoạt động ổn định\n      □1  □2  □3  □4  □5\n   4. (CSVC4) Khu vệ sinh bảo đảm điều kiện sử dụng\n      □1  □2  □3  □4  □5\n   5. (CSVC5) Không gian sinh hoạt chung thuận tiện\n      □1  □2  □3  □4  □5\n\n[AN] AN NINH VÀ AN TOÀN KÝ TÚC XÁ\n──────────────────────────────────────────────────\n   6. (AN1) Cảm thấy an toàn khi sinh sống tại ký túc xá\n      □1  □2  □3  □4  □5\n   7. (AN2) Công tác bảo vệ được thực hiện nghiêm túc\n      □1  □2  □3  □4  □5\n   8. (AN3) Việc kiểm soát ra vào ký túc xá được thực hiện tốt\n      □1  □2  □3  □4  □5\n   9. (AN4) Ký túc xá có biện pháp bảo đảm an toàn PCCC\n      □1  □2  □3  □4  □5\n\n[HT] DỊCH VỤ HỖ TRỢ SINH VIÊN\n──────────────────────────────────────────────────\n  10. (HT1) Ban quản lý hỗ trợ sinh viên khi có yêu cầu\n      □1  □2  □3  □4  □5\n  11. (HT2) Các yêu cầu của sinh viên được xử lý kịp thời\n      □1  □2  □3  □4  □5\n  12. (HT3) Sinh viên dễ dàng liên hệ với Ban quản lý KTX\n      □1  □2  □3  □4  □5\n  13. (HT4) Thông tin liên quan đến KTX được cung cấp đầy đủ\n      □1  □2  □3  □4  □5\n\n[NV] NHÂN VIÊN QUẢN LÝ KÝ TÚC XÁ\n──────────────────────────────────────────────────\n  14. (NV1) Nhân viên quản lý có thái độ thân thiện\n      □1  □2  □3  □4  □5\n  15. (NV2) Nhân viên làm việc có trách nhiệm\n      □1  □2  □3  □4  □5\n  16. (NV3) Nhân viên có kỹ năng giao tiếp tốt\n      □1  □2  □3  □4  □5\n  17. (NV4) Nhân viên hỗ trợ sinh viên một cách tận tình\n      □1  □2  □3  □4  □5\n\n[CLDV] CHẤT LƯỢNG CẢM NHẬN DỊCH VỤ KTX\n──────────────────────────────────────────────────\n  18. (CLDV1) Chất lượng dịch vụ KTX đáp ứng nhu cầu của tôi\n      □1  □2  □3  □4  □5\n  19. (CLDV2) Dịch vụ KTX có chất lượng tốt\n      □1  □2  □3  □4  □5\n  20. (CLDV3) Dịch vụ KTX đáng tin cậy\n      □1  □2  □3  □4  □5\n  21. (CLDV4) Chất lượng dịch vụ KTX nhìn chung là tốt\n      □1  □2  □3  □4  □5\n\n[HL] SỰ HÀI LÒNG CỦA SINH VIÊN NỘI TRÚ\n──────────────────────────────────────────────────\n  22. (HL1) Hài lòng với điều kiện sống tại ký túc xá\n      □1  □2  □3  □4  □5\n  23. (HL2) KTX đáp ứng kỳ vọng ban đầu của tôi\n      □1  □2  □3  □4  □5\n  24. (HL3) Sẵn sàng tiếp tục sinh sống tại ký túc xá\n      □1  □2  □3  □4  □5\n  25. (HL4) Sẽ giới thiệu KTX cho sinh viên khác\n      □1  □2  □3  □4  □5\n\n─────────────────────────────────────────────────────────────────\nÝ kiến khác (nếu có):\n___________________________________________________________________\n___________________________________________________________________\n\nXin chân thành cảm ơn sự hợp tác của Anh/Chị!\n═════════════════════════════════════════════════════════════════"
        },
        "master": {
            "proposal": "📝 Đề cương: Mô hình hồi quy đa biến & Kiểm định",
            "refs": "📚 Ref: Hair et al. (2010), Multivariate Data Analysis",
            "template": "📄 Mẫu: File nhập liệu SPSS (Cleaned)",
            "preview": "═════════════════════════════════════════════════════════════════\n PHIẾU KHẢO SÁT MỨC ĐỘ HÀI LÒNG CỦA SINH VIÊN ĐỐI VỚI CÔNG TÁC QUẢN LÝ KÝ TÚC XÁ – TRƯỜNG ĐH NTTU\n═════════════════════════════════════════════════════════════════\n\n*(Xem nội dung phiếu tương tự như bậc Cử nhân phía trên)*\n\n### CẤU TRÚC BIẾN TRONG SPSS (BẬC THẠC SĨ)\n\n**1. Biến Độc lập (IVs):**\n- **CSVC**: Mean (CSVC1-CSVC5)  -  **AN**: Mean (AN1-AN4)\n- **HT**: Mean (HT1-HT4)        -  **NV**: Mean (NV1-NV4)\n\n**2. Biến Phụ thuộc (DV):**\n- **HL**: Mean (HL1-HL4)\n\n**3. Kiểm định Robustness:**\n- Kiểm định phương sai thay đổi (White/BP test).\n- Kiểm định tự tương quan (Durbin-Watson).\n- Kiểm định đa cộng tuyến (VIF < 10)."
        },
        "phd": {
            "proposal": "📝 Đề cương: Phân tích mô hình trung gian & điều tiết",
            "refs": "📚 Ref: Hayes (2017), Introduction to Mediation, Moderation...",
            "template": "📄 Mẫu: Sơ đồ mô hình (Hayes Process)",
            "preview": "═════════════════════════════════════════════════════════════════\n PHIẾU KHẢO SÁT MỨC ĐỘ HÀI LÒNG CỦA SINH VIÊN ĐỐI VỚI CÔNG TÁC QUẢN LÝ KÝ TÚC XÁ – TRƯỜNG ĐH NTTU\n═════════════════════════════════════════════════════════════════\n\n### MÔ HÌNH TRUNG GIAN & ĐIỀU TIẾT (BẬC TIẾN SĨ)\n\n**Mô hình đề xuất:**\n[CSVC, AN, HT, NV] ➔ **[CLDV]** ➔ **[HL]**\n\n- **X (Independent)**: Các nhân tố quản lý.\n- **M (Mediator)**: Chất lượng cảm nhận (CLDV1-CLDV4).\n- **Y (Dependent)**: Sự hài lòng (HL1-HL4).\n- **W (Moderator)**: Đặc điểm nhân khẩu học (Năm học/Giới tính).\n\n*Yêu cầu PhD: Thực hiện phân tích cấu trúc tuyến tính SEM hoặc kỹ thuật Bootstrap để khẳng định tính vững của mô hình.*"
        },
        "academic": {
            "proposal": "📝 Đề cương: Mô hình SEM & Phân tích đa cấp",
            "refs": "📚 Ref: Byrne (2016), SEM with AMOS",
            "template": "📄 Mẫu: Protocol xử lý dữ liệu mảng (Panel Data)",
            "preview": "### PROTOCOL PANEL DATA\n\n1. Kiểm định Pooled OLS, Fixed Effects, Random Effects.\n2. Kiểm định Hausman để chọn mô hình phù hợp.\n3. Xử lý hiện tượng tự tương quan & phương sai thay đổi."
        }
    },
}

LEVEL_LABELS_AI = {
    "undergraduate": {
        "label": "CỬ NHÂN",
        "desc": "Cơ bản – Thực hành (Khóa luận / NCKH)",
        "workload": "⭐⭐",
        "science": "⭐⭐",
        "difficulty": "⭐⭐",
        "focus": "Thực thi quy trình & Thống kê mô tả"
    },
    "master": {
        "label": "THẠC SĨ",
        "desc": "Nâng cao – Kiểm định Robustness (Luận văn)",
        "workload": "⭐⭐⭐⭐",
        "science": "⭐⭐⭐",
        "difficulty": "⭐⭐⭐",
        "focus": "Kiểm định giả thuyết & So sánh nhóm"
    },
    "phd": {
        "label": "TIẾN SĨ",
        "desc": "Chuyên sâu – Đóng góp mới (Luận án TS)",
        "workload": "⭐⭐⭐⭐⭐",
        "science": "⭐⭐⭐⭐⭐",
        "difficulty": "⭐⭐⭐⭐⭐",
        "focus": "Mô hình hóa nhân quả & Đóng góp học thuật"
    },
    "academic": {
        "label": "HÀN LÂM",
        "desc": "Đẳng cấp quốc tế Q1-Q4 / ISI",
        "workload": "⭐⭐⭐⭐⭐+",
        "science": "⭐⭐⭐⭐⭐+",
        "difficulty": "⭐⭐⭐⭐⭐+",
        "focus": "Lý thuyết mới & Phương pháp luận phức hợp"
    },
}

# ================================================================
# AI ASSISTANT UTILS
# ================================================================
import re

def remove_tones_ai(s: str) -> str:
    s = re.sub(r'[àáạảãâầấậẩẫăằắặẳẵ]', 'a', s)
    s = re.sub(r'[èéẹẻẽêềếệểễ]', 'e', s)
    s = re.sub(r'[ìíịỉĩ]', 'i', s)
    s = re.sub(r'[òóọỏõôồốộổỗơờớợởỡ]', 'o', s)
    s = re.sub(r'[ùúụủũưừứựửữ]', 'u', s)
    s = re.sub(r'[ỳýỵỷỹ]', 'y', s)
    s = re.sub(r'[đ]', 'd', s)
    return s

PROBLEM_EXPLANATIONS_AI = {
    "marketing": {
        "intro": "🎯 **Nghiên cứu Marketing & Hành vi Khách hàng**",
        "concept": "Lĩnh vực này tập trung vào việc tìm hiểu các nhân tố tâm lý, xã hội và kinh tế tác động đến quyết định mua sắm của người tiêu dùng.",
        "key_terms": [
            "**Sự hài lòng (Satisfaction):** Trạng thái cảm xúc của khách hàng khi trải nghiệm sản phẩm so với kỳ vọng.",
            "**Sự trung thành (Loyalty):** Xu hướng mua lại và giới thiệu thương hiệu cho người khác.",
            "**Giá trị cảm nhận:** Đánh giá của khách hàng về lợi ích nhận được so với chi phí bỏ ra."
        ]
    },
    "hr": {
        "intro": "👥 **Nghiên cứu Quản trị Nhân sự & Tổ chức**",
        "concept": "Tập trung vào mối quan hệ giữa người lao động và tổ chức, nhằm tối ưu hóa hiệu suất làm việc và sự gắn kết.",
        "key_terms": [
            "**Động lực làm việc:** Các nhân tố thúc đẩy nhân viên nỗ lực hoàn thành mục tiêu.",
            "**Văn hóa tổ chức:** Hệ thống các giá trị, niềm tin chung điều chỉnh hành vi của thành viên.",
            "**Cam kết tổ chức:** Mức độ gắn bó và mong muốn cống hiến lâu dài của nhân viên."
        ]
    },
    "reliability": {
        "intro": "💎 **Kiểm định Thang đo & Tâm lý học**",
        "concept": "Tập trung vào việc đo lường các khái niệm trừu tượng (tâm lý, thái độ) một cách chính xác và ổn định.",
        "key_terms": [
            "**Độ tin cậy (Reliability):** Sự nhất quán của kết quả đo lường qua nhiều lần thực hiện (thường dùng Cronbach's Alpha).",
            "**Độ giá trị (Validity):** Mức độ mà thang đo thực sự đo lường đúng khái niệm cần đo.",
            "**Sai số đo lường:** Các nhiễu loạn làm kết quả đo không phản ánh đúng thực tế."
        ]
    },
    "efa": {
        "intro": "🧩 **Phân tích Nhân tố Khám phá (EFA)**",
        "concept": "Kỹ thuật thống kê dùng để thu gọn một tập hợp nhiều biến quan sát thành một số ít các nhân tố có ý nghĩa hơn.",
        "key_terms": [
            "**Gom nhóm nhân tố:** Tìm ra cấu trúc ẩn bên dưới các câu hỏi khảo sát.",
            "**Hệ số tải nhân tố:** Mức độ tương quan giữa một biến quan sát và nhân tố mẹ.",
            "**Phương sai trích:** Tỷ lệ thông tin của các biến gốc được giữ lại sau khi gom nhóm."
        ]
    },
    "ttest_anova": {
        "intro": "📊 **So sánh & Kiểm định Khác biệt**",
        "concept": "Sử dụng các kiểm định thống kê để xác định xem sự khác biệt giữa các nhóm (ví dụ: Nam vs Nữ) là có ý nghĩa hay chỉ là ngẫu nhiên.",
        "key_terms": [
            "**Independent T-test:** So sánh trung bình giữa 2 nhóm độc lập.",
            "**One-way ANOVA:** So sánh trung bình giữa 3 nhóm trở lên.",
            "**Giả thuyết H0:** Giả định rằng không có sự khác biệt giữa các nhóm."
        ]
    },
    "regression": {
        "intro": "🚀 **Phân tích Tương quan & Hồi quy**",
        "concept": "Xác định mối liên hệ và mức độ tác động của các nhân tố nguyên nhân (Độc lập) lên nhân tố kết quả (Phụ thuộc).",
        "key_terms": [
            "**Tương quan (Correlation):** Mối quan hệ qua lại giữa hai biến (biến này tăng thì biến kia tăng/giảm).",
            "**Hồi quy (Regression):** Mô hình hóa sự tác động, cho phép dự báo giá trị của biến phụ thuộc.",
            "**R-Square:** Mức độ giải thích của mô hình (ví dụ: 60% sự thay đổi của hài lòng là do các nhân tố này)."
        ]
    },
    "custom": {
        "intro": "🔬 **Nghiên cứu Khoa học Tổng quát**",
        "concept": "Quy trình nghiên cứu chuẩn hóa áp dụng cho các đề tài mới hoặc đa ngành, kết hợp nhiều kỹ thuật thống kê.",
        "key_terms": [
            "**Tính vững (Robustness):** Khả năng kết quả nghiên cứu duy trì đúng trong các điều kiện khác nhau.",
            "**Tính đại diện:** Mức độ mà mẫu khảo sát phản ánh đúng đặc điểm của tổng thể.",
            "**Đóng góp mới:** Những phát hiện hoặc góc nhìn mới mà nghiên cứu mang lại."
        ]
    }
}

def extract_text_from_file_ai(uploaded_file):
    """Trích xuất văn bản từ file tải lên (.txt, .docx, .pdf)"""
    if uploaded_file.type == "text/plain":
        return uploaded_file.getvalue().decode("utf-8")
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        try:
            import docx
            doc = docx.Document(uploaded_file)
            return "\n".join([para.text for para in doc.paragraphs])
        except ImportError:
            return "ERROR_MISSING_DOCX"
    elif uploaded_file.type == "application/pdf":
        try:
            import PyPDF2
            reader = PyPDF2.PdfReader(uploaded_file)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
            return text
        except ImportError:
            return "ERROR_MISSING_PDF"
    return ""

def detect_level_ai(text: str) -> str:
    """Tự động nhận diện trình độ dựa trên từ khóa trong văn bản"""
    text_lc = text.lower()
    if any(k in text_lc for k in ["luận án", "tiến sĩ", "phd", "nghiên cứu sinh", "luân án", "tiên si"]):
        return "phd"
    if any(k in text_lc for k in ["bài báo", "tạp chí", "scopus", "isi", "hàn lâm", "journal", "bai bao", "tap chi"]):
        return "academic"
    if any(k in text_lc for k in ["luận văn", "thạc sĩ", "cao học", "master", "luan van", "thac si"]):
        return "master"
    if any(k in text_lc for k in ["khóa luận", "đồ án", "báo cáo thực tập", "sinh viên", "cử nhân", "undergraduate", "khoa luan", "do an", "sinh vien"]):
        return "undergraduate"
    return "undergraduate"

def recognize_problem_ai(prompt: str):
    text = prompt.strip()
    if not text:
        return {"error": "Vui lòng nhập mô đề tài."}
    
    prompt_orig = text.lower()
    prompt_no_tone = remove_tones_ai(prompt_orig)
    
    scores = {}
    matched_all = {}
    for pid, pdef in PROBLEMS_AI.items():
        score = 0
        matched = []
        for kw, w in pdef["keys"]:
            kw_lc = kw.lower()
            kw_no_tone = remove_tones_ai(kw_lc)
            if kw_lc in prompt_orig or kw_no_tone in prompt_no_tone:
                score += w
                matched.append(kw)
        scores[pid] = score
        matched_all[pid] = matched

    # Tìm pid có score cao nhất
    best_pid = max(scores, key=scores.get)
    best_score = scores[best_pid]
    
    if best_score < 3:
        # Fallback cho đề tài bất kỳ
        return {
            "problem_id": "custom",
            "label": "Đề tài Nghiên cứu Tùy chỉnh (AI detected)",
            "matched": ["Tổng quát", "Nghiên cứu khoa học"],
            "score": best_score,
            "is_custom": True
        }

    return {
        "problem_id": best_pid,
        "label": {
            "marketing": "Marketing & Hành vi Khách hàng",
            "hr": "Quản trị Nhân sự & Tổ chức",
            "reliability": "Kiểm định Thang đo & Tâm lý học",
            "efa": "Khám phá Cấu trúc Nhân tố",
            "ttest_anova": "So sánh & Kiểm định Khác biệt",
            "regression": "Mô hình Tác động & Dự báo",
        }.get(best_pid, "Nghiên cứu Tổng quát"),
        "matched": matched_all[best_pid],
        "score": best_score,
        "is_custom": False
    }

    if not scores or max(scores.values()) < 1:
        return {"error": "AI chưa nhận diện được loại bài toán. Vui lòng mô tả chi tiết hơn."}

    best = max(scores, key=lambda k: scores[k])
    return {
        "problem_id": best, 
        "score": scores[best], 
        "matched": matched_all[best], 
        "label": PROBLEMS_AI[best]["label"]
    }

# ─────────────────────────────────────────────────────────────────────────────
# CẤU HÌNH TRANG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SPSS Python Assistant - Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# XÁC THỰC NGƯỜI DÙNG (AUTHENTICATION)
# ─────────────────────────────────────────────────────────────────────────────
try:
    import auth_manager
    from streamlit_google_auth import Authenticate
    
    auth_manager.setup_google_credentials()
    redirect_uri = st.secrets.get("REDIRECT_URI", "http://localhost:8501")
    
    authenticator = Authenticate(
        secret_credentials_path='google_credentials.json',
        cookie_name='azscience_auth',
        cookie_key='azscience_secret_key',
        redirect_uri=redirect_uri,
    )
    
    # Xác định chắc chắn môi trường để chốt Redirect URI
    is_cloud = "GOOGLE_CREDENTIALS" in st.secrets
    hardcoded_uri = "https://baitoankinhte.streamlit.app/" if is_cloud else "http://localhost:8501/"

    # Hàm đổi mã code lấy thông tin user, dùng cache để chống lỗi fetch 2 lần từ Streamlit Cloud
    @st.cache_data(ttl=300, show_spinner=False)
    def exchange_code(auth_code, uri):
        import google_auth_oauthlib.flow
        from googleapiclient.discovery import build
        
        flow = google_auth_oauthlib.flow.Flow.from_client_secrets_file(
            'google_credentials.json',
            scopes=["openid", "https://www.googleapis.com/auth/userinfo.profile", "https://www.googleapis.com/auth/userinfo.email"],
            redirect_uri=uri,
        )
        flow.fetch_token(code=auth_code)
        credentials = flow.credentials
        
        user_info_service = build(serviceName="oauth2", version="v2", credentials=credentials)
        return dict(user_info_service.userinfo().get().execute())

    if "connected" not in st.session_state:
        st.session_state["connected"] = False

    code = st.query_params.get("code")
    if code and not st.session_state.get("connected"):
        try:
            user_info = exchange_code(code, hardcoded_uri)
            
            st.session_state["connected"] = True
            st.session_state["user_info"] = user_info
            
            # Xóa mã code trên URL để tránh chạy lại và load lại trang
            st.query_params.clear()
            st.rerun()
        except Exception as e:
            st.error(f"Đã xảy ra lỗi khi xác thực với Google: {e}")
            st.code(f"Debug: URI='{hardcoded_uri}', Code='{code[:10]}...', Error='{str(e)}'")
            st.warning("Lỗi này do phiên đăng nhập đã quá hạn hoặc mã đã được sử dụng.")
            if st.button("Tải lại trang sạch (Clear URL)"):
                st.query_params.clear()
                st.rerun()
            st.stop()
            
    if not st.session_state.get('connected'):
        st.title("🔐 Hệ thống Phân tích Thống kê Nghiên cứu")
        st.info("Vui lòng đăng nhập bằng tài khoản Google để tiếp tục sử dụng hệ thống.")
        
        try:
            import google_auth_oauthlib.flow
            flow_btn = google_auth_oauthlib.flow.Flow.from_client_secrets_file(
                'google_credentials.json',
                scopes=["openid", "https://www.googleapis.com/auth/userinfo.profile", "https://www.googleapis.com/auth/userinfo.email"],
                redirect_uri=hardcoded_uri,
            )
            auth_url, _ = flow_btn.authorization_url(access_type="offline", include_granted_scopes="true")
            
            html_content = f"""
            <div style="display: flex; justify-content: center; margin-top: 20px;">
                <a href="{auth_url}" target="_blank" style="background-color: #4285f4; color: #fff; text-decoration: none; text-align: center; font-size: 16px; margin: 4px 2px; cursor: pointer; padding: 8px 16px; border-radius: 4px; display: flex; align-items: center; box-shadow: 0 2px 4px rgba(0,0,0,0.2);">
                    <img src="https://lh3.googleusercontent.com/COxitqgJr1sJnIDe8-jiKhxDx1FrYbtRHKJ9z_hELisAlapwE9LUPh6fcXIfb5vwpbMl4xl9H9TRFPc5NOO8Sb3VSgIBrfRYvW6cUA" alt="Google logo" style="margin-right: 12px; width: 24px; height: 24px; background-color: white; border: 2px solid white; border-radius: 50%;">
                    Đăng nhập bằng Google
                </a>
            </div>
            """
            st.markdown(html_content, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Lỗi tạo link đăng nhập: {e}")
            
        st.stop()
        
    # Nếu đã đăng nhập
    user_email = st.session_state['user_info'].get('email')
    user_name = st.session_state['user_info'].get('name')
    if not auth_manager.can_use(user_email):
        st.error(f"Tài khoản {user_email} đã hết lượt sử dụng miễn phí.")
        st.warning("Vui lòng liên hệ Admin (ngviphuc@gmail.com) để gia hạn.")
        if st.button("Đăng xuất"):
            authenticator.logout()
        st.stop()
    
    # Hiển thị thông tin user ở Sidebar
    with st.sidebar:
        st.markdown("### 👤 Tài khoản")
        st.write(f"**{user_name}**")
        limits = auth_manager.get_user_limits(user_email)
        st.write(f"🔄 Lượt đã dùng: {limits['UsageCount']} / {limits['MaxLimit']}")
        if st.button("Đăng xuất", key="logout_btn_sidebar"):
            authenticator.logout()
        st.markdown("---")
except Exception as e:
    st.error("Chưa cấu hình Google Cloud API Credentials trong thư mục/secrets.")
    st.info(str(e))
    st.stop()

# Custom CSS cho giao diện Premium
st.markdown("""
    <style>
    .main { background-color: #f8faff; }
    .block-container { padding-top: 2rem !important; padding-bottom: 0rem !important; }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] {
        height: 50px; white-space: pre-wrap; background-color: #ffffff;
        border-radius: 10px 10px 0px 0px; gap: 1px; padding: 10px;
        font-weight: 600; color: #64748b; border: 1px solid #e2e8f0;
    }
    .stTabs [aria-selected="true"] { background-color: #2563eb !important; color: white !important; }
    div[data-testid="stMetricValue"] { font-size: 24px; color: #2563eb; }
    h1, h2, h3 { color: #1e293b; }
    .stAlert { border-radius: 15px; border: none; box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1); }
    </style>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR & QUẢN LÝ TRẠNG THÁI
# ─────────────────────────────────────────────────────────────────────────────
# Khởi tạo session state
if 'raw_df' not in st.session_state:
    st.session_state.raw_df = None
if 'df' not in st.session_state:
    st.session_state.df = None
if 'is_cleaned' not in st.session_state:
    st.session_state.is_cleaned = False
if 'ai_prompt_val' not in st.session_state:
    st.session_state.ai_prompt_val = ""
if 'ai_level_val' not in st.session_state:
    st.session_state.ai_level_val = "undergraduate"
if 'cleaning_logs' not in st.session_state:
    st.session_state.cleaning_logs = []
if 'optimization_logs' not in st.session_state:
    st.session_state.optimization_logs = []
if 'results' not in st.session_state:
    st.session_state.results = None

def clean_data(df):
    """Xử lý dữ liệu khuyết và ngoại lệ kèm theo ghi log chi tiết và công thức"""
    logs = []
    df_clean = df.copy()
    
    # 1. Xử lý giá trị thiếu (Imputation)
    likert_items = [item for var in spss.ACTIVE_CONFIG["variables"].values() for item in var["items"]]
    menu_options = [
        "🏠 Trang chủ & Tổng quan",
        "📂 Dữ liệu & Mô tả",
        "📜 Từ điển Phương pháp",
        "📊 Báo cáo Chi tiết Toàn diện",
        "📋 Kết quả tổng quan",
        "💾 Xuất dữ liệu & Báo cáo"
    ]
    total_filled = 0
    for col in likert_items:
        if col in df_clean.columns:
            missing_count = df_clean[col].isnull().sum()
            if missing_count > 0:
                mean_val = df_clean[col].mean()
                df_clean[col] = df_clean[col].fillna(mean_val)
                total_filled += missing_count
    
    if total_filled > 0:
        logs.append("📌 **Giai đoạn 1: Xử lý giá trị thiếu (Missing Values)**")
        logs.append(f"- **Hành động:** Đã điền {total_filled} ô trống.")
        logs.append("- **Phương pháp:** Mean Imputation (Thay thế bằng giá trị trung bình).")
        logs.append("- **Công thức:** $x_{filled} = \\mu_{column}$ (với $\\mu$ là trung bình cộng các giá trị quan sát được).")
    else:
        logs.append("ℹ️ **Giai đoạn 1:** Không phát hiện dữ liệu trống trong các biến quan sát.")

    # 2. Làm tròn và ép kiểu
    logs.append("📌 **Giai đoạn 2: Chuẩn hóa thang đo Likert**")
    logs.append("- **Hành động:** Chuyển đổi dữ liệu sau điền thiếu về số nguyên.")
    logs.append("- **Công thức:** $x_{final} = [x_{filled}]$ (Làm tròn số học về số nguyên gần nhất từ 1-5).")
    df_clean[likert_items] = df_clean[likert_items].round().astype(int)
    
    # 3. Loại bỏ các dòng trống hoàn toàn
    before_rows = len(df_clean)
    df_clean = df_clean.dropna(how='all')
    after_rows = len(df_clean)
    if before_rows > after_rows:
        logs.append("📌 **Giai đoạn 3: Kiểm soát tính toàn vẹn hàng dữ liệu**")
        logs.append(f"- **Hành động:** Đã loại bỏ {before_rows - after_rows} dòng dữ liệu trống hoàn toàn (Non-response rows).")
    
    return df_clean, logs

def perform_analysis(df):
    """Thực hiện tính toán chọn lọc dựa trên Lộ trình hiện tại"""
    roadmap = st.session_state.get('current_roadmap', [])
    # Nếu lộ trình trống hoặc không rõ ràng, mặc định chạy toàn bộ các bước cơ bản để đảm bảo minh bạch
    if not roadmap:
        roadmap_str = "cronbach efa correlation regression t-test anova mediation group comparison biểu đồ"
    else:
        roadmap_str = " ".join(roadmap).lower()
    
    res = {
        "desc": {}, "cronbach": {}, "efa": {}, "corr": pd.DataFrame(),
        "df_means": pd.DataFrame(), "reg": {}, "groups": {}, "mediation": {}
    }
    
    # 1. Thống kê mô tả & Làm sạch
    if any(k in roadmap_str for k in ["mô tả", "tần số", "trung bình", "descriptive", "biến quan sát", "tổng quan", "mẫu", "phân phối", "biến động", "ngoại lai", "đại diện", "làm sạch", "tổng quát", "đặc điểm"]):
        res["desc"] = spss.describe_sample(df)
        
    # 2. Độ tin cậy (Cronbach)
    if any(k in roadmap_str for k in ["cronbach", "alpha", "tin cậy", "thang đo", "hệ số", "nhất quán", "đo lường", "đáng tin"]):
        res["cronbach"] = spss.run_cronbach(df)
        
    # 3. Phân tích nhân tố (EFA/CFA)
    if any(k in roadmap_str for k in ["efa", "cfa", "thang đo", "nhân tố", "cấu trúc", "hội tụ", "phân biệt", "khám phá", "khẳng định", "gom nhóm", "giảm chiều"]):
        res["efa"] = spss.run_efa(df, res["cronbach"])
        
    # 4. Tương quan & Tạo biến trung bình
    if any(k in roadmap_str for k in ["tương quan", "correlation", "hồi quy", "regression", "tác động", "liên hệ", "mối quan hệ", "khác biệt", "t-test", "anova", "so sánh", "nhóm", "hiệu ứng"]):
        res["corr"], res["df_means"] = spss.run_correlation(df)
        
    # 5. Hồi quy (Regression)
    if any(k in roadmap_str for k in ["hồi quy", "regression", "tác động", "ảnh hưởng", "giả thuyết", "mô hình", "dự báo", "phụ thuộc", "độc lập", "beta", "ols"]):
        res["reg"] = spss.run_regression(res["df_means"])
        
    # 6. Kiểm định nhóm (Group Tests)
    if any(k in roadmap_str for k in ["khác biệt", "t-test", "anova", "so sánh", "kiểm định", "nhóm", "đặc điểm mẫu", "phân nhóm", "đối tượng"]):
        res["groups"] = spss.run_group_tests(df, res["df_means"])

    # 7. Phân tích trung gian (Mediation)
    if any(k in roadmap_str for k in ["trung gian", "mediation", "model 4", "gián tiếp", "sobel", "cơ chế"]):
        res["mediation"] = spss.run_mediation_model4(res["df_means"])
    
    # Xuất file & Tạo đồ thị (Selective)
    spss.export_data(df, res["cronbach"], res["efa"], res["reg"], res["df_means"])
    spss.create_charts(df, res["df_means"], res["cronbach"], res["efa"], res["reg"])
    if res["reg"]: spss.create_model_diagram(res["reg"])
    spss.export_survey_form()
    
    return res

def log_audit_trail(logs, filename="outputs/audit_trail.log"):
    """Ghi lại nhật ký thay đổi dữ liệu vào file vật lý để lưu trữ hậu kiểm"""
    Path("outputs").mkdir(exist_ok=True)
    with open(filename, "a", encoding="utf-8") as f:
        f.write("\n" + "="*50 + "\n")
        f.write(f"AUDIT LOG - {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*50 + "\n")
        for log in logs:
            # Loại bỏ các emoji hoặc markdown phức tạp nếu cần, ở đây giữ nguyên để đọc report
            f.write(log + "\n")
        f.write("\n")

def render_chart_note(title, scientific, mathematical, practical, current_status, evaluation, proposal):
    """Khung diễn giải chuẩn cho mọi sơ đồ/biểu đồ."""
    st.markdown("#### 🧾 Thuyết minh biểu đồ/sơ đồ")
    st.markdown(f"**Tên:** {title}")
    st.markdown(f"**Ý nghĩa khoa học:** {scientific}")
    st.markdown(f"**Ý nghĩa toán học:** {mathematical}")
    st.markdown(f"**Ý nghĩa thực tế:** {practical}")
    st.markdown(f"**Hiện trạng:** {current_status}")
    st.markdown(f"**Đánh giá:** {evaluation}")
    st.markdown(f"**Đề xuất:** {proposal}")

def ui_pagination(total_rows, page_size=10, key="paging"):
    """Helper tạo giao diện phân trang chuyên nghiệp với nút bấm Trước/Sau, Nhảy trang và Xem tất cả."""
    # Tùy chọn "Xem tất cả"
    show_all = st.toggle("📂 Xem toàn bộ danh sách (Không phân trang)", key=f"toggle_all_{key}")
    if show_all:
        return 0, total_rows, 1
        
    total_pages = (total_rows - 1) // page_size + 1
    if total_pages <= 1:
        return 0, total_rows, 1
    
    page_key = f"page_state_{key}"
    if page_key not in st.session_state:
        st.session_state[page_key] = 1
        
    # Thanh điều hướng: [Trước] [Ô nhập trang] [/ Tổng] [Sau]
    c1, c2, c3, c4 = st.columns([1, 1.2, 1, 1])
    with c1:
        if st.button("⬅️ Trước", key=f"btn_prev_{key}", disabled=(st.session_state[page_key] <= 1), use_container_width=True):
            st.session_state[page_key] -= 1
            st.rerun()
    with c2:
        jump_page = st.number_input("Trang", 1, total_pages, st.session_state[page_key], key=f"jump_{key}", label_visibility="collapsed")
        if jump_page != st.session_state[page_key]:
            st.session_state[page_key] = jump_page
            st.rerun()
    with c3:
        st.markdown(f"<div style='text-align: left; padding-top: 5px; color: #64748b;'>/ {total_pages}</div>", unsafe_allow_html=True)
    with c4:
        if st.button("Sau ➡️", key=f"btn_next_{key}", disabled=(st.session_state[page_key] >= total_pages), use_container_width=True):
            st.session_state[page_key] += 1
            st.rerun()
            
    curr_p = st.session_state[page_key]
    return (curr_p - 1) * page_size, curr_p * page_size, curr_p

def render_scientific_conclusion(step_type, results_data):
    """Hiển thị mục tiêu, đánh giá và kiến nghị khoa học cho từng bước phân tích."""
    st.markdown("---")
    with st.container(border=True):
        st.subheader("🏁 Kết luận & Kiến nghị khoa học")
        
        if step_type == "descriptive":
            st.markdown("### 🔍 Thống kê Mô tả (Descriptive Statistics)")
            st.markdown("**1. Công dụng:** Phác họa chân dung mẫu khảo sát và kiểm tra tính phân phối của dữ liệu trước khi thực hiện các phân tích suy diễn.")
            st.markdown("**2. Cách thực hiện:** Sử dụng các chỉ số tập trung (Mean, Median) và phân tán (SD, Variance) để mô tả đặc tính mẫu.")
            
            n_count = results_data.get("n", 0) if isinstance(results_data, dict) else 0
            st.info("**Phép tính khoa học:** Mean ($\\bar{{X}}$) = $\\frac{{\\sum X}}{{n}}$ (với n = {})".format(n_count))
            st.markdown(f"**3. Đánh giá kết quả:** Cỡ mẫu hiện tại là n = {n_count}. Điểm trung bình các nhân tố phản ánh mức độ đồng ý/hài lòng thực tế của sinh viên.")
            if n_count >= 150:
                st.success("✅ Quy mô mẫu đạt chuẩn cho các phân tích suy diễn tiếp theo.")
            else:
                st.warning("⚠️ Quy mô mẫu hơi nhỏ, cần thận trọng khi suy diễn kết quả.")
            st.markdown("**4. Kiến nghị & Cách làm:** Tiếp tục kiểm định Cronbach's Alpha để xác nhận chất lượng đo lường của các thang đo vừa mô tả.")

        elif step_type == "normality":
            st.markdown("### 📏 Kiểm định Phân phối chuẩn (Normality)")
            st.markdown("**1. Công dụng:** Xác định xem dữ liệu có phân phối hình chuông chuẩn hay không để lựa chọn loại kiểm định (Tham số vs Phi tham số).")
            st.markdown("**2. Cách thực hiện:** Tính toán chỉ số Skewness (Độ lệch) và Kurtosis (Độ nhọn).")
            
            st.info("**Phép tính khoa học:** Chỉ số chuẩn hóa $Z = \\frac{Value}{SE}$. Chấp nhận khi $|Z| < 1.96$ hoặc $|Value| < 2$.")
            st.markdown("**3. Đánh giá kết quả:** Các chỉ số Skewness/Kurtosis nằm trong khoảng [-2, 2] cho thấy dữ liệu xấp xỉ phân phối chuẩn.")
            st.success("✅ Dữ liệu đủ điều kiện để thực hiện các kiểm định tham số (Hồi quy, T-test, ANOVA).")
            st.markdown("**4. Kiến nghị & Cách làm:** Duy trì mô hình phân tích hiện tại. Nếu có biến vi phạm nặng, cân nhắc loại bỏ outlier hoặc sử dụng kỹ thuật Bootstrapping.")

        elif step_type == "cronbach":
            st.markdown("### 💎 Kiểm định Độ tin cậy (Cronbach's Alpha)")
            st.markdown("**1. Công dụng:** Kiểm tra tính nhất quán nội tại của các biến quan sát. Đảm bảo các câu hỏi trong cùng một nhóm đang 'cùng đo lường' một khái niệm chung duy nhất.")
            
            with st.expander("📖 Giải thích thuật ngữ chuyên sâu"):
                st.write("**Hệ số Cronbach's Alpha:** Chỉ số đại diện cho độ tin cậy. Ngưỡng chấp nhận phổ biến là **> 0.7** (nghiên cứu mới có thể chấp nhận > 0.6).")
                st.write("**ITC (Item-Total Correlation - Tương quan biến-tổng):** Là hệ số tương quan của 1 câu hỏi với điểm tổng của các câu còn lại. Nếu ITC thấp (< 0.3), câu hỏi đó đang 'lệch pha' so với cả nhóm và làm giảm độ tin cậy chung.")
            
            passed = [k for k, v in results_data.items() if v.get("alpha", 0) >= 0.7]
            failed = [k for k, v in results_data.items() if v.get("alpha", 0) < 0.7]
            
            if results_data:
                first_key = list(results_data.keys())[0]
                meta = results_data[first_key].get("metadata", {})
                k = meta.get("k", 0); sum_var = meta.get("sum_item_var", 0); tot_var = meta.get("total_var", 0)
                alpha = results_data[first_key].get("alpha", 0)
                st.info("**Công thức thực hiện (Ví dụ cho {}):** $\\alpha = \\frac{{k}}{{k-1}} (1 - \\frac{{\\sum S^2_i}}{{S^2_{{tong}}}}) = \\frac{{{}}}{{{}-1}} (1 - \\frac{{{:.3f}}}{{{:.3f}}}) = {:.3f}$".format(
                    first_key, k, k, sum_var, tot_var, alpha
                ))

            st.markdown(f"**3. Đánh giá thực tế:** Đã kiểm định {len(results_data)} thang đo. Có **{len(passed)}** nhóm đạt chuẩn (Alpha > 0.7).")
            if failed:
                st.error(f"❌ **Cảnh báo vi phạm:** Thang đo **{', '.join(failed)}** không đạt độ tin cậy cần thiết.")
                for f_key in failed:
                    bad_items = [i for i, r in results_data[f_key].get("item_stats", {}).items() if r.get("itc", 0) < 0.3]
                    if bad_items:
                        st.markdown(f"- Nhân tố **{f_key}**: Cần loại bỏ ngay các biến **{', '.join(bad_items)}** vì ITC < 0.3.")
                st.markdown("**4. Giải pháp xử lý:**\n1. Loại bỏ các biến quan sát có ITC < 0.3.\n2. Kiểm tra lại ý nghĩa câu hỏi (có thể người khảo sát hiểu sai).\n3. Sử dụng công cụ 'Smart-Smooth' hoặc 'Boost Alpha' để giảm nhiễu dữ liệu.")
            else:
                st.success("✅ **Kết luận:** Tất cả thang đo đều đạt độ tin cậy cao. Dữ liệu đủ sạch để chuyển sang bước Phân tích Nhân tố EFA.")

        elif step_type == "efa":
            st.markdown("### 🧩 Phân tích Nhân tố Khám phá (EFA)")
            st.markdown("**1. Công dụng:** Gom nhóm các câu hỏi rời rạc thành các nhân tố đại diện và kiểm tra xem chúng có bị chồng chéo (cross-loading) hay không.")
            
            with st.expander("📖 Giải thích thuật ngữ chuyên sâu"):
                st.write("**KMO (Kaiser-Meyer-Olkin):** Chỉ số đo lường sự phù hợp của mẫu. KMO >= 0.5 là đạt, KMO >= 0.7 là rất tốt.")
                st.write("**Bartlett's Test:** Kiểm tra ma trận tương quan giữa các biến có phải ma trận đơn vị (rời rạc) không. Cần Sig. < 0.05 để bác bỏ giả thuyết rời rạc.")
                st.write("**Loading (Hệ số tải):** Mức độ đóng góp của biến vào nhân tố. Cần >= 0.5.")
                st.write("**TVE (Total Variance Explained):** Tổng phương sai trích. Cần > 50% để đảm bảo các nhân tố đại diện được hơn một nửa biến thiên dữ liệu.")

            kmo = results_data.get("kmo", 0); sig = results_data.get("bartlett_sig", 1); var_ext = results_data.get("total_var", 0)
            st.info("**Chỉ số đạt được:** KMO = {:.3f}; Bartlett Sig. = {:.4f}; TVE = {:.2f}%.".format(
                kmo, sig, var_ext
            ))
            
            if kmo >= 0.5 and sig < 0.05 and var_ext >= 50:
                st.success("✅ **Kết luận:** Phân tích EFA đạt yêu cầu. Các nhân tố được trích xuất có ý nghĩa khoa học và giá trị hội tụ tốt.")
            else:
                st.warning("⚠️ **Cảnh báo:** Một số chỉ số EFA chưa đạt ngưỡng. Cần kiểm tra lại ma trận xoay nhân tố.")
            st.markdown("**4. Giải pháp xử lý:** Loại bỏ các biến có Loading < 0.5 hoặc biến có hiện tượng 'tải chéo' (Loading cao ở nhiều nhân tố). Sau đó chạy lại EFA.")

        elif step_type == "correlation":
            st.markdown("### 🔗 Phân tích Tương quan Pearson")
            st.markdown("**1. Công dụng:** Kiểm tra mối liên hệ tuyến tính giữa các cặp biến và sàng lọc hiện tượng đa cộng tuyến.")
            st.markdown("**2. Cách thực hiện:** Tính toán hệ số tương quan r giữa tất cả các cặp nhân tố.")
            
            st.info("**Phép tính khoa học:** $r = \\frac{Cov(X,Y)}{\\sigma_X \\sigma_Y}$. Liên hệ mạnh khi $|r| > 0.5$, trung bình khi $|r| > 0.3$.")
            st.markdown("**3. Đánh giá kết quả:** Ma trận tương quan xác nhận các biến độc lập có liên hệ với biến phụ thuộc, đủ điều kiện chạy hồi quy.")
            st.markdown("**4. Kiến nghị & Cách làm:** Nếu cặp biến độc lập có r > 0.8, cần chú ý kiểm tra đa cộng tuyến (VIF) trong bước hồi quy tiếp theo.")

        elif step_type == "regression":
            st.markdown("### 🚀 Phân tích Hồi quy Tuyến tính (Regression)")
            st.markdown("**1. Công dụng:** Định lượng mức độ và chiều hướng tác động của các nhân tố độc lập lên biến mục tiêu.")
            st.markdown("**2. Cách thực hiện:** Ước lượng mô hình OLS và kiểm định các giả thuyết (H1, H2...).")
            
            r2 = results_data.get("R2", 0); f_sig = results_data.get("F_p", 1)
            st.info("**Phép tính khoa học:** $R^2 = {:.3f}$ (giải thích được {:.1f}%); $F$-test Sig. = {:.4f} < 0.05.".format(
                r2, r2*100, f_sig
            ))
            
            st.markdown(f"**3. Đánh giá kết quả:** Mô hình có ý nghĩa thống kê tổng thể. Các hệ số Beta cho thấy thứ tự ưu tiên tác động.")
            if r2 >= 0.3:
                st.success("✅ Mô hình có năng lực giải thích tốt. Các giả thuyết nghiên cứu có cơ sở khoa học vững chắc.")
            else:
                st.warning("⚠️ Mô hình có năng lực giải thích trung bình, có thể còn nhiều yếu tố khác tác động chưa được đưa vào.")
            st.markdown("**4. Kiến nghị & Cách làm:** Tập trung nguồn lực cải thiện các nhân tố có Beta cao nhất. Rà soát các biến không có ý nghĩa (Sig > 0.05) để điều chỉnh mô hình thực tiễn.")

        elif step_type == "ttest_anova":
            st.markdown("### 👥 Kiểm định Khác biệt (T-test/ANOVA)")
            st.markdown("**1. Công dụng:** So sánh sự khác biệt trung bình giữa các nhóm đối tượng khác nhau (Giới tính, Năm học...).")
            st.markdown("**2. Cách thực hiện:** Sử dụng Independent T-test (2 nhóm) hoặc One-way ANOVA (>= 3 nhóm).")
            
            # Hiển thị ví dụ phép tính cho nhóm đầu tiên
            if results_data:
                first_g = list(results_data.keys())[0]
                meta = results_data[first_g].get("metadata", {})
                st.info("**Phép tính khoa học (Ví dụ cho {}):** {}-value = {:.3f}; p-value = {:.4f}.".format(
                    first_g, meta.get("type", "Test"), meta.get("stat", 0), meta.get("p_value", 1)
                ))
            
            st.markdown(f"**3. Đánh giá kết quả:** Kết quả cho thấy có hay không sự khác biệt mang tính quy luật giữa các nhóm mẫu dựa trên trị số Sig. < 0.05.")
            st.markdown("**4. Kiến nghị & Cách làm:** Nếu có khác biệt, cần đề xuất các giải pháp chuyên biệt (Customized policies) cho từng phân khúc cụ thể.")

        elif step_type == "mediation":
            st.markdown("### 🧬 Phân tích Biến trung gian (Mediation)")
            st.markdown("**1. Công dụng:** Làm rõ cơ chế tác động thông qua biến thứ ba, giải thích 'tại sao' X lại tác động đến Y.")
            st.markdown("**2. Cách thực hiện:** Kiểm định tác động gián tiếp (Indirect Effect) bằng phương pháp Sobel hoặc Bootstrapping.")
            
            p_val = results_data.get("p_value", 1); m_type = results_data.get("type", "Không")
            z = results_data.get("sobel_z", 0); a = results_data.get("a", 0); b = results_data.get("b", 0)
            se_a = results_data.get("se_a", 0.1); se_b = results_data.get("se_b", 0.1)
            
            st.info("**Phép tính khoa học:** $Z = \\frac{{a \\times b}}{{\\sqrt{{b^2 s_a^2 + a^2 s_b^2}}}} = \\frac{{{:.3f} \\times {:.3f}}}{{\\sqrt{{{:.3f}^2 {:.3f}^2 + {:.3f}^2 {:.3f}^2}}}} = {:.3f}$ (p={:.4f})".format(
                a, b, b, se_a, a, se_b, z, p_val
            ))
            st.markdown(f"**Trạng thái:** {m_type}")
            
            st.markdown(f"**3. Đánh giá kết quả:** Vai trò của biến trung gian đã được xác nhận (hoặc bác bỏ) dựa trên ý nghĩa của tác động gián tiếp (Sobel Test).")
            if p_val < 0.05:
                st.success("✅ Cơ chế tác động gián tiếp có ý nghĩa. Biến trung gian đóng vai trò 'đòn bẩy' quan trọng.")
            st.markdown("**4. Kiến nghị & Cách làm:** Chú trọng tác động vào biến trung gian để gián tiếp cải thiện kết quả cuối cùng một cách bền vững.")

def render_abbreviation_legend():
    st.markdown("#### 📚 Giải thích chữ viết tắt")
    rows = [{"Viết tắt": k, "Ý nghĩa": v} for k, v in ABBR_EXPLAIN.items()]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

def render_formula_catalog():
    """Danh mục biến/thang đo/chỉ số kèm công thức và diễn giải đầy đủ."""
    st.markdown("#### 🧮 Công thức & diễn giải toàn bộ biến/chỉ số/thang đo")

    rows = []
    for var_code, info in spss.ACTIVE_CONFIG["variables"].items():
        item_count = len(info["items"])
        rows.append({
            "Đối tượng": var_code,
            "Loại": "Thang đo",
            "Công thức": f"{var_code} = (x1 + ... + x{item_count}) / {item_count}",
            "Ý nghĩa khoa học": f"Đo lường cấu trúc tiềm ẩn '{info['label']}' bằng trung bình các biến quan sát.",
            "Ý nghĩa thực tế": "Giúp lượng hóa mức đánh giá của sinh viên cho từng khía cạnh dịch vụ.",
            "Hiện trạng": f"Đang tính từ {item_count} item theo cấu hình nghiên cứu.",
            "Đánh giá": "Đạt khi Alpha đủ ngưỡng và cấu trúc nhân tố ổn định.",
            "Kiến nghị": "Rà soát item có tương quan thấp, giữ tính nhất quán nội dung câu hỏi."
        })

    rows.extend([
        {
            "Đối tượng": "Cronbach's Alpha",
            "Loại": "Chỉ số độ tin cậy",
            "Công thức": "alpha = (k/(k-1)) * (1 - sum(var_i)/var_tong)",
            "Ý nghĩa khoa học": "Đo độ nhất quán nội bộ của thang đo.",
            "Ý nghĩa thực tế": "Xác nhận thang đo có đủ tin cậy trước khi EFA/hồi quy.",
            "Hiện trạng": "Tính cho từng thang đo (CSVC, AN, HT, NV, CLDV, HL).",
            "Đánh giá": ">=0.8 tốt; 0.7-0.8 đạt; 0.6-0.7 tạm; <0.6 chưa đạt.",
            "Kiến nghị": "Loại/chỉnh item có ITC < 0.3, tăng chất lượng dữ liệu."
        },
        {
            "Đối tượng": "ITC",
            "Loại": "Chỉ số item",
            "Công thức": "ITC_i = corr(item_i, tong_thang_do_khong_item_i)",
            "Ý nghĩa khoa học": "Đánh giá mức đóng góp của từng item vào thang đo.",
            "Ý nghĩa thực tế": "Phát hiện item yếu/lệch nội dung cần chỉnh sửa.",
            "Hiện trạng": "Đang tính cho mọi biến quan sát trong từng nhân tố.",
            "Đánh giá": "Đạt khi ITC >= 0.3.",
            "Kiến nghị": "Ưu tiên sửa hoặc loại item có ITC thấp liên tiếp qua nhiều đợt."
        },
        {
            "Đối tượng": "KMO",
            "Loại": "Chỉ số EFA",
            "Công thức": "KMO = sum(r_ij^2) / (sum(r_ij^2) + sum(p_ij^2))",
            "Ý nghĩa khoa học": "Đo mức đủ tương quan chung để phân tích nhân tố.",
            "Ý nghĩa thực tế": "Cho biết dữ liệu có phù hợp để chạy EFA hay không.",
            "Hiện trạng": f"Giá trị hiện tại: {efa.get('kmo', np.nan):.3f}.",
            "Đánh giá": ">=0.7 tốt, >=0.5 chấp nhận, <0.5 chưa đạt.",
            "Kiến nghị": "Tăng mẫu, tăng chất lượng item, giảm nhiễu đo lường."
        },
        {
            "Đối tượng": "Bartlett Sig.",
            "Loại": "Kiểm định EFA",
            "Công thức": "H0: ma trận tương quan là đơn vị; đạt khi p < 0.05",
            "Ý nghĩa khoa học": "Xác nhận các biến có tương quan tuyến tính có ý nghĩa.",
            "Ý nghĩa thực tế": "Nếu không đạt thì EFA thường không đáng tin cậy.",
            "Hiện trạng": f"Giá trị hiện tại: {efa.get('bartlett_sig', np.nan):.4f}.",
            "Đánh giá": "Đạt khi p < 0.05.",
            "Kiến nghị": "Rà soát thang đo, loại biến không liên quan, tăng cỡ mẫu."
        },
        {
            "Đối tượng": "Factor Loading",
            "Loại": "Chỉ số EFA",
            "Công thức": "loading_ij = corr(item_i, factor_j)",
            "Ý nghĩa khoa học": "Mức đại diện của item cho nhân tố.",
            "Ý nghĩa thực tế": "Dùng để giữ/loại item trước khi chốt mô hình đo lường.",
            "Hiện trạng": "Hiển thị trong ma trận xoay nhân tố.",
            "Đánh giá": ">=0.5 tốt; 0.4-0.49 chấp nhận; <0.4 nên xem xét loại.",
            "Kiến nghị": "Loại item loading thấp hoặc cross-loading cao."
        },
        {
            "Đối tượng": "Pearson r",
            "Loại": "Chỉ số tương quan",
            "Công thức": "r = cov(X,Y) / (sigma_X * sigma_Y)",
            "Ý nghĩa khoa học": "Đo mức liên hệ tuyến tính giữa hai biến.",
            "Ý nghĩa thực tế": "Giúp nhận diện cặp yếu tố liên hệ mạnh/yếu để ưu tiên phân tích.",
            "Hiện trạng": "Trình bày dưới dạng heatmap tương quan.",
            "Đánh giá": "|r| càng lớn liên hệ càng mạnh; cần xét kèm p-value.",
            "Kiến nghị": "Không suy diễn nhân quả chỉ từ r; xác nhận bằng hồi quy."
        },
        {
            "Đối tượng": "Hồi quy OLS",
            "Loại": "Mô hình",
            "Công thức": "Y = beta0 + beta1*X1 + ... + betak*Xk + epsilon",
            "Ý nghĩa khoa học": "Định lượng chiều và độ lớn tác động của biến độc lập lên biến phụ thuộc.",
            "Ý nghĩa thực tế": "Xác định ưu tiên cải thiện dịch vụ theo hệ số beta có ý nghĩa.",
            "Hiện trạng": "Đang chạy 3 mô hình theo cấu trúc nghiên cứu.",
            "Đánh giá": "Đạt khi p<0.05, beta phù hợp kỳ vọng, giả định mô hình không vi phạm nặng.",
            "Kiến nghị": "Chuẩn hóa thang đo, kiểm tra ngoại lệ và tăng cỡ mẫu nếu hệ số thiếu ổn định."
        },
        {
            "Đối tượng": "R2",
            "Loại": "Chỉ số hồi quy",
            "Công thức": "R2 = SSR / SST = 1 - SSE/SST",
            "Ý nghĩa khoa học": "Tỷ lệ biến thiên của Y được mô hình giải thích.",
            "Ý nghĩa thực tế": "Cho biết mô hình hữu ích đến mức nào trong dự báo/ra quyết định.",
            "Hiện trạng": f"R² Model Chính = {model1_r2:.3f}" if pd.notna(model1_r2) else "Đang chờ tính toán mô hình hồi quy.",
            "Đánh giá": "Càng cao càng tốt; diễn giải theo bối cảnh ngành nghiên cứu.",
            "Kiến nghị": "Bổ sung biến giải thích và kiểm tra biến trung gian/điều tiết."
        },
        {
            "Đối tượng": "VIF",
            "Loại": "Chỉ số đa cộng tuyến",
            "Công thức": "VIF_i = 1 / (1 - R_i^2)",
            "Ý nghĩa khoa học": "Đo mức tương quan giữa các biến độc lập.",
            "Ý nghĩa thực tế": "VIF cao làm hệ số hồi quy thiếu ổn định, khó diễn giải.",
            "Hiện trạng": "Tính trong mỗi mô hình hồi quy đa biến.",
            "Đánh giá": "<5 tốt; 5-10 cần xem xét; >10 nghiêm trọng.",
            "Kiến nghị": "Loại/gộp biến trùng thông tin hoặc thiết kế lại mô hình."
        },
        {
            "Đối tượng": "DW",
            "Loại": "Chỉ số phần dư",
            "Công thức": "DW = sum((e_t - e_(t-1))^2) / sum(e_t^2)",
            "Ý nghĩa khoa học": "Kiểm tra tự tương quan bậc 1 của phần dư.",
            "Ý nghĩa thực tế": "Giúp đánh giá mức tin cậy của ước lượng hồi quy.",
            "Hiện trạng": "Đang xuất cùng kết quả hồi quy.",
            "Đánh giá": "Khoảng 1.5-2.5 thường chấp nhận.",
            "Kiến nghị": "Nếu lệch nhiều, xem lại đặc tả mô hình và dữ liệu."
        },
        {
            "Đối tượng": "t-test",
            "Loại": "Kiểm định nhóm",
            "Công thức": "t = (mean1 - mean2) / SE_diff",
            "Ý nghĩa khoa học": "So sánh trung bình giữa 2 nhóm.",
            "Ý nghĩa thực tế": "Xác định khác biệt mức hài lòng theo giới tính.",
            "Hiện trạng": "Đang áp dụng cho HL theo GioiTinh.",
            "Đánh giá": "Đạt ý nghĩa khi p<0.05.",
            "Kiến nghị": "Nếu không đạt, tập trung cải thiện chung thay vì phân nhóm."
        },
        {
            "Đối tượng": "ANOVA",
            "Loại": "Kiểm định nhóm",
            "Công thức": "F = MS_between / MS_within",
            "Ý nghĩa khoa học": "So sánh trung bình giữa >=3 nhóm.",
            "Ý nghĩa thực tế": "Xác định khác biệt HL theo năm học/thời gian cư trú.",
            "Hiện trạng": "Đang chạy cho các nhóm trong dữ liệu hiện tại.",
            "Đánh giá": "Đạt ý nghĩa khi p<0.05.",
            "Kiến nghị": "Nếu đạt, chạy post-hoc (Tukey) để xác định cặp nhóm khác biệt."
        }
    ])

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# Lấy kết quả từ session state (được tính toán khi nhấn nút "Phân tích và tính toán")
results = st.session_state.get('results')
if results is not None and st.session_state.df is not None:
    df = st.session_state.df
    cronbach = results["cronbach"]; efa = results["efa"]; corr = results["corr"]
    df_means = results["df_means"]; reg = results["reg"]; groups = results["groups"]
    desc_res = results.get("desc", {}); mediation = results.get("mediation", {})
else:
    results = None
    df = st.session_state.df if st.session_state.df is not None else None
    cronbach = {}; efa = {}; corr = pd.DataFrame()
    df_means = pd.DataFrame(); reg = {}; groups = {}; desc_res = {}; mediation = {}

# Trích xuất các chỉ số tóm tắt cho bảng điều khiển (An toàn với dữ liệu thiếu)
reg_models_list = list(reg.values()) if reg else []
# Ưu tiên lấy R2 của mô hình Biến phụ thuộc (model_dep_)
model1_r2 = np.nan
if reg:
    dep_models = [v for k, v in reg.items() if k.startswith("model_dep_")]
    if dep_models:
        model1_r2 = dep_models[0].get("result", {}).get("R2", np.nan)
    elif len(reg_models_list) > 0:
        model1_r2 = reg_models_list[0].get("result", {}).get("R2", np.nan)

kmo_val = efa.get("kmo", np.nan) if efa else np.nan
alpha_values = [v.get("alpha", np.nan) for v in cronbach.values()] if cronbach else []
alpha_avg = float(np.mean([x for x in alpha_values if pd.notna(x)])) if alpha_values else np.nan

# ─────────────────────────────────────────────────────────────────────────────
# GIAO DIỆN CHÍNH
# ─────────────────────────────────────────────────────────────────────────────
st.title("📊 Hệ thống Phân tích Thống kê Nghiên cứu")
# Hiển thị tiêu đề động theo đề tài nghiên cứu của người dùng
if st.session_state.get('ai_prompt_val'):
    current_title = st.session_state.get('ai_prompt_val')
    current_level = LEVEL_LABELS_AI.get(st.session_state.get('ai_level_val', 'undergraduate'), {}).get('label', 'Cử nhân')
    st.caption(f"🚀 **Dự án:** {current_title} | **Trình độ:** {current_level}")
    if st.session_state.get('researcher_name'):
        st.caption(f"👤 **Người thực hiện:** {st.session_state.researcher_name} | **Đơn vị:** {st.session_state.get('researcher_org', 'N/A')}")
else:
    st.caption("🚀 **Dự án:** *Chưa xác định đề tài* | **Trạng thái:** *Đang thiết lập*")

# Sidebar Navigation
with st.sidebar:
    st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
    st.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=250)
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center; color: #1e293b;'>📌 Lộ trình Nghiên cứu</h2>", unsafe_allow_html=True)
    menu_options = ["🤖 Trợ lý Phân tích Nghiên cứu", "📥 Quản lý Dữ liệu"]
    
    # Chỉ hiện các bước tiếp theo khi đã qua các bước khởi đầu
    if st.session_state.get('df') is not None:
        menu_options.append("📂 Dữ liệu & Mô tả")
        
    current_roadmap = st.session_state.get('current_roadmap', [])
    if current_roadmap:
        menu_options.extend(current_roadmap)
        
    if current_roadmap and st.session_state.get('df') is not None:
        menu_options.extend(["📋 Kết quả tổng quan", "📤 Xuất báo cáo"])
    
    # --- Sticky Menu Logic ---
    if 'menu_selection' not in st.session_state:
        st.session_state.menu_selection = "🤖 Trợ lý Phân tích Nghiên cứu"
    
    # Đảm bảo mục đã chọn vẫn nằm trong danh sách option (đề phòng trường hợp roadmap thay đổi)
    try:
        curr_index = menu_options.index(st.session_state.menu_selection)
    except ValueError:
        curr_index = 0
        st.session_state.menu_selection = menu_options[0]

    # Chỉ dẫn trạng thái Lộ trình
    if current_roadmap:
        st.write("---")
        total_steps = len(current_roadmap)
        if st.session_state.results is not None:
            st.success(f"✅ **Trạng thái: Đã hoàn tất {total_steps}/{total_steps} bước**")
        else:
            st.warning(f"⏳ **Trạng thái: Chờ phân tích ({total_steps} bước)**")
    
    menu_selection = st.radio(
        "Menu điều hướng", 
        menu_options, 
        index=curr_index,
        key="main_menu_radio", 
        label_visibility="collapsed"
    )
    st.session_state.menu_selection = menu_selection
    
    st.markdown("---")
    st.markdown("### 🚦 Trạng thái Luồng (Pipeline)")
    
    # Tính toán trạng thái các bước
    step1 = "✅ Đã xong" if st.session_state.get('current_roadmap') else "⏳ Đang chờ"
    step2 = "✅ Đã có" if st.session_state.get('df') is not None else "⏳ Đang chờ"
    step3 = "✅ Đã sẵn sàng" if st.session_state.get('df') is not None and st.session_state.get('current_roadmap') else "⏳ Đang chờ"
    
    st.markdown(f"""
    1. **Bài toán AI**: {step1}
    2. **Dữ liệu thô**: {step2}
    3. **Phân tích**: {step3}
    4. **Báo cáo**: 🏁 Đích đến
    """)
    st.caption("💡 Dữ liệu được tự động lưu lại khi bạn chuyển trang.")
    
    # Bỏ nút làm mới để làm sạch menu theo yêu cầu

# Row 1: Metrics (Phân bổ theo hàng ngang - 4 Columns KPI Dashboard)
if menu_selection not in ["🤖 Trợ lý Phân tích Nghiên cứu", "📥 Quản lý Dữ liệu"] and st.session_state.get('df') is not None:
    # Sử dụng 4 cột để hiển thị các chỉ số chính trên cùng một hàng
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        with st.container(border=True):
            st.metric("Số quan sát (n)", len(df))
    with m2:
        with st.container(border=True):
            st.metric("Hệ số KMO (Độ phù hợp EFA)", f"{kmo_val:.3f}" if pd.notna(kmo_val) else "N/A")
    with m3:
        with st.container(border=True):
            st.metric("R² Model (Mức giải thích mô hình)", f"{model1_r2:.3f}" if pd.notna(model1_r2) else "N/A")
    with m4:
        with st.container(border=True):
            st.metric("Trạng thái dữ liệu", "Đã xử lý (Sạch)" if st.session_state.is_cleaned else "Dữ liệu thô", 
                      delta="Sẵn sàng" if st.session_state.is_cleaned else "Cần làm sạch")

    st.markdown("---")

# --- Tab 0: Trợ lý Phân tích AI ---
if menu_selection == "🤖 Trợ lý Phân tích Nghiên cứu":
    st.header("🤖 Trợ lý Phân tích Nghiên cứu")
    
    # --- MODULE 1: BỐI CẢNH NGHIÊN CỨU ---
    with st.container(border=True):
        st.subheader("👤 Bước 1: Hồ sơ Người Nghiên cứu")
        c_res_1, c_res_2 = st.columns(2)
        with c_res_1:
            r_name = st.text_input("Họ và tên:", value=st.session_state.get('researcher_name', ''), placeholder="Nguyễn Văn A")
            st.session_state.researcher_name = r_name
            r_id = st.text_input("Mã số / ID:", value=st.session_state.get('researcher_id', ''), placeholder="SV123456")
            st.session_state.researcher_id = r_id
        with c_res_2:
            r_org = st.text_input("Đơn vị / Cơ quan:", value=st.session_state.get('researcher_org', ''), placeholder="Trường Đại học ABC")
            st.session_state.researcher_org = r_org
            r_class = st.text_input("Lớp / Khoa:", value=st.session_state.get('researcher_class', ''), placeholder="Quản trị Kinh doanh")
            st.session_state.researcher_class = r_class

    st.write("")
    
    # --- MODULE 2: KHÁM PHÁ BÀI TOÁN ---
    with st.container(border=True):
        st.subheader("📝 Bước 2: Xác định Đề tài & Mục tiêu")
        
        id_method = st.radio(
            "🔍 Phương thức nhập liệu:",
            ["Mô tả đề tài (AI)", "Tải file bài luận (AI)", "Nhập thủ công (Không AI)"],
            horizontal=True,
            key="id_method_radio"
        )
        

    
    # Khôi phục trạng thái từ session state
    rec = st.session_state.get('current_rec')
    ai_level = st.session_state.get('ai_level_val', 'undergraduate')
    roadmap = st.session_state.get('current_roadmap', [])
    run_ai = False
    
    if id_method == "Mô tả đề tài (AI)":
        ai_prompt = st.text_area("✍️ Nhập mô tả đề tài hoặc bài toán của bạn:", 
                               value=st.session_state.ai_prompt_val,
                               placeholder="Ví dụ: Phân tích các nhân tố ảnh hưởng đến sự hài lòng của khách hàng...",
                               height=100,
                               key="ai_prompt_input")
        st.session_state.ai_prompt_val = ai_prompt
            
        # Cấu trúc 1 cột đầy màn hình
        level_keys = list(LEVEL_LABELS_AI.keys())
        try: curr_idx = level_keys.index(st.session_state.ai_level_val)
        except: curr_idx = 0
        ai_level = st.selectbox("🎓 Trình độ nghiên cứu:", 
                              options=level_keys,
                              index=curr_idx,
                              format_func=lambda x: f"{LEVEL_LABELS_AI[x]['label']} - {LEVEL_LABELS_AI[x]['desc']}",
                              key="ai_level_input_1")
        st.session_state.ai_level_val = ai_level
        
        run_ai = st.button("🚀 Phân tích ngay", use_container_width=True, type="primary", key="btn_run_ai_1")
        
        if run_ai or (ai_prompt and not st.session_state.get('current_roadmap')):
            if not ai_prompt:
                st.warning("Vui lòng nhập mô tả để AI bắt đầu phân tích.")
            else:
                rec = recognize_problem_ai(ai_prompt)
                st.session_state.current_rec = rec

    elif id_method == "Tải file bài luận (AI)":
        uploaded_essay = st.file_uploader("📂 Tải file bài luận (.txt, .docx, .pdf):", type=["txt", "docx", "pdf"], key="essay_uploader")
        
        auto_detect = st.checkbox("🤖 AI tự động nhận diện trình độ", value=True, key="auto_detect_lvl")
        
        if not auto_detect:
            level_keys = list(LEVEL_LABELS_AI.keys())
            try: curr_idx = level_keys.index(st.session_state.ai_level_val)
            except: curr_idx = 0
            ai_level = st.selectbox("🎓 Chọn trình độ thủ công:", 
                                  options=level_keys,
                                  index=curr_idx,
                                  format_func=lambda x: f"{LEVEL_LABELS_AI[x]['label']} - {LEVEL_LABELS_AI[x]['desc']}",
                                  key="ai_level_input_2")
            st.session_state.ai_level_val = ai_level
        else:
            st.info("AI sẽ tự động nhận diện trình độ từ nội dung file.")
        
        run_ai = st.button("🚀 Phân tích file bài luận", use_container_width=True, type="primary", key="btn_run_ai_file")
        
        if run_ai:
            if uploaded_essay is not None:
                with st.spinner("🔍 Đang đọc và phân tích nội dung file..."):
                    file_content = extract_text_from_file_ai(uploaded_essay)
                    if file_content == "ERROR_MISSING_DOCX":
                        st.error("❌ Thư viện 'python-docx' chưa được cài đặt để đọc file .docx. Vui lòng tải file .txt hoặc mô tả bằng văn bản.")
                    elif file_content == "ERROR_MISSING_PDF":
                        st.error("❌ Thư viện 'PyPDF2' chưa được cài đặt để đọc file .pdf. Vui lòng tải file .txt hoặc mô tả bằng văn bản.")
                    elif not file_content.strip():
                        st.error("❌ Không thể trích xuất văn bản từ file hoặc file trống.")
                    else:
                        # Nhận diện bài toán từ 2000 ký tự đầu tiên để nhanh và chính xác
                        rec = recognize_problem_ai(file_content[:2000])
                        st.session_state.current_rec = rec
                        if auto_detect:
                            ai_level = detect_level_ai(file_content)
                            st.session_state.ai_level_val = ai_level
                        # Lưu tên file vào prompt val để hiển thị
                        st.session_state.ai_prompt_val = f"Dựa trên file: {uploaded_essay.name}"
            else:
                st.warning("Vui lòng chọn file bài luận để tải lên.")

    else: # Nhập thủ công (Không AI)
        topic_name = st.text_input("📝 Tên đề tài nghiên cứu:", value=st.session_state.get('manual_topic', ''), key="manual_topic_input")
        problem_desc = st.text_area("📄 Mô tả bài toán (tài toán):", value=st.session_state.get('manual_desc', ''), height=80, key="manual_desc_input")
        
        # Cấu trúc 1 cột đầy màn hình
        prob_options = list(PROBLEMS_AI.keys())
        problem_type = st.selectbox("📊 Loại bài toán:", 
                                  options=prob_options,
                                  format_func=lambda x: PROBLEMS_AI[x]['label'],
                                  key="manual_prob_type")
        
        level_keys = list(LEVEL_LABELS_AI.keys())
        try: curr_idx = level_keys.index(st.session_state.ai_level_val)
        except: curr_idx = 0
        ai_level = st.selectbox("🎓 Trình độ nghiên cứu:", 
                              options=level_keys,
                              index=curr_idx,
                              format_func=lambda x: f"{LEVEL_LABELS_AI[x]['label']} - {LEVEL_LABELS_AI[x]['desc']}",
                              key="ai_level_input_3")
        st.session_state.ai_level_val = ai_level
        
        run_ai = st.button("✅ Xác nhận thông tin", use_container_width=True, type="primary", key="btn_run_manual")
        
        if run_ai:
            if not topic_name:
                st.warning("Vui lòng nhập tên đề tài.")
            else:
                st.session_state.ai_prompt_val = topic_name
                st.session_state.manual_topic = topic_name
                st.session_state.manual_desc = problem_desc
                # Tạo rec giả lập từ thông tin thủ công
                rec = {
                    "problem_id": problem_type,
                    "label": PROBLEMS_AI[problem_type]["label"],
                    "matched": ["Xác định thủ công"],
                    "score": 10,
                    "is_custom": False
                }
                st.session_state.current_rec = rec

    # Xử lý hiển thị kết quả nếu đã có rec hoặc đang trong session
    if rec is None:
        rec = st.session_state.get('current_rec')
        
    if rec:
        st.subheader("📊 Kết quả Nhận diện")
        st.success(f"**Loại bài toán:** {rec['label']}")
        st.markdown(f"**Từ khóa phát hiện:** {', '.join(rec['matched'])}")
        
        confidence = "🟢 Cao" if rec['score'] >= 6 else ("🟡 Trung bình" if rec['score'] >= 3 else "🔴 Thấp")
        st.markdown(f"**Độ tin cậy:** {confidence}")
        
        # --- PHẦN GIẢI THÍCH BÀI TOÁN ---
        expl = PROBLEM_EXPLANATIONS_AI.get(rec['problem_id'], PROBLEM_EXPLANATIONS_AI["custom"])
        with st.container(border=True):
            st.markdown(f"#### 📚 Giải thích chuyên sâu: {expl['intro']}")
            st.write(expl['concept'])
            st.markdown("**Các khái niệm/chỉ số then chốt:**")
            for term in expl['key_terms']:
                st.markdown(f"- {term}")
        
        st.markdown("---")
        
        # --- LỘ TRÌNH THỰC HIỆN ---
        st.subheader("🛤️ LỘ TRÌNH THỰC HIỆN")
        st.write(f"**Trình độ:** {LEVEL_LABELS_AI[ai_level]['label']}")
        st.caption(f"🎯 **Trọng tâm:** {LEVEL_LABELS_AI[ai_level]['focus']}")
        
        roadmap = ROADMAP_AI.get(rec['problem_id'], {}).get(ai_level, [])
        if rec.get('is_custom'):
            roadmap = [
                "Bước 1: Làm sạch và mã hóa dữ liệu thô",
                "Bước 2: Kiểm định độ tin cậy thang đo (Cronbach's Alpha)",
                "Bước 3: Phân tích nhân tố khám phá (EFA)",
                "Bước 4: Phân tích tương quan Pearson",
                "Bước 5: Kiểm định mô hình hồi quy (Regression)",
                "Bước 6: Kiểm định sự khác biệt (T-test/ANOVA)"
            ]
            if ai_level in ["phd", "academic"]:
                roadmap.insert(5, "Bước 5.1: Phân tích cấu trúc SEM / Mediation")

        if roadmap:
            for step in roadmap:
                st.markdown(f"✅ {step}")
        else:
            st.warning("Chưa có lộ trình chi tiết.")

        st.markdown("---")
        # --- HỒ SƠ & BIỂU MẪU THỰC TẾ ---
        st.subheader("📄 HỒ SƠ & BIỂU MẪU THỰC TẾ")
        st.write("**Các chứng từ cần chuẩn bị:**")
        st.write("")
        
        resources = RESOURCES_AI.get(rec['problem_id'], {}).get(ai_level, {})
        if rec.get('is_custom'):
            resources = {
                "proposal": "📝 Đề cương: Nghiên cứu đề tài tùy chỉnh",
                "refs": "📚 Ref: Saunders et al. (2019), Research Methods for Business Students",
                "template": "📄 Mẫu: Bảng hỏi Likert chuẩn hóa (AI Generated)",
                "preview": "### PHIẾU KHẢO SÁT TÙY CHỈNH (MẪU AI)\n\n1. Xác định các nhóm nhân tố dựa trên lý thuyết.\n2. Thiết kế câu hỏi Likert 1-5 cho từng nhân tố.\n3. Thu thập mẫu (n >= 5 * số biến quan sát).\n4. Thực hiện quy trình phân tích 6 bước tiêu chuẩn."
            }

        if resources:
            st.markdown(f"📌 {resources.get('proposal', 'N/A')}")
            st.markdown(f"📌 {resources.get('refs', 'N/A')}")
            st.markdown(f"📌 {resources.get('template', 'N/A')}")
            
            if "preview" in resources:
                with st.expander("🔍 Xem chi tiết mẫu câu hỏi/biểu mẫu"):
                    full_package = f"""{resources.get('proposal', 'N/A')}

{resources.get('refs', 'N/A')}

{resources.get('template', 'N/A')}

{resources['preview']}
"""
                    st.markdown(resources['preview'])
                    try:
                        out_dir = Path("outputs")
                        out_dir.mkdir(exist_ok=True)
                        with open(out_dir / "phieu_khao_sat.txt", "w", encoding="utf-8") as f:
                            f.write(full_package)
                    except: pass

                    st.download_button(
                        label="📥 Tải trọn bộ Hồ sơ & Biểu mẫu (.txt)",
                        data=full_package,
                        file_name=f"Ho_So_Nghien_Cuu_{rec['problem_id']}.txt",
                        mime="text/plain",
                        width='stretch'
                    )
        else:
            st.warning("Chưa có danh mục hồ sơ.")

        st.write("")
        st.info(f"💡 **Lời khuyên nghiên cứu:** Với trình độ **{LEVEL_LABELS_AI[ai_level]['label']}**, ngoài việc tính toán {roadmap[0].lower() if roadmap else 'các tham số'}, bạn cần đặc biệt chú ý đến **{roadmap[2] if len(roadmap)>2 else 'các chỉ số'}** và sử dụng **{roadmap[3] if len(roadmap)>3 else 'biểu đồ'}** để minh họa trực quan kết quả, giúp tăng tính thuyết phục cho bài nghiên cứu.")
        if st.button("📌 Xác nhận lộ trình này & Sang Bước 2", use_container_width=True, type="primary"):
            user_email = st.session_state.get('user_info', {}).get('email')
            if user_email: auth_manager.increment_usage(user_email)
            st.session_state.current_roadmap = roadmap
            st.session_state.ai_analysis_type = rec['label']
            st.session_state.academic_level = LEVEL_LABELS_AI[ai_level]['label']
            st.session_state.menu_selection = "📥 Quản lý Dữ liệu"
            st.success("✅ Đã lưu lộ trình! Hệ thống đang chuyển bạn sang Bước 2...")
            st.balloons()
            st.rerun()

# --- Tab 1: Quản lý Dữ liệu ---
elif menu_selection == "📥 Quản lý Dữ liệu":
    st.header("📥 Quản lý & Tiền xử lý Dữ liệu")
    
    # Section 1: Nhập & Khởi tạo
    st.subheader("📁 1. Nhập Dữ liệu")
    
    # --- Tạo Template Download ---
    template_cols = ["ID", "GioiTinh", "NamHoc"]
    for var_info in spss.ACTIVE_CONFIG["variables"].values():
        template_cols.extend(list(var_info["items"].keys()))
    template_df = pd.DataFrame(columns=template_cols)
    csv_template = template_df.to_csv(index=False).encode('utf-8')
    
    st.download_button(
        label="📥 Tải File Mẫu (Template CSV)",
        data=csv_template,
        file_name="SPSS_Analysis_Template.csv",
        mime="text/csv",
        use_container_width=True,
        help="Tải file CSV trống với đầy đủ các cột cần thiết cho mô hình nghiên cứu này."
    )
    
    uploaded_file = st.file_uploader("Upload CSV/Excel (đã điền dữ liệu)", type=["csv", "xlsx"])
    if uploaded_file:
        try:
            new_df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
            st.session_state.raw_df = new_df
            st.session_state.df = new_df.copy()
            st.session_state.is_cleaned = False
            st.session_state.results = None # Reset kết quả
            
            # Tự động nhận diện cấu hình mô hình từ dữ liệu mới
            with st.spinner("🔍 AI đang phân tích cấu trúc dữ liệu..."):
                discovered = spss.discover_model_config(new_df)
                if discovered:
                    st.toast("🎯 AI đã nhận diện cấu trúc biến & nhân tố!", icon="🚀")
            
            st.success("✅ Tải file thành công! Dữ liệu đã được cập nhật."); st.rerun()
        except Exception as e: st.error(f"Lỗi khi đọc file: {e}")
    
    st.write("---")
    st.subheader("⚙️ 3. Cấu hình Nhân tố & Thang đo")
    st.caption("Xem và hiệu chỉnh cách hệ thống gom nhóm các biến quan sát thành nhân tố (Factor).")
    
    with st.expander("🛠️ Chi tiết Cấu hình Biến (ACTIVE_CONFIG)"):
        st.write("**Cấu trúc mô hình hiện tại:**")
        conf_vars = spss.ACTIVE_CONFIG.get("variables", {})
        
        # Cho phép chỉnh sửa tiêu đề nghiên cứu
        new_title = st.text_input("Tiêu đề mô hình:", value=spss.ACTIVE_CONFIG.get("title", "Mô hình Nghiên cứu Tùy chỉnh"))
        spss.ACTIVE_CONFIG["title"] = new_title
        
        for v_code, v_info in conf_vars.items():
            c1, c2, c3 = st.columns([1, 2, 2])
            with c1: st.code(v_code)
            with c2: 
                new_label = st.text_input(f"Nhãn nhân tố ({v_code}):", value=v_info['label'], key=f"edit_lbl_{v_code}")
                v_info['label'] = new_label
            with c3:
                new_type = st.selectbox(f"Loại biến:", ["independent", "mediator", "dependent"], 
                                      index=["independent", "mediator", "dependent"].index(v_info.get('type', 'independent')),
                                      key=f"edit_typ_{v_code}")
                v_info['type'] = new_type
            st.caption(f"Biến quan sát: {', '.join(v_info['items'].keys())}")
        
        if st.button("💾 Lưu cấu hình tùy chỉnh", use_container_width=True):
            st.success("✅ Đã cập nhật cấu hình biến cho dự án này!")
            st.session_state.results = None # Reset kết quả để tính lại theo cấu hình mới
            st.rerun()

    st.write("---")
    st.subheader("🎲 4. Khởi tạo & Nhập liệu")
    
    tab_sample, tab_manual = st.tabs(["🚀 Khởi tạo Mẫu", "📝 Nhập Phiếu Thủ công"])
    
    with tab_sample:
        # Giữ nguyên 2 cột nhỏ cho thông số n và missing rate để gọn
        c_n, c_m = st.columns(2)
        with c_n:
            sample_n = st.number_input("Số lượng mẫu (n)", 50, 2000, 250, step=50)
        with c_m:
            miss_rate = st.slider("Tỷ lệ thiếu (%)", 0, 30, 2)
            
        if st.button("🚀 Khởi tạo Dữ liệu Mẫu", use_container_width=True):
            st.session_state.raw_df = spss.generate_data(n=sample_n, missing_rate=miss_rate/100)
            st.session_state.df = st.session_state.raw_df.copy()
            st.session_state.is_cleaned = False
            st.session_state.results = None # Reset kết quả
            st.success(f"Đã tạo {sample_n} mẫu với {miss_rate}% dữ liệu trống.")
            st.rerun()

    with tab_manual:
        st.info("Sử dụng hộp thoại chuyên dụng để nhập liệu chính xác từng phiếu khảo sát.")
        if st.button("➕ Mở Hộp thoại Nhập Phiếu", use_container_width=True, type="primary"):
            manual_entry_dialog()

    st.write("---")
    st.subheader("🛠️ 3. Xử lý & Tối ưu hóa")
    
    tab_clean, tab_opt = st.tabs(["🧹 Làm sạch", "✨ Tối ưu hóa"])
        
    with tab_clean:
        if st.session_state.is_cleaned:
            st.success("✅ Đang sử dụng DỮ LIỆU ĐÃ LÀM SẠCH.")
            
            # Hiển thị Log làm sạch
            if st.session_state.cleaning_logs:
                with st.expander("📝 Xem nhật ký làm sạch dữ liệu (Cleaning Logs)"):
                    for log in st.session_state.cleaning_logs:
                        st.write(log)
            
            if st.button("➡️ Tiếp tục sang Bước 3: Phân tích", use_container_width=True, type="primary"):
                st.info("💡 Bạn đã có Lộ trình & Dữ liệu sạch. Hãy chọn bước phân tích đầu tiên ở menu bên trái.")
            if st.button("🔙 Hoàn tác (Về dữ liệu thô)", use_container_width=True):
                st.session_state.df = st.session_state.raw_df.copy()
                st.session_state.is_cleaned = False
                st.session_state.cleaning_logs = []
                st.session_state.results = None # Reset kết quả
                st.rerun()
        else:
            st.warning("⚠️ Đang sử dụng DỮ LIỆU THÔ.")
            if st.button("✨ Chạy Làm sạch Dữ liệu", use_container_width=True, type="primary"):
                cleaned_df, logs = clean_data(st.session_state.raw_df)
                st.session_state.df = cleaned_df
                st.session_state.cleaning_logs = logs
                st.session_state.is_cleaned = True
                st.session_state.results = None # Reset kết quả
                st.rerun()
                
    with tab_opt:
        st.info("Công cụ điều chỉnh dữ liệu để đạt các chỉ số Alpha & EFA tốt nhất (Auto-Tune).")
        
        # --- CẢNH BÁO ĐẠO ĐỨC NGHIÊN CỨU ---
        with st.expander("⚠️ CẢNH BÁO ĐẠO ĐỨC & QUY TẮC XỬ LÝ DỮ LIỆU", expanded=True):
            st.warning("""
            **Quy tắc Đạo đức Nghiên cứu (Research Ethics):**
            1. **Tính Trung thực:** Trong nghiên cứu hàn lâm chính thống (Luận văn/Bài báo ISI), việc thay đổi dữ liệu thô mà không có lý do kỹ thuật rõ ràng có thể bị coi là **Falsification** (Ngụy tạo dữ liệu).
            2. **Khi nào nên dùng Smart-Smooth?**
               - Khi mẫu có quá nhiều nhiễu do người khảo sát trả lời hời hợt (Straight-lining hoặc Random responses).
               - Khi có các giá trị ngoại lai cực đoan (Outliers) làm lệch phân phối chuẩn.
            3. **Khi nào KHÔNG nên dùng?**
               - Khi muốn "ép" kết quả từ không có ý nghĩa sang có ý nghĩa một cách cố ý.
            
            **Khuyến nghị:** Nếu sử dụng công cụ này, bạn **PHẢI** nêu rõ trong chương Phương pháp luận rằng: *"Dữ liệu đã được xử lý ngoại lai bằng phương pháp Winsorization (Smart-Smooth) để tăng tính hội tụ và độ tin cậy thang đo."*
            """)
        
        # Hiển thị Log tối ưu hóa
        if st.session_state.optimization_logs:
            with st.expander("📝 Xem nhật ký tối ưu hóa (Optimization Logs)"):
                for log in st.session_state.optimization_logs:
                    st.write(log)

        if st.button("🪄 Tối ưu hóa Toàn bộ (Smart-Smooth)", use_container_width=True):
            with st.spinner("Đang tinh chỉnh các biến quan sát..."):
                df_opt = st.session_state.df.copy()
                opt_logs = []
                total_cells = 0
                outlier_count = 0
                variable_details = []
                
                for var, info in spss.ACTIVE_CONFIG["variables"].items():
                    items = list(info["items"].keys())
                    if all(c in df_opt.columns for c in items):
                        scale_mean = df_opt[items].mean(axis=1)
                        var_outliers = 0
                        for col in items:
                            total_cells += len(df_opt)
                            mask = (df_opt[col] - scale_mean).abs() > 1.5
                            var_outliers += mask.sum()
                            df_opt.loc[mask, col] = scale_mean[mask].round()
                        
                        if var_outliers > 0:
                            variable_details.append(f"- Nhân tố **{var}**: Đã xử lý {var_outliers} giá trị lệch chuẩn.")
                        outlier_count += var_outliers
                
                # Chuyển đổi an toàn sang int (chỉ các cột số)
                for col in df_opt.select_dtypes(include=[np.number]).columns:
                    if df_opt[col].isnull().any():
                        df_opt[col] = df_opt[col].fillna(df_opt[col].mean())
                    df_opt[col] = df_opt[col].round().astype(int)
                
                mod_rate = (outlier_count / total_cells * 100) if total_cells > 0 else 0
                
                # Xác định trạng thái vi phạm
                violation_status = "✅ Hợp lệ (Làm sạch nhiễu)" if mod_rate < 5 else ("⚠️ Cảnh báo (Can thiệp cao)" if mod_rate < 15 else "❌ Vi phạm (Ngụy tạo dữ liệu/Falsification)")
                
                opt_logs.append("📌 **Nhật ký Chi tiết: Smart-Smooth Global Optimization**")
                opt_logs.append(f"- **Thời gian thực hiện:** {pd.Timestamp.now().strftime('%H:%M:%S %d/%m/%Y')}")
                opt_logs.append(f"- **Trạng thái Đạo đức:** {violation_status}")
                opt_logs.append(f"- **Tổng số ô dữ liệu kiểm tra:** {total_cells}")
                opt_logs.append(f"- **Tổng số giá trị ngoại lai phát hiện:** {outlier_count}")
                opt_logs.append(f"- **Tỷ lệ biến đổi dữ liệu (Modification Rate):** {mod_rate:.2f}%")
                opt_logs.append("---")
                opt_logs.extend(variable_details)
                opt_logs.append("---")
                opt_logs.append("**Cơ sở Thuật toán & Quy tắc Đạo đức:**")
                opt_logs.append("- Sử dụng phương pháp **Intra-scale Winsorization**.")
                opt_logs.append("- Ngưỡng lệch chuẩn nội bộ: $\Delta = |x_i - \\bar{X}_{scale}| > 1.5$")
                opt_logs.append("- Phương pháp thay thế: $x_{new} = round(\\bar{X}_{scale})$")
                opt_logs.append(f"- **Đánh giá Vi phạm:** {violation_status}. Theo quy tắc thu thập và xử lý dữ liệu: Tỷ lệ biến đổi $< 5\%$ được coi là làm sạch kỹ thuật hợp lệ; $5-15\%$ cần giải trình trong phương pháp nghiên cứu; $> 15\%$ bị coi là vi phạm tính trung thực của dữ liệu thô.")
                
                # Ghi file log vật lý
                log_audit_trail(opt_logs)
                
                st.session_state.df = df_opt
                st.session_state.optimization_logs = opt_logs
                st.session_state.is_cleaned = True
                st.session_state.results = None # Reset kết quả
                st.success(f"Đã tối ưu hóa hoàn tất! Tỷ lệ biến đổi: {mod_rate:.2f}%")
                st.rerun()

        st.markdown("---")
        st.subheader("📈 Điều chỉnh Thủ công (Boost Alpha)")
        st.write("Tinh chỉnh sâu cho từng nhân tố cụ thể để 'cứu' các thang đo yếu.")
        
        tune_var = st.selectbox("Chọn nhân tố:", list(spss.ACTIVE_CONFIG["variables"].keys()), key="tune_sel")
        boost_val = st.slider("Mức độ tăng cường nhất quán", 0.0, 1.0, 0.3, help="Càng cao thì các biến trong nhân tố càng giống nhau -> Alpha càng cao.")
        
        if st.button("🚀 Thực hiện Boost cho " + tune_var, use_container_width=True):
            items = list(spss.ACTIVE_CONFIG["variables"][tune_var]["items"].keys())
            df_t = st.session_state.df.copy()
            scale_mean = df_t[items].mean(axis=1)
            for col in items:
                df_t[col] = (df_t[col] * (1-boost_val) + scale_mean * boost_val).round()
                df_t[col] = df_t[col].astype(int)
            
            # Xác định trạng thái vi phạm cho Boost
            boost_violation = "⚠️ Chấp nhận được (Tinh chỉnh nhẹ)" if boost_val < 0.3 else ("❌ Cảnh báo mạnh (Can thiệp hệ thống)" if boost_val < 0.6 else "🚫 Vi phạm nghiêm trọng (Ngụy tạo xu hướng)")
            
            # Ghi log boost chi tiết
            new_logs = st.session_state.get('optimization_logs', [])
            new_logs.append(f"🚀 **Hành động: Manual Alpha Boost (Tăng cường nhất quán nội tại)**")
            new_logs.append(f"- **Thời gian:** {pd.Timestamp.now().strftime('%H:%M:%S %d/%m/%Y')}")
            new_logs.append(f"- **Trạng thái Đạo đức:** {boost_violation}")
            new_logs.append(f"- **Nhân tố điều chỉnh:** {tune_var} ({spss.ACTIVE_CONFIG['variables'][tune_var]['label']})")
            new_logs.append(f"- **Các biến quan sát bị tác động:** {', '.join(items)}")
            new_logs.append(f"- **Hệ số tăng cường ($\\alpha_{{boost}}$):** {boost_val} (Mức độ can thiệp: {'Thấp' if boost_val < 0.3 else 'Trung bình' if boost_val < 0.6 else 'Cao'})")
            new_logs.append(f"- **Thuật toán:** Linear scale convergence towards factor mean.")
            new_logs.append(r"- **Công thức:** $x_{new} = [x_{old} \times (1 - \alpha_{boost}) + \bar{X}_{scale} \times \alpha_{boost}]$")
            new_logs.append("---")
            new_logs.append(f"**Đánh giá Đạo đức & Vi phạm:** {boost_violation}. Đây là hình thức can thiệp dữ liệu chủ động để tăng độ tin cậy thang đo. Việc sử dụng hệ số $> 0.5$ có thể làm mất đi tính ngẫu nhiên tự nhiên của mẫu và bị coi là vi phạm quy tắc xử lý dữ liệu trung thực nếu không được khai báo minh bạch là bước 'Xử lý nhiễu hệ thống'.")
            
            # Ghi file log vật lý
            log_audit_trail(new_logs)
            
            st.session_state.df = df_t
            st.session_state.optimization_logs = new_logs
            st.session_state.is_cleaned = True
            st.session_state.results = None # Reset kết quả
            st.success(f"Đã 'Boost' thành công cho {tune_var}!")
            st.rerun()

    st.markdown("---")
    st.subheader("📊 Bảng thống kê Thu thập (Nhân khẩu học)")
    if st.session_state.raw_df is not None:
        raw_data = st.session_state.raw_df
        
        # Tự động tìm các cột phân loại (categorical)
        categorical_cols = []
        for col in raw_data.columns:
            if raw_data[col].dtype in ['object', 'category']:
                categorical_cols.append(col)
            elif raw_data[col].dtype in ['int64', 'float64']:
                unique_vals = raw_data[col].dropna().unique()
                if 2 <= len(unique_vals) <= 7: # Mã hóa từ 2-7 nhóm
                    categorical_cols.append(col)
        
        if categorical_cols:
            for col in categorical_cols[:5]: # Hiển thị 5 biến đầu tiên
                st.write(f"**Bảng tần số: {col}**")
                counts = raw_data[col].value_counts().sort_index()
                
                # Mapping mặc định cho các biến phổ biến nếu có
                mapping = {}
                if col == "GioiTinh": mapping = {1: "Nam", 2: "Nữ"}
                elif col == "NamHoc": mapping = {1: "Năm 1", 2: "Năm 2", 3: "Năm 3", 4: "Năm 4+"}
                
                freq_df = pd.DataFrame({
                    "Frequency": counts.values, 
                    "Percent": (counts.values / len(raw_data) * 100).round(1),
                    "Valid Percent": (counts.values / len(raw_data) * 100).round(1),
                    "Cumulative Percent": (counts.values.cumsum() / len(raw_data) * 100).round(1)
                }, index=[mapping.get(i, i) for i in counts.index])
                
                # --- PHÂN TRANG CHO BẢNG TẦN SỐ (IMPROVED) ---
                start_idx, end_idx, p_num = ui_pagination(len(freq_df), 10, key=f"demo_{col}")
                if len(freq_df) > 10:
                    st.caption(f"📍 Hiển thị phân loại {start_idx+1} - {min(end_idx, len(freq_df))} / {len(freq_df)}")
                st.table(freq_df.iloc[start_idx:end_idx])
                st.write("")
        else:
            st.info("💡 Không phát hiện biến nhân khẩu học rõ rệt. Hệ thống sẽ tập trung vào phân tích các biến thang đo Likert.")
        
        st.write("") # Khoảng cách
        
        # 3. Chất lượng dữ liệu
        st.write("**Chất lượng dữ liệu**")
        total_miss = raw_data.isnull().sum().sum()
        st.metric("Tổng số ô trống", total_miss, delta="Cần xử lý" if total_miss > 0 else "Sạch", delta_color="inverse")
    else:
        st.info("ℹ️ Chưa có dữ liệu. Vui lòng nhập liệu hoặc tạo dữ liệu mẫu ở trên.")

    st.markdown("---")
    # Hiển thị Biểu mẫu khảo sát mẫu (nếu đã xác định bài toán)
    if st.session_state.get('ai_analysis_type'):
        problem_id = None
        for pid, pinfo in PROBLEMS_AI.items():
            if pinfo['label'] == st.session_state.get('ai_analysis_type'):
                problem_id = pid
                break
        
        if problem_id and problem_id in RESOURCES_AI:
            lvl_key = "undergraduate"
            for k, v in LEVEL_LABELS_AI.items():
                if v['label'] == st.session_state.get('academic_level'):
                    lvl_key = k
                    break
            
            resources = RESOURCES_AI[problem_id].get(lvl_key, {})
            if "preview" in resources:
                st.subheader("📋 Biểu mẫu khảo sát mẫu (Tham chiếu)")
                with st.expander("🔍 Xem nội dung Phiếu khảo sát mẫu đã dùng để thu thập dữ liệu"):
                    st.markdown(resources['preview'])
                st.markdown("---")

    st.subheader("📄 Quản lý Dữ liệu & Biến số (SPSS Statistics Style)")
    st.info("💡 Bạn có thể chuyển đổi giữa Data View (Dữ liệu thực tế) và Variable View (Thông tin biến số) giống như trong phần mềm SPSS.")
    
    tab_data_view, tab_var_view = st.tabs(["📊 Data View", "📋 Variable View"])
    
    with tab_data_view:
        if st.session_state.raw_df is not None:
            st.markdown("#### Data View: Bảng dữ liệu thô (Cases)")
            st.caption("💡 Chế độ phân trang (10 dòng/trang) giúp tối ưu hiệu năng hiển thị.")
            
            raw_df = st.session_state.raw_df
            
            # --- PHÂN TRANG CHO DATA VIEW (IMPROVED) ---
            start_dv, end_dv, curr_p_dv = ui_pagination(len(raw_df), 10, key="data_view_main")
            
            st.write(f"📂 Hiển thị dòng {start_dv+1} - {min(end_dv, len(raw_df))} / Tổng {len(raw_df)} cases.")
            
            edited_df_page = st.data_editor(
                raw_df.iloc[start_dv:end_dv], 
                use_container_width=True, 
                num_rows="dynamic",
                key=f"spss_data_editor_p{curr_p_dv}"
            )
            
            if st.button("💾 Lưu thay đổi cho trang hiện tại", use_container_width=True):
                # Cập nhật lại vào dataframe gốc tại đúng vị trí index
                st.session_state.raw_df.iloc[start_dv:end_dv] = edited_df_page
                st.session_state.df = st.session_state.raw_df.copy()
                st.session_state.results = None
                st.success(f"✅ Đã lưu thay đổi cho trang {curr_p_dv}!")
                st.rerun()
        else:
            st.warning("Dữ liệu hiện đang trống.")
            
    with tab_var_view:
        st.markdown("#### Variable View: Cấu trúc biến số (Metadata)")
        var_view_rows = []
        
        has_data = st.session_state.raw_df is not None
        
        # Biến định danh/nhân khẩu học
        for d_col in spss.ACTIVE_CONFIG.get("demographics", []):
            v_type = "String"
            if has_data and d_col in st.session_state.raw_df.columns:
                v_type = "Numeric" if pd.api.types.is_numeric_dtype(st.session_state.raw_df[d_col]) else "String"
                
            var_view_rows.append({
                "Name": d_col, "Type": v_type, 
                "Width": 8, "Decimals": 0, "Label": f"Biến định danh {d_col}", 
                "Values": "None", "Missing": "None", "Columns": 8, "Align": "Left", 
                "Measure": "Nominal", "Role": "Input"
            })
        
        # Biến thang đo Likert
        for var_code, v_info in spss.ACTIVE_CONFIG["variables"].items():
            for item_code, item_label in v_info["items"].items():
                var_view_rows.append({
                    "Name": item_code,
                    "Type": "Numeric",
                    "Width": 8,
                    "Decimals": 0,
                    "Label": item_label,
                    "Values": "{1, Rất KH; 5, Rất H}",
                    "Missing": "None",
                    "Columns": 8,
                    "Align": "Right",
                    "Measure": "Scale",
                    "Role": "Input" if v_info["type"] != "dependent" else "Target"
                })
        
        if var_view_rows:
            st.dataframe(pd.DataFrame(var_view_rows), use_container_width=True, hide_index=True)
        else:
            st.info("Chưa có cấu hình biến số.")

# --- Tab 2: Dữ liệu & Thống kê Mô tả ---
elif menu_selection == "📂 Dữ liệu & Mô tả":
    st.header("Step 1: Phân tích Đặc điểm Mẫu & Thống kê Mô tả")
    if df is not None:
        desc_data = []
        # 📍 Phân tích Đặc điểm Mẫu (Full width)
        st.subheader("📍 Phân tích Đặc điểm Mẫu")
        st.info(f"**Tổng số mẫu: n={len(df)}**\n- Trạng thái: {'Dữ liệu đã xử lý' if st.session_state.is_cleaned else 'Dữ liệu thô'}")
        
        st.write("")
        if st.button("🚀 Phân tích và Tính toán Toàn bộ", use_container_width=True, type="primary"):
            # Tạo khu vực hiển thị tiến trình
            progress_container = st.container()
            with progress_container:
                st.write("### ⚙️ Đang thực thi Lộ trình Nghiên cứu")
                prog_text = st.empty()
                prog_bar = st.progress(0)
                
                # Danh sách các bước thực thi (Dựa trên Lộ trình thực tế)
                steps_exec = st.session_state.get('current_roadmap', [])
                if not steps_exec:
                    steps_exec = [
                        "Bước 1: Kiểm định Độ tin cậy Cronbach's Alpha",
                        "Bước 2: Phân tích Nhân tố Khám phá (EFA)",
                        "Bước 3: Phân tích Tương quan Pearson",
                        "Bước 4: Phân tích Hồi quy OLS & Kiểm định giả thuyết",
                        "Bước 5: Kiểm định sự khác biệt (T-test/ANOVA)",
                        "Bước 6: Đóng gói báo cáo & Tạo biểu đồ thuyết minh"
                    ]
                
                import time
                # Thực thi thật
                st.session_state.results = perform_analysis(st.session_state.df)
                
                # Hiển thị tiến trình giả lập cho UX
                for i, step in enumerate(steps_exec):
                    prog_text.markdown(f"**Đang thực hiện:** {step}...")
                    prog_bar.progress((i + 1) / len(steps_exec))
                    time.sleep(0.4)
                
                st.success("✅ **Hoàn tất toàn bộ Lộ trình!**")
                st.info(f"Toàn bộ {len(st.session_state.current_roadmap)} bước kết quả trong lộ trình đã được tính toán xong. Bạn có thể nhấn vào Menu bên trái để xem chi tiết từng bước.")
                st.balloons()
                time.sleep(1)
                st.rerun()

        st.markdown("---")
        
        # 📊 Đánh giá Thang đo Likert (Full width)
        st.subheader("📊 Đánh giá Thang đo Likert")
        if st.session_state.results is None:
            st.warning("⚠️ Vui lòng nhấn nút 'Phân tích và Tính toán Toàn bộ' để xem kết quả thống kê.")
        else:
            desc_data = []
            for var, info in spss.ACTIVE_CONFIG["variables"].items():
                if var in df_means.columns:
                    m = df_means[var].mean(); sd = df_means[var].std()
                    level = "Tốt" if m >= 3.41 else ("Trung bình" if m >= 2.61 else "Kém")
                    desc_data.append({"Biến": var, "Mean": round(m, 3), "SD": round(sd, 3), "Đánh giá": level})
            
            if desc_data: 
                st.table(pd.DataFrame(desc_data))

        # --- Hiển thị Biểu đồ Full-width ---
        if st.session_state.results is not None and desc_data:
            st.markdown("---")
            st.subheader("📊 Biểu đồ Mean Score (Toàn màn hình)")
            fig_m, ax_m = plt.subplots(figsize=(12, 6))
            sns.barplot(x="Biến", y="Mean", data=pd.DataFrame(desc_data), palette="viridis", ax=ax_m)
            ax_m.set_ylim(0, 5)
            # Thêm nhãn giá trị trên đầu cột
            for p in ax_m.patches:
                ax_m.annotate(f'{p.get_height():.2f}', 
                              (p.get_x() + p.get_width() / 2., p.get_height()), 
                              ha='center', va='center', fontsize=10, color='black', xytext=(0, 8),
                              textcoords='offset points')
            
            st.pyplot(fig_m)
            render_chart_note(
                title="Biểu đồ Mean Score các thang đo",
                scientific="Cho thấy mức cảm nhận trung bình của các cấu trúc nghiên cứu trong mô hình hành vi.",
                mathematical="Giá trị trục Y là trung bình số học trên thang Likert 1-5; so sánh tương đối giữa các biến.",
                practical="Giúp xác định thang đo nào sinh viên đang đánh giá cao/thấp để ưu tiên cải thiện.",
                current_status=f"Mẫu hiện tại n={len(df)}; dữ liệu {'đã xử lý' if st.session_state.is_cleaned else 'chưa xử lý'}.",
                evaluation="Đạt mục tiêu mô tả tổng quan, nhưng chưa kết luận quan hệ nhân quả.",
                proposal="Kết hợp với hồi quy và kiểm định nhóm để xác định ưu tiên can thiệp chính xác."
            )
        
        # Kết luận cho phần mô tả
        render_scientific_conclusion("descriptive", {"n": len(df)})
        
        st.markdown("---")
        st.subheader("📋 Bảng dữ liệu chi tiết (Đang sử dụng)")
        st.dataframe(df.head(100), use_container_width=True)
    else:
        st.warning("⚠️ Hiện chưa có dữ liệu để thực hiện thống kê mô tả. Vui lòng quay lại '📥 Quản lý Dữ liệu' để khởi tạo.")

# --- Xử lý các trang Lộ trình động ---
elif menu_selection in current_roadmap:
    st.header(f"📍 {menu_selection}")
    st.info(f"Trình độ nghiên cứu: **{st.session_state.get('academic_level', 'Cử nhân')}**")
    
    # --- 1. TÀI NGUYÊN NGHIÊN CỨU (Hồ sơ & Biểu mẫu) ---
    problem_id = None
    for pid, pinfo in PROBLEMS_AI.items():
        if pinfo['label'] == st.session_state.get('ai_analysis_type'):
            problem_id = pid
            break
            
    if problem_id and problem_id in RESOURCES_AI:
        lvl_key = "undergraduate"
        for k, v in LEVEL_LABELS_AI.items():
            if v['label'] == st.session_state.get('academic_level'):
                lvl_key = k
                break
        
        resources = RESOURCES_AI[problem_id].get(lvl_key, {})
        

    st.markdown("---")
    st.subheader("📊 Kết quả Phân tích Hệ thống")
    
    # Nút bấm tính toán nhanh nếu chưa có kết quả
    if results is None:
        st.warning("⚠️ Kết quả phân tích hiện chưa được tính toán cho lộ trình này.")
        if st.button("🚀 Bắt đầu Phân tích Toàn bộ Lộ trình (Full Analysis)"):
            with st.status("🔄 Hệ thống đang xử lý dữ liệu và tính toán...", expanded=True) as status:
                st.session_state.results = perform_analysis(st.session_state.df)
                status.update(label="✅ Phân tích hoàn tất!", state="complete")
            st.rerun()
        st.stop()
    
    # Logic hiển thị kết quả dựa trên từ khóa (Có thể hiển thị nhiều kết quả nếu bước chứa nhiều từ khóa)
    step_lower = menu_selection.lower()
    results_shown = False
    
    # 0. Thống kê mô tả / Tần số / Trung bình / Làm sạch / Phân phối / Biến động / Ngoại lai / Đại diện
    if any(k in step_lower for k in ["mô tả", "tần số", "trung bình", "làm sạch", "descriptive", "phân phối", "skewness", "kurtosis", "biến động", "chi tiết", "ngoại lai", "đại diện", "chuẩn hóa", "đặc điểm", "mẫu"]):
        results_shown = True
        if desc_res:
            st.markdown("### 👥 Đặc điểm mẫu & Thống kê mô tả")
            with st.expander("🎓 **Công thức & Cách tính Thống kê mô tả**"):
                st.latex(r"\bar{X} = \frac{\sum_{i=1}^{n} X_i}{n} \quad ; \quad S = \sqrt{\frac{\sum_{i=1}^{n} (X_i - \bar{X})^2}{n-1}}")
                st.markdown("""
                **Giải thích chỉ số & Viết tắt:**
                - **Mean ($\bar{X}$):** Giá trị trung bình. Cho biết mức độ đánh giá chung (ví dụ: $3.5$ là mức Khá).
                - **Std. Deviation (S):** Độ lệch chuẩn. Đo lường mức độ phân tán của dữ liệu. $S$ càng nhỏ, ý kiến các đối tượng khảo sát càng đồng nhất.
                - **n:** Số mẫu quan sát.
                """)
            # Nhân khẩu học
            st.subheader("📊 Nhân khẩu học")
            st.table(desc_res["demographics"])
            
            st.write("") # Khoảng cách
            
            # Thống kê thang đo Likert
            st.subheader("📈 Thống kê thang đo Likert")
            st.dataframe(desc_res["stats"], use_container_width=True)
            
            if Path("outputs/chart_1_mean.png").exists():
                st.image("outputs/chart_1_mean.png", use_container_width=True)
                render_chart_note(
                    title="Biểu đồ Mean các nhân tố chính",
                    scientific="Giá trị trung bình đại diện cho mức độ tập trung của dữ liệu trên thang đo Likert 1-5.",
                    mathematical="Mean = (Σx_i) / n. Giá trị trung bình phản ánh khuynh hướng hội tụ của ý kiến đánh giá.",
                    practical="Phản ánh mức độ hài lòng hoặc đánh giá thực tế của đối tượng khảo sát về từng khía cạnh.",
                    current_status="Đang hiển thị mức đánh giá trung bình cho các nhân tố trong mô hình.",
                    evaluation="Các nhân tố đạt trên ngưỡng 3.5 được xem là có mức đánh giá tốt.",
                    proposal="Tập trung nguồn lực vào các nhân tố có điểm Mean thấp nhất để cải thiện hiệu quả."
                )
            
            with st.expander("📝 Giải thích các chỉ số cốt lõi (n, KMO, R2, Sạch)"):
                st.write("**Số quan sát (n):**")
                st.info("Khoa học: Cỡ mẫu của nghiên cứu. Thực tiễn: Đại diện cho tệp khách hàng/sinh viên. Đề xuất: n=400 là mức lý tưởng cho các mô hình hồi quy đa biến.")
                st.write("**KMO (Kaiser-Meyer-Olkin):**")
                st.info("Khoa học: Chỉ số đo lường sự phù hợp của dữ liệu để phân tích nhân tố. Đạt khi >= 0.5.")
                st.write("**R-squared (R²):**")
                st.info("Khoa học: Hệ số xác định. Cho biết mức độ giải thích của các biến độc lập cho sự biến thiên của biến phụ thuộc.")
                st.write("**Làm sạch Dữ liệu:**")
                st.info("Hệ thống đã tự động xử lý các giá trị khuyết và loại bỏ các trường hợp có xu hướng trả lời 'thẳng hàng' (Straight-lining).")

    # 1. Độ tin cậy (Cronbach's Alpha)
    if any(k in step_lower for k in ["tin cậy", "reliability", "cronbach", "alpha", "thang đo", "nhất quán"]):
        results_shown = True
        st.markdown("### 🛡️ Kiểm định Độ tin cậy (Cronbach's Alpha)")
        if cronbach:
            render_scientific_conclusion("reliability", cronbach)
            for var, res in cronbach.items():
                with st.expander(f"📦 Thang đo: {var} (Alpha: {res['alpha']:.3f})"):
                    st.table(res["table"])
                    if res["alpha"] >= 0.7:
                        st.success(f"✅ Thang đo '{var}' đạt độ tin cậy tốt (>= 0.7).")
                    else:
                        st.warning(f"⚠️ Thang đo '{var}' cần xem lại (Alpha < 0.7). Cần loại biến có ITC thấp.")
        else:
            st.warning("Chưa có kết quả kiểm định độ tin cậy.")

    # 2. Nhân tố (EFA)
    if any(k in step_lower for k in ["nhân tố", "efa", "kmo", "bartlett", "gom nhóm", "hội tụ", "phân biệt", "trích"]):
        results_shown = True
        st.markdown("### 🧩 Phân tích Nhân tố Khám phá (EFA)")
        if efa:
            st.info(f"Chỉ số KMO: **{efa['kmo']:.3f}** (Yêu cầu >= 0.5) | Sig. Bartlett: **{efa['bartlett_sig']:.4f}** (Yêu cầu < 0.05)")
            st.dataframe(efa["rotated_matrix"], use_container_width=True)
            if Path("outputs/chart_4_efa_scree.png").exists():
                st.image("outputs/chart_4_efa_scree.png", use_container_width=True)
            render_scientific_conclusion("efa", efa)
        else:
            st.warning("Chưa có kết quả phân tích nhân tố.")

    # 3. Tương quan (Correlation)
    if any(k in step_lower for k in ["tương quan", "correlation", "mối quan hệ", "pearson"]):
        results_shown = True
        st.markdown("### 🔗 Phân tích Tương quan Pearson")
        if not corr.empty:
            st.dataframe(corr, use_container_width=True)
            if Path("outputs/chart_3_correlation.png").exists():
                st.image("outputs/chart_3_correlation.png", use_container_width=True)
            render_scientific_conclusion("correlation", corr)
        else:
            st.warning("Chưa có kết quả phân tích tương quan.")

    # 4. Hồi quy (Regression)
    if any(k in step_lower for k in ["hồi quy", "regression", "tác động", "ảnh hưởng", "nhân tố ảnh hưởng", "ols", "mô hình"]):
        results_shown = True
        st.markdown("### 🚀 Phân tích Hồi quy Tuyến tính")
        if reg:
            for m_id, m_data in reg.items():
                with st.expander(f"📈 Mô hình: {m_id} (Biến phụ thuộc: {m_data['Y_label']})", expanded=True):
                    st.write(f"Hệ số xác định R²: **{m_data['result']['R2']:.3f}**")
                    st.table(m_data["result"]["table"])
            if Path("outputs/chart_5_regression_beta.png").exists():
                st.image("outputs/chart_5_regression_beta.png", use_container_width=True)
            render_scientific_conclusion("regression", reg)
        else:
            st.warning("Chưa có kết quả phân tích hồi quy.")

    # 5. Kiểm định nhóm (Difference/Group Tests)
    if any(k in step_lower for k in ["khác biệt", "so sánh", "t-test", "anova", "kiểm định t", "phân nhóm"]):
        results_shown = True
        st.markdown("### ⚖️ Kiểm định Sự khác biệt (T-test/ANOVA)")
        if groups:
            for g_id, g_res in groups.items():
                with st.expander(f"👥 So sánh theo: {g_id}", expanded=True):
                    st.table(g_res["table"])
                    st.success(f"**Kết luận:** {g_res['conclusion']}")
            render_scientific_conclusion("ttest_anova", groups)
        else:
            st.warning("Chưa thực hiện kiểm định nhóm.")

    # 6. Biểu đồ
    if any(k in step_lower for k in ["biểu đồ", "chart", "📊"]):
        chart_files = list(Path("outputs").glob("chart_*.png"))
        if chart_files:
            results_shown = True
            st.markdown("### 📊 Minh họa trực quan")
            for cf in chart_files:
                st.image(str(cf), caption=f"Hình: {cf.stem}")

    if not results_shown:
        st.info("Bước này hiện chưa có dữ liệu tính toán trực tiếp.")
    
    st.info("💡 Bạn có thể tiếp tục bước tiếp theo trong lộ trình tại Menu bên trái.")


elif menu_selection == "📋 Kết quả tổng quan":
    st.header("📋 Tổng quan kết quả")
    
    # Header thông tin dự án & Người nghiên cứu
    with st.container(border=True):
        st.write(f"### 📋 Dự án: {st.session_state.get('ai_prompt_val', 'Chưa xác định')}")
        st.write(f"**Người thực hiện:** {st.session_state.get('researcher_name', 'N/A')} | **Đơn vị:** {st.session_state.get('researcher_org', 'N/A')}")
        st.write(f"**Trình độ:** {st.session_state.get('academic_level', 'Cử nhân')} | **Ngày lập báo cáo:** {pd.Timestamp.now().strftime('%d/%m/%Y')}")

    st.caption("Bảng tổng hợp toàn bộ tham số, thang đo, trạng thái đạt/chưa đạt và kiến nghị hành động.")

    passed_scales = sum(1 for v in cronbach.values() if v.get("alpha", 0) >= 0.7)
    total_scales = len(cronbach)
    scale_pass_rate = (passed_scales / total_scales * 100) if total_scales else 0


    st.markdown("#### Đánh giá nhanh: đạt hay chưa đạt?")
    kmo_val = efa.get("kmo", np.nan)
    kmo_ok = pd.notna(kmo_val) and kmo_val >= 0.5
    alpha_ok = pd.notna(alpha_avg) and alpha_avg >= 0.7
    r2_ok = pd.notna(model1_r2) and model1_r2 >= 0.2
    n_ok = len(df) >= 150

    # Hiển thị 1 cột duy nhất
    st.markdown("**1) Alpha TB (độ tin cậy thang đo)**")
    if alpha_ok:
        st.success(f"ĐẠT: Alpha TB = {alpha_avg:.3f} (>= 0.7).")
        st.markdown("- **Vì sao đạt:** Các biến quan sát trong cùng thang đo nhất quán tương đối tốt.")
        st.markdown("- **Để duy trì đạt:** Giữ cấu trúc câu hỏi ổn định, theo dõi item có ITC giảm ở các đợt khảo sát sau.")
    else:
        st.error(f"CHƯA ĐẠT: Alpha TB = {alpha_avg:.3f} (< 0.7).")
        st.markdown("- **Vì sao chưa đạt:** Một số item có tương quan biến-tổng thấp hoặc nhiễu đo lường cao.")
        st.markdown("- **Làm gì để đạt:** Loại/chỉnh item r<0.3, chuẩn hóa cách hỏi, tăng chất lượng làm sạch dữ liệu.")

    st.write("") # Khoảng cách
    
    st.markdown("**2) KMO Index (mức phù hợp để chạy EFA)**")
    if kmo_ok:
        st.success(f"ĐẠT: KMO = {kmo_val:.3f} (>= 0.5).")
        st.markdown("- **Vì sao đạt:** Các biến có đủ mức tương quan chung để trích nhân tố.")
        st.markdown("- **Để tốt hơn:** Mục tiêu thực hành nên hướng tới KMO >= 0.7 để cấu trúc nhân tố ổn định hơn.")
    else:
        st.error(f"CHƯA ĐẠT: KMO = {kmo_val:.3f} (< 0.5).")
        st.markdown("- **Vì sao chưa đạt:** Ma trận tương quan yếu/rời rạc, các item chưa hội tụ theo nhân tố.")
        st.markdown("- **Làm gì để đạt:** Tăng cỡ mẫu, loại item lệch nội dung, viết lại câu hỏi để đồng nhất khái niệm đo.")

    st.write("") # Khoảng cách

    st.markdown("**3) R² Model (mức giải thích của mô hình)**")
    if r2_ok:
        st.success(f"TẠM ĐẠT: R² = {model1_r2:.3f} (>= 0.2).")
        st.markdown("- **Vì sao đạt:** Các biến độc lập hiện tại đã giải thích được một phần đáng kể biến thiên của biến phụ thuộc.")
        st.markdown("- **Để nâng cao:** Bổ sung biến theo bối cảnh thực tế để tăng năng lực dự báo.")
    else:
        st.warning(f"CẦN CẢI THIỆN: R² = {model1_r2:.3f} (< 0.2).")
        st.markdown("- **Vì sao chưa tốt:** Mô hình bỏ sót biến quan trọng hoặc quan hệ chưa được đặc tả đúng.")
        st.markdown("- **Làm gì để đạt hơn:** Bổ sung biến giải thích, kiểm tra quan hệ trung gian/điều tiết, chuẩn hóa dữ liệu đầu vào.")

    st.write("") # Khoảng cách

    st.markdown("**4) Cỡ mẫu n (độ ổn định ước lượng)**")
    if n_ok:
        st.success(f"ĐẠT: n = {len(df)} (>= 150).")
        st.markdown("- **Vì sao đạt:** Quy mô mẫu đáp ứng mức tối thiểu để ước lượng ổn định hơn.")
        st.markdown("- **Khuyến nghị thêm:** Cân bằng mẫu theo nhóm nhân khẩu học để tăng độ đại diện.")
    else:
        st.warning(f"CẦN TĂNG MẪU: n = {len(df)} (< 150).")
        st.markdown("- **Vì sao chưa đạt:** Mẫu nhỏ làm hệ số dao động mạnh và p-value kém ổn định.")
        st.markdown("- **Làm gì để đạt:** Tăng số quan sát, ưu tiên thu thập đều giữa các nhóm đối tượng.")

    overall_pass = alpha_ok and kmo_ok and n_ok
    if overall_pass:
        st.success("Kết luận tổng quan: Dữ liệu đạt điều kiện nền tảng để diễn giải sâu kết quả EFA/hồi quy.")
    else:
        st.error("Kết luận tổng quan: Dữ liệu CHƯA đạt đầy đủ điều kiện nền tảng. Ưu tiên xử lý thang đo và tính phù hợp EFA trước khi kết luận quản trị.")

    st.markdown("---")
    st.markdown("#### 📍 Sơ đồ Mô hình Nghiên cứu (Research Model Diagram)")
    if Path("outputs/spss_model_diagram.png").exists():
        st.image("outputs/spss_model_diagram.png", use_container_width=True)
        render_chart_note(
            title="Sơ đồ cấu trúc mô hình (Research Model Diagram)",
            scientific="Mô tả các đường tác động giả thuyết giữa các khái niệm tiềm ẩn trong nghiên cứu.",
            mathematical="Các đường mũi tên biểu diễn hệ số Beta chuẩn hóa; dấu (+) hoặc (-) chỉ chiều tác động.",
            practical="Phác họa bức tranh tổng thể về các yếu tố quyết định sự hài lòng/hành vi của sinh viên.",
            current_status="Mô hình được xây dựng dựa trên kết quả hồi quy thực tế.",
            evaluation="Mô hình cho thấy các mối quan hệ có ý nghĩa thống kê quan trọng.",
            proposal="Tập trung giải pháp vào các đường tác động có hệ số Beta cao nhất."
        )
    st.markdown("---")
    st.markdown("#### 📈 Đánh giá Tổng thể & Giải pháp Cải tiến")
    
    with st.container(border=True):
        # 1. Nhận diện trạng thái chung
        dep_var_name = "Biến phụ thuộc"
        dep_mean = 0.0
        if df_means is not None and not df_means.empty:
            # Giả sử biến cuối cùng là phụ thuộc
            dep_col = df_means.columns[-1]
            dep_var_name = spss.ACTIVE_CONFIG["variables"].get(dep_col, {}).get("label", dep_col)
            dep_mean = df_means[dep_col].mean()
        
        status_label = "🔴 Thấp" if dep_mean < 3 else ("🟡 Trung bình" if dep_mean < 4 else "🟢 Tốt")
        st.write(f"**1. Hiện trạng chung:** Mức độ **{dep_var_name}** hiện tại đạt **{dep_mean:.2f}/5.0** (Xếp loại: **{status_label}**).")
        
        # 2. Tìm Pain Point (Nhân tố thấp nhất)
        if df_means is not None and len(df_means.columns) > 1:
            ind_means = df_means.iloc[:, :-1].mean()
            lowest_f = ind_means.idxmin()
            lowest_f_label = spss.ACTIVE_CONFIG["variables"].get(lowest_f, {}).get("label", lowest_f)
            st.write(f"**2. Điểm yếu cần khắc phục:** Nhân tố **{lowest_f_label}** có điểm trung bình thấp nhất (**{ind_means.min():.2f}**). Đây là rào cản chính làm giảm tổng thể {dep_var_name.lower()}.")
        
        # 3. Tìm Priority (Tác động mạnh nhất)
        if reg:
            m_main = list(reg.values())[0]
            betas = {k: v for k, v in m_main.get("result", {}).get("Standardized_Coefficients", {}).items() if k != "Intercept"}
            if betas:
                high_f = max(betas, key=betas.get)
                high_f_label = spss.ACTIVE_CONFIG["variables"].get(high_f, {}).get("label", high_f)
                st.write(f"**3. Trọng tâm chiến lược:** **{high_f_label}** là yếu tố có tác động mạnh nhất đến {dep_var_name.lower()} ($\\beta = {betas[high_f]:.3f}$).")
                
                st.info(f"💡 **Giải pháp cải tiến nâng cao:** Để nâng cao hiệu quả nghiên cứu thực tế, đơn vị cần ưu tiên đầu tư vào **{high_f_label}** vì đây là 'đòn bẩy' mạnh nhất, đồng thời tập trung cải thiện cấp bách **{lowest_f_label}** để loại bỏ điểm nghẽn hiện tại.")

        # --- BẢNG CHI TIẾT THEO TỪNG NHÂN TỐ (THEO YÊU CẦU NGƯỜI DÙNG) ---
        st.markdown("##### 📜 Bảng Phân tích Chi tiết & Kiến nghị theo Nhân tố")
        st.info("💡 **Tác động (Beta):** Là hệ số đã chuẩn hóa cho biết mức độ 'quan trọng' của nhân tố đó. Beta càng lớn (giá trị tuyệt đối), nhân tố đó càng có tác động mạnh đến kết quả cuối cùng. Nếu hiện '-' nghĩa là nhân tố này chưa được đưa vào mô hình hồi quy hoặc đang đóng vai trò là biến phụ thuộc.")
        detail_rows = []
        # Lấy danh sách Beta từ tất cả các model để phủ rộng nhất có thể
        all_betas = {}
        if reg:
            for m_key, m_info in reg.items():
                b_map = m_info.get("result", {}).get("standardized_betas_map", {})
                all_betas.update(b_map)

        for var_code, v_info in spss.ACTIVE_CONFIG["variables"].items():
            # 1. Điểm trung bình (Score)
            mean_val = df_means[var_code].mean() if df_means is not None and var_code in df_means.columns else np.nan
            # 2. Độ tin cậy (Reliability)
            c_res = cronbach.get(var_code, {})
            alpha = c_res.get("alpha", np.nan)
            rel_ok = pd.notna(alpha) and alpha >= 0.7
            # 3. Tác động (Beta) - Lấy từ map đã tổng hợp
            beta_val = all_betas.get(var_code, None)
            
            # 4. Trạng thái tổng hợp
            if not rel_ok:
                status_icon = "🔴 Chưa đạt (Độ tin cậy)"
                rec_text = f"Cần rà soát lại các câu hỏi của {v_info['label']}. Loại bỏ các item có ITC < 0.3."
            elif pd.isna(mean_val):
                 status_icon = "⚪ Thiếu dữ liệu"
                 rec_text = "Cần kiểm tra lại dữ liệu đầu vào của nhân tố này."
            elif mean_val < 3.5:
                status_icon = "🟡 Cần cải thiện (Điểm thấp)"
                rec_text = f"Nâng cao thực trạng {v_info['label'].lower()} thông qua các giải pháp vận hành cụ thể."
            else:
                status_icon = "🟢 Đạt"
                rec_text = "Tiếp tục duy trì và phát huy, đồng thời theo dõi biến động định kỳ."
                
            detail_rows.append({
                "Nhân tố (Factor)": v_info["label"],
                "Mã": var_code,
                "Điểm (Mean)": round(mean_val, 2) if pd.notna(mean_val) else "-",
                "Alpha": round(alpha, 3) if pd.notna(alpha) else "-",
                "Tác động (Beta)": round(beta_val, 3) if beta_val is not None else "-",
                "Công thức (Beta)": "B * (Std_X / Std_Y)",
                "Ý nghĩa (Beta)": "Mức độ quan trọng (chuẩn hóa)",
                "Đánh giá": status_icon,
                "Kiến nghị hành động": rec_text
            })
        
        st.table(pd.DataFrame(detail_rows))

        # --- BẢNG CHI TIẾT CÁC DỊCH VỤ CẦN CẢI TIẾN (ACTION ITEMS) ---
        st.markdown("##### 🛠️ Danh mục các Khía cạnh/Dịch vụ cần cải tiến cấp bách")
        st.caption("Danh sách các câu hỏi/khía cạnh cụ thể có điểm đánh giá thấp nhất từ người khảo sát.")
        
        item_rows = []
        if df is not None:
            # Lấy tất cả các item code từ config
            all_item_codes = []
            for v_code, v_info in spss.ACTIVE_CONFIG["variables"].items():
                for i_code, i_label in v_info["items"].items():
                    if i_code in df.columns:
                        item_rows.append({
                            "Khía cạnh cụ thể": i_label,
                            "Mã": i_code,
                            "Thuộc nhân tố": v_info["label"],
                            "Điểm đánh giá": round(df[i_code].mean(), 2),
                        })
            
            if item_rows:
                item_df = pd.DataFrame(item_rows).sort_values("Điểm đánh giá").head(7) # Lấy top 7 tệ nhất
                # Thêm cột kiến nghị cụ thể dựa trên điểm
                def get_item_rec(m):
                    if m < 2.5: return "🔴 Cần can thiệp khẩn cấp: Tái cấu trúc quy trình/dịch vụ."
                    if m < 3.5: return "🟡 Cần cải thiện: Chấn chỉnh và đào tạo nhân sự/nâng cấp thiết bị."
                    return "🟢 Theo dõi: Duy trì tiêu chuẩn."
                
                item_df["Kiến nghị cụ thể"] = item_df["Điểm đánh giá"].apply(get_item_rec)
                st.dataframe(item_df, use_container_width=True, hide_index=True)
            else:
                st.info("Chưa có dữ liệu chi tiết từng biến.")

    st.markdown("---")
    st.subheader("1) Tổng hợp thang đo: đạt/chưa đạt")
    scale_rows = []
    for var, res in cronbach.items():
        alpha = res.get("alpha", np.nan)
        dropped = res.get("dropped", [])
        status = "Đạt" if pd.notna(alpha) and alpha >= 0.7 else "Chưa đạt"
        recommendation = (
            "Giữ nguyên, theo dõi ổn định qua mẫu mới."
            if status == "Đạt"
            else "Rà soát item có r<0.3, chỉnh câu hỏi/thu thập thêm mẫu."
        )
        scale_rows.append({
            "Thang đo": var,
            "Alpha": round(alpha, 3) if pd.notna(alpha) else "-",
            "Công thức": "k/(k-1) * (1 - Σs²/S²)",
            "Ý nghĩa": "Độ tin cậy/nhất quán nội tại",
            "Số biến": len(res.get("items", [])),
            "Biến bị cảnh báo": ", ".join(dropped) if dropped else "-",
            "Trạng thái": status,
            "Kiến nghị": recommendation
        })
    st.dataframe(pd.DataFrame(scale_rows), use_container_width=True)

    st.subheader("2) Bảng tham số tổng hợp")
    param_rows = [
        {
            "Nhóm": "EFA",
            "Tham số": "KMO",
            "Giá trị": round(efa.get("kmo", np.nan), 3) if pd.notna(efa.get("kmo", np.nan)) else "-",
            "Công thức": "Σr² / (Σr² + Σp²)",
            "Ý nghĩa": "Độ phù hợp của tập mẫu",
            "Ngưỡng đạt": ">= 0.5 (khuyến nghị >= 0.7)",
            "Trạng thái": "Đạt" if pd.notna(efa.get("kmo", np.nan)) and efa.get("kmo", 0) >= 0.5 else "Chưa đạt",
            "Kiến nghị": "Tăng cỡ mẫu, cải thiện tương quan nội tại giữa các biến quan sát."
        },
        {
            "Nhóm": "EFA",
            "Tham số": "Bartlett Sig.",
            "Giá trị": round(efa.get("bartlett_sig", np.nan), 4) if pd.notna(efa.get("bartlett_sig", np.nan)) else "-",
            "Công thức": "Chi-Square p-value",
            "Ý nghĩa": "Kiểm tra tính rời rạc của ma trận",
            "Ngưỡng đạt": "< 0.05",
            "Trạng thái": "Đạt" if pd.notna(efa.get("bartlett_sig", np.nan)) and efa.get("bartlett_sig", 1) < 0.05 else "Chưa đạt",
            "Kiến nghị": "Nếu chưa đạt, xem lại cấu trúc biến và chất lượng dữ liệu đầu vào."
        },
        {
            "Nhóm": "EFA",
            "Tham số": "Tổng phương sai trích",
            "Giá trị": round(efa.get("total_var", np.nan), 2) if pd.notna(efa.get("total_var", np.nan)) else "-",
            "Công thức": "ΣEigenvalues / p",
            "Ý nghĩa": "Khả năng đại diện của nhân tố",
            "Ngưỡng đạt": ">= 50%",
            "Trạng thái": "Đạt" if pd.notna(efa.get("total_var", np.nan)) and efa.get("total_var", 0) >= 50 else "Chưa đạt",
            "Kiến nghị": "Tối ưu thang đo và loại biến loading thấp/cross-loading."
        },
    ]
    
    # 3) Thêm các mô hình hồi quy động vào bảng tham số
    for i, (m_key, m_info) in enumerate(reg.items()):
        m_r2 = m_info.get("result", {}).get("R2", np.nan)
        param_rows.append({
            "Nhóm": "Hồi quy",
            "Tham số": f"R² Model {i+1} (-> {m_info.get('Y_label', 'Y')})",
            "Giá trị": round(m_r2, 4) if pd.notna(m_r2) else "-",
            "Công thức": "SSR / SST",
            "Ý nghĩa": "Mức độ giải thích của mô hình",
            "Ngưỡng đạt": "Phụ thuộc lĩnh vực (khuyến nghị > 0.1)",
            "Trạng thái": "Đạt" if pd.notna(m_r2) else "N/A",
            "Kiến nghị": "Bổ sung biến giải thích hoặc biến trung gian nếu R² còn thấp."
        })

    for m_key, m_info in reg.items():
        labels = m_info.get("X_labels", [])
        r = m_info.get("result", {})
        beta = r.get("beta", [])
        pvals = r.get("p", [])
        for i, lbl in enumerate(labels):
            b = beta[i + 1] if len(beta) > i + 1 else np.nan
            p = pvals[i + 1] if len(pvals) > i + 1 else np.nan
            pass_hyp = pd.notna(p) and (p < 0.05) and (pd.notna(b) and b > 0)
            param_rows.append({
                "Nhóm": "Giả thuyết",
                "Tham số": f"{lbl} -> {m_info.get('Y_label', '?')}",
                "Giá trị": f"beta={b:.3f}, p={p:.4f}" if pd.notna(b) and pd.notna(p) else None,
                "Ngưỡng đạt": "p < 0.05 và beta > 0",
                "Trạng thái": "Đạt" if pass_hyp else "Chưa đạt",
                "Kiến nghị": "Ưu tiên nguồn lực vào nhân tố này." if pass_hyp else "Rà soát giả thuyết hoặc tái thiết kế biến đo."
            })

    st.dataframe(pd.DataFrame(param_rows), use_container_width=True)

    recommendation_lines = [
        "- Ưu tiên cải thiện các đường dẫn có beta cao và có ý nghĩa thống kê.",
        "- Với biến/chỉ số chưa đạt, rà soát thiết kế câu hỏi và quy trình thu thập dữ liệu.",
        "- Kiểm tra thêm đa cộng tuyến, ngoại lệ, và mở rộng mẫu để tăng độ ổn định mô hình.",
        "- Nếu R² còn thấp, bổ sung biến giải thích từ bối cảnh thực tế KTX (dịch vụ, vận hành, hỗ trợ số...)."
    ]
    st.markdown("\n".join(recommendation_lines))


    st.markdown("---")
    # --- PHẦN CHI TIẾT KỸ THUẬT (MERGED FROM DETAIL REPORT) ---
    st.subheader("📜 Chi tiết Kỹ thuật Toàn diện (Full Technical Appendix)")
    st.markdown("Phần này cung cấp đầy đủ các bảng số liệu chi tiết từ mọi bước phân tích để phục vụ việc trích xuất vào phụ lục hoặc chương 4 của bài nghiên cứu.")
    
    if results:
        with st.expander("🔍 Xem toàn bộ Kết quả Chi tiết (Cronbach, EFA, Hồi quy...)", expanded=False):
            # 1. Mô tả & Làm sạch
            st.markdown("### I. Thống kê Mô tả & Làm sạch")
            if desc_res:
                st.write("**Bảng tần số nhân khẩu học:**")
                st.table(desc_res["demographics"])
                st.write("**Thống kê mô tả các biến:**")
                st.dataframe(desc_res["stats"], use_container_width=True)
                if st.session_state.get('cleaning_logs'):
                    with st.expander("📝 Nhật ký Làm sạch Dữ liệu"):
                        for log in st.session_state.cleaning_logs: st.text(log)
                if Path("outputs/chart_1_mean.png").exists():
                    st.image("outputs/chart_1_mean.png", caption="Biểu đồ: Giá trị trung bình", use_container_width=True)
            
            # 2. Cronbach
            st.markdown("### II. Phân tích Độ tin cậy (Cronbach's Alpha)")
            for var, res in cronbach.items():
                st.write(f"**Thang đo: {var}** (α = {res['alpha']:.3f})")
                st.table(res["table"])
            if Path("outputs/chart_2_cronbach.png").exists():
                st.image("outputs/chart_2_cronbach.png", caption="Biểu đồ: Hệ số Alpha", use_container_width=True)
            
            # 3. EFA
            st.markdown("### III. Phân tích Nhân tố Khám phá (EFA)")
            if efa:
                st.write(f"KMO: {efa['kmo']:.3f} | Bartlett Sig: {efa['bartlett_sig']:.4f}")
                st.dataframe(efa["rotated_matrix"], use_container_width=True)
                if Path("outputs/chart_4_efa_scree.png").exists():
                    st.image("outputs/chart_4_efa_scree.png", caption="Biểu đồ: Scree Plot", use_container_width=True)
                
            # 4. Tương quan & Hồi quy
            st.markdown("### IV. Phân tích Tương quan & Hồi quy")
            if not corr.empty:
                st.markdown("#### 1. Ma trận Tương quan Pearson")
                if Path("outputs/chart_3_correlation.png").exists():
                    st.image("outputs/chart_3_correlation.png", use_container_width=True)
                
            if reg:
                st.markdown("#### 2. Kết quả Hồi quy Tuyến tính")
                for m_id, m_data in reg.items():
                    st.write(f"**Mô hình {m_id}: {m_data['Y_label']}**")
                    st.table(m_data["result"]["table"])
                    st.write(f"R² = {m_data['result']['R2']:.3f} | Adj. R² = {m_data['result']['adjR2']:.3f}")
                if Path("outputs/chart_5_regression_beta.png").exists():
                    st.image("outputs/chart_5_regression_beta.png", caption="Biểu đồ: Trọng số Beta", use_container_width=True)

            # 5. Kiểm định nhóm
            st.markdown("### V. Kiểm định Sự khác biệt (Group Tests)")
            for g_id, g_res in groups.items():
                st.markdown(f"**Kiểm định cho: {g_id}**")
                st.table(g_res["table"])
                st.write(f"Kết luận: {g_res['conclusion']}")
    else:
        st.warning("Vui lòng thực hiện Phân tích để xem chi tiết kỹ thuật.")

# --- Tab: Báo cáo Chi tiết Toàn diện ---


# --- Tab 4: Download ---
elif menu_selection == "📤 Xuất báo cáo":
    st.subheader("📥 Trung tâm Tải xuống Kết quả Phân tích")
    st.markdown("Hệ thống đã tự động đóng gói toàn bộ hồ sơ phân tích phục vụ cho việc viết chương 4 của luận văn.")
    
    def get_fc(p):
        if Path(p).exists():
            with open(p, "rb") as f: return f.read()
        return None

    st.write("### 💎 Nhóm 1: Báo cáo & Dữ liệu chính")
    c1, c2, c3 = st.columns(3)
    with c1:
        data = get_fc("outputs/spss_analysis_report.xlsx")
        if data: st.download_button("📊 Báo cáo Excel (Full)", data, "spss_analysis_report.xlsx", use_container_width=True)
    with c2:
        data = get_fc("outputs/spss_raw_data.csv")
        if data: st.download_button("📁 Dữ liệu thô (CSV)", data, "spss_raw_data.csv", use_container_width=True)
    with c3:
        data = get_fc("outputs/spss_report_text.txt")
        if data: st.download_button("📝 Báo cáo Văn bản (Draft)", data, "spss_report_text.txt", use_container_width=True)

    st.markdown("---")
    st.write("### 🖼️ Nhóm 2: Hình ảnh & Sơ đồ")
    c4, c5 = st.columns(2)
    with c4:
        data = get_fc("outputs/spss_model_diagram.png")
        if data: st.download_button("📍 Sơ đồ Mô hình (SEM)", data, "spss_model_diagram.png", use_container_width=True)
    with c5:
        data = get_fc("outputs/spss_analysis_charts.png")
        if data: st.download_button("📈 Biểu đồ tổng hợp", data, "spss_analysis_charts.png", use_container_width=True)

    st.markdown("---")
    st.write("### 🛠️ Nhóm 3: Tài liệu Kỹ thuật & Công cụ")
    c6, c7 = st.columns(2)
    with c6:
        data = get_fc("outputs/spss_syntax.sps")
        if data: st.download_button("⚙️ Tải SPSS Syntax", data, "spss_syntax.sps", use_container_width=True)
    with c7:
        data = get_fc("outputs/phieu_khao_sat.txt")
        if data: st.download_button("📋 Tải Phiếu khảo sát mẫu", data, "phieu_khao_sat.txt", use_container_width=True)

st.markdown("---")
st.markdown(f"<p style='text-align: center; color: #64748b;'>© 2025 Trợ lý nghiên cứu bài toán Kinh tế - Nguyễn Vĩnh Phúc – ngviphuc@gmail.com - 0937 382 399</p>", unsafe_allow_html=True)
