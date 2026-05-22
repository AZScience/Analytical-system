import streamlit as st
import pandas as pd
from streamlit_gsheets import GSheetsConnection
import json
import os

def setup_google_credentials():
    """Tạo file google_credentials.json từ st.secrets nếu đang chạy trên Cloud."""
    if "GOOGLE_CREDENTIALS" in st.secrets:
        creds = st.secrets["GOOGLE_CREDENTIALS"]
        if hasattr(creds, "to_dict"):
            creds_dict = creds.to_dict()
        else:
            creds_dict = dict(creds)
        with open('google_credentials.json', 'w') as f:
            json.dump(creds_dict, f, default=dict)
                
def setup_gsheets_credentials():
    """Đảm bảo thiết lập cấu hình gsheets"""
    pass # GSheetsConnection tự đọc từ st.secrets["connections"]["gsheets"]

def get_user_limits(email: str):
    """Lấy thông tin giới hạn của user từ Google Sheets."""
    if email == "ngviphuc@gmail.com":
        return {"Email": email, "UsageCount": 0, "MaxLimit": 9999}
        
    try:
        conn = st.connection("gsheets", type=GSheetsConnection)
        df = conn.read(worksheet="Users", usecols=[0, 1, 2], ttl=0)
        # Đảm bảo cấu trúc cột
        if "Email" not in df.columns:
            df = pd.DataFrame(columns=["Email", "UsageCount", "MaxLimit"])
    except Exception as e:
        st.warning(f"Lỗi đọc dữ liệu giới hạn từ Google Sheets: {e}")
        df = pd.DataFrame(columns=["Email", "UsageCount", "MaxLimit"])
    
    user_row = df[df["Email"] == email]
    if user_row.empty:
        # Nếu là user mới, tặng 3 lượt
        new_row = {"Email": email, "UsageCount": 0, "MaxLimit": 3}
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        try:
            conn.update(worksheet="Users", data=df)
        except Exception as e:
            st.error(f"Lỗi ghi nhận user mới vào Google Sheets: {e}")
        return new_row
    
    row_dict = user_row.iloc[0].to_dict()
    try:
        row_dict["UsageCount"] = int(float(row_dict["UsageCount"]))
        row_dict["MaxLimit"] = int(float(row_dict["MaxLimit"]))
    except Exception:
        row_dict["UsageCount"] = 0
        row_dict["MaxLimit"] = 3
    return row_dict

def increment_usage(email: str):
    """Trừ đi 1 lượt sử dụng của user."""
    if email == "ngviphuc@gmail.com":
        return True # Admin không bị trừ
        
    try:
        conn = st.connection("gsheets", type=GSheetsConnection)
        df = conn.read(worksheet="Users", ttl=0)
    except Exception as e:
        st.error(f"Lỗi đọc dữ liệu Google Sheets để cập nhật lượt dùng: {e}")
        return False
        
    user_idx = df.index[df["Email"] == email].tolist()
    if user_idx:
        idx = user_idx[0]
        try:
            current_val = int(float(df.at[idx, "UsageCount"]))
        except Exception:
            current_val = 0
        df.at[idx, "UsageCount"] = current_val + 1
        try:
            conn.update(worksheet="Users", data=df)
        except Exception as e:
            st.error(f"Lỗi cập nhật số lượt dùng lên Google Sheets: {e}")
            return False
        return True
    return False

def can_use(email: str):
    """Kiểm tra user còn lượt sử dụng không."""
    if email == "ngviphuc@gmail.com":
        return True
    user = get_user_limits(email)
    return int(user["UsageCount"]) < int(user["MaxLimit"])
