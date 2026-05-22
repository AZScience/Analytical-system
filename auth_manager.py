import streamlit as st
import pandas as pd
from streamlit_gsheets import GSheetsConnection
import json
import os

def setup_google_credentials():
    """Tạo file google_credentials.json từ st.secrets nếu đang chạy trên Cloud."""
    if not os.path.exists('google_credentials.json'):
        if "GOOGLE_CREDENTIALS" in st.secrets:
            with open('google_credentials.json', 'w') as f:
                json.dump(dict(st.secrets["GOOGLE_CREDENTIALS"]), f)
                
def setup_gsheets_credentials():
    """Đảm bảo thiết lập cấu hình gsheets"""
    pass # GSheetsConnection tự đọc từ st.secrets["connections"]["gsheets"]

def get_user_limits(email: str):
    """Lấy thông tin giới hạn của user từ Google Sheets."""
    if email == "ngviphuc@gmail.com":
        return {"Email": email, "UsageCount": 0, "MaxLimit": 9999}
        
    try:
        conn = st.connection("gsheets", type=GSheetsConnection)
        df = conn.read(worksheet="Users", usecols=[0, 1, 2])
        # Đảm bảo cấu trúc cột
        if "Email" not in df.columns:
            df = pd.DataFrame(columns=["Email", "UsageCount", "MaxLimit"])
    except Exception as e:
        df = pd.DataFrame(columns=["Email", "UsageCount", "MaxLimit"])
    
    user_row = df[df["Email"] == email]
    if user_row.empty:
        # Nếu là user mới, tặng 3 lượt
        new_row = {"Email": email, "UsageCount": 0, "MaxLimit": 3}
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        try:
            conn.update(worksheet="Users", data=df)
        except Exception:
            pass # Bỏ qua nếu sheet chưa cấu hình ghi
        return new_row
    
    return user_row.iloc[0].to_dict()

def increment_usage(email: str):
    """Trừ đi 1 lượt sử dụng của user."""
    if email == "ngviphuc@gmail.com":
        return True # Admin không bị trừ
        
    try:
        conn = st.connection("gsheets", type=GSheetsConnection)
        df = conn.read(worksheet="Users")
    except Exception:
        return False
        
    user_idx = df.index[df["Email"] == email].tolist()
    if user_idx:
        idx = user_idx[0]
        df.at[idx, "UsageCount"] = int(df.at[idx, "UsageCount"]) + 1
        try:
            conn.update(worksheet="Users", data=df)
        except Exception:
            pass
        return True
    return False

def can_use(email: str):
    """Kiểm tra user còn lượt sử dụng không."""
    if email == "ngviphuc@gmail.com":
        return True
    user = get_user_limits(email)
    return int(user["UsageCount"]) < int(user["MaxLimit"])
