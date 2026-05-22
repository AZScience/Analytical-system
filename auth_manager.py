import streamlit as st
import gspread
from google.oauth2.service_account import Credentials
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
    pass

def get_users_worksheet():
    """Kết nối và lấy worksheet Users bằng gspread trực tiếp (không cache)."""
    if "connections" not in st.secrets or "gsheets" not in st.secrets["connections"]:
        raise Exception("Không tìm thấy cấu hình [connections.gsheets] trong Secrets.")
        
    info = dict(st.secrets["connections"]["gsheets"])
    sa_keys = [
        "type", "project_id", "private_key_id", "private_key",
        "client_email", "client_id", "auth_uri", "token_uri",
        "auth_provider_x509_cert_url", "client_x509_cert_url", "universe_domain"
    ]
    sa_info = {k: info[k] for k in sa_keys if k in info}
    if "private_key" in sa_info and isinstance(sa_info["private_key"], str):
        sa_info["private_key"] = sa_info["private_key"].replace("\\n", "\n")
        
    scopes = [
        'https://www.googleapis.com/auth/spreadsheets',
        'https://www.googleapis.com/auth/drive'
    ]
    creds = Credentials.from_service_account_info(sa_info, scopes=scopes)
    client = gspread.authorize(creds)
    
    spreadsheet_url = info.get("spreadsheet")
    if not spreadsheet_url:
        raise Exception("Không tìm thấy trường 'spreadsheet' trong cấu hình GSheets.")
        
    sh = client.open_by_url(spreadsheet_url)
    return sh.worksheet("Users")

def get_user_limits(email: str):
    """Lấy thông tin giới hạn của user từ Google Sheets."""
    try:
        ws = get_users_worksheet()
        records = ws.get_all_records()
    except Exception as e:
        st.warning(f"Lỗi kết nối hoặc đọc Google Sheets: {e}")
        max_lim = 9999 if email == "ngviphuc@gmail.com" else 3
        return {"Email": email, "UsageCount": 0, "MaxLimit": max_lim}
        
    # Tìm email trong records (không phân biệt hoa thường và khoảng trắng)
    user_row = None
    target_email = email.strip().lower()
    for r in records:
        if str(r.get("Email", "")).strip().lower() == target_email:
            user_row = r
            break
            
    if user_row is None:
        # Nếu chưa tồn tại, tạo mới
        max_lim = 9999 if email == "ngviphuc@gmail.com" else 3
        new_row = [email, 0, max_lim]
        try:
            ws.append_row(new_row)
        except Exception as e:
            st.error(f"Lỗi ghi nhận user mới vào Google Sheets: {e}")
        return {"Email": email, "UsageCount": 0, "MaxLimit": max_lim}
        
    # Nếu tồn tại, parse các giá trị
    try:
        usage_cnt = int(float(user_row.get("UsageCount", 0)))
        max_lim = int(float(user_row.get("MaxLimit", 3)))
    except Exception:
        usage_cnt = 0
        max_lim = 9999 if email == "ngviphuc@gmail.com" else 3
        
    return {"Email": email, "UsageCount": usage_cnt, "MaxLimit": max_lim}

def increment_usage(email: str):
    """Trừ đi 1 lượt sử dụng của user (tăng UsageCount thêm 1)."""
    try:
        ws = get_users_worksheet()
        records = ws.get_all_records()
    except Exception as e:
        st.error(f"Lỗi đọc dữ liệu Google Sheets để cập nhật lượt dùng: {e}")
        return False
        
    # Tìm user
    row_idx = None
    user_row = None
    target_email = email.strip().lower()
    # get_all_records không bao gồm hàng tiêu đề (hàng 1) nên record đầu tiên tương ứng hàng 2
    for idx, r in enumerate(records, start=2):
        if str(r.get("Email", "")).strip().lower() == target_email:
            user_row = r
            row_idx = idx
            break
            
    if row_idx is not None:
        try:
            current_val = int(float(user_row.get("UsageCount", 0)))
        except Exception:
            current_val = 0
        
        # Cập nhật ô UsageCount (cột 2 là UsageCount)
        try:
            ws.update_cell(row_idx, 2, current_val + 1)
            return True
        except Exception as e:
            st.error(f"Lỗi cập nhật số lượt dùng lên Google Sheets: {e}")
            return False
    else:
        # Nếu chưa tồn tại trong sheet
        max_lim = 9999 if email == "ngviphuc@gmail.com" else 3
        new_row = [email, 1, max_lim]
        try:
            ws.append_row(new_row)
            return True
        except Exception as e:
            st.error(f"Lỗi cập nhật số lượt dùng lên Google Sheets: {e}")
            return False
            
    return False

def can_use(email: str):
    """Kiểm tra user còn lượt sử dụng không."""
    if email == "ngviphuc@gmail.com":
        return True
    user = get_user_limits(email)
    return int(user["UsageCount"]) < int(user["MaxLimit"])
