import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime
import streamlit as st

def log_feedback_to_gsheet(name: str, rating: int, comments: str, sheet_name: str = "feedback"):
    try:
        creds_dict = st.secrets["google_sheets"]
        scopes = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive"
        ]
        credentials = Credentials.from_service_account_info(creds_dict, scopes=scopes)
        client = gspread.authorize(credentials)
        sheet = client.open(sheet_name).sheet1

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        sheet.append_row([name or "Anonymous", rating, comments, timestamp])
        st.success("✅ Feedback successfully logged!")

    except Exception as e:
        st.error(f"❌ Feedback logging error: {e}")