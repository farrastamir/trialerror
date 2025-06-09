import streamlit as st
import pandas as pd
import zipfile
import urllib.request
import os
from io import BytesIO

from dashboard_tier import run_tier_dashboard
from dashboard_sosmed import run_sosmed_dashboard

st.set_page_config(layout="wide")
st.title("📰 Topic Summary NoLimit Dashboard")

@st.cache_data(show_spinner=False)
def extract_csv_from_zip(zip_file):
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        csv_files = [f for f in zip_ref.namelist() if f.endswith('.csv')]
        dfs = []
        for f in csv_files:
            with zip_ref.open(f) as file:
                df = pd.read_csv(file, delimiter=';', quotechar='"', on_bad_lines='skip', engine='python')
                dfs.append(df)
        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

st.markdown("### 📁 Pilih sumber data ZIP")

input_type = st.radio("Input ZIP via:", ["Upload File", "Link Download"])
zip_data = None

if input_type == "Upload File":
    uploaded = st.file_uploader("Unggah file ZIP", type="zip")
    if uploaded:
        zip_data = uploaded
else:
    zip_url = st.text_input("Masukkan URL file ZIP")
    if st.button("Download ZIP"):
        if zip_url:
            try:
                response = urllib.request.urlopen(zip_url)
                zip_data = BytesIO(response.read())
            except Exception as e:
                st.error(f"Gagal mengunduh ZIP: {e}")

if zip_data:
    with st.spinner("🔍 Mengekstrak dan membaca file..."):
        df = extract_csv_from_zip(zip_data)
        if df is None or df.empty:
            st.error("❌ Tidak ada data terbaca.")
        else:
            if 'tier' in df.columns:
                run_tier_dashboard(df)
            else:
                run_sosmed_dashboard(df)