import streamlit as st
import pandas as pd
import zipfile
import urllib.request
import os
import io
from collections import Counter
import re

st.set_page_config(layout="wide")
st.title("📰 Topic Summary NoLimit Dashboard")

@st.cache_data(show_spinner=False)
def extract_csv_from_zip(zip_file):
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        csv_files = [f for f in zip_ref.namelist() if f.endswith('.csv')]
        if not csv_files:
            st.error("❌ Tidak ada file .csv dalam ZIP.")
            return []
        dfs = []
        for f in csv_files:
            with zip_ref.open(f) as file:
                try:
                    df = pd.read_csv(file, delimiter=';', quotechar='"', on_bad_lines='skip', engine='python')
                    dfs.append(df)
                except Exception as e:
                    st.warning(f"Gagal membaca {f}: {e}")
        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

if 'show_wordcloud' not in st.session_state:
    st.session_state['show_wordcloud'] = False
if 'dynamic_wordcloud' not in st.session_state:
    st.session_state['dynamic_wordcloud'] = True
if 'reset_filter' not in st.session_state:
    st.session_state['sentiment_filter'] = "All"
    st.session_state['label_filter'] = "All"
    st.session_state['keyword_input'] = ""
    st.session_state['highlight_words'] = ""
    st.session_state['platform_filter'] = "All"
    st.session_state['group_filter'] = "All"
    st.session_state['mode_filter'] = "All"

st.markdown("### 📁 Pilih sumber data")
input_type = st.radio("Input ZIP via:", ["Upload File", "Link Download"])

zip_data = None
if input_type == "Upload File":
    uploaded = st.file_uploader("Unggah file ZIP", type="zip")
    if uploaded:
        zip_data = uploaded
else:
    zip_url = st.text_input("Masukkan URL file ZIP")
    if st.button("Proceed"):
        if zip_url:
            try:
                tmp_path = "/tmp/downloaded.zip"
                urllib.request.urlretrieve(zip_url, tmp_path)
                zip_data = tmp_path
            except Exception as e:
                st.error(f"❌ Gagal mengunduh: {e}")

if 'last_df' not in st.session_state:
    st.session_state['last_df'] = None

if zip_data:
    with st.spinner("Membaca dan memproses data..."):
        df = extract_csv_from_zip(zip_data)
        if not df.empty:
            st.session_state['last_df'] = df.copy()

if st.session_state['last_df'] is not None:
    df = st.session_state['last_df']
    is_sosmed = 'tier' not in df.columns

    if is_sosmed:
        if 'specific_resource' not in df.columns:
            df['specific_resource'] = ''
        if 'object_group' not in df.columns:
            df['object_group'] = ''

        for col in ['content', 'final_sentiment']:
            df[col] = df[col].astype(str).str.strip("'")
        df['label'] = df['label'].fillna('')
        df['post_type'] = df['post_type'].fillna('')
        df['object_group'] = df['object_group'].fillna('')
        df['specific_resource'] = df['specific_resource'].fillna('')
        all_labels = sorted(set([label.strip() for sub in df['label'] for label in sub.split(',') if label.strip()]))
        all_groups = sorted(set([g.strip() for g in df['object_group'].unique() if g.strip()]))
        all_platforms = sorted(set([p.strip() for p in df['specific_resource'].unique() if p.strip()]))
        sentiments_all = sorted(df['final_sentiment'].str.lower().unique())

        st.success("✅ Data berhasil dimuat. Silakan pilih filter di sidebar.")
    else:
        st.warning("Dataset ini tidak memiliki kolom yang sesuai dengan format sosial media.")
else:
    st.info("Silakan upload atau unduh ZIP untuk melihat ringkasan topik.")
