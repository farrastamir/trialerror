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
        for col in ['content', 'final_sentiment']:
            df[col] = df[col].astype(str).str.strip("'")
        df['label'] = df['label'].fillna('')
        df['post_type'] = df['post_type'].fillna('')
        df['object_group'] = df['object_group'].fillna('')
        df['specific_resource'] = df['specific_resource'].fillna('')
        df['url'] = df['url'].fillna('-')

        all_labels = sorted(set([label.strip() for sub in df['label'] for label in sub.split(',') if label.strip()]))
        sentiments_all = sorted(df['final_sentiment'].str.lower().unique())

        # Filter Sidebar
        with st.sidebar:
            if st.button("🧹 Clear Filter"):
                st.session_state['sentiment_filter'] = "All"
                st.session_state['label_filter'] = "All"
                st.session_state['keyword_input'] = ""
                st.session_state['highlight_words'] = ""

            sentiment_filter = st.selectbox("Sentimen", options=["All"] + sentiments_all, index=([
                "All"] + sentiments_all).index(st.session_state['sentiment_filter']))
            st.session_state['sentiment_filter'] = sentiment_filter

            label_filter = st.selectbox("Label", options=["All"] + all_labels, index=([
                "All"] + all_labels).index(st.session_state['label_filter']))
            st.session_state['label_filter'] = label_filter

            keyword_input = st.text_input("Kata kunci (\"frasa\" -exclude)", value=st.session_state['keyword_input'])
            st.session_state['keyword_input'] = keyword_input

            highlight_words = st.text_input("Highlight Kata", value=st.session_state['highlight_words'])
            st.session_state['highlight_words'] = highlight_words

            st.session_state['show_wordcloud'] = st.checkbox("Tampilkan WordCloud",
                                                             value=st.session_state['show_wordcloud'])
            if st.session_state['show_wordcloud']:
                st.session_state['dynamic_wordcloud'] = st.checkbox("Word Cloud Dinamis",
                                                                   value=st.session_state['dynamic_wordcloud'])

        filtered_df = df.copy()
        if sentiment_filter != 'All':
            filtered_df = filtered_df[filtered_df['final_sentiment'].str.lower() == sentiment_filter]
        if label_filter != 'All':
            filtered_df = filtered_df[filtered_df['label'].apply(lambda x: label_filter in [s.strip() for s in x.split(',')])]

        def parse_advanced_keywords(query):
            query = query.strip()
            if not query:
                return [], [], []
            include_groups, exclude_words, exact_phrases = [], [], []
            token_pattern = r'\"[^\"]+\"|\([^\)]+\)|\S+'
            tokens = re.findall(token_pattern, query)
            for tok in tokens:
                if tok.startswith('"') and tok.endswith('"'):
                    exact_phrases.append(tok.strip('"'))
                elif tok.startswith('-'):
                    inner = tok[1:].strip()
                    exclude_words.extend(inner.strip('()').split())
                elif tok.startswith('(') and tok.endswith(')'):
                    or_group = [w.strip() for w in tok.strip('()').split('OR') if w.strip()]
                    include_groups.append(or_group)
                else:
                    include_groups.append([tok.strip()])
            return include_groups, exact_phrases, exclude_words

        def match_advanced(text, includes, phrases, excludes):
            text = text.lower()
            if any(word in text for word in excludes):
                return False
            for phrase in phrases:
                if phrase.lower() not in text:
                    return False
            for group in includes:
                if not any(word.lower() in text for word in group):
                    return False
            return True

        includes, phrases, excludes = parse_advanced_keywords(keyword_input)
        if keyword_input:
            mask = filtered_df['content'].apply(lambda x: match_advanced(x, includes, phrases, excludes))
            filtered_df = filtered_df[mask]

        highlight_tokens = re.findall(r'\"[^\"]+\"|\S+', highlight_words)
        highlight_words_set = set([h.strip('"').lower() for h in highlight_tokens])

        def highlight_text(text):
            for word in highlight_words_set:
                text = re.sub(f"(?i)({re.escape(word)})", r'<mark>\1</mark>', text)
            return text

        # Sample link ambil dari salah satu baris untuk setiap content
        grouped = filtered_df.groupby('content').agg(
            Article=('content', 'count'),
            Sentiment=('final_sentiment', lambda x: x.mode().iloc[0] if not x.mode().empty else '-'),
            Link=('url', lambda x: x.dropna().iloc[0] if not x.dropna().empty else '-')
        ).reset_index().sort_values(by='Article', ascending=False)

        def sentiment_color(sent):
            s = sent.lower()
            if s == 'positive': return f'<span style="color:green;font-weight:bold">{s}</span>'
            if s == 'negative': return f'<span style="color:red;font-weight:bold">{s}</span>'
            if s == 'neutral': return f'<span style="color:gray;font-weight:bold">{s}</span>'
            return sent

        grouped['content'] = grouped['content'].apply(highlight_text)
        grouped['Sentiment'] = grouped['Sentiment'].apply(sentiment_color)
        grouped['Link'] = grouped['Link'].apply(lambda x: f'<a href="{x}" target="_blank">Link</a>' if x != '-' else '-')

        st.markdown("### 📊 Ringkasan Percakapan")
        st.markdown("<style>table { table-layout: fixed; width: 100%; word-wrap: break-word; }</style>", unsafe_allow_html=True)
        st.markdown(grouped[['content', 'Article', 'Sentiment', 'Link']].to_html(escape=False, index=False), unsafe_allow_html=True)

        if st.session_state['show_wordcloud']:
            st.markdown("### ☁️ Word Cloud (Top 500)")
            base_df = filtered_df if st.session_state['dynamic_wordcloud'] else df
            all_text = ' '.join(base_df['content'].tolist())
            tokens = re.findall(r'\b\w{3,}\b', all_text.lower())
            stop_url = "https://raw.githubusercontent.com/stopwords-iso/stopwords-id/master/stopwords-id.txt"
            common_stopwords = set(pd.read_csv(stop_url, header=None)[0].tolist())
            tokens = [word for word in tokens if word not in common_stopwords]
            word_freq = Counter(tokens).most_common(500)
            wc_df = pd.DataFrame(word_freq, columns=['Kata', 'Jumlah'])
            st.dataframe(wc_df, use_container_width=True)
else:
    st.info("Silakan upload atau unduh ZIP untuk melihat ringkasan topik.")
