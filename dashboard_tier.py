import streamlit as st
import pandas as pd
import re
from collections import Counter

def run_tier_dashboard(df):
    st.set_page_config(layout="wide")
    st.title("📰 Topic Summary NoLimit Dashboard")

    if 'show_wordcloud' not in st.session_state:
        st.session_state['show_wordcloud'] = False
    if 'dynamic_wordcloud' not in st.session_state:
        st.session_state['dynamic_wordcloud'] = True
    if 'reset_filter' not in st.session_state:
        st.session_state['sentiment_filter'] = "All"
        st.session_state['label_filter'] = "All"
        st.session_state['keyword_input'] = ""
        st.session_state['highlight_words'] = ""

    for col in ['title', 'body', 'url', 'sentiment']:
        df[col] = df[col].astype(str).str.strip("'")

    df['label'] = df['label'].fillna('')
    df['tier'] = df['tier'].fillna('-')
    df['tier'] = pd.Categorical(df['tier'], categories=['Tier 1', 'Tier 2', 'Tier 3', '-', ''], ordered=True)
    all_labels = sorted(set([label.strip() for sub in df['label'] for label in sub.split(',') if label.strip()]))
    sentiments_all = sorted(df['sentiment'].str.lower().unique())

    if st.sidebar.button("🔄 Clear Filter"):
        st.session_state['sentiment_filter'] = "All"
        st.session_state['label_filter'] = "All"
        st.session_state['keyword_input'] = ""
        st.session_state['highlight_words'] = ""

    sentiment_filter = st.sidebar.selectbox("Sentimen", options=["All"] + sentiments_all, index=(["All"] + sentiments_all).index(st.session_state['sentiment_filter']))
    st.session_state['sentiment_filter'] = sentiment_filter

    label_filter = st.sidebar.selectbox("Label", options=["All"] + all_labels, index=(["All"] + all_labels).index(st.session_state['label_filter']))
    st.session_state['label_filter'] = label_filter

    keyword_input = st.sidebar.text_input("Kata kunci (\"frasa\" -exclude)", value=st.session_state['keyword_input'])
    st.session_state['keyword_input'] = keyword_input

    highlight_words = st.sidebar.text_input("Highlight Kata", value=st.session_state['highlight_words'])
    st.session_state['highlight_words'] = highlight_words

    st.session_state['show_wordcloud'] = st.sidebar.checkbox("Tampilkan WordCloud", value=st.session_state['show_wordcloud'])
    if st.session_state['show_wordcloud']:
        st.session_state['dynamic_wordcloud'] = st.sidebar.checkbox("Word Cloud Dinamis", value=st.session_state['dynamic_wordcloud'])

    filtered_df = df.copy()
    if sentiment_filter != 'All':
        filtered_df = filtered_df[filtered_df['sentiment'].str.lower() == sentiment_filter]
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
        mask = filtered_df['title'].apply(lambda x: match_advanced(x, includes, phrases, excludes)) | \
               filtered_df['body'].apply(lambda x: match_advanced(x, includes, phrases, excludes))
        filtered_df = filtered_df[mask]

    highlight_tokens = re.findall(r'\"[^\"]+\"|\S+', highlight_words)
    highlight_words_set = set([h.strip('"').lower() for h in highlight_tokens])

    def highlight_text(text):
        for word in highlight_words_set:
            text = re.sub(f"(?i)({re.escape(word)})", r'<mark>\1</mark>', text)
        return text

    def get_best_link(sub_df):
        for tier in ['Tier 1', 'Tier 2', 'Tier 3', '-', '']:
            result = sub_df[sub_df['tier'] == tier]['url']
            if not result.empty:
                return result.iloc[0]
        return '-'

    st.sidebar.markdown("### 📊 Statistik")
    sentiments = filtered_df['sentiment'].str.lower()
    st.sidebar.markdown(f"<div style='font-size:18px; font-weight:bold;'>📰 Total Artikel: {filtered_df.shape[0]}</div>", unsafe_allow_html=True)
    st.sidebar.markdown(f"""
        <div style='margin-top:4px;'>
            <span style='color:green;'>🟢 {(sentiments == 'positive').sum()}</span> |
            <span style='color:gray;'>⚪ {(sentiments == 'neutral').sum()}</span> |
            <span style='color:red;'>🔴 {(sentiments == 'negative').sum()}</span>
        </div>
    """, unsafe_allow_html=True)

    grouped = filtered_df.groupby('title').agg(
        Article=('title', 'count'),
        Sentiment=('sentiment', lambda x: x.mode().iloc[0] if not x.mode().empty else '-'),
        Link=('title', lambda x: get_best_link(filtered_df[filtered_df['title'] == x.iloc[0]]))
    ).reset_index().sort_values(by='Article', ascending=False)

    def sentiment_color(sent):
        s = sent.lower()
        if s == 'positive': return f'<span style="color:green;font-weight:bold">{s}</span>'
        if s == 'negative': return f'<span style="color:red;font-weight:bold">{s}</span>'
        if s == 'neutral': return f'<span style="color:gray;font-weight:bold">{s}</span>'
        return sent

    grouped['title'] = grouped['title'].apply(highlight_text)
    grouped['Sentiment'] = grouped['Sentiment'].apply(sentiment_color)
    grouped['Link'] = grouped['Link'].apply(lambda x: f'<a href="{x}" target="_blank">Link</a>' if x != '-' else '-')

    col1, col2 = st.columns([0.7, 0.3])
    with col1:
        st.markdown("### 📊 Ringkasan Topik")
        st.markdown("<div style='overflow-x:auto;'>", unsafe_allow_html=True)
        st.write(grouped[['title', 'Article', 'Sentiment', 'Link']].to_html(escape=False, index=False), unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        if st.session_state['show_wordcloud']:
            st.markdown("### ☁️ Word Cloud (Top 500)")
            base_df = filtered_df if st.session_state['dynamic_wordcloud'] else df
            all_text = ' '.join(base_df['title'].tolist() + base_df['body'].tolist())
            tokens = re.findall(r'\b\w{3,}\b', all_text.lower())
            stop_url = "https://raw.githubusercontent.com/stopwords-iso/stopwords-id/master/stopwords-id.txt"
            common_stopwords = set(pd.read_csv(stop_url, header=None)[0].tolist())
            tokens = [word for word in tokens if word not in common_stopwords]
            word_freq = Counter(tokens).most_common(500)
            wc_df = pd.DataFrame(word_freq, columns=['Kata', 'Jumlah'])
            st.dataframe(wc_df, use_container_width=True)
