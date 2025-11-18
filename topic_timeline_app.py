# enhanced_topic_timeline_app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.colors import qualitative
from datetime import datetime, timedelta
import re
import networkx as nx
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from textblob import TextBlob
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------
# Page config with enhanced styling
# -------------------------
st.set_page_config(
    page_title="ORBIT- Online Risk Behaviour & Intelligence Tracker (ORBIT) Dashboard ",
    layout="wide",
    page_icon="🛡️",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
    }
    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        border-left: 5px solid #667eea;
        margin: 1rem 0;
    }
    .risk-high { border-left-color: #dc3545 !important; }
    .risk-medium { border-left-color: #fd7e14 !important; }
    .risk-low { border-left-color: #28a745 !important; }
    .cohort-section {
        background: #f8f9fa;
        padding: 2rem;
        border-radius: 12px;
        margin: 1rem 0;
        border: 1px solid #e9ecef;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    .stAlert {
        border-radius: 12px;
        border: none;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>🛡️ Advanced Cohort Intelligence & Risk Assessment Platform</h1>
    <p>Specialized analytics for behavioral patterns, language evolution, and risk identification</p>
    <p><em>Designed for counter-extremism, social cohort analysis, and threat assessment</em></p>
</div>
""", unsafe_allow_html=True)

# -------------------------
# Enhanced helper functions
# -------------------------
def guess_time_cols(cols):
    cands = [c for c in cols if re.search(r"time|date|timestamp|created|posted", c, re.I)]
    return cands or list(cols)[:1]

def guess_text_cols(cols):
    cands = [c for c in cols if re.search(r"comment|text|body|content|message|desc", c, re.I)]
    return cands or list(cols)[:1]

def guess_user_cols(cols):
    cands = [c for c in cols if re.search(r"user|author|unique_id|account|handle", c, re.I)]
    return cands or list(cols)[:1]

@st.cache_data(show_spinner=False)
def load_data(path_or_buf):
    return pd.read_csv(path_or_buf)

def ensure_datetime(series):
    s = series.copy()
    if pd.api.types.is_numeric_dtype(s):
        med = s.dropna().median()
        if med > 1e12:
            s = pd.to_datetime(s, unit="ms", errors="coerce")
        elif med > 1e10:
            s = pd.to_datetime(s, unit="ms", errors="coerce")
        else:
            s = pd.to_datetime(s, unit="s", errors="coerce")
    else:
        s = pd.to_datetime(s, errors="coerce")
    return s

def clean_text(s: pd.Series) -> pd.Series:
    """Enhanced text cleaning for extremism detection"""
    t = s.astype(str).str.lower()
    t = t.str.replace(r"http\S+|www\.\S+", " ", regex=True)  # URLs
    t = t.str.replace(r"[^a-z0-9\s']", " ", regex=True)      # Keep alnum + apostrophe
    t = t.str.replace(r"\s+", " ", regex=True).str.strip()
    return t

def calculate_risk_indicators(df, text_col, user_col=None):
    """Calculate various risk indicators for extremism detection"""
    risk_keywords = {
        'violence': ['violence', 'violent', 'attack', 'bomb', 'weapon', 'kill', 'murder', 'death', 'destroy'],
        'hate': ['hate', 'enemy', 'traitor', 'betray', 'revenge', 'punish', 'inferior', 'superior'],
        'us_vs_them': ['us', 'them', 'they', 'those people', 'outsider', 'invader', 'foreign', 'alien'],
        'conspiracy': ['conspiracy', 'plot', 'secret', 'hidden', 'cover', 'truth', 'lie', 'fake', 'control'],
        'urgency': ['now', 'must', 'time', 'urgent', 'action', 'act', 'do something', 'before', 'too late']
    }
    
    risk_scores = {}
    
    for category, keywords in risk_keywords.items():
        pattern = '|'.join(keywords)
        matches = df[text_col].str.contains(pattern, case=False, na=False)
        risk_scores[f'{category}_score'] = matches.sum() / len(df)
    
    # Calculate sentiment polarity
    if 'textblob' not in st.session_state:
        st.session_state.textblob = True
    
    df['sentiment'] = df[text_col].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
    risk_scores['negative_sentiment'] = (df['sentiment'] < -0.1).sum() / len(df)
    
    # User-level risk if user column available
    if user_col and user_col in df.columns:
        user_risk = df.groupby(user_col).agg({
            'sentiment': 'mean',
            text_col: 'count'
        }).rename(columns={text_col: 'post_count'})
        
        # Calculate activity intensity (posts per day)
        if '_time' in df.columns:
            date_range = (df['_time'].max() - df['_time'].min()).days + 1
            user_risk['posts_per_day'] = user_risk['post_count'] / date_range
        
        risk_scores['user_risks'] = user_risk
    
    return risk_scores

@st.cache_data(show_spinner=True)
def run_enhanced_lda(texts, n_topics, max_features, min_df, max_df, random_state):
    """Enhanced LDA with better preprocessing"""
    vectorizer = CountVectorizer(
        stop_words="english",
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        ngram_range=(1, 2)  # Include bigrams
    )
    X = vectorizer.fit_transform(texts)
    
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        learning_method="batch",
        random_state=random_state,
        max_iter=20
    )
    lda.fit(X)
    
    vocab = np.array(vectorizer.get_feature_names_out())
    doc_topic = lda.transform(X)
    
    return lda, vectorizer, vocab, doc_topic

def create_cohort_wordcloud(texts, max_words=100, colormap='plasma'):
    """Create enhanced word cloud for cohort analysis with error handling"""
    try:
        import matplotlib.pyplot as plt
        from wordcloud import WordCloud, STOPWORDS
        
        # Combine all text
        text_combined = ' '.join(texts.astype(str))
        
        # Check if we have any meaningful text
        if len(text_combined.strip()) == 0:
            return None
        
        # Custom stopwords for social media
        custom_stopwords = set(STOPWORDS).union({
            'user', 'video', 'comment', 'like', 'get', 'go', 'see', 'know', 
            'think', 'said', 'say', 'would', 'could', 'one', 'two', 'also',
            'really', 'much', 'way', 'even', 'make', 'made', 'take'
        })
        
        # Pre-process text to ensure we have content
        words_only = ' '.join([word for word in text_combined.split() 
                              if word.lower() not in custom_stopwords and len(word) > 2])
        
        # Final check - do we have any words left?
        if len(words_only.strip()) == 0:
            return None
        
        wordcloud = WordCloud(
            width=1200, height=600,
            max_words=max_words,
            stopwords=custom_stopwords,
            background_color='white',
            colormap=colormap,
            collocations=False,
            relative_scaling=0.5,
            min_word_length=2,
            max_font_size=100
        )
        
        # Generate with error handling
        try:
            wordcloud.generate(words_only)
        except ValueError as e:
            if "We need at least 1 word" in str(e):
                return None
            else:
                raise e
        
        fig, ax = plt.subplots(figsize=(15, 8))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        plt.tight_layout(pad=0)
        
        return fig
        
    except ImportError:
        st.error("WordCloud library not installed. Please install with: pip install wordcloud")
        return None
    except Exception as e:
        st.warning(f"Could not generate word cloud: {str(e)}")
        return None


def analyze_posting_patterns(df, time_col, user_col=None):
    """Analyze posting behavior patterns"""
    df['hour'] = df[time_col].dt.hour
    df['day_of_week'] = df[time_col].dt.day_name()
    df['date'] = df[time_col].dt.date
    
    patterns = {
        'hourly': df.groupby('hour').size(),
        'daily': df.groupby('day_of_week').size(),
        'temporal': df.groupby('date').size()
    }
    
    if user_col and user_col in df.columns:
        patterns['user_activity'] = df.groupby(user_col).agg({
            'hour': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.mean(),
            time_col: 'count'
        }).rename(columns={time_col: 'total_posts'})
    
    return patterns

def _safe_default_index(cols, candidate):
    cols_list = list(cols)
    try:
        idx = cols_list.index(candidate)
        return int(idx)
    except (ValueError, IndexError):
        return 0

def identify_influential_users(df, text_col, user_col, threshold_percentile=90):
    """Identify potentially influential users based on engagement and content"""
    if user_col not in df.columns:
        return pd.DataFrame()
    
    try:
        # Calculate text lengths first
        df_temp = df.copy()
        df_temp['_text_length'] = df_temp[text_col].astype(str).str.len()
        
        # Aggregate user statistics with proper numeric handling
        user_stats = df_temp.groupby(user_col).agg({
            text_col: 'count',                    # Count posts
            '_text_length': 'sum',                # Sum of character lengths
            'sentiment': ['mean', 'std']          # Sentiment stats
        })
        
        # Flatten column names
        user_stats.columns = ['post_count', 'total_chars', 'avg_sentiment', 'sentiment_variability']
        
        # Handle NaN values
        user_stats = user_stats.fillna(0)
        
        # Calculate influence score with safe numeric operations
        user_stats['influence_score'] = (
            user_stats['post_count'].astype(float) * 0.4 + 
            user_stats['total_chars'].astype(float) * 0.0001 +  # Scale down chars
            abs(user_stats['avg_sentiment'].astype(float)) * 0.3
        )
        
        # Get threshold and filter
        if len(user_stats) > 0:
            threshold = np.percentile(user_stats['influence_score'], threshold_percentile)
            influential = user_stats[user_stats['influence_score'] >= threshold].sort_values('influence_score', ascending=False)
            return influential
        else:
            return pd.DataFrame()
            
    except Exception as e:
        st.warning(f"User influence analysis failed: {str(e)}")
        return pd.DataFrame()


# -------------------------
# Enhanced sidebar controls
# -------------------------
st.sidebar.markdown("### 🔧 Data Configuration")

data_src = st.sidebar.radio("Data Source", ["Path to CSV", "Upload CSV"], index=0)

if data_src == "Upload CSV":
    file = st.sidebar.file_uploader("Upload CSV with comments", type=["csv"])
    if not file:
        st.stop()
    df_raw = load_data(file)
else:
    csv_path = st.sidebar.text_input("CSV path", value="full_call_no_null_columns.csv")
    if not csv_path:
        st.stop()
    try:
        df_raw = load_data(csv_path)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

# Enhanced column detection
time_guess_list = guess_time_cols(df_raw.columns)
text_guess_list = guess_text_cols(df_raw.columns)
user_guess_list = guess_user_cols(df_raw.columns)

time_default_idx = _safe_default_index(df_raw.columns, time_guess_list[0])
text_default_idx = _safe_default_index(df_raw.columns, text_guess_list[0])
user_default_idx = _safe_default_index(df_raw.columns, user_guess_list[0])

time_col = st.sidebar.selectbox("Timestamp column", options=df_raw.columns, index=time_default_idx)
text_col = st.sidebar.selectbox("Text column", options=df_raw.columns, index=text_default_idx)
user_col = st.sidebar.selectbox("User column (optional)", options=["None"] + list(df_raw.columns), index=user_default_idx+1)

if user_col == "None":
    user_col = None

st.sidebar.markdown("### 🎯 Analysis Parameters")

# Analysis mode selection
analysis_mode = st.sidebar.selectbox(
    "Analysis Focus",
    ["General Social Analysis", "Counter-Extremism Assessment", "Cohort Behavior Study", "Risk Identification"]
)

n_topics = st.sidebar.slider("Number of topics", 3, 20, 8, 1)
time_gran = st.sidebar.selectbox("Time granularity", ["D (day)", "W (week)", "M (month)"])
gran_map = {"D (day)": "D", "W (week)": "W", "M (month)": "M"}
gran = gran_map[time_gran]

max_features = st.sidebar.slider("Max vocab size", 2000, 20000, 10000, 500)
min_df = st.sidebar.slider("Vectorizer min_df", 1, 20, 3, 1)
max_df = st.sidebar.slider("Vectorizer max_df (as % of docs)", 50, 100, 85, 1) / 100.0

st.sidebar.markdown("### 🚨 Risk Assessment")
risk_threshold = st.sidebar.slider("Risk Alert Threshold", 0.1, 0.9, 0.3, 0.05)
enable_user_analysis = st.sidebar.checkbox("Enable User-Level Analysis", value=True if user_col else False)

# -------------------------
# Data preparation
# -------------------------
df = df_raw.copy()
df["_time"] = ensure_datetime(df[time_col])
df = df[~df["_time"].isna()].copy()
df["_text"] = clean_text(df[text_col])
df = df[df["_text"].str.len() > 0].copy()

if df.empty:
    st.error("No valid rows after parsing timestamp/text. Check your columns.")
    st.stop()

# Enhanced date filters
col_f1, col_f2 = st.columns(2)
with col_f1:
    min_date = st.date_input("Start date", value=pd.to_datetime(df["_time"].min()).date())
with col_f2:
    max_date = st.date_input("End date", value=pd.to_datetime(df["_time"].max()).date())

df = df[(df["_time"] >= pd.to_datetime(min_date)) & (df["_time"] <= pd.to_datetime(max_date))].copy()

if df.empty:
    st.warning("No rows in selected date range.")
    st.stop()

# Calculate risk indicators
with st.spinner("Analyzing risk indicators and behavioral patterns..."):
    risk_indicators = calculate_risk_indicators(df, "_text", user_col)
    posting_patterns = analyze_posting_patterns(df, "_time", user_col)

# -------------------------
# Enhanced KPI Dashboard
# -------------------------
st.subheader("🎯 Intelligence Dashboard")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("Total Posts", f"{len(df):,}")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    if user_col:
        unique_users = df[user_col].nunique()
        st.metric("Unique Users", f"{unique_users:,}")
    else:
        st.metric("Topics", f"{n_topics}")
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    risk_class = "risk-high" if risk_indicators.get('violence_score', 0) > risk_threshold else "risk-low"
    st.markdown(f'<div class="metric-card {risk_class}">', unsafe_allow_html=True)
    st.metric("Violence Risk", f"{risk_indicators.get('violence_score', 0):.3f}")
    st.markdown('</div>', unsafe_allow_html=True)

with col4:
    risk_class = "risk-high" if risk_indicators.get('hate_score', 0) > risk_threshold else "risk-low"
    st.markdown(f'<div class="metric-card {risk_class}">', unsafe_allow_html=True)
    st.metric("Hate Speech Risk", f"{risk_indicators.get('hate_score', 0):.3f}")
    st.markdown('</div>', unsafe_allow_html=True)

with col5:
    avg_sentiment = df['sentiment'].mean()
    risk_class = "risk-high" if avg_sentiment < -0.2 else "risk-medium" if avg_sentiment < 0 else "risk-low"
    st.markdown(f'<div class="metric-card {risk_class}">', unsafe_allow_html=True)
    st.metric("Avg Sentiment", f"{avg_sentiment:.3f}")
    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------
# Risk Assessment Dashboard
# -------------------------
if analysis_mode in ["Counter-Extremism Assessment", "Risk Identification"]:
    st.subheader("🚨 Comprehensive Risk Assessment")
    
    # Risk indicators radar chart
    risk_categories = ['violence_score', 'hate_score', 'us_vs_them_score', 'conspiracy_score', 'urgency_score']
    risk_values = [risk_indicators.get(cat, 0) for cat in risk_categories]
    risk_labels = ['Violence', 'Hate Speech', 'Us vs Them', 'Conspiracy', 'Urgency']
    
    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=risk_values + [risk_values[0]],  # Close the radar
        theta=risk_labels + [risk_labels[0]],
        fill='toself',
        name='Risk Profile',
        line=dict(color='red')
    ))
    
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, max(0.1, max(risk_values) * 1.2)])
        ),
        showlegend=True,
        title="Risk Assessment Radar Chart",
        height=500
    )
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.plotly_chart(fig_radar, use_container_width=True)
    
    with col2:
        st.markdown("### 🎯 Risk Interpretation")
        overall_risk = np.mean(risk_values)
        
        if overall_risk > 0.4:
            st.error("🚨 **HIGH RISK**: Multiple concerning indicators detected")
        elif overall_risk > 0.2:
            st.warning("⚠️ **MEDIUM RISK**: Some concerning patterns identified")
        else:
            st.success("✅ **LOW RISK**: Minimal concerning indicators")
        
        st.markdown("**Risk Factors:**")
        for label, value in zip(risk_labels, risk_values):
            if value > risk_threshold:
                st.markdown(f"- {label}: {value:.3f} ⚠️")
            else:
                st.markdown(f"- {label}: {value:.3f}")

# -------------------------
# Enhanced Topic Analysis
# -------------------------
if len(df) >= 10 and df["_text"].str.len().sum() >= 100:
    with st.spinner("Running enhanced topic modeling..."):
        lda, vectorizer, vocab, doc_topic = run_enhanced_lda(
            texts=df["_text"],
            n_topics=n_topics,
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            random_state=42
        )
    
    df["_topic_vec"] = list(doc_topic)
    df["_dom_topic"] = df["_topic_vec"].apply(lambda x: int(np.argmax(x)))
    df["_bin"] = df["_time"].dt.to_period(gran).dt.to_timestamp()
    
    # Enhanced topic labels
    topics_words = []
    for k, comp in enumerate(lda.components_):
        idx = np.argsort(comp)[-10:][::-1]
        topics_words.append((k, vocab[idx], comp[idx]))
    
    topic_labels = {k: f"T{k:02d}: " + ", ".join(words[:4]) for k, words, _wts in topics_words}
    
    # -------------------------
    # Topic Evolution Timeline
    # -------------------------
    st.subheader("📈 Topic Evolution & Behavioral Shifts")
    
    bin_topic = df.groupby("_bin")["_topic_vec"].apply(lambda rows: np.mean(np.vstack(rows.values), axis=0)).reset_index()
    
    topic_cols = [f"topic_{i}" for i in range(n_topics)]
    topic_matrix = np.vstack(bin_topic["_topic_vec"].values)
    topic_df = pd.DataFrame(topic_matrix, columns=topic_cols)
    topic_df.insert(0, "bin", bin_topic["_bin"].values)
    
    topic_long = topic_df.melt(id_vars="bin", var_name="topic", value_name="proportion")
    topic_long["label"] = (
        topic_long["topic"]
          .str.extract(r"(\d+)", expand=False)
          .astype(int)
          .map(topic_labels)
    )
    topic_long = topic_long.sort_values("bin")
    
    fig_area = go.Figure()
    colors = qualitative.Set3 + qualitative.Pastel
    
    for i in range(n_topics):
        sub = topic_long[topic_long["topic"] == f"topic_{i}"]
        fig_area.add_trace(go.Scatter(
            x=sub["bin"], y=sub["proportion"],
            mode="lines",
            name=topic_labels[i],
            stackgroup="one",
            line=dict(width=2),
            hovertemplate=f"{topic_labels[i]}<br>%{{x|%Y-%m-%d}}<br>Share=%{{y:.3f}}<extra></extra>",
        ))
    
    fig_area.update_layout(
        height=500,
        title="Topic Shift Timeline - Track Language Evolution & Behavioral Changes",
        legend_title="Topics (Key Terms)",
        xaxis_title="Time Period",
        yaxis_title="Topic Dominance",
        hovermode="x unified",
    )
    st.plotly_chart(fig_area, use_container_width=True)
    
    # -------------------------
    # Cohort Language Analysis
    # -------------------------
    st.markdown('<div class="cohort-section">', unsafe_allow_html=True)
    st.subheader("🗣️ Cohort Language Profiling")
    st.markdown("**Designed for briefing stakeholders and onboarding team members on target cohort communication patterns**")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("#### 📊 Communication Fingerprint")
        wordcloud_fig = create_cohort_wordcloud(df["_text"], max_words=150)
        if wordcloud_fig:
            st.pyplot(wordcloud_fig)
        else:
            st.warning("Insufficient text data for word cloud generation")
    
    with col2:
        st.markdown("#### 🎯 Key Language Insights")
        
        # Most common terms
        vectorizer_simple = CountVectorizer(stop_words='english', max_features=20)
        X_simple = vectorizer_simple.fit_transform(df["_text"])
        word_freq = dict(zip(vectorizer_simple.get_feature_names_out(), 
                           np.asarray(X_simple.sum(axis=0)).ravel()))
        
        top_terms = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
        
        st.markdown("**Most Frequent Terms:**")
        for term, freq in top_terms:
            st.markdown(f"- **{term}**: {freq} mentions")
        
        # Average metrics
        avg_length = df[text_col].astype(str).str.len().mean()
        avg_words = df["_text"].str.split().str.len().mean()
        
        st.markdown("**Communication Style:**")
        st.markdown(f"- Avg message length: {avg_length:.0f} chars")
        st.markdown(f"- Avg words per message: {avg_words:.1f}")
        st.markdown(f"- Sentiment tendency: {avg_sentiment:.2f}")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # -------------------------
    # Behavioral Pattern Analysis
    # -------------------------
    st.subheader("⏰ Behavioral Pattern Analysis")
    st.markdown("**Time-based posting behaviors for operational planning and target cohort matching**")
    
    fig_patterns = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Hourly Activity Pattern', 'Day of Week Pattern', 
                       'Activity Timeline', 'Sentiment Over Time'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Hourly pattern
    hourly_data = posting_patterns['hourly']
    fig_patterns.add_trace(
        go.Bar(x=hourly_data.index, y=hourly_data.values, name="Hourly Posts"),
        row=1, col=1
    )
    
    # Daily pattern
    daily_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    daily_data = posting_patterns['daily'].reindex(daily_order, fill_value=0)
    fig_patterns.add_trace(
        go.Bar(x=daily_data.index, y=daily_data.values, name="Daily Posts"),
        row=1, col=2
    )
    
    # Temporal timeline
    temporal_data = posting_patterns['temporal']
    fig_patterns.add_trace(
        go.Scatter(x=temporal_data.index, y=temporal_data.values, 
                  mode='lines+markers', name="Daily Activity"),
        row=2, col=1
    )
    
    # Sentiment timeline
    daily_sentiment = df.groupby(df["_time"].dt.date)['sentiment'].mean()
    fig_patterns.add_trace(
        go.Scatter(x=daily_sentiment.index, y=daily_sentiment.values,
                  mode='lines+markers', name="Daily Sentiment", line=dict(color='red')),
        row=2, col=2
    )
    
    fig_patterns.update_layout(height=600, showlegend=False)
    st.plotly_chart(fig_patterns, use_container_width=True)

# -------------------------
# User-Level Risk Analysis
# -------------------------
# -------------------------
# User-Level Risk Analysis
# -------------------------
if enable_user_analysis and user_col and 'user_risks' in risk_indicators:
    st.subheader("👥 Individual Risk Assessment & User Profiling")
    
    try:
        user_risks = risk_indicators['user_risks']
        
        # Identify high-risk users with safe operations
        if len(user_risks) > 0:
            # Safe quantile calculation
            high_activity_threshold = user_risks['post_count'].quantile(0.9) if 'post_count' in user_risks.columns else float('inf')
            
            # Filter users safely
            negative_sentiment = user_risks[user_risks['sentiment'] < -0.3] if 'sentiment' in user_risks.columns else pd.DataFrame()
            high_volume = user_risks[user_risks['post_count'] > high_activity_threshold] if 'post_count' in user_risks.columns else pd.DataFrame()
            
            st.markdown("#### 🎯 Users of Interest")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**High Volume Posters:**")
                if len(high_volume) > 0:
                    st.dataframe(high_volume.head(10))
                else:
                    st.info("No high-volume users identified")
            
            with col2:
                st.markdown("**Negative Sentiment Users:**")
                if len(negative_sentiment) > 0:
                    st.dataframe(negative_sentiment.head(10))
                else:
                    st.info("No users with consistently negative sentiment")
            
            # User influence analysis with safe error handling
            if len(df) > 50:  # Only if sufficient data
                try:
                    influential_users = identify_influential_users(df, "_text", user_col)
                    
                    if len(influential_users) > 0:
                        st.markdown("#### 🌟 Influential Users (Node Centrality Analysis)")
                        
                        # Create scatter plot with safe data handling
                        plot_data = influential_users.reset_index()
                        
                        fig_influence = px.scatter(
                            plot_data,
                            x='post_count',
                            y='influence_score',
                            size='total_chars',
                            color='avg_sentiment',
                            hover_name=user_col,
                            title="User Influence vs Activity Pattern",
                            color_continuous_scale='RdYlBu_r'
                        )
                        
                        st.plotly_chart(fig_influence, use_container_width=True)
                        st.dataframe(influential_users.head(15))
                    else:
                        st.info("No influential users identified with current thresholds")
                        
                except Exception as e:
                    st.warning(f"Influential user analysis failed: {str(e)}")
        else:
            st.info("No user risk data available for analysis")
            
    except Exception as e:
        st.error(f"User analysis section failed: {str(e)}")
        st.info("This may be due to data quality issues or missing required columns")


# -------------------------
# Export and Download Section
# -------------------------
st.subheader("📥 Intelligence Reports & Data Export")

col1, col2, col3 = st.columns(3)

with col1:
    if 'topic_df' in locals():
        st.download_button(
            "📊 Download Topic Timeline",
            data=topic_df.to_csv().encode("utf-8"),
            file_name=f"topic_timeline_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
        )

with col2:
    # Risk assessment report
    risk_report = pd.DataFrame([risk_indicators]).T
    risk_report.columns = ['Risk_Score']
    st.download_button(
        "🚨 Download Risk Report",
        data=risk_report.to_csv().encode("utf-8"),
        file_name=f"risk_assessment_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
    )

with col3:
    if enable_user_analysis and 'user_risks' in risk_indicators:
        st.download_button(
            "👥 Download User Analysis",
            data=risk_indicators['user_risks'].to_csv().encode("utf-8"),
            file_name=f"user_analysis_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
        )

# -------------------------
# Footer with guidance
# -------------------------
st.markdown("---")
st.markdown("""
<div style='background: #f8f9fa; padding: 20px; border-radius: 10px; margin-top: 20px;'>
<h4>🎯 Platform Guidance</h4>
<p><strong>For Counter-Extremism Operations:</strong></p>
<ul>
<li>Monitor risk indicators trending upward over time</li>
<li>Identify users with high influence scores and negative sentiment patterns</li>
<li>Track language evolution toward more extreme terminology</li>
<li>Use behavioral patterns to predict optimal intervention timing</li>
</ul>

<p><strong>For Cohort Analysis & Stakeholder Briefing:</strong></p>
<ul>
<li>Use word clouds and language insights for rapid cohort understanding</li>
<li>Behavioral patterns help match posting schedules and communication styles</li>
<li>Topic evolution tracks narrative shifts within target communities</li>
<li>Influence analysis identifies key voices within cohorts</li>
</ul>
</div>
""", unsafe_allow_html=True)

st.caption("🛡️ Advanced Cohort Intelligence Platform | Specialized for law enforcement, security analysis, and social research")
