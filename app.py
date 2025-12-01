import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import re
import networkx as nx
from datetime import datetime, timedelta
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# -----------------------------------------------------------------------------
# 1. Page configuration & styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title='Social Toxicity Analytics Platform',
    layout='wide',
    initial_sidebar_state='expanded',
    page_icon='📊'
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    .stAlert {
        border-radius: 8px;
    }
    .section-header {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 4px solid #667eea;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>🛡️ Social Toxicity Analytics Platform</h1>
    <p>Advanced insights into online discourse patterns and toxicity trends</p>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. Enhanced sidebar controls
# -----------------------------------------------------------------------------
st.sidebar.markdown("### 📁 Data Configuration")
DATA_PATH = st.sidebar.text_input(
    'CSV data path',
    value='full_call_no_null_columns.csv',
    help='Path to your consolidated CSV file'
)

st.sidebar.markdown("### 📅 Time Range Filters")
col1, col2 = st.sidebar.columns(2)
with col1:
    min_date = st.date_input('Start Date', None)
with col2:
    max_date = st.date_input('End Date', None)

st.sidebar.markdown("### 🎯 Analysis Parameters")
tox_threshold = st.sidebar.slider(
    'Toxicity Threshold',
    0.0, 1.0, 0.5, 0.01,
    help='Minimum toxicity score for filtering'
)

min_comments = st.sidebar.slider(
    'Minimum Comments per Video',
    1, 100, 5,
    help='Filter videos with fewer comments'
)

st.sidebar.markdown("### 🧭 Navigation")
page = st.sidebar.radio(
    'Select Analysis View',
    ['📊 Executive Dashboard', '🔍 Toxicity Deep Dive', '📈 Trend Analysis', 
     '⏱️ User Timeline', '🗝️ Content Analysis', '🌐 Network Insights']
)

# -----------------------------------------------------------------------------
# 3. Enhanced data loading & preprocessing
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner='Loading and processing data...')
def load_and_prepare(path):
    try:
        df = pd.read_csv(path)
        
        # Parse timestamps with better error handling
        if 'create_time_x' in df.columns:
            df['comment_time'] = pd.to_datetime(
                df['create_time_x'], unit='s', errors='coerce'
            )
        if 'createTime' in df.columns:
            df['video_time'] = pd.to_datetime(
                df['createTime'], unit='s', errors='coerce'
            )
        
        # Handle comment text
        if 'desc_x' in df.columns and df['desc_x'].notna().sum() > 0:
            df['comment'] = df['desc_x'].astype(str)
        elif 'text_x' in df.columns:
            df['comment'] = df['text_x'].astype(str)
        else:
            df['comment'] = ''
        
        # Create comprehensive toxicity flags
        tox_cols = ['toxicity_label', 'severe_toxicity_label', 'obscene_label',
                   'threat_label', 'insult_label', 'identity_attack_label']
        present_cols = [c for c in tox_cols if c in df.columns]
        
        if 'any_toxic' not in df.columns and present_cols:
            df['any_toxic'] = df[present_cols].any(axis=1).astype(int)
        elif 'any_toxic' not in df.columns:
            df['any_toxic'] = 0
        
        # Add derived features
        df['comment_length'] = df['comment'].str.len()
        df['word_count'] = df['comment'].str.split().str.len()
        df['hour'] = df['comment_time'].dt.hour
        df['day_of_week'] = df['comment_time'].dt.day_name()
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()

# Load data
df = load_and_prepare(DATA_PATH)

if df.empty:
    st.error("No data loaded. Please check your file path.")
    st.stop()

# Apply filters
if min_date:
    df = df[df['comment_time'] >= pd.to_datetime(min_date)]
if max_date:
    df = df[df['comment_time'] <= pd.to_datetime(max_date)]

# -----------------------------------------------------------------------------
# 4. Enhanced helper functions
# -----------------------------------------------------------------------------
def create_metric_card(title, value, delta=None, delta_color="normal"):
    """Create a styled metric display"""
    return st.metric(
        label=title,
        value=value,
        delta=delta,
        delta_color=delta_color
    )

def advanced_word_cloud(text_series, max_words=100, colormap='viridis'):
    """Generate an advanced word cloud"""
    stop_words = STOPWORDS.union({'user', 'video', 'comment', 'like', 'get'})
    text = ' '.join(text_series.astype(str).tolist())
    
    if len(text.strip()) == 0:
        st.warning("No text data available for word cloud")
        return
    
    wc = WordCloud(
        width=1200, 
        height=600,
        stopwords=stop_words,
        max_words=max_words,
        background_color='white',
        colormap=colormap,
        collocations=False
    ).generate(text)
    
    fig, ax = plt.subplots(figsize=(15, 8))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)

def sentiment_analysis(text_series, n_topics=5):
    """Perform topic modeling with LDA"""
    if len(text_series) == 0:
        return []
    
    # Clean text
    text_clean = text_series.astype(str).str.lower()
    text_clean = text_clean.str.replace(r'[^\w\s]', '', regex=True)
    
    try:
        vectorizer = CountVectorizer(
            max_features=1000,
            stop_words='english',
            min_df=2,
            max_df=0.95
        )
        X = vectorizer.fit_transform(text_clean)
        
        lda = LatentDirichletAllocation(
            n_components=n_topics,
            random_state=42,
            max_iter=10
        )
        lda.fit(X)
        
        vocab = vectorizer.get_feature_names_out()
        topics = []
        
        for i, comp in enumerate(lda.components_):
            top_words_idx = comp.argsort()[-10:][::-1]
            top_words = [vocab[idx] for idx in top_words_idx]
            topics.append({
                'topic': i+1,
                'words': ', '.join(top_words),
                'weight': comp[top_words_idx[0]]
            })
        
        return topics
    except Exception as e:
        st.error(f"Topic analysis failed: {str(e)}")
        return []

# -----------------------------------------------------------------------------
# 5. Page implementations
# -----------------------------------------------------------------------------

if page == '📊 Executive Dashboard':
    st.markdown('<div class="section-header"><h2>📊 Executive Summary</h2></div>', unsafe_allow_html=True)
    
    # Key Performance Indicators
    col1, col2, col3, col4, col5 = st.columns(5)
    
    total_comments = len(df)
    unique_commenters = df['unique_id_x'].nunique() if 'unique_id_x' in df.columns else 0
    unique_videos = df['aweme_id_x'].nunique() if 'aweme_id_x' in df.columns else 0
    toxic_rate = df['any_toxic'].mean() * 100 if 'any_toxic' in df.columns else 0
    avg_engagement = df.groupby('aweme_id_x').size().mean() if 'aweme_id_x' in df.columns else 0
    
    with col1:
        create_metric_card("Total Comments", f"{total_comments:,}")
    with col2:
        create_metric_card("Unique Users", f"{unique_commenters:,}")
    with col3:
        create_metric_card("Videos Analyzed", f"{unique_videos:,}")
    with col4:
        create_metric_card("Toxicity Rate", f"{toxic_rate:.1f}%")
    with col5:
        create_metric_card("Avg Engagement", f"{avg_engagement:.1f}")
    
    # Dashboard charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Toxicity Distribution")
        if 'toxicity' in df.columns:
            fig = px.histogram(
                df, 
                x='toxicity',
                nbins=50,
                title='Toxicity Score Distribution',
                color_discrete_sequence=['#667eea']
            )
            fig.update_layout(
                showlegend=False,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📊 Toxicity by Type")
        if 'toxicity' in df.columns:
            tox_types = ['severe_toxicity', 'obscene', 'threat', 'insult', 'identity_attack']
            available_types = [t for t in tox_types if t in df.columns]
            
            if available_types:
                type_means = df[available_types].mean()
                fig = px.bar(
                    x=available_types,
                    y=type_means.values,
                    title='Average Toxicity by Type',
                    color=type_means.values,
                    color_continuous_scale='Reds'
                )
                fig.update_layout(
                    showlegend=False,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # Time series analysis
    st.subheader("📅 Activity Timeline")
    if 'comment_time' in df.columns:
        df['date'] = df['comment_time'].dt.date
        daily_stats = df.groupby('date').agg({
            'comment': 'count',
            'toxicity': 'mean' if 'toxicity' in df.columns else lambda x: 0,
            'any_toxic': 'sum' if 'any_toxic' in df.columns else lambda x: 0
        }).reset_index()
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Daily Comment Volume', 'Daily Toxicity Rate'),
            vertical_spacing=0.1
        )
        
        fig.add_trace(
            go.Scatter(
                x=daily_stats['date'],
                y=daily_stats['comment'],
                mode='lines+markers',
                name='Comments',
                line=dict(color='#667eea', width=2)
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=daily_stats['date'],
                y=daily_stats['toxicity'],
                mode='lines+markers',
                name='Avg Toxicity',
                line=dict(color='#e74c3c', width=2)
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            height=500,
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Top toxic content
    st.subheader("🚨 High-Risk Content")
    if 'toxicity' in df.columns:
        toxic_content = df.nlargest(10, 'toxicity')[
            ['comment_time', 'comment', 'toxicity', 'unique_id_x']
        ]
        st.dataframe(toxic_content, use_container_width=True)

elif page == '🔍 Toxicity Deep Dive':
    st.markdown('<div class="section-header"><h2>🔍 Advanced Toxicity Analysis</h2></div>', unsafe_allow_html=True)
    
    # Correlation analysis
    st.subheader("📊 Toxicity Correlations")
    if 'toxicity' in df.columns:
        tox_cols = ['toxicity', 'severe_toxicity', 'obscene', 'threat', 'insult', 'identity_attack']
        available_cols = [c for c in tox_cols if c in df.columns]
        
        if len(available_cols) > 1:
            corr_matrix = df[available_cols].corr()
            
            fig = px.imshow(
                corr_matrix,
                text_auto=True,
                aspect="auto",
                color_continuous_scale='RdBu',
                title="Toxicity Metrics Correlation Matrix"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Scatter plot analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Comment Length vs Toxicity")
        if 'toxicity' in df.columns and 'comment_length' in df.columns:
            fig = px.scatter(
                df.sample(min(1000, len(df))),
                x='comment_length',
                y='toxicity',
                opacity=0.6,
                trendline="ols",
                title="Comment Length vs Toxicity Score"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🕐 Toxicity by Hour")
        if 'toxicity' in df.columns and 'hour' in df.columns:
            hourly_tox = df.groupby('hour')['toxicity'].mean().reset_index()
            fig = px.bar(
                hourly_tox,
                x='hour',
                y='toxicity',
                title="Average Toxicity by Hour of Day"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # User behavior analysis
    st.subheader("👥 User Behavior Patterns")
    if 'unique_id_x' in df.columns and 'toxicity' in df.columns:
        user_stats = df.groupby('unique_id_x').agg({
            'comment': 'count',
            'toxicity': 'mean',
            'any_toxic': 'sum'
        }).reset_index()
        user_stats.columns = ['user_id', 'total_comments', 'avg_toxicity', 'toxic_comments']
        user_stats['toxicity_rate'] = user_stats['toxic_comments'] / user_stats['total_comments']
        
        fig = px.scatter(
            user_stats[user_stats['total_comments'] >= 5],
            x='total_comments',
            y='avg_toxicity',
            size='toxic_comments',
            hover_data=['toxicity_rate'],
            title="User Activity vs Average Toxicity",
            labels={'total_comments': 'Total Comments', 'avg_toxicity': 'Average Toxicity'}
        )
        st.plotly_chart(fig, use_container_width=True)

elif page == '⏱️ User Timeline':
    st.markdown('<div class="section-header"><h2>⏱️ Interactive User Timeline</h2></div>', unsafe_allow_html=True)
    
    # Tab selection for different timeline views
    tab1, tab2 = st.tabs(["📊 Original Timeline Dashboard", "🔍 Multi-User Timeline"])
    
    with tab1:
        # Your original timeline code implementation
        st.subheader("🎬 Video Timeline & Commenters Dashboard")
        
        # — Compute total comments per video
        comment_counts = (
            df.groupby("aweme_id_x")
              .size()
              .rename("total_comments")
              .reset_index()
        )

        # — Build a videos DataFrame with one row per video
        videos = (
            df[["aweme_id_x", "uniqueId", "createTime"]]
            .drop_duplicates("aweme_id_x")
            .merge(comment_counts, on="aweme_id_x")
        )

        # — Sidebar controls for original timeline
        col1, col2 = st.columns(2)
        
        with col1:
            scale = st.slider("Bubble size scale", 0.1, 5.0, 0.5, 0.1, key="bubble_scale")
        
        with col2:
            # — Power Commenters filters
            st.write("**Power Commenters Filter**")
            min_comments = st.number_input("Min total comments", min_value=1, value=2, step=1, key="min_comments")
            min_authors = st.number_input("Min distinct authors", min_value=1, value=2, step=1, key="min_authors")

        # — Compute commenter stats
        stats = (
            df.groupby("unique_id_x")
              .agg(
                  total_comments   = ("aweme_id_x", "count"),
                  distinct_authors = ("uniqueId", "nunique"),
              )
              .reset_index()
        )
        power_users = stats[
            (stats.total_comments >= min_comments) &
            (stats.distinct_authors >= min_authors)
        ].sort_values(["total_comments","distinct_authors"], ascending=False)

        st.subheader("Power Commenters")
        st.write(
            f"Commenters with ≥ {min_comments} comments "
            f"across ≥ {min_authors} distinct authors:"
        )
        st.dataframe(power_users, use_container_width=True)

        # — Identify the main video authors
        video_authors = sorted(videos["uniqueId"].unique())

        # — Commenter selection
        all_commenters = sorted(df["unique_id_x"].unique())
        selected_commenter = st.selectbox(
            "Highlight a commentator on the timeline (or 'None')",
            ["None"] + all_commenters,
            key="selected_commenter"
        )

        # — Build the y-axis categories
        y_categories = video_authors.copy()
        if selected_commenter != "None" and selected_commenter not in y_categories:
            y_categories.append(selected_commenter)
        user_pos = {uid: idx for idx, uid in enumerate(y_categories)}

        # — Create the Plotly figure
        fig = go.Figure()

        # 1) Plot video bubbles sized by total_comments - WITH ERROR FIX
        for author in video_authors:
            vids = videos[videos["uniqueId"] == author]
            
            # Create text with proper timestamp handling
            text_labels = []
            for ts, cnt in zip(vids["createTime"], vids["total_comments"]):
                try:
                    # Check if timestamp is valid
                    if pd.isna(ts) or ts is pd.NaT:
                        formatted_time = "Invalid Date"
                    else:
                        formatted_time = ts.strftime('%Y-%m-%d %H:%M')
                    
                    text_labels.append(
                        f"{author}<br>{formatted_time}<br>"
                        f"Comments: {cnt}"
                    )
                except (AttributeError, ValueError):
                    # Handle any other timestamp formatting errors
                    text_labels.append(
                        f"{author}<br>Invalid Date<br>"
                        f"Comments: {cnt}"
                    )
            
            fig.add_trace(go.Scatter(
                x=vids["createTime"],
                y=[user_pos[author]] * len(vids),
                mode="markers",
                marker=dict(
                    size=vids["total_comments"] * scale,
                    opacity=0.7
                ),
                name=f"{author} (videos)",
                text=text_labels,
                hoverinfo="text",
            ))

        # 2) Draw internal edges (author↔author comments)
        internal = df[
            df["unique_id_x"].isin(video_authors) &
            df["uniqueId"].isin(video_authors) &
            (df["unique_id_x"] != df["uniqueId"])
        ]
        for _, row in internal.iterrows():
            t  = row["create_time_x"]
            y0 = user_pos[row["unique_id_x"]]
            y1 = user_pos[row["uniqueId"]]
            fig.add_trace(go.Scatter(
                x=[t, t],
                y=[y0, y1],
                mode="lines",
                line=dict(width=1, dash="dot", color="gray"),
                hoverinfo="none",
                showlegend=False,
            ))

        # 3) Draw edges for the selected external commentator
        if selected_commenter != "None":
            comments = df[df["unique_id_x"] == selected_commenter]
            for _, row in comments.iterrows():
                t  = row["create_time_x"]
                y0 = user_pos[selected_commenter]
                y1 = user_pos[row["uniqueId"]]
                fig.add_trace(go.Scatter(
                    x=[t, t],
                    y=[y0, y1],
                    mode="lines",
                    line=dict(width=1, dash="solid", color="blue"),
                    hoverinfo="none",
                    showlegend=False,
                ))

        # 4) Final layout tweaks
        fig.update_layout(
            title="Video Authors Timeline with Comment‐Count‐Sized Bubbles",
            xaxis_title="Time",
            yaxis=dict(
                tickmode="array",
                tickvals=list(user_pos.values()),
                ticktext=list(user_pos.keys()),
                title="Account",
            ),
            height=600,
            margin=dict(l=50, r=50, t=60, b=50),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )

        # Render in Streamlit
        st.plotly_chart(fig, use_container_width=True)

        # — Display the highlighted commentator's comments & toxicity
        if selected_commenter != "None":
            st.subheader(f"Comments by {selected_commenter}")
            commenter_df = df[df["unique_id_x"] == selected_commenter][
                ["create_time_x", "text_x", "toxicity"]
            ].sort_values("create_time_x")
            commenter_df = commenter_df.rename(columns={
                "create_time_x": "Timestamp",
                "text_x": "Comment",
                "toxicity": "Toxicity"
            })
            st.dataframe(commenter_df, use_container_width=True)
    
    with tab2:
        # Multi-User Timeline from the comprehensive version
        st.subheader("👤 Multi-User Selection")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if 'unique_id_x' in df.columns:
                # Get user activity stats
                user_activity = df.groupby('unique_id_x').agg({
                    'text_x': 'count',  # Using text_x instead of 'comment'
                    'aweme_id_x': 'nunique',
                    'toxicity': 'mean' if 'toxicity' in df.columns else lambda x: 0
                }).reset_index()
                user_activity.columns = ['user_id', 'total_comments', 'videos_engaged', 'avg_toxicity']
                user_activity = user_activity.sort_values('total_comments', ascending=False)
                
                # Multi-select for users
                selected_users = st.multiselect(
                    "Select users to analyze (top active users shown first):",
                    options=user_activity['user_id'].head(50).tolist(),
                    default=user_activity['user_id'].head(5).tolist(),
                    help="Select multiple users to compare their activity patterns",
                    key="multi_user_select"
                )
        
        with col2:
            st.subheader("📊 Selection Stats")
            if selected_users:
                for user in selected_users:
                    user_data = user_activity[user_activity['user_id'] == user].iloc[0]
                    st.write(f"**User {user}:**")
                    st.write(f"- Comments: {user_data['total_comments']}")
                    st.write(f"- Videos: {user_data['videos_engaged']}")
                    st.write(f"- Avg Toxicity: {user_data['avg_toxicity']:.3f}")
                    st.write("---")
        
        # Multi-user timeline visualization
        if selected_users:
            st.subheader("📈 Multi-User Activity Timeline")
            
            # Prepare timeline data
            timeline_data = []
            video_data = []
            
            for user in selected_users:
                user_comments = df[df['unique_id_x'] == user].copy()
                
                # Get video posting times
                user_videos = user_comments.groupby('aweme_id_x').agg({
                    'create_time_x': 'min',  # First comment time as proxy
                    'text_x': 'count'
                }).reset_index()
                user_videos.columns = ['video_id', 'post_time', 'total_comments']
                
                # Add video bubbles
                for _, video in user_videos.iterrows():
                    video_data.append({
                        'user': user,
                        'video_id': video['video_id'],
                        'post_time': video['post_time'],
                        'total_comments': video['total_comments'],
                        'type': 'video'
                    })
                
                # Add comment connections
                for _, comment in user_comments.iterrows():
                    timeline_data.append({
                        'user': user,
                        'video_id': comment['aweme_id_x'],
                        'comment_time': comment['create_time_x'],
                        'toxicity': float(comment.get('toxicity', 0)),
                        'type': 'comment'
                    })
            
            # Create multi-user timeline plot
            fig2 = go.Figure()
            
            import plotly.express as px
            colors = px.colors.qualitative.Set3[:len(selected_users)]
            
            # Add video bubbles for each user - WITH ERROR FIX
            for i, user in enumerate(selected_users):
                user_videos = [v for v in video_data if v['user'] == user]
                
                if user_videos:
                    # Create safe text labels for multi-user timeline
                    text_labels_multi = []
                    for v in user_videos:
                        try:
                            if pd.isna(v['post_time']) or v['post_time'] is pd.NaT:
                                formatted_time = "Invalid Date"
                            else:
                                formatted_time = v['post_time'].strftime('%Y-%m-%d %H:%M')
                            
                            text_labels_multi.append(f"Video: {v['video_id']}<br>Time: {formatted_time}<br>Comments: {v['total_comments']}")
                        except (AttributeError, ValueError):
                            text_labels_multi.append(f"Video: {v['video_id']}<br>Time: Invalid Date<br>Comments: {v['total_comments']}")
                    
                    fig2.add_trace(go.Scatter(
                        x=[v['post_time'] for v in user_videos],
                        y=[i] * len(user_videos),
                        mode='markers',
                        marker=dict(
                            size=[min(v['total_comments'] * 5, 100) for v in user_videos],
                            color=colors[i],
                            opacity=0.7,
                            line=dict(width=2, color='darkblue')
                        ),
                        name=f'User {user} Videos',
                        text=text_labels_multi,
                        hovertemplate='<b>%{text}</b><br>Posted: %{x}<extra></extra>'
                    ))
            
            # Add comment connections between users
            for i, user in enumerate(selected_users):
                user_comments = [c for c in timeline_data if c['user'] == user]
                
                # Group comments by video
                video_comments = {}
                for comment in user_comments:
                    video_id = comment['video_id']
                    if video_id not in video_comments:
                        video_comments[video_id] = []
                    video_comments[video_id].append(comment)
                
                # Create connections to other users' videos
                for video_id, comments in video_comments.items():
                    # Find if this video belongs to another user in selection
                    video_owner = None
                    owner_index = None
                    for j, other_user in enumerate(selected_users):
                        if any(v['video_id'] == video_id and v['user'] == other_user for v in video_data):
                            video_owner = other_user
                            owner_index = j
                            break
                    
                    if video_owner and video_owner != user:
                        # Get video post time
                        video_post_time = next(v['post_time'] for v in video_data if v['video_id'] == video_id and v['user'] == video_owner)
                        
                        # Draw connection line
                        fig2.add_trace(go.Scatter(
                            x=[video_post_time, video_post_time],
                            y=[owner_index, i],
                            mode='lines',
                            line=dict(
                                width=min(len(comments) * 2, 10),
                                color='red',
                                dash='dash'
                            ),
                            opacity=0.6,
                            showlegend=False,
                            hovertemplate=f'User {user} → User {video_owner}<br>Comments: {len(comments)}<extra></extra>'
                        ))
            
            # Update layout
            fig2.update_layout(
                title="Multi-User Video Interaction Timeline",
                xaxis_title="Time",
                yaxis_title="Users",
                yaxis=dict(
                    tickmode='array',
                    tickvals=list(range(len(selected_users))),
                    ticktext=[f'User {user}' for user in selected_users]
                ),
                height=max(400, len(selected_users) * 80),
                hovermode='closest',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            
            st.plotly_chart(fig2, use_container_width=True)
            
            # Legend and explanation
            st.info("""
            **Multi-User Timeline Legend:**
            - 🔵 **Bubbles**: Videos posted by users (size = total comments received)
            - 📏 **Dashed Lines**: Comments from other users (thickness = number of comments)
            - 🎨 **Colors**: Each user has a unique color
            """)
            
            # Interaction statistics
            st.subheader("📊 Cross-User Interactions")
            
            interaction_matrix = pd.DataFrame(0, index=selected_users, columns=selected_users)
            
            for user in selected_users:
                user_comments = df[df['unique_id_x'] == user]
                for _, comment in user_comments.iterrows():
                    video_id = comment['aweme_id_x']
                    # Find video owner
                    for other_user in selected_users:
                        if other_user != user:
                            other_user_videos = df[df['unique_id_x'] == other_user]['aweme_id_x'].unique()
                            if video_id in other_user_videos:
                                interaction_matrix.loc[user, other_user] += 1
            
            if interaction_matrix.sum().sum() > 0:
                import plotly.express as px
                fig3 = px.imshow(
                    interaction_matrix,
                    text_auto=True,
                    aspect="auto",
                    color_continuous_scale='Blues',
                    title="User Interaction Matrix (Comments on Each Other's Videos)"
                )
                st.plotly_chart(fig3, use_container_width=True)
        
        else:
            st.info("Please select users to analyze their timeline interactions.")

elif page == '📈 Trend Analysis':
    st.markdown('<div class="section-header"><h2>📈 Comprehensive Trend Analysis</h2></div>', unsafe_allow_html=True)
    
    # Time-based analysis
    if 'comment_time' in df.columns:
        df['date'] = df['comment_time'].dt.date
        df['week'] = df['comment_time'].dt.isocalendar().week
        df['month'] = df['comment_time'].dt.month
        
        # Daily trends
        st.subheader("📅 Daily Patterns")
        daily_trends = df.groupby(['date', 'day_of_week']).agg({
            'comment': 'count',
            'toxicity': 'mean' if 'toxicity' in df.columns else lambda x: 0,
            'any_toxic': 'sum' if 'any_toxic' in df.columns else lambda x: 0
        }).reset_index()
        
        fig = px.line(
            daily_trends,
            x='date',
            y='comment',
            color='day_of_week',
            title='Daily Comment Volume by Day of Week'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Weekly aggregation
        st.subheader("📊 Weekly Trends")
        weekly_trends = df.groupby('week').agg({
            'comment': 'count',
            'toxicity': 'mean' if 'toxicity' in df.columns else lambda x: 0,
            'any_toxic': 'sum' if 'any_toxic' in df.columns else lambda x: 0
        }).reset_index()
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Weekly Comment Volume', 'Weekly Toxicity Rate'),
            vertical_spacing=0.1
        )
        
        fig.add_trace(
            go.Bar(x=weekly_trends['week'], y=weekly_trends['comment'], name='Comments'),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=weekly_trends['week'], y=weekly_trends['toxicity'], 
                      mode='lines+markers', name='Avg Toxicity'),
            row=2, col=1
        )
        
        fig.update_layout(height=500, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

elif page == '🗝️ Content Analysis':
    st.markdown('<div class="section-header"><h2>🗝️ Advanced Content Analysis</h2></div>', unsafe_allow_html=True)
    
    # Content filtering options
    col1, col2 = st.columns(2)
    with col1:
        content_filter = st.selectbox(
            "Analyze content by:",
            ["All Comments", "Toxic Comments Only", "Non-Toxic Comments Only"]
        )
    
    with col2:
        word_count_filter = st.slider(
            "Minimum word count:",
            1, 20, 3
        )
    
    # Filter data based on selection
    filtered_df = df.copy()
    if content_filter == "Toxic Comments Only":
        filtered_df = df[df['any_toxic'] == 1]
    elif content_filter == "Non-Toxic Comments Only":
        filtered_df = df[df['any_toxic'] == 0]
    
    filtered_df = filtered_df[filtered_df['word_count'] >= word_count_filter]
    
    if len(filtered_df) == 0:
        st.warning("No data matches the selected filters.")
    else:
        # Word cloud
        st.subheader("☁️ Word Cloud Analysis")
        advanced_word_cloud(filtered_df['comment'])
        
        # Topic modeling
        st.subheader("🏷️ Topic Modeling")
        n_topics = st.slider("Number of topics:", 3, 10, 5)
        
        if len(filtered_df) >= 10:
            topics = sentiment_analysis(filtered_df['comment'], n_topics)
            
            if topics:
                for topic in topics:
                    st.write(f"**Topic {topic['topic']}:** {topic['words']}")
        else:
            st.warning("Not enough data for topic modeling.")
        
        # Content statistics
        st.subheader("📊 Content Statistics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            avg_length = filtered_df['comment_length'].mean()
            st.metric("Average Length", f"{avg_length:.1f} chars")
        
        with col2:
            avg_words = filtered_df['word_count'].mean()
            st.metric("Average Words", f"{avg_words:.1f}")
        
        with col3:
            unique_words = len(set(' '.join(filtered_df['comment'].astype(str)).split()))
            st.metric("Unique Words", f"{unique_words:,}")
        
        # Length vs toxicity analysis
        if 'toxicity' in filtered_df.columns:
            st.subheader("📏 Comment Length vs Toxicity")
            
            # Bin comments by length
            filtered_df['length_bin'] = pd.cut(filtered_df['comment_length'], 
                                             bins=10, labels=False)
            length_toxicity = filtered_df.groupby('length_bin').agg({
                'toxicity': 'mean',
                'comment': 'count'
            }).reset_index()
            
            fig = px.bar(
                length_toxicity,
                x='length_bin',
                y='toxicity',
                title='Average Toxicity by Comment Length',
                labels={'length_bin': 'Length Bin', 'toxicity': 'Average Toxicity'}
            )
            st.plotly_chart(fig, use_container_width=True)

elif page == '🌐 Network Insights':
    st.markdown('<div class="section-header"><h2>🌐 Network Analysis & Insights</h2></div>', unsafe_allow_html=True)
    
    # Network configuration
    st.subheader("⚙️ Network Configuration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        min_connections = st.slider("Minimum connections:", 1, 20, 2)
    
    with col2:
        max_nodes = st.slider("Maximum nodes to display:", 50, 500, 200)
    
    with col3:
        layout_type = st.selectbox("Layout algorithm:", 
                                 ["spring", "circular", "kamada_kawai", "random"])
    
    # Build network
    st.subheader("🕸️ User-Video Network")
    
    if 'unique_id_x' in df.columns and 'aweme_id_x' in df.columns:
        # Create edge list
        edges = df.groupby(['unique_id_x', 'aweme_id_x']).agg({
            'comment': 'count',
            'toxicity': 'mean' if 'toxicity' in df.columns else lambda x: 0
        }).reset_index()
        edges.columns = ['user', 'video', 'weight', 'avg_toxicity']
        
        # Filter by minimum connections
        edges = edges[edges['weight'] >= min_connections]
        
        # Limit nodes
        if len(edges) > max_nodes:
            edges = edges.nlargest(max_nodes, 'weight')
        
        if len(edges) == 0:
            st.warning("No connections meet the minimum threshold.")
        else:
            # Build NetworkX graph
            G = nx.Graph()
            
            # Add nodes and edges
            for _, row in edges.iterrows():
                user_node = f"user_{row['user']}"
                video_node = f"video_{row['video']}"
                
                G.add_node(user_node, type='user', toxicity=row['avg_toxicity'])
                G.add_node(video_node, type='video', toxicity=row['avg_toxicity'])
                G.add_edge(user_node, video_node, weight=row['weight'])
            
            # Calculate layout
            if layout_type == "spring":
                pos = nx.spring_layout(G, k=0.5, iterations=50)
            elif layout_type == "circular":
                pos = nx.circular_layout(G)
            elif layout_type == "kamada_kawai":
                pos = nx.kamada_kawai_layout(G)
            else:
                pos = nx.random_layout(G)
            
            # Create plotly traces
            edge_x = []
            edge_y = []
            edge_weights = []
            
            for edge in G.edges(data=True):
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
                edge_weights.append(edge[2]['weight'])
            
            # Node traces
            user_x = []
            user_y = []
            user_text = []
            user_toxicity = []
            
            video_x = []
            video_y = []
            video_text = []
            video_toxicity = []
            
            for node in G.nodes(data=True):
                x, y = pos[node[0]]
                
                if node[1]['type'] == 'user':
                    user_x.append(x)
                    user_y.append(y)
                    user_text.append(node[0])
                    user_toxicity.append(node[1]['toxicity'])
                else:
                    video_x.append(x)
                    video_y.append(y)
                    video_text.append(node[0])
                    video_toxicity.append(node[1]['toxicity'])
            
            # Create figure
            fig = go.Figure()
            
            # Add edges
            fig.add_trace(go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=0.5, color='#888'),
                hoverinfo='none',
                mode='lines',
                showlegend=False
            ))
            
            # Add user nodes
            fig.add_trace(go.Scatter(
                x=user_x, y=user_y,
                mode='markers',
                hoverinfo='text',
                text=user_text,
                marker=dict(
                    showscale=True,
                    colorscale='Viridis',
                    size=15,
                    color=user_toxicity,
                    colorbar=dict(title="Avg Toxicity"),
                    line_width=2,
                    symbol='circle'
                ),
                name='Users'
            ))
            
            # Add video nodes
            fig.add_trace(go.Scatter(
                x=video_x, y=video_y,
                mode='markers',
                hoverinfo='text',
                text=video_text,
                marker=dict(
                    size=12,
                    color='lightblue',
                    line=dict(width=2, color='darkblue'),
                    symbol='square'
                ),
                name='Videos'
            ))
            
            fig.update_layout(
                title='User-Video Interaction Network',
                showlegend=True,
                hovermode='closest',
                margin=dict(b=20, l=5, r=5, t=40),
                annotations=[
                    dict(
                        text="Users (circles) connected to Videos (squares)<br>Edge thickness = comment frequency",
                        showarrow=False,
                        xref="paper", yref="paper",
                        x=0.005, y=-0.002,
                        xanchor='left', yanchor='bottom',
                        font=dict(size=12)
                    )
                ],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Network statistics
            st.subheader("📊 Network Statistics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Nodes", len(G.nodes()))
            
            with col2:
                st.metric("Total Edges", len(G.edges()))
            
            with col3:
                density = nx.density(G)
                st.metric("Network Density", f"{density:.3f}")
            
            with col4:
                if len(G.nodes()) > 0:
                    avg_degree = sum(dict(G.degree()).values()) / len(G.nodes())
                    st.metric("Average Degree", f"{avg_degree:.1f}")
            
            # Community detection
            st.subheader("🏘️ Community Detection")
            
            try:
                communities = nx.community.greedy_modularity_communities(G)
                st.write(f"**Number of communities detected:** {len(communities)}")
                
                for i, community in enumerate(communities[:5]):  # Show top 5 communities
                    users = [node for node in community if node.startswith('user_')]
                    videos = [node for node in community if node.startswith('video_')]
                    
                    st.write(f"**Community {i+1}:** {len(users)} users, {len(videos)} videos")
                    
                    if len(communities) > 5:
                        st.write("... and more communities")
                        break
                        
            except Exception as e:
                st.warning(f"Community detection failed: {str(e)}")
            
            # Top influencers
            st.subheader("🌟 Top Influencers")
            
            # Calculate centrality measures
            try:
                degree_centrality = nx.degree_centrality(G)
                betweenness_centrality = nx.betweenness_centrality(G)
                closeness_centrality = nx.closeness_centrality(G)
                
                # Create centrality dataframe
                centrality_df = pd.DataFrame({
                    'node': list(degree_centrality.keys()),
                    'degree': list(degree_centrality.values()),
                    'betweenness': list(betweenness_centrality.values()),
                    'closeness': list(closeness_centrality.values())
                })
                
                # Filter for users only
                user_centrality = centrality_df[centrality_df['node'].str.startswith('user_')]
                user_centrality = user_centrality.sort_values('degree', ascending=False).head(10)
                
                st.dataframe(user_centrality, use_container_width=True)
                
            except Exception as e:
                st.warning(f"Centrality calculation failed: {str(e)}")
    
    else:
        st.error("Required columns for network analysis not found.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>🛡️ Social Toxicity Analytics Platform | Built with Streamlit & Plotly</p>
    <p>Real-time insights into online discourse patterns and community behavior</p>
</div>
""", unsafe_allow_html=True)
