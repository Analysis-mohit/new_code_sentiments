import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import re
from collections import Counter
from wordcloud import WordCloud
import io
import base64
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import uuid

# Download NLTK stopwords if not already downloaded
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')
    nltk.download('punkt')

# For sentiment analysis
from textblob import TextBlob
try:
    from transformers import pipeline
    transformer_available = True
except Exception:
    transformer_available = False

# Configure Streamlit page settings
st.set_page_config(
    page_title="📊 PulsePoint - Voice of the Customer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Small CSS tweak
st.markdown("""
<style>
    .main-header {
        font-size: 2.4rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card { 
        padding: 1rem; 
        border-radius: 10px; 
        color: white; 
        text-align: center; 
        margin: 0.5rem 0; 
    }
</style>
""", unsafe_allow_html=True)

class SentimentAnalyzer:
    def __init__(self):
        self.transformer_sentiment = None
        if transformer_available:
            try:
                self.transformer_sentiment = pipeline("sentiment-analysis\", 
                                                    model=\"cardiffnlp/twitter-roberta-base-sentiment-latest")
            except Exception as e:
                st.warning(f\"Could not load Hugging Face Transformer model: {e}. Falling back to TextBlob.")
                self.transformer_sentiment = None

    def analyze_textblob(self, text):
        if pd.isna(text) or text == \"\":
            return \"Neutral\", 0.0
        blob = TextBlob(str(text))
        polarity = blob.sentiment.polarity
        if polarity > 0.1:
            return \"Positive\", polarity
        elif polarity < -0.1:
            return \"Negative\", polarity
        else:
            return \"Neutral\", polarity

    def analyze_transformer(self, text):
        if not self.transformer_sentiment or pd.isna(text) or text == \"\":
            return self.analyze_textblob(text)
        try:
            result = self.transformer_sentiment(str(text)[:512]) 
            label = result[0]['label']
            score = result[0]['score']
            if label in ['LABEL_2', 'POSITIVE']:
                return \"Positive\", score
            elif label in ['LABEL_0', 'NEGATIVE']:
                return \"Negative\", -score
            else:
                return \"Neutral\", 0.0
        except Exception as e:
            st.warning(f\"Transformer analysis failed for a text ({str(text)[:50]}...): {e}. Using TextBlob.\")
            return self.analyze_textblob(text)

@st.cache_data
def load_and_process_data(file_path):
    df = pd.read_csv("new_feedbacks.csv")
    df.columns = df.columns.str.strip()
    # Normalize timestamp
    if 'timestam' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestam'], errors='coerce')
    elif 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    else:
        df['timestamp'] = pd.NaT
    # Standardize column names expected downstream
    if 'feedback' not in df.columns and 'content' in df.columns:
        df['feedback'] = df['content'].astype(str)
    if 'rating' in df.columns:
        df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
    else:
        df['rating'] = np.nan
    if 'escalated_flag' not in df.columns:
        df['escalated_flag'] = False
    else:
        df['escalated_flag'] = df['escalated_flag'].fillna(False)
    if 'user_id' not in df.columns:
        df['user_id'] = [str(uuid.uuid4()) for _ in range(len(df))]
    if 'source' not in df.columns:
        df['source'] = 'unknown'
    if 'customer_type' not in df.columns:
        df['customer_type'] = 'Unknown'
    df['customer_type'] = df['customer_type'].fillna('Unknown').astype(str)
    df['feedback'] = df['feedback'].fillna('').astype(str)
    return df

@st.cache_data
def perform_sentiment_analysis(df, method='TextBlob'):
    analyzer = SentimentAnalyzer()
    if df.empty:
        return df
    sentiments = []
    scores = []
    for _, row in df.iterrows():
        rating = row.get('rating', np.nan)
        text = row.get('feedback', '')
        if pd.notna(rating):
            if rating >= 4:
                sentiments.append('Positive'); scores.append(1.0)
            elif rating == 3:
                sentiments.append('Neutral'); scores.append(0.0)
            elif rating <= 2:
                sentiments.append('Negative'); scores.append(-1.0)
            else:
                pred_sentiment, pred_score = (analyzer.analyze_transformer(text) if method == 'Transformer' and transformer_available else analyzer.analyze_textblob(text))
                sentiments.append(pred_sentiment); scores.append(pred_score)
        else:
            pred_sentiment, pred_score = (analyzer.analyze_transformer(text) if method == 'Transformer' and transformer_available else analyzer.analyze_textblob(text))
            sentiments.append(pred_sentiment); scores.append(pred_score)
    df = df.copy()
    df['predicted_sentiment'] = sentiments
    df['sentiment_score'] = scores
    return df

@st.cache_data
def create_sentiment_buckets(df):
    if df.empty:
        return df
    df = df.copy()
    df['sentiment_bucket'] = pd.cut(df['sentiment_score'], 
                                   bins=[-1, -0.5, -0.1, 0.1, 0.5, 1],
                                   labels=['Very Negative', 'Negative', 'Neutral', 'Positive', 'Very Positive'],
                                   include_lowest=True)
    return df

def get_top_n_ngrams_from_negative_feedback(df, n=10, ngram_max_length=4):
    if df.empty:
        return pd.DataFrame(columns=['Problem', 'Count', 'Distribution (%)'])
    negative_feedback = df[df['predicted_sentiment']=='Negative']['feedback'].tolist()
    if not negative_feedback:
        return pd.DataFrame(columns=['Problem', 'Count', 'Distribution (%)'])
    cleaned_feedback_texts = [re.sub(r'[^a-zA-Z0-9\\s]', '', fb.lower()) for fb in negative_feedback]
    stop_words_set = set(stopwords.words('english'))
    custom_stop_words = {'app','service','driver','phone','customer','team','one','get','would','really','back','much','go','like','just','can','even','still','know','dont','didnt','always','also','never','nothing','please','call','want','make','think','need','issue','problem','support','time','day','use','try','way','thing','people','good','bad','great'}
    stop_words_set.update(custom_stop_words)
    uncategorized_ngrams_counter = Counter()
    for feedback_text in cleaned_feedback_texts:
        words_in_text = word_tokenize(feedback_text)
        filtered_words_in_text = [word for word in words_in_text if word.isalpha() and word not in stop_words_set and len(word)>2]
        for length in range(1, ngram_max_length+1):
            for j in range(len(filtered_words_in_text)-length+1):
                ngram = ' '.join(filtered_words_in_text[j:j+length])
                uncategorized_ngrams_counter[ngram]+=1
    final = [{'Problem': k, 'Count': v} for k,v in uncategorized_ngrams_counter.most_common(n)]
    total = sum(item['Count'] for item in final) if final else 0
    df_final = pd.DataFrame(final)
    if not df_final.empty:
        df_final['Distribution (%)'] = (df_final['Count'] / total) * 100
    return df_final

@st.cache_data
def identify_super_users(df):
    if 'user_id' not in df.columns or df['user_id'].nunique()<=1:
        return {'Super Happy':0,'Super Sad':0,'Normal':0}, pd.DataFrame()
    user_summary = df.groupby('user_id').agg(
        feedback_count=('feedback','size'),
        avg_rating=('rating','mean'),
        avg_sentiment_score=('sentiment_score','mean'),
        has_positive_sentiment=('predicted_sentiment', lambda x: ('Positive' in x.values)),
        has_negative_sentiment=('predicted_sentiment', lambda x: ('Negative' in x.values)),
        has_positive_rating=('rating', lambda x: (x>=4).any()),
        has_negative_rating=('rating', lambda x: (x<=2).any())
    ).reset_index()
    SUPER_HAPPY_MIN_FEEDBACK = 5
    SUPER_SAD_MIN_FEEDBACK = 3
    POSITIVE_SENTIMENT_THRESHOLD = 0.1
    NEGATIVE_SENTIMENT_THRESHOLD = -0.1
    user_summary['user_type']='Normal'
    explicitly_happy = (user_summary['has_positive_sentiment']==True) & (user_summary['has_positive_rating']==True)
    consistently_good_engaged = (user_summary['feedback_count']>=SUPER_HAPPY_MIN_FEEDBACK) & (user_summary['has_negative_rating']==False) & (user_summary['avg_sentiment_score']>POSITIVE_SENTIMENT_THRESHOLD)
    user_summary.loc[explicitly_happy | consistently_good_engaged,'user_type']='Super Happy'
    explicitly_sad = (user_summary['has_negative_sentiment']==True) & (user_summary['has_negative_rating']==True)
    consistently_bad_engaged = (user_summary['feedback_count']>=SUPER_SAD_MIN_FEEDBACK) & (user_summary['has_negative_rating']==True) & (user_summary['avg_sentiment_score']<NEGATIVE_SENTIMENT_THRESHOLD)
    user_summary.loc[(explicitly_sad | consistently_bad_engaged) & (user_summary['user_type']!='Super Happy'),'user_type']='Super Sad'
    counts = user_summary['user_type'].value_counts().to_dict()
    final_counts = {'Super Happy':0,'Super Sad':0,'Normal':0}; final_counts.update(counts)
    return final_counts, user_summary

def style_plotly(fig):
    fig.update_layout(plot_bgcolor='#ffffff', paper_bgcolor='#ffffff', title_x=0.5, font=dict(size=13))
    for trace in fig.data:
        if hasattr(trace, 'marker') and trace.marker:
            trace.marker.line = dict(width=0.5, color='white')
    return fig

def create_common_sections(df, view_title, description, include_wordclouds=True):
    st.markdown(f'<h2 class=\"text-2xl md:text-3xl font-semibold text-slate-800 text-center mb-6\">{view_title}</h2>', unsafe_allow_html=True)
    st.markdown(f\"<p class='text-slate-600 text-center mb-8'>{description}</p>\", unsafe_allow_html=True)
    if df.empty:
        st.warning(\"No data available for the selected filters. Please adjust your date range or check the data source.\")
        return
    col1, col2, col3, col4 = st.columns(4)
    total_feedback = len(df)
    avg_rating = df['rating'].mean() if not df['rating'].isna().all() else 0
    escalated_count = int(df['escalated_flag'].sum())
    sentiment_dist = df['predicted_sentiment'].value_counts()
    with col1:
        st.markdown(f\"\"\"<div class=\"metric-card\" style='background: linear-gradient(135deg,#667eea 0%,#764ba2 100%);'><h4>Total Feedback</h4><h2>{total_feedback:,}</h2></div>\"\"\", unsafe_allow_html=True)
    with col2:
        st.markdown(f\"\"\"<div class=\"metric-card\" style='background: linear-gradient(135deg,#11998e 0%,#38ef7d 100%);'><h4>Avg Rating</h4><h2>{avg_rating:.1f}/5</h2></div>\"\"\", unsafe_allow_html=True)
    with col3:
        st.markdown(f\"\"\"<div class=\"metric-card\" style='background: linear-gradient(135deg,#ff416c 0%,#ff4b2b 100%);'><h4>Escalated</h4><h2>{escalated_count}</h2></div>\"\"\", unsafe_allow_html=True)
    with col4:
        dominant_sentiment = sentiment_dist.index[0] if not sentiment_dist.empty else 'Unknown'
        st.markdown(f\"\"\"<div class=\"metric-card\" style='background: linear-gradient(135deg,#ffecd2 0%,#fcb69f 100%); color:#333;'><h4>Dominant Sentiment</h4><h2>{dominant_sentiment}</h2></div>\"\"\", unsafe_allow_html=True)
    st.markdown(\"---\")
    st.markdown(\"<h3>Sentiment Distribution & Trends</h3>\", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        sentiment_dist_df = sentiment_dist.reset_index(); sentiment_dist_df.columns = ['predicted_sentiment','count']
        fig_pie = px.pie(sentiment_dist_df, values='count', names='predicted_sentiment', title=f\"Sentiment Distribution (Total: {total_feedback:,})\", color='predicted_sentiment', color_discrete_map=GLOBAL_SENTIMENT_COLOR_MAP_GENERAL)
        fig_pie.update_traces(textposition='inside', textinfo='percent+label+value', pull=[0.03]*len(sentiment_dist_df))
        fig_pie = style_plotly(fig_pie)
        st.plotly_chart(fig_pie, use_container_width=True)
    with col2:
        if 'time_period' in df.columns and not df['time_period'].isna().all() and df['time_period'].nunique()>1:
            daily_sentiment = df.groupby(['time_period','predicted_sentiment']).size().unstack(fill_value=0)
            for sentiment_type in ['Positive','Negative','Neutral']:
                if sentiment_type not in daily_sentiment.columns:
                    daily_sentiment[sentiment_type]=0
            daily_sentiment = daily_sentiment.sort_index()
            fig_time = px.line(daily_sentiment.reset_index(), x='time_period', y=['Positive','Negative','Neutral'], title=f\"Sentiment Trends Over Time (Total: {total_feedback:,})\", color_discrete_map=GLOBAL_SENTIMENT_COLOR_MAP_GENERAL)
            fig_time = style_plotly(fig_time)
            st.plotly_chart(fig_time, use_container_width=True)
        else:
            st.info(\"Insufficient data points or 'time_period' not available for sentiment trends in this view. Needs at least two data points for trend analysis.\")
    st.markdown(\"---\")
    if include_wordclouds:
        st.markdown(\"<h3>📝 Common Issues & What Customers Love</h3>\", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        stop_words_set = set(stopwords.words('english'))
        custom_stop_words = {'app','service','driver','phone','customer','team','one','get','would','really'}
        stop_words_set.update(custom_stop_words)
        with col1:
            negative_feedback_text = ' '.join(df[df['predicted_sentiment']=='Negative']['feedback'].tolist())
            if negative_feedback_text.strip():
                cleaned_text = re.sub(r'[^a-zA-Z\\s]','',negative_feedback_text.lower())
                if cleaned_text.strip():
                    fig, ax = plt.subplots(figsize=(5,4))
                    wordcloud = WordCloud(width=400, height=300, background_color='white', colormap='Reds', max_words=40, stopwords=stop_words_set).generate(cleaned_text)
                    ax.imshow(wordcloud, interpolation='bilinear'); ax.axis('off'); ax.set_title('Common Issues (Negative Feedback)', color='red')
                    st.pyplot(fig); plt.close(fig)
                else:
                    st.info(\"No significant negative feedback text to generate word cloud.\")
            else:
                st.info(\"No negative feedback found.\")
        with col2:
            positive_feedback_text = ' '.join(df[df['predicted_sentiment']=='Positive']['feedback'].tolist())
            if positive_feedback_text.strip():
                cleaned_text = re.sub(r'[^a-zA-Z\\s]','',positive_feedback_text.lower())
                if cleaned_text.strip():
                    fig, ax = plt.subplots(figsize=(5,4))
                    wordcloud = WordCloud(width=400, height=300, background_color='white', colormap='Greens', max_words=40, stopwords=stop_words_set).generate(cleaned_text)
                    ax.imshow(wordcloud, interpolation='bilinear'); ax.axis('off'); ax.set_title('What Customers Love (Positive Feedback)', color='green')
                    st.pyplot(fig); plt.close(fig)
                else:
                    st.info(\"No significant positive feedback text to generate word cloud.\")
            else:
                st.info(\"No positive feedback found.\")
        st.markdown('---')
    st.markdown('<h3>🎯 Top 10 Problems Faced by Users</h3>', unsafe_allow_html=True)
    top_problems_df = get_top_n_ngrams_from_negative_feedback(df)
    if not top_problems_df.empty:
        st.dataframe(top_problems_df.style.format({\"Distribution (%)\": \"{:.1f}%\"}), use_container_width=True)
        fig_problems = px.bar(top_problems_df.head(10).sort_values('Distribution (%)', ascending=True), x='Distribution (%)', y='Problem', orientation='h', title=f\"Distribution of Top Problems (Total: {total_feedback:,})\", color_discrete_sequence=px.colors.qualitative.Dark24)
        fig_problems = style_plotly(fig_problems)
        st.plotly_chart(fig_problems, use_container_width=True)
    else:
        st.info(\"No negative feedback or identifiable problems found in the selected data range.\")

def create_overview_dashboard(df):
    create_common_sections(df, \"📈 Overall Sentiment Trends & Key Insights\", \"This section provides a holistic view of customer sentiment across all feedback sources within the selected time period.\")
    st.markdown('---')
    st.markdown('<h3>🎉 User Segmentation</h3>', unsafe_allow_html=True)
    user_counts, user_df = identify_super_users(df)
    if user_df.empty or df['user_id'].nunique()<=1:
        st.info(\"User-level analysis requires 'user_id' column with multiple unique users in the data.\")
    else:
        user_col1, user_col2, user_col3 = st.columns(3)
        user_col1.metric('Super Happy Users', user_counts.get('Super Happy',0))
        user_col2.metric('Normal Users', user_counts.get('Normal',0))
        user_col3.metric('Super Sad Users', user_counts.get('Super Sad',0))
        st.subheader('Sample Super Happy Users'); st.dataframe(user_df[user_df['user_type']=='Super Happy'].head(5)[['user_id','feedback_count','avg_rating','avg_sentiment_score']])
        st.subheader('Sample Super Sad Users'); st.dataframe(user_df[user_df['user_type']=='Super Sad'].head(5)[['user_id','feedback_count','avg_rating','avg_sentiment_score']])
    st.markdown('---')
    st.markdown('<h3>Sentiment by Source</h3>', unsafe_allow_html=True)
    total_feedback = len(df)
    source_sentiment = pd.crosstab(df['source'], df['predicted_sentiment']).reset_index()
    source_sentiment_melted = source_sentiment.melt(id_vars='source', var_name='sentiment', value_name='count')
    fig_bar = px.bar(source_sentiment_melted, x='source', y='count', color='sentiment', title=f\"Sentiment by Source (Total: {total_feedback:,})\", color_discrete_map=GLOBAL_SENTIMENT_COLOR_MAP_GENERAL)
    fig_bar.update_layout(barmode='stack', height=420, margin=dict(l=0,r=0,t=50,b=0), xaxis_title='Source', yaxis_title='Count')
    fig_bar = style_plotly(fig_bar)
    st.plotly_chart(fig_bar, use_container_width=True)

def create_playstore_view(df):
    playstore_df = df[df['source'].str.contains('play_store', case=False, na=False)]
    create_common_sections(playstore_df, \"📱 PlayStore Feedback Analysis\", \"Dive deep into customer sentiment from PlayStore reviews.\")
    if playstore_df.empty:
        return
    st.markdown('---')
    st.markdown('<h3>Rating Distribution</h3>', unsafe_allow_html=True)
    if not playstore_df['rating'].isna().all():
        rating_dist = playstore_df['rating'].value_counts().sort_index()
        rating_df = rating_dist.reset_index(); rating_df.columns=['rating','count']
        fig_rating = px.bar(rating_df, x='rating', y='count', title=f\"PlayStore Rating Distribution (Total: {len(playstore_df):,})\", text='count')
        fig_rating.update_traces(texttemplate='%{text}', textposition='outside', marker_line_width=0.5)
        fig_rating = style_plotly(fig_rating)
        st.plotly_chart(fig_rating, use_container_width=True)
    else:
        st.info('No rating data available for PlayStore reviews.')
    st.markdown('---')
    st.markdown('<h3>🔍 Recent Negative PlayStore Feedback (Action Required)</h3>', unsafe_allow_html=True)
    negative_ps = playstore_df[playstore_df['predicted_sentiment']=='Negative'].sort_values('timestamp', ascending=False).head(10)
    if not negative_ps.empty:
        display_cols = [c for c in ['timestamp','feedback','rating','sentiment_score','escalated_flag'] if c in negative_ps.columns]
        st.dataframe(negative_ps[display_cols], use_container_width=True)
    else:
        st.info('No recent negative PlayStore feedback found.')

def create_trip_feedback_view(df):
    trip_df = df[df['source'].str.contains('Trip', case=False, na=False)]
    create_common_sections(trip_df, \"🚗 Trip Feedback Analysis\", \"Analyze customer feedback specifically related to trip or service experiences.\")
    if trip_df.empty:
        return
    st.markdown('---')
    st.markdown('<h3>Rating vs Sentiment Score</h3>', unsafe_allow_html=True)
    if not trip_df['rating'].isna().all():
        trip_corr = trip_df.groupby('rating')['sentiment_score'].mean().reset_index()
        fig_corr = px.line(trip_corr, x='rating', y='sentiment_score', title=f\"Trip Rating vs Sentiment Score (Total: {len(trip_df):,})\", markers=True)
        fig_corr = style_plotly(fig_corr)
        st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.info('No rating data available for Trip Feedback.')

def create_escalation_view(df):
    escalation_df = df[df['source'].str.contains('Escalation', case=False, na=False)]
    create_common_sections(escalation_df, \"🚨 Escalation Analysis\", \"Focus on critical feedback and escalated cases.\")
    if escalation_df.empty:
        return
    st.markdown('---')
    st.markdown('<h3>Severity Distribution & Trend</h3>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        severity_labels=[]
        for score in escalation_df['sentiment_score']:
            if score < -0.7: severity_labels.append('Critical')
            elif score < -0.3: severity_labels.append('High')
            elif score < 0: severity_labels.append('Medium')
            else: severity_labels.append('Low')
        escalation_df['severity'] = pd.Categorical(severity_labels, categories=['Critical','High','Medium','Low'], ordered=True)
        severity_counts = escalation_df['severity'].value_counts().sort_index()
        fig_severity = px.bar(x=severity_counts.index, y=severity_counts.values, title=f\"Escalation Severity Distribution (Total: {len(escalation_df):,})\", color=severity_counts.index)
        fig_severity = style_plotly(fig_severity)
        st.plotly_chart(fig_severity, use_container_width=True)
    with col2:
        if 'time_period' in escalation_df.columns and not escalation_df['time_period'].isna().all() and escalation_df['time_period'].nunique()>1:
            daily_escalations = escalation_df.groupby('time_period').size().reset_index(name='count').sort_values('time_period')
            fig_trend = px.line(daily_escalations, x='time_period', y='count', title=f\"Escalation Trend Over Time (Total: {len(escalation_df):,})\", color_discrete_sequence=['red'])
            fig_trend = style_plotly(fig_trend)
            st.plotly_chart(fig_trend, use_container_width=True)
        else:
            st.info(\"Insufficient data points or 'time_period' not available for Escalation trends.\")
    st.markdown('---')
    st.markdown('<h3>🧾 Critical Escalations (Immediate Action Required)</h3>', unsafe_allow_html=True)
    critical_escalations = escalation_df[escalation_df['sentiment_score'] < -0.7].head(10)
    if not critical_escalations.empty:
        display_cols = [c for c in ['timestamp','feedback','rating','sentiment_score','escalated_flag'] if c in critical_escalations.columns]
        st.dataframe(critical_escalations[display_cols].sort_values('timestamp', ascending=False), use_container_width=True)
    else:
        st.success('✅ No critical escalations found!')

def main_app():
    st.markdown('<h1 class=\"main-header\">📊 PulsePoint - Voice of the Customer</h1>', unsafe_allow_html=True)
    csv_file_path = \"feedbacks_overall.csv\"
    customer_types_for_filter = ['All','Consigner','Operator','Unknown']
    with st.sidebar:
        st.header(\"⚙️ Configuration\")
        st.info(f\"Data Source: `{csv_file_path}`\")
        sentiment_method = st.selectbox(\"🧠 Sentiment Analysis Method\", [\"TextBlob\"] if not transformer_available else [\"TextBlob\",\"Transformer\"], key='global_sentiment_method')
        st.markdown('---')
        st.header('🗓️ Date & Time Filters')
        today = datetime.now().date(); default_start_date = today - timedelta(days=365)
        start_date = st.date_input('Start Date', value=default_start_date, key='global_start_date')
        end_date = st.date_input('End Date', value=today, key='global_end_date')
        time_granularity = st.selectbox('Time Granularity', ['Daily','Weekly','Monthly'], key='global_time_granularity')
        st.markdown('---')
        selected_customer_type = st.selectbox('👥 Select Customer Type', customer_types_for_filter, key='global_customer_type_filter')
        st.markdown('---')
        st.header('📊 Dashboard Views')
        page_selection = st.radio('Explore Data By:', ['📈 Overview','📱 PlayStore Feedback','🚗 Trip Feedback','🚨 Escalations'], key='page_selection')
        st.markdown('---')
        if st.button('🔄 Refresh Analysis', key='global_refresh_button'):
            st.cache_data.clear(); st.rerun()
    try:
        with st.spinner(f\"📊 Loading and processing your data from {csv_file_path}...\"):
            df = load_and_process_data(csv_file_path)
            # Apply global date filter (normalize timestamp to datetime.date for comparison)
            if 'timestamp' in df.columns and not df['timestamp'].isna().all():
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df_filtered = df[(df['timestamp'].dt.date >= start_date) & (df['timestamp'].dt.date <= end_date)].copy()
                if time_granularity == 'Daily':
                    df_filtered['time_period'] = df_filtered['timestamp'].dt.date
                elif time_granularity == 'Weekly':
                    df_filtered['time_period'] = df_filtered['timestamp'].dt.to_period('W').astype(str)
                else:
                    df_filtered['time_period'] = df_filtered['timestamp'].dt.to_period('M').astype(str)
            else:
                st.warning(\"Timestamp column not found or is empty. Date filtering will be limited.\")
                df_filtered = df.copy(); df_filtered['time_period'] = 'N/A'
            df_processed_all = perform_sentiment_analysis(df_filtered, method=sentiment_method)
            df_processed_all = create_sentiment_buckets(df_processed_all)
            # Apply customer type filter
            if selected_customer_type != 'All':
                df_to_display = df_processed_all[df_processed_all['customer_type'].str.lower()==selected_customer_type.lower()].copy()
            else:
                df_to_display = df_processed_all.copy()
            # Safety check
            if df_to_display.empty:
                st.error('No data available after applying filters. Adjust date range or check file.'); return
            st.success(f\"✅ Processed {len(df_to_display):,} feedback records for the selected filters.\")
            # Download button
            csv_output = df_to_display.to_csv(index=False).encode('utf-8')
            download_file_name = f\"processed_sentiment_data_{page_selection.replace(' ','_')}_{selected_customer_type}.csv\"
            st.download_button(label='⬇️ Download Processed Data (CSV)', data=csv_output, file_name=download_file_name, mime='text/csv')
            st.markdown('---')
            # Summary metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                playstore_count = len(df_to_display[df_to_display['source'].str.contains('play_store', case=False, na=False)])
                st.metric('📱 PlayStore Reviews (Filtered)', playstore_count)
            with col2:
                trip_count = len(df_to_display[df_to_display['source'].str.contains('Trip', case=False, na=False)])
                st.metric('🚗 Trip Feedbacks (Filtered)', trip_count)
            with col3:
                escalation_count = len(df_to_display[df_to_display['source'].str.contains('Escalation', case=False, na=False)])
                st.metric('🚨 Escalations (Filtered)', escalation_count)
            st.markdown('---')
            # --- Raw Feedback Viewer ---
            st.markdown('## 🗂 View Raw Feedbacks')
            with st.expander('🔍 Click to View and Filter Raw Feedbacks', expanded=False):
                raw_source_options = sorted(df_to_display['source'].dropna().unique().tolist())
                raw_customer_options = sorted(df_to_display['customer_type'].dropna().unique().tolist())
                raw_source_filter = st.multiselect('Filter by Source', options=raw_source_options, default=raw_source_options)
                raw_customer_filter = st.multiselect('Filter by Customer Type', options=raw_customer_options, default=raw_customer_options)
                raw_start_date = st.date_input('Start Date (Raw View)', value=start_date, key='raw_start_date')
                raw_end_date = st.date_input('End Date (Raw View)', value=end_date, key='raw_end_date')
                df_raw_filtered = df_to_display[
                    (df_to_display['source'].isin(raw_source_filter)) &
                    (df_to_display['customer_type'].isin(raw_customer_filter)) &
                    (df_to_display['timestamp'].dt.date >= raw_start_date) &
                    (df_to_display['timestamp'].dt.date <= raw_end_date)
                ].copy()
                st.write(f\"Showing {len(df_raw_filtered):,} feedback records\")
                display_cols = [c for c in ['timestamp','source','customer_type','rating','predicted_sentiment','sentiment_score','feedback'] if c in df_raw_filtered.columns]
                st.dataframe(df_raw_filtered[display_cols].sort_values('timestamp', ascending=False), use_container_width=True)
            st.markdown('---')
            # Render selected dashboard view
            if page_selection == '📈 Overview':
                create_overview_dashboard(df_to_display)
            elif page_selection == '📱 PlayStore Feedback':
                create_playstore_view(df_to_display)
            elif page_selection == '🚗 Trip Feedback':
                create_trip_feedback_view(df_to_display)
            elif page_selection == '🚨 Escalations':
                create_escalation_view(df_to_display)
    except FileNotFoundError:
        st.error(f\"❌ Error: The file was not found at the specified path: `{csv_file_path}`\")
    except pd.errors.EmptyDataError:
        st.error(f\"❌ Error: The file at `{csv_file_path}` is empty or has no data.\")
    except Exception as e:
        st.error(f\"❌ An unexpected error occurred during data processing: {str(e)}\")
        with st.expander('🔍 Show Detailed Error Information'):
            st.write(f\"Attempted to load data from: `{csv_file_path}`\")
            try:
                temp_df = pd.read_csv(csv_file_path)
                st.write('First few rows:'); st.dataframe(temp_df.head())
                st.write('Detected columns:'); st.write(list(temp_df.columns))
            except Exception as inner_e:
                st.write(f\"Could not read file head: {inner_e}\")
    st.markdown(\"<p style='text-align: center; margin-top: 30px; color: #777;'>Developed by Yash & Mohit</p>\", unsafe_allow_html=True)

if __name__ == '__main__':
    main_app()
    
   
