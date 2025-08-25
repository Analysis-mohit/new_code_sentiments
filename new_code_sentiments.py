#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
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
from pandas.api.types import is_categorical_dtype, is_datetime64_any_dtype, is_numeric_dtype, is_object_dtype

# Download NLTK stopwords if not already downloaded
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('punkt_tab')

# For sentiment analysis
from textblob import TextBlob
try:
    from transformers import pipeline
    transformer_available = True
except ImportError:
    transformer_available = False

# Configure Streamlit page settings
st.set_page_config(
    page_title="📊 PulsePoint - Voice of the Customer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced custom CSS for better design
st.markdown("""
<style>
    /* Styling for the main header of the dashboard */
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4, #45B7D1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    
    /* General styling for metric cards, providing a gradient background */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
    }
    
    /* Specific styling for positive sentiment cards */
    .sentiment-positive {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
    }
    
    /* Specific styling for negative sentiment cards */
    .sentiment-negative {
        background: linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%);
    }
    
    /* Specific styling for neutral sentiment cards */
    .sentiment-neutral {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        color: #333;
    }
    
    /* Enhanced table styling */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    /* Custom styles for Chart.js containers */
    .chart-container {
        position: relative;
        width: 100%;
        max-width: 700px;
        margin-left: auto;
        margin-right: auto;
        height: 350px;
        max-height: 450px;
    }
    
    /* Adjust chart container height for smaller screens */
    @media (max-width: 768px) {
        .chart-container {
            height: 300px;
            max-height: 350px;
        }
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #f8f9fa;
    }
    
    /* Enhanced button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        border: none;
        padding: 0.5rem 1rem;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    }
    
    /* Success message styling */
    .stSuccess {
        background-color: #d4edda;
        border-radius: 10px;
        padding: 1rem;
    }
    
    /* Error message styling */
    .stError {
        border-radius: 10px;
    }
    
    /* Info message styling */
    .stInfo {
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# --- Global Color Maps for Consistent Sentiment Visualization ---
GLOBAL_SENTIMENT_COLOR_MAP = {
    'Very Positive': '#2ca02c',
    'Positive': '#7fcdbb',
    'Neutral': '#ff7f0e',
    'Negative': '#e377c2',
    'Very Negative': '#d62728'
}

GLOBAL_SENTIMENT_COLOR_MAP_GENERAL = {
    'Positive': '#00CC96',
    'Negative': '#EF553B',
    'Neutral': '#FFA15A'
}

class SentimentAnalyzer:
    def __init__(self):
        self.transformer_sentiment = None
        if transformer_available:
            try:
                self.transformer_sentiment = pipeline("sentiment-analysis", 
                                                    model="cardiffnlp/twitter-roberta-base-sentiment-latest")
            except Exception as e:
                st.warning(f"Could not load Hugging Face Transformer model: {e}. Falling back to TextBlob.")
                self.transformer_sentiment = None

    def analyze_textblob(self, text):
        if pd.isna(text) or text == "":
            return "Neutral", 0.0
        
        blob = TextBlob(str(text))
        polarity = blob.sentiment.polarity
        
        if polarity > 0.1:
            return "Positive", polarity
        elif polarity < -0.1:
            return "Negative", polarity
        else:
            return "Neutral", polarity
    
    def analyze_transformer(self, text):
        if not self.transformer_sentiment or pd.isna(text) or text == "":
            return self.analyze_textblob(text)
        
        try:
            result = self.transformer_sentiment(str(text)[:512])
            label = result[0]['label']
            score = result[0]['score']
            
            if label in ['LABEL_2', 'POSITIVE']:
                return "Positive", score
            elif label in ['LABEL_0', 'NEGATIVE']:
                return "Negative", -score
            else:
                return "Neutral", 0.0
        except Exception as e:
            st.warning(f"Transformer analysis failed for a text ({str(text)[:50]}...): {e}. Using TextBlob.")
            return self.analyze_textblob(text)

@st.cache_data
def load_and_process_data(file_path):
    """
    Enhanced data loading with automatic timestamp detection
    """
    df = pd.read_csv("sentiment_25aug.csv")
    
    # Clean column names
    df.columns = df.columns.str.strip()
    
    # Auto-detect timestamp columns
    timestamp_columns = []
    for col in df.columns:
        if any(word in col.lower() for word in ['time', 'date', 'timestam', 'created', 'updated']):
            timestamp_columns.append(col)
    
    # Try to parse timestamp columns
    timestamp_found = False
    for col in timestamp_columns:
        try:
            df['timestamp'] = pd.to_datetime(df[col], errors='coerce')
            if df['timestamp'].notna().sum() > 0:
                timestamp_found = True
                st.success(f"✅ Found and parsed timestamp column: {col}")
                break
        except:
            continue
    
    if not timestamp_found:
        # Try all columns with datetime format
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    df['timestamp'] = pd.to_datetime(df[col], errors='coerce')
                    if df['timestamp'].notna().sum() > 0:
                        timestamp_found = True
                        st.success(f"✅ Found and parsed timestamp column: {col}")
                        break
                except:
                    continue
    
    if not timestamp_found:
        st.warning("⚠️ No valid timestamp column found. Using default date.")
        df['timestamp'] = pd.Timestamp.now()
    
    # Process other columns
    df['feedback'] = df['feedback'].fillna('').astype(str)
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
    
    if 'escalated_flag' in df.columns:
        df['escalated_flag'] = df['escalated_flag'].fillna(False).astype(str).str.upper()
        df['escalated_flag'] = df['escalated_flag'].map({'TRUE': True, 'FALSE': False}).fillna(False)
    
    if 'user_id' not in df.columns:
        df['user_id'] = df.index.astype(str) + '_anon'
    
    if 'customer_type' not in df.columns:
        df['customer_type'] = 'Consumer'
    df['customer_type'] = df['customer_type'].fillna('Unknown').astype(str).str.strip()
    
    if 'source' not in df.columns:
        df['source'] = 'Unknown'
    
    return df

@st.cache_data
def perform_sentiment_analysis(df, method="TextBlob"):
    analyzer = SentimentAnalyzer()
    
    if df.empty:
        return df
    
    sentiments = []
    scores = []

    for index, row in df.iterrows():
        rating = row['rating']
        text = row['feedback']

        if pd.notna(rating):
            if rating >= 4:
                sentiments.append("Positive")
                scores.append(1.0)
            elif rating == 3:
                sentiments.append("Neutral")
                scores.append(0.0)
            elif rating <= 2:
                sentiments.append("Negative")
                scores.append(-1.0)
            else:
                pred_sentiment, pred_score = (analyzer.analyze_transformer(text) if method == "Transformer" and transformer_available else analyzer.analyze_textblob(text))
                sentiments.append(pred_sentiment)
                scores.append(pred_score)
        else:
            pred_sentiment, pred_score = (analyzer.analyze_transformer(text) if method == "Transformer" and transformer_available else analyzer.analyze_textblob(text))
            sentiments.append(pred_sentiment)
            scores.append(pred_score)
    
    df['predicted_sentiment'] = sentiments
    df['sentiment_score'] = scores
    
    return df

@st.cache_data
def create_sentiment_buckets(df):
    if df.empty:
        return df

    df['sentiment_bucket'] = pd.cut(df['sentiment_score'], 
                                   bins=[-1, -0.5, -0.1, 0.1, 0.5, 1],
                                   labels=['Very Negative', 'Negative', 'Neutral', 'Positive', 'Very Positive'],
                                   include_lowest=True)
    return df

# Enhanced filter function for raw data view
def filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a UI on top of a dataframe to let viewers filter columns
    """
    modify = st.checkbox("Add filters", key="raw_data_filter")
    
    if not modify:
        return df
    
    df = df.copy()
    
    # Try to convert datetimes into a standard format
    for col in df.columns:
        if is_object_dtype(df[col]):
            try:
                df[col] = pd.to_datetime(df[col])
            except Exception:
                pass
        
        if is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.tz_localize(None)
    
    modification_container = st.container()
    
    with modification_container:
        to_filter_columns = st.multiselect("Filter dataframe on", df.columns)
        for column in to_filter_columns:
            left, right = st.columns((1, 20))
            left.write("↳")
            
            # Treat columns with < 10 unique values as categorical
            if is_categorical_dtype(df[column]) or df[column].nunique() < 10:
                user_cat_input = right.multiselect(
                    f"Values for {column}",
                    df[column].unique(),
                    default=list(df[column].unique()),
                )
                df = df[df[column].isin(user_cat_input)]
            elif is_numeric_dtype(df[column]):
                _min = float(df[column].min())
                _max = float(df[column].max())
                step = (_max - _min) / 100
                user_num_input = right.slider(
                    f"Values for {column}",
                    min_value=_min,
                    max_value=_max,
                    value=(_min, _max),
                    step=step,
                )
                df = df[df[column].between(*user_num_input)]
            elif is_datetime64_any_dtype(df[column]):
                user_date_input = right.date_input(
                    f"Values for {column}",
                    value=(
                        df[column].min(),
                        df[column].max(),
                    ),
                )
                if len(user_date_input) == 2:
                    user_date_input = tuple(map(pd.to_datetime, user_date_input))
                    start_date, end_date = user_date_input
                    df = df.loc[df[column].between(start_date, end_date)]
            else:
                user_text_input = right.text_input(
                    f"Substring or regex in {column}",
                )
                if user_text_input:
                    df = df[df[column].astype(str).str.contains(user_text_input, case=False)]
    
    return df

# Rest of the code remains the same...
ISSUE_CATEGORIES_KEYWORDS = {
    "Bilty/LR Issues": ["bilti", "lr", "builty", "billtee"],
    "Payment Issues": ["payment", "charges", "money", "wallet", "credit", "tds", "extra", "amount"],
    "Invoice Issues": ["invoice", "gst", "bill", "billing", "wrong", "missing", "change name"],
    "Driver Behavior/Availability": ["driver", "picking call", "rude", "reached", "extra money", "tripal", "not responding", "absent"],
    "Stage Change Issues": ["trip not started", "stage change", "loaded", "not moving", "started", "moving"],
    "Transit/Halt Issues": ["transit halt", "stopped", "halt", "not running", "police"],
    "Location/Tracking Issues": ["location", "gps", "tracking", "not updated", "showing wrong"],
    "E-Way Bill Issues": ["e-way bill", "eway"],
    "Delay Issues": ["delay", "late", "dispatch"],
    "Damage/Material Issues": ["damage", "material", "scrap", "unloading", "shortage"],
    "Cancellation Issues": ["cancel", "cancelled", "cancel vehicle", "booking cancelled"],
    "Other Vehicle Issues": ["vehicle", "height", "number plate", "tyres", "unfit", "overloaded"],
    "POD Issues": ["pod"],
    "OTP Issues": ["otp"],
    "Extension Request": ["extension", "extend"],
    "Booking Issues": ["booking", "booked"],
    "Service Quality": ["poor service", "bad experience", "unprofessional", "not satisfied"],
    "Customer Support": ["call", "connect", "issue resolved", "support", "response"],
    "Other/Miscellaneous": ["problem", "help", "insurance", "address", "pincode", "resolved"]
}

KEYWORD_TO_CATEGORY = {}
for category, keywords in ISSUE_CATEGORIES_KEYWORDS.items():
    for keyword in keywords:
        KEYWORD_TO_CATEGORY[keyword.lower()] = category

def get_top_n_ngrams_from_negative_feedback(df, n=10, ngram_max_length=4):
    if df.empty:
        return pd.DataFrame(columns=['Problem', 'Count', 'Distribution (%)'])

    negative_feedback = df[df['predicted_sentiment'] == 'Negative']['feedback'].tolist()
    if not negative_feedback:
        return pd.DataFrame(columns=['Problem', 'Count', 'Distribution (%)'])

    cleaned_feedback_texts = [re.sub(r'[^a-zA-Z0-9\s]', '', fb.lower()) for fb in negative_feedback]
    
    stop_words_set = set(stopwords.words('english'))
    custom_stop_words = {
        'app', 'service', 'driver', 'phone', 'customer', 'team', 'ride', 'one', 'get', 'would', 'really', 'back', 'much', 'go', 'like', 'just', 'can', 'even', 'still', 'know', 'dont', 'didnt', 'always', 'also', 'never', 'nothing', 'please', 'call', 'want', 'make', 'think', 'need', 'issue', 'problem', 'support', 'time', 'day', 'use', 'try', 'way', 'thing', 'people', 'good', 'bad', 'great', 'every', 'us', 'said', 'found', 'getting', 'not', 'will', 'vehicle', 'vehical', 'gadi', 'truck', 'rupees', 'rs', 'money', 'sir', 'mam', 'kindly', 'pls', 'plz', 'share', 'provide', 'update', 'check', 'due'
    } 
    stop_words_set.update(custom_stop_words)

    category_counts = Counter()
    sorted_keywords = sorted(KEYWORD_TO_CATEGORY.keys(), key=len, reverse=True)
    feedback_categorized_flags = [False] * len(cleaned_feedback_texts)

    for i, feedback_text in enumerate(cleaned_feedback_texts):
        matched_categories_for_this_feedback = set()
        for keyword in sorted_keywords:
            if keyword in feedback_text:
                category = KEYWORD_TO_CATEGORY[keyword]
                if category not in matched_categories_for_this_feedback:
                    category_counts[category] += 1
                    matched_categories_for_this_feedback.add(category)
        if matched_categories_for_this_feedback:
            feedback_categorized_flags[i] = True

    uncategorized_ngrams_counter = Counter()
    for i, feedback_text in enumerate(cleaned_feedback_texts):
        if not feedback_categorized_flags[i]:
            words_in_text = word_tokenize(feedback_text)
            filtered_words_in_text = [word for word in words_in_text if word.isalpha() and word not in stop_words_set and len(word) > 2]
            
            for length in range(1, ngram_max_length + 1):
                for j in range(len(filtered_words_in_text) - length + 1):
                    ngram = ' '.join(filtered_words_in_text[j:j+length])
                    
                    is_part_of_category_keyword = False
                    for category_keyword in KEYWORD_TO_CATEGORY.keys():
                        if (ngram == category_keyword) or (ngram in category_keyword and len(ngram) > len(category_keyword) / 2) or (category_keyword in ngram and len(category_keyword) > len(ngram) / 2):
                            is_part_of_category_keyword = True
                            break

                    if not is_part_of_category_keyword:
                        uncategorized_ngrams_counter[ngram] += 1

    final_problems_list = []
    for category, count in category_counts.most_common():
        final_problems_list.append({'Problem': category, 'Count': count})
    
    added_problem_texts = {item['Problem'].lower() for item in final_problems_list}
    
    for problem_text, count in uncategorized_ngrams_counter.most_common():
        if problem_text.lower() not in added_problem_texts:
            final_problems_list.append({'Problem': problem_text, 'Count': count})
            added_problem_texts.add(problem_text.lower())
        
        if len(final_problems_list) >= n:
            break

    if not final_problems_list:
        return pd.DataFrame(columns=['Problem', 'Count', 'Distribution (%)'])

    final_problems_list_sorted = sorted(final_problems_list, key=lambda x: x['Count'], reverse=True)
    final_problems_list_sliced = final_problems_list_sorted[:n]

    total_counts_for_dist = sum(item['Count'] for item in final_problems_list_sliced)
    results_df = pd.DataFrame(final_problems_list_sliced)
    results_df['Distribution (%)'] = (results_df['Count'] / total_counts_for_dist) * 100 if total_counts_for_dist > 0 else 0
    results_df = results_df.sort_values(by=['Count', 'Problem'], ascending=[False, True])
    
    return results_df

def identify_super_users(df):
    if 'user_id' not in df.columns or df['user_id'].nunique() <= 1:
        return {'Super Happy': 0, 'Super Sad': 0, 'Normal': 0}, pd.DataFrame()

    user_summary = df.groupby('user_id').agg(
        feedback_count=('feedback', 'size'),
        avg_rating=('rating', 'mean'),
        min_rating=('rating', 'min'),
        max_rating=('rating', 'max'),
        avg_sentiment_score=('sentiment_score', 'mean'),
        has_positive_sentiment=('predicted_sentiment', lambda x: ('Positive' in x.values)),
        has_negative_sentiment=('predicted_sentiment', lambda x: ('Negative' in x.values)),
        has_positive_rating=('rating', lambda x: (x >= 4).any()),
        has_neutral_rating=('rating', lambda x: (x == 3).any()),
        has_negative_rating=('rating', lambda x: (x <= 2).any())
    ).reset_index()

    SUPER_HAPPY_MIN_FEEDBACK = 5
    SUPER_SAD_MIN_FEEDBACK = 3
    POSITIVE_SENTIMENT_THRESHOLD = 0.1
    NEGATIVE_SENTIMENT_THRESHOLD = -0.1

    user_summary['user_type'] = 'Normal'

    explicitly_happy = (user_summary['has_positive_sentiment'] == True) & (user_summary['has_positive_rating'] == True)
    consistently_good_engaged = (user_summary['feedback_count'] >= SUPER_HAPPY_MIN_FEEDBACK) & (user_summary['has_negative_rating'] == False) & (user_summary['avg_sentiment_score'] > POSITIVE_SENTIMENT_THRESHOLD)
    user_summary.loc[explicitly_happy | consistently_good_engaged, 'user_type'] = 'Super Happy'

    explicitly_sad = (user_summary['has_negative_sentiment'] == True) & (user_summary['has_negative_rating'] == True)
    consistently_bad_engaged = (user_summary['feedback_count'] >= SUPER_SAD_MIN_FEEDBACK) & (user_summary['has_negative_rating'] == True) & (user_summary['avg_sentiment_score'] < NEGATIVE_SENTIMENT_THRESHOLD)
    user_summary.loc[(explicitly_sad | consistently_bad_engaged) & (user_summary['user_type'] != 'Super Happy'), 'user_type'] = 'Super Sad'

    user_type_counts = user_summary['user_type'].value_counts().to_dict()
    final_counts = {'Super Happy': 0, 'Super Sad': 0, 'Normal': 0}
    final_counts.update(user_type_counts)

    return final_counts, user_summary

# Dashboard View Functions (keeping the same structure but enhanced)
def create_common_sections(df, view_title, description, include_wordclouds=True):
    st.markdown(f'<h2 class="text-2xl md:text-3xl font-semibold text-slate-800 text-center mb-6">{view_title}</h2>', unsafe_allow_html=True)
    st.markdown(f"<p class='text-slate-600 text-center mb-8'>{description}</p>", unsafe_allow_html=True)

    if df.empty:
        st.warning("No data available for the selected filters. Please adjust your date range or check the data source.")
        return

    col1, col2, col3, col4 = st.columns(4)
    
    total_feedback = len(df)
    avg_rating = df['rating'].mean() if not df['rating'].isna().all() else 0
    escalated_count = df['escalated_flag'].sum() if 'escalated_flag' in df.columns else 0
    sentiment_dist = df['predicted_sentiment'].value_counts()
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>📊 Total Feedback</h3>
            <h2>{total_feedback:,}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>⭐ Avg Rating</h3>
            <h2>{avg_rating:.1f}/5</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>🚨 Escalated</h3>
            <h2>{escalated_count}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        dominant_sentiment = sentiment_dist.index[0] if not sentiment_dist.empty else "Unknown"
        st.markdown(f"""
        <div class="metric-card">
            <h3>🎯 Dominant Sentiment</h3>
            <h2>{dominant_sentiment}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("<h3 class='text-xl md:text-2xl font-semibold text-slate-700 mb-4'>Sentiment Distribution & Trends</h3>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    
    with col1:
        sentiment_dist_df = sentiment_dist.reset_index()
        sentiment_dist_df.columns = ['predicted_sentiment', 'count']

        fig_pie = px.pie(sentiment_dist_df,
                        values='count', names='predicted_sentiment',
                        title="Sentiment Distribution",
                        color='predicted_sentiment',
                        color_discrete_map=GLOBAL_SENTIMENT_COLOR_MAP_GENERAL)
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        fig_pie.update_layout(height=400, margin=dict(l=0, r=0, t=50, b=0))
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        if 'time_period' in df.columns and not df['time_period'].isna().all() and df['time_period'].nunique() > 1:
            daily_sentiment = df.groupby(['time_period', 'predicted_sentiment']).size().unstack(fill_value=0)
            
            for sentiment_type in ['Positive', 'Negative', 'Neutral']:
                if sentiment_type not in daily_sentiment.columns:
                    daily_sentiment[sentiment_type] = 0

            daily_sentiment = daily_sentiment.sort_index()

            fig_time = px.line(daily_sentiment.reset_index(), 
                                x='time_period', 
                                y=['Positive', 'Negative', 'Neutral'],
                                title="Sentiment Trends Over Time",
                                color_discrete_map=GLOBAL_SENTIMENT_COLOR_MAP_GENERAL)
            fig_time.update_xaxes(title_text="Time Period")
            fig_time.update_layout(height=400, margin=dict(l=0, r=0, t=50, b=0))
            st.plotly_chart(fig_time, use_container_width=True)
        else:
            st.info("Insufficient data points or 'time_period' not available for sentiment trends in this view.")

# [Include all other view functions: create_overview_dashboard, create_playstore_view, create_trip_feedback_view, create_escalation_view]
# They remain the same as in your original code, just copy them here...

# Raw Data View Function
def create_raw_data_view(df):
    st.markdown('<h2 class="text-2xl md:text-3xl font-semibold text-slate-800 text-center mb-6">📊 Raw Data Explorer</h2>', unsafe_allow_html=True)
    st.markdown("<p class='text-slate-600 text-center mb-8'>Explore and filter the raw feedback data with advanced filtering options</p>", unsafe_allow_html=True)
    
    if df.empty:
        st.warning("No data available for the selected filters.")
        return
    
    # Display basic info
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", len(df))
    with col2:
        st.metric("Columns", len(df.columns))
    with col3:
        st.metric("Date Range", f"{df['timestamp'].min().strftime('%Y-%m-%d')} to {df['timestamp'].max().strftime('%Y-%m-%d')}" if 'timestamp' in df.columns else "N/A")
    with col4:
        st.metric("Missing Values", df.isnull().sum().sum())
    
    st.markdown("---")
    
    # Column selector
    col1, col2 = st.columns([2, 1])
    with col1:
        selected_columns = st.multiselect(
            "Select columns to display",
            options=df.columns.tolist(),
            default=df.columns.tolist()[:10]  # Show first 10 columns by default
        )
    
    with col2:
        rows_per_page = st.selectbox("Rows per page", [10, 25, 50, 100, 200], index=1)
    
    # Filter the dataframe
    filtered_df = filter_dataframe(df[selected_columns])
    
    # Display statistics
    st.markdown("### 📈 Data Summary")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Numerical Columns Summary:**")
        numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            st.dataframe(filtered_df[numeric_cols].describe())
        else:
            st.info("No numerical columns selected")
    
    with col2:
        st.write("**Categorical Columns Summary:**")
        categorical_cols = filtered_df.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            for col in categorical_cols:
                if col in filtered_df.columns:
                    unique_count = filtered_df[col].nunique()
                    st.write(f"**{col}:** {unique_count} unique values")
                    if unique_count <= 10:
                        st.write(filtered_df[col].value_counts())
    
    st.markdown("---")
    
    # Display the filtered dataframe
    st.markdown(f"### 📋 Filtered Data ({len(filtered_df)} rows)")
    
    # Add export options
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        if st.button("📥 Export to CSV"):
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"filtered_feedback_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    with col2:
        if st.button("📊 Export to Excel"):
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                filtered_df.to_excel(writer, index=False, sheet_name='Feedback Data')
            excel_data = output.getvalue()
            st.download_button(
                label="Download Excel",
                data=excel_data,
                file_name=f"filtered_feedback_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    
    with col3:
        if st.button("🔄 Reset Filters"):
            st.rerun()
    
    # Display the dataframe with pagination
    if len(filtered_df) > 0:
        st.dataframe(filtered_df, use_container_width=True)
        
        # Show paginated view
        total_rows = len(filtered_df)
        total_pages = (total_rows - 1) // rows_per_page + 1
        
        page_number = st.number_input("Page", min_value=1, max_value=total_pages, value=1)
        
        start_idx = (page_number - 1) * rows_per_page
        end_idx = start_idx + rows_per_page
        
        st.dataframe(
            filtered_df.iloc[start_idx:end_idx],
            use_container_width=True,
            height=400
        )
        
        st.write(f"Showing rows {start_idx + 1} to {min(end_idx, total_rows)} of {total_rows}")
    else:
        st.warning("No data matches the current filters.")

# --- Main App Logic ---
def main_app():
    st.markdown('<h1 class="main-header">📊 PulsePoint - Voice of the Customer</h1>', unsafe_allow_html=True)
    
    csv_file_path = "sentiment_25aug.csv"

    customer_types_for_filter = ['All', 'Consigner', 'Operator']

    with st.sidebar:
        st.header("⚙️ Configuration")
        
        st.info(f"Data Source: `{csv_file_path}`") 
        
        sentiment_method = st.selectbox(
            "🧠 Sentiment Analysis Method",
            ["TextBlob"] if transformer_available else ["TextBlob"],
            key="global_sentiment_method",
            help="Choose the sentiment analysis method."
        )
        
        st.markdown("---")
        st.header("🗓️ Date & Time Filters")
        today = datetime.now().date()
        default_start_date = today - timedelta(days=365)

        start_date = st.date_input("Start Date", value=default_start_date, key="global_start_date")
        end_date = st.date_input("End Date", value=today, key="global_end_date")
        time_granularity = st.selectbox("Time Granularity", ["Daily", "Weekly", "Monthly"], key="global_time_granularity")
        
        st.markdown("---")
        selected_customer_type = st.selectbox(
            "👥 Select Customer Type",
            customer_types_for_filter,
            key="global_customer_type_filter"
        )
        
        st.markdown("---")
        st.header("📊 Dashboard Views")
        page_selection = st.radio(
            "Explore Data By:",
            ["📈 Overview", "📱 PlayStore Feedback", "🚗 Trip Feedback", "🚨 Escalations", "📊 Raw Data Explorer"],
            key="page_selection"
        )

        st.markdown("---")
        if st.button("🔄 Refresh Analysis", key="global_refresh_button"):
            st.cache_data.clear()
            st.rerun()

    try:
        with st.spinner(f"📊 Loading and processing your data from {csv_file_path}..."):
            df = load_and_process_data(csv_file_path)
            
            # Apply global date filter
            if 'timestamp' in df.columns and not df['timestamp'].isna().all():
                mask = (df['timestamp'].dt.date >= start_date) & (df['timestamp'].dt.date <= end_date)
                df_filtered = df[mask].copy()
                
                if time_granularity == "Daily":
                    df_filtered['time_period'] = df_filtered['timestamp'].dt.date
                elif time_granularity == "Weekly":
                    df_filtered['time_period'] = df_filtered['timestamp'].dt.to_period('W').astype(str)
                elif time_granularity == "Monthly":
                    df_filtered['time_period'] = df_filtered['timestamp'].dt.to_period('M').astype(str)
            else:
                df_filtered = df.copy() 
                df_filtered['time_period'] = 'N/A' 
                
            df_processed_all = perform_sentiment_analysis(df_filtered, method=sentiment_method)
            df_processed_all = create_sentiment_buckets(df_processed_all)

            # Apply customer type filter
            if selected_customer_type != 'All':
                df_to_display = df_processed_all[df_processed_all['customer_type'] == selected_customer_type].copy()
            else:
                df_to_display = df_processed_all.copy()
        
        if df_to_display.empty:
            st.error("No data available after applying filters. Please adjust your date range or check the input CSV file.")
            return
        else:
            st.success(f"✅ Processed {len(df_to_display)} feedback records successfully!")
            
            # Download button
            csv_output = df_to_display.to_csv(index=False).encode('utf-8')
            download_file_name = f"processed_sentiment_data_{page_selection.replace(' ', '_')}_{selected_customer_type}.csv"
            st.download_button(
                label="⬇️ Download Processed Data (CSV)",
                data=csv_output,
                file_name=download_file_name,
                mime="text/csv"
            )
            st.markdown("---")

            # Display summary metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                playstore_count = len(df_to_display[df_to_display['source'].str.contains('play_store', case=False, na=False)])
                st.metric("📱 PlayStore Reviews", playstore_count)
            with col2:
                trip_count = len(df_to_display[df_to_display['source'].str.contains('Trip', case=False, na=False)])
                st.metric("🚗 Trip Feedbacks", trip_count)
            with col3:
                escalation_count = len(df_to_display[df_to_display['source'].str.contains('Escalation', case=False, na=False)])
                st.metric("🚨 Escalations", escalation_count)
            st.markdown("---")

            # Render selected page
            if page_selection == "📈 Overview":
                create_overview_dashboard(df_to_display)
            elif page_selection == "📱 PlayStore Feedback":
                create_playstore_view(df_to_display)
            elif page_selection == "🚗 Trip Feedback":
                create_trip_feedback_view(df_to_display)
            elif page_selection == "🚨 Escalations":
                create_escalation_view(df_to_display[df_to_display['source'].str.contains('Escalation', case=False, na=False)])
            elif page_selection == "📊 Raw Data Explorer":
                create_raw_data_view(df_to_display)
            
    except FileNotFoundError:
        st.error(f"❌ Error: The file was not found at the specified path: `{csv_file_path}`")
    except pd.errors.EmptyDataError:
        st.error(f"❌ Error: The file at `{csv_file_path}` is empty or has no data.")
    except Exception as e:
        st.error(f"❌ An unexpected error occurred: {str(e)}")
        with st.expander("🔍 Show Detailed Error Information"):
            st.write(f"Detailed error: {e}")

    st.markdown("<p style='text-align: center; margin-top: 50px; color: #777;'>Developed by Yash & Mohit</p>", unsafe_allow_html=True)

if __name__ == "__main__":
    main_app()
