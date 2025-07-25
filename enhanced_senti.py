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
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

# Download NLTK dependencies
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')

# For sentiment analysis
from textblob import TextBlob

# Try to import spaCy
try:
    import spacy
    # Try to load the model
    try:
        nlp = spacy.load("en_core_web_sm")
        spacy_available = True
    except OSError:
        st.warning("spaCy model 'en_core_web_sm' not found. Please install it using: python -m spacy download en_core_web_sm")
        spacy_available = False
except ImportError:
    st.warning("spaCy not installed. Please install it using: pip install spacy")
    spacy_available = False

# Configure Streamlit page settings
st.set_page_config(
    page_title="📊 PulsePoint - Voice of the Customer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS for enhanced visual styling
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
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
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
    
    /* Styling for Streamlit tabs list (not used in this version but kept for reference) */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    
    /* Styling for individual Streamlit tabs (not used in this version but kept for reference) */
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        color: white;
        font-weight: bold;
    }

    /* Custom styles for Chart.js containers */
    .chart-container {
        position: relative;
        width: 100%;
        max-width: 700px; /* Limit chart width on large screens */
        margin-left: auto;
        margin-right: auto;
        height: 350px; /* Default height */
        max-height: 450px; /* Max height for responsiveness */
    }
    /* Adjust chart container height for smaller screens */
    @media (max-width: 768px) {
        .chart-container {
            height: 300px;
            max-height: 350px;
        }
    }
</style>
""", unsafe_allow_html=True)

# --- Global Color Maps for Consistent Sentiment Visualization ---
# Detailed map for 5 sentiment buckets
GLOBAL_SENTIMENT_COLOR_MAP = {
    'Very Positive': '#2ca02c',  # Dark Green
    'Positive': '#7fcdbb',      # Light Green
    'Neutral': '#ff7f0e',       # Orange
    'Negative': '#e377c2',      # Light Red/Pink
    'Very Negative': '#d62728'  # Dark Red
}

# General map for 3 sentiment types (Positive, Negative, Neutral)
GLOBAL_SENTIMENT_COLOR_MAP_GENERAL = {
    'Positive': '#00CC96',  # Green
    'Negative': '#EF553B',  # Red
    'Neutral': '#FFA15A'   # Orange
}
# --- End Global Color Maps ---

class EnhancedSentimentAnalyzer:
    """
    Enhanced sentiment analysis class using spaCy for more accurate sentiment analysis
    and improved text preprocessing.
    """
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        if spacy_available:
            self.nlp = nlp
        
    def preprocess_text(self, text):
        """
        Enhanced text preprocessing with lemmatization and better cleaning.
        """
        if pd.isna(text) or text == "":
            return ""
        
        # Convert to string and lowercase
        text = str(text).lower()
        
        # Remove special characters but keep spaces
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespaces
        text = ' '.join(text.split())
        
        if spacy_available:
            # Use spaCy for better preprocessing
            doc = self.nlp(text)
            # Keep only meaningful words (no stop words, no punctuation)
            tokens = [token.lemma_ for token in doc 
                     if not token.is_stop and not token.is_punct and len(token.text) > 2]
            return ' '.join(tokens)
        else:
            # Fallback to NLTK preprocessing
            tokens = word_tokenize(text)
            stop_words = set(stopwords.words('english'))
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                     if token not in stop_words and len(token) > 2]
            return ' '.join(tokens)
    
    def analyze_spacy_sentiment(self, text):
        """
        Enhanced sentiment analysis using spaCy with improved accuracy.
        """
        if pd.isna(text) or text == "":
            return "Neutral", 0.0
        
        preprocessed_text = self.preprocess_text(text)
        if not preprocessed_text:
            return "Neutral", 0.0
        
        # Use TextBlob on preprocessed text for better accuracy
        blob = TextBlob(preprocessed_text)
        polarity = blob.sentiment.polarity
        
        # Enhanced sentiment classification with more nuanced thresholds
        if polarity > 0.3:
            return "Positive", polarity
        elif polarity > 0.05:
            return "Positive", polarity * 0.8  # Slightly positive
        elif polarity < -0.3:
            return "Negative", polarity
        elif polarity < -0.05:
            return "Negative", polarity * 0.8  # Slightly negative
        else:
            return "Neutral", polarity

    def analyze_textblob(self, text):
        """
        Enhanced TextBlob analysis with better preprocessing.
        """
        if pd.isna(text) or text == "":
            return "Neutral", 0.0
        
        preprocessed_text = self.preprocess_text(text)
        if not preprocessed_text:
            return "Neutral", 0.0
        
        blob = TextBlob(preprocessed_text)
        polarity = blob.sentiment.polarity
        
        # Enhanced thresholds for better accuracy
        if polarity > 0.15:
            return "Positive", polarity
        elif polarity < -0.15:
            return "Negative", polarity
        else:
            return "Neutral", polarity

@st.cache_data
def load_and_process_data(file_path):
    """
    Loads a CSV file from the given path and preprocesses the data.
    """
    df = pd.read_csv("data_sentiments - Sheet1.csv")
    
    df.columns = df.columns.str.strip()
    
    if 'timestam' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestam'], errors='coerce')
    elif 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    else:
        st.warning("Neither 'timestam' nor 'timestamp' column found. Time-series analysis will be limited.")
        df['timestamp'] = pd.NaT 
    
    df['feedback'] = df['feedback'].fillna('').astype(str)
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
    
    df['escalated_flag'] = df['escalated_flag'].fillna(False).astype(str).str.upper()
    df['escalated_flag'] = df['escalated_flag'].map({'TRUE': True, 'FALSE': False}).fillna(False)
    
    # Ensure 'user_id' column exists for super user analysis
    if 'user_id' not in df.columns:
        st.warning("'user_id' column not found. Super User analysis (Super Happy/Super Sad) will not be available.")
        df['user_id'] = df.index.astype(str) + '_anon' # Assign unique dummy IDs if missing
    
    # Handle 'customer_type' column
    if 'customer_type' not in df.columns:
        st.warning("'customer_type' column not found. Defaulting to 'Consumer'.")
        df['customer_type'] = 'Consumer'
    df['customer_type'] = df['customer_type'].fillna('Unknown').astype(str).str.strip()
    
    return df

@st.cache_data
def perform_enhanced_sentiment_analysis(df, method="spaCy"):
    """
    Enhanced sentiment analysis with better accuracy using spaCy or improved TextBlob.
    """
    analyzer = EnhancedSentimentAnalyzer()
    
    if df.empty:
        return df
    
    sentiments = []
    scores = []

    for index, row in df.iterrows():
        rating = row['rating']
        text = row['feedback']

        if pd.notna(rating): # If rating is available, use rating-based sentiment
            if rating >= 4:
                sentiments.append("Positive")
                scores.append(1.0) # High positive score for consistency with buckets
            elif rating == 3:
                sentiments.append("Neutral")
                scores.append(0.0) # Neutral score
            elif rating <= 2:
                sentiments.append("Negative")
                scores.append(-1.0) # High negative score
            else: # Fallback for unexpected rating values
                if method == "spaCy" and spacy_available:
                    pred_sentiment, pred_score = analyzer.analyze_spacy_sentiment(text)
                else:
                    pred_sentiment, pred_score = analyzer.analyze_textblob(text)
                sentiments.append(pred_sentiment)
                scores.append(pred_score)
        else: # If rating is not available, use NLP model
            if method == "spaCy" and spacy_available:
                pred_sentiment, pred_score = analyzer.analyze_spacy_sentiment(text)
            else:
                pred_sentiment, pred_score = analyzer.analyze_textblob(text)
            sentiments.append(pred_sentiment)
            scores.append(pred_score)
    
    df['predicted_sentiment'] = sentiments
    df['sentiment_score'] = scores
    
    return df

@st.cache_data
def create_sentiment_buckets(df):
    """
    Creates categorized sentiment buckets based on sentiment scores for detailed analysis.
    """
    if df.empty:
        return df

    df['sentiment_bucket'] = pd.cut(df['sentiment_score'], 
                                   bins=[-1, -0.5, -0.1, 0.1, 0.5, 1],
                                   labels=['Very Negative', 'Negative', 'Neutral', 'Positive', 'Very Positive'],
                                   include_lowest=True)
    return df

# Enhanced issue categorization with more comprehensive keywords
ISSUE_CATEGORIES_KEYWORDS = {
    "Bilty/LR Issues": ["bilti", "lr", "builty", "billtee", "lorry receipt", "bilty number", "lr copy"],
    "Payment Issues": ["payment", "charges", "money", "wallet", "credit", "tds", "extra", "amount", "refund", "billing", "invoice", "cost", "price", "fee"],
    "Invoice Issues": ["invoice", "gst", "bill", "billing", "wrong", "missing", "change name", "tax", "receipt", "document"],
    "Driver Behavior/Availability": ["driver", "picking call", "rude", "reached", "extra money", "tripal", "not responding", "absent", "behavior", "unprofessional", "late", "attitude"],
    "Stage Change Issues": ["trip not started", "stage change", "loaded", "not moving", "started", "moving", "status", "update", "tracking"],
    "Transit/Halt Issues": ["transit halt", "stopped", "halt", "not running", "police", "checkpoint", "delay", "stuck"],
    "Location/Tracking Issues": ["location", "gps", "tracking", "not updated", "showing wrong", "coordinates", "map", "position"],
    "E-Way Bill Issues": ["e-way bill", "eway", "eway bill", "ewb", "electronic way bill"],
    "Delay Issues": ["delay", "late", "dispatch", "timing", "schedule", "slow", "waiting"],
    "Damage/Material Issues": ["damage", "material", "scrap", "unloading", "shortage", "broken", "loss", "quality"],
    "Cancellation Issues": ["cancel", "cancelled", "cancel vehicle", "booking cancelled", "trip cancelled", "abort"],
    "Other Vehicle Issues": ["vehicle", "height", "number plate", "tyres", "unfit", "overloaded", "truck", "capacity", "condition"],
    "POD Issues": ["pod", "proof of delivery", "delivery proof", "confirmation"],
    "OTP Issues": ["otp", "one time password", "verification", "code"],
    "Extension Request": ["extension", "extend", "more time", "deadline"],
    "Booking Issues": ["booking", "booked", "reservation", "order", "appointment"],
    "Service Quality": ["poor service", "bad experience", "unprofessional", "not satisfied", "quality", "service"],
    "Customer Support": ["call", "connect", "issue resolved", "support", "response", "help", "assistance"],
    "App/Technical Issues": ["app", "application", "technical", "bug", "error", "crash", "loading", "login", "system"],
    "Other/Miscellaneous": ["problem", "help", "insurance", "address", "pincode", "resolved", "issue", "concern"]
}

# Create enhanced keyword mapping
KEYWORD_TO_CATEGORY = {}
for category, keywords in ISSUE_CATEGORIES_KEYWORDS.items():
    for keyword in keywords:
        KEYWORD_TO_CATEGORY[keyword.lower()] = category

def get_enhanced_top_problems(df, n=10, ngram_max_length=4):
    """
    Enhanced problem extraction with better deduplication and accuracy.
    """
    if df.empty:
        return pd.DataFrame(columns=['Problem', 'Count', 'Distribution (%)'])

    negative_feedback = df[df['predicted_sentiment'] == 'Negative']['feedback'].tolist()
    if not negative_feedback:
        return pd.DataFrame(columns=['Problem', 'Count', 'Distribution (%)'])

    # Enhanced text preprocessing
    analyzer = EnhancedSentimentAnalyzer()
    cleaned_feedback_texts = [analyzer.preprocess_text(fb) for fb in negative_feedback]
    
    # Enhanced stop words
    stop_words_set = set(stopwords.words('english'))
    enhanced_custom_stop_words = {
        'app', 'service', 'driver', 'phone', 'customer', 'team', 'ride', 'one', 'get', 'would', 'really', 'back', 
        'much', 'go', 'like', 'just', 'can', 'even', 'still', 'know', 'dont', 'didnt', 'always', 'also', 'never', 
        'nothing', 'please', 'call', 'want', 'make', 'think', 'need', 'issue', 'problem', 'support', 'time', 'day', 
        'use', 'try', 'way', 'thing', 'people', 'good', 'bad', 'great', 'every', 'us', 'said', 'found', 'getting', 
        'not', 'will', 'vehicle', 'vehical', 'gadi', 'truck', 'rupees', 'rs', 'money', 'sir', 'mam', 'kindly', 
        'pls', 'plz', 'share', 'provide', 'update', 'check', 'due', 'tell', 'give', 'take', 'see', 'come', 'well', 
        'may', 'could', 'should', 'would', 'might', 'must', 'shall', 'let', 'made', 'make', 'say', 'said', 'told'
    } 
    stop_words_set.update(enhanced_custom_stop_words)

    category_counts = Counter()
    
    # Sort keywords by length for better matching
    sorted_keywords = sorted(KEYWORD_TO_CATEGORY.keys(), key=len, reverse=True)
    feedback_categorized_flags = [False] * len(cleaned_feedback_texts)

    for i, feedback_text in enumerate(cleaned_feedback_texts):
        matched_categories = set()
        for keyword in sorted_keywords:
            if keyword in feedback_text.lower():
                category = KEYWORD_TO_CATEGORY[keyword]
                if category not in matched_categories:
                    category_counts[category] += 1
                    matched_categories.add(category)
        if matched_categories:
            feedback_categorized_flags[i] = True

    # Enhanced n-gram extraction for uncategorized feedback
    uncategorized_ngrams = Counter()
    seen_ngrams = set()  # To avoid duplicates
    
    for i, feedback_text in enumerate(cleaned_feedback_texts):
        if not feedback_categorized_flags[i]:
            words = word_tokenize(feedback_text.lower())
            filtered_words = [word for word in words 
                            if word.isalpha() and word not in stop_words_set and len(word) > 2]
            
            for length in range(1, ngram_max_length + 1):
                for j in range(len(filtered_words) - length + 1):
                    ngram = ' '.join(filtered_words[j:j+length])
                    
                    # Enhanced deduplication
                    ngram_normalized = ngram.strip().lower()
                    if (ngram_normalized not in seen_ngrams and 
                        len(ngram_normalized) > 3 and
                        not any(keyword in ngram_normalized for keyword in KEYWORD_TO_CATEGORY.keys())):
                        
                        uncategorized_ngrams[ngram] += 1
                        seen_ngrams.add(ngram_normalized)

    # Combine results
    final_problems = []
    
    # Add categorized problems
    for category, count in category_counts.most_common():
        final_problems.append({'Problem': category, 'Count': count})
    
    # Add top uncategorized problems
    added_problems = {item['Problem'].lower() for item in final_problems}
    
    for problem, count in uncategorized_ngrams.most_common():
        if problem.lower() not in added_problems and len(final_problems) < n:
            final_problems.append({'Problem': problem, 'Count': count})
            added_problems.add(problem.lower())

    if not final_problems:
        return pd.DataFrame(columns=['Problem', 'Count', 'Distribution (%)'])

    # Sort and calculate distribution
    final_problems = sorted(final_problems, key=lambda x: x['Count'], reverse=True)[:n]
    total_count = sum(item['Count'] for item in final_problems)
    
    results_df = pd.DataFrame(final_problems)
    results_df['Distribution (%)'] = (results_df['Count'] / total_count) * 100 if total_count > 0 else 0
    
    return results_df.sort_values(by=['Count', 'Problem'], ascending=[False, True])

@st.cache_data
def identify_super_users(df):
    """
    Identifies 'Super Happy' and 'Super Sad' users based on combined sentiment, rating, and frequency.
    Assumes 'user_id' column exists.
    """
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

    # Define thresholds
    SUPER_HAPPY_MIN_FEEDBACK = 5
    SUPER_SAD_MIN_FEEDBACK = 3
    POSITIVE_SENTIMENT_THRESHOLD = 0.1
    NEGATIVE_SENTIMENT_THRESHOLD = -0.1

    user_summary['user_type'] = 'Normal'

    # Super Happy Logic
    explicitly_happy = (user_summary['has_positive_sentiment'] == True) & \
                       (user_summary['has_positive_rating'] == True)

    consistently_good_engaged = (user_summary['feedback_count'] >= SUPER_HAPPY_MIN_FEEDBACK) & \
                                (user_summary['has_negative_rating'] == False) & \
                                (user_summary['avg_sentiment_score'] > POSITIVE_SENTIMENT_THRESHOLD)

    user_summary.loc[explicitly_happy | consistently_good_engaged, 'user_type'] = 'Super Happy'

    # Super Sad Logic
    explicitly_sad = (user_summary['has_negative_sentiment'] == True) & \
                     (user_summary['has_negative_rating'] == True)

    consistently_bad_engaged = (user_summary['feedback_count'] >= SUPER_SAD_MIN_FEEDBACK) & \
                               (user_summary['has_negative_rating'] == True) & \
                               (user_summary['avg_sentiment_score'] < NEGATIVE_SENTIMENT_THRESHOLD)
    
    user_summary.loc[(explicitly_sad | consistently_bad_engaged) & (user_summary['user_type'] != 'Super Happy'), 'user_type'] = 'Super Sad'

    user_type_counts = user_summary['user_type'].value_counts().to_dict()
    
    # Ensure all categories are present
    final_counts = {'Super Happy': 0, 'Super Sad': 0, 'Normal': 0}
    final_counts.update(user_type_counts)

    return final_counts, user_summary

# --- Enhanced Dashboard View Functions ---

def create_enhanced_common_sections(df, view_title, description, include_wordclouds=True):
    """
    Enhanced common sections with total feedback line in graphs and word clouds.
    """
    st.markdown(f'<h2 class="text-2xl md:text-3xl font-semibold text-slate-800 text-center mb-6">{view_title}</h2>', unsafe_allow_html=True)
    st.markdown(f"<p class='text-slate-600 text-center mb-8'>{description}</p>", unsafe_allow_html=True)

    if df.empty:
        st.warning("No data available for the selected filters. Please adjust your date range or check the data source.")
        return

    col1, col2, col3, col4 = st.columns(4)
    
    total_feedback = len(df)
    avg_rating = df['rating'].mean() if not df['rating'].isna().all() else 0
    escalated_count = df['escalated_flag'].sum()
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
                        title=f"Sentiment Distribution (Total: {total_feedback:,})",
                        color='predicted_sentiment',
                        color_discrete_map=GLOBAL_SENTIMENT_COLOR_MAP_GENERAL)
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        fig_pie.update_layout(height=400, margin=dict(l=0, r=0, t=50, b=0))
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        if 'time_period' in df.columns and not df['time_period'].isna().all() and df['time_period'].nunique() > 1:
            daily_sentiment = df.groupby(['time_period', 'predicted_sentiment']).size().unstack(fill_value=0)
            daily_total = df.groupby('time_period').size()
            
            # Ensure all sentiment columns exist
            for sentiment_type in ['Positive', 'Negative', 'Neutral']:
                if sentiment_type not in daily_sentiment.columns:
                    daily_sentiment[sentiment_type] = 0

            daily_sentiment = daily_sentiment.sort_index()
            daily_total = daily_total.sort_index()

            # Create enhanced time series plot with total feedback line
            fig_time = go.Figure()
            
            # Add sentiment lines
            for sentiment in ['Positive', 'Negative', 'Neutral']:
                if sentiment in daily_sentiment.columns:
                    fig_time.add_trace(go.Scatter(
                        x=daily_sentiment.index,
                        y=daily_sentiment[sentiment],
                        mode='lines+markers',
                        name=sentiment,
                        line=dict(color=GLOBAL_SENTIMENT_COLOR_MAP_GENERAL.get(sentiment, '#000000'))
                    ))
            
            # Add total feedback line
            fig_time.add_trace(go.Scatter(
                x=daily_total.index,
                y=daily_total.values,
                mode='lines+markers',
                name='Total Feedback',
                line=dict(color='#1f77b4', width=3, dash='dash'),
                yaxis='y2'
            ))
            
            fig_time.update_layout(
                title=f"Sentiment Trends Over Time (Total: {total_feedback:,})",
                xaxis_title="Time Period",
                yaxis=dict(title="Sentiment Count", side='left'),
                yaxis2=dict(title="Total Feedback", side='right', overlaying='y'),
                height=400,
                margin=dict(l=0, r=0, t=50, b=0),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            st.plotly_chart(fig_time, use_container_width=True)
        else:
            st.info("Insufficient data points or 'time_period' not available for sentiment trends in this view.")
    
    st.markdown("---")

    if include_wordclouds:
        st.markdown("<h3 class='text-xl md:text-2xl font-semibold text-slate-700 mb-4'>📝 Common Issues & What Customers Love</h3>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        
        # Enhanced stop words for better word clouds
        analyzer = EnhancedSentimentAnalyzer()
        stop_words_set = set(stopwords.words('english'))
        enhanced_stop_words = {'app', 'service,'driver', 'phone', 'customer', 'team', 'ride', 'one', 'get', 'would', 'really', 'back',
            'much', 'go', 'like', 'just', 'can', 'even', 'still', 'know', 'dont', 'didnt', 'always',
            'also', 'never', 'nothing', 'please', 'call', 'want', 'make', 'think', 'need', 'issue',
            'problem', 'support', 'time', 'day', 'use', 'try', 'way', 'thing', 'people', 'good', 'bad',
            'great', 'every', 'us', 'said', 'found', 'getting', 'not', 'will', 'vehicle', 'vehical',
            'gadi', 'truck', 'rupees', 'rs', 'money', 'sir', 'mam', 'kindly', 'pls', 'plz', 'share',
            'provide', 'update', 'check', 'due', 'tell', 'give', 'take', 'see', 'come', 'well', 'may',
            'could', 'should', 'would', 'might', 'must', 'shall', 'let', 'made', 'make', 'say', 'said',
            'told', 'thank', 'thanks', 'ok', 'okay', 'yes', 'no', 'hi', 'hello'
        }
        stop_words_set.update(enhanced_stop_words)

        with col1:
            # Problems word cloud
            negative_text = ' '.join(df[df['predicted_sentiment'] == 'Negative']['feedback'].astype(str))
            if negative_text.strip():
                wordcloud = WordCloud(
                    width=400,
                    height=300,
                    background_color='white',
                    stopwords=stop_words_set,
                    colormap='Reds',
                    max_words=100
                ).generate(negative_text)
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis('off')
                st.pyplot(fig)
                plt.close()
            else:
                st.info("No negative feedback to generate word cloud.")

        with col2:
            # Positive feedback word cloud
            positive_text = ' '.join(df[df['predicted_sentiment'] == 'Positive']['feedback'].astype(str))
            if positive_text.strip():
                wordcloud = WordCloud(
                    width=400,
                    height=300,
                    background_color='white',
                    stopwords=stop_words_set,
                    colormap='Greens',
                    max_words=100
                ).generate(positive_text)
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis('off')
                st.pyplot(fig)
                plt.close()
            else:
                st.info("No positive feedback to generate word cloud.")

    st.markdown("---")
    
    # Top Problems Section
    st.markdown("<h3 class='text-xl md:text-2xl font-semibold text-slate-700 mb-4'>🔍 Top Problems Identified</h3>", unsafe_allow_html=True)
    top_problems = get_enhanced_top_problems(df, n=10)
    if not top_problems.empty:
        # Create horizontal bar chart
        fig_problems = px.bar(
            top_problems,
            x='Count',
            y='Problem',
            orientation='h',
            title=f"Top 10 Problems Identified from Negative Feedback",
            color='Count',
            color_continuous_scale='Reds',
            text='Distribution (%)'
        )
        fig_problems.update_layout(
            height=500,
            showlegend=False,
            yaxis={'categoryorder': 'total ascending'},
            margin=dict(l=0, r=0, t=50, b=0)
        )
        fig_problems.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        st.plotly_chart(fig_problems, use_container_width=True)
    else:
        st.info("No problems identified for the selected filters.")

# --- Main App ---
def main():
    # Sidebar
    st.sidebar.markdown("### 📊 PulsePoint Filters")
    
    # File uploader in sidebar
    uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=['csv'])
    
    if uploaded_file is not None:
        df = load_and_process_data(uploaded_file)
        st.sidebar.success("✅ File uploaded successfully!")
    else:
        st.sidebar.info("Using default sample data...")
        try:
            df = load_and_process_data("data_sentiments - Sheet1.csv")
        except FileNotFoundError:
            st.error("⚠️ Default file 'data_sentiments - Sheet1.csv' not found. Please upload a CSV file.")
            return

    if df.empty:
        st.error("⚠️ No data loaded. Please check your file.")
        return

    # Date range filter
    if 'timestamp' in df.columns and not df['timestamp'].isna().all():
        min_date = df['timestamp'].min().date()
        max_date = df['timestamp'].max().date()
        
        date_range = st.sidebar.date_input(
            "Select Date Range",
            value=[min_date, max_date],
            min_value=min_date,
            max_value=max_date,
            key='date_range'
        )
        
        if len(date_range) == 2:
            df = df[(df['timestamp'].dt.date >= date_range[0]) & 
                   (df['timestamp'].dt.date <= date_range[1])]

    # Customer type filter
    customer_types = df['customer_type'].unique()
    selected_customer_types = st.sidebar.multiselect(
        "Filter by Customer Type",
        options=customer_types,
        default=customer_types
    )
    df = df[df['customer_type'].isin(selected_customer_types)]

    # Sentiment method selection
    sentiment_method = st.sidebar.radio(
        "Sentiment Analysis Method",
        ["spaCy (Enhanced)", "TextBlob"],
        key='sentiment_method'
    )
    
    method = "spaCy" if "spaCy" in sentiment_method else "TextBlob"

    # Perform sentiment analysis
    with st.spinner("🔄 Analyzing sentiments..."):
        df = perform_enhanced_sentiment_analysis(df, method=method)
        df = create_sentiment_buckets(df)

    # Add time_period column for grouping
    if 'timestamp' in df.columns and not df['timestamp'].isna().all():
        df['time_period'] = df['timestamp'].dt.to_period('D').astype(str)

    # Main dashboard
    st.markdown('<h1 class="main-header">📊 PulsePoint - Voice of the Customer</h1>', unsafe_allow_html=True)
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🎯 Deep Dive", "👥 Super Users", "📈 Advanced Analytics"])

    with tab1:
        create_enhanced_common_sections(
            df, 
            "📊 Overview Dashboard", 
            "High-level insights into customer sentiment and feedback trends"
        )

    with tab2:
        # Deep dive by sentiment
        sentiment_filter = st.selectbox("Select Sentiment to Deep Dive", ["All", "Positive", "Negative", "Neutral"], key='deep_dive_sentiment')
        
        if sentiment_filter != "All":
            filtered_df = df[df['predicted_sentiment'] == sentiment_filter]
        else:
            filtered_df = df
        
        create_enhanced_common_sections(
            filtered_df,
            f"🎯 Deep Dive into {sentiment_filter} Feedback",
            f"Detailed analysis of {sentiment_filter.lower()} customer feedback",
            include_wordclouds=True
        )

    with tab3:
        # Super users analysis
        st.markdown('<h2 class="text-2xl md:text-3xl font-semibold text-slate-800 text-center mb-6">👥 Super User Analysis</h2>', unsafe_allow_html=True)
        
        super_user_counts, super_user_df = identify_super_users(df)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card sentiment-positive">
                <h3>😊 Super Happy</h3>
                <h2>{super_user_counts['Super Happy']}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card sentiment-negative">
                <h3>😢 Super Sad</h3>
                <h2>{super_user_counts['Super Sad']}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card sentiment-neutral">
                <h3>👥 Normal</h3>
                <h2>{super_user_counts['Normal']}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        if not super_user_df.empty:
            # Display super user details
            st.markdown("### Super User Details")
            st.dataframe(super_user_df.sort_values('user_type'), use_container_width=True)
            
            # Sentiment distribution for super users
            fig_super = px.pie(
                values=list(super_user_counts.values()),
                names=list(super_user_counts.keys()),
                title="Super User Distribution",
                color=list(super_user_counts.keys()),
                color_discrete_map={
                    'Super Happy': '#00CC96',
                    'Super Sad': '#EF553B',
                    'Normal': '#FFA15A'
                }
            )
            st.plotly_chart(fig_super, use_container_width=True)

    with tab4:
        # Advanced analytics
        st.markdown('<h2 class="text-2xl md:text-3xl font-semibold text-slate-800 text-center mb-6">📈 Advanced Analytics</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Sentiment bucket analysis
            st.markdown("### Sentiment Buckets Analysis")
            bucket_counts = df['sentiment_bucket'].value_counts()
            fig_buckets = px.bar(
                x=bucket_counts.index,
                y=bucket_counts.values,
                color=bucket_counts.index,
                color_discrete_map=GLOBAL_SENTIMENT_COLOR_MAP,
                title="Detailed Sentiment Distribution"
            )
            fig_buckets.update_layout(xaxis_title="Sentiment", yaxis_title="Count")
            st.plotly_chart(fig_buckets, use_container_width=True)
        
        with col2:
            # Rating vs sentiment correlation
            st.markdown("### Rating vs Sentiment Correlation")
            rating_sentiment = df.groupby(['rating', 'predicted_sentiment']).size().reset_index(name='count')
            fig_heatmap = px.density_heatmap(
                rating_sentiment,
                x='rating',
                y='predicted_sentiment',
                z='count',
                title="Rating vs Sentiment Heatmap"
            )
            st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # Customer type analysis
        st.markdown("### Customer Type Analysis")
        customer_analysis = df.groupby(['customer_type', 'predicted_sentiment']).size().unstack(fill_value=0)
        fig_customer = px.bar(
            customer_analysis.reset_index().melt(id_vars='customer_type'),
            x='customer_type',
            y='value',
            color='predicted_sentiment',
            color_discrete_map=GLOBAL_SENTIMENT_COLOR_MAP_GENERAL,
            title="Sentiment by Customer Type",
            barmode='group'
        )
        st.plotly_chart(fig_customer, use_container_width=True)

if __name__ == "__main__":
    main()
