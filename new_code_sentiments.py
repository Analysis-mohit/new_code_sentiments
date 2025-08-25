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

# Download NLTK stopwords if not already downloaded
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('punkt_tab') # Ensures punkt_tab is downloaded if missing

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

# Apply custom CSS for enhanced visual styling
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global font styling */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Styling for the main header of the dashboard */
    .main-header {
        font-size: 3.5rem;
        font-weight: 700;
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
        text-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* Enhanced metric cards with glassmorphism effect */
    .metric-card {
        background: linear-gradient(135deg, rgba(255,255,255,0.1), rgba(255,255,255,0.05));
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.18);
        padding: 1.5rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
        transition: all 0.3s ease;
        background-image: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(31, 38, 135, 0.5);
    }
    
    /* Enhanced sentiment cards */
    .sentiment-positive {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        box-shadow: 0 8px 32px rgba(17, 153, 142, 0.3);
    }
    
    .sentiment-negative {
        background: linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%);
        box-shadow: 0 8px 32px rgba(255, 65, 108, 0.3);
    }
    
    .sentiment-neutral {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        color: #333;
        box-shadow: 0 8px 32px rgba(255, 236, 210, 0.3);
    }
    
    /* Enhanced section headers */
    .section-header {
        font-size: 2rem;
        font-weight: 600;
        color: #2d3748;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid;
        border-image: linear-gradient(135deg, #667eea 0%, #764ba2 100%) 1;
    }
    
    /* Custom card container */
    .custom-card {
        background: linear-gradient(135deg, rgba(255,255,255,0.9), rgba(255,255,255,0.7));
        border-radius: 15px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        border: 1px solid rgba(255,255,255,0.2);
    }
    
    /* Enhanced dataframe styling */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(135deg, #f7fafc 0%, #edf2f7 100%);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Raw data section styling */
    .raw-data-header {
        background: linear-gradient(135deg, #4299e1 0%, #3182ce 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px 10px 0 0;
        font-size: 1.5rem;
        font-weight: 600;
        text-align: center;
    }
    
    .filter-container {
        background: rgba(66, 153, 225, 0.05);
        border: 1px solid rgba(66, 153, 225, 0.1);
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    /* Animation keyframes */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .fade-in {
        animation: fadeInUp 0.8s ease-out;
    }
    
    /* Chart container styling */
    .chart-container {
        position: relative;
        width: 100%;
        margin: 1rem 0;
        padding: 1rem;
        background: rgba(255,255,255,0.8);
        border-radius: 15px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
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

class SentimentAnalyzer:
    """
    A class to perform sentiment analysis using TextBlob or a Hugging Face Transformer model.
    """
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
        """
        Analyzes sentiment using TextBlob.
        Returns sentiment label ("Positive", "Negative", "Neutral") and polarity score.
        """
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
        """
        Analyzes sentiment using a Hugging Face Transformer model.
        Falls back to TextBlob if transformer is not initialized or fails.
        """
        if not self.transformer_sentiment or pd.isna(text) or text == "":
            return self.analyze_textblob(text)
        
        try:
            # Truncate text for model input, as some models have input length limits
            result = self.transformer_sentiment(str(text)[:512]) 
            label = result[0]['label']
            score = result[0]['score']
            
            if label in ['LABEL_2', 'POSITIVE']:
                return "Positive", score
            elif label in ['LABEL_0', 'NEGATIVE']:
                return "Negative", -score # Transformer scores are positive for negative label, adjust
            else:
                return "Neutral", 0.0
        except Exception as e:
            st.warning(f"Transformer analysis failed for a text ({str(text)[:50]}...): {e}. Using TextBlob.")
            return self.analyze_textblob(text)

@st.cache_data
def load_and_process_data(file_path):
    """
    Loads a CSV file from the given path and preprocesses the data.
    """
    df = pd.read_csv("sentiment_data_25aug2 - Sheet1.csv")
    
    df.columns = df.columns.str.strip()
    
    # Enhanced timestamp column detection - now includes 'timestamp_'
    timestamp_columns = ['timestamp_', 'timestam', 'timestamp', 'date', 'created_at', 'datetime']
    timestamp_col = None
    
    for col in timestamp_columns:
        if col in df.columns:
            timestamp_col = col
            break
    
    if timestamp_col:
        df['timestamp'] = pd.to_datetime(df[timestamp_col], errors='coerce')
        st.success(f"✅ Found timestamp column: '{timestamp_col}' - Time filtering enabled!")
    else:
        st.warning("No timestamp column found. Time-series analysis will be limited.")
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
def perform_sentiment_analysis(df, method="TextBlob"):
    """
    Turns sentiment analysis on the 'feedback' column of the DataFrame.
    Prioritizes sentiment based on 'rating' if available (5,4=Positive; 3=Neutral; 1,2=Negative).
    """
    analyzer = SentimentAnalyzer()
    
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
            else: # Fallback for unexpected rating values (e.g., if rating is 0 or > 5)
                pred_sentiment, pred_score = (analyzer.analyze_transformer(text) if method == "Transformer" and transformer_available else analyzer.analyze_textblob(text))
                sentiments.append(pred_sentiment)
                scores.append(pred_score)
        else: # If rating is not available, use NLP model
            pred_sentiment, pred_score = (analyzer.analyze_transformer(text) if method == "Transformer" and transformer_available else analyzer.analyze_textblob(text))
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

# Knowledge base for issue categorization (Optimized/Simplified)
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


# Create a flattened list of keywords for quick checking and to avoid re-matching
# and a reverse map to get the category from a keyword
KEYWORD_TO_CATEGORY = {}
for category, keywords in ISSUE_CATEGORIES_KEYWORDS.items():
    for keyword in keywords:
        KEYWORD_TO_CATEGORY[keyword.lower()] = category # Ensure keywords are stored in lowercase

def get_top_n_ngrams_from_negative_feedback(df, n=10, ngram_max_length=4):
    """
    Extracts and ranks top N-grams (problems) from negative feedback,
    categorizing them based on predefined keywords.
    Prioritizes longer, more specific N-grams to make categories distinct.
    """
    if df.empty:
        return pd.DataFrame(columns=['Problem', 'Count', 'Distribution (%)'])

    negative_feedback = df[df['predicted_sentiment'] == 'Negative']['feedback'].tolist()
    if not negative_feedback:
        return pd.DataFrame(columns=['Problem', 'Count', 'Distribution (%)'])

    # Clean and lowercase all feedback texts once
    cleaned_feedback_texts = [re.sub(r'[^a-zA-Z0-9\s]', '', fb.lower()) for fb in negative_feedback]
    
    stop_words_set = set(stopwords.words('english'))
    custom_stop_words = {
        'app', 'service', 'driver', 'phone', 'customer', 'team', 'ride', 'one', 'get', 'would', 'really', 'back', 'much', 'go', 'like', 'just', 'can', 'even', 'still', 'know', 'dont', 'didnt', 'always', 'also', 'never', 'nothing', 'please', 'call', 'want', 'make', 'think', 'need', 'issue', 'problem', 'support', 'time', 'day', 'use', 'try', 'way', 'thing', 'people', 'good', 'bad', 'great', 'every', 'us', 'said', 'found', 'getting', 'not', 'will', 'vehicle', 'vehical', 'gadi', 'truck', 'rupees', 'rs', 'money', 'sir', 'mam', 'kindly', 'pls', 'plz', 'share', 'provide', 'update', 'check', 'due'
    } 
    stop_words_set.update(custom_stop_words)

    category_counts = Counter()
    
    # Sort keywords by length in descending order to prioritize longer, more specific matches
    # This avoids matching "limit" when "credit limit" is present
    sorted_keywords = sorted(KEYWORD_TO_CATEGORY.keys(), key=len, reverse=True)

    feedback_categorized_flags = [False] * len(cleaned_feedback_texts)

    for i, feedback_text in enumerate(cleaned_feedback_texts):
        matched_categories_for_this_feedback = set() # Track categories matched within a single feedback entry
        for keyword in sorted_keywords:
            if keyword in feedback_text:
                category = KEYWORD_TO_CATEGORY[keyword]
                if category not in matched_categories_for_this_feedback:
                    category_counts[category] += 1
                    matched_categories_for_this_feedback.add(category)
        if matched_categories_for_this_feedback:
            feedback_categorized_flags[i] = True # Mark this feedback entry as categorized

    # Now, find top N-grams from the *un-categorized* feedback for specific problems
    uncategorized_ngrams_counter = Counter()
    for i, feedback_text in enumerate(cleaned_feedback_texts):
        if not feedback_categorized_flags[i]: # Only process if not already fully categorized by keywords
            words_in_text = word_tokenize(feedback_text)
            # Filter out stopwords and short words
            filtered_words_in_text = [word for word in words_in_text if word.isalpha() and word not in stop_words_set and len(word) > 2]
            
            # Generate N-grams (up to ngram_max_length)
            for length in range(1, ngram_max_length + 1):
                for j in range(len(filtered_words_in_text) - length + 1):
                    ngram = ' '.join(filtered_words_in_text[j:j+length])
                    
                    # Avoid adding N-grams that are already explicit category keywords or highly similar
                    # This check is crucial for keeping "Other Problems" distinct from predefined categories
                    is_part_of_category_keyword = False
                    for category_keyword in KEYWORD_TO_CATEGORY.keys():
                        if (ngram == category_keyword) or \
                           (ngram in category_keyword and len(ngram) > len(category_keyword) / 2) or \
                           (category_keyword in ngram and len(category_keyword) > len(ngram) / 2):
                            is_part_of_category_keyword = True
                            break

                    if not is_part_of_category_keyword:
                        uncategorized_ngrams_counter[ngram] += 1

    final_problems_list = []
    
    # Add categorized problems first, sorted by count
    for category, count in category_counts.most_common():
        final_problems_list.append({'Problem': category, 'Count': count})
    
    # Add top uncategorized problems, after categorized problems, ensuring uniqueness
    added_problem_texts = {item['Problem'].lower() for item in final_problems_list} # Track problems already added
    
    for problem_text, count in uncategorized_ngrams_counter.most_common():
        if problem_text.lower() not in added_problem_texts:
            final_problems_list.append({'Problem': problem_text, 'Count': count})
            added_problem_texts.add(problem_text.lower())
        
        if len(final_problems_list) >= n:
            break

    if not final_problems_list:
        return pd.DataFrame(columns=['Problem', 'Count', 'Distribution (%)'])

    # Re-sort the combined list by count
    final_problems_list_sorted = sorted(final_problems_list, key=lambda x: x['Count'], reverse=True)
    
    # Take top N
    final_problems_list_sliced = final_problems_list_sorted[:n]

    total_counts_for_dist = sum(item['Count'] for item in final_problems_list_sliced)
    results_df = pd.DataFrame(final_problems_list_sliced)
    results_df['Distribution (%)'] = (results_df['Count'] / total_counts_for_dist) * 100 if total_counts_for_dist > 0 else 0
    results_df = results_df.sort_values(by=['Count', 'Problem'], ascending=[False, True])
    
    return results_df


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
        has_positive_rating=('rating', lambda x: (x >= 4).any()), # Any rating 4 or 5
        has_neutral_rating=('rating', lambda x: (x == 3).any()),
        has_negative_rating=('rating', lambda x: (x <= 2).any())  # Any rating 1 or 2
    ).reset_index()

    # Define thresholds
    SUPER_HAPPY_MIN_FEEDBACK = 5
    SUPER_SAD_MIN_FEEDBACK = 3
    POSITIVE_SENTIMENT_THRESHOLD = 0.1
    NEGATIVE_SENTIMENT_THRESHOLD = -0.1

    user_summary['user_type'] = 'Normal'

    # Super Happy Logic
    # Condition 1: Explicitly Positive
    explicitly_happy = (user_summary['has_positive_sentiment'] == True) & \
                       (user_summary['has_positive_rating'] == True)

    # Condition 2: Consistently Good & Engaged
    consistently_good_engaged = (user_summary['feedback_count'] >= SUPER_HAPPY_MIN_FEEDBACK) & \
                                (user_summary['has_negative_rating'] == False) & \
                                (user_summary['avg_sentiment_score'] > POSITIVE_SENTIMENT_THRESHOLD)

    user_summary.loc[explicitly_happy | consistently_good_engaged, 'user_type'] = 'Super Happy'

    # Super Sad Logic
    # Condition 1: Explicitly Negative
    explicitly_sad = (user_summary['has_negative_sentiment'] == True) & \
                     (user_summary['has_negative_rating'] == True)

    # Condition 2: Consistently Bad & Engaged
    consistently_bad_engaged = (user_summary['feedback_count'] >= SUPER_SAD_MIN_FEEDBACK) & \
                               (user_summary['has_negative_rating'] == True) & \
                               (user_summary['avg_sentiment_score'] < NEGATIVE_SENTIMENT_THRESHOLD)
    
    # Apply Super Sad, but ensure they aren't already marked Super Happy by explicit positive feedback
    # Prioritize "Super Happy" if a user falls into both (e.g., gave one 5-star and one 1-star) - decision based on typical business needs
    user_summary.loc[(explicitly_sad | consistently_bad_engaged) & (user_summary['user_type'] != 'Super Happy'), 'user_type'] = 'Super Sad'


    user_type_counts = user_summary['user_type'].value_counts().to_dict()
    
    # Ensure all categories are present, even if 0
    final_counts = {'Super Happy': 0, 'Super Sad': 0, 'Normal': 0}
    final_counts.update(user_type_counts)

    return final_counts, user_summary

# --- Dashboard View Functions ---

def create_common_sections(df, view_title, description, include_wordclouds=True):
    """
    Helper function to generate common sections for different views.
    """
    st.markdown(f'<div class="section-header fade-in">{view_title}</div>', unsafe_allow_html=True)
    st.markdown(f"<p class='text-slate-600 text-center mb-8 fade-in'>{description}</p>", unsafe_allow_html=True)

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
        <div class="metric-card fade-in">
            <h3>📊 Total Feedback</h3>
            <h2>{total_feedback:,}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card fade-in">
            <h3>⭐ Avg Rating</h3>
            <h2>{avg_rating:.1f}/5</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card fade-in">
            <h3>🚨 Escalated</h3>
            <h2>{escalated_count}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        dominant_sentiment = sentiment_dist.index[0] if not sentiment_dist.empty else "Unknown"
        st.markdown(f"""
        <div class="metric-card fade-in">
            <h3>🎯 Dominant Sentiment</h3>
            <h2>{dominant_sentiment}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("<div class='section-header'>Sentiment Distribution & Trends</div>", unsafe_allow_html=True)
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
        fig_pie.update_layout(height=400, margin=dict(l=0, r=0, t=50, b=0)) # Adjust margins for better fit
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        if 'time_period' in df.columns and not df['time_period'].isna().all() and df['time_period'].nunique() > 1:
            daily_sentiment = df.groupby(['time_period', 'predicted_sentiment']).size().unstack(fill_value=0)
            
            # Ensure all three sentiment columns exist, create if not
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
            st.info("Insufficient data points or 'time_period' not available for sentiment trends in this view. Needs at least two data points for trend analysis.")
    
    st.markdown("---")

    if include_wordclouds:
        st.markdown("<div class='section-header'>📝 Common Issues & What Customers Love</div>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        
        # Define custom stop words, including general terms that don't add much value to word clouds
        stop_words_set = set(stopwords.words('english'))
        custom_stop_words = {
            'app', 'service', 'driver', 'phone', 'customer', 'team', 'ride', 'one', 'get', 'would', 'really', 'back', 'much', 'go', 'like', 'just', 'can', 'even', 'still', 'know', 'dont', 'didnt', 'always', 'also', 'never', 'nothing', 'please', 'call', 'want', 'make', 'think', 'need', 'issue', 'problem', 'support', 'time', 'day', 'use', 'try', 'way', 'thing', 'people', 'good', 'bad', 'great', 'every', 'us', 'said', 'found', 'getting', 'not', 'will', 'vehicle', 'vehical', 'gadi', 'truck', 'rupees', 'rs', 'money', 'sir', 'mam', 'kindly', 'pls', 'plz', 'share', 'provide', 'update', 'check', 'due'
        } 
        stop_words_set.update(custom_stop_words)

        with col1:
            negative_feedback_text = ' '.join(df[df['predicted_sentiment'] == 'Negative']['feedback'].tolist())
            if negative_feedback_text.strip():
                cleaned_text = re.sub(r'[^a-zA-Z\s]', '', negative_feedback_text.lower())
                if cleaned_text.strip():
                    fig, ax = plt.subplots(figsize=(5, 4))
                    wordcloud = WordCloud(width=400, height=300, 
                                        background_color='white',
                                        colormap='Reds',
                                        max_words=30,
                                        stopwords=stop_words_set).generate(cleaned_text)
                    
                    ax.imshow(wordcloud, interpolation='bilinear')
                    ax.axis('off')
                    ax.set_title('Common Issues (Negative Feedback)', fontsize=16, fontweight='bold', color='red')
                    st.pyplot(fig)
                    plt.close(fig)
                else:
                    st.info("No significant negative feedback text to generate word cloud.")
            else:
                st.info("No negative feedback found.")
        
        with col2:
            positive_feedback_text = ' '.join(df[df['predicted_sentiment'] == 'Positive']['feedback'].tolist())
            if positive_feedback_text.strip():
                cleaned_text = re.sub(r'[^a-zA-Z\s]', '', positive_feedback_text.lower())
                if cleaned_text.strip():
                    fig, ax = plt.subplots(figsize=(5, 4))
                    wordcloud = WordCloud(width=400, height=300, 
                                        background_color='white',
                                        colormap='Greens',
                                        max_words=30,
                                        stopwords=stop_words_set).generate(cleaned_text)
                    
                    ax.imshow(wordcloud, interpolation='bilinear')
                    ax.axis('off')
                    ax.set_title('What Customers Love (Positive Feedback)', fontsize=16, fontweight='bold', color='green')
                    st.pyplot(fig)
                    plt.close(fig)
                else:
                    st.info("No significant positive feedback text to generate word cloud.")
            else:
                st.info("No positive feedback found.")
        st.markdown("---")


    st.markdown("<div class='section-header'>🎯 Top 10 Problems Faced by Users</div>", unsafe_allow_html=True)
    top_problems_df = get_top_n_ngrams_from_negative_feedback(df)
    if not top_problems_df.empty:
        st.dataframe(top_problems_df.style.format({"Distribution (%)": "{:.1f}%"}), use_container_width=True)
        fig_problems = px.bar(top_problems_df.head(10).sort_values('Distribution (%)', ascending=True), # Sort for horizontal bar chart
                            x='Distribution (%)', y='Problem', orientation='h',
                            title='Distribution of Top Problems from Negative Feedback',
                            color_discrete_sequence=px.colors.qualitative.Dark24)
        fig_problems.update_layout(height=500, xaxis_title="Distribution (%)", yaxis_title="Problem/Issue", margin=dict(l=0, r=0, t=50, b=0))
        st.plotly_chart(fig_problems, use_container_width=True)
    else:
        st.info("No negative feedback or identifiable problems found in the selected data range.")

def create_overview_dashboard(df):
    """
    Generates the overall sentiment analysis dashboard with key metrics and aggregated visualizations.
    """
    create_common_sections(df, "📈 Overall Sentiment Trends & Key Insights", "This section provides a holistic view of customer sentiment across all feedback sources within the selected time period. Understand the general sentiment distribution, source-specific contributions, and identify top recurring problems.")
    
    st.markdown("---")
    # --- Super User Analysis Section ---
    st.markdown("<div class='section-header'>🎉 User Segmentation: Happy vs. Sad Users 😞</div>", unsafe_allow_html=True)
    user_counts, user_df = identify_super_users(df)
    
    if user_df.empty or df['user_id'].nunique() <= 1:
        st.info("User-level analysis requires 'user_id' column with multiple unique users in the data. Please check your data or filters.")
    else:
        user_col1, user_col2, user_col3 = st.columns(3)
        with user_col1:
            st.markdown(f"""
            <div class="metric-card sentiment-positive fade-in">
                <h3>Super Happy Users</h3>
                <h2>{user_counts.get('Super Happy', 0):,}</h2>
            </div>
            """, unsafe_allow_html=True)
        with user_col2:
            st.markdown(f"""
            <div class="metric-card sentiment-neutral fade-in">
                <h3>Normal Users</h3>
                <h2>{user_counts.get('Normal', 0):,}</h2>
            </div>
            """, unsafe_allow_html=True)
        with user_col3:
            st.markdown(f"""
            <div class="metric-card sentiment-negative fade-in">
                <h3>Super Sad Users</h3>
                <h2>{user_counts.get('Super Sad', 0):,}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True) # Add some space

        # Optionally display a sample of super happy/sad users
        st.subheader("Sample of Super Happy Users")
        super_happy_sample = user_df[user_df['user_type'] == 'Super Happy'].head(5)
        if not super_happy_sample.empty:
            st.dataframe(super_happy_sample[['user_id', 'feedback_count', 'avg_rating', 'avg_sentiment_score']].style.format({"avg_rating": "{:.1f}", "avg_sentiment_score": "{:.2f}"}), use_container_width=True)
        else:
            st.info("No Super Happy users identified in this period.")

        st.subheader("Sample of Super Sad Users")
        super_sad_sample = user_df[user_df['user_type'] == 'Super Sad'].head(5)
        if not super_sad_sample.empty:
            st.dataframe(super_sad_sample[['user_id', 'feedback_count', 'avg_rating', 'avg_sentiment_score']].style.format({"avg_rating": "{:.1f}", "avg_sentiment_score": "{:.2f}"}), use_container_width=True)
        else:
            st.info("No Super Sad users identified in this period.")
            
    st.markdown("---") # Separator after super user analysis
    
    st.markdown("<div class='section-header'>Sentiment by Source</div>", unsafe_allow_html=True)
    source_sentiment = pd.crosstab(df['source'], df['predicted_sentiment']).reset_index()
    source_sentiment_melted = source_sentiment.melt(id_vars='source', var_name='sentiment', value_name='count')

    fig_bar = px.bar(source_sentiment_melted,
                    x='source', y='count',
                    color='sentiment',
                    title="Sentiment by Source",
                    color_discrete_map=GLOBAL_SENTIMENT_COLOR_MAP_GENERAL)
    fig_bar.update_layout(barmode='stack', height=400, margin=dict(l=0, r=0, t=50, b=0), xaxis_title="Source", yaxis_title="Count")
    st.plotly_chart(fig_bar, use_container_width=True)


def create_playstore_view(df):
    """
    Generates the PlayStore-specific dashboard view with key metrics and visualizations.
    """
    playstore_df = df[df['source'].str.contains('play', case=False, na=False)]
    
    create_common_sections(playstore_df, "📱 PlayStore Feedback Analysis", "Dive deep into customer sentiment from PlayStore reviews. Understand ratings distribution, detailed sentiment buckets, and identify recent negative feedback for immediate action.")

    if playstore_df.empty:
        return # Common sections will handle the warning

    st.markdown("---")
    st.markdown("<div class='section-header'>Rating Distribution</div>", unsafe_allow_html=True)
    if not playstore_df['rating'].isna().all():
        rating_dist = playstore_df['rating'].value_counts().sort_index()
        rating_df = rating_dist.reset_index()
        rating_df.columns = ['rating', 'count']

        fig_rating = px.bar(rating_df,
                           x='rating', y='count',
                           title="PlayStore Rating Distribution",
                           color='rating',
                           color_continuous_scale="RdYlGn") 
        fig_rating.update_layout(height=400, xaxis_title="Rating", yaxis_title="Count", margin=dict(l=0, r=0, t=50, b=0))
        st.plotly_chart(fig_rating, use_container_width=True)
    else:
        st.info("No rating data available for PlayStore reviews.")
    
    st.markdown("---")
    st.markdown("<div class='section-header'>🔍 Recent Negative PlayStore Feedback (Action Required)</div>", unsafe_allow_html=True)
    negative_ps = playstore_df[playstore_df['predicted_sentiment'] == 'Negative'].head(10) # Show top 10
    if not negative_ps.empty:
        display_cols = ['timestamp', 'feedback', 'rating', 'sentiment_score', 'escalated_flag']
        available_cols = [col for col in display_cols if col in negative_ps.columns]
        st.dataframe(negative_ps[available_cols].sort_values('timestamp', ascending=False), use_container_width=True)
    else:
        st.info("No recent negative PlayStore feedback found.")


def create_trip_feedback_view(df):
    """
    Generates the Trip Feedback-specific dashboard view with key metrics and visualizations.
    """
    trip_df = df[df['source'].str.contains('trip', case=False, na=False)]
    
    create_common_sections(trip_df, "🚗 Trip Feedback Analysis", "Analyze customer feedback specifically related to trip or service experiences. See satisfaction trends, common issues, and rating correlations.")

    if trip_df.empty:
        return # Common sections will handle the warning
    
    st.markdown("---")
    st.markdown("<div class='section-header'>Rating Correlation</div>", unsafe_allow_html=True)
    if not trip_df['rating'].isna().all():
        trip_corr = trip_df.groupby('rating')['sentiment_score'].mean().reset_index()
        fig_corr = px.line(trip_corr,
                          x='rating', y='sentiment_score',
                          title="Trip Rating vs Sentiment Score Correlation",
                          markers=True)
        fig_corr.update_layout(height=400, xaxis_title="Rating", yaxis_title="Average Sentiment Score", margin=dict(l=0, r=0, t=50, b=0))
        st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.info("No rating data available for Trip Feedback.")


def create_escalation_view(df):
    """
    Generates the Escalation-specific dashboard view with key metrics and visualizations.
    """
    escalation_df = df[df['source'].str.contains('Escalation', case=False, na=False)]
    
    create_common_sections(escalation_df, "🚨 Escalation Analysis", "Focus on critical feedback and escalated cases. Understand severity distribution, track escalation trends, and prioritize immediate actions.")

    if escalation_df.empty:
        return # Common sections will handle the warning
    
    st.markdown("---")
    st.markdown("<div class='section-header'>Severity Distribution & Trend</div>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    
    with col1:
        # Define severity based on sentiment score
        severity_labels = []
        for score in escalation_df['sentiment_score']:
            if score < -0.7:
                severity_labels.append('Critical')
            elif score < -0.3:
                severity_labels.append('High')
            elif score < 0:
                severity_labels.append('Medium')
            else:
                severity_labels.append('Low')
        
        # Ensure categories are ordered for correct plotting
        escalation_df['severity'] = pd.Categorical(severity_labels, categories=['Critical', 'High', 'Medium', 'Low'], ordered=True)
        severity_counts = escalation_df['severity'].value_counts().sort_index()
        
        fig_severity = px.bar(x=severity_counts.index, y=severity_counts.values,
                             title="Escalation Severity Distribution",
                             color=severity_counts.index, # Color based on severity level
                             color_discrete_map={ # Specific colors for severity levels
                                 'Critical': 'darkred',
                                 'High': 'red',
                                 'Medium': 'orange',
                                 'Low': 'lightgreen'
                             })
        fig_severity.update_layout(height=400, xaxis_title="Severity Level", yaxis_title="Count", margin=dict(l=0, r=0, t=50, b=0))
        st.plotly_chart(fig_severity, use_container_width=True)
    
    with col2:
        # Escalation trend over time - uses 'time_period' from main df
        if 'time_period' in escalation_df.columns and not escalation_df['time_period'].isna().all() and escalation_df['time_period'].nunique() > 1:
            daily_escalations = escalation_df.groupby('time_period').size().reset_index(name='count')
            if not daily_escalations.empty:
                daily_escalations = daily_escalations.sort_values('time_period')

                fig_trend = px.line(daily_escalations,
                                   x='time_period', y='count',
                                   title="Escalation Trend Over Time",
                                   color_discrete_sequence=['red'])
                fig_trend.update_xaxes(title_text="Time Period")
                fig_trend.update_layout(height=400, margin=dict(l=0, r=0, t=50, b=0))
                st.plotly_chart(fig_trend, use_container_width=True)
            else:
                st.info("No Escalation trends available for the selected time period. Needs at least two data points for trend analysis.")
        else:
            st.info("Insufficient data points or 'time_period' not available for Escalation trends.")
    
    st.markdown("---")
    st.markdown("<div class='section-header'>🚨 Critical Escalations (Immediate Action Required)</div>", unsafe_allow_html=True)
    critical_escalations = escalation_df[escalation_df['sentiment_score'] < -0.7].head(10) # Show top 10
    if not critical_escalations.empty:
        display_cols = ['timestamp', 'feedback', 'rating', 'sentiment_score', 'escalated_flag']
        available_cols = [col for col in display_cols if col in critical_escalations.columns]
        st.dataframe(critical_escalations[available_cols].sort_values('timestamp', ascending=False), use_container_width=True)
    else:
        st.success("✅ No critical escalations found!")


def create_raw_data_view(df):
    """
    Creates a comprehensive raw data view with advanced filtering capabilities.
    """
    st.markdown('<div class="raw-data-header fade-in">🔍 Raw Data Explorer</div>', unsafe_allow_html=True)
    st.markdown("<p class='text-center text-gray-600 mb-6'>Explore and filter your feedback data with advanced search capabilities</p>", unsafe_allow_html=True)
    
    # Advanced Filters
    with st.container():
        st.markdown('<div class="filter-container">', unsafe_allow_html=True)
        st.markdown("### 🎛️ Advanced Filters")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            # Source filter
            sources = ['All'] + list(df['source'].unique()) if 'source' in df.columns else ['All']
            selected_source = st.selectbox("📱 Source", sources, key="raw_source_filter")
        
        with col2:
            # Sentiment filter
            sentiments = ['All'] + list(df['predicted_sentiment'].unique()) if 'predicted_sentiment' in df.columns else ['All']
            selected_sentiment = st.selectbox("😊 Sentiment", sentiments, key="raw_sentiment_filter")
        
        with col3:
            # Rating filter
            if 'rating' in df.columns and not df['rating'].isna().all():
                rating_options = ['All'] + sorted(df['rating'].dropna().unique().astype(int).tolist())
                selected_rating = st.selectbox("⭐ Rating", rating_options, key="raw_rating_filter")
            else:
                selected_rating = 'All'
        
        with col4:
            # Escalated filter
            escalated_options = ['All', 'True', 'False']
            selected_escalated = st.selectbox("🚨 Escalated", escalated_options, key="raw_escalated_filter")
        
        # Text search
        col1, col2 = st.columns([3, 1])
        with col1:
            search_text = st.text_input("🔍 Search in feedback text", placeholder="Enter keywords to search...", key="raw_search_text")
        
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)  # Add spacing
            search_button = st.button("🔍 Search", key="raw_search_button")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Apply filters
    filtered_df = df.copy()
    
    if selected_source != 'All':
        filtered_df = filtered_df[filtered_df['source'] == selected_source]
    
    if selected_sentiment != 'All':
        filtered_df = filtered_df[filtered_df['predicted_sentiment'] == selected_sentiment]
    
    if selected_rating != 'All':
        filtered_df = filtered_df[filtered_df['rating'] == selected_rating]
    
    if selected_escalated != 'All':
        escalated_bool = selected_escalated == 'True'
        filtered_df = filtered_df[filtered_df['escalated_flag'] == escalated_bool]
    
    if search_text:
        filtered_df = filtered_df[filtered_df['feedback'].str.contains(search_text, case=False, na=False)]
    
    # Display results summary
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    
    total_records = len(filtered_df)
    positive_count = len(filtered_df[filtered_df['predicted_sentiment'] == 'Positive']) if 'predicted_sentiment' in filtered_df.columns else 0
    negative_count = len(filtered_df[filtered_df['predicted_sentiment'] == 'Negative']) if 'predicted_sentiment' in filtered_df.columns else 0
    avg_rating = filtered_df['rating'].mean() if 'rating' in filtered_df.columns and not filtered_df['rating'].isna().all() else 0
    
    with col1:
        st.metric("📊 Total Records", f"{total_records:,}")
    
    with col2:
        st.metric("😊 Positive", f"{positive_count:,}")
    
    with col3:
        st.metric("😞 Negative", f"{negative_count:,}")
    
    with col4:
        st.metric("⭐ Avg Rating", f"{avg_rating:.1f}" if avg_rating > 0 else "N/A")
    
    st.markdown("---")
    
    # Display options
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.markdown("### 📋 Filtered Results")
    
    with col2:
        # Records per page
        records_per_page = st.selectbox("Records per page", [10, 25, 50, 100, 500], index=1, key="raw_records_per_page")
    
    with col3:
        # Download button
        if not filtered_df.empty:
            csv = filtered_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="⬇️ Download CSV",
                data=csv,
                file_name=f"filtered_feedback_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    # Pagination
    if not filtered_df.empty:
        total_pages = max(1, (len(filtered_df) - 1) // records_per_page + 1)
        
        if total_pages > 1:
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                page_number = st.selectbox(
                    f"Page (1-{total_pages})", 
                    range(1, total_pages + 1), 
                    key="raw_page_number"
                )
        else:
            page_number = 1
        
        # Calculate start and end indices
        start_idx = (page_number - 1) * records_per_page
        end_idx = min(start_idx + records_per_page, len(filtered_df))
        
        # Display data
        display_df = filtered_df.iloc[start_idx:end_idx].copy()
        
        # Format timestamp for better display
        if 'timestamp' in display_df.columns:
            display_df['timestamp'] = pd.to_datetime(display_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
        
        # Select columns to display
        display_columns = []
        available_columns = display_df.columns.tolist()
        
        # Prioritize important columns
        priority_columns = ['timestamp', 'feedback', 'rating', 'predicted_sentiment', 'sentiment_score', 'source', 'escalated_flag', 'customer_type', 'user_id']
        
        for col in priority_columns:
            if col in available_columns:
                display_columns.append(col)
        
        # Add remaining columns
        for col in available_columns:
            if col not in display_columns:
                display_columns.append(col)
        
        # Display the dataframe with custom styling
        st.dataframe(
            display_df[display_columns],
            use_container_width=True,
            height=600
        )
        
        # Show pagination info
        st.caption(f"Showing records {start_idx + 1}-{end_idx} of {len(filtered_df)} total records")
        
    else:
        st.warning("No records found matching your filter criteria. Please adjust your filters and try again.")
    
    # Quick insights
    if not filtered_df.empty and len(filtered_df) >= 10:
        st.markdown("---")
        st.markdown("<div class='section-header'>📈 Quick Insights from Filtered Data</div>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Sentiment distribution chart
            if 'predicted_sentiment' in filtered_df.columns:
                sentiment_counts = filtered_df['predicted_sentiment'].value_counts()
                fig_sentiment = px.pie(
                    values=sentiment_counts.values,
                    names=sentiment_counts.index,
                    title="Sentiment Distribution (Filtered Data)",
                    color=sentiment_counts.index,
                    color_discrete_map=GLOBAL_SENTIMENT_COLOR_MAP_GENERAL
                )
                fig_sentiment.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig_sentiment, use_container_width=True)
        
        with col2:
            # Rating distribution chart (if available)
            if 'rating' in filtered_df.columns and not filtered_df['rating'].isna().all():
                rating_counts = filtered_df['rating'].value_counts().sort_index()
                fig_rating = px.bar(
                    x=rating_counts.index,
                    y=rating_counts.values,
                    title="Rating Distribution (Filtered Data)",
                    labels={'x': 'Rating', 'y': 'Count'},
                    color=rating_counts.values,
                    color_continuous_scale="RdYlGn"
                )
                st.plotly_chart(fig_rating, use_container_width=True)
            else:
                # Show source distribution instead
                if 'source' in filtered_df.columns:
                    source_counts = filtered_df['source'].value_counts()
                    fig_source = px.bar(
                        x=source_counts.values,
                        y=source_counts.index,
                        title="Source Distribution (Filtered Data)",
                        labels={'x': 'Count', 'y': 'Source'},
                        orientation='h'
                    )
                    st.plotly_chart(fig_source, use_container_width=True)


# --- Main App Logic ---
def main_app():
    # Updated dashboard title
    st.markdown('<h1 class="main-header fade-in">📊 PulsePoint - Voice of the Customer</h1>', unsafe_allow_html=True)
    
    # --- Predefined File Path ---
    csv_file_path = "sentiment_25aug.csv" 
    # --- End Predefined File Path ---

    # Static list for customer types to ensure only Consigner/Operator are explicitly shown
    customer_types_for_filter = ['All', 'Consigner', 'Operator']

    with st.sidebar:
        st.header("⚙️ Configuration")
        
        st.info(f"Data Source: `{csv_file_path}`") 
        
        sentiment_method = st.selectbox(
            "🧠 Sentiment Analysis Method",
            ["TextBlob"] if not transformer_available else ["TextBlob", "Transformer"],
            key="global_sentiment_method", # Unique key for this widget
            help="Choose the sentiment analysis method. Transformer models are more accurate but slower. Requires 'transformers' library."
        )
        
        st.markdown("---")
        st.header("🗓️ Date & Time Filters")
        today = datetime.now().date()
        default_start_date = today - timedelta(days=365)

        start_date = st.date_input("Start Date", value=default_start_date, key="global_start_date")
        end_date = st.date_input("End Date", value=today, key="global_end_date")

        time_granularity = st.selectbox("Time Granularity", ["Daily", "Weekly", "Monthly"], key="global_time_granularity")
        
        st.markdown("---")
        # Customer type filter applies to all views now, as requested.
        selected_customer_type = st.selectbox(
            "👥 Select Customer Type",
            customer_types_for_filter,
            key="global_customer_type_filter",
            help="Filter all dashboard views by selected customer type."
        )
        st.markdown("---")
        st.header("📊 Dashboard Views")
        page_selection = st.radio(
            "Explore Data By:",
            ["📈 Overview", "📱 PlayStore Feedback", "🚗 Trip Feedback", "🚨 Escalations", "🔍 Raw Data"],
            key="page_selection" # Unique key for this widget
        )

        st.markdown("---")
        if st.button("🔄 Refresh Analysis", key="global_refresh_button"):
            st.cache_data.clear()
            st.rerun()
        
        st.markdown("---")
        st.markdown("### 🎯 Sentiment Scoring (TextBlob Example)")
        st.markdown("• **Very Positive**: Score > 0.5")
        st.markdown("• **Positive**: 0.1 to 0.5")
        st.markdown("• **Neutral**: -0.1 to 0.1")
        st.markdown("• **Negative**: -0.5 to -0.1")
        st.markdown("• **Very Negative**: < -0.5")
        st.markdown("---")
        st.markdown("### 🌟 **Rating-Based Sentiment Logic:**")
        st.markdown("• **Rating 5, 4:** ⭐⭐⭐⭐ = **Positive**")
        st.markdown("• **Rating 3:** ⭐⭐⭐ = **Neutral**")
        st.markdown("• **Rating 1, 2:** ⭐⭐ = **Negative**")
        st.markdown("---")
        st.markdown("### 🧑‍🤝‍🧑 **Super User Logic:**")
        st.markdown("• **Super Happy:** Explicitly positive (Positive sentiment + Rating 4/5) OR Consistently good & engaged (>=5 feedback, no bad ratings, avg sentiment > 0.1).")
        st.markdown("• **Super Sad:** Explicitly negative (Negative sentiment + Rating 1/2) OR Consistently bad & engaged (>=3 feedback, at least one bad rating, avg sentiment < -0.1).")

    try:
        with st.spinner(f"📊 Loading and processing your data from {csv_file_path}..."):
            df = load_and_process_data("sentiment_25aug.csv")
            
            # Apply global date filter
            if 'timestamp' in df.columns and not df['timestamp'].isna().all():
                df_filtered = df[(df['timestamp'].dt.date >= start_date) & 
                                 (df['timestamp'].dt.date <= end_date)].copy()
                
                if time_granularity == "Daily":
                    df_filtered['time_period'] = df_filtered['timestamp'].dt.date
                elif time_granularity == "Weekly":
                    df_filtered['time_period'] = df_filtered['timestamp'].dt.to_period('W').astype(str)
                elif time_granularity == "Monthly":
                    df_filtered['time_period'] = df_filtered['timestamp'].dt.to_period('M').astype(str)
            else:
                st.warning("Timestamp column not found or is empty. Date filtering and time trends will not be available.")
                df_filtered = df.copy() 
                df_filtered['time_period'] = 'N/A' 
                
            df_processed_all = perform_sentiment_analysis(df_filtered, method=sentiment_method)
            df_processed_all = create_sentiment_buckets(df_processed_all)

            # Apply the selected customer type filter to all data
            if selected_customer_type != 'All':
                df_to_display = df_processed_all[df_processed_all['customer_type'] == selected_customer_type].copy()
            else:
                df_to_display = df_processed_all.copy()
        
        if df_to_display.empty:
            st.error("No data available after applying initial filters and view selection. Please adjust your date range, check the input CSV file, or try different customer type filters.")
            return # Exit function if no data
        else:
            st.success(f"✅ Processed {len(df_to_display)} feedback records successfully for the selected view, period, and customer type!")
            
            # --- Download Button for Processed Data ---
            if page_selection != "🔍 Raw Data":  # Don't show download button on raw data page (it has its own)
                csv_output = df_to_display.to_csv(index=False).encode('utf-8')
                download_file_name = f"processed_sentiment_data_{page_selection.replace(' ', '_').replace('📈','').replace('📱','').replace('🚗','').replace('🚨','')}_{selected_customer_type}.csv"
                st.download_button(
                    label="⬇️ Download Processed Data (CSV)",
                    data=csv_output,
                    file_name=download_file_name,
                    mime="text/csv",
                    help="Download the current filtered and processed feedback data."
                )
            st.markdown("---")
            # --- End Download Button ---

            # Display summary metrics on the main dashboard area for all pages except Raw Data
            if page_selection != "🔍 Raw Data":
                col1, col2, col3 = st.columns(3)
                with col1:
                    playstore_count = len(df_to_display[df_to_display['source'].str.contains('play_store', case=False, na=False)])
                    st.metric("📱 PlayStore Reviews (Filtered)", playstore_count)
                with col2:
                    trip_count = len(df_to_display[df_to_display['source'].str.contains('Trip', case=False, na=False)])
                    st.metric("🚗 Trip Feedbacks (Filtered)", trip_count)
                with col3:
                    escalation_count = len(df_to_display[df_to_display['source'].str.contains('Escalation', case=False, na=False)])
                    st.metric("🚨 Escalations (Filtered)", escalation_count)
                st.markdown("---") # Separator after metrics

            # Render the selected page content
            if page_selection == "📈 Overview":
                create_overview_dashboard(df_to_display)
            elif page_selection == "📱 PlayStore Feedback":
                create_playstore_view(df_to_display)
            elif page_selection == "🚗 Trip Feedback":
                create_trip_feedback_view(df_to_display)
            elif page_selection == "🚨 Escalations":
                create_escalation_view(df_to_display)
            elif page_selection == "🔍 Raw Data":
                create_raw_data_view(df_to_display)
            
    except FileNotFoundError:
        st.error(f"❌ Error: The file was not found at the specified path: `{csv_file_path}`")
        st.info("Please ensure your CSV file is located in the same directory as your script, or update the `csv_file_path` variable with the correct absolute path.")
    except pd.errors.EmptyDataError:
        st.error(f"❌ Error: The file at `{csv_file_path}` is empty or has no data.")
        st.info("Please ensure your CSV file contains data.")
    except Exception as e:
        st.error(f"❌ An unexpected error occurred during data processing: {str(e)}")
        st.info("Please verify your CSV file's format. It should contain columns like `feedback_id`, `timestamp_` (or similar), `source`, `user_id`, `rating`, `feedback`, `escalated_flag`, and `customer_type`.")
        
        with st.expander("🔍 Show Detailed Error Information"):
            st.write(f"Attempted to load data from: `{csv_file_path}`")
            st.write("First few rows of your file (if readable):")
            try:
                temp_df = pd.read_csv(csv_file_path)
                st.dataframe(temp_df.head())
                st.write("Detected column names in your file:")
                st.write(list(temp_df.columns))
            except Exception as inner_e:
                st.write(f"Could not read the file or display its head due to an internal error: {inner_e}")
            st.write(f"Detailed error: {e}")

    # Add creators' names at the very bottom
    st.markdown("""
    <div style='text-align: center; margin-top: 50px; padding: 20px; background: linear-gradient(135deg, #f7fafc 0%, #edf2f7 100%); border-radius: 10px;'>
        <p style='color: #4a5568; font-size: 14px; margin: 0;'>🚀 Developed with ❤️ by <strong>Yash & Mohit</strong></p>
        <p style='color: #718096; font-size: 12px; margin: 5px 0 0 0;'>Enhanced PulsePoint Dashboard v2.0</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main_app()
